"""Voice chain: HPF -> denoise (RNNoise) -> VAD-driven expander -> EQ -> compressor -> limiter.

Streaming-safe: Chain.process(block) takes float32 mono @48k of any length and returns the same
length (internal 10 ms framing, 10 ms algorithmic delay). Same class is used offline and live.
usage: python chain.py in.wav out.wav [--no-gate] [--no-eq] [--no-comp]
"""
import sys, threading
from collections import deque
import numpy as np
import soundfile as sf
from pedalboard import (Pedalboard, HighpassFilter, PeakFilter, HighShelfFilter, LowShelfFilter,
                        Compressor, Limiter, Gain)
import ctypes
from pyrnnoise.rnnoise import lib as _rn_lib, create as _rn_create, destroy as _rn_destroy

SR = 48000
FRAME = 480  # 10 ms, RNNoise native
class _RN:
    """Direct binding to RNNoise: pyrnnoise's Python wrapper spends ~6x longer than the native call on conversions."""
    _P = ctypes.POINTER(ctypes.c_float)

    def __init__(self):
        self.st = _rn_create()
        self.buf = np.zeros(FRAME, np.float32)
        self.ptr = self.buf.ctypes.data_as(self._P)

    def __call__(self, fr):
        """fr: float32 in [-1, 1], FRAME samples -> (speech_prob, denoised float32 in [-1, 1])"""
        np.multiply(fr, 32768.0, out=self.buf)
        np.clip(self.buf, -32768.0, 32767.0, out=self.buf)
        prob = _rn_lib.rnnoise_process_frame(self.st, self.ptr, self.ptr)
        return float(prob), self.buf * np.float32(1.0 / 32768.0)

    def __del__(self):
        try: _rn_destroy(self.st)
        except Exception: pass


_RAMP = np.linspace(0.0, 1.0, FRAME, dtype=np.float32)  # shared 0..1 ramp for gain smoothing
DFN_DELAY_FRAMES = 3  # DeepFilterNet3 plugin delay with min_buf=1 under real-time pacing (~30-40 ms)

CFG = dict(
    denoiser="rnnoise", dfn_atten=100.0, dfn_pf=0.0,  # "rnnoise" (fast) | "dfn" (DeepFilterNet3, +70 ms)
    hpf_hz=80, passes=1,
    exp_strength=2.0, exp_max_db=-60.0, exp_attack_ms=5.0, exp_release_ms=120.0,
    # expander / gate (Broadcast measured: ~0 dB above -32 dBFS, ~-6 dB @ -47, ~-22 dB below -56)
    gate_on=True, vad_open=0.55, vad_close=0.30, floor_db=-40.0,
    attack_ms=6.0, release_ms=180.0, hold_ms=120.0,
    # EQ (tunable by ear): low-mid cut, presence, air
    eq_on=True, lowmid_hz=300, lowmid_db=-1.5, presence_hz=3500, presence_db=2.0, air_hz=10000, air_db=2.0,
    # dynamics
    comp_on=True, comp_thresh_db=-22.0, comp_ratio=3.0, comp_attack_ms=8.0, comp_release_ms=120.0,
    makeup_db=3.0, limiter_db=-1.0,
    out_db=-6.0,  # master output volume, applied after the limiter (not part of presets)
)


class Chain:
    def __init__(self, cfg=None, dfn=None):
        self.c = dict(CFG, **(cfg or {}))
        self._adopt = dfn  # reuse a running DFN worker (restarts of the engine shouldn't reload the model)
        c = self.c
        self.dn = _RN()
        self.extra = []
        for _ in range(c["passes"] - 1):
            self.extra.append(_RN())
        # Plugins are always present so every parameter can change live; "off" = neutral setting.
        self.hpf = HighpassFilter(80)
        self.pre = Pedalboard([self.hpf])
        self.eq_low = PeakFilter(300, 0, 0.8)
        self.eq_pres = PeakFilter(3500, 0, 0.9)
        self.eq_air = HighShelfFilter(10000, 0, 0.7)
        self.comp = Compressor(-22, 3, 8, 120)
        self.makeup = Gain(0)
        self.limiter = Limiter(-1.0, 60.0)
        self.out_gain = Gain(0)
        self.post = Pedalboard([self.eq_low, self.eq_pres, self.eq_air, self.comp, self.makeup, self.limiter, self.out_gain])
        self.dfn, self.dfn_status, self._dfn_loading = self._adopt, ("ready" if self._adopt else "off"), False
        self._fifo = deque()
        self.apply()
        self.buf = np.zeros(0, dtype=np.float32)
        self.g = 1.0          # current gate gain (linear)
        self.eg = 0.0         # current expander gain (dB)
        self.open = False
        self.hold = 0
        self.last_prob = 0.0

    def apply(self, **changes):
        """Update parameters live (gate/expander read self.c each frame; plugins are set here)."""
        c = self.c
        c.update(changes)
        self.hpf.cutoff_frequency_hz = float(c["hpf_hz"])
        eq = c["eq_on"]
        self.eq_low.cutoff_frequency_hz, self.eq_low.gain_db = float(c["lowmid_hz"]), float(c["lowmid_db"]) if eq else 0.0
        self.eq_pres.cutoff_frequency_hz, self.eq_pres.gain_db = float(c["presence_hz"]), float(c["presence_db"]) if eq else 0.0
        self.eq_air.cutoff_frequency_hz, self.eq_air.gain_db = float(c["air_hz"]), float(c["air_db"]) if eq else 0.0
        on = c["comp_on"]
        self.comp.threshold_db = float(c["comp_thresh_db"]) if on else 0.0
        self.comp.ratio = float(c["comp_ratio"]) if on else 1.0
        self.comp.attack_ms, self.comp.release_ms = float(c["comp_attack_ms"]), float(c["comp_release_ms"])
        self.makeup.gain_db = float(c["makeup_db"]) if on else 0.0
        self.limiter.threshold_db = float(c["limiter_db"])
        self.out_gain.gain_db = float(c["out_db"])
        want = int(c["passes"]) - 1
        while len(self.extra) < want:
            self.extra.append(_RN())
        del self.extra[want:]
        if c["denoiser"] == "dfn":
            if self.dfn is None and not self._dfn_loading and not self.dfn_status.startswith(("failed", "unavailable")):
                self._start_dfn()
            if self.dfn is not None:
                try: self.dfn.set(c["dfn_atten"], c["dfn_pf"])
                except Exception: pass

    def _gate(self, prob):
        c = self.c
        fdur = FRAME / SR * 1000
        if prob >= c["vad_open"]:
            self.open, self.hold = True, int(c["hold_ms"] / fdur)
        elif prob < c["vad_close"]:
            if self.hold > 0:
                self.hold -= 1
            else:
                self.open = False
        target = 1.0 if self.open else 10 ** (c["floor_db"] / 20)
        tc = c["attack_ms"] if target > self.g else c["release_ms"]
        a = 1 - np.exp(-fdur / tc)
        new = self.g + a * (target - self.g)
        old, self.g = self.g, new
        if abs(new - old) < 1e-4:
            return new  # settled: scalar gain, no per-sample ramp
        return old + (new - old) * _RAMP

    # Level-based downward expander; curve = Broadcast Noise Removal gain vs input frame level (measured).
    _EXP_LVL = np.array([-80, -65, -59, -53, -47, -41, -35, -32, 0.0])
    _EXP_GAIN = np.array([-23, -22, -21, -11, -6, -3, -0.9, 0.0, 0.0])

    def _expander(self, lvl):
        c = self.c
        tgt = float(np.interp(lvl, self._EXP_LVL, self._EXP_GAIN)) * c["exp_strength"]
        tgt = max(tgt, c["exp_max_db"])
        fdur = FRAME / SR * 1000
        tc = c["exp_attack_ms"] if tgt > self.eg else c["exp_release_ms"]  # opening = attack
        new = self.eg + (1 - np.exp(-fdur / tc)) * (tgt - self.eg)
        old, self.eg = self.eg, new
        if abs(new - old) < 0.01:
            return 10 ** (new / 20)  # settled
        return 10 ** ((old + (new - old) * _RAMP) / 20)

    # ---- DeepFilterNet3 (separate worker process; RNNoise stays on as VAD + fallback) ----
    def _start_dfn(self, block=False):
        if self.dfn is not None or self._dfn_loading:
            return
        self._dfn_loading, self.dfn_status = True, "loading"

        def work():
            try:
                from dfn_client import DFNProc
                d = DFNProc(self.c["dfn_atten"], self.c["dfn_pf"])
                self.dfn, self.dfn_status = d, "ready"
                self._fifo.clear()
            except Exception as e:  # missing DLL, worker crash, ...
                self.dfn, self.dfn_status = None, f"unavailable: {e}"
            self._dfn_loading = False
        if block:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    def _stop_dfn(self):
        d, self.dfn = self.dfn, None
        if d: d.close()
        self._fifo.clear()

    def close(self):
        self._stop_dfn()

    def latency_ms(self):
        return 20.0 + (DFN_DELAY_FRAMES * 10.0 if self.dfn is not None and self.c["denoiser"] == "dfn" else 0.0)

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        n_in = len(x)
        self.buf = np.concatenate([self.buf, self.pre(x[None, :], SR, reset=False)[0]])
        frames = []
        while len(self.buf) >= FRAME:
            fr, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            frames.append(fr)
        # 1) RNNoise on every frame: VAD probability always; its audio unless DFN is active
        dens, probs, lvls = [], [], []
        for fr in frames:
            prob, den = self.dn(fr)
            for dn2 in self.extra:  # extra cascade passes: denoise the denoised signal again
                _, den = dn2(den)
            dens.append(den)
            probs.append(prob)
            lvls.append(10.0 * np.log10(float(np.dot(fr, fr)) / FRAME + 1e-18))
        # 2) DeepFilterNet3 replaces the audio (needs >=2 frames per call); decisions are delayed to match it
        use_dfn = self.c["denoiser"] == "dfn" and self.dfn is not None and len(frames) >= 2
        if use_dfn:
            try:
                cat = self.dfn.process(np.concatenate(frames))
                dens = [cat[i * FRAME:(i + 1) * FRAME] for i in range(len(frames))]
                for i in range(len(frames)):  # FIFO of (prob, level): decision for frame i = input DFN_DELAY_FRAMES ago
                    self._fifo.append((probs[i], lvls[i]))
                    probs[i], lvls[i] = self._fifo.popleft() if len(self._fifo) > DFN_DELAY_FRAMES else (0.0, -90.0)
            except Exception as e:
                self._stop_dfn(); self.dfn_status = f"failed: {e}"
                use_dfn = False
        out = []
        for den, p, lvl in zip(dens, probs, lvls):
            self.last_prob = p
            if self.c["gate_on"]:
                den = den * self._gate(p)
            if self.c["exp_strength"] > 0:
                den = den * self._expander(lvl)
            out.append(den)
        y = np.concatenate(out) if out else np.zeros(0, np.float32)
        y = self.post(y[None, :], SR, reset=False)[0] if len(y) else y
        # keep output length == input length by tracking a small delay line
        self._pending = np.concatenate([getattr(self, "_pending", np.zeros(0, np.float32)), y])
        if len(self._pending) >= n_in:
            res, self._pending = self._pending[:n_in], self._pending[n_in:]
        else:
            res = np.concatenate([np.zeros(n_in - len(self._pending), np.float32), self._pending])
            self._pending = np.zeros(0, np.float32)
        return res


def main():
    src, dst = sys.argv[1], sys.argv[2]
    flags = set(sys.argv[3:])
    cfg = {}
    if "--no-gate" in flags: cfg["gate_on"] = False
    if "--no-eq" in flags: cfg["eq_on"] = False
    if "--no-comp" in flags: cfg["comp_on"] = False
    for f in flags:  # --floor=-45 --passes=2 --vad=0.7 --release=120
        if f.startswith("--floor="): cfg["floor_db"] = float(f[8:])
        if f.startswith("--passes="): cfg["passes"] = int(f[9:])
        if f.startswith("--exp="): cfg["exp_strength"] = float(f[6:])
        if f == "--dfn": cfg["denoiser"] = "dfn"
        if f.startswith("--atten="): cfg["dfn_atten"] = float(f[8:])
        if f.startswith("--vad="): cfg["vad_open"] = float(f[6:])
        if f.startswith("--release="): cfg["release_ms"] = float(f[10:])
    x, sr = sf.read(src, dtype="float32")
    assert sr == SR, f"need {SR} Hz, got {sr}"
    if x.ndim > 1: x = x[:, 0]
    ch = Chain(cfg)
    if cfg.get("denoiser") == "dfn":
        while ch._dfn_loading: import time; time.sleep(0.1)
        assert ch.dfn is not None, ch.dfn_status
    y = np.concatenate([ch.process(x[i:i + 960]) for i in range(0, len(x), 960)])
    sf.write(dst, y, SR, subtype="FLOAT")
    print("wrote", dst, len(y) / SR, "s")


if __name__ == "__main__":
    main()
