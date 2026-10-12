"""Threaded audio engine: any input mic -> Chain -> virtual cable. UI-agnostic.

Engine.start(input_name) / stop() / set(**params) / bypass / meters. Sounddevice runs its own audio
threads; the chain is guarded by a lock so parameter changes never race a processing call.
"""
import json, queue, threading, time
from pathlib import Path
import numpy as np
import sounddevice as sd
from chain import Chain, CFG, FRAME, SR

HERE = Path(__file__).parent
BLOCK = FRAME * 2  # 20 ms
MAX_BACKLOG = 3    # blocks queued between mic and cable before the oldest are skipped (clock-drift guard)
OUT_HINT = "CABLE Input"
# Devices we never offer as inputs (our own outputs / Broadcast's virtual mic would loop or double-process)
HIDE = ("CABLE Output", "Stereo Mix", "NVIDIA Broadcast", "Sound Mapper")


def _wasapi():
    return next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])


def list_inputs():
    api = _wasapi()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == api and d["max_input_channels"] > 0 and not any(h in d["name"] for h in HIDE):
            out.append(d["name"])
    return out


def list_outputs():
    api = _wasapi()
    return [d["name"] for d in sd.query_devices() if d["hostapi"] == api and d["max_output_channels"] > 0]


def refresh_devices():
    """Re-enumerate (hot-plug). Only safe while no stream is open."""
    sd._terminate(); sd._initialize()


def _find(name, kind):
    api = _wasapi()
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == api and d[f"max_{kind}_channels"] > 0 and d["name"] == name:
            return i
    for i, d in enumerate(sd.query_devices()):  # fall back to substring
        if d["hostapi"] == api and d[f"max_{kind}_channels"] > 0 and name.lower() in d["name"].lower():
            return i
    raise LookupError(f"{kind} device not found: {name}")


def default_output():
    for n in list_outputs():
        if OUT_HINT in n:
            return n
    return None


class Engine:
    def __init__(self):
        self.chain = Chain()
        self.lock = threading.Lock()
        self.q = queue.Queue(maxsize=50)
        self.in_stream = self.out_stream = None
        self.running = False
        self.bypass = False
        self.error = ""
        self.in_db = self.out_db = -90.0  # smoothed meters
        self.input_name = None
        self.output_name = default_output()
        self.stats = self._new_stats()
        self._t_start, self._warm = 0.0, True

    @staticmethod
    def _new_stats():
        return dict(underrun=0, overflow=0, trimmed=0, in_flags=0, out_flags=0, slow=0, max_q=0, cb_n=0, cb_sum=0.0, cb_max=0.0)

    # ---- parameters -------------------------------------------------------------------
    def set(self, **params):
        with self.lock:
            self.chain.apply(**params)

    def get(self):
        return dict(self.chain.c)

    # ---- audio callbacks --------------------------------------------------------------
    @staticmethod
    def _db(x):
        return 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9) if len(x) else -90.0

    def _cb_in(self, indata, frames, t, status):
        if status: self.stats["in_flags"] += 1
        try:
            self.q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.stats["overflow"] += 1  # drop instead of building latency

    def _cb_out(self, outdata, frames, t, status):
        t0 = time.perf_counter()
        st = self.stats
        if status: st["out_flags"] += 1
        outdata[:] = 0
        try:
            x = self.q.get_nowait()
        except queue.Empty:
            st["underrun"] += 1
            return
        # Mic and cable run on independent clocks, so the backlog slowly drifts. Keep it short:
        # past MAX_BACKLOG blocks, run the oldest ones through the chain (state stays continuous) but don't play them.
        while self.q.qsize() > MAX_BACKLOG:
            try:
                newer = self.q.get_nowait()
            except queue.Empty:
                break
            if not self.bypass:
                with self.lock: self.chain.process(x)
            x = newer; st["trimmed"] += 1
        st["max_q"] = max(st["max_q"], self.q.qsize())
        self.in_db = max(self._db(x), self.in_db - 1.5)
        if self.bypass:
            y = x
        else:
            with self.lock:
                y = self.chain.process(x)
        self.out_db = max(self._db(y), self.out_db - 1.5)
        n = min(len(y), frames)
        outdata[:n, :] = y[:n, None]
        ms = (time.perf_counter() - t0) * 1000
        st["cb_n"] += 1; st["cb_sum"] += ms; st["cb_max"] = max(st["cb_max"], ms)
        if ms > 15: st["slow"] += 1  # callback used >75% of its 20 ms budget

    # ---- lifecycle --------------------------------------------------------------------
    def start(self, input_name, output_name=None):
        self.stop()
        self.error = ""
        self.stats = self._new_stats()
        self.output_name = output_name or self.output_name or default_output()
        if not self.output_name:
            self.error = "Virtual cable not found. Install VB-Cable."
            return False
        try:
            ws = sd.WasapiSettings(auto_convert=True)
            while not self.q.empty():
                self.q.get_nowait()
            old = self.chain
            self.chain = Chain(old.c, dfn=old.dfn)  # fresh denoiser/gate state, same settings, keep DFN worker
            old.dfn = None
            self.in_stream = sd.InputStream(device=_find(input_name, "input"), samplerate=SR, channels=1,
                                            dtype="float32", blocksize=BLOCK, callback=self._cb_in, extra_settings=ws)
            self.out_stream = sd.OutputStream(device=_find(self.output_name, "output"), samplerate=SR, channels=2,
                                              dtype="float32", blocksize=BLOCK, callback=self._cb_out, extra_settings=ws)
            self.in_stream.start()
            t0 = time.time()
            while self.q.qsize() < 2 and time.time() - t0 < 0.5:  # pre-fill so the output never starts on an empty queue
                time.sleep(0.005)
            self.stats = self._new_stats()
            self._t_start, self._warm = time.time(), False
            self.out_stream.start()
            self.input_name, self.running = input_name, True
            return True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.stop()
            return False

    def stop(self):
        for s in (self.in_stream, self.out_stream):
            try:
                if s: s.stop(); s.close()
            except Exception:
                pass
        self.in_stream = self.out_stream = None
        self.running = False
        self.in_db = self.out_db = -90.0

    def shutdown(self):
        self.stop()
        self.chain.close()

    # ---- meters -----------------------------------------------------------------------
    def meters(self):
        c = self.chain
        if self.running and not self._warm and time.time() - self._t_start > 2.0:
            self.stats = self._new_stats(); self._warm = True  # startup clicks aren't "glitches"
        return dict(in_db=self.in_db, out_db=self.out_db, vad=c.last_prob, gate_open=c.open, exp_db=c.eg,
                    running=self.running, dfn=c.dfn_status, latency_ms=c.latency_ms() + self.q.qsize() * 20.0,
                    glitches=self.stats["underrun"] + self.stats["overflow"] + self.stats["trimmed"] + self.stats["out_flags"]
                    + self.stats["in_flags"])

    def diag(self):
        s = dict(self.stats)
        s["cb_avg_ms"] = s["cb_sum"] / s["cb_n"] if s["cb_n"] else 0.0
        return s


# ---- presets --------------------------------------------------------------------------
PRESET_FILE = HERE / "presets.json"
_BASE = {k: CFG[k] for k in CFG}
DEFAULT_PRESETS = {
    "Strong (G)": {},
    "Quality (DFN3)": dict(denoiser="dfn", dfn_pf=0.03, exp_strength=2.0),
    "Broadcast match": dict(exp_strength=1.0, floor_db=-24.0),
    "Gentle": dict(exp_strength=0.7, floor_db=-18.0, comp_on=False),
    "Raw (no processing)": dict(exp_strength=0.0, gate_on=False, eq_on=False, comp_on=False, passes=1, hpf_hz=20),
}


def load_presets():
    p = {k: dict(_BASE, **v) for k, v in DEFAULT_PRESETS.items()}
    if PRESET_FILE.exists():
        try:
            user = json.loads(PRESET_FILE.read_text())
            p.update({k: dict(_BASE, **v) for k, v in user.items()})
        except Exception:
            pass
    return p


def save_user_preset(name, cfg):
    user = {}
    if PRESET_FILE.exists():
        try: user = json.loads(PRESET_FILE.read_text())
        except Exception: pass
    user[name] = cfg
    PRESET_FILE.write_text(json.dumps(user, indent=1))


def delete_user_preset(name):
    if PRESET_FILE.exists():
        user = json.loads(PRESET_FILE.read_text()); user.pop(name, None)
        PRESET_FILE.write_text(json.dumps(user, indent=1))


SETTINGS_FILE = HERE / "settings.json"


def load_settings():
    try: return json.loads(SETTINGS_FILE.read_text())
    except Exception: return {}


def save_settings(d):
    SETTINGS_FILE.write_text(json.dumps(d, indent=1))
