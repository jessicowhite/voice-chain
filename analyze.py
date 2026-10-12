"""Compare raw vs Broadcast recordings: lag, levels, speech/noise split, spectrum, dynamics."""
import sys, json
import numpy as np, soundfile as sf
from scipy import signal
import pyloudnorm as pyln

SR = 48000
db = lambda v: 20 * np.log10(np.maximum(v, 1e-12))


def align(r, b):
    n = min(len(r), len(b), SR * 20)
    c = signal.correlate(b[:n], r[:n], mode="full", method="fft")
    lag = int(np.argmax(np.abs(c)) - (n - 1))  # b delayed by lag vs r
    if lag > 0:
        b = b[lag:]
    else:
        r = r[-lag:]
    m = min(len(r), len(b))
    return r[:m], b[:m], lag


def frames_rms(x, w=int(SR * 0.02)):
    n = len(x) // w
    return np.sqrt(np.mean(x[: n * w].reshape(n, w) ** 2, axis=1))


def ltas(x):
    f, p = signal.welch(x, SR, nperseg=8192)
    return f, 10 * np.log10(p + 1e-20)


def band(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.mean(p[m]))


def main(tag, other="bcast"):
    r, _ = sf.read(f"rec/{tag}_raw.wav"); b, _ = sf.read(f"rec/{tag}_{other}.wav")
    r, b, lag = align(r, b)
    fr, fb = frames_rms(r), frames_rms(b)
    # speech frames defined on the RAW signal: top 30% loudest frames
    thr = np.percentile(fr, 70)
    sp = fr >= thr
    ns = fr <= np.percentile(fr, 20)
    meter = pyln.Meter(SR)
    out = {
        "tag": tag, "lag_ms": lag / SR * 1000,
        "rms_dBFS": [float(db(np.sqrt(np.mean(r ** 2)))), float(db(np.sqrt(np.mean(b ** 2))))],
        "LUFS": [float(meter.integrated_loudness(r)), float(meter.integrated_loudness(b))],
        "peak_dBFS": [float(db(np.abs(r).max())), float(db(np.abs(b).max()))],
        "speech_rms_dB": [float(db(np.sqrt(np.mean(fr[sp] ** 2)))), float(db(np.sqrt(np.mean(fb[sp] ** 2))))],
        "noise_rms_dB": [float(db(np.sqrt(np.mean(fr[ns] ** 2)))), float(db(np.sqrt(np.mean(fb[ns] ** 2))))],
        "crest_dB_speech": [float(db(np.abs(r).max()) - db(np.sqrt(np.mean(fr[sp] ** 2)))),
                            float(db(np.abs(b).max()) - db(np.sqrt(np.mean(fb[sp] ** 2))))],
        # loudness range proxy: p95-p10 of frame levels
        "dyn_range_dB": [float(np.percentile(db(fr), 95) - np.percentile(db(fr), 10)),
                         float(np.percentile(db(fb), 95) - np.percentile(db(fb), 10))],
    }
    # long-term spectrum on speech frames only
    w = int(SR * 0.02)
    idx = np.where(sp)[0]
    rs = np.concatenate([r[i * w:(i + 1) * w] for i in idx]); bs = np.concatenate([b[i * w:(i + 1) * w] for i in idx])
    f, pr = ltas(rs); _, pb = ltas(bs)
    edges = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12000, 20000]
    out["band_diff_dB(bcast-raw)_speech"] = {f"{lo}-{hi}": round(band(f, pb, lo, hi) - band(f, pr, lo, hi), 2)
                                             for lo, hi in zip(edges[:-1], edges[1:])}
    # noise-frame spectrum change
    idn = np.where(ns)[0]
    if len(idn) > 10:
        rn = np.concatenate([r[i * w:(i + 1) * w] for i in idn]); bn = np.concatenate([b[i * w:(i + 1) * w] for i in idn])
        f, pr = ltas(rn); _, pb = ltas(bn)
        out["band_diff_dB(bcast-raw)_noise"] = {f"{lo}-{hi}": round(band(f, pb, lo, hi) - band(f, pr, lo, hi), 2)
                                                for lo, hi in zip(edges[:-1], edges[1:])}
    # gain curve: per-frame bcast level vs raw level (static compressor/expander shape)
    bins = np.arange(-80, 0, 6)
    curve = {}
    for lo in bins:
        m = (db(fr) >= lo) & (db(fr) < lo + 6)
        if m.sum() > 5:
            curve[f"{lo}..{lo+6}"] = [int(m.sum()), round(float(np.median(db(fb[m]) - db(fr[m]))), 2)]
    out["gain_vs_input_level(dB:[n,median_gain])"] = curve
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"rec/{tag}_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    other = "bcast"
    args = sys.argv[1:]
    if args and args[0].startswith("--vs="):
        other = args[0][5:]; args = args[1:]
    for t in args:
        main(t, other)
