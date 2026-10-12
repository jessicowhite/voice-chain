"""Live mic -> Chain -> virtual cable. Pick the cable as your mic in Discord/OBS/etc.

usage: python live.py [--in "Wireless microphone"] [--out "CABLE Input"] [--list] [--monitor]
Opens both ends in WASAPI shared mode at 48 kHz mono (auto_convert), 10 ms blocks.
"""
import argparse, queue, sys, time
import numpy as np
import sounddevice as sd
from chain import Chain, FRAME, SR

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="Wireless microphone")
ap.add_argument("--out", default="CABLE Input")
ap.add_argument("--list", action="store_true")
ap.add_argument("--monitor", action="store_true", help="also print level/VAD meter")
a = ap.parse_args()


def wasapi():
    return next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])


def find(name, kind):
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == wasapi() and d[f"max_{kind}_channels"] > 0 and name.lower() in d["name"].lower():
            return i
    sys.exit(f"{kind} device not found: {name!r}  (run with --list)")


if a.list:
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == wasapi():
            print(i, d["name"], "in" if d["max_input_channels"] else "", "out" if d["max_output_channels"] else "")
    sys.exit()

ch = Chain()
q = queue.Queue(maxsize=50)
BLOCK = FRAME * 2  # 20 ms callback blocks


def cb_in(indata, frames, t, status):
    if status: print("in:", status, file=sys.stderr)
    try: q.put_nowait(indata[:, 0].copy())
    except queue.Full: pass  # drop rather than grow latency


def cb_out(outdata, frames, t, status):
    if status: print("out:", status, file=sys.stderr)
    try:
        y = ch.process(q.get_nowait())
    except queue.Empty:
        y = np.zeros(frames, np.float32)
    n = min(len(y), frames)
    outdata[:] = 0
    outdata[:n, :] = y[:n, None]


ws = sd.WasapiSettings(auto_convert=True)
with sd.InputStream(device=find(a.inp, "input"), samplerate=SR, channels=1, dtype="float32",
                    blocksize=BLOCK, callback=cb_in, extra_settings=ws), \
     sd.OutputStream(device=find(a.out, "output"), samplerate=SR, channels=2, dtype="float32",
                     blocksize=BLOCK, callback=cb_out, extra_settings=ws):
    print(f"live: {a.inp!r} -> chain -> {a.out!r}   Ctrl+C to stop", flush=True)
    try:
        while True:
            time.sleep(0.5)
            if a.monitor:
                print(f"vad {ch.last_prob:4.2f}  gate {'open' if ch.open else 'shut'}  exp {ch.eg:6.1f} dB", end="\r")
    except KeyboardInterrupt:
        print("\nstopped")
