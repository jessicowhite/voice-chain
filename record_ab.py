"""Record raw mic + NVIDIA Broadcast virtual mic simultaneously (same utterance, 48 kHz).

usage: python record_ab.py <label> [seconds]
Finds devices by name on WASAPI so it survives index changes.
"""
import sys, time, winsound
import numpy as np
import sounddevice as sd
import soundfile as sf

RAW_NAME = "Wireless microphone"
BCAST_NAME = "NVIDIA Broadcast"
SR = 48000


def find(name):
    wasapi = next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == wasapi and d["max_input_channels"] > 0 and name in d["name"]:
            return i
    raise SystemExit(f"device not found: {name}")


def main():
    label = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 45
    bufs = {"raw": [], "bcast": []}
    ids = {"raw": find(RAW_NAME), "bcast": find(BCAST_NAME)}
    streams = []
    for k, dev in ids.items():
        def cb(indata, frames, t, status, k=k):
            bufs[k].append(indata[:, 0].copy())
        streams.append(sd.InputStream(device=dev, samplerate=SR, channels=1, dtype="float32",
                                      callback=cb, extra_settings=sd.WasapiSettings(auto_convert=True)))
    for s in streams:
        s.start()
    winsound.Beep(1000, 300)
    print(f"RECORDING {label} for {secs}s ...", flush=True)
    time.sleep(secs)
    for s in streams:
        s.stop(); s.close()
    winsound.Beep(600, 500)
    for k, v in bufs.items():
        x = np.concatenate(v)
        sf.write(f"rec/{label}_{k}.wav", x, SR, subtype="FLOAT")
        print(k, len(x) / SR, "s  peak", float(np.abs(x).max()))


if __name__ == "__main__":
    import os; os.makedirs("rec", exist_ok=True)
    main()
