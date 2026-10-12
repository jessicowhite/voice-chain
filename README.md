# Voice Chain

Real-time microphone cleanup for Windows, built after measuring what NVIDIA Broadcast's *Noise Removal* does.
Any input mic -> noise removal -> EQ -> compressor -> limiter -> volume -> a virtual cable, which other apps use as their mic.

![chain](https://img.shields.io/badge/chain-HPF%20%E2%86%92%20RNNoise%20%E2%86%92%20gate%2Bexpander%20%E2%86%92%20EQ%20%E2%86%92%20comp%20%E2%86%92%20limiter%20%E2%86%92%20volume-76B900)

## Setup

1. Install [VB-Cable](https://vb-audio.com/Cable/) (the virtual audio device; not bundled).
2. `powershell -ExecutionPolicy Bypass -File setup.ps1` (venv, dependencies, Desktop shortcut). Needs Python 3.12+ on PATH.
3. Launch **Voice Chain** from the Desktop, pick your mic, flip **ON / OFF**.
4. In Discord / OBS / etc. set the microphone to **CABLE Output (VB-Audio Virtual Cable)**.
   Leave NVIDIA Broadcast's noise removal off for that mic so the two don't stack.

## Dashboard

| Area | What it does |
|---|---|
| Microphone | Every input device on the PC, switchable live; *Refresh devices* picks up hot-plugged mics |
| ON / OFF, Bypass (A/B) | Master switch; bypass sends the raw mic for instant comparison |
| Live | Input/output meters, voice detector, gate state, latency, glitch counter |
| Noise removal | Strength, gate floor, voice sensitivity, hold, release, passes; optional DeepFilterNet3 engine |
| Tone / EQ | Low cut, low-mid cut, presence, air |
| Dynamics | Compressor threshold/ratio, makeup gain, limiter ceiling |
| Output volume | Master level after the limiter (-40..+6 dB, default -6). Not part of presets |
| Presets | Strong (G, default), Broadcast match, Gentle, Raw, Quality (DFN3); save your own |
| Tray + mini widget | Always-on-top panel with on/off, strength, volume, level meter |

Settings and presets are saved next to the app (`settings.json`, `presets.json`, git-ignored).

## What NVIDIA Broadcast actually does (measured)

Recorded the same mic through Broadcast's virtual mic and the raw mic at the same time, then compared (`record_ab.py`, `analyze.py`):

- Speech passes at ~0 dB; it is **not** a compressor or loudness leveler (speech crest factor unchanged).
- Below about -32 dBFS it applies a downward-expander curve: ~-3 dB at -41, -6 at -47, -11 at -53, ~-22 below -56.
- Non-speech is hard-gated to digital silence (VAD).
- The models are TensorRT engines (`.trtpkg`) in `ProgramData\NVIDIA\NGX\models\nvbcast_afx_*`; the engine is the NVIDIA Audio Effects SDK (`NVAudioEffects.dll`).

This chain reproduces that curve (`exp_strength`, 1.0 = Broadcast-matched, default 2.0 = stronger) on top of RNNoise and a VAD gate.

## Optional: DeepFilterNet3 engine ("Quality")

Off by default (RNNoise is lighter and cuts more noise). To enable, place the official
`deep_filter_ladspa-0.5.6-x86_64-pc-windows-msvc.dll` from the
[DeepFilterNet releases](https://github.com/Rikorose/DeepFilterNet/releases/tag/v0.5.6) at
`tools/dfn/deep_filter_ladspa.dll`. It runs in a separate worker process (the plugin panics if it falls behind real time),
adds ~30-40 ms, and the app falls back to RNNoise automatically if the worker dies.

## Files

| File | Role |
|---|---|
| `app.py` | Dashboard, tray, mini widget |
| `engine.py` | Audio streams, device handling, meters, glitch diagnostics, presets |
| `chain.py` | The DSP chain (also a CLI: `python chain.py in.wav out.wav [--dfn --exp=2.0 ...]`) |
| `dfn*.py` | DeepFilterNet3 plugin wrapper, worker process, client |
| `live.py`, `run_live.bat` | Headless live runner (no UI) |
| `record_ab.py`, `analyze.py` | Measurement tools: record raw + Broadcast simultaneously, compare levels/spectrum/gain curve |

## Performance

About 1% of total CPU (12-thread Ryzen 5 3600); no GPU use. RNNoise is ~0.6 ms per 10 ms frame; the rest is the UI,
which redraws only on change and not at all while hidden in the tray. Algorithmic latency ~20 ms (RNNoise path).

## Credits

[RNNoise](https://github.com/xiph/rnnoise) via [pyrnnoise](https://github.com/pengzhendong/pyrnnoise),
[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), [Pedalboard](https://github.com/spotify/pedalboard),
[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter), [VB-Cable](https://vb-audio.com/Cable/) (VB-Audio, not redistributed).
Not affiliated with NVIDIA.
