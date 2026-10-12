"""Client for dfn_worker.py. DFNProc.process(x) -> y (same length); raises DFNError if the worker is gone."""
import os, struct, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


class DFNError(RuntimeError):
    pass


class DFNProc:
    def __init__(self, atten_db=100.0, pf_beta=0.0):
        py = Path(sys.executable)
        if py.name.lower() == "pythonw.exe":  # console-less parent: use python.exe for real pipes, hide its window
            py = py.with_name("python.exe")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.p = subprocess.Popen([str(py), str(HERE / "dfn_worker.py")], cwd=str(HERE), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags,
                                  env={**os.environ, "PYTHONPATH": str(HERE)})
        if self.p.stdout.read(1) != b"R":
            raise DFNError("worker failed to start")
        self.set(atten_db, pf_beta)

    def _w(self, b):
        try:
            self.p.stdin.write(b); self.p.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise DFNError(f"worker pipe closed: {e}")

    def _r(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                raise DFNError("worker exited")
            buf += chunk
        return buf

    def set(self, atten_db, pf_beta):
        self._w(struct.pack("<Iff", 0xFFFFFFFF, float(atten_db), float(pf_beta)))

    def process(self, x):
        x = np.ascontiguousarray(x, dtype="<f4")
        self._w(struct.pack("<I", len(x)) + x.tobytes())
        return np.frombuffer(self._r(len(x) * 4), dtype="<f4").astype(np.float32)

    def alive(self):
        return self.p.poll() is None

    def close(self):
        try:
            self.p.stdin.close(); self.p.terminate(); self.p.wait(timeout=2)
        except Exception:
            try: self.p.kill()
            except Exception: pass
