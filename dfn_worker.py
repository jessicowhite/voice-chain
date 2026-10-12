"""DeepFilterNet3 worker process (crash isolation: the Rust plugin panics if it falls behind real time).

Protocol on stdin/stdout (binary, little-endian):
  request : uint32 n, then n float32 samples          -> response: n float32 samples
  settings: uint32 0xFFFFFFFF, float32 atten_db, float32 pf_beta   (no response)
Warm-up runs on start; worker prints one byte b'R' when ready.
"""
import os, struct, sys
import numpy as np

# keep fd 1 for the protocol only; anything the plugin prints goes to stderr
proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)
proto_in = sys.stdin.buffer

from dfn import DFN  # noqa: E402


def read_exact(n):
    buf = b""
    while len(buf) < n:
        chunk = proto_in.read(n - len(buf))
        if not chunk:
            raise SystemExit(0)
        buf += chunk
    return buf


def main():
    d = DFN(min_buf=float(os.environ.get("DFN_MINBUF", "1")))
    z = np.zeros(4800, np.float32)
    for _ in range(40):  # ~4 s of silence in big blocks: loads the model without tripping the watchdog
        d.process(z)
    proto_out.write(b"R")
    while True:
        n = struct.unpack("<I", read_exact(4))[0]
        if n == 0xFFFFFFFF:
            atten, pf = struct.unpack("<ff", read_exact(8))
            d.set(atten_db=atten, pf_beta=pf)
            continue
        x = np.frombuffer(read_exact(n * 4), dtype="<f4")
        proto_out.write(d.process(x).astype("<f4").tobytes())


if __name__ == "__main__":
    main()
