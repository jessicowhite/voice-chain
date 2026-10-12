"""DeepFilterNet3 via its official LADSPA plugin DLL (tools/dfn/deep_filter_ladspa.dll), driven with ctypes.

Streaming mono @48 kHz. DFN.process(x) takes any-length float32 and returns the same length; the
plugin itself buffers internally (its own STFT/lookahead delay is reported as DFN.delay_samples).
"""
import ctypes as C
from ctypes import c_ulong, c_int, c_float, c_char_p, c_void_p, POINTER
from pathlib import Path
import numpy as np

DLL = Path(__file__).parent / "tools" / "dfn" / "deep_filter_ladspa.dll"


class _Hint(C.Structure):
    _fields_ = [("h", c_int), ("lo", c_float), ("hi", c_float)]


class _Desc(C.Structure):
    pass


_Desc._fields_ = [
    ("UniqueID", c_ulong), ("Label", c_char_p), ("Properties", c_int), ("Name", c_char_p), ("Maker", c_char_p),
    ("Copyright", c_char_p), ("PortCount", c_ulong), ("PortDescriptors", POINTER(c_int)), ("PortNames", POINTER(c_char_p)),
    ("PortRangeHints", POINTER(_Hint)), ("Impl", c_void_p),
    ("instantiate", C.CFUNCTYPE(c_void_p, POINTER(_Desc), c_ulong)),
    ("connect_port", C.CFUNCTYPE(None, c_void_p, c_ulong, c_void_p)),
    ("activate", C.CFUNCTYPE(None, c_void_p)), ("run", C.CFUNCTYPE(None, c_void_p, c_ulong)),
    ("run_adding", c_void_p), ("set_gain", c_void_p),
    ("deactivate", C.CFUNCTYPE(None, c_void_p)), ("cleanup", C.CFUNCTYPE(None, c_void_p)),
]

_lib = None


def available():
    return DLL.exists()


def _load():
    global _lib
    if _lib is None:
        _lib = C.CDLL(str(DLL))
        _lib.ladspa_descriptor.restype = POINTER(_Desc)
        _lib.ladspa_descriptor.argtypes = [c_ulong]
    return _lib


class DFN:
    MAXBLOCK = 48000  # largest block process() hands the plugin in one run() call

    def __init__(self, sr=48000, atten_db=100.0, min_thresh=-15.0, max_erb=35.0, max_df=35.0, min_buf=0.0, pf_beta=0.0):
        lib = _load()
        self.d = lib.ladspa_descriptor(0).contents  # deep_filter_mono
        self.h = self.d.instantiate(C.pointer(self.d), sr)
        if not self.h:
            raise RuntimeError("DeepFilterNet plugin failed to instantiate")
        self.inb = (c_float * self.MAXBLOCK)()
        self.outb = (c_float * self.MAXBLOCK)()
        self.ctl = [c_float(v) for v in (atten_db, min_thresh, max_erb, max_df, min_buf, pf_beta)]
        self.d.connect_port(self.h, 0, C.addressof(self.inb))
        self.d.connect_port(self.h, 1, C.addressof(self.outb))
        for i, c in enumerate(self.ctl):
            self.d.connect_port(self.h, 2 + i, C.addressof(c))
        self.d.activate(self.h)

    def set(self, atten_db=None, pf_beta=None):
        if atten_db is not None: self.ctl[0].value = float(atten_db)
        if pf_beta is not None: self.ctl[5].value = float(pf_beta)

    def process(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty(len(x), np.float32)
        for s in range(0, len(x), self.MAXBLOCK):
            blk = x[s:s + self.MAXBLOCK]
            C.memmove(self.inb, blk.ctypes.data, blk.nbytes)
            self.d.run(self.h, len(blk))
            C.memmove(out[s:s + len(blk)].ctypes.data, self.outb, blk.nbytes)
        return out

    def close(self):
        try:
            self.d.deactivate(self.h); self.d.cleanup(self.h)
        except Exception:
            pass
        self.h = None
