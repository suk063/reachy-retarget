"""Restricted readers for RoboVerse (MetaSim) trajectory pickles, without torch.

RoboVerse ships trajectories as Python pickles. Unpickling arbitrary data can execute
code, so both readers refuse every global except an explicit allow-list:

* ``*.pkl.gz`` (RLBench, the CALVIN per-task files): plain dicts, lists and floats. No
  global is allowed at all.
* ``*.pkl`` written by the CALVIN converters: torch tensors (``torch.save`` legacy
  format, pickled through ``torch.storage._load_from_bytes``), ``numpy`` scalars/arrays and
  ``collections.OrderedDict``. The legacy torch serialization (magic number, protocol,
  system info, the pickled object with persistent storage ids, the storage key list, then
  each storage as ``int64 numel`` + raw little-endian bytes) is decoded here into numpy
  arrays; tensors become ``np.ndarray`` (``_rebuild_tensor_v2`` with offset/size/stride).
"""
from __future__ import annotations

import collections
import gzip
import io
import pickle
import struct
from pathlib import Path

import numpy as np

_TORCH_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64,
                 "IntStorage": np.int32, "ShortStorage": np.int16, "CharStorage": np.int8,
                 "ByteStorage": np.uint8, "BoolStorage": np.bool_, "HalfStorage": np.float16}


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"global {module}.{name} refused (plain RoboVerse pickle expected)")


class _StorageUnpickler(pickle.Unpickler):
    """The inner pickle of a legacy ``torch.save`` stream (storages as persistent ids)."""

    def __init__(self, f):
        super().__init__(f)
        self.storages: dict[str, str] = {}

    def find_class(self, module, name):
        if module == "torch" and name in _TORCH_DTYPES:
            return name
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"global {module}.{name} refused inside a torch storage")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        _, storage_type, key = pid[:3]
        self.storages[key] = storage_type
        return ("storage", key)


class _Storage:
    def __init__(self, arrays: dict, ref):
        self.arrays, self.ref = arrays, ref


def _load_from_bytes(b: bytes) -> _Storage:
    f = io.BytesIO(b)
    for _ in range(3):              # magic number, protocol version, system info
        pickle.load(f)
    u = _StorageUnpickler(f)
    ref = u.load()
    keys = _PlainUnpickler(f).load()
    arrays = {}
    for k in keys:
        n = struct.unpack("<q", f.read(8))[0]
        dt = np.dtype(_TORCH_DTYPES[u.storages[k]]).newbyteorder("<")
        arrays[k] = np.frombuffer(f.read(n * dt.itemsize), dt)
    return _Storage(arrays, ref)


def _rebuild_tensor_v2(storage, offset, size, stride, *_):
    if not isinstance(storage, _Storage):
        raise pickle.UnpicklingError("tensor without a decoded storage")
    arr = storage.arrays[storage.ref[1]]
    if not size:
        return arr[offset].copy()
    return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                           strides=[s * arr.itemsize for s in stride]).copy()


def _frombuffer(buf, dtype, shape, order):
    return np.frombuffer(buf, dtype).reshape(shape, order=order).copy()


def _reconstruct(*args):
    from numpy._core.multiarray import _reconstruct as rec
    return rec(*args)


def _scalar(dtype, data=None):
    from numpy._core.multiarray import scalar
    return scalar(dtype, data) if data is not None else scalar(dtype)


_ALLOWED = {
    ("torch.storage", "_load_from_bytes"): _load_from_bytes,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy._core.numeric", "_frombuffer"): _frombuffer,
    ("numpy.core.numeric", "_frombuffer"): _frombuffer,
    ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy._core.multiarray", "scalar"): _scalar,
    ("numpy.core.multiarray", "scalar"): _scalar,
}


class _TensorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"global {module}.{name} refused") from None


def load(path) -> object:
    """Read a RoboVerse trajectory pickle (``.pkl.gz`` plain, ``.pkl`` with tensors)."""
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    cls = _PlainUnpickler if path.suffix == ".gz" else _TensorUnpickler
    return cls(io.BytesIO(data)).load()


def load_npy_object(path) -> object:
    """Read a pickled object array saved by ``np.save`` (e.g. ``ann_dict.npy``) safely."""
    from numpy.lib import format as npf

    with open(path, "rb") as f:
        version = npf.read_magic(f)
        shape, _, dtype = (npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0)(f)
        if dtype != np.dtype(object):
            f.seek(0)
            return np.load(f, allow_pickle=False)
        obj = _TensorUnpickler(f).load()
    return obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj


__all__ = ["load", "load_npy_object"]
