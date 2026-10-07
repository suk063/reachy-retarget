"""State-only copies of image-embedding HDF5 files.

Many published demonstration files (MimicGen, LIBERO, DexMimicGen) store camera
observations next to the simulator states. This project never stores images, so a
verified download of a ``kind: images_embedded`` file is rewritten as
``<stem>.state.hdf5``: every group, dataset and attribute is copied except image-like
datasets, then the original is deleted. A dataset is dropped when its full HDF5 path
matches the image guard of :func:`reachy_retarget.schema.io.check_not_image` (names
containing image/rgb/depth/camera/segmentation/...) or when it is a ``uint8`` array
with ``ndim >= 3``. Copied datasets keep their dtype, shape, chunking and filters.

The stripped file carries one extra root attribute, :data:`STRIP_ATTR`, a JSON record
of the original (catalog id, URL, revision, SHA-256, size) and of what was dropped, so
adapters can find the catalog entry of a stripped file without the ledger.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from ..schema.io import check_not_image

STRIP_ATTR = "reachy_retarget_stripped"
SUFFIX = ".state.hdf5"


def stripped_path(path) -> Path:
    """``a/b/stack.hdf5`` -> ``a/b/stack.state.hdf5``."""
    path = Path(path)
    return path.with_name(path.stem + SUFFIX)


def is_image(name: str, ds: h5py.Dataset) -> bool:
    """Same rule as the episode writer's guard, evaluated on a dataset's metadata only."""
    probe = np.empty((0,) * len(ds.shape), dtype=ds.dtype)  # metadata only, no data read
    try:
        check_not_image(name, probe)
    except ValueError:
        return True
    return False


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def strip_images(src, dst, *, original: dict | None = None) -> dict:
    """Write a state-only copy of ``src`` to ``dst`` (atomic) and return a summary.

    ``original`` (catalog id, url, sha256, size, ...) is stored with the summary in the
    root attribute :data:`STRIP_ATTR`. ``src`` is not modified or deleted here.
    """
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    dropped, kept = [], 0
    dropped_bytes = 0
    try:
        with h5py.File(src, "r") as fi, h5py.File(tmp, "w") as fo:
            _copy_attrs(fi, fo)

            def walk(gi: h5py.Group, go: h5py.Group):
                nonlocal kept, dropped_bytes
                for name, obj in gi.items():
                    link = gi.get(name, getlink=True)
                    if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                        go[name] = link
                    elif isinstance(obj, h5py.Group):
                        sub = go.create_group(name)
                        _copy_attrs(obj, sub)
                        walk(obj, sub)
                    elif is_image(obj.name, obj):
                        dropped.append(obj.name)
                        dropped_bytes += obj.id.get_storage_size()
                    else:
                        gi.copy(obj, go, name=name)
                        kept += 1

            walk(fi, fo)
            summary = {
                "original": original or {"file": src.name},
                "dropped_datasets": len(dropped),
                "dropped_names": sorted({_pattern(n) for n in dropped}),
                "dropped_stored_bytes": int(dropped_bytes),
                "kept_datasets": kept,
                "rule": "schema.io.check_not_image: image-like path or uint8 ndim>=3",
                "stripped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            fo.attrs[STRIP_ATTR] = json.dumps(summary, sort_keys=True)
        _verify(src, tmp)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    return summary


def _pattern(name: str) -> str:
    """``/data/demo_12/obs/agentview_image`` -> ``/data/*/obs/agentview_image``."""
    parts = name.split("/")
    return "/".join("*" if p.startswith("demo_") and p[5:].isdigit() else p for p in parts)


def _verify(src: Path, dst: Path) -> None:
    """Every non-image dataset and attribute of ``src`` is present and equal in ``dst``."""
    with h5py.File(src, "r") as fi, h5py.File(dst, "r") as fo:
        def same_attrs(a, b):
            for k, v in a.attrs.items():
                w = b.attrs[k]
                if not np.array_equal(np.asarray(v), np.asarray(w)):
                    raise RuntimeError(f"strip verification: attribute {a.name}@{k} differs")

        same_attrs(fi, fo)

        def check(name, obj):
            if isinstance(obj, h5py.Dataset) and is_image(obj.name, obj):
                if name in fo:
                    raise RuntimeError(f"strip verification: image dataset {name} was kept")
                return
            other = fo.get(name)
            if other is None:
                raise RuntimeError(f"strip verification: {name} missing")
            same_attrs(obj, other)
            if isinstance(obj, h5py.Dataset):
                if obj.shape != other.shape or obj.dtype != other.dtype:
                    raise RuntimeError(f"strip verification: {name} shape/dtype differ")
                if obj.size and not np.array_equal(obj[()], other[()]):
                    raise RuntimeError(f"strip verification: {name} values differ")

        fi.visititems(check)


def read_strip_record(path) -> dict | None:
    """The :data:`STRIP_ATTR` record of a stripped file, or ``None`` for any other file."""
    try:
        with h5py.File(path, "r") as f:
            raw = f.attrs.get(STRIP_ATTR)
    except OSError:
        return None
    if raw is None:
        return None
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


__all__ = ["STRIP_ATTR", "SUFFIX", "is_image", "read_strip_record", "strip_images", "stripped_path"]
