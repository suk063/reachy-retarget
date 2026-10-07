"""Shared offline helpers: placeholder asset archives for recorded robosuite MJCF."""
import re
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np

from reachy_retarget.acquire import CatalogEntry


def placeholder(suffix: str) -> bytes:
    """A tiny valid file of the given type (tetrahedron mesh or 1x1 PNG)."""
    v = np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    if suffix == ".stl":
        return b"\0" * 80 + struct.pack("<I", 4) + b"".join(
            struct.pack("<12fH", 0, 0, 0, *v[a], *v[b], *v[c], 0) for a, b, c in faces)
    if suffix == ".obj":
        return ("".join(f"v {x} {y} {z}\n" for x, y, z in v)
                + "".join(f"f {a+1} {b+1} {c+1}\n" for a, b, c in faces)).encode()
    if suffix == ".msh":
        return struct.pack("<4i", 4, 0, 0, 4) + v.tobytes() + faces.astype(np.int32).tobytes()
    chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0\x80\x80\x80")) + chunk(b"IEND", b""))


def archive_entry(root: Path, family: str, path: str, xml: str, recorded_marker: str, member_prefix: str,
                  asset_marker: str, extra: dict | None = None) -> CatalogEntry:
    """Write ``root/raw/<family>/<path>``: placeholders for every reference of ``xml``
    containing ``recorded_marker``, stored under ``member_prefix + asset_marker``."""
    rels = {f.split(recorded_marker, 1)[1] for f in re.findall(r'file="([^"]+)"', xml) if recorded_marker in f}
    dest = Path(root) / "raw" / family / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w") as z:
        for rel in rels:
            import posixpath
            rel = posixpath.normpath(rel)
            z.writestr(member_prefix + asset_marker + rel, placeholder(Path(rel).suffix))
        for name, data in (extra or {}).items():
            z.writestr(member_prefix + name, data)
    return CatalogEntry(id=f"{family}/{path}", family=family, path=path, url="https://example.invalid/" + path,
                        revision="test", sha256=None, size=None, license="MIT", kind="assets",
                        asset_marker=asset_marker)
