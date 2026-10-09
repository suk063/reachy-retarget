"""Content-addressed asset library of a build output root: ``<out>/assets/<sha256>.<ext>``.

Every mesh and texture an episode needs is stored once per output root under the SHA-256 of
its bytes; episodes reference it as ``{"sha256", "format", "bytes"}``. Identical content gives
an identical name, so files are never rewritten: a new file is written to a temporary name in
the same directory and hard-linked to its final name (``os.link`` fails if the name exists, so
two jobs racing on the same content cannot clobber each other) and the temporary name is
removed. Merging libraries of several output roots (cluster publication) is a plain copy that
skips existing names. A library is never indexed: references carry format and size.

Formats: meshes are MuJoCo binary ``msh`` (see :func:`reachy_retarget.schema.scene_assets.encode_msh`),
textures keep their recorded bytes (``png``, ``jpg``, ``ktx`` ...).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..schema.scene_assets import asset_format


class AssetLibrary:
    """Writer of one library directory, deduplicating within the process by digest."""

    def __init__(self, root):
        self.root = Path(root)
        self._known: dict[str, int] = {}     # sha256 -> bytes, files known to exist
        self.new_files = 0
        self.new_bytes = 0

    def put(self, data: bytes, fmt: str | None = None, name: str = "") -> dict:
        """Store ``data`` (if not yet stored) and return its reference ``{"sha256", "format", "bytes"}``."""
        data = bytes(data)
        fmt = fmt or asset_format(data, name)
        sha = hashlib.sha256(data).hexdigest()
        ref = {"sha256": sha, "format": fmt, "bytes": len(data)}
        if sha in self._known:
            return ref
        path = self.path(ref)
        if not path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            try:
                tmp.write_bytes(data)
                try:
                    os.link(tmp, path)       # no-overwrite publication
                    self.new_files += 1
                    self.new_bytes += len(data)
                except FileExistsError:
                    pass
            finally:
                tmp.unlink(missing_ok=True)
        if path.stat().st_size != len(data):
            raise RuntimeError(f"asset {path} exists with a different size; the library is corrupt")
        self._known[sha] = len(data)
        return ref

    def path(self, ref: dict) -> Path:
        return self.root / f"{ref['sha256']}.{ref['format']}"

    def relative_to(self, directory) -> str:
        """The library path relative to ``directory`` (stored in each episode)."""
        return os.path.relpath(self.root.resolve(), Path(directory).resolve())


_OPEN: dict[str, AssetLibrary] = {}


def open_library(root) -> AssetLibrary:
    """The process-wide writer of library directory ``root`` (one per directory, so its digest
    set deduplicates across all episodes of a build job)."""
    key = str(Path(root).resolve())
    if key not in _OPEN:
        _OPEN[key] = AssetLibrary(root)
    return _OPEN[key]
