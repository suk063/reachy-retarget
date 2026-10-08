"""Subsets of catalogued asset zips, read member by member over HTTP ``Range``.

When a full asset archive does not fit (RoboCasa's object zips are ~11 GB), the members
an episode references can be copied out of the catalogued zips without downloading
them: the zip central directory and each member are read by byte range and every member
is checked against its zip CRC-32. ``<out>/manifest.json`` records, per member, the
source archive id and pinned digest and the member's CRC-32 and SHA-256; members found in
no archive are listed as ``missing``. Image textures are assets, not observations, so this
path is not subject to the image rule of :func:`.fetch`.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import zipfile
import zlib
from pathlib import Path

from .fetch import RESERVE_BYTES, ChecksumMismatch, _ensure_space
from .transports import KeepAliveOpener, RangeFile


def fetch_zip_subset(rels, out_dir, archives, *, reserve=RESERVE_BYTES, opener=None) -> dict:
    """Copy members ``rels`` into ``out_dir`` from ``archives`` = ``[(entry, member_of)]``,
    where ``member_of(rel)`` is the member name of ``rel`` in that zip (or ``None``)."""
    opener = opener or KeepAliveOpener()
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    mpath = base / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"members": {}, "missing": []}
    todo = sorted({posixpath.normpath(r) for r in rels} - set(man["members"]))
    for e, member_of in archives:
        zf = names = None
        for rel in list(todo):
            member = member_of(rel)
            if member is None:
                continue
            if zf is None:
                zf = zipfile.ZipFile(io.BufferedReader(RangeFile(e.url, opener, size=e.size), 1 << 16))
                names = {i.filename: i for i in zf.infolist()}
            info = names.get(member)
            if info is None:
                continue
            _ensure_space(base, info.file_size, reserve, rel)
            data = zf.read(info)
            if zlib.crc32(data) != info.CRC:
                raise ChecksumMismatch(f"{e.id}:{member}: CRC mismatch")
            p = base / rel
            if not p.resolve().is_relative_to(base.resolve()):
                raise ValueError(f"unsafe member path {rel}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            man["members"][rel] = {"archive_id": e.id, "archive_sha256": e.sha256, "archive_note": e.note,
                                   "member": member, "crc32": info.CRC, "size": info.file_size,
                                   "sha256": hashlib.sha256(data).hexdigest()}
            todo.remove(rel)
    man["missing"] = sorted(set(man.get("missing", [])) - set(man["members"]) | set(todo))
    tmp = mpath.with_name(f".manifest.json.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True))
    os.replace(tmp, mpath)
    return man


def fetch_robocasa_asset_subset(rels, root, *, catalog=None, reserve=RESERVE_BYTES, opener=None) -> dict:
    """RoboCasa asset members (paths relative to ``robocasa/models/assets/``) copied from the
    catalogued asset zips into ``<root>/raw/robocasa/asset_subset/``."""
    from ..sources.robocasa import SUBSET_DIR, asset_sources_from_catalog
    from .catalog import load_catalog

    catalog = catalog or load_catalog()
    archives = [(catalog[s["id"]], s["member"]) for s in asset_sources_from_catalog(catalog) if s["kind"] == "zip"]
    return fetch_zip_subset(rels, Path(root) / "raw" / "robocasa" / SUBSET_DIR, archives, reserve=reserve,
                            opener=opener)


__all__ = ["fetch_zip_subset", "fetch_robocasa_asset_subset"]
