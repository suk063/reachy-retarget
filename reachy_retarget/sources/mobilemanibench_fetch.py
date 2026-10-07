"""Explicit acquisition of MobileManiBench state files without storing images.

MobileManiDataset (``arnoldland/MobileManiBench`` on the Hugging Face Hub) ships one
uncompressed tar per robot / skill / object group (``MobileManiDataset/<robot>/<Open|Close>/
<category>/<group>.tar``, 2-46 GB each). Every episode folder in a tar holds six MP4 camera
videos and one ``state_infos.pkl`` (robot, object and action state). The generic
:func:`reachy_retarget.acquire.fetch` works on whole files and would have to store the videos,
so this module

* reads single tar members by HTTP ``Range`` (the tars are not compressed, so a member is one
  contiguous byte range: ``offset`` = first data byte, ``size`` = member size),
* reads single members of ``Assets/Assets.zip`` (the robot URDF) by ``Range`` from the zip
  local header, inflating ``deflate`` members and checking the zip CRC-32,
* checks each member against the SHA-256 recorded by the catalog generator (the publisher only
  hashes whole tars, in ``MobileManiDataset/object_manifest.jsonl``; that digest is kept as
  ``archive_sha256`` for provenance),
* writes the member to ``<root>/raw/mobilemanibench/<member name>`` and records it in the shared
  per-file ledger (``<root>/raw/ledger/mobilemanibench/...``).

Videos are never requested. The member table lives in
``reachy_retarget/acquire/catalog/mobilemanibench.yaml`` under ``mobilemanibench_members``.
:func:`generate_members` (network; used to build that table) walks tar headers with small
range requests and hashes the selected members.

Nothing here touches the network on import.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import struct
import tarfile
import urllib.request
import zlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

from ..acquire.fetch import RESERVE_BYTES, USER_AGENT, CatalogEntry, ChecksumMismatch, InsufficientDisk, _record

FAMILY = "mobilemanibench"
REPO = "https://huggingface.co/datasets/arnoldland/MobileManiBench"


@dataclass(frozen=True)
class MemberEntry:
    id: str                    # "mobilemanibench/<member>"
    path: str                  # member name inside the archive (tar: original tree path), reused locally
    archive: str               # publisher path of the tar / zip
    url: str                   # archive URL at the pinned revision
    archive_sha256: str        # publisher digest of the whole archive (object_manifest.jsonl / HF LFS oid)
    archive_size: int
    offset: int                # first stored byte of the member (tar: data; zip: local file header)
    stored_size: int           # bytes stored in the archive (compressed size for deflate members)
    size: int                  # bytes of the extracted member
    sha256: str                # SHA-256 of the extracted member (computed by the catalog generator)
    compression: str = "none"  # "none" (tar member) or "deflate" (zip member)
    crc32: int | None = None   # zip CRC-32 of the extracted member
    kind: str = "low_dim"
    dataset: str | None = None
    note: str | None = None
    revision: str = ""
    license: str = ""

    def local_path(self, root) -> Path:
        return Path(root) / "raw" / FAMILY / self.path

    def catalog_entry(self) -> CatalogEntry:
        where = (f"bytes [{self.offset}, {self.offset + self.stored_size}) of {self.archive}" if self.compression == "none"
                 else f"zip member at local header offset {self.offset} ({self.stored_size} deflated bytes) of {self.archive}")
        return CatalogEntry(id=self.id, family=FAMILY, path=self.path, url=self.url, revision=self.revision,
                            sha256=self.sha256, size=self.size, license=self.license, kind=self.kind,
                            dataset=self.dataset,
                            note=f"{where} (archive sha256 {self.archive_sha256})" + (f"; {self.note}" if self.note else ""))


def _catalog_text(path=None) -> str:
    if path:
        return Path(path).read_text()
    return resources.files("reachy_retarget.acquire").joinpath("catalog/mobilemanibench.yaml").read_text()


def load_members(path=None) -> dict[str, MemberEntry]:
    """Parse the ``mobilemanibench_members`` section of the catalog file."""
    sec = yaml.safe_load(_catalog_text(path))["mobilemanibench_members"]
    archives = {a["archive"]: a for a in sec["archives"]}
    out = {}
    for m in sec["members"]:
        a = archives[m["archive"]]
        e = MemberEntry(id=f"{FAMILY}/{m['path']}", url=a["url"], archive_sha256=a["sha256"], archive_size=a["size"],
                        revision=sec["revision"], license=a.get("license", sec["license"]), **m)
        if e.id in out:
            raise ValueError(f"duplicate member id {e.id}")
        out[e.id] = e
    return out


def find_member(path, members=None) -> MemberEntry | None:
    """Content-addressed lookup of a local member file."""
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return next((m for m in (members or load_members()).values() if m.sha256 == h), None)


def _get_range(url, offset, size, open_url) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={offset}-{offset + size - 1}"})
    with open_url(req) as resp:
        if getattr(resp, "status", 206) != 206:
            raise RuntimeError(f"{url}: server ignored the byte range (status {resp.status}); refusing to read a whole archive")
        data = resp.read()
    if len(data) != size:
        raise ChecksumMismatch(f"{url}: got {len(data)} bytes for range [{offset}, {offset + size}), expected {size}")
    return data


def _extract(e: MemberEntry, open_url) -> bytes:
    if e.compression == "none":
        return _get_range(e.url, e.offset, e.stored_size, open_url)
    if e.compression != "deflate":
        raise ValueError(f"{e.id}: unsupported compression {e.compression!r}")
    head = _get_range(e.url, e.offset, 30, open_url)
    if head[:4] != b"PK\x03\x04":
        raise ChecksumMismatch(f"{e.id}: no zip local header at offset {e.offset}")
    nl, el = struct.unpack("<HH", head[26:30])
    raw = _get_range(e.url, e.offset + 30 + nl + el, e.stored_size, open_url)
    data = zlib.decompressobj(-15).decompress(raw)
    if e.crc32 is not None and (zlib.crc32(data) & 0xFFFFFFFF) != e.crc32:
        raise ChecksumMismatch(f"{e.id}: zip CRC-32 mismatch")
    return data


def fetch_members(ids, root, *, members=None, reserve=RESERVE_BYTES, opener=None) -> list[dict]:
    """Fetch catalogued members (ids or :class:`MemberEntry`) into ``<root>/raw/mobilemanibench``.

    Returns ledger records. Existing files whose SHA-256 matches are only re-recorded. The
    transfer is refused when the members would leave less than ``reserve`` bytes free.
    """
    members = members or load_members()
    entries = [members[i] if isinstance(i, str) else i for i in ids]
    open_url = opener or urllib.request.urlopen
    root = Path(root)
    need = sum(e.size for e in entries if not e.local_path(root).exists())
    if entries and need:
        probe = root if root.exists() else Path(os.path.abspath(os.sep))
        if shutil.disk_usage(probe).free - need < reserve:
            raise InsufficientDisk(f"{need} bytes would leave < {reserve} bytes free")
    out = []
    for e in entries:
        dest = e.local_path(root)
        if dest.exists():
            h = hashlib.sha256(dest.read_bytes()).hexdigest()
            if h != e.sha256:
                raise ChecksumMismatch(f"existing {dest} has sha256 {h}, expected {e.sha256}")
        else:
            data = _extract(e, open_url)
            h = hashlib.sha256(data).hexdigest()
            if len(data) != e.size or h != e.sha256:
                raise ChecksumMismatch(f"{e.id}: sha256 {h} ({len(data)} bytes) != {e.sha256} ({e.size} bytes)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            part.write_bytes(data)
            os.replace(part, dest)
        out.append(_record(root, e.catalog_entry(), e.sha256, e.size, "catalog"))
    return out


# ---------------------------------------------------------------- catalog generation (network)

IMAGE_SUFFIXES = (".mp4", ".npz", ".png", ".jpg")


def walk_tar(url, *, start=0, stop=None, open_url=None, max_members=None):
    """Yield ``(data_offset, size, name)`` of tar members by reading headers over HTTP Range.

    Only 512-byte headers (plus GNU long-name / pax records) are transferred. ``stop(name)``
    ends the walk when it returns true.
    """
    open_url = open_url or urllib.request.urlopen
    off, n = start, 0
    while max_members is None or n < max_members:
        h = _get_range(url, off, 512, open_url)
        if h == b"\0" * 512:
            return
        ti = tarfile.TarInfo.frombuf(h, "utf-8", "surrogateescape")
        hdr, name, size = 512, ti.name, ti.size
        if ti.type in (tarfile.GNUTYPE_LONGNAME, tarfile.XHDTYPE):
            nblk = (ti.size + 511) // 512
            ext = _get_range(url, off + 512, nblk * 512 + 512, open_url)
            real = tarfile.TarInfo.frombuf(ext[nblk * 512:], "utf-8", "surrogateescape")
            name, size = real.name, real.size
            if ti.type == tarfile.GNUTYPE_LONGNAME:
                name = ext[:ti.size].rstrip(b"\0").decode("utf-8", "surrogateescape")
            else:
                for line in ext[:ti.size].decode("utf-8", "surrogateescape").split("\n"):
                    if " path=" in line:
                        name = line.split(" path=", 1)[1]
                    elif " size=" in line:
                        size = int(line.split(" size=", 1)[1])
            hdr = 512 + nblk * 512 + 512
            ti = real
        if ti.isfile() or ti.type == tarfile.AREGTYPE:
            yield off + hdr, size, name
            n += 1
        if stop is not None and stop(name):
            return
        off += hdr + ((size + 511) // 512) * 512


def generate_members(selections, *, revision, open_url=None) -> list[dict]:
    """Member table rows for ``selections``: dicts ``{archive, want(name) -> kind|None, stop(name) -> bool}``.

    Image members are never selected. Every selected member is downloaded once to hash it.
    """
    open_url = open_url or urllib.request.urlopen
    rows = []
    for sel in selections:
        url = f"{REPO}/resolve/{revision}/{sel['archive']}"
        for offset, size, name in walk_tar(url, stop=sel["stop"], open_url=open_url):
            kind = sel["want"](name)
            if kind is None or name.endswith(IMAGE_SUFFIXES):
                continue
            data = _get_range(url, offset, size, open_url)
            rows.append({"path": name, "archive": sel["archive"], "offset": offset, "stored_size": size, "size": size,
                         "sha256": hashlib.sha256(data).hexdigest(), "compression": "none", "kind": kind,
                         **({"dataset": sel["dataset"]} if sel.get("dataset") else {})})
    return rows


__all__ = ["MemberEntry", "load_members", "find_member", "fetch_members", "walk_tar", "generate_members"]
