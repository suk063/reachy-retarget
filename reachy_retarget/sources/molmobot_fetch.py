"""Explicit acquisition of MolmoBot-Data scene packages without storing images.

MolmoBot-Data (``allenai/molmobot-data``) stores one zstd-compressed tar per scene
("package") as a byte range inside large shard tars. Every package holds the
trajectory HDF5 files (``trajectories_batch_<i>_of_<n>.h5``, state, actions and
metadata only) next to MP4 camera videos. The generic :func:`reachy_retarget.acquire.fetch`
works on whole files and would have to store the videos, so this module

* streams the pinned byte range of a package over HTTP (``Range`` request),
* hashes the compressed range on the fly and compares it with ``range_sha256``,
* decompresses the stream (``zstandard`` if installed, else the ``zstd`` CLI),
* writes only ``.h5`` members (each checked against its catalogued SHA-256) to
  ``<root>/raw/molmobot/<config>/<split>/part<k>/<house>/``; videos are discarded
  in memory and never touch the disk,
* records each written file in the shared ledger (``<root>/raw/ledger.json``).

The package table lives in ``reachy_retarget/acquire/catalog/molmobot.yaml`` under
``molmobot_packages``. Its ``range_sha256`` and member digests were computed by the
catalog generator (the publisher only hashes whole shards); the shard's publisher
digest is kept as ``shard_sha256`` for provenance.

:func:`unpack_robot_assets` turns a fetched MolmoSpaces robot shard (whole file,
publisher-hashed, fetched with :func:`acquire.fetch`) into a plain directory the
adapter can read.

Nothing here touches the network on import.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import threading
import urllib.request
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from ..acquire.fetch import RESERVE_BYTES, USER_AGENT, CatalogEntry, ChecksumMismatch, InsufficientDisk, _record

CHUNK = 1 << 20


@dataclass(frozen=True)
class PackageEntry:
    id: str                    # "molmobot/<config>/<split>/part<k>/<house>"
    config: str
    split: str
    part: int
    package: str               # publisher package name, e.g. "<config>_house_123.tar.zst"
    shard: str                 # "<config>/<split>_shards/<nnnnn>.tar"
    url: str                   # shard URL at the pinned revision
    shard_sha256: str
    shard_size: int
    offset: int
    size: int                  # compressed bytes of the package
    inflated_size: int
    range_sha256: str
    scene_id: str
    scene_family: str
    commercial_valid_episodes: str
    members: tuple = field(default_factory=tuple)  # dicts: name, sha256, size, trajectories, ...
    revision: str = ""
    license: str = ""

    def member_path(self, root, name: str) -> Path:
        house = Path(name).parts[0]
        return Path(root) / "raw" / "molmobot" / self.config / self.split / f"part{self.part}" / house / Path(name).name

    def member_entry(self, root, m: dict) -> CatalogEntry:
        local = self.member_path(root, m["name"]).relative_to(Path(root) / "raw" / "molmobot")
        return CatalogEntry(
            id=f"molmobot/{local.as_posix()}", family="molmobot", path=local.as_posix(), url=self.url,
            revision=self.revision, sha256=m["sha256"], size=m["size"], license=self.license, kind="low_dim",
            dataset=f"molmobot/{self.config}/{self.split}",
            note=(f"member {m['name']} of package {self.package} = bytes [{self.offset}, {self.offset + self.size}) "
                  f"of {self.shard} (range sha256 {self.range_sha256}; shard sha256 {self.shard_sha256})"))


def load_packages(path=None) -> dict[str, PackageEntry]:
    """Parse the ``molmobot_packages`` section of the MolmoBot catalog file."""
    text = Path(path).read_text() if path else \
        resources.files("reachy_retarget.acquire").joinpath("catalog/molmobot.yaml").read_text()
    sec = yaml.safe_load(text)["molmobot_packages"]
    out = {}
    for p in sec["packages"]:
        e = PackageEntry(**{**p, "members": tuple(p["members"])}, revision=sec["revision"], license=sec["license"])
        if e.id in out:
            raise ValueError(f"duplicate package id {e.id}")
        out[e.id] = e
    return out


def find_package(path, packages=None) -> tuple[PackageEntry | None, dict | None]:
    """Content-addressed lookup of a local member file: ``(package, member)``."""
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for pkg in (packages or load_packages()).values():
        for m in pkg.members:
            if m["sha256"] == h:
                return pkg, m
    return None, None


# ---------------------------------------------------------------- decompression

def _zstd_reader(chunks):
    """A readable binary stream of the decompressed ``chunks`` and a finaliser."""
    try:
        import zstandard  # optional dependency
    except ImportError:
        zstandard = None
    if zstandard is not None:
        class _Iter(io.RawIOBase):
            def __init__(self, it):
                self.it, self.buf = it, b""

            def readable(self):
                return True

            def readinto(self, b):
                while not self.buf:
                    try:
                        self.buf = next(self.it)
                    except StopIteration:
                        return 0
                n = min(len(b), len(self.buf))
                b[:n], self.buf = self.buf[:n], self.buf[n:]
                return n

        reader = zstandard.ZstdDecompressor().stream_reader(io.BufferedReader(_Iter(iter(chunks)), CHUNK))
        return reader, lambda abort=False: None
    exe = shutil.which("zstd")
    if exe is None:
        raise RuntimeError("MolmoBot packages are zstd-compressed: install the 'zstandard' package or the zstd CLI")
    proc = subprocess.Popen([exe, "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []

    def feed():
        try:
            for c in chunks:
                proc.stdin.write(c)
        except BaseException as exc:  # surfaced by finish()
            err.append(exc)
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    th = threading.Thread(target=feed, daemon=True)
    th.start()

    def finish(abort=False):
        if abort:
            proc.kill()
            proc.stdout.close()
        th.join()
        rc = proc.wait()
        if abort:  # report the feeder's error (e.g. a refused range) rather than the tar EOF it caused
            return err[0] if err else None
        if err:
            raise err[0]
        if rc != 0:
            raise RuntimeError(f"zstd failed: {proc.stderr.read().decode(errors='replace')}")
    return proc.stdout, finish


# ---------------------------------------------------------------- packages

def _range_chunks(e: PackageEntry, open_url, digest, counter):
    req = urllib.request.Request(e.url, headers={"User-Agent": USER_AGENT,
                                                 "Range": f"bytes={e.offset}-{e.offset + e.size - 1}"})
    with open_url(req) as resp:
        if getattr(resp, "status", 206) != 206:
            raise RuntimeError(f"{e.id}: server ignored the byte range (status {resp.status}); refusing to stream a whole shard")
        for block in iter(lambda: resp.read(CHUNK), b""):
            digest.update(block)
            counter[0] += len(block)
            yield block


def fetch_packages(ids, root, *, packages=None, reserve=RESERVE_BYTES, opener=None) -> list[dict]:
    """Fetch the ``.h5`` members of catalogued packages (ids or :class:`PackageEntry`).

    Returns ledger records of the member files. Existing members whose SHA-256 matches
    are only re-recorded. A package is refused before transfer when its members would
    leave less than ``reserve`` bytes free.
    """
    packages = packages or load_packages()
    entries = [packages[i] if isinstance(i, str) else i for i in ids]
    open_url = opener or urllib.request.urlopen
    root = Path(root)
    out = []
    for e in entries:
        want = {m["name"]: m for m in e.members}
        done = {}
        for name, m in want.items():
            p = e.member_path(root, name)
            if p.exists():
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                if h != m["sha256"]:
                    raise ChecksumMismatch(f"existing {p} has sha256 {h}, expected {m['sha256']}")
                done[name] = p
        if len(done) < len(want):
            dest = e.member_path(root, next(iter(want)))
            dest.parent.mkdir(parents=True, exist_ok=True)
            need = sum(m["size"] for n, m in want.items() if n not in done)
            if shutil.disk_usage(dest.parent).free - need < reserve:
                raise InsufficientDisk(f"{e.id}: {need} bytes would leave < {reserve} bytes free")
            digest, counter, parts = hashlib.sha256(), [0], {}
            stream, finish = _zstd_reader(_range_chunks(e, open_url, digest, counter))
            try:
                with tarfile.open(fileobj=stream, mode="r|") as tar:
                    for mem in tar:  # videos are skipped without being read into a file
                        if not (mem.isfile() and mem.name in want and mem.name not in done):
                            continue
                        part = e.member_path(root, mem.name).with_suffix(".h5.part")
                        part.parent.mkdir(parents=True, exist_ok=True)
                        h = hashlib.sha256()
                        with tar.extractfile(mem) as src, open(part, "wb") as dst:
                            for block in iter(lambda: src.read(CHUNK), b""):
                                h.update(block)
                                dst.write(block)
                        if h.hexdigest() != want[mem.name]["sha256"]:
                            bad = part.with_suffix(".sha256-mismatch")
                            os.replace(part, bad)
                            raise ChecksumMismatch(f"{e.id}:{mem.name}: sha256 {h.hexdigest()} != {want[mem.name]['sha256']}")
                        parts[mem.name] = part
                    for _ in tar:  # drain to the end of the archive
                        pass
                stream.read()  # zero padding after the tar end marker
            except BaseException as exc:
                cause = finish(abort=True)
                for p in parts.values():
                    p.unlink(missing_ok=True)
                if cause is not None and cause is not exc:
                    raise cause from exc
                raise
            finish()
            if counter[0] != e.size or digest.hexdigest() != e.range_sha256:
                for p in parts.values():
                    p.unlink(missing_ok=True)
                raise ChecksumMismatch(f"{e.id}: range sha256 {digest.hexdigest()} ({counter[0]} bytes) "
                                       f"!= {e.range_sha256} ({e.size} bytes)")
            missing = set(want) - set(done) - set(parts)
            if missing:
                raise ChecksumMismatch(f"{e.id}: members not found in the package: {sorted(missing)}")
            for name, part in parts.items():
                final = e.member_path(root, name)
                os.replace(part, final)
                done[name] = final
        for name, m in want.items():
            out.append(_record(root, e.member_entry(root, m), m["sha256"], m["size"], "catalog"))
    return out


# ---------------------------------------------------------------- robot assets

def robot_dir(root, name: str, version: str) -> Path:
    return Path(root) / "raw" / "molmobot" / "robots" / name / version


def unpack_robot_assets(root, entry: CatalogEntry) -> Path:
    """Unpack a fetched MolmoSpaces robot shard into ``raw/molmobot/robots/<name>/<version>/``.

    The shard is a plain tar holding ``<name>.tar.zst``; the inner tar is extracted
    (regular files only, no absolute or parent paths) and ``MANIFEST.json`` records the
    shard digest and every extracted file's SHA-256.
    """
    shard = entry.local_path(root)
    if not shard.exists():
        raise FileNotFoundError(f"{shard} not fetched; run acquire.fetch([{entry.id!r}], root) first")
    parts = Path(entry.path).parts  # molmospaces/mujoco/robots/<name>/<version>/shards/00000.tar
    name, version = parts[3], parts[4]
    dest = robot_dir(root, name, version)
    if (dest / "MANIFEST.json").exists():
        return dest
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files = {}
    with tarfile.open(shard, "r:") as outer:
        inner = outer.getmember(f"{name}.tar.zst")
        src = outer.extractfile(inner)
        stream, finish = _zstd_reader(iter(lambda: src.read(CHUNK), b""))
        try:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for mem in tar:
                    rel = Path(mem.name)
                    if not mem.isfile() or rel.is_absolute() or ".." in rel.parts:
                        continue
                    data = tar.extractfile(mem).read()
                    (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                    (tmp / rel).write_bytes(data)
                    files[rel.as_posix()] = hashlib.sha256(data).hexdigest()
                for _ in tar:
                    pass
            stream.read()
        except BaseException as exc:
            cause = finish(abort=True)
            shutil.rmtree(tmp, ignore_errors=True)
            if cause is not None and cause is not exc:
                raise cause from exc
            raise
        finish()
    (tmp / "MANIFEST.json").write_text(json.dumps(
        {"catalog_id": entry.id, "url": entry.url, "revision": entry.revision, "shard_sha256": entry.sha256,
         "license": entry.license, "files": files}, indent=1, sort_keys=True))
    os.replace(tmp, dest)
    return dest


__all__ = ["PackageEntry", "load_packages", "find_package", "fetch_packages", "unpack_robot_assets", "robot_dir"]
