"""Explicit, verified acquisition of catalogued sources: one entry point for every kind.

Nothing here touches the network on import; :func:`fetch` is the only function that
transfers data. Files land in ``<root>/raw/<family>/...`` and every verified file gets its
own ledger record (:mod:`.ledger`). Rules shared by all kinds (see :mod:`.catalog`):

* **Reserve**: a transfer is refused when it would leave less than :data:`RESERVE_BYTES`
  free on the target filesystem, and aborted (partial data kept or removed, never
  recorded) when free space drops below the reserve while streaming.
* **Verification**: publisher digests are checked whenever they exist (SHA-256, Box SHA-1,
  git blob SHA-1, zip CRC-32), plus sizes, tar headers and decompressed sizes. Without a
  publisher digest the SHA-256 is recorded trust-on-first-use (``sha256_source: tofu``);
  ``verified_by`` in the record lists the checks that passed.
* **No images**: image/video members of tars and packages are discarded in memory, a
  ``file``/``range_member`` entry whose name is an image or video is refused, and
  ``file_images_embedded`` entries are refused unless ``strip_images=True`` (then only the
  state-only ``<stem>.state.hdf5`` copy is kept, see :mod:`.strip`).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .catalog import CatalogEntry, load_catalog, select
from .ledger import read_ledger_record, record
from .strip import read_strip_record, strip_images as _strip, stripped_path
from .transports import (CHUNK, KeepAliveOpener, ResumingStream, check_tar_header, is_image_name, range_chunks,
                         read_range, zstd_stream)

RESERVE_BYTES = 50_000_000_000  # 50 decimal GB
TAR_MANIFEST = "tar_members.json"
PACKAGE_MANIFEST = "package_members.json"


class InsufficientDisk(RuntimeError):
    pass


class ChecksumMismatch(RuntimeError):
    pass


class ImageRefused(PermissionError):
    pass


def _free(path) -> int:
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _ensure_space(where, need: int, reserve: int, what: str) -> None:
    if need and _free(where) - need < reserve:
        raise InsufficientDisk(f"{what}: {need} more bytes would leave < {reserve} bytes free")


def _hash_file(path, git_blob: bool = False) -> dict:
    """SHA-256, SHA-1 and (optionally) git blob SHA-1 of a local file in one pass."""
    size = Path(path).stat().st_size
    h256, h1, hg = hashlib.sha256(), hashlib.sha1(), hashlib.sha1(b"blob %d\0" % size) if git_blob else None
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h256.update(block)
            h1.update(block)
            if hg is not None:
                hg.update(block)
    out = {"sha256": h256.hexdigest(), "sha1": h1.hexdigest(), "size": size}
    if hg is not None:
        out["git_blob_sha1"] = hg.hexdigest()
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _verify_digests(e: CatalogEntry, got: dict, where) -> list[str]:
    """Compare ``got`` with every pinned digest of ``e``; return the checks that passed."""
    checks = []
    if e.size is not None:
        if got["size"] != e.size:
            raise ChecksumMismatch(f"{e.id}: {got['size']} bytes, expected {e.size} ({where})")
        checks.append("size")
    pinned = {"sha256": e.sha256, **e.digests}
    for k, v in pinned.items():
        if v is None or k not in got:
            continue
        if got[k] != v:
            raise ChecksumMismatch(f"{e.id}: {k} {got[k]} != {v} ({where})")
        checks.append(k)
    return checks


def _source(e: CatalogEntry) -> str:
    return "catalog" if e.sha256 else "tofu"


# ---------------------------------------------------------------- kind: file / file_images_embedded

def _download(e: CatalogEntry, part: Path, reserve: int, opener) -> None:
    """Stream ``e.url`` into ``part``, appending to the bytes already there when the server
    honours ``Range`` (otherwise the file restarts)."""
    have = part.stat().st_size if part.exists() else 0
    stream = ResumingStream(e.url, opener, start=have)
    try:
        with open(part, "ab" if have and not stream.restarted else "wb") as f:
            buf = bytearray(CHUNK)
            i = 0
            while n := stream.readinto(buf):
                f.write(memoryview(buf)[:n])
                i += 1
                if i % 64 == 0 and _free(part.parent) < reserve:
                    raise InsufficientDisk(f"{e.id}: free space fell below reserve; partial kept at {part}")
    finally:
        stream.close()


def _fetch_file(e: CatalogEntry, root: Path, reserve, opener, strip: bool) -> list[dict]:
    dest = e.local_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    git = "git_blob_sha1" in e.digests
    if strip and stripped_path(dest).exists() and not dest.exists():
        rec = read_ledger_record(root, e.id)
        have = sha256_file(stripped_path(dest))
        if not rec or rec.get("stripped", {}).get("sha256") != have:
            raise ChecksumMismatch(f"{stripped_path(dest)} (sha256 {have}) is not the recorded "
                                   f"stripped copy of {e.id}")
        return [rec]
    if dest.exists():
        got = _hash_file(dest, git)
        checks = _verify_digests(e, got, f"existing {dest}")
    else:
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        if e.size is not None:
            _ensure_space(dest.parent, e.size - have, reserve, e.id)
        if e.size is None or have < e.size:
            _download(e, part, reserve, opener)
        got = _hash_file(part, git)
        try:
            checks = _verify_digests(e, got, "download")
        except ChecksumMismatch:
            if e.size is None or got["size"] == e.size:  # complete but wrong: keep for inspection
                os.replace(part, part.with_name(dest.name + ".sha256-mismatch"))
            raise
        os.replace(part, dest)
    if strip:
        return [_strip_and_record(root, e, dest, got["sha256"], got["size"], checks)]
    return [record(root, e, got["sha256"], got["size"], _source(e), verified_by=checks)]


def _strip_and_record(root, e: CatalogEntry, dest: Path, digest: str, size: int, checks) -> dict:
    """Write ``<stem>.state.hdf5`` from the verified ``dest``, record both digests, delete ``dest``."""
    out = stripped_path(dest)
    summary = _strip(dest, out, original={"catalog_id": e.id, "url": e.url, "revision": e.revision,
                                          "sha256": digest, "size": size, "license": e.license})
    info = {"path": str(out), "sha256": sha256_file(out), "bytes": out.stat().st_size,
            "original_deleted": True, "original_bytes": size,
            **{k: summary[k] for k in ("dropped_datasets", "dropped_names", "dropped_stored_bytes",
                                       "kept_datasets", "rule")}}
    rec = record(root, e, digest, size, _source(e), stripped=info, verified_by=checks)
    dest.unlink()
    return rec


# ---------------------------------------------------------------- kind: tar_stream

def _fetch_tar_stream(e: CatalogEntry, root: Path, reserve, opener, max_episodes=None) -> list[dict]:
    """Stream the whole tar once, keep non-image members, verify size + SHA-1/SHA-256."""
    dest = e.output_dir(root)
    manifest = dest / TAR_MANIFEST
    sha1 = e.digests.get("sha1")
    if manifest.exists():
        man = json.loads(manifest.read_text())
        if man.get("tar_sha1_verified") and man.get("box_sha1", man.get("sha1")) == sha1 \
                and (e.sha256 is None or man.get("tar_sha256") == e.sha256):
            return [_tar_record(root, e, man, manifest)]
    dest.mkdir(parents=True, exist_ok=True)
    _ensure_space(dest, e.size or 0, reserve, e.id)  # non-image members are at most the tar
    stream = ResumingStream(e.url, opener)
    kept, dropped, episodes, written = [], {"members": 0, "bytes": 0}, [], []
    try:
        with tarfile.open(fileobj=io.BufferedReader(stream, CHUNK), mode="r|") as tar:
            for mem in tar:
                if not mem.isfile():
                    continue
                if is_image_name(mem.name):
                    dropped["members"] += 1
                    dropped["bytes"] += mem.size
                    continue  # stream mode skips the member's data without storing it
                mt = re.search(r"/extras/(episode_\d+)/", mem.name)
                if mt and mt[1] not in episodes:
                    if max_episodes is not None and len(episodes) >= max_episodes:
                        continue
                    episodes.append(mt[1])
                _ensure_space(dest, mem.size, reserve, f"{e.id}:{mem.name}")
                p = dest / mem.name
                if not p.resolve().is_relative_to(dest.resolve()):
                    raise ValueError(f"{e.id}: unsafe member path {mem.name}")
                p.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with tar.extractfile(mem) as src, open(p, "wb") as dst:
                    for block in iter(lambda: src.read(CHUNK), b""):
                        h.update(block)
                        dst.write(block)
                written.append(p)
                kept.append({"member": mem.name, "tar_offset": mem.offset_data, "size": mem.size,
                             "sha256": h.hexdigest()})
        stream.drain()  # trailing zero blocks: the digest covers every byte
    except BaseException:
        for p in written:
            p.unlink(missing_ok=True)
        raise
    finally:
        stream.close()
    bad = (e.size is not None and stream.pos != e.size) or (sha1 and stream.sha1.hexdigest() != sha1) \
        or (e.sha256 and stream.sha256.hexdigest() != e.sha256)
    if bad or not (sha1 or e.sha256):
        for p in written:
            p.unlink(missing_ok=True)
        raise ChecksumMismatch(f"{e.id}: streamed {stream.pos} bytes sha1 {stream.sha1.hexdigest()}, expected "
                               f"{e.size} bytes sha1 {sha1} sha256 {e.sha256}; members removed")
    provenance = {k: e.meta[k] for k in ("shared_link", "box_file_id", "box_file_version") if k in e.meta}
    man = {"tar_id": e.id, "dataset": e.dataset, "url": e.url, **provenance,
           ("box_sha1" if "box_file_id" in e.meta else "sha1"): sha1,
           "tar_size": stream.pos, "tar_sha1_verified": True, "tar_sha256": stream.sha256.hexdigest(),
           "selection": "non-image members" + ("" if max_episodes is None else
                                               f"; extras of the first {max_episodes} episodes in tar order"),
           "extras_episodes": len(episodes), "dropped_image_members": dropped, "members": kept}
    tmp = manifest.with_name(f".{TAR_MANIFEST}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, manifest)
    return [_tar_record(root, e, man, manifest)]


def _tar_record(root, e: CatalogEntry, man: dict, manifest: Path) -> dict:
    sha1 = man.get("box_sha1", man.get("sha1"))
    info = {"path": str(manifest), "sha256": sha256_file(manifest),
            "bytes": sum(m["size"] for m in man["members"]), "original_deleted": True,
            "original_bytes": man["tar_size"], "dropped_datasets": man["dropped_image_members"]["members"],
            "dropped_names": ["videos/**"], "dropped_stored_bytes": man["dropped_image_members"]["bytes"],
            "kept_datasets": len(man["members"]),
            "rule": "tar members under videos/ or images/ or with an image/video extension are never written",
            ("box_sha1_verified" if "box_sha1" in man else "sha1_verified"): sha1}
    checks = ["size", "sha1"] + (["sha256"] if e.sha256 else [])
    return record(root, e, man["tar_sha256"], man["tar_size"], _source(e), stripped=info, verified_by=checks)


# ---------------------------------------------------------------- kind: range_member

def _member_bytes(e: CatalogEntry, opener) -> tuple[bytes, list[str]]:
    s = e.source
    comp = s.get("compression") or "none"
    offset, stored = s["offset"], s.get("stored_size") or e.size
    checks = []
    if comp == "none":
        if s.get("archive_format") == "tar" and offset >= 512:
            raw = read_range(e.url, offset - 512, stored + 512, opener)
            try:
                check_tar_header(raw[:512], s.get("member") or e.path, stored)
            except IOError as exc:
                raise ChecksumMismatch(f"{e.id}: {exc}") from exc
            data, checks = raw[512:], ["tar_header"]
        else:
            data = read_range(e.url, offset, stored, opener)
    elif comp == "deflate":  # zip member: offset = local file header
        head = read_range(e.url, offset, 30, opener)
        if head[:4] != b"PK\x03\x04":
            raise ChecksumMismatch(f"{e.id}: no zip local header at offset {offset}")
        nl, el = int.from_bytes(head[26:28], "little"), int.from_bytes(head[28:30], "little")
        data = zlib.decompressobj(-15).decompress(read_range(e.url, offset + 30 + nl + el, stored, opener))
    elif comp == "zstd":
        stream, finish = zstd_stream(range_chunks(e.url, offset, stored, opener))
        data = stream.read()
        finish()
    else:
        raise ValueError(f"{e.id}: unsupported compression {comp!r}")
    if s.get("crc32") is not None:
        if (zlib.crc32(data) & 0xFFFFFFFF) != s["crc32"]:
            raise ChecksumMismatch(f"{e.id}: CRC-32 mismatch")
        checks.append("crc32")
    return data, checks


def _fetch_range_member(e: CatalogEntry, root: Path, reserve, opener) -> list[dict]:
    dest = e.local_path(root)
    if dest.exists():
        got = _hash_file(dest)
        if e.sha256 is None:  # trust-on-first-use entries: the earlier record must match
            rec = read_ledger_record(root, e.id)
            if not rec or rec.get("sha256_verified") != got["sha256"]:
                raise ChecksumMismatch(f"existing {dest} has no matching ledger record; remove it to refetch")
        checks = _verify_digests(e, got, f"existing {dest}")
        return [record(root, e, got["sha256"], got["size"], _source(e), verified_by=checks)]
    _ensure_space(dest.parent, e.size or e.source.get("stored_size") or 0, reserve, e.id)
    data, checks = _member_bytes(e, opener)
    got = {"sha256": hashlib.sha256(data).hexdigest(), "sha1": hashlib.sha1(data).hexdigest(), "size": len(data)}
    checks += _verify_digests(e, got, "range")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(data)
    os.replace(part, dest)
    return [record(root, e, got["sha256"], got["size"], _source(e), verified_by=checks)]


# ---------------------------------------------------------------- kind: range_package

def _member_entry(e: CatalogEntry, m: dict) -> CatalogEntry:
    s = e.source
    base = os.path.basename(m["name"])
    return CatalogEntry(
        id=e.member_id(m["name"]), family=e.family, path=f"{e.path}/{base}", url=e.url, revision=e.revision,
        sha256=m["sha256"], size=m["size"], license=e.license, kind="range_package", content=e.content,
        dataset=e.dataset, meta={k: v for k, v in m.items() if k not in ("name", "sha256", "size")},
        note=(f"member {m['name']} of package {e.meta.get('package', e.id)} = bytes [{s['offset']}, "
              f"{s['offset'] + s['stored_size']}) of {s.get('archive')} (range sha256 "
              f"{s.get('range_sha256') or 'tofu'}; archive sha256 {s.get('archive_sha256')})"),
        source={"package": e.id, "member": m["name"]})


def _fetch_range_package(e: CatalogEntry, root: Path, reserve, opener) -> list[dict]:
    s = e.source
    keep = tuple(s.get("keep") or (".h5",))
    listed = {m["name"]: m for m in s.get("members") or []}
    dest = e.output_dir(root)
    manifest = dest / PACKAGE_MANIFEST
    if manifest.exists():
        man = json.loads(manifest.read_text())
        if all((dest / os.path.basename(m["name"])).exists() for m in man["members"]):
            for m in man["members"]:  # re-verify the files of a completed package
                p = dest / os.path.basename(m["name"])
                if sha256_file(p) != m["sha256"]:
                    raise ChecksumMismatch(f"existing {p} differs from {manifest}")
            return _package_records(root, e, man, manifest)
    done = {}
    for name, m in listed.items():  # members of an earlier fetch (catalogued digests)
        p = dest / os.path.basename(name)
        if p.exists():
            if sha256_file(p) != m["sha256"]:
                raise ChecksumMismatch(f"existing {p} has sha256 {sha256_file(p)}, expected {m['sha256']}")
            done[name] = {**m, "size": p.stat().st_size}
    if listed and len(done) == len(listed):  # complete from an earlier fetch: nothing to transfer
        man = _package_manifest(e, s.get("range_sha256"), None, done, {"members": 0, "bytes": 0}, keep)
        return _package_records(root, e, man, _write_json(manifest, man))
    dest.mkdir(parents=True, exist_ok=True)
    need = sum(m["size"] for n, m in listed.items() if n not in done) if listed else (s.get("inflated_size")
                                                                                       or s["stored_size"])
    _ensure_space(dest, need, reserve, e.id)
    digest, counter, parts, found, dropped = hashlib.sha256(), [0], {}, {}, {"members": 0, "bytes": 0}
    stream, finish = zstd_stream(range_chunks(e.url, s["offset"], s["stored_size"], opener, digest, counter))
    stream = _CountingReader(stream)
    member_bytes = 0
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for mem in tar:
                if not mem.isfile():
                    continue
                member_bytes += mem.size
                take = mem.name.endswith(keep) and not is_image_name(mem.name) and (not listed or mem.name in listed)
                if not take or mem.name in done:
                    if is_image_name(mem.name):
                        dropped["members"] += 1
                        dropped["bytes"] += mem.size
                    continue  # skipped in stream mode, never written
                part = dest / (os.path.basename(mem.name) + ".part")
                if os.path.basename(mem.name) in {os.path.basename(n) for n in found}:
                    raise ValueError(f"{e.id}: two kept members named {os.path.basename(mem.name)}")
                _ensure_space(dest, mem.size, reserve, f"{e.id}:{mem.name}")
                h = hashlib.sha256()
                with tar.extractfile(mem) as src, open(part, "wb") as dst:
                    for block in iter(lambda: src.read(CHUNK), b""):
                        h.update(block)
                        dst.write(block)
                parts[mem.name] = part
                if mem.name in listed and h.hexdigest() != listed[mem.name]["sha256"]:
                    raise ChecksumMismatch(f"{e.id}:{mem.name}: sha256 {h.hexdigest()} != {listed[mem.name]['sha256']}")
                found[mem.name] = {**listed.get(mem.name, {}), "name": mem.name, "sha256": h.hexdigest(),
                                   "size": mem.size}
            for _ in tar:  # drain to the end of the archive
                pass
        while stream.read(CHUNK):  # zero padding after the tar end marker
            pass
    except BaseException as exc:
        cause = finish(abort=True)
        for p in parts.values():
            p.unlink(missing_ok=True)
        if cause is not None and cause is not exc:
            raise cause from exc
        raise
    finish()
    problems = []
    if counter[0] != s["stored_size"]:
        problems.append(f"range {counter[0]} bytes != {s['stored_size']}")
    if s.get("range_sha256") and digest.hexdigest() != s["range_sha256"]:
        problems.append(f"range sha256 {digest.hexdigest()} != {s['range_sha256']}")
    # The publisher's inflated_size is either the summed member sizes or the whole tar stream
    # (headers and block padding included); both forms occur.
    if s.get("inflated_size") is not None and s["inflated_size"] not in (member_bytes, stream.count):
        problems.append(f"inflated to {stream.count} bytes ({member_bytes} in members) != {s['inflated_size']}")
    missing = set(listed) - set(done) - set(found)
    if missing:
        problems.append(f"members not found in the package: {sorted(missing)}")
    if problems:
        for p in parts.values():
            p.unlink(missing_ok=True)
        raise ChecksumMismatch(f"{e.id}: " + "; ".join(problems))
    for name, part in parts.items():
        os.replace(part, dest / os.path.basename(name))
    man = _package_manifest(e, digest.hexdigest(), member_bytes, {**done, **found}, dropped, keep)
    return _package_records(root, e, man, _write_json(manifest, man))


def _write_json(path: Path, doc: dict) -> Path:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return path


class _CountingReader:
    """File-like wrapper counting the bytes read through it."""

    def __init__(self, f):
        self.f, self.count = f, 0

    def read(self, n=-1):
        block = self.f.read(n)
        self.count += len(block)
        return block


def _package_manifest(e, range_sha256, inflated, members: dict, dropped, keep) -> dict:
    return {"package_id": e.id, "url": e.url, "archive": e.source.get("archive"),
            "archive_sha256": e.source.get("archive_sha256"), "offset": e.source["offset"],
            "stored_size": e.source["stored_size"], "range_sha256": range_sha256,
            "range_sha256_source": "catalog" if e.source.get("range_sha256") else "tofu",
            "member_bytes": inflated, "keep": list(keep), "dropped_image_members": dropped,
            "members": [members[k] for k in sorted(members)]}


def _package_records(root, e: CatalogEntry, man: dict, manifest: Path) -> list[dict]:
    checks = ["size", "tar_stream"] + (["range_sha256"] if e.source.get("range_sha256") else []) \
        + (["inflated_size"] if e.source.get("inflated_size") else [])
    listed = {m["name"] for m in e.source.get("members") or []}
    out = []
    for m in man["members"]:
        me = _member_entry(e, m)
        src = "catalog" if m["name"] in listed else "tofu"
        out.append(record(root, me, m["sha256"], m["size"], src,
                          verified_by=checks + (["sha256"] if src == "catalog" else [])))
    info = {"path": str(manifest), "sha256": sha256_file(manifest), "bytes": sum(m["size"] for m in man["members"]),
            "original_deleted": True, "original_bytes": e.source["stored_size"],
            "dropped_datasets": man["dropped_image_members"]["members"], "dropped_names": ["*.mp4"],
            "dropped_stored_bytes": man["dropped_image_members"]["bytes"], "kept_datasets": len(man["members"]),
            "rule": f"only members ending with {man['keep']} are written; image/video members never"}
    source = "catalog" if e.source.get("range_sha256") else "tofu"
    out.append(record(root, e, man["range_sha256"], e.source["stored_size"], source, stripped=info,
                      verified_by=checks, extra={"members": [_member_entry(e, m).id for m in man["members"]]}))
    return out


# ---------------------------------------------------------------- entry point

def _guard(e: CatalogEntry, strip_images: bool) -> None:
    kind = e.transport
    if kind == "file_images_embedded" and not strip_images:
        raise PermissionError(f"{e.id} embeds images; refusing (pass strip_images=True to keep a state-only copy)")
    if kind in ("file", "range_member", "file_images_embedded") and is_image_name(e.path):
        raise ImageRefused(f"{e.id}: image/video files are never fetched")


def fetch_one(e: CatalogEntry, root, *, reserve=RESERVE_BYTES, strip_images=False, opener=None,
              max_episodes=None) -> list[dict]:
    """Acquire one entry of any kind; returns the ledger records written."""
    _guard(e, strip_images)
    root = Path(root)
    opener = opener or KeepAliveOpener()
    kind = e.transport
    if kind in ("file", "file_images_embedded"):
        return _fetch_file(e, root, reserve, opener, strip=kind == "file_images_embedded")
    if kind == "tar_stream":
        return _fetch_tar_stream(e, root, reserve, opener, max_episodes)
    if kind == "range_member":
        return _fetch_range_member(e, root, reserve, opener)
    if kind == "range_package":
        return _fetch_range_package(e, root, reserve, opener)
    raise ValueError(f"{e.id}: unknown kind {e.kind!r}")


def fetch(ids, root, *, catalog=None, reserve=RESERVE_BYTES, strip_images=False, opener=None, workers=1,
          max_episodes=None, log=None) -> list[dict]:
    """Acquire catalog entries into ``root`` and return their ledger records.

    ``ids`` are catalog ids, prefixes ending in ``/`` (every usable entry below) or
    :class:`CatalogEntry` objects. Entries already present and verified are re-recorded
    without transfer. ``workers`` > 1 fetches entries concurrently (useful for many small
    ``range_member`` / ``file`` entries). ``max_episodes`` limits the ``extras/`` episodes
    kept from ``tar_stream`` entries. All kinds share the reserve, verification and image
    rules described in the module docstring.
    """
    if catalog is None and any(isinstance(i, str) for i in ids):
        catalog = load_catalog()
    entries = select(catalog or {}, list(ids))
    for e in entries:  # refuse before any transfer
        _guard(e, strip_images)
    opener = opener or KeepAliveOpener()

    def one(e):
        recs = fetch_one(e, root, reserve=reserve, strip_images=strip_images, opener=opener,
                         max_episodes=max_episodes)
        if log:
            log(e, recs)
        return recs

    if workers <= 1:
        return [r for e in entries for r in one(e)]
    with ThreadPoolExecutor(workers) as ex:
        return [r for recs in ex.map(one, entries) for r in recs]


# ---------------------------------------------------------------- local lookups

def identify(path, catalog=None) -> tuple[CatalogEntry | None, dict]:
    """Content-addressed lookup of a local source file.

    Returns ``(entry, info)``. For a stripped ``*.state.hdf5`` file the lookup uses the
    original SHA-256 recorded inside it; ``info`` then holds ``sha256`` (the publisher
    file's digest, which the catalog pins), ``file_sha256`` (the local file's digest) and
    ``stripped`` (the strip record, or ``None`` for an unmodified file).
    """
    catalog = catalog if catalog is not None else load_catalog()
    file_digest = sha256_file(path)
    rec = read_strip_record(path) if str(path).endswith((".hdf5", ".h5")) else None
    digest = rec["original"].get("sha256") if rec else file_digest
    entry = next((e for e in catalog.values() if e.sha256 and e.sha256 == digest), None)
    return entry, {"sha256": digest, "file_sha256": file_digest, "stripped": rec}


def find_entry(path, catalog=None) -> CatalogEntry | None:
    """The catalog entry of ``path`` (original or stripped copy), or ``None``."""
    return identify(path, catalog)[0]


def locate_entry(path, catalog=None, digest: str | None = None) -> CatalogEntry | None:
    """Catalog entry of a fetched file: by pinned SHA-256, else by its place under
    ``<root>/raw/<family>/`` when the ledger of that root records the same SHA-256 (entries
    pinned without a SHA-256 and members of range packages). Files whose content matches
    neither (e.g. test subsets placed at a catalogued path) give ``None``."""
    catalog = catalog if catalog is not None else load_catalog()
    digest = digest or sha256_file(path)
    hit = next((e for e in catalog.values() if e.sha256 and e.sha256 == digest), None)
    if hit is not None:
        return hit
    parts = Path(path).resolve().parts
    if "raw" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("raw")
    root, rel = Path(*parts[:i]), "/".join(parts[i + 1:])
    rec = read_ledger_record(root, rel)
    if rec is None or rec.get("sha256_verified") != digest:
        return None
    if rel in catalog:
        return catalog[rel] if catalog[rel].sha256 in (None, digest) else None
    pkg = catalog.get(rel.rsplit("/", 1)[0])
    if pkg is not None and pkg.transport == "range_package":
        name = (rec.get("source") or {}).get("member", parts[-1])
        m = next((m for m in pkg.source.get("members") or [] if m["name"] == name),
                 {"name": name, "sha256": digest, "size": rec.get("bytes")})
        return _member_entry(pkg, m)
    return None


__all__ = ["RESERVE_BYTES", "TAR_MANIFEST", "PACKAGE_MANIFEST", "ChecksumMismatch", "ImageRefused",
           "InsufficientDisk", "fetch", "fetch_one", "find_entry", "identify", "locate_entry", "sha256_file"]
