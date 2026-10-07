"""Explicit, resumable, hash-verified downloads of catalogued source files.

Nothing here touches the network on import. :func:`fetch` is the only function that
transfers data. Files land in ``<root>/raw/<family>/<path>``; every verified file gets
its own JSON ledger record at ``<root>/raw/ledger/<family>/<path>.json`` (one file per
record, so private output roots of concurrent jobs merge by plain no-overwrite copies;
:func:`read_ledger` aggregates them). A transfer is refused when it
would leave less than :data:`RESERVE_BYTES` free on the target filesystem, and it is
aborted (keeping the partial file for resumption) if free space drops below the
reserve while streaming.

Files of ``kind: images_embedded`` are refused unless ``strip_images=True``: the
verified download is then rewritten as a state-only ``<stem>.state.hdf5`` (see
:mod:`.strip`), the ledger records the original URL, SHA-256 and size together with the
stripped file's SHA-256, and the original is deleted.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import yaml

from .strip import read_strip_record, strip_images as _strip, stripped_path

RESERVE_BYTES = 50_000_000_000  # 50 decimal GB
CHUNK = 1 << 20
USER_AGENT = "reachy-retarget-fetch/2"


class InsufficientDisk(RuntimeError):
    pass


class ChecksumMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogEntry:
    id: str                    # "<family>/<path>"
    family: str                # top-level folder under raw/
    path: str                  # publisher-relative path, reused locally
    url: str                   # pinned download URL
    revision: str              # publisher revision (commit, tag or release version)
    sha256: str | None         # publisher-declared digest (HF LFS oid, PyPI digest)
    size: int | None           # bytes
    license: str
    kind: str                  # "low_dim" | "assets" | "images_embedded" ...
    dataset: str | None = None # logical dataset name, e.g. "robomimic/can/ph"
    note: str | None = None
    asset_marker: str | None = None  # kind "assets": recorded MJCF paths containing this
                                     # marker resolve to the archive members after it

    def local_path(self, root) -> Path:
        return Path(root) / "raw" / self.family / self.path


def load_catalog(path=None) -> dict[str, CatalogEntry]:
    """Parse a catalog file, or every packaged ``catalog/*.yaml``, into entries keyed by id."""
    if path:
        texts = [Path(path).read_text()]
    else:
        folder = resources.files(__package__).joinpath("catalog")
        texts = [f.read_text() for f in sorted(folder.iterdir(), key=lambda f: f.name) if f.name.endswith(".yaml")]
    out = {}
    for src in (s for text in texts for s in yaml.safe_load(text)["sources"]):
        shared = {k: src[k] for k in ("family", "revision", "license") if k in src}
        for f in src["files"]:
            e = CatalogEntry(id=f"{shared['family']}/{f['path']}", **{**shared, **f})
            if e.id in out:
                raise ValueError(f"duplicate catalog id {e.id}")
            out[e.id] = e
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


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


LEDGER_DIR = "ledger"
LEGACY_LEDGER = "ledger.json"  # single-file ledger of earlier versions; read-only fallback


def _ledger_dir(root) -> Path:
    return Path(root) / "raw" / LEDGER_DIR


def ledger_record_path(root, entry_id: str) -> Path:
    """``<root>/raw/ledger/<family>/<path>.json`` for catalog id ``<family>/<path>``."""
    return _ledger_dir(root) / f"{entry_id}.json"


def read_ledger_record(root, entry_id: str) -> dict | None:
    p = ledger_record_path(root, entry_id)
    if p.exists():
        return json.loads(p.read_text())
    legacy = Path(root) / "raw" / LEGACY_LEDGER
    return json.loads(legacy.read_text()).get(entry_id) if legacy.exists() else None


def read_ledger(root) -> dict:
    """All ledger records keyed by catalog id (per-file records override the legacy file)."""
    legacy = Path(root) / "raw" / LEGACY_LEDGER
    out = json.loads(legacy.read_text()) if legacy.exists() else {}
    folder = _ledger_dir(root)
    if folder.exists():
        for p in sorted(folder.rglob("*.json")):
            rec = json.loads(p.read_text())
            out[rec["id"]] = rec
    return out


def migrate_ledger(root) -> int:
    """Write per-file records for every entry of a legacy ``raw/ledger.json`` that has none
    yet. The legacy file is left in place. Returns the number of records written."""
    legacy = Path(root) / "raw" / LEGACY_LEDGER
    n = 0
    for eid, rec in (json.loads(legacy.read_text()) if legacy.exists() else {}).items():
        if not ledger_record_path(root, eid).exists():
            _write_record(root, rec)
            n += 1
    return n


def _write_record(root, rec: dict) -> None:
    p = ledger_record_path(root, rec["id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
    os.replace(tmp, p)


def _record(root, entry: CatalogEntry, digest: str, size: int, sha256_source: str,
            stripped: dict | None = None) -> dict:
    local = Path(stripped["path"]) if stripped else entry.local_path(root)
    rec = {**asdict(entry), "local_path": str(local.relative_to(root)),
           "sha256_verified": digest, "sha256_source": sha256_source, "bytes": size,
           "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if stripped:
        rec["stripped"] = {**stripped, "path": str(local.relative_to(root))}
    _write_record(root, rec)
    return rec


def _free(path) -> int:
    return shutil.disk_usage(path).free


def _transfer(e: CatalogEntry, part: Path, have: int, reserve: int, open_url) -> None:
    """Stream ``e.url`` into ``part``, appending from byte ``have`` when possible."""
    req = urllib.request.Request(e.url, headers={"User-Agent": USER_AGENT})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with open_url(req) as resp:
        resumed = have and getattr(resp, "status", 200) == 206  # else the server restarted
        with open(part, "ab" if resumed else "wb") as f:
            for i, block in enumerate(iter(lambda: resp.read(CHUNK), b"")):
                f.write(block)
                if i % 64 == 0 and _free(part.parent) < reserve:
                    raise InsufficientDisk(f"{e.id}: free space fell below reserve; partial kept at {part}")


def _strip_and_record(root, e: CatalogEntry, dest: Path, digest: str, size: int, source: str) -> dict:
    """Write ``<stem>.state.hdf5`` from the verified ``dest``, record both digests, delete ``dest``."""
    out = stripped_path(dest)
    summary = _strip(dest, out, original={"catalog_id": e.id, "url": e.url, "revision": e.revision,
                                          "sha256": digest, "size": size, "license": e.license})
    info = {"path": str(out), "sha256": sha256_file(out), "bytes": out.stat().st_size,
            "original_deleted": True, "original_bytes": size,
            **{k: summary[k] for k in ("dropped_datasets", "dropped_names", "dropped_stored_bytes",
                                       "kept_datasets", "rule")}}
    rec = _record(root, e, digest, size, source, stripped=info)
    dest.unlink()
    return rec


def fetch(ids, root, *, catalog=None, reserve=RESERVE_BYTES, strip_images=False, opener=None) -> list[dict]:
    """Download catalogue entries (ids or :class:`CatalogEntry`) into ``root``.

    Resumes ``*.part`` files with HTTP Range requests, verifies SHA-256 against the
    catalogue (or records a trust-on-first-use digest when the publisher declares none)
    and returns the ledger records. Files whose ``kind`` is ``images_embedded`` are
    refused unless ``strip_images`` is set, because this project never stores images;
    with it, each verified file is replaced by its state-only ``<stem>.state.hdf5``
    copy. An original that is already present locally is verified and stripped without
    any transfer.
    """
    catalog = catalog or load_catalog()
    entries = [catalog[i] if isinstance(i, str) else i for i in ids]
    open_url = opener or urllib.request.urlopen
    root = Path(root)
    out = []
    for e in entries:
        strip = e.kind == "images_embedded"
        if strip and not strip_images:
            raise PermissionError(f"{e.id} embeds images; refusing (pass strip_images=True to keep a "
                                  "state-only copy)")
        dest = e.local_path(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        source = "catalog" if e.sha256 else "tofu"
        if strip and stripped_path(dest).exists() and not dest.exists():
            rec = read_ledger_record(root, e.id)
            have = sha256_file(stripped_path(dest))
            if not rec or rec.get("stripped", {}).get("sha256") != have:
                raise ChecksumMismatch(f"{stripped_path(dest)} (sha256 {have}) is not the recorded "
                                       f"stripped copy of {e.id}")
            out.append(rec)
            continue
        if dest.exists():
            digest = sha256_file(dest)
            if e.sha256 and digest != e.sha256:
                raise ChecksumMismatch(f"existing {dest} has sha256 {digest}, expected {e.sha256}")
            size = dest.stat().st_size
        else:
            part = dest.with_name(dest.name + ".part")
            have = part.stat().st_size if part.exists() else 0
            if e.size is not None and _free(dest.parent) - (e.size - have) < reserve:
                raise InsufficientDisk(f"{e.id}: {e.size - have} more bytes would leave < {reserve} bytes free")
            if e.size is None or have < e.size:
                _transfer(e, part, have, reserve, open_url)
            size = part.stat().st_size
            if e.size is not None and size != e.size:
                raise ChecksumMismatch(f"{e.id}: got {size} bytes, expected {e.size}; partial kept at {part}")
            digest = sha256_file(part)
            if e.sha256 and digest != e.sha256:
                bad = part.with_name(dest.name + ".sha256-mismatch")
                os.replace(part, bad)
                raise ChecksumMismatch(f"{e.id}: sha256 {digest} != {e.sha256}; kept at {bad}")
            os.replace(part, dest)
        if strip:
            out.append(_strip_and_record(root, e, dest, digest, size, source))
        else:
            out.append(_record(root, e, digest, size, source))
    return out
