"""Explicit, resumable, hash-verified downloads of catalogued source files.

Nothing here touches the network on import. :func:`fetch` is the only function that
transfers data. Files land in ``<root>/raw/<family>/<path>``; a JSON ledger at
``<root>/raw/ledger.json`` records every verified file. A transfer is refused when it
would leave less than :data:`RESERVE_BYTES` free on the target filesystem, and it is
aborted (keeping the partial file for resumption) if free space drops below the
reserve while streaming.
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


def find_entry(path, catalog=None) -> CatalogEntry | None:
    """Content-addressed lookup: the catalog entry whose digest matches ``path``."""
    digest = sha256_file(path)
    return next((e for e in (catalog or load_catalog()).values() if e.sha256 == digest), None)


def _ledger_path(root) -> Path:
    return Path(root) / "raw" / "ledger.json"


def read_ledger(root) -> dict:
    p = _ledger_path(root)
    return json.loads(p.read_text()) if p.exists() else {}


def _record(root, entry: CatalogEntry, digest: str, size: int, sha256_source: str) -> dict:
    ledger = read_ledger(root)
    rec = {**asdict(entry), "local_path": str(entry.local_path(root).relative_to(root)),
           "sha256_verified": digest, "sha256_source": sha256_source, "bytes": size,
           "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ledger[entry.id] = rec
    p = _ledger_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1, sort_keys=True))
    os.replace(tmp, p)
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


def fetch(ids, root, *, catalog=None, reserve=RESERVE_BYTES, allow_images=False, opener=None) -> list[dict]:
    """Download catalogue entries (ids or :class:`CatalogEntry`) into ``root``.

    Resumes ``*.part`` files with HTTP Range requests, verifies SHA-256 against the
    catalogue (or records a trust-on-first-use digest when the publisher declares none)
    and returns the ledger records. Files whose ``kind`` is ``images_embedded`` are
    refused unless ``allow_images`` is set, because this project never stores images.
    """
    catalog = catalog or load_catalog()
    entries = [catalog[i] if isinstance(i, str) else i for i in ids]
    open_url = opener or urllib.request.urlopen
    root = Path(root)
    out = []
    for e in entries:
        if e.kind == "images_embedded" and not allow_images:
            raise PermissionError(f"{e.id} embeds images; refusing (allow_images=False)")
        dest = e.local_path(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            digest = sha256_file(dest)
            if e.sha256 and digest != e.sha256:
                raise ChecksumMismatch(f"existing {dest} has sha256 {digest}, expected {e.sha256}")
            out.append(_record(root, e, digest, dest.stat().st_size, "catalog" if e.sha256 else "tofu"))
            continue
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
        out.append(_record(root, e, digest, size, "catalog" if e.sha256 else "tofu"))
    return out
