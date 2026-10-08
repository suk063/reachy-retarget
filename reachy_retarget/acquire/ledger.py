"""Per-file fetch ledger: ``<root>/raw/ledger/<family>/<path>.json``, one JSON record each.

One record per verified file lets concurrent jobs fetch into private roots that are
merged into a shared root by plain no-overwrite copies. Records written by every
earlier version stay readable: a record is a JSON object with at least ``id``,
``local_path`` (relative to the root) and ``sha256_verified``; when ``stripped`` is present
the local file is the stripped copy / extraction manifest and ``stripped.sha256`` is its
digest. The single-file ``raw/ledger.json`` of the first version is read as a fallback.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

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


def write_record(root, rec: dict) -> None:
    p = ledger_record_path(root, rec["id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}-{id(rec)}")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
    os.replace(tmp, p)


def migrate_ledger(root) -> int:
    """Write per-file records for every entry of a legacy ``raw/ledger.json`` that has none
    yet. The legacy file is left in place. Returns the number of records written."""
    legacy = Path(root) / "raw" / LEGACY_LEDGER
    n = 0
    for eid, rec in (json.loads(legacy.read_text()) if legacy.exists() else {}).items():
        if not ledger_record_path(root, eid).exists():
            write_record(root, rec)
            n += 1
    return n


def record(root, entry, digest: str, size: int, sha256_source: str, *, local: Path | None = None,
           stripped: dict | None = None, verified_by=None, extra: dict | None = None) -> dict:
    """Write and return the ledger record of a verified ``entry``.

    ``digest`` is the SHA-256 of the publisher bytes (for a stripped or extracted entry: of
    the whole original); ``sha256_source`` is ``catalog`` (pinned digest matched) or
    ``tofu`` (first-use digest); ``verified_by`` lists the checks that passed."""
    root = Path(root)
    local = Path(stripped["path"]) if stripped else (local or entry.local_path(root))
    rec = {**asdict(entry), "local_path": str(local.relative_to(root)), "sha256_verified": digest,
           "sha256_source": sha256_source, "bytes": size,
           "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for k in ("digests", "source", "meta"):  # keep records of plain files in their earlier shape
        if not rec.get(k):
            rec.pop(k, None)
    if verified_by:
        rec["verified_by"] = sorted(set(verified_by))
    if stripped:
        rec["stripped"] = {**stripped, "path": str(Path(stripped["path"]).relative_to(root))}
    if extra:
        rec.update(extra)
    write_record(root, rec)
    return rec


def record_digest(rec: dict) -> str:
    """SHA-256 the local file of ``rec`` must have (stripped copy / manifest or the file)."""
    return rec["stripped"]["sha256"] if rec.get("stripped") else rec["sha256_verified"]


def same_record(a: dict, b: dict) -> bool:
    """Two records of the same verified local content (fetch time and descriptive
    fields written by different code versions may differ)."""
    keys = ("id", "local_path", "sha256_verified")
    return all(a.get(k) == b.get(k) for k in keys) and record_digest(a) == record_digest(b)


__all__ = ["LEDGER_DIR", "LEGACY_LEDGER", "ledger_record_path", "migrate_ledger", "read_ledger",
           "read_ledger_record", "record", "record_digest", "same_record", "write_record"]
