"""Portable selection lock: public URLs, source versions and integrity metadata."""

import gzip
import json
from pathlib import Path
from .store import now


def export_lock(store, path=None):
    path = Path(path or store.root / "configs/collection-lock.jsonl.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "kind": "header",
                    "schema": "reachy-collection-lock-v1",
                    "created": now(),
                }
            )
            + "\n"
        )
        for row in store.rows("sources"):
            f.write(json.dumps({"kind": "source", **row}) + "\n")
        for row in store.rows("files"):
            # Local extraction hashes differ from the original remote-file hash;
            # keep both identities and never compare them as equivalent.
            f.write(json.dumps({"kind": "file", **row}) + "\n")
    return path


def restore_lock(store, path):
    excluded = ("excluded", "unsuitable", "state_extracted_media_omitted")
    counts = {"sources": 0, "files": 0}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(next(f))
        if header.get("schema") != "reachy-collection-lock-v1":
            raise ValueError("Unsupported collection lock")
        for line in f:
            row = json.loads(line)
            if row["kind"] == "source":
                store.source(
                    row["id"],
                    row["category"],
                    row["url"],
                    row["status"],
                    **json.loads(row["details"]),
                )
                counts["sources"] += 1
            elif row["kind"] == "file":
                store.file(
                    row["source_id"],
                    row["path"],
                    row["url"],
                    row["size"],
                    row["expected_sha256"],
                )
                if row["status"].startswith(excluded):
                    store.update_file(
                        row["source_id"],
                        row["path"],
                        status=row["status"],
                        error=row["error"],
                    )
                counts["files"] += 1
    return counts
