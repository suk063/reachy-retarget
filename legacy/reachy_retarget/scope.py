"""Object-inclusive scope and explicit, provenance-recorded local removal."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .episodes import read_episode
from .store import Store, json_write, now, sha256


# These local selections are known to omit object trajectories. Dataset-family
# names or the word "robot" alone are never evidence for exclusion.
EXCLUDED_SOURCES = {
    "hf__glannuzel__reachy2_pick_place": "Local numeric selection has robot joints but no object state",
    "cmu": "Body-only motion capture selection",
    "lafan1": "Body-only skeletal motion selection",
    "amass": "Body-only motion archive; derived HOI datasets are separate sources",
}


def exclusion_reason(store, source):
    if source in EXCLUDED_SOURCES:
        return EXCLUDED_SOURCES[source]
    tombstones = store.root / "catalog/objectless-exclusions.json"
    if tombstones.exists() and source in json.loads(tombstones.read_text()):
        return "Persistent removal record: objectless selection removed by user"
    rows = store.rows("sources", "id=?", (source,))
    if rows and rows[0]["status"] == "excluded_objectless":
        return "Locally inspected objectless selection removed by user"
    return None


def object_tracks(arrays):
    tracks = {}
    frames = len(arrays.get("time_s", []))
    for key, values in arrays.items():
        parts = key.split("/")
        if len(parts) != 3 or parts[0] != "objects" or parts[2] != "pose":
            continue
        poses = np.asarray(values)
        if poses.shape != (frames, 7):
            continue
        valid = np.array(arrays.get(f"objects/{parts[1]}/valid", np.ones(frames, dtype=bool)), dtype=bool, copy=True)
        if valid.shape != (frames,):
            continue
        valid &= np.isfinite(poses).all(axis=1) & (np.abs(np.linalg.norm(poses[:, 3:], axis=1)-1) < 1e-3)
        if valid.sum() >= 2:
            tracks[parts[1]] = int(valid.sum())
    return tracks


def require_objects(arrays):
    tracks = object_tracks(arrays)
    if not tracks:
        raise ValueError("Excluded robot/body-only episode: no valid time-aligned object pose track")
    return tracks


def _sha256_digest(value):
    if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value):
        return value.lower()
    return None


def _orphaned_raw_blobs(store, source, raw_hashes, ledger_files):
    """Plan deletion of this source's blobs, retaining every shared reference."""
    root = store.root.resolve()
    blob_root = root / "data/blobs"
    # Expected hashes alone do not prove that this source ever acquired a blob.
    digests = {digest for value in [*raw_hashes, *(row.get("sha256") for row in ledger_files)]
               if (digest := _sha256_digest(value))}
    candidates = {}
    for digest in sorted(digests):
        path = blob_root / digest[:2] / digest
        for component in (root / "data", blob_root, path.parent, path):
            if component.is_symlink():
                raise ValueError("Refuse deletion of linked raw blob data")
        if not path.exists():
            continue
        if not path.is_file() or not path.resolve().is_relative_to(blob_root):
            raise ValueError("Refuse deletion outside the content-addressed blob store")
        if sha256(path) != digest:
            raise ValueError(f"Raw blob checksum does not match its address: {path}")
        candidates[digest] = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                              "sha256": digest, "kind": "raw_blob"}

    preserved = []
    references = {}
    for row in store.rows("files", "source_id != ?", (source,)):
        for value in (row.get("sha256"), row.get("expected_sha256")):
            digest = _sha256_digest(value)
            if digest in candidates:
                references.setdefault(digest, set()).add((row["source_id"], row["path"]))
    for digest, uses in sorted(references.items()):
        preserved.append(dict(candidates.pop(digest), reason="Referenced by another source ledger",
                              references=[{"source_id": sid, "path": path} for sid, path in sorted(uses)]))

    # An unregistered raw copy or hardlink is still a reference. Size and inode
    # checks avoid hashing unrelated large payloads; copies need a content hash.
    raw_root = root / "data/raw"
    removed_raw = raw_root / source
    sizes = {entry["bytes"] for entry in candidates.values()}
    inodes = {}
    for digest, entry in candidates.items():
        stat = (root / entry["path"]).stat()
        inodes[(stat.st_dev, stat.st_ino)] = digest
    raw_paths = [raw_root] if raw_root.is_symlink() else raw_root.rglob("*")
    for path in raw_paths:
        if not candidates:
            break
        if path.is_relative_to(removed_raw):
            continue
        if path.is_symlink():
            # Do not follow links into sibling stores, or assume their contents
            # are unreferenced. Keep remaining blobs when inventory is uncertain.
            preserved.extend(dict(entry, reason="Linked raw path prevents complete reference inventory",
                                  references=[{"path": str(path.relative_to(root))}])
                             for entry in candidates.values())
            candidates.clear()
            break
        if not path.is_file():
            continue
        stat = path.stat()
        if stat.st_size not in sizes:
            continue
        digest = inodes.get((stat.st_dev, stat.st_ino)) or sha256(path)
        if digest in candidates:
            preserved.append(dict(candidates.pop(digest), reason="Referenced by another raw path",
                                  references=[{"path": str(path.relative_to(root))}]))
    return list(candidates.values()), {"candidate_hashes": sorted(digests), "preserved": preserved}


def remove_source(store, source, *, include_raw=False):
    """Delete only an explicitly named, fully inspected objectless local selection."""
    if Path(source).name != source or source in (".", "..") or "\\" in source:
        raise ValueError("Source must be one safe path component")
    rows = store.rows("episodes", "source_id=?", (source,))
    normalized = store.root / "data/normalized" / source
    candidates = set(normalized.rglob("*.h5")) if normalized.exists() else set()
    for row in rows:
        path = store.root / row["path"]
        if not path.exists():
            raise ValueError(f"Cannot verify missing normalized episode: {path}")
        candidates.add(path)
    for path in candidates:
        if path.is_symlink() or not path.resolve().is_relative_to(normalized.resolve()):
            raise ValueError("Refuse inspection of linked or external normalized data")
        if object_tracks(read_episode(path)[0]):
            raise ValueError(f"Refuse whole-source removal: object state exists in {path}")
    if not rows and not exclusion_reason(store, source):
        raise ValueError("No inspected episodes or explicit objectless-source evidence")
    directories = [store.root / "data" / stage / source for stage in ("normalized", "retargeted")]
    if include_raw:
        directories.append(store.root / "data/raw" / source)
    root = store.root.resolve()
    files = []
    for directory in directories:
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.resolve().is_relative_to(root / "data"):
            raise ValueError("Refuse deletion outside this store's data directory")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("Refuse to traverse linked source data")
            if path.is_file():
                files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    ledger_files = store.rows("files", "source_id=?", (source,))
    blobs = []
    blob_cleanup = {"requested": include_raw, "candidate_hashes": [], "preserved": []}
    if include_raw:
        raw_prefix = f"data/raw/{source}/"
        raw_hashes = [entry["sha256"] for entry in files if entry["path"].startswith(raw_prefix)]
        blobs, evidence = _orphaned_raw_blobs(store, source, raw_hashes, ledger_files)
        blob_cleanup.update(evidence)
        files.extend(blobs)
    stamp = now().replace(":", "-")
    manifest = {"created": now(), "source_id": source, "include_raw": include_raw,
                "reason": EXCLUDED_SOURCES.get(source, "All inspected normalized episodes lack object state"),
                "authorization": "User explicitly requested deletion including original downloads",
                "files": files, "blob_cleanup": blob_cleanup,
                "ledger_sources": store.rows("sources", "id=?", (source,)),
                "ledger_files": ledger_files, "ledger_episodes": rows}
    record = root / "runs/removals" / f"{source}-{stamp}.json"
    json_write(record, manifest)
    tombstones = root / "catalog/objectless-exclusions.json"
    exclusions = json.loads(tombstones.read_text()) if tombstones.exists() else {}
    exclusions[source] = {"reason": manifest["reason"], "manifest": str(record), "created": now()}
    json_write(tombstones, exclusions)
    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)
    for entry in blobs:
        (root / entry["path"]).unlink()
    with store.connect() as connection:
        connection.execute("DELETE FROM episodes WHERE source_id=?", (source,))
        connection.execute("UPDATE files SET status='excluded_objectless',error=? WHERE source_id=?",
                           ("Removed by user; do not reacquire robot-only selection", source))
        connection.execute("UPDATE sources SET status='excluded_objectless' WHERE id=?", (source,))
    manifest["completed"] = now()
    json_write(record, manifest)
    return {"source": source, "files_removed": len(files), "bytes_removed": sum(x["bytes"] for x in files),
            "blobs_removed": len(blobs), "manifest": str(record)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--include-raw", action="store_true")
    args = parser.parse_args()
    print(json.dumps(remove_source(Store(args.root.resolve()), args.source, include_raw=args.include_raw), indent=2))
