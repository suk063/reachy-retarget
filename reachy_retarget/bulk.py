"""Official Hugging Face Xet batch transport for many small state files.

Uses the public repository's read token, never requests a user account or stores
credentials. Each payload still passes the published LFS SHA256 and size check.
"""

import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import time
from .store import Store, sha256
from .disk import allocation
from .acquire import RESERVE, host_paused, pause_host, file_lock


def inventory(store):
    result = {}
    for path in (store.root / "catalog/evidence").glob("hf__*.json"):
        try:
            data = json.loads(path.read_text())
            for entry in data.get("selected_files", []):
                if isinstance(entry, dict) and entry.get("xetHash"):
                    result[path.stem, entry["path"]] = entry["xetHash"]
        except (ValueError, KeyError):
            continue
    return result


def fetch_xet(store, items, batch_size=256):
    os.environ.setdefault("HF_XET_CACHE", str(store.root / "data/cache/xet"))
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS", "64")
    from hf_xet import XetSession, XetFileInfo

    session = XetSession()
    from huggingface_hub.utils._xet import XetFileData, refresh_xet_connection_info
    import fcntl

    hashes = inventory(store)
    groups = collections.defaultdict(list)
    for item in items:
        key = item["source_id"], item["path"]
        if key in hashes and "#" not in item["url"] and item.get("expected_sha256"):
            groups[item["source_id"]].append(item)
    outcomes = collections.Counter()
    for sid, rows in groups.items():
        if host_paused(store, rows[0]["url"]):
            outcomes["rate_wait"] += len(rows)
            continue
        first = rows[0]
        base, rest = first["url"].split("/resolve/", 1)
        revision = rest.split("/", 1)[0]
        repo = base.split("/datasets/", 1)[1]
        route = f"https://huggingface.co/api/datasets/{repo}/xet-read-token/{revision}"

        def connection():
            return refresh_xet_connection_info(
                file_data=XetFileData(
                    file_hash=hashes[first["source_id"], first["path"]],
                    refresh_route=route,
                ),
                headers={"User-Agent": "reachy-retarget/0.1"},
            )

        try:
            conn = connection()
        except Exception as exc:
            limited = "429" in str(exc)
            if limited:
                pause_host(store, first["url"])
            print(
                f"Xet {sid}: public connection unavailable ({type(exc).__name__}); queue retained",
                flush=True,
            )
            outcomes["rate_wait" if limited else "connection_failed"] += len(rows)
            continue

        def refresh():
            value = connection()
            return value.access_token, value.expiration_unix_epoch

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            if host_paused(store, first["url"]):
                break
            ownership = []
            owned = []
            for item in batch:
                lock = file_lock(store, item)
                if lock is not None:
                    ownership.append(lock)
                    owned.append(item)
                else:
                    outcomes["in_progress"] += 1
            batch = owned
            if not batch:
                continue
            total = sum(x["size"] for x in batch)
            # Reserve known reconstruction bytes without serializing network transfers.
            lock_path = store.root / "catalog/disk-write.lock"
            reservation = allocation(store.root, total + 64 * 1024 * 1024)
            try:
                reservation.__enter__()
            except RuntimeError:
                store.update_files(
                    [
                        (
                            x["source_id"],
                            x["path"],
                            {
                                "status": "storage_wait",
                                "error": "50 GB reserve before Xet allocation; queue retained",
                            },
                        )
                        for x in batch
                    ]
                )
                for lock in ownership:
                    lock.close()
                return dict(outcomes, storage_wait=len(batch))
            try:
                if (
                    shutil.disk_usage(store.root).free - total - 64 * 1024 * 1024
                    < RESERVE
                ):
                    for item in batch:
                        store.update_file(
                            item["source_id"],
                            item["path"],
                            status="storage_wait",
                            error="50 GB reserve before Xet batch",
                        )
                    return dict(outcomes, storage_wait=len(batch))
                infos = []
                store.update_files(
                    [
                        (
                            x["source_id"],
                            x["path"],
                            {"status": "downloading", "error": None},
                        )
                        for x in batch
                    ]
                )
                for item in batch:
                    target = store.root / "data/raw" / item["source_id"] / item["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    part = target.with_name(target.name + ".part")
                    if not (
                        part.exists()
                        and part.stat().st_size == item["size"]
                        and sha256(part) == item["expected_sha256"]
                    ):
                        infos.append(
                            (
                                XetFileInfo(
                                    hash=hashes[item["source_id"], item["path"]],
                                    file_size=item["size"],
                                ),
                                str(part),
                            )
                        )
                failed = None
                for attempt in range(3):
                    try:
                        with session.new_file_download_group(
                            endpoint=conn.endpoint,
                            token=conn.access_token,
                            token_expiry_unix_secs=conn.expiration_unix_epoch,
                            token_refresh_url=route,
                            token_refresh_headers={"User-Agent": "reachy-retarget/0.1"},
                        ) as group:
                            for info, destination in infos:
                                group.start_download_file(info, destination)
                        failed = None
                        break
                    except Exception as exc:
                        failed = (
                            type(exc).__name__
                        )  # Do not persist signed URLs/tokens from exception text.
                        if "429" in str(exc):
                            pause_host(store, first["url"])
                            failed = "HTTP 429"
                            break
                        if attempt < 2:
                            time.sleep(2**attempt)
                updates = []
                for item in batch:
                    key = item["source_id"], item["path"]
                    target = store.root / "data/raw" / key[0] / key[1]
                    part = target.with_name(target.name + ".part")
                    if not part.exists() or part.stat().st_size != item["size"]:
                        updates.append(
                            (
                                *key,
                                dict(
                                    status="retry_wait"
                                    if failed == "HTTP 429"
                                    else "failed",
                                    error=failed or "Xet size mismatch",
                                ),
                            )
                        )
                        outcomes["failed"] += 1
                        continue
                    digest = sha256(part)
                    if digest != item["expected_sha256"]:
                        part.rename(
                            part.with_name(part.name + f".corrupt-{int(time.time())}")
                        )
                        updates.append(
                            (
                                *key,
                                dict(
                                    status="failed",
                                    error="SHA256 mismatch; Xet payload quarantined",
                                ),
                            )
                        )
                        outcomes["failed"] += 1
                        continue
                    blob = store.root / "data/blobs" / digest[:2] / digest
                    blob.parent.mkdir(parents=True, exist_ok=True)
                    if blob.exists():
                        part.unlink()
                    else:
                        part.replace(blob)
                    target.unlink(missing_ok=True)
                    os.link(blob, target)
                    updates.append(
                        (
                            *key,
                            dict(
                                status="downloaded",
                                bytes=item["size"],
                                sha256=digest,
                                error=None,
                            ),
                        )
                    )
                    outcomes["downloaded"] += 1
                store.update_files(updates)
            finally:
                reservation.__exit__(None, None, None)
                for lock in ownership:
                    lock.close()
            print(
                f"Xet {sid} {min(start + batch_size, len(rows))}/{len(rows)} {dict(outcomes)}",
                flush=True,
            )
    return dict(outcomes)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--source", action="append")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    store = Store(args.root)
    rows = store.rows(
        "files", "status IN ('queued','failed','downloading','retry_wait')"
    )
    if args.source:
        rows = [r for r in rows if r["source_id"] in args.source]
    if args.limit:
        rows = rows[: args.limit]
    print(fetch_xet(store, rows), flush=True)
