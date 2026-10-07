"""Resumable bounded-concurrency HTTP acquisition with pinned URLs and hashes."""

import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import threading
import time
import json
import requests
from urllib.parse import urlparse, parse_qs
from .store import sha256, json_write
from .disk import available

RESERVE = 50_000_000_000
_disk_lock = threading.Lock()
_reservations = {}


class StorageFull(RuntimeError):
    pass


class RateLimited(RuntimeError):
    pass


def pause_host(store, url, seconds=900):
    path = store.root / "catalog/network_backoff.json"
    import fcntl

    with (store.root / "catalog/network-backoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads(path.read_text()) if path.exists() else {}
        host = urlparse(url).netloc
        until = max(
            current.get(host, {}).get("retry_after_epoch", 0), time.time() + seconds
        )
        current[host] = {
            "retry_after_epoch": until,
            "reason": "HTTP 429; preserve queue and respect provider limits",
        }
        json_write(path, current)


def host_paused(store, url):
    path = store.root / "catalog/network_backoff.json"
    if not path.exists():
        return False
    return (
        json.loads(path.read_text())
        .get(urlparse(url).netloc, {})
        .get("retry_after_epoch", 0)
        > time.time()
    )


def file_lock(store, item):
    """Nonblocking ownership shared by HTTP and Xet processes."""
    import fcntl

    name = hashlib.sha256(
        (item["source_id"] + "\0" + item["path"]).encode()
    ).hexdigest()
    directory = store.root / "catalog/file-locks"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / name).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def download(store, item, attempts=3, reserve=RESERVE):
    from .scope import exclusion_reason
    reason = exclusion_reason(store, item["source_id"])
    if reason:
        store.update_file(item["source_id"], item["path"], status="excluded_objectless", error=reason)
        return "excluded_objectless"
    lock = file_lock(store, item)
    if lock is None:
        return "in_progress"
    try:
        current = store.rows(
            "files", "source_id=? AND path=?", (item["source_id"], item["path"])
        )
        return _download_unlocked(
            store, current[0] if current else item, attempts, reserve
        )
    finally:
        lock.close()


def _download_unlocked(store, item, attempts=3, reserve=RESERVE):
    key = (item["source_id"], item["path"])
    if host_paused(store, item["url"]):
        return "rate_wait"
    target = store.root / "data/raw" / item["source_id"] / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if target.exists() and item["status"] == "downloaded":
        if target.stat().st_size == item["bytes"]:
            return "cached"
    if item["url"].endswith(
        (
            "#state-hdf5",
            "#state-parquet",
            "#state-zip",
            "#state-tar",
            "#state-tgz",
            "#state-rlds",
        )
    ):
        from .extract import hdf5_state, parquet_state, archive_state

        for attempt in range(attempts):
            try:
                store.update_file(*key, status="downloading", error=None)
                source_url, kind = item["url"].rsplit("#", 1)
                if kind == "state-hdf5":
                    size, digest = hdf5_state(source_url, target, reserve)
                elif kind == "state-parquet":
                    size, digest = parquet_state(source_url, target, reserve=reserve)
                elif kind == "state-rlds":
                    from .rlds import extract

                    size, digest = extract(source_url, target, reserve)
                else:
                    size, digest = archive_state(
                        source_url, target, kind.removeprefix("state-"), reserve
                    )
                store.update_file(
                    *key, status="downloaded", bytes=size, sha256=digest, error=None
                )
                return "downloaded"
            except Exception as e:
                limited = "429" in str(e)
                if isinstance(e, FileNotFoundError):
                    try:
                        limited = (
                            requests.head(
                                item["url"].split("#")[0], timeout=(10, 20)
                            ).status_code
                            == 429
                        )
                    except requests.RequestException:
                        pass
                if limited:
                    pause_host(store, item["url"])
                    store.update_file(
                        *key,
                        status="retry_wait",
                        error="HTTP 429 provider rate limit; queue preserved",
                    )
                    return "rate_wait"
                status = "storage_wait" if "50 GB reserve" in str(e) else "failed"
                store.update_file(*key, status=status, error=f"{type(e).__name__}: {e}")
                if status == "storage_wait":
                    return status
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
        return "failed"
    for attempt in range(attempts):
        try:
            offset = part.stat().st_size if part.exists() else 0
            need = max(0, (item.get("size") or 64 * 1024 * 1024) - offset)
            with _disk_lock:
                if (
                    available(target.parent) - sum(_reservations.values()) - need
                    < reserve
                ):
                    raise StorageFull("50 GB free-space reserve; queue preserved")
                _reservations[key] = need
            headers = {
                "User-Agent": "reachy-retarget/0.1",
                "Accept-Encoding": "identity",
            }
            if offset:
                headers["Range"] = f"bytes={offset}-"
            url = item["url"]
            if url.startswith("https://drive.google.com/uc?"):
                ident = parse_qs(urlparse(url).query)["id"][0]
                url = f"https://drive.usercontent.google.com/download?id={ident}&export=download&confirm=t"
            with requests.get(
                url, headers=headers, stream=True, timeout=(30, 120)
            ) as r:
                if r.status_code == 429:
                    delay = r.headers.get("Retry-After", "900")
                    pause_host(store, url, int(delay) if delay.isdigit() else 900)
                    raise RateLimited("HTTP 429 provider rate limit; queue preserved")
                if r.status_code == 416 and item.get("size") == offset:
                    pass
                else:
                    r.raise_for_status()
                    if offset and r.status_code != 206:
                        offset = 0
                    if r.status_code == 206 and not r.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected resume byte range")
                    content_type = r.headers.get("Content-Type", "")
                    if "text/html" in content_type and target.suffix not in (
                        ".html",
                        ".htm",
                    ):
                        raise ValueError(
                            "HTML response instead of a data file (login/confirmation page)"
                        )
                    total_size = (
                        offset + int(r.headers["Content-Length"])
                        if r.headers.get("Content-Length")
                        else item.get("size")
                    )
                    if total_size:
                        with _disk_lock:
                            other = sum(v for k, v in _reservations.items() if k != key)
                            if (
                                available(target.parent) - other - (total_size - offset)
                                < reserve
                            ):
                                raise StorageFull(
                                    "Response size exceeds remaining disk budget"
                                )
                            _reservations[key] = total_size - offset
                    with open(part, "ab" if offset else "wb") as f:
                        store.update_file(*key, status="downloading", error=None)
                        for chunk in r.iter_content(4 * 1024 * 1024):
                            if not chunk:
                                continue
                            with _disk_lock:
                                if available(target.parent) - len(chunk) < reserve:
                                    raise StorageFull(
                                        "50 GB free-space reserve during transfer"
                                    )
                                # Serialize the final free-space check/write
                                # across independently running fetch commands.
                                import fcntl

                                lock_path = store.root / "catalog/disk-write.lock"
                                with open(lock_path, "a") as disk_lock:
                                    fcntl.flock(disk_lock, fcntl.LOCK_EX)
                                    if available(target.parent) - len(chunk) < reserve:
                                        raise StorageFull(
                                            "50 GB reserve across concurrent fetches"
                                        )
                                    f.write(chunk)
                                    f.flush()
                                _reservations[key] = max(
                                    0, _reservations.get(key, 0) - len(chunk)
                                )
                        f.flush()
                        os.fsync(f.fileno())
            actual_size = part.stat().st_size
            if actual_size < 1024:
                header = part.read_bytes()
                if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ValueError(
                        "Git LFS pointer is metadata, not the data payload; resolve the official LFS media URL and declared hash/size"
                    )
            if item.get("size") is not None and actual_size != item["size"]:
                raise ValueError(f"Size mismatch {actual_size} != {item['size']}")
            digest = sha256(part)
            expected = item.get("expected_sha256")
            if expected and digest != expected:
                part.rename(part.with_name(part.name + f".corrupt-{int(time.time())}"))
                raise ValueError("SHA256 mismatch; corrupt payload quarantined")
            # Content-addressed payloads avoid retaining identical mirror files twice.
            blob = store.root / "data/blobs" / digest[:2] / digest
            blob.parent.mkdir(parents=True, exist_ok=True)
            if blob.exists():
                part.unlink()
            else:
                part.replace(blob)
            if target.exists():
                target.unlink()
            os.link(blob, target)
            store.update_file(
                *key, status="downloaded", bytes=actual_size, sha256=digest, error=None
            )
            return "downloaded"
        except StorageFull as e:
            store.update_file(*key, status="storage_wait", error=str(e))
            return "storage_wait"
        except RateLimited as e:
            store.update_file(*key, status="retry_wait", error=str(e))
            return "rate_wait"
        except Exception as e:
            store.update_file(
                *key,
                status="failed",
                error=f"{type(e).__name__}: {e}",
                bytes=part.stat().st_size if part.exists() else 0,
            )
            if attempt + 1 < attempts:
                time.sleep(min(8, 2**attempt))
        finally:
            with _disk_lock:
                _reservations.pop(key, None)
    return "failed"


def fetch(store, sources=None, workers=4, limit=None, retry_failed=True):
    items = store.rows(
        "files",
        "status IN ('queued','failed','storage_wait','downloading','retry_wait')",
    )
    if not retry_failed:
        items = [x for x in items if x["status"] != "failed"]
    if sources:
        items = [x for x in items if x["source_id"] in sources]
    paused = [x for x in items if host_paused(store, x["url"])]
    items = [x for x in items if not host_paused(store, x["url"])]
    if paused:
        print(
            f"Provider backoff active: {len(paused)} files preserved in queue",
            flush=True,
        )
    items.sort(
        key=lambda x: (
            x["size"] if x["size"] is not None else 10**12,
            x["source_id"],
            x["path"],
        )
    )
    if limit is not None:
        items = items[:limit]
    print(
        f"Queued {len(items)} files; {workers} workers; 50 GB free-space reserve",
        flush=True,
    )
    counts = {}
    # h5py serializes a complete visit under its process-global HDF5 lock.
    # Independent processes allow remote HDF5 reads to proceed concurrently.
    use_processes = any(
        x["url"].endswith(("#state-hdf5", "#state-rlds")) for x in items
    )
    executor = (
        concurrent.futures.ProcessPoolExecutor
        if use_processes
        else concurrent.futures.ThreadPoolExecutor
    )
    options = {}
    if use_processes:
        import multiprocessing

        options["mp_context"] = multiprocessing.get_context("spawn")
    with executor(max_workers=workers, **options) as pool:
        iterator = iter(items)
        pending = {}
        i = 0
        stopped = False

        def submit():
            item = next(iterator, None)
            if item is not None:
                pending[pool.submit(download, store, item)] = item

        for _ in range(workers * 2):
            submit()
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                item = pending.pop(future)
                if future.cancelled():
                    continue
                result = future.result()
                i += 1
                counts[result] = counts.get(result, 0) + 1
                if i % 25 == 0 or result not in ("downloaded", "cached"):
                    print(
                        f"{i}/{len(items)} {counts} {item['source_id']}/{item['path']}",
                        flush=True,
                    )
                if result in ("storage_wait", "rate_wait"):
                    stopped = True
                    for f in pending:
                        f.cancel()
                if not stopped:
                    submit()
    print(counts, flush=True)
    return counts
