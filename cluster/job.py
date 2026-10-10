"""Pod-side job runner: execute one job spec, publish its outputs to the PVC, write a receipt.

Runs from an extracted release directory (``python -m cluster.job <job dir>``). The job
command writes into ``<job dir>/out``; every output file is copied into
``<PVC>/<publish>/`` with a hash check and a no-overwrite rename. Failed jobs still publish
whatever they produced (failed retargets are data) and always write a receipt.

Copies run on ``publish_threads`` threads (spec, default 16): one small file on the CephFS PVC
takes ~1-5 s (create, fsync, read-back hash, link), so a serial copy of a long job's episodes
takes hours. Files matching a ``stream`` glob of the spec (relative to ``out``; for writers that
create each file under a temporary name and rename it into place, so a visible file is complete)
are published while the command still runs; everything else after it exits.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PVC = Path(os.environ.get("REACHY_RETARGET_PVC", "/mnt/reachy-retarget"))
RESERVE = 50e9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True))
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def same_ledger_record(a: Path, b: Path) -> bool:
    """Two fetch-ledger records of the same verified local content: same id, local path and
    digests (the fetch time and descriptive fields written by other code versions may differ;
    see ``reachy_retarget.acquire.ledger.same_record``)."""
    if "ledger" not in a.parts or a.suffix != ".json":
        return False
    ra, rb = (json.loads(p.read_text()) for p in (a, b))

    def key(r):
        return (r.get("id"), r.get("local_path"), r.get("sha256_verified"),
                (r.get("stripped") or {}).get("sha256"))
    return key(ra) == key(rb)


def publish_file(out: Path, src: Path, dest: Path) -> dict:
    """Copy one file under ``out`` to the same relative path under ``dest`` atomically; never
    overwrite."""
    rel = src.relative_to(out)
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(src)
    if target.exists():
        if sha256(target) != digest and not same_ledger_record(src, target):
            raise FileExistsError(f"{target} exists with different content")
    else:
        partial = target.with_name(f".{target.name}.{os.getpid()}.{os.urandom(4).hex()}.partial")
        shutil.copyfile(src, partial)
        with partial.open("rb") as f:
            os.fsync(f.fileno())
        if sha256(partial) != digest:
            raise OSError(f"hash mismatch after copying {rel}")
        try:
            os.link(partial, target)  # fails instead of overwriting a concurrent writer
        except FileExistsError:
            # Another job published the same path meanwhile (shared content-addressed assets):
            # fine when the content is identical, an error otherwise.
            if sha256(target) != digest and not same_ledger_record(src, target):
                raise FileExistsError(f"{target} exists with different content") from None
        finally:
            partial.unlink()
    return {"path": str(rel), "sha256": digest, "bytes": src.stat().st_size}


class Publisher:
    """Publishes files under ``out`` to ``dest`` on a thread pool, each path once."""

    def __init__(self, out: Path, dest: Path, threads: int = 16):
        self.out, self.dest = out, dest
        self.pool = ThreadPoolExecutor(threads)
        self.futures = {}  # relative path -> Future of its record

    def submit(self, paths) -> None:
        for src in sorted(paths):
            rel = str(src.relative_to(self.out))
            if rel not in self.futures:
                self.futures[rel] = self.pool.submit(publish_file, self.out, src, self.dest)

    def stream(self, patterns) -> None:
        """Submit the complete files matching ``patterns`` (globs relative to ``out``)."""
        self.submit(p for pattern in patterns for p in self.out.glob(pattern)
                    if p.is_file() and not p.name.startswith("."))

    def finish(self) -> list[dict]:
        """Submit every remaining file, wait, and return the records sorted by path; raises the
        first copy error."""
        self.submit(p for p in self.out.rglob("*") if p.is_file())
        try:
            return [self.futures[rel].result() for rel in sorted(self.futures)]
        finally:
            self.pool.shutdown(wait=True)


def publish(out: Path, dest: Path, threads: int = 16) -> list[dict]:
    """Copy every file under ``out`` into ``dest`` atomically; never overwrite."""
    return Publisher(out, dest, threads).finish()


def main(job_dir: str) -> int:
    job_dir = Path(job_dir)
    spec = json.loads((job_dir / "spec.json").read_text())
    out = job_dir / "out"
    out.mkdir(exist_ok=True)
    receipt = {"id": spec["id"], "batch": spec["batch"], "pod": os.environ.get("HOSTNAME"),
               "release": spec["release"], "argv": spec["argv"], "started": time.time()}
    write_json(job_dir / "status.json", {**receipt, "state": "running"})
    free = shutil.disk_usage(job_dir).free
    if free < RESERVE:
        receipt.update(state="refused", error=f"local free space {free / 1e9:.1f} GB below reserve")
    else:
        # MALLOC_ARENA_MAX: glibc otherwise keeps a heap arena per thread (MuJoCo compiles
        # meshes on a thread pool), whose freed pages stay resident; jobs share a 4 GiB pod.
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "MALLOC_ARENA_MAX": "2", "REACHY_RETARGET_PVC": str(PVC)}
        argv = [arg.replace("{out}", str(out)).replace("{pvc}", str(PVC)) for arg in spec["argv"]]
        publisher = Publisher(out, PVC / spec["publish"], spec.get("publish_threads", 16))
        deadline = time.time() + spec.get("timeout_s", 6 * 3600)
        with (job_dir / "log.txt").open("wb") as log:
            proc = subprocess.Popen([sys.executable, *argv], stdout=log, stderr=subprocess.STDOUT, env=env)
            while proc.poll() is None:
                if time.time() > deadline:
                    proc.kill()
                    proc.wait()
                    receipt["error"] = f"timeout after {spec.get('timeout_s', 6 * 3600)} s"
                    break
                if spec.get("stream"):
                    publisher.stream(spec["stream"])
                time.sleep(10)
        receipt.update(returncode=proc.returncode, state="succeeded" if proc.returncode == 0 else "failed")
        try:
            receipt["published"] = publisher.finish()
        except Exception as error:  # noqa: BLE001 - recorded in the receipt
            receipt.update(state="publish_failed", error=repr(error))
    receipt["finished"] = time.time()
    receipt["log_tail"] = (job_dir / "log.txt").read_text(errors="replace")[-4000:] \
        if (job_dir / "log.txt").exists() else ""
    receipts = PVC / "receipts" / spec["batch"]
    receipts.mkdir(parents=True, exist_ok=True)
    write_json(receipts / f"{spec['id']}.json", receipt)
    write_json(job_dir / "status.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
