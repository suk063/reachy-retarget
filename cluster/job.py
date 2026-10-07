"""Pod-side job runner: execute one job spec, publish its outputs to the PVC, write a receipt.

Runs from an extracted release directory (``python -m cluster.job <job dir>``). The job
command writes into ``<job dir>/out``; every output file is copied into
``<PVC>/<publish>/`` with a hash check and a no-overwrite rename. Failed jobs still publish
whatever they produced (failed retargets are data) and always write a receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PVC = Path(os.environ.get("REACHY_RETARGET_PVC", "/mnt/reachy-retarget/v2"))
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


def publish(out: Path, dest: Path) -> list[dict]:
    """Copy every file under ``out`` into ``dest`` atomically; never overwrite."""
    published = []
    for src in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = src.relative_to(out)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(src)
        if target.exists():
            if sha256(target) != digest:
                raise FileExistsError(f"{target} exists with different content")
        else:
            partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
            shutil.copyfile(src, partial)
            with partial.open("rb") as f:
                os.fsync(f.fileno())
            if sha256(partial) != digest:
                raise OSError(f"hash mismatch after copying {rel}")
            os.link(partial, target)  # fails instead of overwriting a concurrent writer
            partial.unlink()
        published.append({"path": str(rel), "sha256": digest, "bytes": src.stat().st_size})
    return published


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
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "REACHY_RETARGET_PVC": str(PVC)}
        argv = [arg.replace("{out}", str(out)).replace("{pvc}", str(PVC)) for arg in spec["argv"]]
        with (job_dir / "log.txt").open("wb") as log:
            proc = subprocess.run([sys.executable, *argv], stdout=log, stderr=subprocess.STDOUT, env=env,
                                  timeout=spec.get("timeout_s", 6 * 3600))
        receipt.update(returncode=proc.returncode, state="succeeded" if proc.returncode == 0 else "failed")
        try:
            receipt["published"] = publish(out, PVC / spec["publish"])
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
