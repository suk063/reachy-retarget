"""Upload a locally produced directory to the PVC with sha256 verification (explicit, never automatic).

Used for data that cannot be produced on the pods, e.g. BiGym replay records: they need the
isolated BiGym interpreter (MuJoCo 3.1.5, numpy 1.26, dm_control) and are made locally with
:func:`reachy_retarget.sources.bigym.replay`. The records and their ``assets/`` folder are
self-contained (the BiGym adapter finds ``assets/`` next to the task folders), so uploading
the whole replay folder makes ``python -m reachy_retarget.build --family bigym --path
{pvc}/data/derived/bigym/replay-v1/<Task>`` work on the pods.

Steps (:func:`upload_dir`):

1. hash every local file (sorted relative paths) into ``SHA256SUMS`` (``sha256sum`` format)
   and ``UPLOAD.json`` (source folder, file count, bytes, time, the git revision of this repo);
2. preflight on the pod: the destination must not exist, and the PVC must keep at least
   50 decimal GB free after the transfer (project rule);
3. stream a tar of the folder plus both manifests through ``kubectl exec -i ... tar -x`` into a
   staging folder ``<dest>.partial-<nonce>`` next to the destination (nothing is held in memory);
4. on the pod: ``sha256sum -c SHA256SUMS`` (every listed file present and equal) and a file
   count check (no extra files), then an atomic ``mv`` of the staging folder to ``<dest>``.

A destination that already exists is never overwritten: if it holds the same ``SHA256SUMS``
the upload is reported as already done; otherwise it is refused. A failed upload leaves its
staging folder in place (reported in the error) and never touches ``<dest>``. Files changed
locally while streaming fail the pod-side check.

    python -m cluster.upload data/derived/bigym/replay-v1 derived/bigym/replay-v1
    python -m cluster.upload data/derived/bigym/replay-v1 derived/bigym/replay-v1 --verify-only

Destinations are relative to the PVC data root ``/mnt/reachy-retarget/data``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import shlex
import subprocess
import tarfile
import threading
import time
import uuid
from pathlib import Path

from cluster import k8s

ROOT = Path(__file__).resolve().parents[1]
DATA = f"{k8s.PVC}/data"
RESERVE = 50_000_000_000  # decimal GB that must stay free on the PVC
SUMS, INFO = "SHA256SUMS", "UPLOAD.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_manifest(src) -> dict[str, dict]:
    """``{relative posix path: {"sha256", "bytes"}}`` of every regular file below ``src``."""
    src = Path(src)
    out = {}
    for p in sorted(src.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"{p}: symbolic links are not uploaded")
        if p.is_file():
            rel = p.relative_to(src).as_posix()
            if rel in (SUMS, INFO):
                raise ValueError(f"{p}: reserved name")
            out[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    if not out:
        raise ValueError(f"{src}: no files")
    return out


def sums_text(manifest: dict) -> str:
    return "".join(f"{v['sha256']}  ./{rel}\n" for rel, v in sorted(manifest.items()))


def _git_revision() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dest(dest: str) -> str:
    dest = posixpath.normpath(dest.strip("/"))
    if dest.startswith("..") or dest in (".", "") or dest.startswith("raw/") or dest == "raw":
        raise ValueError(f"{dest!r}: destination must be a new folder below the data root, outside raw/")
    return f"{DATA}/{dest}"


def _write_tar(stream, src: Path, manifest: dict, info: dict) -> None:
    """Write an uncompressed tar of ``manifest``'s files plus both manifests to ``stream``."""
    with tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for name, data in ((SUMS, sums_text(manifest).encode()), (INFO, json.dumps(info, indent=1).encode())):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
            tar.addfile(ti, io.BytesIO(data))
        for rel in sorted(manifest):
            tar.add(src / rel, arcname=rel, recursive=False)


PREFLIGHT = """set -eu
dest={dest}
if [ -e "$dest" ]; then
  echo "exists"
  if [ -f "$dest/{sums}" ]; then sha256sum "$dest/{sums}" | cut -d' ' -f1; else echo none; fi
else
  echo "absent"
  echo none
fi
mkdir -p "$(dirname "$dest")"
df -B1 --output=avail "$(dirname "$dest")" | tail -1 | tr -d ' '
"""

RECEIVE = """set -eu
dest={dest}
stage={stage}
mkdir "$stage"
tar -x -f - -C "$stage" --no-same-owner
cd "$stage"
sha256sum -c --quiet {sums}
n=$(find . -type f ! -name {sums} ! -name {info} | wc -l)
if [ "$n" -ne {count} ]; then echo "file count $n != {count}" >&2; exit 4; fi
if [ -e "$dest" ]; then echo "$dest appeared during the upload" >&2; exit 5; fi
mv "$stage" "$dest"
echo verified
"""

VERIFY = """set -eu
cd {dest}
sha256sum -c --quiet {sums}
find . -type f ! -name {sums} ! -name {info} | wc -l
"""


def stream_exec(pod: str, script: str, write, *, timeout: float = 6 * 3600) -> str:
    """Run ``script`` in ``pod`` with ``write(stdin)`` producing its standard input incrementally."""
    proc = subprocess.Popen(["kubectl", "-n", k8s.NAMESPACE, "exec", "-i", pod, "--", "sh", "-c", script],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = [], []
    readers = [threading.Thread(target=lambda s=s, b=b: b.append(s.read())) for s, b in
               ((proc.stdout, out), (proc.stderr, err))]
    for t in readers:
        t.start()
    try:
        write(proc.stdin)
    finally:
        proc.stdin.close()
    rc = proc.wait(timeout=timeout)
    for t in readers:
        t.join()
    if rc:
        raise RuntimeError(f"{pod}: exit {rc}: {b''.join(err).decode(errors='replace')[-2000:]}")
    return b"".join(out).decode()


def upload_dir(src, dest: str, pod: str | None = None, *, run=None, stream=None, manifest=None) -> dict:
    """Upload local folder ``src`` to ``<PVC>/data/<dest>`` (see the module docstring).

    ``run(pod, script) -> stdout`` and ``stream(pod, script, write) -> stdout`` default to
    :func:`cluster.k8s.run` and :func:`stream_exec`; tests pass fakes."""
    src = Path(src).resolve()
    run = run or (lambda p, s: k8s.run(p, s, timeout=600))
    stream = stream or stream_exec
    pod = pod or k8s.ready_pods()[0]
    target = _dest(dest)
    manifest = manifest or local_manifest(src)
    total = sum(v["bytes"] for v in manifest.values())
    sums_digest = hashlib.sha256(sums_text(manifest).encode()).hexdigest()
    summary = {"src": str(src), "dest": target, "pod": pod, "files": len(manifest), "bytes": total,
               "sums_sha256": sums_digest}

    state, remote_sums, avail = run(pod, PREFLIGHT.format(dest=shlex.quote(target), sums=SUMS)).split()[:3]
    if state == "exists":
        if remote_sums == sums_digest:
            return {**summary, "state": "already_present"}
        raise FileExistsError(f"{target} exists with different content (SHA256SUMS {remote_sums}); not overwritten")
    if int(avail) - total < RESERVE:
        raise OSError(f"PVC free space {int(avail) / 1e9:.1f} GB minus {total / 1e9:.1f} GB would fall below "
                      f"the {RESERVE / 1e9:.0f} GB reserve")

    info = {"source_dir": str(src), "dest": target, "files": len(manifest), "bytes": total,
            "sums_sha256": sums_digest, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repo_revision": _git_revision(), "tool": "cluster.upload"}
    stage = f"{target}.partial-{uuid.uuid4().hex[:8]}"
    script = RECEIVE.format(dest=shlex.quote(target), stage=shlex.quote(stage), sums=SUMS, info=INFO,
                            count=len(manifest))
    t0 = time.time()
    try:
        out = stream(pod, script, lambda fh: _write_tar(fh, src, manifest, info))
    except Exception as error:
        raise RuntimeError(f"upload failed; staging folder left for inspection: {stage}: {error}") from error
    if "verified" not in out.split():
        raise RuntimeError(f"upload not verified (staging folder {stage}): {out[-500:]}")
    return {**summary, "state": "uploaded", "seconds": round(time.time() - t0, 1)}


def verify_remote(dest: str, pod: str | None = None, *, run=None, expected_files: int | None = None) -> dict:
    """Re-check an uploaded folder against its own ``SHA256SUMS`` on the pod."""
    run = run or (lambda p, s: k8s.run(p, s, timeout=3600))
    pod = pod or k8s.ready_pods()[0]
    target = _dest(dest)
    n = int(run(pod, VERIFY.format(dest=shlex.quote(target), sums=SUMS, info=INFO)).split()[-1])
    if expected_files is not None and n != expected_files:
        raise RuntimeError(f"{target}: {n} files, expected {expected_files}")
    return {"dest": target, "files": n, "state": "verified"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="local folder")
    ap.add_argument("dest", help="destination below the PVC data root, e.g. derived/bigym/replay-v1")
    ap.add_argument("--pod", help="worker pod (default: first Ready pod)")
    ap.add_argument("--verify-only", action="store_true", help="only re-check an uploaded folder")
    a = ap.parse_args(argv)
    if a.verify_only:
        n = len(local_manifest(a.src)) if os.path.isdir(a.src) else None
        print(json.dumps(verify_remote(a.dest, a.pod, expected_files=n)))
    else:
        print(json.dumps(upload_dir(a.src, a.dest, a.pod)))


if __name__ == "__main__":
    main()
