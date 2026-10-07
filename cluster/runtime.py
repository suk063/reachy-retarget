"""Build the pod Python runtime and publish immutable source releases to the shared PVC.

Runtime: a venv built once on a pod at its final absolute path ``/tmp/rr2/runtime/<hash>``
and archived to ``<PVC>/runtime/<hash>.tar.gz``; other pods extract it to the same path.
Release: the tracked package sources archived to ``<PVC>/releases/<sha256>.tar.gz``.
Both are content addressed and never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

from cluster import k8s

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ("numpy==2.5.3", "scipy==1.18.1", "h5py==3.16.0", "pyyaml==6.0.3",
                "pyarrow==25.0.1", "mujoco==3.15.0")  # same versions as the local development venv
RELEASE_PATHS = ("reachy_retarget", "cluster", "pyproject.toml", "configs")


def runtime_hash() -> str:
    return hashlib.sha256("\n".join(REQUIREMENTS).encode()).hexdigest()[:16]


def build_runtime(pod: str) -> str:
    """Build the runtime on one pod unless the archive already exists; returns its hash."""
    h = runtime_hash()
    reqs = " ".join(REQUIREMENTS)
    k8s.run(pod, f"""set -eu
archive={k8s.PVC}/runtime/{h}.tar.gz
[ -f "$archive" ] && exit 0
mkdir -p {k8s.PVC}/runtime /tmp/rr2/runtime
rm -rf /tmp/rr2/runtime/{h}
python3 -m venv /tmp/rr2/runtime/{h}
/tmp/rr2/runtime/{h}/bin/pip install -q --no-cache-dir {reqs}
tar -C /tmp/rr2/runtime -czf "$archive.partial" {h}
sync
mv -n "$archive.partial" "$archive"
""", timeout=1800)
    return h


def release_archive() -> tuple[str, bytes]:
    """Deterministic tar.gz of git-tracked release files; returns (sha256, bytes)."""
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", *RELEASE_PATHS],
                           check=True, capture_output=True, text=True).stdout.split()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name in sorted(files):
            data = (ROOT / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    payload = raw.getvalue()
    return hashlib.sha256(payload).hexdigest(), payload


def publish_release(pod: str) -> str:
    """Upload the current tracked sources as an immutable release; returns its sha256."""
    sha, payload = release_archive()
    k8s.run(pod, f"""set -eu
target={k8s.PVC}/releases/{sha}.tar
[ -f "$target" ] && exit 0
mkdir -p {k8s.PVC}/releases
cat > "$target.partial"
[ "$(sha256sum "$target.partial" | cut -d' ' -f1)" = {sha} ]
sync
mv -n "$target.partial" "$target"
""", stdin=payload)
    return sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", help="pod used for building/uploading (default: first Ready)")
    args = parser.parse_args()
    pod = args.pod or k8s.ready_pods()[0]
    print("runtime", build_runtime(pod))
    print("release", publish_release(pod))


if __name__ == "__main__":
    main()
