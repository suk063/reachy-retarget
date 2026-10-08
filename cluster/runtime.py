"""Build the pod Python runtime and publish immutable source releases to the shared PVC.

Runtime: a venv built once on a pod at its final absolute path ``/tmp/rr2/runtime/<hash>``
and archived to ``<PVC>/runtime/<hash>.tar.gz``; other pods extract it to the same path.
Release: the committed package sources (``git archive HEAD``) archived to ``<PVC>/releases/<sha256>.tar.gz``.
Both are content addressed and never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from cluster import k8s

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ("numpy==2.5.3", "scipy==1.18.1", "h5py==3.16.0", "pyyaml==6.0.3",
                "pyarrow==25.0.1", "mujoco==3.15.0", "zstandard==0.25.0")  # same versions as the local development venv
RELEASE_PATHS = ("reachy_retarget", "cluster", "pyproject.toml")


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
tar -C /tmp/rr2/runtime -czf "$archive.$$.partial" {h}
sync
mv -n "$archive.$$.partial" "$archive"
rm -f "$archive.$$.partial"
""", timeout=1800)
    return h


def release_archive() -> tuple[str, bytes]:
    """Deterministic tar of the committed (HEAD) release paths; returns (sha256, bytes).

    Built from HEAD rather than the working tree so uncommitted edits never ship.
    """
    payload = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "HEAD", "--", *RELEASE_PATHS],
                             check=True, capture_output=True).stdout
    return hashlib.sha256(payload).hexdigest(), payload


def publish_release(pod: str) -> str:
    """Upload the current tracked sources as an immutable release; returns its sha256."""
    sha, payload = release_archive()
    k8s.run(pod, f"""set -eu
target={k8s.PVC}/releases/{sha}.tar
[ -f "$target" ] && exit 0
mkdir -p {k8s.PVC}/releases
partial="$target.$$.partial"  # unique per uploader: concurrent pools may publish the same release
cat > "$partial"
[ "$(sha256sum "$partial" | cut -d' ' -f1)" = {sha} ]
sync
mv -n "$partial" "$target"
rm -f "$partial"
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
