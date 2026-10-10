"""Thin kubectl helpers for the persistent worker pods. Never deletes or recreates pods."""
from __future__ import annotations

import json
import subprocess

NAMESPACE = "erl-ucsd"
SELECTOR = "app.kubernetes.io/name=reachy-retarget-worker"
PVC = "/mnt/reachy-retarget"


def ready_pods() -> list[str]:
    """Names of Ready worker pods, sorted."""
    out = subprocess.run(["kubectl", "-n", NAMESPACE, "get", "pods", "-l", SELECTOR, "-o", "json"],
                         check=True, capture_output=True, text=True).stdout
    pods = []
    for item in json.loads(out)["items"]:
        if item["metadata"].get("deletionTimestamp"):
            continue
        conditions = {c["type"]: c["status"] for c in item["status"].get("conditions", [])}
        if conditions.get("Ready") == "True":
            pods.append(item["metadata"]["name"])
    return sorted(pods)


def run(pod: str, script: str, *, stdin: bytes | None = None, timeout: float = 600) -> str:
    """Run a POSIX shell script in a pod and return stdout; raises on a non-zero exit."""
    proc = subprocess.run(["kubectl", "-n", NAMESPACE, "exec", "-i", pod, "--", "sh", "-c", script],
                          input=stdin, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"{pod}: exit {proc.returncode}: {proc.stderr.decode(errors='replace')[-2000:]}")
    return proc.stdout.decode()
