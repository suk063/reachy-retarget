"""Sampling a published dataset folder for local inspection: the pod-side selection, run locally."""
import json
import shlex
import subprocess
import sys

from cluster import k8s, pull


def _record(i, cell, passed, root, attempts):
    files = []
    for a in range(attempts):
        rel = f"tracking/synthetic-v1/e{i}-{a}.h5"
        (root / "episodes" / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / "episodes" / rel).write_bytes(b"x" * (i + 1))
        files.append(f"/tmp/rr2/jobs/j{i}/out/episodes/{rel}")
    return {"scenario": {"cell": cell}, "K": {"passed": passed}, "attempts": [{"file": f} for f in files],
            "file": files[-1]}


def test_select_draws_per_cell_passes_and_failures_with_every_attempt(tmp_path):
    root = tmp_path / "datasets" / "tracking-v1"
    (root / "records" / "tracking").mkdir(parents=True)
    for shard in range(4):
        lines = [_record(100 * shard + i, "ab"[i % 2], i % 3 != 0, root, 1 + (i % 3 == 0)) for i in range(20)]
        (root / "records" / "tracking" / f"s{shard}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))

    def run(pod, script, stdin):
        args = shlex.split(script)[2:]
        args[0] = args[0].replace(k8s.PVC, str(tmp_path))
        return subprocess.run([sys.executable, "-", *args], input=stdin, capture_output=True, check=True).stdout.decode()

    sel = pull.select("datasets/tracking-v1", "pod", shards=2, per_cell=3, failed=2, seed=0, run=run)
    assert sel["record_files"] == ["records/tracking/s0.jsonl", "records/tracking/s2.jsonl"]
    assert sel["cells"] == {"a": [12, 8], "b": [14, 6]}
    recs = [json.loads(r["line"]) for r in sel["records"]]
    assert len(recs) == 10 and sum(not r["K"]["passed"] for r in recs) == 4
    assert len(sel["files"]) == 14  # failed draws carry two attempt files each
    assert all(f.startswith("episodes/tracking/synthetic-v1/") and (root / f).is_file() for f in sel["files"])
    assert sel["bytes"] == sum((root / f).stat().st_size for f in sel["files"])
    assert pull.select("datasets/tracking-v1", "pod", shards=2, per_cell=3, failed=2, seed=0, run=run) == sel
