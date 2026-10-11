"""Sampling a published dataset folder for local inspection: the pod-side selection, run locally."""
import json
import shlex
import shutil
import subprocess
import sys

import pytest

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


def test_pull_all_copies_every_file_with_pod_side_hashes(tmp_path):
    import tarfile

    root = tmp_path / "pvc" / "datasets" / "scenes-x"
    for rel, data in (("scenes/a/scene.xml", b"<mujoco/>"), ("scenes/a/assets/ff", b"\0" * 10),
                      ("records/job.jsonl", b"{}\n")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / ".hidden.partial").write_bytes(b"in flight")

    def run(pod, script, stdin=None):
        script = script.replace(k8s.PVC, str(tmp_path / "pvc"))
        return subprocess.run(["sh", "-c", script], input=stdin, capture_output=True, check=True).stdout.decode()

    def stream(pod, src, files, dest):
        src = src.replace(k8s.PVC, str(tmp_path / "pvc"))
        with tarfile.open(fileobj=None, mode="w", name=tmp_path / "t.tar") as tar:
            for f in files:
                tar.add(f"{src}/{f}", arcname=f)
        with tarfile.open(tmp_path / "t.tar") as tar:
            tar.extractall(dest, filter="data")

    out = pull.pull_all("datasets/scenes-x", tmp_path / "local" / "scenes-x", "pod", run=run, stream=stream)
    dest = tmp_path / "local" / "scenes-x"
    assert out["files"] == 3 and (dest / "scenes/a/assets/ff").read_bytes() == b"\0" * 10
    assert not (dest / ".hidden.partial").exists()
    sums = (dest / "SHA256SUMS").read_text()
    assert "scenes/a/scene.xml" in sums and json.loads((dest / "PULL.json").read_text())["files"] == 3
    with pytest.raises(FileExistsError):
        pull.pull_all("datasets/scenes-x", dest, "pod", run=run, stream=stream)


def test_batches_split_by_size():
    sizes = {"a": 5, "b": 5, "c": 20, "d": 1}
    assert pull.batches(["a", "b", "c", "d"], sizes, 10) == [["a", "b"], ["c"], ["d"]]


def test_select_final_only_takes_each_record_file_without_its_earlier_attempts(tmp_path):
    root = tmp_path / "datasets" / "tracking-v1"
    (root / "records" / "tracking").mkdir(parents=True)
    lines = [_record(i, "ab"[i % 2], True, root, 1 + (i % 3 == 0)) for i in range(12)]
    (root / "records" / "tracking" / "s0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))

    def run(pod, script, stdin):
        args = shlex.split(script)[2:]
        args[0] = args[0].replace(k8s.PVC, str(tmp_path))
        return subprocess.run([sys.executable, "-", *args], input=stdin, capture_output=True, check=True).stdout.decode()

    sel = pull.select("datasets/tracking-v1", "pod", shards=9, per_cell=4, failed=0, seed=0, final_only=True, run=run)
    recs = [json.loads(r["line"]) for r in sel["records"]]
    assert len(recs) == 8 and len(sel["files"]) == 8
    assert sel["files"] == ["episodes/" + r["file"].split("/episodes/", 1)[1] for r in recs]
    assert sel["sizes"] == {f: (root / f).stat().st_size for f in sel["files"]}


def test_transfer_retries_a_cut_stream_on_another_pod_and_resumes(tmp_path):
    import tarfile

    src = tmp_path / "pvc" / "d"
    files = [f"episodes/e{i}.h5" for i in range(6)]
    for i, f in enumerate(files):
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_bytes(bytes([i]) * (10 + i))
    sizes = {f: (src / f).stat().st_size for f in files}

    def run(pod, script, stdin=None):
        return subprocess.run(["sh", "-c", script.replace(k8s.PVC, str(tmp_path / "pvc"))], input=stdin,
                              capture_output=True, check=True).stdout.decode()

    calls = []

    def stream(pod, root, group, dest):
        calls.append((pod, tuple(group)))
        if pod == "bad":
            raise tarfile.ReadError("unexpected end of data")
        for f in group:
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / f, dest / f)

    stage = tmp_path / "stage"
    stage.mkdir()
    out = pull.transfer(f"{k8s.PVC}/d", files, sizes, stage, ["bad", "good"], run=run, stream=stream,
                        batch_bytes=25, workers=1)
    assert sorted(out) == files and all((stage / f).read_bytes() == (src / f).read_bytes() for f in files)
    good = [g for pod, g in calls if pod == "good"]
    assert len(good) == len(set(good)) == len(pull.batches(files, sizes, 25))  # each stream once on "good"
    assert any(pod == "bad" for pod, _ in calls) and all(g in good for pod, g in calls if pod == "bad")
    calls.clear()
    assert pull.transfer(f"{k8s.PVC}/d", files, sizes, stage, ["good"], run=run, stream=stream,
                         batch_bytes=25) == out
    assert calls == []  # resumed: every stream was already done


def test_pull_sample_resumes_with_its_saved_selection(tmp_path):
    root = tmp_path / "datasets" / "tracking-v1"
    (root / "records" / "tracking").mkdir(parents=True)
    lines = [_record(i, "ab"[i % 2], True, root, 1 + (i % 3 == 0)) for i in range(40)]
    (root / "records" / "tracking" / "s0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    selections = []

    def run(pod, script, stdin=None):
        if script.startswith("python3 - "):
            selections.append(pod)
            args = shlex.split(script)[2:]
            args[0] = args[0].replace(k8s.PVC, str(tmp_path))
            cmd = [sys.executable, "-", *args]
        else:
            cmd = ["sh", "-c", script.replace(k8s.PVC, str(tmp_path))]
        return subprocess.run(cmd, input=stdin, capture_output=True, check=True).stdout.decode()

    streams = []

    def stream(pod, src, files, dest):
        if broken and streams:
            raise OSError("interrupted")
        streams.append(tuple(files))
        for f in files:
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src.replace(k8s.PVC, str(tmp_path)) + "/" + f, dest / f)

    dest = tmp_path / "local" / "sample"
    kw = {"per_cell": 6, "failed": 0, "final_only": True, "batch_bytes": 40, "workers": 1, "run": run,
          "stream": stream, "ready": lambda: ["p0", "p1"], "pods": 2}
    broken = True
    with pytest.raises(OSError):
        pull.pull_sample("datasets/tracking-v1", dest, **kw)
    assert not dest.exists() and len(streams) == 1
    broken = False
    out = pull.pull_sample("datasets/tracking-v1", dest, **kw)
    assert selections == ["p0"]  # the second run used the saved selection
    assert out["episode_files"] == 12 and len(streams) == len(set(streams))  # no stream sent twice
    assert sorted(p.name for p in dest.iterdir()) == ["PULL.json", "SHA256SUMS", "episodes", "records"]
    for line in (dest / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split(None, 1)
        assert pull.sha256(dest / rel) == digest
