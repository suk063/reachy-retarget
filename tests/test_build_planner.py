"""Build-manifest planning from PVC listings, in-adapter shard selection, multi-path builds and the
verified directory upload (all offline: synthetic listings, fake kubectl runners)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster import manifests as mf  # noqa: E402
from cluster import upload  # noqa: E402
from reachy_retarget.acquire import CatalogEntry  # noqa: E402
from reachy_retarget.sources import iter_episodes, registry  # noqa: E402

PVC = "/mnt/reachy-retarget/v2/data"
FIX = Path(__file__).parent / "fixtures"


def listing(tmp_path, paths, name="l.txt"):
    p = tmp_path / name
    p.write_text("".join(f"{s} {f}\n" for f, s in paths.items()))
    return p


def behavior_files(tasks=("task-0002",), n=200, meta=True):
    files = {f"{PVC}/raw/behavior/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf": 1,
             f"{PVC}/raw/behavior/2025-challenge-demos/meta/tasks.jsonl": 1}
    for t in tasks:
        for i in range(n):
            files[f"{PVC}/raw/behavior/2025-challenge-demos/data/{t}/episode_{i:08d}.parquet"] = 10
            if meta:
                files[f"{PVC}/raw/behavior/2025-challenge-demos/meta/episodes/{t}/episode_{i:08d}.json"] = 1
    return files


def covered(jobs):
    """{(path, episode position)} built by the jobs, given each unit's episode count."""
    out = []
    for j in jobs:
        a = j["argv"]
        paths = [a[i + 1] for i, x in enumerate(a) if x == "--path"]
        k, n = map(int, a[a.index("--shard") + 1].split("/"))
        out.append((paths, k, n))
    return out


def test_read_listing_normalizes_paths(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text(f"5 {PVC}/raw/a/b.bin\n7 roboverse/x.pkl\nraw/c/d\n3 ./derived/bigym/r/T/u.npz\n\n")
    assert mf.read_listing(p) == {"raw/a/b.bin": 5, "raw/roboverse/x.pkl": 7, "raw/c/d": 0,
                                  "derived/bigym/r/T/u.npz": 3}
    q = tmp_path / "q.txt"
    q.write_text("1 /elsewhere/x\n")
    with pytest.raises(ValueError):
        mf.read_listing(q)


def test_behavior_task_is_sharded_to_target_size(tmp_path):
    files = mf.read_listing(listing(tmp_path, behavior_files(("task-0002", "task-0003"))))
    jobs = mf.build_jobs("behavior", files)
    assert len(jobs) == 14  # 200 episodes -> ceil(200 / 30) = 7 shards per task
    shards = covered(jobs)
    for task in ("task-0002", "task-0003"):
        mine = [(k, n) for paths, k, n in shards if paths == [f"{{pvc}}/data/raw/behavior/2025-challenge-demos/data/{task}"]]
        assert sorted(mine) == [(k, 7) for k in range(7)]
    for j in jobs:
        assert "--physics" not in j["argv"] and j["publish"] == "datasets/w2-v1"
        assert j["argv"][:4] == ["-m", "reachy_retarget.build", "--family", "behavior"]
        assert j["argv"][-2:] == ["--out", "{out}"] and j["_stats"]["episodes"] <= 30
        assert j["id"].startswith("build-w2-behavior-") and j["timeout_s"] >= 3600
    assert [j["id"] for j in mf.build_jobs("behavior", files)] == [j["id"] for j in jobs]  # stable ids
    assert len({j["id"] for j in jobs}) == len(jobs)


def test_behavior_checks_report_missing_inputs(tmp_path):
    files = behavior_files(("task-0002",), n=3)
    files.pop(f"{PVC}/raw/behavior/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf")
    files.pop(f"{PVC}/raw/behavior/2025-challenge-demos/meta/episodes/task-0002/episode_00000001.json")
    files[f"{PVC}/raw/behavior/2025-challenge-demos/meta/episodes/task-0000/episode_00000000.json"] = 1
    problems = mf.check_inputs("behavior", mf.read_listing(listing(tmp_path, files)))
    text = "\n".join(problems)
    assert "r1pro.urdf: missing" in text
    assert "task-0002/episode_00000001.parquet: episode metadata missing" in text
    assert "data/task-0000/: no episode parquet files" in text
    assert "2025-challenge-rawdata" in text


def test_roboverse_packs_small_files_and_sets_physics_per_benchmark(tmp_path):
    files = {f"raw/roboverse/trajs/rlbench/{t}/v2/franka_v2.pkl.gz": 1000 for t in ("a", "b", "c", "d", "close_box")}
    files.update({f"raw/roboverse/trajs/calvin/calvin_traj_ann/env_A_out/task_{i}_v2.pkl": 15 * 49_500
                  for i in range(5)})
    for rel in ("robots/franka/urdf/franka_panda.urdf", "robots/franka/mjcf/panda.xml",
                "robots/franka_calvin/panda_longer_finger.urdf", "trajs/calvin/calvin_traj_ann/ann_dict.npy"):
        files[f"raw/roboverse/{rel}"] = 1
    jobs = mf.build_jobs("roboverse", files)
    calvin = [j for j in jobs if "/calvin/" in " ".join(j["argv"])]
    rlbench = [j for j in jobs if "/rlbench/" in " ".join(j["argv"])]
    assert all("--physics" not in j["argv"] for j in calvin) and all("--physics" in j["argv"] for j in rlbench)
    assert not any("/calvin/" in " ".join(j["argv"]) and "/rlbench/" in " ".join(j["argv"]) for j in jobs)
    box = [j for j in rlbench if "close_box" in " ".join(j["argv"])]
    assert len(box) == 4 and all(j["argv"].count("--path") == 1 for j in box)  # 100 demos -> 4 shards
    small = [j for j in rlbench if j not in box]
    assert [j["argv"].count("--path") for j in small] == [3, 1]  # 10 + 10 + 10, then 10
    assert [j["argv"].count("--path") for j in calvin] == [2, 2, 1]  # 15 estimated windows per file
    assert sum(j["_stats"]["episodes"] for j in jobs) == 4 * 10 + 100 + 5 * 15
    assert mf.check_inputs("roboverse", files) == []
    del files["raw/roboverse/robots/franka/mjcf/panda.xml"]
    assert mf.check_inputs("roboverse", files) == ["raw/roboverse/robots/franka/mjcf/panda.xml: missing"]


def test_molmobot_counts_from_catalog_and_robot_check():
    pkg = CatalogEntry(id="molmobot/Cfg/train/part0/house_1", family="molmobot", path="Cfg/train/part0/house_1",
                       url="u", revision="r", sha256=None, size=None, license="x", kind="range_package",
                       content="low_dim", usable=True,
                       source={"members": [{"name": "house_1/trajectories_batch_1_of_2.h5", "trajectories": 40},
                                           {"name": "house_1/trajectories_batch_2_of_2.h5", "trajectories": 4}]})
    tofu = CatalogEntry(id="molmobot/Cfg/train/part0/house_3", family="molmobot", path="Cfg/train/part0/house_3",
                        url="u", revision="r", sha256=None, size=None, license="x", kind="range_package",
                        content="low_dim", usable=True, meta={"est_trajectories": 12})
    cat = {pkg.id: pkg, tofu.id: tofu}
    files = {"raw/molmobot/Cfg/train/part0/house_1/trajectories_batch_1_of_2.h5": 1,
             "raw/molmobot/Cfg/train/part0/house_1/trajectories_batch_2_of_2.h5": 1,
             "raw/molmobot/Cfg/train/part0/house_2/trajectories_batch_1_of_1.h5": 1}
    units = {u.path: (u.episodes, u.exact) for u in mf.units_molmobot(files, cat)}
    assert units == {"raw/molmobot/Cfg/train/part0/house_1/trajectories_batch_1_of_2.h5": (40, True),
                     "raw/molmobot/Cfg/train/part0/house_1/trajectories_batch_2_of_2.h5": (4, True),
                     "raw/molmobot/Cfg/train/part0/house_2/trajectories_batch_1_of_1.h5": (5, False)}
    tofu_files = {f"raw/molmobot/Cfg/train/part0/house_3/trajectories_batch_{i}_of_3.h5": 1 for i in (1, 2)}
    assert {u.episodes for u in mf.units_molmobot(tofu_files, cat)} == {6}  # est_trajectories 12 over 2 files
    files["raw/molmobot/Cfg/train/part0/house_3/package_members.json"] = 1
    jobs = mf.build_jobs("molmobot", files, catalog=cat)
    assert [(j["argv"].count("--path"), j["argv"][j["argv"].index("--shard") + 1]) for j in jobs] == \
        [(1, "0/2"), (1, "1/2"), (2, "0/1")]
    problems = mf.check_inputs("molmobot", files, cat)
    assert any("raw/molmobot/robots/rby1m/20251224/: robot assets missing" in p for p in problems)
    files["raw/molmobot/robots/rby1m/20251224/MANIFEST.json"] = 1
    files["raw/molmobot/robots/franka_droid/20260127/MANIFEST.json"] = 1
    assert mf.check_inputs("molmobot", files, cat) == []


def test_bigym_variants_publish_separately(tmp_path):
    files = {f"derived/bigym/replay-v1/MovePlate/{i:032x}.npz": 1 for i in range(50)}
    files.update({f"derived/bigym/replay-v1/DrawerTopOpen/{i:032x}.npz": 1 for i in range(20)})
    files.update({f"derived/bigym/replay-v1-clipped/MovePlate/{i:032x}.npz": 1 for i in range(3)})
    files["derived/bigym/replay-v1/assets/mesh.obj"] = 1
    files["derived/bigym/replay-v1/summary.jsonl"] = 1
    files[f"derived/bigym/replay-v1.partial-0123abcd/MovePlate/{0:032x}.npz"] = 1  # failed upload staging: ignored
    jobs = mf.build_jobs("bigym", files)
    clipped = [j for j in jobs if "clipped" in j["argv"][5]]
    assert len(clipped) == 1 and clipped[0]["publish"] == "datasets/w2-v1-bigym-clipped"
    assert all(j["publish"] == "datasets/w2-v1" and "--physics" in j["argv"] for j in jobs if j not in clipped)
    assert sum(j["_stats"]["episodes"] for j in jobs) == 73
    problems = mf.check_inputs("bigym", files)
    assert problems == ["derived/bigym/replay-v1-clipped/assets/: missing (no SceneRef, kinematic-only route)"]


def test_mobilemanibench_and_robocasa_units():
    base = "raw/mobilemanibench/G1_Robot/Open/partnet/box/0001/box_1/train_0"
    files = {f"{base}/trajectories/traj_{t}/episode_{e}/state_infos.pkl": 1 for t in range(2) for e in range(20)}
    files[f"{base}/params/env.yaml"] = 1
    (u,) = mf.units_mobilemanibench(files)
    assert (u.path, u.episodes) == (base, 40)
    assert "env.yaml" not in " ".join(mf.check_inputs("mobilemanibench", files))
    rc = "raw/robocasa/v1.0/pretrain/atomic/Task/20250820"
    files = {f"{rc}/lerobot/extras/episode_{i:06d}/states.npz": 1 for i in range(50)}
    (u,) = mf.units_robocasa(files)
    assert (u.path, u.episodes, u.physics) == (rc, 50, True)
    problems = mf.check_inputs("robocasa", files)
    assert any("dataset_meta.json missing" in p for p in problems) and any("robosuite-1.5.2-py3-none-any.whl: missing" in p for p in problems)


def test_write_build_manifest_and_cli(tmp_path):
    lst = listing(tmp_path, behavior_files(("task-0002",), n=40))
    out = tmp_path / "m.jsonl"
    mf.main(["build", "behavior", "--listing", str(lst), "--out", str(out)])
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 2 and all(set(x) == {"id", "argv", "publish", "timeout_s"} for x in lines)
    s = mf.write_build_manifest("behavior", [lst], out, catalog={})
    assert s["units"] == 1 and s["episodes_est"] == 40 and s["jobs"] == 2


# ---------------------------------------------------------------- adapter selection and multi-path build

def test_selection_is_implemented_by_the_adapters():
    for fam in ("behavior", "mobilemanibench", "robocasa", "roboverse", "molmobot", "bigym"):
        registry.adapter(fam)
        assert fam in registry._SELECTS, fam


def test_bigym_select_skips_unselected_records(monkeypatch):
    from reachy_retarget.sources import bigym
    read = []
    real = bigym.read_record
    monkeypatch.setattr(bigym, "read_record", lambda p: read.append(Path(p).name) or real(p))
    out = list(iter_episodes("bigym", FIX / "bigym", select=lambda i: i == 1, with_scene=False))
    assert len(out) == 3 and out[0] is None and out[2] is None and out[1] is not None
    assert len(read) == 1


def test_roboverse_select_positions():
    from reachy_retarget.sources import roboverse as rv
    files = {rel: FIX / "roboverse" / rel for rel in (rv.FRANKA_URDF, rv.FRANKA_MJCF, rv.CALVIN_URDF, rv.ANN_DICT)}
    path = FIX / "roboverse/trajs/rlbench/pick_and_lift/v2/franka_v2.pkl.gz"
    assert list(iter_episodes("roboverse", path, robot_files=files, select=lambda i: False)) == [None]


def test_build_accepts_several_paths(tmp_path, monkeypatch):
    from reachy_retarget import build

    def fake_iter(family, path, select=None, **kw):
        for i in range(3):
            if not select(i):
                yield None
            elif "bad" in path and i == 1:
                raise OSError("unreadable")
            else:
                yield f"{Path(path).name}:{i}"

    def fake_process(src, **kw):
        return {"family": "f", "episode_id": src, "K": {"passed": True}, "P": None}

    monkeypatch.setattr(build, "iter_episodes", fake_iter)
    monkeypatch.setattr(build, "process_source", fake_process)
    paths = [str(tmp_path / "raw/f/a.bin"), str(tmp_path / "raw/f/bad.bin"), str(tmp_path / "raw/f/c.bin")]
    counts = build.build("f", paths, (0, 1), str(tmp_path / "out"))
    assert counts["episodes"] == 7 and counts["errors"] == 1 and counts["K"] == 7
    recs = sorted(p.name for p in (tmp_path / "out/records/f").iterdir())
    assert recs == ["f__a.bin.0-of-1.jsonl", "f__bad.bin.0-of-1.jsonl", "f__c.bin.0-of-1.jsonl"]
    bad = [json.loads(x) for x in (tmp_path / "out/records/f/f__bad.bin.0-of-1.jsonl").read_text().splitlines()]
    assert [r.get("status") for r in bad] == [None, "read_error"]
    monkeypatch.setattr(sys, "argv", ["build"])
    build.main(["--family", "f", "--path", paths[0], "--path", paths[2], "--shard", "1/2",
                "--out", str(tmp_path / "out2")])
    assert len(list((tmp_path / "out2/records/f").iterdir())) == 2


# ---------------------------------------------------------------- upload

@pytest.fixture
def local_pod(tmp_path, monkeypatch):
    """A fake pod: scripts run locally with ``sh`` and the PVC data root mapped into ``tmp_path``.

    ``sha256sum`` is provided by ``shasum -a 256`` where coreutils are missing (macOS)."""
    pvc = tmp_path / "pvc"
    (pvc / "data/derived").mkdir(parents=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    if shutil.which("sha256sum") is None:
        (bindir / "sha256sum").write_text('#!/bin/sh\nexec shasum -a 256 "$@"\n')
        (bindir / "sha256sum").chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}
    calls = []

    def local(script):
        return script.replace(upload.DATA, str(pvc / "data"))

    def run(pod, script):
        calls.append(("run", pod))
        if "df -B1" in script:  # preflight: emulate the GNU df output used on the pods
            dest = local(script).split("dest=", 1)[1].split("\n", 1)[0].strip("'")
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            exists = Path(dest).exists()
            sums = Path(dest) / upload.SUMS
            digest = upload.sha256(sums) if sums.exists() else "none"
            return f"{'exists' if exists else 'absent'}\n{digest}\n{10**12}\n"
        return subprocess.run(["sh", "-c", local(script)], capture_output=True, text=True, check=True, env=env).stdout

    def stream(pod, script, write):
        calls.append(("stream", pod))
        proc = subprocess.Popen(["sh", "-c", local(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env)
        write(proc.stdin)
        proc.stdin.close()
        out, err = proc.stdout.read(), proc.stderr.read()
        if proc.wait():
            raise RuntimeError(err.decode())
        return out.decode()

    return pvc, run, stream, calls


def replay_dir(tmp_path):
    src = tmp_path / "local/replay-v1"
    (src / "MovePlate").mkdir(parents=True)
    (src / "assets").mkdir()
    (src / "MovePlate/c795.npz").write_bytes(os.urandom(5000))
    (src / "assets/plate.obj").write_text("v 0 0 0\n")
    (src / "summary.jsonl").write_text("{}\n")
    return src


def test_upload_streams_verifies_and_never_overwrites(tmp_path, local_pod):
    pvc, run, stream, calls = local_pod
    src = replay_dir(tmp_path)
    res = upload.upload_dir(src, "derived/bigym/replay-v1", "pod-a", run=run, stream=stream)
    assert res["state"] == "uploaded" and res["files"] == 3
    dest = pvc / "data/derived/bigym/replay-v1"
    for rel in ("MovePlate/c795.npz", "assets/plate.obj", "summary.jsonl"):
        assert (dest / rel).read_bytes() == (src / rel).read_bytes()
    assert (dest / upload.SUMS).read_text() == upload.sums_text(upload.local_manifest(src))
    info = json.loads((dest / upload.INFO).read_text())
    assert info["files"] == 3 and info["sums_sha256"] == res["sums_sha256"]
    assert not list(dest.parent.glob("*.partial-*"))
    assert upload.verify_remote("derived/bigym/replay-v1", "pod-a", run=run, expected_files=3)["files"] == 3
    # the adapter's asset lookup finds assets/ next to the task folders of the uploaded tree
    from reachy_retarget.sources.bigym import _asset_dir
    assert _asset_dir(dest / "MovePlate/c795.npz", None) == dest / "assets"
    # same content again: reported, not re-sent; different content: refused
    assert upload.upload_dir(src, "derived/bigym/replay-v1", "pod-a", run=run, stream=stream)["state"] == \
        "already_present"
    (src / "summary.jsonl").write_text("changed\n")
    with pytest.raises(FileExistsError):
        upload.upload_dir(src, "derived/bigym/replay-v1", "pod-a", run=run, stream=stream)
    assert sum(1 for c in calls if c[0] == "stream") == 1


def test_upload_detects_corruption_and_keeps_staging(tmp_path, local_pod):
    pvc, run, stream, _ = local_pod
    src = replay_dir(tmp_path)
    manifest = upload.local_manifest(src)
    manifest["summary.jsonl"] = {**manifest["summary.jsonl"], "sha256": "0" * 64}  # file changed after hashing
    with pytest.raises(RuntimeError, match="staging folder"):
        upload.upload_dir(src, "derived/bigym/replay-v1", "pod-a", run=run, stream=stream, manifest=manifest)
    assert not (pvc / "data/derived/bigym/replay-v1").exists()
    assert len(list((pvc / "data/derived/bigym").glob("replay-v1.partial-*"))) == 1


def test_upload_refuses_raw_and_low_space(tmp_path, local_pod):
    _, run, stream, _ = local_pod
    src = replay_dir(tmp_path)
    for bad in ("raw/bigym/x", "../x", "/", ""):
        with pytest.raises(ValueError):
            upload.upload_dir(src, bad, "pod-a", run=run, stream=stream)

    def full(pod, script):
        return "absent\nnone\n1000\n"
    with pytest.raises(OSError, match="reserve"):
        upload.upload_dir(src, "derived/bigym/replay-v1", "pod-a", run=full, stream=stream)
