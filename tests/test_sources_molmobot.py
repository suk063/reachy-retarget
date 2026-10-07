"""Offline tests of the MolmoBot adapter and package fetcher.

Fixtures (MolmoBot-Data, ODC-BY 1.0; source package, member sha256 and kept rows are in
each file's ``fixture_note`` attribute):

* ``molmobot_rby1_door_sub.hdf5``: DoorOpeningDataGenConfig val traj_0, 16 of 153 rows.
* ``molmobot_franka_pick_sub.hdf5``: FrankaPickOmniCamConfig val traj_0, 27 of 51 rows.
* ``molmobot_robots_min.zip``: the MolmoSpaces ``rby1m`` and ``franka_droid`` robot XML
  files, the RB-Y1 finger collision meshes and LICENSE files; every other mesh is absent,
  so the adapter's placeholder route is exercised (kinematics stay exact).
"""
import base64
import hashlib
import io
import json
import pickle
import shutil
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from reachy_retarget.acquire import load_catalog
from reachy_retarget.acquire.fetch import ChecksumMismatch, InsufficientDisk, read_ledger
from reachy_retarget.sources import families, iter_episodes
from reachy_retarget.sources import molmobot as mb
from reachy_retarget.sources.molmobot_fetch import PackageEntry, fetch_packages, load_packages

FIX = Path(__file__).parent / "fixtures"
DOOR = FIX / "molmobot_rby1_door_sub.hdf5"
PICK = FIX / "molmobot_franka_pick_sub.hdf5"
ROBOTS = FIX / "molmobot_robots_min.zip"
ASSETS = {"rby1m": ROBOTS, "franka_droid": ROBOTS}


def episode(path, **kw):
    return next(iter_episodes("molmobot", path, robot_assets=ASSETS, packages={}, **kw))


def extra(path, key):
    with h5py.File(path) as f:
        return f["traj_0/obs/extra"][key][()]


def test_registered():
    assert "molmobot" in families()


def test_catalog_entries_and_packages():
    cat = {k: e for k, e in load_catalog().items() if e.family == "molmobot"}
    assert {mb.ROBOTS[n]["catalog_id"] for n in mb.ROBOTS} <= set(cat)
    assert all(e.sha256 and len(e.sha256) == 64 and e.size for e in cat.values())
    pk = load_packages()
    assert len(pk) >= 20
    configs = {p.config for p in pk.values()}
    assert any(c.startswith("RBY1") for c in configs) and any(c.startswith("Franka") for c in configs)
    for p in pk.values():
        assert p.members and all(m["name"].endswith(".h5") and len(m["sha256"]) == 64 for m in p.members)
        assert p.shard.startswith(p.config) and p.offset >= 0 and p.size > 0
        assert p.commercial_valid_episodes  # only commercial-use-listed packages were selected


def test_config_unpickler_never_runs_code():
    class Evil:
        def __reduce__(self):
            return (shutil.rmtree, ("/nonexistent-reachy-test",))

    payload = base64.b64encode(pickle.dumps({"a": Evil(), "b": np.arange(3)})).decode()
    out = mb.decode_config(payload)
    assert out["b"] == [0, 1, 2]
    assert isinstance(out["a"], dict)  # rmtree became an inert stub


def test_robot_geometry_from_partial_assets():
    rby = mb.RobotModel("rby1m", mb.RobotFiles(ROBOTS, "rby1m"))
    fr = mb.RobotModel("franka_droid", mb.RobotFiles(ROBOTS, "franka_droid"))
    assert rby.missing and fr.missing  # placeholder route
    # Values measured with the complete asset packages (docs/sources.md).
    for side in ("left", "right"):
        g = rby.grippers[side]
        np.testing.assert_allclose(g.R_fix, np.diag([-1.0, -1.0, 1.0]), atol=1e-9)
        np.testing.assert_allclose(g.offset, [0, 0, -0.016506], atol=2e-6)
        assert g.width_max == pytest.approx(0.100164, abs=2e-6)
    g = fr.grippers["gripper"]
    np.testing.assert_allclose(g.R_fix, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(g.offset, [0, 0, -0.014821], atol=2e-6)
    assert g.width_max == pytest.approx(0.086878, abs=2e-6)
    np.testing.assert_allclose(g.q_open, [0, 0])


def test_rby1_door_episode():
    ep = episode(DOOR)
    T = ep.length
    assert T == 16 and ep.task == "door_open" and ep.success is True
    assert ep.regime == "mobile_manipulation" and ep.instruction
    assert ep.base.shape == (T, 3) and ep.base_hint is None
    np.testing.assert_allclose(ep.time[1] - ep.time[0], 0.1)
    assert set(ep.effectors) == {"left", "right"}
    assert ep.effectors["left"].side_hint == "left" and ep.effectors["right"].side_hint == "right"
    for k, v in ep.provenance["fk_check"].items():
        assert v["ok"] and v["pos_max_m"] < 1.5e-3, (k, v)
    # Recorded base pose (x, y, yaw) equals the base joints.
    bp = extra(DOOR, "robot_base_pose")
    yaw = 2 * np.arctan2(bp[:, 6], bp[:, 3])
    np.testing.assert_allclose(ep.base[:, :2], bp[:, :2], atol=1e-5)
    np.testing.assert_allclose(np.angle(np.exp(1j * (ep.base[:, 2] - yaw))), 0, atol=1e-5)
    assert np.all(np.abs(np.diff(ep.base[:, 2])) < np.pi)  # unwrapped
    assert ep.torso_height.shape == (T,) and 1.0 < ep.torso_height.min() <= ep.torso_height.max() < 1.6
    (door,) = ep.articulations
    q = ep.articulations[door].qpos[:, 0]
    assert q[0] == pytest.approx(0, abs=1e-6) and q[-1] > 0.6
    h = ep.objects[f"{door}:handle"]
    assert h.valid.all() and h.role == "manipulated"
    yaw_h = 2 * np.arctan2(h.pose[:, 6], h.pose[:, 3])
    turn = np.angle(np.exp(1j * (yaw_h - yaw_h[0])))
    np.testing.assert_allclose(np.abs(turn), q - q[0], atol=2e-3)  # handle swings with the hinge
    for e in ep.effectors.values():
        assert 0 <= e.opening.min() <= e.opening.max() <= 1
        np.testing.assert_allclose(np.linalg.det(e.pose[:, :3, :3]), 1, atol=1e-9)
    assert ep.lineage["generated"] is False and ep.lineage["synthetic"] is True
    assert ep.scene is None and "molmospaces" in ep.provenance["scene_ref"]


def test_franka_pick_episode():
    ep = episode(PICK)
    T = ep.length
    assert T == 27 and ep.task == "pick" and ep.regime == "tabletop" and ep.base is None
    assert ep.torso_height is None and ep.base_hint.shape == (3,)
    bp = extra(PICK, "robot_base_pose")
    np.testing.assert_allclose(ep.base_hint[:2], bp[0, :2], atol=1e-6)
    (eff,) = ep.effectors.values()
    assert ep.provenance["fk_check"]["gripper"]["ok"]
    # Grasp center = recorded TCP (grasp_site, pedestal frame) moved 14.8 mm toward the palm.
    tcp = extra(PICK, "tcp_pose")
    Tb = np.stack([mb._pose7_T(p) for p in bp])
    site = np.stack([Tb[t] @ mb._pose7_T(tcp[t]) for t in range(T)])
    d = np.einsum("tij,tj->ti", np.transpose(site[:, :3, :3], (0, 2, 1)), eff.pose[:, :3, 3] - site[:, :3, 3])
    np.testing.assert_allclose(d, np.tile([0, 0, -0.014821], (T, 1)), atol=1.5e-3)
    assert eff.opening[0] > 0.95 and eff.opening.min() < 0.5
    (name,) = [k for k, o in ep.objects.items() if o.role == "manipulated"]
    obj = ep.objects[name]
    assert obj.valid[0] and not obj.valid[1:].any()
    np.testing.assert_allclose(obj.pose[0], extra(PICK, "obj_start")[0], atol=1e-5)
    tr = mb.rigid_grasp_track(ep, name)
    s, r = tr.geometry["held_rows"]
    assert tr.valid[s:r].all() and r == T
    goal = np.asarray(ep.provenance["pickup_obj_goal_pose"][:3])
    err = ep.provenance["final_task_info"]["position_error"]
    assert abs(np.linalg.norm(tr.pose[-1, :3] - goal) - err) < 0.01


def test_traj_selection_and_missing_robot(tmp_path):
    assert list(iter_episodes("molmobot", DOOR, robot_assets=ASSETS, packages={}, trajs=["traj_9"])) == []
    with pytest.raises(FileNotFoundError):
        next(iter_episodes("molmobot", DOOR, packages={}, root=tmp_path))


# ---------------------------------------------------------------- package fetcher

def _zstd_available():
    try:
        import zstandard  # noqa: F401
        return True
    except ImportError:
        return shutil.which("zstd") is not None


def _compress(raw: bytes) -> bytes:
    try:
        import zstandard
        return zstandard.ZstdCompressor().compress(raw)
    except ImportError:
        import subprocess
        return subprocess.run([shutil.which("zstd"), "-c", "-q"], input=raw, capture_output=True, check=True).stdout


class _Resp(io.BytesIO):
    status = 206

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.mark.skipif(not _zstd_available(), reason="needs zstandard or the zstd CLI")
def test_fetch_packages_keeps_only_h5(tmp_path):
    h5 = DOOR.read_bytes()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in [("house_1/trajectories_batch_1_of_1.h5", h5),
                           ("house_1/episode_00000000_head_camera_batch_1_of_1.mp4", b"\x00" * 4096)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    blob = _compress(buf.getvalue())
    shard = b"X" * 100 + blob + b"Y" * 50
    member = {"name": "house_1/trajectories_batch_1_of_1.h5", "sha256": hashlib.sha256(h5).hexdigest(), "size": len(h5)}
    e = PackageEntry(id="molmobot/Cfg/val/part0/house_1", config="Cfg", split="val", part=0,
                     package="Cfg_house_1.tar.zst", shard="Cfg/val_shards/00000.tar", url="https://example.invalid/s.tar",
                     shard_sha256="0" * 64, shard_size=len(shard), offset=100, size=len(blob), inflated_size=0,
                     range_sha256=hashlib.sha256(blob).hexdigest(), scene_id="part0_house_1", scene_family="test",
                     commercial_valid_episodes="*", members=(member,), revision="r", license="ODC-BY-1.0")
    seen = []

    def opener(req):
        a, b = map(int, req.get_header("Range").split("=")[1].split("-"))
        seen.append((a, b))
        return _Resp(shard[a:b + 1])

    recs = fetch_packages([e], tmp_path, reserve=0, opener=opener)
    out = tmp_path / "raw/molmobot/Cfg/val/part0/house_1/trajectories_batch_1_of_1.h5"
    assert out.read_bytes() == h5 and seen == [(100, 100 + len(blob) - 1)]
    assert not list(tmp_path.rglob("*.mp4")) and not list(tmp_path.rglob("*.part"))
    assert recs[0]["sha256_verified"] == member["sha256"]
    assert "molmobot/Cfg/val/part0/house_1/trajectories_batch_1_of_1.h5" in read_ledger(tmp_path)
    # Existing verified member: no transfer.
    fetch_packages([e], tmp_path, reserve=0, opener=lambda r: pytest.fail("network used"))
    # Corrupt range digest: nothing is kept.
    bad = PackageEntry(**{**e.__dict__, "id": "molmobot/Cfg/val/part0/house_2", "range_sha256": "f" * 64,
                          "members": ({**member, "name": "house_1/trajectories_batch_1_of_1.h5"},)})
    shutil.rmtree(tmp_path / "raw/molmobot/Cfg")
    with pytest.raises(ChecksumMismatch):
        fetch_packages([bad], tmp_path, reserve=0, opener=opener)
    assert not list((tmp_path / "raw/molmobot").rglob("*.h5*"))
    with pytest.raises(InsufficientDisk):
        fetch_packages([e], tmp_path, reserve=10 ** 18, opener=opener)


@pytest.mark.skipif(not _zstd_available(), reason="needs zstandard or the zstd CLI")
def test_fetch_refuses_ignored_range(tmp_path):
    class Full(_Resp):
        status = 200

    e = next(iter(load_packages().values()))
    with pytest.raises(Exception, match="ignored the byte range"):
        fetch_packages([e], tmp_path, reserve=0, opener=lambda r: Full(b"abc"))
    assert not list(tmp_path.rglob("*.h5"))


def test_fixture_notes_record_sources():
    for p in (DOOR, PICK):
        with h5py.File(p) as f:
            note = f.attrs["fixture_note"]
        assert "ODC-BY" in note and "sha256" in note
    assert json.loads(mb.RobotFiles(ROBOTS, "rby1m").read("MANIFEST.json"))["shard_sha256"]
