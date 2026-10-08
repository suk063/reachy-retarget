"""Offline tests of the BEHAVIOR-1K 2025 challenge adapter on a tiny fixture.

Fixture (``tests/fixtures/behavior``, MIT), mirroring the published layout: 51 rows (every
200th frame, the contiguous frames 1160-1199 where the radio is carried, and the last
frame) of ``2025-challenge-demos`` ``task-0000/episode_00000010.parquet`` (turning_on_radio),
its trimmed episode metadata JSON, the unchanged skill annotation, the first line of
``meta/tasks.jsonl``, the ``terminated``/``reward``/``truncated`` rows of the raw HDF5 and
the pinned ``r1pro.urdf`` without visual/collision/inertial elements. Source URLs and
sha256 are in each file's ``fixture_note``.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from reachy_retarget.acquire import load_catalog
from reachy_retarget.sources import families, iter_episodes
from reachy_retarget.sources.behavior import EEF_TO_CONTRACT, STATE, R1ProFK, carry_intervals, task_info_layout

FIX = Path(__file__).parent / "fixtures" / "behavior"
DEMOS = FIX / "2025-challenge-demos"
PARQUET = DEMOS / "data" / "task-0000" / "episode_00000010.parquet"
URDF = FIX / "omnigibson-robot-assets" / "models" / "r1pro" / "urdf" / "r1pro.urdf"


def episode(**kw):
    eps = list(iter_episodes("behavior", PARQUET, urdf=URDF, **kw))
    assert len(eps) == 1
    return eps[0]


def test_registry_and_catalog():
    assert "behavior" in families()
    cat = {k: e for k, e in load_catalog().items() if e.family == "behavior"}
    pqs = [e for e in cat.values() if e.path.endswith(".parquet")]
    assert len(pqs) == 10000 and all(e.sha256 and len(e.sha256) == 64 and e.size for e in pqs)
    for e in cat.values():  # JSON files are plain git files: pinned by git blob sha1 when no sha256 was computed
        assert e.size and (e.sha256 or len(e.digests["git_blob_sha1"]) == 40) and e.kind == "file"
    assert not any(s in e.path for e in cat.values() for s in ("videos/", ".mp4", "episodes_stats"))
    for e in pqs:  # every parquet has its metadata and annotation catalogued
        tail = e.path.split("/data/")[1][:-len(".parquet")]
        assert f"behavior/2025-challenge-demos/meta/episodes/{tail}.json" in cat
        assert f"behavior/2025-challenge-demos/annotations/{tail}.json" in cat
    raw = [e for e in cat.values() if "2025-challenge-rawdata/" in e.path]
    assert len(raw) == 80  # the earlier subset's raw HDF5 (success flags); not part of the full-scale fetch
    assert "behavior/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf" in cat
    assert len({e.dataset for e in pqs}) == 50 and len({e.meta["episode_index"] for e in pqs}) == 10000


def test_task_info_layout():
    meta = json.loads((DEMOS / "meta/episodes/task-0000/episode_00000010.json").read_text())
    lay = task_info_layout(meta["task_obs_keys"])
    assert lay["dim"] == 46
    assert lay["objects"]["agent.n.01_1"] == {"real": (0, 1), "pos": (1, 4), "ori_cos": (4, 7), "ori_sin": (7, 10)}
    assert lay["objects"]["radio_receiver.n.01_1"]["in_gripper_right"] == (21, 22)
    with pytest.raises(ValueError):
        task_info_layout(["foo_bar"])


def test_episode_contents():
    ep = episode()
    T = 51
    assert ep.length == T and ep.regime == "mobile_manipulation" and ep.scene is None
    assert ep.task == "turning_on_radio" and ep.instruction.startswith("Turn on the radio")
    assert ep.episode_id == "task-0000/episode_00000010" and ep.dataset == "behavior/2025-challenge/turning_on_radio"
    assert ep.success is True and ep.provenance["success"]["first_success_step"] is not None
    assert set(ep.effectors) == {"left", "right"} and ep.effectors["left"].side_hint == "left"
    assert ep.base.shape == (T, 3) and np.allclose(ep.base_hint, ep.base[0])
    assert ep.torso_height.shape == (T,) and 0.8 < ep.torso_height.min() < ep.torso_height.max() < 2.0
    assert not ep.articulations and ep.lineage["human"] and not ep.lineage["generated"]
    assert ep.provenance["catalog_id"] is None  # fixture is a subset, not the catalogued file
    assert {k: o.role for k, o in ep.objects.items()} == {
        "radio_89": "manipulated", "coffee_table_koagbh_0": "support", "floors_ulujpr_0": "support"}
    radio = ep.objects["radio_89"]
    assert radio.valid.all() and np.allclose(np.linalg.norm(radio.pose[:, 3:], axis=1), 1)
    assert radio.geometry["category"] == "radio" and radio.geometry["bddl_inst"] == "radio_receiver.n.01_1"
    for e in ep.effectors.values():
        assert 0 <= e.opening.min() and e.opening.max() <= 1 and np.allclose(e.width, e.opening * 0.10)


def test_poses_match_urdf_fk_and_state():
    ep = episode()
    for side in ("left", "right"):
        assert ep.provenance["fk_check"][side]["max_position_error_m"] < 1e-5
        assert ep.provenance["fk_check"][side]["max_rotation_error_rad"] < 1e-5
    assert ep.provenance["base_yaw_consistency_rad"] < 1e-6
    assert ep.provenance["agent_task_info_vs_robot_pos_m"] < 1e-5
    # independent recomposition: world grasp pose = base pose * FK(eef) * contract rotation
    S = np.stack(pq.read_table(PARQUET, columns=["observation.state"])["observation.state"].to_numpy(zero_copy_only=False))
    fk = R1ProFK(URDF).forward(S)
    from scipy.spatial.transform import Rotation
    yaw = ep.base[:, 2]
    rpy = np.arctan2(S[:, slice(*STATE["robot_ori_sin"])], S[:, slice(*STATE["robot_ori_cos"])])
    assert np.allclose(rpy[:, 2], yaw, atol=1e-6)
    B = np.tile(np.eye(4), (len(S), 1, 1))
    B[:, :3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    B[:, :3, 3] = S[:, slice(*STATE["robot_pos"])]
    B[:, 2, 3] += ep.provenance["world_z_offset_m"]
    for side in ("left", "right"):
        W = B @ fk[side]
        G = ep.effectors[side].pose
        assert np.abs(G[:, :3, 3] - W[:, :3, 3]).max() < 1e-5
        assert np.abs(G[:, :3, :3] - W[:, :3, :3] @ EEF_TO_CONTRACT).max() < 1e-5
        # +z (approach) leaves the gripper link toward the fingertips; +y points finger 1 -> finger 2
        assert np.all(np.einsum("ti,ti->t", G[:, :3, 2], W[:, :3, 2]) > 0.999)


def test_floor_at_z0():
    ep = episode()
    off = ep.provenance["world_z_offset_m"]
    assert -0.01 < off < 0 and abs(np.median(ep.provenance["base_z_range_m"])) < 1e-3
    lo, hi = ep.provenance["source_base_z_range_m"]
    assert hi - lo < 2e-3
    TI = np.stack(pq.read_table(PARQUET, columns=["observation.task_info"])["observation.task_info"].to_numpy(zero_copy_only=False))
    assert np.allclose(ep.objects["radio_89"].pose[:, :3], TI[:, 11:14] + [0, 0, off], atol=1e-6)
    # floor object root sits below the walking surface; the robot's lowest grasp stays above it
    assert ep.objects["floors_ulujpr_0"].pose[0, 2] < 0 < min(e.pose[:, 2, 3].min() for e in ep.effectors.values())


def test_carry_label_and_grasp_center():
    ep = episode()
    iv = ep.provenance["carry_intervals"]["radio_89"]["right"]
    assert iv and all(b > a for a, b in iv)
    assert ep.provenance["in_gripper_true_frames"] == 0
    # the carried radio stays at a fixed offset from the right grasp center
    G, o = ep.effectors["right"].pose, ep.objects["radio_89"].pose
    a, b = iv[0]
    rel = np.einsum("tji,tj->ti", G[a:b, :3, :3], o[a:b, :3] - G[a:b, :3, 3])
    assert np.ptp(rel, axis=0).max() < 0.02 and np.linalg.norm(rel, axis=1).max() < 0.25
    # a static object is never carried
    still = carry_intervals(o[:, :3] * 0 + o[:1, :3], np.ones(ep.length, bool), ep.effectors["right"], ep.time)
    assert still == []


def test_without_urdf_and_raw(tmp_path):
    root = tmp_path / "2025-challenge-demos"
    shutil.copytree(DEMOS, root)
    ep = next(iter_episodes("behavior", root / "data" / "task-0000" / "episode_00000010.parquet", root=tmp_path))
    assert ep.success is None and ep.torso_height is None and ep.provenance["fk_check"] is None
    ref = episode()
    for side in ("left", "right"):
        assert np.allclose(ep.effectors[side].pose, ref.effectors[side].pose)
    # folder input yields every episode of the task folder
    assert len(list(iter_episodes("behavior", root / "data" / "task-0000", root=tmp_path))) == 1


def test_missing_metadata_is_an_error(tmp_path):
    root = tmp_path / "2025-challenge-demos"
    shutil.copytree(DEMOS, root)
    (root / "meta/episodes/task-0000/episode_00000010.json").unlink()
    with pytest.raises(FileNotFoundError):
        list(iter_episodes("behavior", root / "data/task-0000/episode_00000010.parquet", root=tmp_path))
