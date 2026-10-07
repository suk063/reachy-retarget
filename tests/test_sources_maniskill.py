"""Offline tests of the ManiSkill3 adapter on tiny subsets of the published demos.

Fixtures (``tests/fixtures/maniskill``, Apache-2.0): episode subsets of the pinned
``haosulab/ManiSkill_Demonstrations`` files, with source URL, revision and sha256 in the
h5 ``fixture_note`` attribute and the JSON ``fixture_note`` field:

* ``pickcube_mp``: PickCube-v1 motion planning ``traj_0``, ``traj_1`` (complete).
* ``peginsertion_mp``: PegInsertionSide-v1 motion planning ``traj_76``, ``traj_100`` (an
  exact duplicate pair with the same seed in the source file) and ``traj_0``, first 21 states.
* ``tworobot_rl``: TwoRobotPickCube-v1 RL ``traj_0``, first 31 states.

``panda_v2.urdf`` / ``panda_v3.urdf`` are the catalogued ManiSkill URDFs (commit baab60ed).
"""
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from reachy_retarget.acquire import load_catalog
from reachy_retarget.sources import families, iter_episodes
from reachy_retarget.sources.maniskill import (TASKS, PandaGripper, fk_crosscheck_mujoco,
                                               peg_insertion_geometry, pose_to_matrix)

FIX = Path(__file__).parent / "fixtures" / "maniskill"
URDFS = {"panda": FIX / "panda_v2.urdf", "panda_wristcam": FIX / "panda_v3.urdf"}


def episodes(name, **kw):
    return list(iter_episodes("maniskill", FIX / name / "trajectory.h5", urdfs=URDFS, **kw))


def states(name, key, group):
    with h5py.File(FIX / name / "trajectory.h5") as f:
        return {k: v[()] for k, v in f[key]["env_states"][group].items()}


def test_registry_and_catalog():
    assert "maniskill" in families()
    cat = {k: e for k, e in load_catalog().items() if e.family == "maniskill"}
    h5 = [e for e in cat.values() if e.path.endswith(".h5")]
    assert len(h5) == 31 and all(e.sha256 and len(e.sha256) == 64 and e.size for e in cat.values())
    assert all(e.path.endswith(".json") is False or e.kind == "metadata" for e in cat.values())
    # every trajectory has its metadata, and excluded embodiments are not catalogued
    assert all(e.id[:-3] + ".json" in cat for e in h5)
    assert not any(t in e.path for e in cat.values() for t in ("PushT", "DrawTriangle", "AnymalC", ".mp4", ".zip", ".pt"))


def test_panda_contract_frame_from_urdf():
    for path in URDFS.values():
        g = PandaGripper(path)
        np.testing.assert_allclose(g.R_fix, np.diag([-1.0, -1.0, 1.0]))
        np.testing.assert_allclose(g.offset, [0, 0, 0.0584 + 0.04525], atol=1e-12)
        np.testing.assert_allclose(g.center_in_tcp, [0, 0, 0.00025], atol=1e-12)
        assert g.width_max == pytest.approx(0.08) and g.ignored_mimic == ["panda_finger_joint2"]
    # Width is the finger joint sum, independent of mimic resolution.
    g = PandaGripper(URDFS["panda"])
    q = np.zeros((3, 9))
    q[:, 7], q[:, 8] = [0.0, 0.01, 0.04], [0.0, 0.03, 0.02]
    out = g.forward(q, np.broadcast_to(np.eye(4), (3, 4, 4)))
    np.testing.assert_allclose(out["width"], q[:, 7] + q[:, 8], atol=1e-12)


def test_pickcube_episode():
    eps = episodes("pickcube_mp")
    assert [e.episode_id for e in eps] == ["traj_0", "traj_1"]
    ep = eps[0]
    arts = states("pickcube_mp", "traj_0", "articulations")["panda"]
    assert ep.task == "PickCube-v1" and ep.length == len(arts) == 75 and ep.scene is None
    np.testing.assert_allclose(ep.time[1], 0.05)
    assert ep.success is True and ep.instruction is None
    assert ep.lineage == {"generated": False, "source_type": "motionplanning", "human": False,
                          "initial_state": "maniskill/PickCube-v1/env_seed/0", "duplicate_of": None}
    np.testing.assert_allclose(ep.base_hint, [-0.615, 0, 0], atol=1e-6)
    (eff,) = ep.effectors.values()
    np.testing.assert_allclose(eff.width, arts[:, 20] + arts[:, 21], atol=1e-6)
    assert eff.opening.max() > 0.99 and 0.4 < eff.opening.min() < 0.5
    assert set(ep.objects) == {"cube", "table-workspace"}
    cube = ep.objects["cube"]
    assert cube.role == "manipulated" and cube.geometry["half_extents"] == [0.02] * 3
    assert ep.objects["table-workspace"].role == "support"
    assert ep.provenance["goal_markers"]["goal_site"]["radius"] == 0.025
    checks = ep.provenance["state_checks"]["panda"]
    assert checks["first_action_minus_qpos_max_rad"] == 0.0  # pd_joint_pos confirms joint order
    # While the cube is held, its center sits at the grasp center, on the closing axis.
    held = (cube.pose[:, 2] > 0.03) & (eff.opening < 0.7)
    assert held.sum() > 10
    rel = np.einsum("tji,tj->ti", eff.pose[held, :3, :3], cube.pose[held, :3] - eff.pose[held, :3, 3])
    assert np.abs(rel).max() < 0.006 and np.abs(rel[:, 1]).max() < 0.001


def test_effector_axes():
    """+z from the palm toward the fingertips, +y from finger 1 toward finger 2."""
    ep = episodes("pickcube_mp", limit=1)[0]
    s = states("pickcube_mp", "traj_0", "articulations")["panda"]
    g = PandaGripper(URDFS["panda"])
    out = g.forward(s[:, 13:22], pose_to_matrix(s[:, :7]))
    G = next(iter(ep.effectors.values())).pose
    for t in range(0, len(G), 7):
        local = lambda T: G[t, :3, :3].T @ (T[:3, 3] - G[t, :3, 3])
        hand = local(out["tcp"][t]) - [0, 0, 0.1034]  # tcp is 0.1034 m ahead of the hand
        f1, f2 = local(out["fingers"][0][t]), local(out["fingers"][1][t])
        assert hand[2] < -0.1 and abs(hand[0]) < 1e-9 and abs(hand[1]) < 1e-9
        assert f1[1] < 0 < f2[1] or (s[t, 20] < 1e-4 and s[t, 21] < 1e-4)


def test_fk_crosscheck_mujoco():
    pytest.importorskip("mujoco")
    s = states("pickcube_mp", "traj_0", "articulations")["panda"]
    g = PandaGripper(URDFS["panda"])
    q, root = s[:, 13:22], pose_to_matrix(s[:, :7])
    rep = fk_crosscheck_mujoco(g.urdf_path, q, root, g.forward(q, root))
    assert rep["available"] and rep["max_position_error_m"] < 1e-12 and rep["max_rotation_error_rad"] < 1e-9


def test_peg_insertion_geometry_and_duplicates():
    eps = {e.episode_id: e for e in episodes("peginsertion_mp")}
    assert eps["traj_100"].lineage["duplicate_of"] == "traj_76"
    assert eps["traj_76"].lineage["duplicate_of"] is None and eps["traj_0"].lineage["duplicate_of"] is None
    for e in eps.values():
        peg = e.objects["peg"]
        assert peg.role == "manipulated" and "regenerated" in e.provenance["geometry_status"]["peg"]
        np.testing.assert_allclose(peg.pose[0, 2], peg.geometry["half_extents"][1], atol=1e-6)
        assert e.objects["box_with_hole"].role == "receptacle"
        assert np.ptp(e.objects["box_with_hole"].pose, axis=0).max() < 1e-6  # kinematic
    peg, box = peg_insertion_geometry(0)
    assert 0.085 <= peg["half_extents"][0] <= 0.125 and 0.015 <= peg["half_extents"][1] <= 0.025
    assert box["hole"]["half_width"] == pytest.approx(peg["half_extents"][1] + 0.003)


def test_two_robots():
    (ep,) = episodes("tworobot_rl")
    assert ep.length == 31 and set(ep.effectors) == {"panda_wristcam-agent-0", "panda_wristcam-agent-1"}
    assert all(e.side_hint is None for e in ep.effectors.values())
    assert ep.provenance["effector_env_roles"] == {"panda_wristcam-agent-0": "left_agent",
                                                   "panda_wristcam-agent-1": "right_agent"}
    bases = ep.provenance["robot_bases_xy_yaw"]
    np.testing.assert_allclose(bases["panda_wristcam-agent-0"], [0, -0.75, np.pi / 2], atol=1e-3)
    np.testing.assert_allclose(bases["panda_wristcam-agent-1"], [0, 0.75, -np.pi / 2], atol=1e-3)
    np.testing.assert_allclose(ep.base_hint, bases["panda_wristcam-agent-0"])
    assert ep.provenance["source_type"] == "rl" and ep.lineage["human"] is False
    assert "first_action_minus_qpos_max_rad" not in ep.provenance["state_checks"]["panda_wristcam-agent-0"]


def test_rejects_non_gripper_and_bad_states(tmp_path):
    src = FIX / "pickcube_mp"
    for name in ("trajectory.h5", "trajectory.json"):
        shutil.copy(src / name, tmp_path / name)
    with h5py.File(tmp_path / "trajectory.h5", "a") as f:
        f["traj_0/env_states/articulations"].move("panda", "panda_stick")
    with pytest.raises(ValueError, match="panda_stick"):
        next(iter_episodes("maniskill", tmp_path / "trajectory.h5", urdfs=URDFS))
    with h5py.File(tmp_path / "trajectory.h5", "a") as f:
        f["traj_0/env_states/articulations"].move("panda_stick", "panda")
        s = f["traj_0/env_states/articulations/panda"]
        a = s[()]
        a[:, [16, 18]] = a[:, [18, 16]]  # swap joint4 and joint6: leaves the URDF limits
        s[...] = a
    with pytest.raises(ValueError, match="joint order"):
        next(iter_episodes("maniskill", tmp_path / "trajectory.h5", urdfs=URDFS))
    with pytest.raises(FileNotFoundError):
        next(iter_episodes("maniskill", src / "trajectory.h5", catalog={}))


def test_task_table_covers_catalogue():
    cat = load_catalog()
    tasks = {e.path.split("/")[1] for e in cat.values() if e.family == "maniskill" and e.path.startswith("demos/")}
    assert tasks == set(TASKS)
    meta = json.loads((FIX / "pickcube_mp" / "trajectory.json").read_text())
    assert "fixture_note" in meta and "sha256" in meta["fixture_note"]
