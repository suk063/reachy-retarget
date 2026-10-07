"""Tier-K checks on hand-built episodes."""
import numpy as np

from reachy_retarget.retarget.assign import posture
from reachy_retarget.retarget.config import RetargetConfig
from reachy_retarget.robot import UPPER, Reachy
from reachy_retarget.schema import DT, ObjectTrack, ReachyEpisode, Reference
from reachy_retarget.schema.rotations import planar_pose, pose_to_vec7
from reachy_retarget.validate.kinematic import check

CFG = RetargetConfig()


def episode(q, *, ref_offset=0.0, grasp=None, objects=None):
    """Episode whose right TCP reference equals the achieved TCP (shifted by ref_offset in z)."""
    q = np.asarray(q, float)
    T = len(q)
    fk = Reachy.load().fk_base(q)
    ref = planar_pose(q[:, :3]) @ fk["right_tcp"]
    ref[:, 2, 3] += ref_offset
    labels = np.full((T, 2), -1, np.int16)
    if grasp is not None:
        labels[:, 1] = grasp
    return ReachyEpisode(
        family="t", dataset="t/k", episode_id="0", task="t", time=np.arange(T) * DT, q=q, qd=np.zeros_like(q),
        tcp_base={s: fk[f"{s}_tcp"] for s in ("left", "right")}, head_base=fk["head"],
        gripper_opening=np.full((T, 2), 0.5), gripper_width=np.full((T, 2), 0.05), source_time=np.arange(T) * 0.05,
        tier={"K": {"passed": False, "reasons": []}, "P": None}, retarget_config="x",
        reference=Reference(tcp={"right": ref}), objects=objects or {}, validation={"grasp_object": labels},
        extra={"grasp_object_ids": sorted(k for k, o in (objects or {}).items() if o.role == "manipulated")})


def still(T=10):
    return np.tile(posture("ready"), (T, 1))


def test_clean_episode_passes():
    k = check(episode(still()), CFG)
    assert k["passed"] and k["reasons"] == []
    assert k["metrics"]["right_max_pos_residual"] < 1e-9
    assert k["frames"]["tcp_pos_residual"].shape == (10, 2) and np.isnan(k["frames"]["tcp_pos_residual"][:, 0]).all()


def test_residual_tolerance_is_stricter_while_holding():
    assert check(episode(still(), ref_offset=0.004), CFG)["passed"]
    held = check(episode(still(), ref_offset=0.004, grasp=np.zeros(10, int),
                         objects={"cube": ObjectTrack(np.tile([0.4, -0.2, 0.9, 1, 0, 0, 0], (10, 1)),
                                                      np.ones(10, bool))}), CFG)
    assert not held["passed"] and any("position residual" in r for r in held["reasons"])


def test_speed_and_joint_limits():
    q = still()
    q[5:, 13] += 0.1  # 0.1 rad in one 20 ms step: 5 rad/s
    k = check(episode(q), CFG)
    assert any("speed limit" in r for r in k["reasons"])
    q = still()
    q[:, 15] = UPPER[15] + 0.01
    k = check(episode(q), CFG)
    assert any("joint limit" in r for r in k["reasons"])


def test_self_clearance():
    q = still()
    q[:, 12] = 1.2  # right forearm into the back bar
    k = check(episode(q), CFG)
    assert any("self-clearance" in r for r in k["reasons"]) and k["metrics"]["min_self_clearance"] < 0


def test_base_footprint_against_static_geometry():
    T = 10
    table = ObjectTrack(np.tile([0.2, 0, 0.375, 1, 0, 0, 0], (T, 1)), np.ones(T, bool), "support",
                        {"kind": "box", "half_extents": [0.1, 0.5, 0.375]})
    floor = ObjectTrack(np.tile([0, 0, -0.01, 1, 0, 0, 0], (T, 1)), np.ones(T, bool), "support",
                        {"kind": "box", "half_extents": [5, 5, 0.01]})
    mesh = ObjectTrack(np.tile([3, 0, 0, 1, 0, 0, 0], (T, 1)), np.ones(T, bool), "fixture", {"kind": "mesh"})
    k = check(episode(still(T), objects={"table": table, "floor": floor, "shelf": mesh}), CFG)
    assert any("footprint" in r for r in k["reasons"])
    assert any("shelf" in n for n in k["metrics"]["footprint_notes"])
    k = check(episode(still(T), objects={"floor": floor}), CFG)
    assert k["passed"], k["reasons"]


def test_grasp_drift_detected():
    T = 10
    q = still(T)
    gc = planar_pose(q[:, :3]) @ Reachy.load().fk_base(q)["right_grasp"]
    pose = pose_to_vec7(gc)
    steady = ObjectTrack(pose.copy(), np.ones(T, bool), "manipulated")
    assert check(episode(q, grasp=np.zeros(T, int), objects={"cube": steady}), CFG)["passed"]
    slipping = pose.copy()
    slipping[:, 0] += np.linspace(0, 0.03, T)
    k = check(episode(q, grasp=np.zeros(T, int), objects={"cube": ObjectTrack(slipping, np.ones(T, bool))}), CFG)
    assert any("relative pose drifts" in r for r in k["reasons"])
    assert abs(k["metrics"]["max_grasp_drift_pos"] - 0.03) < 1e-9
