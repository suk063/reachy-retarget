"""End-to-end retargeting of synthetic source episodes (no data files, no network)."""
import numpy as np
import pytest

from reachy_retarget.retarget import RetargetConfig, retarget
from reachy_retarget.retarget.gaze import gaze_error
from reachy_retarget.robot import LOWER, UPPER, VELOCITY, Reachy
from reachy_retarget.schema import Effector, ObjectTrack, SourceEpisode, read_episode, write_episode
from reachy_retarget.schema.rotations import pose_to_vec7, se3_inv, so3_log, vec7_to_pose

HZ_SRC = 20.0
CUBE = 0.02  # half extent
BOX = {"kind": "box", "half_extents": [CUBE] * 3}
TABLE = ObjectTrack(np.tile([0.65, 0.0, 0.375, 1, 0, 0, 0], (1, 1)), np.ones(1, bool), "support",
                    {"kind": "box", "half_extents": [0.35, 0.6, 0.375]})  # top at 0.75, edge at x = 0.3


def top_down(p, yaw=0.0):
    """Grasp-center pose: approach straight down, closing axis rotated by yaw from world +y."""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    T = np.eye(4)
    T[:3, :3] = np.c_[np.cross(y, z), y, z]
    T[:3, 3] = p
    return T


def _segment(a, b, n):
    return np.linspace(a, b, n, endpoint=False)


def pick_place(start, goal, *, lift=0.15, offset=(0.0, 0.0, 0.0)):
    """Grasp-center path, opening and width of a top-down pick-and-place (T = 90 at 20 Hz),
    plus the held-cube track (object centre at the grasp center while closed)."""
    start, goal, off = np.asarray(start, float), np.asarray(goal, float), np.asarray(offset, float)
    up = np.array([0, 0, lift])
    path = np.concatenate([
        _segment(start + up, start, 15), np.repeat(start[None], 12, 0),          # approach, close
        _segment(start, start + up, 12), _segment(start + up, goal + up, 20),  # lift, carry
        _segment(goal + up, goal, 12), np.repeat(goal[None], 10, 0),           # lower, open
        np.linspace(goal, goal + up, 9)]) + off                               # retreat
    T = len(path)
    opening = np.ones(T)
    opening[15:27] = np.linspace(1, 0.3, 12)
    opening[27:69] = 0.3
    opening[69:79] = np.linspace(0.3, 1, 10)
    width = np.clip(0.08 * opening / 1.0, 2 * CUBE, None)
    width[opening > 0.99] = 0.08
    held = np.zeros(T, bool)
    held[21:72] = True
    obj = np.empty((T, 7))
    obj[:, 3:] = [1, 0, 0, 0]
    first, last = np.flatnonzero(held)[[0, -1]]
    obj[:first, :3] = path[first]
    obj[held, :3] = path[held]
    obj[last + 1:, :3] = path[last]
    poses = np.stack([top_down(p) for p in path])
    return poses, opening, width, obj


def single_arm(z_shift=0.0):
    poses, opening, width, obj = pick_place((0.40, -0.12, 0.77), (0.42, 0.08, 0.77), offset=(0, 0, z_shift))
    T = len(poses)
    return SourceEpisode(
        family="synthetic", dataset="synthetic/pick", episode_id=f"single{z_shift:g}", task="pick_place",
        time=np.arange(T) / HZ_SRC, effectors={"hand": Effector(poses, opening, width)},
        objects={"cube": ObjectTrack(obj, np.ones(T, bool), "manipulated", BOX),
                 "table": ObjectTrack(np.repeat(TABLE.pose, T, 0), np.ones(T, bool), "support", TABLE.geometry)},
        base_hint=np.array([0.0, 0.0, 0.0]), license="CC0-1.0", provenance={"generator": "tests"},
        lineage={"seed": f"single{z_shift:g}", "generated": True}, instruction="put the cube to the left")


def bimanual():
    lp, lo, lw, lobj = pick_place((0.36, 0.26, 0.77), (0.38, 0.14, 0.77))
    rp, ro, rw, robj = pick_place((0.36, -0.26, 0.77), (0.38, -0.14, 0.77))
    T = len(lp)
    return SourceEpisode(
        family="synthetic", dataset="synthetic/bimanual", episode_id="0", task="two_cubes",
        time=np.arange(T) / HZ_SRC, effectors={"a": Effector(rp, ro, rw), "b": Effector(lp, lo, lw)},
        objects={"left_cube": ObjectTrack(lobj, np.ones(T, bool), "manipulated", BOX),
                 "right_cube": ObjectTrack(robj, np.ones(T, bool), "manipulated", BOX),
                 "table": ObjectTrack(np.repeat(TABLE.pose, T, 0), np.ones(T, bool), "support", TABLE.geometry)},
        base_hint=np.array([0.0, 0.0, 0.0]), lineage={"seed": "bimanual"})


def mobile():
    """The source base drives 1 m forward while carrying the hand, then the hand picks."""
    poses, opening, width, obj = pick_place((0.40, -0.12, 0.77), (0.42, 0.08, 0.77))
    drive = 40
    T = drive + len(poses)
    base = np.zeros((T, 3))
    base[:drive, 0] = np.linspace(-1.0, 0.0, drive)
    rel = poses[0].copy()
    rel[:3, 3] -= base[drive]
    hand = np.concatenate([np.stack([rel.copy() for _ in range(drive)]), poses])
    hand[:drive, :3, 3] += np.c_[base[:drive, :2], np.zeros(drive)]
    cube = np.concatenate([np.repeat(obj[:1], drive, 0), obj])
    return SourceEpisode(
        family="synthetic", dataset="synthetic/mobile", episode_id="0", task="drive_and_pick",
        time=np.arange(T) / HZ_SRC,
        effectors={"hand": Effector(hand, np.r_[np.ones(drive), opening], np.r_[np.full(drive, 0.08), width])},
        objects={"cube": ObjectTrack(cube, np.ones(T, bool), "manipulated", BOX),
                 "table": ObjectTrack(np.repeat(TABLE.pose, T, 0), np.ones(T, bool), "support", TABLE.geometry)},
        base=base, regime="mobile_manipulation", lineage={"seed": "mobile"})


@pytest.fixture(scope="module")
def single():
    return retarget(single_arm())


@pytest.fixture(scope="module")
def both():
    return retarget(bimanual())


@pytest.fixture(scope="module")
def moving():
    return retarget(mobile())


def _speed_ok(ep):
    dq = np.abs(np.diff(ep.q, axis=0)) / np.diff(ep.time)[:, None]
    return np.all(dq <= VELOCITY * (1 + 1e-6))


def test_single_arm_pick_place(single):
    cfg = RetargetConfig()
    assert single.status == "ok", single.reasons
    ep = single.episode
    assert ep.tier["K"] == {"passed": True, "reasons": []}, ep.tier
    side = "right" if "right_arm" in ep.body_parts else "left"
    assert set(ep.reference.tcp) == {side}
    i = ("left", "right").index(side)
    assert np.nanmax(ep.validation["tcp_pos_residual"][:, i]) < cfg.tcp_pos_tol
    assert np.nanmax(ep.validation["tcp_rot_residual"][:, i]) < cfg.tcp_rot_tol
    assert _speed_ok(ep)
    assert np.all(ep.q[:, 3:] >= LOWER[3:] - 1e-9) and np.all(ep.q[:, 3:] <= UPPER[3:] + 1e-9)
    assert ep.validation["self_clearance"].min() >= cfg.min_self_clearance
    # The gripper closes on the cube (squeeze below contact) and reopens.
    held = ep.validation["grasp_object"][:, i] >= 0
    assert held.sum() > 20
    open_w = ep.gripper_width[:, i]
    assert open_w[held].max() < 2 * CUBE < open_w[~held].max()
    # Object pose in the Reachy grasp-center frame stays constant while held.
    G = ep.tcp_world[side] @ Reachy.load().grasp_center(side)
    rel = se3_inv(G[held]) @ vec7_to_pose(ep.objects["cube"].pose[held])
    assert np.linalg.norm(rel[:, :3, 3] - rel[0, :3, 3], axis=1).max() < 0.005
    assert np.linalg.norm(so3_log(np.swapaxes(rel[0, :3, :3], 0, 1) @ rel[:, :3, :3]), axis=1).max() < 0.03
    # Fixed base: the base never moves and stays clear of the table.
    assert np.ptp(ep.q[:, :3], axis=0).max() == 0 and "base" not in ep.body_parts
    assert ep.extra["tier_k_metrics"]["min_footprint_clearance"] >= 0
    assert ep.retarget_config == cfg.digest() and ep.lineage["seed"] == "single0"
    assert single.diagnostics["ik"]["ms_per_source_frame"] < 50


def test_bimanual_assignment_by_lateral_position(both):
    assert both.status == "ok", both.reasons
    ep = both.episode
    assert both.diagnostics["assignment"]["sides"] == {"a": "right", "b": "left"}
    assert {"left_arm", "right_arm", "head"} <= set(ep.body_parts)
    assert ep.tier["K"]["passed"], ep.tier["K"]["reasons"]
    for i, cube in enumerate(("left_cube", "right_cube")):
        held = ep.validation["grasp_object"][:, i]
        assert np.array(ep.extra["grasp_object_ids"])[held[held >= 0]].tolist() == [cube] * int((held >= 0).sum())


def test_mobile_source_moves_the_base(moving):
    assert moving.status == "ok", moving.reasons
    ep = moving.episode
    assert "base" in ep.body_parts and ep.regime == "mobile_manipulation"
    assert np.ptp(ep.q[:, 0]) > 0.8
    assert np.abs(ep.q[:, :2] - ep.reference.base[:, :2]).max() < 0.15
    assert ep.tier["K"]["passed"], ep.tier["K"]["reasons"]
    assert _speed_ok(ep)


def test_unreachable_target_fails_k_but_returns_episode():
    res = retarget(single_arm(z_shift=2.2))
    assert res.status == "ok" and res.episode is not None
    k = res.episode.tier["K"]
    assert not k["passed"]
    assert any("TCP position residual" in r for r in k["reasons"])


def test_more_than_two_effectors_fails():
    src = single_arm()
    e = src.effectors["hand"]
    src.effectors.update(x=e, y=e)
    res = retarget(src)
    assert res.status == "failed" and res.episode is None and "effectors" in res.reasons[0]


def test_gaze_points_at_the_held_object(single):
    """The head turns toward the held cube. Where the cube lies below the neck's pitch range,
    pitch saturates at its limit (minus the margin) and yaw is still optimal."""
    cfg = RetargetConfig()
    ep = single.episode
    side = 1 if "right_arm" in ep.body_parts else 0
    held = ep.validation["grasp_object"][:, side] >= 0
    cube = ep.objects["cube"].pose[held, :3]
    q = ep.q[held]
    err = gaze_error(q, cube)
    saturated = q[:, 18] >= UPPER[18] - cfg.gaze_margin - 1e-6
    assert np.all(saturated | (err < 0.05))
    for dyaw in (-0.05, 0.05):
        qy = q.copy()
        qy[:, 19] += dyaw
        assert np.all(gaze_error(qy, cube) > err - 2e-3)
    neck = ep.q[:, 17:20]
    assert np.all(neck >= LOWER[17:20]) and np.all(neck <= UPPER[17:20])
    assert np.allclose(neck[:, 0], 0) and "head" in ep.body_parts
    assert np.ptp(neck[:, 2]) > 0.1  # the head follows the cube sideways


def test_round_trip(single, tmp_path):
    ep = single.episode
    path = write_episode(tmp_path / "ep.h5", ep)
    back = read_episode(path)
    assert back.tier == ep.tier and back.extra == ep.extra and back.body_parts == ep.body_parts
    np.testing.assert_allclose(back.q, ep.q)
    np.testing.assert_allclose(back.source_time, ep.source_time)
    np.testing.assert_array_equal(back.validation["grasp_object"], ep.validation["grasp_object"])
    np.testing.assert_allclose(pose_to_vec7(back.reference.tcp["right" if "right_arm" in ep.body_parts else "left"]),
                               pose_to_vec7(ep.reference.tcp["right" if "right_arm" in ep.body_parts else "left"]),
                               atol=1e-12)
    assert set(back.objects) == {"cube", "table"}


def _problem(src, side="right", cfg=None):
    from reachy_retarget.retarget.assign import posture
    from reachy_retarget.retarget.placement import PlacementProblem
    from reachy_retarget.retarget.targets import grasp_labels, source_closed
    cfg = cfg or RetargetConfig()
    labels = {k: grasp_labels(e.pose[:, :3, 3], source_closed(e, cfg, src.time), src.objects, cfg, effector=e,
                              times=src.time) for k, e in src.effectors.items()}
    return PlacementProblem(src, {next(iter(src.effectors)): side}, cfg, posture("ready"), labels)


def test_placement_candidates_are_moved_out_of_the_table():
    """Ring candidates inside the table are moved straight back from the targets to the footprint
    margin (heading kept); without the projection most of them overlap the table."""
    src = single_arm()
    raw = _problem(src, cfg=RetargetConfig(placement_project=False))
    moved = _problem(src)
    a, b = raw.candidates(), moved.candidates()
    clear_a = np.array([raw.footprint_clearance(p) for p in a])
    clear_b = np.array([moved.footprint_clearance(p) for p in b])
    assert (clear_a < 0).sum() > len(a) // 3
    assert (clear_b >= 0).mean() > 0.9  # the rest would need more than placement_project_max
    moved_far = np.linalg.norm(b[:, :2] - a[:, :2], axis=1)
    assert ((clear_b >= 0) | (moved_far >= RetargetConfig().placement_project_max - 0.02)).all()
    np.testing.assert_allclose(a[:, 2], b[:, 2])
    fixed = clear_a >= 0
    np.testing.assert_allclose(a[fixed], b[fixed])


def test_arm_contact_measures_arm_links_inside_scene_boxes():
    """The placement arm screen: a fixture box around the forearm of the scored posture is found,
    the grasped cube near the hand during its grasp window is not."""
    from dataclasses import replace as dc_replace

    src = single_arm()
    prob = _problem(src)
    rows = np.arange(len(prob.kf))
    q = np.tile(prob.nominal, (len(rows), 1))
    q[:, :3] = [0.0, 0.0, 0.0]
    assert prob.arm_contact("right", q, rows).max() == 0.0
    elbow = prob._spheres().sphere_centers(q[:1])[0][0][prob._arm_spheres("right")].mean(axis=0)
    T = src.length
    post = ObjectTrack(np.tile(np.r_[elbow, 1, 0, 0, 0], (T, 1)), np.ones(T, bool), "fixture",
                       {"kind": "box", "half_extents": [0.03, 0.03, 0.03]})
    blocked = _problem(dc_replace(src, objects={**src.objects, "post": post}))
    assert (blocked.arm_contact("right", q, rows) > 0.02).all()
