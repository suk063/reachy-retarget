"""Unit tests of the retargeting building blocks."""
import numpy as np
import pytest

from reachy_retarget.retarget import RetargetConfig, footprint
from reachy_retarget.retarget.assign import bimanual_sides, posture, tuck_posture
from reachy_retarget.retarget.gaze import eye_axes
from reachy_retarget.retarget.targets import FLIP, finger_angles, grasp_labels, tcp_targets
from reachy_retarget.retarget.wbik import FrameSolver, tcp_errors
from reachy_retarget.robot import BASE_FOOTPRINT_RADIUS, Reachy, gripper, min_clearance
from reachy_retarget.schema import Effector, ObjectTrack, SourceEpisode

CFG = RetargetConfig()


def test_config_digest_is_stable_and_sensitive():
    assert CFG.digest() == RetargetConfig().digest() and CFG.digest().startswith("sha256:")
    assert RetargetConfig(tcp_pos_tol=0.004).digest() != CFG.digest()
    with pytest.raises(ValueError):
        RetargetConfig(rate_hz=30)


def test_tcp_targets_put_the_grasp_center_on_the_source_pose():
    T = 3
    pose = np.tile(np.eye(4), (T, 1, 1))
    pose[:, :3, 3] = [0.4, -0.2, 0.8]
    src = SourceEpisode("f", "f/d", "0", "t", np.arange(T) * 0.1, {"h": Effector(pose, np.ones(T))})
    robot = Reachy.load()
    for flip in (False, True):
        X = tcp_targets(src, {"h": "right"}, {"right": flip})["right"]
        gc = X @ robot.grasp_center("right")
        np.testing.assert_allclose(gc, pose @ (FLIP if flip else np.eye(4)), atol=1e-12)


def test_grasp_labels_and_finger_squeeze():
    T = 5
    pts = np.array([[0.0, 0, 0], [0, 0, 0], [0.2, 0, 0], [0, 0, 0.03], [0, 0, 0.03]])
    closed = np.array([False, True, True, True, True])
    cube = ObjectTrack(np.tile([0, 0, 0, 1, 0, 0, 0.0], (T, 1)), np.ones(T, bool), "manipulated",
                       {"kind": "box", "half_extents": [0.02] * 3})
    table = ObjectTrack(np.tile([0, 0, -1, 1, 0, 0, 0.0], (T, 1)), np.ones(T, bool), "support")
    lab = grasp_labels(pts, closed, {"table": table, "cube": cube}, CFG)
    assert lab.tolist() == [-1, 0, -1, 0, 0]
    width = np.full(T, 0.04)
    eff = Effector(np.tile(np.eye(4), (T, 1, 1)), np.r_[1.0, np.zeros(T - 1)], width)
    angle = finger_angles(eff, lab >= 0, CFG)
    contact = gripper.width_to_angle(0.04)
    assert angle[0] == pytest.approx(contact) and angle[1] == pytest.approx(contact - CFG.squeeze_angle)
    assert gripper.angle_to_width(angle[1]) < 0.04
    eff = Effector(eff.pose, np.linspace(0, 1, T))
    np.testing.assert_allclose(finger_angles(eff, lab >= 0, CFG), gripper.opening_to_angle(np.linspace(0, 1, T)))


def test_bimanual_side_rules():
    T = 2

    def eff(y, hint=None):
        p = np.tile(np.eye(4), (T, 1, 1))
        p[:, :3, 3] = [0.5, y, 0.8]
        return Effector(p, np.ones(T), side_hint=hint)

    def src(effs, hint=None):
        return SourceEpisode("f", "f/d", "0", "t", np.arange(T) * 0.1, effs, base_hint=hint)

    assert bimanual_sides(src({"a": eff(0.2), "b": eff(-0.2)}, np.zeros(3)))[0] == {"a": "left", "b": "right"}
    # Facing -x, +world y is on the robot's right.
    assert bimanual_sides(src({"a": eff(0.2), "b": eff(-0.2)}, np.array([1.0, 0, np.pi])))[0] == {
        "a": "right", "b": "left"}
    assert bimanual_sides(src({"a": eff(0.2, "right"), "b": eff(-0.2)}))[0] == {"a": "right", "b": "left"}


def test_postures_and_eyes():
    assert min_clearance(tuck_posture(CFG)) >= CFG.self_clearance_margin
    assert np.allclose(posture("ready")[:3], 0)
    eye, axis = eye_axes()
    np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-6)  # head +x is the optical axis
    assert eye[0] > 0 and abs(eye[1]) < 1e-9


def test_footprint_clearance():
    table = ObjectTrack(np.array([[1.0, 0, 0.4, 1, 0, 0, 0]]), np.ones(1, bool), "support",
                        {"kind": "box", "half_extents": [0.2, 0.5, 0.4]})
    cube = ObjectTrack(np.array([[0.0, 2.0, 0.8, 1, 0, 0, 0]]), np.ones(1, bool), "manipulated",
                       {"kind": "box", "half_extents": [0.05, 0.05, 0.05]})
    obs = footprint.obstacles({"table": table, "cube": cube}, CFG)
    d = footprint.clearance(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.5]]), obs)
    np.testing.assert_allclose(d[0], 0.8 - BASE_FOOTPRINT_RADIUS)
    assert d[1] < 0
    # the cube (z 0.75-0.85) is at the height of the tripod column, not of the base disc
    np.testing.assert_allclose(d[2], 0.5 - np.hypot(0.05, 0.05) - footprint.body_radius(0.75, 0.85))
    assert footprint.body_radius(0.75, 0.85) < BASE_FOOTPRINT_RADIUS
    assert len(footprint.obstacles({"table": table, "cube": cube}, CFG, static_only=True).points) == 0


def test_footprint_table_top_and_aabb():
    """An elevated table top (no legs) blocks only the column; aabb centers are honoured."""
    top = ObjectTrack(np.array([[1.0, 0, 0.8, 1, 0, 0, 0]]), np.ones(1, bool), "support",
                      {"kind": "aabb", "center": [0.0, 0.0, -0.025], "half_extents": [0.2, 0.5, 0.025],
                       "frame": "body"})
    obs = footprint.obstacles({"top": top}, CFG)
    assert obs.heights[0] == pytest.approx((0.75, 0.8))
    d = footprint.clearance(np.array([[0.5, 0.0]]), obs)
    np.testing.assert_allclose(d[0], 0.3 - footprint.body_radius(0.75, 0.8))
    assert footprint.box_geometry({"kind": "aabb", "half_extents": [1, 1, 1], "frame": "world"}) is None


def test_frame_solver_respects_margins_and_box():
    robot = Reachy.load()
    q0 = posture("ready")
    target = robot.fk(q0)["right_tcp"].copy()
    target[:3, 3] += [0.05, -0.03, -0.1]
    solver = FrameSolver(CFG, ("right",), False, q0)
    q, _ = solver.solve(q0, {"right": target}, q0, np.zeros(3), 50)
    p, r = tcp_errors(q[None], {"right": target[None]})["right"]
    assert p[0] < 1e-3 and r[0] < 1e-2
    assert np.array_equal(q[3:10], q0[3:10]) and np.array_equal(q[:3], np.zeros(3))
    box = (q0 - 0.01, q0 + 0.01)
    qb, _ = solver.solve(q0, {"right": target}, q0, np.zeros(3), 50, box)
    assert np.all(np.abs(qb - q0) <= 0.01 + 1e-12)


def test_box_lsq_matches_bvls():
    from scipy.optimize import lsq_linear

    from reachy_retarget.retarget.wbik import box_lsq
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(3, 15))
        A = np.vstack([rng.normal(size=(n + 6, n)), 0.02 * np.eye(n)])
        b = rng.normal(size=len(A))
        lo, hi = -rng.uniform(0, 0.3, n), rng.uniform(0, 0.3, n)
        x, y = box_lsq(A, b, lo, hi), lsq_linear(A, b, bounds=(lo, hi), method="bvls", tol=1e-12).x
        assert np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)
        assert np.sum((A @ x - b) ** 2) <= np.sum((A @ y - b) ** 2) * (1 + 1e-9) + 1e-12


def test_source_closed_detects_a_grasp_stalled_above_half_opening():
    from reachy_retarget.retarget.targets import source_closed
    t = np.arange(40) * 0.05
    opening = np.r_[np.linspace(0.5, 1.0, 5), np.ones(10), np.linspace(1.0, 0.6, 5), np.full(15, 0.6),
                    np.linspace(0.6, 1.0, 5)]
    e = Effector(np.tile(np.eye(4), (40, 1, 1)), opening)
    closed = source_closed(e, CFG, t)
    assert not closed[:15].any()          # initial opening from a half-closed state, then open
    assert closed[21:34].all()            # stalled at 0.6 on an object
    assert not closed[36:].any()          # released
    assert not (opening < CFG.closed_opening).any()  # a fixed threshold would miss it


def test_grasp_offsets_keep_the_grasp_center_and_pad_planes():
    from reachy_retarget.retarget.targets import offset_matrix
    M = offset_matrix((True, 12.0, -30.0))
    np.testing.assert_allclose(M[:3, 3], 0)
    tilt = offset_matrix((False, 0.0, 45.0))
    np.testing.assert_allclose(tilt[:3, 1], [0, 1, 0], atol=1e-12)  # closing axis (pad normal) kept
    np.testing.assert_allclose(offset_matrix(True), FLIP, atol=1e-12)


def test_object_centric_hand_carries_a_slipping_object_rigidly():
    from reachy_retarget.retarget.targets import object_centric
    from reachy_retarget.schema.rotations import se3_inv, vec7_to_pose
    T = 30
    t = np.arange(T) * 0.1
    hand = np.tile(np.eye(4), (T, 1, 1))
    hand[:, 2, 3] = 0.8 + 0.01 * np.arange(T)
    obj = np.tile([0, 0, 0.8, 1, 0, 0, 0.0], (T, 1))
    obj[:, 2] = hand[:, 2, 3]
    obj[15:, 2] -= 0.002 * np.arange(15)  # slides 28 mm down the pads while held
    labels = np.where((t >= 0.5) & (t < 2.5), 0, -1)
    src = SourceEpisode("f", "f/d", "0", "t", t, {"h": Effector(hand, np.ones(T))},
                        objects={"o": ObjectTrack(obj, np.ones(T, bool), "manipulated",
                                                  {"kind": "box", "half_extents": [0.02] * 3})})
    path, info = object_centric(src, "h", labels, CFG)
    held = labels >= 0
    rel = se3_inv(path[held]) @ vec7_to_pose(obj[held])
    np.testing.assert_allclose(rel[:, :3, 3], rel[:1, :3, 3].repeat(held.sum(), 0), atol=1e-9)
    assert max(v[0] for v in info.values()) > 0.015  # the hand shifts with the 18 mm slip
    np.testing.assert_allclose(path[~held & (t > 2.9)], hand[~held & (t > 2.9)])  # back on the source path


def test_free_base_respects_the_footprint_constraint():
    """A target beyond a table edge pulls a free base toward the table; the linearized footprint
    constraint keeps the column clearance >= cfg.footprint_margin."""
    table = ObjectTrack(np.array([[1.0, 0, 0.775, 1, 0, 0, 0]]), np.ones(1, bool), "support",
                        {"kind": "box", "half_extents": [0.4, 0.6, 0.025]})
    obs = footprint.obstacles({"table": table}, CFG)
    q = posture("ready")
    q[:3] = [0.3, 0.0, 0.0]
    target = Reachy.load().fk(q)["right_tcp"].copy()
    target[0, 3] += 0.5  # 0.5 m further over the table than the arm reaches from here
    solver = FrameSolver(CFG, ("right",), True, posture("ready"), obstacles=obs)
    out = solver.solve(q, {"right": target}, q, q[:3], 60)[0]
    assert out[0] > 0.3 + 0.05  # the base did move toward the target
    assert footprint.clearance(out[None, :2], obs)[0] >= CFG.footprint_margin - 1e-3


def test_evaluate_demo_ranges():
    from reachy_retarget.evaluate import parse_demos
    assert parse_demos("0-2,5") == ["demo_0", "demo_1", "demo_2", "demo_5"]
    assert parse_demos(None) is None and parse_demos("all") is None
