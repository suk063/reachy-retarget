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
    np.testing.assert_allclose(d[2], 0.5 - np.hypot(0.05, 0.05) - BASE_FOOTPRINT_RADIUS)
    assert len(footprint.obstacles({"table": table, "cube": cube}, CFG, static_only=True).points) == 0


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
