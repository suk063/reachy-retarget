"""Unit tests of the retargeting building blocks."""
import numpy as np
import pytest

from reachy_retarget.retarget import RetargetConfig, footprint
from reachy_retarget.retarget.assign import bimanual_sides, posture, tuck_posture
from reachy_retarget.retarget.gaze import eye_axes
from reachy_retarget.retarget.targets import (FLIP, finger_angles, grasp_labels, object_distance, object_extent,
                                              tcp_targets)
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


def test_union_of_boxes_has_an_envelope_and_exact_parts():
    """ManiSkill's L tool (kind "boxes"): box_geometry gives the enclosing box, distances use
    the exact parts, the extent a hand can hold is the thinnest part."""
    geom = {"kind": "boxes", "frame": "body",
            "boxes": [{"center": [0.1, 0, 0], "half_extents": [0.1, 0.025, 0.025]},
                      {"center": [0.175, 0.05, 0], "half_extents": [0.025, 0.05, 0.025]}]}
    c, h = footprint.box_geometry(geom)
    np.testing.assert_allclose(c, [0.1, 0.0375, 0])
    np.testing.assert_allclose(h, [0.1, 0.0625, 0.025])
    assert len(footprint.box_parts(geom)) == 2
    assert footprint.box_parts({**geom, "boxes": [{"center": [0, 0], "half_extents": [1, 1, 1]}]}) == []
    assert footprint.box_parts({"kind": "box", "half_extents": [1, 2, 3]})[0][1].tolist() == [1, 2, 3]
    track = ObjectTrack(np.tile([1.0, 0, 0, 1, 0, 0, 0], (3, 1)), np.ones(3, bool), "manipulated", geom)
    pts = np.array([[1.05, 0.06, 0.0],    # inside the envelope, outside both parts
                    [1.1, 0.0, 0.0],      # inside the handle
                    [1.175, 0.12, 0.0]])  # 2 cm beyond the hook
    np.testing.assert_allclose(object_distance(pts, track), [0.035, 0.0, 0.02], atol=1e-12)
    np.testing.assert_allclose(object_extent(track, np.arange(3), np.eye(3)), 0.05)


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


def test_source_closed_prefers_the_recorded_command():
    from reachy_retarget.retarget.targets import source_closed
    t = np.arange(40) * 0.05
    opening = np.r_[np.ones(10), np.linspace(1.0, 0.6, 5), np.full(20, 0.6), np.linspace(0.6, 1.0, 5)]
    command = np.r_[np.zeros(10), np.ones(25), np.zeros(5)]
    e = Effector(np.tile(np.eye(4), (40, 1, 1)), opening, command=command)
    closed = source_closed(e, CFG, t)
    assert not closed[:15].any()          # commanded at frame 10, but the fingers still close
    assert closed[15:35].all()            # stalled on the object until the release command
    assert not closed[35:].any()          # released by the command, before the fingers open
    np.testing.assert_array_equal(source_closed(e, CFG), command >= 0.5)  # without times: the command
    with pytest.raises(ValueError, match="command"):
        SourceEpisode("f", "f/d", "0", "t", t, {"h": Effector(e.pose, opening, command=2 * command)})


def test_cylinder_grasped_along_its_axis_is_rotationally_symmetric():
    from reachy_retarget.retarget.targets import (contact_width, finger_penetration, grasp_symmetry,
                                                  object_distance, offset_candidates)
    T = 10
    t = np.arange(T) * 0.1
    hand = np.tile(np.diag([1.0, -1.0, -1.0, 1.0]), (T, 1, 1))  # approach -z (top grasp), closing -y
    hand[:, :3, 3] = [0.0, 0.0, 0.9]
    can = ObjectTrack(np.tile([0, 0, 0.9, np.cos(0.2), 0, 0, np.sin(0.2)], (T, 1)), np.ones(T, bool), "manipulated",
                      {"kind": "cylinder", "radius": 0.026, "half_length": 0.04, "axis": "z"})
    src = SourceEpisode("f", "f/d", "0", "t", t, {"h": Effector(hand, np.full(T, 0.6))}, objects={"Can": can})
    labels = np.zeros(T, int)
    sym = grasp_symmetry(src, "h", labels, CFG)
    assert sym["rotational"] and sym["align_deg"] == 0.0
    thetas = sorted({o[1] for o in offset_candidates(sym, CFG)})
    assert thetas == [0.0, 45.0, 90.0, 135.0]
    for theta in thetas:  # the pads span the diameter whatever the turn
        np.testing.assert_allclose(contact_width(src, "h", (False, theta, 0.0), labels), 0.052, atol=1e-9)
    # exact cylinder distance: a point beside the rim, not the bounding-box corner
    p = np.tile([0.03 / np.sqrt(2), 0.03 / np.sqrt(2), 0.9], (T, 1))
    np.testing.assert_allclose(object_distance(p, can), 0.004, atol=1e-9)
    depth = finger_penetration(src, "h", "right", (False, 45.0, 0.0), np.full(T, -1), np.full(T, 0.0), CFG)
    assert np.all(depth >= 0)
    # a side grasp (approach across the axis) keeps the box rules
    side = np.tile(np.eye(4), (T, 1, 1))
    side[:, :3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # approach +x
    src2 = SourceEpisode("f", "f/d", "0", "t", t, {"h": Effector(side, np.full(T, 0.6))}, objects={"Can": can})
    assert not grasp_symmetry(src2, "h", labels, CFG)["rotational"]


def _cube(T, xyz=(0.0, 0.0, 0.0)):
    return ObjectTrack(np.tile([*xyz, 1, 0, 0, 0.0], (T, 1)), np.ones(T, bool), "manipulated",
                       {"kind": "box", "half_extents": [0.02] * 3})


def test_closed_empty_hand_and_brief_contact_are_not_grasps():
    T = 20
    t = np.arange(T) * 0.05
    pose = np.tile(np.eye(4), (T, 1, 1))
    pose[:, :3, 3] = [0.0, 0.0, 0.03]               # 1 cm above the cube top
    closed = np.ones(T, bool)
    objects = {"cube": _cube(T)}
    fist = Effector(pose, np.zeros(T), width=np.zeros(T))  # fingers closed on nothing: a push
    assert (grasp_labels(pose[:, :3, 3], closed, objects, CFG, effector=fist, times=t) == -1).all()
    held = Effector(pose, np.full(T, 0.5), width=np.full(T, 0.037))  # stalled on the 40 mm cube
    assert (grasp_labels(pose[:, :3, 3], closed, objects, CFG, effector=held, times=t) == 0).all()
    brief = closed & (np.arange(T) >= 5) & (np.arange(T) < 8)     # 0.15 s < grasp_min_duration_s
    assert (grasp_labels(pose[:, :3, 3], brief, objects, CFG, effector=held, times=t) == -1).all()
    flicker = closed.copy()
    flicker[[3, 6, 7, 12]] = False                                # jittering stall: gaps <= grasp_gap_s
    assert (grasp_labels(pose[:, :3, 3], flicker, objects, CFG, effector=held, times=t) == 0).all()
    flicker[8:14] = False                                         # a 0.3 s release is kept
    lab = grasp_labels(pose[:, :3, 3], flicker, objects, CFG, effector=held, times=t)
    assert (lab[8:14] == -1).all() and (lab[14:] == 0).all()
    # an aabb is an envelope (a ring nut): only its thinnest extent bounds the held part
    ring = ObjectTrack(objects["cube"].pose, np.ones(T, bool), "manipulated",
                       {"kind": "aabb", "half_extents": [0.06, 0.044, 0.01]})
    thin = Effector(pose, np.full(T, 0.3), width=np.full(T, 0.02))
    assert (grasp_labels(pose[:, :3, 3], closed, {"nut": ring}, CFG, effector=thin, times=t) == 0).all()


def test_orientation_is_strict_near_objects_and_free_far_from_them():
    from reachy_retarget.retarget.targets import hand_object_distance, orientation_weight
    T = 40
    t = np.arange(T) * 0.05
    pose = np.tile(np.eye(4), (T, 1, 1))
    pose[:, 0, 3] = np.linspace(0.4, 0.0, T)        # approaches the cube from 40 cm
    src = SourceEpisode("f", "f/d", "0", "t", t, {"h": Effector(pose, np.ones(T))}, objects={"cube": _cube(T)})
    d = hand_object_distance(src, "h")
    np.testing.assert_allclose(d[[0, -1]], [0.38, 0.0], atol=1e-12)
    none = np.full(T, -1)
    assert np.all(orientation_weight(t, none, CFG) == 1)  # without distances: strict everywhere
    w = orientation_weight(t, none, CFG, d)
    assert w[0] == CFG.free_rot_weight and np.all(w[d <= CFG.contact_strict_distance] == 1)
    assert np.all(np.diff(w) >= 0)
    grasp = np.where(t >= 1.5, 0, -1)
    wg = orientation_weight(t, grasp, CFG, np.full(T, 1.0))  # far (by distance) but held: strict
    assert np.all(wg[grasp >= 0] == 1)


def test_closed_hand_push_frames():
    from reachy_retarget.retarget.targets import push_labels
    T = 30
    t = np.arange(T) * 0.05
    pose = np.tile(np.eye(4), (T, 1, 1))
    x = np.r_[np.full(10, 0.1), np.linspace(0.1, 0.3, 20)]      # moves +x after 0.5 s
    pose[:, 0, 3] = x - 0.045                                   # 25 mm behind the cube's -x face
    cube = _cube(T)
    cube.pose[:, 0] = x
    src = SourceEpisode("f", "f/d", "0", "t", t, {"h": Effector(pose, np.zeros(T), width=np.zeros(T))},
                        objects={"cube": cube})
    push = push_labels(src, "h", np.full(T, -1), CFG, closed=np.ones(T, bool))
    assert (push[11:29] == 0).all() and (push[:9] == -1).all()
    assert (push_labels(src, "h", np.full(T, -1), CFG, closed=np.zeros(T, bool)) == -1).all()


def test_per_segment_offsets_turn_between_segment_windows():
    from reachy_retarget.retarget.targets import grasp_segments, offset_matrix, offset_track
    t = np.arange(100) * 0.05
    lab = np.full(100, -1)
    lab[10:30], lab[30:40], lab[70:90] = 0, 1, 1
    segs = grasp_segments(lab)
    assert segs == [(10, 30), (30, 40), (70, 90)]
    a, b = (False, 0.0, 0.0), (True, 0.0, 30.0)
    M = offset_track(t, [(10, 30, a), (30, 40, a), (70, 90, b)], a, CFG)
    np.testing.assert_allclose(M[:45], np.broadcast_to(offset_matrix(a), (45, 4, 4)), atol=1e-12)
    np.testing.assert_allclose(M[60:], np.broadcast_to(offset_matrix(b), (40, 4, 4)), atol=1e-12)
    turn = np.linalg.norm(np.diff(M[:, :3, :3], axis=0), axis=(1, 2))
    assert turn[45:60].max() < 0.5  # a gradual turn in the free time between the windows


def test_touch_labels_mark_a_closed_hand_on_an_object():
    from reachy_retarget.retarget.targets import touch_labels
    T = 10
    pose = np.tile(np.eye(4), (T, 1, 1))
    pose[:, 2, 3] = np.r_[np.full(5, 0.03), np.full(5, 0.2)]  # on the cube top, then 18 cm above
    src = SourceEpisode("f", "f/d", "0", "t", np.arange(T) * 0.05, {"h": Effector(pose, np.zeros(T))},
                        objects={"cube": _cube(T)})
    lab = touch_labels(src, "h", np.full(T, -1), CFG, np.ones(T, bool))
    assert lab.tolist() == [0] * 5 + [-1] * 5


def test_release_ramp_opens_at_the_finger_speed_from_the_source_opening():
    from reachy_retarget.retarget.targets import release_ramp, release_starts
    # source clock: grasp until row 3, plateau (Panda command ramping out of its squeeze), then opening
    src = np.array([0.80, 0.80, 0.80, 0.85, 0.85, 0.85, 0.86, 1.00, 1.40, 1.80, 1.80])
    lab = np.array([0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1])
    assert release_starts(src, lab, 0.005) == [5]  # last plateau row before the fingers leave
    t = np.arange(len(src)) * 0.1
    q, events = release_ramp(src, t, [5], lab, rate=3.0)
    assert events == [(5, 10)]
    np.testing.assert_allclose(q[:6], src[:6])
    np.testing.assert_allclose(q[6:8], [1.15, 1.45])           # 3 rad/s from the plateau
    assert np.all(q >= src) and q[9] == pytest.approx(1.8)     # never below the source, capped at its peak
    # a release the next grasp interrupts is capped at the angle reached before it
    lab2 = lab.copy()
    lab2[8:] = 0
    q2, _ = release_ramp(src, t, [5], lab2, rate=3.0)
    np.testing.assert_allclose(q2[6:], [1.0, 1.0, 1.4, 1.8, 1.8])


def _drop_source(fall_to, end_rows=4, bin_walls=False):
    """A cube carried at 5 cm, released at row 6 and falling onto a table box; optional walls."""
    T = 6 + 3 + end_rows
    t = np.arange(T) * 0.05
    z = np.r_[np.full(6, 0.05 + fall_to), [0.05 + fall_to / 2, 0.05, 0.05], np.full(end_rows, 0.05)][:T]
    pose = np.tile([0.0, 0, 0, 1, 0, 0, 0], (T, 1))
    pose[:, 2] = z
    hand = np.tile(np.eye(4), (T, 1, 1))
    hand[:, 2, 3] = 0.05 + fall_to  # the source hand stays where it released the cube
    # the hand approaches along -z: the grasp-center frame's +z points down
    hand[:, :3, :3] = np.diag([1.0, -1.0, -1.0])
    objects = {"cube": ObjectTrack(pose, np.ones(T, bool), "manipulated", {"kind": "box", "half_extents": [0.02] * 3}),
               "table": ObjectTrack(np.tile([0, 0, 0.015, 1, 0, 0, 0.0], (T, 1)), np.ones(T, bool), "support",
                                    {"kind": "box", "half_extents": [0.3, 0.3, 0.015]})}
    if bin_walls:
        objects["wall"] = ObjectTrack(np.tile([0.028, 0, 0.05, 1, 0, 0, 0.0], (T, 1)), np.ones(T, bool), "fixture",
                                      {"kind": "box", "half_extents": [0.005, 0.1, 0.03]})
    eff = Effector(hand, np.r_[np.zeros(6), np.ones(T - 6)], np.r_[np.full(6, 0.04), np.full(T - 6, 0.08)])
    src = SourceEpisode("f", "f/d", "0", "t", t, {"h": eff}, objects=objects)
    lab = np.r_[np.zeros(6, int), np.full(T - 6, -1)]
    return src, lab


def test_short_drops_are_placed_and_long_drops_or_insertions_are_not():
    from reachy_retarget.retarget.targets import place_labels
    src, lab = _drop_source(0.02)
    out, events = place_labels(src, "h", lab, CFG)
    assert len(events) == 1 and events[0][0] == 6 and events[0][1] == 8  # held until it has settled (row 7)
    assert events[0][2] == pytest.approx(0.02) and (out[:8] == 0).all() and (out[8:] == -1).all()
    _, events = place_labels(src, "h", lab, RetargetConfig(place_drops=True, place_max_drop=0.01))
    assert events == []                       # longer than place_max_drop: kept as a drop
    src, lab = _drop_source(0.002)
    assert place_labels(src, "h", lab, CFG)[1] == []   # no drop
    src, lab = _drop_source(0.02, bin_walls=True)
    assert place_labels(src, "h", lab, CFG)[1] == []   # a wall beside the landed cube: an insertion


def test_object_extent_of_a_union_of_boxes_is_its_thinnest_part():
    # robosuite multi-geom objects (the MimicGen mug: 32 convex parts, held by its 9 mm handle)
    track = ObjectTrack(np.tile([0, 0, 0, 1, 0, 0, 0.0], (2, 1)), np.ones(2, bool), "manipulated",
                        {"kind": "boxes", "frame": "body",
                         "boxes": [{"center": [0, 0, 0], "half_extents": [0.04, 0.04, 0.045]},
                                   {"center": [0.05, 0, 0], "half_extents": [0.0045, 0.005, 0.03]}]})
    assert object_extent(track, np.arange(2), np.tile([0, 1.0, 0], (2, 1))) == pytest.approx([0.009, 0.009])
