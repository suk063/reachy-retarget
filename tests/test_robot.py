import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from reachy_retarget import robot as R
from reachy_retarget.robot import gripper as G
from reachy_retarget.robot.resources import profile

AGENT = Path("/Users/sunghwan/workspace/reachy-agent")
RNG = np.random.default_rng(0)


def ready(finger=0.5):
    q = np.zeros(22)
    posture = profile()["postures"]["ready"]
    q[R.LEFT_ARM], q[R.RIGHT_ARM] = np.radians(posture["left"]), np.radians(posture["right"])
    q[R.GRIPPERS] = finger
    return q


def random_q(n):
    lo, hi = R.LOWER.copy(), R.UPPER.copy()
    lo[:3], hi[:3] = [-2, -2, -np.pi], [2, 2, np.pi]
    return lo + RNG.random((n, 22)) * (hi - lo)


@pytest.fixture(scope="module")
def agent():
    if not AGENT.exists():
        pytest.skip("reachy-agent checkout not available")
    sys.path.insert(0, str(AGENT))
    try:
        from robot import collision, model
        yield model, collision
    finally:
        sys.path.remove(str(AGENT))


def as_dict(q):
    return {n: float(v) for n, v in zip(R.JOINTS[3:], q[3:])}


def test_constants():
    assert len(R.JOINTS) == 22 and R.JOINTS[R.NECK] == ("neck_roll", "neck_pitch", "neck_yaw")
    assert R.JOINTS[R.GRIPPERS] == ("l_hand_finger", "r_hand_finger")
    assert np.isinf(R.LOWER[:3]).all() and np.isinf(R.UPPER[:3]).all()
    assert np.allclose(R.UPPER[[7, 8, 14, 15]], np.radians(30), atol=1e-4)
    assert np.allclose(R.VELOCITY[R.NECK], np.radians(30)) and np.allclose(R.VELOCITY[3:17], 1.0)
    assert hashlib.sha256(R.URDF_FILE.read_bytes()).hexdigest() == R.URDF_SHA256


def test_fk_parity_with_reachy_agent(agent):
    model, _ = agent
    m, robot = model.load(), R.Reachy.load()
    for q in random_q(50):
        poses = robot.fk_base(q)
        ref = model.task_poses(as_dict(q), m)
        assert np.allclose(poses["left_tcp"], ref["left"], atol=1e-9)
        assert np.allclose(poses["right_tcp"], ref["right"], atol=1e-9)
        assert np.allclose(poses["head"], m.transform("base_link", "head", as_dict(q)), atol=1e-9)
        assert np.allclose(poses["torso"], model.torso_in_base(m), atol=1e-9)
        world = robot.fk(q)
        assert np.allclose(world["left_tcp"], R.planar(*q[:3]) @ poses["left_tcp"], atol=1e-12)


def test_fk_batched_matches_single():
    robot, Q = R.Reachy.load(), random_q(8)
    batch = robot.fk(Q)
    for i, q in enumerate(Q):
        for frame, T in robot.fk(q).items():
            assert np.allclose(batch[frame][i], T, atol=1e-12)


def test_single_configuration_fast_path_is_bit_identical():
    tree = R.SelfCollision.load().tree
    for q in random_q(50)[:, 3:]:
        batched = tree.node_poses(q[None])
        assert all(np.array_equal(T, B[0]) for T, B in zip(tree.node_poses(q), batched))
        T, J = tree.jacobian(q, "l_hand_palm_link")
        assert np.array_equal(T, tree.jacobian(q, "l_hand_palm_link", tree.node_poses(q))[0])


def pose_error(A, B):
    dR = A[:3, :3] @ B[:3, :3].T
    w = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    return np.r_[A[:3, 3] - B[:3, 3], w]


@pytest.mark.parametrize("frame", R.FRAMES)
def test_jacobian_matches_finite_differences(frame):
    robot, eps = R.Reachy.load(), 1e-6
    for q in random_q(5):
        T, J = robot.jacobian(q, frame)
        assert np.allclose(T, robot.fk(q)[frame], atol=1e-12)
        for k in range(20):
            dq = np.zeros(22)
            dq[k] = eps
            num = (pose_error(robot.fk(q + dq)[frame], robot.fk(q - dq)[frame])) / (2 * eps)
            assert np.allclose(J[:, k], num, atol=1e-6), (frame, R.JOINTS[k])


def test_tree_mimic_jacobian_and_reference_transform():
    tree, urdf, eps = G._tree("left"), R.Reachy.load().urdf, 1e-6
    for f in np.linspace(G.LOWER, G.UPPER, 5):
        for link in tree.targets:
            T, J = tree.jacobian([f], link)
            assert np.allclose(T, urdf.transform("l_hand_palm_link", link, {"l_hand_finger": f}), atol=1e-12)
            num = pose_error(tree.fk([f + eps])[link], tree.fk([f - eps])[link]) / (2 * eps)
            assert np.allclose(J[:, 0], num, atol=1e-6)


def test_grasp_center_axes_and_offset():
    robot = R.Reachy.load()
    for side in ("left", "right"):
        G_ = robot.grasp_center(side)
        assert np.allclose(G_[:3, :3].T @ G_[:3, :3], np.eye(3)) and np.isclose(np.linalg.det(G_[:3, :3]), 1)
        assert np.allclose(G_[:3, 3], [-0.010196, 0, -0.046775], atol=1e-6)
        poses = robot.fk(ready())
        gc = poses[f"{side}_grasp"]
        assert gc[0, 2] > 0.99  # ready posture: forearm horizontal, approach points forward
        assert gc[1, 1] < -0.99  # closing axis from the robot-left finger toward the robot-right one
        assert np.allclose(robot.tcp_from_grasp_center(gc, side), poses[f"{side}_tcp"], atol=1e-12)
        # The grasp center lies between the pads of the posed fingers (straight fingers).
        s = side[0]
        q = ready(G.STRAIGHT_ANGLE)
        palm = robot.urdf.transform("base_link", f"{s}_hand_palm_link", {**as_dict(q), "tripod_joint": 0})
        pads = G.pad_centers(G.STRAIGHT_ANGLE, side) @ palm[:3, :3].T + palm[:3, 3]
        assert np.allclose(pads.mean(0), robot.fk_base(q)[f"{side}_grasp"][:3, 3], atol=1e-9)
        closing = pads[0] - pads[1]
        assert np.allclose(closing / np.linalg.norm(closing), robot.fk_base(q)[f"{side}_grasp"][:3, 1], atol=1e-9)


def test_gripper_mapping():
    a = np.linspace(G.LOWER, G.UPPER, 101)
    assert (G.LOWER, G.UPPER) == (-0.0803, 2.27)
    assert np.allclose(R.opening_to_angle(R.angle_to_opening(a)), a)
    assert R.angle_to_opening(G.UPPER) == 1 and R.angle_to_opening(G.LOWER) == 0
    w = R.angle_to_width(a)
    assert np.all(np.diff(w) >= 0) and w[0] == 0 and np.isclose(w[-1], R.MAX_WIDTH)
    assert 0.09 < R.MAX_WIDTH < 0.11 and 0 < G.CONTACT_ANGLE < 0.1
    widths = np.linspace(0, R.MAX_WIDTH, 50)
    assert np.allclose(R.angle_to_width(R.width_to_angle(widths)), widths, atol=1e-6)
    assert np.allclose(G.pad_centers(a, "left"), G.pad_centers(a, "right"))


def test_self_clearance(agent):
    _, collision = agent
    q = ready()
    d, pair = R.self_clearance(q)
    assert d > 0.009, (d, pair)
    assert np.isclose(d, collision.CollisionGeometry().clearance(as_dict(q)), atol=1e-9)
    head_hit = q.copy()
    head_hit[R.LEFT_ARM] = np.radians([-90, 0, 0, -130, 0, 0, 0])  # left hand folded into the head
    crossed = q.copy()  # both forearms swung inward into each other (a cross-arm pair)
    crossed[R.LEFT_ARM], crossed[R.RIGHT_ARM] = np.radians([[-60, -29, 0, -100, 0, 0, 0], [-60, 29, 0, -100, 0, 0, 0]])
    for bad, link in ((head_hit, "head"), (crossed, "r_elbow_forearm_link")):
        d_bad, pair = R.self_clearance(bad)
        assert d_bad < -0.01 and link in pair, (d_bad, pair)
        assert np.isclose(d_bad, collision.CollisionGeometry().clearance(as_dict(bad)), atol=1e-9)
    assert np.allclose(R.min_clearance(np.stack([q, head_hit, crossed])),
                       [d, R.self_clearance(head_hit)[0], R.self_clearance(crossed)[0]])
    moved = q.copy()
    moved[:3] = [1.0, -2.0, 0.7]
    assert np.isclose(R.self_clearance(moved)[0], d)
    centers, radii, link = R.sphere_centers(q)
    assert centers.shape == (len(radii), 3) and link.shape == radii.shape
    assert R.BASE_FOOTPRINT_RADIUS == 0.25


def test_performance():
    robot, q = R.Reachy.load(), ready()
    robot.jacobian(q, "left_grasp")
    n = 200
    start = time.perf_counter()
    for _ in range(n):
        robot.fk(q)
        robot.jacobian(q, "left_grasp")
    assert (time.perf_counter() - start) / n < 1e-3


def test_joint_order_matches_schema():
    from reachy_retarget.robot import JOINTS as robot_joints
    from reachy_retarget.schema.episode import JOINTS as schema_joints
    assert tuple(robot_joints) == tuple(schema_joints)
