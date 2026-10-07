import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from reachy_retarget import feasible_maniskill as feasible


class Robot:
    def __init__(self, fail_at=None):
        self.arm_ids = self.arm_v = np.arange(3, 17)
        self.active = np.arange(17)
        self.calls = 0
        self.fail_at = fail_at
        self.r = SimpleNamespace(geometry=lambda q: (.02, .04, None))

    def pack(self, arms, base):
        return np.r_[base, arms]

    def fk(self, q):
        return np.tile(np.eye(4), (2, 1, 1))

    def ik(self, targets, q, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("solver interrupted")
        q = q.copy()
        q[-1] = targets[1, 0, 3]
        return q, .0005, .005


def fixture(monkeypatch):
    xml = '<mujoco><worldbody><body name="base_link"><joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/><joint name="yaw" axis="0 0 1"/><geom name="robot" type="box" size=".1 .1 .1" pos="0 0 .5"/></body><body name="table" pos="5 0 .5"><geom name="table" type="box" size=".2 .2 .2"/></body><body name="cube" pos="8 0 1"><freejoint name="cube_free"/><geom name="cube" type="box" size=".02 .02 .02"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)

    def initialize(model, data, robot, q, mimics, grip):
        np.testing.assert_array_equal(data.qpos[3:], model.qpos0[3:])
        data.qpos[:3] = q[:3]
        mujoco.mj_forward(model, data)

    monkeypatch.setattr(feasible, 'initialize', initialize)
    times = np.arange(3) * .01
    reference = np.zeros((3, 17))
    reference[:, 3:10] = np.arange(3)[:, None] * .001
    hands = np.tile(np.eye(4), (3, 1, 1))
    hands[:, 0, 3] = np.arange(3) * .001
    objects = np.tile(np.eye(4), (3, 1, 1))
    details = dict(task='PickCube-v1', object_id='cube', effective_candidate={'controller': 'joint_reference'})
    return (model, xml, {'objects': {'cube': {'body': 'cube', 'joint': 'cube_free'}}, 'mimics': {}},
            {}, {}, details, np.zeros(17), times, reference, np.array([2., -.06, -.06]), hands, objects)


def test_robot_only_correction_preserves_targets_clock_scene_and_source(monkeypatch, tmp_path):
    original = fixture(monkeypatch)
    robot = Robot()
    active = robot.active.copy()
    before = original[8].copy()
    result = feasible.prepare(original, robot, tmp_path / 'admitted', [.25, -.653, np.pi])
    for index in (0, 1, 2, 3, 4, 7, 9, 10, 11):
        assert result[index] is original[index]
    assert result[5]['effective_candidate'] == original[5]['effective_candidate']
    assert 'robot_initialization' not in original[5]
    np.testing.assert_array_equal(original[8], before)
    np.testing.assert_array_equal(result[8][:, 3:10], before[:, 3:10])
    np.testing.assert_array_equal(result[8][:, :3], np.tile([.25, -.653, np.pi], (3, 1)))
    np.testing.assert_array_equal(result[6][:3], [.25, -.653, np.pi])
    np.testing.assert_array_equal(robot.active, active)
    assert result[5]['robot_initialization']['admitted']
    assert not result[5]['robot_initialization']['physics_validated']


def test_colliding_base_rejected_with_complete_computed_artifact(monkeypatch, tmp_path):
    original = fixture(monkeypatch)
    with pytest.raises(ValueError, match='initialization rejected'):
        feasible.prepare(original, Robot(), tmp_path / 'collision', [5, 0, 0])
    report = json.loads((tmp_path / 'collision/result.json').read_text())
    assert report['completed_frames'] == 3
    assert not report['admission_checks']['environment']
    with np.load(tmp_path / 'collision/robot-initialization.npz') as archive:
        assert archive['q'].shape == (3, 17)
        np.testing.assert_array_equal(archive['original_object_goals'], original[11])


def test_interrupted_solver_retains_partial_solution_and_restores_solver(monkeypatch, tmp_path):
    original = fixture(monkeypatch)
    robot = Robot(fail_at=2)
    active = robot.active.copy()
    with pytest.raises(ValueError, match='initialization rejected'):
        feasible.prepare(original, robot, tmp_path / 'interrupted', [.25, -.653, np.pi])
    np.testing.assert_array_equal(robot.active, active)
    report = json.loads((tmp_path / 'interrupted/result.json').read_text())
    assert report['completed_frames'] == 1
    assert 'solver interrupted' in report['exception']
    assert not report['admitted']
    with np.load(tmp_path / 'interrupted/robot-initialization.npz') as archive:
        assert archive['q'].shape == (1, 17)
        np.testing.assert_array_equal(archive['time_s'], original[7])


def test_explicit_can_base_trajectory_preserves_source_clock_and_other_inputs(monkeypatch, tmp_path):
    from reachy_retarget import feasible_base
    original = fixture(monkeypatch)
    original[5]['task'] = 'PickPlaceCan'
    track = np.array([[.2, .3, .0], [.21, .3, .001], [.22, .3, .002]])
    robot = Robot()
    result = feasible_base.prepare(original, robot, tmp_path / 'can', track)
    np.testing.assert_array_equal(result[8][:, :3], track)
    np.testing.assert_array_equal(result[8][:, 3:10], original[8][:, 3:10])
    np.testing.assert_array_equal(result[6][:3], track[0])
    for index in (0, 1, 2, 3, 4, 7, 9, 10, 11):
        assert result[index] is original[index]
    report = result[5]['robot_initialization']
    assert report['base_policy'] == 'explicit original-clock base trajectory'
    assert report['admitted'] and not report['physics_validated']
    with np.load(tmp_path / 'can/robot-initialization.npz') as artifact:
        np.testing.assert_array_equal(artifact['q'][:, :3], track)
    with pytest.raises(ValueError, match='PickCube-v1'):
        feasible.prepare(original, robot, tmp_path / 'wrong-task', track)


def test_explicit_arm_seed_is_used_recorded_and_does_not_modify_inactive_arm(monkeypatch, tmp_path):
    from reachy_retarget import feasible_base
    original = fixture(monkeypatch)
    original[5]['task'] = 'PickPlaceCan'
    seed = np.arange(7) * .01
    result = feasible_base.prepare(original, Robot(), tmp_path / 'seed', [0, 0, 0],
                                   right_arm_seed=seed)
    # The mock IK changes only the last active joint. The other six prove that
    # the requested branch seed survives packing, rather than being discarded.
    np.testing.assert_array_equal(result[8][0, 10:16], seed[:6])
    np.testing.assert_array_equal(result[8][:, 3:10], original[8][:, 3:10])
    assert result[5]['robot_initialization']['right_arm_seed'] == seed.tolist()
    with pytest.raises(ValueError, match='seven finite'):
        feasible_base.prepare(original, Robot(), tmp_path / 'bad-seed', [0, 0, 0],
                              right_arm_seed=[np.nan] * 7)


@pytest.mark.parametrize("task", ["Lift", "Stack_D0", "UnseenRigidBoxTask"])
def test_shape_admission_routes_generic_boxes_without_changing_source(monkeypatch, tmp_path, task):
    from reachy_retarget import feasible_base
    original = fixture(monkeypatch)
    original[5].update(task=task, alignment_variant="constant_tcp_attachment",
                       grasp_attachment={"requires_full_ik_and_collision_admission": True},
                       pad_alignment={"inferred_contact_angle_rad": .9})
    result = feasible_base.prepare(original, Robot(), tmp_path / "box", [0, 0, 0])
    assert result[5]["robot_initialization"]["admitted"]
    assert "shape_eligibility" in result[5]["robot_initialization"]
    with np.load(tmp_path / "box/robot-initialization.npz") as artifact:
        np.testing.assert_array_equal(artifact["gripper_aperture_envelopes_rad"],
                                      [[2., 2.], [-.06, 2.], [-.06, .9]])
    for index in (0, 1, 2, 3, 4, 7, 9, 10, 11):
        assert result[index] is original[index]
    assert json.loads((tmp_path / "box/result.json").read_text())["shape_eligibility"] == result[5]["robot_initialization"]["shape_eligibility"]


def test_generic_admission_rejects_uncalibrated_attachment(monkeypatch, tmp_path):
    from reachy_retarget import feasible_base
    original = fixture(monkeypatch)
    original[5]["task"] = "Lift"
    with pytest.raises(ValueError, match="calibrated constant box attachment"):
        feasible_base.prepare(original, Robot(), tmp_path / "uncalibrated", [0, 0, 0])
    assert not (tmp_path / "uncalibrated").exists()


def test_intent_admission_uses_declared_aperture_and_keeps_inactive_hand_open(monkeypatch, tmp_path):
    from reachy_retarget import feasible_base
    original = fixture(monkeypatch)
    original[5].update(task='Lift', alignment_variant='constant_tcp_attachment',
                       grasp_attachment={'requires_full_ik_and_collision_admission': True},
                       pad_alignment={'inferred_contact_angle_rad': .9},
                       effective_candidate={'gripper_closed_target': .85, 'force_feedback': False})
    previous = feasible.initialize; observed = []
    def record(model, data, robot, q, mimics, grip):
        observed.append(grip); previous(model, data, robot, q, mimics, grip)
    monkeypatch.setattr(feasible, 'initialize', record)
    feasible_base.prepare(original, Robot(), tmp_path/'bounded', [0., 0., 0.])
    assert all(g['l_hand_finger'] == 2. and g['r_hand_finger'] >= .85 for g in observed)
    with np.load(tmp_path/'bounded/robot-initialization.npz') as artifact:
        np.testing.assert_array_equal(artifact['gripper_aperture_envelopes_rad'],
                                      [[2., 2.], [.85, 2.], [.85, .9]])
    original[5]['effective_candidate']['force_feedback'] = True
    with pytest.raises(ValueError, match='force-feedback'):
        feasible_base.prepare(original, Robot(), tmp_path/'force', [0., 0., 0.])
