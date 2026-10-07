"""Actual isolated MuJoCo fixture checks; no downloads or hardware backend."""
from types import SimpleNamespace
import json

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget import mobile_fixture_ik as mobile


class Robot:
    arm_ids = np.arange(4, 18)
    identity = 'synthetic-test-robot'
    pin = SimpleNamespace(log3=lambda matrix: Rotation.from_matrix(matrix).as_rotvec())

    def __init__(self):
        self.r = SimpleNamespace(model=SimpleNamespace(lowerPositionLimit=np.full(18, -2.),
                                                       upperPositionLimit=np.full(18, 2.)),
                                 geometry=lambda q: (.1, .1, None))

    def pack(self, arms, base):
        return np.r_[base[:2], np.cos(base[2]), np.sin(base[2]), arms]

    def base(self, q):
        return np.r_[q[:2], np.arctan2(q[3], q[2])]

    def fk(self, q):
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[1, :3, 3] = [q[0] + q[11], q[1], 1.]
        return poses


def prepared(monkeypatch, obstacle_x=2.):
    xml = f'''<mujoco><worldbody>
      <geom name="fixture" type="box" pos="{obstacle_x} 0 1" size=".1 .1 .1"/>
      <body name="base_link" pos="0 0 1"><joint name="robot_x" type="slide" axis="1 0 0"/>
        <geom name="robot" type="sphere" size=".02"/></body>
      <body name="object" pos="4 0 1"><freejoint name="object_free"/>
        <geom name="object_geom" type="sphere" size=".02"/></body>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    object_q0 = model.qpos0[1:].copy()

    def initialize(model, data, robot, q, mimics, grip):
        assert isinstance(grip, dict) and grip['l_hand_finger'] == 2.
        # Robot-only kinematic updates, just as the production initializer.
        data.qpos[0] = q[0] + q[11]
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.qpos[1:], object_q0)
        assert data.time == 0.

    monkeypatch.setattr(mobile, 'initialize', initialize)
    n = 4
    refs = np.zeros((n, 17))
    refs[:, 3:10] = np.linspace(0., .1, n)[:, None]
    goals = np.tile(np.eye(4), (n, 1, 1))
    goals[:, 0, 3] = np.linspace(0., .03, n)
    goals[:, 2, 3] = 1.
    objects = goals.copy(); objects[:, 0, 3] = 4.
    metadata = {'source_urls': ['https://example.invalid/offline-fixture'], 'source_revision': 'test'}
    details = {'object_id': 'object', 'pad_alignment': {'inferred_contact_angle_rad': 1.},
               'effective_candidate': {'gripper_closed_target': .998, 'force_feedback': False}}
    robot = Robot()
    value = (model, xml, {'objects': {'object': {'body': 'object'}}, 'mimics': {}},
             {}, metadata, details, robot.pack(refs[0, 3:], refs[0, :3]),
             np.arange(n) * .01, refs, np.array([2., -.06, -.06, 2.]), goals, objects)
    return robot, value


def test_complete_clock_and_left_arm_retained_without_object_state_assignments(monkeypatch, tmp_path):
    robot, source = prepared(monkeypatch)
    original = [a.copy() for a in source[7:12]]
    result = mobile.prepare(source, robot, tmp_path/'ok', anchor_index=1,
        anchor_reference=source[8][1], base_bounds_xyyaw=[[-.2, .2], [-.2, .2], [-.2, .2]],
        support_pairs=[])
    report = result[5]['robot_mobile_fixture_ik']
    assert report['admitted'] and report['completed_frames'] == report['total_frames'] == 4
    assert report['max_ik_errors'][0] < 1e-5 and not report['physics_validated']
    for i in (7, 9, 10, 11):
        np.testing.assert_array_equal(result[i], source[i])
    for saved, actual in zip(original, source[7:12]):
        np.testing.assert_array_equal(saved, actual)
    np.testing.assert_array_equal(result[8][:, 3:10], source[8][:, 3:10])
    assert report['source_metadata'] == source[4]
    assert 'robot_initialization' not in source[5]
    assert report['source_array_sha256']['time_s'] == mobile.array_digest(source[7])
    assert report['scene_sha256'] and report['artifact_sha256']


def test_true_fixture_obstruction_is_retained_as_failed_attempt(monkeypatch, tmp_path):
    robot, source = prepared(monkeypatch, obstacle_x=0.)
    with pytest.raises(ValueError, match='Mobile fixture IK rejected'):
        mobile.prepare(source, robot, tmp_path/'blocked', anchor_index=1,
            anchor_reference=source[8][1], base_bounds_xyyaw=[[-.2, .2], [-.2, .2], [-.2, .2]],
            max_iterations=20, max_active_set_updates=2, support_pairs=[])
    report = json.loads((tmp_path/'blocked/result.json').read_text())
    assert not report['admitted'] and report['events']
    assert report['completed_frames'] == 4
    assert (tmp_path/'blocked/robot-mobile-path.npz').exists()
    assert not report['admission_checks']['ik'] or not report['admission_checks']['positive_fixture_clearance']


def test_missing_or_retimed_clock_cannot_be_silently_reinterpreted(monkeypatch, tmp_path):
    robot, source = prepared(monkeypatch)
    source[5]['robot_control_retiming'] = {'source_clock_artifact': 'not-read'}
    with pytest.raises(ValueError, match='original-clock'):
        mobile.prepare(source, robot, tmp_path/'retimed', anchor_index=1,
            anchor_reference=source[8][1], base_bounds_xyyaw=[[-.2, .2], [-.2, .2], [-.2, .2]], support_pairs=[])


def test_early_rejection_retains_failed_configuration_and_every_source_row(monkeypatch, tmp_path):
    robot, source = prepared(monkeypatch, obstacle_x=0.)
    output = tmp_path/'early'
    with pytest.raises(ValueError, match='Mobile fixture IK rejected'):
        mobile.prepare(source, robot, output, anchor_index=1,
            anchor_reference=source[8][1], base_bounds_xyyaw=[[-.2, .2], [-.2, .2], [-.2, .2]],
            max_iterations=20, max_active_set_updates=2, support_pairs=[],
            stop_on_first_invalid=True)
    report = json.loads((output/'result.json').read_text())
    assert not report['admitted'] and not report['admission_checks']['completed']
    assert report['completed_frames'] == 1 and report['total_frames'] == 4
    assert report['events'][0]['frame'] == 1
    with np.load(output/'robot-mobile-path.npz') as saved:
        assert np.isfinite(saved['q'][1]).all() and np.isnan(saved['q'][0]).all()
        for name, value in zip(('original_time_s', 'original_reference',
                               'original_gripper_intent', 'original_hand_goals',
                               'original_object_goals'), source[7:12]):
            np.testing.assert_array_equal(saved[name], value)
        np.testing.assert_array_equal(saved['reference'][[0, 2, 3]], source[8][[0, 2, 3]])


def test_numeric_binding_includes_dtype_shape_and_values():
    a = np.arange(4, dtype='float64')
    assert mobile.array_digest(a) != mobile.array_digest(a.astype('float32'))
    assert mobile.array_digest(a) != mobile.array_digest(a.reshape(2, 2))
    assert mobile.array_digest(a) != mobile.array_digest(a + 1.)


def test_explicit_pose_constraints_resolve_scalar_objective_tradeoff(monkeypatch, tmp_path):
    robot, source = prepared(monkeypatch)
    source[10][:, :3, 3] = [0., 0., 1.]

    def coupled_fk(q):
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[1, :3, 3] = [q[11], 0., 1.]
        poses[1, :3, :3] = Rotation.from_rotvec([0., 0., 5.*(q[11]-.005)]).as_matrix()
        return poses

    robot.fk = coupled_fk
    kwargs = dict(anchor_index=1, anchor_reference=source[8][1],
                  base_bounds_xyyaw=[[-.2, .2], [-.2, .2], [-.2, .2]],
                  support_pairs=[], stop_on_first_invalid=True)
    # The least-squares optimum prefers ~3.77mm position error to a slightly
    # larger rotation error, although the unchanged2mm/20mrad box is feasible.
    with pytest.raises(ValueError, match='Mobile fixture IK rejected'):
        mobile.prepare(source, robot, tmp_path/'scalar', **kwargs)
    result = mobile.prepare(source, robot, tmp_path/'constrained',
                            pose_tolerance_constraints=True, **kwargs)
    report = result[5]['robot_mobile_fixture_ik']
    assert report['admitted'] and report['pose_tolerance_constraints']
    assert report['max_ik_errors'][0] < .002 and report['max_ik_errors'][1] < .02
    for index in (7, 9, 10, 11):
        np.testing.assert_array_equal(result[index], source[index])
