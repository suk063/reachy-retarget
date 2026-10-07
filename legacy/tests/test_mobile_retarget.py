"""Offline end-to-end archive checks; the IK backend is a deterministic stub."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget import mobile_retarget, robot
from reachy_retarget.agent_dataset import inspect_archive
from reachy_retarget.episodes import matrices_to_pose, pose_to_matrices


class MockRobot:
    errors = (.001, .01)
    fail_at = None

    def __init__(self, root):
        self.q = np.r_[0., 0., 1., 0., np.zeros(14)]
        self.arm_ids = np.arange(4, 18)
        self.arm_v = np.arange(3, 17)
        self.identity = {'backend': 'offline-test-stub'}
        self.calls = 0
        self.last_target = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
        self.last_target[:, 2, 3] = .8
        self.last_target[:, 1, 3] = [.2, -.2]

    def fk(self, q):
        return self.last_target.copy()

    def ik(self, target, q, active_hands, iterations):
        assert active_hands == (0, 1)
        assert iterations == 60
        np.testing.assert_array_equal(self.active, self.arm_v)
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('fixture IK failure')
        self.last_target = target.copy()
        result = q.copy()
        result[self.arm_ids] = self.calls / 10
        return result, *self.errors


@pytest.fixture
def record():
    base = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    base[:, :3, :3] = Rotation.from_euler('z', np.array([.4, .6, .9])[:, None]).as_matrix()
    base[:, :3, 3] = [[2., 3., 0.], [2.1, 3.2, 0.], [2.3, 3.4, 0.]]
    left = base.copy()
    left[:, :3, 3] += [.3, .2, .8]
    right = base.copy()
    right[:, :3, 3] += [.3, -.2, .8]
    cup = base.copy()
    cup[:, :3, 3] += [.4, .2, .75]
    return {'arrays': {'time_s': np.array([1., 1.1, 1.3]),
                       'source/robot_root_pose': matrices_to_pose(base),
                       'hand/left_pose': matrices_to_pose(left),
                       'hand/right_pose': matrices_to_pose(right),
                       'source/gripper_action': np.array([[-1., 1.], [0., 0.], [1., -1.]]),
                       'objects/cup/pose': matrices_to_pose(cup)},
            'metadata': {'source_sequence': 'task/demo_0', 'source_group': 'task/demo_0',
                         'source_urls': ['https://example.invalid/fixture.hdf5'],
                         'source_revision': 'offline-fixture',
                         'objects': {'cup': {'pose_frame': 'world', 'task_role': 'manipulated'}},
                         'derived_fields': {'source_marker': 'source-only derivation'},
                         'state_control_coverage': {'source_only': True},
                         'simulation_assumptions': ['Source fixture is not a physical rollout.']},
            'status': 'normalized_source', 'missing_fields': ['source/neck_joint_state']}


@pytest.fixture(autouse=True)
def offline_robot(monkeypatch):
    monkeypatch.setattr(robot, 'Robot', MockRobot)
    monkeypatch.setattr(MockRobot, 'errors', (.001, .01))
    monkeypatch.setattr(MockRobot, 'fail_at', None)


def check_raw(attempt, clock, completed, failure=None):
    path = attempt / 'raw-ik.npz'
    manifest = json.loads((attempt / 'raw-ik.json').read_text())
    assert manifest['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest['completed_frames'] == completed
    assert manifest['requested_frames'] == len(clock)
    assert manifest['all_frames_computed'] is (completed == len(clock))
    assert manifest['physics_validated'] is False
    assert manifest['failure'] == failure
    with np.load(path, allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw['original_clock_s'], clock)
        assert raw['robot_qpos'].shape == (completed, 18)
        assert raw['ik_error_m_rad'].shape == (completed, 2)
        assert raw['achieved_hand_matrix'].shape == (completed, 2, 4, 4)
        assert raw['target_hand_matrix'].shape == (len(clock), 2, 4, 4)
        return {name: value.copy() for name, value in raw.items()}


@pytest.mark.parametrize('passed,errors', [(True, (.001, .01)), (False, (.03, .2))])
def test_all_frames_export_passed_and_failed_candidates(tmp_path, monkeypatch, record, passed, errors):
    monkeypatch.setattr(MockRobot, 'errors', errors)
    batch_shapes = []

    class StrictSingleAxisRotation:
        @staticmethod
        def from_euler(seq, angles, **kwargs):
            values = np.asarray(angles)
            if values.ndim:
                # Reproduce the newer SciPy batch requirement to catch regressions
                # even when this test runs on an older SciPy release.
                assert values.shape[-1] == len(seq)
                batch_shapes.append(values.shape)
            return Rotation.from_euler(seq, angles, **kwargs)

    monkeypatch.setattr(mobile_retarget, 'Rotation', StrictSingleAxisRotation)
    attempt = tmp_path / 'attempt'
    report = mobile_retarget.retarget_momagen(record, tmp_path / 'export', 'momagen', 'demo', attempt)
    assert batch_shapes == [(3, 1)]
    assert report['status'] == ('kinematic_candidate' if passed else 'kinematic_failed')
    assert report['kinematic_passed'] is passed
    assert report['physics_validated'] is False
    assert report['frames'] == 3
    raw = check_raw(attempt, record['arrays']['time_s'], 3)
    np.testing.assert_allclose(raw['ik_error_m_rad'], np.tile(errors, (3, 1)))
    output = Path(report['retarget_archive'])
    archive = inspect_archive(output)
    assert archive['rows'] == 3 and archive['timed'] is True
    assert archive['policy_ready'] is False
    assert not any(name.startswith('observation/') for name in archive['fields'])
    assert not {'action', 'applied_controls', 'physics_state'} & archive['fields'].keys()
    with h5py.File(output, 'r') as handle:
        metadata = json.loads(handle.attrs['metadata_json'])
        assert metadata['source_metadata']['state_control_coverage'] == {'source_only': True}
        assert 'state_control_coverage' not in metadata
        assert 'source_marker' not in metadata['derived_fields']
        assert metadata['status'] == report['status']
        np.testing.assert_array_equal(handle['timestamp'][()], record['arrays']['time_s'])
        np.testing.assert_array_equal(handle['derived/robot_qpos'][()], raw['robot_qpos'])
        np.testing.assert_allclose(handle['derived/gripper_joint_target_rad'][()],
                                   [[-.06, 2.], [.97, .97], [2., -.06]])
        reconstructed = pose_to_matrices(handle['derived/base_pose'][()])
        np.testing.assert_allclose(reconstructed, raw['source_base_reference_matrix'], atol=1e-12)
        expected_objects = raw['placement'] @ pose_to_matrices(record['arrays']['objects/cup/pose'])
        np.testing.assert_allclose(pose_to_matrices(handle['reference/objects/cup/pose'][()]),
                                   expected_objects, atol=1e-12)
    assert json.loads((attempt / 'kinematic-validation.json').read_text())['kinematic_passed'] is passed


def test_export_failure_keeps_complete_raw_ik(tmp_path, monkeypatch, record):
    attempt = tmp_path / 'attempt'

    def failing_export(*args, **kwargs):
        check_raw(attempt, record['arrays']['time_s'], 3)
        raise RuntimeError('fixture export failure')

    monkeypatch.setattr(mobile_retarget, 'write_archive', failing_export)
    with pytest.raises(RuntimeError, match='fixture export failure'):
        mobile_retarget.retarget_momagen(record, tmp_path / 'export', 'momagen', 'demo', attempt)
    check_raw(attempt, record['arrays']['time_s'], 3)
    assert not (attempt / 'kinematic-validation.json').exists()


def test_partial_ik_failure_keeps_completed_frames_and_original_clock(tmp_path, monkeypatch, record):
    monkeypatch.setattr(MockRobot, 'fail_at', 2)
    attempt = tmp_path / 'attempt'
    with pytest.raises(RuntimeError, match='fixture IK failure'):
        mobile_retarget.retarget_momagen(record, tmp_path / 'export', 'momagen', 'demo', attempt)
    check_raw(attempt, record['arrays']['time_s'], 1,
              {'type': 'RuntimeError', 'message': 'fixture IK failure'})
    assert not (tmp_path / 'export').exists()


def task_policy(record):
    text = json.dumps(dict(name='task', task=dict(task_spec=dict(phase_1=dict(
        arm_left=dict(subtask_1=dict(arm='left', object_ref='cup', attached_obj=None)),
        arm_right=dict(subtask_1=dict(arm='right', object_ref=None, attached_obj=None)))))))
    return dict(kind='declared_task_hands_mobile_v1', source_task_config_text=text,
        source_task_config_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_task_config_url='https://github.com/ChengshuLi/MoMaGen/blob/offline-fixture/momagen/datasets/base_configs/task.json',
        source_revision='offline-fixture', base_xy_bounds=[[-1., 1.], [-1., 1.]], seed_yaw_deg=0.)


def test_task_hand_policy_binds_source_identity_and_keeps_attached_hands(record):
    from reachy_retarget.momagen_task_hands import validate
    p = task_policy(record)
    assert validate(p, record['metadata'])['active_hands'] == [0]
    for key, value in [('source_task_config_sha256', '0'*64), ('source_revision', 'wrong'),
                       ('source_task_config_url', 'https://example.invalid/config'),
                       ('seed_yaw_deg', float('nan')), ('base_xy_bounds', [[0, 0], [0, 1]])]:
        with pytest.raises(ValueError):
            validate(dict(p, **{key: value}), record['metadata'])
    config = json.loads(p['source_task_config_text'])
    config['task']['task_spec']['phase_1']['arm_right']['subtask_1']['attached_obj'] = 'plate'
    text = json.dumps(config)
    p.update(source_task_config_text=text, source_task_config_sha256=hashlib.sha256(text.encode()).hexdigest())
    assert validate(p, record['metadata'])['active_hands'] == [0, 1]


@pytest.mark.parametrize('errors,passed', [((.001, .01), True), ((.003, .01), False)])
def test_task_hand_export_preserves_unreachable_idle_targets_and_source_objects(tmp_path, monkeypatch, record, errors, passed):
    class TaskRobot(MockRobot):
        def ik(self, target, q, active_hands, iterations, base_xy_bounds):
            assert active_hands == (0,) and iterations == 100
            np.testing.assert_array_equal(self.active, np.r_[np.arange(3), self.arm_v[:7]])
            np.testing.assert_array_equal(base_xy_bounds, [[-1., 1.], [-1., 1.]])
            self.last_target[0] = target[0]
            result = q.copy()
            result[self.arm_ids[:7]] += .01
            # Deliberately do not change the unconstrained idle arm.
            return result, *errors
    monkeypatch.setattr(robot, 'Robot', TaskRobot)
    record['arrays']['hand/right_pose'][:, 2] = -2.  # Unreachable idle source motion.
    original_right = record['arrays']['hand/right_pose'].copy()
    report = mobile_retarget.retarget_momagen(record, tmp_path/'export', 'momagen', 'task-only',
        tmp_path/'attempt', planning_policy=task_policy(record))
    assert report['kinematic_passed'] is passed
    assert report['position_threshold_m'] == .002
    assert report['fixture_clearance_checked'] is False
    assert report['physics_validated'] is False
    np.testing.assert_array_equal(record['arrays']['hand/right_pose'], original_right)
    with h5py.File(report['retarget_archive']) as h:
        q = h['derived/robot_qpos'][()]
        np.testing.assert_array_equal(q[:, 11:18], 0.)
        raw = np.load(tmp_path/'attempt/raw-ik.npz')
        np.testing.assert_array_equal(h['timestamp'][()], record['arrays']['time_s'])
        np.testing.assert_allclose(pose_to_matrices(h['derived/target_hand_pose'][()])[:, 1],
                                   raw['target_hand_matrix'][:, 1])
        expected = raw['placement'] @ pose_to_matrices(record['arrays']['objects/cup/pose'])
        np.testing.assert_allclose(pose_to_matrices(h['reference/objects/cup/pose'][()]), expected)
        assert 'observation' not in h and 'applied_controls' not in h
    assert inspect_archive(report['retarget_archive'])['policy_ready'] is False
