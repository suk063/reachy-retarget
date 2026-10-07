"""Offline pipeline stage preservation with real common-archive publication."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reachy_retarget import legacy_pipeline as pipeline
from reachy_retarget.agent_dataset import inspect_archive


def write_hdf(path, values, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as handle:
        handle.attrs['metadata_json'] = json.dumps(metadata)
        for name, value in values.items():
            handle[name] = value


@pytest.fixture
def job(tmp_path, monkeypatch):
    metadata = {'source_id': 'source', 'episode_id': 'episode', 'source_sequence': 'task/demo_0',
                'source_group': 'original_recording', 'split': 'train',
                'source_urls': ['https://example.invalid/original'], 'source_revision': 'fixture-pin',
                'provenance': [{'sha256': 'a' * 64, 'path': 'original-source'}],
                'objects': {'Can': {'task_role': 'manipulated'}},
                'env_args': {'env_name': 'PickPlaceCan'}, 'simulation_assumptions': []}
    values = {'time_s': np.array([0., .05, .1]),
              'objects/Can/pose': np.tile([0., 0., 1., 1., 0., 0., 0.], (3, 1))}
    write_hdf(tmp_path / 'workspace/source.h5', values, metadata)
    (tmp_path / 'workspace/source.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(pipeline, '_task_objects', lambda: {'PickPlaceCan': 'Can'})

    def retarget(store, row):
        assert row['split'] == 'train'
        assert row['source_provenance']['provenance'][0]['sha256'] == 'a' * 64
        output = store.root / 'data/retargeted/source/episode/motion.h5'
        write_hdf(output, {'time_s': np.array([0., .01, .02]),
                          'robot/joint_position': np.ones((3, 14)),
                          'robot/base_pose_xyyaw': np.zeros((3, 3)),
                          'objects/Can/pose': values['objects/Can/pose'],
                          'derived/gripper/right_position': np.zeros(3)}, {})
        return {'status': 'kinematic_fail', 'time_dilation': 2., 'physics_validated': False}

    monkeypatch.setattr(pipeline, '_retarget', retarget)
    return dict(workspace='workspace', source='source.h5', dataset='legacy-source', id='run-1',
                source_provenance={'source_urls': ['https://example.invalid/original'],
                                   'source_revision': 'fixture-pin',
                                   'provenance': metadata['provenance']})


def state_record(attempt):
    metadata = {'source_sequence': 'task/demo_0', 'source_group': 'original_recording',
                'source_urls': ['https://example.invalid/original'], 'source_revision': 'fixture-pin',
                'objects': {'Can': {'task_role': 'manipulated'}},
                'derived_fields': {}, 'simulation_assumptions': [], 'missing_fields': [],
                'physics_validated': False}
    path = attempt / 'state-attempt.hdf5'
    write_hdf(path, {'timestamp': np.array([0., .01, .02]),
                     'observation/joint_position': np.arange(12).reshape(3, 4)}, metadata)
    return path


def physics_stubs(monkeypatch, *, rejected=False, task_pass=False):
    def plans(store, row, output):
        if rejected:
            (output / 'alignment-rejected.json').write_text(json.dumps({'reason': 'fixture unreachable'}))
        return 'candidate', 'original-before', None if rejected else 'corrected-after'

    def rollout(store, row, label, candidate, plan):
        assert candidate == 'candidate'
        assert plan == ('original-before' if rejected else 'corrected-after')
        assert label.endswith('-baseline' if rejected else '-aligned')
        attempt = store.root / 'runs/dynamics' / label / row['id']
        path = state_record(attempt)
        (attempt / 'replay.h5').touch()
        return {'status': 'physical_pass' if task_pass else 'physical_fail',
                'physics_validated': task_pass, 'state_recording': {'path': str(path)}}

    monkeypatch.setattr(pipeline, '_paired_plans', plans)
    monkeypatch.setattr(pipeline, '_rollout', rollout)
    monkeypatch.setattr(pipeline, '_audit', lambda attempt: {'actuator_replay_pass': True})


def test_unsupported_task_still_retargets_and_exports_derived_only(tmp_path, monkeypatch, job):
    monkeypatch.setattr(pipeline, '_task_objects', lambda: {})
    result = pipeline.run(tmp_path, job)
    assert result['status'] == 'retargeted_physics_blocked'
    assert result['retargeted'] is True and result['physics_validated'] is False
    assert result['kinematic_report']['status'] == 'kinematic_fail'
    report = inspect_archive(result['archives']['kinematic'])
    assert report['rows'] == 3
    assert 'derived/joint_position' in report['fields']
    assert not any(key.startswith('observation/') for key in report['fields'])
    with h5py.File(result['archives']['kinematic']) as handle:
        meta = json.loads(handle.attrs['metadata_json'])
        assert meta['source_group'] == 'original_recording'
        assert meta['source_revision'] == 'fixture-pin'
        assert meta['kinematic_report']['time_dilation'] == 2.
        np.testing.assert_array_equal(handle['source/time_s'], [0., .05, .1])
    assert Path(result['pipeline_report']).is_file()


def test_rejected_pad_plan_runs_uncorrected_diagnostic_and_preserves_failed_rollout(tmp_path, monkeypatch, job):
    physics_stubs(monkeypatch, rejected=True)
    result = pipeline.run(tmp_path, job)
    assert result['status'] == 'physical_rollout_recorded'
    assert result['diagnostic_baseline'] is True
    assert result['stages']['pad_alignment']['rejection']['reason'] == 'fixture unreachable'
    assert result['physics_report']['status'] == 'physical_fail'
    assert result['actuator_replay_pass'] is True and result['physics_validated'] is False
    assert set(result['archives']) == {'kinematic', 'physics'}
    assert inspect_archive(result['archives']['physics'])['rows'] == 3
    with h5py.File(result['archives']['physics']) as handle:
        np.testing.assert_array_equal(handle['observation/joint_position'], np.arange(12).reshape(3, 4))
        assert json.loads(handle.attrs['metadata_json'])['diagnostic_baseline'] is True


def test_audit_failure_does_not_discard_measured_recording_or_claim_validation(tmp_path, monkeypatch, job):
    physics_stubs(monkeypatch, task_pass=True)

    def audit(attempt):
        raise ValueError('fixture replay mismatch')

    monkeypatch.setattr(pipeline, '_audit', audit)
    result = pipeline.run(tmp_path, job)
    assert result['physics_task_pass'] is True and result['physics_validated'] is False
    assert result['stages']['actuator_audit']['status'] == 'failed'
    assert set(result['archives']) == {'kinematic', 'physics'}
    with h5py.File(result['archives']['physics']) as handle:
        assert json.loads(handle.attrs['metadata_json'])['physics_validated'] is False


def test_rollout_exception_preserves_ik_and_partial_recording(tmp_path, monkeypatch, job):
    physics_stubs(monkeypatch)

    def rollout(store, row, label, candidate, plan):
        state_record(store.root / 'runs/dynamics' / label / row['id'])
        raise RuntimeError('fixture simulation interrupted')

    monkeypatch.setattr(pipeline, '_rollout', rollout)
    result = pipeline.run(tmp_path, job)
    assert result['retargeted'] is True and result['physics_validated'] is False
    assert result['stages']['physics']['status'] == 'failed'
    assert set(result['archives']) == {'kinematic', 'physics'}
    assert Path(result['motion_artifact']).exists()


def test_ik_exception_after_motion_write_preserves_partial_archive(tmp_path, monkeypatch, job):
    original = pipeline._retarget

    def failing_ik(store, row):
        original(store, row)
        raise RuntimeError('fixture report failure')

    monkeypatch.setattr(pipeline, '_retarget', failing_ik)
    result = pipeline.run(tmp_path, job)
    assert result['status'] == 'ik_failed' and result['retargeted'] is False
    assert set(result['archives']) == {'kinematic'}
    assert inspect_archive(result['archives']['kinematic'])['rows'] == 3


def test_isolated_workspace_cannot_overwrite_prior_retarget(tmp_path, job):
    motion_dir = tmp_path / 'workspace/data/retargeted/source/episode'
    motion_dir.mkdir(parents=True)
    marker = motion_dir / 'motion.h5'
    marker.write_bytes(b'preserve me')
    result = pipeline.run(tmp_path, job)
    assert result['status'] == 'source_failed'
    assert marker.read_bytes() == b'preserve me'
