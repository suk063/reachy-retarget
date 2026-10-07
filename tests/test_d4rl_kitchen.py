"""Noisy D4RL observations are not exact states, velocities or measured time."""

import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
import pytest

from reachy_retarget.adapters import d4rl_kitchen as adapter
from reachy_retarget.agent_dataset import export_normalized, inspect_archive


def _xml():
    robot = '<body name="panda0_link0" pos="0 0 1">'
    for i in range(9):
        robot += f'<body name="robot_link_{i}" pos="0 0 .03"><joint name="panda_joint_{i}"/><geom type="sphere" size=".01"/>'
    robot += '<site name="end_effector" pos="0 0 .05"/>' + '</body>' * 10
    objects = ''
    for index in range(9, 23):
        name = {11: 'knob 2', 12: 'Burner 2', 17: 'lightswitchbaseroot', 18: 'lightblock_hinge',
                19: 'slidecabinet', 22: 'microwave'}.get(index, 'unrelated_' + str(index))
        site = {11: 'knob2_site', 17: 'light_site', 19: 'slide_site', 22: 'microhandle_site'}.get(index)
        objects += f'<body name="{name}" pos="{index * .1} 1 1"><joint name="object_joint_{index}"/><geom type="sphere" size=".02"/>'
        if site:
            objects += f'<site name="{site}" pos=".03 0 0"/>'
        objects += '</body>'
    objects += '<body name="kettle" pos="1 1 1"><freejoint/><geom type="sphere" size=".03"/><site name="kettle_site" pos=".05 0 0"/></body>'
    return '<mujoco><include file="defaults.xml"/><option timestep=".002"/><compiler angle="radian"/><worldbody>' + robot + objects + '</worldbody></mujoco>'


def fixture(tmp_path, monkeypatch):
    source = tmp_path / 'source' / adapter.SOURCE_REVISION
    model_path = source / adapter.MODEL_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text(_xml())
    (model_path.parent / 'defaults.xml').write_text('<mujoco><default class="fixture"><geom rgba="1 0 0 1"/></default></mujoco>')
    code = source / 'd4rl/kitchen/adept_envs/franka/kitchen_multitask_v0.py'
    code.write_text('class KitchenV0:\n    def __init__(self, robot_params={}, frame_skip=40):\n        pass\n')
    files = [{"path": str(path.relative_to(tmp_path)), "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "git_blob_sha1": None,
              "url": 'https://example.invalid/source'} for path in source.rglob('*') if path.is_file()]
    (tmp_path / 'acquisition.json').write_text(json.dumps({"source_revision": adapter.SOURCE_REVISION, "files": files}))
    model, _ = adapter.compile_model(tmp_path)
    obs = np.zeros((4, 60), dtype=np.float32)
    obs[:, :30] = model.qpos0
    obs[:, :9] += np.arange(4)[:, None] * .01
    obs[:, 26] = 1.01  # Recorded noisy quaternion must survive unchanged.
    spec = dict(adapter.DATASETS['complete'])
    raw = tmp_path / 'raw' / spec['filename']
    raw.parent.mkdir()
    with h5py.File(raw, 'w') as f:
        f['observations'] = obs
        f['actions'] = np.arange(36).reshape(4, 9).astype(np.float32)
        f['rewards'] = [0., 1., 0., 1.]
        f['terminals'] = [False, True, False, True]
        f['timeouts'] = np.zeros(4, dtype=bool)
        f['infos'] = [0, 0, 0, 0]  # Reused IDs must not merge terminal segments.
    spec.update(bytes=raw.stat().st_size, sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    monkeypatch.setitem(adapter.DATASETS, 'complete', spec)
    return obs, model_path


def test_recorded_markers_and_unclosed_tail_define_segments():
    result = adapter.segments(np.array([False, True, False, False, False]), np.array([False, False, False, True, False]))
    assert [(s['start'], s['stop']) for s in result] == [(0, 2), (2, 4), (4, 5)]
    assert result[0]['terminal'] and result[1]['timeout']
    assert not result[2]['closed_by_recorded_marker']
    with pytest.raises(ValueError, match='boolean'):
        adapter.segments(np.zeros(3), np.zeros(3))


def test_legacy_named_defaults_compile_without_changing_original_files(tmp_path, monkeypatch):
    _, path = fixture(tmp_path, monkeypatch)
    original = path.read_bytes()
    model, identity = adapter.compile_model(tmp_path)
    assert (model.nq, model.nv) == (30, 29)
    assert identity['nominal_control_period_s'] == .08
    assert path.read_bytes() == original
    assert '<default><default class="fixture">' in identity['compatible_xml']


def test_noisy_configuration_fk_is_task_scoped_and_untimed_by_default(tmp_path, monkeypatch):
    original, _ = fixture(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('No dynamics/action replay during observation FK')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
    record = adapter.normalize(tmp_path)
    arrays, meta = record['arrays'], record['metadata']
    np.testing.assert_array_equal(arrays['source/observations'], original[:2])
    assert np.all(arrays['source/kettle_quaternion_original_norm'] > 1)
    np.testing.assert_allclose(np.linalg.norm(arrays['objects/kettle/pose'][:, 3:], axis=1), 1)
    assert 'timestamp' not in arrays and 'timestamp' in record['missing_fields']
    assert not any('velocity' in key or 'qvel' in key for key in arrays)
    assert not any('unrelated_' in str(row) for row in meta['objects'].values())
    assert set(meta['objects']) == {'microwave', 'kettle', 'light_switch', 'slide_cabinet'}
    assert meta['source_row_interval']['stop'] == 2
    assert meta['independent_demonstration_count'] is None
    assert meta['physics_validated'] is False
    output = export_normalized(record, tmp_path / 'common', 'd4rl_kitchen', 'noisy-fixture')
    assert inspect_archive(output)['timed'] is False


def test_nominal_time_requires_explicit_choice_and_retains_measured_time_missing(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    record = adapter.normalize(tmp_path, episode=1, nominal_timing=True)
    np.testing.assert_allclose(record['arrays']['timestamp'], [0., .08])
    np.testing.assert_array_equal(record['arrays']['source/frame_index'], [2, 3])
    assert record['metadata']['timing']['measured'] is False
    assert 'recorded simulator timestamps' in record['missing_fields']
    assert 'timestamp' not in record['missing_fields']
    assert 'timestamp' in record['metadata']['derived_fields']


def test_source_tampering_fails_before_reconstruction(tmp_path, monkeypatch):
    _, path = fixture(tmp_path, monkeypatch)
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='checksum mismatch'):
        adapter.normalize(tmp_path)


def test_explicit_ik_view_retains_source_uncertainty_and_no_reachy_commands(tmp_path, monkeypatch):
    original, _ = fixture(tmp_path, monkeypatch)
    record = adapter.ik_view(tmp_path)
    arrays, meta = record['arrays'], record['metadata']
    np.testing.assert_array_equal(arrays['hand/right_pose'], arrays['source/right_tcp_pose_estimate'])
    np.testing.assert_allclose(arrays['time_s'], [0., .08])
    np.testing.assert_array_equal(arrays['source/gripper_aperture_observed_m'], original[:2, 7:9].sum(axis=1))
    assert meta['timing']['measured'] is False and meta['physics_validated'] is False
    assert 'exact simulator qpos' in record['missing_fields']
    assert not any(key.startswith(('observation/', 'command/', 'robot/')) for key in arrays)
