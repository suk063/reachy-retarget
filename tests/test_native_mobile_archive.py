import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from reachy_retarget.adapters.native_mobile import normalize, _joint_layout
from reachy_retarget.mobile_pilot import SOURCES


def ledger(root, source):
    folder = root / 'data/raw' / (source + '_native_pilot')
    rows = []
    for path in folder.rglob('*'):
        if path.is_file():
            rows.append({'path': str(path.relative_to(folder)), 'size': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / 'catalog').mkdir(exist_ok=True)
    (root / 'catalog' / (source + '-pilot.json')).write_text(json.dumps({**SOURCES[source], 'downloaded': rows}))


def robocasa(root):
    folder = root / 'data/raw/robocasa_native_pilot/lerobot/extras'
    ep = folder / 'episode_000000'
    ep.mkdir(parents=True)
    xml = '''<mujoco><worldbody>
      <body name="robot0_base"><joint name="robot_base_x" type="slide"/><body><joint name="robot_arm"/></body></body>
      <body name="cab_main"><body><joint name="cab_hinge"/></body></body>
      <body name="counter_main"/>
      <body name="obj_main"><joint name="obj_joint" type="free"/></body>
      <body name="distr_counter_main"><joint name="distr_joint" type="free"/></body>
    </worldbody></mujoco>'''
    _, joints, nq, nv = _joint_layout(xml)
    states = np.zeros((3, 1 + nq + nv))
    states[:, 0] = [.5, .55, .65]
    for joint in joints:
        if joint['type'] == 'free':
            states[:, 1 + joint['qpos_address'] + 3] = 1
    primary = next(joint for joint in joints if joint['body'] == 'obj_main')
    states[:, 1 + primary['qpos_address']] = [1, 2, 3]
    states[:, 1] = [4, 5, 6]
    states[:, 1 + nq] = [7, 8, 9]
    np.savez(ep / 'states.npz', states=states)
    (ep / 'model.xml.gz').write_bytes(gzip.compress(xml.encode()))
    (ep / 'ep_meta.json').write_text(json.dumps({'lang': 'Put cereal in cabinet.',
        'object_cfgs': [{'name': 'obj', 'info': {'cat': 'cereal'}}, {'name': 'distr_counter'}],
        'fixture_refs': {'cab': 'cab', 'counter': 'counter'}}))
    (folder / 'dataset_meta.json').write_text(json.dumps({'env': 'PickPlaceCounterToCabinet'}))
    ledger(root, 'robocasa')
    return ep


def bigym(root):
    folder = root / 'data/raw/bigym_native_pilot/MovePlate'
    folder.mkdir(parents=True)
    values = {'info_demo_action': np.arange(45, dtype='<f8').reshape(3, 15),
              'termination': np.array([False, False, True]),
              'truncation': np.zeros(3, dtype=bool), 'reward': np.empty(0, dtype='<f8')}
    metadata = {'uuid': 'real-source-seed-id', 'seed': 7,
                'package_versions': {'mujoco': '3.1.5', 'bigym': '4.0.0'},
                'environment_data': {'env_name': 'MovePlate', 'action_mode_name': 'JointPositionActionMode',
                                     'action_mode_absolute': True, 'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_rz']}}
    header = {'__metadata__': {key: json.dumps(value) for key, value in metadata.items()}}
    payload = b''
    for key, value in values.items():
        start = len(payload)
        payload += value.tobytes()
        header[key] = {'dtype': 'BOOL' if value.dtype == bool else 'F64',
                       'shape': list(value.shape), 'data_offsets': [start, len(payload)]}
    hbytes = json.dumps(header).encode()
    path = folder / 'pilot.safetensors'
    path.write_bytes(struct.pack('<Q', len(hbytes)) + hbytes + payload)
    ledger(root, 'bigym')
    return path


def test_robocasa_preserves_clock_gaps_and_measured_joint_state(tmp_path):
    robocasa(tmp_path)
    record = normalize(tmp_path, 'robocasa')
    arrays = record['arrays']
    np.testing.assert_allclose(arrays['timestamp'], [0, .05, .15])
    np.testing.assert_array_equal(arrays['source/simulator_time_s'], [.5, .55, .65])
    np.testing.assert_array_equal(arrays['objects/obj/pose'][:, 0], [1, 2, 3])
    np.testing.assert_array_equal(arrays['source/joint_position'][:, 0], [4, 5, 6])
    np.testing.assert_array_equal(arrays['source/joint_velocity'][:, 0], [7, 8, 9])
    assert arrays['objects/cab/joint_position'].shape == (3, 1)
    assert 'source/action' not in arrays
    assert not any('distr_' in key or 'image' in key for key in arrays)
    assert record['metadata']['source_joint_names'] == ['robot_base_x', 'robot_arm']
    assert record['metadata']['object_coverage']['missing_task_object_pose_names'] == ['cab', 'counter']
    assert record['metadata']['physics_validated'] is False


def test_bigym_stays_untimed_and_actions_are_not_observations(tmp_path):
    bigym(tmp_path)
    record = normalize(tmp_path, 'bigym')
    arrays = record['arrays']
    assert 'timestamp' not in arrays and 'time_s' not in arrays
    assert arrays['source/action'].shape == (3, 15)
    assert arrays['source/reward'].shape == (0,)
    assert record['metadata']['timing']['frame_index_is_timestamp'] is False
    assert record['metadata']['source_configuration']['seed'] == 7
    assert not any(key.startswith('objects/') or key.startswith('robot/') for key in arrays)
    assert record['status'] == 'blocked_source_replay_and_timestamp'


def test_native_source_integrity_is_required(tmp_path):
    ep = robocasa(tmp_path)
    (ep / 'ep_meta.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum/length'):
        normalize(tmp_path, 'robocasa')


@pytest.mark.parametrize('bad_clock,bad_quat', [(True, False), (False, True)])
def test_bad_recorded_values_fail_closed(tmp_path, bad_clock, bad_quat):
    ep = robocasa(tmp_path)
    with np.load(ep / 'states.npz') as f:
        states = f['states']
    if bad_clock:
        states[1, 0] = states[0, 0]
    if bad_quat:
        _, joints, _, _ = _joint_layout(gzip.decompress((ep / 'model.xml.gz').read_bytes()))
        joint = next(j for j in joints if j['body'] == 'obj_main')
        states[1, 1 + joint['qpos_address'] + 3] = 0
    np.savez(ep / 'states.npz', states=states)
    ledger(tmp_path, 'robocasa')
    with pytest.raises(ValueError):
        normalize(tmp_path, 'robocasa')


def test_invalid_safetensors_extent_rejected(tmp_path):
    path = bigym(tmp_path)
    raw = path.read_bytes()
    length = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    header['info_demo_action']['data_offsets'][1] += 8
    hbytes = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(hbytes)) + hbytes + raw[8 + length:])
    ledger(tmp_path, 'bigym')
    with pytest.raises(ValueError, match='extent'):
        normalize(tmp_path, 'bigym')


@pytest.mark.parametrize('source', ['robocasa', 'bigym'])
def test_pilot_record_exports_to_the_common_archive(tmp_path, source):
    from reachy_retarget.agent_dataset import export_normalized, inspect_archive

    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    (robocasa if source == 'robocasa' else bigym)(inputs)
    record = normalize(inputs, source)
    output = export_normalized(record, tmp_path / 'output', dataset_id=source, episode_id='source-pilot')
    report = inspect_archive(output)
    assert report['rows'] == 3
    assert report['timed'] is (source == 'robocasa')
    assert report['policy_ready'] is False
    assert report['rgb_required'] is False
    if source == 'bigym':
        assert 'timestamp' in report['missing_fields']
        assert report['fields']['source/source/reward']['shape'] == [0]
        assert 'timestamp' not in report['fields']
    else:
        assert report['fields']['observation/objects/obj/pose']['shape'] == [3, 7]
        assert not any('distr_counter/pose' in key for key in report['fields'])
