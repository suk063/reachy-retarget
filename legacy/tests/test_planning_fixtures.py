import json
from copy import deepcopy
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget import planning_fixtures as fixtures
from reachy_retarget.episodes import pose_to_matrices, matrices_to_pose, interpolate_pose


def sample(tmp_path):
    xml = '''<mujoco><worldbody>
      <body name="base_link"><joint name="x" type="slide" axis="1 0 0"/>
        <geom name="robot" type="sphere" size=".01" pos="0 0 -.5"/></body>
      <body name="A" pos="4 0 1"><freejoint name="A_free"/><geom type="box" size=".02 .02 .02"/></body>
      <body name="B" pos="5 0 1"><freejoint name="B_free"/><geom type="box" size=".02 .02 .02"/></body>
      <body name="decor" pos="6 0 1"><freejoint name="decor_free"/><geom type="sphere" size=".02"/></body>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('z', .3).as_matrix()
    transform[:3, 3] = [.1, -.2, .0]
    native = np.arange(5)*.01
    clock = np.arange(7)*.01
    arrays = {'time_s': native}
    for name, x in [('A', 4.), ('B', 5.)]:
        poses = np.tile(np.eye(4), (len(native), 1, 1))
        poses[:, :3, 3] = [x, 0, 1]
        if name == 'B':
            poses[:, 2, 3] -= np.arange(len(native))*.0025
            poses[:, :3, :3] = Rotation.from_euler('z', native[:, None]).as_matrix()
        else:
            poses[:, 0, 3] += native*.1
        arrays[f'objects/{name}/pose'] = matrices_to_pose(np.linalg.inv(transform)@poses)
    metadata = {'objects': {n: {'pose_frame': 'world'} for n in ('A', 'B')},
                'source_group': 'test/recorded/1', 'provenance': [{'sha256': 'recorded-test-source'}]}
    source = tmp_path/'source.h5'
    with h5py.File(source, 'w') as f:
        f.attrs['metadata_json'] = json.dumps(metadata)
        for key, array in arrays.items():
            f.create_dataset(key, data=array)
    source.with_suffix('.json').write_text(json.dumps({'source_sequence': 'one'}))
    objects = transform@pose_to_matrices(interpolate_pose(native, arrays['objects/A/pose'], clock))
    hands = np.tile(np.eye(4), (len(clock), 1, 1))
    hands[:, 0, 3] = np.arange(len(clock))*.005
    reference = np.zeros((len(clock), 17))
    reference[:, -3:] = hands[:, :3, 3]
    manifest = {'objects': {n: {'body': n, 'joint': n+'_free'} for n in ('A', 'B')}, 'mimics': {}}
    details = {'object_id': 'A', 'task': 'PickCube-v1', 'evaluation_hold_s': .02,
               'pad_alignment': {'inferred_contact_angle_rad': .9},
               'effective_candidate': {'gripper_closed_target': .85}}
    prepared = (model, xml, manifest, arrays, metadata, details, reference[0], clock,
                reference, np.array([2., 2., 2., 2., -.06, -.06, 2.]), hands, objects)
    return prepared, source, transform


def test_bound_forecast_isolated_data_exact_world_transform_and_endpoint_hold(tmp_path):
    original, source, transform = sample(tmp_path)
    model = original[0]
    unrelated = mujoco.MjData(model)
    unchanged = unrelated.qpos.copy()
    original[5]['robot_initialization'] = {'admitted': True, 'old_reset_query': True}
    bound = fixtures.bind(original, source, tmp_path/'bind', fixture_ids=['B'])
    for index in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11):
        assert bound[index] is original[index]
    assert 'robot_initialization' not in bound[5]
    assert original[5]['robot_initialization']['admitted']
    query = fixtures.make_query(bound)
    assert query.data is not unrelated
    address = int(model.joint('B_free').qposadr[0])
    expected = transform@pose_to_matrices(interpolate_pose(original[3]['time_s'], original[3]['objects/B/pose'], original[7]))
    for frame in range(len(original[7])):
        query.synchronize(frame)
        np.testing.assert_allclose(pose_to_matrices(query.data.qpos[address:address+7][None])[0], expected[frame], atol=1e-14)
        np.testing.assert_array_equal(query.data.qpos[:address], model.qpos0[:address])
        np.testing.assert_array_equal(query.data.qpos[address+7:], model.qpos0[address+7:])
    np.testing.assert_array_equal(unrelated.qpos, unchanged)
    np.testing.assert_array_equal(model.qpos0, unchanged)
    np.testing.assert_array_equal(query.poses[0][-1], query.poses[0][-2])
    output = tmp_path/'query'; output.mkdir()
    report = query.save(output)
    assert report['fixture_ids'] == ['B'] and report['rows'] == len(original[7])
    assert not bound[5][fixtures.KEY]['physics_validated']


def test_actual_retiming_and_terminal_hold_preserve_native_correspondence(tmp_path):
    from reachy_retarget import feasible_timing, terminal_hold
    original, source, transform = sample(tmp_path)
    original[8][:, -3] = np.arange(7)*.01
    bound = fixtures.bind(original, source, tmp_path/'bind', fixture_ids=['B'])
    retimed = feasible_timing.prepare(bound, tmp_path/'timing', arm_speed_rad_s=.7)
    extended = terminal_hold.apply(retimed, tmp_path/'hold', duration_s=.2)
    query = fixtures.make_query(extended)
    with np.load(extended[5]['robot_control_retiming']['source_clock_artifact']) as f:
        mapped = f['source_reference_time_s']
    expected = transform@pose_to_matrices(interpolate_pose(original[3]['time_s'], original[3]['objects/B/pose'], mapped))
    np.testing.assert_allclose(pose_to_matrices(query.poses[0]), expected, atol=1e-14)
    assert len(query.times) > len(original[7])
    assert query.source_times[-1] == original[3]['time_s'][-1]
    np.testing.assert_array_equal(query.poses[0][-1], query.poses[0][-21])


@pytest.mark.parametrize('corruption', ['missing_pose', 'nan_pose', 'frame_zero', 'truncated_clock', 'active_reference', 'robot_object', 'articulated'])
def test_binding_rejects_unknown_or_inconsistent_source_with_saved_reason(tmp_path, corruption):
    original, source, _ = sample(tmp_path)
    p = list(original)
    if corruption in ('missing_pose', 'nan_pose', 'frame_zero'):
        with h5py.File(source, 'r+') as f:
            if corruption == 'missing_pose':
                del f['objects/B/pose']
            else:
                a = f['objects/B/pose'][()]
                a[1 if corruption == 'nan_pose' else 0, 0] += np.nan if corruption == 'nan_pose' else .01
                f['objects/B/pose'][...] = a
                p[3]['objects/B/pose'] = a
    elif corruption == 'truncated_clock':
        p[7] = p[7][1:]
    elif corruption == 'active_reference':
        p[11] = p[11].copy(); p[11][2, 0, 3] += .001
    elif corruption == 'robot_object':
        p[2]['objects']['B'] = {'body': 'base_link', 'joint': 'x'}
    else:
        xml = p[1].replace('<body name="B"', '<body name="B"').replace(
            '<freejoint name="B_free"/>', '<freejoint name="B_free"/><body name="child"><joint/><geom type="sphere" size=".01"/></body>')
        p[0], p[1] = mujoco.MjModel.from_xml_string(xml), xml
    with pytest.raises(ValueError):
        fixtures.bind(tuple(p), source, tmp_path/'bad', fixture_ids=['B'])
    assert not json.loads((tmp_path/'bad/result.json').read_text())['admitted']


@pytest.mark.parametrize('change', ['source', 'metadata', 'artifact', 'scene', 'reset', 'clock', 'active_reference', 'unsupported_placement'])
def test_query_rejects_stale_bindings(tmp_path, change):
    original, source, _ = sample(tmp_path)
    p = list(fixtures.bind(original, source, tmp_path/'bind', fixture_ids=['B']))
    if change == 'source':
        with h5py.File(source, 'r+') as f:
            f.attrs['changed'] = True
    elif change == 'metadata':
        source.with_suffix('.json').write_text('{}')
    elif change == 'artifact':
        with open(p[5][fixtures.KEY]['artifact'], 'ab') as f:
            f.write(b'changed')
    elif change == 'scene':
        p[1] += ' '
    elif change == 'reset':
        p[0].qpos0[int(p[0].joint('B_free').qposadr[0])] += .01
    elif change == 'clock':
        p[7] = p[7]*2
    elif change == 'active_reference':
        p[11] = p[11].copy(); p[11][2, 0, 3] += .01
    else:
        p[5]['robot_supported_placement'] = {'admitted': True}
    with pytest.raises(ValueError):
        fixtures.make_query(tuple(p))


@pytest.mark.parametrize('bad_map', ['missing', 'hash', 'original', 'current', 'reversed', 'truncated'])
def test_invalid_retiming_map_fails_closed(tmp_path, bad_map):
    original, source, _ = sample(tmp_path)
    p = list(fixtures.bind(original, source, tmp_path/'bind', fixture_ids=['B']))
    clock = p[7].copy(); mapped = clock.copy()
    if bad_map == 'current': clock = clock*2
    if bad_map == 'reversed': mapped[2:4] = mapped[2:4][::-1]
    if bad_map == 'truncated': mapped[-1] -= .01
    path = tmp_path/'time-map.npz'
    np.savez(path, original_time_s=p[7]*2 if bad_map == 'original' else p[7],
             control_time_s=clock, source_reference_time_s=mapped)
    p[5]['robot_control_retiming'] = dict(original_interval_complete=True, source_clock_artifact=str(path), artifact_sha256=fixtures.sha256(path))
    if bad_map == 'missing': p[5]['robot_control_retiming'].pop('source_clock_artifact')
    if bad_map == 'hash': p[5]['robot_control_retiming']['artifact_sha256'] = 'bad'
    with pytest.raises(ValueError): fixtures.make_query(tuple(p))


def test_forecast_refuses_integrated_or_moving_query_and_active_object(tmp_path):
    original, source, _ = sample(tmp_path)
    with pytest.raises(ValueError, match='other than'):
        fixtures.bind(original, source, tmp_path/'bad', fixture_ids=['A'])
    bound = fixtures.bind(original, source, tmp_path/'bind', fixture_ids=['B'])
    query = fixtures.make_query(bound)
    query.data.time = .002
    with pytest.raises(ValueError, match='forbidden'): query.synchronize(0)
    query.data.time = 0
    query.data.qvel[0] = .1
    with pytest.raises(ValueError, match='forbidden'): query.synchronize(0)


class Robot:
    def __init__(self):
        self.arm_ids = self.arm_v = np.arange(3, 17)
        self.active = np.arange(17)
        self.r = SimpleNamespace(geometry=lambda q: (.02, .04, None))
    def pack(self, arms, base): return np.r_[base, arms]
    def fk(self, q):
        poses = np.tile(np.eye(4), (2, 1, 1)); poses[1, :3, 3] = q[-3:]; return poses
    def ik(self, goals, q, **kwargs):
        q = q.copy(); q[-3:] = goals[1, :3, 3]; return q, 0., 0.


@pytest.mark.parametrize('admission', ['base', 'approach', 'clearance'])
def test_every_admission_queries_source_fixture_not_reset(monkeypatch, tmp_path, admission):
    from reachy_retarget import feasible_maniskill, grasp_approach, geometry_clearance, physics
    original, source, _ = sample(tmp_path)
    bound = fixtures.bind(original, source, tmp_path/'bind', fixture_ids=['B'])
    address = int(bound[0].joint('B_free').qposadr[0])
    seen = []
    def initialize(model, data, robot, q, mimics, grip):
        data.qpos[0] = q[0]; mujoco.mj_forward(model, data)
    monkeypatch.setattr(physics, 'initialize', initialize)
    monkeypatch.setattr(feasible_maniskill, 'initialize', initialize)
    monkeypatch.setattr(grasp_approach, 'initialize', initialize)
    if admission == 'clearance':
        old = geometry_clearance.RobotFixtureClearance.inspect
        def inspect(self, data):
            seen.append(data.qpos[address:address+7].copy()); return old(self, data)
        monkeypatch.setattr(geometry_clearance.RobotFixtureClearance, 'inspect', inspect)
        result = geometry_clearance.prepare(bound, Robot(), tmp_path/'admission')
        key = 'robot_fixture_clearance'
    else:
        module = feasible_maniskill if admission == 'base' else grasp_approach
        old = module.collision_depths
        def inspect(model, data, *args):
            seen.append(data.qpos[address:address+7].copy()); return old(model, data, *args)
        monkeypatch.setattr(module, 'collision_depths', inspect)
        result = module.prepare(bound, Robot(), tmp_path/'admission', [0, 0, 0]) if admission == 'base' else module.prepare(bound, Robot(), tmp_path/'admission')
        key = 'robot_initialization' if admission == 'base' else 'robot_grasp_approach'
    assert seen[0][2] == pytest.approx(1.)
    assert seen[-1][2] == pytest.approx(.99)
    assert result[5][key]['planning_fixture_forecast']['fixture_ids'] == ['B']
    assert result[5][key]['admitted'] and not result[5][key]['physics_validated']
    np.testing.assert_array_equal(original[0].qpos0, bound[0].qpos0)
