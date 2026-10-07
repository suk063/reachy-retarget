"""Source task-fixture forecasts for isolated kinematic collision queries.

The forecast is an explicit planning assumption, never a simulator controller.
Physical rollouts must reset every free object once and integrate it normally.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from .episodes import read_episode, interpolate_pose, pose_to_matrices, matrices_to_pose
from .store import json_write, sha256


KEY = 'planning_fixture_forecast'


def _clock(value, name):
    value = np.asarray(value, float)
    if value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all() or np.any(np.diff(value) <= 0):
        raise ValueError(name+' must be a complete finite increasing clock')
    return value


def _poses(value, count, name):
    value = np.asarray(value, float)
    if (value.shape != (count, 7) or not np.isfinite(value).all()
            or not np.allclose(np.linalg.norm(value[:, 3:], axis=1), 1., rtol=0, atol=1e-6)):
        raise ValueError(name+' requires complete finite world poses with unit wxyz quaternions')
    return value


def _free_joint(model, spec):
    body = model.body(spec['body']).id
    joint = model.joint(spec['joint']).id
    if (body <= 0 or int(model.jnt_bodyid[joint]) != body
            or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE)):
        raise ValueError('Forecast requires a verified free task-object root')
    # Pose-only forecasts cannot reconstruct articulated descendants or robot
    # links. Require a separate rigid root in the original compiled scene.
    descendants = {body}
    for child in range(body+1, model.nbody):
        if int(model.body_parentid[child]) in descendants:
            descendants.add(child)
    if (int(model.body_parentid[body]) != 0
            or any(int(model.jnt_bodyid[j]) in descendants and j != joint for j in range(model.njnt))
            or any(model.body(b).name == 'base_link' for b in descendants)):
        raise ValueError('Pose-only forecast requires a rigid, nonrobot task-object subtree')
    return joint, int(model.jnt_qposadr[joint])


def bind(prepared, source_path, output, *, fixture_ids):
    """Bind unchanged normalized poses before retiming and robot corrections."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _bind(prepared, source_path, output, fixture_ids=fixture_ids)
    except (ValueError, KeyError, OSError) as error:
        json_write(output/'result.json', dict(admitted=False, physics_validated=False,
                                             reason=str(error), scope='Planning forecast binding only'))
        raise ValueError(str(error)) from error


def _bind(prepared, source_path, output, *, fixture_ids):
    model, xml, manifest, arrays, metadata, details = prepared[:6]
    if KEY in details or any(details.get(k) for k in
            ('robot_control_retiming', 'robot_supported_placement', 'robot_terminal_hold')):
        raise ValueError('Bind fixture forecast once on the original complete reference clock')
    if (not isinstance(fixture_ids, (list, tuple)) or not fixture_ids
            or any(not isinstance(key, str) for key in fixture_ids)
            or len(set(fixture_ids)) != len(fixture_ids)):
        raise ValueError('Explicit distinct task fixture IDs are required')
    active = details['object_id']
    if active in fixture_ids or not set(fixture_ids) <= set(manifest['objects']):
        raise ValueError('Forecast IDs must be task objects other than the manipulated object')
    source_path = Path(source_path).resolve()
    sidecar = source_path.with_suffix('.json')
    source, original_metadata = read_episode(source_path)
    source_time = _clock(source['time_s'], 'Source time')
    control_time = _clock(prepared[7], 'Original control time')
    hold = float(details.get('evaluation_hold_s', 0.))
    if (not np.isfinite(hold) or hold < 0
            or not np.isclose(control_time[0], source_time[0], atol=1e-9, rtol=0)
            or not np.isclose(control_time[-1], source_time[-1]+hold, atol=1e-8, rtol=0)
            or details.get('source_start_s', 0.) != 0.
            or details.get('source_time_scale', 1.) != 1.
            or details.get('velocity_dilation', 1.) != 1.):
        raise ValueError('Original source clock plus explicit endpoint evaluation hold is required')
    if not np.array_equal(source_time, arrays['time_s']):
        raise ValueError('Prepared source clock differs from the bound HDF5')
    active_key = 'objects/'+active+'/pose'
    source_active = _poses(source[active_key], len(source_time), active_key)
    if not np.array_equal(source_active, arrays[active_key]):
        raise ValueError('Prepared active-object source differs from bound HDF5')
    if original_metadata['objects'][active].get('pose_frame') != 'world':
        raise ValueError('Explicit world-frame source object poses are required')
    native_active = pose_to_matrices(interpolate_pose(source_time, source_active, control_time))
    reference = np.asarray(prepared[11])
    if reference.shape != native_active.shape or not np.isfinite(reference).all():
        raise ValueError('Complete active-object reference required to bind scene transform')
    transform = reference[0] @ np.linalg.inv(native_active[0])
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-8, rtol=0)
            or not np.allclose(reference, transform@native_active, atol=1e-7, rtol=0)):
        raise ValueError('Source clock and a single rigid scene transform must reproduce every active-object reference')
    _, active_q = _free_joint(model, manifest['objects'][active])
    if not np.allclose(pose_to_matrices(model.qpos0[active_q:active_q+7][None])[0], reference[0], atol=1e-7, rtol=0):
        raise ValueError('Active-object frame zero differs from original compiled reset')
    values = dict(native_source_time_s=source_time, binding_control_time_s=control_time,
                  binding_active_object_reference=reference, source_to_scene=transform)
    bindings = []
    for index, oid in enumerate(fixture_ids):
        key = 'objects/'+oid+'/pose'
        poses = _poses(source[key], len(source_time), key)
        if (not np.array_equal(poses, arrays[key])
                or original_metadata['objects'][oid].get('pose_frame') != 'world'):
            raise ValueError('Prepared fixture poses must equal the bound world-frame source')
        joint, address = _free_joint(model, manifest['objects'][oid])
        reset = model.qpos0[address:address+7].copy()
        if not np.allclose(pose_to_matrices(reset[None]), transform@pose_to_matrices(poses[:1]), atol=1e-7, rtol=0):
            raise ValueError('Fixture frame zero differs from original compiled reset: '+oid)
        field = 'fixture_pose_'+str(index)
        values[field] = poses
        bindings.append(dict(object_id=oid, body=manifest['objects'][oid]['body'],
                             joint=manifest['objects'][oid]['joint'], source_channel=key,
                             artifact_field=field, original_reset_pose_world=reset.tolist(),
                             qpos_address=address, joint_id=joint))
    artifact = output/'source-fixtures.npz'
    np.savez_compressed(artifact, **values)
    report = dict(schema='planning-task-fixture-forecast-v1', status='bound', physics_validated=False,
                  source_path=str(source_path), source_sha256=sha256(source_path),
                  source_sidecar=str(sidecar) if sidecar.exists() else None,
                  source_sidecar_sha256=sha256(sidecar) if sidecar.exists() else None,
                  source_group=original_metadata.get('source_group'), source_sequence=original_metadata.get('source_sequence'),
                  source_provenance=original_metadata.get('provenance'),
                  source_metadata_objects=original_metadata['objects'],
                  scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                  source_to_scene=transform.tolist(), active_object_id=active,
                  native_source_rows=len(source_time), binding_control_rows=len(control_time),
                  evaluation_hold_s=hold, fixtures=bindings,
                  original_fixture_initial_joint_positions=deepcopy(manifest.get('fixture_initial_joint_positions', {})),
                  artifact=str(artifact), artifact_sha256=sha256(artifact),
                  assumption='Inactive task objects follow their unchanged recorded source poses for kinematic planning only. Retiming uses the bound source-clock map; evaluation holds repeat the final recorded pose.',
                  physical_policy='Forecast is never applied to physical data. All task objects reset once and then remain fully dynamic; actual contacts and original gates decide success.')
    result = list(prepared)
    result[5] = deepcopy(details)
    invalidated = {}
    for name in ('robot_initialization', 'robot_grasp_approach', 'robot_fixture_clearance'):
        if name in result[5]:
            invalidated[name] = result[5].pop(name)
    report['invalidated_previous_admissions'] = invalidated
    report['requires_fresh_full_ik_and_collision_admission'] = True
    json_write(output/'result.json', report)
    result[5][KEY] = report
    return tuple(result)


class KinematicFixtureQuery:
    """Own isolated, nonintegrated MjData; never accept rollout data from callers."""
    def __init__(self, prepared):
        self.model = model = prepared[0]
        binding = prepared[5][KEY]
        if (sha256(binding['artifact']) != binding['artifact_sha256']
                or sha256(binding['source_path']) != binding['source_sha256']
                or hashlib.sha256(prepared[1].encode()).hexdigest() != binding['scene_sha256']):
            raise ValueError('Planning fixture source, scene, or artifact checksum mismatch')
        sidecar = Path(binding['source_path']).with_suffix('.json')
        if ((sha256(sidecar) if sidecar.exists() else None) != binding['source_sidecar_sha256']):
            raise ValueError('Planning fixture source metadata checksum mismatch')
        if prepared[5].get('robot_supported_placement'):
            raise ValueError('Supported placement requires its own verified fixture-forecast admission')
        with np.load(binding['artifact'], allow_pickle=False) as f:
            values = {key: f[key] for key in f.files}
        original = _clock(values['binding_control_time_s'], 'Bound reference time')
        current = _clock(prepared[7], 'Current reference time')
        timing = prepared[5].get('robot_control_retiming')
        timing_binding = None
        if timing:
            if (not timing.get('original_interval_complete')
                    or not timing.get('source_clock_artifact')
                    or sha256(timing['source_clock_artifact']) != timing.get('artifact_sha256')):
                raise ValueError('Complete checksum-bound retiming map required for fixture forecast')
            with np.load(timing['source_clock_artifact'], allow_pickle=False) as f:
                if 'original_time_s' not in f or not np.array_equal(f['original_time_s'], original):
                    raise ValueError('Fixture retiming map belongs to a different original clock')
                if not np.array_equal(f['control_time_s'], current):
                    raise ValueError('Fixture retiming map does not cover current control clock')
                mapped = f['source_reference_time_s'].copy()
            timing_binding = dict(path=timing['source_clock_artifact'], sha256=timing['artifact_sha256'])
        else:
            if not np.array_equal(current, original):
                raise ValueError('Changed reference clock requires an explicit retiming map')
            mapped = current
        if (mapped.shape != current.shape or not np.isfinite(mapped).all()
                or np.any(np.diff(mapped) < -1e-10)
                or not np.isclose(mapped[0], original[0], atol=1e-9, rtol=0)
                or not np.isclose(mapped[-1], original[-1], atol=1e-9, rtol=0)
                or mapped.min() < original[0]-1e-9 or mapped.max() > original[-1]+1e-9):
            raise ValueError('Fixture map must preserve the complete original reference interval')
        expected = pose_to_matrices(interpolate_pose(original, matrices_to_pose(values['binding_active_object_reference']), mapped))
        if not np.allclose(expected, prepared[11], atol=1e-7, rtol=0):
            raise ValueError('Retiming correspondence does not reproduce active-object references')
        source_time = values['native_source_time_s']
        self.poses, self.addresses = [], []
        for fixture in binding['fixtures']:
            oid = fixture['object_id']
            _, address = _free_joint(model, prepared[2]['objects'][oid])
            if (address != fixture['qpos_address']
                    or not np.array_equal(model.qpos0[address:address+7], fixture['original_reset_pose_world'])):
                raise ValueError('Fixture identity or original reset changed after binding')
            self.addresses.append(address)
            native = interpolate_pose(source_time, values[fixture['artifact_field']], mapped)
            self.poses.append(matrices_to_pose(values['source_to_scene']@pose_to_matrices(native)))
        self.data = mujoco.MjData(model)
        from .object_scene import initialize_fixtures
        initialize_fixtures(model, self.data, prepared[2])
        self.times, self.source_times = current, np.clip(mapped, source_time[0], source_time[-1])
        self.metadata = dict(binding_artifact_sha256=binding['artifact_sha256'],
                             source_sha256=binding['source_sha256'], fixture_ids=[x['object_id'] for x in binding['fixtures']],
                             source_clock_map=timing_binding, rows=len(current),
                             scope=binding['assumption'], physical_policy=binding['physical_policy'])

    def synchronize(self, index):
        if self.data.time != 0 or np.any(self.data.qvel != 0):
            raise ValueError('Fixture forecasts are forbidden on integrated or moving simulation data')
        if not 0 <= index < len(self.times):
            raise IndexError('Fixture query frame is outside the complete reference')
        for address, poses in zip(self.addresses, self.poses):
            self.data.qpos[address:address+7] = poses[index]
        mujoco.mj_forward(self.model, self.data)

    def save(self, output):
        path = Path(output)/'planning-fixture-query.npz'
        np.savez_compressed(path, control_time_s=self.times, native_source_time_s=self.source_times,
                            **{'fixture_pose_'+str(i): p for i, p in enumerate(self.poses)})
        return dict(self.metadata, query_artifact=str(path), query_artifact_sha256=sha256(path))


def make_query(prepared):
    return KinematicFixtureQuery(prepared) if KEY in prepared[5] else None
