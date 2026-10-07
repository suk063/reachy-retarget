"""Geometry-qualified acquisition on a fixed cylindrical pad patch.

This adapter preserves the complete original arrays. It supplies explicitly
derived robot ramps and read-only measured guards to the shared rendezvous
clock/controller. It never steps physics or assigns a moving object's state.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .cylindrical_grasp import finite_pad_faces
from .feasible_maniskill import collision_depths
from .geometry_clearance import RobotFixtureClearance, reachy_wheel_floor_pairs
from .pad_alignment import FINGERS, pad_surfaces
from .physics import initialize
from .store import json_write, sha256


def _unit(value):
    value = np.asarray(value, float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-12:
        raise ValueError('A finite nonzero three-vector is required')
    return value / np.linalg.norm(value)


def calibrated_object_tcp(details):
    """Recover the saved calibration without indexing a stale source clock."""
    calibration, cylinder = details['pad_alignment'], details['cylindrical_grasp']
    points = np.asarray(calibration['pad']['points_tcp'], float)
    midpoint = np.asarray(calibration['pad']['midpoint_tcp'], float)
    if points.shape != (2, 3) or not np.allclose(points.mean(0), midpoint, atol=1e-9, rtol=0):
        raise ValueError('The saved finite pad midpoint must match its two contacts')
    jaw = _unit(points[1] - points[0])
    longitudinal = _unit(cylinder['longitudinal_axis_tcp'])
    object_jaw = _unit(cylinder['closing_axis_after_object'])
    object_longitudinal = _unit(cylinder['longitudinal_axis_after_object'])
    if max(abs(jaw @ longitudinal), abs(object_jaw @ object_longitudinal)) > 1e-6:
        raise ValueError('Calibrated jaw and longitudinal axes must be orthogonal')
    tcp_basis = np.c_[jaw, longitudinal, np.cross(jaw, longitudinal)]
    object_basis = np.c_[object_jaw, object_longitudinal, np.cross(object_jaw, object_longitudinal)]
    transform = np.eye(4)
    transform[:3, :3] = object_basis @ tcp_basis.T
    transform[:3, 3] = np.asarray(cylinder['desired_pad_midpoint_object_m']) - transform[:3, :3] @ midpoint
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6, rtol=0):
        raise ValueError('Saved calibration does not define a rigid transform')
    return transform


def _array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def binding(prepared):
    return dict(scene_sha256=hashlib.sha256(prepared[1].encode()).hexdigest(),
                arrays={name: _array_hash(prepared[index]) for name, index in
                        [('time_s', 7), ('reference', 8), ('intent', 9), ('hand_goals', 10), ('object_goals', 11)]})


def _smooth(value):
    value = np.clip(np.asarray(value, float), 0., 1.)
    return np.clip(value**3 * (10. + value * (-15. + 6. * value)), 0., 1.)


def _ramp_intervals(duration_s, timestep_s):
    # Uniform clocks formed by arange can have a median interval a few ULPs
    # below 10 ms. Do not turn an exact ten-interval request into eleven.
    ratio = duration_s / timestep_s
    nearest = round(ratio)
    if np.isclose(ratio, nearest, atol=1e-10, rtol=0):
        ratio = nearest
    return max(2, int(np.ceil(ratio)))


def select_index(times, source_times, commands, hands, objects, object_reset, details,
                 *, radial_m=.001, axial_m=.002, lookahead_s=1., lift_limit_m=.02,
                 placement_start=None, up=(0., 0., 1.)):
    """Select the first nearby source acquisition row, never a lifted anchor."""
    times, source_times, commands = map(np.asarray, (times, source_times, commands))
    if (times.ndim != 1 or source_times.shape != times.shape or commands.shape != times.shape
            or not np.isfinite(source_times).all() or np.any(np.diff(source_times) < -1e-10)):
        raise ValueError('A complete nondecreasing source-reference clock is required')
    closed = np.flatnonzero(commands < 0)
    if not len(closed):
        raise ValueError('An explicit source close interval is required')
    first = int(closed[0])
    if first == 0:
        raise ValueError('An original open acquisition prefix is required')
    release = first + next((i for i, c in enumerate(commands[first:]) if c >= 0), len(commands)-first)
    stop = min(release, len(times) if placement_start is None else int(placement_start))
    midpoint = np.asarray(details['pad_alignment']['pad']['midpoint_tcp'])
    patch = np.asarray(details['cylindrical_grasp']['desired_pad_midpoint_object_m'])
    axis = _unit(details['cylindrical_grasp']['shape_eligibility']['axis_object'])
    projected = (hands[:, :3, :3] @ midpoint + hands[:, :3, 3] - object_reset[:3, 3]) @ object_reset[:3, :3]
    delta = projected-patch
    axial = delta @ axis
    radial = np.linalg.norm(delta-axial[:, None]*axis, axis=1)
    lift = (objects[:, :3, 3]-objects[0, :3, 3]) @ _unit(up)
    indices = np.arange(len(times))
    valid = ((indices >= first) & (indices < stop) & (source_times-source_times[first] <= lookahead_s)
             & (radial <= radial_m) & (abs(axial) <= axial_m) & (lift <= lift_limit_m))
    found = np.flatnonzero(valid)
    if not len(found):
        raise ValueError('No pre-lift source row reaches the fixed calibrated patch within the declared bounds')
    return first, int(found[0]), release, dict(radial_error_m=radial, axial_error_m=axial, source_object_lift_m=lift)


def ramps(reference, hand_goals, acquisition_index, fixed_reference, robot, count):
    """Continuous joint ramps with exact original suffix and source-row coverage."""
    if count < 2 or acquisition_index+count >= len(reference):
        raise ValueError('The exit ramp requires at least two retained subsequent source rows')
    weights = _smooth(np.linspace(0., 1., count+1))
    offset = np.asarray(fixed_reference)-reference[acquisition_index]
    if not np.array_equal(offset[:10], np.zeros(10)):
        raise ValueError('Acquisition correction may change only the active right arm')
    entry = reference[acquisition_index] + weights[:, None]*offset
    indices = np.arange(acquisition_index+1, acquisition_index+count+1)
    exit_rows = reference[indices] + (1-weights[1:, None])*offset
    entry[0] = reference[acquisition_index]
    entry[-1] = fixed_reference
    exit_rows[-1] = reference[indices[-1]]
    def goals(rows):
        return np.asarray([robot.fk(robot.pack(row[3:], row[:3]))[1] for row in rows])
    entry_goals, exit_goals = goals(entry), goals(exit_rows)
    entry_goals[0] = hand_goals[acquisition_index]
    exit_goals[-1] = hand_goals[indices[-1]]
    return dict(entry_reference=entry, entry_hand_goals=entry_goals,
                exit_reference=exit_rows, exit_hand_goals=exit_goals, exit_source_indices=indices)


def _body_subtree(model, root):
    ids = {model.body(root).id}
    for i in range(1, model.nbody):
        if int(model.body_parentid[i]) in ids:
            ids.add(i)
    return ids


def _object_clearance(model, data, object_body, *, cap=.10):
    robots = _body_subtree(model, 'base_link')
    objects = _body_subtree(model, object_body)
    distal = {model.body(n).id for n in FINGERS}
    robot_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in robots]
    object_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in objects]
    whole, other, pair = cap, cap, None
    for a in robot_geoms:
        for b in object_geoms:
            if not ((int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
                    or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))):
                continue
            distance = float(mujoco.mj_geomDistance(model, data, a, b, cap, None))
            whole = min(whole, distance)
            if int(model.geom_bodyid[a]) not in distal and distance < other:
                other, pair = distance, [model.geom(a).name, model.geom(b).name]
    return dict(whole_robot_object_gap_m=whole, non_distal_object_gap_m=other,
                closest_non_distal_object_pair=pair)


def _geometry_guard(model, data, metadata, calibrated_faces):
    body = model.body(metadata['object_body']).id
    site = model.site('r_arm_tip_tcp').id
    tcp_r, tcp_t = data.site_xmat[site].reshape(3, 3), data.site_xpos[site]
    obj_r, obj_t = data.xmat[body].reshape(3, 3), data.xpos[body]
    fixed = np.asarray(metadata['fixed_hand_goal'])
    axis = np.asarray(metadata['axis_object'])
    patch = np.asarray(metadata['patch_object_m'])
    midpoint = np.asarray(metadata['calibrated_midpoint_tcp_m'])
    forecast_midpoint = obj_r.T @ (tcp_t+tcp_r@midpoint-obj_t)
    delta = forecast_midpoint-patch
    axial = float(delta @ axis)
    radial = float(np.linalg.norm(delta-axial*axis))
    bounds = []
    for face in calibrated_faces:
        xyz = (np.asarray(face).reshape(-1, 3) @ tcp_r.T+tcp_t-obj_t) @ obj_r
        z = xyz @ axis
        bounds.append([float(z.min()), float(z.max())])
    pad = pad_surfaces(model, data)
    actual_midpoint = obj_r.T @ (tcp_t+tcp_r@pad['midpoint_tcp']-obj_t)
    actual_bounds = []
    for face in finite_pad_faces(model, data):
        xyz = (np.asarray(face).reshape(-1, 3) @ tcp_r.T+tcp_t-obj_t) @ obj_r
        z = xyz @ axis
        actual_bounds.append([float(z.min()), float(z.max())])
    gap = _object_clearance(model, data, metadata['object_body'])
    position = float(np.linalg.norm(tcp_t-fixed[:3, 3]))
    rotation = float(Rotation.from_matrix(fixed[:3, :3] @ tcp_r.T).magnitude())
    allowed = metadata['wall_interval_with_guard_m']
    checks = dict(tcp_position=position <= metadata['tcp_position_tolerance_m'],
                  tcp_rotation=rotation <= metadata['tcp_rotation_tolerance_rad'],
                  calibrated_contact_radial=radial <= metadata['radial_tolerance_m'],
                  calibrated_contact_axial=abs(axial) <= metadata['axial_tolerance_m'],
                  finite_calibrated_pad_wall=bool(np.min(bounds, axis=0)[0] >= allowed[0]
                                                  and np.max(bounds, axis=0)[1] <= allowed[1]),
                  finite_actual_pad_wall=bool(np.min(actual_bounds, axis=0)[0] >= allowed[0]
                                              and np.max(actual_bounds, axis=0)[1] <= allowed[1]),
                  non_distal_object_clearance=gap['non_distal_object_gap_m'] >= metadata['non_distal_clearance_m'])
    actual_angle = float(data.qpos[model.joint('r_hand_finger').qposadr[0]])
    near_contact = abs(actual_angle-metadata['contact_angle_rad']) <= .02
    actual_delta = actual_midpoint-patch
    actual_axial = float(actual_delta @ axis)
    actual_radial = float(np.linalg.norm(actual_delta-actual_axial*axis))
    if near_contact:
        checks['actual_contact_midpoint'] = (actual_radial <= metadata['radial_tolerance_m']
                                              and abs(actual_axial) <= metadata['axial_tolerance_m'])
    forces = {name: 0. for name in FINGERS}
    for i, contact in enumerate(data.contact):
        bodies = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
        if body not in bodies:
            continue
        for name in FINGERS:
            if model.body(name).id in bodies:
                force = np.zeros(6); mujoco.mj_contactForce(model, data, i, force)
                forces[name] += max(0., float(force[0]))
    return dict(alignment=all(checks.values()), checks=checks,
                geometry_safe=bool(checks['finite_calibrated_pad_wall'] and checks['finite_actual_pad_wall']
                                   and checks['non_distal_object_clearance']),
                target_alignment=bool(checks['tcp_position'] and checks['tcp_rotation']),
                patch_alignment=bool(checks['calibrated_contact_radial'] and checks['calibrated_contact_axial']
                                     and checks.get('actual_contact_midpoint', True)),
                failed_checks=[key for key, passed in checks.items() if not passed],
                tcp_position_error_m=position, tcp_rotation_error_rad=rotation,
                calibrated_contact_midpoint_actual_object_m=forecast_midpoint.tolist(),
                calibrated_contact_radial_error_m=radial, calibrated_contact_axial_error_m=axial,
                calibrated_face_axial_bounds_m=bounds,
                actual_aperture_face_axial_bounds_m=actual_bounds,
                actual_aperture_pad_midpoint_object_m=actual_midpoint.tolist(), actual_aperture_rad=actual_angle,
                actual_midpoint_guard_applicable=near_contact,
                pad_normal_force_N=forces,
                bilateral_force=all(v >= metadata['minimum_pad_force_N'] for v in forces.values()),
                **gap)


def guard(model, data, plan):
    """Read current measured state; do not forward, step, or mutate anything."""
    return _geometry_guard(model, data, plan['metadata'],
                           [plan['arrays']['calibrated_pad0_triangles_tcp'], plan['arrays']['calibrated_pad1_triangles_tcp']])


def verify(prepared, plan):
    if binding(prepared) != plan['metadata']['input_binding']:
        raise ValueError('Acquisition plan is bound to different source/reference/model arrays')
    path = Path(plan['metadata']['artifact'])
    if not path.is_file() or sha256(path) != plan['metadata']['artifact_sha256']:
        raise ValueError('Acquisition numeric artifact is missing or changed')
    for key, value in plan['arrays'].items():
        if _array_hash(value) != plan['metadata']['plan_array_hashes'][key]:
            raise ValueError('Acquisition in-memory arrays changed: ' + key)
    return True


def prepare(prepared, robot, output, *, source_reference_time_s=None,
            ramp_duration_s=.10, radial_tolerance_m=.001, axial_tolerance_m=.002,
            tcp_position_tolerance_m=.001, tcp_rotation_tolerance_rad=.01,
            lookahead_s=1., source_lift_limit_m=.02, minimum_fixture_gap_m=.003,
            non_distal_clearance_m=.003, minimum_pad_force_N=.5,
            max_correction_m=.02, max_correction_rad=.175, stable_s=.1, timeout_s=2.):
    """Return unchanged original arrays plus an admitted, explicitly bound plan."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/Path(__file__).name).write_text(Path(__file__).read_text())
    arrays, samples, metadata = {}, [], dict(physics_validated=False, admitted=False)
    old_active = robot.active.copy()
    failure = None
    try:
        model, xml, manifest, _, _, details, _, times, reference, commands, hands, objects = prepared
        if details.get('task') != 'PickPlaceCan' or 'cylindrical_grasp' not in details:
            raise ValueError('An explicit calibrated cylindrical Can plan is required')
        values = [ramp_duration_s, radial_tolerance_m, axial_tolerance_m, tcp_position_tolerance_m,
                  tcp_rotation_tolerance_rad, lookahead_s, source_lift_limit_m, minimum_fixture_gap_m,
                  non_distal_clearance_m, minimum_pad_force_N, max_correction_m, max_correction_rad, stable_s, timeout_s]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Positive finite acquisition bounds are required')
        dt = float(np.median(np.diff(times)))
        if dt <= 0 or not np.allclose(np.diff(times), dt, atol=1e-8, rtol=0):
            raise ValueError('Acquisition requires an explicit uniform control clock')
        if source_reference_time_s is None:
            if details.get('robot_control_retiming') or details.get('robot_supported_placement'):
                raise ValueError('Retimed or inserted plans require their full source-reference time map')
            source_reference_time_s = times
        source_times = np.asarray(source_reference_time_s, float)
        spec = manifest['objects'][details['object_id']]
        data = mujoco.MjData(model)
        object_address = int(model.joint(spec['joint']).qposadr[0])
        untouched_object = data.qpos[object_address:object_address+7].copy()
        original_q = robot.pack(reference[0, 3:], reference[0, :3])
        angle = float(details['pad_alignment']['inferred_contact_angle_rad'])
        initialize(model, data, robot, original_q, manifest['mimics'], {'l_hand_finger': 2., 'r_hand_finger': angle})
        body = model.body(spec['body']).id
        reset = np.eye(4); reset[:3, :3] = data.xmat[body].reshape(3, 3); reset[:3, 3] = data.xpos[body]
        faces = finite_pad_faces(model, data)
        object_tcp = calibrated_object_tcp(details)
        fixed = reset @ object_tcp
        placement_start = details.get('robot_supported_placement', {}).get('phase_frames', {}).get('traverse', [len(times)])[0]
        first, acquisition, release, selection = select_index(times, source_times, commands, hands, objects, reset, details,
            radial_m=radial_tolerance_m, axial_m=axial_tolerance_m, lookahead_s=lookahead_s,
            lift_limit_m=source_lift_limit_m, placement_start=placement_start, up=-model.opt.gravity)
        shift = float(np.linalg.norm(fixed[:3, 3]-hands[acquisition, :3, 3]))
        turn = float(Rotation.from_matrix(fixed[:3, :3] @ hands[acquisition, :3, :3].T).magnitude())
        if shift > max_correction_m or turn > max_correction_rad:
            raise ValueError('The fixed patch requires more than the declared robot correction bounds')
        count = _ramp_intervals(ramp_duration_s, dt)
        if acquisition+count >= min(release, placement_start):
            raise ValueError('The complete exit blend must remain inside original closed source intent')
        robot.active = robot.arm_v[7:]
        q = robot.pack(reference[acquisition, 3:], reference[acquisition, :3])
        targets = robot.fk(q); targets[1] = fixed
        q, pe, re = robot.ik(targets, q, active_hands=(1,), iterations=500, joint_margin=.031)
        if pe > .002 or re > .02:
            raise ValueError('Fixed calibrated patch IK is unreachable')
        fixed_reference = reference[acquisition].copy(); fixed_reference[10:] = q[robot.arm_ids[7:]]
        arrays = ramps(reference, hands, acquisition, fixed_reference, robot, count)
        arrays.update(original_time_s=np.asarray(times), original_source_reference_time_s=source_times,
                      original_reference=np.asarray(reference), original_hand_goals=np.asarray(hands),
                      original_object_goals=np.asarray(objects), original_gripper_intent=np.asarray(commands),
                      calibrated_pad0_triangles_tcp=faces[0], calibrated_pad1_triangles_tcp=faces[1],
                      **{'selection_'+k: v for k, v in selection.items()})
        wall = details['cylindrical_grasp']['shape_eligibility']
        metadata.update(schema='cylindrical-acquisition-v1', source_id=details.get('source_episode'),
            input_binding=binding(prepared), object_body=spec['body'], object_joint=spec['joint'],
            first_close_index=first, acquisition_index=acquisition, exit_source_indices=arrays['exit_source_indices'].tolist(),
            unchanged_suffix_start_index=int(arrays['exit_source_indices'][-1])+1,
            control_time_s=float(times[acquisition]), source_reference_time_s=float(source_times[acquisition]),
            timestep_s=dt, ramp_duration_s=count*dt, stable_s=stable_s, timeout_s=timeout_s,
            fixed_hand_goal=fixed.tolist(), fixed_object_relative_tcp=object_tcp.tolist(), object_reset_pose=reset.tolist(),
            calibrated_midpoint_tcp_m=details['pad_alignment']['pad']['midpoint_tcp'],
            patch_object_m=details['cylindrical_grasp']['desired_pad_midpoint_object_m'],
            axis_object=wall['axis_object'], wall_interval_with_guard_m=(np.asarray(wall['straight_wall_interval_m'])+[.001, -.001]).tolist(),
            contact_angle_rad=angle, radial_tolerance_m=radial_tolerance_m, axial_tolerance_m=axial_tolerance_m,
            tcp_position_tolerance_m=tcp_position_tolerance_m, tcp_rotation_tolerance_rad=tcp_rotation_tolerance_rad,
            non_distal_clearance_m=non_distal_clearance_m, minimum_pad_force_N=minimum_pad_force_N,
            correction_translation_m=shift, correction_rotation_rad=turn,
            robot_only_target_correction=True, original_arrays_unchanged=True,
            clock_policy='Repeat acquisition index during entry/settle/close; consume every exit_source_indices row exactly once, then the unchanged original suffix',
            ff_policy='Zero velocity feedforward during entry/settle/close/exit; original controller elsewhere',
            guard_scope='Saved calibrated-aperture CAD at measured TCP forecasts closing patch; current-aperture midpoint is additionally checked near contact; actual object pose is read only')
        checker = RobotFixtureClearance(model, active_object_bodies=(spec['body'],),
            support_pairs=reachy_wheel_floor_pairs(model), minimum_m=minimum_fixture_gap_m)
        robot_bodies = set(checker.metadata['robot_body_ids'])
        closed_target = float(details['effective_candidate']['gripper_closed_target'])
        if details['effective_candidate'].get('force_feedback'):
            raise ValueError('Acquisition admission requires an explicit bounded closure target')
        checks = dict(complete=True, ik=True, fixtures=True, non_distal_object=True, open_object=True,
                      self_collision=True, self_clearance=True, joint_margin=True, fixed_patch_finite_support=True)
        cases = [(i, reference[i], hands[i], [2.], 'delayed_open') for i in range(first, acquisition+1)]
        cases += [(acquisition, row, goal, [2.], 'entry') for row, goal in zip(arrays['entry_reference'], arrays['entry_hand_goals'])]
        cases += [(acquisition, arrays['entry_reference'][-1], fixed, np.linspace(closed_target, 2., 17), 'closure_sweep')]
        cases += [(int(i), row, goal, np.linspace(min(closed_target, angle), max(closed_target, angle), 5), 'exit')
                  for i, row, goal in zip(arrays['exit_source_indices'], arrays['exit_reference'], arrays['exit_hand_goals'])]
        for index, row, goal, apertures, phase in cases:
            q = robot.pack(row[3:], row[:3]); actual = robot.fk(q)[1]
            error = [float(np.linalg.norm(actual[:3, 3]-goal[:3, 3])), float(Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).magnitude())]
            sphere_gap, margin, _ = robot.r.geometry(q)
            for aperture in apertures:
                initialize(model, data, robot, q, manifest['mimics'], {'l_hand_finger': 2., 'r_hand_finger': float(aperture)})
                fixture = checker.inspect(data); gap = _object_clearance(model, data, spec['body'])
                collision, pairs = collision_depths(model, data, robot_bodies, body)
                row_checks = dict(ik=bool(np.all(np.asarray(error) <= [.002, .02])), fixtures=fixture['admitted'],
                    non_distal_object=gap['non_distal_object_gap_m'] >= non_distal_clearance_m,
                    open_object=bool(aperture != 2. or gap['whole_robot_object_gap_m'] >= 0),
                    self_collision=collision['self'] <= .002, self_clearance=sphere_gap >= .009, joint_margin=margin >= .025)
                for name, value in row_checks.items(): checks[name] &= bool(value)
                samples.append(dict(source_index=index, phase=phase, aperture_rad=float(aperture), ik_errors=error,
                    checks=row_checks, fixture=fixture, collision_depths=collision, collision_pairs=pairs,
                    self_clearance_m=float(sphere_gap), joint_margin_rad=float(margin), **gap))
        initialize(model, data, robot, robot.pack(fixed_reference[3:], fixed_reference[:3]), manifest['mimics'],
                   {'l_hand_finger': 2., 'r_hand_finger': angle})
        qualification = _geometry_guard(model, data, metadata, faces)
        checks['fixed_patch_finite_support'] = bool(qualification['alignment'])
        if not np.array_equal(untouched_object, data.qpos[object_address:object_address+7]):
            raise AssertionError('Acquisition planning changed the original object reset')
        metadata.update(admission_checks=checks, admitted=all(checks.values()),
                        completed_geometry_samples=len(samples), fixed_patch_guard=qualification,
                        object_state_unchanged=True)
        chain = np.vstack([reference[acquisition], arrays['entry_reference'], arrays['exit_reference'], reference[acquisition+count+1]])
        metadata['peak_ramp_reference_speed_xyyaw_and_arms'] = np.max(abs(np.diff(chain, axis=0))/dt, axis=0).tolist()
        if not metadata['admitted']:
            raise ValueError('Cylindrical acquisition geometry rejected')
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
    finally:
        robot.active = old_active
    artifact = output/'acquisition-plan.npz'
    np.savez_compressed(artifact, **arrays)
    metadata.update(artifact=str(artifact), artifact_sha256=sha256(artifact), exception=failure,
                    plan_array_hashes={k: _array_hash(v) for k, v in arrays.items()},
                    implementation_sha256=sha256(Path(__file__)), samples=samples)
    if failure is not None: metadata['admitted'] = False
    json_write(output/'result.json', metadata)
    if failure is not None:
        raise ValueError('Cylindrical acquisition rejected; inspect ' + str(output/'result.json'))
    result = list(prepared); result[5] = deepcopy(prepared[5]); result[5]['cylindrical_acquisition'] = metadata
    plan = dict(metadata=metadata, arrays=arrays)
    verify(tuple(result), plan)
    return tuple(result), plan
