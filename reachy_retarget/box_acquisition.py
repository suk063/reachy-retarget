"""Measured acquisition at an unchanged native box-component grasp patch.

The complete original references remain immutable. Explicit robot-only entry
and exit ramps feed the shared acquisition clock; measured guards never step
physics or assign an object's state.
"""
from copy import deepcopy
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from .cylindrical_acquisition import (_array_hash, _ramp_intervals, _object_clearance,
                                      _unit, binding, ramps, verify)
from .cylindrical_grasp import finite_pad_faces
from .feasible_maniskill import collision_depths
from .geometry_clearance import RobotFixtureClearance, reachy_wheel_floor_pairs
from .grasp_region import component
from .pad_alignment import FINGERS, pad_surfaces
from .physics import initialize
from .store import json_write, sha256


def polygon_area(points):
    points = np.asarray(points, float)
    if len(points) < 3:
        return 0.
    return abs(float(np.sum(points[:, 0]*np.roll(points[:, 1], -1)
                            - points[:, 1]*np.roll(points[:, 0], -1)))*.5)


def clipped_triangle_area(triangle, half):
    """Exact convex polygon clipping against the original rectangular face."""
    polygon = [p.copy() for p in np.asarray(triangle, float)]
    for axis in range(2):
        for sign in (-1, 1):
            if not polygon:
                return 0.
            clipped = []
            previous = polygon[-1]
            old_distance = half[axis]-sign*previous[axis]
            for point in polygon:
                distance = half[axis]-sign*point[axis]
                if (old_distance >= 0) != (distance >= 0):
                    clipped.append(previous+(point-previous)*old_distance/(old_distance-distance))
                if distance >= 0:
                    clipped.append(point)
                previous, old_distance = point, distance
            polygon = clipped
    return polygon_area(polygon)


def projected_support(faces, object_tcp, points_tcp, metadata):
    geometry = metadata['component']
    center = np.asarray(geometry['center_object_m'])
    axes = np.asarray(geometry['axes_object'])
    half = np.asarray(geometry['half_size_m'])
    tangent = [i for i in range(3) if i != metadata['face_axis']]
    points = (np.asarray(points_tcp)@object_tcp[:3, :3].T+object_tcp[:3, 3]-center)@axes
    margins = half[tangent]-abs(points[:, tangent])
    pads = []
    for face in faces:
        coordinates = (face@object_tcp[:3, :3].T+object_tcp[:3, 3]-center)@axes
        total = sum(polygon_area(triangle[:, tangent]) for triangle in coordinates)
        supported = sum(clipped_triangle_area(triangle[:, tangent], half[tangent]) for triangle in coordinates)
        if not np.isfinite(total+supported) or total <= 1e-12:
            raise ValueError('Finite pad face must have a nonzero projection on the native box face')
        pads.append(dict(projected_area_m2=total, supported_area_m2=supported,
                         supported_fraction=float(np.clip(supported/total, 0., 1.))))
    return dict(pads=pads, contact_center_edge_margin_m=float(margins.min()),
                contact_centers_component_m=points.tolist(),
                scope='Exact projected CAD triangle overlap; not a measured contact-pressure area')


def select_index(times, source_times, commands, hands, objects, object_reset, details,
                 *, radial_m=.002, axial_m=.001, lookahead_s=1., lift_limit_m=.015,
                 placement_start=None, up=(0., 0., 1.)):
    times, source_times, commands = map(np.asarray, (times, source_times, commands))
    if (times.ndim != 1 or source_times.shape != times.shape or commands.shape != times.shape
            or not np.isfinite(source_times).all() or np.any(np.diff(source_times) < -1e-10)):
        raise ValueError('A complete nondecreasing source-reference clock is required')
    closed = np.flatnonzero(commands < 0)
    if not len(closed) or closed[0] == 0:
        raise ValueError('An original open prefix and explicit source close interval are required')
    first = int(closed[0])
    release = first+next((i for i, c in enumerate(commands[first:]) if c >= 0), len(commands)-first)
    stop = min(release, len(times) if placement_start is None else int(placement_start))
    proposal = details['grasp_attachment']; geometry = proposal['shape_eligibility']
    midpoint = np.asarray(details['pad_alignment']['pad']['midpoint_tcp'])
    patch = np.asarray(geometry['center_object_m'])
    axis = np.asarray(geometry['axes_object'])[:, proposal['face_axis']]
    projected = (hands[:, :3, :3]@midpoint+hands[:, :3, 3]-object_reset[:3, 3])@object_reset[:3, :3]
    delta = projected-patch
    axial = delta@axis
    radial = np.linalg.norm(delta-axial[:, None]*axis, axis=1)
    lift = (objects[:, :3, 3]-objects[0, :3, 3])@_unit(up)
    indices = np.arange(len(times))
    valid = ((indices >= first) & (indices < stop) & (source_times-source_times[first] <= lookahead_s)
             & (radial <= radial_m) & (abs(axial) <= axial_m) & (lift <= lift_limit_m))
    found = np.flatnonzero(valid)
    if not len(found):
        raise ValueError('No pre-lift source row reaches the fixed native box patch within declared bounds')
    return first, int(found[0]), release, dict(radial_error_m=radial, axial_error_m=axial, source_object_lift_m=lift)


def _geometry_guard(model, data, metadata, faces):
    body = model.body(metadata['object_body']).id
    site = model.site('r_arm_tip_tcp').id
    tcp = np.eye(4); tcp[:3, :3] = data.site_xmat[site].reshape(3, 3); tcp[:3, 3] = data.site_xpos[site]
    obj = np.eye(4); obj[:3, :3] = data.xmat[body].reshape(3, 3); obj[:3, 3] = data.xpos[body]
    relative = np.linalg.inv(obj)@tcp
    fixed, attachment = np.asarray(metadata['fixed_hand_goal']), np.asarray(metadata['fixed_object_relative_tcp'])
    forecast = relative[:3, :3]@np.asarray(metadata['calibrated_midpoint_tcp_m'])+relative[:3, 3]
    delta = forecast-np.asarray(metadata['patch_object_m'])
    axis = np.asarray(metadata['axis_object']); axial = float(delta@axis)
    radial = float(np.linalg.norm(delta-axial*axis))
    position = float(np.linalg.norm(tcp[:3, 3]-fixed[:3, 3]))
    rotation = float(Rotation.from_matrix(fixed[:3, :3]@tcp[:3, :3].T).magnitude())
    relative_rotation = float(Rotation.from_matrix(attachment[:3, :3]@relative[:3, :3].T).magnitude())
    support = projected_support(faces, relative, metadata['calibrated_points_tcp_m'], metadata)
    gap = _object_clearance(model, data, metadata['object_body'])
    checks = dict(tcp_position=position <= metadata['tcp_position_tolerance_m'],
                  tcp_rotation=rotation <= metadata['tcp_rotation_tolerance_rad'],
                  relative_rotation=relative_rotation <= metadata['relative_rotation_tolerance_rad'],
                  calibrated_contact_radial=radial <= metadata['radial_tolerance_m'],
                  calibrated_contact_axial=abs(axial) <= metadata['axial_tolerance_m'],
                  finite_projected_support=min(p['supported_fraction'] for p in support['pads']) >= metadata['minimum_supported_fraction'],
                  contact_center_edge_margin=support['contact_center_edge_margin_m'] >= metadata['minimum_center_edge_margin_m'],
                  non_distal_object_clearance=gap['non_distal_object_gap_m'] >= metadata['non_distal_clearance_m'])
    pad = pad_surfaces(model, data)
    actual = relative[:3, :3]@pad['midpoint_tcp']+relative[:3, 3]
    angle = float(data.qpos[model.joint('r_hand_finger').qposadr[0]])
    near = abs(angle-metadata['contact_angle_rad']) <= .02
    if near:
        difference = actual-np.asarray(metadata['patch_object_m']); along = float(difference@axis)
        checks['actual_contact_midpoint'] = bool(abs(along) <= metadata['axial_tolerance_m']
            and np.linalg.norm(difference-along*axis) <= metadata['radial_tolerance_m'])
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
                geometry_safe=bool(checks['finite_projected_support'] and checks['contact_center_edge_margin'] and checks['non_distal_object_clearance']),
                target_alignment=bool(checks['tcp_position'] and checks['tcp_rotation']),
                patch_alignment=bool(checks['calibrated_contact_radial'] and checks['calibrated_contact_axial'] and checks['relative_rotation'] and checks.get('actual_contact_midpoint', True)),
                failed_checks=[key for key, passed in checks.items() if not passed],
                tcp_position_error_m=position, tcp_rotation_error_rad=rotation,
                object_relative_tcp_rotation_error_rad=relative_rotation,
                calibrated_contact_midpoint_actual_object_m=forecast.tolist(),
                calibrated_contact_radial_error_m=radial, calibrated_contact_axial_error_m=axial,
                finite_projected_support=support, actual_aperture_pad_midpoint_object_m=actual.tolist(),
                actual_aperture_rad=angle, actual_midpoint_guard_applicable=near,
                pad_normal_force_N=forces,
                bilateral_force=all(value >= metadata['minimum_pad_force_N'] for value in forces.values()), **gap)


def guard(model, data, plan):
    """Read-only actual patch, box-face support, orientation and per-pad forces."""
    return _geometry_guard(model, data, plan['metadata'],
                           [plan['arrays']['calibrated_pad0_triangles_tcp'], plan['arrays']['calibrated_pad1_triangles_tcp']])


def refine_fixed_patch(robot, q, target, object_reset, midpoint, patch, axis,
                       *, position_m, rotation_rad, axial_m, radial_m,
                       joint_margin=.031):
    """Find guard margin with fixed base and unchanged right-arm joint bounds.

    A weighted pose residual can miss a feasible position/rotation tolerance
    box. The explicit constraints retain each independent guard, then minimize
    the largest normalized residual. Full scene admission remains mandatory.
    """
    original = np.asarray(q, float).copy()
    ids = np.asarray(robot.arm_ids[7:])
    limits = np.array([position_m, rotation_rad, axial_m, radial_m], float)
    if (len(ids) != 7 or not np.isfinite(limits).all() or np.any(limits <= 0)
            or not np.isfinite(joint_margin) or joint_margin <= 0):
        raise ValueError('Seven right-arm joints and positive guard bounds are required')
    bounds = np.column_stack([robot.r.model.lowerPositionLimit[ids]+joint_margin,
                              robot.r.model.upperPositionLimit[ids]-joint_margin])
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('Right-arm joint margin leaves invalid bounds')
    inverse_reset = np.linalg.inv(object_reset)
    axis = _unit(axis)

    def errors(values):
        candidate = original.copy(); candidate[ids] = values
        actual = robot.fk(candidate)[1]
        relative = inverse_reset@actual
        delta = relative[:3, :3]@midpoint+relative[:3, 3]-patch
        along = float(delta@axis)
        return np.array([np.linalg.norm(actual[:3, 3]-target[:3, 3]),
                         Rotation.from_matrix(target[:3, :3]@actual[:3, :3].T).magnitude(),
                         abs(along), np.linalg.norm(delta-along*axis)])/limits

    seed = np.clip(original[ids], bounds[:, 0], bounds[:, 1])
    before = errors(seed)
    feasible = minimize(lambda values: float(np.sum((values-seed)**2)), seed,
        method='SLSQP', bounds=bounds,
        constraints=[{'type': 'ineq', 'fun': lambda values: .99-errors(values)}],
        options={'maxiter': 1200, 'ftol': 1e-12})
    initial = np.r_[feasible.x, max(errors(feasible.x))]
    maximum = max(2., float(initial[-1]))
    refined = minimize(lambda values: float(values[-1]), initial,
        method='SLSQP', bounds=[tuple(row) for row in bounds]+[(0., maximum)],
        constraints=[{'type': 'ineq', 'fun': lambda values: values[-1]-errors(values[:-1])}],
        options={'maxiter': 1200, 'ftol': 1e-12})
    # Independently score all attempted values; optimizer status is not admission.
    attempts = [seed, feasible.x, refined.x[:-1]]
    eligible = [values for values in attempts if np.isfinite(values).all()
                and np.all(values >= bounds[:, 0]-1e-12)
                and np.all(values <= bounds[:, 1]+1e-12)]
    selected = min(eligible, key=lambda values: float(np.max(errors(values))))
    result = original.copy(); result[ids] = selected
    after = errors(selected)
    report = dict(method='fixed_base_right_arm_explicit_guard_constraints_minimax',
        normalized_error_names=['tcp_position', 'tcp_rotation', 'patch_axial', 'patch_radial'],
        normalized_errors_before=before.tolist(), normalized_errors_after=after.tolist(),
        maximum_normalized_error=float(max(after)), joint_margin_rad=joint_margin,
        base_and_inactive_arm_unchanged=bool(np.array_equal(np.delete(result, ids), np.delete(original, ids))),
        within_guard_bounds=bool(np.isfinite(after).all() and np.all(after <= 1.)),
        stages=[dict(success=bool(stage.success), message=str(stage.message), iterations=int(stage.nit))
                for stage in [feasible, refined]])
    return result, report


def prepare(prepared, robot, output, *, source_reference_time_s=None,
            ramp_duration_s=.10, radial_tolerance_m=.002, axial_tolerance_m=.001,
            tcp_position_tolerance_m=.001, tcp_rotation_tolerance_rad=.01,
            lookahead_s=1., source_lift_limit_m=.015, minimum_fixture_gap_m=.003,
            non_distal_clearance_m=.003, minimum_pad_force_N=.1,
            max_correction_m=.02, max_correction_rad=.175, stable_s=.1, timeout_s=2.,
            minimum_supported_fraction=.99, minimum_center_edge_margin_m=.01, relative_rotation_tolerance_rad=.01):
    """Return unchanged original arrays plus an admitted, explicitly bound plan."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/Path(__file__).name).write_text(Path(__file__).read_text())
    arrays, samples, metadata = {}, [], dict(physics_validated=False, admitted=False)
    old_active = robot.active.copy()
    failure = None
    try:
        model, xml, manifest, _, _, details, _, times, reference, commands, hands, objects = prepared
        if details.get('task') != 'Threading_D0' or details.get('grasp_attachment', {}).get('method') != 'native_grasp_box_component':
            raise ValueError('An explicitly calibrated native Threading handle component is required')
        if details.get('robot_control_retiming') or details.get('robot_supported_placement'):
            raise ValueError('Box acquisition requires the original calibrated reference clock; remapped anchors are not supported')
        values = [ramp_duration_s, radial_tolerance_m, axial_tolerance_m, tcp_position_tolerance_m,
                  tcp_rotation_tolerance_rad, lookahead_s, source_lift_limit_m, minimum_fixture_gap_m,
                  non_distal_clearance_m, minimum_pad_force_N, max_correction_m, max_correction_rad, stable_s, timeout_s,
                  minimum_supported_fraction, minimum_center_edge_margin_m, relative_rotation_tolerance_rad]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Positive finite acquisition bounds are required')
        if minimum_supported_fraction > 1:
            raise ValueError('Projected supported fraction must be at most one')
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
        proposal = details['grasp_attachment']
        geometry = component(model, spec['body'], spec['joint'], proposal['shape_eligibility']['source_grasp_point_object_m'])
        if geometry != proposal['shape_eligibility']:
            raise ValueError('Bound native box component differs from the unchanged scene')
        anchor = int(details['pad_alignment']['retarget_anchor_frame'])
        if not 0 <= anchor < len(hands):
            raise ValueError('Original calibration anchor is outside the bound reference clock')
        object_tcp = np.linalg.inv(objects[anchor])@hands[anchor]
        patch = np.asarray(geometry['center_object_m'])
        mapped = object_tcp[:3, :3]@np.asarray(details['pad_alignment']['pad']['midpoint_tcp'])+object_tcp[:3, 3]
        if not np.allclose(mapped, patch, atol=1e-6, rtol=0):
            raise ValueError('Saved calibrated midpoint does not lie on the native component center')
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
        q, refinement = refine_fixed_patch(robot, q, fixed, reset,
            np.asarray(details['pad_alignment']['pad']['midpoint_tcp']), patch,
            np.asarray(geometry['axes_object'])[:, proposal['face_axis']],
            position_m=tcp_position_tolerance_m, rotation_rad=min(tcp_rotation_tolerance_rad, relative_rotation_tolerance_rad),
            axial_m=axial_tolerance_m, radial_m=radial_tolerance_m)
        metadata['fixed_patch_ik_refinement'] = refinement
        actual_fixed = robot.fk(q)[1]
        pe = float(np.linalg.norm(actual_fixed[:3, 3]-fixed[:3, 3]))
        re = float(Rotation.from_matrix(fixed[:3, :3]@actual_fixed[:3, :3].T).magnitude())
        if pe > .002 or re > .02:
            raise ValueError('Fixed calibrated patch IK is unreachable')
        fixed_reference = reference[acquisition].copy(); fixed_reference[10:] = q[robot.arm_ids[7:]]
        arrays = ramps(reference, hands, acquisition, fixed_reference, robot, count)
        arrays.update(original_time_s=np.asarray(times), original_source_reference_time_s=source_times,
                      original_reference=np.asarray(reference), original_hand_goals=np.asarray(hands),
                      original_object_goals=np.asarray(objects), original_gripper_intent=np.asarray(commands),
                      calibrated_pad0_triangles_tcp=faces[0], calibrated_pad1_triangles_tcp=faces[1],
                      **{'selection_'+k: v for k, v in selection.items()})
        axis = np.asarray(geometry['axes_object'])[:, proposal['face_axis']]
        metadata.update(schema='box-acquisition-v1', source_id=details.get('source_episode'),
            input_binding=binding(prepared), object_body=spec['body'], object_joint=spec['joint'],
            first_close_index=first, acquisition_index=acquisition, exit_source_indices=arrays['exit_source_indices'].tolist(),
            unchanged_suffix_start_index=int(arrays['exit_source_indices'][-1])+1,
            control_time_s=float(times[acquisition]), source_reference_time_s=float(source_times[acquisition]),
            timestep_s=dt, ramp_duration_s=count*dt, stable_s=stable_s, timeout_s=timeout_s,
            fixed_hand_goal=fixed.tolist(), fixed_object_relative_tcp=object_tcp.tolist(), object_reset_pose=reset.tolist(),
            calibrated_midpoint_tcp_m=details['pad_alignment']['pad']['midpoint_tcp'],
            patch_object_m=patch.tolist(), component=geometry, face_axis=proposal['face_axis'],
            axis_object=axis.tolist(), calibrated_points_tcp_m=details['pad_alignment']['pad']['points_tcp'],
            minimum_supported_fraction=minimum_supported_fraction, minimum_center_edge_margin_m=minimum_center_edge_margin_m,
            relative_rotation_tolerance_rad=relative_rotation_tolerance_rad,
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
            raise ValueError('Box acquisition geometry rejected')
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
        raise ValueError('Box acquisition rejected; inspect ' + str(output/'result.json'))
    result = list(prepared); result[5] = deepcopy(prepared[5]); result[5]['box_acquisition'] = metadata
    plan = dict(metadata=metadata, arrays=arrays)
    verify(tuple(result), plan)
    return tuple(result), plan
