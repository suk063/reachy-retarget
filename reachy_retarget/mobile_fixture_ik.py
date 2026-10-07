"""Source-bound mobile/right-arm IK with explicit fixture constraints.

This is isolated kinematic planning. It neither steps physics nor assigns any
object state. Successful planning still requires independent physical gates.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

from .dynamics_audit import bind_scene_assets
from .feasible_maniskill import collision_depths, intent_apertures, intent_parameters
from .geometry_clearance import RobotFixtureClearance, reachy_wheel_floor_pairs
from .physics import initialize
from .store import json_write, sha256


def array_digest(value):
    """Bind dtype, shape and exact numeric bytes, including original clocks."""
    value = np.ascontiguousarray(value)
    if value.dtype.kind not in 'biuf':
        raise ValueError('Numeric source arrays are required')
    header = json.dumps([value.dtype.str, list(value.shape)]).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def prepare(prepared, robot, output, *, anchor_index, anchor_reference,
            base_bounds_xyyaw, minimum_fixture_gap_m=.003,
            max_iterations=160, max_active_set_updates=4, support_pairs=None,
            stop_on_first_invalid=False, pose_tolerance_constraints=False):
    """Re-solve the complete path from a declared manipulation anchor.

    ``anchor_reference`` is planar base xyz-yaw followed by the 14 arm angles
    (17 scalars total). It is a seed, never accepted without a fresh check.
    The original inactive left-arm reference remains fixed at every frame.
    ``base_bounds_xyyaw`` gives explicit three pairs of optimization bounds.
    Five aperture samples include the declared closed target/contact interval
    and intent transitions. Continuous swept-volume clearance is not asserted.
    Optional early rejection retains the full original arrays and the failed
    configuration; it never admits the unvisited part of a path.
    Explicit pose constraints use the existing admission tolerances. They can
    resolve an unnecessary position/rotation tradeoff in the scalar objective.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, xml, manifest = prepared[:3]
    details = deepcopy(prepared[5])
    times, reference, commands, hands, objects = map(np.asarray, prepared[7:12])
    n = len(times)
    anchor = np.asarray(anchor_reference, float)
    bounds = np.asarray(base_bounds_xyyaw, float)
    if (type(anchor_index) is not int or not 0 <= anchor_index < n
            or anchor.shape != (17,) or not np.isfinite(anchor).all()
            or bounds.shape != (3, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError('Explicit finite anchor and planar base bounds are required')
    if (n < 2 or times.shape != (n,) or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0) or reference.shape != (n, 17)
            or commands.shape != (n,) or hands.shape != (n, 4, 4)
            or objects.shape != (n, 4, 4)
            or not all(np.isfinite(v).all() for v in (reference, commands, hands, objects))):
        raise ValueError('Complete aligned finite source arrays are required')
    if (not isinstance(max_iterations, int) or not 1 <= max_iterations <= 500
            or not isinstance(max_active_set_updates, int) or not 1 <= max_active_set_updates <= 8):
        raise ValueError('Optimization budgets must be bounded positive integers')
    if type(stop_on_first_invalid) is not bool:
        raise ValueError('stop_on_first_invalid must be an explicit boolean')
    if type(pose_tolerance_constraints) is not bool:
        raise ValueError('pose_tolerance_constraints must be an explicit boolean')
    if any(details.get(key) for key in ('robot_supported_placement', 'robot_control_retiming',
                                        'planning_fixture_forecast')):
        raise ValueError('Mobile planning requires original-clock targets before placement/retiming/fixture forecast')
    contact, target = intent_parameters(details)
    active_name = manifest['objects'][details['object_id']]['body']
    active = model.body(active_name).id
    robots = {model.body('base_link').id}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in robots:
            robots.add(body)
    if support_pairs is None:
        support_pairs = reachy_wheel_floor_pairs(model)
    checker = RobotFixtureClearance(model, active_object_bodies=(active_name,),
                                    support_pairs=support_pairs, minimum_m=minimum_fixture_gap_m)
    assets = bind_scene_assets(xml, output)
    if manifest.get('scene_asset_hashes') is not None and assets != manifest['scene_asset_hashes']:
        raise ValueError('Compiled scene assets differ from the bound source model')
    source_binding = {name: array_digest(value) for name, value in zip(
        ('time_s', 'reference', 'gripper_intent', 'hand_goals', 'object_goals'), prepared[7:12])}
    data = mujoco.MjData(model)
    arm_ids = robot.arm_ids[7:]
    lower = robot.r.model.lowerPositionLimit[arm_ids] + .031
    upper = robot.r.model.upperPositionLimit[arm_ids] - .031
    optimization_bounds = list(map(tuple, bounds)) + list(zip(lower, upper))
    qanchor = robot.pack(anchor[3:], anchor[:3])
    solutions = np.full((n, len(qanchor)), np.nan)
    errors = np.full((n, 2), np.nan)
    depths = np.full((n, 2), np.nan)
    fixture_gaps = np.full(n, np.nan)
    self_gaps, margins = np.full(n, np.nan), np.full(n, np.nan)
    apertures_saved = np.full((n, 2), np.nan)
    pairs, events = [], []
    current = qanchor.copy()
    sequence = list(range(anchor_index, -1, -1)) + list(range(anchor_index + 1, n))
    failure = None
    started = time.monotonic()
    try:
        for index in sequence:
            if index == anchor_index + 1:
                current = solutions[anchor_index].copy()
            qbase = robot.pack(reference[index, 3:], reference[index, :3])
            qbase[:4] = current[:4]
            qbase[arm_ids] = current[arm_ids]
            x0 = np.r_[robot.base(qbase), qbase[arm_ids]]
            x = x0.copy()
            goal = hands[index]
            apertures = intent_apertures(commands, index, contact, target)
            apertures_saved[index] = [apertures.min(), apertures.max()]

            def configuration(value):
                q = qbase.copy()
                q[:4] = [value[0], value[1], np.cos(value[2]), np.sin(value[2])]
                q[arm_ids] = value[3:]
                return q

            def objective(value):
                actual = robot.fk(configuration(value))[1]
                error = np.r_[goal[:3, 3] - actual[:3, 3],
                              .35 * robot.pin.log3(goal[:3, :3] @ actual[:3, :3].T)]
                return float(error @ error + 1e-8 * np.sum((value - x0) ** 2))

            def pose_constraint(value):
                actual = robot.fk(configuration(value))[1]
                position = goal[:3, 3] - actual[:3, 3]
                rotation = robot.pin.log3(goal[:3, :3] @ actual[:3, :3].T)
                # A tiny numerical interior buffer prevents equality-roundoff
                # from being mistaken for an independently admitted path.
                return np.array([1. - (position @ position)/(.002-1e-7)**2,
                                 1. - (rotation @ rotation)/(.02-1e-7)**2])

            def constraint(value):
                distances = []
                for aperture in np.unique([apertures.min(), apertures.max()]):
                    initialize(model, data, robot, configuration(value), manifest['mimics'],
                               {'l_hand_finger': 2., 'r_hand_finger': float(aperture)})
                    distances.extend(mujoco.mj_geomDistance(model, data, g, h, 2., None)
                                     - minimum_fixture_gap_m - 1e-5 for g, h in pairs)
                return np.asarray(distances)

            optimizer_trace = []
            for outer in range(max_active_set_updates):
                constraints = [dict(type='ineq', fun=constraint)] if pairs else []
                if pose_tolerance_constraints:
                    constraints.append(dict(type='ineq', fun=pose_constraint))
                optimized = minimize(objective, x, method='SLSQP', bounds=optimization_bounds,
                    constraints=constraints, options=dict(maxiter=max_iterations, ftol=1e-12))
                x = optimized.x
                q = configuration(x)
                added, collision = [], np.zeros(2)
                nearest, worst = np.inf, None
                for aperture in apertures:
                    initialize(model, data, robot, q, manifest['mimics'],
                               {'l_hand_finger': 2., 'r_hand_finger': float(aperture)})
                    found, _ = collision_depths(model, data, robots, active)
                    collision = np.maximum(collision, [found['environment'], found['self']])
                    gap = checker.inspect(data)
                    if gap['minimum_signed_distance_or_lower_bound_m'] < nearest:
                        nearest, worst = gap['minimum_signed_distance_or_lower_bound_m'], gap
                    if not gap['admitted']:
                        ids = tuple(model.geom(name).id for name in gap['closest_geom_pair'])
                        if ids not in pairs:
                            added.append(ids)
                pairs.extend(dict.fromkeys(added))
                optimizer_trace.append(dict(iteration=outer, success=bool(optimized.success),
                                             message=str(optimized.message), added_pairs=len(set(added))))
                if not added:
                    break
            current = q
            actual = robot.fk(q)[1]
            error = np.array([np.linalg.norm(goal[:3, 3] - actual[:3, 3]),
                              np.linalg.norm(robot.pin.log3(goal[:3, :3] @ actual[:3, :3].T))])
            self_gap, margin, _ = robot.r.geometry(q)
            solutions[index], errors[index], depths[index] = q, error, collision
            fixture_gaps[index], self_gaps[index], margins[index] = nearest, self_gap, margin
            if (np.any(error > [.002, .02]) or np.any(collision > .002)
                    or nearest < minimum_fixture_gap_m or self_gap < .009 or margin < .025):
                events.append(dict(frame=index, time_s=float(times[index]),
                    closed=bool(commands[index] < 0), ik_errors=error.tolist(),
                    collision_depths_m=collision.tolist(), fixture=worst,
                    self_clearance_m=float(self_gap), joint_margin_rad=float(margin),
                    optimizer_trace=optimizer_trace))
                if stop_on_first_invalid:
                    failure = 'Early rejection at original frame ' + str(index)
                    break
    except Exception as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
    complete = bool(np.isfinite(solutions).all() and failure is None)
    changed = reference.copy()
    valid = np.isfinite(solutions).all(axis=1)
    for i in np.flatnonzero(valid):
        changed[i] = np.r_[robot.base(solutions[i]), solutions[i, robot.arm_ids]]
    if complete:
        changed[:, 2] = np.unwrap(changed[:, 2])
    artifact = output / 'robot-mobile-path.npz'
    np.savez_compressed(artifact, original_time_s=times, original_reference=reference,
        reference=changed, original_hand_goals=hands, original_object_goals=objects,
        original_gripper_intent=commands, q=solutions, ik_errors=errors,
        collision_depths_m=depths, fixture_clearance_m=fixture_gaps,
        self_clearance_m=self_gaps, joint_margin_rad=margins,
        gripper_aperture_envelopes_rad=apertures_saved)
    checks = dict(completed=complete, ik=bool(complete and np.all(errors <= [.002, .02])),
        environment=bool(complete and np.all(depths[:, 0] <= .002)),
        self_collision=bool(complete and np.all(depths[:, 1] <= .002)),
        positive_fixture_clearance=bool(complete and np.all(fixture_gaps >= minimum_fixture_gap_m)),
        joint_margin=bool(complete and np.all(margins >= .025)),
        self_clearance=bool(complete and np.all(self_gaps >= .009)),
        inactive_left_reference=bool(np.array_equal(changed[:, 3:10], reference[:, 3:10])),
        source_arrays_unchanged=all(source_binding[name] == array_digest(value) for name, value in zip(
            source_binding, prepared[7:12])))
    report = dict(admitted=all(checks.values()), admission_checks=checks,
        completed_frames=int(valid.sum()), total_frames=n, exception=failure, events=events,
        method='Anchored backward/forward mobile/right-arm SLSQP with an active set of compiled fixture-distance constraints',
        anchor_index=anchor_index, anchor_reference=anchor.tolist(), base_bounds_xyyaw=bounds.tolist(),
        source_array_sha256=source_binding, scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        source_metadata=deepcopy(prepared[4]), frozen_parent_binding=details.get('frozen_plan_reuse'),
        scene_asset_hashes=assets, artifact_sha256=sha256(artifact), artifact=str(artifact),
        physics_validated=False, object_state_assignment='none; isolated robot-only FK queries',
        fixture_assumption='Original compiled reset; task fixtures are not predicted or rewritten',
        source_clock_scope='Every complete original control row retained; this module does not retime',
        controller_identity=getattr(robot, 'identity', None),
        minimum_fixture_gap_m=minimum_fixture_gap_m, support_pairs=list(support_pairs),
        gripper_closed_target_rad=target, inferred_contact_angle_rad=contact,
        aperture_scope='Five samples across declared contact/command interval and intent transitions; inactive left2rad; continuous sweep not asserted',
        max_iterations=max_iterations, max_active_set_updates=max_active_set_updates,
        stop_on_first_invalid=stop_on_first_invalid,
        pose_tolerance_constraints=pose_tolerance_constraints,
        pose_constraint_tolerances=[.002-1e-7, .02-1e-7] if pose_tolerance_constraints else None,
        incomplete_path_scope='Unvisited rows retain original references; no full-path feasibility claim',
        displacement_regularizer=1e-8, wall_s=time.monotonic() - started,
        constrained_pairs=[[model.geom(g).name, model.geom(h).name] for g, h in pairs])
    if complete:
        report.update(max_ik_errors=errors.max(0).tolist(), max_collision_depths_m=depths.max(0).tolist(),
            min_fixture_clearance_m=float(fixture_gaps.min()), min_self_clearance_m=float(self_gaps.min()),
            min_joint_margin_rad=float(margins.min()),
            peak_original_clock_reference_speed=np.max(np.abs(np.diff(changed, axis=0) /
                                                              np.diff(times)[:, None]), axis=0).tolist())
    json_write(output / 'result.json', report)
    if not report['admitted']:
        raise ValueError('Mobile fixture IK rejected; inspect ' + str(output / 'result.json'))
    # Earlier admissions concern a different joint path. Retain their reports
    # as provenance, never as evidence for this newly solved path.
    invalidated = {key: details.pop(key) for key in ('robot_initialization', 'robot_grasp_approach',
        'robot_fixture_clearance') if key in details}
    details['mobile_ik_prior_admissions'] = invalidated
    details['robot_mobile_fixture_ik'] = report
    details['robot_initialization'] = report
    result = list(prepared)
    result[5], result[6], result[8] = details, solutions[0].copy(), changed
    return tuple(result)
