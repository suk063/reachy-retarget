"""Repair only new robot samples introduced by an explicit source time map.

Original source knots, hand/object targets, intents and clocks remain immutable.
This is a planning correction, never a simulation-state assignment or success.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .feasible_maniskill import collision_depths
from .geometry_clearance import RobotFixtureClearance, reachy_wheel_floor_pairs
from .mobile_placement_ik import Solver
from .object_scene import initialize_fixtures
from .physics import initialize
from .store import json_write, sha256

ACTIVE = np.r_[np.arange(3), np.arange(10, 17)]


class RepairRejected(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def repair_reference(reference, times, source_clock, original_times, original_reference,
                     speed, inspect, solve, *, minimum_m=.003, planar_speed=None,
                     maximum_correction=(.002, .002, .01, .01, .01, .01, .01, .01, .01, .01)):
    """Bounded callback-based repair with original-knot and neighbor checks.

    ``inspect(ref)`` returns the exact admitted fixture lower bound. ``solve``
    receives (row, ref, lower, upper), with bounds only for ACTIVE coordinates.
    It must return one complete reference. No collision or speed gate is relaxed.
    """
    reference, times, source_clock, original_times, original_reference, speed = map(
        lambda x: np.asarray(x, float),
        (reference, times, source_clock, original_times, original_reference, speed))
    maximum_correction = np.asarray(maximum_correction, float)
    if planar_speed is not None and (isinstance(planar_speed,bool) or not np.isfinite(planar_speed) or planar_speed<=0):
        raise ValueError('A finite positive planar speed limit is required')
    if (reference.ndim != 2 or reference.shape[1] != 17 or len(reference) < 2
            or original_reference.shape != (len(original_times), 17)
            or times.shape != (len(reference),) or source_clock.shape != times.shape
            or speed.shape != (17,) or maximum_correction.shape != (10,)
            or not all(np.isfinite(x).all() for x in (reference, times, source_clock,
                original_times, original_reference, speed, maximum_correction))
            or len(original_times) < 2 or np.any(np.diff(times) <= 0)
            or np.any(np.diff(original_times) <= 0) or np.any(np.diff(source_clock) < 0)
            or np.any(speed <= 0) or np.any(maximum_correction <= 0)
            or not np.isfinite(minimum_m) or not 0 < minimum_m <= .05):
        raise ValueError('Finite aligned original/retimed arrays, bounds and complete clocks required')
    expected = np.column_stack([np.interp(source_clock, original_times, original_reference[:, j])
                                for j in range(17)])
    if not np.allclose(reference, expected, rtol=0, atol=1e-11):
        raise ValueError('Input is not the bound linear retiming of original robot knots')
    if not np.allclose(source_clock[[0, -1]], original_times[[0, -1]], rtol=0, atol=1e-11):
        raise ValueError('Complete original source interval required')
    original_gaps = np.array([inspect(row) for row in original_reference])
    before = np.array([inspect(row) for row in reference])
    changed = reference.copy()
    protected = np.min(abs(source_clock[:, None]-original_times[None, :]), axis=1) <= 1e-11
    evidence = dict(original_gaps_m=original_gaps, before_gaps_m=before,
                    corrected_reference=changed, protected_rows=protected, repaired_rows=[],
                    status='checking', minimum_m=float(minimum_m))
    if not np.isfinite(original_gaps).all() or not np.isfinite(before).all():
        raise RepairRejected('Non-finite fixture query', evidence)
    if np.any(original_gaps < minimum_m):
        evidence['status'] = 'original_knot_rejected'
        raise RepairRejected('Original source knots already violate fixture clearance', evidence)
    for i in np.flatnonzero(before < minimum_m):
        if protected[i] or i == 0 or i == len(reference)-1:
            evidence['status'] = 'protected_row_rejected'
            raise RepairRejected('A protected original knot cannot be repaired as interpolation', evidence)
        lower = np.maximum.reduce([reference[i, ACTIVE]-maximum_correction,
            changed[i-1, ACTIVE]-speed[ACTIVE]*(times[i]-times[i-1]),
            changed[i+1, ACTIVE]-speed[ACTIVE]*(times[i+1]-times[i])])
        upper = np.minimum.reduce([reference[i, ACTIVE]+maximum_correction,
            changed[i-1, ACTIVE]+speed[ACTIVE]*(times[i]-times[i-1]),
            changed[i+1, ACTIVE]+speed[ACTIVE]*(times[i+1]-times[i])])
        # Saturated linear segments can differ by a few floating-point ulps.
        # Collapse only that numerical interval; the complete speed gate below
        # still uses the original declared limit and its existing tolerance.
        roundoff = (lower > upper) & (lower-upper <= 1e-12)
        midpoint = (lower+upper)*.5
        lower[roundoff] = midpoint[roundoff]
        upper[roundoff] = midpoint[roundoff]
        if np.any(lower > upper):
            evidence['status'] = 'neighbor_speed_bounds_rejected'
            evidence['failed_row'] = int(i)
            raise RepairRejected('No correction fits both unchanged neighbor speed limits', evidence)
        try:
            discs = None if planar_speed is None else np.array([
                [*changed[i-1,:2],planar_speed*(times[i]-times[i-1])],
                [*changed[i+1,:2],planar_speed*(times[i+1]-times[i])]])
            options = {} if discs is None else dict(planar_constraints=discs)
            proposed = np.asarray(solve(int(i), changed[i].copy(), lower, upper,**options), float)
        except Exception as exc:
            evidence['status'] = 'solver_exception'
            raise RepairRejected('Interpolation solver failed: '+str(exc), evidence) from exc
        valid = (proposed.shape == (17,) and np.isfinite(proposed).all()
                 and np.array_equal(proposed[3:10], reference[i, 3:10])
                 and np.all(proposed[ACTIVE] >= lower-1e-11)
                 and np.all(proposed[ACTIVE] <= upper+1e-11))
        if valid and discs is not None:
            valid=bool(np.all(np.linalg.norm(proposed[:2]-discs[:,:2],axis=1)<=discs[:,2]+1e-11))
        gap = float(inspect(proposed)) if valid else float('nan')
        evidence['repaired_rows'].append(dict(row=int(i), source_time_s=float(source_clock[i]),
            before_gap_m=float(before[i]), after_gap_m=gap if np.isfinite(gap) else None,
            maximum_coordinate_change=float(np.max(abs(proposed-reference[i]))) if valid else None,
            bounds_valid=bool(valid)))
        if not valid or not np.isfinite(gap) or gap < minimum_m:
            evidence['status'] = 'correction_rejected'
            raise RepairRejected('Proposed interpolation correction failed independent checks', evidence)
        changed[i] = proposed
    after = np.array([inspect(row) for row in changed])
    peak = np.max(abs(np.diff(changed, axis=0))/np.diff(times)[:, None], axis=0)
    planar_peak=float(np.max(np.linalg.norm(np.diff(changed[:,:2],axis=0),axis=1)/np.diff(times)))
    evidence.update(after_gaps_m=after, peak_reference_speed=peak,
                    planar_speed_limit_m_s=planar_speed,planar_speed_peak_m_s=planar_peak,
                    maximum_correction_by_coordinate=np.max(abs(changed-reference), axis=0))
    evidence['status'] = 'admitted' if (np.isfinite(after).all() and np.all(after >= minimum_m)
        and np.all(peak <= speed+1e-7) and (planar_speed is None or planar_peak<=planar_speed+1e-7)
        and np.array_equal(changed[protected], reference[protected])) else 'final_check_rejected'
    if evidence['status'] != 'admitted':
        raise RepairRejected('Final complete source fixture/speed/protected-row check failed', evidence)
    return changed, evidence


def prepare(prepared, robot, output, *, minimum_m=.003, max_iterations=160,
            enforce_planar_neighbor_speed=False):
    """Call immediately after feasible_timing, before inserted placement phases."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    evidence = {}; solver = None; failure = None; completed_frames = 0
    try:
        details = prepared[5]
        if details.get('robot_supported_placement') or details.get('retimed_source_clearance'):
            raise ValueError('Repair requires an uninserted, unrepaired source time map')
        timing = details.get('robot_control_retiming', {})
        planar_limit=timing.get('planar_speed_norm_limit_m_s')
        if type(enforce_planar_neighbor_speed) is not bool or (enforce_planar_neighbor_speed and planar_limit is None):
            raise ValueError('Planar neighbor enforcement requires an explicit boolean and declared timing norm cap')
        artifact = Path(timing.get('source_clock_artifact', ''))
        if not artifact.is_file() or sha256(artifact) != timing.get('artifact_sha256'):
            raise ValueError('Checksum-bound original retiming artifact required')
        with np.load(artifact, allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        required = {'original_time_s', 'original_reference', 'retimed_reference',
                    'control_time_s', 'source_reference_time_s'}
        if not required <= arrays.keys() or not np.array_equal(arrays['control_time_s'], prepared[7]) or not np.array_equal(arrays['retimed_reference'], prepared[8]):
            raise ValueError('Retiming artifact does not bind the complete current robot arrays')
        model, _, manifest = prepared[:3]
        data = mujoco.MjData(model); initialize_fixtures(model, data, manifest)
        active = manifest['objects'][details['object_id']]['body']
        supports = reachy_wheel_floor_pairs(model)
        checker = RobotFixtureClearance(model, active_object_bodies=[active],
                                        support_pairs=supports, minimum_m=minimum_m)
        def inspect(ref):
            q = robot.pack(ref[3:], ref[:3])
            initialize(model, data, robot, q, manifest['mimics'],
                       {'l_hand_finger': 2., 'r_hand_finger': 2.})
            return checker.inspect(data)['minimum_signed_distance_or_lower_bound_m']
        reference = np.asarray(prepared[8])
        limits = np.array(timing['speed_limits'], float)
        maximum = np.r_[.002, .002, .01, np.full(7, .01)]
        base_bounds = np.column_stack([reference[:, :3].min(0)-maximum[:3],
                                       reference[:, :3].max(0)+maximum[:3]])
        solver = Solver(prepared, robot, base_bounds_xyyaw=base_bounds,
            minimum_fixture_gap_m=minimum_m, max_iterations=max_iterations,
            pose_tolerance_constraints=True, minimum_motion=True)
        initialize_fixtures(model, solver.data, manifest)
        joint_bounds = np.asarray(solver.bounds, float)
        def solve(i, ref, lower, upper, *, planar_constraints=None):
            bounded_lower = np.maximum(lower, joint_bounds[:, 0])
            bounded_upper = np.minimum(upper, joint_bounds[:, 1])
            if np.any(bounded_lower > bounded_upper):
                raise ValueError('Original neighbor bounds conflict with robot joint limits')
            solver.bounds = list(zip(bounded_lower, bounded_upper))
            solver.base_yaw = float(ref[2])
            q = solver.solve(prepared[10][i], robot.pack(ref[3:], ref[:3]), [2.],
                             phase='retimed_source_only', weight=float(prepared[7][i]),
                             planar_constraints=planar_constraints)
            value = ref.copy(); value[:3] = solver.reference_base(q)
            value[10:] = q[robot.arm_ids[7:]]
            return value
        changed, evidence = repair_reference(reference, prepared[7], arrays['source_reference_time_s'],
            arrays['original_time_s'], arrays['original_reference'], limits, inspect, solve,
            minimum_m=minimum_m, maximum_correction=maximum,
            planar_speed=planar_limit if enforce_planar_neighbor_speed else None)
        errors = np.empty((len(changed), 2)); depths = errors.copy()
        gaps = np.empty(len(changed)); margins = gaps.copy()
        bodies = set(checker.metadata['robot_body_ids']); active_id = model.body(active).id
        for i, ref in enumerate(changed):
            q = robot.pack(ref[3:], ref[:3]); hand = robot.fk(q)[1]
            errors[i] = [np.linalg.norm(hand[:3, 3]-prepared[10][i, :3, 3]),
                Rotation.from_matrix(prepared[10][i, :3, :3]@hand[:3, :3].T).magnitude()]
            initialize(model, data, robot, q, manifest['mimics'],
                       {'l_hand_finger': 2., 'r_hand_finger': 2.})
            collision, _ = collision_depths(model, data, bodies, active_id)
            depths[i] = [collision['environment'], collision['self']]
            gaps[i], margins[i], _ = robot.r.geometry(q)
            completed_frames += 1
        planar = np.linalg.norm(np.diff(changed[:, :2], axis=0), axis=1)/np.diff(prepared[7])
        planar_limit = timing.get('planar_speed_norm_limit_m_s')
        checks = dict(complete=True, fixture=bool(np.all(evidence['after_gaps_m'] >= minimum_m)),
            hand_ik=bool(np.all(errors <= [.002, .02])), environment=bool(np.max(depths[:, 0]) <= .002),
            self_collision=bool(np.max(depths[:, 1]) <= .002), self_clearance=bool(np.min(gaps) >= .009),
            joint_margin=bool(np.min(margins) >= .025), reference_speed=bool(np.all(evidence['peak_reference_speed'] <= limits+1e-7)),
            planar_speed=bool(planar_limit is None or np.max(planar) <= planar_limit+1e-7),
            inactive_arm_exact=bool(np.array_equal(changed[:, 3:10], reference[:, 3:10])),
            protected_knots_exact=bool(np.array_equal(changed[evidence['protected_rows']], reference[evidence['protected_rows']])))
        evidence.update(errors=errors, collision_depths=depths, self_clearance_m=gaps,
                        joint_margin_rad=margins, admission_checks=checks)
        if not all(checks.values()):
            evidence['status']='full_source_check_rejected'
            raise RepairRejected('Full repaired source IK/collision/speed admission failed', evidence)
    except Exception as exc:
        failure = type(exc).__name__+': '+str(exc)
        if isinstance(exc, RepairRejected): evidence = exc.evidence
    numeric = {k: v for k, v in evidence.items() if isinstance(v, np.ndarray)}
    numeric.update(input_reference=np.asarray(prepared[8]), input_time_s=np.asarray(prepared[7]),
                   unchanged_hand_targets=np.asarray(prepared[10]), unchanged_object_targets=np.asarray(prepared[11]),
                   unchanged_gripper_intent=np.asarray(prepared[9]))
    if 'arrays' in locals(): numeric.update({'time_map_'+k: v for k, v in arrays.items()})
    result_artifact = output/'repair.npz'; np.savez_compressed(result_artifact, **numeric)
    report = {k: v for k, v in evidence.items() if not isinstance(v, np.ndarray)}
    report.update(admitted=failure is None, physics_validated=False, exception=failure,
        completed_frames=completed_frames, total_frames=len(prepared[7]),
        artifact=str(result_artifact), artifact_sha256=sha256(result_artifact),
        inherited_time_map_sha256=timing.get('artifact_sha256') if 'timing' in locals() else None,
        scene_sha256=hashlib.sha256(prepared[1].encode()).hexdigest(),
        aperture_rad=2., maximum_correction_xy_yaw_arm=[.002, .002, .01, *([.01]*7)],
        source_knots_immutable=True, hand_object_clock_intent_unchanged=True,
        enforce_planar_neighbor_speed=enforce_planar_neighbor_speed,
        scope='Only new retimed robot samples; original source and coarse reference retained. Complete all-open source fixture proof supplements unchanged physics gates.')
    if solver is not None: json_write(output/'solver-records.json', solver.records)
    json_write(output/'result.json', report)
    if failure is not None: raise ValueError('Retimed source repair rejected; inspect '+str(output/'result.json'))
    result = list(prepared); result[8] = changed; result[5] = deepcopy(prepared[5])
    result[5]['retimed_source_clearance'] = report
    result[5]['joint_reference_valid'] = True
    result[5]['requires_full_ik_and_collision_admission'] = False
    return tuple(result)
