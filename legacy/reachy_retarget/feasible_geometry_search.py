"""Finite, deterministic robot geometry search with no physical outcome tuning.

Sparse screening is a bounded search heuristic, never full-path admission or
proof of impossibility. Every screened rejection and every full-path attempt is
retained. A selected plan still requires independent actuator-only validation.
"""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np

SEED9 = [-.07174286951069012, -.902745775685186, .18542113837357807,
         -1.361150358195674, -.20982220218473058, .03744438167939799,
         -1.1980294239033369]
POLICY = 'can-geometry-bank-v1'


def describe_bank():
    """Return a source-independent bank; IDs remain stable across chunks."""
    bank = []
    for index, (x, y, degrees, seed) in enumerate(product(
            (0., .1, .2), (-.15, 0., .15), (-15., 0., 15.), ('source', 'seed9'))):
        yaw = float(np.deg2rad(degrees))
        cost = float(np.hypot(np.hypot(x, y), .3*yaw))
        bank.append(dict(id=index, base_offset_xyyaw=[x, y, yaw], seed=seed,
                         rank=[cost, int(seed == 'seed9'), index]))
    return dict(policy=POLICY, candidates=bank, candidate_count=len(bank),
                rank_semantics='Lexicographic base displacement with 0.3 m yaw radius, seed order, stable base ID, placement correction, stable placement ID',
                sparse_screening='Approximate IK/collision rejection; false negatives possible, never a claim of source infeasibility',
                physics_validated=False)


def chunk_candidates(index, count):
    if not isinstance(index, int) or not isinstance(count, int) or not 0 <= index < count <= 54:
        raise ValueError('Require integer chunk index in [0, count), count <= 54')
    return sorted((item for item in describe_bank()['candidates'] if item['id'] % count == index),
                  key=lambda item: item['rank'])


def screen_indices(prepared):
    """Include boundaries, intent transitions and position/reference extrema."""
    n = len(prepared[7])
    if n < 3:
        raise ValueError('Geometry search requires at least three original rows')
    indices = set(np.linspace(0, n-1, min(25, n), dtype=int).tolist())
    transitions = np.flatnonzero(np.diff(np.asarray(prepared[9]) < 0)) + 1
    for i in transitions:
        indices.update((int(i-1), int(i)))
    values = np.column_stack([prepared[10][:, :3, 3], prepared[8][:, :3]])
    indices.update(np.argmin(values, axis=0).tolist())
    indices.update(np.argmax(values, axis=0).tolist())
    return np.array(sorted(indices), dtype=int)


def placement_bank(prepared):
    """Forty-five posture candidates inside the verified original task region."""
    from scipy.spatial.transform import Rotation
    contract = prepared[5]['task_contract']
    lower, upper = np.asarray(contract['bin_lower']), np.asarray(contract['bin_upper'])
    frame = np.asarray(prepared[5]['world_placement'])
    center = (lower+upper)/2
    local_offsets = [(0., 0.), (.025, -.025), (.025, .025), (-.025, -.025), (-.025, .025)]
    closed = np.flatnonzero(prepared[9] < 0)
    releases = np.flatnonzero((np.arange(len(prepared[9])) > closed[0]) & (prepared[9] >= 1.9999))
    if not len(releases):
        raise ValueError('Placement bank requires an original release transition')
    release = int(releases[0]); original_xy = prepared[11][release, :2, 3]
    result = []
    for index, ((dx, dy), pitch, yaw) in enumerate(product(local_offsets, (30., 40., 50.), (-20., 0., 20.))):
        local = center + np.array([dx, dy, 0.])
        if not np.all((local > lower) & (local < upper)):
            continue
        xy = (frame[:3, :3] @ local + frame[:3, 3])[:2]
        rotation = [0., pitch, yaw]
        cost = float(np.linalg.norm(xy-original_xy) + .1*Rotation.from_euler('xyz', rotation, degrees=True).magnitude())
        result.append(dict(id=index, rank=[cost, index], parameters=dict(
            placement_xy_world=xy.tolist(), base_offset_xyyaw=[.15, 0., 0.],
            rotation_xyz_deg=rotation, release_aperture_rad=1.4,
            extra_descent_m=.002, retreat_speed_factor=1.)))
    return sorted(result, key=lambda item: item['rank'])


def select(results, *, expected_chunks=None, expected_rolls=None):
    """Choose by the declared robot correction rank, never rollout outcomes."""
    if any(r.get('search_complete') is False or r.get('status') == 'search_incomplete' for r in results):
        raise ValueError('Incomplete search errors prevent a minimum-rank selection')
    if expected_chunks is not None:
        if expected_rolls is None:
            actual = [r['chunk_index'] for r in results]
            expected = list(range(expected_chunks))
        else:
            actual = [(round(float(r['grasp_roll']['angle_rad']), 12), r['chunk_index']) for r in results]
            expected = [(round(float(angle), 12), index) for angle in expected_rolls for index in range(expected_chunks)]
        if sorted(actual) != sorted(expected):
            raise ValueError('Complete, unique chunk results are required for selection')
    selected = [r['selected'] for r in results if r.get('selected')]
    return min(selected, key=lambda item: item['selection_rank']) if selected else None


def _sample(prepared, indices):
    result = list(prepared)
    for i in range(7, 12):
        result[i] = np.asarray(prepared[i])[indices]
    result[5] = deepcopy(prepared[5])
    result[5]['geometry_search_screening_only'] = True
    return tuple(result)


def _screen_placement(prepared, robot, parameters, output):
    """Inspect nine IK poses before attempting a complete inserted phase."""
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp
    from .feasible_maniskill import collision_depths
    from .physics import initialize
    from .store import json_write
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    commands, hands, objects = prepared[9:12]
    first_close = np.flatnonzero(commands < 0)[0]
    release = int(np.flatnonzero((np.arange(len(commands)) > first_close) & (commands >= 1.9999))[0])
    anchor = hands[release].copy()
    position = objects[-1, :3, 3].copy(); position[:2] = parameters['placement_xy_world']
    rotation = Rotation.from_euler('xyz', parameters['rotation_xyz_deg'], degrees=True).as_matrix()
    bottom = anchor.copy(); bottom[:3, :3] = rotation @ anchor[:3, :3]
    bottom[:3, 3] = position + rotation @ (anchor[:3, 3]-objects[release, :3, 3])
    bottom[2, 3] -= parameters['extra_descent_m']
    traverse = bottom.copy(); traverse[2, 3] = max(anchor[2, 3], bottom[2, 3])
    model, _, manifest = prepared[:3]; data = mujoco.MjData(model)
    robot_bodies = {model.body('base_link').id}
    for child in range(1, model.nbody):
        if int(model.body_parentid[child]) in robot_bodies:
            robot_bodies.add(child)
    body = model.body(manifest['objects'][prepared[5]['object_id']]['body']).id
    initial = prepared[8][release]; q = robot.pack(initial[3:], initial[:3])
    old_active = robot.active.copy(); robot.active = robot.arm_v[7:]
    records, configs, failure = [], [], None
    try:
        for name, first, last in [('traverse', anchor, traverse), ('lower', traverse, bottom)]:
            for weight in np.linspace(0., 1., 5):
                if name == 'lower' and weight == 0:
                    continue
                target = first.copy(); target[:3, 3] = (1-weight)*first[:3, 3]+weight*last[:3, 3]
                target[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(np.stack([first[:3, :3], last[:3, :3]])))(weight).as_matrix()
                base = initial[:3] + np.asarray(parameters['base_offset_xyyaw'])*(weight if name == 'traverse' else 1.)
                q = robot.pack(q[robot.arm_ids], base); goals = robot.fk(q); goals[1] = target
                q, pe, re = robot.ik(goals, q, active_hands=(1,), iterations=150, joint_margin=.031)
                depth = {'environment': 0., 'self': 0.}; worst = {}
                close = prepared[5]['effective_candidate']['gripper_closed_target']
                for aperture in np.linspace(close, parameters['release_aperture_rad'], 5):
                    initialize(model, data, robot, q, manifest['mimics'],
                               {'l_hand_finger': 2., 'r_hand_finger': float(aperture)})
                    current, pairs = collision_depths(model, data, robot_bodies, body)
                    for key in depth:
                        if current[key] > depth[key]:
                            depth[key] = current[key]; worst[key] = pairs[key]
                gap, margin, _ = robot.r.geometry(q)
                passed = pe <= .002 and re <= .02 and max(depth.values()) <= .002 and gap >= .009 and margin >= .025
                records.append(dict(phase=name, weight=float(weight), position_error_m=float(pe), rotation_error_rad=float(re),
                                    depths_m=depth, worst_collision_pairs=worst, clearance_m=float(gap),
                                    joint_margin_rad=float(margin), passed=bool(passed)))
                configs.append(q.copy())
                if not passed:
                    break
            if records and not records[-1]['passed']:
                break
    except Exception as exc:
        failure = type(exc).__name__+': '+str(exc)
    finally:
        robot.active = old_active
    np.savez_compressed(output/'sampled-q.npz', q=np.asarray(configs))
    report = dict(screening_only=True, physics_validated=False, exception=failure,
                  passed=bool(failure is None and len(records) == 9 and all(r['passed'] for r in records)),
                  parameters=parameters, samples=records,
                  limitation='Sparse sequential IK may reject a different feasible branch; no full-path or physical claim')
    json_write(output/'result.json', report)
    return report


def run(root, job):
    """Run one staged bank chunk. Never perform dataset transfers or physics."""
    from .cluster_pipeline import under
    from .feasible_experiments import variant
    from . import feasible_base, feasible_timing, frozen_plan, grasp_approach, supported_placement
    from .robot import Robot
    from .store import json_write, sha256
    root = Path(root); workspace = under(root, job['workspace'])
    config = job['geometry_search']; index = config['chunk_index']; count = config['chunk_count']
    candidates = chunk_candidates(index, count)
    output = workspace/'geometry-search'; output.mkdir(parents=True, exist_ok=False)
    (output/Path(__file__).name).write_text(Path(__file__).read_text())
    robot = Robot(workspace); source = under(workspace, job['source'])
    candidate, original = frozen_plan.load(workspace/'frozen-plan', source, robot, output)
    if original[5].get('task') != 'PickPlaceCan':
        raise ValueError('This declared geometry bank is scoped to Can pick/place')
    screening_indices = screen_indices(original)
    np.savez_compressed(output/'source-reference.npz', time_s=original[7], reference=original[8],
                        commands=original[9], hand_goals=original[10], object_goals=original[11],
                        screening_indices=screening_indices)
    roll_parameters = job.get('grasp_roll')
    if 'cylindrical_grasp' in job:
        if roll_parameters is not None:
            raise ValueError('Compare cylindrical finite-pad alignment separately from a roll bank')
        from . import cylindrical_grasp
        original = cylindrical_grasp.apply(original, robot, output/'cylindrical-grasp', **job['cylindrical_grasp'])
        # The new aperture follows the exact unchanged mesh width. Admission
        # must cover the actual selected command, not the old mechanical stop.
        candidate, original = variant(candidate, original, 'contact_002')
    if roll_parameters is not None:
        from . import grasp_roll
        angle = float(roll_parameters['angle_rad'])
        if not any(np.isclose(angle, np.deg2rad(value), rtol=0, atol=1e-12)
                   for value in (-45., -30., -15., 15., 30., 45.)):
            raise ValueError('The declared roll search bank is +/-15,30,45 degrees')
        original = grasp_roll.apply(original, output/'grasp-roll', **roll_parameters)
    report = dict(policy=POLICY, chunk_index=index, chunk_count=count,
                  source_episode=Path(job['source']).stem, evaluation_split=job['evaluation_split'],
                  status='no_admitted_candidate', search_complete=True, physics_tested=False, physics_validated=False,
                  source_sha256=sha256(source), candidates=[], selected=None,
                  bank=describe_bank(), source_reference_artifact=str(output/'source-reference.npz'))
    if roll_parameters is not None:
        report.update(policy=POLICY+'-pad-line-roll-v1', grasp_roll=roll_parameters,
                      rank_semantics='Absolute pad-line roll, then the original declared base/placement rank, then signed roll as stable tie breaker')
    if 'cylindrical_grasp' in job:
        report.update(policy=POLICY+'-finite-cylindrical-pad-v1', cylindrical_grasp=original[5]['cylindrical_grasp'])
    if 'geometry_clearance' in job:
        report['geometry_clearance'] = job['geometry_clearance']
    def save():
        errors = [entry for entry in report['candidates']
                  if entry['status'] == 'error' or any(s.get('status') == 'error' for s in entry.get('stages', []))]
        report['error_count'] = len(errors)
        report['search_complete'] = not errors
        if errors:
            report['status'] = 'search_incomplete'
        json_write(output/'result.json', report)
    save()
    for number, item in enumerate(candidates):
        entry = dict(candidate=item, status='running', stages=[]); report['candidates'].append(entry)
        folder = output/f"base-{item['id']:02d}"; folder.mkdir()
        seed = original[8][0, 10:].copy() if item['seed'] == 'source' else np.asarray(SEED9)
        track = original[8][:, :3] + np.asarray(item['base_offset_xyyaw'])
        stage = 'base-screen'
        try:
            feasible_base.prepare(_sample(original, screening_indices), robot, folder/stage,
                                  track[screening_indices], right_arm_seed=seed)
            entry['stages'].append(dict(stage=stage, status='screen_passed', path=str(folder/stage)))
            stage = 'base-full'
            base = feasible_base.prepare(original, robot, folder/stage, track, right_arm_seed=seed)
            entry['stages'].append(dict(stage=stage, status='full_admission_passed', path=str(folder/stage)))
            # Cache this exact full-source admission before resampling/insertion.
            cached_base = frozen_plan.save_admitted(base, candidate, source, robot, folder/'admitted-base')
            stage = 'grasp-approach'
            approached = grasp_approach.prepare(base, robot, folder/stage, lift_m=.12, traverse_fraction=.5)
            stage = 'retiming'
            timed = feasible_timing.prepare(approached, folder/stage, arm_speed_rad_s=.8)
            control, timed = variant(candidate, timed, 'contact_002')
            entry['stages'].append(dict(stage=stage, status='passed', path=str(folder/stage)))
            placements = placement_bank(timed) if config.get('placement', True) else [None]
            for placement in placements:
                if placement is None:
                    ready = timed; placement_rank = [0., -1]; params = None
                else:
                    placement_rank = placement['rank']; params = placement['parameters']
                    location = folder/f"placement-{placement['id']:02d}"
                    screened = _screen_placement(timed, robot, params, location/'screen')
                    attempt = dict(stage='placement', id=placement['id'], rank=placement_rank,
                                   path=str(location), status='screen_rejected')
                    if screened.get('exception'):
                        attempt.update(status='error', reason=screened['exception'])
                    entry['stages'].append(attempt); save()
                    if not screened['passed']:
                        continue
                    try:
                        ready = supported_placement.prepare(timed, robot, location/'full', **params)
                    except ValueError as exc:
                        if not (location/'full/result.json').exists():
                            raise
                        details = json.loads((location/'full/result.json').read_text())
                        attempt.update(status='error' if details.get('exception') else 'full_admission_rejected', reason=str(exc))
                        save(); continue
                    attempt['status'] = 'full_admission_passed'
                if 'geometry_clearance' in job:
                    from . import geometry_clearance
                    clearance_output = (location if placement is not None else folder)/'fixture-clearance'
                    try:
                        ready = geometry_clearance.prepare(ready, robot, clearance_output,
                            support_pairs=geometry_clearance.reachy_wheel_floor_pairs(ready[0]),
                            **job['geometry_clearance'])
                    except ValueError as exc:
                        path = clearance_output/'result.json'
                        if not path.exists():
                            raise
                        details = json.loads(path.read_text())
                        if placement is not None:
                            attempt.update(status='error' if details.get('exception') else 'clearance_rejected',
                                           reason=str(exc), clearance_report=str(path))
                        else:
                            entry['stages'].append(dict(stage='fixture-clearance', status='error' if details.get('exception') else 'clearance_rejected', reason=str(exc)))
                        save(); continue
                # The cached base is independent of closure. Downstream replays
                # the selected approach/timing/placement after choosing closure.
                rank = item['rank'] + placement_rank
                if roll_parameters is not None:
                    rank = [abs(float(roll_parameters['angle_rad']))] + rank + [float(roll_parameters['angle_rad'])]
                final_cache = None
                if placement is not None:
                    final_control = replace(control, contact_release=False, calibrate_attachment=False,
                                            arm_velocity_feedforward=0.)
                    final_cache = frozen_plan.save_admitted(ready, final_control, source, robot, folder/'admitted-complete')
                selected = dict(selection_rank=rank, base_candidate=item,
                                placement_parameters=params, admitted_base_plan=str(cached_base),
                                admitted_plan=None if final_cache is None else str(final_cache),
                                full_geometric_admission_artifact=str(folder),
                                source_episode=report['source_episode'],
                                downstream_grasp_approach=dict(lift_m=.12, traverse_fraction=.5),
                                downstream_retiming=dict(arm_speed_rad_s=.8),
                                grasp_roll=roll_parameters,
                                cylindrical_grasp=job.get('cylindrical_grasp'),
                                geometry_clearance=job.get('geometry_clearance'),
                                closure_admission='contact_002 aperture envelope only; each closure requires final full admission',
                                physics_validated=False)
                report.update(status='kinematic_candidate', selected=selected)
                entry['status'] = 'admitted'; save()
                for other in candidates[number+1:]:
                    report['candidates'].append(dict(candidate=other, status='not_attempted_higher_declared_rank'))
                save(); return report
            entry['status'] = 'placement_bank_exhausted'
        except ValueError as exc:
            entry.update(status='rejected', failed_stage=stage, reason=str(exc))
            path = folder/stage/'result.json'
            if path.exists():
                entry['failure_report'] = str(path)
                if json.loads(path.read_text()).get('exception'):
                    entry['status'] = 'error'
            else:
                entry['status'] = 'error'
        except Exception as exc:
            entry.update(status='error', failed_stage=stage, reason=type(exc).__name__+': '+str(exc))
        save()
    return report
