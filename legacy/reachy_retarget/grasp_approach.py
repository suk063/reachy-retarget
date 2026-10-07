"""Robot-only approach above an unchanged grasp pose in an unchanged scene.

The complete original time interval and source objects are retained. Admission
is kinematic; an actuator-only rollout is still required for physical success.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .feasible_maniskill import collision_depths, intent_parameters, intent_apertures
from .physics import initialize
from .store import json_write, sha256


def targets(times, hands, commands, *, lift_m=.12, traverse_fraction=.5):
    """Traverse above the closing pose, then descend on its vertical line."""
    times, hands, commands = map(np.asarray, (times, hands, commands))
    if (times.ndim != 1 or len(times) < 3 or hands.shape != (len(times), 4, 4)
            or commands.shape != times.shape or not np.isfinite(times).all()
            or not np.isfinite(hands).all() or not np.isfinite(commands).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Finite, aligned poses and an increasing complete clock are required")
    if not np.isfinite(lift_m) or lift_m <= 0 or not 0 < traverse_fraction < 1:
        raise ValueError("Positive lift and an interior traverse fraction are required")
    closed = np.flatnonzero(commands < 0)
    if not len(closed) or closed[0] < 2 or np.any(commands[:closed[0]] < 1.999):
        raise ValueError("An open pregrasp prefix followed by explicit closed intent is required")
    close = int(closed[0])
    first, last = hands[0], hands[close]
    above = last.copy()
    above[2, 3] = max(float(last[2, 3] + lift_m), float(first[2, 3]))
    pivot = times[0] + traverse_fraction * (times[close] - times[0])
    phase = np.clip((times[:close+1] - times[0]) / (pivot - times[0]), 0, 1)
    # Near an endpoint, roundoff can produce 1+2e-16 and make Slerp reject
    # an otherwise valid 100 Hz source clock (for example close frame255).
    smooth = lambda x: np.clip(x**3 * (10 - 15*x + 6*x*x), 0., 1.)
    weight = smooth(phase)
    corrected = hands.copy()
    corrected[:close+1, :3, 3] = ((1-weight[:, None])*first[:3, 3]
                                  + weight[:, None]*above[:3, 3])
    corrected[:close+1, :3, :3] = Slerp(
        [0., 1.], Rotation.from_matrix(np.stack([first[:3, :3], last[:3, :3]])))(weight).as_matrix()
    descent = smooth(np.clip((times[:close+1]-pivot)/(times[close]-pivot), 0, 1))
    corrected[:close+1, 2, 3] -= descent * (above[2, 3]-last[2, 3])
    # Explicit copies preserve exact endpoints and the whole original suffix.
    corrected[0] = hands[0]
    corrected[close:] = hands[close:]
    return corrected, dict(close_frame=close, close_time_s=float(times[close]),
                           traverse_end_time_s=float(pivot), above_pose=above.tolist(),
                           requested_lift_m=float(lift_m), traverse_fraction=float(traverse_fraction))


def propose(prepared, output, *, lift_m=.12, traverse_fraction=.5,
            gripper_admission='open_envelope'):
    """Stage TCP targets before base/IK admission; certify no robot trajectory.

    This lets an invalid original open-hand prefix be replaced before the full
    path solver examines it. Call ``prepare`` after base admission to retain its
    independent all-frame and open-hand/object checks, then retime and simulate.
    """
    if gripper_admission not in ('open_envelope', 'intent_envelope'):
        raise ValueError('Unknown gripper admission policy')
    corrected, policy = targets(prepared[7], prepared[10], prepared[9],
                                lift_m=lift_m, traverse_fraction=traverse_fraction)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/Path(__file__).name).write_text(Path(__file__).read_text())
    artifact = output/'original-and-proposed-targets.npz'
    np.savez_compressed(artifact, time_s=prepared[7], original_hand_targets=prepared[10],
                        proposed_hand_targets=corrected, original_reference=prepared[8],
                        original_object_targets=prepared[11], original_gripper_intent=prepared[9])
    details = deepcopy(prepared[5])
    prior = {key: details.pop(key) for key in ('robot_initialization', 'robot_grasp_approach',
             'robot_supported_placement', 'robot_fixture_clearance') if key in details}
    report = dict(policy, admitted=False, physics_validated=False,
                  joint_reference_valid=False, requires_full_ik_and_collision_admission=True,
                  requires_open_robot_object_admission=True, requires_retiming=True,
                  completed_frames=0, total_frames=len(corrected),
                  requested_followup_gripper_admission=gripper_admission,
                  artifact=str(artifact), artifact_sha256=sha256(artifact),
                  invalidated_prior_admissions=prior,
                  scope='Target proposal only; exact initial/closing targets, entire closed suffix, source clock, object references and gripper intent retained')
    details['robot_grasp_approach_proposal'] = report
    details['joint_reference_valid'] = False
    details['requires_full_ik_and_collision_admission'] = True
    details['frozen_factors'] = [x for x in details.get('frozen_factors', [])
                               if x not in ('arm/base reference', 'initial physical state')]
    details['trajectory_seed'] = 'Proposed vertical pregrasp targets awaiting complete IK and collision admission'
    json_write(output/'result.json', report)
    result = list(prepared)
    result[5], result[10] = details, corrected
    return tuple(result)


def _admitted_reference_binding(prepared, corrected, gripper_admission):
    """Bind reuse to the exact proposed targets and complete mobile artifact."""
    proposal = prepared[5].get('robot_grasp_approach_proposal', {})
    mobile = prepared[5].get('robot_mobile_fixture_ik', {})
    count = len(prepared[7])
    checks = mobile.get('admission_checks', {})
    if (not mobile.get('admitted') or mobile.get('completed_frames') != count
            or mobile.get('total_frames') != count or not checks
            or not all(value is True for value in checks.values())
            or checks.get('positive_fixture_clearance') is not True):
        raise ValueError('Reference reuse requires complete positive-clearance mobile admission')
    if mobile.get('scene_sha256') != hashlib.sha256(prepared[1].encode()).hexdigest():
        raise ValueError('Mobile admission belongs to a different scene')
    if proposal.get('requested_followup_gripper_admission') != gripper_admission:
        raise ValueError('Reference reuse requires the declared follow-up aperture policy')
    bindings = []
    specifications = [
        (proposal, {'time_s': prepared[7], 'proposed_hand_targets': prepared[10],
                    'original_object_targets': prepared[11], 'original_gripper_intent': prepared[9]}),
        (mobile, {'original_time_s': prepared[7], 'reference': prepared[8],
                  'original_hand_goals': prepared[10], 'original_object_goals': prepared[11],
                  'original_gripper_intent': prepared[9]})]
    for report, expected in specifications:
        artifact = Path(report.get('artifact', ''))
        if not artifact.is_file() or sha256(artifact) != report.get('artifact_sha256'):
            raise ValueError('Reference reuse requires checksum-bound proposal and mobile artifacts')
        with np.load(artifact, allow_pickle=False) as saved:
            if any(key not in saved or not np.array_equal(saved[key], value)
                   for key, value in expected.items()):
                raise ValueError('Reference reuse artifacts do not match current complete arrays')
        bindings.append(dict(artifact=str(artifact), sha256=report['artifact_sha256']))
    if not np.array_equal(corrected, prepared[10]):
        raise ValueError('Requested approach differs from the already admitted proposal')
    return dict(proposal=bindings[0], mobile=bindings[1], reference_exact=True,
                source_arrays_exact=True, scene_sha256=mobile['scene_sha256'])


def prepare(prepared, robot, output, *, lift_m=.12, traverse_fraction=.5,
            gripper_admission='open_envelope', preserve_admitted_reference=False):
    """Re-solve only the prefix, preserving the existing closed joint path.

    Invoke after base/branch correction and before any local speed retiming.
    Every frame is inspected; pregrasp robot/object overlap is also rejected.
    The manipulated object stays at reset. Explicitly bound inactive fixture
    forecasts affect isolated planning data only, never physical validation.
    """
    if type(preserve_admitted_reference) is not bool:
        raise ValueError('preserve_admitted_reference must be an explicit boolean')
    if gripper_admission not in ('open_envelope', 'intent_envelope'):
        raise ValueError('Unknown gripper admission policy')
    contact_angle, closed_target = (intent_parameters(prepared[5])
        if gripper_admission == 'intent_envelope' else (None, None))
    corrected, policy = targets(prepared[7], prepared[10], prepared[9],
                                lift_m=lift_m, traverse_fraction=traverse_fraction)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/Path(__file__).name).write_text(Path(__file__).read_text())
    model, _, manifest = prepared[:3]
    reference = np.array(prepared[8], copy=True)
    close = policy["close_frame"]
    proposal = output/'proposal.npz'
    np.savez_compressed(proposal, time_s=prepared[7], original_hand_targets=prepared[10],
                        hand_targets=corrected, original_reference=prepared[8],
                        original_object_targets=prepared[11], original_gripper_intent=prepared[9])
    previous_active = robot.active.copy()
    robot.active = robot.arm_v[7:]
    q = robot.pack(reference[close, 3:], reference[close, :3])
    solutions = np.full((len(reference), len(q)), np.nan)
    errors = np.full((len(reference), 2), np.nan)
    depths = np.full((len(reference), 3), np.nan)
    gaps = np.full(len(reference), np.nan)
    margins = gaps.copy()
    aperture_envelopes = np.full((len(reference), 2), np.nan)
    worst = {'environment': None, 'self': None}
    worst_depth = {'environment': 0., 'self': 0.}
    data = mujoco.MjData(model)
    body = model.body(manifest['objects'][prepared[5]['object_id']]['body']).id
    robot_bodies = {model.body('base_link').id}
    for child in range(1, model.nbody):
        if int(model.body_parentid[child]) in robot_bodies:
            robot_bodies.add(child)
    failure = None
    completed_frames = 0
    fixture_query = None
    reuse_binding = None
    try:
        if preserve_admitted_reference:
            reuse_binding = _admitted_reference_binding(prepared, corrected, gripper_admission)
        from .planning_fixtures import make_query
        fixture_query = make_query(prepared)
        if fixture_query is not None:
            data = fixture_query.data
        for index in (() if preserve_admitted_reference else range(close-1, -1, -1)):
            q = robot.pack(q[robot.arm_ids], reference[index, :3])
            q[robot.arm_ids[:7]] = reference[index, 3:10]
            goal = robot.fk(q)
            goal[1] = corrected[index]
            q, _, _ = robot.ik(goal, q, active_hands=(1,), iterations=150, joint_margin=.031)
            reference[index, 10:] = q[robot.arm_ids[7:]]
            solutions[index] = q
        for index in range(len(reference)):
            q = robot.pack(reference[index, 3:], reference[index, :3])
            solutions[index] = q
            actual = robot.fk(q)[1]
            errors[index] = [np.linalg.norm(actual[:3, 3]-corrected[index, :3, 3]),
                             Rotation.from_matrix(corrected[index, :3, :3] @ actual[:3, :3].T).magnitude()]
            apertures = (intent_apertures(prepared[9], index, contact_angle, closed_target)
                         if gripper_admission == 'intent_envelope' else np.array([2.]))
            aperture_envelopes[index] = [apertures.min(), apertures.max()]
            collision = {'environment': 0., 'self': 0.}
            object_depth = 0.
            for aperture in apertures:
                initialize(model, data, robot, q, manifest['mimics'],
                           {'l_hand_finger': 2., 'r_hand_finger': float(aperture)})
                if fixture_query is not None:
                    fixture_query.synchronize(index)
                current, pairs = collision_depths(model, data, robot_bodies, body)
                for kind in collision:
                    collision[kind] = max(collision[kind], current[kind])
                    if current[kind] > worst_depth[kind]:
                        worst_depth[kind] = current[kind]
                        worst[kind] = dict(frame=index, time_s=float(prepared[7][index]),
                                           aperture_rad=float(aperture), pair=pairs[kind])
                # At the first closing boundary the envelope also includes
                # intentional closed-finger contact. The approach condition is
                # specifically clearance while open, through that boundary.
                if index <= close and aperture >= 1.999:
                    for contact in data.contact:
                        pair = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
                        if body in pair and pair & robot_bodies:
                            object_depth = max(object_depth, -float(contact.dist))
            depths[index] = [collision['environment'], collision['self'], object_depth]
            gaps[index], margins[index], _ = robot.r.geometry(q)
            completed_frames += 1
    except Exception as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
    finally:
        robot.active = previous_active
    artifact = output/'approach-admission.npz'
    np.savez_compressed(artifact, time_s=prepared[7], reference=reference, q=solutions,
                        ik_errors=errors, collision_depths=depths, self_clearance_m=gaps,
                        joint_margin_rad=margins, gripper_aperture_envelopes_rad=aperture_envelopes)
    complete = failure is None and np.isfinite(errors).all() and np.isfinite(depths).all()
    checks = dict(complete=bool(complete),
                  hand_ik=bool(complete and np.all(errors <= [.002, .02])),
                  environment=bool(complete and np.max(depths[:, 0]) <= .002),
                  self_collision=bool(complete and np.max(depths[:, 1]) <= .002),
                  open_robot_object_no_overlap=bool(complete and np.max(depths[:, 2]) <= 1e-7),
                  self_clearance=bool(complete and np.min(gaps) >= .009),
                  joint_margin=bool(complete and np.min(margins) >= .025))
    report = dict(policy, exception=failure, admission_checks=checks, admitted=all(checks.values()),
                  completed_frames=completed_frames, total_frames=len(reference),
                  physics_validated=False, proposal_sha256=sha256(proposal), artifact_sha256=sha256(artifact),
                  preserve_admitted_reference=preserve_admitted_reference,
                  reused_admission_binding=reuse_binding,
                  gripper_admission=gripper_admission, gripper_closed_target_rad=closed_target,
                  inactive_left_gripper_rad=2., worst_collision_pairs=worst,
                  aperture_scope='Five samples across each declared numeric aperture interval and opening/closing transition; physical tracking remains independently gated',
                  scope='Robot pregrasp TCP path and right-arm prefix only; original closure pose and closed suffix retained',
                  object_check_scope='Manipulated object reset unchanged; open-aperture samples through first closing boundary checked for overlap. Closed grasp contact is physically gated.')
    if preserve_admitted_reference:
        report['scope'] = 'Exact admitted mobile reference retained; complete approach, open-object, aperture, IK and collision checks rerun without a second prefix solve'
    if fixture_query is not None:
        report['planning_fixture_forecast'] = fixture_query.save(output)
    if complete:
        report.update(max_ik_errors=errors.max(0).tolist(), max_collision_depths_m=depths.max(0).tolist(),
                      min_self_clearance_m=float(gaps.min()), min_joint_margin_rad=float(margins.min()))
    json_write(output/'result.json', report)
    if not report['admitted']:
        raise ValueError('Grasp approach rejected; inspect ' + str(output/'result.json'))
    result = list(prepared)
    details = deepcopy(prepared[5])
    details['robot_grasp_approach'] = report
    details['frozen_factors'] = [x for x in details.get('frozen_factors', [])
                               if x not in ('arm/base reference', 'initial physical state')]
    details['trajectory_seed'] = 'Original closed hand/object path with an admitted vertical pregrasp approach'
    result[5], result[6], result[8], result[10] = details, solutions[0], reference, corrected
    return tuple(result)
