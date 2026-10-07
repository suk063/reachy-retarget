"""Propose an elevated robot return on the complete, unchanged source clock.

Only rows after the recorded release intent can change. This is a target
proposal, not proof of actual release, IK, clearance or physical success.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from .store import json_write, sha256


def _smooth(value):
    x = np.clip(value, 0., 1.)
    return np.clip(x**3 * (10. - 15.*x + 6.*x*x), 0., 1.)


def propose(prepared, output, *, lift_m=.03, raise_fraction=.2,
            lower_fraction=.25, terminal_lift_m=0., hold_alignment_tolerance_s=0.):
    """Add a bounded world-Z excursion during the source-intent-open suffix.

    Call before retiming, placement insertion and complete base/IK admission.
    The declared evaluation hold stays a stationary tail. A nonzero terminal
    correction is explicit; preserving an endpoint does not certify its safety.
    """
    values = np.asarray([lift_m, raise_fraction, lower_fraction, terminal_lift_m,
                         hold_alignment_tolerance_s], float)
    if (not np.isfinite(values).all() or not 0 < lift_m <= .15
            or not 0 <= terminal_lift_m <= lift_m
            or not 0 <= hold_alignment_tolerance_s <= .1
            or not 0 < raise_fraction <= .5 or not 0 < lower_fraction <= .5
            or raise_fraction + lower_fraction > 1.):
        raise ValueError('Require bounded positive lift, valid ramp fractions and terminal lift')
    details = deepcopy(prepared[5])
    if any(key in details for key in ('robot_control_retiming', 'robot_supported_placement',
                                     'robot_unloaded_return_proposal', 'robot_terminal_hold')):
        raise ValueError('Return proposal must precede retiming, placement and terminal insertion')
    times, reference, commands, hands, objects = map(np.asarray, prepared[7:12])
    n = len(times)
    if (times.shape != (n,) or n < 4 or commands.shape != (n,)
            or hands.shape != (n, 4, 4) or objects.shape != (n, 4, 4)
            or len(reference) != n or not np.isfinite(times).all()
            or not np.isfinite(hands).all() or not np.isfinite(commands).all()
            or not np.allclose(np.diff(times), .01, atol=1e-9, rtol=0)):
        raise ValueError('Complete finite 100 Hz source-reference arrays are required')
    closed = np.flatnonzero(commands < 0)
    if (not len(closed) or closed[-1] >= n - 2
            or not np.all(commands[closed[0]:closed[-1]+1] < 0)):
        raise ValueError('Exactly one contiguous recorded grasp and a release suffix are required')
    release = int(closed[-1] + 1)
    if not np.all(commands[release:] >= 1.2):
        raise ValueError('All return rows must have explicit open gripper intent')
    hold_s = float(details.get('evaluation_hold_s', 0.))
    if not np.isfinite(hold_s) or hold_s < 0:
        raise ValueError('Invalid declared evaluation hold')
    return_end = int(np.argmin(abs(times - (times[-1] - hold_s))))
    declared_return_end = return_end
    # Existing pad-alignment release blending can finish a few rows inside the
    # declared evaluation hold. Preserve those rows, and explicitly bind the
    # actual stationary tail rather than claiming it started earlier.
    if hold_alignment_tolerance_s:
        moving = np.flatnonzero(np.max(abs(hands-hands[-1]), axis=(1, 2)) > 1e-10)
        actual_start = int(moving[-1]+1) if len(moving) else 0
        if actual_start >= n or times[actual_start]-times[return_end] > hold_alignment_tolerance_s+1e-9:
            raise ValueError('Observed stationary tail exceeds the explicit hold-alignment tolerance')
        return_end = max(return_end, actual_start)
    if (abs(times[declared_return_end] - (times[-1]-hold_s)) > 1e-7
            or return_end <= release + 1
            or not np.allclose(hands[return_end:], hands[-1], atol=1e-10, rtol=0)):
        raise ValueError('A nonempty return and an exact stationary declared evaluation hold are required')
    u = (times - times[release]) / (times[return_end] - times[release])
    height = lift_m * _smooth(u / raise_fraction)
    height -= (lift_m-terminal_lift_m) * _smooth((u-(1.-lower_fraction))/lower_fraction)
    height[:release+1] = 0.
    height[return_end:] = terminal_lift_m
    changed = hands.copy()
    changed[:, 2, 3] += height
    changed_rows = np.flatnonzero(height != 0.)
    if not np.array_equal(changed[:release+1], hands[:release+1]):
        raise AssertionError('The source grasp or release boundary was changed')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifact = output/'original-and-proposed-return.npz'
    translation = np.zeros((n, 3)); translation[:, 2] = height
    np.savez_compressed(artifact, original_time_s=times, original_reference=reference,
                        original_gripper_intent=commands, original_hand_goals=hands,
                        original_object_goals=objects, hand_goals=changed,
                        translation_world_m=translation)
    report = dict(lift_m=float(lift_m), raise_fraction=float(raise_fraction),
                  lower_fraction=float(lower_fraction), terminal_lift_m=float(terminal_lift_m),
                  original_release_frame=release, original_release_time_s=float(times[release]),
                  original_return_end_frame=return_end,
                  original_return_end_time_s=float(times[return_end]),
                  declared_return_end_frame=declared_return_end,
                  hold_alignment_tolerance_s=float(hold_alignment_tolerance_s),
                  observed_stationary_hold_s=float(times[-1]-times[return_end]),
                  declared_evaluation_hold_s=hold_s, changed_source_frames=changed_rows.tolist(),
                  original_source_row_count=n, complete_source_clock_preserved=True,
                  object_reference_rows_preserved_exactly=True,
                  gripper_intent_preserved_exactly=True,
                  closed_hand_targets_preserved_exactly=True,
                  release_boundary_target_preserved_exactly=True,
                  hand_orientations_preserved_exactly=True,
                  terminal_hand_target_preserved_exactly=bool(terminal_lift_m == 0.),
                  terminal_target_safety='Not asserted; complete subsequent IK and fixture admission required',
                  artifact=str(artifact), artifact_sha256=sha256(artifact),
                  joint_reference_valid=False, physics_validated=False,
                  requires_full_ik_and_collision_admission=True,
                  requires_retiming=True,
                  scope='Robot world-Z target excursion during recorded open intent; actual release must remain independently support/contact gated. No source/object/scene/control-clock change.')
    prior = {key: details.pop(key) for key in ('robot_initialization', 'robot_grasp_approach',
             'robot_fixture_clearance') if key in details}
    if prior:
        report['invalidated_prior_admissions'] = prior
    details['robot_unloaded_return_proposal'] = report
    details['joint_reference_valid'] = False
    details['requires_full_ik_and_collision_admission'] = True
    details['frozen_factors'] = [value for value in details.get('frozen_factors', [])
                               if value != 'arm/base reference']
    details['trajectory_seed'] = 'Original grasp targets with an explicit open-return robot excursion awaiting full admission'
    json_write(output/'result.json', report)
    result = list(prepared)
    result[5], result[10] = details, changed
    return tuple(result)
