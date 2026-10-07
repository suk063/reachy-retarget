"""Append an explicitly derived terminal robot hold to a complete retimed plan.

Every existing target and timestamp is preserved exactly. Added object poses
are reference labels only; no simulator state is set. The source-clock map
saturates at its existing final value rather than inventing source samples.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from .store import json_write, sha256


def apply(prepared, output, *, duration_s=.2):
    """Append 10 ms control intervals after existing retiming; keep all inputs."""
    if not np.isfinite(duration_s) or not 0 < duration_s <= 2.:
        raise ValueError('An explicit terminal hold between zero and two seconds is required')
    ticks = int(round(duration_s/.01))
    if ticks < 1 or not np.isclose(duration_s, ticks*.01, atol=1e-10, rtol=0):
        raise ValueError('Terminal hold must be an integer number of 10 ms control intervals')
    original = np.asarray(prepared[7])
    if len(original) < 2 or not np.allclose(np.diff(original), .01, atol=1e-10, rtol=0):
        raise ValueError('A complete regular 100 Hz robot control clock is required')
    details = deepcopy(prepared[5])
    if details.get('robot_terminal_hold'):
        raise ValueError('Refuse to silently append a second terminal hold')
    candidate = details.get('effective_candidate', {})
    if (details.get('robot_supported_placement')
            or any(candidate.get(key) for key in ('controlled_place', 'contact_release', 'calibrate_attachment'))):
        raise ValueError('Placement or adaptive phase controllers require a separately verified terminal phase extension')
    timing = details.get('robot_control_retiming')
    if not timing:
        raise ValueError('An explicit existing source-clock map is required; retime first')
    parent = Path(timing['source_clock_artifact'])
    if sha256(parent) != timing['artifact_sha256']:
        raise ValueError('Existing retiming artifact checksum mismatch')
    with np.load(parent, allow_pickle=False) as values:
        clocks = {key: values[key].copy() for key in values.files}
    if (not np.array_equal(clocks['control_time_s'], original)
            or clocks['source_reference_time_s'].shape != original.shape):
        raise ValueError('Source-clock map does not cover the complete current reference')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = list(prepared)
    result[7] = np.r_[original, original[-1]+np.arange(1, ticks+1)*.01]
    for index in (8, 9, 10, 11):
        values = np.asarray(prepared[index])
        if len(values) != len(original) or not np.isfinite(values).all():
            raise ValueError('Finite complete reference arrays are required')
        result[index] = np.concatenate([values, np.repeat(values[-1:], ticks, axis=0)])
    clocks['control_time_s'] = result[7]
    clocks['source_reference_time_s'] = np.r_[clocks['source_reference_time_s'],
        np.repeat(clocks['source_reference_time_s'][-1], ticks)]
    clocks['retimed_reference'] = result[8]
    artifact = output/'time-map.npz'
    np.savez_compressed(artifact, **clocks)
    report = dict(duration_s=float(duration_s), added_control_intervals=ticks,
                  original_control_rows=len(original), total_control_rows=len(result[7]),
                  original_end_s=float(original[-1]), derived_end_s=float(result[7][-1]),
                  terminal_gripper_intent=float(prepared[9][-1]),
                  original_interaction_phases=deepcopy(details.get('interaction')),
                  derived_phase=dict(mode='terminal_robot_hold', start_control_row=len(original),
                                     stop_control_row=len(result[7])),
                  inherited_source_clock=timing, artifact_sha256=sha256(artifact),
                  source_clock_semantics='Original endpoint held; no additional native source observations',
                  physics_validated=False,
                  scope='Append only final robot joint/hand targets and gripper intent; object poses are unchanged reference labels, never prescribed simulator state')
    timing = deepcopy(timing)
    timing.update(source_clock_artifact=str(artifact), artifact_sha256=sha256(artifact),
                  duration_s=float(result[7][-1]-result[7][0]),
                  duration_ratio=float((result[7][-1]-result[7][0])/timing['original_duration_s']),
                  terminal_hold_s=float(duration_s))
    details.update(robot_terminal_hold=report, robot_control_retiming=timing,
                   duration_s=float(result[7][-1]))
    result[5] = details
    json_write(output/'result.json', report)
    return tuple(result)
