"""Introduce robot grasp corrections smoothly while the source gripper is open.

Preserve the original initial hand pose/base, then reach the declared corrected
grasp before closure. The entire corrected closed suffix, source clock, objects
and gripper intent remain unchanged. This proposes targets; it never admits IK.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .store import json_write, sha256


def weights(seconds, commands, *, complete_fraction=.8):
    """Smoothly introduce a correction by an explicit fraction of open time."""
    seconds, commands = np.asarray(seconds, float), np.asarray(commands, float)
    if (seconds.ndim != 1 or commands.shape != seconds.shape or len(seconds) < 3
            or not np.isfinite(seconds).all() or not np.isfinite(commands).all()
            or np.any(np.diff(seconds) <= 0) or not .1 <= complete_fraction <= 1.):
        raise ValueError('An increasing complete clock and bounded open-prefix fraction are required')
    closed = np.flatnonzero(commands < 0)
    if not len(closed) or closed[0] < 2 or np.any(commands[:closed[0]] < 1.999):
        raise ValueError('An explicitly open prefix before the first negative contact intent is required')
    first = int(closed[0])
    complete = seconds[0]+complete_fraction*(seconds[first]-seconds[0])
    phase = np.clip((seconds-seconds[0])/(complete-seconds[0]), 0., 1.)
    # At phase=1-2e-16 the polynomial can round to 1+2e-16. Its
    # mathematical range is [0,1]; preserve that range for SE(3) interpolation.
    return np.clip(phase**3*(10.-15.*phase+6.*phase**2), 0., 1.)


def blend(original, corrected, weight):
    """Interpolate each pregrasp target correction; preserve endpoints exactly."""
    original, corrected, weight = (np.asarray(value, dtype=float) for value in (original, corrected, weight))
    if (original.shape != corrected.shape or original.ndim != 3
            or original.shape[1:] != (4, 4) or weight.shape != original.shape[:1]
            or not all(np.isfinite(x).all() for x in (original, corrected, weight))
            or np.any((weight < 0) | (weight > 1))):
        raise ValueError('Aligned finite rigid-pose arrays and weights in [0,1] required')
    for poses in (original, corrected):
        rotations = poses[:, :3, :3]
        if (not np.allclose(poses[:, 3], [0., 0., 0., 1.], atol=1e-10, rtol=0)
                or not np.allclose(rotations.swapaxes(1, 2)@rotations, np.eye(3), atol=1e-7, rtol=0)
                or np.any(np.linalg.det(rotations) < .999999)):
            raise ValueError('Rigid right-handed target frames are required')
    result = np.array(corrected, copy=True)
    relative = original[:, :3, :3].swapaxes(1, 2)@corrected[:, :3, :3]
    scaled = Rotation.from_rotvec(Rotation.from_matrix(relative).as_rotvec()*weight[:, None])
    result[:, :3, :3] = original[:, :3, :3]@scaled.as_matrix()
    result[:, :3, 3] = original[:, :3, 3]*(1.-weight[:, None])+corrected[:, :3, 3]*weight[:, None]
    result[weight == 0.] = original[weight == 0.]
    result[weight == 1.] = corrected[weight == 1.]
    return result


def apply(prepared, original_hand_targets, output, *, complete_fraction=.8):
    """Return a pending full-admission target proposal with original lineage."""
    if prepared[5].get('robot_control_retiming'):
        raise ValueError('Introduce geometric corrections before retiming to preserve source alignment')
    amount = weights(prepared[7], prepared[9], complete_fraction=complete_fraction)
    changed = blend(original_hand_targets, prepared[10], amount)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifact = output/'open-prefix-correction.npz'
    np.savez_compressed(artifact, original_time_s=prepared[7], original_reference=prepared[8],
                        original_hand_targets=original_hand_targets,
                        full_corrected_hand_targets=prepared[10], hand_targets=changed,
                        correction_weight=amount, original_object_targets=prepared[11],
                        original_gripper_intent=prepared[9])
    details = deepcopy(prepared[5])
    previous = {key: details.pop(key) for key in ('robot_initialization', 'robot_grasp_approach',
                'robot_supported_placement', 'robot_fixture_clearance') if key in details}
    report = dict(complete_fraction=float(complete_fraction),
                  first_closed_frame=int(np.flatnonzero(prepared[9] < 0)[0]),
                  full_correction_frame=int(np.flatnonzero(amount == 1.)[0]),
                  initial_hand_pose_preserved=True, corrected_closed_suffix_preserved=True,
                  prior_admissions_invalidated=previous, artifact_sha256=sha256(artifact),
                  physics_validated=False, joint_reference_valid=False,
                  requires_full_ik_and_collision_admission=True,
                  scope='Blend only newly proposed robot hand correction during explicit open source intent; original first pose and complete corrected closed suffix retained. No source/object/clock/intent edits.')
    details['open_prefix_correction'] = report
    details['joint_reference_valid'] = False
    details['requires_full_ik_and_collision_admission'] = True
    result = list(prepared)
    result[5], result[10] = details, changed
    json_write(output/'result.json', report)
    return tuple(result)


def base_track(prepared, offset_xyyaw, *, complete_fraction=.8):
    """Apply the same declared open-prefix envelope to an explicit base offset."""
    offset = np.asarray(offset_xyyaw, float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('A finite explicit base XY/yaw offset is required')
    amount = weights(prepared[7], prepared[9], complete_fraction=complete_fraction)
    return prepared[8][:, :3]+amount[:, None]*offset
