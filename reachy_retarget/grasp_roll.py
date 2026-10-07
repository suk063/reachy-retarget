"""Rotate robot targets around their calibrated pad line without moving that line.

This geometric proposal never edits the scene, object states, source clock or
contact intent. A new complete IK/clearance admission and actuator rollout are
required. Positive rotation follows the right-hand rule about pad0 -> pad1.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .store import json_write, sha256


def rotate_about_pad_line(hands, midpoint_tcp, closing_axis_tcp, angle_rad):
    """Return corrected poses and a constant right attachment, bounded to 90 deg."""
    poses = np.asarray(hands, dtype=float)
    point, axis = np.asarray(midpoint_tcp, float), np.asarray(closing_axis_tcp, float)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses)
            or point.shape != (3,) or axis.shape != (3,)
            or not all(np.isfinite(x).all() for x in (poses, point, axis))
            or np.linalg.norm(axis) < 1e-10 or not np.isfinite(angle_rad)
            or abs(angle_rad) > np.pi/2 + 1e-12):
        raise ValueError('Finite poses, pad line and a rotation bounded to 90 degrees required')
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3, :], [0., 0., 0., 1.], atol=1e-10, rtol=0)
            or not np.allclose(rotations.swapaxes(1, 2) @ rotations, np.eye(3), atol=1e-7, rtol=0)
            or np.any(np.linalg.det(rotations) < .999999)):
        raise ValueError('Hand poses must be rigid right-handed transforms')
    attachment = np.eye(4)
    attachment[:3, :3] = Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle_rad).as_matrix()
    attachment[:3, 3] = point - attachment[:3, :3] @ point
    return poses @ attachment, attachment


def apply(prepared, output, *, angle_rad):
    """Apply to any existing calibrated grasp; save invalidated admission history."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    details = deepcopy(prepared[5])
    calibration = details['pad_alignment']
    points = np.asarray(calibration['pad']['points_tcp'], float)
    midpoint = np.asarray(calibration['pad']['midpoint_tcp'], float)
    if points.shape != (2, 3) or not np.allclose(points.mean(0), midpoint, atol=1e-9, rtol=0):
        raise ValueError('The verified opposing pad points must define their declared midpoint')
    changed, attachment = rotate_about_pad_line(prepared[10], midpoint, points[1]-points[0], angle_rad)
    original_points = prepared[10][:, None, :3, :3] @ points[None, :, :, None]
    original_points = original_points[..., 0] + prepared[10][:, None, :3, 3]
    new_points = changed[:, None, :3, :3] @ points[None, :, :, None]
    new_points = new_points[..., 0] + changed[:, None, :3, 3]
    displacement = float(np.linalg.norm(new_points-original_points, axis=2).max())
    if displacement > 1e-10:
        raise AssertionError('A pad-line roll changed the calibrated contact trajectory')
    artifact = output/'pad-line-roll.npz'
    np.savez_compressed(artifact, original_time_s=prepared[7], original_reference=prepared[8],
                        original_hand_goals=prepared[10], hand_goals=changed,
                        original_object_goals=prepared[11], original_gripper_intent=prepared[9],
                        attachment=attachment, calibrated_pad_points_world=original_points)
    report = dict(angle_rad=float(angle_rad), axis_convention='TCP pad0 -> pad1, right-hand positive',
                  pad_midpoint_tcp_m=midpoint.tolist(), closing_axis_tcp=(points[1]-points[0]).tolist(),
                  attachment=attachment.tolist(), max_calibrated_pad_displacement_m=displacement,
                  max_tcp_translation_m=float(np.linalg.norm(changed[:, :3, 3]-prepared[10][:, :3, 3], axis=1).max()),
                  inferred_contact_angle_rad=calibration['inferred_contact_angle_rad'],
                  artifact_sha256=sha256(artifact), physics_validated=False,
                  joint_reference_valid=False, requires_full_ik_and_collision_admission=True,
                  scope='One constant robot TCP attachment; both calibrated pad-point world paths, complete source clock, object references, geometry and contact intent retained')
    # A prior accepted pose path cannot admit this newly rotated robot path.
    prior = {key: details.pop(key) for key in ('robot_initialization', 'robot_grasp_approach',
             'robot_supported_placement', 'robot_fixture_clearance') if key in details}
    if prior:
        report['invalidated_prior_admissions'] = prior
    details['grasp_roll'] = report
    previous_attachment = details.get('grasp_attachment')
    details['grasp_attachment'] = dict(previous_attachment or {},
        requires_full_ik_and_collision_admission=True,
        pending_pad_line_roll=True)
    details['alignment_variant'] = 'constant_tcp_attachment'
    details['joint_reference_valid'] = False
    details['requires_full_ik_and_collision_admission'] = True
    json_write(output/'result.json', report)
    result = list(prepared)
    result[5], result[10] = details, changed
    return tuple(result)
