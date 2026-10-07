"""Source effectors -> Reachy TCP targets, grasp labels and finger commands.

Adapters express every effector at its grasp center (approach +z, closing +y), and the
Reachy grasp center (``Reachy.grasp_center``) uses the same convention, so a TCP target is
``grasp_center_pose @ inv(T_tip_gc)``. Source and Reachy share the world frame, which
preserves the hand–object relative pose by construction.

A parallel-jaw grasp is symmetric under a half turn about the approach axis, so each side
may use the flipped frame (``flip=True``) for the whole episode; the choice is made by
placement scoring and recorded as a derived label.
"""
from __future__ import annotations

import numpy as np

from ..robot import Reachy, angle_to_opening, angle_to_width, gripper
from ..schema.rotations import quat_to_matrix
from .config import RetargetConfig

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])  # half turn about the grasp-center approach axis
IDLE_FINGER = gripper.CONTACT_ANGLE     # inactive hands: fingers just touching, no squeeze


def tcp_targets(src, sides, flips=None):
    """World TCP targets {side: (T, 4, 4)} for ``sides`` = {effector key: side}.

    ``flips`` = {side: bool} applies the half-turn symmetry of the gripper.
    """
    robot, flips = Reachy.load(), flips or {}
    return {side: robot.tcp_from_grasp_center(src.effectors[key].pose @ (FLIP if flips.get(side) else np.eye(4)), side)
            for key, side in sides.items()}


def manipulated_ids(objects):
    """Sorted ids of the manipulated objects; grasp labels index into this list."""
    return sorted(k for k, o in objects.items() if o.role == "manipulated")


def object_distance(points, track, valid_rows=None):
    """Distance (T,) from points (T, 3) to an object: to its box if ``geometry`` is a box
    with half_extents, else to its centre. NaN where the object is invalid."""
    valid = track.valid if valid_rows is None else valid_rows
    out = np.full(len(points), np.nan)
    if not valid.any():
        return out
    pose = track.pose[valid]
    rel = np.einsum("tji,tj->ti", quat_to_matrix(pose[:, 3:]), points[valid] - pose[:, :3])
    half = np.asarray(track.geometry.get("half_extents", ()), float)
    if track.geometry.get("kind") == "box" and half.shape == (3,):
        rel = rel - np.clip(rel, -half, half)
    out[valid] = np.linalg.norm(rel, axis=-1)
    return out


def grasp_labels(grasp_points, closed, objects, cfg: RetargetConfig):
    """Index of the grasped manipulated object per frame (-1 = none), for one hand.

    grasp_points: (T, 3) grasp-center positions; closed: (T,) bool commanded-closed state.
    A frame is grasping when the hand is closed and the nearest manipulated object lies
    within ``cfg.grasp_contact_distance`` (box surface or centre).
    """
    ids = manipulated_ids(objects)
    labels = np.full(len(grasp_points), -1)
    if not ids:
        return labels
    d = np.stack([object_distance(grasp_points, objects[k]) for k in ids])
    d = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(d, axis=0)
    hit = closed & (d[nearest, np.arange(len(nearest))] <= cfg.grasp_contact_distance)
    labels[hit] = nearest[hit]
    return labels


def source_closed(effector, cfg: RetargetConfig):
    """Commanded-closed state (T,) of a source effector."""
    return effector.opening < cfg.closed_opening


def finger_angles(effector, grasping, cfg: RetargetConfig):
    """Reachy finger angle (T,) for one source effector.

    With a source width: the angle reproducing that pad separation (clipped to Reachy's
    stroke), minus ``cfg.squeeze_angle`` while grasping so the fingers press on the object.
    Without width: the normalized opening mapped linearly onto the joint range.
    """
    if effector.width is None:
        return gripper.opening_to_angle(effector.opening)
    angle = gripper.width_to_angle(effector.width)
    return np.where(grasping, np.maximum(angle - cfg.squeeze_angle, gripper.LOWER), angle)


def gripper_state(angles):
    """(opening (T, 2), width (T, 2)) of finger angles (T, 2)."""
    return angle_to_opening(angles), angle_to_width(angles)

