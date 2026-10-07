"""Reachy 2 parallel gripper: joint angle <-> normalized opening <-> pad separation width.

Geometry (URDF, identical for both hands; frame = `{l,r}_hand_palm_link`, +z toward the fingertips):
the commanded joint `{l,r}_hand_finger` f drives, through mimic tags, the two proximal links
(pivots at palm (-0.0102, +/-0.0379, 0.0700), length 0.0500) by p = 0.554 - 0.4689 f and the
distal links by -p, so the distal links stay parallel to the palm (a parallel-jaw linkage).
The primary finger (`*_hand_distal_link`) sits at palm +y, the mimic finger
(`*_hand_distal_mimic_link`, rotated by pi about z) at palm -y. Increasing f opens the hand:
f = 2.27 (URDF upper limit) is fully open, f = -0.0803 (lower limit) presses the fingers together.

Pads: in the distal frame the flat inner pad face is the plane Y = -PAD_INSET facing the other
finger and spans Z in PAD_SPAN (from the `pincette_distal` visual/collider meshes: their node
transform maps mesh y = 0 to Y = -0.0125 and the pad plate covers Z 7.4-46.0 mm; reachy-agent's
nominal fingertip endpoint is Z = 46.1 mm). The width is the separation of the two pad faces along
palm y; the rigid model's faces meet at f = CONTACT_ANGLE (about 0.044 rad), below which the
real gripper only squeezes (width 0).

Opening is normalized linearly in joint angle over the URDF limits (0 = lower limit = closed,
1 = upper limit = open), which matches the actuator's position-control range; use the width
functions where a metric aperture is needed.
"""
import functools

import numpy as np

from .resources import urdf
from .urdf import KinematicTree

PAD_INSET = 0.0125           # m, distal-frame -Y offset of the inner pad face
PAD_SPAN = (0.0074, 0.0461)  # m, distal-frame Z extent of the pad face
PAD_DEPTH = 0.5 * (PAD_SPAN[0] + PAD_SPAN[1])  # pad centre along the distal axis
LOWER, UPPER = urdf().limits["r_hand_finger"]
# Angle at which the proximal links are parallel to the approach axis (p = 0); the grasp center
# is defined there. Pad depth changes by at most 8.6 mm over the whole stroke (1 - cos p).
_src, _mult, _off = urdf().source("r_hand_finger_proximal")
STRAIGHT_ANGLE = -_off / _mult


@functools.cache
def _tree(side):
    s = side[0]
    return KinematicTree(urdf(), f"{s}_hand_palm_link", [f"{s}_hand_distal_link", f"{s}_hand_distal_mimic_link"],
                         [f"{s}_hand_finger"])


def pad_centers(angle, side="right"):
    """Pad-face centres of the primary and mimic fingers in the palm frame: (..., 2, 3)."""
    angle = np.asarray(angle, float)
    poses = _tree(side).fk(angle[..., None])
    local = np.array([0.0, -PAD_INSET, PAD_DEPTH, 1.0])
    return np.stack([poses[k] @ local for k in _tree(side).targets], axis=-2)[..., :3]


def _signed_width(angle):
    p = pad_centers(angle)
    return p[..., 0, 1] - p[..., 1, 1]


_TABLE_ANGLE = np.linspace(LOWER, UPPER, 2001)
_TABLE_WIDTH = _signed_width(_TABLE_ANGLE)
assert np.all(np.diff(_TABLE_WIDTH) > 0), "pad width must increase with the finger angle"
CONTACT_ANGLE = float(np.interp(0.0, _TABLE_WIDTH, _TABLE_ANGLE))
MAX_WIDTH = float(_TABLE_WIDTH[-1])


def angle_to_opening(angle):
    """Normalized opening in [0, 1] (1 = open) of `{l,r}_hand_finger` angles (rad)."""
    return np.clip((np.asarray(angle, float) - LOWER) / (UPPER - LOWER), 0.0, 1.0)


def opening_to_angle(opening):
    """Finger joint angle (rad) for a normalized opening (clipped to [0, 1])."""
    return LOWER + np.clip(np.asarray(opening, float), 0.0, 1.0) * (UPPER - LOWER)


def angle_to_width(angle):
    """Pad separation (m, >= 0) at finger angles (rad); 0 at and below CONTACT_ANGLE."""
    return np.maximum(np.interp(angle, _TABLE_ANGLE, _TABLE_WIDTH), 0.0)


def width_to_angle(width):
    """Finger angle (rad) giving a pad separation (m, clipped to [0, MAX_WIDTH]); 0 -> CONTACT_ANGLE."""
    return np.interp(np.clip(width, 0.0, MAX_WIDTH), _TABLE_WIDTH, _TABLE_ANGLE)


# Distal finger collider box in the distal frame (pincette_distal_collider hull, see robot.mjcf).
DISTAL_BOX = ((-0.015, 0.015), (-PAD_INSET, 0.015), (0.0053, 0.0461))


def finger_points(angle, side="right", n=(3, 2, 4)):
    """Sample points (..., 2 * prod(n), 3) on the two distal finger boxes (``DISTAL_BOX``, a grid
    of ``n`` points per axis) in the grasp-center frame (palm axes, origin at the pad-centre
    midpoint at STRAIGHT_ANGLE, see ``Reachy.grasp_center``)."""
    angle = np.asarray(angle, float)
    grid = np.stack(np.meshgrid(*[np.linspace(a, b, k) for (a, b), k in zip(DISTAL_BOX, n)], indexing="ij"),
                    axis=-1).reshape(-1, 3)
    local = np.c_[grid, np.ones(len(grid))]
    poses = _tree(side).fk(angle[..., None])
    pts = np.concatenate([np.einsum("...ij,nj->...ni", poses[k], local)[..., :3] for k in _tree(side).targets], axis=-2)
    return pts - _gc_offset(side)


@functools.cache
def _gc_offset(side):
    return pad_centers(STRAIGHT_ANGLE, side).mean(axis=0)
