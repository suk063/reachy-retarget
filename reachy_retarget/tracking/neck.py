"""Full 3-DoF head-rotation tracking for the neck (``neck_roll``, ``neck_pitch``, ``neck_yaw``).

The manipulation retargeting only points the head (:mod:`reachy_retarget.retarget.gaze`: pitch and
yaw, roll 0). Tracking follows a head *rotation*, as reachy-control's controller does
(``neck_R = base_from_world.R @ R_head``, frame ``head_tip``). ``head_tip`` is ``head`` translated
by (0.0564, 0, 0.062) m without rotation, so the rotation of either frame is the same; the look-at
origin is the ``head_tip`` origin and the viewing axis its +X (reachy-control ``util/geometry.look_at``).

The torso is rigid on the base (the tripod stays at 0), so the head rotation in ``base_link``
depends on the three neck joints only. :func:`solve` runs a batched, bound-projected Gauss-Newton
on them for every row at once, then limits the rate to the neck speed.
"""
from __future__ import annotations

import functools

import numpy as np

from ..robot import LOWER, UPPER, VELOCITY
from ..robot.reachy import FIXED, NECK, planar
from ..robot.resources import urdf
from ..robot.urdf import KinematicTree
from ..schema.rotations import so3_log

NECK_JOINTS = ("neck_roll", "neck_pitch", "neck_yaw")
HEAD_TIP = urdf().transform("head", "head_tip")  # constant: head_tip in the head frame


@functools.cache
def _tree():
    return KinematicTree(urdf(), "base_link", ["head"], NECK_JOINTS, FIXED)


def head_base(neck) -> np.ndarray:
    """Head frame pose(s) (..., 4, 4) in ``base_link`` for neck angles (..., 3)."""
    return _tree().fk(np.asarray(neck, float))["head"]


def head_tip_base(neck) -> np.ndarray:
    return head_base(neck) @ HEAD_TIP


def _jacobian(neck):
    """(head poses (N, 4, 4), angular Jacobians (N, 3, 3) in base_link) for neck angles (N, 3)."""
    tree = _tree()
    P = tree.node_poses(neck)
    J = np.zeros((len(neck), 3, 3))
    for i in tree.chains["head"]:
        _, _, axis, _, c, mult, _ = tree.nodes[i]
        J[:, :, c] += mult * (P[i][..., :3, :3] @ axis)
    return P[tree.index["head"]], J


def limits(margin: float):
    return LOWER[NECK] + margin, UPPER[NECK] - margin


def solve_base(R_base, seed=None, margin: float = 0.03, iterations: int = 25):
    """Neck angles (N, 3) whose head rotation in ``base_link`` is closest to ``R_base`` (N, 3, 3),
    within the neck limits minus ``margin``. Rows are independent (the neck range is small enough
    for a unique solution)."""
    R_base = np.asarray(R_base, float)
    lo, hi = limits(margin)
    q = np.zeros((len(R_base), 3)) if seed is None else np.clip(np.array(seed, float), lo, hi)
    for _ in range(iterations):
        P, J = _jacobian(q)
        e = so3_log(R_base @ np.swapaxes(P[:, :3, :3], 1, 2))  # base-frame rotation error
        H = np.swapaxes(J, 1, 2) @ J + 1e-6 * np.eye(3)
        step = np.linalg.solve(H, (np.swapaxes(J, 1, 2) @ e[..., None]))[..., 0]
        new = np.clip(q + step, lo, hi)
        if np.abs(new - q).max() < 1e-10:
            q = new
            break
        q = new
    return q


def rate_limit(neck, start, dt, scale: float = 1.0):
    """Clip the per-row change of neck angles (N, 3) to ``scale`` x the neck speed, starting from
    ``start`` (3,): a forward pass, then a backward pass so a late target is approached in time."""
    step = VELOCITY[NECK] * scale * np.asarray(dt, float).reshape(-1, 1)
    out = np.array(neck, float)
    out[0] = start
    for t in range(1, len(out)):
        out[t] = np.clip(out[t], out[t - 1] - step[t - 1], out[t - 1] + step[t - 1])
    for t in range(len(out) - 2, 0, -1):
        out[t] = np.clip(out[t], out[t + 1] - step[t], out[t + 1] + step[t])
    out[0] = start
    for t in range(1, len(out)):  # the backward pass may have broken the forward bound at the start
        out[t] = np.clip(out[t], out[t - 1] - step[t - 1], out[t - 1] + step[t - 1])
    return out


def solve(q, R_world, *, margin: float = 0.03, iterations: int = 25, start=None, dt=None):
    """Neck angles (T, 3) tracking world head rotations ``R_world`` (T, 3, 3) for body postures
    ``q`` (T, 22) (only the base yaw matters), rate-limited from ``start`` (default: the solution of
    the first row) when the row spacing ``dt`` (T-1,) is given."""
    q = np.asarray(q, float)
    W = planar(q[:, 0], q[:, 1], q[:, 2])[:, :3, :3]
    R_base = np.swapaxes(W, 1, 2) @ np.asarray(R_world, float)
    neck = solve_base(R_base, q[:, NECK], margin, iterations)
    if dt is not None:
        neck = rate_limit(neck, neck[0] if start is None else start, dt)
    return neck


def rotation_error(R_a, R_b) -> np.ndarray:
    """Angle (T,) between two rotation sequences."""
    return np.linalg.norm(so3_log(np.swapaxes(R_a, -1, -2) @ R_b), axis=-1)


def look_at(origin, target, previous=None):
    """Rotation(s) whose +X points from ``origin`` to ``target`` (..., 3) with +Y horizontal
    (reachy-control ``util/geometry.look_at``, batched; near the vertical the previous +Y is kept)."""
    origin, target = np.broadcast_arrays(np.asarray(origin, float), np.asarray(target, float))
    x = target - origin
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    x = np.where(n > 1e-9, x / np.maximum(n, 1e-12), [1.0, 0.0, 0.0])
    y = np.cross([0.0, 0.0, 1.0], x)
    ny = np.linalg.norm(y, axis=-1, keepdims=True)
    hint = np.broadcast_to([0.0, 1.0, 0.0], x.shape) if previous is None else np.asarray(previous)[..., :, 1]
    alt = hint - x * np.sum(x * hint, axis=-1, keepdims=True)
    y = np.where(ny > 1e-5, y, alt)
    y /= np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-12)
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def gaze_neck(points_world, base, *, margin: float, start, dt, iterations: int = 25, rate_scale: float = 0.8):
    """Neck angles (T, 3) that point ``head_tip``'s +X at world points (T, 3) from the base path
    (T, 3), clamped to the limits minus ``margin`` and rate-limited to ``rate_scale`` x the neck speed
    from ``start`` (3,).

    The look-at origin is the ``head_tip`` origin at the solved neck (two passes)."""
    W = planar(base[:, 0], base[:, 1], base[:, 2])
    p = np.einsum("tij,tj->ti", np.linalg.inv(W), np.c_[points_world, np.ones(len(points_world))])[:, :3]
    neck = np.tile(np.asarray(start, float), (len(p), 1))
    for _ in range(2):
        origin = head_tip_base(neck)[:, :3, 3]
        neck = solve_base(look_at(origin, p), neck, margin, iterations)
    return rate_limit(neck, start, dt, rate_scale)
