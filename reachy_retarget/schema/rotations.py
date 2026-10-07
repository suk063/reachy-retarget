"""Vectorized rotation and rigid-transform helpers.

Conventions: quaternions are ``wxyz`` (scipy uses ``xyzw`` internally), rotation6d is
the first two matrix columns concatenated ``(r00, r10, r20, r01, r11, r21)`` exactly as
in reachy-agent ``robot/actions.py::rotation_values``, 7-vectors are ``xyz + wxyz``,
SE(3) twists are ``(v, w)`` and planar poses are ``(x, y, yaw)``. All functions accept
arbitrary leading batch dimensions.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _batched(fn, a, inner):
    a = np.asarray(a, float)
    lead = a.shape[: a.ndim - len(inner)]
    out = fn(a.reshape(-1, *inner))
    return out.reshape(*lead, *out.shape[1:])


def wrap_angle(a):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def quat_to_matrix(q):
    """wxyz quaternion (..., 4) -> rotation matrix (..., 3, 3); q is normalized."""
    return _batched(lambda x: Rotation.from_quat(x[:, [1, 2, 3, 0]]).as_matrix(), q, (4,))


def matrix_to_quat(R):
    """Rotation matrix (..., 3, 3) -> unit wxyz quaternion with w >= 0."""
    q = _batched(lambda x: Rotation.from_matrix(x).as_quat()[:, [3, 0, 1, 2]], R, (3, 3))
    return np.where(q[..., :1] < 0, -q, q)


def rot6d(R):
    """First two columns of R concatenated: (..., 3, 3) -> (..., 6)."""
    R = np.asarray(R)
    return np.concatenate((R[..., :, 0], R[..., :, 1]), axis=-1)


def rot6d_to_matrix(v):
    """Gram–Schmidt inverse of :func:`rot6d`; raises on degenerate input."""
    v = np.asarray(v, float)
    if v.shape[-1:] != (6,) or not np.isfinite(v).all():
        raise ValueError("expected finite rotation6d values")
    x, y = v[..., :3], v[..., 3:]
    nx = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(nx < 1e-6):
        raise ValueError("degenerate rotation6d first axis")
    x = x / nx
    y = y - x * np.sum(x * y, axis=-1, keepdims=True)
    ny = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(ny < 1e-6):
        raise ValueError("degenerate rotation6d second axis")
    y = y / ny
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def so3_log(R):
    """Rotation matrix (..., 3, 3) -> rotation vector (..., 3)."""
    return _batched(lambda x: Rotation.from_matrix(x).as_rotvec(), R, (3, 3))


def so3_exp(w):
    """Rotation vector (..., 3) -> rotation matrix (..., 3, 3)."""
    return _batched(lambda x: Rotation.from_rotvec(x).as_matrix(), w, (3,))


def pose_to_vec7(T):
    """Homogeneous pose (..., 4, 4) -> xyz + wxyz (..., 7)."""
    T = np.asarray(T, float)
    return np.concatenate((T[..., :3, 3], matrix_to_quat(T[..., :3, :3])), axis=-1)


def vec7_to_pose(v):
    """xyz + wxyz (..., 7) -> homogeneous pose (..., 4, 4)."""
    v = np.asarray(v, float)
    T = np.zeros((*v.shape[:-1], 4, 4))
    T[..., :3, :3] = quat_to_matrix(v[..., 3:])
    T[..., :3, 3] = v[..., :3]
    T[..., 3, 3] = 1.0
    return T


def se3_inv(T):
    T = np.asarray(T, float)
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = R
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", R, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _hat(w):
    W = np.zeros((*w.shape[:-1], 3, 3))
    W[..., 0, 1], W[..., 0, 2], W[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    return W - np.swapaxes(W, -1, -2)


def se3_log(T):
    """Pose (..., 4, 4) -> twist (..., 6) = (v, w) with exp(hat(v, w)) = T."""
    T = np.asarray(T, float)
    w = so3_log(T[..., :3, :3])
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    small = th < 1e-6
    ths = np.where(small, 1.0, th)
    # (1 - th sin th / (2 (1 - cos th))) / th^2, with its series limit 1/12 near zero.
    c = np.where(small, 1 / 12, (1 - ths * np.sin(ths) / (2 * (1 - np.cos(ths)))) / ths**2)
    W = _hat(w)
    V_inv = np.eye(3) - 0.5 * W + c * W @ W
    return np.concatenate((np.einsum("...ij,...j->...i", V_inv, T[..., :3, 3]), w), axis=-1)


def se3_exp(xi):
    """Twist (..., 6) = (v, w) -> pose (..., 4, 4)."""
    xi = np.asarray(xi, float)
    v, w = xi[..., :3], xi[..., 3:]
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    small = th < 1e-6
    ths = np.where(small, 1.0, th)
    a = np.where(small, 0.5, (1 - np.cos(ths)) / ths**2)
    b = np.where(small, 1 / 6, (ths - np.sin(ths)) / ths**3)
    W = _hat(w)
    T = np.zeros((*xi.shape[:-1], 4, 4))
    T[..., :3, :3] = so3_exp(w)
    T[..., :3, 3] = np.einsum("...ij,...j->...i", np.eye(3) + a * W + b * W @ W, v)
    T[..., 3, 3] = 1.0
    return T


def planar_pose(p):
    """Planar pose (..., 3) = (x, y, yaw) -> homogeneous pose (..., 4, 4) at z = 0."""
    p = np.asarray(p, float)
    c, s = np.cos(p[..., 2]), np.sin(p[..., 2])
    T = np.zeros((*p.shape[:-1], 4, 4))
    T[..., 0, 0], T[..., 0, 1], T[..., 1, 0], T[..., 1, 1] = c, -s, s, c
    T[..., 0, 3], T[..., 1, 3] = p[..., 0], p[..., 1]
    T[..., 2, 2] = T[..., 3, 3] = 1.0
    return T


def se2_relative(start, end):
    """Body-frame displacement (dx, dy, dyaw) from world pose start to end; dyaw wrapped."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    d = end[..., :2] - start[..., :2]
    c, s = np.cos(start[..., 2]), np.sin(start[..., 2])
    return np.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1],
                     wrap_angle(end[..., 2] - start[..., 2])), axis=-1)


def se2_compose(start, delta):
    """World pose after the body-frame displacement ``delta`` from world pose ``start``."""
    start, delta = np.asarray(start, float), np.asarray(delta, float)
    c, s = np.cos(start[..., 2]), np.sin(start[..., 2])
    return np.stack((start[..., 0] + c * delta[..., 0] - s * delta[..., 1],
                     start[..., 1] + s * delta[..., 0] + c * delta[..., 1],
                     start[..., 2] + delta[..., 2]), axis=-1)


def _se2_coeffs(angle):
    a = np.sinc(angle / np.pi)                              # sin(t) / t
    b = 0.5 * angle * np.sinc(angle / (2 * np.pi)) ** 2     # (1 - cos t) / t
    return a, b


def se2_log(delta, dt):
    """Constant body twist (vx, vy, wz) whose exponential over ``dt`` reaches ``delta``."""
    delta = np.asarray(delta, float)
    angle = wrap_angle(delta[..., 2])
    a, b = _se2_coeffs(angle)
    n = a * a + b * b
    vx = (a * delta[..., 0] + b * delta[..., 1]) / n
    vy = (-b * delta[..., 0] + a * delta[..., 1]) / n
    return np.stack((vx, vy, angle), axis=-1) / dt


def se2_exp(twist, dt):
    """Body-frame displacement after holding body twist (vx, vy, wz) for ``dt``."""
    v = np.asarray(twist, float) * dt
    a, b = _se2_coeffs(v[..., 2])
    return np.stack((a * v[..., 0] - b * v[..., 1], b * v[..., 0] + a * v[..., 1], v[..., 2]), axis=-1)
