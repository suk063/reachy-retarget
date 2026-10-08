"""Gaze (docs/design.md, step 5): point the head at the currently manipulated object.

Gaze point per frame, in priority order: the objects held by active hands (grasp label
set), else the objects the active hands approach next (label of the next grasp phase),
else the midpoint of the active hands' grasp centers, else a point 1.5 m ahead of the base at
0.6 m height (no active hand: navigation), ahead along the direction of travel (toward the base
position ``cfg.gaze_lookahead_m`` further along its path) while the base moves, else along its
heading. The point path is Gaussian-smoothed in time.

The head's viewing axis is the optical (+z) axis of the URDF eye cameras expressed in the
``head`` frame (it equals head +x); the viewpoint is the midpoint of the two eyes. Neck
roll is held at 0; pitch and yaw are solved by batched Gauss-Newton, clipped to the neck
limits minus a margin, scaled toward 0 where the turned head would come closer to an arm than
the IK clearance margin, and rate limited to the scaled neck speed on the source clock (so
gaze never forces time dilation).
"""
from __future__ import annotations

import functools

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..robot import LOWER, UPPER, VELOCITY, Reachy, planar
from ..robot.reachy import NECK
from ..schema.rotations import se3_inv, so3_exp
from .config import RetargetConfig

PITCH, YAW = NECK.start + 1, NECK.start + 2


@functools.cache
def eye_axes():
    """(viewpoint (3,), viewing axis (3,)) in the ``head`` frame from the eye cameras."""
    u = Reachy.load().urdf
    cams = [u.transform("head", f"{s}_camera_optical") for s in ("left", "right")]
    axis = np.mean([T[:3, 2] for T in cams], axis=0)
    return np.mean([T[:3, 3] for T in cams], axis=0), axis / np.linalg.norm(axis)


def _next_label(labels):
    """For each frame, the label of the next frame at or after it with a label >= 0 (else -1)."""
    out = np.full(len(labels), -1)
    nxt = -1
    for t in range(len(labels) - 1, -1, -1):
        if labels[t] >= 0:
            nxt = labels[t]
        out[t] = nxt
    return out


def gaze_points(hands, labels, ids, objects, base, cfg: RetargetConfig, times):
    """World gaze points (T, 3) and the rule used per frame (0 held, 1 approached, 2 hands,
    3 ahead).

    hands: {side: (T, 3)} grasp-center positions of active hands; labels: {side: (T,)}
    indices into ``ids`` (object ids); objects: {id: ObjectTrack}; base: (T, 3).
    """
    T = len(base)
    out, rule = np.zeros((T, 3)), np.full(T, 3)
    ahead = None if hands else travel_direction(base, cfg.gaze_lookahead_m)
    nxt = {s: _next_label(lab) for s, lab in labels.items()}
    for t in range(T):
        for r, lab in ((0, labels), (1, nxt)):
            pts = [objects[ids[lab[s][t]]].pose[t, :3] for s in hands
                   if lab[s][t] >= 0 and objects[ids[lab[s][t]]].valid[t]]
            if pts:
                out[t], rule[t] = np.mean(pts, axis=0), r
                break
        else:
            if hands:
                out[t], rule[t] = np.mean([h[t] for h in hands.values()], axis=0), 2
            else:
                d = ahead[t]
                out[t] = (base[t, 0] + 1.5 * d[0], base[t, 1] + 1.5 * d[1], 0.6)
    sigma = cfg.gaze_smoothing_s / np.median(np.diff(times))
    return gaussian_filter1d(out, sigma, axis=0, mode="nearest"), rule


def travel_direction(base, lookahead):
    """Unit floor directions (T, 2) to look along while moving: toward the base position
    ``lookahead`` metres further along its path (where the base is going), else the heading
    (standing, turning in place, or the last ``lookahead`` metres of the path)."""
    base = np.asarray(base, float)
    out = np.c_[np.cos(base[:, 2]), np.sin(base[:, 2])]
    step = np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    j = np.searchsorted(s, s + lookahead)
    ok = j < len(base)
    d = base[np.minimum(j, len(base) - 1), :2] - base[:, :2]
    n = np.linalg.norm(d, axis=1)
    ok &= n > 0.5 * lookahead  # mostly straight progress, not a turn on the spot
    out[ok] = d[ok] / n[ok, None]
    return out


def _view(q):
    """(viewpoint (T, 3), viewing axis (T, 3), head pose (T, 4, 4)) in base_link."""
    H = Reachy.load().fk_base(q)["head"]
    eye, axis = eye_axes()
    return H[:, :3, :3] @ eye + H[:, :3, 3], H[:, :3, :3] @ axis, H


def solve_neck(q, points, times, cfg: RetargetConfig, iterations=10):
    """Neck angles (T, 3) pointing the head at world ``points`` (T, 3) for body postures q
    (T, 22), and the gaze reference head pose (T, 4, 4) in the world: the achieved head
    pose rotated minimally so that its viewing axis passes through the point."""
    q = np.array(q, float)
    q[:, NECK] = 0.0
    lo, hi = LOWER[[PITCH, YAW]] + cfg.gaze_margin, UPPER[[PITCH, YAW]] - cfg.gaze_margin
    W = planar(q[:, 0], q[:, 1], q[:, 2])
    p_base = np.einsum("tij,tj->ti", se3_inv(W), np.c_[points, np.ones(len(points))])[:, :3]
    h = 1e-5
    for _ in range(iterations):
        eye, axis, _ = _view(q)
        want = p_base - eye
        want /= np.linalg.norm(want, axis=1, keepdims=True)
        J = np.empty((len(q), 3, 2))
        for i, col in enumerate((PITCH, YAW)):
            qh = q.copy()
            qh[:, col] += h
            J[:, :, i] = (_view(qh)[1] - axis) / h
        r = want - axis
        JtJ = np.einsum("tki,tkj->tij", J, J) + 1e-6 * np.eye(2)
        step = np.linalg.solve(JtJ, np.einsum("tki,tk->ti", J, r)[..., None])[..., 0]
        # Projected Gauss-Newton: a joint at its bound that the step pushes outward is held
        # and the other one is re-solved alone.
        cur = q[:, [PITCH, YAW]]
        held = ((cur <= lo + 1e-12) & (step < 0)) | ((cur >= hi - 1e-12) & (step > 0))
        for i in range(2):
            alone = held[:, i] & ~held[:, 1 - i]
            Jo = J[alone, :, 1 - i]
            step[alone, 1 - i] = np.sum(Jo * r[alone], axis=1) / (np.sum(Jo * Jo, axis=1) + 1e-6)
        step[held] = 0.0
        q[:, [PITCH, YAW]] = np.clip(cur + np.clip(step, -0.5, 0.5), lo, hi)
    vmax = VELOCITY[[PITCH, YAW]] * cfg.velocity_scale
    dt = np.diff(times)[:, None]
    cleared = _keep_clear(q, cfg)
    # Magnitude envelope: where clearance forces the neck toward 0, start returning early
    # enough for the rate limit (backward pass), then rate limit forward.
    bound = np.abs(cleared)
    for t in range(len(bound) - 2, -1, -1):
        bound[t] = np.minimum(bound[t], bound[t + 1] + vmax * dt[t])
    neck = np.clip(q[:, [PITCH, YAW]], -bound, bound)
    for t in range(1, len(neck)):
        neck[t] = neck[t - 1] + np.clip(neck[t] - neck[t - 1], -vmax * (times[t] - times[t - 1]),
                                        vmax * (times[t] - times[t - 1]))
    q[:, [PITCH, YAW]] = neck
    eye, axis, H = _view(q)
    want = p_base - eye
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    c = np.cross(axis, want)
    angle = np.arctan2(np.linalg.norm(c, axis=1), np.sum(axis * want, axis=1))
    n = np.linalg.norm(c, axis=1, keepdims=True)
    rotvec = np.where(n > 1e-12, c / np.maximum(n, 1e-12), 0.0) * angle[:, None]
    ref = H.copy()
    ref[:, :3, :3] = so3_exp(rotvec) @ H[:, :3, :3]
    return q[:, NECK], W @ ref


def _keep_clear(q, cfg: RetargetConfig, steps=8):
    """Neck (pitch, yaw) (T, 2) scaled toward 0 (the posture the arm IK was solved with) on frames
    where turning the head would bring the self-clearance below ``cfg.self_clearance_margin``
    (or below its value with the neck at 0, if that is lower): bisection on the scale."""
    from ..robot import min_clearance
    neck = q[:, [PITCH, YAW]].copy()
    q0 = q.copy()
    q0[:, [PITCH, YAW]] = 0.0
    floor = np.minimum(min_clearance(q0), cfg.self_clearance_margin)
    bad = min_clearance(q) < floor - 1e-9
    if not bad.any():
        return neck
    lo, hi = np.zeros(bad.sum()), np.ones(bad.sum())
    qb = q[bad].copy()
    for _ in range(steps):
        mid = (lo + hi) / 2
        qb[:, [PITCH, YAW]] = neck[bad] * mid[:, None]
        ok = min_clearance(qb) >= floor[bad] - 1e-9
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    neck[bad] *= lo[:, None]
    return neck


def gaze_error(q, points):
    """Angle (T,) between the viewing axis and the direction to world ``points``."""
    q = np.asarray(q, float)
    W = planar(q[:, 0], q[:, 1], q[:, 2])
    p_base = np.einsum("tij,tj->ti", se3_inv(W), np.c_[points, np.ones(len(points))])[:, :3]
    eye, axis, _ = _view(q)
    want = p_base - eye
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    return np.arccos(np.clip(np.sum(axis * want, axis=1), -1, 1))
