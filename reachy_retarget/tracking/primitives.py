"""Path pieces for synthetic tracking references (numpy only, independent of any controller).

Ideas follow reachy-control ``experiment/scenarios.py`` and ``manipulation.py`` (smoothstep
phases, SE(3) paths with linear translation and geodesic rotation, +X look-at) and
``rl_tracking/data.py`` (min-jerk blends, joint-space witnesses); the code is written here and
imports nothing from that repository.

* :class:`Track`: a hand pose timeline built from segments ``(duration, u -> (n, 4, 4))``.
* Hand primitives (``reach``, ``line``, ``circle``, ``raster``, ``press``, ``hinge``, ``twist``,
  ``pour``, ``oscillate``, ``wander``): each returns ``(duration, fn)`` starting at the current
  pose ``P``; durations follow from linear and angular speeds (``speed["lin"]`` m/s,
  ``speed["ang"]`` rad/s) and the min-jerk peak factor.
* :class:`BasePath` and base primitives (``drive``, ``arc``, ``turn``, ``spline``) for SE(2) paths
  with unwrapped yaw.
* :func:`gripper_profile`: piecewise-constant openings ramped at the finger speed.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from ..schema.rotations import so3_exp, so3_log

PEAK = 1.875  # peak / mean speed of a min-jerk profile


def min_jerk(u):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def bump(u):
    """0 -> 1 -> 0 with zero slope at both ends (sin^2)."""
    return np.sin(np.pi * np.clip(np.asarray(u, float), 0.0, 1.0)) ** 2


def rot(axis, angle):
    """Rotation(s) about a unit ``axis`` (3,) by ``angle`` (...,)."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return so3_exp(np.asarray(angle, float)[..., None] * axis)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def random_unit(rng, horizontal=False):
    v = rng.normal(size=3)
    if horizontal:
        v[2] = 0.0
    return unit(v)


def _poses(R, p):
    n = len(p)
    out = np.zeros((n, 4, 4))
    out[:, :3, :3] = R
    out[:, :3, 3] = p
    out[:, 3, 3] = 1.0
    return out


def duration(dist=0.0, angle=0.0, speed=None, minimum=0.3):
    """Min-jerk duration (s) whose mean speeds stay at the linear/angular speeds / PEAK."""
    return max(minimum, PEAK * max(dist / speed["lin"], abs(angle) / speed["ang"]))


class Track:
    """Pose timeline of one hand (or of a carried object frame)."""

    def __init__(self, start):
        self.start = np.array(start, float)
        self.end = self.start.copy()
        self.t = 0.0
        self.segments: list[tuple[float, float, object]] = []  # (t0, duration, fn)
        self.events: list[tuple[float, str]] = []

    def add(self, dur, fn, kind="move"):
        dur = float(dur)
        if dur <= 0:
            return self
        self.segments.append((self.t, dur, fn))
        self.events.append((self.t, f"{kind}_start"))
        self.t += dur
        self.end = fn(np.array([1.0]))[0]
        self.events.append((self.t, f"{kind}_end"))
        return self

    def hold(self, dur):
        if dur > 0:
            P = self.end.copy()
            self.add(dur, lambda u, P=P: np.broadcast_to(P, (len(u), 4, 4)).copy(), "hold")
        return self

    def sample(self, times):
        times = np.asarray(times, float)
        out = np.broadcast_to(self.end, (len(times), 4, 4)).copy()
        if not self.segments:
            out[:] = self.start
            return out
        starts = np.array([s[0] for s in self.segments])
        k = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(starts) - 1)
        for i in np.unique(k):
            t0, dur, fn = self.segments[i]
            rows = np.flatnonzero((k == i) & (times <= t0 + dur + 1e-12))
            if len(rows):
                out[rows] = fn((times[rows] - t0) / dur)
        return out


# ---------------------------------------------------------------------------- hand primitives


def reach(P, Q, speed):
    """Straight line and geodesic rotation from P to Q (min-jerk)."""
    P, Q = np.array(P, float), np.array(Q, float)
    rel = so3_log(P[:3, :3].T @ Q[:3, :3])
    dist = np.linalg.norm(Q[:3, 3] - P[:3, 3])

    def fn(u):
        s = min_jerk(u)
        return _poses(P[:3, :3] @ so3_exp(s[:, None] * rel), P[:3, 3] + s[:, None] * (Q[:3, 3] - P[:3, 3]))
    return duration(dist, np.linalg.norm(rel), speed, 0.4), fn


def line(P, direction, length, speed, back=True):
    d = unit(direction) * length

    def fn(u):
        s = bump(u) if back else min_jerk(u)
        return _poses(np.broadcast_to(P[:3, :3], (len(u), 3, 3)), P[:3, 3] + s[:, None] * d)
    # bump: peak speed pi * length / dur
    dur = max(0.4, np.pi * length / speed["lin"]) if back else duration(length, 0, speed)
    return dur, fn


def circle(P, radius, normal, turns, speed):
    n = unit(normal)
    e1 = unit(np.cross(n, [1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.cross(n, [0.0, 1.0, 0.0]))
    e2 = np.cross(n, e1)
    center = P[:3, 3] - radius * e1

    def fn(u):
        a = 2 * np.pi * turns * min_jerk(u)
        p = center + radius * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2)
        return _poses(np.broadcast_to(P[:3, :3], (len(u), 3, 3)), p)
    return duration(2 * np.pi * radius * turns, 0, speed), fn


def raster(P, axis_a, axis_b, width, height, rows, speed):
    """Zig-zag over a ``width`` (along ``axis_a``) x ``height`` (along ``axis_b``) patch in ``rows``
    passes, then straight back to the start."""
    a, b = unit(axis_a), unit(axis_b)
    rows = max(int(rows), 2)
    pts = []
    for r in range(rows):
        y = height * r / (rows - 1)
        xs = (0.0, width) if r % 2 == 0 else (width, 0.0)
        pts += [xs[0] * a + y * b, xs[1] * a + y * b]
    pts.append(np.zeros(3))
    pts = np.array(pts)
    legs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.r_[True, legs > 1e-9]]
    legs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    durs = np.array([duration(L, 0, speed, 0.2) for L in legs])
    edges = np.r_[0, np.cumsum(durs)] / durs.sum()

    def fn(u):
        u = np.asarray(u, float)
        k = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(legs) - 1)
        s = min_jerk((u - edges[k]) / (edges[k + 1] - edges[k]))
        p = P[:3, 3] + pts[k] + s[:, None] * (pts[k + 1] - pts[k])
        return _poses(np.broadcast_to(P[:3, :3], (len(u), 3, 3)), p)
    return float(durs.sum()), fn


def press(P, depth, reps, speed):
    """Along the approach axis (TCP -z: from the wrist toward the fingertips) and back, ``reps`` times."""
    a = -P[:3, 2]

    def fn(u):
        s = bump(np.mod(np.asarray(u) * reps, 1.0 + 1e-12))
        s = np.where(np.asarray(u) >= 1.0, 0.0, s)
        return _poses(np.broadcast_to(P[:3, :3], (len(u), 3, 3)), P[:3, 3] + s[:, None] * depth * a)
    return max(0.5, reps * np.pi * depth / speed["lin"]), fn


def hinge(P, pivot_offset, axis, angle, speed, back=False):
    """Rotate the hand rigidly about an axis through ``P.p + pivot_offset`` (a door or lid)."""
    pivot = P[:3, 3] + np.asarray(pivot_offset, float)
    axis = unit(axis)
    r = np.linalg.norm(np.cross(axis, P[:3, 3] - pivot))

    def fn(u):
        s = bump(u) if back else min_jerk(u)
        R = rot(axis, s * angle)
        return _poses(R @ P[:3, :3], pivot + np.einsum("nij,j->ni", R, P[:3, 3] - pivot))
    dur = duration(r * abs(angle) * (2 if back else 1), abs(angle) * (2 if back else 1), speed)
    return dur, fn


def twist(P, angle, speed, back=True):
    """Rotate about the TCP z axis (exactly the wrist yaw joint)."""
    def fn(u):
        s = bump(u) if back else min_jerk(u)
        return _poses(P[:3, :3] @ rot([0.0, 0.0, 1.0], s * angle), np.broadcast_to(P[:3, 3], (len(u), 3)))
    return duration(0, abs(angle) * (2 if back else 1), speed), fn


def pour(P, center, axis, angle, speed, hold_s=0.0):
    """Tilt about a world ``axis`` through ``center`` (the grasp center) and back, holding the tilt
    for ``hold_s``."""
    axis = unit(axis)
    tilt = duration(0, abs(angle), speed)
    total = 2 * tilt + hold_s

    def fn(u):
        t = np.asarray(u) * total
        s = np.where(t < tilt, min_jerk(t / tilt), np.where(t < tilt + hold_s, 1.0,
                                                               1.0 - min_jerk((t - tilt - hold_s) / tilt)))
        R = rot(axis, s * angle)
        return _poses(R @ P[:3, :3], center + np.einsum("nij,j->ni", R, P[:3, 3] - center))
    return total, fn


def oscillate(P, amp_pos, amp_rot, freq, cycles):
    """Sinusoidal position (amp (3,), m) and rotation (amp (3,), rad, about world axes) offsets with
    a sin^2 envelope."""
    amp_pos, amp_rot = np.asarray(amp_pos, float), np.asarray(amp_rot, float)
    dur = cycles / freq

    def fn(u):
        u = np.asarray(u, float)
        w = np.sin(2 * np.pi * cycles * u)[:, None] * bump(u)[:, None]
        return _poses(so3_exp(w * amp_rot) @ P[:3, :3], P[:3, 3] + w * amp_pos)
    return dur, fn


def wander(P, rng, amp_pos, amp_rot, knots, speed):
    """Smooth random offsets (cubic spline through ``knots`` random points, zero at both ends)."""
    k = max(int(knots), 1)
    off_p = np.r_[[np.zeros(3)], rng.uniform(-1, 1, (k, 3)) * amp_pos, [np.zeros(3)]]
    off_r = np.r_[[np.zeros(3)], rng.uniform(-1, 1, (k, 3)) * amp_rot, [np.zeros(3)]]
    s = np.linspace(0, 1, k + 2)
    sp, sr = CubicSpline(s, off_p, bc_type="clamped"), CubicSpline(s, off_r, bc_type="clamped")
    path = np.linalg.norm(np.diff(sp(np.linspace(0, 1, 200)), axis=0), axis=1).sum()
    ang = np.linalg.norm(np.diff(sr(np.linspace(0, 1, 200)), axis=0), axis=1).sum()
    dur = 1.5 * max(path / speed["lin"], ang / speed["ang"], 0.5)

    def fn(u):
        u = np.asarray(u, float)
        return _poses(so3_exp(sr(u)) @ P[:3, :3], P[:3, 3] + sp(u))
    return dur, fn


def mirror(X):
    """Reflect poses (..., 4, 4) about the x-z plane of their frame: the right TCP at the mirrored
    joints of a left-arm posture (signs [1, -1, -1, 1, -1, 1, -1]) is ``mirror(left TCP)`` (to the
    URDF's ~2 um left/right asymmetry)."""
    M = np.diag([1.0, -1.0, 1.0, 1.0])
    return M @ X @ M


MIRROR_SIGNS = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


# ---------------------------------------------------------------------------- base paths


class BasePath:
    """SE(2) timeline (x, y, unwrapped yaw)."""

    def __init__(self, start):
        self.start = np.array(start, float)
        self.end = self.start.copy()
        self.t = 0.0
        self.segments: list[tuple[float, float, object]] = []
        self.events: list[tuple[float, str]] = []

    def add(self, dur, fn, kind="drive"):
        if dur <= 0:
            return self
        self.segments.append((self.t, float(dur), fn))
        self.events.append((self.t, f"{kind}_start"))
        self.t += float(dur)
        self.end = fn(np.array([1.0]))[0]
        self.events.append((self.t, f"{kind}_end"))
        return self

    def hold(self, dur):
        if dur > 0:
            b = self.end.copy()
            self.add(dur, lambda u, b=b: np.broadcast_to(b, (len(u), 3)).copy(), "stop")
        return self

    def sample(self, times):
        times = np.asarray(times, float)
        out = np.broadcast_to(self.end, (len(times), 3)).copy()
        if not self.segments:
            out[:] = self.start
            return out
        starts = np.array([s[0] for s in self.segments])
        k = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(starts) - 1)
        for i in np.unique(k):
            t0, dur, fn = self.segments[i]
            rows = np.flatnonzero((k == i) & (times <= t0 + dur + 1e-12))
            if len(rows):
                out[rows] = fn((times[rows] - t0) / dur)
        return out


def drive(b, distance, direction, dyaw, speed):
    """Holonomic straight move: ``distance`` m along body direction ``direction`` (rad from the
    initial heading) while the heading turns by ``dyaw``."""
    b = np.array(b, float)
    head = b[2] + direction
    d = distance * np.array([np.cos(head), np.sin(head)])
    dur = max(0.5, PEAK * max(abs(distance) / speed["base"], abs(dyaw) / speed["yaw"]))

    def fn(u):
        s = min_jerk(u)
        return np.c_[b[0] + s * d[0], b[1] + s * d[1], b[2] + s * dyaw]
    return dur, fn


def arc(b, radius, angle, speed):
    """Drive along a circle of ``radius`` (m, > 0 = turning left) by ``angle`` (rad) with the
    heading tangent to it."""
    b = np.array(b, float)
    c = b[:2] + radius * np.array([-np.sin(b[2]), np.cos(b[2])])
    sign = np.sign(radius) or 1.0
    length = abs(radius * angle)
    dur = max(0.5, PEAK * max(length / speed["base"], abs(angle) / speed["yaw"]))

    def fn(u):
        a = sign * min_jerk(u) * abs(angle)
        yaw = b[2] + a
        return np.c_[c[0] + radius * np.sin(yaw), c[1] - radius * np.cos(yaw), yaw]
    return dur, fn


def turn(b, dyaw, speed):
    return drive(b, 0.0, 0.0, dyaw, speed)


def spline(b, waypoints, speed, heading="tangent"):
    """Smooth path through world ``waypoints`` (k, 2) from ``b``. ``heading``: ``tangent`` (face
    the travel direction; the caller turns the base toward the initial tangent, ``fn(0)``'s yaw,
    first) or ``fixed`` (keep b's heading, holonomic)."""
    b = np.array(b, float)
    pts = np.r_[[b[:2]], np.asarray(waypoints, float)]
    pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]]
    if len(pts) < 2:
        return 0.0, None
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    cs = CubicSpline(s, pts, bc_type="natural") if len(pts) > 2 else None

    def xy(q):
        return cs(q) if cs is not None else pts[0] + np.outer(q / s[-1], pts[1] - pts[0])
    fine = np.linspace(0, s[-1], 400)
    p = xy(fine)
    length = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    if heading == "tangent":
        g = np.gradient(p, axis=0)
        yaw_fine = np.unwrap(np.arctan2(g[:, 1], g[:, 0]))
        yaw_fine += 2 * np.pi * np.round((b[2] - yaw_fine[0]) / (2 * np.pi))
        yaw_of = PchipInterpolator(fine, yaw_fine)
        turn_total = np.abs(np.diff(yaw_fine)).sum()
    else:
        yaw_of, turn_total = None, 0.0
    dur = max(0.5, PEAK * max(length / speed["base"], turn_total / speed["yaw"]))

    def fn(u):
        q = min_jerk(u) * s[-1]
        yaw = yaw_of(q) if yaw_of is not None else np.full(len(q), b[2])
        return np.c_[xy(q), yaw]
    return dur, fn


def body_speed_ratio(base, dt, limits):
    """Peak body-frame speed of a base path (T, 3) as a fraction of ``limits`` (3,)."""
    d = np.diff(base, axis=0)
    yaw = (base[:-1, 2] + base[1:, 2]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    body = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]] / dt
    return float(np.max(np.abs(body) / limits)) if len(body) else 0.0


# ---------------------------------------------------------------------------- grippers


def gripper_profile(times, levels, rate):
    """Opening (T,) from ``levels`` [(t, opening)] (piecewise constant from each t), ramped at most
    ``rate`` (opening units per second)."""
    times = np.asarray(times, float)
    target = np.full(len(times), levels[0][1], float)
    for t, v in sorted(levels):
        target[times >= t] = v
    out = target.copy()
    dt = np.diff(times)
    for i in range(1, len(out)):
        out[i] = np.clip(target[i], out[i - 1] - rate * dt[i - 1], out[i - 1] + rate * dt[i - 1])
    return np.clip(out, 0.0, 1.0)
