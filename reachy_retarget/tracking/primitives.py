"""Path pieces for synthetic tracking references (numpy only, independent of any controller).

Ideas follow reachy-control ``experiment/scenarios.py`` and ``manipulation.py`` (smoothstep
phases, SE(3) paths with linear translation and geodesic rotation, +X look-at) and
``rl_tracking/data.py`` (min-jerk blends, joint-space witnesses); the code is written here and
imports nothing from that repository.

* :class:`Track`: a hand pose timeline built from segments ``(duration, u -> (n, 4, 4))``.
* Hand motions (``reach``, ``curved_reach``, ``line``, ``approach``, ``hinge``, ``twist``, ``tilt``):
  each goes once from the current pose ``P`` to one goal and returns ``(duration, fn)``; durations
  follow from linear and angular speeds (``speed["lin"]`` m/s, ``speed["ang"]`` rad/s) and the
  min-jerk peak factor.
* :class:`BasePath` and base motions (``drive``, ``arc``, ``turn``, ``curve``) for SE(2) paths with
  unwrapped yaw, each one drive to one goal.
* :func:`gripper_profile`: piecewise-constant openings ramped at the finger speed.
"""
from __future__ import annotations

import numpy as np

from ..schema.rotations import so3_exp, so3_log

PEAK = 1.875  # peak / mean speed of a min-jerk profile


def min_jerk(u):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


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


# ---------------------------------------------------------------------------- hand motions
# Every motion goes once from the current pose P to one goal: no return, repetition or cycle.


def reach(P, Q, speed):
    """Straight line and geodesic rotation from P to Q (min-jerk)."""
    P, Q = np.array(P, float), np.array(Q, float)
    rel = so3_log(P[:3, :3].T @ Q[:3, :3])
    dist = np.linalg.norm(Q[:3, 3] - P[:3, 3])

    def fn(u):
        s = min_jerk(u)
        return _poses(P[:3, :3] @ so3_exp(s[:, None] * rel), P[:3, 3] + s[:, None] * (Q[:3, 3] - P[:3, 3]))
    return duration(dist, np.linalg.norm(rel), speed, 0.4), fn


def curved_reach(P, Q, bulge, speed):
    """P to Q along a quadratic Bezier bowed by ``bulge`` (3,) at its middle (one smooth stroke)."""
    P, Q = np.array(P, float), np.array(Q, float)
    rel = so3_log(P[:3, :3].T @ Q[:3, :3])
    a, c = P[:3, 3], Q[:3, 3]
    b = (a + c) / 2 + 2 * np.asarray(bulge, float)  # control point: the curve's middle is (a+c)/2 + bulge
    t = np.linspace(0, 1, 200)[:, None]
    length = np.linalg.norm(np.diff((1 - t) ** 2 * a + 2 * t * (1 - t) * b + t ** 2 * c, axis=0), axis=1).sum()

    def fn(u):
        s = min_jerk(u)[:, None]
        return _poses(P[:3, :3] @ so3_exp(s * rel), (1 - s) ** 2 * a + 2 * s * (1 - s) * b + s ** 2 * c)
    return duration(length, np.linalg.norm(rel), speed, 0.4), fn


def line(P, direction, length, speed):
    """Translate by ``length`` along ``direction`` with the orientation kept (push, pull, lift, slide)."""
    d = unit(direction) * length

    def fn(u):
        return _poses(np.broadcast_to(P[:3, :3], (len(u), 3, 3)), P[:3, 3] + min_jerk(u)[:, None] * d)
    return duration(length, 0, speed), fn


def approach(P, depth, speed):
    """Move along the approach axis (TCP -z, wrist toward fingertips) by ``depth`` (< 0: retreat)."""
    return line(P, -P[:3, 2] * np.sign(depth or 1.0), abs(depth), speed)


def hinge(P, pivot_offset, axis, angle, speed):
    """Rotate the hand rigidly about an axis through ``P.p + pivot_offset`` (a door or lid)."""
    pivot = P[:3, 3] + np.asarray(pivot_offset, float)
    axis = unit(axis)
    r = np.linalg.norm(np.cross(axis, P[:3, 3] - pivot))

    def fn(u):
        R = rot(axis, min_jerk(u) * angle)
        return _poses(R @ P[:3, :3], pivot + np.einsum("nij,j->ni", R, P[:3, 3] - pivot))
    return duration(r * abs(angle), abs(angle), speed), fn


def twist(P, angle, speed):
    """Rotate about the TCP z axis (exactly the wrist yaw joint)."""
    def fn(u):
        return _poses(P[:3, :3] @ rot([0.0, 0.0, 1.0], min_jerk(u) * angle), np.broadcast_to(P[:3, 3], (len(u), 3)))
    return duration(0, abs(angle), speed), fn


def tilt(P, center, axis, angle, speed):
    """Tilt by ``angle`` about a world ``axis`` through ``center`` (the grasp center), one way."""
    axis = unit(axis)
    center = np.asarray(center, float)
    r = np.linalg.norm(np.cross(axis, P[:3, 3] - center))

    def fn(u):
        R = rot(axis, min_jerk(u) * angle)
        return _poses(R @ P[:3, :3], center + np.einsum("nij,j->ni", R, P[:3, 3] - center))
    return duration(r * abs(angle), abs(angle), speed), fn


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


def curve(b, goal_xy, goal_yaw, speed):
    """One smooth drive to ``goal_xy`` along a cubic Bezier leaving along the current heading and
    arriving along ``goal_yaw``; the heading follows the tangent (no turn in place first)."""
    b = np.array(b, float)
    p0, p3 = b[:2], np.asarray(goal_xy, float)
    span = np.linalg.norm(p3 - p0)
    h0 = np.array([np.cos(b[2]), np.sin(b[2])])
    h3 = np.array([np.cos(goal_yaw), np.sin(goal_yaw)])
    p1, p2 = p0 + h0 * span / 3, p3 - h3 * span / 3
    t = np.linspace(0, 1, 400)[:, None]
    pts = (1 - t) ** 3 * p0 + 3 * t * (1 - t) ** 2 * p1 + 3 * t ** 2 * (1 - t) * p2 + t ** 3 * p3
    d = 3 * (1 - t) ** 2 * (p1 - p0) + 6 * t * (1 - t) * (p2 - p1) + 3 * t ** 2 * (p3 - p2)  # exact tangent
    yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    yaw += 2 * np.pi * np.round((b[2] - yaw[0]) / (2 * np.pi))
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    dur = max(0.5, PEAK * max(seg[-1] / speed["base"], np.abs(np.diff(yaw)).sum() / speed["yaw"]))

    def fn(u):
        q = np.interp(min_jerk(u) * seg[-1], seg, np.arange(len(seg)) / (len(seg) - 1))
        return np.c_[np.interp(q, t[:, 0], pts[:, 0]), np.interp(q, t[:, 0], pts[:, 1]), np.interp(q, t[:, 0], yaw)]
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
