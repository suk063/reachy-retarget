"""Phase-preserving time scaling and 50 Hz resampling (docs/design.md, step 7).

Each source interval keeps its geometry and is only ever slowed down: its new duration is
the largest of the source duration and ``|dq_j| / (cfg.velocity_scale * VELOCITY_j)`` over
all joints (base yaw unwrapped). A sliding maximum over ``cfg.dilation_window_s`` smooths
the dilation (never below the per-interval requirement). The source steps into and out of
each grasp last at least ``cfg.grasp_dwell_s`` (fingers settle before the arm moves the
object). Output rows sit every ``DT`` on
the dilated clock; each row maps to a source time (piecewise linear, so linear joint
interpolation keeps every interval's speed) and the last row holds the final source frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..robot import VELOCITY
from ..schema.episode import DT
from ..schema.rotations import pose_to_vec7, quat_to_matrix, so3_exp, so3_log
from ..schema.source import ObjectTrack
from .config import RetargetConfig


@dataclass
class Clock:
    """Map from output rows to source rows: row k lies at ``fraction[k]`` between source
    rows ``index[k]`` and ``index[k] + 1``."""

    time: np.ndarray          # (N,) output time, uniform DT
    source_time: np.ndarray   # (N,) source time of each output row (non-decreasing)
    index: np.ndarray         # (N,) left source row
    fraction: np.ndarray      # (N,) in [0, 1]
    dilation: np.ndarray      # (T-1,) new / source duration of each source interval

    def diagnostics(self, source_times) -> dict:
        span = float(source_times[-1] - source_times[0])
        return {"source_duration": span, "duration": float(self.time[-1]),
                "duration_ratio": float(self.time[-1] / span), "max_dilation": float(self.dilation.max()),
                "dilated_intervals": int(np.sum(self.dilation > 1 + 1e-9))}


def dilation(times, q, cfg: RetargetConfig, grasp_events=None, slides=None):
    """Per-interval slow-down factor (T-1,) >= 1 so q (T, 22) respects scaled speed limits.

    ``grasp_events``: source rows where a grasp starts or ends; the interval ending at a grasp
    start and the one starting at a grasp end last at least ``cfg.grasp_dwell_s`` so the fingers
    settle on (or off) the object before the arm moves it (the source gripper stalls on the
    object within one control step, Reachy's position-servoed fingers in about 0.1-0.2 s).
    The dwell is not spread by the sliding maximum.

    ``slides`` (T, n): source positions (m) of articulated slide joints (drawers). They move no
    faster than ``cfg.slide_speed``: a drawer's damping resists the hand with a force proportional
    to its speed, and Reachy's position-servoed arm, pulling a robosuite drawer (damping 100 N s/m)
    at the source's 0.14 m/s, lagged its reference by 3-4 cm (MimicGen MugCleanup tcp_tracking)."""
    dt = np.diff(times)
    required = np.max(np.abs(body_increments(q)) / (VELOCITY * cfg.velocity_scale), axis=1) / dt
    if slides is not None and np.size(slides) and cfg.slide_speed:
        s = np.nan_to_num(np.asarray(slides, float).reshape(len(times), -1))
        required = np.maximum(required, np.max(np.abs(np.diff(s, axis=0)), axis=1) / cfg.slide_speed / dt)
    required = np.maximum(required, 1.0)
    window = max(1, round(cfg.dilation_window_s / np.median(dt)))
    out = np.maximum(maximum_filter1d(required, 2 * window + 1, mode="nearest"), required)
    for kind, row in grasp_events or ():
        i = row - 1 if kind == "start" else row
        if 0 <= i < len(dt):
            out[i] = max(out[i], cfg.grasp_dwell_s / dt[i])
    return out


def body_increments(q) -> np.ndarray:
    """Row increments (T-1, 22) of q with the base translation in the base frame (at the interval's
    mid yaw): Reachy's base limits (0.61 m/s) hold per body axis, as the whole-body controller and
    the tier-P ``base_speed`` gate measure them (world-axis increments of 0.58 m/s each are 0.82 m/s
    along a diagonal heading)."""
    q = np.asarray(q, float)
    dq = np.diff(q, axis=0)
    yaw = (q[:-1, 2] + q[1:, 2]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    dq[:, 0], dq[:, 1] = c * dq[:, 0] + s * dq[:, 1], -s * dq[:, 0] + c * dq[:, 1]
    return dq


def slide_positions(src) -> np.ndarray | None:
    """(T, n) source positions of the articulated joints of ``src`` whose MJCF type is ``slide``
    (read from the scene XML, no simulator), or None."""
    if src.scene is None or not src.articulations:
        return None
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(src.scene.mjcf)
    except ET.ParseError:
        return None
    kinds = {j.get("name"): j.get("type", "hinge") for j in root.iter("joint") if j.get("name")}
    cols = [a.qpos[:, i] for a in src.articulations.values() for i, n in enumerate(a.joint_names)
            if kinds.get(n) == "slide"]
    return np.stack(cols, axis=1) if cols else None


def grasp_events(labels) -> list[tuple[str, int]]:
    """[("start" | "end", source row)] of the grasp segments of label arrays {side: (T,)}:
    the first held row of a segment and the first row after it."""
    out = []
    for lab in labels:
        held = np.asarray(lab) >= 0
        d = np.diff(np.r_[0, held.astype(int), 0])
        out += [("start", int(i)) for i in np.flatnonzero(d == 1) if i > 0]
        out += [("end", int(i)) for i in np.flatnonzero(d == -1) if i < len(held)]
    return out


def clock(times, q, cfg: RetargetConfig, grasp_events=None, slides=None) -> Clock:
    """Output clock for source times (T,) and source-rate joint values q (T, 22)
    (``grasp_events``, ``slides``: see :func:`dilation`)."""
    times = np.asarray(times, float)
    d = dilation(times, q, cfg, grasp_events, slides)
    warped = np.r_[0.0, np.cumsum(np.diff(times) * d)]
    n = int(np.ceil(warped[-1] / DT - 1e-9)) + 1
    t = np.arange(n) * DT
    w = np.minimum(t, warped[-1])
    index = np.clip(np.searchsorted(warped, w, side="right") - 1, 0, len(times) - 2)
    fraction = np.clip((w - warped[index]) / (warped[index + 1] - warped[index]), 0.0, 1.0)
    source = times[index] + fraction * (times[index + 1] - times[index])
    return Clock(t, np.maximum.accumulate(source), index, fraction, d)


def linear(c: Clock, values):
    """Linear resampling of per-row values (T, ...) onto the output clock."""
    values = np.asarray(values, float)
    u = c.fraction.reshape(-1, *([1] * (values.ndim - 1)))
    a = values[c.index]
    return a + u * (values[c.index + 1] - a)


def poses(c: Clock, X):
    """Resample poses (T, 4, 4): linear translation, geodesic (slerp) rotation."""
    X = np.asarray(X, float)
    A, B = X[c.index], X[c.index + 1]
    rel = so3_log(np.swapaxes(A[:, :3, :3], 1, 2) @ B[:, :3, :3])
    out = np.zeros((len(c.time), 4, 4))
    out[:, :3, :3] = A[:, :3, :3] @ so3_exp(c.fraction[:, None] * rel)
    out[:, :3, 3] = linear(c, X[:, :3, 3])
    out[:, 3, 3] = 1.0
    return out


def row_valid(c: Clock, valid):
    """Output validity: every source row the output row interpolates from must be valid."""
    valid = np.asarray(valid, bool)
    left = valid[c.index] | (c.fraction >= 1.0)
    right = valid[c.index + 1] | (c.fraction <= 0.0)
    return left & right


def labels(c: Clock, lab):
    """Integer labels: kept where both neighbouring source rows agree (or the row is exact), else -1."""
    lab = np.asarray(lab)
    a, b = lab[c.index], lab[c.index + 1]
    return np.where(c.fraction <= 0.0, a, np.where(c.fraction >= 1.0, b, np.where(a == b, a, -1)))


def object_track(c: Clock, track: ObjectTrack) -> ObjectTrack:
    """Resample an object track; invalid output rows are NaN (no interpolation across them)."""
    valid = row_valid(c, track.valid)
    pose = np.full((len(c.time), 7), np.nan)
    if valid.any():
        X = np.zeros((len(track.pose), 4, 4))
        ok = track.valid
        X[:] = np.eye(4)
        X[ok, :3, :3] = quat_to_matrix(track.pose[ok, 3:])
        X[ok, :3, 3] = track.pose[ok, :3]
        sub = Clock(c.time[valid], c.source_time[valid], c.index[valid], c.fraction[valid], c.dilation)
        pose[valid] = pose_to_vec7(poses(sub, X))
    return ObjectTrack(pose, valid, track.role, dict(track.geometry))


def velocity(q, dt=DT):
    """Central finite differences (one-sided at the ends): (N, 22) rates of q (N, 22).

    Central differences average two neighbouring interval speeds, so they never exceed
    the larger of them."""
    return np.gradient(np.asarray(q, float), dt, axis=0)
