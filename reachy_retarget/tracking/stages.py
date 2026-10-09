"""Cut a recorded demonstration into single stages (docs/tracking.md, "Source paths").

A stage is a time window in which every body part makes at most one movement toward one goal, the
rule of the synthetic scenarios. Recorded demonstrations chain several (approach, close the gripper,
carry, open, retreat), so they are cut:

1. **Movements per part** (:func:`movements`): a part moves while its smoothed speed is above a
   threshold (hand: 2 cm/s or 0.15 rad/s relative to the base; base: 2 cm/s or 0.05 rad/s; gripper:
   0.2 opening/s). Pauses shorter than ``join`` (0.3 s) do not split a movement, and movements
   shorter than ``min_duration`` (0.15 s) that travel less than ``min_travel`` are noise. A movement
   that turns back without stopping (a slow-down below 30 % of its peak speed where the direction
   changes by more than 90°, e.g. a reach and return) is two movements: one goal each.
2. **Stages** (:func:`cuts`): scanning movements by start time, a new stage begins when a part starts
   a second movement within the current stage. The cut lies within ``lead`` (0.5 s) before that
   movement (after the part's previous one), where the other parts move least (within 20 % of the
   window's activity range of the least; ties: the latest).

A movement of another part that is running at a cut continues in the next stage, where it counts as
that part's one movement, unless it ends within ``tail`` (0.3 s): demonstrations overlap their actions
(the hand still settles as the gripper starts to close), and such a settling tail is not an action.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..schema.rotations import so3_log


@dataclass(frozen=True)
class Thresholds:
    lin: float = 0.02          # m/s
    ang: float = 0.15          # rad/s (hands); the base uses base_ang
    base_ang: float = 0.05     # rad/s
    grip: float = 0.2          # opening units / s
    grip_travel: float = 0.1   # opening units: smaller gripper changes are noise
    join: float = 0.3          # s: shorter pauses do not split a movement
    min_duration: float = 0.15  # s
    min_travel: float = 0.01   # m (or rad / opening units): shorter, slower movements are noise
    smooth: float = 0.1        # s: moving-average window of the speeds
    turn_speed: float = 0.3    # fraction of the peak speed below which a direction change splits
    turn_angle: float = 90.0   # degrees
    lead: float = 0.5          # s: a cut lies at most this long before the movement that forces it
    tail: float = 0.3          # s: a movement ending this soon after a cut does not count in the new stage


def _smooth(x, times, window):
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    n = max(1, int(round(window / dt)))
    return uniform_filter1d(x, n, axis=0, mode="nearest") if n > 1 else x


def _runs(moving, times, th: Thresholds, travel):
    """Index runs [start, end) of ``moving``, joined across short pauses, without noise runs."""
    d = np.diff(np.r_[0, moving.astype(int), 0])
    runs = [[int(a), int(b)] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    joined = []
    for r in runs:
        if joined and times[min(r[0], len(times) - 1)] - times[joined[-1][1] - 1] < th.join:
            joined[-1][1] = r[1]
        else:
            joined.append(r)
    return [(a, b) for a, b in joined
            if times[b - 1] - times[a] >= th.min_duration or travel(a, b) >= th.min_travel]


def _split_turns(runs, velocity, speed, times, th: Thresholds):
    """Split runs where the motion slows down and changes direction (one goal per movement)."""
    out = []
    window = max(1, int(round(0.15 / max(float(np.median(np.diff(times))), 1e-6))))
    for a, b in runs:
        start = a
        peak = float(speed[a:b].max()) if b > a else 0.0
        for k in range(a + 1, b - 1):
            if not (speed[k] <= speed[k - 1] and speed[k] <= speed[k + 1] and speed[k] < th.turn_speed * peak):
                continue
            before = velocity[max(start, k - window):k].mean(axis=0)
            after = velocity[k + 1:min(b, k + 1 + window)].mean(axis=0)
            nb, na = np.linalg.norm(before), np.linalg.norm(after)
            if nb < 1e-9 or na < 1e-9:
                continue
            angle = np.degrees(np.arccos(np.clip(before @ after / (nb * na), -1, 1)))
            if (angle > th.turn_angle and times[k] - times[start] >= th.min_duration
                    and times[b - 1] - times[k] >= th.min_duration):
                out.append((start, k))
                start = k
        out.append((start, b))
    return out


def movements(times, *, pose=None, planar=None, scalar=None, th: Thresholds = Thresholds()):
    """Movements [(start, end)] (row indices, end exclusive) of one part.

    ``pose``: (T, 4, 4) hand poses relative to the base; ``planar``: (T, 3) base x, y, yaw;
    ``scalar``: (T,) gripper opening."""
    times = np.asarray(times, float)
    if pose is not None:
        p = pose[:, :3, 3]
        v = _smooth(np.gradient(p, times, axis=0), times, th.smooth)
        rel = so3_log(np.swapaxes(pose[:-1, :3, :3], 1, 2) @ pose[1:, :3, :3])
        w = np.r_[np.linalg.norm(rel, axis=1) / np.diff(times), 0.0]
        w = _smooth(w, times, th.smooth)
        speed = np.linalg.norm(v, axis=1)
        moving = (speed > th.lin) | (w > th.ang)

        def travel(a, b):
            return max(np.linalg.norm(p[b - 1] - p[a]), float(np.linalg.norm(so3_log(pose[a, :3, :3].T @ pose[b - 1, :3, :3]))))
        runs = _runs(moving, times, th, travel)
        return _split_turns(runs, v, speed, times, th)
    if planar is not None:
        b_ = np.asarray(planar, float)
        v = _smooth(np.gradient(b_[:, :2], times, axis=0), times, th.smooth)
        w = _smooth(np.abs(np.gradient(np.unwrap(b_[:, 2]), times)), times, th.smooth)
        speed = np.linalg.norm(v, axis=1)
        moving = (speed > th.lin) | (w > th.base_ang)

        def travel(a, b):
            return max(np.linalg.norm(b_[b - 1, :2] - b_[a, :2]), abs(b_[b - 1, 2] - b_[a, 2]))
        runs = _runs(moving, times, th, travel)
        return _split_turns(runs, v, speed, times, th)
    s = np.asarray(scalar, float)
    rate = _smooth(np.gradient(s, times), times, th.smooth)
    moving = np.abs(rate) > th.grip
    runs = _runs(moving, times, th, lambda a, b: abs(s[b - 1] - s[a]))
    runs = [(a, b) for a, b in runs if np.ptp(s[a:b]) >= th.grip_travel]
    # an opening that reverses (close then open) is two gripper actions
    return _split_turns(runs, rate[:, None], np.abs(rate), times, th)


def cuts(times, parts: dict[str, list[tuple[int, int]]], activity: np.ndarray | None = None,
         th: Thresholds = Thresholds()) -> list[int]:
    """Stage boundaries (row indices, first = 0, last = T - 1) such that every part starts at most one
    movement per stage. A new stage begins shortly before a part's second movement: at the instant of
    least ``activity`` (T,) within ``th.lead`` seconds before it (within 20 % of the window's range of the
    least; the latest such instant). A movement still
    running at a cut continues in the new stage, where it counts as that part's movement unless it ends
    within ``th.tail`` seconds (a settling tail)."""
    times = np.asarray(times, float)
    T = len(times)
    activity = np.zeros(T) if activity is None else np.asarray(activity, float)
    events = sorted((a, b, name) for name, runs in parts.items() for a, b in runs)
    bounds = [0]
    used: dict[str, tuple[int, int]] = {}
    for a, b, name in events:
        if name in used and a > bounds[-1]:
            lo = max(used[name][1] - 1, bounds[-1] + 1, int(np.searchsorted(times, times[a] - th.lead)))
            window = np.arange(min(lo, a), a + 1)
            act = activity[window]
            best = window[np.flatnonzero(act <= act.min() + 0.2 * np.ptp(act) + 1e-9)]
            cut = int(best[-1])
            bounds.append(cut)
            used = {n: r for n, r in used.items()
                    if n != name and r[1] > cut and times[min(r[1], T) - 1] - times[cut] > th.tail}
        used[name] = (a, b)
    bounds.append(T - 1)
    return sorted(set(bounds))


def activity(times, parts_signals) -> np.ndarray:
    """Sum over parts of the normalized speed (T,), for choosing cut instants."""
    total = np.zeros(len(times))
    for speed, scale in parts_signals:
        total += np.asarray(speed, float) / scale
    return total
