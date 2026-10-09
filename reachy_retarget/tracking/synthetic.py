"""Procedural tracking scenarios (docs/tracking.md, "Synthetic scenarios").

Every scenario is one independent seed (``<namespace>/<split>/<index>``, hashed as in reachy-control
``rl_tracking/data.py``) and samples these axes independently:

* body-usage **cell** (:data:`CELLS`; stratified: every block of ``len(CELLS)`` consecutive
  indices holds each cell once, in a seeded order),
* **neck mode** (:data:`NECK_MODES`; ``off`` = no head reference),
* **length** bucket (:data:`LENGTHS`), **speed** class (:data:`SPEEDS`), **start** posture
  (:data:`STARTS`), **workspace region** per hand (:data:`REGIONS`), **gripper** style per hand
  (:data:`GRIPPER_STYLES`), and the world start pose of the base.

Hand keyframes are postures from a joint-space random walk (each keyframe is reachable and clear of
self-collision on its own); task-space primitives (:mod:`.primitives`) run between them. ``witness``
cells instead follow a smooth joint/base path whose forward kinematics is the reference (feasible
by construction, as in ``rl_tracking``). Head references are always the rotation of a neck path
inside the limits and the neck speed relative to the nominal base, so they are reachable whenever
the base follows its nominal path.

:func:`generate` builds a :class:`~.reference.TrackingReference` for one scenario and one
:class:`Attempt` (``size`` scales motion magnitudes, ``time`` stretches every duration); the
retry policy lives in :mod:`.build`.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..robot import LOWER, UPPER, VELOCITY, Reachy, gripper, min_clearance, planar
from ..robot.reachy import GRIPPERS, LEFT_ARM, NECK, RIGHT_ARM
from ..robot.resources import profile, urdf
from ..schema.episode import DT, SIDES
from ..validate.kinematic import REACH_Z
from . import neck as neck_mod
from . import primitives as pr
from .reference import TrackingReference, jsonable as _jsonable, motion_parts, regime_of

GENERATOR = "tracking.synthetic/v1"
DATASET = "tracking/synthetic-v1"

STATIONARY = ("hold", "left", "right", "bimanual_independent", "bimanual_symmetric", "bimanual_rigid",
              "handover", "neck_only", "witness_arms")
MOBILE = ("navigation", "mobile_carry", "mobile_reach", "mobile_world_hold", "mobile_free",
          "witness_whole_body")
CELLS = STATIONARY + MOBILE
NECK_MODES = ("off", "base_hold", "world_hold", "look_at_point", "look_at_hand", "scan", "look_ahead",
              "random")
NECK_WEIGHTS = (0.25, 0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1)
LENGTHS = {"short": (2.0, 6.0), "medium": (6.0, 20.0), "long": (20.0, 60.0)}
LENGTH_WEIGHTS = (0.4, 0.4, 0.2)
# Linear/angular hand speeds, base cruise (m/s), base yaw rate (rad/s) and witness knot spacing (s).
SPEEDS = {
    "slow": dict(lin=(0.06, 0.12), ang=(0.25, 0.5), base=(0.2, 0.3), yaw=(0.3, 0.5), knot=(2.5, 4.0)),
    "normal": dict(lin=(0.12, 0.25), ang=(0.5, 0.9), base=(0.3, 0.42), yaw=(0.5, 0.9), knot=(1.5, 2.5)),
    "fast": dict(lin=(0.25, 0.4), ang=(0.9, 1.3), base=(0.42, 0.55), yaw=(0.9, 1.4), knot=(0.9, 1.5)),
}
SPEED_WEIGHTS = (0.3, 0.45, 0.25)
STARTS = ("home", "ready", "rest", "random")
START_WEIGHTS = (0.3, 0.2, 0.15, 0.35)
REGIONS = ("any", "low", "mid", "high", "front", "side", "cross")
REGION_WEIGHTS = (0.3, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1)
GRIPPER_STYLES = ("open", "closed", "toggle", "cycle", "partial")
GRIPPER_WEIGHTS = (0.15, 0.15, 0.35, 0.15, 0.2)
HAND_PRIMITIVES = ("line", "circle", "raster", "press", "hinge", "twist", "pour", "oscillate", "wander")

SHOULDER = {s: urdf().transform("base_link", f"{s[0]}_shoulder_ball_link")[:3, 3] for s in SIDES}
TCP_REACH = 0.665       # m, TCP to shoulder ball (sampled maximum 0.66, the straight arm)
KEY_MARGIN = 0.12       # rad inside the joint limits for keyframes and witnesses
KEY_CLEARANCE = 0.025   # m self-clearance of keyframes
WITNESS_CLEARANCE = 0.015
NECK_GEN_MARGIN = 0.07  # rad inside the neck limits for generated head references
SPEED_FRACTION = 0.85   # witness joint and base speeds stay below this fraction of the limits
ARM_STEP = np.array([0.6, 0.4, 0.6, 0.6, 0.35, 0.35, 0.8])  # keyframe random walk, rad per joint
FINGER_RATE = 0.9 * VELOCITY[GRIPPERS.start] / (gripper.UPPER - gripper.LOWER)  # opening units / s


@dataclass(frozen=True)
class Scenario:
    namespace: str
    split: str
    index: int
    cell: str
    neck: str
    length: str
    speed: str
    start: str
    regions: tuple[str, str]
    grippers: tuple[str, str]
    base_start: tuple[float, float, float]

    @property
    def source_id(self) -> str:
        return f"{self.namespace}/{self.split}/{self.index:06d}"

    @property
    def seed(self) -> int:
        return seed_of(self.source_id)

    @property
    def episode_id(self) -> str:
        return f"{self.namespace}-{self.split}-{self.index:06d}"


@dataclass(frozen=True)
class Attempt:
    size: float = 1.0   # motion magnitudes (keyframe steps, primitive extents, small base moves)
    time: float = 1.0   # every duration


def seed_of(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def _choice(rng, options, weights):
    w = np.asarray(weights, float)
    return options[int(rng.choice(len(options), p=w / w.sum()))]


def scenario(namespace: str, split: str, index: int) -> Scenario:
    """The deterministic scenario of one index (stratified cells, independent other axes)."""
    block, pos = divmod(index, len(CELLS))
    order = np.random.default_rng(seed_of(f"{namespace}/{split}/cells/{block}")).permutation(len(CELLS))
    cell = CELLS[order[pos]]
    rng = np.random.default_rng(seed_of(f"{namespace}/{split}/{index:06d}/axes"))
    neck = _choice(rng, NECK_MODES, NECK_WEIGHTS)
    mobile = cell in MOBILE
    if neck == "look_ahead" and not mobile:
        neck = "look_at_point"
    if cell == "neck_only" and neck in ("off", "base_hold"):
        neck = _choice(rng, ("look_at_point", "scan", "random", "world_hold"), (1, 1, 1, 0.5))
    length = _choice(rng, tuple(LENGTHS), LENGTH_WEIGHTS)
    speed = _choice(rng, tuple(SPEEDS), SPEED_WEIGHTS)
    start = _choice(rng, STARTS, START_WEIGHTS)
    if cell in ("navigation", "mobile_carry", "bimanual_rigid") and start == "rest":
        start = "home" if cell != "navigation" else "rest"
    regions = tuple(_choice(rng, REGIONS, REGION_WEIGHTS) for _ in SIDES)
    if cell in ("bimanual_symmetric", "handover"):
        regions = tuple("any" if r == "cross" else r for r in regions)
    grippers = tuple(_choice(rng, GRIPPER_STYLES, GRIPPER_WEIGHTS) for _ in SIDES)
    base_start = (float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)), float(rng.uniform(-np.pi, np.pi)))
    return Scenario(namespace, split, index, cell, neck, length, speed, start, regions, grippers, base_start)


# ---------------------------------------------------------------------------- postures


def _profile_posture(name):
    p = profile()["postures"][name]
    q = np.zeros(22)
    q[LEFT_ARM], q[RIGHT_ARM] = np.radians(p["left"]), np.radians(p["right"])
    return q


def _clip_arm(q_arm, side):
    sl = LEFT_ARM if side == "left" else RIGHT_ARM
    return np.clip(q_arm, LOWER[sl] + KEY_MARGIN, UPPER[sl] - KEY_MARGIN)


class Context:
    """Per-attempt generation state: rng, speeds, scale factors, robot model."""

    def __init__(self, sc: Scenario, attempt: Attempt):
        self.sc, self.attempt = sc, attempt
        self.rng = np.random.default_rng(sc.seed)
        self.robot = Reachy.load()
        self.speed = {k: float(self.rng.uniform(*v)) for k, v in SPEEDS[sc.speed].items()}
        for k in ("lin", "ang", "base", "yaw"):
            self.speed[k] /= attempt.time
        self.speed["knot"] *= attempt.time
        self.size = attempt.size
        self.target = float(self.rng.uniform(*LENGTHS[sc.length]))
        self.params: dict = {"speed": dict(self.speed), "target_duration_s": self.target}

    def uniform(self, lo, hi, size=None):
        return self.rng.uniform(lo, hi, size)

    def dwell(self):
        return float(self.rng.uniform(0.1, 0.5)) * self.attempt.time


def start_posture(ctx: Context) -> np.ndarray:
    """q (22,) at the origin of the base frame (the base pose is added by the caller)."""
    sc, rng = ctx.sc, ctx.rng
    kind = sc.start
    if kind == "home":  # reachy-control experiment start (scenarios.initial_configuration)
        q = np.zeros(22)
        for side, sl, sign in (("left", LEFT_ARM, 1.0), ("right", RIGHT_ARM, -1.0)):
            q[sl] = [rng.uniform(-0.5, -0.34), sign * 0.10, sign * 0.10, rng.uniform(-1.4, -1.3), 0, 0, 0]
    elif kind in ("ready", "rest"):
        q = _profile_posture(kind)
    else:
        q = _profile_posture("ready")
        for _ in range(200):
            cand = q.copy()
            for side, sl in (("left", LEFT_ARM), ("right", RIGHT_ARM)):
                cand[sl] = _clip_arm(q[sl] + rng.uniform(-1, 1, 7) * ARM_STEP * 1.3, side)
            if sc.cell == "bimanual_symmetric":
                cand[RIGHT_ARM] = cand[LEFT_ARM] * pr.MIRROR_SIGNS
            fk = ctx.robot.fk_base(cand)
            z = [fk[f"{s}_grasp"][2, 3] for s in SIDES]
            if min(z) > 0.65 and max(z) < 1.45 and min_clearance(cand) > 0.03:
                q = cand
                break
    q[NECK] = 0.0
    return q


# ---------------------------------------------------------------------------- keyframes


def _region_ok(p, side, region):
    """Region predicate on TCP positions (N, 3) in the base frame."""
    sign = 1.0 if side == "left" else -1.0
    x, y, z = p[:, 0], p[:, 1] * sign, p[:, 2]
    ok = {"any": np.ones(len(p), bool), "low": z < 0.85, "mid": (z >= 0.85) & (z <= 1.15), "high": z > 1.15,
          "front": x > 0.42, "side": y > 0.38, "cross": y < -0.02}[region]
    return ok & (x > 0.12)


def keyframe(ctx: Context, side: str, q: np.ndarray, region: str, min_move=0.05, n=192, extra=None):
    """A reachable, self-clear arm posture near ``q``'s arm (joint random walk), biased to a region.

    ``q``: (22,) full posture in the base frame (the other arm and the neck are kept for the
    clearance check). ``extra``: optional predicate on the TCP (N, 4, 4) poses. Returns (q', pose)."""
    rng = ctx.rng
    sl = LEFT_ARM if side == "left" else RIGHT_ARM
    cur = ctx.robot.fk_base(q)[f"{side}_tcp"]
    for relax in range(4):
        cand = np.tile(q, (n, 1))
        step = ARM_STEP * ctx.size * (1.0 + 0.3 * relax)
        cand[:, sl] = _clip_arm(q[sl] + rng.uniform(-1, 1, (n, 7)) * step, side)
        fk = ctx.robot.fk_base(cand)
        P, G = fk[f"{side}_tcp"], fk[f"{side}_grasp"]
        ok = (G[:, 2, 3] > REACH_Z[0] + 0.12) & (G[:, 2, 3] < 1.6)
        ok &= np.linalg.norm(P[:, :3, 3] - cur[:3, 3], axis=1) > min_move * ctx.size
        ok &= _region_ok(P[:, :3, 3], side, region if relax < 2 else "any")
        if extra is not None:
            ok &= extra(P)
        idx = np.flatnonzero(ok)
        if len(idx):
            idx = idx[rng.permutation(len(idx))[:24]]
            clear = min_clearance(cand[idx])
            good = idx[clear > KEY_CLEARANCE]
            if len(good):
                k = int(good[0])
                return cand[k], P[k]
    return q.copy(), cur


# ---------------------------------------------------------------------------- hand chains


def primitive(ctx: Context, P, kind, wrist_yaw=0.0):
    """One task-space primitive at pose P (4x4, in the chain's frame; z is up in every frame).
    ``wrist_yaw``: the current wrist yaw (a twist about the TCP z axis is exactly a wrist-yaw turn)."""
    rng, s, sp = ctx.rng, ctx.size, ctx.speed
    if kind == "line":
        return pr.line(P, pr.random_unit(rng), rng.uniform(0.04, 0.18) * s, sp, back=rng.random() < 0.7)
    if kind == "circle":
        n = [0, 0, 1] if rng.random() < 0.4 else pr.random_unit(rng)
        return pr.circle(P, rng.uniform(0.025, 0.09) * s, n, int(rng.integers(1, 3)), sp)
    if kind == "raster":
        a = pr.random_unit(rng, horizontal=True)
        b = np.cross([0, 0, 1], a) if rng.random() < 0.5 else np.array([0.0, 0.0, -1.0])
        return pr.raster(P, a, b, rng.uniform(0.06, 0.16) * s, rng.uniform(0.03, 0.1) * s,
                         int(rng.integers(2, 5)), sp)
    if kind == "press":
        return pr.press(P, rng.uniform(0.02, 0.07) * s, int(rng.integers(1, 4)), sp)
    if kind == "hinge":
        axis = [0, 0, 1] if rng.random() < 0.6 else pr.random_unit(rng, horizontal=True)
        r = rng.uniform(0.15, 0.4)
        off = pr.unit(np.cross(axis, pr.random_unit(rng))) * r
        return pr.hinge(P, off, axis, rng.choice([-1, 1]) * rng.uniform(0.15, 0.45) * s, sp,
                        back=rng.random() < 0.5)
    if kind == "twist":
        sign = rng.choice([-1, 1])
        room = (UPPER[9] - KEY_MARGIN - wrist_yaw) if sign > 0 else (wrist_yaw - LOWER[9] - KEY_MARGIN)
        return pr.twist(P, sign * min(rng.uniform(0.3, 1.0) * s, max(room, 0.05)), sp, back=True)
    if kind == "pour":
        gc = (P @ ctx.robot.grasp_center("left"))[:3, 3]
        return pr.pour(P, gc, pr.random_unit(rng, horizontal=True), rng.uniform(0.3, 0.9) * s, sp,
                       hold_s=rng.uniform(0.0, 1.0))
    if kind == "oscillate":
        return pr.oscillate(P, pr.random_unit(rng) * rng.uniform(0.01, 0.04) * s,
                            pr.random_unit(rng) * rng.uniform(0.03, 0.15) * s,
                            rng.uniform(0.4, 1.0) / ctx.attempt.time, int(rng.integers(2, 6)))
    if kind == "wander":
        return pr.wander(P, rng, rng.uniform(0.02, 0.08) * s, rng.uniform(0.05, 0.3) * s,
                         int(rng.integers(2, 5)), sp)
    raise ValueError(kind)


def hand_chain(ctx: Context, side, q, until, region, *, anchor=None, start_track=None, primitives=True,
               extra=None):
    """Keyframe reaches and primitives until ``until`` seconds. ``q`` (22,) is the base-frame
    posture at the start; ``anchor`` (4x4) maps base-frame poses into the chain's frame (identity
    for base-relative chains). Returns (Track, final q, list of primitive kinds)."""
    rng = ctx.rng
    A = np.eye(4) if anchor is None else anchor
    sl = LEFT_ARM if side == "left" else RIGHT_ARM
    track = start_track or pr.Track(A @ ctx.robot.fk_base(q)[f"{side}_tcp"])
    kinds = []
    while track.t < until:
        q_new, P_new = keyframe(ctx, side, q, region if rng.random() < 0.75 else "any", extra=extra)
        dur, fn = pr.reach(track.end, A @ P_new, ctx.speed)
        if track.t + dur > until + 4.0 and track.t > 0:
            break
        track.add(dur, fn, "reach")
        kinds.append("reach")
        q = q_new
        track.hold(ctx.dwell())
        if primitives and rng.random() < 0.7:
            kind = str(rng.choice(HAND_PRIMITIVES))
            wy = q[sl.stop - 1]
            dur, fn = primitive(ctx, track.end, kind, wy)
            if track.t + dur <= until + 4.0:
                track.add(dur, fn, kind)
                kinds.append(kind)
                track.hold(ctx.dwell())
    return track, q, kinds


# ---------------------------------------------------------------------------- base paths


def base_path(ctx: Context, start, until, *, small=False, kinds=None):
    """Chained SE(2) segments until ``until`` seconds (``small``: short repositioning moves)."""
    rng, sp = ctx.rng, ctx.speed
    path = pr.BasePath(start)
    used = []
    options = kinds or ("drive", "arc", "turn", "spline", "strafe")
    while path.t < until or not path.segments:
        kind = str(rng.choice(options))
        b = path.end
        if small:  # out from the start pose and back again
            if len(used) % 2 == 0:
                dur, fn = pr.drive(b, rng.uniform(0.05, 0.3) * ctx.size, rng.uniform(-np.pi, np.pi),
                                   rng.uniform(-0.35, 0.35) * ctx.size, sp)
            else:
                d = path.start[:2] - b[:2]
                dur, fn = pr.drive(b, np.linalg.norm(d), np.arctan2(d[1], d[0]) - b[2], path.start[2] - b[2], sp)
            kind = "reposition"
        elif kind == "drive":
            dur, fn = pr.drive(b, rng.uniform(0.3, 2.0), rng.choice([0.0, 0.0, np.pi, rng.uniform(-np.pi, np.pi)]),
                               rng.uniform(-0.6, 0.6), sp)
        elif kind == "strafe":
            dur, fn = pr.drive(b, rng.uniform(0.2, 1.0), rng.choice([-np.pi / 2, np.pi / 2]), 0.0, sp)
        elif kind == "arc":
            dur, fn = pr.arc(b, rng.choice([-1, 1]) * rng.uniform(0.5, 2.0), rng.uniform(0.4, 2.0), sp)
        elif kind == "turn":
            dur, fn = pr.turn(b, rng.choice([-1, 1]) * rng.uniform(0.4, np.pi), sp)
        else:
            heading = b[2]
            pts, p = [], b[:2].copy()
            for _ in range(int(rng.integers(2, 5))):
                heading += rng.uniform(-1.2, 1.2)
                p = p + rng.uniform(0.5, 1.5) * np.array([np.cos(heading), np.sin(heading)])
                pts.append(p.copy())
            mode = "tangent" if rng.random() < 0.7 else "fixed"
            dur, fn = pr.spline(b, pts, sp, mode)
            if fn is None:
                continue
            if mode == "tangent":  # turn toward the spline's initial tangent first
                dyaw = float(fn(np.array([0.0]))[0, 2] - b[2])
                if abs(dyaw) > 1e-9:
                    d_turn, f_turn = pr.turn(b, dyaw, sp)
                    path.add(_fit_base(d_turn, f_turn), f_turn, "turn")
                    used.append("turn")
                    dur, fn = pr.spline(path.end, pts, sp, mode)
        dur = _fit_base(dur, fn)
        if path.t + dur > until + 6.0 and path.segments:
            break
        path.add(dur, fn, kind)
        used.append(kind)
        if rng.random() < 0.5:
            path.hold(rng.uniform(0.2, 1.0) * ctx.attempt.time)
    return path, used


def _fit_base(dur, fn):
    """Stretch a base segment's duration until its body-frame speeds stay within SPEED_FRACTION of
    the limits (tangent-heading splines turn fast in tight bends)."""
    for _ in range(6):
        u = np.linspace(0.0, 1.0, max(int(np.ceil(dur / DT)), 2) + 1)
        ratio = pr.body_speed_ratio(fn(u), dur / (len(u) - 1), VELOCITY[:3]) / SPEED_FRACTION
        if ratio <= 1.0:
            break
        dur *= ratio * 1.02
    return dur


# ---------------------------------------------------------------------------- witnesses


def _pchip(knot_t, values, times):
    return PchipInterpolator(knot_t, values, axis=0)(np.clip(times, knot_t[0], knot_t[-1]))


def joint_witness(ctx: Context, q0, cols, until, tries: int = 40):
    """Smooth joint path (PCHIP through random-walk knots, no overshoot) of the columns ``cols``
    starting at ``q0``, inside the limits minus KEY_MARGIN, below SPEED_FRACTION of the speed limits
    and with self-clearance >= WITNESS_CLEARANCE (checked every 0.1 s; knots are redrawn, and the
    random-walk step shrinks every 10 failed draws). Returns (knot times, knot values (K, len(cols)))."""
    rng = ctx.rng
    k = max(2, int(np.ceil(until / ctx.speed["knot"])))
    lo, hi = LOWER[cols] + KEY_MARGIN, UPPER[cols] - KEY_MARGIN
    step = np.array([ARM_STEP[(c - 3) % 7] if c < 17 else 0.35 for c in cols])
    vmax = VELOCITY[cols] * SPEED_FRACTION
    best = None
    for attempt in range(tries):
        scale = ctx.size * 0.8 ** (attempt // 10)
        knot_t = np.r_[0.0, np.cumsum(rng.uniform(0.7, 1.3, k) * ctx.speed["knot"])]
        vals = [q0[cols]]
        for _ in range(k):
            vals.append(np.clip(vals[-1] + rng.uniform(-1, 1, len(cols)) * step * scale, lo, hi))
        vals = np.array(vals)
        # stretch knot intervals whose joint speed would exceed SPEED_FRACTION of the limits
        dt_need = np.max(np.abs(np.diff(vals, axis=0)) / vmax * 1.5, axis=1)  # PCHIP peaks ~1.5x the mean
        knot_t = np.r_[0.0, np.cumsum(np.maximum(np.diff(knot_t), dt_need))]
        t = np.arange(0.0, knot_t[-1] + 1e-9, 0.1)
        q = np.tile(q0, (len(t), 1))
        q[:, cols] = _pchip(knot_t, vals, t)
        clear = float(min_clearance(q).min())
        if best is None or clear > best[0]:
            best = (clear, knot_t, vals)
        if clear >= WITNESS_CLEARANCE:
            break
    ctx.params["witness_clearance_m"] = best[0]
    return best[1], best[2]


# ---------------------------------------------------------------------------- neck references


def neck_path(ctx: Context, mode, times, base, tcp_world, q_start, active_sides):
    """Neck angles (T, 3) of the head reference (None for ``off``), inside the limits minus
    NECK_GEN_MARGIN and the neck speed (relative to the nominal ``base``)."""
    rng = ctx.rng
    start = q_start[NECK]
    dt = np.diff(times)
    lo, hi = neck_mod.limits(NECK_GEN_MARGIN)
    T = len(times)
    W = planar(base[:, 0], base[:, 1], base[:, 2])
    if mode == "off":
        return None
    if mode == "base_hold":
        target = rng.uniform(lo, hi) * 0.7
        return neck_mod.rate_limit(np.tile(target, (T, 1)), start, dt, 0.8)
    if mode == "random":
        span = times[-1]
        k = max(2, int(np.ceil(span / max(ctx.speed["knot"], 1.0))))
        kt = np.r_[0.0, np.sort(rng.uniform(0, span, k - 1)), span]
        vals = np.r_[[start], rng.uniform(lo, hi, (k, 3))]
        return neck_mod.rate_limit(_pchip(kt, vals, times), start, dt, 0.8)
    if mode == "world_hold":
        R0 = W[0, :3, :3] @ neck_mod.head_base(start)[:3, :3]
        q = np.zeros((T, 22))
        q[:, :3] = base
        q[:, NECK] = start
        n = neck_mod.solve(q, np.broadcast_to(R0, (T, 3, 3)), margin=NECK_GEN_MARGIN)
        return neck_mod.rate_limit(n, start, dt, 0.8)
    if mode in ("look_at_point", "scan", "look_ahead"):
        if mode == "look_at_point":
            k = int(rng.integers(1, 4))
            switch = np.sort(rng.uniform(0, times[-1], k - 1))
            local = np.c_[rng.uniform(0.8, 3.0, k), rng.uniform(-1.2, 1.2, k), rng.uniform(0.3, 1.7, k)]
            world = (W[0] @ np.c_[local, np.ones(k)].T).T[:, :3]
            idx = np.searchsorted(switch, times, side="right")
            pts = world[idx]
        elif mode == "scan":
            period = rng.uniform(3.0, 8.0) * ctx.attempt.time
            ay, az, z0 = rng.uniform(0.3, 1.0), rng.uniform(0.05, 0.3), rng.uniform(0.9, 1.4)
            u = 2 * np.pi * times / period
            local = np.c_[np.full(T, 1.5), ay * np.sin(u), z0 + az * np.sin(2 * u), np.ones(T)]
            pts = np.einsum("tij,tj->ti", W, local)[:, :3]
        else:  # look ahead along the travel direction (heading when the base does not translate)
            v = np.gradient(base[:, :2], times, axis=0)
            from scipy.ndimage import gaussian_filter1d
            v = gaussian_filter1d(v, 10, axis=0, mode="nearest")
            speed = np.linalg.norm(v, axis=1, keepdims=True)
            heading = np.c_[np.cos(base[:, 2]), np.sin(base[:, 2])]
            w = np.clip(speed / 0.1, 0, 1)
            d = w * v / np.maximum(speed, 1e-9) + (1 - w) * heading
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            pts = np.c_[base[:, :2] + 1.5 * d, np.full(T, rng.uniform(0.6, 1.3))]
        return neck_mod.gaze_neck(pts, base, margin=NECK_GEN_MARGIN, start=start, dt=dt)
    if mode == "look_at_hand":
        robot = ctx.robot
        gc = {s: (tcp_world[s] @ robot.grasp_center(s))[:, :3, 3] for s in SIDES}
        sides = active_sides or list(SIDES)
        if len(sides) == 1:
            pts = gc[sides[0]]
        else:
            period = rng.uniform(2.0, 5.0)
            w = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * times / period + rng.uniform(0, 2 * np.pi)))
            from scipy.ndimage import gaussian_filter1d
            w = gaussian_filter1d(w, 12, mode="nearest")
            pts = (1 - w)[:, None] * gc["left"] + w[:, None] * gc["right"]
        return neck_mod.gaze_neck(pts, base, margin=NECK_GEN_MARGIN, start=start, dt=dt)
    raise ValueError(mode)


# ---------------------------------------------------------------------------- grippers


def gripper_levels(ctx: Context, style, events, end):
    """[(t, opening)] for one hand from its style and its track events."""
    rng = ctx.rng
    open_level = float(rng.uniform(0.7, 1.0))
    closed = float(gripper.angle_to_opening(gripper.width_to_angle(rng.uniform(0.0, 0.07))))
    if style == "open":
        return [(0.0, open_level)]
    if style == "closed":
        return [(0.0, closed)]
    if style == "cycle":
        period = rng.uniform(1.5, 4.0) * ctx.attempt.time
        t, state, out = 0.0, rng.random() < 0.5, []
        while t < end:
            out.append((t, open_level if state else closed))
            state = not state
            t += period / 2
        return out
    arrivals = [t for t, k in events if k == "reach_end"] or [end * 0.4]
    state = rng.random() < 0.6
    out = [(0.0, open_level if state else closed)]
    for t in arrivals:
        if style == "partial":
            out.append((t, float(rng.uniform(0.1, 0.9))))
        elif rng.random() < 0.7:
            state = not state
            out.append((t, open_level if state else closed))
    return out


# ---------------------------------------------------------------------------- cells


def _hold(P):
    return pr.Track(P)


def _world(B, X):
    """World poses (T, 4, 4) of base-frame poses X (T, 4, 4) on base path B (T, 3)."""
    return planar(B[:, 0], B[:, 1], B[:, 2]) @ X


def build_cell(ctx: Context, q0):
    """Hand tracks, base path and metadata of the scenario's cell.

    Returns a dict with ``tracks`` {side: (Track, frame)} (frame ``world`` or ``base``), ``base``
    (BasePath), ``active`` sides, ``events`` {side: [...]}, optional ``witness`` (knot times,
    values, columns), optional ``gripper_override`` {side: [(t, opening)]} and ``params``."""
    sc, rng, robot = ctx.sc, ctx.rng, ctx.robot
    cell = sc.cell
    B0 = planar(*sc.base_start)
    fkb = robot.fk_base(q0)
    start_b = {s: fkb[f"{s}_tcp"] for s in SIDES}
    start_w = {s: B0 @ start_b[s] for s in SIDES}
    base = pr.BasePath(np.array(sc.base_start))
    out = {"tracks": {}, "base": base, "active": [], "params": {}, "gripper_override": {}}
    until = ctx.target

    if cell == "hold":
        out["tracks"] = {s: (_hold(start_w[s]), "world") for s in SIDES}
        base.hold(until)
    elif cell in ("left", "right"):
        other = "right" if cell == "left" else "left"
        tr, _, kinds = hand_chain(ctx, cell, q0, until, sc.regions[SIDES.index(cell)], anchor=B0)
        out["tracks"] = {cell: (tr, "world"), other: (_hold(start_w[other]), "world")}
        out["active"], out["params"]["primitives"] = [cell], {cell: kinds}
    elif cell == "bimanual_independent":
        q, kinds = q0.copy(), {}
        for s in SIDES:  # the left chain is planned with the right arm at its start posture
            tr, q_end, kinds[s] = hand_chain(ctx, s, q, until, sc.regions[SIDES.index(s)], anchor=B0)
            out["tracks"][s] = (tr, "world")
        out["active"], out["params"]["primitives"] = list(SIDES), kinds
    elif cell == "bimanual_symmetric":
        tr, _, kinds = hand_chain(ctx, "left", q0, until, sc.regions[0], anchor=np.eye(4),
                                  extra=lambda P: P[:, 1, 3] > 0.12)
        mirrored = pr.Track(pr.mirror(tr.start))
        for t0, dur, fn in tr.segments:
            mirrored.add(dur, (lambda u, fn=fn: pr.mirror(fn(u))), "mirror")
        mirrored.events = list(tr.events)
        out["tracks"] = {"left": (tr, "base"), "right": (mirrored, "base")}
        out["active"], out["params"]["primitives"] = list(SIDES), {"left": kinds, "right": "mirror"}
    elif cell == "bimanual_rigid":
        out.update(_rigid_object(ctx, start_b, until, mobile=False))
        base.hold(until)
    elif cell == "handover":
        out.update(_handover(ctx, q0, B0))
    elif cell == "neck_only":
        out["tracks"] = {s: (_hold(start_w[s]), "world") for s in SIDES}
        base.hold(until)
    elif cell == "navigation":
        path, used = base_path(ctx, sc.base_start, until)
        out["base"] = path
        out["tracks"] = {s: (_hold(start_b[s]), "base") for s in SIDES}
        out["params"]["base_segments"] = used
    elif cell == "mobile_carry":
        path, used = base_path(ctx, sc.base_start, until)
        out["base"] = path
        out["params"]["base_segments"] = used
        if rng.random() < 0.6:
            out.update(_rigid_object(ctx, start_b, path.t, mobile=True))
        else:
            carrier = str(rng.choice(SIDES))
            other = "right" if carrier == "left" else "left"
            tr, _, kinds = hand_chain(ctx, carrier, q0, path.t, "any", anchor=np.eye(4), primitives=False)
            out["tracks"] = {carrier: (tr, "base"), other: (_hold(start_b[other]), "base")}
            out["active"] = [carrier]
            out["gripper_override"][carrier] = [(0.0, float(gripper.angle_to_opening(
                gripper.width_to_angle(rng.uniform(0.01, 0.06)))))]
            out["params"]["carry"] = {"hand": carrier, "primitives": kinds}
    elif cell == "mobile_reach":
        out.update(_mobile_reach(ctx, q0, start_b, until))
    elif cell == "mobile_world_hold":
        out.update(_mobile_world_hold(ctx, q0, start_b, start_w, until))
    elif cell == "mobile_free":
        path, used = base_path(ctx, sc.base_start, until)
        out["base"] = path
        q, kinds = q0.copy(), {}
        for s in SIDES:
            tr, _, kinds[s] = hand_chain(ctx, s, q, path.t, sc.regions[SIDES.index(s)], anchor=np.eye(4))
            out["tracks"][s] = (tr, "base")
        out["active"], out["params"]["base_segments"], out["params"]["primitives"] = list(SIDES), used, kinds
    elif cell in ("witness_arms", "witness_whole_body"):
        sides = list(SIDES) if rng.random() < 0.6 else [str(rng.choice(SIDES))]
        cols = np.concatenate([np.arange(3, 10) if s == "left" else np.arange(10, 17) for s in sides])
        if cell == "witness_whole_body":
            path, used = base_path(ctx, sc.base_start, until)
            out["base"], out["params"]["base_segments"] = path, used
            until = path.t
        else:
            base.hold(until)
        knot_t, vals = joint_witness(ctx, q0, cols, until)
        out["witness"] = (knot_t, vals, cols)
        out["active"] = sides
        out["tracks"] = {}
    else:
        raise ValueError(cell)
    return out


def _rigid_object(ctx: Context, start_b, until, mobile):
    """Both hands hold a virtual box: hand = object frame @ constant grasp; the object frame moves
    by small lifts, slides, yaw turns and tilts (base frame; carried along the base path when
    ``mobile``)."""
    rng, sp = ctx.rng, ctx.speed
    mid = 0.5 * (start_b["left"][:3, 3] + start_b["right"][:3, 3])
    O0 = np.eye(4)
    O0[:3, 3] = mid
    grasp = {s: np.linalg.inv(O0) @ start_b[s] for s in SIDES}
    obj = pr.Track(O0)
    moves = []
    while obj.t < until:
        P = obj.end
        kind = str(rng.choice(("lift", "slide", "yaw", "tilt", "oscillate", "hold")))
        s = ctx.size
        if kind == "lift":
            dur, fn = pr.line(P, [0, 0, rng.choice([-1, 1])], rng.uniform(0.03, 0.15) * s, sp,
                              back=rng.random() < 0.5)
        elif kind == "slide":
            dur, fn = pr.line(P, pr.random_unit(rng, horizontal=True), rng.uniform(0.03, 0.12) * s, sp,
                              back=rng.random() < 0.5)
        elif kind == "yaw":
            dur, fn = pr.twist(P, rng.choice([-1, 1]) * rng.uniform(0.1, 0.4) * s, sp, back=True)
        elif kind == "tilt":
            dur, fn = pr.pour(P, P[:3, 3], pr.random_unit(rng, horizontal=True),
                              rng.uniform(0.1, 0.35) * s, sp, hold_s=rng.uniform(0, 0.8))
        elif kind == "oscillate":
            dur, fn = pr.oscillate(P, pr.random_unit(rng) * rng.uniform(0.01, 0.03) * s,
                                   np.zeros(3), rng.uniform(0.4, 0.9), int(rng.integers(2, 5)))
        else:
            obj.hold(rng.uniform(0.3, 1.5))
            moves.append("hold")
            continue
        if obj.t + dur > until + 4.0 and obj.t > 0:
            break
        obj.add(dur, fn, kind)
        moves.append(kind)
        obj.hold(ctx.dwell())
    tracks = {}
    for s in SIDES:
        tr = pr.Track(O0 @ grasp[s])
        for t0, dur, fn in obj.segments:
            tr.add(dur, (lambda u, fn=fn, G=grasp[s]: fn(u) @ G), "object")
        tr.events = list(obj.events)
        tracks[s] = (tr, "base")
    closed = float(gripper.angle_to_opening(gripper.CONTACT_ANGLE))
    return {"tracks": tracks, "active": list(SIDES), "params": {"rigid_object": moves, "mobile": mobile},
            "gripper_override": {s: [(0.0, closed)] for s in SIDES}}


def _handover(ctx: Context, q0, B0):
    """Giver brings its hand near the midline, the receiver meets it 13-22 cm away (TCP to TCP), closes, the giver
    opens, both retreat to new keyframes."""
    rng = ctx.rng
    giver = str(rng.choice(SIDES))
    taker = "right" if giver == "left" else "left"
    sign = 1.0 if giver == "left" else -1.0
    q_g, P_g = keyframe(ctx, giver, q0, "any", extra=lambda P: (sign * P[:, 1, 3] > -0.02)
                        & (sign * P[:, 1, 3] < 0.12) & (P[:, 0, 3] > 0.3) & (P[:, 2, 3] > 0.8) & (P[:, 2, 3] < 1.25))
    q_mid = q0.copy()
    q_mid[LEFT_ARM if giver == "left" else RIGHT_ARM] = q_g[LEFT_ARM if giver == "left" else RIGHT_ARM]
    near = lambda P: (np.linalg.norm(P[:, :3, 3] - P_g[:3, 3], axis=1) > 0.13) & (  # noqa: E731
        np.linalg.norm(P[:, :3, 3] - P_g[:3, 3], axis=1) < 0.22)
    q_t, P_t = keyframe(ctx, taker, q_mid, "any", extra=near, n=512)
    sp = ctx.speed
    tg = pr.Track(B0 @ ctx.robot.fk_base(q0)[f"{giver}_tcp"])
    tt = pr.Track(B0 @ ctx.robot.fk_base(q0)[f"{taker}_tcp"])
    d, fn = pr.reach(tg.end, B0 @ P_g, sp)
    tg.add(d, fn, "reach")
    tt.hold(d * rng.uniform(0.3, 1.0))
    d2, fn2 = pr.reach(tt.end, B0 @ P_t, sp)
    tt.add(d2, fn2, "reach")
    meet = max(tg.t, tt.t)
    tg.hold(meet - tg.t)
    tt.hold(meet - tt.t)
    close_s = 1.0 / FINGER_RATE
    tg.hold(close_s * 2 + 0.3)
    tt.hold(close_s * 2 + 0.3)
    q_both = q_mid.copy()
    q_both[LEFT_ARM if taker == "left" else RIGHT_ARM] = q_t[LEFT_ARM if taker == "left" else RIGHT_ARM]
    for tr, s, qs in ((tg, giver, q_both), (tt, taker, q_both)):
        q_away, P_away = keyframe(ctx, s, qs, "any")
        d, fn = pr.reach(tr.end, B0 @ P_away, sp)
        tr.add(d, fn, "reach")
    open_l = float(rng.uniform(0.7, 1.0))
    closed = float(gripper.angle_to_opening(gripper.width_to_angle(rng.uniform(0.01, 0.05))))
    levels = {giver: [(0.0, closed), (meet + close_s + 0.2, open_l)],
              taker: [(0.0, open_l), (meet, closed)]}
    return {"tracks": {giver: (tg, "world"), taker: (tt, "world")}, "active": list(SIDES),
            "gripper_override": levels, "params": {"giver": giver, "meet_s": meet}}


def _mobile_reach(ctx: Context, q0, start_b, until):
    """Drive (hands base-relative), stop, manipulate in the world at the stop, repeat."""
    rng = ctx.rng
    sides = list(SIDES) if rng.random() < 0.4 else [str(rng.choice(SIDES))]
    path = pr.BasePath(np.array(ctx.sc.base_start))
    tracks = {s: pr.Track(planar(*ctx.sc.base_start) @ start_b[s]) for s in SIDES}
    segs, prims = [], {s: [] for s in SIDES}
    q = q0.copy()
    phases = []
    while path.t < until or not phases:
        leg, used = base_path(ctx, path.end, rng.uniform(2.0, 6.0))
        t_drive0 = path.t
        for t0, dur, fn in leg.segments:
            path.add(dur, fn, "drive")
        segs += used
        t_stop = path.t
        # hands: base-relative hold while driving (sampled on the final base path later)
        for s in SIDES:
            tracks[s].add(t_stop - t_drive0, (lambda u, s=s: np.full((len(u), 4, 4), np.nan)), "drive")
        anchor = planar(*path.end)
        manip = rng.uniform(3.0, 12.0)
        t_manip_end = t_stop
        for s in sides:
            tr = pr.Track(anchor @ ctx.robot.fk_base(q)[f"{s}_tcp"])
            tr, _, kinds = hand_chain(ctx, s, q, manip, ctx.sc.regions[SIDES.index(s)], anchor=anchor,
                                        start_track=tr)
            # return to the carry posture before driving on
            d, fn = pr.reach(tr.end, anchor @ ctx.robot.fk_base(q)[f"{s}_tcp"], ctx.speed)
            tr.add(d, fn, "reach")
            prims[s] += kinds
            for t0, dur, fn in tr.segments:
                tracks[s].add(dur, fn, "manip")
            t_manip_end = max(t_manip_end, t_stop + tr.t)
        for s in SIDES:
            tracks[s].hold(t_manip_end - tracks[s].t)
        path.hold(t_manip_end - path.t)
        phases.append((t_drive0, t_stop, t_manip_end))
        if path.t > until or rng.random() < 0.3:
            break
    return {"base": path, "tracks": {s: (tracks[s], "mixed") for s in SIDES}, "active": sides,
            "params": {"base_segments": segs, "primitives": prims, "phases": phases},
            "carry_base": {s: start_b[s] for s in SIDES}}


def _mobile_world_hold(ctx: Context, q0, start_b, start_w, until):
    """Hand(s) hold or move along a world path (door, wiping) while the base repositions a little."""
    rng = ctx.rng
    holders = list(SIDES) if rng.random() < 0.35 else [str(rng.choice(SIDES))]
    path, used = base_path(ctx, ctx.sc.base_start, min(until, rng.uniform(4.0, 15.0)), small=True)
    tracks, prims = {}, {}
    for s in SIDES:
        if s in holders:
            tr = pr.Track(start_w[s])
            while tr.t < path.t:
                kind = str(rng.choice(("hold", "hinge", "line", "wander")))
                if kind == "hold":
                    tr.hold(rng.uniform(0.5, 2.0))
                else:
                    d, fn = primitive(ctx, tr.end, kind)
                    tr.add(d, fn, kind)
                prims.setdefault(s, []).append(kind)
            tracks[s] = (tr, "world")
        else:
            tracks[s] = (_hold(start_b[s]), "base")
    return {"base": path, "tracks": tracks, "active": holders,
            "params": {"base_segments": used, "primitives": prims, "holders": holders}}


# ---------------------------------------------------------------------------- assembly


def generate(sc: Scenario, attempt: Attempt = Attempt()) -> TrackingReference:
    """The tracking reference of one scenario attempt (deterministic)."""
    ctx = Context(sc, attempt)
    rng, robot = ctx.rng, ctx.robot
    q0 = start_posture(ctx)
    if sc.cell == "bimanual_symmetric":
        q0[RIGHT_ARM] = q0[LEFT_ARM] * pr.MIRROR_SIGNS
    cell = build_cell(ctx, q0)
    base_track: pr.BasePath = cell["base"]
    tracks = cell["tracks"]
    witness = cell.get("witness")
    span = max([base_track.t] + [tr.t for tr, _ in tracks.values()]
               + ([witness[0][-1]] if witness is not None else []))
    span += float(rng.uniform(0.5, 1.0)) * attempt.time  # final hold
    T = int(np.ceil(span / DT)) + 1
    times = np.arange(T) * DT
    base = base_track.sample(times)
    q_start = q0.copy()
    q_start[:3] = base[0]
    wq = None
    if witness is not None:
        knot_t, vals, cols = witness
        wq = np.tile(q_start, (T, 1))
        wq[:, :3] = base
        wq[:, cols] = _pchip(knot_t, vals, times)
        fkb = robot.fk_base(wq)
        tcp = {s: _world(base, fkb[f"{s}_tcp"]) for s in SIDES}
    else:
        tcp = {}
        for s, (tr, frame) in tracks.items():
            X = tr.sample(times)
            if frame == "base":
                X = _world(base, X)
            elif frame == "mixed":  # NaN rows = base-relative carry pose
                hold = np.isnan(X[:, 0, 0])
                X[hold] = _world(base[hold], np.broadcast_to(cell["carry_base"][s], (int(hold.sum()), 4, 4)))
            tcp[s] = X
    neck = neck_path(ctx, sc.neck, times, base, tcp, q_start, cell["active"])
    head = None
    if neck is not None:
        Wr = planar(base[:, 0], base[:, 1], base[:, 2])[:, :3, :3]
        head = Wr @ neck_mod.head_base(neck)[:, :3, :3]
        if wq is not None:
            wq[:, NECK] = neck
    # grippers
    opening = np.zeros((T, 2))
    for i, s in enumerate(SIDES):
        levels = cell["gripper_override"].get(s)
        if levels is None:
            events = tracks[s][0].events if s in tracks else [(t, "reach_end") for t in
                                                               np.arange(1.0, span, rng.uniform(1.5, 4.0))]
            levels = gripper_levels(ctx, sc.grippers[i], events, span)
        opening[:, i] = pr.gripper_profile(times, levels, FINGER_RATE)
    q_start[GRIPPERS] = gripper.opening_to_angle(opening[0])
    if wq is not None:
        wq[:, GRIPPERS] = gripper.opening_to_angle(opening)
    parts = motion_parts(tcp, base, head)
    params = dict(ctx.params, **cell["params"], attempt=asdict(attempt), active=cell["active"],
                  base_speed_ratio=pr.body_speed_ratio(base, DT, VELOCITY[:3]))
    return TrackingReference(
        dataset=DATASET, episode_id=sc.episode_id, task=sc.cell, time=times, tcp=tcp, base=base,
        opening=opening, head=head, q_start=q_start, retime=False, regime=regime_of(parts), body_parts=parts,
        lineage={"seed": sc.source_id, "generated": True, "generator": GENERATOR},
        license="generated (no third-party data)",
        provenance={"generator": GENERATOR, "namespace": sc.namespace, "split": sc.split, "index": sc.index},
        witness=wq,
        extra={"scenario": asdict(sc), "params": _jsonable(params)})


def prescreen(ref: TrackingReference) -> list[str]:
    """Cheap reachability reasons (empty = go to IK): grasp-center heights outside Reachy's reach band
    and TCP targets farther from the shoulder than the arm reaches, at the nominal base."""
    robot = Reachy.load()
    reasons = []
    W = planar(ref.base[:, 0], ref.base[:, 1], ref.base[:, 2])
    for s in SIDES:
        z = (ref.tcp[s] @ robot.grasp_center(s))[:, 2, 3]
        if z.min() < REACH_Z[0] or z.max() > REACH_Z[1]:
            reasons.append(f"{s} grasp center height outside the reach band")
        local = np.einsum("tij,tj->ti", np.linalg.inv(W), np.c_[ref.tcp[s][:, :3, 3], np.ones(ref.length)])[:, :3]
        d = np.linalg.norm(local - SHOULDER[s], axis=1)
        if d.max() > TCP_REACH:
            reasons.append(f"{s} TCP target {d.max():.2f} m from the shoulder")
    return reasons


