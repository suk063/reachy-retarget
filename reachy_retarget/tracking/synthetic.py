"""Procedural tracking scenarios (docs/tracking.md, "Synthetic scenarios").

**One scenario = one stage.** Every body part does at most one action, going once from its start
to one goal (no return, repetition, cycle or sequence of steps). Several parts may act at the same
time within the stage, e.g. the base drives while an arm reaches and the head turns. The stage is
framed by a short still start and a still end.

Every scenario is one independent seed (``<namespace>/<split>/<index>``, hashed as in reachy-control
``rl_tracking/data.py``) and samples these axes independently:

* body-usage **cell** (:data:`CELLS`; stratified: every block of ``len(CELLS)`` consecutive
  indices holds each cell once, in a seeded order),
* **hand motion** per active hand (:data:`HAND_MOTIONS`) and **base motion** (:data:`BASE_MOTIONS`),
* **neck mode** (:data:`NECK_MODES`; ``off`` = no head reference),
* **extent** (motion magnitude, :data:`EXTENTS`), **speed** class (:data:`SPEEDS`), **start**
  posture (:data:`STARTS`), **workspace region** per hand (:data:`REGIONS`), **gripper** action per
  hand (:data:`GRIPPER_ACTIONS`, at most one transition) and the world start pose of the base.

Reach goals are postures from a joint-space step (each goal is reachable and clear of self-collision
on its own). ``witness`` cells instead make one smooth joint-space move whose forward kinematics is
the reference (feasible by construction, as in ``rl_tracking``). Head references are always the
rotation of a neck path inside the limits and the neck speed relative to the nominal base.

:func:`generate` builds a :class:`~.reference.TrackingReference` for one scenario and one
:class:`Attempt` (``size`` scales motion magnitudes, ``time`` stretches every duration); the
retry policy lives in :mod:`.build`.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from ..robot import LOWER, UPPER, VELOCITY, Reachy, gripper, min_clearance, planar
from ..robot.reachy import GRIPPERS, LEFT_ARM, NECK, RIGHT_ARM
from ..robot.resources import profile, urdf
from ..schema.episode import DT, SIDES
from ..schema.rotations import so3_exp, so3_log
from ..validate.kinematic import REACH_Z
from . import neck as neck_mod
from . import primitives as pr
from .reference import TrackingReference, jsonable as _jsonable, motion_parts, regime_of

GENERATOR = "tracking.synthetic/v1"
DATASET = "tracking/synthetic-v1"

STATIONARY = ("gripper_only", "neck_only", "left", "right", "bimanual_independent", "bimanual_symmetric",
              "bimanual_rigid", "handover", "witness_arms")
MOBILE = ("navigation", "mobile_carry", "mobile_reach", "mobile_world_hold", "mobile_free",
          "witness_whole_body")
CELLS = STATIONARY + MOBILE
HAND_MOTIONS = ("reach", "curved_reach", "line", "approach", "hinge", "twist", "tilt")
HAND_WEIGHTS = (0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1)
BASE_MOTIONS = ("drive", "strafe", "turn", "arc", "curve")
BASE_WEIGHTS = (0.3, 0.15, 0.15, 0.2, 0.2)
NECK_MODES = ("off", "hold", "world_hold", "look_at_point", "look_at_hand", "look_ahead", "shift")
NECK_WEIGHTS = (0.25, 0.1, 0.1, 0.15, 0.15, 0.1, 0.15)
OBJECT_MOTIONS = ("lift", "slide", "turn", "tilt")
# Motion magnitude per extent: arm joint step scale, translation (m), rotation (rad), base travel (m),
# base turn (rad).
EXTENTS = {
    "small": dict(step=0.45, lin=(0.04, 0.09), ang=(0.15, 0.4), base=(0.2, 0.6), turn=(0.3, 0.8)),
    "medium": dict(step=0.75, lin=(0.09, 0.17), ang=(0.4, 0.75), base=(0.6, 1.5), turn=(0.8, 1.6)),
    "large": dict(step=1.0, lin=(0.17, 0.28), ang=(0.75, 1.1), base=(1.5, 3.0), turn=(1.6, 3.1)),
}
EXTENT_WEIGHTS = (0.35, 0.4, 0.25)
# Linear/angular hand speeds, base cruise (m/s), base yaw rate (rad/s), witness joint-speed fraction.
SPEEDS = {
    "slow": dict(lin=(0.06, 0.12), ang=(0.25, 0.5), base=(0.2, 0.3), yaw=(0.3, 0.5), joint=(0.25, 0.4)),
    "normal": dict(lin=(0.12, 0.25), ang=(0.5, 0.9), base=(0.3, 0.42), yaw=(0.5, 0.9), joint=(0.4, 0.6)),
    "fast": dict(lin=(0.25, 0.4), ang=(0.9, 1.3), base=(0.42, 0.55), yaw=(0.9, 1.4), joint=(0.6, 0.8)),
}
SPEED_WEIGHTS = (0.3, 0.45, 0.25)
STARTS = ("home", "ready", "rest", "random")
START_WEIGHTS = (0.3, 0.2, 0.15, 0.35)
REGIONS = ("any", "low", "mid", "high", "front", "side", "cross")
REGION_WEIGHTS = (0.3, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1)
GRIPPER_ACTIONS = ("keep_open", "keep_closed", "close", "open", "partial")
GRIPPER_WEIGHTS = (0.25, 0.15, 0.25, 0.2, 0.15)

SHOULDER = {s: urdf().transform("base_link", f"{s[0]}_shoulder_ball_link")[:3, 3] for s in SIDES}
TCP_REACH = 0.665       # m, TCP to shoulder ball (sampled maximum 0.66, the straight arm)
KEY_MARGIN = 0.12       # rad inside the joint limits for goals and witnesses
KEY_CLEARANCE = 0.025   # m self-clearance of goal postures
WITNESS_CLEARANCE = 0.015
NECK_GEN_MARGIN = 0.07  # rad inside the neck limits for generated head references
SPEED_FRACTION = 0.85   # base speeds stay below this fraction of the limits
ARM_STEP = np.array([0.6, 0.4, 0.6, 0.6, 0.35, 0.35, 0.8])  # goal step at extent "large", rad per joint
FINGER_RATE = 0.9 * VELOCITY[GRIPPERS.start] / (gripper.UPPER - gripper.LOWER)  # opening units / s


@dataclass(frozen=True)
class Scenario:
    namespace: str
    split: str
    index: int
    cell: str
    neck: str
    extent: str
    speed: str
    start: str
    regions: tuple[str, str]
    grippers: tuple[str, str]
    hand_motions: tuple[str, str]
    base_motion: str | None
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
    size: float = 1.0   # motion magnitudes
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
    mobile = cell in MOBILE
    neck = _choice(rng, NECK_MODES, NECK_WEIGHTS)
    if neck == "look_ahead" and not mobile:
        neck = "look_at_point"
    if cell == "neck_only":  # the neck is the action
        neck = _choice(rng, ("look_at_point", "shift"), (1, 1))
    extent = _choice(rng, tuple(EXTENTS), EXTENT_WEIGHTS)
    speed = _choice(rng, tuple(SPEEDS), SPEED_WEIGHTS)
    start = _choice(rng, STARTS, START_WEIGHTS)
    if cell in ("mobile_carry", "bimanual_rigid") and start == "rest":
        start = "home"
    regions = tuple(_choice(rng, REGIONS, REGION_WEIGHTS) for _ in SIDES)
    if cell in ("bimanual_symmetric", "handover"):
        regions = tuple("any" if r == "cross" else r for r in regions)
    grippers = tuple(_choice(rng, GRIPPER_ACTIONS, GRIPPER_WEIGHTS) for _ in SIDES)
    if cell == "gripper_only":  # the gripper is the action
        grippers = tuple(_choice(rng, ("close", "open", "partial"), (1, 1, 1)) for _ in SIDES)
    hand_motions = tuple(_choice(rng, HAND_MOTIONS, HAND_WEIGHTS) for _ in SIDES)
    if cell == "mobile_reach":  # the hand reaches a world goal while the base drives
        hand_motions = tuple("reach" if m not in ("reach", "curved_reach") else m for m in hand_motions)
    base_motion = _choice(rng, BASE_MOTIONS, BASE_WEIGHTS) if mobile else None
    base_start = (float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)), float(rng.uniform(-np.pi, np.pi)))
    return Scenario(namespace, split, index, cell, neck, extent, speed, start, regions, grippers, hand_motions,
                    base_motion, base_start)


# ---------------------------------------------------------------------------- postures


def _profile_posture(name):
    p = profile()["postures"][name]
    q = np.zeros(22)
    q[LEFT_ARM], q[RIGHT_ARM] = np.radians(p["left"]), np.radians(p["right"])
    return q


def _arm(side):
    return LEFT_ARM if side == "left" else RIGHT_ARM


def _clip_arm(q_arm, side):
    sl = _arm(side)
    return np.clip(q_arm, LOWER[sl] + KEY_MARGIN, UPPER[sl] - KEY_MARGIN)


class Context:
    """Per-attempt generation state: rng, speeds, magnitudes, robot model."""

    def __init__(self, sc: Scenario, attempt: Attempt):
        self.sc, self.attempt = sc, attempt
        self.rng = np.random.default_rng(sc.seed)
        self.robot = Reachy.load()
        self.speed = {k: float(self.rng.uniform(*v)) for k, v in SPEEDS[sc.speed].items()}
        for k in ("lin", "ang", "base", "yaw", "joint"):
            self.speed[k] /= attempt.time
        self.size = attempt.size
        self.extent = EXTENTS[sc.extent]
        self.params: dict = {"speed": dict(self.speed)}

    def lin(self):
        return float(self.rng.uniform(*self.extent["lin"])) * self.size

    def ang(self):
        return float(self.rng.uniform(*self.extent["ang"])) * self.size

    def delay(self, longest):
        """Start offset of a part acting together with another: within the first 30 % of the stage."""
        return float(self.rng.uniform(0.0, 0.3)) * longest


def start_posture(ctx: Context) -> np.ndarray:
    """q (22,) at the origin of the base frame (the base pose is added by the caller)."""
    sc, rng = ctx.sc, ctx.rng
    kind = sc.start
    if kind == "home":  # reachy-control experiment start (scenarios.initial_configuration)
        q = np.zeros(22)
        for sl, sign in ((LEFT_ARM, 1.0), (RIGHT_ARM, -1.0)):
            q[sl] = [rng.uniform(-0.5, -0.34), sign * 0.10, sign * 0.10, rng.uniform(-1.4, -1.3), 0, 0, 0]
    elif kind in ("ready", "rest"):
        q = _profile_posture(kind)
    else:
        q = _profile_posture("ready")
        for _ in range(200):
            cand = q.copy()
            for side in SIDES:
                cand[_arm(side)] = _clip_arm(q[_arm(side)] + rng.uniform(-1, 1, 7) * ARM_STEP * 1.3, side)
            if sc.cell == "bimanual_symmetric":
                cand[RIGHT_ARM] = cand[LEFT_ARM] * pr.MIRROR_SIGNS
            fk = ctx.robot.fk_base(cand)
            z = [fk[f"{s}_grasp"][2, 3] for s in SIDES]
            if min(z) > 0.65 and max(z) < 1.45 and min_clearance(cand) > 0.03:
                q = cand
                break
    if sc.cell == "bimanual_symmetric":
        q[RIGHT_ARM] = q[LEFT_ARM] * pr.MIRROR_SIGNS
    q[NECK] = 0.0
    return q


# ---------------------------------------------------------------------------- goals


def _region_ok(p, side, region):
    """Region predicate on TCP positions (N, 3) in the base frame."""
    sign = 1.0 if side == "left" else -1.0
    x, y, z = p[:, 0], p[:, 1] * sign, p[:, 2]
    ok = {"any": np.ones(len(p), bool), "low": z < 0.85, "mid": (z >= 0.85) & (z <= 1.15), "high": z > 1.15,
          "front": x > 0.42, "side": y > 0.38, "cross": y < -0.02}[region]
    return ok & (x > 0.12)


def goal_posture(ctx: Context, side: str, q: np.ndarray, region: str, n=192, extra=None):
    """A reachable, self-clear arm posture one joint-space step from ``q``'s arm, biased to a region.

    ``q``: (22,) full posture in the base frame (the other arm and the neck are kept for the
    clearance check). ``extra``: optional predicate on the TCP (N, 4, 4) poses. Returns (q', pose)."""
    rng = ctx.rng
    sl = _arm(side)
    cur = ctx.robot.fk_base(q)[f"{side}_tcp"]
    min_move = {"small": 0.03, "medium": 0.08, "large": 0.15}[ctx.sc.extent] * ctx.size
    for relax in range(4):
        cand = np.tile(q, (n, 1))
        step = ARM_STEP * ctx.extent["step"] * ctx.size * (1.0 + 0.3 * relax)
        cand[:, sl] = _clip_arm(q[sl] + rng.uniform(-1, 1, (n, 7)) * step, side)
        fk = ctx.robot.fk_base(cand)
        P, G = fk[f"{side}_tcp"], fk[f"{side}_grasp"]
        ok = (G[:, 2, 3] > REACH_Z[0] + 0.12) & (G[:, 2, 3] < 1.6)
        ok &= np.linalg.norm(P[:, :3, 3] - cur[:3, 3], axis=1) > min_move / (1.0 + relax)
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


# ---------------------------------------------------------------------------- one hand motion


def hand_motion(ctx: Context, side, q, kind, region, anchor=np.eye(4), extra=None):
    """One single-goal hand motion from the posture ``q`` (22,): ``(duration, fn, info)``. ``anchor``
    maps base-frame poses into the motion's frame (world for a still base, identity for
    base-relative motions)."""
    rng, sp = ctx.rng, ctx.speed
    P = anchor @ ctx.robot.fk_base(q)[f"{side}_tcp"]
    info = {"kind": kind}
    if kind in ("reach", "curved_reach"):
        _, goal = goal_posture(ctx, side, q, region, extra=extra)
        Q = anchor @ goal
        if kind == "reach":
            dur, fn = pr.reach(P, Q, sp)
        else:
            d = Q[:3, 3] - P[:3, 3]
            side_dir = np.cross(pr.unit(d) if np.linalg.norm(d) > 1e-6 else [1.0, 0, 0], pr.random_unit(rng))
            bulge = pr.unit(side_dir) * rng.uniform(0.15, 0.35) * max(np.linalg.norm(d), 0.05)
            dur, fn = pr.curved_reach(P, Q, bulge, sp)
            info["bulge_m"] = float(np.linalg.norm(bulge))
        info["distance_m"] = float(np.linalg.norm(Q[:3, 3] - P[:3, 3]))
        return dur, fn, info
    if kind == "line":
        direction = {0: [0, 0, 1], 1: [0, 0, -1]}.get(int(rng.integers(0, 4)), None)
        direction = pr.random_unit(rng) if direction is None else np.array(direction, float)
        length = ctx.lin()
        info.update(direction=direction.tolist(), length_m=length)
        return (*pr.line(P, direction, length, sp), info)
    if kind == "approach":
        depth = ctx.lin() * rng.choice([1.0, 1.0, -1.0])
        info["depth_m"] = float(depth)
        return (*pr.approach(P, depth, sp), info)
    if kind == "hinge":
        axis = np.array([0.0, 0, 1]) if rng.random() < 0.6 else pr.random_unit(rng, horizontal=True)
        radius = rng.uniform(0.15, 0.4)
        offset = pr.unit(np.cross(axis, pr.random_unit(rng))) * radius
        angle = rng.choice([-1, 1]) * ctx.ang()
        info.update(axis=axis.tolist(), radius_m=radius, angle_rad=float(angle))
        return (*pr.hinge(P, offset, axis, angle, sp), info)
    if kind == "twist":
        wrist_yaw = q[_arm(side).stop - 1]
        sign = rng.choice([-1, 1])
        room = (UPPER[9] - KEY_MARGIN - wrist_yaw) if sign > 0 else (wrist_yaw - LOWER[9] - KEY_MARGIN)
        if room < 0.1:
            sign, room = -sign, (UPPER[9] - KEY_MARGIN - wrist_yaw) if sign < 0 else (wrist_yaw - LOWER[9] - KEY_MARGIN)
        angle = sign * min(ctx.ang(), max(room, 0.05))
        info["angle_rad"] = float(angle)
        return (*pr.twist(P, angle, sp), info)
    if kind == "tilt":
        center = (P @ ctx.robot.grasp_center(side))[:3, 3]
        axis = pr.random_unit(rng, horizontal=True)
        angle = rng.choice([-1, 1]) * ctx.ang()
        info.update(axis=axis.tolist(), angle_rad=float(angle))
        return (*pr.tilt(P, center, axis, angle, sp), info)
    raise ValueError(kind)


def single(start_pose, delay, motion=None):
    """A Track holding ``start_pose`` for ``delay`` seconds, then doing ``motion`` (duration, fn) once."""
    tr = pr.Track(start_pose)
    tr.hold(delay)
    if motion is not None:
        tr.add(motion[0], motion[1], "action")
    return tr


# ---------------------------------------------------------------------------- one base motion


def base_motion(ctx: Context, start, kind):
    """One single-goal base drive: (duration, fn, info)."""
    rng, sp, e = ctx.rng, ctx.speed, ctx.extent
    b = np.array(start, float)
    dist = float(rng.uniform(*e["base"])) * ctx.size
    turn = float(rng.uniform(*e["turn"])) * ctx.size * rng.choice([-1, 1])
    if ctx.sc.cell == "mobile_world_hold":  # a hand stays on a world pose: only a short repositioning
        scale = {"small": 0.4, "medium": 0.7, "large": 1.0}[ctx.sc.extent] * ctx.size
        dist = float(rng.uniform(0.05, 0.3)) * scale
        turn = float(rng.uniform(0.1, 0.35)) * scale * np.sign(turn)
        if kind in ("arc", "curve"):  # no room for a curved drive: a straight repositioning instead
            kind = "drive"
    if kind == "curve" and dist < 0.3:  # a short Bezier would swing its heading back and forth
        kind = "drive"
    if kind == "drive":
        direction = float(rng.choice([0.0, 0.0, np.pi, rng.uniform(-np.pi, np.pi)]))
        dyaw = turn * rng.uniform(0.0, 0.5) if rng.random() < 0.5 else 0.0
        dur, fn = pr.drive(b, dist, direction, dyaw, sp)
        info = {"distance_m": dist, "direction_rad": direction, "dyaw_rad": dyaw}
    elif kind == "strafe":
        direction = float(rng.choice([-np.pi / 2, np.pi / 2]))
        dur, fn = pr.drive(b, dist * 0.6, direction, 0.0, sp)
        info = {"distance_m": dist * 0.6, "direction_rad": direction}
    elif kind == "turn":
        dur, fn = pr.turn(b, turn, sp)
        info = {"dyaw_rad": turn}
    elif kind == "arc":
        angle = abs(turn)
        radius = float(np.sign(turn)) * max(dist / max(angle, 0.3), 0.4)
        dur, fn = pr.arc(b, radius, angle, sp)
        info = {"radius_m": radius, "angle_rad": angle}
    elif kind == "curve":
        heading = b[2] + rng.uniform(-0.8, 0.8)
        goal = b[:2] + dist * np.array([np.cos(heading), np.sin(heading)])
        goal_yaw = b[2] + rng.uniform(-1.0, 1.0) * min(abs(turn), 1.2)
        dur, fn = pr.curve(b, goal, goal_yaw, sp)
        info = {"distance_m": dist, "goal_yaw_change_rad": float(goal_yaw - b[2])}
    else:
        raise ValueError(kind)
    info["kind"] = kind
    return _fit_base(dur, fn), fn, info


def _fit_base(dur, fn):
    """Stretch a base motion's duration until its body-frame speeds stay within SPEED_FRACTION of
    the limits."""
    for _ in range(6):
        u = np.linspace(0.0, 1.0, max(int(np.ceil(dur / DT)), 2) + 1)
        ratio = pr.body_speed_ratio(fn(u), dur / (len(u) - 1), VELOCITY[:3]) / SPEED_FRACTION
        if ratio <= 1.0:
            break
        dur *= ratio * 1.02
    return dur


# ---------------------------------------------------------------------------- witnesses


def joint_move(ctx: Context, q0, sides, tries: int = 40):
    """One smooth joint-space move of the arms ``sides`` from ``q0`` to one goal posture: min-jerk,
    inside the limits minus KEY_MARGIN, self-clearance >= WITNESS_CLEARANCE every 0.1 s (goals are
    redrawn otherwise). Returns (goal (22,), duration)."""
    rng = ctx.rng
    cols = np.concatenate([np.arange(_arm(s).start, _arm(s).stop) for s in sides])
    lo, hi = LOWER[cols] + KEY_MARGIN, UPPER[cols] - KEY_MARGIN
    step = np.tile(ARM_STEP, len(sides)) * ctx.extent["step"]
    best = None
    for attempt in range(tries):
        scale = ctx.size * 0.8 ** (attempt // 10)
        goal = q0.copy()
        goal[cols] = np.clip(q0[cols] + rng.uniform(-1, 1, len(cols)) * step * scale, lo, hi)
        u = pr.min_jerk(np.linspace(0, 1, 21))[:, None]
        clear = float(min_clearance(q0 + u * (goal - q0)).min())
        if best is None or clear > best[0]:
            best = (clear, goal)
        if clear >= WITNESS_CLEARANCE:
            break
    clear, goal = best
    ctx.params["witness_clearance_m"] = clear
    dq = np.abs(goal - q0)[cols]
    dur = max(0.6, pr.PEAK * float(np.max(dq / (VELOCITY[cols] * ctx.speed["joint"]))))
    return goal, dur


# ---------------------------------------------------------------------------- neck references


def _one_action(neck, times, start, stage, dt):
    return neck_mod.one_action(neck, times, start, stage, dt, rate_scale=0.8)


def neck_path(ctx: Context, mode, times, base, tcp_world, q_start, active_sides, stage):
    """Neck angles (T, 3) of the head reference (None for ``off``): one action within the stage,
    inside the limits minus NECK_GEN_MARGIN and the neck speed (relative to the nominal ``base``).
    Gaze-following modes keep following their target when that is one continuous movement, else they
    become one turn toward where the gaze ends (:func:`.neck.one_action`)."""
    rng = ctx.rng
    start = q_start[NECK]
    dt = np.diff(times)
    lo, hi = neck_mod.limits(NECK_GEN_MARGIN)
    T = len(times)
    W = planar(base[:, 0], base[:, 1], base[:, 2])
    t0, t1 = stage
    if mode == "off":
        return None
    if mode == "hold":
        return np.tile(start, (T, 1))
    if mode == "shift":  # one min-jerk turn of the head to a new orientation
        goal = rng.uniform(lo, hi) * {"small": 0.4, "medium": 0.7, "large": 1.0}[ctx.sc.extent]
        dur = max(0.5, pr.PEAK * float(np.max(np.abs(goal - start) / (VELOCITY[NECK] * 0.8))))
        begin = t0 + ctx.delay(max(t1 - t0, dur))
        s = pr.min_jerk((times - begin) / dur)
        return neck_mod.rate_limit(start + s[:, None] * (goal - start), start, dt, 0.8)
    if mode == "world_hold":
        R0 = W[0, :3, :3] @ neck_mod.head_base(start)[:3, :3]
        q = np.zeros((T, 22))
        q[:, :3] = base
        q[:, NECK] = start
        n = neck_mod.solve(q, np.broadcast_to(R0, (T, 3, 3)), margin=NECK_GEN_MARGIN)
        return _one_action(neck_mod.rate_limit(n, start, dt, 0.8), times, start, stage, dt)
    if mode == "look_at_point":  # one gaze shift to a fixed world point, held to the end
        local = np.array([rng.uniform(0.8, 3.0), rng.uniform(-1.2, 1.2), rng.uniform(0.3, 1.7), 1.0])
        begin = t0 + ctx.delay(t1 - t0)
        pts = np.where((times >= begin)[:, None], (W[0] @ local)[:3], np.nan)
        # before the shift the head keeps its start orientation: look where it already looks
        head0 = W[0] @ neck_mod.head_tip_base(start)
        pts[np.isnan(pts[:, 0])] = head0[:3, 3] + 2.0 * head0[:3, 0]
        return _one_action(neck_mod.gaze_neck(pts, base, margin=NECK_GEN_MARGIN, start=start, dt=dt), times, start,
                           stage, dt)
    if mode == "look_ahead":  # along the travel direction (heading when the base does not translate)
        from scipy.ndimage import gaussian_filter1d
        v = gaussian_filter1d(np.gradient(base[:, :2], times, axis=0), 10, axis=0, mode="nearest")
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        heading = np.c_[np.cos(base[:, 2]), np.sin(base[:, 2])]
        w = np.clip(speed / 0.1, 0, 1)
        d = w * v / np.maximum(speed, 1e-9) + (1 - w) * heading
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        pts = np.c_[base[:, :2] + 1.5 * d, np.full(T, rng.uniform(0.6, 1.3))]
        return _one_action(neck_mod.gaze_neck(pts, base, margin=NECK_GEN_MARGIN, start=start, dt=dt), times, start,
                           stage, dt)
    if mode == "look_at_hand":  # follow the acting hand (both: their midpoint)
        robot = ctx.robot
        gc = {s: (tcp_world[s] @ robot.grasp_center(s))[:, :3, 3] for s in SIDES}
        sides = active_sides or list(SIDES)
        pts = np.mean([gc[s] for s in sides], axis=0)
        return _one_action(neck_mod.gaze_neck(pts, base, margin=NECK_GEN_MARGIN, start=start, dt=dt), times, start,
                           stage, dt)
    raise ValueError(mode)


# ---------------------------------------------------------------------------- grippers


def gripper_levels(ctx: Context, action, t_change):
    """[(t, opening)] for one hand: constant, or one transition at ``t_change``."""
    rng = ctx.rng
    open_level = float(rng.uniform(0.7, 1.0))
    closed = float(gripper.angle_to_opening(gripper.width_to_angle(rng.uniform(0.0, 0.07))))
    if action == "keep_open":
        return [(0.0, open_level)]
    if action == "keep_closed":
        return [(0.0, closed)]
    if action == "close":
        return [(0.0, open_level), (t_change, closed)]
    if action == "open":
        return [(0.0, closed), (t_change, open_level)]
    if action == "partial":
        a = float(rng.uniform(0.0, 1.0))
        b = float(np.clip(a + rng.choice([-1, 1]) * rng.uniform(0.25, 0.6), 0.0, 1.0))
        return [(0.0, a), (t_change, b)]
    raise ValueError(action)


# ---------------------------------------------------------------------------- cells


def _world(B, X):
    """World poses (T, 4, 4) of base-frame poses X (T, 4, 4) on base path B (T, 3)."""
    return planar(B[:, 0], B[:, 1], B[:, 2]) @ X


def build_cell(ctx: Context, q0, t0):
    """Tracks and base path of the scenario's single stage, starting at ``t0`` (the still start).

    Returns ``tracks`` {side: (Track, frame)} with frame ``world`` (a still base), ``base``
    (base-relative) or ``blend`` (a world goal reached from the base-relative start while the base
    drives: see :func:`generate`), ``base`` (BasePath), ``active`` sides, ``arrivals`` {side: time}
    (when the hand action ends: the gripper acts there), optional ``witness`` and ``params``."""
    sc, rng, robot = ctx.sc, ctx.rng, ctx.robot
    cell = sc.cell
    B0 = planar(*sc.base_start)
    fkb = robot.fk_base(q0)
    start_b = {s: fkb[f"{s}_tcp"] for s in SIDES}
    start_w = {s: B0 @ start_b[s] for s in SIDES}
    base = pr.BasePath(np.array(sc.base_start))
    base.hold(t0)
    out = {"tracks": {s: (single(start_w[s], 0.0), "world") for s in SIDES}, "base": base, "active": [],
           "arrivals": {}, "params": {}, "gripper_override": {}}
    region = dict(zip(SIDES, sc.regions))
    motion = dict(zip(SIDES, sc.hand_motions))

    def arm_action(side, delay, anchor, frame, extra=None):
        dur, fn, info = hand_motion(ctx, side, q0, motion[side], region[side], anchor, extra)
        P0 = anchor @ start_b[side]
        out["tracks"][side] = (single(P0, t0 + delay, (dur, fn)), frame)
        out["arrivals"][side] = t0 + delay + dur
        out["active"].append(side)
        out["params"].setdefault("hand", {})[side] = dict(info, start_s=t0 + delay, duration_s=dur)
        return dur

    if cell == "gripper_only":
        for s in SIDES:
            out["arrivals"][s] = t0 + ctx.delay(1.0)
    elif cell == "neck_only":
        pass
    elif cell in ("left", "right"):
        arm_action(cell, 0.0, B0, "world")
    elif cell == "bimanual_independent":
        first = str(rng.choice(SIDES))
        d = arm_action(first, 0.0, B0, "world")
        arm_action("right" if first == "left" else "left", ctx.delay(d), B0, "world")
    elif cell == "bimanual_symmetric":
        arm_action("left", 0.0, np.eye(4), "base", extra=lambda P: P[:, 1, 3] > 0.12)
        tr = out["tracks"]["left"][0]
        mirrored = pr.Track(pr.mirror(tr.start))
        for _, dur, fn in tr.segments:
            mirrored.add(dur, (lambda u, fn=fn: pr.mirror(fn(u))), "mirror")
        out["tracks"] = {"left": (tr, "base"), "right": (mirrored, "base")}
        out["arrivals"]["right"] = out["arrivals"]["left"]
        out["active"] = list(SIDES)
        out["params"]["hand"]["right"] = "mirror of left"
    elif cell == "bimanual_rigid":
        out.update(_rigid_object(ctx, start_b, B0, t0, frame="world"))
    elif cell == "handover":
        out.update(_handover(ctx, q0, B0, t0))
    elif cell == "witness_arms":
        sides = list(SIDES) if rng.random() < 0.6 else [str(rng.choice(SIDES))]
        goal, dur = joint_move(ctx, q0, sides)
        out["witness"] = (goal, t0, dur)
        out["active"] = sides
        out["arrivals"] = {s: t0 + dur for s in sides}
        out["params"]["witness"] = {"sides": sides, "duration_s": dur}
    elif cell in MOBILE:
        bdur, bfn, binfo = base_motion(ctx, sc.base_start, sc.base_motion)
        base.add(bdur, bfn, "drive")
        out["params"]["base"] = dict(binfo, start_s=t0, duration_s=bdur)
        out["tracks"] = {s: (single(start_b[s], 0.0), "base") for s in SIDES}
        if cell == "mobile_carry":
            out["gripper_override"] = {s: "keep_closed" for s in SIDES}
            if rng.random() < 0.5:  # the carried box also moves once (lift, slide, turn, tilt)
                rigid = _rigid_object(ctx, start_b, np.eye(4), t0 + ctx.delay(bdur), frame="base")
                out["params"].update(rigid.pop("params"))
                out.update(rigid)
        elif cell == "mobile_reach":
            sides = list(SIDES) if rng.random() < 0.4 else [str(rng.choice(SIDES))]
            B_end = planar(*bfn(np.array([1.0]))[0])
            for s in sides:  # a world goal near the destination, reached as the base arrives
                _, goal = goal_posture(ctx, s, q0, region[s])
                out["tracks"][s] = (("blend", start_b[s], B_end @ goal, t0, bdur), "blend")
                out["arrivals"][s] = t0 + bdur
                out["active"].append(s)
            out["params"]["hand"] = {s: {"kind": "reach_world_goal_while_driving"} for s in sides}
        elif cell == "mobile_world_hold":  # a hand keeps (or moves along) a world pose while the base moves
            holders = list(SIDES) if rng.random() < 0.35 else [str(rng.choice(SIDES))]
            for s in holders:
                P0 = B0 @ start_b[s]
                if rng.random() < 0.5:
                    out["tracks"][s] = (single(P0, 0.0), "world")
                    out["params"].setdefault("hand", {})[s] = {"kind": "world_hold"}
                else:
                    kind = str(rng.choice(("line", "hinge")))
                    dur, fn, info = hand_motion(ctx, s, q0, kind, region[s], B0)
                    delay = ctx.delay(bdur)
                    out["tracks"][s] = (single(P0, t0 + delay, (dur, fn)), "world")
                    out["params"].setdefault("hand", {})[s] = dict(info, start_s=t0 + delay)
                out["active"].append(s)
                out["arrivals"][s] = t0 + bdur
        elif cell == "mobile_free":  # base-relative hand actions while the base drives
            sides = list(SIDES) if rng.random() < 0.5 else [str(rng.choice(SIDES))]
            for s in sides:
                arm_action(s, ctx.delay(bdur), np.eye(4), "base")
        elif cell == "witness_whole_body":
            sides = list(SIDES) if rng.random() < 0.6 else [str(rng.choice(SIDES))]
            goal, dur = joint_move(ctx, q0, sides)
            delay = ctx.delay(bdur)
            out["witness"] = (goal, t0 + delay, dur)
            out["active"] = sides
            out["arrivals"] = {s: t0 + delay + dur for s in sides}
            out["params"]["witness"] = {"sides": sides, "start_s": t0 + delay, "duration_s": dur}
    else:
        raise ValueError(cell)
    return out


def _rigid_object(ctx: Context, start_b, anchor, t_start, frame):
    """Both hands hold a virtual box (hand = object frame @ constant grasp); the box moves once:
    a lift or lower, a slide, a turn about the vertical or a tilt."""
    rng, sp = ctx.rng, ctx.speed
    mid = 0.5 * (start_b["left"][:3, 3] + start_b["right"][:3, 3])
    O0 = np.eye(4)
    O0[:3, 3] = mid
    O0 = anchor @ O0
    grasp = {s: np.linalg.inv(O0) @ (anchor @ start_b[s]) for s in SIDES}
    kind = str(rng.choice(OBJECT_MOTIONS))
    if kind == "lift":
        dur, fn = pr.line(O0, [0, 0, rng.choice([-1, 1])], ctx.lin() * 0.8, sp)
    elif kind == "slide":
        dur, fn = pr.line(O0, pr.random_unit(rng, horizontal=True), ctx.lin() * 0.7, sp)
    elif kind == "turn":
        dur, fn = pr.twist(O0, rng.choice([-1, 1]) * ctx.ang() * 0.5, sp)
    else:
        dur, fn = pr.tilt(O0, O0[:3, 3], pr.random_unit(rng, horizontal=True), ctx.ang() * 0.45, sp)
    tracks = {}
    for s in SIDES:
        tracks[s] = (single(O0 @ grasp[s], t_start, (dur, (lambda u, G=grasp[s]: fn(u) @ G))), frame)
    return {"tracks": tracks, "active": list(SIDES), "arrivals": {s: t_start + dur for s in SIDES},
            "params": {"rigid_object": {"kind": kind, "start_s": t_start, "duration_s": dur}},
            "gripper_override": {s: "keep_closed" for s in SIDES}}


def _handover(ctx: Context, q0, B0, t0):
    """Both hands move once to meet (TCPs 13-22 cm apart near the midline); at the meeting the taker
    closes and the giver opens (one gripper action each)."""
    rng, sp = ctx.rng, ctx.speed
    giver = str(rng.choice(SIDES))
    taker = "right" if giver == "left" else "left"
    sign = 1.0 if giver == "left" else -1.0
    q_g, P_g = goal_posture(ctx, giver, q0, "any", extra=lambda P: (sign * P[:, 1, 3] > -0.02)
                            & (sign * P[:, 1, 3] < 0.12) & (P[:, 0, 3] > 0.3) & (P[:, 2, 3] > 0.8)
                            & (P[:, 2, 3] < 1.25))
    q_mid = q0.copy()
    q_mid[_arm(giver)] = q_g[_arm(giver)]
    near = lambda P: (np.linalg.norm(P[:, :3, 3] - P_g[:3, 3], axis=1) > 0.13) & (  # noqa: E731
        np.linalg.norm(P[:, :3, 3] - P_g[:3, 3], axis=1) < 0.22)
    _, P_t = goal_posture(ctx, taker, q_mid, "any", extra=near, n=512)
    fkb = ctx.robot.fk_base(q0)
    dg, fg = pr.reach(B0 @ fkb[f"{giver}_tcp"], B0 @ P_g, sp)
    dt_, ft = pr.reach(B0 @ fkb[f"{taker}_tcp"], B0 @ P_t, sp)
    delay = ctx.delay(dg)
    meet = t0 + max(dg, delay + dt_)
    open_l = float(rng.uniform(0.7, 1.0))
    closed = float(gripper.angle_to_opening(gripper.width_to_angle(rng.uniform(0.01, 0.05))))
    return {"tracks": {giver: (single(B0 @ fkb[f"{giver}_tcp"], t0, (dg, fg)), "world"),
                       taker: (single(B0 @ fkb[f"{taker}_tcp"], t0 + delay, (dt_, ft)), "world")},
            "active": list(SIDES), "arrivals": {giver: meet, taker: meet},
            "gripper_override": {giver: [(0.0, closed), (meet, open_l)], taker: [(0.0, open_l), (meet, closed)]},
            "params": {"giver": giver, "meet_s": meet}}


# ---------------------------------------------------------------------------- assembly


def generate(sc: Scenario, attempt: Attempt = Attempt()) -> TrackingReference:
    """The tracking reference of one scenario attempt (deterministic)."""
    ctx = Context(sc, attempt)
    rng, robot = ctx.rng, ctx.robot
    q0 = start_posture(ctx)
    t0 = float(rng.uniform(0.2, 0.5)) * attempt.time  # still start
    cell = build_cell(ctx, q0, t0)
    base_track: pr.BasePath = cell["base"]
    tracks = cell["tracks"]
    witness = cell.get("witness")
    ends = [base_track.t] + [t[0].t if isinstance(t[0], pr.Track) else t[0][3] + t[0][4] for t in tracks.values()]
    ends += list(cell["arrivals"].values())
    if witness is not None:
        ends.append(witness[1] + witness[2])
    stage_end = max(ends + [t0 + 0.5])
    changes = [s for i, s in enumerate(SIDES)
               if not str(cell["gripper_override"].get(s, sc.grippers[i])).startswith("keep")]
    if changes:  # let a gripper action finish within the stage (a full stroke takes 1 / FINGER_RATE)
        stage_end = max([stage_end] + [cell["arrivals"].get(s, t0) + 1.0 / FINGER_RATE for s in changes])
    span = stage_end + float(rng.uniform(0.5, 1.0)) * attempt.time  # still end
    T = int(np.ceil(span / DT)) + 1
    times = np.arange(T) * DT
    base = base_track.sample(times)
    q_start = q0.copy()
    q_start[:3] = base[0]
    wq = None
    if witness is not None:
        goal, w0, wdur = witness
        s = pr.min_jerk((times - w0) / wdur)[:, None]
        wq = q_start + s * (goal - q0)
        wq[:, :3] = base
        fkb = robot.fk_base(wq)
        tcp = {s_: _world(base, fkb[f"{s_}_tcp"]) for s_ in SIDES}
    else:
        tcp = {}
        for s, (tr, frame) in tracks.items():
            if frame == "blend":  # world goal reached from the base-relative start while the base drives
                _, X_b, goal, b0, bdur = tr
                A = _world(base, np.broadcast_to(X_b, (T, 4, 4)))
                u = pr.min_jerk((times - b0) / bdur)[:, None]
                rel = np.linalg.inv(A) @ goal
                M = np.broadcast_to(np.eye(4), (T, 4, 4)).copy()
                M[:, :3, :3] = so3_exp(u * so3_log(rel[:, :3, :3]))
                M[:, :3, 3] = u * rel[:, :3, 3]
                tcp[s] = A @ M
                continue
            X = tr.sample(times)
            tcp[s] = _world(base, X) if frame == "base" else X
    neck = neck_path(ctx, sc.neck, times, base, tcp, q_start, cell["active"], (t0, stage_end))
    head = None
    if neck is not None:
        Wr = planar(base[:, 0], base[:, 1], base[:, 2])[:, :3, :3]
        head = Wr @ neck_mod.head_base(neck)[:, :3, :3]
        if wq is not None:
            wq[:, NECK] = neck
    # grippers: at most one transition per hand, when its action ends (or within the stage)
    opening = np.zeros((T, 2))
    for i, s in enumerate(SIDES):
        levels = cell["gripper_override"].get(s, sc.grippers[i])
        if isinstance(levels, str):
            t_change = cell["arrivals"].get(s, t0 + ctx.delay(stage_end - t0))
            levels = gripper_levels(ctx, levels, t_change)
        opening[:, i] = pr.gripper_profile(times, levels, FINGER_RATE)
    q_start[GRIPPERS] = gripper.opening_to_angle(opening[0])
    if wq is not None:
        wq[:, GRIPPERS] = gripper.opening_to_angle(opening)
    parts = motion_parts(tcp, base, head)
    params = dict(ctx.params, **cell["params"], attempt=asdict(attempt), active=cell["active"],
                  stage_s=[t0, stage_end], base_speed_ratio=pr.body_speed_ratio(base, DT, VELOCITY[:3]))
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
