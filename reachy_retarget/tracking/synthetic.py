"""Procedural tracking scenarios (docs/tracking.md, "Synthetic scenarios").

**One scenario = one stage.** Every body part does at most one action, going once from its start
to one goal (no return, repetition, cycle or sequence of steps). Several parts may act at the same
time within the stage, e.g. the base drives while an arm reaches and the head turns. The stage is
framed by a short still start and a still end.

Every scenario is one independent seed (``<namespace>/<split>/<index>``, hashed as in reachy-control
``rl_tracking/data.py``) and samples these axes independently:

* body-usage **cell** (:data:`CELLS_V2`; stratified: every block of ``len(CELLS_V2)`` consecutive
  indices holds each slot once, in a seeded order),
* **hand motion** per active hand (:data:`HAND_MOTIONS`) and **base motion** (:data:`BASE_MOTIONS`),
* **neck mode** (:data:`NECK_MODES`; ``off`` = no head reference),
* **extent** (motion magnitude, :data:`EXTENTS`), **speed** class (:data:`SPEEDS`), **start**
  posture (:data:`STARTS`), **workspace region** per hand (:data:`REGIONS`), **gripper** action per
  hand (:data:`GRIPPER_ACTIONS`, at most one transition) and the world start pose of the base.

Head references are always the rotation of a neck path inside the limits and the neck speed
relative to the nominal base.

:func:`generate` builds a :class:`~.reference.TrackingReference` for one scenario and one
:class:`Attempt` (``size`` scales motion magnitudes, ``time`` stretches every duration); the
retry policy lives in :mod:`.build`.

The generator is version 2 (:data:`VERSIONS`; dataset ``tracking/synthetic-v2``; version 1, a different
cell set, is retired). Most cells have explicit, body-relative goals, sampled for coverage rather than
as a step from the start posture, and indices form **sibling groups**:

* ``reach_target``: each active hand reaches a TCP position drawn uniformly in :data:`TARGET_BOX`
  (base frame, y outward), through a self-clear posture found among joint-space candidates; when 16
  draws find none (about 1.5 % of hands) the hand takes a joint-space step goal instead and its
  ``drawn_base`` is None (:func:`target_posture`);
* ``gaze_target``: the head turns to a gaze direction (yaw, pitch of the head +X axis in base_link)
  drawn uniformly in :data:`GAZE_YAW` x :data:`GAZE_PITCH`;
* ``joint_move``: one or two named arm joints, or one neck joint, move by a drawn amount and the
  others hold; the episode executes the joint path itself (``extra["exact_witness"]``);
* ``drive_target``: the base goes straight (holonomic) to a body-frame goal (dx, dy, dyaw);
* ``left``, ``right`` (one hand motion, :data:`HAND_MOTIONS`, whose reach goals are postures one
  joint-space step from the start, reachable and clear of self-collision), ``bimanual_independent``
  (one such motion per hand, staggered start) and ``navigation`` (one base motion,
  :data:`BASE_MOTIONS`; the arms hold).

Arms, head and grippers that are not the cell's action hold (grippers keep their state; the base
of the stationary target cells is not freed for the tracker, :data:`FIXED_BASE`). With
``group_size`` k > 1, indices ``k g .. k g + k - 1`` form sibling group g: every member has the
group's axes, start posture, speeds, base pose and still start (the group's seed), then draws its
own goals (its own seed), so the same start leads to different motions. Members share the lineage
seed (the group) and are not independent demonstrations.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from ..robot import LOWER, UPPER, VELOCITY, Reachy, gripper, min_clearance, planar
from ..robot.reachy import GRIPPERS, JOINTS, LEFT_ARM, NECK, RIGHT_ARM
from ..robot.resources import profile, urdf
from ..schema.episode import DT, SIDES
from ..validate.kinematic import REACH_Z
from . import neck as neck_mod
from . import primitives as pr
from .reference import TrackingReference, jsonable as _jsonable, motion_parts, regime_of

# Stratification slots (module docstring): reach_target and gaze_target fill two of ten each.
CELLS_V2 = ("reach_target", "reach_target", "gaze_target", "gaze_target", "left", "right", "bimanual_independent",
            "navigation", "joint_move", "drive_target")
VERSIONS = {2: {"generator": "tracking.synthetic/v2", "dataset": "tracking/synthetic-v2", "cells": CELLS_V2}}
MOBILE = ("navigation", "drive_target")                      # the cells whose base moves
FIXED_BASE = ("reach_target", "gaze_target", "joint_move")   # the tracker keeps the base on its path
KEEP_GRIPPERS = ("reach_target", "gaze_target", "joint_move", "drive_target", "navigation")
TARGET_BOX = ((0.15, 0.60), (0.0, 0.55), (0.55, 1.30))       # TCP x, outward y, z (m, base frame)
TARGET_TOL = 0.04                                            # m, reached TCP to the drawn position
GAZE_YAW, GAZE_PITCH = (-0.9, 0.9), (-0.4, 0.35)             # rad, head +X axis in base_link
GAZE_TOL = 0.02                                              # rad, solved gaze to the drawn one
JOINT_DELTA = (0.1, 0.6)                                     # rad, |change| of a moved joint
DRIVE_GOAL = ((-0.5, 2.5), (-1.0, 1.0), (-1.6, 1.6))         # dx, dy (m), dyaw (rad) in the start body frame
HAND_MOTIONS = ("reach", "curved_reach", "line", "approach", "hinge", "twist", "tilt")
HAND_WEIGHTS = (0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1)
BASE_MOTIONS = ("drive", "strafe", "turn", "arc", "curve")
BASE_WEIGHTS = (0.3, 0.15, 0.15, 0.2, 0.2)
NECK_MODES = ("off", "hold", "world_hold", "look_at_point", "look_at_hand", "look_ahead", "shift")
NECK_WEIGHTS = (0.25, 0.1, 0.1, 0.15, 0.15, 0.1, 0.15)
# Motion magnitude per extent: arm joint step scale, translation (m), rotation (rad), base travel (m),
# base turn (rad).
EXTENTS = {
    "small": dict(step=0.45, lin=(0.04, 0.09), ang=(0.15, 0.4), base=(0.2, 0.6), turn=(0.3, 0.8)),
    "medium": dict(step=0.75, lin=(0.09, 0.17), ang=(0.4, 0.75), base=(0.6, 1.5), turn=(0.8, 1.6)),
    "large": dict(step=1.0, lin=(0.17, 0.28), ang=(0.75, 1.1), base=(1.5, 3.0), turn=(1.6, 3.1)),
}
EXTENT_WEIGHTS = (0.35, 0.4, 0.25)
# Linear/angular hand speeds, base cruise (m/s), base yaw rate (rad/s), joint_move joint-speed fraction.
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
KEY_MARGIN = 0.12       # rad inside the joint limits for goals and joint moves
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
    version: int = 2
    group: int | None = None    # sibling group (group_size > 1)
    member: int | None = None

    @property
    def source_id(self) -> str:
        return f"{self.namespace}/{self.split}/{self.index:06d}"

    @property
    def group_id(self) -> str | None:
        return None if self.group is None else f"{self.namespace}/{self.split}/g{self.group:06d}"

    @property
    def seed(self) -> int:
        """The start's seed: shared by the members of a sibling group."""
        return seed_of(self.group_id or self.source_id)

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


def scenario(namespace: str, split: str, index: int, version: int = 2, group_size: int = 1) -> Scenario:
    """The deterministic scenario of one index (stratified cells, independent other axes; with
    ``group_size`` > 1, of its sibling group)."""
    cells = VERSIONS[version]["cells"]
    group, member = divmod(index, group_size) if group_size > 1 else (None, None)
    unit = index if group is None else group
    block, pos = divmod(unit, len(cells))
    order = np.random.default_rng(seed_of(f"{namespace}/{split}/cells/{block}")).permutation(len(cells))
    cell = cells[order[pos]]
    key = f"{index:06d}" if group is None else f"g{group:06d}"
    rng = np.random.default_rng(seed_of(f"{namespace}/{split}/{key}/axes"))
    # Every axis is drawn for every cell, then overridden below: the draw stream of the built datasets.
    mobile = cell in MOBILE
    neck = _choice(rng, NECK_MODES, NECK_WEIGHTS)
    if neck == "look_ahead" and not mobile:
        neck = "look_at_point"
    extent = _choice(rng, tuple(EXTENTS), EXTENT_WEIGHTS)
    speed = _choice(rng, tuple(SPEEDS), SPEED_WEIGHTS)
    start = _choice(rng, STARTS, START_WEIGHTS)
    regions = tuple(_choice(rng, REGIONS, REGION_WEIGHTS) for _ in SIDES)
    grippers = tuple(_choice(rng, GRIPPER_ACTIONS, GRIPPER_WEIGHTS) for _ in SIDES)
    hand_motions = tuple(_choice(rng, HAND_MOTIONS, HAND_WEIGHTS) for _ in SIDES)
    base_motion = _choice(rng, BASE_MOTIONS, BASE_WEIGHTS) if mobile else None
    base_start = (float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)), float(rng.uniform(-np.pi, np.pi)))
    if cell in KEEP_GRIPPERS:  # only the cell's action moves: grippers keep their drawn state
        grippers = tuple("keep_open" if g in ("keep_open", "open") else "keep_closed" for g in grippers)
    if cell == "gaze_target":
        neck = "gaze_target"
    elif cell == "joint_move":
        neck = "joint" if rng.random() < 0.25 else "off"
    elif cell in ("reach_target", "drive_target"):
        neck = "off"
    if cell == "drive_target":
        base_motion = "drive_target"
    return Scenario(namespace, split, index, cell, neck, extent, speed, start, regions, grippers, hand_motions,
                    base_motion, base_start, version, group, member)


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
            fk = ctx.robot.fk_base(cand)
            z = [fk[f"{s}_grasp"][2, 3] for s in SIDES]
            if min(z) > 0.65 and max(z) < 1.45 and min_clearance(cand) > 0.03:
                q = cand
                break
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


def goal_posture(ctx: Context, side: str, q: np.ndarray, region: str, n=192):
    """A reachable, self-clear arm posture one joint-space step from ``q``'s arm, biased to a region.

    ``q``: (22,) full posture in the base frame (the other arm and the neck are kept for the
    clearance check). Returns (q', pose)."""
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
        idx = np.flatnonzero(ok)
        if len(idx):
            idx = idx[rng.permutation(len(idx))[:24]]
            clear = min_clearance(cand[idx])
            good = idx[clear > KEY_CLEARANCE]
            if len(good):
                k = int(good[0])
                return cand[k], P[k]
    return q.copy(), cur


def target_posture(ctx: Context, side: str, q: np.ndarray, n=1024, tries=16):
    """A self-clear arm posture whose TCP is within TARGET_TOL of a position drawn uniformly in
    TARGET_BOX (base frame, y outward), among joint-space candidates around ``q``'s arm (the other arm
    and the neck kept for the clearance check): ``(q', pose, drawn position)``; a step goal
    (:func:`goal_posture`) with ``drawn`` None when no draw is reached."""
    rng, robot = ctx.rng, ctx.robot
    sl = _arm(side)
    lo, hi = np.array(TARGET_BOX).T
    sign = np.array([1.0, 1.0 if side == "left" else -1.0, 1.0])
    for _ in range(tries):
        drawn = rng.uniform(lo, hi) * sign
        cand = np.tile(q, (n, 1))
        cand[:, sl] = _clip_arm(q[sl] + rng.uniform(-1, 1, (n, 7)) * ARM_STEP * 1.6, side)
        fk = robot.fk_base(cand)
        P, G = fk[f"{side}_tcp"], fk[f"{side}_grasp"]
        d = np.linalg.norm(P[:, :3, 3] - drawn, axis=1)
        idx = np.flatnonzero((d < TARGET_TOL) & (G[:, 2, 3] > REACH_Z[0] + 0.12) & (G[:, 2, 3] < 1.6))
        if len(idx):
            idx = idx[np.argsort(d[idx])][:24]
            good = idx[min_clearance(cand[idx]) > KEY_CLEARANCE]
            if len(good):
                k = int(good[0])
                return cand[k], P[k], drawn
    goal, pose = goal_posture(ctx, side, q, "any")
    return goal, pose, None


# ---------------------------------------------------------------------------- one hand motion


def hand_motion(ctx: Context, side, q, kind, region, anchor=np.eye(4)):
    """One single-goal hand motion from the posture ``q`` (22,): ``(duration, fn, info)``. ``anchor``
    maps base-frame poses into the motion's frame (world for a still base, identity for
    base-relative motions)."""
    rng, sp = ctx.rng, ctx.speed
    P = anchor @ ctx.robot.fk_base(q)[f"{side}_tcp"]
    info = {"kind": kind}
    if kind in ("reach", "curved_reach"):
        _, goal = goal_posture(ctx, side, q, region)
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


# ---------------------------------------------------------------------------- joint moves


def _clear_goal(q0, draw, tries):
    """``(clearance, goal, info)`` of the first ``draw(attempt)`` (``(goal, info)``, or None to skip the
    attempt) whose min-jerk path from ``q0`` keeps self-clearance >= WITNESS_CLEARANCE at 21 points,
    else of the clearest goal drawn."""
    u = pr.min_jerk(np.linspace(0, 1, 21))[:, None]
    best = None
    for attempt in range(tries):
        drawn = draw(attempt)
        if drawn is None:
            continue
        goal, info = drawn
        clear = float(min_clearance(q0 + u * (goal - q0)).min())
        if best is None or clear > best[0]:
            best = (clear, goal, info)
        if clear >= WITNESS_CLEARANCE:
            break
    if best is None:
        raise ValueError("no joint move inside the limits")
    return best


def _joint_dur(ctx: Context, q0, goal, cols):
    """Min-jerk duration of a joint move whose fastest joint peaks at the drawn fraction of its speed
    limit (at least 0.6 s)."""
    return max(0.6, pr.PEAK * float(np.max(np.abs(goal - q0)[cols] / (VELOCITY[cols] * ctx.speed["joint"]))))


def named_joint_move(ctx: Context, q0, neck: bool, tries: int = 40):
    """One smooth joint-space move of one or two named joints of one arm, or of one neck joint, by
    JOINT_DELTA each; the other joints hold. Inside the limits minus KEY_MARGIN (NECK_GEN_MARGIN for
    the neck), self-clear along the path (:func:`_clear_goal`; the deltas shrink by 0.85 every 10 draws,
    moves under half the smallest delta are redrawn). Returns (goal (22,), duration, info)."""
    rng = ctx.rng
    if neck:
        cols, count, margin, side = np.arange(NECK.start, NECK.stop), 1, NECK_GEN_MARGIN, None
    else:
        side = str(rng.choice(SIDES))
        cols, count, margin = np.arange(_arm(side).start, _arm(side).stop), int(rng.choice([1, 2], p=[.6, .4])), KEY_MARGIN

    def draw(attempt):
        chosen = np.sort(rng.choice(cols, count, replace=False))
        delta = rng.choice([-1.0, 1.0], count) * rng.uniform(*JOINT_DELTA, count) * ctx.size * 0.85 ** (attempt // 10)
        goal = q0.copy()
        goal[chosen] = np.clip(q0[chosen] + delta, LOWER[chosen] + margin, UPPER[chosen] - margin)
        return None if np.abs(goal[chosen] - q0[chosen]).min() < 0.5 * JOINT_DELTA[0] else (goal, chosen)

    clear, goal, chosen = _clear_goal(q0, draw, tries)
    dur = _joint_dur(ctx, q0, goal, chosen)
    info = {"side": side, "joints": [JOINTS[c] for c in chosen], "start_rad": q0[chosen].tolist(),
            "goal_rad": goal[chosen].tolist(), "delta_rad": (goal[chosen] - q0[chosen]).tolist(), "clearance_m": clear}
    return goal, dur, info


def gaze_goal(ctx: Context, start):
    """Neck angles (3,) turning the head +X axis to a direction drawn uniformly in GAZE_YAW x GAZE_PITCH
    (base_link), solved within the limits minus NECK_GEN_MARGIN to GAZE_TOL; ``(goal, info)``."""
    rng = ctx.rng
    for _ in range(40):
        yaw, pitch = float(rng.uniform(*GAZE_YAW)), float(rng.uniform(*GAZE_PITCH))
        c, s, cp, sp_ = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        R = np.array([[c * cp, -s, -c * sp_], [s * cp, c, -s * sp_], [sp_, 0.0, cp]])  # Rz(yaw) Ry(-pitch)
        goal = neck_mod.solve_base(R[None], seed=start[None], margin=NECK_GEN_MARGIN)[0]
        x = neck_mod.head_base(goal)[:3, 0]
        achieved = (float(np.arctan2(x[1], x[0])), float(np.arcsin(np.clip(x[2], -1, 1))))
        if max(abs(achieved[0] - yaw), abs(achieved[1] - pitch)) < GAZE_TOL:
            return goal, {"drawn_rad": [yaw, pitch], "achieved_rad": list(achieved)}
    raise ValueError("no reachable gaze target")


# ---------------------------------------------------------------------------- neck references


def _one_action(neck, times, start, stage, dt):
    return neck_mod.one_action(neck, times, start, stage, dt, rate_scale=0.8)


def _neck_turn_dur(start, goal):
    """Duration of a min-jerk neck turn peaking at 0.8 x the neck speed limit (at least 0.5 s)."""
    return max(0.5, pr.PEAK * float(np.max(np.abs(goal - start) / (VELOCITY[NECK] * 0.8))))


def _neck_turn(times, start, goal, begin, dur):
    """Neck angles (T, 3) of one min-jerk turn from ``start`` to ``goal`` starting at ``begin``, rate-limited."""
    s = pr.min_jerk((times - begin) / dur)
    return neck_mod.rate_limit(start + s[:, None] * (goal - start), start, np.diff(times), 0.8)


def neck_path(ctx: Context, mode, times, base, tcp_world, q_start, active_sides, stage, turn=None):
    """Neck angles (T, 3) of the head reference (None for ``off``): one action within the stage,
    inside the limits minus NECK_GEN_MARGIN and the neck speed (relative to the nominal ``base``).
    Gaze-following modes keep following their target when that is one continuous movement, else they
    become one turn toward where the gaze ends (:func:`.neck.one_action`). ``turn``: the
    ``(goal, begin, duration)`` of a ``gaze_target`` cell's head turn (:func:`build_cell`)."""
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
    if mode == "gaze_target":  # the turn of build_cell, inside the stage
        return _neck_turn(times, start, *turn)
    if mode == "shift":  # one min-jerk turn of the head to a new orientation
        goal = rng.uniform(lo, hi) * {"small": 0.4, "medium": 0.7, "large": 1.0}[ctx.sc.extent]
        dur = _neck_turn_dur(start, goal)
        return _neck_turn(times, start, goal, t0 + ctx.delay(max(t1 - t0, dur)), dur)
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


def gripper_levels(ctx: Context, action, t_change, rng=None):
    """[(t, opening)] for one hand: constant, or one transition at ``t_change``; levels drawn from
    ``rng`` (default the context's)."""
    rng = ctx.rng if rng is None else rng
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

    Returns ``tracks`` {side: (Track, frame)} with frame ``world`` (a still base) or ``base``
    (base-relative), ``base`` (BasePath), ``active`` sides, ``arrivals`` {side: time} (when the hand
    action ends: the gripper acts there), optional ``witness`` (the ``joint_move`` path: goal, start,
    duration), ``neck_turn`` (the ``gaze_target`` head turn: goal, start, duration) and ``params``."""
    sc, rng, robot = ctx.sc, ctx.rng, ctx.robot
    cell = sc.cell
    B0 = planar(*sc.base_start)
    fkb = robot.fk_base(q0)
    start_b = {s: fkb[f"{s}_tcp"] for s in SIDES}
    start_w = {s: B0 @ start_b[s] for s in SIDES}
    base = pr.BasePath(np.array(sc.base_start))
    base.hold(t0)
    out = {"tracks": {s: (single(start_w[s], 0.0), "world") for s in SIDES}, "base": base, "active": [],
           "arrivals": {}, "params": {}}
    region = dict(zip(SIDES, sc.regions))
    motion = dict(zip(SIDES, sc.hand_motions))

    def arm_action(side, delay, anchor, frame):
        dur, fn, info = hand_motion(ctx, side, q0, motion[side], region[side], anchor)
        P0 = anchor @ start_b[side]
        out["tracks"][side] = (single(P0, t0 + delay, (dur, fn)), frame)
        out["arrivals"][side] = t0 + delay + dur
        out["active"].append(side)
        out["params"].setdefault("hand", {})[side] = dict(info, start_s=t0 + delay, duration_s=dur)
        return dur

    if cell == "gaze_target":  # one head turn to an absolute gaze direction; the stage lasts until it ends
        goal, info = gaze_goal(ctx, q0[NECK])
        dur = _neck_turn_dur(q0[NECK], goal)
        out["neck_turn"] = (goal, t0, dur)
        out["params"]["gaze_target"] = dict(info, start_s=t0, duration_s=dur)
        out["arrivals"] = {"head": t0 + dur}
    elif cell == "reach_target":
        sides = list(SIDES) if rng.random() < 1 / 3 else [str(rng.choice(SIDES))]
        q_now, delay = q0.copy(), 0.0
        for s in sides:
            goal, pose, drawn = target_posture(ctx, s, q_now)
            q_now[_arm(s)] = goal[_arm(s)]  # the second hand's goal stays clear of the first one's
            P0 = B0 @ start_b[s]
            dur, fn = pr.reach(P0, B0 @ pose, ctx.speed)
            out["tracks"][s] = (single(P0, t0 + delay, (dur, fn)), "world")
            out["arrivals"][s] = t0 + delay + dur
            out["active"].append(s)
            out["params"].setdefault("hand", {})[s] = {
                "kind": "reach_target", "target_base": pose[:3, 3].tolist(),
                "drawn_base": None if drawn is None else drawn.tolist(), "start_s": t0 + delay, "duration_s": dur}
            delay = ctx.delay(dur)  # drawn after the last hand too: the draw stream of the built datasets
    elif cell == "joint_move":
        goal, dur, info = named_joint_move(ctx, q0, sc.neck == "joint")
        out["witness"] = (goal, t0, dur)
        out["active"] = [info["side"]] if info["side"] else []
        out["arrivals"] = {info["side"]: t0 + dur} if info["side"] else {}
        out["params"]["joint_move"] = dict(info, start_s=t0, duration_s=dur)
    elif cell == "drive_target":
        dx, dy, dyaw = (float(rng.uniform(*r)) for r in DRIVE_GOAL)
        dur, fn = pr.drive(np.array(sc.base_start, float), float(np.hypot(dx, dy)), float(np.arctan2(dy, dx)), dyaw,
                           ctx.speed)
        dur = _fit_base(dur, fn)
        base.add(dur, fn, "drive")
        out["params"]["base"] = {"kind": "drive_target", "goal_body": [dx, dy, dyaw], "start_s": t0, "duration_s": dur}
        out["tracks"] = {s: (single(start_b[s], 0.0), "base") for s in SIDES}
    elif cell in ("left", "right"):
        arm_action(cell, 0.0, B0, "world")
    elif cell == "bimanual_independent":
        first = str(rng.choice(SIDES))
        d = arm_action(first, 0.0, B0, "world")
        arm_action("right" if first == "left" else "left", ctx.delay(d), B0, "world")
    elif cell == "navigation":
        bdur, bfn, binfo = base_motion(ctx, sc.base_start, sc.base_motion)
        base.add(bdur, bfn, "drive")
        out["params"]["base"] = dict(binfo, start_s=t0, duration_s=bdur)
        out["tracks"] = {s: (single(start_b[s], 0.0), "base") for s in SIDES}
    else:
        raise ValueError(cell)
    return out


# ---------------------------------------------------------------------------- assembly


def generate(sc: Scenario, attempt: Attempt = Attempt()) -> TrackingReference:
    """The tracking reference of one scenario attempt (deterministic)."""
    ctx = Context(sc, attempt)
    rng, robot = ctx.rng, ctx.robot
    q0 = start_posture(ctx)
    t0 = float(rng.uniform(0.2, 0.5)) * attempt.time  # still start
    start_rng = rng  # siblings share the start (also the gripper levels); each draws its own goals
    if sc.group is not None:
        ctx.rng = rng = np.random.default_rng(seed_of(f"{sc.source_id}/goals"))
    cell = build_cell(ctx, q0, t0)
    base_track: pr.BasePath = cell["base"]
    tracks = cell["tracks"]
    witness = cell.get("witness")
    ends = [base_track.t] + [tr.t for tr, _ in tracks.values()]
    ends += list(cell["arrivals"].values())
    if witness is not None:
        ends.append(witness[1] + witness[2])
    stage_end = max(ends + [t0 + 0.5])
    changes = [s for s, action in zip(SIDES, sc.grippers) if not action.startswith("keep")]
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
            X = tr.sample(times)
            tcp[s] = _world(base, X) if frame == "base" else X
    if sc.neck == "joint":  # a neck joint move: the head follows the witness
        neck = wq[:, NECK].copy()
    else:
        neck = neck_path(ctx, sc.neck, times, base, tcp, q_start, cell["active"], (t0, stage_end),
                         cell.get("neck_turn"))
    head = None
    if neck is not None:
        Wr = planar(base[:, 0], base[:, 1], base[:, 2])[:, :3, :3]
        head = Wr @ neck_mod.head_base(neck)[:, :3, :3]
        if wq is not None:
            wq[:, NECK] = neck
    # grippers: at most one transition per hand, when its action ends (or within the stage)
    opening = np.zeros((T, 2))
    for i, s in enumerate(SIDES):
        # the default delay is drawn even when the hand has an arrival: the draw stream of the built datasets
        t_change = cell["arrivals"].get(s, t0 + ctx.delay(stage_end - t0))
        levels = gripper_levels(ctx, sc.grippers[i], t_change, start_rng)
        opening[:, i] = pr.gripper_profile(times, levels, FINGER_RATE)
    q_start[GRIPPERS] = gripper.opening_to_angle(opening[0])
    if wq is not None:
        wq[:, GRIPPERS] = gripper.opening_to_angle(opening)
    parts = motion_parts(tcp, base, head)
    params = dict(ctx.params, **cell["params"], attempt=asdict(attempt), active=cell["active"],
                  stage_s=[t0, stage_end], base_speed_ratio=pr.body_speed_ratio(base, DT, VELOCITY[:3]))
    version = VERSIONS[sc.version]
    lineage = {"seed": sc.group_id or sc.source_id, "generated": True, "generator": version["generator"]}
    if sc.group is not None:
        lineage.update(group=sc.group_id, member=sc.member)
    extra = {"scenario": asdict(sc), "params": _jsonable(params)}
    if sc.cell == "joint_move":  # the joint path is the episode (no IK): the named joints are exact
        extra["exact_witness"] = True
    return TrackingReference(
        dataset=version["dataset"], episode_id=sc.episode_id, task=sc.cell, time=times, tcp=tcp, base=base,
        opening=opening, head=head, q_start=q_start, retime=False, regime=regime_of(parts), body_parts=parts,
        lineage=lineage, license="generated (no third-party data)",
        provenance={"generator": version["generator"], "namespace": sc.namespace, "split": sc.split, "index": sc.index},
        witness=wq, extra=extra)


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
