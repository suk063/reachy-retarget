"""Arm assignment (docs/design.md, step 2) and the postures of unused body parts.

* Bimanual sources: ``side_hint`` when given, else the effector with the larger mean
  lateral coordinate (+y) in the source robot frame (``base_hint``, the source base path, or
  a heading estimated from the mean approach direction) becomes the left arm.
* Single-arm sources: both arms are placed and scored (:mod:`.placement`); the lower cost
  wins and ties go to the right arm.
* The inactive arm holds the ``rest`` posture of robot.toml (arms hanging beside the torso),
  whose self-clearance is checked against the IK margin. The nominal posture of active arms
  (posture regularization and IK seed) is ``ready``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..robot import min_clearance
from ..robot.reachy import GRIPPERS, LEFT_ARM, RIGHT_ARM
from ..robot.resources import profile
from .config import RetargetConfig
from .placement import Placement, PlacementProblem
from .targets import IDLE_FINGER


def posture(name):
    """(22,) q at the origin with both arms in robot.toml posture ``name``, neck 0, idle fingers."""
    p = profile()["postures"][name]
    q = np.zeros(22)
    q[LEFT_ARM], q[RIGHT_ARM] = np.radians(p["left"]), np.radians(p["right"])
    q[GRIPPERS] = IDLE_FINGER
    return q


def tuck_posture(cfg: RetargetConfig):
    """The collision-free posture held by inactive arms (``rest``); raises if it is not clear."""
    q = posture("rest")
    clearance = float(min_clearance(q))
    if clearance < cfg.self_clearance_margin:
        raise RuntimeError(f"rest posture self-clearance {clearance:.4f} m is below the IK margin")
    return q


def _source_frame(src):
    """(origin xy, yaw) of the source robot for lateral side decisions."""
    if src.base_hint is not None:
        return src.base_hint[:2], src.base_hint[2]
    if src.base is not None:
        return src.base[0, :2], src.base[0, 2]
    approach = np.mean([e.pose[:, :2, 2].mean(axis=0) for e in src.effectors.values()], axis=0)
    origin = np.mean([e.pose[:, :2, 3].mean(axis=0) for e in src.effectors.values()], axis=0)
    return origin - approach, float(np.arctan2(approach[1], approach[0]))


def bimanual_sides(src):
    """{effector key: side} for a two-effector source and the rule that decided it."""
    keys = sorted(src.effectors)
    hints = [src.effectors[k].side_hint for k in keys]
    if set(hints) == {"left", "right"}:
        return dict(zip(keys, hints)), "side_hint"
    if sum(h is not None for h in hints) == 1:
        i = 0 if hints[0] is not None else 1
        other = "left" if hints[i] == "right" else "right"
        return {keys[i]: hints[i], keys[1 - i]: other}, "side_hint (one effector)"
    origin, yaw = _source_frame(src)
    lateral = []
    for k in keys:
        d = src.effectors[k].pose[:, :2, 3] - origin
        lateral.append(float(np.mean(-np.sin(yaw) * d[:, 0] + np.cos(yaw) * d[:, 1])))
    left = int(np.argmax(lateral))
    return {keys[left]: "left", keys[1 - left]: "right"}, "lateral position"


@dataclass
class Assignment:
    sides: dict[str, str]         # effector key -> side
    placement: Placement
    rule: str
    scores: dict[str, float]      # per candidate side (single-arm sources)


def assign(src, cfg: RetargetConfig, labels) -> Assignment:
    """Assign effectors to arms and place the base. ``labels``: {effector key: grasp labels}."""
    nominal = posture("ready")
    keys = sorted(src.effectors)
    if len(keys) > 2:
        raise ValueError(f"{len(keys)} effectors; Reachy has two arms")
    if not keys:
        base = np.array(src.base, float)
        return Assignment({}, Placement(base, True, np.zeros(3), {}, 0.0, nominal), "no effectors", {})
    if len(keys) == 2:
        sides, rule = bimanual_sides(src)
        problem = PlacementProblem(src, sides, cfg, nominal, labels)
        return Assignment(sides, problem.segment_offsets(problem.refine(problem.search())), rule, {})
    best, scores = None, {}
    for side in ("right", "left"):
        problem = PlacementProblem(src, {keys[0]: side}, cfg, nominal, labels)
        placement = problem.search()
        scores[side] = placement.cost
        if best is None or placement.cost < best[1].cost:
            best = (problem, placement)
    problem, placement = best
    return Assignment(dict(problem.sides), problem.segment_offsets(problem.refine(placement)), "reachability score",
                      scores)
