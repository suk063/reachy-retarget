"""Tracking references from the hand paths of source datasets (docs/tracking.md, "Source paths").

Unlike the manipulation retargeting, the source hand path is the reference itself: the Reachy TCP
target is the source grasp-center pose times the fixed Reachy grasp-center offset
(:meth:`Reachy.tcp_from_grasp_center`); objects, grasp labels and grasp re-selection are ignored,
and the scene is empty (no footprint obstacles). What is chosen here:

* **sides**: ``side_hint``; bimanual sources without hints by the lateral position in the source
  robot frame; a single-arm source is tracked with both arms when ``both_arms`` (the arm with the
  better reachability score first, the other as a variant: ``variant_of``),
* **base**: fixed-base sources get one base pose (a grid around the target centroid and the source
  base hint, scored by a reachability proxy, the best few by keyframe IK); mobile sources follow the
  source base path composed with a constant SE(2) offset chosen the same way,
* **idle hand**: holds the ``home`` carry pose relative to the base,
* **neck**: points ``head_tip`` at the active hand(s) (along the travel direction without one),
* **gripper**: the source opening; **timing**: slowed down where Reachy's limits need it,
* **stages**: a demonstration chains several actions, so it is cut into stages (:mod:`.stages`): in
  each stage every hand, the base and each gripper make at most one movement toward one goal, and
  the head makes one movement (the rule of the synthetic scenarios). Each stage is one episode
  (``<episode>-stageNN``; all stages share the source's lineage seed).

``lineage.seed`` is the source's seed: these episodes are not independent of that source's
manipulation retarget.
"""
from __future__ import annotations

import numpy as np

from ..robot import Reachy, planar
from ..schema.episode import SIDES
from ..validate.kinematic import REACH_Z
from . import neck as neck_mod
from .config import TrackingConfig
from .reference import TrackingReference, jsonable, motion_parts, regime_of

GENERATOR = "tracking.source/v1"
CARRY = np.array([-0.42, 0.10, 0.10, -1.35, 0.0, 0.0, 0.0])  # reachy-control experiment start (left)
SHOULDER_Y = 0.2


def carry_posture() -> np.ndarray:
    q = np.zeros(22)
    q[3:10] = CARRY
    q[10:17] = CARRY * np.array([1, -1, -1, 1, -1, 1, -1])
    return q


def _frame(src):
    if src.base_hint is not None:
        return src.base_hint
    if src.base is not None:
        return src.base[0]
    return np.zeros(3)


def assign_sides(src) -> dict[str, str]:
    """{effector key: side} for up to two effectors."""
    keys = list(src.effectors)[:2]
    hints = {k: src.effectors[k].side_hint for k in keys}
    if len(keys) == 1:
        return {keys[0]: hints[keys[0]] or "right"}
    if all(hints.values()) and len(set(hints.values())) == 2:
        return dict(hints)
    x, y, yaw = _frame(src)
    lateral = {k: np.mean(-np.sin(yaw) * (src.effectors[k].pose[:, 0, 3] - x)
                          + np.cos(yaw) * (src.effectors[k].pose[:, 1, 3] - y)) for k in keys}
    left = max(keys, key=lateral.get)
    return {k: "left" if k == left else "right" for k in keys}


def _keyframes(T, n=12):
    return np.unique(np.linspace(0, T - 1, n).round().astype(int))


def _proxy(targets, base, rows):
    """Reachability proxy cost of TCP targets {side: (T, 4, 4)} at base poses (len(rows), 3)."""
    robot = Reachy.load()
    W = planar(base[:, 0], base[:, 1], base[:, 2])
    cost = 0.0
    for side, X in targets.items():
        L = np.linalg.inv(W) @ X[rows]
        sh = np.array([-0.01, SHOULDER_Y if side == "left" else -SHOULDER_Y, 1.166])
        d = np.linalg.norm(L[:, :3, 3] - sh, axis=1)
        z = (X[rows] @ robot.grasp_center(side))[:, 2, 3]
        cost += np.sum(np.maximum(d - 0.55, 0)) * 10 + np.sum(np.maximum(0.15 - L[:, 0, 3], 0)) * 10
        cost += np.sum(np.maximum(REACH_Z[0] - z, 0) + np.maximum(z - REACH_Z[1], 0)) * 10
    return cost


def _ik_score(targets, base, rows, cfg: TrackingConfig):
    """Mean normalized keyframe residual of a fixed-base IK at ``base`` (len(rows), 3)."""
    from ..retarget.wbik import FrameSolver
    ik = cfg.ik()
    nominal = carry_posture()
    solver = FrameSolver(ik, tuple(targets), False, nominal)
    q = nominal.copy()
    errs = []
    for i, r in enumerate(rows):
        frame = {s: X[r] for s, X in targets.items()}
        q[:3] = base[i]
        q = (solver.cold_solve(q, frame, base[i], ik.ik_first_max_iter)[0] if i == 0 else
             solver.solve(q, frame, q, base[i], 30)[0])
        errs.append(min(solver.error(q, frame), 10.0))
    return float(np.mean(errs) + 0.5 * np.max(errs))


def place_base(src, targets, cfg: TrackingConfig, n_eval=4):
    """Nominal base path (T, 3) and placement diagnostics."""
    T = src.length
    rows = _keyframes(T)
    if src.base is not None:
        b = src.base.copy()
        b[:, 2] = np.unwrap(b[:, 2])
        single = list(targets)[0] if len(targets) == 1 else None
        lat = 0.0 if single is None else (-SHOULDER_Y if single == "left" else SHOULDER_Y)
        cands = []
        for dx in (-0.3, -0.15, 0.0, 0.15):
            for dy in (-0.15, 0.0, 0.15):
                off = np.array([dx, dy + lat])
                c, s = np.cos(b[:, 2]), np.sin(b[:, 2])
                path = b.copy()
                path[:, 0] += c * off[0] - s * off[1]
                path[:, 1] += s * off[0] + c * off[1]
                cands.append((path, {"offset_body_xy": off.tolist()}))
    else:
        pts = np.concatenate([X[:, :3, 3] for X in targets.values()])
        centroid = pts[:, :2].mean(axis=0)
        single = list(targets)[0] if len(targets) == 1 else None
        lat = 0.0 if single is None else (-SHOULDER_Y if single == "left" else SHOULDER_Y)
        cands = []
        yaws = np.linspace(-np.pi, np.pi, 16, endpoint=False)
        if src.base_hint is not None:
            hint = src.base_hint
            yaws = np.r_[hint[2], yaws]
        for yaw in yaws:
            for r in (0.3, 0.4, 0.5, 0.6):
                head = np.array([np.cos(yaw), np.sin(yaw)])
                side = np.array([-np.sin(yaw), np.cos(yaw)])
                xy = centroid - r * head + lat * side
                cands.append((np.tile([xy[0], xy[1], yaw], (T, 1)), {"pose": [xy[0], xy[1], float(yaw)]}))
    costs = [_proxy(targets, path[rows], rows) for path, _ in cands]
    order = np.argsort(costs)[:n_eval]
    scored = [(_ik_score(targets, cands[i][0][rows], rows, cfg), int(i)) for i in order]
    best = min(scored)
    path, info = cands[best[1]]
    return path, dict(info, ik_score=best[0], proxy=float(costs[best[1]]), candidates=len(cands))


def references(src, cfg: TrackingConfig | None = None, both_arms: bool = True) -> list[TrackingReference]:
    """The tracking references of one SourceEpisode: one per stage and arm assignment."""
    cfg = cfg or TrackingConfig()
    robot = Reachy.load()
    times = src.time - src.time[0]
    T = src.length
    navigation = src.regime == "navigation" or not src.effectors
    sides_of = {} if navigation else assign_sides(src)
    variants = [sides_of]
    if both_arms and len(sides_of) == 1:
        (key, side), = sides_of.items()
        variants.append({key: "left" if side == "right" else "right"})
    out = []
    scored = []
    for sides in variants:
        targets = {side: robot.tcp_from_grasp_center(src.effectors[k].pose, side) for k, side in sides.items()}
        if targets:
            base, place = place_base(src, targets, cfg)
        else:
            base, place = np.asarray(src.base, float).copy(), {"source_base": True}
            base[:, 2] = np.unwrap(base[:, 2])
        scored.append((place.get("ik_score", 0.0), sides, targets, base, place))
    if len(scored) == 2:
        scored.sort(key=lambda s: s[0])
    # stages: cut once per demonstration (on the first variant; the arm choice does not change when
    # the source moves), every variant uses the same cuts
    _, sides0, targets0, base0, _ = scored[0]
    bounds, stage_parts = stage_bounds(src, times, sides0, targets0, base0)
    out = []
    primary = {}
    for n, (_, sides, targets, base, place) in enumerate(scored):
        W = planar(base[:, 0], base[:, 1], base[:, 2])
        carry = robot.fk_base(carry_posture())
        tcp = {s: targets[s] if s in targets else W @ carry[f"{s}_tcp"] for s in SIDES}
        opening = np.ones((T, 2))
        for k, s in sides.items():
            opening[:, SIDES.index(s)] = np.clip(src.effectors[k].opening, 0, 1)
        # neck: look at the active hand(s), else along the travel direction
        dt = np.diff(times)
        if targets:
            pts = np.mean([(X @ robot.grasp_center(s))[:, :3, 3] for s, X in targets.items()], axis=0)
        else:
            v = np.gradient(base[:, :2], times, axis=0)
            spd = np.linalg.norm(v, axis=1, keepdims=True)
            heading = np.c_[np.cos(base[:, 2]), np.sin(base[:, 2])]
            w = np.clip(spd / 0.1, 0, 1)
            d = w * v / np.maximum(spd, 1e-9) + (1 - w) * heading
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            pts = np.c_[base[:, :2] + 1.5 * d, np.full(T, 1.0)]
        gaze = neck_mod.gaze_neck(pts, base, margin=0.07, start=np.zeros(3), dt=dt)
        suffix = "-".join(sorted(sides.values())) if sides else "base"
        dataset = f"tracking/source/{src.dataset}"
        for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            rows = slice(a, b + 1)
            t = times[rows]
            # one head action per stage: the gaze path of the stage, or one turn toward where it ends
            neck = neck_mod.one_action(gaze[rows], t, gaze[a], (t[0], t[-1]), np.diff(t))
            head = W[rows, :3, :3] @ neck_mod.head_base(neck)[:, :3, :3]
            stage_tcp = {s: X[rows] for s, X in tcp.items()}
            parts = motion_parts(stage_tcp, base[rows], head)
            base_id = f"{src.episode_id}-{suffix}" if len(sides) == 1 else src.episode_id
            episode_id = f"{base_id}-stage{k:02d}" if len(bounds) > 2 else base_id
            ref = TrackingReference(
                dataset=dataset, episode_id=episode_id, task=src.task, time=src.time[rows], tcp=stage_tcp,
                base=base[rows], opening=opening[rows], head=head, q_start=None, retime=True,
                regime="navigation" if navigation else regime_of(parts), body_parts=parts,
                lineage=dict(src.lineage, tracking_of=src.uid, stage=k, stages=len(bounds) - 1),
                variant_of=primary.get(k), license=src.license, provenance=dict(src.provenance),
                instruction=src.instruction,
                extra={"generator": GENERATOR,
                       "source": {"family": src.family, "dataset": src.dataset, "episode_id": src.episode_id,
                                  "uid": src.uid, "regime": src.regime, "success": src.success},
                       "stage": {"index": k, "count": len(bounds) - 1, "source_rows": [int(a), int(b)],
                                 "source_time": [float(src.time[a]), float(src.time[b])],
                                 "movements": {name: [[int(x), int(y)] for x, y in runs if x < b and y > a]
                                               for name, runs in stage_parts.items()}},
                       "params": jsonable({"sides": sides, "placement": place, "neck": "look_at_hand"
                                           if targets else "look_ahead",
                                           "notes": ["objects, grasp labels and scene geometry are ignored",
                                                     "one stage of the demonstration (stages.cuts)"]})})
            if n == 0:
                primary[k] = f"{dataset}/{episode_id}"
            out.append(ref)
    return out


def stage_bounds(src, times, sides, targets, base):
    """Stage boundaries (source rows) of one demonstration and the movements of each part."""
    from . import stages
    W_inv = np.linalg.inv(planar(base[:, 0], base[:, 1], base[:, 2]))
    parts, speeds = {}, []
    for side, X in targets.items():
        L = W_inv @ X
        parts[f"{side}_hand"] = stages.movements(times, pose=L)
        speeds.append((np.linalg.norm(np.gradient(L[:, :3, 3], times, axis=0), axis=1), 0.05))
    for key, side in sides.items():
        o = np.clip(src.effectors[key].opening, 0, 1)
        parts[f"{side}_gripper"] = stages.movements(times, scalar=o)
        speeds.append((np.abs(np.gradient(o, times)), 0.5))
    if src.base is not None:
        parts["base"] = stages.movements(times, planar=base)
        speeds.append((np.linalg.norm(np.gradient(base[:, :2], times, axis=0), axis=1), 0.05))
    bounds = stages.cuts(times, parts, stages.activity(times, speeds))
    return bounds, parts
