"""SourceEpisode -> ReachyEpisode (docs/design.md, "Retargeting method").

Steps: grasp labels, object-centric hand paths for grasps the source does not hold rigidly,
arm assignment with base placement and grasp offsets, TCP targets, two-pass whole-body IK on
the source clock (free orientation away from grasps, then strict; + light smoothing; base
assistance when a fixed placement leaves frames out of tolerance), finger commands, gaze,
time scaling and 50 Hz resampling (+ residual refinement on the output clock), episode
assembly and tier K.
Episodes that fail tier K are still returned (failed retargets are saved too); the status
is ``failed`` only when no trajectory could be produced.
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field, replace

import numpy as np

from ..robot import Reachy, min_clearance
from ..robot.reachy import GRIPPERS, NECK
from ..schema.episode import DT, SIDES, ReachyEpisode, Reference
from ..schema.rotations import so3_log
from ..schema.source import Articulation, SourceEpisode
from ..validate import kinematic
from . import footprint, timing
from .assign import assign, posture, tuck_posture
from .config import RetargetConfig
from .gaze import gaze_error, gaze_points, solve_neck
from .placement import diagnostics as placement_diagnostics
from .targets import (blend_orientation, contact_width, finger_angles, grasp_labels, gripper_state,
                      hand_object_distance, held_masks, manipulated_ids, object_centric, orientation_weight,
                      offset_track, push_labels, source_closed, tcp_targets, _runs)
from .wbik import ARM_COLUMNS, refine, smooth, solve_trajectory, tcp_errors


@dataclass
class RetargetResult:
    episode: ReachyEpisode | None
    status: str                   # "ok" (a trajectory exists; see episode.tier) or "failed"
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _jsonable(x):
    """Recursively convert numpy values to JSON types; non-finite floats become None."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def _notes(src: SourceEpisode, sides):
    notes = []
    if src.torso_height is not None:
        notes.append("source torso_height ignored: Reachy's tripod stays at 0")
    if len(sides) == 1:
        hint = next(iter(src.effectors.values())).side_hint
        if hint is not None and hint != next(iter(sides.values())):
            notes.append(f"single-arm source side_hint {hint!r} overridden by the reachability score")
    return notes


def _bad_frames(cfg, q, targets, labels):
    """Frames whose TCP residual exceeds the tier-K tolerance (grasp tolerances while holding)
    or whose self-clearance is below the tier-K minimum."""
    bad = np.zeros(len(q), bool)
    for side, (pos, rot) in tcp_errors(q, targets).items():
        hold = labels[side] >= 0
        bad |= pos > np.where(hold, cfg.grasp_tcp_pos_tol, cfg.tcp_pos_tol)
        bad |= rot > np.where(hold, cfg.grasp_tcp_rot_tol, cfg.tcp_rot_tol)
    return int(np.sum(bad | (min_clearance(q) < cfg.min_self_clearance)))


def _ik(cfg, robot, targets, weights, labels, base_ref, base_free, q0, nominal, box=None, obstacles=None):
    """Two-pass whole-body IK on the source clock (free orientation away from grasps, then strict
    tracking of the blended reference) and smoothing; returns q, the blended targets and stats."""
    targets = dict(targets)
    relaxed = {s: w for s, w in weights.items() if (w < 1).any()}
    orient = {}
    if relaxed:
        # Pass 1: orientation weight lowered away from grasps; its achieved TCP rotation becomes
        # the reference there (blended into the source rotation near grasps), tracked strictly.
        q_free, _ = solve_trajectory(cfg, targets, base_ref, base_free, q0, nominal, rot_scale=weights,
                                     max_iter=cfg.ik_free_iter, box=box, obstacles=obstacles)
        fk_free = robot.fk(q_free)
        for side in relaxed:
            src_t = targets[side]
            targets[side] = blend_orientation(src_t, fk_free[f"{side}_tcp"], weights[side])
            dev = np.linalg.norm(so3_log(np.swapaxes(src_t[:, :3, :3], 1, 2) @ targets[side][:, :3, :3]), axis=1)
            orient[side] = {"relaxed_frames": int(np.sum(weights[side] < 1)),
                            "max_reference_deviation_rad": float(dev.max())}
    q, iters = solve_trajectory(cfg, targets, base_ref, base_free, q0, nominal, box=box, obstacles=obstacles)
    q, smoothed = smooth(cfg, q, targets, base_free)
    return {"q": q, "iters": iters, "smoothed": smoothed, "targets": targets, "orientation": orient,
            "bad_frames": _bad_frames(cfg, q, targets, labels)}


def retarget(src: SourceEpisode, cfg: RetargetConfig | None = None) -> RetargetResult:
    """Retarget one source episode onto Reachy 2 (default configuration if ``cfg`` is None)
    and run tier K."""
    cfg = cfg or RetargetConfig()
    t_start = _time.perf_counter()
    robot = Reachy.load()
    closed_src = {k: source_closed(e, cfg, src.time) for k, e in src.effectors.items()}
    labels_src = {k: grasp_labels(e.pose[:, :3, 3], closed_src[k], src.objects, cfg, effector=e, times=src.time)
                  for k, e in src.effectors.items()}
    oc_diag, push_diag = {}, {}
    if cfg.object_centric:
        # Hands carry held objects rigidly along the source object path (targets.object_centric).
        effectors = {}
        for k, e in src.effectors.items():
            lab = labels_src[k]
            if cfg.push_centric:
                # closed-hand pushes follow the object like grasps (the relative pose at the push onset is kept)
                push = push_labels(src, k, lab, cfg, closed_src[k])
                lab = np.where(lab >= 0, lab, push)
                push_diag[k] = {"push_frames": int(np.sum(push >= 0)),
                                "segments": [[int(a), int(b)] for a, b in _runs(push >= 0)]}
            pose, oc_diag[k] = object_centric(src, k, lab, cfg)
            effectors[k] = replace(e, pose=pose)
        src = replace(src, effectors=effectors)
    try:
        tuck = tuck_posture(cfg)
        assignment = assign(src, cfg, labels_src)
    except (ValueError, RuntimeError) as err:
        return RetargetResult(None, "failed", [str(err)], {"stage": "assignment"})
    sides, place = assignment.sides, assignment.placement
    t_place = _time.perf_counter()

    # Whole-body IK on the source clock.
    nominal = posture("ready")
    offsets = {side: offset_track(src.time, place.segments[side], o, cfg) if side in place.segments else o
               for side, o in place.offsets.items()}
    targets = tcp_targets(src, sides, offsets)
    labels = {side: labels_src[key] for key, side in sides.items()}
    q0 = tuck.copy()
    for side in targets:
        q0[ARM_COLUMNS[side]] = place.seed[ARM_COLUMNS[side]]
    base_ref = place.base.copy()
    base_ref[:, 2] = np.unwrap(base_ref[:, 2])
    weights = {side: orientation_weight(src.time, labels[side], cfg,
                                        hand_object_distance(src, key) if cfg.orientation_by_distance else None)
               for key, side in sides.items()}
    fixed = _ik(cfg, robot, targets, weights, labels, base_ref, place.mobile, q0, nominal)
    best, base_free = fixed, place.mobile
    base_diag = {"assist": "mobile source" if place.mobile else "not needed" if fixed["bad_frames"] == 0
                 else "disabled"}
    if not place.mobile and fixed["bad_frames"] and cfg.base_assist:
        # A single placement does not cover the trajectory: let the base move (penalized
        # deviation from the placement, bounded box, footprint clearance as an IK constraint).
        span = np.r_[cfg.base_assist_range, cfg.base_assist_range, cfg.base_assist_yaw]
        lo, hi = np.full(22, -np.inf), np.full(22, np.inf)
        lo[:3], hi[:3] = base_ref[0] - span, base_ref[0] + span
        cfg_b = replace(cfg, w_base=cfg.w_base_assist)
        obs = footprint.obstacles(src.objects, cfg, held=held_masks(src.objects, labels.values()))
        assisted = _ik(cfg_b, robot, targets, weights, labels, base_ref, True, q0, nominal, box=(lo, hi),
                       obstacles=obs)
        has_obstacles = bool(obs.polygons) or len(obs.points) > 0
        clear = float(footprint.clearance(assisted["q"][:, :2], obs).min()) if has_obstacles else np.inf
        base_diag = {"assist": "rejected", "fixed_bad_frames": fixed["bad_frames"],
                     "assisted_bad_frames": assisted["bad_frames"], "footprint_clearance": clear,
                     "base_travel_m": float(np.ptp(assisted["q"][:, :2], axis=0).max()),
                     "base_yaw_range_rad": float(np.ptp(assisted["q"][:, 2]))}
        if clear >= 0 and assisted["bad_frames"] < fixed["bad_frames"]:
            best, base_free = assisted, True
            base_diag["assist"] = "used"
    q_src, iters, smoothed, targets, orient_diag = (best["q"], best["iters"], best["smoothed"], best["targets"],
                                                    best["orientation"])
    t_ik = _time.perf_counter()
    if not np.isfinite(q_src).all():
        return RetargetResult(None, "failed", ["IK produced non-finite joint values"], {"stage": "ik"})

    # Fingers and gaze.
    for key, side in sides.items():
        q_src[:, GRIPPERS.start + SIDES.index(side)] = finger_angles(
            src.effectors[key], labels[side] >= 0, cfg, contact_width(src, key, offsets[side], labels[side]))
    ids = manipulated_ids(src.objects)
    fk = robot.fk(q_src)
    hands = {side: fk[f"{side}_grasp"][:, :3, 3] for side in targets}
    points, rule = gaze_points(hands, labels, ids, src.objects, q_src[:, :3], cfg, src.time)
    q_src[:, NECK], head_ref_src = solve_neck(q_src, points, src.time, cfg)
    gaze_err = gaze_error(q_src, points)

    # Time scaling, resampling and refinement on the output clock.
    clock = timing.clock(src.time, q_src, cfg, timing.grasp_events(labels.values()))
    q = timing.linear(clock, q_src)
    targets_out = {s: timing.poses(clock, X) for s, X in targets.items()}
    base_out = timing.linear(clock, base_ref)
    assist = base_free and not place.mobile
    q, refined = refine(replace(cfg, w_base=cfg.w_base_assist) if assist else cfg, q, targets_out, base_out,
                        base_free, nominal, DT, obstacles=obs if assist else None)
    qd = timing.velocity(q)
    t_time = _time.perf_counter()

    grasp_out = np.full((len(q), 2), -1, dtype=np.int16)
    for side, lab in labels.items():
        grasp_out[:, SIDES.index(side)] = timing.labels(clock, lab)
    base_moves = bool(np.any(np.ptp(q[:, :3], axis=0) > 1e-3))
    neck_moves = bool(np.any(np.ptp(q[:, NECK], axis=0) > 1e-3))
    body_parts = (tuple(f"{s}_arm" for s in targets) + (("head",) if neck_moves else ())
                  + (("base",) if base_moves else ()))
    fkb = robot.fk_base(q)
    opening, width = gripper_state(q[:, GRIPPERS])
    n_src = src.length
    diag = {
        "assignment": {"sides": sides, "rule": assignment.rule, "scores": assignment.scores},
        "placement": placement_diagnostics(place),
        "free_orientation": orient_diag,
        "base_assist": base_diag,
        "push": push_diag,
        "object_centric": {k: {seg: {"max_hand_shift_m": v[0], "max_hand_turn_rad": v[1]} for seg, v in d.items()}
                           for k, d in oc_diag.items()},
        "timing": clock.diagnostics(src.time),
        "ik": {"source_frames": n_src, "output_frames": len(q), "mean_iterations": float(iters.mean()),
               "max_iterations": int(iters.max()), "ms_per_source_frame": (t_ik - t_place) / n_src * 1e3,
               "smoothing_kept_fraction": smoothed, "refined_output_frames": refined},
        "gaze": {"rule_counts": {name: int(np.sum(rule == i)) for i, name in
                                 enumerate(("held", "approached", "hands", "ahead"))},
                 "max_error": float(gaze_err.max()), "median_error": float(np.median(gaze_err))},
        "seconds": {"placement": t_place - t_start, "ik": t_ik - t_place, "timing_refine": t_time - t_ik},
    }
    episode = ReachyEpisode(
        family=src.family, dataset=src.dataset, episode_id=src.episode_id, task=src.task,
        time=clock.time, q=q, qd=qd,
        tcp_base={s: fkb[f"{s}_tcp"] for s in SIDES}, head_base=fkb["head"],
        gripper_opening=opening, gripper_width=width, source_time=clock.source_time,
        tier={"K": {"passed": False, "reasons": ["not validated"]}, "P": None},
        retarget_config=cfg.digest(), instruction=src.instruction, regime=src.regime, license=src.license,
        provenance=dict(src.provenance), lineage=dict(src.lineage), body_parts=body_parts,
        reference=Reference(tcp=targets_out, head=timing.poses(clock, head_ref_src), base=base_out),
        objects={k: timing.object_track(clock, o) for k, o in src.objects.items()},
        articulations={k: Articulation(list(a.joint_names), timing.linear(clock, a.qpos))
                       for k, a in src.articulations.items()},
        validation={"grasp_object": grasp_out},
        extra={"retarget": _jsonable(diag), "retarget_config": cfg.to_dict(), "grasp_object_ids": ids,
               "grasp_flips": _jsonable(place.flips),
               "grasp_offsets": _jsonable({s: {"flip": bool(o[0]), "theta_deg": o[1], "tilt_deg": o[2]}
                                           for s, o in place.offsets.items()}),
               "grasp_offset_segments": _jsonable(placement_diagnostics(place)["segment_grasp_offsets"]),
               "notes": _notes(src, sides)})
    k = kinematic.check(episode, cfg)
    episode.validation.update(k["frames"])
    episode.tier = {"K": {"passed": k["passed"], "reasons": k["reasons"]}, "P": None}
    episode.extra["tier_k_metrics"] = _jsonable(k["metrics"])
    diag["seconds"]["total"] = _time.perf_counter() - t_start
    return RetargetResult(episode, "ok", list(k["reasons"]), _jsonable(diag))
