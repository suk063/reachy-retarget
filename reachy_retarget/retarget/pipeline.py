"""SourceEpisode -> ReachyEpisode (docs/design.md, "Retargeting method").

Steps: grasp labels (short drops after a release extended to the object's rest: placed drops),
object-centric hand paths for grasps the source does not hold rigidly, arm assignment with base
placement and grasp offsets, TCP targets, two-pass whole-body IK on the source clock (free
orientation away from grasps, then strict; + light smoothing; base assistance when a fixed
placement leaves frames out of tolerance), finger commands, gaze, time scaling and 50 Hz
resampling (+ residual refinement on the output clock), releases opened at the finger speed
limit, episode assembly and tier K.
Episodes that fail tier K are still returned (failed retargets are saved too); the status
is ``failed`` only when no trajectory could be produced.
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field, replace

import numpy as np

from ..robot import VELOCITY, Reachy, gripper, min_clearance
from ..robot.reachy import GRIPPERS, NECK
from ..schema.episode import DT, SIDES, ReachyEpisode, Reference
from ..schema.rotations import so3_log
from ..schema.source import Articulation, SourceEpisode
from ..validate import kinematic
from . import footprint, timing
from .assign import assign, posture, stow_posture, tuck_posture
from .config import RetargetConfig
from .gaze import gaze_error, gaze_points, solve_neck
from .placement import diagnostics as placement_diagnostics
from .targets import (approach_width, blend_orientation, contact_width, finger_angles, grasp_labels, gripper_state,
                      hand_object_distance, held_masks, idle_distance, manipulated_ids, object_centric,
                      orientation_weight, position_weight,
                      offset_track, place_labels, push_labels, release_ramp, release_starts, source_closed,
                      straight_approach,
                      tcp_targets, _runs)
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


def _ik(cfg, robot, targets, weights, labels, base_ref, base_free, q0, nominal, box=None, obstacles=None,
        pos_weights=None):
    """Two-pass whole-body IK on the source clock (free orientation away from grasps and, with
    ``pos_weights``, free position of idle hands; then strict tracking of the blended reference) and
    smoothing; returns q, the blended targets and stats."""
    targets = dict(targets)
    pos_weights = pos_weights or {}
    relaxed = {s for s, w in weights.items() if (w < 1).any()} | {s for s, w in pos_weights.items() if (w < 1).any()}
    orient = {}
    if relaxed:
        # Pass 1: orientation (and idle position) weights lowered away from grasps; its achieved TCP
        # pose becomes the reference there (blended into the source pose near grasps), tracked strictly.
        q_free, _ = solve_trajectory(cfg, targets, base_ref, base_free, q0, nominal, rot_scale=weights,
                                     max_iter=cfg.ik_max_iter if pos_weights else cfg.ik_free_iter, box=box,
                                     obstacles=obstacles,
                                     pos_scale=pos_weights or None)
        fk_free = robot.fk(q_free)
        for side in sorted(relaxed):
            src_t = targets[side]
            X_free = fk_free[f"{side}_tcp"]
            targets[side] = blend_orientation(src_t, X_free, weights[side])
            dev = np.linalg.norm(so3_log(np.swapaxes(src_t[:, :3, :3], 1, 2) @ targets[side][:, :3, :3]), axis=1)
            orient[side] = {"relaxed_frames": int(np.sum(weights[side] < 1)),
                            "max_reference_deviation_rad": float(dev.max())}
            if side in pos_weights:
                # an idle hand's reference is where the weighted pass put it (feasible by construction;
                # a linear blend toward an unreachable source point inside the ramp is not)
                wp = pos_weights[side]
                targets[side][:, :3, 3] = np.where((wp < 1)[:, None], X_free[:, :3, 3], src_t[:, :3, 3])
                shift = np.linalg.norm(targets[side][:, :3, 3] - src_t[:, :3, 3], axis=1)
                orient[side].update(idle_frames=int(np.sum(pos_weights[side] < 1)),
                                    idle_shifted_frames=int(np.sum(shift > cfg.tcp_pos_tol)),
                                    max_reference_shift_m=float(shift.max()))
    q, iters = solve_trajectory(cfg, targets, base_ref, base_free, q0, nominal, box=box, obstacles=obstacles)
    q, smoothed = smooth(cfg, q, targets, base_free, obstacles)
    return {"q": q, "iters": iters, "smoothed": smoothed, "targets": targets, "orientation": orient,
            "bad_frames": _bad_frames(cfg, q, targets, labels)}


def retarget(src: SourceEpisode, cfg: RetargetConfig | None = None) -> RetargetResult:
    """Retarget one source episode onto Reachy 2 (default configuration if ``cfg`` is None)
    and run tier K."""
    cfg = cfg or RetargetConfig()
    t_start = _time.perf_counter()
    robot = Reachy.load()
    dropped = []
    if cfg.navigation_arms_rest and src.regime == "navigation" and src.base is not None and src.effectors:
        # Navigation: the source hand pose carries no task (RoboCasa NavigateKitchen holds the Panda
        # hand fixed on the base); only the base and the head move, the arms rest.
        dropped = sorted(src.effectors)
        src = replace(src, effectors={})
    closed_src = {k: source_closed(e, cfg, src.time) for k, e in src.effectors.items()}
    labels_src = {k: grasp_labels(e.pose[:, :3, 3], closed_src[k], src.objects, cfg, effector=e, times=src.time)
                  for k, e in src.effectors.items()}
    place_diag = {}
    if cfg.place_drops and cfg.object_centric:
        # short drops after a release are placed: the hand follows the object to rest, then opens
        for k in src.effectors:
            labels_src[k], place_diag[k] = place_labels(src, k, labels_src[k], cfg)
    oc_diag, push_diag, straight_diag = {}, {}, {}
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
            placed = {e for _, e, _, _ in place_diag.get(k, ())}
            pose, oc_diag[k] = object_centric(src, k, lab, cfg, force_ends=placed)
            if cfg.straight_approach:
                # picks are approached straight along the approach axis (pushes are kept as they are)
                pose, straight_diag[k] = straight_approach(pose, src.time, labels_src[k], cfg, src.objects,
                                                           frozen=(lab >= 0) & (labels_src[k] < 0))
            effectors[k] = replace(e, pose=pose)
        src = replace(src, effectors=effectors)
    scene_obs = None
    if cfg.scene_footprint and src.scene is not None:
        # Static source-scene geometry (house walls, counters, cabinets) for the base footprint.
        near = [e.pose[:, :2, 3] for e in src.effectors.values()]
        near += [src.base[:, :2]] if src.base is not None else []
        near += [src.base_hint[None, :2]] if src.base_hint is not None else []
        if near:
            scene_obs = footprint.scene_obstacles(src.scene, cfg, np.concatenate(near), cfg.scene_footprint_radius)
    try:
        tuck = stow_posture(cfg) if dropped else tuck_posture(cfg)
        assignment = assign(src, cfg, labels_src, scene_obs)
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
                                        hand_object_distance(src, key, cfg.orientation_articulated,
                                                             cfg.orientation_articulated_motion)
                                        if cfg.orientation_by_distance else None)
               for key, side in sides.items()}
    pos_weights = ({side: position_weight(src.time, labels[side], cfg, idle_distance(src, key, cfg))
                    for key, side in sides.items()} if cfg.idle_position and place.mobile else {})
    static_obs = footprint.obstacles(src.objects, cfg, static_only=True)
    if scene_obs is not None:
        static_obs = footprint.merge_scene(static_obs, scene_obs)
    rest_arm = place.mobile and len(targets) < 2
    if rest_arm:  # a resting arm travels beside the torso
        static_obs.bands = footprint.BODY_PROFILE + (footprint.arm_band(tuck),)
    # a mobile base keeps its footprint clear of static geometry (linearized IK constraint)
    mobile_obs = static_obs if place.mobile and (static_obs.polygons or len(static_obs.points)) else None
    fixed = _ik(cfg, robot, targets, weights, labels, base_ref, place.mobile, q0, nominal, pos_weights=pos_weights,
                obstacles=mobile_obs)
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
        if scene_obs is not None:
            obs = footprint.merge_scene(obs, scene_obs)
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
        col = GRIPPERS.start + SIDES.index(side)
        q_src[:, col] = finger_angles(
            src.effectors[key], labels[side] >= 0, cfg, contact_width(src, key, offsets[side], labels[side]))
        if cfg.approach_narrow:
            cap = approach_width(src, key, offsets[side], labels[side], cfg)
            q_src[:, col] = np.minimum(q_src[:, col], gripper.width_to_angle(cap))
        if cfg.place_hold_squeeze:
            # placed drops: the source fingers open while Reachy carries the object down; keep the
            # squeeze of the source release until the object has settled (it opens there, see below)
            for b, e, _, _ in place_diag.get(key, ()):
                q_src[b:e, col] = q_src[b - 1, col]
    ids = manipulated_ids(src.objects)
    fk = robot.fk(q_src)
    hands = {side: fk[f"{side}_grasp"][:, :3, 3] for side in targets}
    points, rule = gaze_points(hands, labels, ids, src.objects, q_src[:, :3], cfg, src.time)
    q_src[:, NECK], head_ref_src = solve_neck(q_src, points, src.time, cfg)
    gaze_err = gaze_error(q_src, points)

    # Time scaling, resampling and refinement on the output clock.
    clock = timing.clock(src.time, q_src, cfg, timing.grasp_events(labels.values()),
                         timing.slide_positions(src) if cfg.slide_speed else None)
    q = timing.linear(clock, q_src)
    targets_out = {s: timing.poses(clock, X) for s, X in targets.items()}
    base_out = timing.linear(clock, base_ref)
    assist = base_free and not place.mobile
    # Refinement recovers interpolation residuals; output rows between two source rows whose position
    # the source-clock IK already left outside the tolerance (unreachable targets) are not re-solved (BiGym
    # SaucepanToHob: 42 of 68 s were spent re-solving such rows without effect).
    hopeless = None
    if targets:
        out_src = np.max([p for p, _ in tcp_errors(q_src, targets).values()], axis=0) > cfg.refine_skip_pos
        hopeless = out_src[clock.index] & out_src[np.minimum(clock.index + 1, len(out_src) - 1)]
    q, refined = refine(replace(cfg, w_base=cfg.w_base_assist) if assist else cfg, q, targets_out, base_out,
                        base_free, nominal, DT, obstacles=obs if assist else mobile_obs, skip=hopeless)
    grasp_out = np.full((len(q), 2), -1, dtype=np.int16)
    for side, lab in labels.items():
        grasp_out[:, SIDES.index(side)] = timing.labels(clock, lab)
    release_diag = {}
    if cfg.release_ramp:
        # Releases open at the finger speed limit once the source fingers leave the object.
        for side, lab in labels.items():
            col = GRIPPERS.start + SIDES.index(side)
            key = next(k for k, s in sides.items() if s == side)
            src_rows = release_starts(q_src[:, col], lab, cfg.release_start_angle, cfg.release_at_command)
            placed = sorted(e for _, e, _, _ in place_diag.get(key, ())) if cfg.place_hold_squeeze else []
            if placed:
                # a placed segment opens from its last held row (the source fingers are open by then)
                held = lab >= 0
                ends = np.flatnonzero(held[:-1] & ~held[1:]) + 1
                src_rows = [r for r in src_rows if int(ends[ends <= r + 1].max(initial=-1)) not in placed]
                src_rows = sorted(set(src_rows) | {e - 1 for e in placed if e < len(lab)})
            starts = [int(np.searchsorted(clock.source_time, src.time[r], side="right")) - 1 for r in src_rows]
            exact_out = {int(np.searchsorted(clock.source_time, src.time[e - 1], side="right")) - 1 for e in placed}
            q[:, col], events = release_ramp(q[:, col], clock.time, starts, grasp_out[:, SIDES.index(side)],
                                             VELOCITY[col] * cfg.velocity_scale, exact=exact_out,
                                             rise=cfg.squeeze_angle + cfg.release_start_angle
                                             if cfg.release_at_command else 0.0)
            release_diag[side] = {"source_rows": src_rows, "output_rows": events}
    qd = timing.velocity(q)
    t_time = _time.perf_counter()
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
        "release_ramp": release_diag,
        "placed_drops": place_diag,
        "straight_approach": straight_diag,
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
               "grasp_offsets": _jsonable({s: {"flip": bool(o[0]), "theta_deg": o[1], "tilt_deg": o[2],
                                               "lift_m": o[3] if len(o) > 3 else 0.0}
                                           for s, o in place.offsets.items()}),
               "grasp_offset_segments": _jsonable(placement_diagnostics(place)["segment_grasp_offsets"]),
               "notes": _notes(src, sides) + ([f"navigation regime: source effectors {dropped} not retargeted, "
                                                "arms in the rest posture"] if dropped else [])})
    if rest_arm:
        episode.extra["footprint_bands"] = [list(b) for b in static_obs.bands]
    if scene_obs is not None and scene_obs.polygons:
        # the scene geometry near the final base path, for the tier-K footprint check
        verts, rad = footprint._packed(scene_obs)
        d = footprint._polygons_distance(q[:: max(1, len(q) // 400), :2], verts).min(axis=0) - rad
        keep = np.flatnonzero(d < 1.0)
        episode.extra["scene_footprint"] = {
            "polygons": [np.round(scene_obs.polygons[i], 4).tolist() for i in keep],
            "heights": [list(np.round(scene_obs.heights[i], 4)) for i in keep],
            "source": "colliding environment geoms of the source scene (footprint.scene_obstacles), "
                      "within 1 m of the base path"}
    k = kinematic.check(episode, cfg)
    episode.validation.update(k["frames"])
    episode.tier = {"K": {"passed": k["passed"], "reasons": k["reasons"]}, "P": None}
    episode.extra["tier_k_metrics"] = _jsonable(k["metrics"])
    diag["seconds"]["total"] = _time.perf_counter() - t_start
    return RetargetResult(episode, "ok", list(k["reasons"]), _jsonable(diag))
