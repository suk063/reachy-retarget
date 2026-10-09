"""TrackingReference -> ReachyEpisode (docs/tracking.md, "Pipeline").

1. First frame: ``q_start`` when given (synthetic references start at the robot's state), else a
   multi-start cold solve.
2. Whole-body IK of both TCPs on every row with full position and orientation weight (no
   relaxation, no grasp offsets): :class:`reachy_retarget.retarget.wbik.FrameSolver` with the base
   free, pulled toward the nominal base path (``w_base``) and boxed around it; then the light
   smoothing and residual refinement of :mod:`reachy_retarget.retarget.wbik`.
3. Neck: 3-DoF head-rotation tracking (:mod:`.neck`), rate-limited.
4. Fingers: reference opening, rate-limited at the finger speed.
5. Timing: references with ``retime`` (source paths) are slowed down where the speed limits need it
   and resampled to 50 Hz (:mod:`reachy_retarget.retarget.timing`), then refined on the output clock.
6. Episode assembly with the reference stored unmodified, and tier K (:mod:`.validate`).
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field, replace

import numpy as np

from ..retarget import timing
from ..retarget.wbik import FrameSolver, refine, smooth, tcp_errors
from ..robot import LOWER, UPPER, VELOCITY, Reachy, gripper
from ..robot.reachy import GRIPPERS, NECK
from ..schema.episode import DT, SIDES, ReachyEpisode, Reference
from ..schema.rotations import so3_exp, so3_log
from . import neck as neck_mod
from . import validate
from .config import TrackingConfig
from .reference import TrackingReference, jsonable


@dataclass
class TrackResult:
    episode: ReachyEpisode | None
    status: str                   # "ok" (a trajectory exists; see episode.tier) or "failed"
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _nominal(ref: TrackingReference) -> np.ndarray:
    """Posture regularization target: the start posture (keeps the IK on the start branch)."""
    if ref.q_start is not None:
        return ref.q_start.copy()
    from ..robot.resources import profile
    p = profile()["postures"]["ready"]
    q = np.zeros(22)
    q[3:10], q[10:17] = np.radians(p["left"]), np.radians(p["right"])
    return q


def solve_arms(ref: TrackingReference, cfg: TrackingConfig):
    """Whole-body IK on the reference clock: (q (T, 22), diagnostics).

    Every frame is boxed around the nominal base path; references that are not retimed are also
    boxed by the speed limits around the previous frame (``cfg.velocity_scale``), as a real tracker
    would move: a reference faster than Reachy is then a tracking residual, never a joint jump."""
    ik = cfg.ik()
    nominal = _nominal(ref)
    solver = FrameSolver(ik, SIDES, cfg.base_free, nominal)
    T = ref.length
    q = np.empty((T, 22))
    iters = np.zeros(T, int)
    box_lo, box_hi = LOWER.copy(), UPPER.copy()
    span = np.array([cfg.base_box_xy, cfg.base_box_xy, cfg.base_box_yaw])
    cur = nominal.copy() if ref.q_start is None else ref.q_start.copy()
    cur[:3] = ref.base[0]
    frame0 = {s: ref.tcp[s][0] for s in SIDES}
    cold = False
    if ref.q_start is None or solver.error(cur, frame0) > 1.0:
        cold = True
        cur = solver.cold_solve(cur, frame0, ref.base[0], ik.ik_first_max_iter)[0]
    vstep = VELOCITY[:20] * cfg.velocity_scale * float(np.median(np.diff(ref.time)))
    for t in range(T):
        box_lo[:3], box_hi[:3] = ref.base[t] - span, ref.base[t] + span
        if t and not ref.retime:
            # the largest world-axis box inside the body-axis speed square at the current heading
            c, s_ = abs(np.cos(cur[2])), abs(np.sin(cur[2]))
            vstep[:2] = VELOCITY[:2] * cfg.velocity_scale * float(np.median(np.diff(ref.time))) / (c + s_)
            box_lo[:20] = np.maximum(np.r_[ref.base[t] - span, LOWER[3:20]], cur[:20] - vstep)
            box_hi[:20] = np.minimum(np.r_[ref.base[t] + span, UPPER[3:20]], cur[:20] + vstep)
        limit = ik.ik_first_max_iter if t == 0 else ik.ik_max_iter
        cur, iters[t] = solver.solve(cur, {s: X[t] for s, X in ref.tcp.items()}, cur, ref.base[t], limit,
                                     box=(box_lo, box_hi))
        q[t] = cur
    kept = 0.0
    if ref.retime:  # speed-boxed frames are already continuous; smoothing could break the speed box
        q, kept = smooth(ik, q, ref.tcp, cfg.base_free)
    return q, {"cold_start": cold, "mean_iterations": float(iters.mean()), "max_iterations": int(iters.max()),
               "smoothing_kept_fraction": kept}


def _refine(ik, q, targets, base_ref, nominal, base_free=True):
    """:func:`reachy_retarget.retarget.wbik.refine` on the output clock, then refined rows that
    break a speed limit are put back (``refine`` keeps a base axis whose neighbours are already
    farther apart than its world-axis box but still moves the other axis, which can exceed the
    body-axis base limit)."""
    new, refined = refine(ik, q, targets, base_ref, base_free, nominal, DT)
    for _ in range(10):
        ratio = np.max(np.abs(timing.body_increments(new)) / DT / VELOCITY, axis=1)
        over = np.flatnonzero(ratio > 1.0)
        if not len(over):
            break
        rows = np.unique(np.r_[over, over + 1])
        changed = rows[np.any(new[rows] != q[rows], axis=1)]
        if not len(changed):
            break
        new[changed] = q[changed]
    return new, refined


def finger_angles(opening, times, start=None):
    """Finger angles (T, 2) of reference openings, rate-limited at 0.95 x the finger speed."""
    target = gripper.opening_to_angle(opening)
    out = target.copy()
    if start is not None:
        out[0] = start
    step = VELOCITY[GRIPPERS] * 0.95 * np.diff(times)[:, None]
    for t in range(1, len(out)):
        out[t] = np.clip(target[t], out[t - 1] - step[t - 1], out[t - 1] + step[t - 1])
    return out


def _resample_rot(c, R):
    """Slerp resampling of rotations (N, 3, 3) on a timing clock."""
    A, B = R[c.index], R[c.index + 1]
    rel = so3_log(np.swapaxes(A, 1, 2) @ B)
    return A @ so3_exp(c.fraction[:, None] * rel)


def track(ref: TrackingReference, cfg: TrackingConfig | None = None) -> TrackResult:
    cfg = cfg or TrackingConfig()
    ik = cfg.ik()
    robot = Reachy.load()
    t_start = _time.perf_counter()
    q_src, ik_diag = solve_arms(ref, cfg)
    if not np.isfinite(q_src).all():
        return TrackResult(None, "failed", ["IK produced non-finite joint values"], {"stage": "ik"})
    t_ik = _time.perf_counter()
    times = ref.time - ref.time[0]
    dt = np.diff(times)
    # without a start state (source paths) the neck starts where the first head target puts it
    start_neck = ref.q_start[NECK] if ref.q_start is not None else None
    if ref.head is not None:
        q_src[:, NECK] = neck_mod.solve(q_src, ref.head, margin=cfg.neck_margin, iterations=cfg.neck_iterations,
                                        start=start_neck, dt=dt * cfg.velocity_scale)
    else:
        q_src[:, NECK] = 0.0 if start_neck is None else start_neck
    start_finger = ref.q_start[GRIPPERS] if ref.q_start is not None else None
    q_src[:, GRIPPERS] = finger_angles(ref.opening, times, start_finger)

    targets, base_ref, head_ref, opening = ref.tcp, ref.base, ref.head, ref.opening
    timing_diag = None
    if ref.retime:
        # body-axis base speeds of the resampled path can exceed the per-interval bound when the base
        # turns within a dilated interval: re-time with a smaller speed fraction until they hold
        scaled = ik
        for _ in range(4):
            clock = timing.clock(times, q_src, scaled)
            q = timing.linear(clock, q_src)
            ratio = float(np.max(np.abs(timing.body_increments(q)) / DT / VELOCITY)) if len(q) > 1 else 0.0
            if ratio <= 0.999:
                break
            scaled = replace(scaled, velocity_scale=scaled.velocity_scale * 0.98 / ratio)
        targets = {s: timing.poses(clock, X) for s, X in ref.tcp.items()}
        base_ref = timing.linear(clock, ref.base)
        head_ref = None if ref.head is None else _resample_rot(clock, ref.head)
        out_time, source_time = clock.time, clock.source_time + ref.time[0]
        timing_diag = clock.diagnostics(times)
    else:
        if not np.allclose(dt, DT, atol=1e-6):
            raise ValueError(f"a reference that is not retimed must be sampled at {1 / DT:g} Hz")
        q = q_src
        out_time, source_time = times, ref.time.copy()
    q, refined = _refine(ik, q, targets, base_ref, _nominal(ref), cfg.base_free)
    if ref.retime and head_ref is not None:  # re-track the head on the refined base yaw
        q[:, NECK] = neck_mod.solve(q, head_ref, margin=cfg.neck_margin, iterations=cfg.neck_iterations,
                                    start=q[0, NECK], dt=np.diff(out_time) * cfg.velocity_scale)
    qd = timing.velocity(q)
    fkb = robot.fk_base(q)
    angles = q[:, GRIPPERS]
    head_world = None
    if head_ref is not None:
        # the reference head pose: tracked rotation at the achieved head position (not tracked)
        from ..robot import planar
        W = planar(q[:, 0], q[:, 1], q[:, 2])
        head_world = W @ fkb["head"]
        head_world[:, :3, :3] = head_ref
    errors = tcp_errors(q, targets)
    diag = {
        "ik": dict(ik_diag, refined_output_frames=int(refined), reference_frames=ref.length,
                   output_frames=len(q)),
        "timing": timing_diag,
        "max_tcp_position_residual_m": {s: float(e[0].max()) for s, e in errors.items()},
        "max_tcp_rotation_residual_rad": {s: float(e[1].max()) for s, e in errors.items()},
        "seconds": {"ik": t_ik - t_start, "total": None},
    }
    extra = {"tracking": jsonable(dict(ref.extra, frames={"head_tip_in_head": neck_mod.HEAD_TIP.tolist()},
                                       diagnostics=diag)),
             "tracking_config": cfg.to_dict(), "grasp_object_ids": [],
             "notes": ["pure tracking episode: no objects; the reference is tracked unmodified"]}
    validation = {"grasp_object": np.full((len(q), 2), -1, dtype=np.int16)}
    if ref.witness is not None and not ref.retime:
        validation["witness_q"] = ref.witness
    episode = ReachyEpisode(
        family="tracking", dataset=ref.dataset, episode_id=ref.episode_id, task=ref.task,
        time=out_time, q=q, qd=qd, tcp_base={s: fkb[f"{s}_tcp"] for s in SIDES}, head_base=fkb["head"],
        gripper_opening=gripper.angle_to_opening(angles), gripper_width=gripper.angle_to_width(angles),
        source_time=source_time, tier={"K": {"passed": False, "reasons": ["not validated"]}, "P": None},
        retarget_config=cfg.digest(), instruction=ref.instruction, regime=ref.regime, license=ref.license,
        provenance=dict(ref.provenance), lineage=dict(ref.lineage), variant_of=ref.variant_of,
        body_parts=ref.body_parts, reference=Reference(tcp=targets, head=head_world, base=base_ref),
        validation=validation, extra=extra)
    k = validate.check(episode, cfg)
    episode.validation.update(k["frames"])
    episode.tier = {"K": {"passed": k["passed"], "reasons": k["reasons"]}, "P": None}
    episode.extra["tier_k_metrics"] = jsonable(k["metrics"])
    diag["seconds"]["total"] = _time.perf_counter() - t_start
    episode.extra["tracking"]["diagnostics"]["seconds"]["total"] = diag["seconds"]["total"]
    return TrackResult(episode, "ok", list(k["reasons"]), jsonable(diag))
