"""Source effectors -> Reachy TCP targets, grasp labels and finger commands.

Adapters express every effector at its grasp center (approach +z, closing +y), and the
Reachy grasp center (``Reachy.grasp_center``) uses the same convention, so a TCP target is
``grasp_center_pose @ inv(T_tip_gc)``. Source and Reachy share the world frame, which
preserves the hand–object relative pose by construction.

Grasp re-selection (object-centric). The task is to reproduce the object trajectory, not the
source hand pose, so each side may use the source grasp-center frame composed with a constant
offset rotation ``G = (flip, theta, beta)``: ``E' = E @ Rz(pi * flip + theta) @ Ry(beta)``. The
origin (grasp center) is unchanged, so the object path is identical and the object pose in the
hand stays constant during every grasp. Allowed offsets:

* ``flip``: a half turn about the approach axis, always a symmetry of a parallel-jaw grasp;
* ``beta``: a tilt about the closing axis (+y). It leaves the pad planes (normal +y) unchanged,
  so the pads press the same object faces; only the contact patch moves on the pad. Limited to
  ``cfg.grasp_tilts_deg``;
* ``theta``: a rotation about the approach axis that aligns the closing axis with the nearest
  face normal of the grasped object's box (``align_deg``, see :func:`grasp_symmetry`; the pads
  then press flat on two faces), plus quarter turns when every grasped object is symmetric
  under them: the mean approach axis in the object frame lies within
  ``cfg.symmetry_axis_tol_deg`` of a principal axis of its box/aabb geometry and the two other
  half extents agree within ``cfg.symmetry_extent_tol`` (relative). A ``cylinder`` grasped
  along its axis (mean approach within ``cfg.symmetry_axis_tol_deg``, e.g. the robomimic Can
  from the top) is symmetric under any turn: no face alignment, and ``theta`` steps by
  ``cfg.cylinder_theta_step_deg``. The derived symmetry and alignment are recorded with the
  choice.

Object geometry: ``box``/``aabb`` records and body-frame ``cylinder`` records (radius,
half_length, axis); distances, contact widths and finger depths use the exact cylinder.

The choice is made by placement scoring and recorded as a derived label
(``extra["grasp_offsets"]``).
"""
from __future__ import annotations

import numpy as np

from ..robot import Reachy, angle_to_opening, angle_to_width, gripper
from ..schema.rotations import quat_to_matrix
from .config import RetargetConfig
from .footprint import box_geometry, cylinder_geometry

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])  # half turn about the grasp-center approach axis
IDLE_FINGER = gripper.CONTACT_ANGLE     # inactive hands: fingers just touching, no squeeze


def offset_matrix(offset) -> np.ndarray:
    """4x4 grasp-center offset of ``offset`` = bool flip or (flip, theta_deg, beta_deg)."""
    flip, theta, beta = (offset, 0.0, 0.0) if isinstance(offset, (bool, np.bool_)) else offset
    a, b = np.pi * bool(flip) + np.radians(theta), np.radians(beta)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1.0, 0], [-np.sin(b), 0, np.cos(b)]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Ry
    return M


def tcp_targets(src, sides, offsets=None):
    """World TCP targets {side: (T, 4, 4)} for ``sides`` = {effector key: side}.

    ``offsets`` = {side: bool flip or (flip, theta_deg, beta_deg)} applies a grasp offset
    (module docstring); missing sides use the source frame.
    """
    robot, offsets = Reachy.load(), offsets or {}
    return {side: robot.tcp_from_grasp_center(src.effectors[key].pose @ offset_matrix(offsets.get(side, False)), side)
            for key, side in sides.items()}


def grasp_symmetry(src, key, labels, cfg: RetargetConfig) -> dict:
    """Grasp geometry of effector ``key`` against the box/aabb of each grasped object (see the
    module docstring): ``{"quarter_turn": bool, "align_deg": float, "objects": {id: reason}}``.

    ``align_deg`` is the rotation about the approach axis that turns the closing axis onto the
    nearest object face normal (median over the grasp frames), applied when every grasped
    object agrees within ``cfg.align_agree_deg`` and it is below ``cfg.align_max_deg``. The source
    gripper's narrow pads tolerate misaligned grasps (the robomimic Panda holds the Lift cube
    10-13 deg off its faces); Reachy's 30 x 39 mm flat pads then touch two opposite edges and
    the object slowly turns between them.
    """
    ids = manipulated_ids(src.objects)
    lab = np.asarray(labels)
    out, ok, aligns = {}, bool((lab >= 0).any()), []
    rotational = ok
    E = src.effectors[key].pose
    for k in np.unique(lab[lab >= 0]):
        oid = ids[k]
        track = src.objects[oid]
        rows = np.flatnonzero((lab == k) & track.valid)
        box = box_geometry(track.geometry)
        if box is None or not len(rows):
            out[oid], ok, rotational = "no box geometry", False, False
            aligns.append(None)
            continue
        R = quat_to_matrix(track.pose[rows, 3:])
        a = np.einsum("tji,tj->ti", R, E[rows, :3, 2]).mean(axis=0)
        a /= np.linalg.norm(a)
        cyl = cylinder_geometry(track.geometry)
        if cyl is not None:
            angle = float(np.degrees(np.arccos(min(1.0, abs(a[cyl[1]])))))
            if angle <= cfg.symmetry_axis_tol_deg:
                out[oid] = f"approach {angle:.0f} deg from the cylinder axis {'xyz'[cyl[1]]}: symmetric under any turn"
                aligns.append(0.0)
                continue
        rotational = False
        axis = int(np.argmax(np.abs(a)))
        angle = float(np.degrees(np.arccos(min(1.0, abs(a[axis])))))
        other = np.delete(box[1], axis)
        rel = abs(other[0] - other[1]) / max(other.max(), 1e-9)
        sym = angle <= cfg.symmetry_axis_tol_deg and rel <= cfg.symmetry_extent_tol
        # signed rotation about the hand approach axis from the closing axis to the nearest face
        # normal (object axes other than the approach-like one), per grasp frame
        z, y = E[rows, :3, 2], E[rows, :3, 1]
        best = None
        for j in (j for j in range(3) if j != axis):
            n = R[:, :, j] - z * np.sum(R[:, :, j] * z, axis=1, keepdims=True)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            n *= np.sign(np.sum(n * y, axis=1, keepdims=True) + 1e-12)
            ang = np.degrees(np.arctan2(np.sum(np.cross(y, n) * z, axis=1), np.sum(y * n, axis=1)))
            med = float(np.median(ang))
            if best is None or abs(med) < abs(best):
                best = med
        aligns.append(best)
        out[oid] = (f"approach {angle:.0f} deg from object axis {'xyz'[axis]}, cross-section extents differ "
                    f"by {rel:.0%}: {'quarter-turn symmetric' if sym else 'half-turn only'}; closing axis "
                    f"{best:+.1f} deg from the nearest face normal")
        ok &= sym
    align = 0.0
    if aligns and all(a is not None for a in aligns) and np.ptp(aligns) <= cfg.align_agree_deg:
        mean = float(np.mean(aligns))
        align = mean if abs(mean) <= cfg.align_max_deg else 0.0
    return {"quarter_turn": ok, "rotational": rotational, "align_deg": round(align, 3), "objects": out}


def offset_candidates(symmetry: dict, cfg: RetargetConfig) -> list[tuple[bool, float, float]]:
    """Candidate grasp offsets (flip, theta_deg, beta_deg) for one side, source frame first."""
    a = float(symmetry.get("align_deg", 0.0))
    if symmetry.get("rotational"):
        thetas = tuple(a + t for t in np.arange(0.0, 180.0, cfg.cylinder_theta_step_deg))  # flips add the rest
    elif symmetry.get("quarter_turn"):
        thetas = (a, a + 90.0)
    else:
        thetas = (a,)
    out = [(f, t, float(b)) for b in sorted(cfg.grasp_tilts_deg, key=abs) for t in thetas for f in (False, True)]
    return out


def manipulated_ids(objects):
    """Sorted ids of the manipulated objects; grasp labels index into this list."""
    return sorted(k for k, o in objects.items() if o.role == "manipulated")


def object_distance(points, track, valid_rows=None):
    """Distance (T,) from points (T, 3) to an object: to its cylinder or box if ``geometry`` is a
    body-frame cylinder, box or aabb (:func:`.footprint.box_geometry`), else to its origin. NaN
    where the object is invalid."""
    valid = track.valid if valid_rows is None else valid_rows
    out = np.full(len(points), np.nan)
    if not valid.any():
        return out
    pose = track.pose[valid]
    rel = np.einsum("tji,tj->ti", quat_to_matrix(pose[:, 3:]), points[valid] - pose[:, :3])
    box, cyl = box_geometry(track.geometry), cylinder_geometry(track.geometry)
    if cyl is not None:
        radial, axial = _cylinder_coords(rel, cyl)
        rel = np.stack([np.maximum(radial - cyl[2], 0.0), np.maximum(np.abs(axial) - cyl[3], 0.0)], axis=-1)
    elif box is not None:
        rel = rel - box[0]
        rel = rel - np.clip(rel, -box[1], box[1])
    out[valid] = np.linalg.norm(rel, axis=-1)
    return out


def _cylinder_coords(rel, cyl):
    """(radial distance, axial coordinate) of object-frame points ``rel`` (..., 3) in a cylinder
    record ``(center, axis, radius, half_length)``."""
    center, axis = cyl[0], cyl[1]
    rel = rel - center
    axial = rel[..., axis]
    radial = np.linalg.norm(np.delete(rel, axis, axis=-1), axis=-1)
    return radial, axial


def grasp_labels(grasp_points, closed, objects, cfg: RetargetConfig):
    """Index of the grasped manipulated object per frame (-1 = none), for one hand.

    grasp_points: (T, 3) grasp-center positions; closed: (T,) bool commanded-closed state.
    A frame is grasping when the hand is closed and the nearest manipulated object lies
    within ``cfg.grasp_contact_distance`` (box surface or centre).
    """
    ids = manipulated_ids(objects)
    labels = np.full(len(grasp_points), -1)
    if not ids:
        return labels
    d = np.stack([object_distance(grasp_points, objects[k]) for k in ids])
    d = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(d, axis=0)
    hit = closed & (d[nearest, np.arange(len(nearest))] <= cfg.grasp_contact_distance)
    labels[hit] = nearest[hit]
    return labels


def source_closed(effector, cfg: RetargetConfig, times=None):
    """Closed (holding) state (T,) of a source effector.

    The hand holds an object once its fingers stall on it, not while they are still closing.
    With a recorded gripper command (``effector.command``, 1 = close), each run of command
    >= 0.5 is closed from the frame where the fingers, having started to close, no longer
    close faster than ``cfg.opening_rate`` per second (the first such frame more than
    ``cfg.closed_drop`` below the open level when there is one: robomimic Lift demo_4 pauses at
    opening 0.92 on the way down), up to the release command: the command
    fixes the grasp intent and its release, the opening only when contact is made (robomimic
    Lift: the command leads the stall by 6-8 frames while the hand still descends about 1 cm).

    Without a command the state is inferred from the opening: a parallel gripper that closes
    on an object stalls at the object's width, which may be well above half its stroke
    (robosuite's Panda holds the Lift cube at opening 0.52 and the Can at 0.62), so a fixed
    threshold misses most grasps. A frame is closed when the opening is below
    ``cfg.closed_opening``, or when it is more than ``cfg.closed_drop`` below the episode's
    open level (95th percentile) and stalled (|rate| <= ``cfg.opening_rate``), and in both cases
    not opening faster than ``cfg.opening_rate`` (the release, or the initial opening from a
    half-closed start state). Without ``times`` only the command or the threshold applies.
    """
    o = np.asarray(effector.opening, float)
    if effector.command is not None:
        cmd = np.asarray(effector.command, float) >= 0.5
        if times is None or len(o) < 3:
            return cmd
        closing = np.gradient(o, np.asarray(times, float)) < -cfg.opening_rate
        dropped = o < float(np.quantile(o, 0.95)) - cfg.closed_drop
        closed = np.zeros(len(o), bool)
        d = np.diff(np.r_[0, cmd.astype(int), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            moving = np.flatnonzero(closing[a:b])
            if not len(moving):          # already closed when commanded (or a stalled start)
                closed[a:b] = True
                continue
            m0 = a + moving[0]
            # prefer a stall well below the open level (a pause on the way down is not contact)
            stop = np.flatnonzero(~closing[m0:b] & dropped[m0:b])
            if not len(stop):
                stop = np.flatnonzero(~closing[m0:b])
            if len(stop):
                closed[m0 + stop[0]:b] = True
        return closed
    closed = o < cfg.closed_opening
    if times is None or len(o) < 3:
        return closed
    open_level = float(np.quantile(o, 0.95))
    rate = np.gradient(o, np.asarray(times, float))
    stalled = (o < open_level - cfg.closed_drop) & (np.abs(rate) <= cfg.opening_rate)
    return (closed | stalled) & (rate <= cfg.opening_rate)


def finger_angles(effector, grasping, cfg: RetargetConfig, contact_width=None):
    """Reachy finger angle (T,) for one source effector.

    With a source width: the angle reproducing that pad separation (clipped to Reachy's
    stroke), minus ``cfg.squeeze_angle`` while grasping so the fingers press on the object.
    While grasping, the separation is at most ``contact_width`` (T,) when given: the extent of
    the held object's box along Reachy's (re-selected) closing axis, so that a grasp turned
    flat onto the object's faces still closes on it (a misaligned source grasp stalls wider).
    Without width: the normalized opening mapped linearly onto the joint range.
    """
    if effector.width is None:
        return gripper.opening_to_angle(effector.opening)
    width = np.asarray(effector.width, float)
    if contact_width is not None:
        width = np.where(grasping, np.minimum(width, contact_width), width)
    angle = gripper.width_to_angle(width)
    return np.where(grasping, np.maximum(angle - cfg.squeeze_angle, gripper.LOWER), angle)


def contact_width(src, key, offset, labels) -> np.ndarray:
    """Extent (T,) of the held object's cylinder/box/aabb along the closing axis of effector ``key``'s
    grasp frame composed with ``offset`` (inf where nothing is held or without box geometry)."""
    ids = manipulated_ids(src.objects)
    lab = np.asarray(labels)
    out = np.full(len(lab), np.inf)
    y = (src.effectors[key].pose @ offset_matrix(offset))[:, :3, 1]
    for k in np.unique(lab[lab >= 0]):
        track = src.objects[ids[k]]
        box = box_geometry(track.geometry)
        rows = np.flatnonzero((lab == k) & track.valid)
        if box is None or not len(rows):
            continue
        yo = np.einsum("tji,tj->ti", quat_to_matrix(track.pose[rows, 3:]), y[rows])
        cyl = cylinder_geometry(track.geometry)
        if cyl is not None:
            c = np.minimum(np.abs(yo[:, cyl[1]]), 1.0)
            out[rows] = 2.0 * (cyl[2] * np.sqrt(1.0 - c ** 2) + cyl[3] * c)
        else:
            out[rows] = 2.0 * np.abs(yo) @ box[1]
    return out


def gripper_state(angles):
    """(opening (T, 2), width (T, 2)) of finger angles (T, 2)."""
    return angle_to_opening(angles), angle_to_width(angles)



SOLID_ROLES = ("manipulated", "support", "fixture")


def finger_penetration(src, key, side, offset, labels, finger, cfg: RetargetConfig) -> np.ndarray:
    """Per-frame depth (T,) by which Reachy's distal fingers, placed on effector ``key``'s
    grasp-center path composed with grasp ``offset`` and opened to ``finger`` (T,) angles,
    would sink into the box/aabb geometry of scene objects (0 = clear).

    Manipulated objects are skipped while this hand holds them (``labels``, the squeeze presses
    the pads into them by design); support and fixture boxes are always checked; receptacles
    (hollow bins whose aabb is an envelope) are not. Object aabbs are envelopes, so concave
    objects (e.g. a nut ring) report false depth; callers compare offsets relative to each other.
    """
    E = src.effectors[key].pose @ offset_matrix(offset)
    pts = gripper.finger_points(finger, side)
    world = np.einsum("tij,tnj->tni", E[:, :3, :3], pts) + E[:, None, :3, 3]
    ids = manipulated_ids(src.objects)
    depth = np.zeros(len(E))
    for name, track in src.objects.items():
        box = box_geometry(track.geometry)
        if box is None or track.role not in SOLID_ROLES:
            continue
        rows = track.valid.copy()
        if track.role == "manipulated" and name in ids:
            rows &= np.asarray(labels) != ids.index(name)
        if not rows.any():
            continue
        R = quat_to_matrix(track.pose[rows, 3:])
        rel = np.einsum("tji,tnj->tni", R, world[rows] - track.pose[rows, None, :3])
        cyl = cylinder_geometry(track.geometry)
        if cyl is not None:
            radial, axial = _cylinder_coords(rel, cyl)
            inside = np.minimum(cyl[2] - radial, cyl[3] - np.abs(axial))  # > 0 inside the cylinder
        else:
            inside = np.min(box[1] - np.abs(rel - box[0]), axis=-1)  # > 0 inside the box
        depth[rows] = np.maximum(depth[rows], np.maximum(inside, 0.0).max(axis=-1))
    return depth


def orientation_weight(times, labels, cfg: RetargetConfig) -> np.ndarray:
    """Orientation weight (T,) in [cfg.free_rot_weight, 1] of one hand's TCP target.

    1 inside every grasp window (a grasp segment of ``labels`` extended by
    ``cfg.approach_window_s`` before and ``cfg.retreat_window_s`` after it), decaying linearly
    to ``cfg.free_rot_weight`` over ``cfg.orientation_blend_s`` away from the windows. A hand
    without grasps keeps weight 1 (its orientation may matter for non-prehensile contact).
    """
    times = np.asarray(times, float)
    lab = np.asarray(labels) >= 0
    if not lab.any():
        return np.ones(len(times))
    d = np.diff(np.r_[0, lab.astype(int), 0])
    dist = np.full(len(times), np.inf)
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        lo, hi = times[a] - cfg.approach_window_s, times[b - 1] + cfg.retreat_window_s
        dist = np.minimum(dist, np.maximum(0.0, np.maximum(lo - times, times - hi)))
    w = 1.0 - dist / max(cfg.orientation_blend_s, 1e-9)
    return np.clip(w, cfg.free_rot_weight, 1.0)


def blend_orientation(X_src, X_free, w):
    """TCP targets (T, 4, 4) with the position of ``X_src`` and the rotation
    R_free @ exp(w log(R_free^T R_src)) (w = 1: source rotation, w = 0: ``X_free``'s)."""
    from ..schema.rotations import so3_exp, so3_log
    out = np.array(X_src, float)
    Rf = np.asarray(X_free)[:, :3, :3]
    rel = so3_log(np.swapaxes(Rf, 1, 2) @ out[:, :3, :3])
    out[:, :3, :3] = Rf @ so3_exp(np.asarray(w)[:, None] * rel)
    return out


def held_masks(objects, labels) -> dict[str, np.ndarray]:
    """{manipulated object id: (T,) bool} of the frames where any hand of ``labels`` (an iterable
    of grasp-label arrays) holds it."""
    ids = manipulated_ids(objects)
    labels = [np.asarray(lab) for lab in labels]
    if not labels:
        return {}
    return {oid: np.any([lab == k for lab in labels], axis=0) for k, oid in enumerate(ids)}


def _blend_pose(A, B, s):
    """Poses (T, 4, 4) interpolated from A (s = 0) to B (s = 1): linear position, geodesic rotation."""
    from ..schema.rotations import so3_exp, so3_log
    out = np.array(A, float)
    rel = so3_log(np.swapaxes(A[:, :3, :3], 1, 2) @ B[:, :3, :3])
    out[:, :3, :3] = A[:, :3, :3] @ so3_exp(np.asarray(s)[:, None] * rel)
    out[:, :3, 3] = A[:, :3, 3] + np.asarray(s)[:, None] * (B[:, :3, 3] - A[:, :3, 3])
    return out


def object_centric(src, key, labels, cfg: RetargetConfig):
    """Grasp-center path (T, 4, 4) of effector ``key`` that carries each held object rigidly.

    During a grasp segment in which the source object moves in its gripper by more than
    ``cfg.object_centric_fraction`` of the tier-K grasp tolerances (robomimic Square: the nut
    slides 5-44 mm while pushed onto the peg), the hand pose is the object pose composed with
    the grasp relative pose at the segment's ``cfg.grasp_settle_frames``-th frame, so the
    object path is reproduced with a constant pose in the hand. Segments the source holds
    rigidly keep the source hand path. Before and after a
    segment the world-frame correction between this path and the source hand path at the segment
    ends decays to identity over ``cfg.approach_window_s`` / ``cfg.retreat_window_s``.
    Returns (poses, {segment: max correction (m, rad)}).
    """
    from ..schema.rotations import se3_inv, so3_log, vec7_to_pose
    E = np.array(src.effectors[key].pose, float)
    out = E.copy()
    times = np.asarray(src.time, float)
    ids = manipulated_ids(src.objects)
    lab = np.asarray(labels)
    held = lab >= 0
    d = np.diff(np.r_[0, held.astype(int), 0])
    info = {}
    corrections = []  # (row, world correction 4x4, window seconds, direction)
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        k = lab[a]
        track = src.objects[ids[k]]
        if not track.valid[a:b].all() or np.any(lab[a:b] != k):
            continue
        O = vec7_to_pose(track.pose[a:b])
        r = a + min(cfg.grasp_settle_frames, b - a - 1)
        G = se3_inv(vec7_to_pose(track.pose[r])) @ E[r]
        seg = O @ G
        dev = se3_inv(E[a:b]) @ seg
        shift = float(np.linalg.norm(dev[:, :3, 3], axis=1).max())
        turn = float(np.linalg.norm(so3_log(dev[:, :3, :3]), axis=1).max())
        if shift <= cfg.object_centric_fraction * cfg.grasp_rel_pos_tol and \
                turn <= cfg.object_centric_fraction * cfg.grasp_rel_rot_tol:
            info[f"{ids[k]}@{a}-{b}"] = (0.0, 0.0)  # the source holds it rigidly: keep its hand path
            continue
        info[f"{ids[k]}@{a}-{b}"] = (shift, turn)
        out[a:b] = seg
        corrections.append((a, seg[0] @ se3_inv(E[a]), cfg.approach_window_s, -1))
        corrections.append((b - 1, seg[-1] @ se3_inv(E[b - 1]), cfg.retreat_window_s, 1))
    for row, C, window, direction in corrections:
        span = np.flatnonzero(~held & (direction * (times - times[row]) > 0)
                              & (direction * (times - times[row]) < window))
        if not len(span):
            continue
        s = 1.0 - np.abs(times[span] - times[row]) / window  # 1 at the segment end, 0 at the window edge
        corr = _blend_pose(np.broadcast_to(np.eye(4), (len(span), 4, 4)), np.broadcast_to(C, (len(span), 4, 4)), s)
        out[span] = corr @ out[span]
    return out, info
