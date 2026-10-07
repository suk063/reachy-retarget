"""Geometric gripper mapping and bounded mobile IK in an unchanged scene.

All results are derived targets. A small IK residual is not collision,
actuation, contact, or task-success validation. Source clocks and task objects
are immutable; the robot's planar placement/path may be corrected explicitly.
"""

from pathlib import Path
import json
import time

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.spatial.transform import Rotation

from .episodes import matrices_to_pose, pose_to_matrices


URDF_SHA256 = "63a1ecab3312a72c35f19a4bb1142005d74e4d7101da87822d02fea96696d448"
SOURCE_REVISIONS = {
    "robocasa": "456174f62b89b8fca99eaaf33949c29fec9cfc2a",
}


def tool_rotation(source_closing_axis, *, jaw_sign=1):
    """Map Reachy -Z approach/+Y jaw separation to a source +Z approach.

    Parallel jaws have an unsigned separation axis. The sign is one declared,
    constant 180-degree roll hypothesis; it is never changed frame by frame.
    """
    close = np.asarray(source_closing_axis, dtype=float) * jaw_sign
    if jaw_sign not in (-1, 1) or close.shape != (3,) or not np.isfinite(close).all():
        raise ValueError("A finite source closing axis and constant jaw sign are required")
    approach = np.array([0., 0., 1.])
    close = close - approach * (close @ approach)
    if np.linalg.norm(close) < 1e-8:
        raise ValueError("Closing and approach axes must be independent")
    y = close / np.linalg.norm(close)
    z = -approach
    return np.column_stack((np.cross(y, z), y, z))


def _pose(values, count=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 7 or (count is not None and len(values) != count)
            or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values[:, 3:], axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Aligned finite xyz/wxyz poses with unit quaternions are required")
    return pose_to_matrices(values)


def source_axes(record, dataset, hand_keys, *, jaw_sign=None):
    """Verify the pinned gripper and measured finger separation before mapping."""
    a, m = record["arrays"], record["metadata"]
    if dataset == "bigym":
        from .native_replay.bigym import SOURCE_REVISION
        expected = SOURCE_REVISION
        names = m.get("body_names", [])
        closing = np.array([0., -1., 0.])
        pairs = {side: (f"h1/robotiq_2f85_{side}/left_pad",
                        f"h1/robotiq_2f85_{side}/right_pad") for side in hand_keys}
        evidence = "Pinned Robotiq 2F85 pinch site lies on gripper +Z; native pad separation is -Y."
        source_files = ["bigym/envs/xmls/robotiq_2f85/2f85.xml"]
        url = "https://github.com/NeuracoreAI/bigym/blob/"
    elif dataset == "robocasa":
        expected = SOURCE_REVISIONS[dataset]
        if hand_keys != {"right": "source/right_tcp_pose"}:
            raise ValueError("Verified RoboCasa gripper contract covers the right grip site only")
        names = m.get("source_robot_body_names", [])
        closing = np.array([-1., 0., 0.])
        pairs = {"right": ("gripper0_right_finger_joint1_tip", "gripper0_right_finger_joint2_tip")}
        evidence = "Saved source MJCF: grip_site on eef +Z; finger Y slide rotated 90 degrees gives unsigned X separation."
        source_files = []
        url = "https://github.com/robocasa/robocasa/blob/"
    else:
        raise ValueError("No verified source gripper geometry for " + dataset)
    if m.get("source_revision") != expected:
        raise ValueError("Source gripper calibration requires the verified native revision")
    bodies = np.asarray(a.get("source/robot_body_poses"))
    if bodies.ndim != 3 or bodies.shape[1:] != (len(names), 7):
        raise ValueError("Measured native robot-body poses are required for axis verification")
    rotations, result = {}, {}
    for side, key in hand_keys.items():
        left, right = pairs[side]
        if left not in names or right not in names:
            raise ValueError("Missing measured finger bodies for " + side)
        hand = _pose(a[key], len(bodies))
        separation = bodies[0, names.index(left), :3] - bodies[0, names.index(right), :3]
        local = hand[0, :3, :3].T @ separation
        local[2] = 0.  # Verify the jaw separation transverse to the documented approach.
        if np.linalg.norm(local) < 1e-6:
            raise ValueError("Initial source jaws cannot establish a closing axis")
        local /= np.linalg.norm(local)
        if abs(local @ closing) < .995:
            raise ValueError("Measured source closing axis contradicts the pinned model")
        sign = (jaw_sign or {}).get(side, 1)
        rotations[side] = tool_rotation(closing, jaw_sign=sign)
        result[side] = {"source_channel": key, "approach_axis_source": [0., 0., 1.],
            "unsigned_closing_axis_source": closing.tolist(), "measured_initial_closing_axis_source": local.tolist(),
            "measured_body_pair": [left, right], "jaw_sign": sign,
            "source_to_reachy_tcp_rotation": rotations[side].tolist(), "axis_evidence": evidence,
            "source_model_xml_sha256": m.get("source_model_xml_sha256"),
            "source_urls": [url + expected + "/" + p for p in source_files],
            "source_contact_semantics": "Documented grip/pinch site proxy; no measured contact patch is asserted"}
    return rotations, result


def _pad_surfaces(model, data, side):
    """Both hands: area-weighted opposing original distal collision faces."""
    import mujoco
    prefix = "l" if side == "left" else "r"
    tcp = model.site(prefix + "_arm_tip_tcp").id
    rotation, origin = data.site_xmat[tcp].reshape(3, 3), data.site_xpos[tcp]
    bodies = [model.body(prefix + suffix).id for suffix in ("_hand_distal_link", "_hand_distal_mimic_link")]
    points, areas = [], []
    for i, body in enumerate(bodies):
        inward = data.xpos[bodies[1-i]] - data.xpos[body]
        inward /= np.linalg.norm(inward)
        centers, weights = [], []
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError("Original distal collision meshes are required")
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            tri = vertices[model.mesh_face[start:start+count]]
            normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
            area = np.linalg.norm(normals, axis=1)
            selected = normals @ inward > .95 * area
            centers.extend(tri[selected].mean(1)); weights.extend(area[selected]/2)
        if not weights or sum(weights) < 1e-8:
            raise ValueError("No opposing collision faces for " + side)
        points.append(rotation.T @ (np.average(centers, axis=0, weights=weights)-origin))
        areas.append(float(sum(weights)))
    points = np.asarray(points)
    return {"points_tcp_m": points.tolist(), "midpoint_tcp_m": points.mean(0).tolist(),
            "gap_m": float(np.linalg.norm(points[1]-points[0])), "face_area_m2": areas}


def calibration_apertures(nominal_gripper_rad=None):
    """Explicit per-hand CAD sampling settings, never inferred grasp labels."""
    values = {"left": .63, "right": .63}
    if nominal_gripper_rad is not None:
        if not isinstance(nominal_gripper_rad, dict) or set(nominal_gripper_rad)-set(values):
            raise ValueError("Pad aperture calibration requires explicit left/right joint angles")
        values.update(nominal_gripper_rad)
    if any(not np.isfinite(v) or not -.06 <= v <= 2. for v in values.values()):
        raise ValueError("Pad calibration angle is outside the verified gripper sampling range")
    return values


def angle_for_pad_gap(gap_function, gap_m):
    """Invert the measured CAD aperture only inside its verified range."""
    if not np.isfinite(gap_m) or gap_m <= 0:
        raise ValueError("A finite positive observed object width is required")
    angles = np.linspace(-.06, 2., 17)
    gaps = np.asarray([gap_function(angle) for angle in angles])
    if not np.isfinite(gaps).all():
        raise ValueError("CAD aperture must be finite before inversion")
    # The fully commanded closure can geometrically cross the opposing face
    # centers. Invert only the physical opening branch above minimum aperture.
    index = int(np.argmin(gaps))
    lower = angles[index]
    if 0 < index < len(angles)-1:
        minimum = minimize_scalar(gap_function, bounds=(angles[index-1], angles[index+1]),
                                  method="bounded", options={"xatol": 1e-12})
        lower = float(minimum.x)
    branch = np.asarray([gap_function(angle) for angle in np.linspace(lower, angles[-1], 17)])
    if np.any(np.diff(branch) < -1e-8):
        raise ValueError("CAD opening branch must be monotone before inversion")
    if not branch[0] <= gap_m <= branch[-1]:
        raise ValueError("Observed object width lies outside the reachable CAD aperture")
    angle = brentq(lambda value: gap_function(value)-gap_m, lower, angles[-1], xtol=1e-12)
    if abs(gap_function(angle)-gap_m) > 1e-6:
        raise ValueError("Discontinuous CAD face selection cannot calibrate the requested aperture")
    return float(angle)


def calibrate_pads(root, robot, nominal_gripper_rad=None, target_gap_m=None):
    """Measure both hands of the pinned CAD; never use home hand orientation."""
    import mujoco
    import xml.etree.ElementTree as ET
    from .physics import prepare_robot, initialize
    tree, manifest = prepare_robot(Path(root))
    if manifest["source_urdf_sha256"] != URDF_SHA256:
        raise ValueError("Unverified Reachy tool-axis geometry")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    data = mujoco.MjData(model)
    nominal = calibration_apertures(nominal_gripper_rad)
    target_gap_m = {} if target_gap_m is None else dict(target_gap_m)
    if set(target_gap_m)-set(nominal) or set(target_gap_m) & set(nominal_gripper_rad or {}):
        raise ValueError("Supply one aperture definition per explicit left/right hand")
    for side, gap in target_gap_m.items():
        def measured_gap(angle):
            initialize(model, data, robot, robot.q, manifest["mimics"], grip=angle)
            return _pad_surfaces(model, data, side)["gap_m"]
        nominal[side] = angle_for_pad_gap(measured_gap, gap)
    evidence = {side: [] for side in ("left", "right")}
    for angle in sorted({-.06, .63, 2., *nominal.values()}):
        initialize(model, data, robot, robot.q, manifest["mimics"], grip=angle)
        for side in evidence:
            evidence[side].append(dict(gripper_rad=angle, **_pad_surfaces(model, data, side)))
    offsets, details = {}, {}
    for side, samples in evidence.items():
        selected = next(sample for sample in samples if sample["gripper_rad"] == nominal[side])
        mid = np.asarray(selected["midpoint_tcp_m"])
        offsets[side] = mid
        variation = max(np.linalg.norm(np.asarray(s["midpoint_tcp_m"])-mid) for s in samples)
        details[side] = {"nominal_gripper_rad": nominal[side], "constant_pad_midpoint_tcp_m": mid.tolist(),
            "nominal_gap_m": selected["gap_m"], "sampled_gripper_configurations": samples,
            "requested_object_width_m": target_gap_m.get(side),
            "sampled_open_nominal_closed": [next(sample for sample in samples if sample["gripper_rad"] == angle)
                                             for angle in (-.06, nominal[side], 2.)],
            "sampled_midpoint_variation_m": float(variation),
            "approach_axis_tcp": [0., 0., -1.], "unsigned_closing_axis_tcp": [0., 1., 0.]}
    return offsets, {"urdf_sha256": URDF_SHA256, "mesh_sha256": manifest["meshes"], "hands": details,
        "axis_evidence": "Pinned URDF arm_tip fixed transform: translation [0,0,0.10], rotation Rx(pi) from palm.",
        "method": "Area-weighted inward collision-face centers, 0.95 cosine threshold; fixed nominal aperture.",
        "scope": "CAD surface proxy, not calibrated deformable pad contact; sampled variation is not a continuous bound."}


def contact_references(arrays, home, base_key, hand_keys, rotations, offsets):
    """Apply one source-world gauge and preserve each source grip-site center."""
    if not hand_keys or set(hand_keys)-{"left", "right"}:
        raise ValueError("Explicit source hand channels required")
    base = _pose(arrays[base_key])
    yaw = np.arctan2(base[0, 1, 0], base[0, 0, 0])
    placement = np.eye(4)
    placement[:3, :3] = Rotation.from_euler("z", -yaw).as_matrix()
    placement[:2, 3] = -(placement[:3, :3] @ base[0, :3, 3])[:2]
    base = placement @ base
    targets = np.tile(home, (len(base), 1, 1, 1))
    contact = targets.copy()
    attachments = np.tile(np.eye(3), (2, 1, 1))
    active = []
    for side, key in hand_keys.items():
        h = ("left", "right").index(side)
        source = placement @ _pose(arrays[key], len(base))
        rotation, offset = np.asarray(rotations[side]), np.asarray(offsets[side])
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(rotation), 1.) or offset.shape != (3,)
                or not np.isfinite(offset).all()):
            raise ValueError("A proper constant geometric tool transform is required")
        contact[:, h] = source
        targets[:, h] = source
        targets[:, h, :3, :3] = source[:, :3, :3] @ rotation
        targets[:, h, :3, 3] -= np.einsum("nij,j->ni", targets[:, h, :3, :3], offset)
        attachments[h] = rotation
        active.append(h)
    return base, targets, contact, placement, attachments, tuple(sorted(active))


def free_phase_weights(arrays, hands, objects, center, specification):
    """Blend only in observed open/far phases, preserving every closed frame."""
    clock = np.asarray(arrays[specification["clock_key"]], dtype=float)
    channel = specification["gripper_command"]
    commands = np.asarray(arrays[channel["array"]])[:, channel["column"]]
    if (clock.shape != (len(hands),) or commands.shape != clock.shape
            or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0)):
        raise ValueError("Aligned increasing source clock and declared gripper commands required")
    closed = commands == channel["closed_value"]
    indices = np.flatnonzero(closed)
    if len(indices) == 0 or not closed[indices[0]:indices[-1]+1].all():
        raise ValueError("One verified contiguous closure phase is required")
    near, far = float(specification["clear_distance_m"]), float(specification["return_distance_m"])
    if not 0 < near < far:
        raise ValueError("Positive ordered geometric clearance distances required")
    com = objects[:, :3, 3]+np.einsum("nij,j->ni", objects[:, :3, :3], center)
    distance = np.linalg.norm(hands[:, :3, 3]-com, axis=1)
    first, last = int(indices[0]), int(indices[-1])
    before = np.flatnonzero((np.arange(len(hands)) < first-1) & (distance >= near))
    if not len(before):
        raise ValueError("No observed distant open approach available for blending")
    start, full = int(before[-1]), first-1
    def smooth(a, b):
        u = np.clip((clock-clock[a])/(clock[b]-clock[a]), 0, 1)
        return u*u*u*(10+u*(-15+6*u))
    weights = smooth(start, full)
    # Source release alone is not withdrawal. If the hand never clears the
    # object, keep the relocated strategy through the available tail.
    retreat_start = retreat_end = None
    after = np.flatnonzero((np.arange(len(hands)) > last) & (distance >= near))
    if len(after):
        candidate = int(after[0])
        ends = np.flatnonzero((np.arange(len(hands)) > candidate) & (distance >= far))
        if len(ends) and (distance[candidate:] >= near).all():
            retreat_start, retreat_end = candidate, int(ends[0])
            weights *= 1-smooth(retreat_start, retreat_end)
    if not np.all(weights[closed] == 1):
        raise ValueError("Blend must preserve the complete verified closed interval")
    return weights, dict(method="source_clock_quintic_open_far_phase_blend",
        specification=specification, approach_start_frame=start, full_before_closure_frame=full,
        closure_frames=[first, last], retreat_start_frame=retreat_start, retreat_end_frame=retreat_end,
        approach_clock_s=[float(clock[start]), float(clock[full])],
        closed_phase_weight_exactly_one=True, no_observed_withdrawal=retreat_start is None,
        minimum_closed_COM_distance_m=float(distance[closed].min()),
        clearance_scope="Source pinch-to-object COM distance only; not a hand-mesh or scene collision admission")


def gravity_aligned_rim(arrays, hand_keys, specification):
    """Relocate a planar-rim grasp above observed object COM, by geometry.

    The selected object-local rotation moves the contact position. The default
    also rotates the hand; explicit evidence can instead retain source hand
    orientation. An optional open/far-phase blend preserves distant approach.
    Only the selected robot hand reference changes. Native
    object poses and original hand arrays are never written. This is a new grasp
    strategy, not a small TCP calibration or a measured command.
    """
    side, key = specification["hand"], specification["object_key"]
    if side not in hand_keys or key not in arrays or not specification.get("geometry_evidence"):
        raise ValueError("Observed task object, selected hand and retained geometry evidence required")
    center = np.asarray(specification["center_of_mass_local_m"], dtype=float)
    normal = np.asarray(specification["plane_normal_local"], dtype=float)
    gravity = np.asarray(specification["gravity_world_m_s2"], dtype=float)
    if (any(v.shape != (3,) or not np.isfinite(v).all() for v in (center, normal, gravity))
            or not np.isclose(np.linalg.norm(normal), 1., atol=1e-7) or np.linalg.norm(gravity) == 0):
        raise ValueError("Finite COM, unit planar normal and nonzero gravity are required")
    objects = _pose(arrays[key])
    hands = _pose(arrays[hand_keys[side]], len(objects))
    frame = specification["anchor_frame"]
    if not isinstance(frame, (int, np.integer)) or not 0 <= frame < len(objects):
        raise ValueError("An observed anchor frame is required")
    obj, hand = objects[frame], hands[frame]
    local = obj[:3, :3].T @ (hand[:3, 3]-obj[:3, 3])
    radial = local-center
    up = obj[:3, :3].T @ (-gravity/np.linalg.norm(gravity))
    radial_plane, up_plane = radial-normal*(normal@radial), up-normal*(normal@up)
    if min(np.linalg.norm(radial_plane), np.linalg.norm(up_plane)) < 1e-6:
        raise ValueError("Rim radius and gravity projection must define a stable planar direction")
    x, y = radial_plane/np.linalg.norm(radial_plane), up_plane/np.linalg.norm(up_plane)
    angle = float(np.arctan2(normal@np.cross(x, y), np.clip(x@y, -1., 1.)))
    rotation = Rotation.from_rotvec(normal*angle).as_matrix()
    local_transform = np.eye(4)
    local_transform[:3, :3] = rotation
    local_transform[:3, 3] = center-rotation@center
    inverse = np.tile(np.eye(4), (len(objects), 1, 1))
    inverse[:, :3, :3] = objects[:, :3, :3].transpose(0, 2, 1)
    inverse[:, :3, 3] = -np.einsum("nij,nj->ni", inverse[:, :3, :3], objects[:, :3, 3])
    blend = None
    weights = np.ones(len(objects))
    if specification.get("free_phase_blend") is not None:
        weights, blend = free_phase_weights(arrays, hands, objects, center, specification["free_phase_blend"])
    transforms = np.tile(local_transform, (len(objects), 1, 1))
    transforms[:, :3, :3] = Rotation.from_rotvec(weights[:, None]*normal*angle).as_matrix()
    transforms[:, :3, 3] = center-np.einsum("nij,j->ni", transforms[:, :3, :3], center)
    relocated = objects @ transforms @ inverse @ hands
    orientation_policy = specification.get("orientation_policy", "rotate_with_position")
    if orientation_policy not in ("rotate_with_position", "preserve_source"):
        raise ValueError("An explicit supported hand orientation policy is required")
    if orientation_policy == "preserve_source":
        if not specification.get("orientation_evidence"):
            raise ValueError("Preserving orientation at a relocated contact requires geometry evidence")
        relocated[:, :3, :3] = hands[:, :3, :3]
    relocated[weights == 0] = hands[weights == 0]
    adjusted = dict(arrays)
    adjusted[hand_keys[side]] = matrices_to_pose(relocated)
    after = rotation@radial+center
    return adjusted, dict(method="constant_object_local_planar_rim_rotation_above_COM" if blend is None else "planar_rim_rotation_above_COM_with_open_far_phase_blend",
        hand=side, object_key=key, anchor_frame=int(frame), geometry_evidence=specification["geometry_evidence"],
        center_of_mass_local_m=center.tolist(), plane_normal_local=normal.tolist(),
        gravity_world_m_s2=gravity.tolist(), object_local_grasp_transform=local_transform.tolist(),
        rim_rotation_rad=angle, source_pinch_local_m=local.tolist(), derived_pinch_local_m=after.tolist(),
        anchor_grasp_relocation_m=float(np.linalg.norm(after-local)),
        max_hand_reference_translation_change_m=float(np.linalg.norm(relocated[:, :3, 3]-hands[:, :3, 3], axis=1).max()),
        original_source_arrays_preserved=True, object_reference_changes=False,
        free_phase_blend=blend, relocation_weight=weights.tolist(),
        orientation_policy=orientation_policy, orientation_evidence=specification.get("orientation_evidence"),
        semantics="Explicit new robot grasp position around unchanged observed task object; orientation follows the separately declared policy")


def reach_preflight(robot, base, targets, active, *, translation_limit_m=.5):
    """Optimistic full-frame necessary tip/wrist bounds, not an IK solution."""
    robot.fk(robot.q)
    model, data = robot.r.model, robot.r.data
    origin = data.oMi[model.getJointId("root_joint")]
    rows, report = {}, {}
    for hand in active:
        side, prefix = ("left", "l") if hand == 0 else ("right", "r")
        shoulder = model.getJointId(prefix+"_shoulder_pitch")
        shoulder_local = origin.rotation.T @ (data.oMi[shoulder].translation-origin.translation)
        frame = model.frames[robot.r.hands[hand]]
        current, length, visited = int(frame.parentJoint), 0., set()
        while current != shoulder:
            if current == 0 or current in visited:
                raise ValueError("Tool must descend from its declared shoulder")
            visited.add(current)
            length += float(np.linalg.norm(model.jointPlacements[current].translation))
            current = int(model.parents[current])
        attachment = np.eye(4)
        attachment[:3, :3], attachment[:3, 3] = frame.placement.rotation, frame.placement.translation
        offset = float(np.linalg.norm(frame.placement.translation))
        wrist = targets[:, hand] @ np.linalg.inv(attachment)
        summaries = {}
        for name, point, reach, tolerance in (
                ("tip", targets[:, hand, :3, 3], length+offset, .02),
                ("wrist", wrist[:, :3, 3], length, .02+2*offset*np.sin(.15/2))):
            outside_box = np.maximum(abs(point[:, :2]-base[:, :2, 3])-translation_limit_m, 0.)
            horizontal = np.maximum(np.linalg.norm(outside_box, axis=1)-np.linalg.norm(shoulder_local[:2]), 0.)
            excess = np.hypot(horizontal, point[:, 2]-shoulder_local[2])-reach
            bad = np.flatnonzero(excess > tolerance+1e-7)
            rows[side+"/"+name+"_reach_excess_m"] = excess
            summaries[name] = dict(maximum_chain_reach_m=reach, shoulder_height_m=float(shoulder_local[2]),
                max_necessary_reach_excess_m=float(excess.max()), admission_tolerance_m=tolerance,
                proven_reject_frames=bad.tolist(), proven_reject_count=len(bad))
        report[side] = summaries
    return rows, dict(hands=report, frames=len(targets),
        proven_infeasible=any(value["proven_reject_count"] for side in report.values() for value in side.values()),
        scope="Necessary optimistic reach bounds only; full allowed XY box plus unrestricted shoulder yaw. Wrist bound includes allowed tip position/orientation error. No collision or IK success claim.")


def solve_frames(robot, clock, base, targets, active, *, translation_limit_m=.5,
                 yaw_limit_rad=1.2, base_axis_speed_m_s=.6, base_yaw_speed_rad_s=1.2,
                 max_starts=8, max_evaluations=120, seed=527, audit=None, progress=None):
    """Warm TRF solve; deterministic bounded restarts only after a bad residual.

    Base speed constraints apply after the initial derived placement. Arm speed
    is measured/reported, not assumed dynamically achievable. A source jump can
    make the base tube/speed intersection empty; then retain the failure rather
    than violate speed by silently relocating the robot.
    """
    audit = [] if audit is None else audit
    clock = np.asarray(clock, dtype=float)
    if clock.ndim != 1 or len(clock) != len(base) or len(clock) < 2 or np.any(np.diff(clock) <= 0) or not np.isfinite(clock).all():
        raise ValueError("Finite increasing unchanged source clock required")
    if any(x <= 0 for x in (translation_limit_m, yaw_limit_rad, base_axis_speed_m_s, base_yaw_speed_rad_s, max_starts, max_evaluations)):
        raise ValueError("Positive bounded solver settings required")
    qids = np.concatenate([robot.arm_ids[7*h:7*(h+1)] for h in active])
    lower = np.asarray(robot.r.model.lowerPositionLimit)[qids]+.031
    upper = np.asarray(robot.r.model.upperPositionLimit)[qids]-.031
    base_reference = np.column_stack((base[:, :2, 3], np.unwrap(np.arctan2(base[:, 1, 0], base[:, 0, 0]))))
    rng = np.random.default_rng(seed)
    last = np.r_[base_reference[0], robot.q[qids]]
    limits = np.array([translation_limit_m, translation_limit_m, yaw_limit_rad])
    speed = np.array([base_axis_speed_m_s, base_axis_speed_m_s, base_yaw_speed_rad_s])
    qrows, errors, achieved, speeds, bound_failures = [], [], [], [], []

    def unpack(x):
        q = robot.q.copy()
        q[:4] = [x[0], x[1], np.cos(x[2]), np.sin(x[2])]
        q[qids] = x[3:]
        return q

    for index, target in enumerate(targets):
        lo, hi = base_reference[index]-limits, base_reference[index]+limits
        if index:
            reach = speed*(clock[index]-clock[index-1])
            lo, hi = np.maximum(lo, last[:3]-reach), np.minimum(hi, last[:3]+reach)
            if np.any(lo >= hi):
                bound_failures.append(index)
                lo, hi = last[:3]-reach, last[:3]+reach
        lo, hi = np.r_[lo, lower], np.r_[hi, upper]
        start = np.clip(last, lo+1e-10, hi-1e-10)
        best = None
        def residual(x):
            actual = robot.fk(unpack(x))
            task = np.concatenate([np.r_[actual[h, :3, 3]-target[h, :3, 3],
                .35*robot.pin.log3(target[h, :3, :3] @ actual[h, :3, :3].T)] for h in active])
            # Tiny continuity cost chooses a nearby solution inside redundancy.
            # It is far below the explicit residual admission thresholds.
            return np.r_[task, 1e-5*(x-last)]
        for trial in range(max_starts):
            if trial:
                start = np.r_[np.clip(last[:3], lo[:3]+1e-10, hi[:3]-1e-10), rng.uniform(lower, upper)]
            solved = least_squares(residual, start, bounds=(lo, hi), max_nfev=max_evaluations,
                                   ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q = unpack(solved.x)
            actual = robot.fk(q)
            pe = max(float(np.linalg.norm(actual[h, :3, 3]-target[h, :3, 3])) for h in active)
            re = max(float(np.linalg.norm(robot.pin.log3(target[h, :3, :3]@actual[h, :3, :3].T))) for h in active)
            score = max(pe/.02, re/.15)
            audit.append({"frame": index, "start": trial, "nfev": int(solved.nfev),
                          "position_error_m": pe, "rotation_error_rad": re, "qpos": q.tolist(),
                          "solver_status": int(solved.status)})
            if progress is not None:
                progress("trial", audit[-1])
            if best is None or score < best[0]:
                best = (score, solved.x.copy(), q.copy(), pe, re, actual.copy())
            # Restarts are reserved for failed frames, not every small residual.
            if best[0] <= 1:
                break
        _, x, q, pe, re, actual = best
        if index:
            speeds.append((x[:3]-last[:3])/(clock[index]-clock[index-1]))
        last = x
        qrows.append(q); errors.append([pe, re]); achieved.append(actual)
        if progress is not None:
            progress("frame", {"completed_frames": index+1, "qpos": qrows,
                                "errors": errors, "achieved": achieved})
    qrows, errors = np.asarray(qrows), np.asarray(errors)
    derived_base = np.column_stack((qrows[:, :2], np.unwrap(np.arctan2(qrows[:, 3], qrows[:, 2]))))
    return {"qpos": qrows, "errors": errors, "achieved": np.asarray(achieved),
        "base_xyyaw": derived_base, "base_correction_xyyaw": derived_base-base_reference,
        "base_interval_velocity_xyyaw": np.asarray(speeds), "base_tube_failure_frames": bound_failures,
        "joint_interval_velocity": np.diff(qrows[:, robot.arm_ids], axis=0)/np.diff(clock)[:, None], "audit": audit}


def retarget(record, root, dataset, episode_id, attempt, *, clock_key, base_key,
             hand_keys, gripper_targets=None, mapping_metadata=None, jaw_sign=None,
             pad_calibration_gripper_rad=None, pad_calibration_gap_m=None,
             pad_calibration_evidence=None, grasp_relocation=None,
             reject_proven_unreachable=True, **solver_settings):
    """Export all attempts and a common derived archive, including IK failures."""
    from .robot import Robot, ARMS
    from .mobile_retarget import _save_raw_ik
    from .agent_dataset import write_archive
    from .cluster_pipeline import atomic
    path = Path(attempt)
    path.mkdir(parents=True, exist_ok=True)
    if any((path/name).exists() for name in ("raw-ik.npz", "solver-attempts.json", "kinematic-preflight.json")):
        raise FileExistsError("Immutable feasible-pose attempt already exists")
    a, m = record["arrays"], record["metadata"]
    if (pad_calibration_gripper_rad is not None or pad_calibration_gap_m is not None) and not (isinstance(pad_calibration_evidence, dict) and pad_calibration_evidence):
        raise ValueError("An aperture calibration override needs explicit retained evidence")
    clock = np.asarray(a[clock_key])
    if not any(k.startswith("objects/") and k.endswith("/pose") for k in a):
        raise ValueError("Observed task-object poses are required")
    robot = Robot(path/"runtime-identity")
    rotations, axes = source_axes(record, dataset, hand_keys, jaw_sign=jaw_sign)
    offsets, pads = calibrate_pads(path/"pad-calibration", robot, pad_calibration_gripper_rad, pad_calibration_gap_m)
    pads["selection_evidence"] = pad_calibration_evidence
    home = robot.fk(robot.q)
    original_contacts = contact_references(a, home, base_key, hand_keys, rotations, offsets)[2]
    reference_arrays, relocation = (a, None) if grasp_relocation is None else gravity_aligned_rim(a, hand_keys, grasp_relocation)
    base, targets, contacts, placement, attachments, active = contact_references(
        reference_arrays, home, base_key, hand_keys, rotations, offsets)
    atomic(path/"geometric-calibration.json", {"source": axes, "reachy": pads, "grasp_relocation": relocation})
    bounds, preflight = reach_preflight(robot, base, targets, active,
        translation_limit_m=solver_settings.get("translation_limit_m", .5))
    np.savez_compressed(path/"reference-preflight.npz", original_clock_s=clock,
        target_hand_matrix=targets, source_base_reference_matrix=base, placement=placement, **bounds)
    atomic(path/"kinematic-preflight.json", preflight)
    if preflight["proven_infeasible"] and reject_proven_unreachable:
        result = dict(status="kinematic_preflight_rejected", kinematic_passed=False,
            retargeted=False, physics_validated=False, full_reference_frames=len(clock),
            preflight=preflight, source_group=m["source_group"],
            missing_fields=["full IK not attempted after a necessary reach-bound rejection"])
        atomic(path/"kinematic-validation.json", result)
        return result
    audit = []
    last_progress, last_checkpoint, completed = time.monotonic(), time.monotonic(), 0
    checkpoint_folder = path/"checkpoints"
    checkpoint_folder.mkdir()
    trial_stream = (path/"solver-trials.jsonl").open("x")
    def progress(event, values):
        nonlocal last_progress, last_checkpoint, completed
        now = time.monotonic()
        if event == "trial":
            trial_stream.write(json.dumps(values, allow_nan=False)+"\n")
            if now-last_progress >= 5:
                trial_stream.flush()
                atomic(path/"solver-progress.json", dict(status="running", completed_frames=completed,
                    source_frames=len(clock), trial_count=len(audit), latest_trial=values,
                    kinematic_passed=False, physics_validated=False))
                last_progress = now
        else:
            completed = values["completed_frames"]
            if completed % 100 == 0 or completed == len(clock) or now-last_checkpoint >= 15:
                np.savez_compressed(checkpoint_folder/f"frames-{completed:06d}.npz",
                    original_clock_s=clock[:completed], robot_qpos=np.asarray(values["qpos"]),
                    errors_m_rad=np.asarray(values["errors"]), achieved_hand_matrix=np.asarray(values["achieved"]))
                trial_stream.flush()
                last_checkpoint = now
    try:
        solved = solve_frames(robot, clock, base, targets, active, audit=audit, progress=progress, **solver_settings)
    except Exception as exc:
        atomic(path/"solver-attempts.json", {"attempts": audit, "failure": str(exc), "physics_validated": False})
        atomic(path/"solver-progress.json", dict(status="solver_failed", completed_frames=completed,
            source_frames=len(clock), trial_count=len(audit), failure=str(exc), physics_validated=False))
        raise
    finally:
        trial_stream.close()
    atomic(path/"solver-attempts.json", {"attempts": audit, "physics_validated": False})
    q, errors, achieved = solved["qpos"], solved["errors"], solved["achieved"]
    raw = _save_raw_ik(path, clock, q, errors, achieved, targets, base, placement, attachments, m,
                      clock_semantics=(mapping_metadata or {}).get("clock", "Unchanged input timestamps"))
    valid = (errors[:, 0] <= .02) & (errors[:, 1] <= .15)
    passed = bool(valid.all() and not solved["base_tube_failure_frames"])
    report = {"status": "kinematic_candidate" if passed else "kinematic_failed", "retargeted": True,
        "kinematic_passed": passed, "physics_validated": False, "frames": len(clock),
        "kinematic_valid_frames": int(valid.sum()), "active_hands": list(hand_keys),
        "max_position_error_m": float(errors[:, 0].max()), "max_rotation_error_rad": float(errors[:, 1].max()),
        "position_threshold_m": .02, "rotation_threshold_rad": .15,
        "solver_attempt_count": len(audit), "solver_settings": solver_settings,
        "preflight": preflight,
        "base_tube_failure_frames": solved["base_tube_failure_frames"],
        "base_max_abs_correction_xyyaw": np.abs(solved["base_correction_xyyaw"]).max(0).tolist(),
        "base_max_abs_interval_velocity_xyyaw": np.abs(solved["base_interval_velocity_xyyaw"]).max(0).tolist(),
        "base_max_planar_speed_m_s": float(np.linalg.norm(solved["base_interval_velocity_xyyaw"][:, :2], axis=1).max()),
        "joint_max_abs_interval_velocity_rad_s": np.abs(solved["joint_interval_velocity"]).max(0).tolist(),
        "placement": placement.tolist(), "source_axes": axes, "pad_calibration": pads,
        "grasp_relocation": relocation,
        "source_mapping": mapping_metadata or {}, "raw_ik_artifact": str(raw),
        "source_task_success": m.get("source_task_success"),
        "physical_limitations": ["No robot/scene or self-collision admission", "No actuator tracking or contact verification",
                                 "Arm velocity, acceleration and torque feasibility not enforced", "No inferred neck target"]}
    out = {"timestamp": clock, "derived/robot_qpos": q, "derived/joint_names": np.asarray(ARMS),
        "derived/robot_joint_position": q[:, robot.arm_ids], "derived/ik_error_m_rad": errors,
        "derived/base_pose_xyyaw": solved["base_xyyaw"], "derived/base_correction_xyyaw": solved["base_correction_xyyaw"],
        "derived/target_hand_pose": matrices_to_pose(targets), "derived/achieved_hand_pose": matrices_to_pose(achieved),
        "reference/source_base_pose": matrices_to_pose(base), "reference/source_contact_pose": matrices_to_pose(original_contacts),
        "source/original_base_pose": np.array(a[base_key], copy=True)}
    if relocation is not None:
        out["derived/relocated_contact_pose"] = matrices_to_pose(contacts)
        out["derived/grasp_relocation_weight"] = np.asarray(relocation["relocation_weight"])
    for side, key in hand_keys.items():
        out["source/original_"+side+"_contact_pose"] = np.array(a[key], copy=True)
    for key, value in a.items():
        if key.startswith("objects/"):
            out["source/"+key] = np.array(value, copy=True)
            if key.endswith(("/pose", "_pose")):
                out["reference/"+key] = matrices_to_pose(placement@_pose(value, len(clock)))
    for side, values in (gripper_targets or {}).items():
        values = np.asarray(values)
        if side not in hand_keys or values.shape != clock.shape or not np.isfinite(values).all():
            raise ValueError("Verified gripper target must align with its active hand and clock")
        out["derived/"+side+"_gripper_joint_target_rad"] = values
    meta = {"source_sequence": m["source_sequence"], "source_group": m["source_group"],
        "source_urls": m.get("source_urls", []), "source_revision": m.get("source_revision"),
        "source_metadata": m, "source_mapping": mapping_metadata or {}, "robot_identity": robot.identity,
        "objects": m["objects"], "status": report["status"], "kinematic_validation": report,
        "grasp_relocation": relocation,
        "derived_fields": {"derived/*": "Geometric contact-reference IK; desired targets, not measured or issued controls",
            "reference/*": "One common planar world gauge; original source references are retained under source/*",
            "timestamp": "Selected original clock without resampling or retiming"},
        "missing_fields": list(record.get("missing_fields", []))+["measured Reachy observations", "Reachy applied controls",
            "contact/physics validation", "source contact-patch to deformable Reachy pad calibration", "mapped Reachy neck target"],
        "simulation_assumptions": list(m.get("simulation_assumptions", []))+[
            "Robot planar placement/path is derived within recorded bounds; source task objects and clock are unchanged.",
            "Fixed CAD pad-face midpoint at each declared calibration angle; sampled aperture-dependent motion is recorded, not corrected during the trajectory.",
            "Native pinch/grip site is a geometric reference, not evidence of actual source contact.",
            "All documented source hands constrained; inactive hand arrays contain home placeholders.",
            "IK feasibility does not establish collision, contact, dynamics, or task success."]}
    if relocation is not None:
        meta["derived_fields"]["derived/relocated_contact_pose"] = "Explicit geometry-derived new robot grasp; original hand references remain separately retained"
        meta["simulation_assumptions"].append("Selected hand uses a declared object-local rim rotation above COM, optionally blended only in observed open/far phases; this is a new grasp, not original source contact.")
    archive = write_archive(root, dataset, episode_id+"-ik", out, meta)
    atomic(path/"kinematic-validation.json", report)
    atomic(path/"solver-progress.json", dict(status=report["status"], completed_frames=len(clock),
        source_frames=len(clock), trial_count=len(audit), kinematic_passed=passed, physics_validated=False))
    return dict(report, retarget_archive=str(archive))
