"""Constant robot TCP attachment for an unchanged single-box grasp reference.

This optional calibration changes only right-hand targets. It does not solve IK,
execute physics, change gripper intent, or assign object states. The caller must
run every target through IK/collision admission before an actuator rollout.
"""

from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import brentq

from .pad_alignment import FINGERS, pad_surfaces
from .physics import initialize
from .store import json_write, sha256


def single_box_geometry(model, *, object_body, object_joint):
    """Inspect a compiled free rigid box without changing model or data.

    Eligibility depends only on physical shape and articulation, never dataset
    or task names. Explicit contact pairs count as collision geometry even when
    a geom has zero collision masks. Visual-only geometry is otherwise ignored.
    All returned values are JSON-safe, suitable for an admission artifact.
    """
    body = model.body(object_body).id
    joint = model.joint(object_joint).id
    if (body <= 0 or int(model.jnt_bodyid[joint]) != body
            or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE)):
        raise ValueError("Single-box calibration requires the declared free joint on the object root")
    descendants = {body}
    for child in range(body + 1, model.nbody):
        if int(model.body_parentid[child]) in descendants:
            descendants.add(child)
    joints = [i for i in range(model.njnt) if int(model.jnt_bodyid[i]) in descendants]
    if joints != [joint]:
        raise ValueError("Single-box calibration requires a rigid subtree without articulated descendants")
    paired = set(map(int, model.pair_geom1)) | set(map(int, model.pair_geom2))
    geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in descendants
             and (model.geom_contype[i] or model.geom_conaffinity[i] or i in paired)]
    if len(geoms) != 1 or int(model.geom_type[geoms[0]]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("Constant face calibration requires one verified source box collision geom")
    geom = geoms[0]
    half_size = model.geom_size[geom]
    if not np.isfinite(half_size).all() or np.any(half_size <= 0):
        raise ValueError("Box half-sizes must be finite and positive")

    def rotation(quaternion):
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        return matrix.reshape(3, 3)

    axes, center = rotation(model.geom_quat[geom]), model.geom_pos[geom].copy()
    child = int(model.geom_bodyid[geom])
    while child != body:
        local = rotation(model.body_quat[child])
        axes, center = local @ axes, model.body_pos[child] + local @ center
        child = int(model.body_parentid[child])
    return dict(eligible=True, object_body=object_body, object_body_id=body,
                object_joint=object_joint, object_joint_id=joint, object_joint_type="free",
                collision_geom=model.geom(geom).name, collision_geom_id=geom,
                box_half_size_m=half_size.tolist(), box_center_object_m=center.tolist(),
                box_axes_object=axes.tolist(), rigid_body_ids=sorted(descendants),
                rigid_body_names=[model.body(i).name for i in sorted(descendants)],
                eligibility="One unchanged source box collision geom in a free, nonarticulated object subtree")


def derive_attachment(hand, object_pose, jaw_tcp, midpoint_tcp, box_axes_object,
                      center_object, *, align_box_axis=True):
    """Return the constant right-multiplied SE(3) and its face selection."""
    hand, object_pose = np.asarray(hand, float), np.asarray(object_pose, float)
    axes = np.asarray(box_axes_object, float)
    center = np.asarray(center_object, float)
    jaw = np.array(jaw_tcp, dtype=float, copy=True)
    midpoint = np.asarray(midpoint_tcp, float)
    if (hand.shape != (4, 4) or object_pose.shape != (4, 4) or axes.shape != (3, 3)
            or center.shape != (3,) or jaw.shape != (3,) or midpoint.shape != (3,)
            or not all(np.isfinite(x).all() for x in (hand, object_pose, axes, center, jaw, midpoint))
            or np.linalg.norm(jaw) < 1e-10):
        raise ValueError("Finite rigid poses, box axes, pad midpoint and closing axis required")
    for rotation in (hand[:3, :3], object_pose[:3, :3], axes):
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0) or np.linalg.det(rotation) < .999999:
            raise ValueError("Frame rotations must be orthonormal and right-handed")
    jaw /= np.linalg.norm(jaw)
    current = object_pose[:3, :3].T @ hand[:3, :3] @ jaw
    projections = axes.T @ current
    index = int(np.argmax(np.abs(projections)))
    desired_axis = axes[:, index] * (1 if projections[index] >= 0 else -1)
    cosine = float(np.clip(current @ desired_axis, -1, 1))
    rotation = np.eye(3)
    if align_box_axis:
        cross = np.cross(current, desired_axis)
        skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
        rotation += skew + skew @ skew / (1 + cosine)
    goal = np.array(hand, copy=True)
    goal[:3, :3] = object_pose[:3, :3] @ rotation @ object_pose[:3, :3].T @ hand[:3, :3]
    target_midpoint = object_pose[:3, :3] @ center + object_pose[:3, 3]
    goal[:3, 3] = target_midpoint - goal[:3, :3] @ midpoint
    attachment = np.linalg.inv(hand) @ goal
    return attachment, dict(box_face_axis=index, closing_axis_before_object=current.tolist(),
                            closing_axis_after_object=(object_pose[:3, :3].T @ goal[:3, :3] @ jaw).tolist(),
                            rotation_change_rad=float(np.arccos(cosine)) if align_box_axis else 0.,
                            desired_pad_midpoint_object_m=center.tolist())


CONTACT_REFERENCES = ("pad_area_centroid", "inward_contact_band")


def inward_contact_band(model, data, *, band_m=.0005):
    """Inward-most distal collision vertices of each pad, in the TCP frame.

    ``pad_surfaces`` averages the flat inward-facing faces. When the opposing
    distal meshes are not parallel at the contact aperture, MuJoCo's convex
    mesh/box contacts occur only where the pads protrude furthest inward. This
    returns the centroid of every compiled collision vertex within ``band_m`` of
    each pad's maximum inward extent, along the same body-to-body inward axis
    used by ``pad_surfaces``. Read-only; geometry and state are unchanged.
    """
    if not np.isfinite(band_m) or not 0 < band_m <= .005:
        raise ValueError("Contact band must be a finite positive width up to 5 mm")
    tcp = model.site("r_arm_tip_tcp").id
    rotation, origin = data.site_xmat[tcp].reshape(3, 3), data.site_xpos[tcp]
    bodies = [model.body(name).id for name in FINGERS]
    points, counts, extents = [], [], []
    for index, body in enumerate(bodies):
        inward = data.xpos[bodies[1-index]]-data.xpos[body]
        inward /= np.linalg.norm(inward)
        vertices = []
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError("Contact band extraction requires the original distal collision mesh")
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices.append(model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T
                            + data.geom_xpos[geom])
        if not vertices:
            raise ValueError("No distal collision mesh")
        vertices = np.vstack(vertices)
        depth = vertices @ inward
        band = vertices[depth >= depth.max()-band_m]
        points.append(rotation.T @ (band.mean(0)-origin))
        counts.append(int(len(band)))
        extents.append(np.ptp((band-origin) @ rotation, axis=0).tolist())
    points = np.asarray(points)
    return {"points_tcp": points, "midpoint_tcp": points.mean(0), "band_m": float(band_m),
            "vertex_counts": counts, "band_extent_tcp_m": extents}


def apply(prepared, robot, output, *, align_box_axis=True, center_box=True,
          contact_reference="pad_area_centroid", contact_band_m=.0005, contact_offset_m=None,
          vertical_offset_m=0.):
    """Calibrate targets, then pass the result to feasible_maniskill.prepare.

    A single original box collision geom is required. The gap-matching gripper
    angle is recalculated from actual opposing pad surfaces; source gripper
    intent remains byte-identical. No per-object constants or physics tuning.

    ``contact_reference="inward_contact_band"`` (opt-in) moves the centered
    reference from the flat-pad area centroid toward the inward-most distal
    vertex band (the pads' measured taper direction), because measured
    original-model PickCube contacts occur only on that tip side. Without
    ``contact_offset_m`` the band centroid itself is centered; otherwise the
    reference lies the declared distance along that unit direction.

    ``vertical_offset_m`` (default zero) raises the centered target along world
    up at the anchor. For a horizontal closing axis this changes no gravity
    torque about that axis; it is an explicit robot-only fixture clearance.
    """
    if contact_reference not in CONTACT_REFERENCES:
        raise ValueError("Unknown pad contact reference")
    if contact_offset_m is not None and (not np.isfinite(contact_offset_m) or not 0 <= contact_offset_m <= .03):
        raise ValueError("Contact offset must lie in [0, 30] mm")
    if not np.isfinite(vertical_offset_m) or not -.015 <= vertical_offset_m <= .015:
        raise ValueError("Vertical offset must lie in [-15, 15] mm")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    model, _, manifest, _, _, original_details, q0, times, _, commands, hands, objects = prepared
    details = deepcopy(original_details)
    anchor = int(details["pad_alignment"]["retarget_anchor_frame"])
    if not 0 <= anchor < len(times):
        raise ValueError("Saved pad anchor is outside the unchanged source clock")
    object_info = manifest["objects"][details["object_id"]]
    eligibility = single_box_geometry(model, object_body=object_info["body"], object_joint=object_info["joint"])
    bid, geom = eligibility["object_body_id"], eligibility["collision_geom_id"]
    data = mujoco.MjData(model)
    initial_object_state = data.qpos.copy()
    initialize(model, data, robot, q0, manifest["mimics"], .63)
    box_axes = data.xmat[bid].reshape(3, 3).T @ data.geom_xmat[geom].reshape(3, 3)
    box_center = data.xmat[bid].reshape(3, 3).T @ (data.geom_xpos[geom] - data.xpos[bid])
    probe = pad_surfaces(model, data)
    jaw = probe["points_tcp"][1] - probe["points_tcp"][0]
    jaw /= np.linalg.norm(jaw)
    current = objects[anchor, :3, :3].T @ hands[anchor, :3, :3] @ jaw
    face = int(np.argmax(np.abs(box_axes.T @ current)))
    if align_box_axis:
        width = 2 * float(model.geom_size[geom, face])
    else:
        # Extent between parallel support planes in the current closing axis.
        # For a skewed box this differs from a line through an off-center point.
        width = 2 * float(np.abs(box_axes.T @ current) @ model.geom_size[geom])
    def gap(angle):
        initialize(model, data, robot, q0, manifest["mimics"], angle)
        return pad_surfaces(model, data)["gap_m"] - width
    if gap(-.06) * gap(2.) > 0:
        raise ValueError("Unchanged source box is outside the original gripper aperture")
    angle = float(brentq(gap, -.06, 2., xtol=1e-10))
    gap(angle)
    pad = pad_surfaces(model, data)
    jaw = pad["points_tcp"][1] - pad["points_tcp"][0]
    jaw /= np.linalg.norm(jaw)
    center = box_center if center_box else np.asarray(details["pad_alignment"]["section"]["center_object"])
    if vertical_offset_m:
        up = -np.asarray(model.opt.gravity, float)
        up /= np.linalg.norm(up)
        center = np.asarray(center, float)+objects[anchor, :3, :3].T @ up*float(vertical_offset_m)
    area_midpoint = np.asarray(pad["midpoint_tcp"], float)
    contact = None
    reference_midpoint = area_midpoint
    if contact_reference == "inward_contact_band":
        contact = inward_contact_band(model, data, band_m=contact_band_m)
        direction = contact["midpoint_tcp"]-area_midpoint
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6:
            raise ValueError("Pad contact band coincides with the area centroid; no taper direction")
        offset = distance if contact_offset_m is None else float(contact_offset_m)
        reference_midpoint = area_midpoint+direction/distance*offset
    attachment, selection = derive_attachment(hands[anchor], objects[anchor], jaw, reference_midpoint,
                                               box_axes, center, align_box_axis=align_box_axis)
    if selection["box_face_axis"] != face:
        raise ValueError("Gripper aperture changed the selected box face; explicit recalibration required")
    corrected = np.asarray(hands) @ attachment
    object_joint = model.joint(manifest["objects"][details["object_id"]]["joint"]).id
    qadr = int(model.jnt_qposadr[object_joint])
    if not np.array_equal(data.qpos[qadr:qadr + 7], initial_object_state[qadr:qadr + 7]):
        raise AssertionError("Pad calibration changed the source object reset")
    artifact = output / "attachment-targets.npz"
    np.savez_compressed(artifact, time_s=times, original_hand_goals=hands, corrected_hand_goals=corrected,
                        original_object_goals=objects, original_gripper_intent=commands, attachment=attachment)
    report = dict(selection, shape_eligibility=eligibility,
                  align_box_axis=bool(align_box_axis), center_box=bool(center_box),
                  anchor_frame=anchor, anchor_time_s=float(times[anchor]),
                  original_collision_geom=model.geom(geom).name, box_half_size_m=model.geom_size[geom].tolist(),
                  box_center_object_m=box_center.tolist(), box_axes_object=box_axes.tolist(),
                  contact_width_m=width, inferred_contact_angle_rad=angle,
                  pad_midpoint_tcp_m=pad["midpoint_tcp"].tolist(), attachment=attachment.tolist(),
                  contact_reference=contact_reference, vertical_offset_m=float(vertical_offset_m),
                  contact_reference_midpoint_tcp_m=reference_midpoint.tolist(),
                  **({} if contact is None else dict(
                      contact_band={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in contact.items()},
                      contact_offset_m=float(np.linalg.norm(reference_midpoint-area_midpoint)))),
                  original_inferred_angle_rad=details["pad_alignment"]["inferred_contact_angle_rad"],
                  artifact_sha256=sha256(artifact), physics_validated=False,
                  joint_reference_valid=False, requires_full_ik_and_collision_admission=True,
                  policy="One constant right-multiplied robot TCP attachment on the complete original hand trajectory; object references, clock, geometry, gripper intent unchanged")
    json_write(output / "result.json", report)
    details["previous_pad_alignment"] = deepcopy(details["pad_alignment"])
    if contact is not None:
        # Downstream centering checks use the declared contact reference.
        pad = dict(pad, area_midpoint_tcp=area_midpoint, midpoint_tcp=reference_midpoint)
    details["pad_alignment"].update(inferred_contact_angle_rad=angle,
                                   pad={key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in pad.items()},
                                   section={"width_m": width, "center_object": np.asarray(center).tolist(),
                                            "collision_regions": [model.geom(geom).name]})
    details.update(grasp_attachment=report, alignment_variant="constant_tcp_attachment",
                   trajectory_seed="Original right-hand motion composed with one declared constant TCP attachment; IK pending")
    result = list(prepared)
    result[5], result[10] = details, corrected
    return tuple(result)
