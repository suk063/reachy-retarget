"""Finite CAD pad alignment to an unchanged, verified cylindrical mesh wall.

This proposes one constant TCP attachment. It neither solves IK nor executes
physics, and every previous robot admission becomes invalid. The cylinder is a
geometric description of the existing MuJoCo convex mesh, never a replacement
collision primitive or an assignment to the object's state.
"""
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from .pad_alignment import FINGERS, pad_surfaces
from .physics import initialize
from .store import json_write, sha256


def _unit(vector):
    vector = np.array(vector, dtype=float, copy=True)
    length = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or length < 1e-10:
        raise ValueError("A finite nonzero direction is required")
    return vector / length


def mesh_wall(vertices):
    """Qualify a long round convex mesh using its actual parallel side facets.

    Repeated long hull edges identify the axis, avoiding PCA bias from unequal
    tessellation at the cap and internal geometry. Every side facet must have
    a common straight interval; the radial polygon must surround a near-circle.
    """
    vertices = np.asarray(vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Finite collision mesh vertices required")
    hull = ConvexHull(vertices)
    faces = hull.simplices
    edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]],
                                         faces[:, [2, 0]]]), axis=1), axis=0)
    vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    long = lengths >= .5 * float(pdist(vertices[hull.vertices]).max())
    directions = vectors[long] / lengths[long, None]
    if len(directions) < 8:
        raise ValueError("No repeated long cylindrical side edges")
    agreement = np.abs(directions @ directions.T) > 1 - 1e-8
    index = int(np.argmax(agreement.sum(1)))
    if agreement[index].sum() < 8:
        raise ValueError("Long mesh edges do not establish a common cylinder axis")
    selected = directions[agreement[index]]
    selected *= np.sign(selected @ directions[index])[:, None]
    axis = _unit(selected.mean(0))
    # Deterministic sign only; either physical axis direction describes the wall.
    axis *= 1 if axis[np.argmax(abs(axis))] >= 0 else -1
    sides = np.abs(hull.equations[:, :3] @ axis) < 1e-5
    normals = hull.equations[sides, :3]
    groups = {}
    for face, equation in zip(faces[sides], hull.equations[sides]):
        groups.setdefault(tuple(np.round(equation, 6)), []).extend(face)
    if len(groups) < 12:
        raise ValueError("Cylinder requires at least twelve distinct surrounding side facets")
    intervals = [(float((vertices[ids] @ axis).min()),
                  float((vertices[ids] @ axis).max())) for ids in groups.values()]
    lower, upper = max(x[0] for x in intervals), min(x[1] for x in intervals)
    radial_x = np.eye(3)[np.argmin(abs(axis))]
    radial_x = _unit(radial_x - axis * (radial_x @ axis))
    radial_y = np.cross(axis, radial_x)
    boundary = np.unique(vertices[faces[sides]].reshape(-1, 3), axis=0)
    xy = boundary @ np.c_[radial_x, radial_y]
    cx, cy, constant = np.linalg.lstsq(np.c_[2*xy, np.ones(len(xy))],
                                      (xy*xy).sum(1), rcond=None)[0]
    radius = float(np.sqrt(constant + cx*cx + cy*cy))
    residual = float(np.max(abs(np.linalg.norm(xy-[cx, cy], axis=1)-radius)))
    angles = np.sort(np.unique(np.round(np.arctan2(normals @ radial_y, normals @ radial_x), 7)))
    max_angle = float(np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])))
    if (not radius > 0 or residual > max(5e-5, radius*.005)
            or upper-lower < 2*radius or max_angle > np.pi/4 + 1e-6):
        raise ValueError("Mesh does not establish a complete long circular straight wall")
    center = radial_x*cx + radial_y*cy
    return dict(axis_object=axis.tolist(), axis_origin_object_m=center.tolist(),
                straight_wall_interval_m=[lower, upper], radius_fit_m=radius,
                max_radius_fit_residual_m=residual, side_facet_count=len(groups),
                max_side_normal_gap_rad=max_angle, hull_vertex_count=len(hull.vertices),
                mesh_vertex_count=len(vertices),
                geometry_scope="Actual compiled convex mesh side facets; fitted circle describes eligibility only")


def collision_mesh(model, data, *, object_body, object_joint):
    """Extract one unchanged rigid free-object mesh in its root body frame."""
    body, joint = model.body(object_body).id, model.joint(object_joint).id
    if (body <= 0 or int(model.jnt_bodyid[joint]) != body
            or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE)):
        raise ValueError("Cylindrical alignment requires the declared root free joint")
    descendants = {body}
    for child in range(body+1, model.nbody):
        if int(model.body_parentid[child]) in descendants:
            descendants.add(child)
    if [i for i in range(model.njnt) if int(model.jnt_bodyid[i]) in descendants] != [joint]:
        raise ValueError("Cylindrical alignment requires a nonarticulated rigid object")
    paired = set(map(int, model.pair_geom1)) | set(map(int, model.pair_geom2))
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in descendants
             and (model.geom_contype[g] or model.geom_conaffinity[g] or g in paired)]
    if len(geoms) != 1 or int(model.geom_type[geoms[0]]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise ValueError("Cylindrical alignment requires one original collision mesh")
    geom = geoms[0]
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T
    vertices = (vertices + data.geom_xpos[geom] - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
    report = mesh_wall(vertices)
    report.update(object_body=object_body, object_joint=object_joint,
                  collision_geom=model.geom(geom).name, collision_geom_id=geom,
                  eligible=True, rigid_body_ids=sorted(descendants))
    return vertices, report


def finite_pad_faces(model, data):
    """Return the same inward CAD triangles used by pad_surfaces, in TCP axes."""
    tcp = model.site("r_arm_tip_tcp").id
    rotation, origin = data.site_xmat[tcp].reshape(3, 3), data.site_xpos[tcp]
    bodies = [model.body(name).id for name in FINGERS]
    result = []
    for index, body in enumerate(bodies):
        inward = _unit(data.xpos[bodies[1-index]]-data.xpos[body])
        selected = []
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError("Original distal CAD collision meshes are required")
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            triangles = vertices[model.mesh_face[start:start+count]]
            normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            selected.extend(triangles[normals @ inward > .95*np.linalg.norm(normals, axis=1)])
        if not selected:
            raise ValueError("No finite inward CAD pad face")
        result.append((np.asarray(selected)-origin) @ rotation)
    return result


def longitudinal_axis(triangles, jaw):
    """Area-integrated face covariance, projected off the closing line."""
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                  triangles[:, 2]-triangles[:, 0]), axis=1)/2
    center = np.average(triangles.mean(1), axis=0, weights=area)
    tri = triangles-center
    sums = tri.sum(1)
    moments = (np.einsum('nvi,nvj->nij', tri, tri) + np.einsum('ni,nj->nij', sums, sums))/12
    projection = np.eye(3)-np.outer(jaw, jaw)
    covariance = projection @ np.average(moments, axis=0, weights=area) @ projection
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= values[-2]*1.02:
        raise ValueError("Finite pad face has no distinct longitudinal direction")
    return _unit(vectors[:, -1])


def dominant_pad_planes(faces):
    """Qualify area-dominant CAD plane normals without discarding finite edges."""
    normals, reports = [], []
    for triangles in faces:
        triangles = np.asarray(triangles, float)
        cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        area = np.linalg.norm(cross, axis=1)/2
        keep = area > 1e-12
        triangles, cross, area = triangles[keep], cross[keep], area[keep]
        directions = cross/(2*area[:, None])
        agreement = directions @ directions.T >= np.cos(np.deg2rad(1.))
        selected = agreement[int(np.argmax(agreement @ area))]
        normal = _unit(np.sum(cross[selected], axis=0))
        selected = directions @ normal >= np.cos(np.deg2rad(1.))
        normal = _unit(np.sum(cross[selected], axis=0))
        fraction = float(area[selected].sum()/area.sum())
        center = np.average(triangles[selected].mean(1), axis=0, weights=area[selected])
        residual = float(np.max(abs((triangles[selected]-center) @ normal)))
        spread = float(np.arccos(np.clip(directions[selected] @ normal, -1, 1)).max())
        if fraction < .95 or residual > .0005 or spread > np.deg2rad(1.)+1e-12:
            raise ValueError("No sufficiently dominant finite CAD pad plane (95% area, 1 degree, 0.5 mm)")
        normals.append(normal)
        reports.append(dict(normal_tcp=normal.tolist(), center_tcp_m=center.tolist(),
            area_fraction=fraction, selected_area_m2=float(area[selected].sum()),
            full_inward_area_m2=float(area.sum()), max_plane_vertex_residual_m=residual,
            max_selected_normal_angle_rad=spread))
    cross = np.cross(*normals)
    if np.linalg.norm(cross) < 1e-4:
        raise ValueError("Nearly parallel pad planes do not identify a unique common tangent")
    return _unit(cross), reports


def nearest_opposed_facets(vertices, axis, jaw):
    """Find the least axial rotation onto a qualified original side facet.

    A nearly opposite facet must exist in the actual convex mesh. This is a
    geometric proposal, not evidence of a physical contact patch or stability.
    """
    axis, jaw = _unit(axis), _unit(jaw)
    jaw = _unit(jaw-axis*(jaw @ axis))
    normals = ConvexHull(np.asarray(vertices, float)).equations[:, :3]
    normals = normals[abs(normals @ axis) < 1e-5]
    if len(normals) < 2:
        raise ValueError("No opposed straight-wall facets")
    normals -= np.outer(normals @ axis, axis)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    dots = normals @ normals.T
    partners = np.argmin(dots, axis=1)
    opposed_error = np.arccos(np.clip(-dots[np.arange(len(normals)), partners], -1, 1))
    eligible = np.flatnonzero(opposed_error <= np.deg2rad(.1))
    if not len(eligible):
        raise ValueError("No opposed straight-wall facets within 0.1 degree")
    angles = np.arctan2(np.cross(jaw, normals) @ axis, normals @ jaw)
    # Signed-angle tie breaking avoids dependence on hull triangle enumeration.
    index = min(eligible, key=lambda i: (round(abs(float(angles[i])), 12), float(angles[i])))
    angle = float(angles[index])
    correction = Rotation.from_rotvec(axis*angle).as_matrix()
    return correction, dict(signed_rotation_rad=angle,
        selected_normal_object=normals[index].tolist(),
        opposed_normal_object=normals[partners[index]].tolist(),
        opposed_normal_error_rad=float(opposed_error[index]),
        opposed_normal_tolerance_rad=float(np.deg2rad(.1)),
        eligible_side_triangles=len(eligible),
        scope="Minimum cylinder-axis rotation to an original convex side facet with an opposed facet; physical contact area remains unverified")


def derive_attachment(hand, object_pose, pad, faces, wall, vertices, original_center, *, end_margin_m=.001,
                      orientation_policy='jaw_level', azimuth_policy='source', azimuth_offset_deg=0.):
    """Align finite pads, optionally snapping azimuth to original opposed facets."""
    if not np.isfinite(end_margin_m) or not 0 <= end_margin_m <= .005:
        raise ValueError("Finite pad end margin must be between zero and five millimetres")
    hand, object_pose = np.asarray(hand, float), np.asarray(object_pose, float)
    for pose in (hand, object_pose):
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-10)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
                or np.linalg.det(pose[:3, :3]) < .999999):
            raise ValueError("Finite right-handed rigid anchor poses are required")
    jaw = _unit(np.asarray(pad['points_tcp'])[1]-np.asarray(pad['points_tcp'])[0])
    lengths = [longitudinal_axis(np.asarray(face), jaw) for face in faces]
    lengths[1] *= 1 if lengths[0] @ lengths[1] >= 0 else -1
    length = _unit(sum(lengths))
    axis = _unit(wall['axis_object'])
    before = object_pose[:3, :3].T @ hand[:3, :3]
    original_jaw = before @ jaw
    transverse = original_jaw-axis*(original_jaw @ axis)
    if np.linalg.norm(transverse) < .25:
        raise ValueError("Source closing line is too axial to retain a defined radial azimuth")
    desired_jaw = _unit(transverse)
    desired_length = axis * (1 if (before @ length) @ axis >= 0 else -1)
    if orientation_policy not in ('jaw_level', 'dominant_pad_planes'):
        raise ValueError("Unknown cylindrical grasp orientation policy")
    if azimuth_policy not in ('source', 'nearest_opposed_facets'):
        raise ValueError("Unknown cylindrical grasp azimuth policy")
    if not np.isfinite(azimuth_offset_deg) or abs(azimuth_offset_deg) > 180.:
        raise ValueError('Explicit cylindrical robot grasp azimuth must be within +/-180 degrees')
    # The CAD face is only slightly longer than it is wide. Forcing its longest
    # direction vertical can add a ~90-degree roll even when the finite shorter
    # dimension already supports the complete axial contact corridor. Start
    # with the unique shortest jaw-leveling rotation; add no unnecessary roll.
    cross = np.cross(original_jaw, desired_jaw)
    skew = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]],
                     [-cross[1], cross[0], 0.]])
    leveling = np.eye(3)+skew+skew @ skew/(1+original_jaw @ desired_jaw)
    desired_rotation = leveling @ before
    plane_report = None
    if orientation_policy == 'dominant_pad_planes':
        common, planes = dominant_pad_planes(faces)
        common *= 1 if (before @ common) @ axis >= 0 else -1
        current = before @ common
        cross = np.cross(current, axis)
        skew = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]],
                         [-cross[1], cross[0], 0.]])
        correction = np.eye(3)+skew+skew @ skew/(1+current @ axis)
        desired_rotation = correction @ before
        actual_jaw = desired_rotation @ jaw
        desired_jaw = _unit(actual_jaw-axis*(actual_jaw @ axis))
        normals = np.array([p['normal_tcp'] for p in planes])
        plane_report = dict(planes=planes, common_tangent_tcp=common.tolist(),
            normal_dot_cylinder_axis_before=(normals @ before.T @ axis).tolist(),
            normal_dot_cylinder_axis_after=(normals @ desired_rotation.T @ axis).tolist(),
            common_tangent_alignment_error_rad=float(np.arccos(np.clip((desired_rotation @ common) @ axis, -1, 1))),
            closing_axis_axial_residual=float(actual_jaw @ axis),
            scope="Area-dominant CAD plane tangency; all exact inward triangles remain in finite-patch checks")
    azimuth_report = None
    if azimuth_offset_deg:
        correction = Rotation.from_rotvec(axis*np.deg2rad(azimuth_offset_deg)).as_matrix()
        desired_rotation = correction @ desired_rotation
        desired_jaw = correction @ desired_jaw
    if azimuth_policy == 'nearest_opposed_facets':
        correction, azimuth_report = nearest_opposed_facets(vertices, axis, desired_jaw)
        desired_rotation = correction @ desired_rotation
        desired_jaw = correction @ desired_jaw
    fully_aligned = (np.c_[desired_jaw, desired_length, np.cross(desired_jaw, desired_length)]
                     @ np.c_[jaw, length, np.cross(jaw, length)].T)
    midpoint = np.asarray(pad['midpoint_tcp'], float)
    relative_faces = [(np.asarray(face)-midpoint) @ desired_rotation.T for face in faces]
    axial_offsets = [face @ axis for face in relative_faces]
    lower, upper = wall['straight_wall_interval_m']
    supported = [lower+end_margin_m-min(float(x.min()) for x in axial_offsets),
                 upper-end_margin_m-max(float(x.max()) for x in axial_offsets)]
    if supported[0] > supported[1]:
        raise ValueError("The finite CAD pads do not fit entirely on the straight wall")
    original_height = float(np.asarray(original_center, float) @ axis)
    height = float(np.clip(original_height, *supported))
    center = np.asarray(wall['axis_origin_object_m']) + height*axis
    # Exact unchanged polygon support width, not the fitted circle diameter.
    side_vertices = np.asarray(vertices)[(np.asarray(vertices) @ axis >= lower-1e-6)
                                        & (np.asarray(vertices) @ axis <= upper+1e-6)]
    projection = side_vertices @ desired_jaw
    lo, hi = float(projection.min()), float(projection.max())
    center += desired_jaw*((lo+hi)/2-center @ desired_jaw)
    goal = np.array(hand, copy=True)
    goal[:3, :3] = object_pose[:3, :3] @ desired_rotation
    goal[:3, 3] = object_pose[:3, :3] @ center + object_pose[:3, 3] - goal[:3, :3] @ midpoint
    attachment = np.linalg.inv(hand) @ goal
    world_faces = [face+center for face in relative_faces]
    bounds = [[float((face @ axis).min()), float((face @ axis).max())] for face in world_faces]
    # The opposed contact line must lie inside each actual finite projected face.
    tangent = np.cross(axis, desired_jaw)
    projected, corridors = [], []
    for index, face in enumerate(relative_faces):
        xy = face.reshape(-1, 3) @ np.c_[tangent, axis]
        hull = ConvexHull(xy)
        support = side_vertices[abs(projection-(lo if index == 0 else hi)) < 1e-6]
        lateral = (support-center) @ tangent
        # A polygon facet can contribute a whole support segment. Check both
        # segment ends, rather than substituting the fitted circle's tangent.
        coordinates = np.array([[float(lateral.min()), 0.], [float(lateral.max()), 0.]])
        margin = float(np.min(-(coordinates @ hull.equations[:, :2].T+hull.equations[:, 2])))
        if margin < end_margin_m:
            raise ValueError("The radial contact line is too close to a finite CAD pad edge")
        projected.append(margin)
        normal, offset = hull.equations[:, :2], hull.equations[:, 2]
        lows, highs = [], []
        for x in coordinates[:, 0]:
            intercept = normal[:, 0]*x+offset+end_margin_m
            slope = normal[:, 1]
            lows.append(float(np.max(-intercept[slope < -1e-10]/slope[slope < -1e-10])))
            highs.append(float(np.min(-intercept[slope > 1e-10]/slope[slope > 1e-10])))
        corridor = [max(lows)+height, min(highs)+height]
        if corridor[0] >= corridor[1]:
            raise ValueError("No finite supported axial contact corridor")
        corridors.append(dict(support_tangent_interval_m=coordinates[:, 0].tolist(),
                              supported_axis_interval_m=corridor))
    actual_jaw = desired_rotation @ jaw
    report = dict(closing_axis_before_object=original_jaw.tolist(), closing_axis_after_object=actual_jaw.tolist(),
                  transverse_width_direction_object=desired_jaw.tolist(),
                  orientation_policy=orientation_policy,
                  azimuth_policy=azimuth_policy, facet_azimuth=azimuth_report,
                  requested_azimuth_offset_deg=float(azimuth_offset_deg),
                  azimuth_offset_scope='Robot grasp around the unchanged original cylinder. Zero requested offset preserves the input azimuth before the separately declared native-facet projection; that projection can still change the final grasp angle. Changed grasps require fresh whole-path IK, fixture, contact and physical validation.',
                  longitudinal_axis_tcp=length.tolist(), longitudinal_axis_after_object=(desired_rotation @ length).tolist(),
                  pad_long_axis_alignment_applied=False,
                  pad_long_axis_alignment_reason="The shortest jaw-leveling rotation already passes complete finite-pad wall and contact-corridor checks; no extra pad roll is required",
                  rotation_if_longest_pad_axis_fully_aligned_rad=float(Rotation.from_matrix(fully_aligned @ before.T).magnitude()),
                  original_midpoint_height_m=original_height, selected_midpoint_height_m=height,
                  supported_midpoint_interval_m=supported, pad_axial_bounds_object_m=bounds,
                  finite_pad_end_margin_m=float(end_margin_m), projected_contact_line_edge_margin_m=projected,
                  finite_contact_corridors=corridors,
                  desired_pad_midpoint_object_m=center.tolist(), contact_width_m=hi-lo,
                  axial_contact_centroid_separation_m=float(abs(actual_jaw @ axis)*pad['gap_m']),
                  rotation_change_rad=float(Rotation.from_matrix(desired_rotation @ before.T).magnitude()),
                  static_finite_patch_checks=dict(full_pad_axial_extent_inside_straight_wall=True,
                      radial_contact_line_inside_finite_pad=True, jaw_perpendicular_to_cylinder_axis=True),
                  patch_scope="Geometric axial support and finite CAD face projection; actual contact forces and contact area require physical rollout")
    if plane_report is not None:
        report['dominant_pad_planes'] = plane_report
        report['static_finite_patch_checks'].pop('jaw_perpendicular_to_cylinder_axis')
        report['static_finite_patch_checks']['common_pad_plane_tangent_aligned_to_cylinder_axis'] = bool(
            np.max(np.abs(plane_report['normal_dot_cylinder_axis_after'])) < 1e-8)
        report['static_finite_patch_checks']['jaw_axial_residual_explicitly_reported'] = True
        report['pad_long_axis_alignment_reason'] = "Area-dominant finite pad planes set their common tangent; no longest-axis PCA roll is imposed"
    return attachment, report


def post_anchor_corridor(hands, objects, commands, anchor, faces, wall, *, end_margin_m=.001):
    """Intersect exact linear axial bounds for one constant TCP translation.

    Only the contiguous source closed-intent interval containing the original
    anchor is used, beginning at that anchor. This is a reference-geometry
    constraint, not an assertion that physical acquisition occurred there.
    """
    hands, objects, commands = np.asarray(hands), np.asarray(objects), np.asarray(commands)
    if (hands.ndim != 3 or hands.shape[1:] != (4, 4) or objects.shape != hands.shape
            or commands.shape != (len(hands),) or not 0 <= anchor < len(hands)
            or not all(np.isfinite(v).all() for v in (hands, objects, commands))
            or commands[anchor] >= 0):
        raise ValueError("A finite source grasp anchor inside recorded closed intent is required")
    axis = _unit(wall['axis_object'])
    release = anchor + next((i for i, value in enumerate(commands[anchor:]) if value >= 0),
                            len(commands)-anchor)
    indices = np.arange(anchor, release)
    h, obj = hands[indices], objects[indices]
    rotation = obj[:, :3, :3].transpose(0, 2, 1) @ h[:, :3, :3]
    translation = np.einsum('nji,nj->ni', obj[:, :3, :3], h[:, :3, 3]-obj[:, :3, 3])
    points = np.concatenate([np.asarray(face).reshape(-1, 3) for face in faces])
    axial = np.einsum('nki,vi,k->nv', rotation, points, axis) + (translation @ axis)[:, None]
    bounds = np.c_[axial.min(1), axial.max(1)]
    # The direction is constant in the corrected TCP frame, not in world axes.
    direction_tcp = hands[anchor, :3, :3].T @ objects[anchor, :3, :3] @ axis
    gain = np.einsum('nij,j->ni', rotation, direction_tcp) @ axis
    lower, upper = np.asarray(wall['straight_wall_interval_m']) + [end_margin_m, -end_margin_m]
    delta_bounds = np.empty((len(indices), 2))
    for i, response in enumerate(gain):
        lo, hi = lower-bounds[i, 0], upper-bounds[i, 1]
        if abs(response) <= 1e-12:
            if not (lo <= 1e-10 and hi >= -1e-10):
                raise ValueError("No constant TCP translation can support a zero-response source frame")
            delta_bounds[i] = [-np.inf, np.inf]
        else:
            delta_bounds[i] = sorted([lo/response, hi/response])
    interval = [float(delta_bounds[:, 0].max()), float(delta_bounds[:, 1].min())]
    if interval[0] > interval[1]+1e-10:
        raise ValueError("No common finite-pad axial support interval across the post-anchor grasp")
    delta = float(np.clip(0., *interval))
    after = bounds + gain[:, None]*delta
    if np.any(after[:, 0] < lower-1e-9) or np.any(after[:, 1] > upper+1e-9):
        raise AssertionError("The constant translation did not satisfy every declared axial bound")
    attachment = np.eye(4)
    attachment[:3, 3] = direction_tcp*delta
    report = dict(anchor_frame=int(anchor), closed_frame_stop_exclusive=int(release),
        completed_frames=len(indices), total_frames=len(indices),
        original_anchor_axis_translation_interval_m=interval, additional_anchor_axis_translation_m=delta,
        constant_translation_corrected_tcp_m=attachment[:3, 3].tolist(),
        minimum_axial_translation_response=float(gain.min()), maximum_axial_translation_response=float(gain.max()),
        pad_axial_bounds_before_m=[float(bounds[:, 0].min()), float(bounds[:, 1].max())],
        pad_axial_bounds_after_m=[float(after[:, 0].min()), float(after[:, 1].max())],
        allowed_wall_interval_with_guard_m=[float(lower), float(upper)],
        every_post_anchor_closed_reference_axially_supported=True,
        scope="Exact finite-pad axial reference bounds after the source anchor; physical acquisition, radial contact, IK and fixture clearance remain separate")
    arrays = dict(corridor_frame_indices=indices, corridor_axial_response=gain,
                  corridor_pad_axial_bounds_before_m=bounds, corridor_pad_axial_bounds_after_m=after,
                  corridor_per_frame_translation_interval_m=delta_bounds)
    return attachment, report, arrays


def axial_offset(hands, objects, commands, anchor, faces, wall, offset_m, *, end_margin_m=.001):
    """Qualify an explicit constant TCP shift against every closed source row.

    The caller supplies a geometry-derived offset; this function performs no
    search and never clamps an unsupported request. This is an axial finite
    surface check, not a physical grasp or robot/fixture admission.
    """
    if not np.isfinite(offset_m) or not 0 < abs(offset_m) <= .02:
        raise ValueError('Nonzero axial offset must be within +/-20 millimetres')
    if not np.isfinite(end_margin_m) or not 0 <= end_margin_m <= .005:
        raise ValueError('Finite pad end margin must be between zero and five millimetres')
    hands, objects, commands = map(np.asarray, (hands, objects, commands))
    if (hands.shape != objects.shape or hands.ndim != 3 or hands.shape[1:] != (4, 4)
            or commands.shape != (len(hands),) or not np.isfinite(hands).all()
            or not np.isfinite(objects).all() or not np.isfinite(commands).all()
            or not isinstance(anchor, (int, np.integer)) or not 0 <= anchor < len(hands)
            or commands[anchor] >= 0 or commands[0] < 0):
        raise ValueError('Axial offset requires aligned poses, an open initial row and a closed anchor')
    axis = _unit(wall['axis_object'])
    origin = np.asarray(wall['axis_origin_object_m'])
    direction = hands[anchor, :3, :3].T @ objects[anchor, :3, :3] @ axis
    indices = np.flatnonzero(commands < 0)
    vertices = np.concatenate(faces).reshape(-1, 3)
    lower, upper = np.asarray(wall['straight_wall_interval_m']) + [end_margin_m, -end_margin_m]
    before, gains, intervals = [], [], []
    for index in indices:
        relative = np.linalg.inv(objects[index]) @ hands[index]
        axial = (vertices @ relative[:3, :3].T + relative[:3, 3] - origin) @ axis
        gain = float(axis @ relative[:3, :3] @ direction)
        extent = [float(axial.min()), float(axial.max())]
        lo, hi = lower-extent[0], upper-extent[1]
        if abs(gain) < 1e-12:
            if lo > 1e-10 or hi < -1e-10:
                raise ValueError('Closed finite pad has unsupported zero-response axial bounds')
            interval = [-np.inf, np.inf]
        else:
            interval = sorted([lo/gain, hi/gain])
        before.append(extent); gains.append(gain); intervals.append(interval)
    before, gains, intervals = map(np.asarray, (before, gains, intervals))
    allowed = [float(intervals[:, 0].max()), float(intervals[:, 1].min())]
    after = before + gains[:, None]*offset_m
    if (offset_m < allowed[0]-1e-10 or offset_m > allowed[1]+1e-10
            or np.any(after[:, 0] < lower-1e-10) or np.any(after[:, 1] > upper+1e-10)):
        raise ValueError('Requested axial offset exceeds full closed finite-pad support; no clamping')
    shift = np.eye(4)
    shift[:3, 3] = direction*offset_m
    report = dict(requested_offset_m=float(offset_m), supported_offset_interval_m=allowed,
        closed_source_rows=indices.tolist(), completed_frames=len(indices), total_frames=len(indices),
        constant_translation_tcp_m=shift[:3, 3].tolist(),
        finite_pad_axial_bounds_before_m=[float(before[:, 0].min()), float(before[:, 1].max())],
        finite_pad_axial_bounds_after_m=[float(after[:, 0].min()), float(after[:, 1].max())],
        allowed_wall_interval_with_guard_m=[float(lower), float(upper)],
        every_closed_source_row_axially_supported=True,
        scope='Explicit robot attachment only; every closed source row checked against unchanged native wall and finite CAD faces; fresh complete IK/fixture and physical validation required')
    arrays = dict(axial_offset_closed_source_rows=indices, axial_offset_response=gains,
                  axial_offset_bounds_before_m=before, axial_offset_bounds_after_m=after,
                  axial_offset_per_frame_interval_m=intervals)
    return shift, report, arrays


def apply(prepared, robot, output, *, end_margin_m=.001, height_policy='anchor', orientation_policy='jaw_level',
          azimuth_policy='source', azimuth_offset_deg=0., axial_offset_m=0.,
          preserve_pre_offset_initial_target=False):
    """Propose one geometry-derived attachment, keeping all source arrays exact."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/Path(__file__).name).write_text(Path(__file__).read_text())
    model, _, manifest, _, _, old_details, q0, times, _, commands, hands, objects = prepared
    details = deepcopy(old_details)
    if (not np.isfinite(axial_offset_m) or abs(axial_offset_m) > .02
            or type(preserve_pre_offset_initial_target) is not bool
            or bool(axial_offset_m) != preserve_pre_offset_initial_target
            or (axial_offset_m and height_policy != 'anchor')):
        raise ValueError('Nonzero axial offset within +/-20 mm requires explicit pre-offset initial-target preservation and anchor height policy')
    if height_policy not in ('anchor', 'post_anchor_closed_corridor'):
        raise ValueError("Unknown cylindrical grasp height policy")
    if details.get('robot_control_retiming') or details.get('robot_supported_placement'):
        raise ValueError("Cylindrical calibration requires the original untimed, uninserted anchor clock")
    calibration = details['pad_alignment']
    anchor = int(calibration['retarget_anchor_frame'])
    if not 0 <= anchor < len(times):
        raise ValueError("The original grasp anchor must lie on the complete source clock")
    data = mujoco.MjData(model)
    original_state = data.qpos.copy()
    def set_grip(angle):
        initialize(model, data, robot, q0, manifest['mimics'],
                   {'l_hand_finger': 2., 'r_hand_finger': float(angle)})
    set_grip(calibration['inferred_contact_angle_rad'])
    info = manifest['objects'][details['object_id']]
    vertices, wall = collision_mesh(model, data, object_body=info['body'], object_joint=info['joint'])
    original_center = np.asarray(calibration['section']['center_object'])
    def proposed(angle):
        set_grip(angle)
        pad = pad_surfaces(model, data)
        faces = finite_pad_faces(model, data)
        attachment, report = derive_attachment(hands[anchor], objects[anchor], pad, faces,
                                               wall, vertices, original_center, end_margin_m=end_margin_m,
                                               orientation_policy=orientation_policy, azimuth_policy=azimuth_policy,
                                               azimuth_offset_deg=azimuth_offset_deg)
        return pad, faces, attachment, report
    def gap(angle):
        pad, _, _, report = proposed(angle)
        return pad['gap_m']-report['contact_width_m']
    if gap(-.06)*gap(2.) > 0:
        raise ValueError("Unchanged cylindrical mesh is outside the original gripper aperture")
    angle = float(brentq(gap, -.06, 2., xtol=1e-10))
    pad, faces, attachment, report = proposed(angle)
    corrected = np.asarray(hands) @ attachment
    extras = {}
    if height_policy == 'post_anchor_closed_corridor':
        shift, corridor, extras = post_anchor_corridor(corrected, objects, commands, anchor,
                                                       faces, wall, end_margin_m=end_margin_m)
        original_report = deepcopy(report)
        anchor_attachment = attachment.copy()
        extras.update(anchor_only_attachment=anchor_attachment, anchor_only_corrected_hand_goals=corrected)
        center = (np.asarray(report['desired_pad_midpoint_object_m'])
                  + np.asarray(wall['axis_object'])*corridor['additional_anchor_axis_translation_m'])
        attachment, report = derive_attachment(hands[anchor], objects[anchor], pad, faces, wall, vertices,
                                               center, end_margin_m=end_margin_m, orientation_policy=orientation_policy,
                                               azimuth_policy=azimuth_policy, azimuth_offset_deg=azimuth_offset_deg)
        if not np.allclose(attachment, anchor_attachment @ shift, rtol=0, atol=1e-10):
            raise AssertionError("Anchor finite-patch recheck changed the declared corridor correction")
        corrected = np.asarray(hands) @ attachment
        report.update(original_midpoint_height_m=original_report['original_midpoint_height_m'],
                      anchor_only_midpoint_height_m=original_report['selected_midpoint_height_m'],
                      post_anchor_closed_corridor=corridor)
    if axial_offset_m:
        previous_report, previous_attachment = deepcopy(report), attachment.copy()
        previous_targets = corrected.copy()
        shift, axial_report, axial_arrays = axial_offset(corrected, objects, commands, anchor,
            faces, wall, axial_offset_m, end_margin_m=end_margin_m)
        center = np.asarray(report['desired_pad_midpoint_object_m']) + np.asarray(wall['axis_object'])*axial_offset_m
        rechecked, report = derive_attachment(hands[anchor], objects[anchor], pad, faces, wall, vertices,
            center, end_margin_m=end_margin_m, orientation_policy=orientation_policy,
            azimuth_policy=azimuth_policy, azimuth_offset_deg=azimuth_offset_deg)
        attachment = previous_attachment @ shift
        if not np.allclose(rechecked, attachment, rtol=0, atol=1e-10):
            raise ValueError('Requested axial offset was changed by anchor calibration; no clamping')
        # Use the explicit constant composition, preserving the prior rotation.
        corrected = previous_targets @ shift
        corrected[0] = previous_targets[0]
        report.update(original_midpoint_height_m=previous_report['original_midpoint_height_m'],
                      pre_axial_offset_report=previous_report, explicit_axial_offset=axial_report,
                      preserve_pre_offset_initial_target=True,
                      open_prefix_requires_full_approach_and_ik=True)
        extras.update(axial_arrays, pre_axial_offset_attachment=previous_attachment,
                      pre_axial_offset_hand_goals=previous_targets,
                      explicit_axial_offset_attachment=shift)
    qadr = int(model.jnt_qposadr[model.joint(info['joint']).id])
    if not np.array_equal(data.qpos[qadr:qadr+7], original_state[qadr:qadr+7]):
        raise AssertionError("Cylindrical calibration changed the source object reset")
    artifact = output/'attachment-targets.npz'
    np.savez_compressed(artifact, original_time_s=times, original_reference=prepared[8],
                        original_hand_goals=hands, corrected_hand_goals=corrected,
                        original_object_goals=objects, original_gripper_intent=commands,
                        attachment=attachment, object_collision_vertices=vertices,
                        pad0_triangles_tcp=faces[0], pad1_triangles_tcp=faces[1], **extras)
    report.update(shape_eligibility=wall, anchor_frame=anchor, anchor_time_s=float(times[anchor]),
                  static_finite_patch_scope="Original grasp anchor only; source relative-pose drift and actual physical contact remain separate checks",
                  height_policy=height_policy,
                  inferred_contact_angle_rad=angle, original_inferred_angle_rad=calibration['inferred_contact_angle_rad'],
                  pad_midpoint_tcp_m=pad['midpoint_tcp'].tolist(), attachment=attachment.tolist(),
                  artifact_sha256=sha256(artifact), physics_validated=False, joint_reference_valid=False,
                  requires_full_ik_and_collision_admission=True,
                  policy="One constant TCP attachment from original mesh and finite CAD pads; complete clock, object references, object physics and gripper intent unchanged")
    prior = {key: details.pop(key) for key in ('robot_initialization', 'robot_grasp_approach',
             'robot_supported_placement', 'robot_fixture_clearance') if key in details}
    if prior:
        report['invalidated_prior_admissions'] = prior
    details['previous_pad_alignment'] = deepcopy(calibration)
    details['pad_alignment'].update(inferred_contact_angle_rad=angle,
        pad={key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in pad.items()},
        section=dict(width_m=report['contact_width_m'], center_object=report['desired_pad_midpoint_object_m'],
                     collision_regions=[wall['collision_geom']]))
    details.update(cylindrical_grasp=report, grasp_attachment=report, alignment_variant='constant_tcp_attachment',
                   joint_reference_valid=False, requires_full_ik_and_collision_admission=True)
    json_write(output/'result.json', report)
    result = list(prepared)
    result[5], result[10] = details, corrected
    return tuple(result)
