"""Calibrate a robot grasp on a verified box component of an unchanged object.

The native grasp point chooses the component; an explicit bounded axis bank
chooses the opposing faces. Other collision pieces remain present when measuring
the closing section. This is a robot target proposal, never a physics result.
"""
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import brentq

from .pad_alignment import collision_section, pad_surfaces
from .physics import initialize
from .store import json_write, sha256


def component(model, object_body, object_joint, point_object):
    """Choose the smallest source box containing the demonstrated grasp point."""
    body = model.body(object_body).id
    joint = model.joint(object_joint).id
    point = np.asarray(point_object, float)
    if (body <= 0 or int(model.jnt_bodyid[joint]) != body
            or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE)
            or point.shape != (3,) or not np.isfinite(point).all()):
        raise ValueError('A verified free object and finite native grasp point are required')
    subtree = {body}
    for child in range(body+1, model.nbody):
        if int(model.body_parentid[child]) in subtree:
            subtree.add(child)
    if [i for i in range(model.njnt) if int(model.jnt_bodyid[i]) in subtree] != [joint]:
        raise ValueError('Component calibration requires a rigid nonarticulated object')
    paired = set(map(int, model.pair_geom1)) | set(map(int, model.pair_geom2))
    physical = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in subtree
                and (model.geom_contype[i] or model.geom_conaffinity[i] or i in paired)]
    # The legacy rollout's active-object collision classifier uses the root.
    if any(int(model.geom_bodyid[i]) != body for i in physical):
        raise ValueError('This rollout requires collision components on the free object root')
    eligible = []
    for geom in physical:
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        axes = np.empty(9); mujoco.mju_quat2Mat(axes, model.geom_quat[geom])
        axes = axes.reshape(3, 3)
        center, half = model.geom_pos[geom], model.geom_size[geom]
        local = axes.T @ (point-center)
        if np.all(np.abs(local) <= half+1e-7):
            eligible.append((float(np.prod(half)), geom, axes))
    if not eligible:
        raise ValueError('Native grasp point is outside every supported source box component')
    _, geom, axes = min(eligible, key=lambda item: item[:2])
    return dict(object_body=object_body, object_joint=object_joint, object_body_id=body,
                component_geom_id=geom, component_geom=model.geom(geom).name,
                center_object_m=model.geom_pos[geom].tolist(),
                axes_object=axes.tolist(), half_size_m=model.geom_size[geom].tolist(),
                source_grasp_point_object_m=point.tolist(),
                selection='Smallest original collision box containing the native grasp point',
                other_collision_components_preserved=True)


def attachment(hand, object_pose, jaw_tcp, midpoint_tcp, axes, center, face_axis):
    """Use the minimum rotation aligning the closing line with a declared face."""
    if type(face_axis) is not int or face_axis not in (0, 1, 2):
        raise ValueError('face_axis must be an explicit integer from 0 to 2')
    jaw = np.asarray(jaw_tcp, float); jaw = jaw/np.linalg.norm(jaw)
    current = object_pose[:3, :3].T @ hand[:3, :3] @ jaw
    desired = np.asarray(axes)[:, face_axis].copy()
    if current @ desired < 0:
        desired *= -1
    cosine = float(np.clip(current @ desired, -1, 1))
    cross = np.cross(current, desired)
    skew = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]],
                     [-cross[1], cross[0], 0.]])
    rotate = np.eye(3)+skew+skew@skew/(1+cosine)
    goal = np.array(hand, copy=True)
    goal[:3, :3] = object_pose[:3, :3] @ rotate @ object_pose[:3, :3].T @ hand[:3, :3]
    goal[:3, 3] = object_pose[:3, 3]+object_pose[:3, :3]@center-goal[:3, :3]@midpoint_tcp
    return np.linalg.inv(hand)@goal, float(np.arccos(cosine))


def apply(prepared, robot, output, *, face_axis):
    """Retain explicit geometric rejections as well as admitted proposals."""
    try:
        return _apply(prepared, robot, output, face_axis=face_axis)
    except ValueError as error:
        path = Path(output)/'result.json'
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            json_write(path, dict(admitted=False, physics_validated=False,
                                  face_axis=face_axis, reason=str(error)))
        raise


def _apply(prepared, robot, output, *, face_axis):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    model, _, manifest = prepared[:3]
    details = deepcopy(prepared[5]); calibration = details['pad_alignment']
    spec = manifest['objects'][details['object_id']]
    eligibility = component(model, spec['body'], spec['joint'],
                            calibration['source_grasp_point_object_m'])
    if type(face_axis) is not int or face_axis not in (0, 1, 2):
        raise ValueError('face_axis must be an explicit integer from 0 to 2')
    center, axes = np.asarray(eligibility['center_object_m']), np.asarray(eligibility['axes_object'])
    data = mujoco.MjData(model); object_address = int(model.joint(spec['joint']).qposadr[0])
    initial_object = data.qpos[object_address:object_address+7].copy()
    initialize(model, data, robot, prepared[6], manifest['mimics'], .63)
    section = collision_section(model, data, eligibility['object_body_id'], center, axes[:, face_axis])
    if eligibility['component_geom'] not in section['collision_regions']:
        raise ValueError('Selected physical closing section does not contain the native grasp component')
    def gap(angle):
        initialize(model, data, robot, prepared[6], manifest['mimics'], angle)
        return pad_surfaces(model, data)['gap_m']-section['width_m']
    if gap(-.06)*gap(2.) > 0:
        json_write(output/'result.json', dict(admitted=False, shape_eligibility=eligibility,
            face_axis=face_axis, section={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in section.items()},
            reason='Original physical section exceeds original gripper aperture', physics_validated=False))
        raise ValueError('Component closing section is outside the unchanged gripper aperture')
    angle = float(brentq(gap, -.06, 2., xtol=1e-10)); gap(angle)
    pad = pad_surfaces(model, data)
    anchor = int(calibration['retarget_anchor_frame'])
    transform, rotation = attachment(prepared[10][anchor], prepared[11][anchor],
        pad['points_tcp'][1]-pad['points_tcp'][0], pad['midpoint_tcp'], axes,
        np.asarray(section['center_object']), face_axis)
    changed = np.asarray(prepared[10])@transform
    max_shift = float(np.linalg.norm(changed[:, :3, 3]-prepared[10][:, :3, 3], axis=1).max())
    if max_shift > .15:
        raise ValueError('Component TCP correction exceeds the declared 150 mm bound')
    if not np.array_equal(data.qpos[object_address:object_address+7], initial_object):
        raise AssertionError('Robot calibration changed the source object reset')
    artifact = output/'component-attachment.npz'
    np.savez_compressed(artifact, original_time_s=prepared[7], original_reference=prepared[8],
                        original_hand_goals=prepared[10], hand_goals=changed,
                        original_object_goals=prepared[11], original_gripper_intent=prepared[9], attachment=transform)
    section = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in section.items()}
    report = dict(method='native_grasp_box_component', shape_eligibility=eligibility,
                  face_axis=face_axis, rotation_change_rad=rotation, max_tcp_translation_m=max_shift,
                  section=section, inferred_contact_angle_rad=angle, attachment=transform.tolist(),
                  artifact_sha256=sha256(artifact), physics_validated=False, joint_reference_valid=False,
                  requires_full_ik_and_collision_admission=True,
                  scope='One constant robot TCP attachment; complete original clock, object references and geometry retained')
    json_write(output/'result.json', report)
    details['previous_pad_alignment'] = deepcopy(calibration)
    calibration.update(inferred_contact_angle_rad=angle, section=section,
                       pad={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in pad.items()})
    details.update(grasp_attachment=report, alignment_variant='constant_tcp_attachment')
    result = list(prepared); result[5], result[10] = details, changed
    return tuple(result)


def admit(prepared, robot, output, base_xyyaw, *, right_arm_seed=None):
    """Verify the component binding, then inspect the entire corrected path."""
    from .feasible_maniskill import _prepare
    proposal = prepared[5]['grasp_attachment']
    if proposal.get('method') != 'native_grasp_box_component':
        raise ValueError('Full component admission requires the explicit region proposal')
    spec = prepared[2]['objects'][prepared[5]['object_id']]
    actual = component(prepared[0], spec['body'], spec['joint'],
                       proposal['shape_eligibility']['source_grasp_point_object_m'])
    if actual != proposal['shape_eligibility']:
        raise ValueError('Source component changed after calibration')
    return _prepare(prepared, robot, output, base_xyyaw, expected_task=prepared[5]['task'],
                    right_arm_seed=right_arm_seed, gripper_admission='intent_envelope')
