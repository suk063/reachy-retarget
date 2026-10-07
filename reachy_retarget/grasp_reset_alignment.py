"""Align a source grasp to the unchanged physical reset object's center.

A source object can already have moved before the target robot first touches
it. Correct the robot path by one constant translation; never prescribe that
unrealized source movement to the simulated object. Full IK admission follows.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import mujoco
from .object_scene import initialize_fixtures
from .store import json_write, sha256


def apply(prepared, output, *, center_tolerance_m=.001, max_translation_m=.03,
          max_reference_lift_m=.005):
    if not 0 < center_tolerance_m <= max_translation_m <= .1:
        raise ValueError('Require explicit small positive grasp-alignment bounds')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    details=deepcopy(prepared[5]);pad=details['pad_alignment']
    midpoint=np.asarray(pad['pad']['midpoint_tcp']);center=np.asarray(pad['section']['center_object'])
    hands,objects,commands=prepared[10],prepared[11],prepared[9]
    pads=hands[:,:3,:3]@midpoint+hands[:,:3,3]
    centers=objects[:,:3,:3]@center+objects[:,:3,3]
    error=np.linalg.norm(pads-centers,axis=1)
    model=prepared[0];data=mujoco.MjData(model);initialize_fixtures(model,data,prepared[2])
    body=model.body(prepared[2]['objects'][details['object_id']]['body']).id
    reset_center=data.xpos[body]+data.xmat[body].reshape(3,3)@center
    reset_error=float(np.linalg.norm(centers[0]-reset_center))
    up=-model.opt.gravity/np.linalg.norm(model.opt.gravity)
    lift=(centers-centers[0])@up
    candidates=np.flatnonzero((commands<0)&(error<=center_tolerance_m)&(lift<=max_reference_lift_m))
    anchor=int(candidates[0]) if len(candidates) else None
    translation=np.zeros(3) if anchor is None else reset_center-centers[anchor]
    admitted=anchor is not None and np.linalg.norm(translation)<=max_translation_m and reset_error<=1e-7
    changed=np.array(hands,copy=True);changed[:,:3,3]+=translation
    artifact=output/'reset-alignment.npz'
    np.savez_compressed(artifact,time_s=prepared[7],original_hand_targets=hands,
        derived_hand_targets=changed,original_object_targets=objects,original_gripper_intent=commands,
        reference_pad_center_error_m=error,translation_world_m=translation)
    report=dict(admitted=bool(admitted),physics_validated=False,anchor_frame=anchor,
        anchor_time_s=None if anchor is None else float(prepared[7][anchor]),
        translation_world_m=translation.tolist(),translation_norm_m=float(np.linalg.norm(translation)),
        center_tolerance_m=center_tolerance_m,max_translation_m=max_translation_m,
        max_reference_lift_m=max_reference_lift_m,reference_reset_center_error_m=reset_error,
        physical_reset_center_world_m=reset_center.tolist(),
        artifact=str(artifact),artifact_sha256=sha256(artifact),joint_reference_valid=False,
        requires_full_ik_and_collision_admission=True,
        policy='One constant robot TCP translation aligns the first central closed-intent source grasp with the original physical reset object; complete trajectory shape, object references/state and clock retained')
    json_write(output/'result.json',report)
    if not admitted:raise ValueError('Reset-object grasp alignment rejected; inspect '+str(output/'result.json'))
    details['grasp_reset_alignment']=report
    result=list(prepared);result[5]=details;result[10]=changed
    return tuple(result)
