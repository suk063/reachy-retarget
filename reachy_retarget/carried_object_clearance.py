"""Isolated rigid carried-object/fixture checks for inserted placement paths.

This is a planning assumption, never a simulated object constraint. Only private
derived geom transforms are changed; model fields, qpos, qvel and clock are not.
"""
from pathlib import Path
from copy import deepcopy
import hashlib

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry_clearance import RobotFixtureClearance, _subtree
from .placement_hand_targets import _pose, _release
from .store import json_write, sha256


class CarriedObjectQuery:
    def __init__(self, prepared, *, support_geoms, minimum_m=.003):
        support_geoms=tuple(support_geoms)
        self.model=prepared[0]
        self.data=mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)
        manifest,details=prepared[2],prepared[5]
        active=manifest['objects'][details['object_id']]['body']
        body=self.model.body(active).id;bodies=_subtree(self.model,active)
        joints=[j for j in range(self.model.njnt) if int(self.model.jnt_bodyid[j]) in bodies]
        if (len(joints)!=1 or int(self.model.jnt_type[joints[0]])!=int(mujoco.mjtJoint.mjJNT_FREE)
                or int(self.model.jnt_bodyid[joints[0]])!=body):
            raise ValueError('A rigid active object with one free root and no articulation is required')
        self.geoms=np.array([g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) in bodies])
        root=np.eye(4);root[:3,:3]=self.data.xmat[body].reshape(3,3);root[:3,3]=self.data.xpos[body]
        self.local_positions=(self.data.geom_xpos[self.geoms]-root[:3,3])@root[:3,:3]
        self.local_rotations=np.einsum('ij,njk->nik',root[:3,:3].T,self.data.geom_xmat[self.geoms].reshape(-1,3,3))
        options=dict(robot_body_root=active,active_object_bodies=('base_link',),minimum_m=minimum_m)
        self.all_pairs=RobotFixtureClearance(self.model,**options)
        physical_geoms=sorted({g for g,_ in self.all_pairs.pairs})
        self.object_radius_m=float(max(np.linalg.norm(self.local_positions[np.flatnonzero(self.geoms==g)[0]])
                                       +self.model.geom_rbound[g] for g in physical_geoms))
        floor_ids=[]
        for name in support_geoms:
            g=self.model.geom(name).id;kind=int(self.model.geom_type[g]);rotation=self.data.geom_xmat[g].reshape(3,3)
            if (kind not in (int(mujoco.mjtGeom.mjGEOM_BOX),int(mujoco.mjtGeom.mjGEOM_PLANE))
                    or abs(rotation[2,2])<.999999
                    or (kind==int(mujoco.mjtGeom.mjGEOM_BOX)
                        and self.model.geom_size[g,2]>=min(self.model.geom_size[g,:2]))):
                raise ValueError('Support exemptions must name horizontal floor planes or thin floor boxes')
            if not any(h==g for _,h in self.all_pairs.pairs):
                raise ValueError('Support geometry is not a physical object/fixture pair')
            floor_ids.append(g)
        if not floor_ids:
            raise ValueError('Explicit intended support geometry is required')
        pairs=[(self.model.geom(g).name,self.model.geom(h).name)
               for g,h in self.all_pairs.pairs if h in floor_ids]
        self.walls=RobotFixtureClearance(self.model,**options,support_pairs=pairs)
        # One explicit reference transform, never a measured attachment assertion.
        placement=details.get('robot_supported_placement')
        if placement:
            original=Path(placement['original_artifact'])
            if sha256(original)!=placement['original_artifact_sha256']:
                raise ValueError('Original placement inputs differ from their bound hash')
            with np.load(original) as saved:
                index=placement['original_release_frame']
                hand,object_pose=saved['original_hand_targets'][index],saved['original_object_targets'][index]
        else:
            index=_release(prepared);hand,object_pose=prepared[10][index],prepared[11][index]
        self.hand_to_object=np.linalg.inv(_pose(hand))@_pose(object_pose)
        self.initial_qpos,self.initial_qvel=self.data.qpos.copy(),self.data.qvel.copy()
        self.metadata=dict(active_body=active,source_release_frame=int(index),
            hand_to_object=self.hand_to_object.tolist(),minimum_m=float(minimum_m),
            conservative_physical_body_radius_m=self.object_radius_m,
            intended_support_geoms=list(support_geoms),excluded_support_pairs=pairs,
            support_policy='Only explicitly named horizontal support surfaces may contact during lower; all other fixtures remain constrained',
            scene_xml_sha256=hashlib.sha256(prepared[1].encode()).hexdigest(),
            source_body_frame='Compiled object root frame, including mesh/geom rotations and rigid child bodies',
            assumption='Perfect rigid source grasp relative to desired TCP; actual grip and support require independent physics',
            state_mutation='Private derived object geom_xpos/geom_xmat only; no qpos/qvel/time/model writes')

    def inspect(self, hand_pose, *, allow_support=False):
        goal=_pose(hand_pose)@self.hand_to_object
        self.data.geom_xpos[self.geoms]=self.local_positions@goal[:3,:3].T+goal[:3,3]
        self.data.geom_xmat[self.geoms]=np.einsum('ij,njk->nik',goal[:3,:3],self.local_rotations).reshape(-1,9)
        found=(self.walls if allow_support else self.all_pairs).inspect(self.data)
        if (self.data.time!=0 or not np.array_equal(self.data.qpos,self.initial_qpos)
                or not np.array_equal(self.data.qvel,self.initial_qvel)):
            raise ValueError('Isolated carried-object query changed simulation state')
        return dict(found,allow_intended_support=bool(allow_support),object_pose=goal.tolist())


def measured_attachment_bound(hand_poses, object_poses, ideal_hand_to_object, *,
                              frame_indices, sample_times_s, object_radius_m,
                              eligibility, provenance):
    """Bound observed attachment error in an explicitly eligible measured window.

    The caller must establish bilateral lifted grasp, absence of fixture contact,
    and pre-placement timing for the supplied frames. This is an empirical bound,
    not a prediction that future grip drift or tracking error stays inside it.
    """
    hands,objects=np.asarray(hand_poses,float),np.asarray(object_poses,float)
    frames,times=np.asarray(frame_indices),np.asarray(sample_times_s,float)
    count=len(hands)
    if (hands.shape!=(count,4,4) or objects.shape!=hands.shape or count<1
            or frames.shape!=(count,) or frames.dtype.kind not in 'iu'
            or np.any(frames<0) or np.any(frames[1:]<=frames[:-1])
            or times.shape!=(count,) or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or not np.isfinite(object_radius_m) or object_radius_m<=0
            or not isinstance(provenance,dict) or not provenance):
        raise ValueError('Aligned measured poses, explicit increasing frames/clock, positive radius and provenance required')
    required={'bilateral_lifted_grasp','no_fixture_contact','pre_placement'}
    if not isinstance(eligibility,dict) or set(eligibility)!=required:
        raise ValueError('Explicit measured acquisition eligibility evidence required')
    for value in eligibility.values():
        value=np.asarray(value)
        if value.shape!=(count,) or value.dtype.kind!='b' or not value.all():
            raise ValueError('Every measured frame must satisfy all acquisition eligibility conditions')
    ideal=_pose(ideal_hand_to_object)
    actual=np.array([np.linalg.inv(_pose(h))@_pose(o) for h,o in zip(hands,objects)])
    translation=np.linalg.norm(actual[:,:3,3]-ideal[:3,3],axis=1)
    angles=Rotation.from_matrix(actual[:,:3,:3]@ideal[:3,:3].T).magnitude()
    bounds=translation+2*float(object_radius_m)*np.sin(angles/2)
    digests={}
    for name,value in [('hand_poses',hands),('object_poses',objects),('frames',frames),('times',times),('ideal',ideal)]:
        digests[name]=hashlib.sha256(str((value.dtype.str,value.shape)).encode()+value.tobytes()).hexdigest()
    return dict(frame_indices=frames.tolist(),sample_times_s=times.tolist(),
        ideal_hand_to_object=ideal.tolist(),measured_hand_to_object=actual.tolist(),
        translation_error_m=translation.tolist(),rotation_error_rad=angles.tolist(),
        conservative_physical_body_radius_m=float(object_radius_m),
        observed_mesh_displacement_bound_m=bounds.tolist(),
        maximum_observed_mesh_displacement_bound_m=float(bounds.max()),
        worst_frame=int(frames[np.argmax(bounds)]),input_array_sha256=digests,
        eligibility={k:np.asarray(v).tolist() for k,v in eligibility.items()},provenance=deepcopy(provenance),
        formula='||t_measured-t_ideal|| + 2*body_radius*sin(relative_rotation_angle/2)',
        scope='Observed eligible acquisition window only; future grip drift and robot tracking require separate reserves and physical validation',
        object_state_assignment='none',physics_validated=False)


def prepare(prepared, output, *, support_geoms, minimum_m=.003):
    """Check every inserted loaded reference; physical support guard remains required."""
    policy=prepared[5].get('robot_supported_placement',{})
    if not policy.get('admitted'):
        raise ValueError('Complete admitted robot placement is required before carried-object admission')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    query=CarriedObjectQuery(prepared,support_geoms=support_geoms,minimum_m=minimum_m)
    records=[]
    for phase in ('traverse','lower'):
        start,end=policy['phase_frames'][phase]
        for i in range(start,end+1):
            records.append(dict(frame=i,phase=phase,**query.inspect(prepared[10][i],allow_support=phase=='lower')))
    worst=min(records,key=lambda r:r['minimum_signed_distance_or_lower_bound_m'])
    path=output/'loaded-reference.npz'
    np.savez_compressed(path,time_s=prepared[7],reference=prepared[8],hand_targets=prepared[10],
                        source_object_targets=prepared[11],checked_frames=[r['frame'] for r in records],
                        distance_m=[r['minimum_signed_distance_or_lower_bound_m'] for r in records])
    report=dict(query.metadata,admitted=all(r['admitted'] for r in records),checked_rows=len(records),
        worst=worst,artifact_sha256=sha256(path),physics_validated=False,
        scope='Every inserted traverse/lower control row under explicit rigid-carry assumption; original source stages are not newly certified')
    json_write(output/'result.json',report)
    if not report['admitted']:
        raise ValueError('Carried-object fixture admission rejected; inspect '+str(output/'result.json'))
    return report
