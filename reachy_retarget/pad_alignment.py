"""Translation-only pad alignment of an immutable kinematic retarget export.

This is a paired experiment, not a new trajectory optimizer. Robot base, hand
orientation, source clock, objects, and gripper commands remain fixed. Collision
surface centroids are pad proxies, not annotations of measured rubber contact.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import brentq

from .episodes import read_episode, pose_to_matrices
from .physics import initialize
from .object_scene import build_scene, initialize_fixtures
from .interaction import contact_contract
from .store import Store, json_write, sha256, now
from .task_contracts import TASK_OBJECTS, source_grasp, scene_contract


FINGERS = ("r_hand_distal_link", "r_hand_distal_mimic_link")
_IMPLEMENTATION = Path(__file__).read_text()


def pad_surfaces(model, data):
    """Area-weighted opposing collision faces, expressed in the TCP frame."""
    tcp = model.site("r_arm_tip_tcp").id
    rotation = data.site_xmat[tcp].reshape(3, 3)
    origin = data.site_xpos[tcp]
    bodies = [model.body(name).id for name in FINGERS]
    points, areas, reach = [], [], 0.
    for index, body in enumerate(bodies):
        inward = data.xpos[bodies[1-index]]-data.xpos[body]
        inward /= np.linalg.norm(inward)
        centers, weights = [], []
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError("Pad surface extraction requires the original distal collision mesh")
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            reach = max(reach,float(np.linalg.norm(vertices-origin,axis=1).max()))
            start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            triangles = vertices[model.mesh_face[start:start+count]]
            normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            area = np.linalg.norm(normal, axis=1)
            selected = (normal @ inward) > .95*area
            centers.extend(triangles[selected].mean(1))
            weights.extend(area[selected]/2)
        if not weights or sum(weights) < 1e-8:
            raise ValueError("No opposing distal collision surface")
        center = np.average(centers, axis=0, weights=weights)
        points.append(rotation.T @ (center-origin))
        areas.append(float(sum(weights)))
    points = np.asarray(points)
    return {"points_tcp": points, "midpoint_tcp": points.mean(0),
            "gap_m": float(np.linalg.norm(points[1]-points[0])), "area_m2": areas,
            "distal_reach_m":reach}


def convex_line_interval(vertices, point, direction):
    """Intersect a line with one convex collision region; return signed bounds."""
    hull = ConvexHull(vertices)
    normal, offset = hull.equations[:, :3], hull.equations[:, 3]
    intercept, slope = normal @ point+offset, normal @ direction
    parallel = np.abs(slope) < 1e-10
    if np.any(intercept[parallel] > 1e-8):
        return None
    lo = max((-intercept[slope < -1e-10]/slope[slope < -1e-10]), default=-np.inf)
    hi = min((-intercept[slope > 1e-10]/slope[slope > 1e-10]), default=np.inf)
    return (float(lo), float(hi)) if np.isfinite([lo, hi]).all() and hi > lo else None


def collision_section(model, data, body, point, direction):
    """Section near the demonstrated EEF origin, in the active object frame.

    Meshes use their compiled convex vertices; primitive round regions use a
    documented 64-sided proxy for this geometric estimate only. No physics
    geometry is replaced. Disjoint intersected regions are kept separate.
    """
    descendants = {int(body)}
    for child in range(int(body)+1, model.nbody):
        if int(model.body_parentid[child]) in descendants:
            descendants.add(child)
    inverse = data.xmat[body].reshape(3, 3).T
    sections = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) not in descendants or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        kind, size = int(model.geom_type[geom]), model.geom_size[geom]
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[geom]); start = int(model.mesh_vertadr[mesh])
            vertices = model.mesh_vert[start:start+int(model.mesh_vertnum[mesh])]
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            vertices = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*size
        elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            angle = np.arange(64)*2*np.pi/64
            vertices = np.array([[size[0]*np.cos(t),size[0]*np.sin(t),z]
                                 for t in angle for z in (-size[1],size[1])])
        else:
            continue
        vertices = ((vertices @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom])-data.xpos[body]) @ inverse.T
        interval = convex_line_interval(vertices, point, direction)
        if interval is not None:
            lo, hi = interval
            sections.append((lo, hi, model.geom(geom).name))
    if not sections:
        raise ValueError("The demonstrated grasp line misses supported collision regions; refusing a guessed center")
    # Overlapping convex pieces form one physical cross-section, not a thinner
    # alternative grasp. Choose the connected component nearest the source point.
    merged = []
    for lo, hi, name in sorted(sections):
        if merged and lo <= merged[-1][1]+1e-7:
            merged[-1][1] = max(merged[-1][1], hi); merged[-1][2].append(name)
        else:
            merged.append([lo, hi, [name]])
    lo, hi, names = min(merged, key=lambda x: max(x[0], -x[1], 0))
    return {"width_m": hi-lo, "center_object": point+(lo+hi)/2*direction,
            "collision_regions": names, "line_bounds_m": [lo,hi]}


def blend_weights(times, closed, transition_s=.5, *, ready_time=None):
    times, closed = np.asarray(times), np.asarray(closed, dtype=bool)
    if times.ndim != 1 or closed.shape != times.shape or transition_s <= 0:
        raise ValueError("Invalid contact timeline")
    indices = np.flatnonzero(closed)
    if not len(indices) or not np.all(closed[indices[0]:indices[-1]+1]):
        raise ValueError("Exactly one continuous grasp interval is required")
    first, last = indices[0], indices[-1]
    if first == 0:
        raise ValueError("A common initial state requires an open pregrasp interval")
    ready_time = float(times[first]) if ready_time is None else float(ready_time)
    if not times[0] < ready_time <= times[first]:
        raise ValueError("Alignment must complete in the free pregrasp interval")
    start = max(float(times[0]), ready_time-transition_s)
    x = np.clip((times-start)/(ready_time-start), 0, 1)
    weight = x**3*(10-15*x+6*x*x)
    if last < len(times)-1:
        x = np.clip((times-times[last+1])/transition_s, 0, 1)
        weight *= 1-x**3*(10-15*x+6*x*x)
    return weight


def translate_only(hands, objects, delta_object, weights):
    result = np.array(hands, copy=True)
    result[:, :3, 3] += np.einsum("nij,j->ni", objects[:, :3, :3], delta_object)*weights[:, None]
    return result


def paired_plans(store, row, robot, output, *, state_profile=False):
    """Read an existing retarget once; solve only translated right-arm targets."""
    from .dynamics import Candidate

    source, metadata = read_episode(store.root/row["path"])
    task = metadata.get("env_args", {}).get("env_name")
    oid = TASK_OBJECTS[task]
    motion = store.root/"data/retargeted"/row["source_id"]/row["id"]/"motion.h5"
    validation_path = motion.with_name("validation.json")
    validation = json.loads(validation_path.read_text())
    if validation["source_hdf5_sha256"] != sha256(store.root/row["path"]):
        raise ValueError("Frozen retarget does not match this source episode")
    placement = np.asarray(validation["world_placement"])
    _, _, closure, anchor, local, _ = source_grasp(source,metadata,oid,False)
    with h5py.File(motion) as f:
        times = f["time_s"][:]; arms = f["robot/joint_position"][:]; base = f["robot/base_pose_xyyaw"][:]
        hands = pose_to_matrices(f["robot/hand_pose_world"][:, 1])
        objects = pose_to_matrices(f[f"objects/{oid}/pose"][:])
        if "derived/gripper/right_position" in f:
            commands = f["derived/gripper/right_position"][:]
            gripper_source = "frozen retarget export"
        else:
            index = np.clip(np.searchsorted(source["time_s"]*validation["time_dilation"],times,side="right")-1,0,len(closure)-1)
            commands = np.where(closure[index] > 0,-.06,2.)
            gripper_source = "missing in legacy retarget export; derived from verified source closure identically for both conditions"
    if not np.allclose(np.diff(times), .01, atol=1e-8, rtol=0):
        raise ValueError("Frozen retarget must use the 100 Hz clock")
    # Identical post-episode measurement hold. No demonstrated time is rescaled.
    hold = 200
    seconds = np.r_[times, times[-1]+np.arange(1,hold+1)*.01]
    extend = lambda x: np.concatenate([x, np.repeat(x[-1:],hold,axis=0)])
    arms, base, hands, objects, commands = map(extend, (arms,base,hands,objects,commands))
    xml, manifest = build_scene(store.root, metadata, source, placement, state_profile=state_profile)
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    q0 = robot.pack(arms[0],base[0])
    initialize(model,data,robot,q0,manifest["mimics"],2.)
    initialize_fixtures(model,data,manifest)
    contract = scene_contract(source,metadata,model,data,placement)
    frame = int(np.argmin(np.abs(seconds-source["time_s"][anchor]*validation["time_dilation"])))
    reference = np.c_[base,arms]
    candidate = Candidate(automatic_grasp=False, object_relative=False, force_feedback=False,
                          contact_release=False, calibrate_attachment=False, close_hold=0,
                          controlled_place=False, mobile_base=False, integral_compensation=False)
    details = {"task":task,"object_id":oid,"effective_candidate":asdict(candidate),
               "world_placement":placement.tolist(),"interaction":contact_contract(closure),
               "task_contract":contract,"duration_s":float(seconds[-1]),
               "source_start_frame":0,"source_start_s":0.,"source_window_policy":"complete original source interval",
               "source_time_scale":validation["time_dilation"],"velocity_dilation":1.,
               "trajectory_seed":"existing kinematic retarget; no approach, release, base or timing replanning",
               "frozen_motion_sha256":sha256(motion),"frozen_validation_sha256":sha256(validation_path),
               "gripper_source":gripper_source,
               "evaluation_hold_s":2.}
    before = (model,xml,manifest,source,metadata,{**details,"alignment_variant":"uncorrected"},q0,
              seconds,reference,commands,hands,objects)
    # Always retain the unchanged baseline even if geometric calibration fails.
    try:
        initialize(model,data,robot,q0,manifest["mimics"],.63)
        nominal = pad_surfaces(model,data)
        jaw = nominal["points_tcp"][1]-nominal["points_tcp"][0]; jaw /= np.linalg.norm(jaw)
        direction = objects[frame,:3,:3].T @ hands[frame,:3,:3] @ jaw
        body = model.body(manifest["objects"][oid]["body"]).id
        section = collision_section(model,data,body,local,direction)
        def gap(angle):
            initialize(model,data,robot,q0,manifest["mimics"],angle)
            return pad_surfaces(model,data)["gap_m"]-section["width_m"]
        if gap(-.06)*gap(2.) > 0:
            raise ValueError("Object section is outside the unchanged gripper aperture range")
        angle = float(brentq(gap,-.06,2.,xtol=1e-10))
        gap(angle)
        pad = pad_surfaces(model,data)
        target = objects[frame,:3,:3] @ section["center_object"] + objects[frame,:3,3]
        baseline_pad = hands[frame,:3,:3] @ pad["midpoint_tcp"] + hands[frame,:3,3]
        delta = objects[frame,:3,:3].T @ (target-baseline_pad)
        if np.linalg.norm(delta) > .10:
            raise ValueError("Required translation exceeds the 100 mm calibration bound")
        initialize(model,data,robot,q0,manifest["mimics"],2.)
        open_pad = pad_surfaces(model,data)
        grasp_track = np.einsum('nij,j->ni',objects[:,:3,:3],section['center_object'])+objects[:,:3,3]
        distance = np.linalg.norm(hands[:,:3,3]-grasp_track,axis=1)
        envelope = open_pad['distal_reach_m']+section['width_m']/2
        near = np.flatnonzero(distance <= envelope)
        close = np.flatnonzero(commands < 0)[0]
        ready = min(close,int(near[0]) if len(near) else close)
        weights = blend_weights(seconds,commands < 0,ready_time=seconds[ready])
        corrected = translate_only(hands,objects,delta,weights)
        changed = reference.copy(); errors = []
        original_active = robot.active.copy(); robot.active = robot.arm_v[7:]
        residual = np.zeros(7)
        try:
            for i in range(len(seconds)):
                if weights[i] == 0:
                    errors.append([0.,0.]); residual[:]=0; continue
                q = robot.pack(arms[i],base[i]); q[robot.arm_ids[7:]] += residual
                targets = robot.fk(q); targets[1] = corrected[i]
                solved, pe, re = robot.ik(targets,q,active_hands=(1,),iterations=100,joint_margin=.031)
                changed[i,10:] = solved[robot.arm_ids[7:]]
                residual = changed[i,10:]-reference[i,10:]
                errors.append([pe,re])
        finally:
            robot.active = original_active
        alignment = {"source_anchor_frame":anchor,"retarget_anchor_frame":frame,
                     "source_grasp_point_object_m":local.tolist(),
                     "section":{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in section.items()},
                     "inferred_contact_angle_rad":angle,
                     "pad":{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in pad.items()},
                     "translation_object_m":delta.tolist(),"translation_norm_m":float(np.linalg.norm(delta)),
                     "alignment_ready_time_s":float(seconds[ready]),"precontact_envelope_m":envelope,
                     "max_ik_position_error_m":float(np.max(np.array(errors)[:,0])),
                     "max_ik_rotation_error_rad":float(np.max(np.array(errors)[:,1])),
                     "pad_definition":"area-weighted inward distal collision faces, proxy checked against source visual insert",
                     "gripper_policy":"source commands unchanged; inferred aperture is calibration only"}
        json_write(output/"alignment.json",alignment)
        np.savez_compressed(output/"paired-targets.npz",time_s=seconds,before=hands,after=corrected,
                            objects=objects,gripper=commands,weight=weights,reference_before=reference,
                            reference_after=changed,ik_errors=errors)
        # Preserve the proposed correction and failed IK, but never describe it
        # as a translation-only executed target when the solver cannot reach it.
        if alignment["max_ik_position_error_m"] > .002 or alignment["max_ik_rotation_error_rad"] > .02:
            raise ValueError("Translated target is unreachable with the frozen base and orientation")
        assert np.array_equal(changed[:,:10],reference[:,:10])
        assert np.array_equal(corrected[:,:3,:3],hands[:,:3,:3])
        after = (model,xml,manifest,source,metadata,{**details,"alignment_variant":"pad_translation","pad_alignment":alignment},
                 q0,seconds,changed,commands,corrected,objects)
        return candidate, before, after
    except ValueError as exc:
        json_write(output/"alignment-rejected.json",{"reason":str(exc),"physics_validated":False})
        return candidate, before, None


def pair_checks(before_path, after_path, target_path):
    """Verify the factors held constant in the executed A/B experiment."""
    checks = {"identical_scene":sha256(before_path/"scene.xml")==sha256(after_path/"scene.xml")}
    with h5py.File(before_path/"plan.h5") as a, h5py.File(after_path/"plan.h5") as b:
        for key in ("time_s","gripper","object_goals","initial/qpos","initial/qvel","initial/ctrl"):
            checks[key] = bool(np.array_equal(a[key][:],b[key][:]))
        checks["base_and_inactive_arm_reference"] = bool(np.array_equal(a["reference"][:,:10],b["reference"][:,:10]))
        # Matrices were identical before serialization; compare quaternion signs
        # as rotations rather than treating q and -q as different orientations.
        ar=pose_to_matrices(a["hand_goals"][:])[:,:3,:3]
        br=pose_to_matrices(b["hand_goals"][:])[:,:3,:3]
        checks["target_orientation"] = bool(np.allclose(ar,br,atol=1e-14,rtol=0))
    model=mujoco.MjModel.from_xml_path(str(before_path/"scene.xml"))
    from .physics import BASE, ARMS, GRIPPERS
    fixed=[model.actuator(n).id for n in (*BASE,*ARMS[:7],*GRIPPERS)]
    with h5py.File(before_path/"replay.h5") as a, h5py.File(after_path/"replay.h5") as b:
        count=min(len(a["time_s"]),len(b["time_s"]))
        checks["base_inactive_arm_gripper_actuator_commands"] = bool(np.array_equal(
            a["simulation/actuator_control"][:count][:,fixed],b["simulation/actuator_control"][:count][:,fixed]))
    with np.load(target_path) as targets:
        free=targets['weight']==0
        checks['uncorrected_free_interval_targets'] = bool(np.array_equal(targets['before'][free],targets['after'][free]))
        checks['uncorrected_free_interval_joint_reference'] = bool(np.array_equal(targets['reference_before'][free],targets['reference_after'][free]))
    if not all(checks.values()):
        raise ValueError("Paired experiment changed a frozen factor: "+str(checks))
    return checks


def run(root, label, episode_ids):
    from .dynamics import rollout
    from .dynamics_audit import verify
    from .robot import Robot

    store = Store(Path(root).resolve()); out = store.root/"runs/pad-alignment"/label
    out.mkdir(parents=True,exist_ok=False)
    summary = {"created":now(),"protocol":"frozen-retarget-pad-translation-v1","episodes":[],
               "source_interval":"complete", "search":False}
    rows = store.rows("episodes")
    if set(episode_ids)-{r["id"] for r in rows}:
        raise ValueError("Requested episode is absent from the local ledger")
    for row in rows:
        if row["id"] not in episode_ids: continue
        folder = out/row["id"]; folder.mkdir()
        result = {"episode":row["id"],"source_sequence":row["source_sequence"],"variants":{}}
        try:
            candidate,before,after = paired_plans(store,row,Robot(store.root),folder)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            result.update(status="input_rejected",reason=f"{type(exc).__name__}: {exc}")
            summary["episodes"].append(result);json_write(out/"summary.json",summary)
            json_write(folder/"input-rejected.json",result)
            continue
        for name, prepared in (("before",before),("after",after)):
            if prepared is None:
                result["variants"][name] = {"status":"alignment_rejected","physics_validated":False,
                    **json.loads((folder/"alignment-rejected.json").read_text())}; continue
            report = rollout(store,row,f"pad_alignment/{label}/{name}",candidate,
                             prepared_plan=prepared,extra_sources={Path(__file__).name:_IMPLEMENTATION})
            directory = store.root/"runs/dynamics/pad_alignment"/label/name/row["id"]
            audit = verify(directory)
            result["variants"][name] = {"path":str(directory.relative_to(store.root)),
                "status":report["status"],"actuator_replay_pass":audit["actuator_replay_pass"],
                "bilateral_grasp_acquired":report["gates"]["bilateral_lifted_grasp"],
                "grasp_drift_available":report["gates"]["bilateral_lifted_grasp"],
                **{k:report[k] for k in ("failure_reasons","max_lift_m","max_hand_object_penetration_m",
                    "bilateral_carry_fraction","max_grasp_drift_m","max_grasp_drift_deg","final_stable_s")}}
        if after is not None:
            result["frozen_factor_checks"] = pair_checks(
                store.root/result["variants"]["before"]["path"],store.root/result["variants"]["after"]["path"],folder/"paired-targets.npz")
        summary["episodes"].append(result);json_write(out/"summary.json",summary)
        print(json.dumps(result),flush=True)
    summary["finished"] = now();json_write(out/"summary.json",summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--label",required=True)
    parser.add_argument("--episode",action="append",required=True)
    args = parser.parse_args();run(args.root,args.label,args.episode)


if __name__ == "__main__":
    main()
