"""Read-only contact placement and servo-limit diagnosis of saved rollouts.

The model and objects are reset once from the bound recording. Every subsequent
state comes from recorded robot controls; no physics or acceptance gates change.
"""
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .dynamics_audit import _load_recording, _verify_assets, _verify_no_object_assistance
from .robot import ARMS


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(attempt):
    attempt = Path(attempt)
    report = json.loads((attempt / "result.json").read_text())
    scene = attempt / "scene.xml"
    if _sha(scene) != report["scene_sha256"]:
        raise ValueError("Saved scene checksum differs from report")
    _verify_assets(scene, report["physics"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    objects = report["physics"]["objects"]
    free_joints = _verify_no_object_assistance(model, objects)
    with h5py.File(attempt / "replay.h5", "r") as handle:
        arrays, substeps = _load_recording(handle, model, report, objects)
        extra = {key: handle[key][()] for key in (
            "metrics/grasp_drift_m_rad", "command/applied_joint_reference",
            "target/applied_right_hand_pose_world", "command/gripper_position")}
    with h5py.File(attempt / "plan.h5", "r") as handle:
        intent = handle["gripper"][()]
    data = mujoco.MjData(model)
    data.qpos[:] = arrays["initial/qpos"]
    data.qvel[:] = arrays["initial/qvel"]
    data.ctrl[:] = arrays["initial/ctrl"]
    for joint in free_joints:
        q = int(model.jnt_qposadr[joint])
        if not np.allclose(data.qpos[q:q+7], model.qpos0[q:q+7], atol=1e-10, rtol=0):
            raise ValueError("Object reset differs from unchanged scene")
    mujoco.mj_forward(model, data)
    active = model.body(objects[report["plan"]["object_id"]]["body"]).id
    fingers = {model.body(name).id: name for name in
               ("r_hand_distal_link", "r_hand_distal_mimic_link")}
    tcp = model.site("r_arm_tip_tcp").id
    joints = np.array([model.joint(name).id for name in ARMS])
    qpos = model.jnt_qposadr[joints]
    actuators = np.array([model.actuator(name).id for name in ARMS])
    limits = model.jnt_range[joints]
    grip = model.actuator("r_hand_finger").id
    minimum = {key: {"margin_rad": float("inf")} for key in ("reference", "actuator_target", "measured")}
    max_error = dict(position_m=0., rotation_rad=0.)
    replay_error = dict(qpos=0., qvel=0.)
    records = []
    mimic_peaks = {}
    joint_equalities = [i for i in range(model.neq) if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT]

    def margin_event(key, values, frame):
        distances = np.minimum(values-limits[:, 0], limits[:, 1]-values)
        index = int(np.argmin(distances))
        if float(distances[index]) < minimum[key]["margin_rad"]:
            minimum[key] = dict(margin_rad=float(distances[index]), joint=ARMS[index],
                                position_rad=float(values[index]), limits_rad=limits[index].tolist(),
                                time_s=float(data.time), frame=frame)

    for frame, count in enumerate(substeps):
        data.ctrl[:] = arrays["simulation/actuator_control"][frame]
        margin_event("reference", extra["command/applied_joint_reference"][frame, 3:], frame)
        margin_event("actuator_target", data.ctrl[actuators], frame)
        contact_rows = []
        forces = {name: 0. for name in fingers.values()}
        for step in range(int(count)):
            mujoco.mj_step(model, data)
            margin_event("measured", data.qpos[qpos], frame)
            for equality in joint_equalities:
                first, second = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
                a = float(data.qpos[model.jnt_qposadr[first]]-model.qpos0[model.jnt_qposadr[first]])
                b = (float(data.qpos[model.jnt_qposadr[second]]-model.qpos0[model.jnt_qposadr[second]])
                     if second >= 0 else 0.)
                residual = a-float(np.polynomial.polynomial.polyval(b, model.eq_data[equality, :5]))
                name = model.equality(equality).name or str(equality)
                if abs(residual) > mimic_peaks.get(name, {}).get("absolute_residual_rad", -1):
                    mimic_peaks[name] = dict(absolute_residual_rad=abs(residual), time_s=float(data.time),
                                            joint1=model.joint(first).name,
                                            joint2=model.joint(second).name if second >= 0 else None)
            if step != int(count)-1:
                continue
            # mj_step collision positions and xmat describe the collision state
            # preceding integration, so transform both in that same frame.
            object_rotation = data.xmat[active].reshape(3, 3)
            for index, contact in enumerate(data.contact):
                pair = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
                if active not in pair or not pair.intersection(fingers):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                if force[0] <= 0:
                    continue
                body = next(iter(pair.intersection(fingers)))
                forces[fingers[body]] += float(force[0])
                contact_rows.append(dict(body=fingers[body], normal_force_n=float(force[0]),
                    position_object_m=(object_rotation.T @ (contact.pos-data.xpos[active])).tolist(),
                    normal_object=(object_rotation.T @ contact.frame[:3]).tolist(),
                    geoms=[model.geom(contact.geom1).name, model.geom(contact.geom2).name],
                    contact_dimension=int(contact.dim),
                    depth_m=max(0., -float(contact.dist))))
        mujoco.mj_forward(model, data)
        centers = {}
        for name in fingers.values():
            rows = [row for row in contact_rows if row["body"] == name]
            if rows:
                centers[name] = np.average([r["position_object_m"] for r in rows], axis=0,
                                           weights=[r["normal_force_n"] for r in rows]).tolist()
        target = extra["target/applied_right_hand_pose_world"][frame]
        actual_rotation = data.site_xmat[tcp].reshape(3, 3)
        target_rotation = Rotation.from_quat(target[3:], scalar_first=True).as_matrix()
        delta = data.site_xpos[tcp]-target[:3]
        error = dict(position_m=float(np.linalg.norm(delta)),
                     rotation_rad=float(Rotation.from_matrix(target_rotation.T @ actual_rotation).magnitude()),
                     position_delta_world_m=delta.tolist())
        max_error["position_m"] = max(max_error["position_m"], error["position_m"])
        max_error["rotation_rad"] = max(max_error["rotation_rad"], error["rotation_rad"])
        records.append(dict(frame=frame, interval_end_s=float(data.time),
                            contact_collision_time_s=float(data.time-model.opt.timestep),
                            drift_m_rad=extra["metrics/grasp_drift_m_rad"][frame].tolist(),
                            tcp_tracking_error=error, pad_normal_force_sum_n=forces,
                            force_weighted_contact_center_object_m=centers,
                            contacts=contact_rows, gripper_target_rad=float(data.ctrl[grip])))
        for key in replay_error:
            replay_error[key] = max(replay_error[key], float(np.max(np.abs(
                getattr(data, key)-arrays["simulation/"+key][frame]), initial=0.)))
    drift = extra["metrics/grasp_drift_m_rad"]
    valid = np.flatnonzero(np.isfinite(drift).all(axis=1))
    closing = np.flatnonzero(intent < 1.9999)
    selected = {"final": len(records)-1}
    if len(closing): selected["requested_closure"] = int(closing[0])
    if len(valid):
        selected["drift_anchor"] = int(valid[0])
        selected["maximum_translation_drift"] = int(valid[np.argmax(drift[valid, 0])])
        selected["maximum_rotation_drift"] = int(valid[np.argmax(drift[valid, 1])])
        bad = valid[(drift[valid, 0] > .003) | (drift[valid, 1] > np.deg2rad(3))]
        if len(bad): selected["first_drift_gate_exceedance"] = int(bad[0])
    return dict(attempt=str(attempt), scene_sha256=_sha(scene),
                report_sha256=_sha(attempt / "result.json"), replay_sha256=_sha(attempt / "replay.h5"),
                implementation_sha256=_sha(__file__), runtime_mujoco_version=mujoco.__version__,
                candidate=report["candidate"], reported_failures=report["failure_reasons"],
                model_options={"disableflags": int(model.opt.disableflags),
                    "enableflags": int(model.opt.enableflags), "cone": int(model.opt.cone),
                    "timestep": float(model.opt.timestep), "impratio": float(model.opt.impratio),
                    "solver": int(model.opt.solver), "iterations": int(model.opt.iterations),
                    "disabled": [k for k, v in mujoco.mjtDisableBit.__members__.items()
                                 if k != "mjNDISABLE" and model.opt.disableflags & int(v)]},
                maximum_joint_equality_residuals=mimic_peaks,
                reported_metrics={k: report.get(k) for k in (
                    "max_grasp_drift_m", "max_grasp_drift_deg", "bilateral_carry_fraction",
                    "max_hand_object_penetration_m", "final_stable_s", "min_joint_margin_rad")},
                minimum_arm_margins=minimum, maximum_tcp_tracking_error=max_error,
                selected_intervals={k: records[v] for k, v in selected.items()},
                frames=len(records), replay_max_absolute_error=replay_error,
                semantics="Recorded-controls-only independent replay; original states, model and gates unchanged. Contact centers use collision-state object frame; TCP errors use interval-end state.")
