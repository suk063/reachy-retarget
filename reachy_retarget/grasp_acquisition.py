"""Separate pregrasp object displacement from pad tracking in exact replay."""
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .dynamics_audit import _load_recording, _verify_assets, _verify_no_object_assistance
from .pad_alignment import pad_surfaces
from .store import sha256


def diagnose(attempt, *, window_s=(1.9, 2.5)):
    """Replay controls unchanged and return acquisition geometry observations."""
    attempt = Path(attempt)
    report = json.loads((attempt / "result.json").read_text())
    scene = attempt / "scene.xml"
    if sha256(scene) != report["scene_sha256"]:
        raise ValueError("Scene checksum differs from its report")
    _verify_assets(scene, report["physics"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    objects = report["physics"]["objects"]
    joints = _verify_no_object_assistance(model, objects)
    with h5py.File(attempt / "replay.h5", "r") as handle:
        arrays, steps = _load_recording(handle, model, report, objects)
        hands = handle["target/applied_right_hand_pose_world"][()]
        object_goals = handle["target/object_pose_world"][()]
    with h5py.File(attempt / "plan.h5", "r") as handle:
        intent = handle["gripper"][()]
    data = mujoco.MjData(model)
    for key in ("qpos", "qvel", "ctrl"):
        getattr(data, key)[:] = arrays["initial/"+key]
    for joint in joints:
        q = int(model.jnt_qposadr[joint])
        if not np.allclose(data.qpos[q:q+7], model.qpos0[q:q+7], atol=1e-10, rtol=0):
            raise ValueError("Object reset differs from source scene")
    mujoco.mj_forward(model, data)
    active = model.body(objects[report["plan"]["object_id"]]["body"]).id
    original_rotation = data.xmat[active].reshape(3, 3).copy()
    original_position = data.xpos[active].copy()
    tcp = model.site("r_arm_tip_tcp").id
    grip_q = int(model.joint("r_hand_finger").qposadr[0])
    grip_act = model.actuator("r_hand_finger").id
    fingers = {model.body(name).id for name in ("r_hand_distal_link", "r_hand_distal_mimic_link")}
    robot = {model.body("base_link").id}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in robot:
            robot.add(body)
    calibrated = np.asarray(report["plan"]["pad_alignment"]["pad"]["midpoint_tcp"])
    events, frames, maximum_error = {}, [], dict(qpos=0., qvel=0.)
    for frame, count in enumerate(steps):
        data.ctrl[:] = arrays["simulation/actuator_control"][frame]
        interval_contacts = []
        for _ in range(int(count)):
            mujoco.mj_step(model, data)
            contacts, touched = [], set()
            rotation = data.xmat[active].reshape(3, 3)
            for index, contact in enumerate(data.contact):
                pair = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
                other = pair.intersection(robot)
                if active not in pair or not other:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                if force[0] <= 1e-6:
                    continue
                touched.update(pair.intersection(fingers))
                body = next(iter(other))
                contacts.append(dict(body=model.body(body).name, distal_finger=body in fingers,
                    normal_force_n=float(force[0]), depth_m=max(0., -float(contact.dist)),
                    position_original_object_m=(original_rotation.T @ (contact.pos-original_position)).tolist(),
                    position_actual_object_m=(rotation.T @ (contact.pos-data.xpos[active])).tolist()))
            kinds = []
            if contacts:
                kinds.append("first_robot_object_force")
                if touched == fingers: kinds.append("first_bilateral_finger_force")
                if any(not c["distal_finger"] for c in contacts): kinds.append("first_non_distal_robot_force")
                if intent[frame] >= 1.9999: kinds.append("first_force_with_open_intent")
            for kind in kinds:
                if kind not in events:
                    events[kind] = dict(collision_state_time_s=float(data.time-model.opt.timestep),
                        substep_end_s=float(data.time), frame=frame, source_contact_intent=float(intent[frame]),
                        actual_gripper_rad=float(data.qpos[grip_q]), issued_gripper_rad=float(data.ctrl[grip_act]),
                        object_translation_original_m=(original_rotation.T @ (data.xpos[active]-original_position)).tolist(),
                        contacts=contacts)
            interval_contacts = contacts
        mujoco.mj_forward(model, data)
        for key in maximum_error:
            maximum_error[key] = max(maximum_error[key], float(np.max(np.abs(
                getattr(data, key)-arrays["simulation/"+key][frame]), initial=0.)))
        if not window_s[0] <= data.time <= window_s[1]:
            continue
        pad = pad_surfaces(model, data)
        actual_hand_rotation = data.site_xmat[tcp].reshape(3, 3)
        actual_object_rotation = data.xmat[active].reshape(3, 3)
        reference_hand_rotation = Rotation.from_quat(hands[frame, 3:], scalar_first=True).as_matrix()
        reference_object_rotation = Rotation.from_quat(object_goals[frame, 3:], scalar_first=True).as_matrix()
        actual_midpoint_world = actual_hand_rotation @ pad["midpoint_tcp"]+data.site_xpos[tcp]
        reference_midpoint_world = reference_hand_rotation @ calibrated+hands[frame, :3]
        frames.append(dict(frame=frame, interval_end_s=float(data.time),
            actual_pad_midpoint_original_object_m=(original_rotation.T @ (actual_midpoint_world-original_position)).tolist(),
            actual_pad_midpoint_actual_object_m=(actual_object_rotation.T @ (actual_midpoint_world-data.xpos[active])).tolist(),
            reference_pad_midpoint_original_object_m=(original_rotation.T @ (reference_midpoint_world-original_position)).tolist(),
            reference_pad_midpoint_reference_object_m=(reference_object_rotation.T @ (reference_midpoint_world-object_goals[frame, :3])).tolist(),
            object_translation_original_m=(original_rotation.T @ (data.xpos[active]-original_position)).tolist(),
            object_rotation_original_rotvec_rad=Rotation.from_matrix(original_rotation.T @ actual_object_rotation).as_rotvec().tolist(),
            tcp_tracking_error_original_object_m=(original_rotation.T @ (data.site_xpos[tcp]-hands[frame, :3])).tolist(),
            actual_gripper_rad=float(data.qpos[grip_q]), issued_gripper_rad=float(data.ctrl[grip_act]),
            source_contact_intent=float(intent[frame]), contacts_at_last_collision_state=interval_contacts))
    return dict(attempt=str(attempt), window_s=list(window_s), original_object_position_world_m=original_position.tolist(),
        original_object_rotation_world=original_rotation.tolist(), events=events, frames=frames,
        replay_max_absolute_error=maximum_error, implementation_sha256=sha256(__file__),
        scene_sha256=sha256(scene), report_sha256=sha256(attempt / "result.json"),
        replay_sha256=sha256(attempt / "replay.h5"),
        semantics="Independent unchanged-controls replay; original object frame is the physical reset. Only initial qpos/qvel/ctrl assigned. Contacts use preintegration collision state; per-frame poses use interval-end FK.")
