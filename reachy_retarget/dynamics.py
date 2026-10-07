"""Simulation-in-the-loop object-relative Reachy retargeting.

Only actuator commands drive validation. Object states are initialized by the
scene builder and subsequently read, never overwritten. Every search candidate,
including rejected plans and unsuccessful rollouts, has a separate artifact.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .episodes import read_episode, pose_to_matrices, matrices_to_pose, interpolate_pose
from .physics import initialize, measured, ARMS, BASE, GRIPPERS
from .robot import Robot
from .store import Store, json_write, sha256, now
from .object_scene import build_scene, initialize_fixtures
from .dynamics_audit import bind_scene_assets
from .interaction import contact_contract, contact_gates, geometry_seed, add_search_sensors, supported_release
from . import dynamics_maniskill
from .task_contracts import (TASK_OBJECTS, REFERENCES, _validate_source_arrays,
                             validate_task_contract, source_grasp, source_window, task_state,
                             source_vertical_offset, scene_contract)

# Capture the code actually loaded for a search process. Later source edits
# must not replace snapshots of an already running attempt.
_IMPLEMENTATION_SOURCES = {name: Path(__file__).with_name(name).read_text()
                           for name in ("dynamics.py", "object_scene.py", "physics.py", "robot.py", "dynamics_maniskill.py", "dynamics_audit.py", "task_contracts.py", "interaction.py", "trajectory_search.py", "state_recording.py", "state_observations.py", "agent_dataset.py", "servo_feedforward.py", "servo_envelope.py")}




@dataclass(frozen=True)
class Candidate:
    depth: float = .025
    height: float = .0
    tilt: float = 15.
    yaw: float = 0.
    time_scale: float = 3.
    close_duration: float = .5
    close_hold: float = .5
    grasp_force: float = 2.
    force_feedback: bool = False
    placement_x: float = .50
    placement_y: float = -.10
    geometry_center: bool = True
    controller: str = "joint_reference"
    object_relative: bool = True
    mobile_base: bool = False
    align_object_axes: bool = True
    scene_heading: float = 0.
    base_x_max: float = 100.
    base_y_min: float = -100.
    balanced_force: bool = False
    calibrate_attachment: bool = False
    grasp_local_x: float = 0.
    grasp_local_y: float = 0.
    soft_force_feedback: bool = False
    controlled_place: bool = False
    source_start_s: float = 0.
    place_clearance: float = .04
    automatic_grasp: bool = True
    grasp_region_rank: int = 0
    force_scale: float = 1.
    contact_release: bool = True
    integral_compensation: bool = True
    gripper_closed_target: float = -.06
    force_preclose_target: float | None = None
    arm_velocity_feedforward: float = 0.
    base_velocity_feedforward: float = 0.
    post_acquisition_base_velocity_feedforward: float = 0.
    source_prefix_velocity_feedforward: float = 0.
    source_prefix_base_velocity_feedforward: float = 0.
    servo_joint_margin_rad: float = 0.










def plan(store, row, candidate, robot, *, state_profile=False):
    import mujoco

    a, metadata = read_episode(store.root / row["path"])
    task = metadata.get("env_args", {}).get("env_name")
    if task not in TASK_OBJECTS:
        raise ValueError(f"No complete dynamics task adapter: {task}")
    a, source_start, original_times = source_window(a, metadata, candidate.source_start_s)
    oid = TASK_OBJECTS[task]
    E, O, grip, anchor, grasp_local, region = source_grasp(a, metadata, oid, candidate.geometry_center)
    grasp_local = grasp_local + [candidate.grasp_local_x, candidate.grasp_local_y, 0.]
    yaw = candidate.yaw
    placement = np.eye(4)
    placement[:3,:3] = Rotation.from_euler("z",candidate.scene_heading,degrees=True).as_matrix()
    placement[:2, 3] = [candidate.placement_x, candidate.placement_y]
    placement[2, 3] = source_vertical_offset(metadata)
    xml, manifest = build_scene(store.root, metadata, a, placement, state_profile=state_profile)
    xml = add_search_sensors(xml, manifest["objects"][oid]["body"])
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    initialize(model, data, robot, robot.q, manifest["mimics"], .63)
    initialize_fixtures(model, data, manifest)
    task_contract = scene_contract(a, metadata, model, data, placement)
    interaction = contact_contract(grip)
    if sum(p["mode"] == "grasp" for p in interaction["phases"]) != 1:
        raise ValueError("Repeated grasps need a multi-phase trajectory adapter; refusing to truncate interactions")
    site = model.site("r_arm_tip_tcp").id
    R = data.site_xmat[site].reshape(3, 3)
    fingers = [model.site(n+"_endpoint").id for n in ("r_hand_distal_link", "r_hand_distal_mimic_link")]
    grasp_geometry = None
    if candidate.automatic_grasp:
        body = model.body(manifest["objects"][oid]["body"]).id
        jaw_local = R.T @ (data.site_xpos[fingers[1]]-data.site_xpos[fingers[0]])
        source_local = (np.linalg.inv(O[anchor]) @ E[anchor])[:3, 3]
        grasp_geometry = geometry_seed(model, data, body, source_local,
                                      (placement @ O[anchor])[:3, :3], jaw_local,
                                      rank=candidate.grasp_region_rank)
        grasp_local = grasp_geometry["point"] + [candidate.grasp_local_x, candidate.grasp_local_y, 0.]
        yaw += grasp_geometry["yaw_deg"] if candidate.align_object_axes else 0.
        region = grasp_geometry["policy"] + ": " + str(grasp_geometry["collision_geom"])
        candidate = replace(candidate, grasp_force=grasp_geometry["normal_force_n"]*candidate.force_scale,
                            controlled_place=interaction["release_required"])
    offset = R.T @ (data.site_xpos[fingers].mean(0)-data.site_xpos[site])
    offset += [candidate.height, 0, candidate.depth]
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_euler("ZY", [yaw, candidate.tilt-90], degrees=True).as_matrix()
    grasp[:3, 3] = (placement @ O[anchor] @ np.r_[grasp_local, 1])[:3]-grasp[:3, :3] @ offset
    attachment = np.linalg.inv(placement @ E[anchor]) @ grasp
    goals = placement @ E @ attachment
    close = int(np.flatnonzero(grip > 0)[0])
    if candidate.object_relative:
        object_attachment = np.linalg.inv(placement @ O[anchor]) @ grasp
        goals = placement @ O @ object_attachment
        # Reconstruct a robot-specific pregrasp and withdrawal. During the
        # manipulation preserve the recorded object's full SE(3) reference;
        # no reference pose is assigned to the freely simulated object.
        approach = goals[close].copy()
        back = approach[:3,:3] @ np.array([0.,0.,.10]) + np.array([0.,0.,.10])
        start_rotation = Rotation.from_euler("ZY", [yaw,-90.],degrees=True).as_matrix()
        rotation_delta = Rotation.from_matrix(start_rotation.T @ approach[:3,:3]).as_rotvec()
        for i in range(close):
            x = a["time_s"][i]/max(a["time_s"][close],1e-9)
            blend = x**3*(10-15*x+6*x*x)
            goals[i] = approach
            goals[i,:3,3] += (1-blend)*back
            rotation_weight = np.clip((a["time_s"][i]-(a["time_s"][close]-.5))/.5,0,1)
            rotation_weight = rotation_weight**3*(10-15*rotation_weight+6*rotation_weight**2)
            goals[i,:3,:3] = start_rotation @ Rotation.from_rotvec(rotation_weight*rotation_delta).as_matrix()
        opened = np.flatnonzero((np.arange(len(grip)) > close) & (grip < 0))
        if len(opened):
            release = int(opened[0])
            retreat = goals[release].copy()
            for i in range(release+1,len(goals)):
                x = np.clip((a["time_s"][i]-a["time_s"][release])/.5,0,1)
                blend = x**3*(10-15*x+6*x*x)
                goals[i] = retreat
                goals[i,:3,3] += blend*np.array([-.04,0.,.10])
    # A source-clock pause permits physical closure before lifting. It changes
    # commanded robot timing only; source state and moving object stay intact.
    times = np.asarray(a["time_s"])*candidate.time_scale
    object_goals = placement @ O
    if candidate.close_hold:
        times = np.insert(times, close+1, times[close]+candidate.close_hold)
        times[close+2:] += candidate.close_hold
        goals = np.insert(goals, close+1, goals[close], axis=0)
        grip = np.insert(grip, close+1, grip[close])
        object_goals = np.insert(object_goals, close+1, object_goals[close], axis=0)
    if candidate.controlled_place:
        if not candidate.object_relative:
            raise ValueError("Controlled placement requires an object-relative grasp reference")
        releases = np.flatnonzero((np.arange(len(grip)) > close) & (grip < 0))
        if not len(releases):
            raise ValueError("Controlled placement requires a demonstrated release")
        release = int(releases[0])
        final_object = placement @ O[-1]
        placed = final_object @ object_attachment
        above = placed.copy()
        above[2,3] = max(goals[release,2,3], placed[2,3]+candidate.place_clearance)
        suffix = goals[release+1:].copy()
        suffix[:,:3,3] += placed[:3,3]-goals[release,:3,3]
        suffix[:,:3,:3] = placed[:3,:3]
        # Reach the demonstrated final support pose before opening, rather
        # than reproducing a source release that relies on a ballistic drop.
        goals = np.concatenate([goals[:release], [goals[release],above,placed,placed],suffix])
        object_goals = np.concatenate([object_goals[:release],
            [object_goals[release],above @ np.linalg.inv(object_attachment),final_object,final_object],
            np.repeat(final_object[None],len(suffix),axis=0)])
        times = np.r_[times[:release],times[release]+np.array([0.,1.,2.5,3.]),times[release+1:]+3.]
        grip = np.r_[grip[:release],1.,1.,1.,-1.,grip[release+1:]]
    targets = np.tile(robot.fk(robot.q), (len(goals), 1, 1, 1))
    targets[:, 1] = goals
    # Keep the inactive arm in its rest configuration; solve a mobile right arm.
    # A fixed initial base is preferred, then planar assistance is permitted.
    active = robot.active.copy()
    robot.active = robot.arm_v[7:]
    rng = np.random.default_rng(5)
    candidates = []
    for i in range(16):
        q = robot.q.copy()
        if i:
            ids = robot.arm_ids[7:]
            q[ids] = rng.uniform(robot.r.model.lowerPositionLimit[ids]+.06,
                                robot.r.model.upperPositionLimit[ids]-.06)
        q, pe, re = robot.ik(targets[0], q, active_hands=(1,), iterations=240, joint_margin=.07)
        gap, margin, _ = robot.r.geometry(q)
        candidates.append((pe+.3*re+10*max(0,.015-gap), q.copy(), pe, re))
        if pe < .001 and re < .01 and gap > .015:
            break
    _, q, initial_pe, initial_re = min(candidates, key=lambda x: x[0])
    robot.active = np.r_[np.arange(3), robot.arm_v[7:]] if candidate.mobile_base else robot.arm_v[7:]
    configurations, errors = [], []
    try:
        for target in targets:
            q, pe, re = robot.ik(target, q, active_hands=(1,), iterations=100, joint_margin=.07,
                                base_xy_bounds=((-100., candidate.base_x_max), (candidate.base_y_min, 100.)))
            configurations.append(q.copy())
            errors.append((pe, re))
    finally:
        robot.active = active
    q = np.asarray(configurations)
    controls = np.c_[np.array([robot.base(v) for v in q]), q[:, robot.arm_ids]]
    controls[:, 2] = np.unwrap(controls[:, 2])
    caps = np.r_[.40, .40, 1.2, np.full(14, .65)]
    velocity_ratio = np.max(np.abs(np.diff(controls, axis=0)/np.diff(times)[:, None])/caps)
    # Motion-dependent dilation provides headroom for servo overshoot. No caps
    # are weakened; actual simulated velocities are independently checked.
    dilation = max(1., float(velocity_ratio))
    if dilation > 8:
        raise ValueError(f"Discontinuous joint plan requires excessive dilation: {dilation}")
    times *= dilation
    seconds = np.arange(0., times[-1]+2.001, .01)
    reference = np.stack([np.interp(seconds, times, values) for values in controls.T], axis=1)
    from .contact import ramp_closure
    index = np.clip(np.searchsorted(times, seconds, side="right")-1, 0, len(times)-1)
    commands = ramp_closure(seconds, np.where(grip[index] > 0, -.06, 2.), candidate.close_duration)
    target_hands = pose_to_matrices(interpolate_pose(times, matrices_to_pose(goals), seconds))
    target_objects = pose_to_matrices(interpolate_pose(times, matrices_to_pose(object_goals), seconds))
    details = {"interaction": interaction, "effective_candidate": asdict(candidate),
               "geometry_seed": ({k: (v.tolist() if isinstance(v,np.ndarray) else v) for k,v in grasp_geometry.items()} if grasp_geometry else None),
               "task": task, "task_contract": task_contract, "object_id": oid, "source_anchor_frame": anchor+source_start,
               "window_anchor_frame": anchor,
               "source_start_frame": source_start, "source_start_s": float(original_times[source_start]),
               "source_window_policy": "same source episode; explicit stationary open-gripper prefix crop; no independent demonstration count" if source_start else "complete original source interval",
               "grasp_region": region, "grasp_point_in_object": grasp_local.tolist(),
               "wrist_attachment": attachment.tolist(), "world_placement": placement.tolist(),
               "source_time_scale": candidate.time_scale, "velocity_dilation": dilation,
               "grasp_yaw_deg": yaw, "planning_joint_margin_rad": .07,
               "duration_s": float(seconds[-1]), "initial_ik_position_error_m": initial_pe,
               "initial_ik_rotation_error_rad": initial_re,
               "max_ik_position_error_m": float(np.max(np.array(errors)[:, 0])),
               "max_ik_rotation_error_rad": float(np.max(np.array(errors)[:, 1])),
               "controller": candidate.controller,
               "release_policy": "move above final source support pose, lower, hold 0.5 s, then open" if candidate.controlled_place else "source release clock",
               "trajectory_seed": "object-relative pose with reconstructed pregrasp/retreat" if candidate.object_relative else "source end-effector attachment",
               "base_planning": "planar mobile" if candidate.mobile_base else "stationary base",
               "base_x_max_m": candidate.base_x_max,
               "base_y_min_m": candidate.base_y_min,
               "task_reference": REFERENCES[task], "final_hold_policy": "preserve source final gripper state for two seconds"}
    return model, xml, manifest, a, metadata, details, q[0], seconds, reference, commands, target_hands, target_objects




def rollout(store, row, label, candidate, *, control_path=None, prepared_plan=None, extra_sources=None, state_profile=False,
            acquisition_plan=None):
    """One uninterrupted actuator-only validation; preserve every attempt."""
    import mujoco
    out = store.root / "runs/dynamics" / label / row["id"]
    if out.exists():
        raise FileExistsError(f"Attempt already exists: {out}")
    out.mkdir(parents=True)
    started = time.monotonic()
    robot = Robot(store.root)
    report = {"created": now(), "source_episode": row["id"], "source_id": row["source_id"],
              "validation_protocol": "reachy-object-dynamics-v1",
              "source_sequence": row["source_sequence"], "candidate": asdict(candidate),
              "status": "plan_rejected", "physics_tested": False, "physics_validated": False, "success": False,
              "source_hdf5_sha256": sha256(store.root/row["path"]),
              "controller_identity": robot.identity, "controller_backend": robot.backend,
              "object_state_assignment": "reset only; actuator commands only during all physics steps",
              "runtime": {n: importlib.metadata.version(n) for n in ("mujoco", "numpy", "pin")}}
    for name,code in _IMPLEMENTATION_SOURCES.items():
        (out/name).write_text(code)
    report["implementation_sha256"] = {n: sha256(out/n) for n in _IMPLEMENTATION_SOURCES}
    for name, code in (extra_sources or {}).items():
        if Path(name).name != name:
            raise ValueError("Implementation snapshots must be plain filenames")
        (out/name).write_text(code)
        report["implementation_sha256"][name] = sha256(out/name)
    try:
        model, xml, manifest, a, metadata, details, q0, seconds, reference, commands, goals, object_goals = (
            plan(store, row, candidate, robot, state_profile=state_profile) if prepared_plan is None else prepared_plan)
    except Exception as exc:
        report["failure_reasons"] = [f"{type(exc).__name__}: {exc}"]
        json_write(out/"result.json", report)
        return report
    candidate = Candidate(**details["effective_candidate"])
    acquisition_clock = None
    original_intent = commands.copy()
    if acquisition_plan is not None:
        from . import acquisition_clock as clock_policy, acquisition_policy
        if (prepared_plan is None or control_path is not None or candidate.force_feedback
                or candidate.contact_release or candidate.calibrate_attachment):
            raise ValueError('Acquisition requires an isolated admitted reference and bounded closure controller')
        acquisition_adapter = acquisition_policy.adapter(acquisition_plan['metadata'])
        acquisition_adapter.verify(prepared_plan, acquisition_plan)
        for module in (acquisition_policy, clock_policy, acquisition_adapter):
            path = Path(module.__file__)
            (out/path.name).write_text(path.read_text())
            report['implementation_sha256'][path.name] = sha256(out/path.name)
        policy = acquisition_plan['metadata']
        if not np.isclose(policy['timestep_s'], .01, atol=1e-12, rtol=0):
            raise ValueError('Acquisition guard clock must match actual 100 Hz control intervals')
        acquisition_clock = clock_policy.AcquisitionClock(
            source_rows=len(seconds), first_close_index=int(policy['first_close_index']),
            acquisition_index=int(policy['acquisition_index']),
            entry_rows=len(acquisition_plan['arrays']['entry_reference']),
            exit_source_indices=[int(i) for i in acquisition_plan['arrays']['exit_source_indices']],
            timestep=policy['timestep_s'], wait_limit_s=policy['timeout_s'], stable_s=policy['stable_s'])
    report.update(plan=details, physics=manifest)
    report["physics"]["scene_asset_hashes"] = bind_scene_assets(xml, out)
    json_write(out/"attempt.json", report)
    (out/"scene.xml").write_text(xml)
    report["scene_sha256"] = sha256(out/"scene.xml")
    data = mujoco.MjData(model)
    substeps = round(.01/model.opt.timestep)
    if substeps < 1 or not np.isclose(substeps*model.opt.timestep,.01,atol=1e-12,rtol=0):
        raise ValueError('Physics timestep must exactly divide the 100 Hz control interval')
    initialize(model, data, robot, q0, manifest["mimics"], 2.)
    initialize_fixtures(model, data, manifest)
    initial_state = {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy()}
    state_recorder = None
    if state_profile:
        from .state_recording import RolloutRecorder, _source_provenance
        if not manifest.get("neck_joints_enabled"):
            raise ValueError("Complete state recording requires the neck-enabled model profile")
        _source_provenance(metadata, row.get('source_provenance'))
        state_recorder = RolloutRecorder(model, dict(manifest, scene_sha256=report["scene_sha256"]), details, out)
    prescribed = None
    if control_path is not None:
        with np.load(control_path, allow_pickle=False) as supplied:
            prescribed = supplied["control"].copy()
            expected_scene = str(supplied["scene_sha256"])
            expected_initial = supplied["initial_qpos"]
            if "contact_closed" in supplied:
                commands = np.where(supplied["contact_closed"], -.06, 2.)
        if (prescribed.shape != (len(seconds), model.nu) or not np.isfinite(prescribed).all()
                or expected_scene != report["scene_sha256"]
                or not np.array_equal(expected_initial, initial_state["qpos"])):
            raise ValueError("Optimized control is not bound to this exact scene and initial state")
        report["optimized_control"] = {"path": str(control_path), "sha256": sha256(control_path)}
    with h5py.File(out/"plan.h5", "w") as f:
        f["time_s"] = seconds
        f["reference"] = reference
        f["gripper"] = commands
        if acquisition_clock is not None:
            f['original_gripper_intent'] = original_intent
        f["object_goals"] = matrices_to_pose(object_goals)
        f["hand_goals"] = matrices_to_pose(goals)
        for key, value in initial_state.items():
            f["initial/"+key] = value

    oid, task = details["object_id"], details["task"]
    placement = np.asarray(details["world_placement"])
    object_bodies = {k: model.body(v["body"]).id for k, v in manifest["objects"].items()}
    active_body = object_bodies[oid]
    active_geoms = set(np.flatnonzero(model.geom_bodyid == active_body))
    robot_bodies = {model.body("base_link").id}
    for bid in range(1,model.nbody):
        if int(model.body_parentid[bid]) in robot_bodies:
            robot_bodies.add(bid)
    finger_names = ("r_hand_distal_link", "r_hand_distal_mimic_link")
    finger_bodies = {model.body(n).id for n in finger_names}
    hand_bodies = {i for i in robot_bodies if "hand_" in model.body(i).name}
    dofs = np.array([model.joint(n).dofadr[0] for n in (*BASE, *ARMS)])
    actuators = np.array([model.actuator(n).id for n in (*BASE, *ARMS)])
    grip_act = model.actuator("r_hand_finger").id
    grip_q = model.joint("r_hand_finger").qposadr[0]
    limit_joints = np.array([model.joint(n).id for n in ARMS])
    limit_qpos = model.jnt_qposadr[limit_joints]
    tcp = model.site("r_arm_tip_tcp").id
    initial = data.xpos[active_body].copy()
    records = {k: [] for k in ("time_s", "simulation/qpos", "simulation/qvel", "simulation/actuator_control",
               "simulation/hand_pose_world", "command/gripper_position", "metrics/bilateral_contact",
               "metrics/hand_object_penetration_m", "metrics/task_satisfied", "metrics/grasp_drift_m_rad",
               "metrics/object_reference_error_m_rad", "command/applied_joint_reference",
               "target/applied_right_hand_pose_world")}
    for name in object_bodies:
        records[f"objects/{name}/pose"] = []
    if acquisition_clock is not None:
        for key in ('source/reference_index', 'source/reference_time_s', 'source/original_gripper_intent',
                    'command/effective_gripper_intent', 'acquisition/phase', 'acquisition/guard_json'):
            records[key] = []
    maxima = dict(hand_depth=0., environment_depth=0., object_environment_depth=0., self_depth=0., lift=0., drift_m=0., drift_rad=0.)
    forbidden_object_robot_contacts = 0
    peak = np.zeros(len(dofs))
    min_margin, min_gap = np.inf, np.inf
    steady = best_steady = 0.
    drift_anchor = None
    carry_bilateral, closure = [], 2.
    failures = []
    normal_force = candidate.grasp_force
    gripper_contact = False
    previous_bilateral = 0.
    support_force = 0.
    support_force_total = 0.
    support_release_step = None
    closed_frames = np.flatnonzero(commands < 0)
    releases = np.flatnonzero((np.arange(len(commands)) > closed_frames[0]) & (commands >= 1.9999)) if len(closed_frames) else []
    release_frame = int(releases[0]) if len(releases) else len(commands)
    object_weight = float(model.body_subtreemass[active_body]*np.linalg.norm(model.opt.gravity))
    placement_controller = None
    placement_updates = None
    if details.get('robot_supported_placement'):
        from .supported_placement import Controller
        if candidate.contact_release or candidate.calibrate_attachment or candidate.arm_velocity_feedforward or prescribed is not None:
            raise ValueError('Supported placement requires its isolated admitted reference policy')
        placement_controller=Controller(details,reference,goals,commands)
        placement_updates = (placement_controller if acquisition_clock is None else
                             clock_policy.SourceIndexedController(placement_controller))
    ff_offset = np.zeros(len(actuators))
    from .servo_feedforward import velocity_offsets
    velocity_ff = velocity_offsets(model, actuators, reference, seconds, candidate.arm_velocity_feedforward)
    if candidate.base_velocity_feedforward:
        if placement_controller is not None or acquisition_clock is not None or prescribed is not None:
            raise ValueError('Whole-source base feedforward requires an unmodified source clock without inserted feedback phases or prescribed controls')
        from .servo_feedforward import base_velocity_offsets
        velocity_ff += base_velocity_offsets(model, actuators, reference, seconds,
                                             candidate.base_velocity_feedforward)
        details['base_feedforward'] = dict(scale=candidate.base_velocity_feedforward,boundary_taper_s=.1,
            maximum_world_xy_offset_m=.05, maximum_yaw_offset_rad=.1,
            scope='Bounded base kv/kp velocity compensation over the complete reference clock; original references and model gains unchanged')
    if candidate.source_prefix_velocity_feedforward:
        if placement_controller is None:
            raise ValueError('Source-prefix feedforward requires an explicit placement boundary')
        from .servo_feedforward import source_prefix_offsets
        prefix_end=placement_controller.phases['traverse'][0]
        velocity_ff=source_prefix_offsets(model,actuators,reference,seconds,
            candidate.source_prefix_velocity_feedforward,prefix_end)
        details['source_prefix_feedforward']=dict(scale=candidate.source_prefix_velocity_feedforward,
            stop_frame=prefix_end,boundary_taper_s=.1,maximum_position_offset_rad=.2,
            scope='Arm velocity compensation before inserted placement only; zero throughout support feedback and original suffix')
    if candidate.source_prefix_base_velocity_feedforward:
        if placement_controller is None:
            raise ValueError('Source-prefix base compensation requires an explicit placement boundary')
        from .servo_feedforward import source_prefix_base_offsets
        prefix_end=placement_controller.phases['traverse'][0]
        velocity_ff+=source_prefix_base_offsets(model,actuators,reference,seconds,
            candidate.source_prefix_base_velocity_feedforward,prefix_end)
        details['source_prefix_base_feedforward']=dict(scale=candidate.source_prefix_base_velocity_feedforward,
            stop_frame=prefix_end,boundary_taper_s=.1,maximum_world_xy_offset_m=.05,maximum_yaw_offset_rad=.1,
            scope='Base kv/kp velocity compensation before inserted placement only; original references unchanged, zero throughout support feedback and original suffix')
    post_acquisition_base_ff = None
    if candidate.post_acquisition_base_velocity_feedforward:
        if (acquisition_clock is None or placement_controller is not None or prescribed is not None
                or candidate.base_velocity_feedforward or candidate.source_prefix_base_velocity_feedforward):
            raise ValueError('Post-acquisition base feedforward requires isolated measured acquisition and its original source suffix')
        from .servo_feedforward import PostAcquisitionBaseOffsets
        resume_index = int(acquisition_plan['arrays']['exit_source_indices'][-1])+1
        post_acquisition_base_ff = PostAcquisitionBaseOffsets(model, actuators, reference, seconds,
            candidate.post_acquisition_base_velocity_feedforward, resume_index)
        details['post_acquisition_base_feedforward'] = dict(
            scale=candidate.post_acquisition_base_velocity_feedforward, resume_index=resume_index,
            boundary_taper_s=.1, maximum_world_xy_offset_m=.05, maximum_yaw_offset_rad=.1,
            scope='Zero before and throughout acquisition/exit; unchanged source base velocity with actual-time/source-progress resume taper; model and original references unchanged')
    from . import servo_envelope
    servo_bounds = None
    if candidate.servo_joint_margin_rad:
        lower, upper = servo_envelope.limits(model, actuators[3:], candidate.servo_joint_margin_rad)
        servo_bounds = np.r_[np.full(3, -np.inf), lower], np.r_[np.full(3, np.inf), upper]
    applied_reference = reference[0].copy()
    correction_age = 0.
    correction_velocity = np.zeros(len(actuators))
    robot.active = np.r_[np.arange(3), robot.arm_v[7:]] if candidate.mobile_base else robot.arm_v[7:]
    try:
        frames = (acquisition_clock.frames() if acquisition_clock is not None else range(len(seconds)))
        for frame in frames:
            step = frame.reference_index if acquisition_clock is not None else frame
            desired_reference = reference[step]
            applied_hand_goal = goals[step].copy()
            if placement_controller is not None:
                desired_reference,applied_hand_goal,commands[step]=placement_updates.update(
                    step,support_force_total,object_weight,object_position_world=data.xpos[active_body].copy(),
                    grasp_intact=bool(drift_anchor is not None and previous_bilateral>=.95))
            intent, frame_ff = commands[step], velocity_ff[step]
            if acquisition_clock is not None:
                desired_reference, applied_hand_goal, intent, frame_ff = clock_policy.command(
                    frame, desired_reference, applied_hand_goal, intent, frame_ff, acquisition_plan)
            if post_acquisition_base_ff is not None:
                frame_ff = frame_ff+post_acquisition_base_ff.update(step, frame.phase, float(data.time))
            if candidate.calibrate_attachment and drift_anchor is not None and commands[step] < 0:
                correction_age += .01
                calibrated_goal = object_goals[step] @ np.linalg.inv(drift_anchor)
                blend = min(1., correction_age)
                target = robot.fk(measured(model, data, robot))
                delta = Rotation.from_matrix(goals[step,:3,:3].T @ calibrated_goal[:3,:3]).as_rotvec()
                target[1,:3,:3] = goals[step,:3,:3] @ Rotation.from_rotvec(blend*delta).as_matrix()
                target[1,:3,3] = (1-blend)*goals[step,:3,3]+blend*calibrated_goal[:3,3]
                applied_hand_goal = target[1].copy()
                corrected, _, _ = robot.ik(target, measured(model,data,robot), active_hands=(1,),
                    iterations=25,joint_margin=.07,
                    base_xy_bounds=((-100.,candidate.base_x_max),(candidate.base_y_min,100.)))
                desired_reference = np.r_[robot.base(corrected),corrected[robot.arm_ids]]
            if candidate.calibrate_attachment:
                caps = np.r_[.25,.25,.8,np.full(14,.5)]
                velocity = np.clip((desired_reference-applied_reference)/.01,-caps,caps)
                acceleration = np.r_[.25,.25,1.,np.full(14,1.)]
                correction_velocity += np.clip(velocity-correction_velocity,-acceleration*.01,acceleration*.01)
                applied_reference += correction_velocity*.01
            else:
                applied_reference = desired_reference.copy()
            data.ctrl[actuators] = (applied_reference+ff_offset+frame_ff if servo_bounds is None
                                    else servo_envelope.target(applied_reference, ff_offset, frame_ff, servo_bounds))
            if (candidate.contact_release and prescribed is None and support_release_step is None
                    and details["interaction"]["release_required"]
                    and supported_release(data.xpos[active_body], object_goals[-1,:3,3], support_force, object_weight,
                                          lifted=maxima["lift"] > .015,
                                          near_release=release_frame-200 <= step < release_frame)):
                support_release_step = step
                opening = np.minimum(2., closure+(np.arange(len(commands)-step)+1)*.01*2/.3)
                commands[step:] = np.maximum(commands[step:], opening)
                intent = commands[step]
            if (candidate.force_feedback and commands[step] < 1.9999 and support_release_step is None
                    and (placement_controller is None or commands[step] < 0)):
                if (candidate.force_preclose_target is not None and
                        (step == 0 or commands[step-1] >= 1.9999)):
                    # Set a geometric feedforward actuator target once at the
                    # source closure transition; no simulated pose is changed.
                    closure = float(np.clip(candidate.force_preclose_target, -.06, 2.))
                # Contact force changes only a gripper actuator reference.
                change = np.clip(.01*(normal_force-candidate.grasp_force),-.015,.015)
                if candidate.soft_force_feedback:
                    change = (np.clip(.004*(normal_force-candidate.grasp_force),-.005,.005)
                              if gripper_contact else -.015)
                closure = float(np.clip(closure+change, -.06, 2.))
            else:
                # Keep source contact intent separate from a robot-specific
                # aperture target. Positive contact angles must not erase the
                # closed/grasp phase used by the unchanged physical gates.
                closure = float(candidate.gripper_closed_target if intent < 0 else intent)
            data.ctrl[grip_act] = closure
            if prescribed is not None:
                data.ctrl[:] = prescribed[step]
                closure = float(prescribed[step, grip_act])
            if state_recorder is not None:
                state_recorder.begin(data, {
                    "format": "reachy-retarget-position-servo-state-v1", "identity": robot.identity,
                    "joints": {name: float(data.qpos[model.joint(name).qposadr[0]])
                               for name in state_recorder.metadata["command_joint_names"]},
                    "targets": {"joint_position": data.ctrl.tolist(),
                                "right_tcp_reference_world": matrices_to_pose(applied_hand_goal).tolist()},
                    "servo_targets": {"joints": dict(zip(state_recorder.metadata["command_joint_names"], data.ctrl.tolist()))},
                    "memory": {"ff_offset": ff_offset.tolist(), "velocity_feedforward": frame_ff.tolist(), "applied_reference": applied_reference.tolist(),
                               "correction_velocity": correction_velocity.tolist(), "correction_age": correction_age,
                               "closure": closure, "normal_force": normal_force, "gripper_contact": bool(gripper_contact),
                               "support_force": support_force, "support_release_step": support_release_step,
                               "support_force_total_interval_mean_N": support_force_total,
                               "supported_placement": None if placement_controller is None else placement_controller.state,
                               "drift_anchor": None if drift_anchor is None else drift_anchor.tolist(),
                               "candidate": asdict(candidate), "source_step": step,
                               **({} if post_acquisition_base_ff is None else {
                                   'post_acquisition_base_feedforward': post_acquisition_base_ff.state}),
                               **({} if acquisition_clock is None else {'acquisition_phase': frame.phase,
                                   'source_reference_time_s': float(seconds[step]),
                                   'original_gripper_intent': float(original_intent[step]),
                                   'effective_gripper_intent': float(intent)})}})
            bilateral_substeps = 0
            hand_contact = False
            forces = {b: 0. for b in finger_bodies}
            interval_depth = 0.
            support_force = 0.
            support_force_total = 0.
            for substep in range(substeps):
                if state_recorder is not None:
                    state_recorder.before_step(data)
                mujoco.mj_step(model, data)
                velocity = data.qvel[dofs].copy()
                yaw = data.qpos[model.joint("base_yaw").qposadr[0]]
                velocity[:2] = Rotation.from_euler("z", -yaw).as_matrix()[:2,:2] @ velocity[:2]
                peak = np.maximum(peak, np.abs(velocity))
                joint_positions = data.qpos[limit_qpos]
                min_margin = min(min_margin, float(np.min(np.minimum(
                    joint_positions-model.jnt_range[limit_joints,0],
                    model.jnt_range[limit_joints,1]-joint_positions))))
                touched = set()
                substep_support_sum = 0.
                for index, c in enumerate(data.contact):
                    b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                    pair, depth = {b1,b2}, max(0., -float(c.dist))
                    if active_body in pair and pair & hand_bodies:
                        hand_contact = True
                        interval_depth = max(interval_depth, depth)
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, index, force)
                        for b in pair & finger_bodies:
                            if force[0] > 1e-6:
                                touched.add(b)
                            forces[b] = max(forces[b], float(force[0]))
                    if pair <= robot_bodies:
                        maxima["self_depth"] = max(maxima["self_depth"], depth)
                    elif pair & robot_bodies and active_body not in pair:
                        maxima["environment_depth"] = max(maxima["environment_depth"], depth)
                    if active_body in pair and not pair & robot_bodies:
                        maxima["object_environment_depth"] = max(maxima["object_environment_depth"],depth)
                        upward = c.frame[2] * (1 if b2 == active_body else -1)
                        if upward > .5:
                            reaction = np.zeros(6)
                            mujoco.mj_contactForce(model,data,index,reaction)
                            support_force = max(support_force, float(reaction[0]*upward))
                            substep_support_sum += max(0.,float(reaction[0]*upward))
                    if active_body in pair and pair & (robot_bodies-hand_bodies):
                        forbidden_object_robot_contacts += 1
                bilateral_substeps += int(touched == finger_bodies)
                support_force_total += substep_support_sum/float(substeps)
            if state_recorder is not None:
                state_recorder.end(data)
            bilateral = bilateral_substeps / float(substeps)
            previous_bilateral = bilateral
            normal_force = min(forces.values()) if candidate.balanced_force else max(forces.values())
            gripper_contact = max(forces.values()) > 1e-6
            mujoco.mj_forward(model, data)
            guard_result = {}
            if acquisition_clock is not None and frame.phase in ('settle', 'close'):
                guard_result = acquisition_adapter.guard(model, data, acquisition_plan)
                acquisition_clock.observe(alignment=guard_result['alignment'],
                    bilateral_force=bool(guard_result['bilateral_force'] and bilateral == 1.))
            qm = measured(model, data, robot)
            # Bounded integral servo correction compensates static contact load
            # without overwriting joints or applying forces to the object.
            actual_controls = np.r_[robot.base(qm), qm[robot.arm_ids]]
            if candidate.integral_compensation:
                if servo_bounds is None:
                    ff_offset += .02*(applied_reference-actual_controls)
                    ff_offset = np.clip(ff_offset, -.03, .03)
                else:
                    ff_offset = servo_envelope.integrate(ff_offset, applied_reference-actual_controls,
                                                         applied_reference, frame_ff, servo_bounds)
            gap, margin, _ = robot.r.geometry(qm)
            min_gap, min_margin = min(min_gap,gap), min(min_margin,margin)
            p = data.xpos[active_body].copy()
            lift = float(p[2]-initial[2]); maxima["lift"] = max(maxima["lift"],lift)
            maxima["hand_depth"] = max(maxima["hand_depth"], interval_depth)
            hand_matrix = np.eye(4); hand_matrix[:3,:3] = data.site_xmat[tcp].reshape(3,3);hand_matrix[:3,3] = data.site_xpos[tcp]
            obj_matrix = pose_to_matrices(np.r_[p,data.xquat[active_body]])
            object_error = np.linalg.inv(object_goals[step]) @ obj_matrix
            reference_error = [float(np.linalg.norm(object_error[:3,3])),
                               float(Rotation.from_matrix(object_error[:3,:3]).magnitude())]
            relative = np.linalg.inv(hand_matrix) @ obj_matrix
            drift = [np.nan,np.nan]
            if intent < 0 and lift > .015 and bilateral == 1. and drift_anchor is None:
                drift_anchor = relative.copy()
            if drift_anchor is not None and intent < 0:
                delta = np.linalg.inv(drift_anchor) @ relative
                drift = [float(np.linalg.norm(delta[:3,3])),float(Rotation.from_matrix(delta[:3,:3]).magnitude())]
                maxima["drift_m"] = max(maxima["drift_m"],drift[0]);maxima["drift_rad"] = max(maxima["drift_rad"],drift[1])
                carry_bilateral.append(bilateral)
            body_vel = np.zeros(6)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,active_body,body_vel,0)
            closed = float(data.qpos[grip_q]) < 1.7
            task_ok = task_state(task,oid,model,data,object_bodies,placement,closed,hand_contact,initial,
                                 contract=details["task_contract"])
            stable = (task_ok and np.linalg.norm(body_vel[3:]) < .02 and np.linalg.norm(body_vel[:3]) < .2
                      and (not details["interaction"]["terminal_contact"] or bilateral == 1.))
            steady = steady+.01 if stable else 0.;best_steady=max(best_steady,steady)
            records["time_s"].append(float(data.time))
            records["simulation/qpos"].append(data.qpos.copy());records["simulation/qvel"].append(data.qvel.copy())
            records["simulation/actuator_control"].append(data.ctrl.copy())
            records["simulation/hand_pose_world"].append(matrices_to_pose(robot.fk(qm)))
            records["command/gripper_position"].append(closure)
            records["metrics/bilateral_contact"].append(bilateral)
            records["metrics/hand_object_penetration_m"].append(interval_depth)
            records["metrics/task_satisfied"].append(task_ok)
            records["metrics/grasp_drift_m_rad"].append(drift)
            records["metrics/object_reference_error_m_rad"].append(reference_error)
            records["command/applied_joint_reference"].append(applied_reference.copy())
            records["target/applied_right_hand_pose_world"].append(matrices_to_pose(applied_hand_goal))
            if acquisition_clock is not None:
                records['source/reference_index'].append(int(step))
                records['source/reference_time_s'].append(float(seconds[step]))
                records['source/original_gripper_intent'].append(float(original_intent[step]))
                records['command/effective_gripper_intent'].append(float(intent))
                records['acquisition/phase'].append(frame.phase)
                records['acquisition/guard_json'].append(json.dumps(guard_result))
            for name,b in object_bodies.items():
                records[f"objects/{name}/pose"].append(np.r_[data.xpos[b],data.xquat[b]])
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Nonfinite dynamic state")
            if maxima["self_depth"] > .01 or maxima["environment_depth"] > .01:
                raise RuntimeError("Early stop: severe robot collision")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    fraction = float(np.mean(carry_bilateral)) if carry_bilateral else 0.
    gates = {"task_stable_final_1s": steady >= 1.,
             **contact_gates(details["interaction"]["mode"], acquired=drift_anchor is not None,
                             fraction=fraction, translation=maxima["drift_m"], rotation=maxima["drift_rad"]),
             "hand_object_penetration_1mm": maxima["hand_depth"] <= .001,
             "robot_environment_penetration_2mm": maxima["environment_depth"] <= .002,
             "object_environment_penetration_2mm": maxima["object_environment_depth"] <= .002,
             "no_nonhand_object_robot_contact": forbidden_object_robot_contacts == 0,
             "robot_self_penetration_2mm": maxima["self_depth"] <= .002,
             "actual_arm_speed_1rad_s": bool(np.max(peak[3:]) <= 1.001),
             "actual_base_speed_limits": bool(np.all(peak[:3] <= np.r_[.611,.611,np.deg2rad(114)+.001])),
             "joint_margin_25mrad": min_margin >= .025, "self_clearance_9mm": min_gap >= .009,
             "rollout_complete": (len(records["time_s"]) == len(seconds) if acquisition_clock is None else
                 acquisition_clock.complete and clock_policy.complete_reference_coverage(records['source/reference_index'], len(seconds)))}
    if acquisition_clock is not None:
        gates['measured_acquisition_before_carry'] = acquisition_clock.wait.phase == 'carry'
        report['acquisition_execution'] = dict(acquisition_clock.report(),
            total_repeated_reference_frames=len(records['source/reference_index'])-len(set(records['source/reference_index'])),
            total_inserted_duration_s=.01*(len(records['source/reference_index'])-len(set(records['source/reference_index']))))
    if post_acquisition_base_ff is not None:
        report['post_acquisition_base_feedforward'] = dict(post_acquisition_base_ff.state)
    if placement_controller is not None:
        gates['placement_opened_after_measured_support']=(placement_controller.supported and placement_controller.opened
            and placement_controller.opening_verified and not placement_controller.failed)
        report['supported_placement']=placement_controller.state
    gates = {k: bool(v) for k, v in gates.items()}
    failures.extend(k for k,v in gates.items() if not v)
    report.update(status="physical_fail" if failures else "physical_pass", success=not failures,
                  physics_tested=bool(records["time_s"]),
                  physics_validated=not failures, failure_reasons=failures, gates=gates,
                  max_lift_m=maxima["lift"], final_stable_s=steady, best_stable_s=best_steady,
                  max_hand_object_penetration_m=maxima["hand_depth"], max_environment_penetration_m=maxima["environment_depth"],
                  max_object_environment_penetration_m=maxima["object_environment_depth"],
                  forbidden_object_robot_contacts=forbidden_object_robot_contacts,
                  max_self_penetration_m=maxima["self_depth"], max_grasp_drift_m=maxima["drift_m"],
                  max_grasp_drift_deg=float(np.rad2deg(maxima["drift_rad"])), bilateral_carry_fraction=fraction,
                  peak_base_velocity=peak[:3].tolist(), peak_arm_velocity=peak[3:].tolist(),
                  min_joint_margin_rad=min_margin,min_self_clearance_m=min_gap,
                  final_object_reference_error_m_rad=(records["metrics/object_reference_error_m_rad"][-1]
                                                     if records["time_s"] else None),
                  recorded_frames=len(records["time_s"]),
                  simulated_s=float(data.time), wall_s=time.monotonic()-started,
                  supported_release_time_s=(float(seconds[support_release_step]) if support_release_step is not None else None))
    with h5py.File(out/"plan.h5", "a") as f:
        f["gripper"][:] = commands
    with h5py.File(out/"replay.h5","w") as f:
        for key,value in initial_state.items():
            f["initial/"+key] = value
        for key,values in records.items():
            dtype = h5py.string_dtype() if key in ('acquisition/phase', 'acquisition/guard_json') else None
            f.create_dataset(key,data=values,dtype=dtype,compression="gzip")
        indices = (np.asarray(records['source/reference_index'], dtype=int) if acquisition_clock is not None
                   else np.arange(len(records['time_s'])))
        f["target/right_hand_pose_world"] = matrices_to_pose(goals[indices])
        f["target/object_pose_world"] = matrices_to_pose(object_goals[indices])
        f["command/joint_reference"] = reference[indices]
        f.attrs["metadata_json"] = json.dumps(report)
        f.attrs["units"] = "m,rad,s; poses xyz+wxyz; simulation interval-end states"
    if state_recorder is not None:
        try:
            report["state_recording"] = state_recorder.finish(
                data, metadata, report, row["id"], source_provenance=row.get("source_provenance"))
        except Exception as error:
            # Metadata/export failure must not discard a completed simulation.
            reason='State recording failed: '+type(error).__name__+': '+str(error)
            report.update(status='recording_failed',success=False,physics_validated=False)
            report['failure_reasons'].append(reason)
            report['gates']['state_recording_complete']=False
            report['state_recording']=dict(status='recording_failed',error=reason,
                path=str(out/'state-attempt.hdf5') if (out/'state-attempt.hdf5').exists() else None)
        with h5py.File(out/'replay.h5','r+') as f:f.attrs['metadata_json']=json.dumps(report)
    json_write(out/"result.json",report)
    print(json.dumps({k:report[k] for k in ("source_sequence","status","failure_reasons","max_lift_m","final_stable_s","max_grasp_drift_m","max_grasp_drift_deg","max_hand_object_penetration_m","wall_s")}),flush=True)
    return report


def objective(report):
    """Rank physical failures; no ranking score can override a validation gate."""
    if report["status"] == "plan_rejected":
        return 1e6
    score = 0.
    score += 1000. if not report["gates"]["task_stable_final_1s"] else 0.
    score += 300.*(1-report["bilateral_carry_fraction"])
    error = report.get("final_object_reference_error_m_rad")
    if error is not None:
        score += 100.*error[0] + 10.*error[1]
    score += 100.*max(0., report["max_hand_object_penetration_m"]/.001-1)
    score += 100.*max(0., report["max_environment_penetration_m"]/.002-1)
    score += 100.*max(0., report["max_self_penetration_m"]/.002-1)
    score += 20.*max(0., report["max_grasp_drift_m"]/.003-1)
    score += 20.*max(0., report["max_grasp_drift_deg"]/3.-1)
    score += 10.*len(report["failure_reasons"])
    return float(score)


def search(store, row, label, seed, budget=32):
    """Deterministic simulation-guided coordinate search over robot actions.

    No candidate changes scene/object physical parameters or pass thresholds.
    Stop at the first complete pass. Search episodes must be reported apart
    from the subsequent fixed-parameter held-out source episodes.
    """
    changes = [("tilt",v) for v in (0.,5.,10.,20.)]
    changes += [("height",v) for v in (-.005,.005,.01,-.01,.015)]
    changes += [("depth",v) for v in (.015,.020,.030,.035)]
    changes += [("grasp_force",v) for v in (.8,1.,2.,3.)]
    changes += [("yaw",v) for v in (-5.,5.,-10.,10.)]
    changes += [("placement_y",v) for v in (-.3,-.1,0.)]
    changes += [("scene_heading",v) for v in (90.,-90.,180.)]
    best, best_score, best_report, history = seed, np.inf, None, []
    seen = set()
    for index in range(budget):
        candidate = seed if index == 0 else replace(best, **dict([changes[(index-1)%len(changes)]]))
        key = json.dumps(asdict(candidate),sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        attempt_label = label+f"/candidate_{index:03d}"
        report = rollout(store,row,attempt_label,candidate)
        score = objective(report)
        history.append({"attempt": attempt_label, "candidate": asdict(candidate), "score": score,
                        "status": report["status"], "failure_reasons": report["failure_reasons"]})
        if score < best_score:
            best, best_score = candidate, score
            best_report = report
        json_write(store.root/"runs/dynamics"/label/"search.json",
                   {"source_episode":row["id"],"source_sequence":row["source_sequence"],"seed":asdict(seed),
                    "selected":asdict(best),"best_score":best_score,"attempts":history,
                    "policy":"Development-only actuator/grasp/timing search. Fixed scene and gates. All failures retained."})
        if report["success"]:
            return report
    return best_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--task",choices=sorted(TASK_OBJECTS),action="append")
    parser.add_argument("--limit",type=int,default=1)
    parser.add_argument("--label",required=True)
    parser.add_argument("--search",action="store_true")
    parser.add_argument("--budget",type=int,default=32)
    for name,field in Candidate.__dataclass_fields__.items():
        if isinstance(field.default,bool):
            parser.add_argument("--"+name.replace("_","-"),action=argparse.BooleanOptionalAction,default=field.default)
        else:
            parser.add_argument("--"+name.replace("_","-"),type=type(field.default),default=field.default)
    args=parser.parse_args();store=Store(args.root.resolve())
    candidate=Candidate(**{name:getattr(args,name) for name in Candidate.__dataclass_fields__})
    counts={};reports=[]
    for row in store.rows("episodes"):
        _,metadata=read_episode(store.root/row["path"])
        task=metadata.get("env_args",{}).get("env_name")
        if task not in (args.task or TASK_OBJECTS) or counts.get(task,0)>=args.limit:continue
        counts[task]=counts.get(task,0)+1
        if args.search:
            reports.append(search(store,row,args.label+"/"+task+"/"+row["id"],candidate,args.budget))
        else:
            reports.append(rollout(store,row,args.label,candidate))
        json_write(store.root/"runs/dynamics"/args.label/"summary.json",reports)


if __name__ == "__main__":
    main()
