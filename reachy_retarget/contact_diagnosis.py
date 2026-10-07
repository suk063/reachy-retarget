"""Read-only diagnosis of saved actuator-driven contact attempts.

Replay resets once to the recorded initial state, then writes only recorded
robot actuator controls. It never adjusts object state, geometry, physics
parameters, acceptance gates, or any saved artifact. Contact sampling matches
the producer: MuJoCo's contacts immediately after each mj_step, before the
interval-end mj_forward. These contacts describe the step's collision state.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from .dynamics_audit import _load_recording, _verify_assets, _verify_no_object_assistance


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _subtree(model, root):
    result = {int(root)}
    for body in range(int(root) + 1, model.nbody):
        if int(model.body_parentid[body]) in result:
            result.add(body)
    return result


def contact_category(bodies, *, robot, hands, active):
    """Classify task-object contacts without treating other objects as robot."""
    bodies = set(bodies)
    if bodies <= active:
        return "object_self"
    if bodies & active:
        if bodies & hands:
            return "hand_object"
        if bodies & robot:
            return "nonhand_object_robot"
        return "object_environment"
    if bodies <= robot:
        return "robot_self"
    if bodies & robot:
        return "robot_environment"
    return "other"


def diagnose(attempt):
    """Return an integrity-bound, substep contact audit; never write files."""
    attempt = Path(attempt)
    report = json.loads((attempt / "result.json").read_text())
    scene = attempt / "scene.xml"
    if _sha(scene) != report["scene_sha256"]:
        raise ValueError("Saved scene checksum differs from its report")
    _verify_assets(scene, report["physics"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    objects = report["physics"]["objects"]
    object_joints = _verify_no_object_assistance(model, objects)
    with h5py.File(attempt / "replay.h5", "r") as handle:
        arrays, steps = _load_recording(handle, model, report, objects)
        optional = {key: handle[key][()] for key in (
            "metrics/bilateral_contact", "metrics/grasp_drift_m_rad",
            "metrics/task_satisfied", "command/gripper_position") if key in handle}
    plan = report["plan"]
    active_root = model.body(objects[plan["object_id"]]["body"]).id
    active = _subtree(model, active_root)
    robot = _subtree(model, model.body("base_link").id)
    hands = {body for body in robot if "hand_" in model.body(body).name}
    finger_names = ("r_hand_distal_link", "r_hand_distal_mimic_link")
    fingers = {model.body(name).id: name for name in finger_names}
    data = mujoco.MjData(model)
    data.qpos[:] = arrays["initial/qpos"]
    data.qvel[:] = arrays["initial/qvel"]
    data.ctrl[:] = arrays["initial/ctrl"]
    for joint in object_joints:
        index = int(model.jnt_qposadr[joint])
        if not np.allclose(data.qpos[index:index + 7], model.qpos0[index:index + 7], atol=1e-10, rtol=0):
            raise ValueError("Object reset differs from saved source scene")
    mujoco.mj_forward(model, data)
    scalar_joints = [i for i in range(model.njnt) if model.jnt_type[i] in
                     (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
    control_q = np.asarray([model.jnt_qposadr[int(model.actuator_trnid[i, 0])] for i in range(model.nu)])
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    grip = model.actuator("r_hand_finger").id
    grip_q = int(model.jnt_qposadr[model.joint("r_hand_finger").id])
    initial = dict(qvel_max_abs=float(np.max(np.abs(data.qvel), initial=0.)),
                   joint_velocity={model.joint(i).name: float(data.qvel[model.jnt_dofadr[i]]) for i in scalar_joints},
                   actuator_position_errors=dict(zip(actuator_names, (data.ctrl - data.qpos[control_q]).tolist())),
                   first_control_position_errors=dict(zip(actuator_names,
                       (arrays["simulation/actuator_control"][0] - data.qpos[control_q]).tolist())))
    events, initial_contacts = {}, []
    force_peaks = {name: {"normal_force_sum_n": 0., "time_s": None} for name in fingers.values()}
    peak_speed = {model.joint(i).name: {"absolute_velocity": 0., "time_s": None} for i in scalar_joints}
    peak_grip_force = 0.
    qpos_error = qvel_error = 0.

    def sample_contacts(initial_sample=False):
        sums = {body: 0. for body in fingers}
        for index, contact in enumerate(data.contact):
            ids = [int(contact.geom1), int(contact.geom2)]
            bodies = [int(model.geom_bodyid[i]) for i in ids]
            category = contact_category(bodies, robot=robot, hands=hands, active=active)
            if category == "other":
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            depth = max(0., -float(contact.dist))
            event = dict(time_s=float(data.time), geoms=[model.geom(i).name for i in ids],
                         bodies=[model.body(i).name for i in bodies], depth_m=depth,
                         normal_force_n=float(force[0]), position_world_m=contact.pos.tolist(),
                         gripper_target_rad=float(data.ctrl[grip]), gripper_position_rad=float(data.qpos[grip_q]))
            if initial_sample:
                initial_contacts.append(dict(category=category, **event))
                continue
            series = events.setdefault(category, {"first": event, "maximum_depth": event, "contact_samples": 0})
            series["contact_samples"] += 1
            if depth > series["maximum_depth"]["depth_m"]:
                series["maximum_depth"] = event
            if category == "hand_object" and force[0] > 0:
                for body in set(bodies) & fingers.keys():
                    sums[body] += float(force[0])
        if not initial_sample:
            for body, force in sums.items():
                if force > force_peaks[fingers[body]]["normal_force_sum_n"]:
                    force_peaks[fingers[body]] = dict(normal_force_sum_n=force, time_s=float(data.time))

    sample_contacts(initial_sample=True)
    for frame, count in enumerate(steps):
        data.ctrl[:] = arrays["simulation/actuator_control"][frame]
        for _ in range(int(count)):
            mujoco.mj_step(model, data)
            sample_contacts()
            peak_grip_force = max(peak_grip_force, abs(float(data.actuator_force[grip])))
            for joint in scalar_joints:
                speed = abs(float(data.qvel[model.jnt_dofadr[joint]]))
                name = model.joint(joint).name
                if speed > peak_speed[name]["absolute_velocity"]:
                    peak_speed[name] = dict(absolute_velocity=speed, time_s=float(data.time))
        mujoco.mj_forward(model, data)
        qpos_error = max(qpos_error, float(np.max(np.abs(data.qpos - arrays["simulation/qpos"][frame]), initial=0.)))
        qvel_error = max(qvel_error, float(np.max(np.abs(data.qvel - arrays["simulation/qvel"][frame]), initial=0.)))
    times = arrays["time_s"]
    metrics = {}
    if "metrics/bilateral_contact" in optional:
        value = optional["metrics/bilateral_contact"]
        indices = np.flatnonzero(value == 1.)
        metrics.update(first_full_bilateral_interval_end_s=float(times[indices[0]]) if len(indices) else None,
                       full_bilateral_intervals=int(len(indices)), intervals=int(len(value)))
    if "metrics/grasp_drift_m_rad" in optional:
        drift = optional["metrics/grasp_drift_m_rad"]
        valid = np.flatnonzero(np.isfinite(drift).all(axis=1))
        metrics["first_recorded_drift_anchor_interval_end_s"] = float(times[valid[0]]) if len(valid) else None
        metrics["grasp_drift_available"] = bool(len(valid))
    command = optional.get("command/gripper_position")
    if command is not None:
        metrics["issued_gripper_range_rad"] = [float(np.min(command)), float(np.max(command))]
    plan_path = attempt / "plan.h5"
    if plan_path.exists():
        with h5py.File(plan_path, "r") as f:
            closing = np.flatnonzero(f["gripper"][()] < 1.9999)
            metrics["first_requested_closure_plan_time_s"] = float(f["time_s"][closing[0]]) if len(closing) else None
    return dict(attempt=str(attempt), report_sha256=_sha(attempt / "result.json"),
                replay_sha256=_sha(attempt / "replay.h5"), scene_sha256=_sha(scene),
                source_hdf5_sha256=report.get("source_hdf5_sha256"), source_sequence=report.get("source_sequence"),
                runtime_mujoco_version=mujoco.__version__, recorded_runtime=report.get("runtime"),
                source_window=plan.get("source_start_s"), candidate=report.get("candidate"),
                plan_summary={key: plan.get(key) for key in ("alignment_variant", "source_time_scale",
                    "velocity_dilation", "trajectory_seed", "frozen_motion_sha256")},
                initial=initial, initial_contacts=initial_contacts, contacts=events,
                finger_force_peaks=force_peaks, peak_gripper_actuator_force=peak_grip_force,
                peak_scalar_joint_speed=peak_speed, recorded_contact_metrics=metrics,
                replay=dict(max_qpos_error=qpos_error, max_qvel_error=qvel_error,
                            exact_within_1e7=bool(max(qpos_error, qvel_error) < 1e-7)),
                reported_gates=report.get("gates"), reported_failures=report.get("failure_reasons"),
                reported_physics_validated=bool(report.get("physics_validated")),
                semantics="Independent actuator-only replay diagnosis; original gates/results unchanged; zero drift without an acquired anchor is unavailable, not success")


def passive_prefix_convergence(attempt, *, duration_s=.15,
                               timesteps=(.002, .001, .0005, .00025)):
    """Probe numerical convergence without changing contact/material parameters.

    Each variant starts from the identical saved reset once and applies the
    identical recorded controls at the saved control boundaries. This is an
    explicitly separate numerical experiment, not an amendment to a validation
    result. A prefix with robot/object contact is reported as nonpassive.
    """
    attempt = Path(attempt)
    report = json.loads((attempt / "result.json").read_text())
    scene = attempt / "scene.xml"
    if _sha(scene) != report["scene_sha256"]:
        raise ValueError("Saved scene checksum differs from its report")
    _verify_assets(scene, report["physics"])
    original = mujoco.MjModel.from_xml_path(str(scene))
    _verify_no_object_assistance(original, report["physics"]["objects"])
    with h5py.File(attempt / "replay.h5", "r") as f:
        arrays, _ = _load_recording(f, original, report, report["physics"]["objects"])
    interval = float(arrays["time_s"][0])
    frames = round(duration_s / interval)
    if (frames < 1 or frames > len(arrays["time_s"])
            or not np.isclose(frames * interval, duration_s, atol=1e-10, rtol=0)):
        raise ValueError("Convergence prefix must fit complete recorded control intervals")
    variants = []
    for timestep in timesteps:
        count = round(interval / timestep) if np.isfinite(timestep) and timestep > 0 else 0
        if count < 1 or not np.isclose(count * timestep, interval, atol=1e-12, rtol=0):
            raise ValueError("Numerical timestep must divide the recorded control interval")
        for tighten_solver in (False, True):
            model = mujoco.MjModel.from_xml_path(str(scene))
            model.opt.timestep = timestep
            if tighten_solver:
                model.opt.iterations = max(500, model.opt.iterations)
                model.opt.tolerance = min(1e-12, model.opt.tolerance)
            data = mujoco.MjData(model)
            data.qpos[:] = arrays["initial/qpos"]
            data.qvel[:] = arrays["initial/qvel"]
            data.ctrl[:] = arrays["initial/ctrl"]
            mujoco.mj_forward(model, data)
            robot = _subtree(model, model.body("base_link").id)
            oid = report["plan"]["object_id"]
            root = model.body(report["physics"]["objects"][oid]["body"]).id
            active = _subtree(model, root)
            depth, contact_time, geoms, robot_contact = 0., None, None, False
            boundary_poses = []
            reset_pose = np.r_[data.xpos[root], data.xquat[root]].tolist()
            for frame in range(frames):
                data.ctrl[:] = arrays["simulation/actuator_control"][frame]
                for _ in range(count):
                    mujoco.mj_step(model, data)
                    for contact in data.contact:
                        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
                        if not bodies & active:
                            continue
                        if bodies & robot:
                            robot_contact = True
                        elif -contact.dist > depth:
                            depth, contact_time = -float(contact.dist), float(data.time)
                            geoms = [model.geom(int(i)).name for i in (contact.geom1, contact.geom2)]
                mujoco.mj_forward(model, data)
                boundary_poses.append(np.r_[data.xpos[root], data.xquat[root]].tolist())
            variants.append(dict(timestep_s=timestep, iterations=int(model.opt.iterations),
                                 tolerance=float(model.opt.tolerance), solver=int(model.opt.solver),
                                 integrator=int(model.opt.integrator), initial_object_pose=reset_pose,
                                 max_object_environment_depth_m=depth, maximum_time_s=contact_time,
                                 maximum_geoms=geoms, robot_object_contact=robot_contact,
                                 object_pose_at_control_boundaries=boundary_poses))
    return dict(attempt=str(attempt), scene_sha256=_sha(scene), replay_sha256=_sha(attempt / "replay.h5"),
                duration_s=duration_s, control_interval_s=interval, variants=variants,
                changed_fields=["option.timestep", "option.iterations", "option.tolerance"],
                immutable_fields="Object/robot geometry, mass, inertia, friction, solref/solimp, gravity, initial qpos/qvel, recorded controls, and acceptance gates",
                semantics="Independent numerical-prefix experiments; no physical success claim or original artifact mutation")


def passive_admission(attempt, *, duration_s=.15):
    """Hold the recorded initial actuator targets to isolate startup contacts.

    This separate initial-hold experiment neither excludes a source from the
    denominator nor changes its original rollout. Passing the short prefix is
    necessary evidence only, not a prediction of complete task feasibility.
    """
    attempt = Path(attempt)
    report = json.loads((attempt / "result.json").read_text())
    scene = attempt / "scene.xml"
    if _sha(scene) != report["scene_sha256"]:
        raise ValueError("Saved scene checksum differs from its report")
    _verify_assets(scene, report["physics"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    joints = _verify_no_object_assistance(model, report["physics"]["objects"])
    with h5py.File(attempt / "replay.h5", "r") as f:
        arrays, _ = _load_recording(f, model, report, report["physics"]["objects"])
    count = round(duration_s / model.opt.timestep)
    if count < 1 or not np.isclose(count * model.opt.timestep, duration_s, atol=1e-10, rtol=0):
        raise ValueError("Admission duration must align with the original model timestep")
    data = mujoco.MjData(model)
    data.qpos[:] = arrays["initial/qpos"]
    data.qvel[:] = arrays["initial/qvel"]
    data.ctrl[:] = arrays["initial/ctrl"]
    for joint in joints:
        index = int(model.jnt_qposadr[joint])
        if not np.allclose(data.qpos[index:index + 7], model.qpos0[index:index + 7], atol=1e-10, rtol=0):
            raise ValueError("Object reset differs from saved source scene")
    mujoco.mj_forward(model, data)
    robot = _subtree(model, model.body("base_link").id)
    hands = {body for body in robot if "hand_" in model.body(body).name}
    oid = report["plan"]["object_id"]
    active = _subtree(model, model.body(report["physics"]["objects"][oid]["body"]).id)
    collisions = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    active_geoms = [i for i in collisions if int(model.geom_bodyid[i]) in active]
    environment_geoms = [i for i in collisions if int(model.geom_bodyid[i]) not in robot | active]
    gravity = np.asarray(model.opt.gravity)
    gravity_axis = gravity / np.linalg.norm(gravity) if np.linalg.norm(gravity) else None
    distances = []
    for first in active_geoms:
        for second in environment_geoms:
            if not ((model.geom_contype[first] & model.geom_conaffinity[second]) or
                    (model.geom_contype[second] & model.geom_conaffinity[first])):
                continue
            segment = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(model, data, first, second, 1., segment))
            if distance >= 1.:
                continue
            direction = (segment[3:] - segment[:3]) / distance if abs(distance) > 1e-12 else None
            if direction is None:
                # A touching support has a zero-length closest-point segment.
                # Recover its orientation from the actual reset contact normal.
                for contact in data.contact:
                    if {int(contact.geom1), int(contact.geom2)} == {first, second}:
                        direction = np.asarray(contact.frame[:3]) * (1 if int(contact.geom1) == first else -1)
                        break
            alignment = float(direction @ gravity_axis) if direction is not None and gravity_axis is not None else None
            distances.append(dict(distance_m=distance, geoms=[model.geom(i).name for i in (first, second)],
                                  closest_points_world_m=segment.tolist(), gravity_alignment=alignment))
    distances.sort(key=lambda row: row["distance_m"])
    supports = [row for row in distances if row["gravity_alignment"] is not None and row["gravity_alignment"] > .9]
    peaks, initial_peaks, first_contact = {}, {}, {}
    robot_object_contact_time = None

    def sample(initial=False):
        nonlocal robot_object_contact_time
        for contact in data.contact:
            gids = [int(contact.geom1), int(contact.geom2)]
            bodies = [int(model.geom_bodyid[i]) for i in gids]
            category = contact_category(bodies, robot=robot, hands=hands, active=active)
            if category in ("other", "object_self"):
                continue
            event = dict(time_s=float(data.time), depth_m=max(0., -float(contact.dist)),
                         geoms=[model.geom(i).name for i in gids], bodies=[model.body(i).name for i in bodies])
            first_contact.setdefault(category, event)
            destination = initial_peaks if initial else peaks
            if category not in destination or event["depth_m"] > destination[category]["depth_m"]:
                destination[category] = event
            if category in ("hand_object", "nonhand_object_robot") and robot_object_contact_time is None:
                robot_object_contact_time = float(data.time)

    sample(initial=True)
    completed, reason = True, None
    for _ in range(count):
        mujoco.mj_step(model, data)
        sample()
        bad_warnings = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                        mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                or any(data.warning[int(warning)].number for warning in bad_warnings)):
            completed, reason = False, "Nonfinite state or MuJoCo numerical warning in initial-hold experiment"
            break
    object_peak = peaks.get("object_environment", {"depth_m": 0., "time_s": None})
    isolated = robot_object_contact_time is None
    failed = object_peak["depth_m"] > .002
    return dict(attempt=str(attempt), source_sequence=report.get("source_sequence"),
                source_hdf5_sha256=report.get("source_hdf5_sha256"), scene_sha256=_sha(scene),
                replay_sha256=_sha(attempt / "replay.h5"), original_recorded_frames=report.get("recorded_frames"),
                original_gates=report.get("gates"), original_physics_validated=bool(report.get("physics_validated")),
                duration_requested_s=duration_s, duration_completed_s=float(data.time), completed=completed, failure=reason,
                initial_qvel_max_abs=float(np.abs(arrays["initial/qvel"]).max(initial=0.)),
                initial_nearest_environment=distances[0] if distances else None,
                initial_support_gap=supports[0] if supports else None,
                support_gap_semantics="Closest signed collision-surface distance whose object-to-environment segment is aligned with gravity; not a universal tabletop-height approximation",
                initial_contact_peaks=initial_peaks, first_contacts=first_contact, contact_peaks=peaks,
                robot_object_contact_time_s=robot_object_contact_time, isolated_passive_object=isolated,
                passive_object_environment_gate_pass=bool(completed and not failed),
                classification=("numerical_failure" if not completed else
                    "isolated_passive_source_model_gate_failure" if failed and isolated else
                    "object_environment_failure_with_robot_contact" if failed else
                    "no_passive_object_gate_failure_in_short_prefix"),
                initial_robot_placement_collision=any(initial_peaks.get(k, {}).get("depth_m", 0.) > .002
                    for k in ("robot_environment", "robot_self", "nonhand_object_robot")),
                control_policy="Constant recorded initial actuator targets for the entire prefix",
                semantics="Separate unchanged-model initial-hold diagnostic; does not remove sources or alter any original failure/pass")


def global_compliance_sensitivity(attempt, *, duration_s=.15,
                                  timesteps=(.001, .0005), time_constant_s=.004):
    """Explicit alternative-contact-model experiment, never source-faithful.

    The same cap applies to every geom and explicit contact pair, preserving
    damping ratio and all solimp/friction fields. Contact override is unused.
    This tests model sensitivity; it cannot promote an original validation pass.
    """
    attempt = Path(attempt)
    report = json.loads((attempt / "result.json").read_text())
    scene = attempt / "scene.xml"
    if _sha(scene) != report["scene_sha256"]:
        raise ValueError("Saved scene checksum differs from its report")
    _verify_assets(scene, report["physics"])
    original = mujoco.MjModel.from_xml_path(str(scene))
    joints = _verify_no_object_assistance(original, report["physics"]["objects"])
    with h5py.File(attempt / "replay.h5", "r") as f:
        arrays, _ = _load_recording(f, original, report, report["physics"]["objects"])
    for joint in joints:
        q = int(original.jnt_qposadr[joint])
        if not np.allclose(arrays["initial/qpos"][q:q + 7], original.qpos0[q:q + 7], atol=1e-10, rtol=0):
            raise ValueError("Object reset differs from saved source scene")
    if not np.isfinite(time_constant_s) or time_constant_s <= 0:
        raise ValueError("Positive global time constant required")
    variants = []
    for timestep in timesteps:
        if not np.isfinite(timestep) or timestep <= 0 or time_constant_s < 2 * timestep:
            raise ValueError("Positive timestep and time constant >= 2*timestep required")
        count = round(duration_s / timestep)
        if count < 1 or not np.isclose(count * timestep, duration_s, atol=1e-10, rtol=0):
            raise ValueError("Sensitivity duration must align with timestep")
        model = mujoco.MjModel.from_xml_path(str(scene))
        snapshot = {name: value.copy() for name in dir(model)
                    if isinstance(value := getattr(model, name), np.ndarray)}
        for name in ("geom_solref", "pair_solref"):
            values = getattr(model, name)
            if len(values) and np.any(values[:, 0] <= 0):
                raise ValueError("Direct stiffness/damping solref requires a separate declared profile")
            values[:, 0] = np.minimum(values[:, 0], time_constant_s)
        model.opt.timestep = timestep
        changed = [name for name, before in snapshot.items()
                   if not np.array_equal(before, getattr(model, name))]
        if set(changed) - {"geom_solref", "pair_solref"}:
            raise AssertionError("Global compliance probe changed an undeclared model array")
        fixed_digest = hashlib.sha256()
        for name, values in sorted(snapshot.items()):
            if name not in {"geom_solref", "pair_solref"}:
                fixed_digest.update(name.encode() + str(values.shape).encode() + values.tobytes())
        if (not np.array_equal(model.geom_solref[:, 1], snapshot["geom_solref"][:, 1])
                or not np.array_equal(model.pair_solref[:, 1], snapshot["pair_solref"][:, 1])):
            raise AssertionError("Contact damping ratios must remain unchanged")
        data = mujoco.MjData(model)
        data.qpos[:] = arrays["initial/qpos"]
        data.qvel[:] = arrays["initial/qvel"]
        data.ctrl[:] = arrays["initial/ctrl"]
        mujoco.mj_forward(model, data)
        robot = _subtree(model, model.body("base_link").id)
        hands = {body for body in robot if "hand_" in model.body(body).name}
        active = _subtree(model, model.body(report["physics"]["objects"][report["plan"]["object_id"]]["body"]).id)
        peaks, robot_contact, numerical_failure = {}, False, False
        for _ in range(count):
            mujoco.mj_step(model, data)
            for contact in data.contact:
                gids = [int(contact.geom1), int(contact.geom2)]
                bodies = [int(model.geom_bodyid[i]) for i in gids]
                category = contact_category(bodies, robot=robot, hands=hands, active=active)
                if category in ("other", "object_self"):
                    continue
                depth = max(0., -float(contact.dist))
                if category not in peaks or depth > peaks[category]["depth_m"]:
                    peaks[category] = dict(depth_m=depth, time_s=float(data.time),
                                           geoms=[model.geom(i).name for i in gids])
                robot_contact |= category in ("hand_object", "nonhand_object_robot")
            bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                   mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
            if any(data.warning[int(warning)].number for warning in bad):
                numerical_failure = True
                break
        variants.append(dict(timestep_s=timestep, duration_completed_s=float(data.time),
                             contact_peaks=peaks, robot_object_contact=robot_contact,
                             numerical_failure=numerical_failure, changed_model_array_fields=changed,
                             unchanged_model_array_field_count=len(snapshot) - 2,
                             unchanged_model_arrays_sha256=fixed_digest.hexdigest(),
                             solref_changes={name: {"before": snapshot[name].tolist(), "after": getattr(model, name).tolist()}
                                            for name in ("geom_solref", "pair_solref")}))
    return dict(attempt=str(attempt), source_sequence=report.get("source_sequence"),
                scene_sha256=_sha(scene), replay_sha256=_sha(attempt / "replay.h5"),
                source_faithful=False, validation_result_promoted=False,
                declared_global_rule=f"Cap every positive geom_solref/pair_solref time constant at {time_constant_s}s; preserve damping ratio",
                preserved="All other compiled model arrays, solimp, friction, mass/inertia, geometry, initial qpos/qvel, and constant initial actuator controls",
                contact_override_used=False, initial_state_sha256={key: hashlib.sha256(arrays["initial/" + key].tobytes()).hexdigest()
                    for key in ("qpos", "qvel", "ctrl")}, variants=variants,
                semantics="Short passive-prefix sensitivity to an explicitly different global contact-compliance model, not an original source-fidelity or task success")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.attempt), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
