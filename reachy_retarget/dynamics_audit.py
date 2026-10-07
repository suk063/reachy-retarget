"""Independent actuator-only replay of a saved dynamic attempt, without IK."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np

from .robot import ARMS, NECK
from .store import json_write, sha256


ROBOT_ACTUATED_JOINTS = frozenset((*ARMS, *NECK, "base_x", "base_y", "base_yaw",
                                    "l_hand_finger", "r_hand_finger"))


def bind_scene_assets(xml, base_dir):
    """Hash all external scene assets before a run; inline assets bind through XML.

    Store the returned mapping as ``physics.scene_asset_hashes``. Generated
    collision meshes must be bound themselves, not only their source DAE files.
    This reads local assets only and rejects unexpanded external MJCF includes.
    """
    root = ET.fromstring(xml)
    if root.findall(".//include") or root.findall(".//attach") or root.findall("./asset/model"):
        raise ValueError("Scene includes must be expanded before asset binding (including attached models)")
    if len(root.findall("compiler")) > 1:
        raise ValueError("Multiple compiler elements require explicit asset resolution")
    base_dir = Path(base_dir).resolve()
    compiler = root.find("compiler")
    options = {} if compiler is None else compiler.attrib
    result = {"meshes": {}, "textures": {}, "hfields": {}}
    for tag, category in (("mesh", "meshes"), ("texture", "textures"), ("hfield", "hfields")):
        for node in root.findall(f"./asset/{tag}"):
            directory = options.get("texturedir" if tag == "texture" else "meshdir",
                                    options.get("assetdir", ""))
            for key, filename in node.attrib.items():
                if key != "file" and not (tag == "texture" and key in
                        {"fileleft", "fileright", "filefront", "fileback", "fileup", "filedown"}):
                    continue
                if options.get("strippath", "false").lower() == "true":
                    filename = Path(filename).name
                path = Path(filename)
                if not path.is_absolute():
                    path = base_dir / directory / path
                path = path.resolve()
                if not path.is_file():
                    raise ValueError(f"Referenced scene asset is missing: {path}")
                result[category][str(path)] = sha256(path)
    return result


def _verify_assets(scene, physics):
    actual = bind_scene_assets(scene.read_text(), scene.parent)
    expected = physics.get("scene_asset_hashes")
    if expected is None:
        if any(actual.values()):
            raise ValueError("Unbound scene assets: physics.scene_asset_hashes is required for external files")
        expected = {key: {} for key in actual}
    if set(expected) != set(actual):
        raise ValueError("Scene asset binding categories differ")
    for category, files in actual.items():
        if set(files) != set(expected[category]):
            raise ValueError(f"Referenced scene {category} differ from bound manifest")
        for path, digest in files.items():
            if digest != expected[category][path]:
                raise ValueError(f"Scene asset checksum changed: {path}")
    return actual


def _object_subtrees(model, objects):
    if not isinstance(objects, dict) or not objects:
        raise ValueError("Audit requires a nonempty object manifest")
    roots, free_joints = set(), set()
    for spec in objects.values():
        body = model.body(spec["body"]).id
        joint = model.joint(spec["joint"]).id
        if body == 0 or body in roots or joint in free_joints:
            raise ValueError("Object body and joint mappings must be distinct")
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_bodyid[joint] != body:
            raise ValueError("Moving objects must retain free joints on their declared bodies")
        roots.add(body)
        free_joints.add(joint)
    bodies = set(roots)
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor:
            if ancestor in roots:
                bodies.add(body)
                break
            ancestor = int(model.body_parentid[ancestor])
    joints = {joint for joint in range(model.njnt) if int(model.jnt_bodyid[joint]) in bodies}
    return bodies, joints, free_joints


def _verify_no_object_assistance(model, objects):
    bodies, joints, free_joints = _object_subtrees(model, objects)
    if any(model.body_gravcomp[body] != 0 or model.body_mocapid[body] != -1 for body in bodies):
        raise ValueError("Object or descendant gravity compensation or mocap is forbidden")
    joint_transmissions = {int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)}
    for actuator in range(model.nu):
        if int(model.actuator_trntype[actuator]) not in joint_transmissions:
            raise ValueError("Only whitelisted robot-joint actuators are supported; site/body/tendon actuators are forbidden")
        joint = int(model.actuator_trnid[actuator, 0])
        if joint in joints or model.joint(joint).name not in ROBOT_ACTUATED_JOINTS:
            raise ValueError("Object, descendant or nonrobot joint actuator is forbidden")
    # A passive spring tendon can assist an object even without an actuator or
    # equality, so audit its full wrap path as well.
    for tendon in range(model.ntendon):
        start = int(model.tendon_adr[tendon])
        for wrap in range(start, start + int(model.tendon_num[tendon])):
            kind, index = int(model.wrap_type[wrap]), int(model.wrap_objid[wrap])
            touches = kind == int(mujoco.mjtWrap.mjWRAP_JOINT) and index in joints
            if kind == int(mujoco.mjtWrap.mjWRAP_SITE):
                touches |= int(model.site_bodyid[index]) in bodies
            if kind in (int(mujoco.mjtWrap.mjWRAP_SPHERE), int(mujoco.mjtWrap.mjWRAP_CYLINDER)):
                touches |= int(model.geom_bodyid[index]) in bodies
                side = int(model.wrap_prm[wrap])
                if side >= 0:
                    touches |= int(model.site_bodyid[side]) in bodies
            if touches:
                raise ValueError("Object or descendant tendon constraints are forbidden")
    for equality in range(model.neq):
        kind = int(model.eq_type[equality])
        ids = {int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])} - {-1}
        if kind == int(mujoco.mjtEq.mjEQ_JOINT):
            forbidden = bool(ids & joints)
        elif kind in (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT)):
            objtype = int(model.eq_objtype[equality]) if hasattr(model, "eq_objtype") else int(mujoco.mjtObj.mjOBJ_BODY)
            if objtype == int(mujoco.mjtObj.mjOBJ_SITE):
                ids = {int(model.site_bodyid[index]) for index in ids}
            elif objtype != int(mujoco.mjtObj.mjOBJ_BODY):
                raise ValueError("Unsupported equality constraints require an explicit audit")
            forbidden = bool(ids & bodies)
        else:
            # Reachy uses joint mimic equalities only; fail closed for tendon,
            # flex and future constraint representations.
            raise ValueError("Unsupported equality constraints require an explicit audit")
        if forbidden:
            raise ValueError("Object or descendant constraints are forbidden")
    return free_joints


def _load_recording(handle, model, report, objects):
    if model.na:
        raise ValueError("Stateful actuator activation is not covered by this recording schema")
    if "time_s" not in handle:
        raise ValueError("Missing recorded dataset: time_s")
    times = handle["time_s"][()]
    if times.ndim != 1 or not len(times):
        raise ValueError("Recorded timeline must be a nonempty one-dimensional array")
    if not np.isfinite(times).all():
        raise ValueError("Nonfinite recorded timeline")
    if times[0] <= 0 or not np.all(np.diff(times) > 0):
        raise ValueError("Recorded times must be positive and strictly increasing")
    n = len(times)
    shapes = {"initial/qpos": (model.nq,), "initial/qvel": (model.nv,), "initial/ctrl": (model.nu,),
              "simulation/qpos": (n, model.nq), "simulation/qvel": (n, model.nv),
              "simulation/actuator_control": (n, model.nu),
              **{f"objects/{oid}/pose": (n, 7) for oid in objects}}
    arrays = {"time_s": times}
    for key, shape in shapes.items():
        if key not in handle:
            raise ValueError(f"Missing recorded dataset: {key}")
        values = handle[key][()]
        if values.shape != shape:
            raise ValueError(f"Recorded dataset shape mismatch for {key}: {values.shape} != {shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite recorded control or state: {key}")
        arrays[key] = values
    if set(handle.get("objects", {})) != set(objects):
        raise ValueError("Recorded object set differs from scene manifest")
    optional_shapes = {
        "simulation/hand_pose_world": (n, 2, 7), "target/hand_pose_world": (n, 2, 7),
        "target/right_hand_pose_world": (n, 7), "command/gripper_position": (n,),
        "command/joint_reference": (n, 3 + len(ARMS)),
        "metrics/bilateral_contact": (n,), "metrics/hand_object_penetration_m": (n,),
        "metrics/task_satisfied": (n,), "metrics/grasp_drift_m_rad": (n, 2),
    }
    def check_frame_length(name, value):
        if not isinstance(value, h5py.Dataset) or name.startswith("simulation/model_"):
            return
        if name.split("/", 1)[0] not in {"simulation", "objects", "target", "command", "metrics"}:
            return
        if value.ndim < 1 or value.shape[0] != n:
            raise ValueError(f"Recorded frame count mismatch for {name}")
        if name in optional_shapes and value.shape != optional_shapes[name]:
            raise ValueError(f"Recorded dataset shape mismatch for {name}")
    handle.visititems(check_frame_length)
    # The producer records each fixed control interval from reset time zero.
    # Binding the whole grid rejects deleted interior frames, not just a short
    # final frame count. Preserve NaN drift before grasp; it is not state input.
    hz = report["physics"].get("control_hz")
    if hz is not None and (not np.isfinite(hz) or hz <= 0):
        raise ValueError("Invalid recorded control frequency")
    interval = 1 / hz if hz is not None else float(times[0])
    if not np.allclose(times, np.arange(1, n + 1) * interval, atol=1e-8, rtol=0):
        raise ValueError("Recorded timeline has incomplete control intervals")
    simulated = report.get("simulated_s")
    if simulated is not None:
        if (not np.isscalar(simulated) or not np.isfinite(simulated) or simulated <= 0
                or not np.isclose(times[-1], simulated, atol=1e-8, rtol=0)
                or n != round(simulated / interval)):
            raise ValueError("Recorded frame count/timeline differs from reported simulated_s")
    if "recorded_frames" in report and report["recorded_frames"] != n:
        raise ValueError("Recorded frame count differs from report")
    if not np.isfinite(model.opt.timestep) or model.opt.timestep <= 0:
        raise ValueError("Invalid physics timestep")
    steps = np.rint(np.diff(np.r_[0., times]) / model.opt.timestep).astype(np.int64)
    if (np.any(steps < 1) or not np.allclose(steps * model.opt.timestep,
                                           np.diff(np.r_[0., times]), atol=1e-9, rtol=0)):
        raise ValueError("Recorded times are not physics-step aligned")
    return arrays, steps


def verify(attempt):
    attempt = Path(attempt)
    report = json.loads((attempt / "result.json").read_text())
    scene = attempt / "scene.xml"
    if sha256(scene) != report["scene_sha256"]:
        raise ValueError("Scene checksum changed")
    asset_hashes = _verify_assets(scene, report["physics"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    objects = report["physics"]["objects"]
    object_joints = _verify_no_object_assistance(model, objects)
    with h5py.File(attempt / "replay.h5", "r") as handle:
        arrays, steps = _load_recording(handle, model, report, objects)
    acquisition = _load_acquisition(attempt, report, len(steps))
    data = mujoco.MjData(model)
    # Reset once; only recorded robot actuator controls are applied afterward.
    data.qpos[:] = arrays["initial/qpos"]
    data.qvel[:] = arrays["initial/qvel"]
    data.ctrl[:] = arrays["initial/ctrl"]
    for joint in object_joints:
        q = model.jnt_qposadr[joint]
        if not np.allclose(data.qpos[q:q + 7], model.qpos0[q:q + 7], atol=1e-10, rtol=0):
            raise ValueError("Initial object state differs from saved source scene")
    mujoco.mj_forward(model, data)
    qpos_error = qvel_error = object_error = 0.
    for index, count in enumerate(steps):
        data.ctrl[:] = arrays["simulation/actuator_control"][index]
        for _ in range(int(count)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Nonfinite replayed dynamics")
        qpos_error = max(qpos_error, float(np.max(np.abs(data.qpos - arrays["simulation/qpos"][index]))))
        qvel_error = max(qvel_error, float(np.max(np.abs(data.qvel - arrays["simulation/qvel"][index]))))
        for oid, spec in objects.items():
            bid = model.body(spec["body"]).id
            measured = np.r_[data.xpos[bid], data.xquat[bid]]
            object_error = max(object_error, float(np.max(np.abs(measured - arrays[f"objects/{oid}/pose"][index]))))
        if acquisition is not None and acquisition['phases'][index] in ('settle', 'close'):
            from .acquisition_policy import adapter
            actual_guard = adapter(acquisition['plan']['metadata']).guard(model, data, acquisition['plan'])
            if actual_guard != acquisition['guards'][index]:
                raise ValueError('Independent measured acquisition guard differs from its recording')
    result = {"attempt": str(attempt), "scene_sha256": sha256(scene), "frames": len(steps),
              "scene_asset_hashes": asset_hashes, "final_recorded_time_s": float(arrays["time_s"][-1]),
              "reported_duration_checked": "simulated_s" in report,
              "max_qpos_error": qpos_error, "max_qvel_error": qvel_error, "max_object_pose_error": object_error,
              "actuator_replay_pass": bool(max(qpos_error, qvel_error, object_error) < 1e-7),
              "object_welds_actuators": False, "object_state_assignments": "initial reset only",
              "meaning": "Control replay equivalence; task and contact success remain separate recorded gates"}
    if acquisition is not None:
        result['acquisition_guard_audit'] = dict(passed=True, source_coverage_complete=acquisition['complete'],
            measured_guards_recomputed=True, source_rows=acquisition['rows'],
            scope='Every executed original reference index/intent/clock verified, pauses replayed from measured guards, and guards independently recomputed in actuator-only physics')
    policy=report.get('plan',{}).get('robot_supported_placement')
    states_path=attempt/'state-attempt.hdf5'
    if policy and states_path.exists():
        with h5py.File(states_path) as handle:
            states=[json.loads(value) for value in handle['controller_state_json'].asstr()[()]]
        first=states[0]['memory'].get('supported_placement') if states else None
        if first and first.get('support_guard_version')==1:
            from .supported_placement import audit_opening_guard
            with h5py.File(attempt/'replay.h5') as handle:
                bilateral=handle['metrics/bilateral_contact'][()]
            oid=report['plan']['object_id']
            result['placement_opening_guard_audit']=audit_opening_guard(states,arrays[f'objects/{oid}/pose'],bilateral,policy,
                reference_indices=None if acquisition is None else acquisition['indices'])
    result['independent_validation_pass']=bool(result['actuator_replay_pass'] and
        result.get('placement_opening_guard_audit',{}).get('passed',True))
    json_write(attempt / "actuator-replay-audit.json", result)
    return result


def _load_acquisition(attempt, report, frames):
    """Bind a variable command clock to its entire immutable prepared plan."""
    execution = report.get('acquisition_execution')
    if execution is None:
        return None
    from .acquisition_clock import AcquisitionClock, complete_reference_coverage, dilation_kwargs
    from .acquisition_policy import from_details
    _, metadata = from_details(report['plan'])
    if not np.isclose(metadata['timestep_s'], .01, atol=1e-12, rtol=0):
        raise ValueError('Acquisition guard clock must match actual 100 Hz control intervals')
    artifact = Path(metadata['artifact'])
    if sha256(artifact) != metadata['artifact_sha256']:
        raise ValueError('Acquisition plan artifact checksum differs')
    with np.load(artifact, allow_pickle=False) as saved:
        plan_arrays = {key: saved[key].copy() for key in saved.files}
    with h5py.File(attempt/'plan.h5') as handle:
        times = handle['time_s'][()]
        intent = handle['original_gripper_intent'][()]
        executed_intent = handle['gripper'][()]
        from .episodes import matrices_to_pose
        expected = {'time_s': plan_arrays['original_time_s'],
                    'original_gripper_intent': plan_arrays['original_gripper_intent'],
                    'reference': plan_arrays['original_reference'],
                    'hand_goals': matrices_to_pose(plan_arrays['original_hand_goals']),
                    'object_goals': matrices_to_pose(plan_arrays['original_object_goals'])}
        if any(not np.array_equal(handle[key][()], value) for key, value in expected.items()):
            raise ValueError('Complete prepared source plan differs from bound acquisition originals')
        if executed_intent.shape != times.shape or not np.isfinite(executed_intent).all():
            raise ValueError('Executed source-indexed gripper intent is incomplete')
    with h5py.File(attempt/'replay.h5') as handle:
        indices = handle['source/reference_index'][()]
        source_times = handle['source/reference_time_s'][()]
        original_intent = handle['source/original_gripper_intent'][()]
        effective_intent = handle['command/effective_gripper_intent'][()]
        phases = handle['acquisition/phase'].asstr()[()].tolist()
        guards = [json.loads(value) for value in handle['acquisition/guard_json'].asstr()[()]]
        bilateral = handle['metrics/bilateral_contact'][()]
    if (indices.shape != (frames,) or indices.dtype.kind not in 'iu' or not frames
            or indices[0] != 0 or np.any(indices < 0) or np.any(indices >= len(times))
            or np.any((np.diff(indices) < 0) | (np.diff(indices) > 1))):
        raise ValueError('Acquisition source index recording skips or reorders original rows')
    if (source_times.shape != indices.shape or original_intent.shape != indices.shape
            or effective_intent.shape != indices.shape or len(phases) != frames or len(guards) != frames
            or not np.array_equal(source_times, times[indices])
            or not np.array_equal(original_intent, intent[indices])):
        raise ValueError('Acquisition source clock or original intent differs from prepared plan')
    clock = AcquisitionClock(source_rows=len(times), first_close_index=int(metadata['first_close_index']),
        acquisition_index=int(metadata['acquisition_index']), entry_rows=len(plan_arrays['entry_reference']),
        exit_source_indices=plan_arrays['exit_source_indices'].tolist(), timestep=metadata['timestep_s'],
        wait_limit_s=metadata['timeout_s'], stable_s=metadata['stable_s'],
        **dilation_kwargs(metadata))
    iterator = clock.frames()
    for i in range(frames):
        try:
            frame = next(iterator)
        except StopIteration as exc:
            raise ValueError('Recorded acquisition continues past its guarded command clock') from exc
        if frame.reference_index != indices[i] or frame.phase != phases[i]:
            raise ValueError('Recorded acquisition phase/index bypasses a measured guard')
        required_intent = (2. if frame.phase in ('delayed_open', 'entry', 'settle')
                           else executed_intent[frame.reference_index])
        if effective_intent[i] != required_intent:
            raise ValueError('Effective gripper intent differs from the source-indexed acquisition policy')
        if frame.phase == 'close' and effective_intent[i] >= 0:
            raise ValueError('Measured acquisition close phase has no close intent')
        if frame.phase in ('settle', 'close'):
            clock.observe(alignment=guards[i]['alignment'],
                bilateral_force=bool(guards[i]['bilateral_force'] and bilateral[i] == 1.))
    try:
        next(iterator)
    except StopIteration:
        pass
    complete = complete_reference_coverage(indices.tolist(), len(times)) and clock.complete
    if bool(report['gates']['rollout_complete']) != complete:
        raise ValueError('Acquisition source coverage disagrees with rollout_complete')
    if bool(report['gates']['measured_acquisition_before_carry']) != (clock.wait.phase == 'carry'):
        raise ValueError('Reported acquisition gate bypasses measured guards')
    if execution['original_reference_rows'] != len(times):
        raise ValueError('Acquisition original source denominator differs')
    return dict(plan={'metadata': metadata, 'arrays': plan_arrays}, phases=phases, guards=guards, indices=indices,
                complete=complete, rows=len(times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    print(json.dumps(verify(parser.parse_args().attempt), indent=2))
