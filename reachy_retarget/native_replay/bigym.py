"""Explicit offline BiGym MovePlate action replay; source H1 state, never Reachy.

The simulator is imported only by replay(). Dependencies/assets must already be
acquired. No DemoStore, network downloader, renderer, or robot SDK is used.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from ..adapters.native_mobile import _provenance, _require_recorded, _safetensors


SOURCE_REVISION = "52070fa73e8dd88a3d76eed4be575a6d01497931"
SOURCE_URL = "https://github.com/NeuracoreAI/bigym/tree/" + SOURCE_REVISION
MUJOCO_VERSION = "3.1.5"
CONTROL_FREQUENCY = 500
TASK_OBJECTS = ("plate", "rack_start", "rack_target", "table")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pilot(root):
    """Verify acquisition provenance before decoding this bounded native pilot."""
    folder, manifest, records = _provenance(root, "bigym")
    paths = sorted(folder.rglob("*.safetensors"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one acquired native BiGym pilot")
    _require_recorded(paths[0], records)
    native, tensors = _safetensors(paths[0])
    config = native.get("environment_data", {})
    expected = {"env_name": "MovePlate", "action_mode_name": "JointPositionActionMode",
                "floating_base": True, "action_mode_absolute": True,
                "floating_dofs": ["pelvis_x", "pelvis_y", "pelvis_rz"]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Unverified source task/action configuration")
    if config.get("observation_config", {}).get("cameras") != []:
        raise ValueError("Native numeric replay requires the recorded no-camera configuration")
    if type(native.get("seed")) is not int or not 0 <= native["seed"] < 2**32:
        raise ValueError("Missing or invalid recorded reset seed")
    actions = tensors.get("info_demo_action")
    if actions is None or actions.ndim != 2 or actions.shape[1] != 15 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Expected finite native 15-component actions")
    if native.get("package_versions", {}).get("mujoco") != MUJOCO_VERSION:
        raise ValueError("This replay recipe was verified only for recorded MuJoCo 3.1.5")
    return native, tensors, records, manifest


def _source_provenance(source_root):
    source_root = Path(source_root).resolve()
    manifest = source_root.parent / "acquisition.json"
    records = json.loads(manifest.read_text())
    source = next((row for row in records if row.get("name") == "bigym"), None)
    if not source or source.get("revision") != SOURCE_REVISION:
        raise ValueError("Pinned historical BiGym source acquisition is required")
    archive = source_root.parent / "bigym.tar.gz"
    if not archive.is_file() or _sha(archive) != source["sha256"]:
        raise ValueError("Historical source archive checksum mismatch")
    files = []
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files.append({"path": str(path.relative_to(source_root)), "sha256": _sha(path), "bytes": path.stat().st_size})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"acquisition": records, "source_tree_sha256": digest, "source_files": files}


def _create_environment(native, source_root):
    """Mirror official Metadata.get_env + DemoPlayer reset without a downloader."""
    sys.path.insert(0, str(Path(source_root).resolve()))
    import mujoco
    import bigym
    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from bigym.envs.move_plates import MovePlate
    from bigym.utils.observation_config import ObservationConfig

    if not Path(bigym.__file__).resolve().is_relative_to(Path(source_root).resolve()):
        raise ValueError("A different BiGym package was already imported")
    if mujoco.__version__ != MUJOCO_VERSION:
        raise ValueError("Use isolated MuJoCo 3.1.5; shared/newer runtime is not this recipe")
    config = native["environment_data"]
    env = MovePlate(action_mode=JointPositionActionMode(absolute=True, floating_base=True,
        floating_dofs=[PelvisDof(name) for name in config["floating_dofs"]]),
        observation_config=ObservationConfig(**config["observation_config"]),
        render_mode=None, control_frequency=CONTROL_FREQUENCY, start_seed=native["seed"])
    env.reset(seed=native["seed"])
    return env


def _name(element):
    return element.mjcf.full_identifier


def action_contract(env):
    """Read exact compiled order; absolute mode does not make base increments absolute."""
    columns = []
    for actuator in env.robot.floating_base.all_actuators:
        joint = actuator.joint
        columns.append({"index": len(columns), "name": joint.full_identifier,
                        "kind": "base_position_target_increment", "reference": "previous actuator target",
                        "unit": "rad" if joint.type == "hinge" else "m"})
    for actuator in env.robot.limb_actuators:
        columns.append({"index": len(columns), "name": actuator.joint.full_identifier,
                        "kind": "absolute_joint_position_target", "unit": "rad"})
    for side, gripper in env.robot.grippers.items():
        columns.append({"index": len(columns), "name": side.name.lower() + "_gripper",
                        "kind": "normalized_gripper_command", "range": np.asarray(gripper.range).tolist(),
                        "actuators": [actuator.full_identifier for actuator in gripper.actuators]})
    if len(columns) != 15 or tuple(env.action_space.shape) != (15,):
        raise ValueError("Compiled action schema differs from recorded 15-component pilot")
    return columns


def _selection(env):
    """Select native task objects and genuine H1 references, excluding other scene props."""
    import mujoco
    model = env.mojo.model

    def identifier(kind, name):
        index = mujoco.mj_name2id(model, kind, name)
        if index < 0:
            raise ValueError("Missing compiled source reference: " + name)
        return index

    roots = {"plate": _name(env.plates[0].body), "rack_start": _name(env.rack_start.body),
             "rack_target": _name(env.rack_target.body), "table": _name(env.table.body)}
    objects = {key: identifier(mujoco.mjtObj.mjOBJ_BODY, name) for key, name in roots.items()}
    base_name = _name(env.robot.pelvis)
    base = identifier(mujoco.mjtObj.mjOBJ_BODY, base_name)
    robot = {base}
    for i in range(base + 1, model.nbody):
        if int(model.body_parentid[i]) in robot:
            robot.add(i)
    joints = [i for i in range(model.njnt) if int(model.jnt_bodyid[i]) in robot and int(model.jnt_type[i]) in (2, 3)]
    sites = {side.name.lower(): identifier(mujoco.mjtObj.mjOBJ_SITE, _name(gripper.wrist_site))
             for side, gripper in env.robot.grippers.items()}
    pinch_sites = {side.name.lower(): identifier(mujoco.mjtObj.mjOBJ_SITE, _name(gripper._pinch_site))
                   for side, gripper in env.robot.grippers.items() if gripper._pinch_site is not None}
    # The official wrist site is an EEF reference, not an inferred pad/contact frame.
    if set(sites) != {"left", "right"}:
        raise ValueError("Both genuine source wrist sites are required")
    cameras = {camera.mjcf.name: identifier(mujoco.mjtObj.mjOBJ_CAMERA, _name(camera)) for camera in env.robot.cameras}
    return {"objects": objects, "object_names": roots, "base": base, "base_name": base_name,
            "robot_body_ids": sorted(robot), "joint_ids": joints, "sites": sites,
            "pinch_sites": pinch_sites, "cameras": cameras}


def _pose(position, matrix):
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(np.asarray(matrix).reshape(3, 3)).as_quat()[[3, 0, 1, 2]]
    return np.r_[position, quat]


def capture_state(model, data, selection):
    """Capture actual numeric caches; no object-state assignment or physics step."""
    import mujoco
    # mj_step can leave position caches at the pre-integration state. Refresh
    # kinematics only; do not rerun the solver or alter integration warm-start.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    mujoco.mj_camlight(model, data)
    joints = selection["joint_ids"]
    qadr = np.asarray([model.jnt_qposadr[i] for i in joints], dtype=int)
    vadr = np.asarray([model.jnt_dofadr[i] for i in joints], dtype=int)
    base = selection["base"]
    base_twist = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, base, base_twist, 0)
    arrays = {"timestamp": np.asarray(data.time), "source/joint_position": data.qpos[qadr].copy(),
              "source/joint_velocity": data.qvel[vadr].copy(), "source/qpos": data.qpos.copy(),
              "source/qvel": data.qvel.copy(), "source/actuator_control": data.ctrl.copy(),
              "source/actuator_force": data.actuator_force.copy(),
              "source/base_pose": _pose(data.xpos[base], data.xmat[base]),
              "source/base_twist_world": base_twist,
              "source/robot_body_poses": np.stack([_pose(data.xpos[i], data.xmat[i]) for i in selection["robot_body_ids"]])}
    for label, index in selection["objects"].items():
        arrays[f"objects/{label}/pose"] = _pose(data.xpos[index], data.xmat[index])
        twist = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, index, twist, 0)
        arrays[f"objects/{label}/twist_world"] = twist
    for side, index in selection["sites"].items():
        arrays[f"source/{side}_eef_pose"] = _pose(data.site_xpos[index], data.site_xmat[index])
        twist = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, index, twist, 0)
        arrays[f"source/{side}_eef_twist_world"] = twist
    for side, index in selection.get("pinch_sites", {}).items():
        arrays[f"source/{side}_pinch_pose"] = _pose(data.site_xpos[index], data.site_xmat[index])
    for label, index in selection["cameras"].items():
        arrays[f"source/cameras/{label}/pose"] = _pose(data.cam_xpos[index], data.cam_xmat[index].reshape(3, 3) @ np.diag([1., -1., -1.]))
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    arrays["source/integration_state"] = state
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite native replay state")
    return arrays


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def replay(pilot_root, output_dir, source_root, max_steps=None):
    """Save one immutable replay attempt, including partial states on failure.

    Recorded actions are consumed once in order at the official native 500 Hz.
    Frames are post-action; the reset state is saved separately, not fabricated
    as a recorded action row. Source success does not validate Reachy dynamics.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    arrays, frames, env = {}, [], None
    metadata = {"source_id": "bigym", "source_urls": [SOURCE_URL], "source_revision": SOURCE_REVISION,
                "physics_validated": False, "reachy_physics_validated": False, "retargeted": False,
                "rgb_stored": False, "simulation_assumptions": [], "derived_fields": {},
                "source_replay_completed": False, "source_task_success": False,
                "replay_implementation_sha256": _sha(__file__), "replay_python_version": sys.version}
    missing = ["Reachy retargeting and physical validation", "exact recorded BiGym 4.0.0 source revision",
               "source articulated neck state/control; H1 has no articulated neck in this model"]
    error = None
    try:
        native, tensors, records, manifest = load_pilot(pilot_root)
        provenance = _source_provenance(source_root)
        _json(output / "source-provenance.json", provenance)
        actions = tensors["info_demo_action"]
        if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
            raise ValueError("max_steps must be a positive integer")
        limit = min(len(actions), max_steps) if max_steps else len(actions)
        metadata.update(source_sequence="MovePlate/" + native["uuid"], source_group="bigym/MovePlate/" + native["uuid"],
            source_urls=[record["source_url"] for record in records] + [SOURCE_URL], source_files=records,
            source_configuration=native, recorded_package_versions=native["package_versions"],
            source_tree_sha256=provenance["source_tree_sha256"], source_actions_total=len(actions),
            requested_steps=limit, acquisition_code_revision=manifest["code_revision"])
        env = _create_environment(native, source_root)
        import mujoco
        import bigym
        model, data = env.mojo.model, env.mojo.data
        if not np.isclose(model.opt.timestep, 1 / CONTROL_FREQUENCY, atol=1e-12) or env.control_frequency != CONTROL_FREQUENCY:
            raise ValueError("Native timestep differs from official maximum-frequency replay")
        selection = _selection(env)
        contract = action_contract(env)
        names = [model.joint(i).name for i in selection["joint_ids"]]
        groups = {side + "_arm": [i for i, name in enumerate(names) if any(
            row["name"] == name and row["kind"] == "absolute_joint_position_target" and side + "_" in name for row in contract)] for side in ("left", "right")}
        groups.update(base=[i for i, name in enumerate(names) if any(row["name"] == name and row["kind"] == "base_position_target_increment" for row in contract)], neck=[])
        for side, gripper in env.robot.grippers.items():
            prefix = gripper.body.mjcf.full_identifier
            groups[side.name.lower() + "_gripper"] = [i for i, name in enumerate(names) if name.startswith(prefix)]
        details = [{"name": model.joint(i).name, "model_joint_id": i,
                    "qpos_address": int(model.jnt_qposadr[i]), "dof_address": int(model.jnt_dofadr[i]),
                    "type": "hinge" if int(model.jnt_type[i]) == 3 else "slide",
                    "units": {"position": "rad" if int(model.jnt_type[i]) == 3 else "m",
                              "velocity": "rad/s" if int(model.jnt_type[i]) == 3 else "m/s"}}
                   for i in selection["joint_ids"]]
        metadata.update(source_runtime_versions={name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "dm-control", "gymnasium", "mojo")},
            replay_bigym_version=bigym.__version__, action_channels=contract, joint_names=names,
            source_joint_names=names, joint_groups=groups, joint_details=details,
            base_body=selection["base_name"], body_names=[model.body(i).name for i in selection["robot_body_ids"]],
            eef_sites={side: model.site(i).name for side, i in selection["sites"].items()},
            pinch_sites={side: model.site(i).name for side, i in selection["pinch_sites"].items()},
            objects={label: {"native_body_name": name, "pose_frame": "source_world", "quaternion_order": "wxyz"} for label, name in selection["object_names"].items()},
            task_object_pose_coverage={"required_task_objects": list(TASK_OBJECTS), "saved_task_objects": list(TASK_OBJECTS),
                "all_task_object_poses_saved": True, "unrelated_scene_objects_required": False},
            timing={"source": "actual mjData.time after each executed action", "native_control_frequency_hz": CONTROL_FREQUENCY,
                    "frequency_evidence": SOURCE_URL + "/demonstrations/demo_player.py", "observation_timing": "post_action",
                    "original_action_start_time_channel": "source/action_start_time_s", "resampling_performed": False},
            action_semantics="3 base target increments + 10 absolute source arm targets + 2 normalized gripper commands",
            frames={"poses": "source world", "quaternion_order": "wxyz", "twist_order": "wx,wy,wz,vx,vy,vz"},
            model_dimensions={"nq": model.nq, "nv": model.nv, "nu": model.nu},
            reset_positions_semantics="Original header retained; official replay resets with robot config + recorded seed, not this zero placeholder",
            simulation_assumptions=["Official historical source reports BiGym 4.1.0; exact recorded 4.0.0 code is not public in available history",
                "Native floating H1 uses source position-actuated pelvis and animated legs; this is not mobile-base locomotion validation",
                "Source object state is assigned only by the official reset; subsequent motion comes from unchanged native action stepping",
                "Wrist EEF sites are source references, not calibrated Reachy pad contact frames",
                "Head camera optical pose is available; articulated neck pose/commands are absent"],
            derived_fields={"timestamp": "reconstructed by native MuJoCo replay, not present in original lightweight recording",
                "source/*state and objects/*": "simulated from recorded source actions and reset seed; original observations unavailable for comparison"})
        initial = capture_state(model, data, selection)
        np.savez_compressed(output / "reset-state.npz", **initial)
        mujoco.mj_saveModel(model, str(output / "source-model.mjb"), None)
        # No rendering is requested. Preserve compiled numeric MJCF as model evidence.
        (output / "source-model.xml").write_text(env.mojo.root_element.mjcf.to_xml_string())
        for i in range(limit):
            start = float(data.time)
            env.step(actions[i].copy(), fast=True)
            row = capture_state(model, data, selection)
            if not np.isclose(float(row["timestamp"]) - start, 1 / CONTROL_FREQUENCY, atol=1e-10):
                raise ValueError("Native clock reset/divergence during replay")
            # Official validate_in_env reads reward/success after fast stepping.
            # Do not additionally call historical env.fail: upstream later fixed
            # that accessor because its floating-base position read diverged replay.
            row.update({"source/action": actions[i].copy(), "source/frame_index": np.asarray(i, dtype=np.int64),
                        "source/action_start_time_s": np.asarray(start), "source/replay_success": np.asarray(env.success),
                        "source/replay_reward": np.asarray(env.reward)})
            for key in ("termination", "truncation"):
                if key in tensors and tensors[key].shape == (len(actions),):
                    row["source/recorded_" + key] = tensors[key][i].copy()
            frames.append(row)
        metadata["source_replay_completed"] = len(frames) == len(actions)
        metadata["source_task_success"] = any(bool(row["source/replay_success"]) for row in frames)
        if limit != len(actions):
            missing.append("remaining source actions not replayed in this bounded attempt")
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        missing.append("source replay failed: " + type(exc).__name__ + ": " + str(exc))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                metadata["close_error"] = {"type": type(exc).__name__, "message": str(exc)}
                if error is None:
                    error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                    missing.append("native environment close failed; sampled prefix preserved")
    if frames:
        arrays = {key: np.stack([row[key] for row in frames]) for key in frames[0]}
    status = "failed_source_replay" if error else "source_replayed_partial" if not metadata["source_replay_completed"] else "source_replayed"
    metadata.update(source_actions_replayed=len(frames), source_error=error, missing_fields=missing,
                    status=status, source_task_success_reproduced=bool(metadata["source_task_success"]))
    np.savez_compressed(output / "states.npz", **arrays)
    metadata["output_sha256"] = {path.name: _sha(path) for path in output.iterdir() if path.is_file()}
    _json(output / "metadata.json", metadata)
    return {"arrays": arrays, "metadata": metadata, "status": status, "missing_fields": missing}


def normalize(output_dir):
    """Reload saved numeric replay outputs, validating all attempt checksums."""
    output = Path(output_dir)
    metadata = json.loads((output / "metadata.json").read_text())
    for name, checksum in metadata["output_sha256"].items():
        if Path(name).name != name or _sha(output / name) != checksum:
            raise ValueError("Replay artifact checksum mismatch")
    with np.load(output / "states.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return {"arrays": arrays, "metadata": metadata, "status": metadata["status"], "missing_fields": metadata["missing_fields"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    result = replay(args.pilot_root, args.output_dir, args.source_root, args.max_steps)
    print(json.dumps({"status": result["status"], "frames": result["metadata"]["source_actions_replayed"],
                      "source_success": result["metadata"]["source_task_success"], "error": result["metadata"]["source_error"]}, indent=2))
    return 2 if result["status"] == "failed_source_replay" else 0


if __name__ == "__main__":
    raise SystemExit(main())
