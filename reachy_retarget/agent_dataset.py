"""Common offline archives and an evidence-gated bridge to reachy-agent v5.

No simulator, hardware SDK, network client, or sibling package is imported.
An archive is never relabelled as a successful reachy-agent episode. Actual v5
recordings can be imported without inventing images or controller history.
"""

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import shutil

import h5py
import numpy as np

from .episodes import read_episode
from .store import json_write, sha256

SCHEMA = "reachy-retarget-agent-archive-v1"
STATE_CONTRACT = "reachy-retarget-state-observations-v1"
V5_SCHEMA = "reachy-mujoco-episode-v5"
ACTION_CONTRACT = "bimanual-head-observation-torso-camera-rotation6d-binary-grippers-reachy-control-v5"
AGENT_REVISION = "eb39c2e0f133e8955acfb949513512eaae193db1"
ROTATION_NAMES = ("r00", "r10", "r20", "r01", "r11", "r21")
ACTION_NAMES = tuple(f"{s}_{k}" for s in ("left", "right") for k in ("x", "y", "z", *ROTATION_NAMES)) + tuple(
    f"head_{k}" for k in ROTATION_NAMES) + ("left_gripper_open", "right_gripper_open")
STAGES = ("navigate", "approach", "grasp", "lift", "transfer", "release", "retreat", "verify")
OBSERVATIONS = ("joint_position", "joint_velocity", "base_pose", "base_twist", "pickup_pose",
                "receptacle_pose", "left_tcp_pose", "right_tcp_pose", "body_poses")
V5_FIELDS = ("action", "applied_controls", "physics_state", "controller_state_json", "timestamp",
             "stage_index", "state_tick", "image_tick", *(f"observation/{k}" for k in OBSERVATIONS),
             "observation/cameras/torso/pose", "observation/cameras/torso/rgb")
STATE_TABLE_FIELDS = ("timestamp", *(f"observation/{key}" for key in OBSERVATIONS),
                      "observation/head_pose", "observation/cameras/torso/pose")


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe dataset/episode identifier: {value!r}")
    return value


def _channel(value):
    if not isinstance(value, str) or not value or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"Unsafe HDF5 channel: {value!r}")
    return value


@contextmanager
def _lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read_arrays(handle):
    arrays = {}
    def read(name, item):
        if isinstance(item, h5py.Dataset):
            arrays[name] = item.asstr()[()] if item.dtype.kind in "OUS" else item[()]
    handle.visititems(read)
    return arrays


def _required_fields(metadata):
    cameras = metadata.get("cameras", {})
    return tuple(dict.fromkeys((*V5_FIELDS, *(f"observation/cameras/{name}/{field}" for name in cameras for field in ("pose", "rgb")))))


def _assessment(arrays, metadata, attributes):
    """Structural checks only; no claim of controller replay or image correctness."""
    errors = []
    required = _required_fields(metadata)
    missing = sorted(set(required) - set(arrays))
    if metadata.get("schema") != V5_SCHEMA or attributes.get("schema") != V5_SCHEMA:
        errors.append("schema is not reachy-agent v5")
    if metadata.get("action_contract") != ACTION_CONTRACT or attributes.get("action_contract") != ACTION_CONTRACT:
        errors.append("action contract differs from v5")
    provenance = metadata.get("provenance", {})
    controller = provenance.get("controller", {})
    if controller.get("mode") != "reachy-control" or not controller.get("revision"):
        errors.append("recorded reachy-control identity/revision is missing")
    if not re.fullmatch(r"[a-f0-9]{64}", provenance.get("manifest_sha256", "")):
        errors.append("prepared scene manifest SHA256 is missing")
    if tuple(metadata.get("action_names", ())) != ACTION_NAMES:
        errors.append("action columns differ from the 26D contract")
    if metadata.get("fps") != 10 or metadata.get("observation_timing") != "synchronized-pre-action-v1":
        errors.append("requires synchronized pre-action observations at 10 Hz")
    if metadata.get("physics_state_spec") != "mjSTATE_INTEGRATION" or metadata.get("quaternion_order") != "wxyz":
        errors.append("physics state or quaternion convention is missing/different")
    if tuple(metadata.get("stages", ())) != STAGES:
        errors.append("stage names differ from v5")
    cameras = metadata.get("cameras", {})
    if "torso" not in cameras:
        errors.append("calibrated torso camera is missing")
    for name, spec in cameras.items():
        if (np.shape(spec.get("K")) != (3, 3) or spec.get("size") != [256, 256]
                or "D" not in spec or "preprocessing" not in spec):
            errors.append(f"camera calibration incomplete: {name}")
    for key in ("joint_names", "body_names", "actuator_names"):
        names = metadata.get(key, [])
        if not names or len(set(names)) != len(names):
            errors.append(f"missing/duplicate {key}")
    count = len(arrays["timestamp"]) if "timestamp" in arrays and np.ndim(arrays["timestamp"]) else 0
    if count < 2:
        errors.append("at least two rows are required for timing verification")
    timestep = provenance.get("simulation", {}).get("timestep", 0)
    if not isinstance(timestep, (int, float)) or not np.isfinite(timestep) or timestep <= 0:
        errors.append("positive simulation timestep is missing")
        substeps = None
    else:
        substeps = round(.1 / timestep)
        if substeps < 1 or not np.isclose(substeps * timestep, .1, atol=1e-10, rtol=0):
            errors.append("physics timestep does not divide a 10 Hz row")
    shapes = {"action": (26,), "stage_index": (), "state_tick": (), "image_tick": (), "timestamp": (),
              "controller_state_json": (), "observation/base_twist": (3,),
              "observation/joint_position": (len(metadata.get("joint_names", [])),),
              "observation/joint_velocity": (len(metadata.get("joint_names", [])),),
              "observation/body_poses": (len(metadata.get("body_names", [])), 7)}
    shapes.update({f"observation/{k}": (7,) for k in OBSERVATIONS if k.endswith("_pose")})
    for name in cameras:
        shapes[f"observation/cameras/{name}/pose"] = (7,)
        shapes[f"observation/cameras/{name}/rgb"] = (256, 256, 3)
    if substeps:
        shapes["applied_controls"] = (substeps, len(metadata.get("actuator_names", [])))
    for key in required:
        if key not in arrays:
            continue
        value = arrays[key]
        if not np.shape(value) or np.shape(value)[0] != count:
            errors.append(f"row count mismatch: {key}")
        if key in shapes and np.shape(value)[1:] != shapes[key]:
            errors.append(f"shape mismatch: {key}")
        if key.endswith("/rgb"):
            if value.dtype != np.uint8:
                errors.append(f"RGB is not uint8: {key}")
        elif value.dtype.kind in "biuf" and not np.isfinite(value[...]).all():
            errors.append(f"nonfinite values: {key}")
        if key.endswith("_pose") or key.endswith("/pose") or key == "observation/body_poses":
            if np.shape(value)[-1:] == (7,) and not np.allclose(np.linalg.norm(value[..., 3:], axis=-1), 1, atol=1e-5):
                errors.append(f"nonunit pose quaternion: {key}")
    if "physics_state" in arrays and (np.ndim(arrays["physics_state"]) != 2 or np.shape(arrays["physics_state"])[-1] < 1):
        errors.append("physics_state must be a nonempty matrix; model-dependent width needs runtime validation")
    if count >= 2 and "timestamp" in arrays and np.shape(arrays["timestamp"]) == (count,):
        timestamp = arrays["timestamp"][...]
        if not np.allclose(np.diff(timestamp), .1, atol=1e-6, rtol=0):
            errors.append("timestamps are not contiguous 10 Hz")
        if all(k in arrays and np.shape(arrays[k]) == (count,) for k in ("state_tick", "image_tick")):
            state, image = arrays["state_tick"][...], arrays["image_tick"][...]
            if (arrays["state_tick"].dtype.kind not in "iu" or arrays["image_tick"].dtype.kind not in "iu"
                    or not np.array_equal(state, image) or np.any(np.diff(state) <= 0)):
                errors.append("image/state ticks are not synchronized increasing integers")
            if substeps and not np.allclose(state * timestep, timestamp, atol=1e-6, rtol=0):
                errors.append("ticks differ from physics timestamps")
    if "action" in arrays and np.shape(arrays["action"]) == (count, 26):
        action = arrays["action"][...]
        if not np.isin(action[:, 24:], (0, 1)).all():
            errors.append("gripper labels must be binary")
        for offset in (3, 12, 18):
            first, second = action[:, offset:offset+3], action[:, offset+3:offset+6]
            if np.any(np.linalg.norm(first, axis=1) < 1e-6) or np.any(np.linalg.norm(np.cross(first, second), axis=1) < 1e-6):
                errors.append("degenerate rotation6d action")
    if "stage_index" in arrays:
        value = arrays["stage_index"][...]
        if value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value >= len(STAGES)):
            errors.append("invalid stage indices")
    if "controller_state_json" in arrays and np.shape(arrays["controller_state_json"]) == (count,):
        identity = None
        for i, raw in enumerate(arrays["controller_state_json"]):
            try:
                state = json.loads(raw)
                if not set(state) >= {"base", "joints", "targets", "memory", "servo_targets", "identity", "format"}:
                    raise ValueError("missing controller history")
                if (not state["joints"] or set(state["joints"]) != set(state["servo_targets"]["joints"])
                        or not state["targets"] or not state["memory"] or np.shape(state["base"]) != (3,)
                        or not state["identity"] or not state["format"]):
                    raise ValueError("incomplete controller history")
                json.dumps(state, allow_nan=False)
                current = (state["identity"], state["format"])
                if identity is not None and current != identity:
                    raise ValueError("controller identity changes within an episode")
                identity = current
            except (ValueError, TypeError, KeyError) as error:
                errors.append(f"invalid controller snapshot at row {i}: {error}")
                break
    return {"structurally_compatible": not missing and not errors, "missing_fields": missing,
            "errors": errors, "rows": count, "physics_validated": False, "policy_ready": False,
            "remaining_checks": ["Hash-bound task-specific physics/controller/image validation",
                                 "Matching prepared scene and policy map",
                                 "At least two successful source-independent episodes for training"]}


def assess_v5(path, metadata=None):
    """Inspect saved v5 data without importing reachy-agent or a robot SDK."""
    with h5py.File(path, "r") as handle:
        metadata = metadata or json.loads(handle.attrs.get("metadata_json", "{}"))
        names = []
        handle.visititems(lambda n, obj: names.append(n) if isinstance(obj, h5py.Dataset) else None)
        result = _assessment({n: handle[n] for n in names}, metadata, dict(handle.attrs))
        result["recorded_success"] = bool(handle.attrs.get("success", False))
    result["episode_sha256"] = sha256(path)
    return result


def assess_state_observations(source, metadata=None):
    """Check measured state coverage independently of RGB and native v5.

    Accept a stacked StateObserver record, an array mapping plus metadata, or a
    canonical HDF5 path. This checks stored structure/conventions, not dynamics,
    controller execution, success, or whether a policy loader accepts the data.
    Undefined pickup/receptacle roles remain missing and are reported separately
    from coverage of the explicitly declared task objects.
    """
    if isinstance(source, (str, Path)):
        with h5py.File(source, "r") as handle:
            arrays = _read_arrays(handle)
            meta = metadata or json.loads(handle.attrs.get("metadata_json", "{}"))
    elif "arrays" in source:
        arrays, meta = source["arrays"], metadata or source["metadata"]
    else:
        arrays, meta = source, metadata or {}
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    errors = []
    if meta.get("state_contract") != STATE_CONTRACT:
        errors.append("Missing/unknown measured-state contract")
    if meta.get("quaternion_order") != "wxyz":
        errors.append("State poses require xyz+wxyz")
    frames = meta.get("frames", {})
    if frames.get("twist_order") != ["wx", "wy", "wz", "vx", "vy", "vz"]:
        errors.append("Spatial twist ordering is not declared angular-first")
    required = {"timestamp": (), "observation/joint_position": (len(meta.get("joint_names", [])),),
                "observation/joint_velocity": (len(meta.get("joint_names", [])),),
                "observation/base_pose": (7,), "observation/base_twist": (3,),
                "observation/base_twist_world": (6,), "observation/base_twist_local": (6,),
                "observation/body_poses": (len(meta.get("body_names", [])), 7)}
    dimensions = meta.get("model_dimensions", {})
    for quantity, size in (("qpos", "nq"), ("qvel", "nv")):
        if not isinstance(dimensions.get(size), int) or dimensions[size] < 1:
            errors.append(f"Missing model dimension: {size}")
        else:
            required["observation/" + quantity] = (dimensions[size],)
    for part in ("left_tcp", "right_tcp", "head"):
        required[f"observation/{part}_pose"] = (7,)
        required[f"observation/{part}_pose_base"] = (7,)
        for suffix in ("world", "base", "relative_base"):
            required[f"observation/{part}_twist_{suffix}"] = (6,)
    cameras, objects = meta.get("cameras", {}), meta.get("objects", {})
    if not cameras:
        errors.append("No camera pose mappings are declared")
    for name in cameras:
        required[f"observation/cameras/{name}/pose"] = (7,)
    if not objects or meta.get("task_object_coverage_complete") is not True:
        errors.append("Task-object mappings/coverage are incomplete")
    selected_body_ids = set(meta.get("robot_body_ids", []))
    for spec in objects.values():
        selected_body_ids.update(spec.get("body_ids", []))
    if set(meta.get("body_ids", [])) != selected_body_ids or len(meta.get("body_ids", [])) != len(meta.get("body_names", [])):
        errors.append("Body pose inventory differs from robot and declared task-object subtrees")
    if any(row.get("model_body_id") not in selected_body_ids for row in meta.get("joint_details", [])):
        errors.append("Direct joint observations contain joints outside robot/task scope")
    for key in arrays:
        if key.startswith("observation/objects/") and key.split("/")[2] not in objects:
            errors.append(f"Undeclared task object pose/state channel: {key}")
    for name, spec in objects.items():
        if len(spec.get("root_body_ids", [])) == 1:
            required[f"observation/objects/{name}/pose"] = (7,)
        for part in spec.get("part_channels", {}):
            required[f"observation/objects/{name}/parts/{part}/pose"] = (7,)
        for quantity, width in (("qpos", "qpos_width"), ("qvel", "dof_width")):
            required[f"observation/objects/{name}/{quantity}"] = (sum(joint[width] for joint in spec.get("joints", [])),)
    undefined_roles = []
    for role in ("pickup", "receptacle"):
        if role in meta.get("object_roles", {}):
            required[f"observation/{role}_pose"] = (7,)
        else:
            undefined_roles.append(f"observation/{role}_pose")
    if (meta.get("scalar_joint_scope") != "robot_and_task_object_subtrees"
            or meta.get("scalar_joint_coverage_complete") is not True or meta.get("required_joint_coverage_complete") is not True):
        errors.append("Declared robot/task/required joint coverage is incomplete")
    missing_joints = list(meta.get("missing_required_joints", []))
    names = meta.get("joint_names", [])
    if len(names) != len(set(names)) or len(meta.get("joint_details", [])) != len(names):
        errors.append("Joint names/details are not aligned and unique")
    for group, indices in meta.get("joint_groups", {}).items():
        if any(not isinstance(i, int) or i < 0 or i >= len(names) for i in indices):
            errors.append(f"Invalid joint group indices: {group}")
    times = arrays.get("timestamp")
    count = len(times) if times is not None and times.ndim == 1 else 0
    if not count or not np.isfinite(times).all() or (count > 1 and np.any(np.diff(times) <= 0)):
        errors.append("Expected a stacked sequence of finite increasing measured timestamps")
    missing = sorted(set(required) - set(arrays))
    for key, shape in required.items():
        if key not in arrays:
            continue
        value = arrays[key]
        if value.shape != (count, *shape):
            errors.append(f"Shape mismatch: {key}; expected {(count, *shape)}, got {value.shape}")
            continue
        if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
            errors.append(f"Nonfinite/nonnumeric state: {key}")
        if shape[-1:] == (7,) and (key.endswith("pose") or key.endswith("pose_base") or key.endswith("body_poses")):
            if not np.allclose(np.linalg.norm(value[..., 3:], axis=-1), 1, atol=1e-5):
                errors.append(f"Nonunit pose quaternion: {key}")
    command_fields = sorted(key for key in arrays if key.startswith("command/"))
    command_semantics = meta.get("command_semantics", {})
    missing_semantics = [key for key in command_fields if not command_semantics.get(key)]
    control_errors = []
    for key in command_fields:
        value = arrays[key]
        if not value.ndim or len(value) != count or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
            control_errors.append(f"Invalid recorded command: {key}")
    alternative_commands = ("command/joint_position", "command/joint_velocity", "command/left_tcp_pose",
                            "command/right_tcp_pose", "command/head_pose", "command/base_twist")
    mode_fields = {name: ("command/" + name,) for name in ("joint_position", "joint_velocity", "left_tcp_pose",
                  "right_tcp_pose", "head_pose", "base_pose", "base_twist", "actuator_control")}
    mode_fields["cartesian_pose"] = ("command/left_tcp_pose", "command/right_tcp_pose", "command/head_pose")
    modes, profile = meta.get("issued_control_modes", []), meta.get("command_profile", {})
    issued_fields, explicit_profile = [], False
    if isinstance(profile, dict) and "required_fields" in profile:
        profile_fields = profile["required_fields"]
        if (not isinstance(profile_fields, list) or not profile_fields
                or any(not isinstance(key, str) or not (key.startswith("command/") or key in ("controller_state_json", "applied_controls")) for key in profile_fields)
                or not any(key.startswith("command/") for key in profile_fields)):
            control_errors.append("command_profile.required_fields must name the actually issued command channels")
            issued_fields = []
        else:
            issued_fields = [key for key in profile_fields if key.startswith("command/")]
            explicit_profile = True
            if isinstance(modes, list) and modes and all(mode in mode_fields for mode in modes):
                implied = {key for mode in modes for key in mode_fields[mode]}
                if not implied <= set(issued_fields):
                    control_errors.append("Command profile omits channels of declared issued_control_modes")
    elif isinstance(modes, list) and modes:
        if any(mode not in mode_fields for mode in modes):
            control_errors.append("Unknown issued control mode; provide an explicit command_profile.required_fields")
        else:
            issued_fields = sorted({key for mode in modes for key in mode_fields[mode]})
            explicit_profile = True
    missing_controls = [key for key in (*issued_fields, "controller_state_json", "applied_controls") if key not in arrays]
    if not explicit_profile:
        missing_controls.append("issued_control_modes_or_command_profile")
    if "applied_controls" in arrays:
        value = arrays["applied_controls"]
        if (value.ndim != 3 or value.shape[0] != count or value.shape[-1] != dimensions.get("nu")
                or value.shape[1] < 1 or not np.isfinite(value).all()):
            control_errors.append("applied_controls must retain every physics substep, with named actuator width")
    if "controller_state_json" in arrays:
        values = arrays["controller_state_json"]
        if values.shape != (count,):
            control_errors.append("Controller snapshots are not row-aligned")
        else:
            for i, raw in enumerate(values):
                try:
                    value = json.loads(raw)
                    if not isinstance(value, dict) or not set(value) >= {"identity", "format", "joints", "targets", "memory", "servo_targets"}:
                        raise ValueError("Incomplete controller snapshot")
                    json.dumps(value, allow_nan=False)
                except (ValueError, TypeError):
                    control_errors.append(f"Invalid controller snapshot at row {i}")
                    break
    complete = not errors and not missing
    issued_missing_semantics = [key for key in issued_fields if not command_semantics.get(key)]
    actual_controls_complete = explicit_profile and not missing_controls and not issued_missing_semantics and not control_errors
    alternative_missing = [key for key in alternative_commands if key not in arrays or not command_semantics.get(key)]
    return {"state_contract": STATE_CONTRACT, "rows": count, "state_observations_complete": complete,
            "table_fields_complete": complete and not undefined_roles, "rgb_required": False,
            "missing_fields": sorted(set(missing + undefined_roles + [f"required_joint/{name}" for name in missing_joints])),
            "undefined_task_roles": undefined_roles, "errors": errors,
            "recorded_command_fields": command_fields, "command_fields_without_semantics": missing_semantics,
            "issued_control_modes": modes, "command_profile": profile, "required_issued_command_fields": issued_fields,
            "issued_command_fields_without_semantics": issued_missing_semantics,
            "missing_control_fields": missing_controls, "control_errors": control_errors,
            "complete_recorded_command_profile": actual_controls_complete,
            "actual_controls_complete": actual_controls_complete,
            "control_recording_complete": actual_controls_complete,
            "all_control_modes_available": not alternative_missing and not control_errors,
            "alternative_control_fields_missing": alternative_missing,
            "physics_validated": False, "policy_ready": False, "native_v5_compatibility_assessed": False}


def write_archive(root, dataset_id, episode_id, arrays, metadata):
    """Write one immutable common-format episode; absent fields stay absent.

    Camera/head/controller fields must come from recorded data when supplied.
    Original source arrays belong under ``source/``. Actual v5 observations use
    their exact v5 names. Other objects use ``observation/objects/<id>/pose`` and
    articulated states use ``observation/fixtures/<id>/joint_position``.
    """
    dataset_id, episode_id = _identifier(dataset_id), _identifier(episode_id)
    arrays = {_channel(k): np.asarray(v) for k, v in arrays.items()}
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    timed = "timestamp" in arrays
    index_channel = "timestamp" if timed else next((key for key in ("source/frame_index", "source/waypoint_index") if key in arrays), None)
    if not timed and ("timestamp" not in metadata.get("missing_fields", []) or index_channel is None):
        raise ValueError("Untimed archives require an explicit missing timestamp and source frame/waypoint index")
    timeline = arrays[index_channel]
    if (timeline.dtype.kind not in "iuf" or timeline.ndim != 1 or len(timeline) < 2
            or not np.isfinite(timeline).all() or np.any(np.diff(timeline) <= 0)):
        raise ValueError("Archive requires at least two finite increasing timestamps or explicit untimed indices")
    if not metadata.get("source_sequence") or not metadata.get("source_group"):
        raise ValueError("source_sequence and source_group are required to avoid duplicate counting")
    if not metadata.get("objects"):
        raise ValueError("Object interaction metadata is required; robot-only episodes are excluded")
    required_metadata = {"source_urls": list, "missing_fields": list, "derived_fields": dict, "simulation_assumptions": list}
    for key, kind in required_metadata.items():
        if not isinstance(metadata.get(key), kind):
            raise ValueError(f"{key} must be a {kind.__name__}")
    if "source_revision" not in metadata:
        raise ValueError("source_revision must be recorded, or explicitly null")
    for key, value in arrays.items():
        if value.dtype.kind not in "biufOUS":
            raise ValueError(f"Unsupported array dtype for {key}: {value.dtype}")
        if key.startswith(("observation/", "target/", "result/")) and (not value.ndim or len(value) != len(timeline)):
            raise ValueError(f"Canonical row channel is not aligned to timestamp: {key}")
    masks = metadata.get("validity_masks", {})
    for key, mask in masks.items():
        if (key not in arrays or mask not in arrays or arrays[mask].dtype.kind != "b"
                or not arrays[mask].ndim or arrays[mask].ndim > arrays[key].ndim
                or arrays[mask].shape != arrays[key].shape[:arrays[mask].ndim]):
            raise ValueError(f"Invalid validity-mask mapping: {key} -> {mask}")
        if arrays[key].dtype.kind in "iuf" and not np.isfinite(arrays[key][arrays[mask]]).all():
            raise ValueError(f"Mask marks nonfinite samples valid: {key}")
    for key, value in arrays.items():
        if key.startswith(("observation/", "target/")) and value.dtype.kind == "f" and not np.isfinite(value).all() and key not in masks:
            raise ValueError(f"Nonfinite canonical channel requires an explicit validity mask: {key}")
    missing = sorted(set(metadata["missing_fields"]) | (set(STATE_TABLE_FIELDS) - set(arrays)))
    metadata.update(schema=SCHEMA, dataset_id=dataset_id, episode_id=episode_id,
                    target_schema=V5_SCHEMA, target_action_contract=ACTION_CONTRACT,
                    reachy_agent_revision=AGENT_REVISION, policy_ready=False, missing_fields=missing,
                    rgb_required=False, rgb_stored=any(key.endswith("/rgb") for key in arrays),
                    storage_profile="recorded-modalities" if any(key.endswith("/rgb") for key in arrays) else "state-only",
                    missing_native_v5_fields=sorted(set(V5_FIELDS) - set(arrays)),
                    timed=timed, row_count=len(timeline), index_channel=index_channel,
                    export_assessment="archive_only; no new physics or controller validation performed")
    metadata["fields"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                               "nonfinite_count": int((~np.isfinite(value)).sum()) if value.dtype.kind in "f" else 0}
                          for key, value in arrays.items()}
    folder = Path(root) / "datasets" / dataset_id
    destination = folder / "episodes" / f"{episode_id}.hdf5"
    partial = destination.with_suffix(".hdf5.part")
    with _lock(folder):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or partial.exists():
            raise FileExistsError(f"Immutable export already exists: {destination}")
        dataset_metadata = {"schema": SCHEMA, "dataset_id": dataset_id, "target_schema": V5_SCHEMA,
                            "target_action_contract": ACTION_CONTRACT, "reachy_agent_revision": AGENT_REVISION,
                            "episode_metadata": "Per-episode HDF5 metadata_json and hash-bound JSON sidecar",
                            "policy_ready": False}
        meta_path = folder / "metadata.json"
        if meta_path.exists() and json.loads(meta_path.read_text()) != dataset_metadata:
            raise ValueError("Dataset root uses different metadata/schema")
        if not meta_path.exists():
            json_write(meta_path, dataset_metadata)
        with h5py.File(partial, "x") as handle:
            handle.attrs.update(schema=SCHEMA, metadata_json=json.dumps(metadata, allow_nan=False), policy_ready=False)
            for key, value in arrays.items():
                if value.dtype.kind in "OUS":
                    handle.create_dataset(key, data=value.astype(object), dtype=h5py.string_dtype("utf-8"))
                else:
                    handle.create_dataset(key, data=value, compression="gzip" if value.ndim and value.size else None)
        partial.rename(destination)
        json_write(destination.with_suffix(".json"), dict(metadata, hdf5_sha256=sha256(destination)))
    return destination


def inspect_archive(path):
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != SCHEMA:
            raise ValueError("Not a common retarget archive")
        metadata = json.loads(handle.attrs["metadata_json"])
        names = []
        handle.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        if set(names) != set(metadata["fields"]):
            raise ValueError("Archive field inventory differs from metadata")
        for key, spec in metadata["fields"].items():
            if list(handle[key].shape) != spec["shape"]:
                raise ValueError(f"Archive shape differs from metadata: {key}")
        report = {"schema": SCHEMA, "rows": metadata["row_count"], "timed": metadata["timed"], "policy_ready": False,
                  "dataset_id": metadata["dataset_id"], "episode_id": metadata["episode_id"],
                  "missing_fields": metadata["missing_fields"], "fields": metadata["fields"],
                  "rgb_required": metadata.get("rgb_required", False),
                  "missing_native_v5_fields": metadata.get("missing_native_v5_fields", [])}
    sidecar = json.loads(path.with_suffix(".json").read_text())
    report["sha256"] = sha256(path)
    if sidecar != dict(metadata, hdf5_sha256=report["sha256"]):
        raise ValueError("Archive/sidecar checksum or metadata mismatch")
    return report


def export_normalized(source, root, dataset_id=None, episode_id=None):
    """Export a normalized HDF5 path or {arrays, metadata, status, missing_fields}.

    Original arrays are retained under source/. Object/fixture observation aliases
    preserve their source frame and validity; source hands never become Reachy
    TCP observations, and source commands never become v5 actions.
    """
    if isinstance(source, dict):
        original = source["arrays"]
        meta = dict(source["metadata"])
        meta["status"] = source.get("status", meta.get("status", "normalized"))
        meta["missing_fields"] = source.get("missing_fields", meta.get("missing_fields", meta.get("missing", [])))
    else:
        original, meta = read_episode(source)
        meta["normalized_source_sha256"] = sha256(source)
    arrays = {f"source/{key}": value for key, value in original.items()}
    timestamps = original.get("time_s", original.get("timestamp"))
    if timestamps is not None:
        arrays["timestamp"] = timestamps
    for key in ("frame_index", "waypoint_index"):
        if key in original or "source/" + key in original:
            arrays["source/" + key] = original.get(key, original.get("source/" + key))
    aliases = {}
    validity = {}
    for key, value in original.items():
        if key.startswith(("objects/", "fixtures/")):
            target = "observation/" + key
            arrays[target] = value
            aliases[target] = {"source_field": key, "operation": "identity", "frame": "source_world"}
            if key.endswith("/pose") and key[:-4] + "valid" in original:
                validity[target] = "observation/" + key[:-4] + "valid"
    for key, mask in meta.get("validity_masks", {}).items():
        if key in original and mask in original:
            validity["source/" + key] = "source/" + mask
            if "observation/" + key in aliases:
                # A source mask may be outside the object hierarchy; retain its
                # actual source channel instead of inventing an observation alias.
                mapped_mask = "observation/" + mask if "observation/" + mask in arrays else "source/" + mask
                validity["observation/" + key] = mapped_mask
        elif key in arrays and mask in arrays:
            validity[key] = mask
        else:
            raise ValueError(f"Declared source validity mask is absent: {key} -> {mask}")
    sequence = meta.get("source_sequence", meta.get("sequence", episode_id))
    dataset_id = dataset_id or meta.get("source_id")
    episode_id = episode_id or meta.get("episode_id")
    metadata = {"source_sequence": sequence, "source_group": meta.get("source_group", f"{dataset_id}/{sequence}"),
                "source_urls": meta.get("source_urls", [meta["source_url"]] if meta.get("source_url") else []),
                "source_revision": meta.get("source_revision", meta.get("revision")),
                "missing_fields": list(meta.get("missing_fields", meta.get("missing", []))),
                "derived_fields": {**meta.get("derived_fields", {}), **aliases}, "validity_masks": validity,
                "simulation_assumptions": meta.get("simulation_assumptions", []),
                "objects": meta.get("objects", {}), "source_metadata": meta,
                "status": meta.get("status", "normalized"), "observation_frame": "source_world"}
    if not metadata["source_urls"]:
        metadata["missing_fields"].append("provenance/source_urls")
    if metadata["source_revision"] is None:
        metadata["missing_fields"].append("provenance/source_revision")
    return write_archive(root, dataset_id, episode_id, arrays, metadata)


def import_v5(source_episode, root, dataset_id, episode_id=None, *, validation_evidence=None):
    """Archive real v5 rows and optionally publish the original file byte-for-byte.

    Native publication requires a passed, hash-bound external validation report:
    {episode_sha256, status: passed, validator: simulation.validate, checks: ...}.
    This function does not run that validator or infer physics success itself.
    """
    source_episode = Path(source_episode)
    assessment = assess_v5(source_episode)
    if not assessment["structurally_compatible"]:
        raise ValueError(f"Incomplete/incompatible v5 recording: {assessment}")
    digest = assessment["episode_sha256"]
    evidence = None
    if validation_evidence is not None:
        evidence_path = Path(validation_evidence)
        evidence = json.loads(evidence_path.read_text())
        checks = evidence.get("checks", [])
        if (evidence.get("episode_sha256") != digest or evidence.get("status") != "passed"
                or evidence.get("validator") != "simulation.validate" or not checks
                or any(row.get("ok") is not True for row in checks) or not assessment["recorded_success"]):
            raise ValueError("Publication needs successful, hash-bound simulation.validate evidence")
        evidence = dict(evidence, report_sha256=sha256(evidence_path))
    with h5py.File(source_episode, "r") as handle:
        arrays = _read_arrays(handle)
        original_metadata = json.loads(handle.attrs["metadata_json"])
        original_episode_id = str(handle.attrs["episode_id"])
    if sha256(source_episode) != digest:
        raise ValueError("Source episode changed during import")
    episode_id = episode_id or original_episode_id
    metadata = {"source_sequence": original_episode_id, "source_group": f"reachy-agent/{original_episode_id}/{digest}",
                "source_urls": [], "source_revision": AGENT_REVISION,
                "missing_fields": [], "derived_fields": {}, "simulation_assumptions": ["Original simulator settings are retained in source_metadata.provenance.simulation"],
                "objects": {"pickup": {"channel": "observation/pickup_pose"},
                            "receptacle": {"channel": "observation/receptacle_pose"}},
                "source_metadata": original_metadata, "source_episode_sha256": digest,
                "v5_structure": assessment, "external_validation": evidence,
                "status": "recorded_v5", "independent_new_demonstration": False}
    archive = write_archive(root, dataset_id, episode_id, arrays, metadata)
    result = {"archive": str(archive), "native_episode": None, "assessment": assessment}
    if evidence is not None:
        # Scene/controller-specific roots prevent mixing incompatible maps or controllers.
        native_root = Path(root) / "datasets" / _identifier(dataset_id) / "policy-v5"
        with _lock(native_root):
            metadata_path = native_root / "metadata.json"
            if metadata_path.exists() and json.loads(metadata_path.read_text()) != original_metadata:
                raise ValueError("Native policy root has a different prepared scene/controller contract")
            episodes = native_root / "episodes"
            episodes.mkdir(exist_ok=True)
            for existing in episodes.glob("[0-9]*.json"):
                if json.loads(existing.read_text()).get("sha256") == digest:
                    raise ValueError("This exact source episode is already published")
            index = max((int(p.stem) for p in episodes.glob("[0-9]*.hdf5")), default=-1) + 1
            destination = episodes / f"{index:06d}.hdf5"
            partial = destination.with_suffix(".hdf5.part")
            with source_episode.open("rb") as source_handle, partial.open("xb") as target:
                shutil.copyfileobj(source_handle, target)
            if sha256(partial) != digest:
                raise ValueError("Source episode changed during native publication")
            if not metadata_path.exists():
                json_write(metadata_path, original_metadata)
            partial.rename(destination)
            json_write(destination.with_suffix(".json"), {"sha256": digest, "source_group": metadata["source_group"],
                       "external_validation": evidence, "source_episode_index_preserved": True})
            result["native_episode"] = str(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export-normalized", "inspect", "assess-v5", "assess-state", "import-v5"))
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/mnt/reachy-retarget"))
    parser.add_argument("--dataset")
    parser.add_argument("--episode-id")
    parser.add_argument("--validation-evidence", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_archive(args.episode)
    elif args.command == "assess-v5":
        result = assess_v5(args.episode)
    elif args.command == "assess-state":
        result = assess_state_observations(args.episode)
    elif args.command == "import-v5":
        if not args.dataset:
            parser.error("import-v5 requires --dataset")
        result = import_v5(args.episode, args.root, args.dataset, args.episode_id, validation_evidence=args.validation_evidence)
    else:
        result = {"archive": str(export_normalized(args.episode, args.root, args.dataset, args.episode_id))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
