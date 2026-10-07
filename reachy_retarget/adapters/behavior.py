"""Read BEHAVIOR raw replay records without importing OmniGibson or using a network.

Serialized states have a scene- and version-dependent layout.  They are retained,
never guessed to be MuJoCo qpos or decoded using a fixed numerical offset.
"""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

CODE_REVISION = "a8247a8cc1633fe1ca0cc66aa07243d46c64f155"
DATA_REVISION = "f9d2901112993d75684223820eec70722806c2c4"
BASE = "https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata"
PILOT = {
    "path": "task-0002/episode_00021890.hdf5",
    "url": BASE + "/resolve/" + DATA_REVISION + "/task-0002/episode_00021890.hdf5",
    "bytes": 667124,
    "sha256": "b01e7b8183e2c738edf1f5e65bba53163fee30713970845631401e20e6cd1a75",
}
SOURCES = [
    "https://behavior.stanford.edu/challenge/dataset.html",
    "https://github.com/StanfordVL/BEHAVIOR-1K/blob/" + CODE_REVISION
    + "/OmniGibson/omnigibson/envs/hdf5_data_wrapper.py",
    "https://github.com/StanfordVL/BEHAVIOR-1K/issues/2245#issuecomment-4597125008",
    "https://behavior.stanford.edu/getting_started/important_concepts.html",
]


def describe():
    """Return an explicit acquisition and replay plan; perform no side effects."""
    return {
        "dataset": "behavior", "adapter": "omnigibson-raw-hdf5-v1",
        "source_revision": DATA_REVISION, "source_code_revision": CODE_REVISION,
        "source_urls": SOURCES, "fetch": [dict(PILOT)],
        "destination": "/mnt/reachy-retarget/data/raw/behavior",
        "availability": "public raw HDF5; 20000 release episodes, not acquired coverage",
        "native_fields": ["data/demo_*/state", "state_size", "action", "reward",
                          "terminated", "truncated", "data.attrs.config", "data.attrs.scene_file"],
        "native_replay": {
            "runtime_recipe": "configs/native-replay-behavior.json",
            "container": "docker.io/stanfordvl/behavior@sha256:3dda0241dae8c6afc5dc3b5d29f9d6e52035d96cc68e8117db37723e7fbb674e",
            "container_source_revision": "bd049de3119acdcdf2334fe9e1ebe060fa20c108",
            "cpu_only_replay_supported": False,
            "image_manifest_verified_runtime_not_launched": True,
            "source_repository": "https://github.com/StanfordVL/BEHAVIOR-1K",
            "inspected_replay_code_revision": CODE_REVISION,
            "recorded_pilot_revision": "c1736c23fe8083bb210da158008ad0bfa1638423",
            "recorded_pilot_versions": {"omnigibson": "3.7.0-alpha", "bddl": "3.7.0a0", "og_dataset": "1.2.0rc21"},
            "entry_point": "OmniGibson/scripts/learning/replay_obs.py",
            "argv": ["--data_folder", "/mnt/reachy-retarget/native/behavior", "--demo_id", "21890", "--output_format", "hdf5"],
            "expected_input_relative_path": "2026-challenge-rawdata/task-0002/episode_00021890.hdf5",
            "dependencies": ["isolated compatible OmniGibson and Isaac Sim runtime", "GPU and compatible NVIDIA driver",
                             "BEHAVIOR scene/object assets", "2026-challenge-task-instances metadata/scene JSONs"],
            "pose_callback": "reachy_retarget.adapters.behavior.capture_replay_frame(env, object_names)",
            "callback_hook": "HDF5PlaybackWrapper.playback_episode(post_state_update_callback=callback)",
            "callback_status": "read-only frame extractor implemented; source-runtime orchestration requires dependencies",
            "compatibility": "Recorded revision differs from current replay code. Compare restored objects and state lengths before accepting a migrated replay.",
            "source_state_writes": "Only source reference reconstruction; never use this state replay during target MuJoCo validation",
        },
        "pipeline": [
            "fetch an explicit pinned raw member; check length and SHA-256",
            "inspect recorded simulator/assets versions and controller configuration",
            "replay source state in isolated compatible OmniGibson runtime; export measured world object/base/EEF poses",
            "retain rigid and articulated task objects; route particle/material tasks to unsupported representation",
            "retain licensed BEHAVIOR assets inside OmniGibson; require separate rights before any asset export to MuJoCo",
            "retarget measured hand trajectories with minimal Reachy pad translation; retain source base/timing reference",
            "validate actuator-only Reachy dynamics with source assisted grasp disabled; preserve every attempt",
            "write common Reachy episode and provenance/validation sidecars",
        ],
        "agent_review": ["recorded-version compatibility", "object-role and articulation mapping",
                         "missing object states", "minimal pad alignment", "MuJoCo contact failures"],
        "missing_fields": ["decoded world object/base/EEF trajectories", "decoded named robot joint position/velocity",
                           "licensed scene assets in GPU worker", "permission for cross-simulator asset use", "Reachy physical validation"],
        "source_grasp": "assisted during collection/evaluation; not evidence of a physical Reachy grasp",
        "license": {"code": "MIT", "raw_data": "no explicit dataset license in checked HF metadata",
                    "assets": "BEHAVIOR Data Bundle EULA: noncommercial academic use inside OmniGibson only; asset extraction/redistribution prohibited"},
        "status": "native_inspection_available_replay_required",
    }


def _json_attr(value, label):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        result = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid JSON metadata: " + label) from exc
    if not isinstance(result, dict):
        raise ValueError("Expected object metadata: " + label)
    return result


def _hashes(path):
    size = Path(path).stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            git.update(block)
    return {"bytes": size, "sha256": sha.hexdigest(), "git_blob_sha1": git.hexdigest()}


def _header(handle):
    if "data" not in handle or not isinstance(handle["data"], h5py.Group):
        raise ValueError("Expected native OmniGibson data group")
    data = handle["data"]
    config = _json_attr(data.attrs.get("config"), "data.config")
    scene = _json_attr(data.attrs.get("scene_file"), "data.scene_file")
    frequency = config.get("env", {}).get("action_frequency")
    if not isinstance(frequency, (int, float)) or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Missing or invalid recorded action_frequency; no default is inferred")
    episodes = sorted((k for k in data if k.startswith("demo_") and isinstance(data[k], h5py.Group)),
                      key=lambda s: (int(s[5:]) if s[5:].isdigit() else 10**12, s))
    if not episodes:
        raise ValueError("No native demo groups")
    return data, config, scene, float(frequency), episodes


def _shape_tree(group):
    result = {}
    group.visititems(lambda key, obj: result.update({key: {"shape": list(obj.shape), "dtype": str(obj.dtype)}})
                     if isinstance(obj, h5py.Dataset) else None)
    return result


def _scene_objects(scene, config):
    """Read named scene/task roles; no poses or collision properties are inferred."""
    registry = scene.get("objects_info", {}).get("init_info", {})
    roles = scene.get("metadata", {}).get("task", {}).get("inst_to_name", {})
    # Fixed ground remains scene geometry, not an episode object-pose feature.
    names = {name for role, name in roles.items() if not role.startswith(("agent.n.", "floor.n."))}
    robot_names = {r.get("name") for r in config.get("robots", [])}
    result = {}
    for name in sorted(names - robot_names):
        spec = registry.get(name, {})
        if spec.get("class_module", "").startswith("omnigibson.robots"):
            continue
        args = spec.get("args", {})
        result[name] = {
            "source_roles": [role for role, target in roles.items() if target == name],
            "source_class": spec.get("class_name"), "decoded_pose_available": False,
            "pose_source": "serialized state; compatible simulator replay required",
            "source_asset": {k: args[k] for k in ("category", "model", "scale", "fixed_base", "expected_file_hash") if k in args},
        }
    return result


def object_coverage(scene, config, decoded_names=(), selected_names=None, selection_evidence=None):
    """Audit declared task objects; scene inventories are provenance, not targets."""
    registry = scene.get("objects_info", {}).get("init_info", {})
    if not isinstance(registry, dict):
        raise ValueError("Invalid scene object registry metadata")
    robot_names = {r.get("name") for r in config.get("robots", [])}
    scene_names = {name for name, spec in registry.items() if name not in robot_names
                   and not spec.get("class_module", "").startswith("omnigibson.robots")}
    task_roles = scene.get("metadata", {}).get("task", {}).get("inst_to_name", {})
    source_scope = set(task_roles.values()) - robot_names
    ground_names = {name for role, name in task_roles.items() if role.startswith("floor.n.")}
    task_names = source_scope - ground_names if selected_names is None else set(selected_names) - robot_names - ground_names
    decoded = set(decoded_names)
    return {
        "scope": "declared task-related object pose features only",
        "selection_evidence": selection_evidence or "recorded BDDL task scope excluding robot and fixed world ground",
        "scene_registry_available": bool(registry),
        "scene_object_names": sorted(scene_names), "scene_object_count": len(scene_names),
        "scene_pose_completeness_required": False,
        "task_object_names": sorted(task_names), "task_object_count": len(task_names),
        "task_scope_available": bool(task_roles) or selected_names is not None,
        "source_bddl_scope_names": sorted(source_scope),
        "excluded_fixed_ground_pose_names": sorted(ground_names),
        "other_source_task_scope_names_for_semantic_review": sorted(source_scope - ground_names - task_names),
        "decoded_object_pose_names": sorted(decoded), "decoded_object_pose_count": len(decoded),
        "missing_task_object_pose_names": sorted(task_names - decoded),
        "decoded_names_outside_scene_registry": sorted(decoded - scene_names),
        "all_task_object_poses_decoded": bool(task_names) and task_names <= decoded,
        "articulated_link_pose_coverage": "not decoded; object root pose does not establish moving link/joint coverage",
    }


def capture_replay_frame(env, object_names):
    """Read a restored source simulator frame through the official object API.

    Call this only from the explicit source-replay stage. It does not import any
    simulator, advance time, create constraints, or write a body's state.
    """
    def numpy_value(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=float)

    def pose(obj):
        position, quaternion_xyzw = obj.get_position_orientation()
        position, quaternion_xyzw = numpy_value(position), numpy_value(quaternion_xyzw)
        if position.shape != (3,) or quaternion_xyzw.shape != (4,):
            raise ValueError("Unexpected source pose dimensions")
        output = np.r_[position, quaternion_xyzw[[3, 0, 1, 2]]]
        if not np.isfinite(output).all() or not np.isclose(np.linalg.norm(output[3:]), 1, atol=1e-4):
            raise ValueError("Invalid restored source pose")
        return output

    if len(env.robots) != 1:
        raise ValueError("Replay frame extractor requires exactly one source robot")
    robot = env.robots[0]
    arrays = {"source/robot_root_pose": pose(robot)}
    for arm in ("left", "right"):
        if arm in robot.eef_links:
            arrays["hand/" + arm + "_pose"] = pose(robot.eef_links[arm])
    for name in object_names:
        if name == robot.name:
            raise ValueError("Robot cannot be declared as an interaction object")
        obj = env.scene.object_registry("name", name)
        arrays["objects/" + name + "/pose"] = pose(obj)
        joint_positions = numpy_value(obj.get_joint_positions())
        if joint_positions.size:
            arrays["objects/" + name + "/joint_position"] = joint_positions
    return arrays


def inspect(path):
    """Inspect local raw metadata, including T/T+1 alignment and assistance."""
    path = Path(path)
    identity = _hashes(path)
    with h5py.File(path, "r") as handle:
        data, config, scene, frequency, episodes = _header(handle)
        shapes = {name: _shape_tree(data[name]) for name in episodes}
        modes = [r.get("grasping_mode", "unspecified") for r in config.get("robots", [])]
        return {
            "dataset": "behavior", "path": str(path), **identity,
            "publisher_identity_verified": identity["sha256"] == PILOT["sha256"],
            "source_revision": DATA_REVISION if identity["sha256"] == PILOT["sha256"] else None,
            "status": "native_inspected_replay_required", "physics_validated": False,
            "episodes": shapes, "episode_count": len(episodes), "action_frequency_hz": frequency,
            "recorded_versions": scene.get("versions", {}),
            "source_grasping_modes": modes, "task": config.get("task", {}),
            "objects": _scene_objects(scene, config),
            "object_coverage": object_coverage(scene, config),
            "scene": config.get("scene", {}), "serialized_state_decoded": False,
            "missing_fields": describe()["missing_fields"],
            "notes": ["Raw and LeRobot releases are views of the same recordings.",
                      "Object state is serialized, not absent; do not classify this as robot-only.",
                      "Episode counts/num_samples attrs are not substituted for actual array lengths."],
        }


def normalize(path, episode=None):
    """Preserve native arrays and timing; fail closed on unavailable pose decoding.

    A blocked result still includes useful raw observations for the common source
    archive. It must not enter a retargeting or training success collection.
    """
    path = Path(path)
    summary = inspect(path)
    with h5py.File(path, "r") as handle:
        data, config, scene, frequency, episodes = _header(handle)
        if episode is None:
            if len(episodes) != 1:
                raise ValueError("Select a native demo explicitly: " + ", ".join(episodes))
            episode = episodes[0]
        if episode not in episodes:
            raise ValueError("Unknown native episode: " + str(episode))
        group = data[episode]
        for key in ("state", "state_size", "action"):
            if key not in group:
                raise ValueError("Missing native dataset: " + key)
        states, sizes, actions = (group[key][()] for key in ("state", "state_size", "action"))
        if states.ndim != 2 or actions.ndim != 2 or len(states) < 2:
            raise ValueError("Native state/action matrices need at least two state frames")
        if len(states) not in (len(actions), len(actions) + 1):
            raise ValueError("Unsupported native state/action length relationship")
        if sizes.shape != (len(states),) or sizes.dtype.kind not in "iu" or np.any(sizes <= 0) or np.any(sizes > states.shape[1]):
            raise ValueError("Invalid serialized state_size")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite native state/action")
        times = np.arange(len(states), dtype=float) / frequency
        arrays = {"time_s": times, "timestamp": times, "source/state": states,
                  "source/state_size": sizes, "source/action": actions,
                  "source/action_time_s": np.arange(len(actions), dtype=float) / frequency}
        for name in ("reward", "terminated", "truncated"):
            if name in group:
                value = group[name][()]
                if value.shape != (len(actions),):
                    raise ValueError("Native transition field is not action-aligned: " + name)
                arrays["source/" + name] = value
        native_name = PILOT["path"] if summary["publisher_identity_verified"] else path.parent.name + "/" + path.name
        sequence = native_name + "/" + episode
        metadata = {
            "source_format": "BEHAVIOR-OmniGibson-HDF5", "source_sequence": sequence,
            "source_group": "behavior/" + sequence,
            "scenario": config.get("task", {}).get("activity_name"),
            "source_revision": summary["source_revision"], "source_urls": SOURCES + [PILOT["url"]]
            if summary["publisher_identity_verified"] else SOURCES,
            "provenance": [{"path": str(path), "sha256": summary["sha256"]}],
            "source_configuration": config, "source_scene_file": scene,
            "recorded_versions": scene.get("versions", {}), "fps": frequency,
            "objects": _scene_objects(scene, config),
            "object_coverage": object_coverage(scene, config),
            "state_control_coverage": {
                "source_raw_actions_preserved": True, "source_serialized_states_preserved": True,
                "left_arm": {"world_eef_pose_decoded": False, "named_joint_state_decoded": False},
                "right_arm": {"world_eef_pose_decoded": False, "named_joint_state_decoded": False},
                "neck": {"named_joint_state_decoded": False, "named_control_decoded": False},
                "base": {"world_pose_decoded": False, "velocity_decoded": False, "named_control_decoded": False},
                "reachy_state_or_control": "not present in source adapter output",
            },
            "source_grasping_modes": summary["source_grasping_modes"], "physics_validated": False,
            "derived_fields": {"time_s": "Frame index / recorded env.action_frequency; no timestamps were recorded"},
            "simulation_assumptions": ["Source state replay is a reference extraction operation, not target dynamics validation.",
                                       "Source assistance must never be used in Reachy physical validation."],
            "timing": {"state_frames": len(states), "action_frames": len(actions),
                       "terminal_state_recorded": len(states) == len(actions) + 1},
            "missing": summary["missing_fields"],
        }
    return {"arrays": arrays, "metadata": metadata, "status": "blocked_native_state_decode",
            "missing_fields": summary["missing_fields"]}
