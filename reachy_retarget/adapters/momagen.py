"""Offline MoMaGen source-reference adapter for the published bimanual HDF5s.

Only recorded achieved EEF/object/base transforms are normalized. Source action
commands remain source commands and assisted grasps never imply target success.
"""

from pathlib import Path

import h5py
import numpy as np

from ..episodes import matrices_to_pose
from . import behavior

SOURCE_REVISION = "5da621667669b15b97d7a220d086283b3c8ad0e5"
BASE = "https://raw.githubusercontent.com/ChengshuLi/MoMaGen/" + SOURCE_REVISION + "/"
BLOB_RECORDS = [
    ("r1_bringing_water", 15012078, "4848856cb1d2814a105c297e58407df503bf0445", "demo_0", 3324),
    ("r1_clean_pan", 103232338, "7e96e872d77d95fb15e7209507d6a131499e47a7", "demo_4", 1751),
    ("r1_dishes_away", 88217546, "39f4ca5c4e132480bb0cb6b30be2cc9b3657e1e6", "demo_2", 2788),
    ("r1_pick_cup", 29948093, "8a505c38722f8127eddbdbf40eb6cdd322381285", "demo_14", 909),
    ("r1_picking_up_trash", 7632853, "1352dbb41f95d4a14c9accb1efc4568c405fcaf7", "demo_0", 2483),
    ("r1_tidy_table", 25037752, "eba6186ea0560b2dcea01199d866313134599ae0", "demo_1", 1273),
]
# Named references in the pinned base_configs task_spec object_ref/attached_obj.
# The recorder also retains explicitly tracked task supports/targets; those stay.
MANIPULATION_REFERENCES = {
    "r1_bringing_water": {"beer_bottle_595", "fridge_dszchb_0"},
    "r1_clean_pan": {"frying_pan_602", "scrub_brush_601"},
    "r1_dishes_away": {"plate_601", "plate_602", "plate_603"},
    "r1_pick_cup": {"coffee_cup_7"},
    "r1_picking_up_trash": {"can_of_soda_261", "trash_can_262"},
    "r1_tidy_table": {"teacup_601", "drop_in_sink_awvzkn_0"},
}
SOURCE_URLS = [
    BASE + "momagen/env_interfaces/omnigibson.py",
    BASE + "momagen/scripts/prepare_src_dataset.py",
    BASE + "momagen/utils/file_utils.py",
    BASE + "docs/tutorials/generating-data.md",
    BASE + "LICENSE", BASE + "LICENSE.NVIDIA",
    "https://github.com/StanfordVL/BEHAVIOR-1K/blob/2ca5503895b2c81e02226dbe11c3fee1a68b5d6c/OmniGibson/omnigibson/controllers/multi_finger_gripper_controller.py",
    "https://github.com/StanfordVL/BEHAVIOR-1K/blob/2ca5503895b2c81e02226dbe11c3fee1a68b5d6c/OmniGibson/omnigibson/robots/r1.py",
]
MISSING = ["Reachy measured joints/actions", "permission for cross-simulator BEHAVIOR asset use",
           "source physical contact forces", "named source arm/base/neck joint positions and velocities",
           "Reachy physical validation"]


def describe():
    records = [{"path": "momagen/datasets/processed_source_demos/" + name + ".hdf5",
                "url": BASE + "momagen/datasets/processed_source_demos/" + name + ".hdf5",
                "bytes": size, "git_blob_sha1": sha, "sha256": None,
                "enriched_episode": episode, "reference_frames": frames}
               for name, size, sha, episode, frames in BLOB_RECORDS]
    # The small can/trash-bin episode is a mobile rigid-object pilot.
    records.sort(key=lambda r: r["bytes"])
    return {
        "dataset": "momagen", "adapter": "momagen-bimanual-datagen-info-v1",
        "source_revision": SOURCE_REVISION, "source_urls": SOURCE_URLS,
        "fetch": records, "destination": "/mnt/reachy-retarget/data/raw/momagen",
        "availability": "six published enriched references across six HDF5 files; 27 total native groups",
        "native_fields": ["datagen_info/eef_pose (N,8,4; left then right)",
                          "datagen_info/base_pose (N,4,4)", "datagen_info/object_poses/* (N,4,4)",
                          "datagen_info/gripper_action (N,2)", "state", "state_size", "action"],
        "pipeline": [
            "fetch pinned source HDF5, verify Git blob SHA-1 and save independent SHA-256",
            "inspect every group; preserve unannotated attempts without inventing object poses or success labels",
            "normalize achieved world EEF/base/object transforms and original frame timing directly",
            "keep full navigation/bimanual reference and source attachment diagnostics",
            "retain BEHAVIOR assets within OmniGibson; require separate rights before cross-simulator asset conversion",
            "map source EEF frames to Reachy pads using translation-first alignment; diagnose reachability separately",
            "validate target actuator dynamics without source assisted grasp; keep failed attempts",
            "write common Reachy episode with source reference and explicit validation status",
        ],
        "agent_review": ["object vs robot-link role", "mobile-base reachability", "bimanual contact handoff",
                         "minimal pad alignment", "articulated fixture fidelity", "source assistance removal"],
        "generator": {"command": "python momagen/scripts/generate_dataset.py --config CONFIG --num_demos N --bimanual --folder OUTPUT --seed SEED",
                      "source_reference_required": True,
                      "submodules": {"BEHAVIOR-1K": "2ca5503895b2c81e02226dbe11c3fee1a68b5d6c",
                                     "robomimic": "34460828098dc8ca0694f330cf48daeb96a4ef3a"},
                      "note": "Generated variants retain source-demo ancestry and are not independent human demonstrations."},
        "license": {"code": "MIT with NVIDIA portions under LICENSE.NVIDIA",
                    "source_demos": "files published in code repository; no separate data-specific grant found",
                    "assets": "BEHAVIOR Data Bundle terms permit noncommercial academic use inside OmniGibson; extraction/redistribution prohibited"},
        "source_grasp": "all six inspected source configurations explicitly use assisted grasp",
        "status": "enriched_reference_normalization_available", "missing_fields": list(MISSING),
    }


def _known_identity(identity):
    return next((row for row in describe()["fetch"] if row["git_blob_sha1"] == identity["git_blob_sha1"]), None)


def _coverage(scene, config, recorded_names, source_stem):
    declared = set(recorded_names) | MANIPULATION_REFERENCES.get(source_stem, set())
    coverage = behavior.object_coverage(
        scene, config, recorded_names, selected_names=declared,
        selection_evidence="Native datagen_info task-tracked scene objects plus pinned task_spec object_ref/attached_obj names; robot links and fixed world ground excluded",
    )
    coverage["manipulation_reference_names"] = sorted(MANIPULATION_REFERENCES.get(source_stem, set()))
    coverage["missing_manipulation_reference_pose_names"] = sorted(MANIPULATION_REFERENCES.get(source_stem, set()) - set(recorded_names))
    coverage["task_config_schema_verified"] = source_stem in MANIPULATION_REFERENCES
    coverage["task_config_url"] = BASE + "momagen/datasets/base_configs/" + source_stem + ".json" if source_stem in MANIPULATION_REFERENCES else None
    return coverage


def inspect(path):
    """Return native schema diagnostics without importing the source simulator."""
    path = Path(path)
    identity = behavior._hashes(path)
    known = _known_identity(identity)
    with h5py.File(path, "r") as handle:
        data, config, scene, frequency, episodes = behavior._header(handle)
        rows, enriched = {}, []
        coverage_by_episode = {}
        for name in episodes:
            schema = behavior._shape_tree(data[name])
            rows[name] = schema
            if all(key in schema for key in ("datagen_info/eef_pose", "datagen_info/base_pose",
                                             "datagen_info/gripper_action")) and any(
                    key.startswith("datagen_info/object_poses/") for key in schema):
                enriched.append(name)
            robot_names = {r.get("name") for r in config.get("robots", [])} | {"torso_link4"}
            pose_names = {key.removeprefix("datagen_info/object_poses/") for key in schema
                          if key.startswith("datagen_info/object_poses/")}
            source_stem = Path(known["path"]).stem if known else path.stem
            coverage_by_episode[name] = _coverage(scene, config, pose_names - robot_names, source_stem)
        return {
            "dataset": "momagen", "path": str(path), **identity,
            "publisher_identity_verified": known is not None,
            "source_revision": SOURCE_REVISION if known else None,
            "publisher_url": known["url"] if known else None,
            "status": "native_inspected", "physics_validated": False,
            "episodes": rows, "episode_count": len(episodes), "enriched_episodes": enriched,
            "unannotated_episodes": [name for name in episodes if name not in enriched],
            "object_coverage_by_episode": coverage_by_episode,
            "action_frequency_hz": frequency, "recorded_versions": scene.get("versions", {}),
            "source_grasping_modes": [r.get("grasping_mode", "unspecified") for r in config.get("robots", [])],
            "task": config.get("task", {}), "missing_fields": list(MISSING),
            "notes": ["Unannotated groups retain source states; they are not classified as robot-only.",
                      "source_og and processed_source_demos are alternate views of the same recordings.",
                      "Metadata inspection does not check transform values; normalize validates every frame."],
        }


def _pose(matrices, key, frames=None):
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or (frames is not None and len(matrices) != frames):
        raise ValueError("Invalid transform shape: " + key)
    if not np.isfinite(matrices).all() or not np.allclose(matrices[:, 3], [0, 0, 0, 1], atol=1e-5, rtol=0):
        raise ValueError("Nonfinite or nonhomogeneous transform: " + key)
    rotations = matrices[:, :3, :3]
    if not np.allclose(rotations.swapaxes(1, 2) @ rotations, np.eye(3), atol=1e-4, rtol=0) or not np.allclose(
            np.linalg.det(rotations), 1, atol=1e-4, rtol=0):
        raise ValueError("Nonrigid recorded transform: " + key)
    return matrices_to_pose(matrices)


def normalize(path, episode=None):
    """Normalize exactly one source group; do not crop, retime, or synthesize data."""
    path = Path(path)
    summary = inspect(path)
    if episode is None:
        if len(summary["enriched_episodes"]) != 1:
            raise ValueError("Select an enriched native episode explicitly: " + ", ".join(summary["enriched_episodes"]))
        episode = summary["enriched_episodes"][0]
    if episode not in summary["episodes"]:
        raise ValueError("Unknown native episode: " + str(episode))
    if episode not in summary["enriched_episodes"]:
        result = behavior.normalize(path, episode)
        result["metadata"].update(source_format="MoMaGen-raw-OmniGibson-HDF5",
                                  source_revision=summary["source_revision"], source_urls=SOURCE_URLS,
                                  source_group="momagen/" + path.stem + "/" + episode)
        return result
    with h5py.File(path, "r") as handle:
        data, config, scene, frequency, _ = behavior._header(handle)
        group = data[episode]
        info = group["datagen_info"]
        interface = info.attrs.get("env_interface_type")
        if isinstance(interface, bytes):
            interface = interface.decode()
        if interface != "omnigibson_bimanual":
            raise ValueError("Unverified EEF order for source interface: " + str(interface))
        eef = info["eef_pose"][()]
        if eef.ndim != 3 or eef.shape[1:] != (8, 4) or len(eef) < 2:
            raise ValueError("Expected at least two stacked left/right EEF transforms")
        frames = len(eef)
        time_s = np.arange(frames, dtype=float) / frequency
        grip = info["gripper_action"][()]
        if grip.shape != (frames, 2) or not np.isfinite(grip).all():
            raise ValueError("Expected aligned finite left/right gripper source commands")
        arrays = {"time_s": time_s, "timestamp": time_s,
                  "hand/left_pose": _pose(eef[:, :4], "eef_left", frames),
                  "hand/right_pose": _pose(eef[:, 4:], "eef_right", frames),
                  "source/robot_root_pose": _pose(info["base_pose"][()], "base_pose", frames),
                  "source/gripper_action": grip,
                  "source/left_gripper_action": grip[:, 0], "source/right_gripper_action": grip[:, 1]}
        robot_names = {r.get("name") for r in config.get("robots", [])}
        robot_names.add("torso_link4")  # Explicit robot-link exception in the pinned source interface.
        roles = scene.get("metadata", {}).get("task", {}).get("inst_to_name", {})
        fixed_ground_names = {name for role, name in roles.items() if role.startswith("floor.n.")}
        objects, robot_links = {}, []
        for name in sorted(info["object_poses"]):
            if name in fixed_ground_names:
                continue
            pose = _pose(info["object_poses"][name][()], "object_poses/" + name, frames)
            if name in robot_names:
                arrays["source/robot_links/" + name + "/pose"] = pose
                robot_links.append(name)
                continue
            arrays["objects/" + name + "/pose"] = pose
            objects[name] = {"pose_frame": "world", "pose_source": "recorded achieved datagen_info/object_poses",
                             "geometry_source": "source_scene_file plus separately acquired BEHAVIOR assets",
                             "motion_type": "not inferred from name or pose constancy"}
        if not objects:
            raise ValueError("No recorded scene-object trajectories after excluding verified robot links")
        # Keep raw state and all transition fields at their original lengths.
        for key in ("action", "state", "state_size", "reward", "terminated", "truncated"):
            if key in group:
                value = group[key][()]
                if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                    raise ValueError("Invalid numeric native field: " + key)
                arrays["source/" + key] = value
        if "source/action" in arrays:
            if len(arrays["source/action"]) != frames:
                raise ValueError("Enriched reference is not aligned with recorded source actions")
            arrays["source/action_time_s"] = time_s.copy()
        if "subtask_term_signals" in info:
            for name, dataset in info["subtask_term_signals"].items():
                values = dataset[()]
                if len(values) != frames or values.dtype.kind not in "biuf" or not np.isfinite(values).all():
                    raise ValueError("Invalid subtask termination signal: " + name)
                arrays["source/subtask_term_signals/" + name] = values
        source_stem = Path(summary["publisher_url"]).stem if summary["publisher_url"] else path.stem
        sequence = source_stem + "/" + episode
        missing = list(MISSING)
        coverage = _coverage(scene, config, objects, source_stem)
        if not coverage["all_task_object_poses_decoded"]:
            missing.append("declared task-related object pose trajectories")
        if not summary["publisher_identity_verified"]:
            missing.append("publisher file identity not verified")
        metadata = {
            "source_format": "MoMaGen-bimanual-datagen-info-HDF5", "source_sequence": sequence,
            "source_group": "momagen/" + sequence, "source_revision": summary["source_revision"],
            "scenario": config.get("task", {}).get("activity_name"),
            "source_urls": SOURCE_URLS + ([summary["publisher_url"]] if summary["publisher_url"] else []),
            "schema_reference_revision": SOURCE_REVISION, "fps": frequency,
            "provenance": [{"path": str(path), "sha256": summary["sha256"], "git_blob_sha1": summary["git_blob_sha1"]}],
            "source_configuration": config, "source_scene_file": scene,
            "recorded_versions": scene.get("versions", {}), "objects": objects,
            "object_coverage": coverage,
            "state_control_coverage": {
                "source_raw_actions_preserved": "source/action" in arrays,
                "source_serialized_states_preserved": "source/state" in arrays,
                "left_arm": {"world_eef_pose_decoded": True, "named_joint_state_decoded": False, "gripper_command_decoded": True},
                "right_arm": {"world_eef_pose_decoded": True, "named_joint_state_decoded": False, "gripper_command_decoded": True},
                "neck": {"named_joint_state_decoded": False, "named_control_decoded": False},
                "base": {"world_pose_decoded": True, "velocity_decoded": False, "named_control_decoded": False},
                "reachy_state_or_control": "not present in source adapter output",
            },
            "excluded_robot_link_roles": robot_links, "physics_validated": False,
            "source_grasping_modes": summary["source_grasping_modes"],
            "gripper_semantics": {
                "ordering": ["left", "right"], "quantity": "source controller command, not measured pad gap or contact",
                "published_reference_controllers": "MultiFingerGripperController, smooth, default input/output limits",
                "published_command_direction": "-1 closed / +1 open; intermediate values are source position targets",
                "verification_scope": "six pinned source files; custom configurations must be reviewed separately",
                "reachy_mapping": "not performed by source normalization",
            },
            "derived_fields": {"time_s": "Frame index / recorded env.action_frequency; no original timestamps supplied",
                               "*_pose": "Recorded homogeneous world transform converted to xyz+wxyz, no FK or pose estimation"},
            "simulation_assumptions": ["Achieved source poses are references, not Reachy states.",
                                       "Source assistance and reset state replay are not allowed in target dynamics validation.",
                                       "Source gripper commands have not been converted to Reachy pad gap, angle, or force."],
            "canonical_units": {"length": "m", "angle": "rad", "time": "s", "quaternion": "wxyz"},
            "missing": missing,
        }
    return {"arrays": arrays, "metadata": metadata, "status": "normalized_source_reference", "missing_fields": missing}
