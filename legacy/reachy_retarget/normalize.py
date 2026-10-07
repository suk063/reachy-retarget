"""Explicit adapters for documented source schemas."""

import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import h5py
from .episodes import write_episode
from .store import json_write, sha256


def lerobot(store, source, limit):
    root = store.root / "data/raw" / source
    info_path = next(
        (
            p
            for p in (root / "meta/info.json", root / "meta_data/info.json")
            if p.exists()
        ),
        None,
    )
    if not info_path:
        raise ValueError("No LeRobot info.json; schema needs manual adapter evidence")
    info = json.loads(info_path.read_text())
    features = info["features"]
    robot_type = str(info.get("robot_type", "")).lower()
    is_reachy = robot_type == "reachy2"
    state_key = "observation.state"
    if state_key not in features:
        raise ValueError("No state feature")
    feature = features[state_key]
    names = feature.get("names")
    if isinstance(names, dict):
        names = next(iter(names.values()))
    names = list(names) if names else None
    if names and len(names) != feature["shape"][0]:
        raise ValueError("Feature-name count contradicts state shape")
    # Source adapter convention: LeRobot Reachy2 reads SDK present_position,
    # explicitly documented as degrees, including the gripper motor angle.
    documented_reachy = (
        is_reachy
        and names
        and all(n.endswith(".pos") for n in names)
        and any("shoulder_pitch" in n for n in names)
    )
    if is_reachy and not documented_reachy:
        raise ValueError(
            "Reachy joint names/SDK convention unresolved; raw data retained"
        )
    buffers = {}
    provenance = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        schema = pq.read_schema(path)
        cols = [
            n
            for n in schema.names
            if n
            in (
                "observation.state",
                "action",
                "timestamp",
                "episode_index",
                "task_index",
                "frame_index",
            )
        ]
        if not {state_key, "timestamp", "episode_index"} <= set(cols):
            continue
        tab = pq.read_table(path, columns=cols).to_pydict()
        ep = np.array(tab["episode_index"])
        for i in np.unique(ep):
            if limit and len(buffers) >= limit and int(i) not in buffers:
                continue
            ix = np.flatnonzero(ep == i)
            buffers.setdefault(int(i), {k: [] for k in tab})
            for key, values in tab.items():
                buffers[int(i)][key].extend([values[j] for j in ix])
            provenance.setdefault(int(i), []).append(
                {"path": str(path.relative_to(store.root)), "sha256": sha256(path)}
            )
        if (
            limit
            and len(buffers) >= limit
            and info.get("codebase_version", "").startswith("v2")
        ):
            break
    results = []
    for i, tab in sorted(buffers.items()):
        order = np.argsort(tab["timestamp"])
        t = np.array(tab["timestamp"], float)[order]
        t -= t[0]
        state = np.array(tab[state_key])[order]
        if state.shape[1:] != tuple(feature["shape"]):
            raise ValueError("Actual state shape contradicts metadata")
        arrays = {"time_s": t, "source/state": state}
        if names:
            arrays["source/state_names"] = names
        if "action" in tab:
            arrays["source/action"] = np.array(tab["action"])[order]
        meta = {
            "source_format": info.get("codebase_version"),
            "robot_type": robot_type,
            "provenance": provenance[i],
            "fps": info.get("fps"),
            "objects": {},
            "missing": ["object_pose", "object_mesh", "contact", "base_pose"],
            "base_reference": "No measured world base pose in selected channels; FK uses a declared stationary reference base, not observed world motion.",
            "state_semantics": "source-native; see source feature schema",
            "source_features": features,
        }
        if documented_reachy:
            indices = [j for j, n in enumerate(names) if "gripper" not in n]
            arrays["robot/joint_names"] = [
                names[j].removesuffix(".pos") for j in indices
            ]
            arrays["robot/joint_position"] = np.deg2rad(state[:, indices])
            if "action" in tab and features["action"].get("names") == names:
                arrays["command/joint_position"] = np.deg2rad(
                    arrays["source/action"][:, indices]
                )
            gi = [j for j, n in enumerate(names) if "gripper" in n]
            if gi:
                arrays["source/gripper_motor_position_rad"] = np.deg2rad(state[:, gi])
                arrays["source/gripper_names"] = [names[j] for j in gi]
            meta.update(
                state_semantics="Reachy2 SDK joint positions converted degrees to URDF radians",
                unit_evidence="https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/reachy2/robot_reachy2.py",
                gripper_mapping="SDK motor radians preserved; not equated with finger joint angle",
            )
        from .scope import require_objects
        require_objects(arrays)
        results.append(write_episode(store, source, f"episode_{i:06d}", arrays, meta))
    return results


def robomimic(store, source, limit):
    root = store.root / "data/raw" / source
    paths = sorted(root.rglob("*.hdf5"))
    # Proficient-human can is the fixed contact benchmark, before other tasks.
    paths.sort(key=lambda p: ("can/ph" not in p.as_posix(), str(p)))
    results = []
    for path in paths:
        raw_hash = sha256(path)
        relative = path.relative_to(root)
        sequence_base = relative.parent if path.name == "low_dim_v15.hdf5" else relative.with_suffix("")
        with h5py.File(path, "r") as f:
            env = json.loads(f["data"].attrs.get("env_args", "{}"))
            if env.get("env_version") not in ("1.5.1", "1.4.1"):
                raise ValueError("Unverified robosuite state version")
            fps = env.get("env_kwargs", {}).get("control_freq")
            if not fps or fps <= 0:
                raise ValueError("Missing documented control frequency")
            demos = sorted(f["data"], key=lambda k: int(k.rsplit("_", 1)[-1]))
            for name in demos[: limit or None]:
                g = f["data"][name]
                n = len(g["actions"])
                obs = g["obs"]
                arrays = {
                    "time_s": np.arange(n) / fps,
                    "source/action": g["actions"][()],
                    "source/simulator_state": g["states"][()],
                }
                for k in obs:
                    if obs[k].ndim < 3:
                        arrays["source/observation/" + k] = obs[k][()]
                if {"robot0_eef_pos", "robot0_eef_quat"} <= set(obs):
                    arrays["hand/right_pose"] = np.c_[
                        obs["robot0_eef_pos"][()],
                        obs["robot0_eef_quat"][()][:, [3, 0, 1, 2]],
                    ]
                if {"robot1_eef_pos", "robot1_eef_quat"} <= set(obs):
                    arrays["hand/left_pose"] = np.c_[obs["robot1_eef_pos"][()], obs["robot1_eef_quat"][()][:, [3, 0, 1, 2]]]
                model = g.attrs.get("model_file", "")
                from .simstate import free_object_poses
                object_arrays, objects, object_evidence = free_object_poses(model, arrays["source/simulator_state"], env["env_name"])
                arrays.update(object_arrays)
                # robosuite 1.5.1 _create_obj_sensors puts relative EEF sensors
                # first, followed by world object xyz and xyzw (last seven).
                if "can" in path.parts and "object" in obs:
                    obj = obs["object"][()]
                    if env.get("env_version") != "1.5.1" or obj.shape[1] != 14:
                        raise ValueError(
                            "Unverified PickPlaceCan object observation ordering"
                        )
                    sensor_pose = np.c_[
                        obj[:, 7:10], obj[:, 10:14][:, [3, 0, 1, 2]]
                    ]
                    np.testing.assert_allclose(arrays["objects/Can/pose"], sensor_pose, atol=1e-6)
                    objects["Can"]["asset"] = "robosuite/models/assets/objects/can.xml"
                    object_evidence["independent_can_sensor_check"] = True
                if "robot0_gripper_qpos" in obs:
                    arrays["source/gripper_qpos"] = obs["robot0_gripper_qpos"][()]
                meta = {
                    "source_format": "robomimic-hdf5",
                    "robot_type": "Panda",
                    "env_args": env,
                    "objects": objects,
                    "object_state_decoding": object_evidence,
                    "model_xml": model,
                    "fps": fps,
                    "source_group": source
                    + "/"
                    + str(sequence_base)
                    + "/"
                    + name,
                    "provenance": [
                        {"path": str(path.relative_to(store.root)), "sha256": raw_hash}
                    ],
                    "hand_assignment": "robot0 to Reachy right; robot1 to Reachy left when present; shared world frame",
                    "quaternion_evidence": "robosuite observables use xyzw; converted to wxyz",
                    "action_semantics": "source OSC command, not a Reachy joint command",
                }
                seq = str(sequence_base) + "/" + name
                results.append(write_episode(store, source, seq, arrays, meta))
    if not results:
        raise ValueError("No supported downloaded robosuite HDF5")
    return results


def run(store, sources=None, limit=5):
    sources = sources or [
        "hf__robomimic__robomimic_datasets",
        "parahome",
        "humoto",
    ]
    report = []
    for source in sources:
        try:
            from .scope import EXCLUDED_SOURCES
            if source in EXCLUDED_SOURCES:
                raise ValueError("Excluded objectless selection: " + EXCLUDED_SOURCES[source])
            if source == "humoto":
                from .human import humoto

                paths = humoto(store, source, limit)
            elif source == "parahome":
                from .human import parahome

                paths = parahome(store, source, limit)
            elif "robomimic" in source or "mimicgen" in source:
                paths = robomimic(store, source, limit)
            elif source == "hf__haosulab__ManiSkill_Demonstrations":
                from .maniskill import maniskill
                paths = maniskill(store, source, limit)
            else:
                paths = lerobot(store, source, limit)
            row = {"source": source, "status": "normalized", "episodes": len(paths)}
        except Exception as e:
            row = {
                "source": source,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            }
        report.append(row)
        print(row, flush=True)
    json_write(store.root / "runs/normalization.json", report)
    return report
