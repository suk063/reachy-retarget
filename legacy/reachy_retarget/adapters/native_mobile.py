"""Archive already acquired RoboCasa/BiGym pilots without network or RGB reads.

This boundary retains observed native values and task-only object references.
A source archive is not a Reachy rollout or a successful physics validation.
"""

import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

from ..mobile_pilot import SOURCES, safe_member


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def _provenance(root, source):
    root = Path(root).resolve()
    folder = root / "data/raw" / (source + "_native_pilot")
    manifest = _load_json(root / "catalog" / (source + "-pilot.json"))
    spec = SOURCES[source]
    if manifest.get("url") != spec["url"] or manifest.get("code_revision") != spec["code_revision"]:
        raise ValueError("Pilot manifest source URL/revision differs from pinned acquisition contract")
    records = []
    for row in manifest.get("downloaded", []):
        relative = safe_member(row["path"])
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Pilot source member resolves outside its raw directory")
        if not path.is_file() or path.stat().st_size != row["size"] or _sha(path) != row["sha256"]:
            raise ValueError("Pilot source checksum/length mismatch: " + str(relative))
        records.append({"path": str(path), "source_member": str(relative), "bytes": row["size"],
                        "sha256": row["sha256"], "source_url": row.get("source_url", manifest["url"] + "#member=" + str(relative))})
    if not records:
        raise ValueError("No acquired member checksums in pilot manifest")
    return folder, manifest, records


def _require_recorded(path, records):
    if str(Path(path).resolve()) not in {r["path"] for r in records}:
        raise ValueError("Source member has no acquisition provenance: " + str(path))


def _joint_layout(xml):
    """Validate the expanded MJCF traversal used by time+qpos+qvel records."""
    tree = ET.fromstring(xml)
    if tree.tag != "mujoco" or tree.find("worldbody") is None:
        raise ValueError("Expected expanded MuJoCo worldbody")
    if any(tree.find(".//" + tag) is not None for tag in ("include", "attach", "replicate", "composite", "flexcomp", "frame")):
        raise ValueError("Unexpanded model needs a compiled source adapter")
    if any(j.get("type", "hinge") != "hinge" for j in tree.findall(".//default/joint")):
        raise ValueError("Joint-type defaults need a compiled source adapter")
    if any(c.get("alignfree") == "true" for c in tree.findall("compiler")) or any(j.get("align") == "true" for j in tree.findall(".//freejoint")):
        raise ValueError("Aligned free-joint coordinates require a compiled source adapter")
    nq = nv = 0
    entries = []
    names = set()
    world = tree.find("worldbody")

    def visit(body, top, parent):
        nonlocal nq, nv
        joints = [j for j in body if j.tag in ("joint", "freejoint")]
        for joint in joints:
            name = joint.get("name")
            kind = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
            if not name or name in names or kind not in ("free", "ball", "hinge", "slide"):
                raise ValueError("Missing, duplicate or unknown native joint")
            names.add(name)
            if kind == "free" and (parent is not world or len(joints) != 1):
                raise ValueError("Free object must have a single joint directly under worldbody")
            qs, vs = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}[kind]
            entries.append({"name": name, "type": kind, "qpos_address": nq, "qvel_address": nv,
                            "nq": qs, "nv": vs, "body": body.get("name"), "top_body": top})
            nq += qs
            nv += vs
        for child in body.findall("body"):
            visit(child, top, body)

    for body in world.findall("body"):
        visit(body, body.get("name"), world)
    return tree, entries, nq, nv


def _metadata(source, manifest, records, sequence):
    spec = SOURCES[source]
    return {
        "source_id": source + "_native_pilot", "source_format": source + "-native-pilot",
        "source_sequence": sequence, "source_group": source + "/" + sequence,
        "source_urls": sorted({r["source_url"] for r in records} | {spec["url"], spec["registry_url"]}),
        "source_revision": spec["code_revision"], "source_release": spec["release"],
        "provenance": records, "source_bytes": sum(r["bytes"] for r in records),
        "whole_archive_checksum_verified": manifest.get("whole_archive_checksum_verified", False),
        "derived_fields": {}, "simulation_assumptions": [], "physics_validated": False,
        "retargeted": False, "rgb_saved": False,
    }


def _robocasa(root):
    folder, manifest, records = _provenance(root, "robocasa")
    lerobot = folder / "lerobot"
    episode = lerobot / "extras/episode_000000"
    paths = [episode / "states.npz", episode / "model.xml.gz", episode / "ep_meta.json", lerobot / "extras/dataset_meta.json"]
    for path in paths:
        _require_recorded(path, records)
    ep = _load_json(paths[2])
    dataset = _load_json(paths[3])
    if dataset.get("env") != "PickPlaceCounterToCabinet":
        raise ValueError("Unverified RoboCasa pilot task selection")
    xml = gzip.decompress(paths[1].read_bytes()).decode("utf-8")
    tree, joints, nq, nv = _joint_layout(xml)
    with np.load(paths[0], allow_pickle=False) as packed:
        if packed.files != ["states"]:
            raise ValueError("Unverified RoboCasa NPZ fields")
        states = packed["states"]
    if states.ndim != 2 or states.shape[1] != 1 + nq + nv or len(states) < 2 or not np.isfinite(states).all():
        raise ValueError("RoboCasa state width must match time+nq+nv; no action/activation fields inferred")
    clock = states[:, 0]
    if not np.all(np.diff(clock) > 0):
        raise ValueError("Recorded simulator clock must be strictly increasing")
    times = clock - clock[0]
    arrays = {"timestamp": times, "time_s": times, "source/simulator_time_s": clock.copy(),
              "source/state": states.copy(), "source/frame_index": np.arange(len(states), dtype=np.int64)}
    candidates = [obj for obj in ep.get("object_cfgs", []) if obj.get("name") == "obj"]
    if len(candidates) != 1:
        raise ValueError("Pinned task requires one declared manipulated object named obj")
    primary = candidates[0]
    free = [joint for joint in joints if joint["type"] == "free" and joint["top_body"] == "obj_main"]
    if len(free) != 1:
        raise ValueError("Cannot resolve recorded task object free joint")
    start = 1 + free[0]["qpos_address"]
    pose = states[:, start:start + 7].copy()
    if not np.allclose(np.linalg.norm(pose[:, 3:], axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Recorded task object quaternion is not unit length")
    arrays["objects/obj/pose"] = pose
    objects = {"obj": {"role": "manipulated_object", "category": primary.get("info", {}).get("cat"),
                       "source_body": "obj_main", "source_joint": free[0]["name"],
                       "decoded_pose_available": True, "pose_frame": "world", "quaternion": "wxyz",
                       "pose_source": "recorded MuJoCo free-joint qpos", "source_config": primary}}
    missing = ["source task fixture world/link poses", "source EEF forward kinematics", "original source action array not acquired",
               "source assets and replay fidelity verification", "Reachy retargeting and physical validation"]
    for role, name in sorted(ep.get("fixture_refs", {}).items()):
        if role not in {"cab", "counter"} or not isinstance(name, str):
            continue
        selected = [joint for joint in joints if joint["top_body"] == name + "_main"]
        objects[name] = {"role": "target_cabinet" if role == "cab" else "source_support",
                         "source_body": name + "_main", "decoded_pose_available": False,
                         "pose_source": "embedded MJCF; compiled source FK required",
                         "joint_names": [joint["name"] for joint in selected]}
        if selected:
            if any(joint["nq"] != 1 for joint in selected):
                raise ValueError("Task fixture articulation needs non-scalar joint adapter")
            arrays["objects/" + name + "/joint_position"] = states[:, [1 + joint["qpos_address"] for joint in selected]].copy()
    robot_joints = [joint for joint in joints if joint["top_body"] == "robot0_base"]
    scalar_robot = bool(robot_joints) and all(joint["nq"] == joint["nv"] == 1 for joint in robot_joints)
    if scalar_robot:
        arrays["source/joint_position"] = states[:, [1 + joint["qpos_address"] for joint in robot_joints]].copy()
        arrays["source/joint_velocity"] = states[:, [1 + nq + joint["qvel_address"] for joint in robot_joints]].copy()
    metadata = _metadata("robocasa", manifest, records, "PickPlaceCounterToCabinet/20250819/episode_000000")
    metadata.update(scenario=dataset["env"], instruction=ep.get("lang"), objects=objects,
                    source_configuration=dataset, source_episode_metadata=ep,
                    source_model_xml=xml, source_model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                    source_joint_names=[joint["name"] for joint in robot_joints] if scalar_robot else [],
                    source_joint_layout={"nq": nq, "nv": nv, "layout": "time,qpos,qvel"},
                    object_coverage={"scope": "manipulated obj plus explicit source/target fixture_refs; distractors excluded",
                                     "task_object_names": sorted(objects), "decoded_object_pose_names": ["obj"],
                                     "missing_task_object_pose_names": sorted(set(objects) - {"obj"}),
                                     "all_task_object_poses_decoded": len(objects) == 1,
                                     "excluded_distractor_names": [o["name"] for o in ep.get("object_cfgs", []) if o.get("name", "").startswith("distr_")]},
                    state_control_coverage={"source_named_robot_joint_positions": scalar_robot,
                                            "source_named_robot_joint_velocities": scalar_robot,
                                            "source_actions_available": False, "source_eef_poses_decoded": False,
                                            "source_base_world_pose_decoded": False,
                                            "reachy_state_or_control": "absent"}, missing=missing)
    metadata["derived_fields"] = {"timestamp": "Recorded MuJoCo simulator time minus its first value; irregular gaps preserved",
                                   "objects/obj/pose": "Exact qpos slice from expanded named free-joint traversal; no interpolation or simulation",
                                   "source/joint_position": "Recorded named scalar robot joints from expanded MJCF traversal",
                                   "source/joint_velocity": "Recorded named scalar robot velocities from expanded MJCF traversal"}
    metadata["simulation_assumptions"] = ["State layout width checked against expanded XML; source simulator/assets not executed",
                                           "Source states are references, not Reachy measured states or physical success"]
    return {"arrays": arrays, "metadata": metadata, "status": "blocked_source_fk_assets_and_actions", "missing_fields": missing}


def _safetensors(path):
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError("Truncated safetensors file")
    header_size = struct.unpack("<Q", raw[:8])[0]
    if header_size > len(raw) - 8:
        raise ValueError("Invalid safetensors header length")
    header = json.loads(raw[8:8 + header_size])
    metadata = {key: json.loads(value) for key, value in header.get("__metadata__", {}).items()}
    payload = memoryview(raw)[8 + header_size:]
    tensors = {}
    intervals = []
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        dtype = {"F64": "<f8", "F32": "<f4", "I64": "<i8", "I32": "<i4", "BOOL": "?"}.get(spec.get("dtype"))
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if dtype is None or not isinstance(shape, list) or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("Unsupported safetensors dtype or shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(n, int) for n in offsets):
            raise ValueError("Invalid safetensors offsets")
        start, end = offsets
        count = math.prod(shape)
        if start < 0 or end < start or end > len(payload) or end - start != count * np.dtype(dtype).itemsize:
            raise ValueError("Safetensors extent differs from declared shape")
        if end > start:
            intervals.append((start, end))
        tensors[key] = np.frombuffer(payload[start:end], dtype=dtype).reshape(shape).copy()
    if any(left[1] > right[0] for left, right in zip(sorted(intervals), sorted(intervals)[1:])):
        raise ValueError("Overlapping safetensors payloads")
    return metadata, tensors


def _bigym(root):
    folder, manifest, records = _provenance(root, "bigym")
    files = sorted(folder.rglob("*.safetensors"))
    if len(files) != 1:
        raise ValueError("Expected one explicitly acquired BiGym pilot")
    _require_recorded(files[0], records)
    native, tensors = _safetensors(files[0])
    env = native.get("environment_data", {})
    if env.get("env_name") != "MovePlate" or env.get("action_mode_name") != "JointPositionActionMode" or env.get("action_mode_absolute") is not True:
        raise ValueError("Unverified BiGym pilot task/control mode")
    action = tensors.get("info_demo_action")
    if action is None or action.ndim != 2 or action.shape[1] != 15 or len(action) < 2 or not np.isfinite(action).all():
        raise ValueError("Invalid recorded BiGym action vectors")
    arrays = {"source/action": action, "source/frame_index": np.arange(len(action), dtype=np.int64)}
    for key in ("termination", "truncation", "reward"):
        if key in tensors:
            value = tensors[key]
            if value.shape not in ((len(action),), (0,)) or not np.isfinite(value).all():
                raise ValueError("Invalid source transition field: " + key)
            arrays["source/" + key] = value
    missing = ["timestamp", "source control frequency not recorded in acquired pilot", "source robot/object/base poses require pinned action replay",
               "source named joint position/velocity and action-channel mapping", "source simulator/assets",
               "Reachy retargeting and physical validation"]
    metadata = _metadata("bigym", manifest, records, "MovePlate/" + str(native.get("uuid", files[0].stem)))
    metadata.update(scenario="MovePlate", source_configuration=native,
                    source_environment=env, source_package_versions=native.get("package_versions", {}),
                    objects={"plate": {"role": "task_manipulated_plate", "decoded_pose_available": False,
                                       "native_body_name": None, "pose_source": "not present in lightweight sample; source reset/action replay required"}},
                    object_coverage={"scope": "MovePlate manipulated plate only; no unrelated scene-object poses stored",
                                     "task_object_names": ["plate"], "decoded_object_pose_names": [],
                                     "missing_task_object_pose_names": ["plate"], "all_task_object_poses_decoded": False},
                    state_control_coverage={"source_actions_available": True, "source_action_quantity": "mixed native actions: pelvis target increments, absolute arm targets and gripper commands; not observed state",
                                            "source_named_robot_joint_states": False, "source_eef_poses_decoded": False,
                                            "source_base_world_pose_decoded": False, "source_neck_state_or_control_decoded": False,
                                            "reachy_state_or_control": "absent"},
                    timing={"status": "untimed", "frames": len(action), "frame_index_is_timestamp": False,
                            "reason": "The acquired lightweight header has neither a timestamp tensor nor a control frequency; no rate is inferred"},
                    missing=missing)
    metadata["simulation_assumptions"] = ["Reset seed and native mixed-action vectors are preserved without importing BiGym",
                                           "No control rate, measured state, object pose or Reachy action is inferred"]
    return {"arrays": arrays, "metadata": metadata, "status": "blocked_source_replay_and_timestamp", "missing_fields": missing}


def normalize(root, source):
    """Return the common archive record for an already acquired native pilot."""
    if source == "robocasa":
        return _robocasa(root)
    if source == "bigym":
        return _bigym(root)
    raise ValueError("Supported native pilots are robocasa and bigym")
