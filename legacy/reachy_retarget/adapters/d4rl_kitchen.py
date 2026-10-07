"""Explicit acquisition and offline state reconstruction of historical D4RL kitchen.

Dataset downloads occur only in fetch(). Source Python is evidence, never
imported: no gym, dm_control, mujoco_py, robot SDK, renderer or robot connection.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time

import h5py
import numpy as np


SOURCE_REVISION = "89141a689b0353b0dac3da5cba60da4b1b16254d"
SOURCE_BASE = "https://raw.githubusercontent.com/Farama-Foundation/D4RL/" + SOURCE_REVISION + "/"
DATA_BASE = "https://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/"
DATASETS = {
    "complete": {"filename": "mini_kitchen_microwave_kettle_light_slider-v0.hdf5", "bytes": 556544,
                 "sha256": "cd797c38cd52dfbe3f960cef73935cc724847abd189e47a703a7a2315ed60a81",
                 "tasks": ["microwave", "kettle", "light switch", "slide cabinet"]},
    "partial": {"filename": "kitchen_microwave_kettle_light_slider-v0.hdf5", "bytes": 19558459,
                "sha256": "57bfd9c5fe88a7cf702bbaf8b7cd800d30cd869e7f73416007238d26cd859800",
                "tasks": ["microwave", "kettle", "light switch", "slide cabinet"]},
    "mixed": {"filename": "kitchen_microwave_kettle_bottomburner_light-v0.hdf5", "bytes": 19560760,
              "sha256": "a30382a50278fdaea2e2d88e5f2836a95bcf9b581d252617872fd6537302721b",
              "tasks": ["microwave", "kettle", "bottom burner", "light switch"]},
}
MODEL_PATH = "d4rl/kitchen/adept_envs/franka/assets/franka_kitchen_jntpos_act_ab.xml"
RESERVE_BYTES = 50_000_000_000


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(cluster_root):
    """Acquire exact historical files and pinned model assets under one shared lock.

    No runtime/pod changes. Existing valid originals are reused; mismatches are
    rejected, never replaced. Failed partial downloads are retained as attempts.
    """
    import fcntl
    import requests

    cluster_root = Path(cluster_root)
    root = cluster_root / "native/d4rl_kitchen"
    root.mkdir(parents=True, exist_ok=True)
    lockpath = cluster_root / "queue/transfer.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if shutil.disk_usage(cluster_root).free - 60_000_000 < RESERVE_BYTES:
            raise OSError("D4RL acquisition would violate 50 decimal GB reserve")
        tree_url = "https://api.github.com/repos/Farama-Foundation/D4RL/git/trees/" + SOURCE_REVISION + "?recursive=1"
        response = requests.get(tree_url, timeout=60)
        response.raise_for_status()
        if len(response.content) > 5_000_000:
            raise ValueError("Source tree exceeds bounded manifest size")
        tree = response.json()
        if tree.get("truncated"):
            raise ValueError("Source asset manifest is truncated")
        source = [row for row in tree["tree"] if row["type"] == "blob" and
                  (row["path"].startswith("d4rl/kitchen/") or row["path"] == "LICENSE")]
        if not source or len(source) > 200 or sum(row["size"] for row in source) > 20_000_000:
            raise ValueError("Pinned kitchen source exceeds 20 MB/200 file acquisition budget")
        records = []

        def acquire(path, url, expected_size, expected_sha=None, expected_git=None):
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                if shutil.disk_usage(cluster_root).free - expected_size < RESERVE_BYTES:
                    raise OSError("50 decimal GB reserve before D4RL transfer")
                partial = path.with_name(path.name + ".attempt-" + str(time.time_ns()))
                with requests.get(url, stream=True, timeout=60) as stream:
                    stream.raise_for_status()
                    total = 0
                    with partial.open("xb") as output:
                        for chunk in stream.iter_content(1024 * 1024):
                            if not chunk:
                                continue
                            total += len(chunk)
                            if total > expected_size or shutil.disk_usage(cluster_root).free - len(chunk) < RESERVE_BYTES:
                                raise OSError("Transfer exceeds pinned size or disk reserve")
                            output.write(chunk)
                verify(partial, expected_size, expected_sha, expected_git)
                partial.rename(path)
            digest = verify(path, expected_size, expected_sha, expected_git)
            records.append({"path": str(path.relative_to(root)), "url": url, "bytes": expected_size,
                            "sha256": digest, "git_blob_sha1": expected_git})

        def verify(path, expected_size, expected_sha, expected_git):
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            git = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if len(data) != expected_size or expected_sha and digest != expected_sha or expected_git and git != expected_git:
                raise ValueError("Pinned D4RL file checksum/size mismatch: " + str(path))
            return digest

        for spec in DATASETS.values():
            acquire(root / "raw" / spec["filename"], DATA_BASE + spec["filename"], spec["bytes"], spec["sha256"])
        for row in source:
            relative = PurePosixPath(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe source asset path")
            acquire(root / "source" / SOURCE_REVISION / str(relative), SOURCE_BASE + str(relative), row["size"], expected_git=row["sha"])
        manifest = {"dataset": "d4rl_kitchen", "source_revision": SOURCE_REVISION, "source_tree_url": tree_url,
                    "source_tree_response_sha256": hashlib.sha256(response.content).hexdigest(),
                    "files": records, "total_bytes": sum(row["bytes"] for row in records),
                    "physics_validated": False, "original_capture_count_assessed": False}
        target = root / "acquisition.json"
        if target.exists() and json.loads(target.read_text()) != manifest:
            raise ValueError("Existing immutable acquisition manifest differs")
        if not target.exists():
            target.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


TASKS = {
    "bottom burner": {"indices": [11, 12], "roots": ["knob 2", "Burner 2"], "site": "knob2_site"},
    "light switch": {"indices": [17, 18], "roots": ["lightswitchbaseroot", "lightblock_hinge"], "site": "light_site"},
    "slide cabinet": {"indices": [19], "roots": ["slidecabinet"], "site": "slide_site"},
    "microwave": {"indices": [22], "roots": ["microwave"], "site": "microhandle_site"},
    "kettle": {"indices": list(range(23, 30)), "roots": ["kettle"], "site": "kettle_site"},
}


def segments(terminals, timeouts):
    """Use only recorded terminal/timeout flags; retain an unclosed tail."""
    terminals, timeouts = np.asarray(terminals), np.asarray(timeouts)
    if terminals.ndim != 1 or terminals.shape != timeouts.shape or terminals.dtype.kind != "b" or timeouts.dtype.kind != "b":
        raise ValueError("Recorded boolean terminal/timeout arrays must align")
    result, start = [], 0
    for end in np.flatnonzero(terminals | timeouts):
        result.append({"start": start, "stop": int(end + 1), "terminal": bool(terminals[end]),
                       "timeout": bool(timeouts[end]), "closed_by_recorded_marker": True})
        start = int(end + 1)
    if start < len(terminals):
        result.append({"start": start, "stop": len(terminals), "terminal": False, "timeout": False,
                       "closed_by_recorded_marker": False})
    return result


def _manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "acquisition.json").read_text())
    if manifest.get("source_revision") != SOURCE_REVISION:
        raise ValueError("D4RL source revision differs from the pinned adapter")
    records = {}
    for row in manifest.get("files", []):
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
            raise ValueError("D4RL acquisition manifest checksum mismatch: " + str(path))
        if row.get("git_blob_sha1"):
            payload = path.read_bytes()
            if hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest() != row["git_blob_sha1"]:
                raise ValueError("Pinned source Git blob changed")
        records[str(path)] = row
    return manifest, records


def compile_model(root):
    """Compile full official geometry with recorded legacy XML syntax fixes.

    No source file is edited. Named defaults are nested below one unnamed main
    default; include contents and compiler attributes keep their source order.
    """
    import ast
    import xml.etree.ElementTree as ET
    import mujoco

    root = Path(root).resolve()
    _, records = _manifest(root)
    source = root / "source" / SOURCE_REVISION
    model_path = source / MODEL_PATH
    dependencies = []

    def expand(path, active=()):
        path = path.resolve()
        if str(path) not in records or path in active:
            raise ValueError("Unverified or cyclic model include: " + str(path))
        dependencies.append(str(path))
        tree = ET.parse(path).getroot()
        def visit(parent):
            for index, node in reversed(list(enumerate(parent))):
                if node.tag == "include":
                    included = expand(path.parent / node.attrib["file"], (*active, path))
                    parent.remove(node)
                    for child in reversed(list(included)):
                        parent.insert(index, child)
                else:
                    visit(node)
        visit(tree)
        return tree

    tree = expand(model_path)
    main = ET.Element("default")
    for node in tree.findall("default"):
        tree.remove(node)
        if node.get("class"):
            main.append(node)
        else:
            # Actual pinned files use unnamed wrappers around named classes.
            if any(child.tag != "default" for child in node):
                raise ValueError("Unverified global default merge semantics")
            main.extend(list(node))
    tree.insert(0, main)
    compiler = {}
    for node in tree.findall("compiler"):
        compiler.update(node.attrib)
        tree.remove(node)
    for key in ("meshdir", "texturedir"):
        if key in compiler:
            compiler[key] = str((model_path.parent / compiler[key]).resolve())
    ET.SubElement(tree, "compiler", compiler)
    # Resolve and verify every mesh/texture file before asking MuJoCo to read it.
    asset_paths = []
    for element in tree.findall("asset/mesh") + tree.findall("asset/texture") + tree.findall("asset/hfield"):
        if "file" not in element.attrib:
            continue
        directory = compiler.get("texturedir" if element.tag == "texture" else "meshdir", str(model_path.parent))
        path = (Path(directory) / element.attrib["file"]).resolve()
        if str(path) not in records:
            raise ValueError("Model references an asset outside the verified source manifest")
        asset_paths.append(str(path))
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    if (model.nq, model.nv) != (30, 29):
        raise ValueError("D4RL model does not match 30 qpos /29 qvel state schema")
    code_path = source / "d4rl/kitchen/adept_envs/franka/kitchen_multitask_v0.py"
    if str(code_path) not in records:
        raise ValueError("Missing pinned source timing implementation")
    code = ast.parse(code_path.read_text())
    klass = next(node for node in code.body if isinstance(node, ast.ClassDef) and node.name == "KitchenV0")
    init = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    defaults = dict(zip([arg.arg for arg in init.args.args][-len(init.args.defaults):], init.args.defaults))
    frame_skip = ast.literal_eval(defaults["frame_skip"])
    if frame_skip != 40 or not np.isclose(model.opt.timestep, .002):
        raise ValueError("Pinned nominal source control period changed")
    return model, {"source_model_path": str(model_path), "compatible_xml": xml,
                   "compatible_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                   "xml_dependencies": dependencies, "geometry_asset_paths": sorted(set(asset_paths)),
                   "compatibility_changes": ["Expand verified includes without changing their order",
                       "Nest legacy top-level named default classes below the unnamed main default",
                       "Merge compiler attributes in source order and resolve original asset directories"],
                   "nominal_control_period_s": float(frame_skip * model.opt.timestep),
                   "nominal_frame_skip": frame_skip, "nominal_physics_timestep_s": float(model.opt.timestep),
                   "timing_evidence_url": SOURCE_BASE + str(code_path.relative_to(source)),
                   "runtime_mujoco_version": mujoco.__version__, "original_collection_runtime_version": None}


def inspect(root, variant="complete"):
    """Inspect saved markers, never guess episodes from a horizon or label ID."""
    root, spec = Path(root), DATASETS[variant]
    path = root / "raw" / spec["filename"]
    if not path.is_file() or path.stat().st_size != spec["bytes"] or _sha(path) != spec["sha256"]:
        raise ValueError("Historical D4RL raw checksum/size mismatch")
    with h5py.File(path, "r") as f:
        required = {"observations", "actions", "rewards", "terminals", "timeouts", "infos"}
        if set(f) != required or f["observations"].ndim != 2 or f["observations"].shape[1] != 60:
            raise ValueError("Unsupported historical kitchen field/schema")
        count = len(f["observations"])
        if f["actions"].shape != (count, 9) or any(f[key].shape != (count,) for key in required - {"observations", "actions"}):
            raise ValueError("Historical source rows are not aligned")
        bounds = segments(f["terminals"][()], f["timeouts"][()])
        return {"dataset": "d4rl_kitchen", "variant": variant, "path": str(path), "sha256": spec["sha256"],
                "rows": count, "segments": bounds, "segment_count": len(bounds),
                "source_label_count": len(np.unique(f["infos"][()])),
                "source_labels_are_unique_demonstration_ids": False,
                "recorded_timestamps": False, "exact_simulator_qpos": False, "recorded_qvel": False,
                "physics_validated": False}


def normalize(root, variant="complete", episode=0, *, nominal_timing=False):
    """FK estimates from noisy recorded configuration; no dynamics execution.

    Default output is untimed. ``nominal_timing=True`` explicitly chooses the
    pinned environment's 80 ms control period as a reconstruction assumption,
    not recovered timestamp evidence. Native indices/observations remain intact.
    """
    import mujoco
    from scipy.spatial.transform import Rotation

    root = Path(root).resolve()
    info = inspect(root, variant)
    if not isinstance(episode, int) or episode < 0 or episode >= len(info["segments"]):
        raise ValueError("Episode index is outside recorded terminal/timeout segments")
    model, identity = compile_model(root)
    manifest = json.loads((root / "acquisition.json").read_text())
    selection = info["segments"][episode]
    start, stop = selection["start"], selection["stop"]
    with h5py.File(info["path"], "r") as f:
        native = {key: f[key][start:stop] for key in f}
    if any(value.dtype.kind not in "biuf" or not np.isfinite(value).all() for value in native.values()):
        raise ValueError("Nonfinite/nonnumeric native source values")
    observations = native["observations"]
    qpos = observations[:, :30].astype(float).copy()
    norms = np.linalg.norm(qpos[:, 26:30], axis=1)
    if np.any(norms < .5) or np.any(norms > 1.5):
        raise ValueError("Kettle quaternion observation cannot be safely interpreted")
    qpos[:, 26:30] /= norms[:, None]
    arrays = {"source/" + key: value for key, value in native.items()}
    arrays.update({"source/frame_index": np.arange(start, stop, dtype=np.int64),
                   "source/joint_position": observations[:, :9].copy(),
                   "source/observed_configuration": observations[:, :30].copy(),
                   "source/goal": observations[:, 30:].copy(), "source/kettle_quaternion_original_norm": norms,
                   "source/nominal_time_s": np.arange(stop - start, dtype=float) * identity["nominal_control_period_s"]})
    if nominal_timing:
        arrays["timestamp"] = arrays["source/nominal_time_s"].copy()
    def body(name):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 1:
            raise ValueError("Missing task/robot body: " + name)
        return index
    def descendants(index):
        ids = {index}
        for child in range(index + 1, model.nbody):
            if int(model.body_parentid[child]) in ids:
                ids.add(child)
        return sorted(ids)
    def pose(pos, mat):
        return np.r_[pos, Rotation.from_matrix(np.asarray(mat).reshape(3, 3)).as_quat()[[3, 0, 1, 2]]]
    selected, objects = {}, {}
    for task in DATASETS[variant]["tasks"]:
        spec, label = TASKS[task], task.replace(" ", "_")
        roots = [body(name) for name in spec["roots"]]
        ids = sorted(set().union(*(set(descendants(index)) for index in roots)))
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec["site"])
        if site < 0 or int(model.site_bodyid[site]) not in ids:
            raise ValueError("Task interaction site is outside its declared bodies")
        selected[label] = (roots[0], ids, site)
        objects[label] = {"role": "benchmark_task_element", "native_task": task, "body_names": [model.body(i).name for i in ids],
                          "root_body_names": spec["roots"], "interaction_site": spec["site"],
                          "configuration_indices": spec["indices"], "pose_frame": "world", "quaternion": "wxyz",
                          "pose_source": "FK estimate from recorded noisy configuration; not exact simulator state"}
        arrays["objects/" + label + "/configuration"] = observations[:, spec["indices"]].copy()
        goal = observations[:, np.asarray(spec["indices"]) + 30]
        distance = np.linalg.norm(observations[:, spec["indices"]] - goal, axis=1)
        arrays["source/task_goal_distance/" + label] = distance
        arrays["source/task_goal_reached_observation/" + label] = distance < .3
    data, values = mujoco.MjData(model), {}
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    robot = descendants(body("panda0_link0"))
    if tcp < 0:
        raise ValueError("Missing source end-effector site")
    def append(key, value):
        values.setdefault(key, []).append(value)
    for row in qpos:
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        append("source/right_tcp_pose_estimate", pose(data.site_xpos[tcp], data.site_xmat[tcp]))
        append("source/base_pose", pose(data.xpos[robot[0]], data.xmat[robot[0]]))
        append("source/robot_body_poses_estimate", np.asarray([pose(data.xpos[i], data.xmat[i]) for i in robot]))
        for label, (anchor, ids, site) in selected.items():
            append("objects/" + label + "/pose", pose(data.xpos[anchor], data.xmat[anchor]))
            append("objects/" + label + "/interaction_pose", pose(data.site_xpos[site], data.site_xmat[site]))
            for index in ids:
                append("objects/" + label + "/parts/body_" + str(index) + "/pose", pose(data.xpos[index], data.xmat[index]))
    arrays.update({name: np.asarray(value) for name, value in values.items()})
    fingerprint = hashlib.sha256(np.ascontiguousarray(observations[:, :30]).tobytes() + native["actions"].tobytes()).hexdigest()
    missing = ["recorded simulator timestamps", "exact simulator qpos", "recorded joint/object qvel",
               "controller memory and physics-substep controls", "original collection simulator revision",
               "Reachy retargeting and physical validation"]
    if not nominal_timing:
        missing.insert(0, "timestamp")
    meta = {"source_id": "d4rl_kitchen", "source_format": "historical-d4rl-kitchen-hdf5", "source_revision": SOURCE_REVISION,
            "source_urls": [DATA_BASE + DATASETS[variant]["filename"], SOURCE_BASE + "d4rl/kitchen/kitchen_envs.py",
                            SOURCE_BASE + "d4rl/kitchen/__init__.py", SOURCE_BASE + MODEL_PATH],
            "source_sequence": variant + "/segment_" + str(episode), "source_group": "d4rl-kitchen/observed-segment/" + fingerprint,
            "source_corpus_group": "historical-d4rl-kitchen-demonstrations-overlapping-releases",
            "deduplication": "Exact noisy-configuration/action segment fingerprint only; overlapping crops and reordered releases are not independently counted",
            "independent_demonstration_count": None, "variant": variant, "source_row_interval": selection,
            "segmentation": "Recorded terminals OR timeouts only; infos labels are retained but reused and not unique capture IDs",
            "source_dataset_sha256": info["sha256"], "provenance": manifest["files"], "objects": objects,
            "task_object_scope": "Only benchmark task elements named by the pinned environment; unrelated fixtures omitted from direct poses",
            "source_robot_type": "fixed-base Franka Panda", "source_joint_names": [model.joint(i).name for i in range(9)],
            "source_robot_body_names": [model.body(i).name for i in robot], "source_model": identity,
            "state_semantics": "observations[:30] are noisy robot/object configuration; observations[30:] are task goals, not velocities",
            "action_semantics": "Source normalized 9D velocity requests; clip/scale, cached noisy robot observation and limits convert them to position actuator targets; raw applied controls are absent",
            "timing": {"measured": False, "nominal_period_s": identity["nominal_control_period_s"],
                       "nominal_timing_explicitly_selected": bool(nominal_timing),
                       "evidence": "Pinned KitchenV0 frame_skip40 and source XML timestep0.002; actual recorded timestamp/dropout information absent"},
            "derived_fields": {key: "FK of noisy recorded configuration; source poses are estimates" for key in values},
            "simulation_assumptions": ["No stepping or source action execution; qvel is not fabricated",
                "Normalize noisy kettle quaternion for FK only; preserve its original components/norm",
                "Current MuJoCo compiles syntax-compatible original assets; original collection simulator version is unknown"],
            "missing_fields": missing, "rgb_stored": False, "physics_validated": False, "retargeted": False,
            "policy_ready": False, "object_state_assignment": "Isolated FK reconstruction of recorded noisy reference; never physical validation"}
    meta["derived_fields"].update({"source/task_goal_distance/*": "Euclidean distance to recorded task goal at observation time",
                                    "source/task_goal_reached_observation/*": "Derived observation predicate distance<0.3; not recorded physical/task success",
                                    "source/nominal_time_s": "Frame index times documented nominal80ms source control period; not measured time"})
    if nominal_timing:
        meta["derived_fields"]["timestamp"] = "Explicit nominal-time reconstruction using pinned environment80ms period; measured timing remains missing"
        meta["simulation_assumptions"].append("Nominal uniform timing selected explicitly; dropped/irregular source samples cannot be verified")
    return {"arrays": arrays, "metadata": meta, "status": "noisy_source_configuration_fk_estimated", "missing_fields": missing}


def ik_view(root, variant="complete", episode=0):
    """Explicit nominal-clock/source-TCP aliases for a separate Reachy IK stage.

    All source uncertainty remains. No world placement, IK, contact success or
    Reachy gripper command is created by this conversion.
    """
    record = normalize(root, variant, episode, nominal_timing=True)
    arrays, meta = record["arrays"], record["metadata"]
    arrays["time_s"] = arrays["timestamp"].copy()
    arrays["hand/right_pose"] = arrays["source/right_tcp_pose_estimate"].copy()
    arrays["source/robot_root_pose"] = arrays["source/base_pose"].copy()
    arrays["source/gripper_aperture_observed_m"] = arrays["source/joint_position"][:, 7:9].sum(axis=1)
    meta.update(robot_type="Panda", reference_profile="noisy-d4rl-source-fk-nominal-time-ik-v1",
                source_sequence=meta["source_sequence"] + "/nominal-time-ik-reference",
                gripper_reference="Sum of two noisy source finger-slide observations; no Reachy target or grasp-contact label inferred",
                reference_world_placement="None; original source world frame retained. Any downstream rigid placement must also transform all task objects.")
    meta["derived_fields"].update({"time_s": meta["derived_fields"]["timestamp"],
                                   "hand/right_pose": "Alias of noisy-configuration source end_effector site FK estimate",
                                   "source/robot_root_pose": "Alias of the fixed source Panda model base pose",
                                   "source/gripper_aperture_observed_m": "Sum of source finger joint7 and joint8 observations; noise and out-of-range values retained"})
    record["status"] = "nominal_time_noisy_source_ik_reference"
    return record
