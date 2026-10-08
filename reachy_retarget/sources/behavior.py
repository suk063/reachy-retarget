"""Adapter for the BEHAVIOR-1K 2025 challenge demonstrations (``behavior-1k/2025-challenge-demos``).

10,000 human teleoperation episodes (JoyLo) of 50 household tasks on a Galaxea R1 Pro
(holonomic base, 4-DoF torso, two 7-DoF arms, parallel grippers) in OmniGibson / Isaac
Sim. The LeRobot-v2.1 layout is used as published:

* ``data/task-XXXX/episode_N.parquet``: one row per frame at 30 Hz with ``timestamp``,
  ``observation.state`` (256-d proprioception), ``observation.task_info`` (privileged
  low-dim task observation), ``action`` (23-d) and ``observation.cam_rel_poses``. The
  observations were produced by replaying the recorded sim state every frame without
  stepping physics (joint efforts are invalid; poses are exact).
* ``meta/episodes/task-XXXX/episode_N.json``: the OmniGibson ``config`` and
  ``scene_file`` (versions, BDDL ``inst_to_name``, object init args) and
  ``task_obs_keys`` (the key order of ``observation.task_info``).
* ``annotations/task-XXXX/episode_N.json``: skill segments (frames, objects).
* optional ``2025-challenge-rawdata/task-XXXX/episode_N.hdf5``: the raw recording of the
  same episode; only ``terminated`` / ``reward`` are read (success).

``observation.state`` layout = ``PROPRIOCEPTION_INDICES["R1Pro"]`` of
``omnigibson/learning/utils/eval_utils.py`` (see :data:`STATE`). ``eef_{side}_pos/quat``
is the pose of ``{side}_eef_link`` relative to ``base_link`` (quaternion x, y, z, w);
``robot_pos`` / ``robot_ori_{cos,sin}`` is the world pose of ``base_link`` (Euler
roll-pitch-yaw from ``T.quat2euler``). ``observation.task_info`` concatenates, per BDDL
instance in ``task_obs_keys`` order, ``_real`` (1), ``_pos`` (3, world), ``_ori_cos`` /
``_ori_sin`` (3 + 3, roll-pitch-yaw of the object root link) and, except for the agent,
``_in_gripper_left`` / ``_in_gripper_right`` (1 + 1; ``IsGraspingState``: 1 true,
-1 false, 0 unknown). In the release these flags are -1 on every frame (the replay does
not re-create the assisted-grasp constraint), so carrying is a derived label
(:func:`carry_intervals`).

Grasp center: the recorded ``{side}_eef_link`` pose. In OmniGibson this frame is
``{side}_gripper_link * (0, 0, -0.06), Ry(pi)`` (``r1pro_source_cfg.yaml``); its +z points
along the fingers, its y axis is the finger articulation axis, and the assisted-grasp
points are placed symmetrically about its origin (``manipulation_robot.py``), so its
origin is the point between the pads. The contract frame keeps +z and flips x and y so
that +y points from finger 1 (``*_gripper_finger_link1``, on the eef +y side) to finger
2: ``R_contract = R_eef @ diag(-1, -1, 1)``. Width = ``q_finger1 + q_finger2``
(0-0.10 m; the finger inner faces meet at q = 0), opening = width / 0.10.

The recorded eef poses are checked per episode against forward kinematics of the pinned
``r1pro.urdf`` (``provenance["fk_check"]``) when that URDF is available; the URDF also
gives ``torso_height`` (world z of ``torso_link4``, the arm and head mount).

World frame: the OmniGibson world translated along z so that the floor the robot stands
on is at z = 0. The R1 Pro ``base_link`` origin is the bottom of the robot
(``misc/metadata.json``: ``base_link_offset`` z = half the bounding-box height), so the
offset is minus the median ``base_link`` world z of the episode (a few mm on the ground
floor; houses with two floors would differ), recorded as ``provenance["world_z_offset_m"]``.

No ``SceneRef``: the scene is an OmniGibson/Isaac Sim house from the BEHAVIOR Data
Bundle (encrypted USD assets under the BEHAVIOR license), not a MuJoCo model.
Articulated scene joints (cabinet, fridge, microwave doors) are not in the low-dim
release; they exist only in the serialized raw sim state, which is not decoded here.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..acquire import load_catalog, locate_entry, sha256_file
from ..schema.source import Effector, ObjectTrack, SourceEpisode
from .registry import register

OG_RECORDING_COMMIT = "875b51866ac8fdd27f9ed51ad43011fc10c16bd0"  # scene_file versions.omnigibson.git_hash
OG_LAYOUT_TAG = "v3.7.2"
OG_LAYOUT_COMMIT = "88454bd04f75dc57c00ab1f1a00bcde1ff505950"
_GH = "https://github.com/StanfordVL/BEHAVIOR-1K/blob/"
EVIDENCE = {
    "state_layout": _GH + OG_LAYOUT_COMMIT + "/OmniGibson/omnigibson/learning/utils/eval_utils.py#L59-L120",
    "proprioception": _GH + OG_RECORDING_COMMIT + "/OmniGibson/omnigibson/robots/manipulation_robot.py#L640-L662",
    "base_pose": _GH + OG_RECORDING_COMMIT + "/OmniGibson/omnigibson/robots/robot_base.py#L331-L370",
    "relative_poses": _GH + OG_RECORDING_COMMIT + "/OmniGibson/omnigibson/utils/usd_utils.py#L931-L938",
    "task_obs": _GH + OG_RECORDING_COMMIT + "/OmniGibson/omnigibson/tasks/behavior_task.py#L428-L469",
    "quat2euler": _GH + OG_RECORDING_COMMIT + "/OmniGibson/omnigibson/utils/transform_utils.py#L677-L702",
    "eef_convention": _GH + OG_RECORDING_COMMIT + "/OmniGibson/omnigibson/robots/manipulation_robot.py#L262-L420",
    "r1pro_links": _GH + OG_RECORDING_COMMIT + "/OmniGibson/omnigibson/robots/r1pro.py",
    "dataset_doc": _GH + OG_LAYOUT_COMMIT + "/docs/challenge/dataset.md",
}
LICENSE = "MIT (2025-challenge-demos dataset card); scene/object assets: BEHAVIOR Data Bundle license"

# observation.state slices (PROPRIOCEPTION_INDICES["R1Pro"]).
STATE = {
    "joint_qpos": (0, 28), "robot_pos": (140, 143), "robot_ori_cos": (143, 146), "robot_ori_sin": (146, 149),
    "robot_2d_ori": (149, 150), "arm_left_qpos": (158, 165), "eef_left_pos": (186, 189), "eef_left_quat": (189, 193),
    "gripper_left_qpos": (193, 195), "arm_right_qpos": (197, 204), "eef_right_pos": (225, 228),
    "eef_right_quat": (228, 232), "gripper_right_qpos": (232, 234), "trunk_qpos": (236, 240), "base_qpos": (244, 247),
}
STATE_DIM = 256
TASK_KEY_DIMS = {"real": 1, "pos": 3, "ori_cos": 3, "ori_sin": 3, "in_gripper_left": 1, "in_gripper_right": 1}
SIDES = ("left", "right")
EEF_TO_CONTRACT = np.diag([-1.0, -1.0, 1.0])
EEF_IN_GRIPPER = ((0.0, 0.0, -0.06), (0.0, 1.0, 0.0, 0.0))  # r1pro_source_cfg.yaml eef_vis_links (xyz, quat xyzw)
WIDTH_MAX = 0.10
TORSO = [f"torso_joint{i}" for i in range(1, 5)]
ARM = {s: [f"{s}_arm_joint{i}" for i in range(1, 8)] for s in SIDES}
FINGERS = {s: [f"{s}_gripper_finger_joint{i}" for i in (1, 2)] for s in SIDES}
URDF_PATH = "omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"
PLACE_IN = ("place in", "pour", "insert")
PLACE_ON = ("place on", "place under", "pick up from", "hang", "attach", "push to", "sweep surface", "wipe hard")


# ---------------------------------------------------------------- R1 Pro kinematics

class R1ProFK:
    """Forward kinematics of the R1 Pro URDF from the recorded joint vector (relative to ``base_link``)."""

    def __init__(self, urdf_path):
        from ..robot.urdf import URDF, KinematicTree
        self.urdf_path = Path(urdf_path)
        self.sha256 = sha256_file(self.urdf_path)
        model = URDF(self.urdf_path)
        self.names = TORSO + ARM["left"] + ARM["right"] + FINGERS["left"] + FINGERS["right"]
        missing = [n for n in self.names if n not in model.by_name]
        if missing:
            raise ValueError(f"{self.urdf_path.name}: not an R1 Pro URDF (missing {missing})")
        self.tree = KinematicTree(model, "base_link", ["left_gripper_link", "right_gripper_link", "torso_link4"],
                                  self.names)
        E = np.eye(4)
        E[:3, 3] = EEF_IN_GRIPPER[0]
        E[:3, :3] = Rotation.from_quat(EEF_IN_GRIPPER[1]).as_matrix()
        self.eef_in_gripper = E

    def forward(self, S: np.ndarray) -> dict:
        q = np.concatenate([S[:, slice(*STATE[k])] for k in
                            ("trunk_qpos", "arm_left_qpos", "arm_right_qpos", "gripper_left_qpos", "gripper_right_qpos")], 1)
        fk = self.tree.fk(q)
        return {"left": fk["left_gripper_link"] @ self.eef_in_gripper,
                "right": fk["right_gripper_link"] @ self.eef_in_gripper, "torso_link4": fk["torso_link4"]}

    def describe(self) -> dict:
        return {"urdf": self.urdf_path.name, "urdf_sha256": self.sha256, "root": "base_link",
                "eef_link": "{side}_gripper_link * translate(0, 0, -0.06) * Ry(pi)",
                "joint_order": self.names}


@lru_cache(maxsize=2)
def _fk(path: str) -> R1ProFK:
    return R1ProFK(path)


# ---------------------------------------------------------------- helpers

def _pose(pos, quat_xyzw) -> np.ndarray:
    T = np.broadcast_to(np.eye(4), pos.shape[:-1] + (4, 4)).copy()
    q = quat_xyzw / np.linalg.norm(quat_xyzw, axis=-1, keepdims=True)
    T[..., :3, :3] = Rotation.from_quat(q).as_matrix()
    T[..., :3, 3] = pos
    return T


def _rpy(cos, sin) -> np.ndarray:
    """Roll-pitch-yaw from their cosines and sines; R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return np.arctan2(sin, cos)


def _rpy_matrix(rpy) -> np.ndarray:
    return Rotation.from_euler("xyz", rpy).as_matrix()


def _list_column(table, name) -> np.ndarray:
    col = table.column(name).combine_chunks()
    flat = col.flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, float).reshape(len(col), -1)


def task_info_layout(keys: list[str]) -> dict:
    """``{bddl_inst: {field: (start, stop)}}`` from ``task_obs_keys`` (concatenation order)."""
    out, i = {}, 0
    for key in keys:
        field = next((f for f in TASK_KEY_DIMS if key.endswith("_" + f)), None)
        if field is None:
            raise ValueError(f"unknown task_obs key {key!r}")
        inst = key[: -len(field) - 1]
        out.setdefault(inst, {})[field] = (i, i + TASK_KEY_DIMS[field])
        i += TASK_KEY_DIMS[field]
    return {"objects": out, "dim": i}


def _intervals(mask: np.ndarray) -> list[list[int]]:
    """[start, stop) frame intervals where mask is true."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return [[int(a), int(b)] for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])]


CARRY = {"window_s": 0.33, "min_object_speed_m_s": 0.03, "max_relative_speed_m_s": 0.03, "max_distance_m": 0.25,
         "max_opening": 0.97, "min_duration_s": 0.5}


def carry_intervals(pos: np.ndarray, valid: np.ndarray, eff: Effector, time: np.ndarray) -> list[list[int]]:
    """Derived label: [start, stop) frames where an object is carried by one gripper.

    The released ``in_gripper`` flags are uninformative (the replay restores state without
    the assisted-grasp constraint, so they stay at -1), so carrying is inferred: the object
    moves (``min_object_speed``), its position in the grasp frame is still
    (``max_relative_speed``), it is within ``max_distance`` of the grasp center and the
    gripper is not fully open. Speeds are central differences over ``window_s``.
    """
    T = len(time)
    if T < 3 or not valid.any():
        return []
    k = max(1, int(round(CARRY["window_s"] / np.median(np.diff(time)) / 2)))
    G = eff.pose
    rel = np.einsum("tji,tj->ti", G[:, :3, :3], pos - G[:, :3, 3])
    lo, hi = np.clip(np.arange(T) - k, 0, T - 1), np.clip(np.arange(T) + k, 0, T - 1)
    dt = time[hi] - time[lo]
    v_obj = np.linalg.norm(pos[hi] - pos[lo], axis=1) / dt
    v_rel = np.linalg.norm(rel[hi] - rel[lo], axis=1) / dt
    ok = (valid & valid[lo] & valid[hi] & (v_obj > CARRY["min_object_speed_m_s"]) & (v_rel < CARRY["max_relative_speed_m_s"])
          & (np.linalg.norm(rel, axis=1) < CARRY["max_distance_m"]) & (eff.opening < CARRY["max_opening"]))
    n_min = CARRY["min_duration_s"] / np.median(np.diff(time))
    merged = []
    for a, b in _intervals(ok):  # bridge short dropouts (regrasp jitter, teleop pauses)
        if merged and a - merged[-1][1] < n_min:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [iv for iv in merged if iv[1] - iv[0] >= n_min]


def _synset(inst: str) -> str:
    return re.sub(r"_\d+$", "", inst)


def _demos_root(path: Path) -> Path:
    # <root>/data/task-XXXX/episode_N.parquet
    return path.parents[2]


def _data_root(path: Path, root):
    if root is not None:
        return Path(root)
    return next((p.parent for p in path.resolve().parents if p.name == "raw"), None)


def _locate(rel: str, path: Path, root, catalog) -> Path | None:
    base = _data_root(path, root)
    if base is None:
        return None
    for e in catalog.values():
        if e.family == "behavior" and e.path == rel:
            local = e.local_path(base)
            return local if local.exists() else None
    return None


def _skills(annotation: dict | None) -> list[dict]:
    if not annotation:
        return []
    out = []
    for s in annotation.get("skill_annotation", []):
        out.append({"skill": s.get("skill_description"), "type": s.get("skill_type"),
                    "frames": s.get("frame_duration"), "objects": s.get("object_id"),
                    "manipulating": s.get("manipulating_object_id")})
    return out


def _roles(objects: dict, skills: list[dict], grasped: set, moved: set) -> dict:
    """Role per scene object name. Order: grasped/manipulating -> placement target -> floor -> fixed -> moved."""
    manipulating, inside, on = set(), set(), set()
    for s in skills:
        for names in s.get("manipulating") or []:
            manipulating.update([names] if isinstance(names, str) else names)
        descs = s.get("skill") or []
        for desc, objs in zip(descs, s.get("objects") or []):
            targets = objs[1:] if len(objs) > 1 else []
            if any(desc.startswith(p) for p in PLACE_IN):
                inside.update(targets)
            elif any(desc.startswith(p) for p in PLACE_ON):
                on.update(targets)
    out = {}
    for name, info in objects.items():
        if name in grasped or name in manipulating:
            out[name] = "manipulated"
        elif name in inside:
            out[name] = "receptacle"
        elif name in on or info["synset"] == "floor.n.01":
            out[name] = "support"
        elif info.get("fixed_base"):
            out[name] = "fixture"
        elif name in moved:
            out[name] = "manipulated"
        else:
            out[name] = "fixture"
    return out


# ---------------------------------------------------------------- adapter

@register("behavior")
def read_behavior(path: Path, *, family: str, root=None, urdf=None, raw=None, catalog=None, limit=None,
                  fk_check: bool = True):
    """Yield the :class:`SourceEpisode` of one ``episode_N.parquet`` (or of every parquet in a folder).

    Sibling ``meta/episodes``, ``annotations`` and ``meta/tasks.jsonl`` files are found from
    the LeRobot layout. ``urdf`` is the R1 Pro URDF (default: the catalogued
    ``omnigibson-robot-assets`` copy under the data root inferred from ``path``); without
    it there is no FK check and no torso height. ``raw`` is the raw HDF5 of the same
    episode (default: the sibling ``2025-challenge-rawdata`` file if present); without
    it ``success`` is ``None``.
    """
    path = Path(path)
    catalog = catalog if catalog is not None else load_catalog()
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if limit is not None:
        files = files[:limit]
    if urdf is None:
        urdf = _locate(URDF_PATH, files[0] if files else path, root, catalog)
    for f in files:
        yield _episode(f, family, root, urdf, raw if len(files) == 1 else None, catalog, fk_check)


def _episode(path: Path, family, root, urdf, raw, catalog, fk_check) -> SourceEpisode:
    import pyarrow.parquet as pq

    m = re.fullmatch(r"episode_(\d+)\.parquet", path.name)
    t = re.fullmatch(r"task-(\d+)", path.parent.name)
    if not (m and t):
        raise ValueError(f"{path}: expected data/task-XXXX/episode_N.parquet")
    ep_name, task_dir = path.stem, path.parent.name
    demos = _demos_root(path)
    meta_path = demos / "meta" / "episodes" / task_dir / f"{ep_name}.json"
    ann_path = demos / "annotations" / task_dir / f"{ep_name}.json"
    tasks_path = demos / "meta" / "tasks.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"episode metadata {meta_path} is required (task_obs_keys)")
    meta = json.loads(meta_path.read_text())
    config = json.loads(meta["config"]) if isinstance(meta["config"], str) else meta["config"]
    scene_file = json.loads(meta["scene_file"]) if isinstance(meta["scene_file"], str) else meta["scene_file"]
    annotation = json.loads(ann_path.read_text()) if ann_path.exists() else None
    tasks = {}
    if tasks_path.exists():
        for line in tasks_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                tasks[d["task_index"]] = d

    table = pq.read_table(path, columns=["index", "episode_index", "task_index", "timestamp",
                                         "observation.state", "observation.task_info"])
    S = _list_column(table, "observation.state")
    TI = _list_column(table, "observation.task_info")
    time = np.asarray(table.column("timestamp").to_numpy(), float)
    episode_index = int(table.column("episode_index")[0].as_py())
    task_index = int(table.column("task_index")[0].as_py())
    T = len(time)
    if S.shape != (T, STATE_DIM):
        raise ValueError(f"{path.name}: observation.state is {S.shape}, expected (T, {STATE_DIM})")
    if not (np.all(np.isfinite(S[:, :112])) and np.all(np.isfinite(S[:, 140:]))):
        raise ValueError(f"{path.name}: non-finite robot state")
    if episode_index != int(m.group(1)):
        raise ValueError(f"{path.name}: episode_index column {episode_index} != file name")

    robot = config["robots"][0]
    if robot.get("type") != "R1Pro":
        raise ValueError(f"{path.name}: robot {robot.get('type')!r} is not R1Pro")
    task_cfg = config.get("task", {})
    task_name = task_cfg.get("activity_name") or tasks.get(task_index, {}).get("task_name") or f"task-{task_index}"
    instruction = tasks.get(task_index, {}).get("task")

    # base_link world pose
    rpy = _rpy(S[:, slice(*STATE["robot_ori_cos"])], S[:, slice(*STATE["robot_ori_sin"])])
    base_T = np.broadcast_to(np.eye(4), (T, 4, 4)).copy()
    base_T[:, :3, :3] = _rpy_matrix(rpy)
    base_T[:, :3, 3] = S[:, slice(*STATE["robot_pos"])]
    # Floor at z = 0: base_link's origin is the bottom of the robot (metadata.json base_link_offset
    # z = half the bbox height), so the floor the robot stands on is at the base_link height.
    raw_base_z = base_T[:, 2, 3].copy()
    floor_z = float(np.median(raw_base_z))
    base_T[:, 2, 3] -= floor_z
    yaw2d = S[:, STATE["robot_2d_ori"][0]]
    base = np.column_stack([base_T[:, 0, 3], base_T[:, 1, 3], yaw2d])
    yaw_err = float(np.abs(np.angle(np.exp(1j * (yaw2d - rpy[:, 2])))).max())

    # effectors from the recorded relative eef poses
    effectors, rel, quat_norm = {}, {}, {}
    for side in SIDES:
        qx = S[:, slice(*STATE[f"eef_{side}_quat"])]
        quat_norm[side] = float(np.abs(np.linalg.norm(qx, axis=1) - 1).max())
        rel[side] = _pose(S[:, slice(*STATE[f"eef_{side}_pos"])], qx)
        W = base_T @ rel[side]
        G = W.copy()
        G[:, :3, :3] = W[:, :3, :3] @ EEF_TO_CONTRACT
        width = S[:, slice(*STATE[f"gripper_{side}_qpos"])].sum(1)
        effectors[side] = Effector(pose=G, opening=np.clip(width / WIDTH_MAX, 0, 1), width=np.clip(width, 0, None),
                                   side_hint=side)

    # FK check and torso height from the pinned URDF
    fk_info, torso_height, robot_desc = None, None, None
    if urdf is not None and fk_check:
        fk = _fk(str(urdf))
        poses = fk.forward(S)
        fk_info = {}
        for side in SIDES:
            dp = np.linalg.norm(poses[side][:, :3, 3] - rel[side][:, :3, 3], axis=1)
            Rm = np.einsum("tji,tjk->tik", rel[side][:, :3, :3], poses[side][:, :3, :3])
            dr = Rotation.from_matrix(Rm).magnitude()
            fk_info[side] = {"max_position_error_m": float(dp.max()), "max_rotation_error_rad": float(dr.max())}
        torso_height = (base_T @ poses["torso_link4"])[:, 2, 3]
        robot_desc = fk.describe()

    # task-relevant objects from observation.task_info
    keys = meta["task_obs_keys"]
    layout = task_info_layout(keys)
    if TI.shape[1] != layout["dim"]:
        raise ValueError(f"{path.name}: task_info has {TI.shape[1]} values, task_obs_keys describe {layout['dim']}")
    inst_to_name = scene_file.get("metadata", {}).get("task", {}).get("inst_to_name", {})
    registry = scene_file.get("objects_info", {}).get("init_info", {})
    agent_inst = next((i for i, n in inst_to_name.items() if n == robot.get("name")), "agent.n.01_1")
    agent_check = None
    objects, grasp_intervals, info, grasped, moved = {}, {}, {}, set(), set()
    in_gripper_true = 0
    raw_tracks = {}
    for inst, fields in layout["objects"].items():
        sl = {k: slice(*v) for k, v in fields.items()}
        real = TI[:, sl["real"]][:, 0] > 0.5
        pos = TI[:, sl["pos"]]
        o_rpy = _rpy(TI[:, sl["ori_cos"]], TI[:, sl["ori_sin"]])
        if inst == agent_inst:
            agent_check = float(np.abs(pos - S[:, slice(*STATE["robot_pos"])]).max())
            continue
        pos = pos - np.array([0.0, 0.0, floor_z])
        name = inst_to_name.get(inst, inst)
        args = registry.get(name, {}).get("args", {})
        info[name] = {"bddl_inst": inst, "synset": _synset(inst), "category": args.get("category"),
                      "model": args.get("model"), "scale": args.get("scale"), "fixed_base": bool(args.get("fixed_base", False)),
                      "class": registry.get(name, {}).get("class_name"), "rooms": args.get("in_rooms")}
        for side in SIDES:
            if f"in_gripper_{side}" in sl:
                in_gripper_true += int((TI[:, sl[f"in_gripper_{side}"]][:, 0] > 0.5).sum())
        g = {side: iv for side in SIDES if (iv := carry_intervals(pos, real, effectors[side], time))}
        if g:
            grasp_intervals[name] = g
            grasped.add(name)
        if real.any() and np.ptp(pos[real], axis=0).max() > 0.02:
            moved.add(name)
        raw_tracks[name] = (pos, o_rpy, real)

    skills = _skills(annotation)
    roles = _roles(info, skills, grasped, moved)
    for name, (pos, o_rpy, real) in raw_tracks.items():
        pose = np.zeros((T, 7))
        pose[:, 3] = 1.0
        q = Rotation.from_euler("xyz", o_rpy[real]).as_quat()  # x, y, z, w
        pose[real, :3] = pos[real]
        pose[real, 3:] = q[:, [3, 0, 1, 2]]
        objects[name] = ObjectTrack(pose=pose, valid=real, role=roles[name],
                                    geometry={"kind": "asset", "frame": "object root link (OmniGibson Pose state)",
                                              **{k: v for k, v in info[name].items() if v is not None}})

    # success from the raw recording, if present
    if raw is None:
        cand = demos.parent / "2025-challenge-rawdata" / task_dir / f"{ep_name}.hdf5"
        raw = cand if cand.exists() else None
    success, success_info = None, {"source": None}
    if raw is not None:
        import h5py
        with h5py.File(raw, "r") as h:
            g = h["data"][sorted(k for k in h["data"] if k.startswith("demo_"))[0]]
            term = np.asarray(g["terminated"][()], bool)
            reward = np.asarray(g["reward"][()], float)
            n_raw = int(len(term))
        success = bool(term[-1])
        success_info = {"source": "raw HDF5 data/demo_0/terminated[-1] (BehaviorTask success)", "file": str(raw),
                        "sha256": sha256_file(raw), "first_success_step": int(np.argmax(term)) if term.any() else None,
                        "reward_sum": float(reward.sum()), "raw_steps": n_raw, "parquet_frames": T}

    digest = sha256_file(path)
    entry = locate_entry(path, catalog, digest)
    meta_digest = sha256_file(meta_path)
    meta_entry = locate_entry(meta_path, catalog, meta_digest)  # JSON: git blob sha1 pinned, found by path
    dataset = entry.dataset if entry and entry.dataset else f"behavior/2025-challenge/{task_name}"
    versions = scene_file.get("versions", {})
    scene_args = scene_file.get("init_info", {}).get("args", {})
    ann_meta = (annotation or {}).get("meta_data", {})
    provenance = {
        "file": str(path), "sha256": digest, "url": entry.url if entry else None,
        "revision": entry.revision if entry else None, "catalog_id": entry.id if entry else None,
        "metadata_file": str(meta_path), "metadata_sha256": meta_digest,
        "metadata_catalog_id": meta_entry.id if meta_entry else None,
        "annotation_file": str(ann_path) if annotation else None,
        "episode_index": episode_index, "task_index": task_index, "task_name": task_name,
        "activity_definition_id": task_cfg.get("activity_definition_id"),
        "activity_instance_id": task_cfg.get("activity_instance_id"),
        "scene_model": scene_args.get("scene_model") or config.get("scene", {}).get("scene_model"),
        "scene_instance": config.get("scene", {}).get("scene_instance"),
        "versions": versions, "robot_type": meta.get("robot_type"), "robot_name": robot.get("name"),
        "grasping_mode": robot.get("grasping_mode"), "controllers": {k: v.get("name") for k, v in
                                                                      robot.get("controller_config", {}).items()},
        "frames": T, "n_steps_meta": meta.get("n_steps"), "fps": 30,
        "time_source": "parquet timestamp (30 Hz replay frames)",
        "state_layout": {k: list(v) for k, v in STATE.items()},
        "state_layout_source": f"PROPRIOCEPTION_INDICES['R1Pro'] at BEHAVIOR-1K {OG_LAYOUT_TAG}",
        "task_obs_keys": keys,
        "inst_to_name": inst_to_name,
        "effectors": {"frame": "recorded {side}_eef_link pose (relative to base_link) composed with the base_link world pose",
                      "eef_to_contract_rotation": EEF_TO_CONTRACT.tolist(),
                      "opening": "q_finger1 + q_finger2 (m) / 0.10", "width_max_m": WIDTH_MAX,
                      "eef_quat_norm_error": quat_norm},
        "fk_check": fk_info, "robot_model": robot_desc,
        "base": "base_link world x, y and robot_2d_ori (yaw)", "world_z_offset_m": -floor_z,
        "world_z_offset_rule": "all poses translated by -median(base_link world z): base_link origin is the "
                               "bottom of the R1 Pro (floor contact), so the floor it stands on is at z = 0",
        "source_base_z_range_m": [float(raw_base_z.min()), float(raw_base_z.max())],
        "base_z_range_m": [float(base_T[:, 2, 3].min()),
                                                                                   float(base_T[:, 2, 3].max())],
        "base_yaw_consistency_rad": yaw_err, "agent_task_info_vs_robot_pos_m": agent_check,
        "torso_height": "world z of torso_link4 (arm and head mount) by URDF FK" if torso_height is not None else None,
        "object_roles_rule": "carried (derived) or annotated manipulating -> manipulated; target of "
                             "'place in'/'pour'/'insert' -> receptacle; target of 'place on'/'pick up from'/... or floor "
                             "-> support; fixed_base -> fixture; moved > 2 cm -> manipulated; else fixture",
        "carry_intervals": grasp_intervals,
        "carry_rule": {**CARRY, "label": "derived (not recorded); [start, stop) frames per object and side"},
        "in_gripper_true_frames": in_gripper_true,
        "in_gripper_note": "task_info in_gripper flags are -1 throughout the replayed release; not used",
        "skills": skills, "annotation_valid_duration": ann_meta.get("valid_duration"),
        "success": success_info,
        "missing": ["articulated scene joint positions (only in the serialized raw sim state, not decoded)",
                    "object geometry (BEHAVIOR Data Bundle assets, encrypted, not redistributable)",
                    "non-task scene objects", "particle systems (water, dust, etc.)",
                    "joint efforts (invalid in the release)"],
        "scene": "none: OmniGibson/Isaac Sim house from the BEHAVIOR Data Bundle; no MuJoCo scene",
        "license_detail": {"demos": "MIT (dataset card)", "rawdata": "MIT (dataset card)",
                           "robot_assets": "MIT (omnigibson-robot-assets dataset card)",
                           "scene_and_object_assets": "BEHAVIOR Data Bundle license (non-commercial, no redistribution)"},
        "evidence": EVIDENCE,
    }
    lineage = {"generated": False, "human": True, "source_type": "teleoperation (JoyLo)",
               "initial_state": f"behavior/{task_name}/instance/{task_cfg.get('activity_instance_id')}",
               "raw_source": f"behavior-1k/2025-challenge-rawdata/{task_dir}/{ep_name}.hdf5",
               "derived": "observations replayed from the raw recording by the publisher"}
    return SourceEpisode(
        family=family, dataset=dataset, episode_id=f"{task_dir}/{ep_name}", task=task_name, time=time,
        effectors=effectors, objects=objects, base=base, base_hint=base[0].copy(), torso_height=torso_height,
        articulations={}, scene=None, instruction=instruction, success=success, regime="mobile_manipulation",
        license=LICENSE, provenance=provenance, lineage=lineage)


__all__ = ["read_behavior", "R1ProFK", "task_info_layout", "STATE", "EEF_TO_CONTRACT"]
