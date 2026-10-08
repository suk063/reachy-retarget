"""Adapter for MobileManiBench / MobileManiDataset (``arnoldland/MobileManiBench``), G1 robot only.

MobileManiDataset (arXiv 2602.05233, Microsoft Research Asia) holds Isaac Sim 4.5 / Isaac Lab
rollouts of per-object PPO policies ("MobileManiRL") on two mobile robots. Only the AgiBot G1
(genie ``G1_120s``, parallel gripper) is read here; the XHand dexterous-hand robot is refused.
Skills: open / close articulated PartNet-Mobility and UniDoor objects, pull / push carts and
chairs (``open``/``close`` of groups ``cart``/``chair``), pick YCB objects (``Open/ycb``).

Layout (one folder per episode, as extracted from the per-group tars; see
``range_member`` catalog entries fetched with :func:`reachy_retarget.acquire.fetch`)::

    <robot>/<Open|Close>/<partnet|unidoor|ycb>/<group>/<NNNN>/<object>/train_0/
        params/env.yaml                                  # Isaac Lab env config of the policy
        trajectories/traj_<k>/scene_infos.json           # room of trajectory batch k
        trajectories/traj_<k>/episode_<j>/state_infos.pkl  # this adapter's input
        trajectories/traj_<k>/episode_<j>/*.mp4          # camera videos (never fetched)

``state_infos.pkl`` (``unimanip/utils/env_model.py`` ``_get_observations``, pinned commit
:data:`CODE_COMMIT`) is a dict of float32 arrays with one row per recorded frame (every env
step, 30 Hz): ``time`` (step counter from 1), ``success`` (success flag), ``action`` (7: hand
position/rotation deltas and gripper), ``object`` (grasp point position, roll-pitch-yaw, goal
position), ``robot_base`` / ``robot_hand`` (``base_link`` / ``gripper_r_center_link``:
position, roll-pitch-yaw, linear and angular velocity), ``robot_body`` (48 bodies x 12, order
:data:`BODIES`), ``robot_joint`` (36 joints x position/velocity/acceleration, order
:data:`JOINTS`), ``robot_joint_target`` (36), camera poses, and ``init`` (robot and object root
states, object joint positions, room pose). Positions are in the env frame (env origin
subtracted; one env, origin 0). Euler angles are roll-pitch-yaw with R = Rz Ry Rx. Body and
joint orders are the ones listed in ``g1_robot_env.py`` and are checked here by FK.

* **Effector** ``right`` (the only arm the policies move; the left arm holds its initial pose):
  orientation of the recorded ``gripper_r_center_link`` (URDF: ``gripper_r_base_link`` ·
  translate(0, 0, 0.1) · Rz(-pi/2)), whose +z is the approach axis and whose +y points from the
  outer to the inner finger pad; position = the recorded midpoint of the two pad links
  (``gripper_r_{inner,outer}_link5``, mesh ``*_Pad_Link``) projected on the center link's z
  axis (the pads sit 0.085-0.107 m in front of the center link depending on the four-bar finger
  angle). ``width`` = distance between the recorded pad link origins; ``opening`` =
  ``idx81_gripper_r_outer_joint1`` clipped to [0, 1] (URDF limit; 0 closed, 1 open, pad
  separation 0.002 -> 0.106 m by FK). The recorded center link pose is checked against FK of the
  pinned ``G1_120s.urdf`` (``provenance["fk_check"]``).
* **Base**: ``base_link`` x, y, yaw. The G1 USD of the release adds two joints between a fixed
  ``Root`` and ``base_link``: ``slider_basex`` (yaw about the root) and ``slider_basey``
  (translation along the rotated heading), so ``base_link`` moves on a polar rig about the
  spawn point (checked per episode, ``provenance["base_check"]``). ``torso_height`` = world z
  of ``arm_base_link`` (arm mount; the lift/pitch joints stay at their initial values).
* **World**: the Isaac Lab env frame. The ground plane and ground box top are at z = 0; the
  robot root floats at z = 0.01 (gravity disabled) and ``base_link`` at z = 0.02. No offset.
* **Objects**: the recorded *grasp point* only. For articulated objects this is the handle
  pose (publisher's closed-handle pose on the grasp link, composed with the link pose each
  frame); for YCB objects it is the body COM pose. The object root pose and joint positions are
  recorded only at t0 (``init``); the root track (fixture) is valid at t0 only. The goal
  position (``object[:, 6:9]``) is kept in provenance. The support stage / table under
  tabletop objects and the room are not recorded per episode.
* **Articulations** (revolute / prismatic grasp joints): *derived* from the handle track,
  because the release does not store per-frame object joint positions: theta(t) = rotation of
  the handle about its dominant axis since t0 (revolute) or translation along the dominant
  direction (prismatic); qpos = q0 + theta for ``open`` and q0 - theta for ``close`` with
  q0 = the recorded initial value of the grasp joint (largest |init joint|; all are 0.001 for
  open tasks). Residuals of the single-axis fit are in ``provenance["articulation"]``.
* **Success**: per-frame flag; the recorder deletes every episode without success
  (``env_model.py`` ``_reset_idx``), so every released episode has ``success = True`` and the
  release says nothing about the failure rate.
* **Instruction**: the publisher's VLA prompt (``load_object_prompt`` in
  ``unimanip/utils/general_utils.py``): ``<skill> <object>[ at <part>]`` from
  ``configs/data/analysis_<category>.yaml``.

No ``SceneRef``: rooms (GenieSim / IsaacSim USD) and objects (PartNet-Mobility, UniDoor, YCB
USD in ``Assets/Assets.zip``) are Isaac Sim assets, not MuJoCo models.
"""
from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from ..acquire import load_catalog, locate_entry, sha256_file
from ..schema.source import Articulation, Effector, ObjectTrack, SourceEpisode
from .registry import register

FAMILY = "mobilemanibench"
HF_REVISION = "88bc86eed0162c3b9a27bc962ce2d4815cbf2e59"
CODE_COMMIT = "135466638d379c642f0c3f1742e6c20091b203c7"
_GH = f"https://github.com/DexHand/MobileManiBench/blob/{CODE_COMMIT}/"
EVIDENCE = {
    "recording": _GH + "unimanip/utils/env_model.py#L1334-L1390",
    "state_definitions": _GH + "unimanip/utils/env_model.py#L596-L700",
    "grasp_point": _GH + "unimanip/utils/env_model.py#L420-L500",
    "success_filter": _GH + "unimanip/utils/env_model.py#L1283-L1312",
    "body_joint_order": _GH + "source/isaaclab_tasks/isaaclab_tasks/direct/g1_robot/g1_robot_env.py#L85-L96",
    "euler": _GH + "unimanip/utils/general_utils.py#L340-L362",
    "prompt": _GH + "unimanip/utils/general_utils.py#L678-L695",
    "paper": "https://arxiv.org/abs/2602.05233",
    "dataset": f"https://huggingface.co/datasets/arnoldland/MobileManiBench/tree/{HF_REVISION}",
}
LICENSE = "MIT (dataset card of arnoldland/MobileManiBench); code BSD-3-Clause; object assets per upstream (PartNet-Mobility, YCB)"
FPS = 30.0
BODIES = [
    "Root", "basex", "base_link", "body_link1", "body_link2", "arm_base_link", "head_link1", "arm_l_base_link",
    "arm_r_base_link", "head_link2", "arm_l_link1", "arm_r_link1", "arm_l_link2", "arm_r_link2", "arm_l_link3",
    "arm_r_link3", "arm_l_link4", "arm_r_link4", "arm_l_link5", "arm_r_link5", "arm_l_link6", "arm_r_link6",
    "arm_l_end_link", "arm_r_end_link", "gripper_l_base_link", "gripper_r_base_link", "gripper_l_inner_link1",
    "gripper_l_outer_link1", "gripper_l_center_link", "gripper_r_inner_link1", "gripper_r_outer_link1",
    "gripper_r_center_link", "gripper_l_inner_link3", "gripper_l_outer_link3", "gripper_r_inner_link3",
    "gripper_r_outer_link3", "gripper_l_inner_link4", "gripper_l_outer_link4", "gripper_r_inner_link4",
    "gripper_r_outer_link4", "gripper_l_inner_link5", "gripper_l_inner_link2", "gripper_l_outer_link5",
    "gripper_l_outer_link2", "gripper_r_inner_link5", "gripper_r_inner_link2", "gripper_r_outer_link5",
    "gripper_r_outer_link2"]
JOINTS = [
    "slider_basex", "slider_basey", "idx01_body_joint1", "idx02_body_joint2", "idx11_head_joint1", "idx12_head_joint2",
    "idx21_arm_l_joint1", "idx61_arm_r_joint1", "idx22_arm_l_joint2", "idx62_arm_r_joint2", "idx23_arm_l_joint3",
    "idx63_arm_r_joint3", "idx24_arm_l_joint4", "idx64_arm_r_joint4", "idx25_arm_l_joint5", "idx65_arm_r_joint5",
    "idx26_arm_l_joint6", "idx66_arm_r_joint6", "idx27_arm_l_joint7", "idx67_arm_r_joint7",
    "idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1", "idx71_gripper_r_inner_joint1",
    "idx81_gripper_r_outer_joint1", "idx32_gripper_l_inner_joint3", "idx42_gripper_l_outer_joint3",
    "idx72_gripper_r_inner_joint3", "idx82_gripper_r_outer_joint3", "idx33_gripper_l_inner_joint4",
    "idx43_gripper_l_outer_joint4", "idx73_gripper_r_inner_joint4", "idx83_gripper_r_outer_joint4",
    "idx54_gripper_l_inner_joint0", "idx53_gripper_l_outer_joint0", "idx94_gripper_r_inner_joint0",
    "idx93_gripper_r_outer_joint0"]
B = {n: i for i, n in enumerate(BODIES)}
J = {n: i for i, n in enumerate(JOINTS)}
FK_JOINTS = ["idx01_body_joint1", "idx02_body_joint2"] + [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
GRIPPER_JOINT = "idx81_gripper_r_outer_joint1"
PADS = ("gripper_r_inner_link5", "gripper_r_outer_link5")
URDF_PATH = "Assets/g1_robot_rotate/G1_120s.urdf"
ANALYSIS_PATH = "code/unimanip/configs/data/analysis_{}.yaml"
SKILL_ALIASES = {("cart", "open"): "pull", ("cart", "close"): "push", ("chair", "open"): "pull",
                 ("chair", "close"): "push", ("ycb", "open"): "pick"}


# ---------------------------------------------------------------- reading

class _StateUnpickler(pickle.Unpickler):
    """Only numpy array reconstruction is allowed (``state_infos.pkl`` holds dicts of arrays)."""

    _ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
                ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
                ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"state_infos.pkl: refusing global {module}.{name}")
        if module.startswith("numpy.core.") and not hasattr(np, "core"):
            module = module.replace("numpy.core.", "numpy._core.", 1)
        return super().find_class(module, name)


def load_state(path) -> dict:
    with open(path, "rb") as f:
        return _StateUnpickler(f).load()


class _EnvLoader(yaml.SafeLoader):
    pass


_EnvLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: list(loader.construct_sequence(node)))


def load_env_yaml(path) -> dict:
    return yaml.load(Path(path).read_text(), Loader=_EnvLoader)


def parse_object_name(name: str) -> dict:
    """``7119-joint_0-REVOLUTE-link_0-handle_0`` -> id/joint/type/link/handle; YCB names pass through."""
    m = re.fullmatch(r"(\w+?)-(joint_\d+)-(REVOLUTE|PRISMATIC|None)-(link_\d+)-(handle_\d+)", name)
    if not m:
        return {"id": name, "joint": None, "joint_type": None, "link": None, "handle": None}
    return dict(zip(("id", "joint", "joint_type", "link", "handle"), m.groups()))


def _layout(path: Path) -> dict:
    parts = path.parts
    try:
        i = len(parts) - 1 - parts[::-1].index("trajectories")
    except ValueError:
        raise ValueError(f"{path}: expected .../<robot>/<task>/<category>/<group>/<NNNN>/<object>/train_N/"
                         "trajectories/traj_K/episode_J/state_infos.pkl") from None
    if i < 7 or len(parts) - i != 4 or path.name != "state_infos.pkl":
        raise ValueError(f"{path}: not a MobileManiDataset episode file")
    robot, task, category, group, index, obj, train = parts[i - 7:i]
    return {"robot": robot, "task": task, "category": category, "group": group, "index": index, "object": obj,
            "train": train, "traj": parts[i + 1], "episode": parts[i + 2],
            "train_dir": Path(*parts[:i]), "traj_dir": Path(*parts[:i + 2])}


def _data_root(path: Path, root):
    if root is not None:
        return Path(root)
    return next((p.parent for p in path.resolve().parents if p.name == "raw"), None)


def _local(rel: str, path: Path, root) -> Path | None:
    base = _data_root(path, root)
    if base is None:
        return None
    p = base / "raw" / FAMILY / rel
    return p if p.exists() else None


@lru_cache(maxsize=8)
def _analysis(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def instruction_for(skill: str, category: str, group: str, obj: str, analysis: dict | None) -> tuple[str, str | None]:
    """The publisher's prompt ``<skill> <object>[ at <part>]`` and the analysis entry used."""
    info = None
    if analysis is not None:
        info = (analysis.get(group) or {}).get(obj.replace("-", "/") if category != "ycb" else obj)
    if not info:
        noun = obj.split("_", 1)[1] if category == "ycb" and "_" in obj else group
        return f"{skill} {noun.replace('_', ' ')}", None
    bits = str(info).split("/")
    noun = bits[1] if len(bits) == 2 else f"{bits[1]} at {bits[2]}"
    return f"{skill} {noun.replace('_', ' ')}", str(info)


# ---------------------------------------------------------------- kinematics

def _poses(arr: np.ndarray) -> np.ndarray:
    """(..., >=6) position + roll-pitch-yaw -> (..., 4, 4)."""
    lead = arr.shape[:-1]
    T = np.broadcast_to(np.eye(4), lead + (4, 4)).copy()
    T[..., :3, :3] = Rotation.from_euler("xyz", arr[..., 3:6].reshape(-1, 3)).as_matrix().reshape(lead + (3, 3))
    T[..., :3, 3] = arr[..., :3]
    return T


def _pose7(T: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    return np.column_stack([T[:, :3, 3], q[:, [3, 0, 1, 2]]])


class G1FK:
    """FK of ``gripper_r_center_link`` relative to ``base_link`` from the recorded joint vector."""

    def __init__(self, urdf_path):
        from ..robot.urdf import URDF, KinematicTree
        self.urdf_path = Path(urdf_path)
        self.sha256 = sha256_file(self.urdf_path)
        model = URDF(self.urdf_path)
        missing = [n for n in FK_JOINTS + [GRIPPER_JOINT] if n not in model.by_name]
        if missing:
            raise ValueError(f"{self.urdf_path.name}: not the G1_120s URDF (missing {missing})")
        self.tree = KinematicTree(model, "base_link", ["gripper_r_center_link", "arm_base_link"], FK_JOINTS)

    def forward(self, qpos: np.ndarray) -> dict:
        return self.tree.fk(qpos[:, [J[n] for n in FK_JOINTS]])

    def describe(self) -> dict:
        return {"urdf": self.urdf_path.name, "urdf_sha256": self.sha256, "root": "base_link",
                "target": "gripper_r_center_link", "joint_order": FK_JOINTS,
                "note": "slider_basex/slider_basey are USD-only joints between Root and base_link"}


@lru_cache(maxsize=2)
def _fk(path: str) -> G1FK:
    return G1FK(path)


def derive_articulation(handle: np.ndarray, joint_type: str) -> dict:
    """Single-axis displacement of the grasp link since t0 from the (T, 4, 4) handle track."""
    R0, p0 = handle[0, :3, :3], handle[0, :3, 3]
    rv = Rotation.from_matrix(np.einsum("tij,kj->tik", handle[:, :3, :3], R0)).as_rotvec()  # R_t R_0^T
    d = handle[:, :3, 3] - p0
    vec = rv if joint_type == "REVOLUTE" else d
    k = int(np.argmax(np.linalg.norm(vec, axis=1)))
    mag = float(np.linalg.norm(vec[k]))
    if mag < 1e-6:
        return {"theta": np.zeros(len(handle)), "axis_world": None, "max_motion": mag, "off_axis_residual": 0.0,
                "other_residual": float(np.linalg.norm(d if joint_type == "REVOLUTE" else rv, axis=1).max())}
    axis = vec[k] / mag
    theta = vec @ axis
    off = float(np.linalg.norm(vec - theta[:, None] * axis, axis=1).max())
    out = {"theta": theta, "axis_world": axis.tolist(), "max_motion": mag, "off_axis_residual": off}
    if joint_type == "PRISMATIC":
        out["rotation_residual_rad"] = float(np.linalg.norm(rv, axis=1).max())
    else:  # hinge line from the circle traced by the handle: fit centre in the plane normal to the axis
        P = handle[:, :3, 3] - (handle[:, :3, 3] @ axis)[:, None] * axis
        if np.ptp(theta) > 0.2:
            A = np.column_stack([2 * P, np.ones(len(P))])
            sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
            c = sol[:3]
            c = c - (c @ axis) * axis
            r = np.linalg.norm(P - c, axis=1)
            out.update(hinge_point_world=(c + (handle[0, :3, 3] @ axis) * axis).tolist(),
                       handle_radius_m=float(r.mean()), radius_spread_m=float(np.ptp(r)))
    return out


# ---------------------------------------------------------------- adapter

@register("mobilemanibench", select=True)
def read_mobilemanibench(path: Path, *, family: str, root=None, urdf=None, catalog=None, limit=None,
                         fk_check: bool = True, select=None):
    """Yield the :class:`SourceEpisode` of one ``state_infos.pkl`` (or of every one below a folder).

    ``params/env.yaml`` and ``scene_infos.json`` are found from the extracted tar layout.
    ``urdf`` is the G1 URDF (default: the catalogued ``Assets/g1_robot_rotate/G1_120s.urdf``
    under the data root inferred from ``path``); without it there is no FK check. The prompt
    tables (``code/unimanip/configs/data/analysis_*.yaml``) are found the same way.
    """
    path = Path(path)
    catalog = catalog if catalog is not None else load_catalog()
    files = sorted(path.rglob("state_infos.pkl")) if path.is_dir() else [path]
    if limit is not None:
        files = files[:limit]
    for i, f in enumerate(files):
        if select is not None and not select(i):
            yield None  # another shard's episode: not read
            continue
        u = urdf if urdf is not None else _local(URDF_PATH, f, root)
        yield _episode(f, family, root, u, catalog, fk_check)


def _episode(path: Path, family, root, urdf, catalog, fk_check) -> SourceEpisode:
    lay = _layout(path)
    if lay["robot"] != "G1_Robot":
        raise ValueError(f"{path}: robot {lay['robot']!r} is not supported (only the G1 parallel gripper; "
                         "XHand dexterous-hand episodes are excluded)")
    env_path = lay["train_dir"] / "params" / "env.yaml"
    if not env_path.exists():
        raise FileNotFoundError(f"{env_path} is required (robot, object and task configuration)")
    env = load_env_yaml(env_path)
    if env.get("robot_name") != "g1_robot" or env.get("robot_hand_body_name") != "gripper_r_center_link":
        raise ValueError(f"{env_path}: robot {env.get('robot_name')!r} / hand {env.get('robot_hand_body_name')!r} "
                         "is not the G1 right gripper")
    scene_path = lay["traj_dir"] / "scene_infos.json"
    scene_infos = json.loads(scene_path.read_text()) if scene_path.exists() else None
    d = load_state(path)

    time_steps = np.asarray(d["time"], float).reshape(-1)
    T = len(time_steps)
    rb = np.asarray(d["robot_body"], float)
    rj = np.asarray(d["robot_joint"], float)
    obj = np.asarray(d["object"], float)
    if rb.shape != (T, len(BODIES), 12) or rj.shape != (T, len(JOINTS), 3) or obj.shape != (T, 9):
        raise ValueError(f"{path}: unexpected shapes body {rb.shape}, joint {rj.shape}, object {obj.shape}")
    if not (np.all(np.isfinite(rb)) and np.all(np.isfinite(rj[:, :, 0])) and np.all(np.isfinite(obj))):
        raise ValueError(f"{path}: non-finite state")
    layout_check = {"robot_base_eq_body_base_link": float(np.abs(d["robot_base"] - rb[:, B["base_link"]]).max()),
                    "robot_hand_eq_body_center_link": float(np.abs(d["robot_hand"] - rb[:, B["gripper_r_center_link"]]).max())}
    if max(layout_check.values()) > 1e-6:
        raise ValueError(f"{path}: robot_body order does not match the documented body list ({layout_check})")
    time = time_steps / FPS
    qpos = rj[:, :, 0]

    # base_link world pose and the polar slider rig check
    base_T = _poses(rb[:, B["base_link"]])
    base = np.column_stack([rb[:, B["base_link"], 0], rb[:, B["base_link"], 1], rb[:, B["base_link"], 5]])
    root_pose = rb[:, B["Root"]]
    yaw_pred = root_pose[:, 5] + np.pi / 2 + qpos[:, J["slider_basex"]]
    pos_pred = root_pose[:, :2] + qpos[:, J["slider_basey"], None] * np.column_stack([np.cos(yaw_pred), np.sin(yaw_pred)])
    base_check = {"max_yaw_error_rad": float(np.abs(np.angle(np.exp(1j * (yaw_pred - base[:, 2])))).max()),
                  "max_xy_error_m": float(np.abs(pos_pred - base[:, :2]).max()),
                  "root_motion_m": float(np.ptp(root_pose[:, :3], axis=0).max()),
                  "base_z_range_m": [float(base_T[:, 2, 3].min()), float(base_T[:, 2, 3].max())],
                  "max_tilt_rad": float(np.abs(rb[:, B["base_link"], 3:5]).max()),
                  "rule": "base_link yaw = Root yaw + pi/2 + slider_basex; xy = Root xy + slider_basey * heading"}

    # effector: center link orientation, pad midpoint on its z axis
    C = _poses(rb[:, B["gripper_r_center_link"]])
    pads = rb[:, [B[PADS[0]], B[PADS[1]]], :3]
    mid_rel = np.einsum("tji,tj->ti", C[:, :3, :3], pads.mean(1) - C[:, :3, 3])
    G = C.copy()
    G[:, :3, 3] = C[:, :3, 3] + C[:, :3, 2] * mid_rel[:, 2:3]
    sep_rel = np.einsum("tji,tj->ti", C[:, :3, :3], pads[:, 0] - pads[:, 1])
    width = np.linalg.norm(pads[:, 0] - pads[:, 1], axis=1)
    q_grip = qpos[:, J[GRIPPER_JOINT]]
    effectors = {"right": Effector(pose=G, opening=np.clip(q_grip, 0.0, 1.0), width=width, side_hint="right")}
    effector_info = {
        "frame": "recorded gripper_r_center_link orientation; position = recorded pad-link midpoint projected on its z axis",
        "pad_depth_m": [float(mid_rel[:, 2].min()), float(mid_rel[:, 2].max())],
        "pad_midpoint_lateral_offset_max_m": float(np.linalg.norm(mid_rel[:, :2], axis=1).max()),
        "pad_separation_along_y_min_fraction": float((sep_rel[:, 1] / np.maximum(width, 1e-9)).min()),
        "opening": f"clip({GRIPPER_JOINT}, 0, 1); 0 closed, 1 open (URDF limit)",
        "gripper_joint_range": [float(q_grip.min()), float(q_grip.max())],
        "width": "distance between recorded gripper_r_{inner,outer}_link5 (pad link) origins",
        "width_range_m": [float(width.min()), float(width.max())],
        "left_arm": "not an effector: idle at its initial pose (left arm motion "
                    f"{float(np.ptp(qpos[:, [J[f'idx2{i}_arm_l_joint{i}'] for i in range(1, 8)]], axis=0).max()):.2e} rad)"}

    # FK check against the pinned URDF; torso height (arm mount) from the recorded body
    fk_info, robot_desc = None, None
    if urdf is not None and fk_check:
        fk = _fk(str(urdf))
        poses = fk.forward(qpos)
        rel = np.linalg.inv(base_T) @ C
        F = poses["gripper_r_center_link"]
        dp = np.linalg.norm(F[:, :3, 3] - rel[:, :3, 3], axis=1)
        dr = Rotation.from_matrix(np.einsum("tji,tjk->tik", rel[:, :3, :3], F[:, :3, :3])).magnitude()
        A = np.linalg.inv(base_T) @ _poses(rb[:, B["arm_base_link"]])
        fk_info = {"right": {"max_position_error_m": float(dp.max()), "max_rotation_error_rad": float(dr.max())},
                   "arm_base_link_max_position_error_m":
                       float(np.linalg.norm(poses["arm_base_link"][:, :3, 3] - A[:, :3, 3], axis=1).max())}
        robot_desc = fk.describe()
    torso_height = rb[:, B["arm_base_link"], 2].copy()

    # task naming
    action = str(env.get("action_type") or lay["task"].lower())
    category, group = lay["category"], lay["group"]
    skill = SKILL_ALIASES.get((group, action), action)
    oinfo = parse_object_name(lay["object"])
    analysis_path = _local(ANALYSIS_PATH.format(category), path, root)
    analysis = _analysis(str(analysis_path)) if analysis_path else None
    instruction, analysis_entry = instruction_for(skill, category, group, lay["object"], analysis)
    noun = oinfo["id"].split("_", 1)[1] if category == "ycb" and "_" in oinfo["id"] else group
    task = f"{skill}_{noun}"

    # objects
    init = d.get("init", {})
    init_obj = init.get("object", {})
    root_state = np.asarray(init_obj.get("root_link_state_w", np.full((1, 13), np.nan)), float).reshape(-1)
    init_joint = np.asarray(init_obj["joint_pos"], float).reshape(-1) if "joint_pos" in init_obj else None
    handle_T = _poses(obj[:, :6])
    objects, articulations, art_info = {}, {}, None
    obj_cfg = env.get("object_infos") or {}
    geom = {"kind": "asset", "category": category, "group": group, "object": lay["object"], "asset_id": oinfo["id"],
            "scale": obj_cfg.get("scale"), "usd": env.get("object_usd_path")}
    if category == "ycb":
        name = noun
        objects[name] = ObjectTrack(pose=_pose7(handle_T), valid=np.ones(T, bool), role="manipulated",
                                    geometry={**geom, "frame": "rigid body center of mass (body_com_state_w)",
                                              "ycb_mass_kg": (env.get("control_params") or {}).get("ycb_mass")})
    else:
        base_name = f"{group}_{oinfo['id']}"
        objects[f"{base_name}_{oinfo['handle'] or 'handle'}"] = ObjectTrack(
            pose=_pose7(handle_T), valid=np.ones(T, bool), role="manipulated",
            geometry={"kind": "point", "frame": f"publisher grasp point ({oinfo['handle']} closed-handle pose on "
                                                f"{oinfo['link']}, composed with the link pose)",
                      "part_of": base_name, "link": oinfo["link"], "joint": oinfo["joint"]})
        if oinfo["joint_type"] in ("REVOLUTE", "PRISMATIC"):
            rpose = np.zeros((T, 7))
            rpose[:, 3] = 1.0
            valid = np.zeros(T, bool)
            if np.all(np.isfinite(root_state[:7])):
                rpose[:] = root_state[:7]
                valid[0] = True
            objects[base_name] = ObjectTrack(pose=rpose, valid=valid, role="fixture",
                                             geometry={**geom, "frame": "articulation root link",
                                                       "measured_rows": "t0 only (init root_link_state_w)"})
            der = derive_articulation(handle_T, oinfo["joint_type"])
            sign = 1.0 if action == "open" else -1.0
            k0 = int(np.argmax(np.abs(init_joint))) if init_joint is not None and init_joint.size else None
            q0 = float(init_joint[k0]) if k0 is not None else 0.0
            articulations[base_name] = Articulation(joint_names=[f"{base_name}:{oinfo['joint']}"],
                                                    qpos=(q0 + sign * der["theta"])[:, None])
            art_info = {"label": "derived from the recorded handle track (no per-frame joint positions in the release)",
                        "joint_type": oinfo["joint_type"], "q0": q0, "q0_init_index": k0,
                        "init_joint_pos": None if init_joint is None else init_joint.tolist(),
                        "sign": sign, "rule": "qpos = q0 + sign * theta, sign = +1 open / -1 close; theta = motion "
                                              "of the handle since t0 about/along its dominant axis",
                        **{k: v for k, v in der.items() if k != "theta"},
                        "final_qpos": float(q0 + sign * der["theta"][-1])}

    # success
    s = np.asarray(d["success"], float).reshape(-1) > 0.5
    success = bool(s.any())
    first = int(np.argmax(s)) if s.any() else None

    digest = sha256_file(path)
    entry = locate_entry(path, catalog, digest)
    src = entry.source if entry is not None else {}
    stored = src.get("stored_size") or (entry.size if entry is not None else None)
    goal = obj[0, 6:9]
    provenance = {
        "file": str(path), "sha256": digest, "catalog_id": entry.id if entry else None,
        "sha256_source": None if entry is None else ("catalog" if entry.sha256 else "tofu (see the fetch ledger)"),
        "url": entry.url if entry else None,
        "archive": src.get("archive"), "archive_sha256": src.get("archive_sha256"),
        "byte_range": [src["offset"], src["offset"] + stored] if "offset" in src else None,
        "revision": HF_REVISION, "code_commit": CODE_COMMIT,
        "env_yaml": str(env_path), "env_yaml_sha256": sha256_file(env_path),
        "scene_infos": scene_infos, "room_init": {k: np.asarray(v).tolist() for k, v in init.get("room", {}).items()},
        "robot": lay["robot"], "task_folder": lay["task"], "category": category, "group": group,
        "object_index": lay["index"], "object": lay["object"], "object_infos": obj_cfg,
        "trajectory": lay["traj"], "episode": lay["episode"], "train_run": lay["train"],
        "frames": T, "fps": FPS, "time_source": "time = step counter / 30 (record_period 1/30 s; sim dt 1/60, decimation 2)",
        "body_order": "g1_robot_env.py comment, checked: robot_base == body base_link, robot_hand == body "
                      "gripper_r_center_link, and URDF FK below",
        "layout_check": layout_check, "base_check": base_check, "effectors": effector_info,
        "fk_check": fk_info, "robot_model": robot_desc,
        "torso_height": "world z of arm_base_link (recorded body)",
        "torso_joints": {n: [float(qpos[:, J[n]].min()), float(qpos[:, J[n]].max())]
                         for n in ("idx01_body_joint1", "idx02_body_joint2")},
        "head_joints": {n: float(np.median(qpos[:, J[n]])) for n in ("idx11_head_joint1", "idx12_head_joint2")},
        "world": "Isaac Lab env frame (single env, origin 0); ground plane / ground box top at z = 0; no offset",
        "object_goal_position": goal.tolist(),
        "object_root_init": root_state.tolist(), "object_init_joint_pos": None if init_joint is None else init_joint.tolist(),
        "articulation": art_info,
        "instruction_source": {"analysis_file": str(analysis_path) if analysis_path else None, "entry": analysis_entry,
                               "rule": "load_object_prompt: '<skill> <object>[ at <part>]'"},
        "action": {"dim": int(np.asarray(d["action"]).shape[1]), "frame": env.get("action_frame"),
                   "pos_scale": (env.get("control_params") or {}).get("action_pos_scale"),
                   "rot_scale": (env.get("control_params") or {}).get("action_rot_scale")},
        "success": {"source": "state_infos success flag (env_success_flag)", "first_success_frame": first,
                    "success_frames": int(s.sum()),
                    "selection": "the recorder deletes episodes without success; failures are not released"},
        "missing": ["per-frame object joint positions (articulations are derived from the handle track)",
                    "per-frame object root pose (t0 only)", "object, room and support-stage geometry (Isaac USD assets)",
                    "failed rollouts (deleted by the recorder)"],
        "scene": "none: Isaac Sim 4.5 / Isaac Lab scene (GenieSim/IsaacSim room USD + PartNet-Mobility/UniDoor/YCB USD "
                 "from Assets/Assets.zip, stage cuboid under tabletop objects); no MuJoCo scene",
        "license_detail": {"dataset": "MIT (dataset card, arnoldland/MobileManiBench)",
                           "code": "BSD-3-Clause (DexHand/MobileManiBench LICENSE)",
                           "assets": "Assets.zip redistributes PartNet-Mobility (SAPIEN terms, non-commercial research), "
                                     "UniDoor and YCB assets under their own upstream terms; not covered by the card"},
        "evidence": EVIDENCE,
    }
    lineage = {"generated": True, "human": False, "source_type": "PPO policy rollout (MobileManiRL, Isaac Lab)",
               "seed": f"{FAMILY}/{lay['robot']}/{lay['task']}/{category}/{group}/{lay['index']}/{lay['object']}",
               "policy": f"{lay['train']}/model_*.pt (one policy per robot, skill and object)",
               "independence": "episodes are separate rollouts with randomized room, robot and object initial states; "
                               "not variants of one seed episode",
               "variant_of": None}
    return SourceEpisode(
        family=family, dataset=f"{FAMILY}/{lay['robot']}/{lay['task']}/{category}/{group}",
        episode_id=f"{lay['index']}/{lay['object']}/{lay['traj']}/{lay['episode']}", task=task, time=time,
        effectors=effectors, objects=objects, base=base, base_hint=base[0].copy(), torso_height=torso_height,
        articulations=articulations, scene=None, instruction=instruction, success=success,
        regime="mobile_manipulation", license=LICENSE, provenance=provenance, lineage=lineage)


__all__ = ["read_mobilemanibench", "load_state", "load_env_yaml", "parse_object_name", "instruction_for",
           "derive_articulation", "G1FK", "BODIES", "JOINTS"]
