"""Adapter for RoboVerse / MetaSim migrated trajectories (``RoboVerseOrg/roboverse_data``).

RoboVerse stores every migrated benchmark in one simulator-agnostic *v2* layout::

    {robot_name: [episode, ...]}
    episode = {"init_state": {...}, "actions": [...], "states": [state, ...], ...}
    state   = {entity: {"pos": xyz, "rot": wxyz, "dof_pos": {joint: q}?}, ...}

``states`` hold one entry per recorded step for every entity (robot root pose and joint
positions, object root poses, articulation joint positions); there are no velocities, no
end-effector pose and no images. Files are pickles: see :mod:`.roboverse_pickle` (restricted
unpicklers, no torch). Two benchmarks are supported; both are new relative to the other
families of this project (RoboVerse's LIBERO and ManiSkill folders duplicate sources we
already read natively and are not catalogued):

**RLBench** (``trajs/rlbench/<task>/v2/franka_v2.pkl.gz``, 80 tasks, 10 demos each except
``close_box`` with 100). Collected by RLBench's motion planner in CoppeliaSim/PyRep with
``github.com/Fisher-Wang/RLBench`` (``tools/collect_demo.py``): robot joint positions and
object poses were recorded per simulation step in CoppeliaSim, where RLBench grasps by
*parenting* the object to the gripper (kinematic attachment, no friction grasp). The states
were not re-simulated. RoboVerse shifted every object by ``-0.75`` m in z ("the height of the
table in RLBench") and placed its IsaacSim Franka at the RLBench arm pose minus
``(-0.0413, 0.0053, 0.8197)`` with a hand-tuned orientation correction. Here every z is
shifted back by ``+0.75`` m (``RLBENCH_TABLE_HEIGHT``) so the floor is at z = 0 and the table
top at 0.75 m. The recorded robot states are calibrated before FK (``RLBENCH_JOINT4_OFFSET``,
identity root rotation; derived from the rigidity of parented objects, switchable with
``rlbench_calibration=False``). The table itself is not part of the RoboVerse data and is synthesized as a
static support box (top at 0.75 m, cropped to the objects' footprint plus a margin). Objects
come from :data:`.roboverse_rlbench.RLBENCH_OBJECTS` (RoboVerse task configs): primitives get
their geometry, converted RLBench meshes keep their USD asset reference; visual-only
(``XFORM``) entities that never move are goal markers (provenance only).

**CALVIN** (``trajs/calvin/calvin_traj_ann/env_<X>_out/task_<N>_v2.pkl``). Language-annotated
windows (64 states, 30 Hz) of CALVIN's human VR-teleoperated play data, converted by RoboVerse
from CALVIN ``robot_obs``/``scene_obs`` (not re-simulated; the recording itself is a PyBullet
simulation with friction grasps). ``task_<N>`` is the index of the sentence in
``ann_dict.npy``; ``env_meta.source_dir = env_X/episode_<k>_<ann_idx>_<start>_<end>`` gives the
CALVIN frame range, recorded as lineage (CALVIN windows overlap and are crops of one play
stream). RoboVerse's converter maps ``scene_obs`` blocks in the fixed order red, blue, pink,
but CALVIN orders them as listed in each scene config (A: pink, blue, red; C: blue, red,
pink): the colours are remapped here (``CALVIN_BLOCK_ORDER``; verified by language/motion
agreement, see docs/sources.md). Blocks are boxes from CALVIN's block URDFs scaled by the
scene ``global_scaling`` 0.8; the table is a mesh with four joints (slide, drawer, button,
switch), recorded as the AABB of its scaled base mesh plus the asset reference, so no MuJoCo
scene is attached.

Grasp centers (contract: +z approach, +y closing, origin between the pads), from robot
models pinned in the catalog:

* RLBench ``franka``: forward kinematics of RoboVerse ``robots/franka/urdf/franka_panda.urdf``
  (franka_ros kinematics) at the recorded root pose; the grasp center is the midpoint of the
  two fingertip pad boxes ``fingertip_pad_collision_1`` of RoboVerse's
  ``robots/franka/mjcf/panda.xml`` (mujoco_menagerie), i.e. 0.1029 m from ``panda_hand``.
* CALVIN: ``robots/franka_calvin/panda_longer_finger.urdf`` (CALVIN's robot,
  ``tcp_link_id: 15`` = link ``tcp``, 0.14 m from ``panda_hand``), base fixed at
  ``(-0.34, -0.46, 0.24)``. The longer DIGIT fingers have no pad primitive, so CALVIN's own
  TCP is used as the grasp center.

Rotation: ``R_contract = R_hand @ diag(-1, -1, 1)`` (closing from ``panda_leftfinger`` to
``panda_rightfinger`` = -y of the hand). ``width = q_finger1 + q_finger2`` (0-0.08 m; CALVIN
stores half the measured opening width per finger), ``opening = width / 0.08``.
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..acquire import load_catalog, sha256_file
from ..robot.urdf import URDF, KinematicTree
from ..schema.source import Articulation, Effector, ObjectTrack, SourceEpisode
from . import primitive_scene
from .registry import register
from .roboverse_pickle import load as load_pickle
from .roboverse_pickle import load_npy_object
from .roboverse_rlbench import PRIMITIVE_DEFAULT_MASS_KG, RLBENCH_OBJECTS, ROBOVERSE_COMMIT

HF_REVISION = "fab63ccaaed54f413901f86edc3fa1ab77a96500"
HF = f"https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/blob/{HF_REVISION}/"
RV = f"https://github.com/RoboVerseOrg/RoboVerse/blob/{ROBOVERSE_COMMIT}/"
FW_RLBENCH = "https://github.com/Fisher-Wang/RLBench/blob/070249f48c4ea8683aa079a0e0c30da96a29096e/"
CALVIN_ENV = "https://github.com/mees/calvin_env/blob/797142c588c21e76717268b7b430958dbd13bf48/"
EVIDENCE = {
    "v2_format": RV + "packages/metasim/metasim/utils/demo_util/demo_util_v2.py (pos, rot wxyz, dof_pos)",
    "rlbench_collection": FW_RLBENCH + "tools/collect_demo.py (CoppeliaSim states; objects z - 0.75; robot "
                                       "pos - (-0.0413, 0.0053, 0.8197) and quaternion correction)",
    "rlbench_readme": HF + "trajs/rlbench/README.md (RLBench motion-planning demos)",
    "rlbench_tasks": RV + "roboverse_pack/tasks/rlbench/",
    "calvin_converter": RV + "roboverse_pack/tasks/calvin/data_preparation/convert_data_batch.py "
                             "(scene_obs -> table joints, red/blue/pink blocks in fixed order; robot_obs)",
    "calvin_robot_base": RV + "roboverse_pack/tasks/calvin/base_table.py (default_position -0.34, -0.46, 0.24)",
    "calvin_scenes": CALVIN_ENV + "conf/scene/calvin_scene_{A,B,C,D}.yaml (movable_objects order, "
                                  "global_scaling 0.8, block URDF sizes)",
    "calvin_tcp": CALVIN_ENV + "conf/robot/panda_longer_finger.yaml (tcp_link_id 15 = link 'tcp')",
    "calvin_freq": CALVIN_ENV + "conf/env/play_table_env.yaml (control_freq 30)",
    "calvin_width": CALVIN_ENV + "calvin_env/robot/robot.py (gripper_opening_width = sum of finger joints)",
}
JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
TO_CONTRACT = np.diag([-1.0, -1.0, 1.0])
WIDTH_MAX = 0.08
FRANKA_URDF = "robots/franka/urdf/franka_panda.urdf"
FRANKA_MJCF = "robots/franka/mjcf/panda.xml"
CALVIN_URDF = "robots/franka_calvin/panda_longer_finger.urdf"
ANN_DICT = "trajs/calvin/calvin_traj_ann/ann_dict.npy"

# ---------------------------------------------------------------- RLBench constants
RLBENCH_TABLE_HEIGHT = 0.75     # collect_demo.py: object z - 0.75 ("the height of the table in RLBench")
RLBENCH_DT = 0.05               # one CoppeliaSim step (RLBench/PyRep default 50 ms) per recorded state
RLBENCH_TABLE_CROP_MARGIN = 0.10
# Calibration of the RoboVerse RLBench robot states (derived here, see docs/sources.md): with
# the recorded root quaternion and joint values, objects that RLBench holds by parenting
# (exactly rigid in the source) drift by 18 mm (median; p90 55 mm) relative to the hand and
# sit up to 3 cm off the closing axis. Replacing the root rotation by the identity (RoboVerse's
# hand-tuned quaternion correction leaves a 0.3 deg tilt and 0.7 deg yaw) and adding 0.0698 rad
# to panda_joint4 (the CoppeliaSim Panda joint-4 zero differs from franka_ros by the joint-4
# upper limit) makes held objects rigid to 1.25 mm (median; p90 2.9 mm, 171 holds, 12 tasks)
# and centred on the closing axis (|offset| p90 3 mm).
RLBENCH_JOINT4_OFFSET = 0.0698
MOVED_M = 0.01                  # an entity moving more than this is "manipulated"
# PhysX default material (RoboVerse sets no friction on RLBench objects; IsaacSim default
# 0.5/0.5/0); the CoppeliaSim parameters of the original recording are not in the data.
RLBENCH_MATERIAL = {"static_friction": 0.5, "dynamic_friction": 0.5, "restitution": 0.0, "density": 1000.0}
RLBENCH_ASSUMPTIONS = [
    "world = RoboVerse RLBench world shifted +0.75 m in z (RLBench floor z = 0, table top z = 0.75)",
    "table not in the RoboVerse data: synthesized static box from the floor to z = 0.75, cropped in x/y to the "
    "objects' footprint + 0.10 m",
    f"time step {RLBENCH_DT} s per recorded state (RLBench / CoppeliaSim default scene dt; not recorded in the file)",
    "grasps in the source are RLBench object parenting (kinematic attachment), not friction",
    "tier-P parameters: MetaSim primitive default mass 0.1 kg, PhysX default material 0.5/0.5/0; the CoppeliaSim "
    "parameters of the recording are unknown",
]

# ---------------------------------------------------------------- CALVIN constants
CALVIN_FREQ_HZ = 30.0
CALVIN_SCALING = 0.8
CALVIN_ROBOT_BASE = (-0.34, -0.46, 0.24)
# scene -> CALVIN movable_objects order (= scene_obs order) and block URDF sizes (unscaled, metres)
CALVIN_BLOCK_ORDER = {"A": ("pink", "blue", "red"), "B": ("red", "blue", "pink"),
                      "C": ("blue", "red", "pink"), "D": ("red", "blue", "pink")}
CALVIN_BLOCK_URDF = {"A": {"pink": "small", "blue": "big", "red": "middle"},
                     "B": {"red": "small", "blue": "big", "pink": "middle"},
                     "C": {"blue": "small", "red": "big", "pink": "middle"},
                     "D": {"red": "middle", "blue": "small", "pink": "big"}}
CALVIN_BLOCK_SIZE = {"small": (0.05, 0.05, 0.05), "middle": (0.07, 0.05, 0.05), "big": (0.1, 0.05, 0.05)}
ROBOVERSE_BLOCK_ORDER = ("red", "blue", "pink")   # convert_data_batch.py: scene_obs[6:12], [12:18], [18:24]
CALVIN_TABLE_JOINTS = ["base__slide", "base__drawer", "base__button", "base__switch"]
# AABB of calvin_table_<X>/meshes/base_link.obj (desk + cabinet, link origin at the table pose,
# identical for A-D) x global_scaling 0.8: x +-0.44, y +-0.18, z 0..0.6478 m.
CALVIN_TABLE_AABB = {"center": [0.0, 0.0, 0.3239], "half_extents": [0.44, 0.18, 0.3239]}


def _rotation_wxyz(q) -> np.ndarray:
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


def pose_to_matrix(p) -> np.ndarray:
    """(..., 7) xyz + wxyz -> (..., 4, 4)."""
    p = np.asarray(p, float)
    T = np.broadcast_to(np.eye(4), p.shape[:-1] + (4, 4)).copy()
    T[..., :3, :3] = _rotation_wxyz(p[..., 3:7])
    T[..., :3, 3] = p[..., :3]
    return T


# ---------------------------------------------------------------- grasp-center models

def mjcf_pad_center(mjcf_path) -> np.ndarray:
    """Center of ``fingertip_pad_collision_1`` in the left finger frame (menagerie panda.xml)."""
    root = ET.parse(mjcf_path).getroot()
    for d in root.iter("default"):
        if d.get("class") == "fingertip_pad_collision_1":
            return np.array([float(v) for v in d.find("geom").get("pos").split()])
    raise ValueError(f"{mjcf_path}: no fingertip_pad_collision_1 default class")


class FrankaGrasp:
    """Panda FK from a URDF with the grasp center either between pad boxes or at a TCP link."""

    def __init__(self, urdf_path, *, pad_center=None, tcp_link=None):
        if (pad_center is None) == (tcp_link is None):
            raise ValueError("give exactly one of pad_center or tcp_link")
        self.urdf_path = Path(urdf_path)
        self.sha256 = sha256_file(self.urdf_path)
        model = URDF(self.urdf_path)
        missing = [j for j in JOINTS if j not in model.by_name]
        if missing:
            raise ValueError(f"{self.urdf_path.name}: missing joints {missing}")
        self.ignored_mimic = sorted(model.mimic)   # both fingers are recorded independently
        model.mimic = {}
        self.limits = np.array([model.limits[j] for j in JOINTS])
        self.tcp_link = tcp_link
        self.pad = None if pad_center is None else np.asarray(pad_center, float)
        targets = ["panda_hand", "panda_leftfinger", "panda_rightfinger"] + ([tcp_link] if tcp_link else [])
        self.tree = KinematicTree(model, "panda_link0", targets, JOINTS)
        q_open = np.r_[np.zeros(7), 0.04, 0.04]
        self.center_in_hand = (np.linalg.inv(self.tree.fk(q_open)["panda_hand"]) @
                               np.r_[self._center(self.tree.fk(q_open[None]))[0], 1])[:3]

    def _center(self, fk) -> np.ndarray:
        if self.tcp_link:
            return fk[self.tcp_link][:, :3, 3]
        # URDF right finger frame = left frame mirrored by the finger axis (y): pad (x, -y, z)
        left = fk["panda_leftfinger"][:, :3, :3] @ self.pad + fk["panda_leftfinger"][:, :3, 3]
        right = fk["panda_rightfinger"][:, :3, :3] @ (self.pad * [-1, -1, 1]) + fk["panda_rightfinger"][:, :3, 3]
        return (left + right) / 2

    def forward(self, q: np.ndarray, root: np.ndarray) -> dict:
        """Grasp poses in world for joint positions (T, 9) and root poses (T, 4, 4)."""
        fk = self.tree.fk(q)
        H = root @ fk["panda_hand"]
        G = np.array(H)
        G[:, :3, :3] = H[:, :3, :3] @ TO_CONTRACT
        c = self._center(fk)
        G[:, :3, 3] = np.einsum("tij,tj->ti", root[:, :3, :3], c) + root[:, :3, 3]
        return {"grasp": G, "hand": H, "width": np.clip(q[:, 7] + q[:, 8], 0.0, None)}

    def describe(self) -> dict:
        return {"urdf": str(self.urdf_path.name), "urdf_sha256": self.sha256, "joint_order": JOINTS,
                "palm_link": "panda_hand", "grasp_center": (f"link {self.tcp_link}" if self.tcp_link else
                                                           "midpoint of fingertip_pad_collision_1 of both fingers"),
                "pad_center_in_finger_m": None if self.pad is None else self.pad.tolist(),
                "grasp_center_in_hand_open_m": self.center_in_hand.round(6).tolist(),
                "hand_to_contract_rotation": TO_CONTRACT.tolist(), "ignored_mimic": self.ignored_mimic,
                "width": "q_finger1 + q_finger2 (m)", "width_max_m": WIDTH_MAX}


@lru_cache(maxsize=4)
def _franka(urdf: str, mjcf: str) -> FrankaGrasp:
    return FrankaGrasp(urdf, pad_center=mjcf_pad_center(mjcf))


@lru_cache(maxsize=4)
def _calvin_robot(urdf: str) -> FrankaGrasp:
    return FrankaGrasp(urdf, tcp_link="tcp")


def fk_crosscheck_mujoco(gr: FrankaGrasp, q: np.ndarray, root: np.ndarray, grasp: np.ndarray,
                         mjcf_path=None) -> dict:
    """Independent grasp-center FK with MuJoCo (optional dependency).

    With ``mjcf_path`` (RLBench), RoboVerse's own MJCF Panda is compiled (meshes stripped) and
    the world centers of the two ``fingertip_pad_collision_1`` geoms are averaged; otherwise
    the same URDF is imported by MuJoCo and the TCP link is compared.
    """
    try:
        import mujoco
    except ImportError:
        return {"available": False, "reason": "mujoco is not installed"}
    if mjcf_path is not None:
        r = ET.parse(mjcf_path).getroot()
        for parent in list(r.iter()):
            for child in list(parent):
                if child.tag == "mesh" or (child.tag == "geom" and child.get("class") in ("visual", "collision")
                                           and child.get("mesh")) or child.tag in ("material", "texture"):
                    parent.remove(child)
        for g in r.iter("geom"):
            g.attrib.pop("material", None)
        for tag in ("actuator", "keyframe", "tendon", "equality", "contact"):
            for el in r.findall(tag):
                r.remove(el)
        m = mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))
        pads = [i for i in range(m.ngeom) if np.allclose(m.geom_size[i], [0.0085, 0.004, 0.0085])]
        if len(pads) != 2:
            return {"available": False, "reason": f"expected 2 pad geoms, found {len(pads)}"}
        source = f"{Path(mjcf_path).name} pad geoms (menagerie Panda, independent of the URDF)"
    else:
        r = ET.parse(gr.urdf_path).getroot()
        for link in r.findall("link"):
            for tag in ("visual", "collision"):
                for e in link.findall(tag):
                    link.remove(e)
        ET.SubElement(ET.SubElement(r, "mujoco"), "compiler", fusestatic="false")
        m = mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))
        pads = None
        source = f"{gr.urdf_path.name} imported by MuJoCo, link {gr.tcp_link}"
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in JOINTS]
    dp = 0.0
    for t in range(len(q)):
        d.qpos[adr] = q[t]
        mujoco.mj_kinematics(m, d)
        c = d.geom_xpos[pads].mean(0) if pads is not None else d.xpos[m.body(gr.tcp_link).id]
        c = root[t, :3, :3] @ c + root[t, :3, 3]
        dp = max(dp, float(np.linalg.norm(c - grasp[t, :3, 3])))
    return {"available": True, "library": f"mujoco {mujoco.__version__}", "reference": source, "frames": len(q),
            "max_grasp_center_error_m": dp}


# ---------------------------------------------------------------- shared helpers

def _entity(states, name, T, z_offset=0.0):
    pose = np.array([list(map(float, s[name]["pos"])) + list(map(float, s[name]["rot"])) for s in states], float)
    if pose.shape != (T, 7) or not np.all(np.isfinite(pose)):
        raise ValueError(f"entity {name}: bad pose array {pose.shape}")
    pose[:, 2] += z_offset
    pose[:, 3:] /= np.linalg.norm(pose[:, 3:], axis=1, keepdims=True)
    return pose


def _dofs(states, name) -> tuple[list[str], np.ndarray]:
    names = list(states[0][name].get("dof_pos") or {})
    return names, np.array([[float(s[name]["dof_pos"][j]) for j in names] for s in states], float)


def _robot_arrays(states, rob, T, z_offset):
    q = np.array([[float(s[rob]["dof_pos"][j]) for j in JOINTS] for s in states], float)
    if q.shape != (T, 9) or not np.all(np.isfinite(q)):
        raise ValueError(f"robot {rob}: bad joint array {q.shape}")
    return q, pose_to_matrix(_entity(states, rob, T, z_offset))


def _limit_violation(gr, q) -> float:
    return float(np.max(np.maximum(gr.limits[:, 0] - q, q - gr.limits[:, 1]).clip(0)))


def _yaw(R) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def held_object_check(grasp: np.ndarray, width: np.ndarray, objects: dict, lift_m=0.02, open_width=0.075,
                      max_dist=0.15) -> dict:
    """Relative pose of each lifted object in the grasp frame while the gripper holds it.

    Frames: the object is more than ``lift_m`` above its first pose, the gripper is closed
    (width < ``open_width``) and the object center is within ``max_dist`` of the grasp center;
    the first contiguous run of such frames is used. A held object stays near the grasp center
    (closing-axis offset about zero) with a nearly constant relative pose. Reports per object
    the frames used, the median offset in the grasp frame and the spread (max - min) per axis.
    """
    out = {}
    for oid, o in objects.items():
        p = o.pose
        near = np.linalg.norm(p[:, :3] - grasp[:, :3, 3], axis=1) < max_dist
        run = np.flatnonzero(o.valid & near & (p[:, 2] > p[o.valid][0, 2] + lift_m) & (width < open_width))
        breaks = np.flatnonzero(np.diff(run) > 1)
        run = run[:breaks[0] + 1] if len(breaks) else run
        if len(run) < 3:
            continue
        rel = np.linalg.inv(grasp[run]) @ pose_to_matrix(p[run])
        rp = rel[:, :3, 3]
        rot = Rotation.from_matrix(rel[:, :3, :3])
        out[oid] = {"frames": int(len(run)), "first_frame": int(run[0]),
                    "median_offset_m": np.median(rp, 0).round(5).tolist(), "spread_m": np.ptp(rp, 0).round(5).tolist(),
                    "rotation_spread_rad": round(float((rot * rot[0].inv()).magnitude().max()), 5)}
    return out


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=lambda a: np.asarray(a).tolist()).encode()).hexdigest()


def _locate(rel: str, path: Path, root, catalog) -> Path | None:
    if root is None:
        root = next((p.parent for p in path.resolve().parents if p.name == "raw"), None)
    if root is None:
        return None
    e = catalog.get(f"roboverse/{rel}")
    local = e.local_path(root) if e else Path(root) / "raw" / "roboverse" / rel
    return local if local.exists() else None


# ---------------------------------------------------------------- reader

@register("roboverse")
def read_roboverse(path: Path, *, family: str, demos=None, limit=None, root=None, catalog=None,
                   robot_files=None, crosscheck: bool = False, rlbench_calibration: bool = True):
    """Yield one :class:`SourceEpisode` per episode of a RoboVerse v2 trajectory file.

    ``demos`` selects episode indices, ``limit`` stops after that many. ``robot_files`` maps
    catalog-relative robot file paths (``FRANKA_URDF``, ``FRANKA_MJCF``, ``CALVIN_URDF``,
    ``ANN_DICT``) to local paths; by default they are looked up under the data root
    inferred from ``path``. ``crosscheck`` adds an independent MuJoCo FK comparison.
    ``rlbench_calibration`` applies :data:`RLBENCH_JOINT4_OFFSET` and an identity root rotation to
    RLBench robot states before FK (default; ``False`` uses the recorded values as they are).
    """
    path = Path(path)
    catalog = catalog if catalog is not None else load_catalog()
    digest = sha256_file(path)
    entry = next((e for e in catalog.values() if e.sha256 == digest), None)
    rel = entry.path if entry else _relative(path)
    files = dict(robot_files or {})

    def need(name):
        if name not in files:
            files[name] = _locate(name, path, root, catalog)
        if files[name] is None:
            raise FileNotFoundError(f"{name} not found; fetch roboverse/{name} or pass robot_files=")
        return Path(files[name])

    data = load_pickle(path)
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"{path.name}: expected one robot key, got {list(data) if isinstance(data, dict) else type(data)}")
    (robot_key, episodes), = data.items()
    idx = list(range(len(episodes))) if demos is None else [int(i) for i in demos]
    if limit is not None:
        idx = idx[:limit]
    m_rl = re.search(r"trajs/rlbench/([^/]+)/v2/([^/]+)_v2\.pkl\.gz$", rel)
    m_cv = re.search(r"trajs/calvin/calvin_traj_ann/env_([A-D])(_val)?_out/task_(\d+)_v2\.pkl$", rel)
    common = {"file": str(path), "sha256": digest, "url": entry.url if entry else None,
              "revision": entry.revision if entry else None, "catalog_id": entry.id if entry else None,
              "catalog_path": rel, "robot_key": robot_key, "episodes_in_file": len(episodes),
              "format": "MetaSim v2 states (pos, rot wxyz, dof_pos)", "evidence": EVIDENCE}
    if m_rl:
        if robot_key != "franka":
            raise ValueError(f"{rel}: only the franka RLBench demonstrations are supported, got {robot_key!r}")
        gr = _franka(str(need(FRANKA_URDF)), str(need(FRANKA_MJCF)))
        for i in idx:
            yield _rlbench_episode(episodes[i], i, m_rl.group(1), gr, family, entry, common, files, crosscheck,
                                  rlbench_calibration)
    elif m_cv:
        scene, val, n = m_cv.group(1), bool(m_cv.group(2)), int(m_cv.group(3))
        gr = _calvin_robot(str(need(CALVIN_URDF)))
        ann = {v: k for k, v in load_npy_object(need(ANN_DICT)).items()}
        for i in idx:
            yield _calvin_episode(episodes[i], i, scene, val, n, ann.get(n), gr, family, entry, common, crosscheck)
    else:
        raise ValueError(f"{rel}: not a supported RoboVerse trajectory (RLBench franka_v2.pkl.gz or "
                         "CALVIN calvin_traj_ann task_<N>_v2.pkl)")


def _relative(path: Path) -> str:
    parts = path.resolve().parts
    return "/".join(parts[parts.index("trajs"):]) if "trajs" in parts else path.name


def _rlbench_episode(ep, i, task, gr, family, entry, common, files, crosscheck, calibrate=True):
    states = ep["states"]
    T = len(states)
    time = np.arange(T) * RLBENCH_DT
    q_rec, root_rec = _robot_arrays(states, "franka", T, RLBENCH_TABLE_HEIGHT)
    q, root = q_rec.copy(), root_rec.copy()
    if calibrate:
        q[:, 3] += RLBENCH_JOINT4_OFFSET
        root[:, :3, :3] = np.eye(3)
    poses = gr.forward(q, root)
    eff = {"franka": Effector(pose=poses["grasp"], width=poses["width"],
                              opening=np.clip(poses["width"] / WIDTH_MAX, 0, 1))}
    table = RLBENCH_OBJECTS.get(task, {})
    objects, arts, markers, geometry_status, physics = {}, {}, {}, {}, {}
    for name in sorted(n for n in states[0] if n != "franka"):
        pose = _entity(states, name, T, RLBENCH_TABLE_HEIGHT)
        moved = float(np.linalg.norm(pose[:, :3] - pose[0, :3], axis=1).max())
        phys, geom = table.get(name, (None, None))
        physics[name] = phys
        if states[0][name].get("dof_pos"):
            joints, qpos = _dofs(states, name)
            arts[name] = Articulation(joint_names=[f"{name}/{j}" for j in joints], qpos=qpos)
        if phys == "XFORM" and moved < MOVED_M:
            markers[name] = {"pose_t0": pose[0].round(6).tolist(), "geometry": _geometry(geom, name)}
            continue
        g = _geometry(geom, name)
        geometry_status[name] = ("primitive (RoboVerse task config)" if g.get("kind") in ("box", "sphere", "cylinder")
                                 else "mesh asset reference" if g.get("kind") == "mesh" else
                                 "unknown: object not in the RoboVerse task config")
        role = "manipulated" if moved > MOVED_M else "fixture"
        objects[name] = ObjectTrack(pose=pose, valid=np.ones(T, bool), role=role, geometry=g)
    # synthesized table (RoboVerse models the RLBench table top as its ground plane)
    xy = np.concatenate([o.pose[:, :2] for o in objects.values()] + [poses["grasp"][:, :2, 3]])
    c = xy.mean(0)
    table_geom = {"kind": "box", "frame": "body", "center": [0.0, 0.0, -RLBENCH_TABLE_HEIGHT / 2],
                  "half_extents": [1.5, 1.5, RLBENCH_TABLE_HEIGHT / 2], "body": "table",
                  "note": "synthesized: RLBench table top at z = 0.75 (RoboVerse ground plane); extent unknown, "
                          "cropped to the objects' footprint"}
    tpose = np.tile(np.r_[c, RLBENCH_TABLE_HEIGHT, 1.0, 0.0, 0.0, 0.0], (T, 1))
    objects["table"] = ObjectTrack(pose=tpose, valid=np.ones(T, bool), role="support", geometry=table_geom)
    objects, adaptations = primitive_scene.crop_supports(objects, RLBENCH_TABLE_CROP_MARGIN, ids=["table"])
    scene, scene_note = _rlbench_scene(objects, physics)
    held = {k: v for k, v in objects.items() if v.role == "manipulated"}
    check = {"max_qpos_limit_violation": _limit_violation(gr, q_rec),
             "root_motion_m": float(np.ptp(root[:, :3, 3], axis=0).max()),
             "held_objects": held_object_check(poses["grasp"], poses["width"], held)}
    if calibrate:
        raw = gr.forward(q_rec, root_rec)
        check["held_objects_uncalibrated"] = held_object_check(raw["grasp"], raw["width"], held)
        check["calibration_grasp_shift_max_m"] = float(np.linalg.norm(raw["grasp"][:, :3, 3] - poses["grasp"][:, :3, 3],
                                                                      axis=1).max())
    if crosscheck:
        check["fk_crosscheck"] = fk_crosscheck_mujoco(gr, q, root, poses["grasp"], mjcf_path=files.get(FRANKA_MJCF))
    R0 = root[0, :3, :3]
    base_hint = np.array([root[0, 0, 3], root[0, 1, 3], _yaw(R0)])
    provenance = {
        **common, "benchmark": "rlbench", "task": task, "episode_index": i, "episode_digest": _digest(states),
        "robot": "franka (RoboVerse IsaacSim Franka placed to match the CoppeliaSim Panda)",
        "robot_root_world": root[0, :3, 3].round(6).tolist(), "robot_base_z_m": float(root[0, 2, 3]),
        "robot_root_quat_recorded_wxyz": np.asarray(states[0]["franka"]["rot"], float).round(6).tolist(),
        "calibration": ({"applied": True, "panda_joint4_offset_rad": RLBENCH_JOINT4_OFFSET,
                         "root_rotation": "identity (recorded quaternion replaced)",
                         "evidence": "held objects (parented in RLBench) become rigid in the grasp frame; "
                                     "see state_checks held_objects vs held_objects_uncalibrated"}
                        if calibrate else {"applied": False}),
        "effectors": {"franka": gr.describe()}, "state_checks": check,
        "time_source": f"state index x {RLBENCH_DT} s (assumed RLBench scene dt)",
        "world_frame": "RoboVerse RLBench world + 0.75 m in z (floor z = 0, table top z = 0.75)",
        "world_z_offset_m": RLBENCH_TABLE_HEIGHT, "assumptions": RLBENCH_ASSUMPTIONS,
        "objects_physics_type": physics, "goal_markers": markers, "geometry_status": geometry_status,
        "role_rule": f"manipulated if the object moves more than {MOVED_M} m in the episode, else fixture",
        "actions": f"{len(ep.get('actions') or [])} dof_pos_target rows (joint targets = recorded joint positions)",
        "state_route": "kinematic: CoppeliaSim recording (RLBench motion planner, parenting grasps), not re-simulated",
        "success_source": "RLBench keeps only demos whose task success condition was met (demo generation)",
        "scene": scene_note, "scene_adaptations": adaptations,
        "scene_physical": None if scene is None else primitive_scene.scene_provenance(scene)}
    lineage = {"generated": False, "human": False, "source_type": "motion_planning", "upstream": "RLBench",
               "upstream_note": "RoboVerse 'preview' subset of RLBench demos re-collected with RLBench's planner; "
                                "independent of the original RLBench dataset files"}
    return SourceEpisode(
        family=family, dataset=entry.dataset if entry and entry.dataset else f"roboverse/rlbench/{task}",
        episode_id=f"ep{i:03d}", task=f"rlbench/{task}", time=time, effectors=eff, objects=objects,
        base_hint=base_hint, articulations=arts, scene=scene, instruction=None, success=True,
        regime="tabletop", license=entry.license if entry else "unknown", provenance=provenance, lineage=lineage)


def _geometry(geom, name) -> dict:
    if geom is None:
        return {}
    kind = geom[0]
    if kind == "box":
        return {"kind": "box", "half_extents": list(map(float, geom[1])), "frame": "body", "body": name}
    if kind == "sphere":
        return {"kind": "sphere", "radius": float(geom[1]), "frame": "body", "body": name}
    if kind == "cylinder":
        return {"kind": "cylinder", "radius": float(geom[1]), "half_length": float(geom[2]) / 2, "axis": "z",
                "frame": "body", "body": name}
    return {"kind": "mesh", "asset": f"roboverse_data/{geom[1]}", "format": "usd", "body": name,
            "note": "converted RLBench asset (RLBench licence)"}


def _rlbench_scene(objects, physics):
    phys = {}
    for oid, o in objects.items():
        if oid == "table":
            phys[oid] = {"body_type": "static", "source": "synthesized table"}
            continue
        if o.geometry.get("kind") not in ("box", "sphere", "cylinder"):
            return None, f"none: object {oid!r} is not a primitive ({o.geometry.get('kind') or 'unknown geometry'})"
        if physics.get(oid) in ("GEOM", "XFORM") or o.role != "manipulated" and physics.get(oid) != "RIGIDBODY":
            phys[oid] = {"body_type": "static", "source": f"RoboVerse physics {physics.get(oid)}"}
        else:
            g = primitive_scene.geoms_of(oid, o.geometry)
            vol = sum(primitive_scene._volume(x) for x in g)
            phys[oid] = {"body_type": "dynamic", "density": PRIMITIVE_DEFAULT_MASS_KG / vol,
                         "source": "MetaSim primitive default mass 0.1 kg"}
    try:
        ref = primitive_scene.build(objects, floor=True, physical={
            "source": {"roboverse_commit": ROBOVERSE_COMMIT, "assumptions": RLBENCH_ASSUMPTIONS},
            "defaults": RLBENCH_MATERIAL, "floor": {"z": 0.0, "source": "RLBench floor"},
            "option": {"timestep": 0.005, "gravity": [0.0, 0.0, -9.81]}, "objects": phys})
    except primitive_scene.UnrepresentableObject as e:
        return None, f"none: {e}"
    return ref, "primitive_scene: primitives + synthesized table (see provenance['scene_physical'])"


def _calvin_episode(ep, i, scene, val, task_index, sentence, gr, family, entry, common, crosscheck):
    states = ep["states"]
    T = len(states)
    time = np.arange(T) / CALVIN_FREQ_HZ
    q, root = _robot_arrays(states, "franka", T, 0.0)
    poses = gr.forward(q, root)
    eff = {"franka": Effector(pose=poses["grasp"], width=poses["width"],
                              opening=np.clip(poses["width"] / WIDTH_MAX, 0, 1))}
    remap = dict(zip(ROBOVERSE_BLOCK_ORDER, CALVIN_BLOCK_ORDER[scene]))   # RoboVerse colour -> true colour
    objects, arts = {}, {}
    for rv_colour, colour in remap.items():
        pose = _entity(states, f"{rv_colour}_cube", T)
        size = np.array(CALVIN_BLOCK_SIZE[CALVIN_BLOCK_URDF[scene][colour]]) * CALVIN_SCALING
        moved = float(np.linalg.norm(pose[:, :3] - pose[0, :3], axis=1).max())
        oid = f"block_{colour}"
        objects[oid] = ObjectTrack(pose=pose, valid=np.ones(T, bool), role="manipulated" if moved > MOVED_M else "fixture",
                                   geometry={"kind": "box", "half_extents": (size / 2).round(6).tolist(), "frame": "body",
                                             "body": oid, "roboverse_entity": f"{rv_colour}_cube",
                                             "asset": f"calvin blocks/block_{colour}_{CALVIN_BLOCK_URDF[scene][colour]}.urdf "
                                                      f"x {CALVIN_SCALING}"})
    joints, qpos = _dofs(states, "table")
    tpose = _entity(states, "table", T)
    objects["table"] = ObjectTrack(pose=tpose, valid=np.ones(T, bool), role="fixture",
                                   geometry={"kind": "aabb", "frame": "body", **CALVIN_TABLE_AABB,
                                             "asset": f"roboverse_data/assets/calvin/calvin_table_{scene}/urdf/"
                                                      f"calvin_table_{scene}.urdf", "format": "urdf",
                                             "scale": CALVIN_SCALING, "body": "table",
                                             "note": "articulated desk (slide, drawer, button, switch); geometry is the "
                                                     "AABB of the scaled base_link collision mesh, moving parts excluded"})
    arts["table"] = Articulation(joint_names=[f"table/{j}" for j in joints], qpos=qpos)
    src = (ep.get("env_meta") or {}).get("source_dir", "")
    m = re.search(r"episode_(\d+)_(\d+)_(\d+)_(\d+)$", src)
    frames = [int(m.group(3)), int(m.group(4))] if m else None
    check = {"max_qpos_limit_violation": _limit_violation(gr, q),
             "root_motion_m": float(np.ptp(root[:, :3, 3], axis=0).max()),
             "min_raw_width_m": float(np.min(q[:, 7] + q[:, 8])),
             "held_objects": held_object_check(poses["grasp"], poses["width"],
                                               {k: v for k, v in objects.items() if v.role == "manipulated"})}
    if crosscheck:
        check["fk_crosscheck"] = fk_crosscheck_mujoco(gr, q, root, poses["grasp"])
    env = f"env_{scene}" + ("_val" if val else "")
    provenance = {
        **common, "benchmark": "calvin", "calvin_scene": scene, "split": "validation" if val else "training",
        "task_index": task_index, "episode_index": i, "episode_digest": _digest(states),
        "source_dir": src, "calvin_frames": frames, "annotation_index": int(m.group(2)) if m else None,
        "effectors": {"franka": gr.describe()}, "state_checks": check,
        "time_source": f"state index / {CALVIN_FREQ_HZ} Hz (CALVIN control_freq)",
        "world_frame": "CALVIN PyBullet world (floor plane at z = 0, robot base at (-0.34, -0.46, 0.24))",
        "block_colour_remap": {f"{k}_cube": f"block_{v}" for k, v in remap.items()},
        "block_remap_rule": "RoboVerse assumes scene_obs order red, blue, pink; CALVIN uses the scene config order",
        "init_state_note": "init_state is RoboVerse's PyBullet reset state, not a recorded frame; unused",
        "state_route": "kinematic: converted CALVIN robot_obs/scene_obs (human VR teleoperation in PyBullet)",
        "success_source": "derived: CALVIN language windows are labelled where CALVIN's task oracle detected the task",
        "scene": "none: CALVIN table is an articulated mesh (calvin_table URDF), not primitive geometry",
        "light_states": "dropped by RoboVerse (scene_obs lightbulb / green light not converted)"}
    lineage = {"generated": False, "human": True, "source_type": "teleoperation", "upstream": "CALVIN",
               "stream": f"calvin/{env}", "frames": frames,
               "independence": "windows are crops of one continuous play stream; windows of the same stream "
                               "with overlapping frames are not independent demonstrations"}
    return SourceEpisode(
        family=family, dataset=entry.dataset if entry and entry.dataset else f"roboverse/calvin/{env}",
        episode_id=f"task{task_index}_ep{i:03d}", task=f"calvin/{sentence or task_index}", time=time, effectors=eff,
        objects=objects, base_hint=np.array([root[0, 0, 3], root[0, 1, 3], _yaw(root[0, :3, :3])]),
        articulations=arts, scene=None, instruction=sentence, success=True, regime="tabletop",
        license=entry.license if entry else "unknown", provenance=provenance, lineage=lineage)


__all__ = ["read_roboverse", "FrankaGrasp", "held_object_check", "fk_crosscheck_mujoco", "pose_to_matrix",
           "CALVIN_BLOCK_ORDER", "RLBENCH_TABLE_HEIGHT", "mjcf_pad_center"]
