"""Adapter for ManiSkill3 demonstrations (``haosulab/ManiSkill_Demonstrations``).

Each ``trajectory*.h5`` holds one group ``traj_<id>`` per episode with ``actions`` (T rows)
and ``env_states`` (T + 1 rows, the terminal state included):

* ``env_states/actors/<name>``: 13 values = position, quaternion (w, x, y, z), linear and
  angular velocity (``Actor.get_state``).
* ``env_states/articulations/<name>``: root position, root quaternion (w, x, y, z), root
  linear and angular velocity, then ``qpos`` and ``qvel`` of the active joints
  (``Articulation.get_state``). For the Panda: ``panda_joint1..7``,
  ``panda_finger_joint1``, ``panda_finger_joint2`` (31 values).

The sibling ``*.json`` carries the env id, controller, source type and per-episode seed,
step count and success flag. The demos are recorded with ``obs_mode=none``: no TCP pose,
no images. Nothing here imports SAPIEN or ManiSkill.

Grasp center: forward kinematics of the pinned Panda URDF (``panda_v2.urdf`` for robot
uid ``panda``, ``panda_v3.urdf`` for ``panda_wristcam``; kinematically identical) from the
recorded joint positions and root pose. As in the robosuite adapter the contract frame is
measured from the model with the fingers open: +z = palm (``panda_hand`` origin) ->
midpoint of the two fingertip pad boxes, +y = finger-1 pad (``panda_leftfinger``) ->
finger-2 pad (``panda_rightfinger``), origin = that pad midpoint, a fixed offset in the
hand frame. For the Panda this is ``R_contract = R_hand @ diag(-1, -1, 1)`` and a grasp
center 0.25 mm beyond ``panda_hand_tcp``. Width = pad separation along +y minus the
closed separation (= ``q_finger1 + q_finger2``, 0-0.08 m); opening = width / 0.08.

World frame: ManiSkill's origin is on the table top (ground at z = -0.9196429). Following the
SourceEpisode contract (floor at z = 0), robot roots, actor tracks, goal markers and the scene
are translated by ``WORLD_Z_OFFSET`` = +0.9196429 m in z (recorded in the provenance).

Scene: SAPIEN scenes of the catalogued tasks are primitives only, so a MuJoCo ``SceneRef`` is
built with ``primitive_scene`` (floor, table and kinematic fixtures as static bodies, dynamic
actors as free bodies with their densities, ManiSkill default PhysX material) when every
actor's geometry is known; otherwise ``scene`` is ``None`` and ``provenance["scene"]`` says
why. Physical constants and their sources: ``PHYSICAL`` and ``PHYSICAL_SOURCES``.
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from ..acquire import load_catalog, sha256_file
from ..robot.urdf import URDF, KinematicTree
from ..schema.source import Effector, ObjectTrack, SourceEpisode
from . import primitive_scene
from .registry import register

MANISKILL_COMMIT = "baab60ede2e89167c1b7aaed41a9aa8e690a9d1e"
SOURCE = f"https://github.com/haosulab/ManiSkill/blob/{MANISKILL_COMMIT}/mani_skill/"
EVIDENCE = {
    "actor_state": SOURCE + "utils/structs/actor.py#L132-L140",
    "articulation_state": SOURCE + "utils/structs/articulation.py#L259-L265",
    "panda_agent": SOURCE + "agents/robots/panda/panda.py",
    "sim_config_defaults": SOURCE + "utils/structs/types.py#L72-L82",
    "table": SOURCE + "utils/scene_builder/table/scene_builder.py#L19-L41",
    "tasks": SOURCE + "envs/tasks/tabletop/",
}
PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
ROBOT_URDF = {"panda": "panda_v2.urdf", "panda_wristcam": "panda_v3.urdf"}
NON_GRIPPER_ROBOTS = {"panda_stick": "stick end effector, no gripper",
                      "anymal_c": "quadruped, no gripper"}
# ManiSkill SimConfig defaults (sim 100 Hz, control 20 Hz); no catalogued task overrides
# them and the JSON files do not record them.
DEFAULT_SIM_FREQ, DEFAULT_CONTROL_FREQ = 100, 20


def _box(*half):
    return {"kind": "box", "half_extents": [float(h) for h in half], "frame": "body"}


def _parts(*parts, note=None):
    """Union of boxes ``(center_xyz, half_extents)`` in the body frame, plus its AABB."""
    c = np.array([p[0] for p in parts], float)
    h = np.array([p[1] for p in parts], float)
    lo, hi = (c - h).min(0), (c + h).max(0)
    out = {"kind": "boxes", "frame": "body",
           "boxes": [{"center": list(map(float, a)), "half_extents": list(map(float, b))} for a, b in parts],
           "aabb": {"center": ((lo + hi) / 2).round(6).tolist(), "half_extents": ((hi - lo) / 2).round(6).tolist()}}
    if note:
        out["note"] = note
    return out


CUBE = _box(0.02, 0.02, 0.02)
PEG_FIXED = _box(0.12, 0.025, 0.025)
TABLE = {"kind": "box", "frame": "body", "center": [0.0, 0.0, 0.9196429 / 2],
         "half_extents": [2.418 / 2, 1.209 / 2, 0.9196429 / 2],
         "note": "TableSceneBuilder collision box; the table top is the ManiSkill plane z = 0, i.e. "
                 "z = 0.9196429 after the WORLD_Z_OFFSET translation"}


def _charger():
    base, peg, gap = (0.02, 0.015, 0.012), (8e-3, 0.75e-3, 3.2e-3), 7e-3
    return _parts(((-base[0], 0, 0), base), ((peg[0], gap, 0), peg), ((peg[0], -gap, 0), peg),
                  note="charger: base box and two prongs along +x")


def _receptacle():
    peg, rec, gap = (8e-3, 0.75e-3, 3.2e-3), (1e-2, 5e-2, 5e-2), 7e-3
    sy, sz = 0.5 * (rec[1] - peg[1] - gap), 0.5 * (rec[2] - peg[2])
    dx, dy, dz = -rec[0], peg[1] + gap + sy, peg[2] + sz
    return _parts(((dx, 0, dz), (rec[0], rec[1], sz)), ((dx, 0, -dz), (rec[0], rec[1], sz)),
                  ((dx, dy, 0), (rec[0], sy, rec[2])), ((dx, -dy, 0), (rec[0], sy, rec[2])),
                  ((-rec[0], 0, 0), (rec[0], gap - peg[1], peg[2])),
                  note="receptacle: wall with two prong slots (kinematic)")


def _l_tool():
    handle, hook, width, height = 0.2, 0.05, 0.05, 0.05
    return _parts(((handle / 2, 0, 0), (handle / 2, width / 2, height / 2)),
                  ((handle - hook / 2, width, 0), (hook / 2, width, height / 2)),
                  note="L-shaped tool: handle along +x, hook toward +y")


# ---------------------------------------------------------------- physical parameters
# Recorded commits of the catalogued JSONs: baab60ed, 652ad935, ecc579b7 (PickCube teleop, older
# repository layout), e77e4ff3, 9d5e0e01, 95ea99d4, ee5f8826. DefaultMaterialsConfig, SceneConfig
# and the per-task actor builders were compared at every commit except ecc579b7 (path not
# present): identical values.
SAPIEN_TAG = "3.0.0b1"  # setup.py pins sapien==3.0.0.b1 at baab60ed
SAPIEN = f"https://github.com/haosulab/SAPIEN/blob/{SAPIEN_TAG}/"
PHYSICAL_SOURCES = {
    "default_material": SOURCE + "utils/structs/types.py#L63-L67 (DefaultMaterialsConfig: static 0.3, "
                                 "dynamic 0.3, restitution 0; applied by envs/sapien_env.py "
                                 "physx.set_default_material)",
    "density": SAPIEN + "python/py_package/wrapper/actor_builder.py (add_*_collision density=1000, "
                        "material=None -> physx.get_default_material())",
    "scene_config": SOURCE + "utils/structs/types.py#L36-L48 (gravity -9.81, contact_offset 0.02, "
                             "rest_offset 0, solver 15 position / 1 velocity iterations, TGS, PCM)",
    "sim_freq": SOURCE + "utils/structs/types.py#L80 (sim_freq 100 Hz)",
    "table": SOURCE + "utils/scene_builder/table/scene_builder.py#L19-L53 (kinematic box, ground plane at "
                      "altitude -table_height = -0.9196429)",
    "ground": SOURCE + "utils/building/ground.py#L18-L44 (static plane, default material)",
    "cube": SOURCE + "utils/building/actors/common.py (build_cube: dynamic, default density and material)",
    "sphere": SOURCE + "utils/building/actors/common.py (build_sphere: dynamic, default density and material)",
    "twocolor_peg": SOURCE + "utils/building/actors/common.py (build_twocolor_peg: one box collision)",
    "l_shape_tool": SOURCE + "envs/tasks/tabletop/pull_cube_tool.py (_build_l_shaped_tool: handle "
                             "density 500, hook default 1000)",
    "charger": SOURCE + "envs/tasks/tabletop/plug_charger.py (charger dynamic, receptacle build_kinematic)",
    "peg_insertion": SOURCE + "envs/tasks/tabletop/peg_insertion_side.py (peg dynamic, box_with_hole "
                              "build_kinematic)",
    "actor_damping": "PhysX SDK PxRigidDynamic defaults (linear 0, angular 0.05); not set by SAPIEN 3.0.0b1",
}
TABLE_HEIGHT = 0.9196429
# ManiSkill's origin is on the table top and its ground plane is at z = -TABLE_HEIGHT. The
# SourceEpisode contract puts the floor at z = 0, so every pose (robot roots -> effectors,
# actors, goal markers) and the scene are translated by +WORLD_Z_OFFSET in z.
WORLD_Z_OFFSET = TABLE_HEIGHT
# The 2.418 x 1.209 m table leaves the objects about 0.5 m from any edge Reachy's base can
# reach. Per episode it is cropped in its own x/y (never in height or top surface) to the
# region covering every object's footprint over the whole episode plus this margin
# (primitive_scene.crop_supports), both for the tier-K footprint and the tier-P scene;
# provenance["scene_adaptations"] records the original and cropped extents.
TABLE_CROP_MARGIN = 0.10
DEFAULT_MATERIAL = {"static_friction": 0.3, "dynamic_friction": 0.3, "restitution": 0.0, "density": 1000.0}


def _dyn(src, density=None):
    return {"body_type": "dynamic", **({"density": density} if density is not None else {}), "source": src}


_STATIC_KINEMATIC = {"body_type": "static", "source": "kinematic actor (never moves in the demos)"}
# Per env: actor -> body type and density (default material everywhere).
PHYSICAL = {
    "PickCube-v1": {"cube": _dyn("cube")},
    "StackCube-v1": {"cubeA": _dyn("cube"), "cubeB": _dyn("cube")},
    "PegInsertionSide-v1": {"peg": _dyn("peg_insertion"), "box_with_hole": {**_STATIC_KINEMATIC, "source": "peg_insertion"}},
    "PlugCharger-v1": {"charger": _dyn("charger"), "receptacle": {**_STATIC_KINEMATIC, "source": "charger"}},
    "PullCube-v1": {"cube": _dyn("cube")},
    "PushCube-v1": {"cube": _dyn("cube")},
    "PullCubeTool-v1": {"l_shape_tool": _dyn("l_shape_tool", [500.0, 1000.0]), "cube": _dyn("cube")},
    "LiftPegUpright-v1": {"peg": _dyn("twocolor_peg")},
    "PokeCube-v1": {"peg": _dyn("twocolor_peg"), "cube": _dyn("cube")},
    "RollBall-v1": {"ball": _dyn("sphere")},
    "StackPyramid-v1": {"cubeA": _dyn("cube"), "cubeB": _dyn("cube"), "cubeC": _dyn("cube")},
    "TwoRobotPickCube-v1": {"cube": _dyn("cube")},
    "TwoRobotStackCube-v1": {"cubeA": _dyn("cube"), "cubeB": _dyn("cube")},
}
STATIC_PHYSICAL = {"table-workspace": {**_STATIC_KINEMATIC, "source": "table"}}
# Render colours of task actors (geom rgba of the stored scene meshes), only where the task source
# sets a constant colour; other actors keep MuJoCo's default grey with rgba null in the provenance.
# The table's source visual is a textured GLB (TableSceneBuilder table.glb, not catalogued): the
# stored table is its collision box (cropped, see TABLE_CROP_MARGIN) in the default grey.
VISUALS = {
    "PickCube-v1": {"cube": {"rgba": [1.0, 0.0, 0.0, 1.0],
                             "source": SOURCE + "envs/tasks/tabletop/pick_cube.py (build_cube color=[1, 0, 0, 1])"}},
    "StackCube-v1": {"cubeA": {"rgba": [1.0, 0.0, 0.0, 1.0],
                               "source": SOURCE + "envs/tasks/tabletop/stack_cube.py (cubeA color=[1, 0, 0, 1])"},
                     "cubeB": {"rgba": [0.0, 1.0, 0.0, 1.0],
                               "source": SOURCE + "envs/tasks/tabletop/stack_cube.py (cubeB color=[0, 1, 0, 1])"}},
}


def scene_for(env_id: str, objects: dict) -> tuple:
    """(SceneRef | None, note) for an episode's objects (ids as produced by the adapter)."""
    table = {**PHYSICAL.get(env_id, {}), **STATIC_PHYSICAL}
    missing = sorted(set(objects) - set(table))
    if env_id not in PHYSICAL or missing:
        return None, f"none: no physical parameters for {missing or env_id}"
    try:
        ref = primitive_scene.build(objects, floor=True, physical={
            "source": {"maniskill_commit": MANISKILL_COMMIT, "sapien": SAPIEN_TAG, **PHYSICAL_SOURCES},
            "defaults": DEFAULT_MATERIAL,
            "floor": {"z": -TABLE_HEIGHT + WORLD_Z_OFFSET, "source": "ground"},
            "option": {"timestep": 1.0 / DEFAULT_SIM_FREQ, "gravity": [0.0, 0.0, -9.81]},
            "objects": {k: v for k, v in table.items() if k in objects}},
            visual={k: v for k, v in VISUALS.get(env_id, {}).items() if k in objects})
    except primitive_scene.UnrepresentableObject as e:
        return None, f"none: {e}"
    return ref, "primitive_scene: SAPIEN scene rebuilt from primitives (see provenance['scene_physical'])"


# Per env: actor -> (role, geometry). Geometry constants are those of the task source at
# the ManiSkill commit recorded in each JSON (checked identical across those commits).
# "markers" are non-colliding goal visuals: kept in provenance, not as objects.
TASKS = {
    "PickCube-v1": {"objects": {"cube": ("manipulated", CUBE)}, "markers": {"goal_site": {"kind": "sphere", "radius": 0.025}}},
    "StackCube-v1": {"objects": {"cubeA": ("manipulated", CUBE), "cubeB": ("support", CUBE)}},
    "PegInsertionSide-v1": {"objects": {"peg": ("manipulated", "peg_insertion_peg"), "box_with_hole": ("receptacle", "peg_insertion_box")}},
    "PlugCharger-v1": {"objects": {"charger": ("manipulated", _charger()), "receptacle": ("receptacle", _receptacle())}},
    "PullCube-v1": {"objects": {"cube": ("manipulated", CUBE)}, "markers": {"goal_region": {"kind": "disc", "radius": 0.1}}},
    "PushCube-v1": {"objects": {"cube": ("manipulated", CUBE)}, "markers": {"goal_region": {"kind": "disc", "radius": 0.1}}},
    "PullCubeTool-v1": {"objects": {"l_shape_tool": ("manipulated", _l_tool()), "cube": ("manipulated", CUBE)}},
    "LiftPegUpright-v1": {"objects": {"peg": ("manipulated", PEG_FIXED)}},
    "PokeCube-v1": {"objects": {"peg": ("manipulated", PEG_FIXED), "cube": ("manipulated", CUBE)},
                    "markers": {"goal_region": {"kind": "disc", "radius": 0.05}}},
    "RollBall-v1": {"objects": {"ball": ("manipulated", {"kind": "sphere", "radius": 0.035})},
                    "markers": {"goal_region": {"kind": "disc", "radius": 0.1}}},
    "StackPyramid-v1": {"objects": {"cubeA": ("manipulated", CUBE), "cubeB": ("support", CUBE), "cubeC": ("manipulated", CUBE)}},
    "TwoRobotPickCube-v1": {"objects": {"cube": ("manipulated", CUBE)}, "markers": {"goal_site": {"kind": "sphere", "radius": 0.025}}},
    "TwoRobotStackCube-v1": {"objects": {"cubeA": ("manipulated", CUBE), "cubeB": ("manipulated", CUBE)},
                             "markers": {"goal_region": {"kind": "disc", "radius": 0.06}}},
}
STATIC_ACTORS = {"table-workspace": ("support", TABLE)}


def peg_insertion_geometry(seed: int, clearance: float = 0.003) -> tuple[dict, dict]:
    """Peg and box geometry of PegInsertionSide-v1, regenerated from the episode seed.

    ``_load_scene`` draws, from ``RandomState(episode_seed)``: half length ~ U(0.085, 0.125),
    radius ~ U(0.015, 0.025), hole center = 0.5 (length - radius) U(-1, 1)^2. The caller
    checks the result against the recorded initial peg height (= radius).
    """
    rng = np.random.RandomState(seed)
    length, radius = rng.uniform(0.085, 0.125), rng.uniform(0.015, 0.025)
    center = 0.5 * (length - radius) * rng.uniform(-1, 1, size=2)
    peg = {**_box(length, radius, radius), "head": "+x half (inserted end)"}
    inner, outer, depth = radius + clearance, length, length
    t = (outer - inner) * 0.5
    hc = center * 0.5
    off = t + inner
    box = _parts(((0, off + hc[0], 0), (depth, t - hc[0], outer)), ((0, -off + hc[0], 0), (depth, t + hc[0], outer)),
                 ((0, 0, off + hc[1]), (depth, outer, t - hc[1])), ((0, 0, -off + hc[1]), (depth, outer, t + hc[1])),
                 note="box with a square hole along x (kinematic)")
    box["hole"] = {"axis": "x", "center_yz": center.tolist(), "half_width": float(inner), "clearance": clearance}
    return peg, box


# ---------------------------------------------------------------- Panda kinematics

def _pad_box(urdf_path: Path, link: str) -> np.ndarray:
    """Center of the fingertip collision box (largest z) of a finger link, in its frame."""
    root = ET.parse(urdf_path).getroot()
    el = next(e for e in root.findall("link") if e.get("name") == link)
    best = None
    for c in el.findall("collision"):
        if c.find("geometry/box") is None:
            continue
        o = c.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
        if best is None or xyz[2] > best[2]:
            best = xyz
    if best is None:
        raise ValueError(f"{urdf_path.name}: no box collision on {link}")
    return best


class PandaGripper:
    """Panda arm + parallel gripper FK from a URDF, expressed in the contract frame."""

    def __init__(self, urdf_path):
        self.urdf_path = Path(urdf_path)
        self.sha256 = sha256_file(self.urdf_path)
        model = URDF(self.urdf_path)
        missing = [j for j in PANDA_JOINTS if j not in model.by_name]
        if missing:
            raise ValueError(f"{self.urdf_path.name}: not a Panda URDF (missing {missing})")
        # The URDF declares panda_finger_joint2 as a mimic of joint1, but SAPIEN simulates
        # both fingers as independent active joints (both are in the recorded qpos and they
        # differ under contact), so the mimic tag is ignored here.
        self.ignored_mimic = sorted(model.mimic)
        model.mimic = {}
        self.limits = np.array([model.limits[j] for j in PANDA_JOINTS])
        self.tree = KinematicTree(model, "panda_link0",
                                  ["panda_hand", "panda_hand_tcp", "panda_leftfinger", "panda_rightfinger"],
                                  PANDA_JOINTS)
        self.pads = [_pad_box(self.urdf_path, "panda_leftfinger"), _pad_box(self.urdf_path, "panda_rightfinger")]
        q_open = np.r_[np.zeros(7), self.limits[7:, 1]]
        q_closed = np.r_[np.zeros(7), self.limits[7:, 0]]
        H, p1, p2 = self._hand_and_pads(q_open)
        R = H[:3, :3]
        mid = (p1 + p2) / 2
        z = R.T @ (mid - H[:3, 3])
        z /= np.linalg.norm(z)
        y = R.T @ (p2 - p1)
        y -= z * (y @ z)
        y /= np.linalg.norm(y)
        fix = np.column_stack([np.cross(y, z), y, z])
        snapped = np.round(fix)
        self.R_fix = snapped if np.abs(fix - snapped).max() < 1e-3 and abs(np.linalg.det(snapped) - 1) < 1e-9 else fix
        self.offset = R.T @ (mid - H[:3, 3])          # grasp center in the hand frame
        y_w = R @ self.R_fix[:, 1]
        sep_open = (p2 - p1) @ y_w
        H, p1, p2 = self._hand_and_pads(q_closed)
        self.sep_closed = float((p2 - p1) @ (H[:3, :3] @ self.R_fix[:, 1]))
        self.width_max = float(sep_open - self.sep_closed)
        tcp = self.tree.fk(q_open)["panda_hand_tcp"]
        self.center_in_tcp = (np.linalg.inv(tcp) @ np.r_[H[:3, 3] + R @ self.offset, 1])[:3]

    def _hand_and_pads(self, q):
        fk = self.tree.fk(q)
        pads = [fk[f][..., :3, :3] @ p + fk[f][..., :3, 3] for f, p in
                zip(("panda_leftfinger", "panda_rightfinger"), self.pads)]
        return fk["panda_hand"], pads[0], pads[1]

    def forward(self, qpos: np.ndarray, root: np.ndarray) -> dict:
        """Batched poses in world for qpos (T, 9) and root poses (T, 4, 4)."""
        fk = self.tree.fk(qpos)
        H = root @ fk["panda_hand"]
        G = np.array(H)
        G[:, :3, :3] = H[:, :3, :3] @ self.R_fix
        G[:, :3, 3] = H[:, :3, 3] + H[:, :3, :3] @ self.offset
        _, p1, p2 = self._hand_and_pads(qpos)
        sep = np.einsum("ti,ti->t", p2 - p1, fk["panda_hand"][:, :3, :3] @ self.R_fix[:, 1])
        return {"grasp": G, "tcp": root @ fk["panda_hand_tcp"], "width": np.maximum(0.0, sep - self.sep_closed),
                "fingers": [root @ fk["panda_leftfinger"], root @ fk["panda_rightfinger"]]}

    def describe(self) -> dict:
        return {"urdf": self.urdf_path.name, "urdf_sha256": self.sha256, "joint_order": PANDA_JOINTS,
                "tcp_link": "panda_hand_tcp", "palm_link": "panda_hand", "ignored_mimic": self.ignored_mimic,
                "finger1": "panda_leftfinger", "finger2": "panda_rightfinger",
                "hand_to_contract_rotation": self.R_fix.round(6).tolist(),
                "grasp_center_in_hand_frame_m": self.offset.round(6).tolist(),
                "grasp_center_in_tcp_frame_m": self.center_in_tcp.round(6).tolist(),
                "width_max_m": round(self.width_max, 6)}


@lru_cache(maxsize=4)
def _gripper(path: str) -> PandaGripper:
    return PandaGripper(path)


def fk_crosscheck_mujoco(urdf_path, qpos: np.ndarray, root: np.ndarray, poses: dict) -> dict:
    """Independent FK of the same URDF with MuJoCo's URDF importer (optional dependency).

    Compares ``panda_hand_tcp`` and both finger link poses with :meth:`PandaGripper.forward`.
    """
    try:
        import mujoco
    except ImportError:
        return {"available": False, "reason": "mujoco is not installed"}
    r = ET.parse(urdf_path).getroot()
    for link in r.findall("link"):
        for tag in ("visual", "collision"):
            for e in link.findall(tag):
                link.remove(e)
    ET.SubElement(ET.SubElement(r, "mujoco"), "compiler", fusestatic="false")
    m = mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in PANDA_JOINTS]
    bodies = {"tcp": m.body("panda_hand_tcp").id, "f1": m.body("panda_leftfinger").id,
              "f2": m.body("panda_rightfinger").id}
    ours = {"tcp": poses["tcp"], "f1": poses["fingers"][0], "f2": poses["fingers"][1]}
    dp = dr = 0.0
    for t in range(len(qpos)):
        d.qpos[adr] = qpos[t]
        mujoco.mj_kinematics(m, d)
        for k, b in bodies.items():
            T = np.eye(4)
            T[:3, :3] = d.xmat[b].reshape(3, 3)
            T[:3, 3] = d.xpos[b]
            T = root[t] @ T
            dp = max(dp, float(np.linalg.norm(T[:3, 3] - ours[k][t][:3, 3])))
            dr = max(dr, float(Rotation.from_matrix(T[:3, :3].T @ ours[k][t][:3, :3]).magnitude()))
    return {"available": True, "library": f"mujoco {mujoco.__version__} URDF import", "frames": len(qpos),
            "links": ["panda_hand_tcp", "panda_leftfinger", "panda_rightfinger"],
            "max_position_error_m": dp, "max_rotation_error_rad": dr}


# ---------------------------------------------------------------- episodes

def pose_to_matrix(p: np.ndarray) -> np.ndarray:
    """(T, 7) xyz + wxyz -> (T, 4, 4)."""
    p = np.asarray(p, float)
    T = np.broadcast_to(np.eye(4), p.shape[:-1] + (4, 4)).copy()
    q = p[..., 3:7] / np.linalg.norm(p[..., 3:7], axis=-1, keepdims=True)
    T[..., :3, :3] = Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(p.shape[:-1] + (3, 3))
    T[..., :3, 3] = p[..., :3]
    return T


def _robot_uid(articulation: str) -> str:
    return articulation.split("-agent-")[0]


def _locate_urdf(uid: str, path: Path, root, catalog) -> Path | None:
    if root is None:
        root = next((p.parent for p in path.resolve().parents if p.name == "raw"), None)
    if root is None:
        return None
    for e in catalog.values():
        if e.family == "maniskill" and e.path.endswith("/" + ROBOT_URDF[uid]):
            local = e.local_path(root)
            if local.exists():
                return local
    return None


def _source_type(meta: dict, path: Path) -> str:
    st = meta.get("source_type")
    if st:
        return st
    folder = path.parent.name
    return {"rl": "rl", "motionplanning": "motionplanning", "teleop": "teleoperation"}.get(folder, "unknown")


def _digest(g: h5py.Group) -> str:
    h = hashlib.sha256()

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            h.update(name.encode())
            h.update(np.ascontiguousarray(obj[()]).tobytes())
    g.visititems(visit)
    return h.hexdigest()


def _base_name(actor: str, table: dict) -> str:
    """``peg_0`` -> ``peg`` when only the suffixed name is recorded (single-env builds)."""
    if actor in table:
        return actor
    stripped = re.sub(r"_\d+$", "", actor)
    return stripped if stripped in table else actor


@register("maniskill")
def read_maniskill(path: Path, *, family: str, demos=None, limit=None, root=None, urdfs=None,
                   catalog=None, crosscheck: bool = False):
    """Yield one :class:`SourceEpisode` per ``traj_<id>`` of a ManiSkill ``trajectory*.h5``.

    ``demos`` selects group keys (default: all, by numeric id), ``limit`` stops after that
    many episodes. ``urdfs`` maps robot uid -> URDF path (default: the catalogued
    ``maniskill/assets/...`` files under the data root inferred from ``path``).
    ``crosscheck`` adds an independent MuJoCo FK comparison to each episode's provenance.
    """
    path = Path(path)
    catalog = catalog if catalog is not None else load_catalog(tables=False)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    digest, meta_digest = sha256_file(path), sha256_file(meta_path)
    entry = next((e for e in catalog.values() if e.sha256 == digest), None)
    meta_entry = next((e for e in catalog.values() if e.sha256 == meta_digest), None)
    env_info = meta["env_info"]
    env_id, kwargs = env_info["env_id"], env_info.get("env_kwargs", {})
    sim_cfg = kwargs.get("sim_config") or kwargs.get("sim_cfg") or {}
    control_freq = float(sim_cfg.get("control_freq", DEFAULT_CONTROL_FREQ))
    sim_freq = float(sim_cfg.get("sim_freq", DEFAULT_SIM_FREQ))
    freq_source = "env_kwargs sim_config" if "control_freq" in sim_cfg else \
        f"ManiSkill SimConfig default (sim {DEFAULT_SIM_FREQ} Hz, control {DEFAULT_CONTROL_FREQ} Hz); not recorded in the JSON"
    source_type = _source_type(meta, path)
    episodes = {f"traj_{e['episode_id']}": e for e in meta["episodes"]}
    dataset = entry.dataset if entry and entry.dataset else f"maniskill/{env_id}/{path.parent.name}"
    task = TASKS.get(env_id)
    urdfs = dict(urdfs or {})

    seeds: dict[int, list[str]] = {}
    for k, e in episodes.items():
        seeds.setdefault(e.get("episode_seed"), []).append(k)

    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1])) if demos is None else list(demos)
        if limit is not None:
            keys = keys[:limit]
        digests: dict[str, str] = {}

        def duplicate_of(key):
            group = sorted(seeds.get(episodes[key].get("episode_seed"), [key]), key=lambda k: int(k.split("_")[-1]))
            if len(group) < 2:
                return None
            for k in group:
                if k not in digests:
                    digests[k] = _digest(f[k])
            return next((k for k in group if int(k.split("_")[-1]) < int(key.split("_")[-1])
                         and digests[k] == digests[key]), None)

        for key in keys:
            if key not in episodes:
                raise KeyError(f"{key} is not listed in {meta_path.name}")
            g = f[key]
            arts = sorted(g["env_states/articulations"])
            grippers = {}
            for a in arts:
                uid = _robot_uid(a)
                if uid in NON_GRIPPER_ROBOTS:
                    raise ValueError(f"{env_id}: robot {uid} is not supported ({NON_GRIPPER_ROBOTS[uid]})")
                if uid not in ROBOT_URDF:
                    raise ValueError(f"{env_id}: unknown robot {uid!r}")
                if uid not in urdfs:
                    urdfs[uid] = _locate_urdf(uid, path, root, catalog)
                if urdfs[uid] is None:
                    raise FileNotFoundError(f"URDF for robot {uid} not found; fetch "
                                            f"maniskill/assets/ManiSkill-{MANISKILL_COMMIT[:8]}/{ROBOT_URDF[uid]} or pass urdfs=")
                grippers[a] = _gripper(str(urdfs[uid]))
            yield _episode(g, key, episodes[key], family, dataset, env_id, kwargs, meta, task, grippers,
                           control_freq, sim_freq, freq_source, source_type, entry, meta_entry, digest,
                           meta_digest, path, duplicate_of(key), crosscheck)


def _episode(g, key, ep, family, dataset, env_id, kwargs, meta, task, grippers, control_freq, sim_freq,
             freq_source, source_type, entry, meta_entry, digest, meta_digest, path, dup, crosscheck):
    actions = np.asarray(g["actions"][()])
    T = len(actions) + 1
    time = np.arange(T) / control_freq
    effectors, bases, checks, roots_z = {}, {}, {}, {}
    arts = sorted(grippers)
    for a, gr in grippers.items():
        s = np.asarray(g[f"env_states/articulations/{a}"][()], float)
        if s.shape != (T, 13 + 2 * len(PANDA_JOINTS)):
            raise ValueError(f"{key}/{a}: expected ({T}, 31) articulation states, got {s.shape}")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"{key}/{a}: non-finite state")
        q = s[:, 13:22]
        out_of_limits = float(np.max(np.maximum(gr.limits[:, 0] - q, q - gr.limits[:, 1]).clip(0)))
        if out_of_limits > 0.02:
            raise ValueError(f"{key}/{a}: qpos leaves the URDF limits by {out_of_limits:.3f}; joint order unverified")
        root = pose_to_matrix(s[:, :7])
        root[:, 2, 3] += WORLD_Z_OFFSET
        poses = gr.forward(q, root)
        effectors[a] = Effector(pose=poses["grasp"], width=poses["width"],
                                opening=np.clip(poses["width"] / gr.width_max, 0, 1))
        R0 = root[0, :3, :3]
        bases[a] = [float(root[0, 0, 3]), float(root[0, 1, 3]), float(np.arctan2(R0[1, 0], R0[0, 0]))]
        roots_z[a] = float(root[0, 2, 3])
        check = {"max_qpos_limit_violation_rad_or_m": out_of_limits,
                 "root_motion_m": float(np.ptp(s[:, :3], axis=0).max())}
        mode = ep.get("control_mode")
        mode = mode.get(a.replace("-agent-", "-")) if isinstance(mode, dict) else mode
        if mode == "pd_joint_pos" and len(arts) == 1 and actions.shape[1] >= 7:
            check["first_action_minus_qpos_max_rad"] = float(np.abs(actions[0, :7] - q[0, :7]).max())
        if crosscheck:
            check["fk_crosscheck"] = fk_crosscheck_mujoco(gr.urdf_path, q, root, poses)
        checks[a] = check

    objects, markers, unknown = {}, {}, []
    table = (task or {}).get("objects", {})
    seed = ep.get("episode_seed")
    geometry_status = {}
    for actor in sorted(g["env_states/actors"]):
        s = np.asarray(g[f"env_states/actors/{actor}"][()], float)
        if s.shape != (T, 13):
            raise ValueError(f"{key}/{actor}: expected ({T}, 13) actor states, got {s.shape}")
        name = _base_name(actor, {**table, **STATIC_ACTORS, **(task or {}).get("markers", {})})
        pose = s[:, :7].copy()
        pose[:, 2] += WORLD_Z_OFFSET
        pose[:, 3:] /= np.linalg.norm(pose[:, 3:], axis=1, keepdims=True)
        if name in (task or {}).get("markers", {}):
            markers[name] = {**task["markers"][name], "actor": actor, "pose_t0": pose[0].round(6).tolist(),
                             "moves": bool(np.ptp(pose[:, :3], axis=0).max() > 1e-6)}
            continue
        if name in STATIC_ACTORS:
            role, geom = STATIC_ACTORS[name]
        elif name in table:
            role, geom = table[name]
        else:
            role, geom = "manipulated", {}
            unknown.append(actor)
        if isinstance(geom, str):  # seed-dependent PegInsertionSide geometry
            peg, box = peg_insertion_geometry(int(seed))
            init_z = float(g[f"env_states/actors/{_find(g, 'peg')}"][0, 2])
            ok = abs(init_z - peg["half_extents"][1]) < 1e-5
            geom = (peg if geom.endswith("peg") else box) if ok else {}
            geometry_status[name] = ("regenerated from episode_seed; initial peg height equals the radius "
                                     f"(|dz| = {abs(init_z - peg['half_extents'][1]):.1e} m)") if ok else \
                "unknown: seed regeneration does not match the recorded initial peg height"
        else:
            geometry_status[name] = "task constant" if geom else "unknown"
        objects[name] = ObjectTrack(pose=pose, valid=np.ones(T, bool), role=role,
                                    geometry={**geom, "actor": actor, "body": name} if geom else {"actor": actor})

    adaptations = []
    if "table-workspace" in objects:   # minimal recorded fixture adaptation (see TABLE_CROP_MARGIN)
        objects, adaptations = primitive_scene.crop_supports(objects, TABLE_CROP_MARGIN, ids=["table-workspace"])
    scene, scene_note = (None, f"none: unknown actors {unknown}") if unknown else scene_for(env_id, objects)
    first = arts[0] if arts else None
    succ = np.asarray(g["success"][()], bool) if "success" in g else None
    provenance = {
        "file": str(path), "sha256": digest, "url": entry.url if entry else None,
        "revision": entry.revision if entry else None, "catalog_id": entry.id if entry else None,
        "metadata_file": str(path.with_suffix(".json")), "metadata_sha256": meta_digest,
        "metadata_catalog_id": meta_entry.id if meta_entry else None,
        "demo_key": key, "episode_id": ep.get("episode_id"), "episode_seed": seed,
        "reset_kwargs": ep.get("reset_kwargs"), "elapsed_steps": ep.get("elapsed_steps"),
        "env_id": env_id, "env_kwargs": kwargs, "max_episode_steps": meta["env_info"].get("max_episode_steps"),
        "maniskill_commit": meta.get("commit_info", {}).get("commit_id"),
        "source_type": source_type, "source_desc": meta.get("source_desc"),
        "control_mode": ep.get("control_mode"), "action_shape": list(actions.shape),
        "control_freq_hz": control_freq, "sim_freq_hz": sim_freq, "freq_source": freq_source,
        "time_source": "state index / control_freq (T actions, T + 1 states incl. terminal)",
        "state_layout": {"actor": "p(3) q_wxyz(4) v(3) w(3)",
                         "articulation": "root p(3) q_wxyz(4) v(3) w(3), qpos(9), qvel(9)"},
        "robots": {a: _robot_uid(a) for a in grippers},
        "robot_bases_xy_yaw": bases, "robot_base_z": roots_z,
        "world_z_offset_m": WORLD_Z_OFFSET,
        "world_frame": ("ManiSkill world translated by +0.9196429 m in z (TableSceneBuilder table height): "
                        "floor at z = 0, table top at z = 0.9196429; applied to robot roots (effectors), "
                        "actors, goal markers and the scene"),
        "effectors": {a: gr.describe() for a, gr in grippers.items()},
        "effector_env_roles": ({arts[0]: "left_agent", arts[1]: "right_agent"} if len(arts) == 2 else None),
        "state_checks": checks,
        "opening": "pad separation along +y minus closed separation (= q_finger1 + q_finger2), / 0.08 m",
        "goal_markers": markers, "unknown_actors": unknown, "geometry_status": geometry_status,
        "success_source": "JSON episodes[].success",
        "success_at_end": None if succ is None else bool(succ[-1]),
        "success_any": None if succ is None else bool(succ.any()),
        "scene": scene_note,
        "scene_adaptations": adaptations,
        "scene_physical": None if scene is None else primitive_scene.scene_provenance(scene),
        "evidence": EVIDENCE,
    }
    lineage = {"generated": False, "source_type": source_type, "human": source_type == "teleoperation",
               "initial_state": f"maniskill/{env_id}/env_seed/{seed}", "duplicate_of": dup}
    return SourceEpisode(
        family=family, dataset=dataset, episode_id=key, task=env_id, time=time, effectors=effectors,
        objects=objects, base_hint=None if first is None else np.array(bases[first]),
        scene=scene, instruction=None, success=bool(ep["success"]) if "success" in ep else None,
        regime="tabletop", license=entry.license if entry else "unknown",
        provenance=provenance, lineage=lineage)


def _find(g, base: str) -> str:
    return next(a for a in g["env_states/actors"] if a == base or re.fullmatch(rf"{base}_\d+", a))


__all__ = ["read_maniskill", "scene_for", "PHYSICAL", "PHYSICAL_SOURCES", "PandaGripper", "fk_crosscheck_mujoco", "peg_insertion_geometry", "TASKS",
           "pose_to_matrix"]
