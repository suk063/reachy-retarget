"""Adapter for BiGym demonstrations (Unitree H1 + two Robotiq 2F-85, floating base).

BiGym's published demonstrations (``bigym_data`` release ``v0.9.0``,
``demonstrations.zip``) are lightweight recordings: 500 Hz actions, a reset seed and
environment metadata, no states. States exist only after replaying the actions in the
BiGym environment, which needs pinned MuJoCo 3.1.5 / numpy 1.26 / dm_control and so
runs in an isolated interpreter: :mod:`reachy_retarget.sources.bigym_replay` (a
standalone script) writes one *replay record* ``<task>/<uuid>.npz`` per recording with
sampled ``qpos``/``qvel``, the env's success check, and the complete seeded scene MJCF;
mesh/texture files go to the shared ``assets/`` folder next to the task folders.
:func:`replay` launches that script explicitly.

This adapter only reads replay records. It compiles the recorded MJCF with this
project's MuJoCo, sets ``qpos`` per frame, runs forward kinematics and reads:

* **Effectors** ``left``/``right``: the grasp center of each 2F-85. The contract frame
  is derived from model geometry at the gripper's reference (fully open) configuration:
  approach +z = palm body origin -> midpoint of the two pad bodies' collision boxes,
  closing +y = left pad -> right pad, grasp center = that pad midpoint, a fixed offset
  in the palm frame. ``opening`` = 1 - normalized driver-joint angle (both driver
  joints averaged; 0 rad = open, 0.8 rad = closed), ``width`` = gap between the inner
  pad faces along +y (pad box centers minus the boxes' half thickness).
* **Base**: the H1 pelvis body world pose as ``(x, y, yaw)``; pelvis height goes to
  ``torso_height``. ``regime = "mobile_manipulation"``.
* **Objects**: every free body outside the robot (task props) as ``manipulated`` (props
  disabled by BiGym, i.e. without any collision geometry, are only listed in
  ``provenance["inactive_free_bodies"]``), plus
  static world-child bodies with collision geometry (furniture) as fixtures.
* **Articulations**: hinge/slide joints outside the robot, grouped by root body
  (cabinets, drawers, dishwasher door/trays).
* **Success**: the BiGym success check observed during replay (any step, the rule of
  ``DemoPlayer.validate_in_env``); a failed or partial replay gives ``success=False``
  or ``None`` and keeps its error in provenance. The release carries no per-demo success
  label, only recorded termination flags (success *or* failure at recording time),
  which are reported separately.
* **Scene** for tier P: the seeded MJCF, the asset files and robot prefix ``h1/``.
* **World frame**: BiGym's floor plane is at z = 0 (checked per record and recorded in
  provenance), as the contract requires; otherwise tracks are translated and the scene
  is omitted.

When assets are missing (e.g. unit-test fixtures), mesh geoms become placeholders: the
kinematics stay exact, no scene is attached and object geometry is omitted
(``provenance["state_route"] == "bigym_replay_kinematic_only"``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..acquire import load_catalog
from ..schema.source import Articulation, Effector, ObjectTrack, SceneRef, SourceEpisode
from .contact_reference import ObjectEnvironmentDepth
from .registry import register

FAMILY = "bigym"
DEMO_ZIP_ID = "bigym/v0.9.0/demonstrations.zip"
CODE_ID_PREFIX = "bigym/code/"
RECORD_VERSION = "bigym-replay-record-v1"
ROBOT_PREFIX = "h1/"
REPLAY_SCRIPT = Path(__file__).with_name("bigym_replay.py")
DEFAULT_REPLAY_PYTHON = "data/envs/bigym-replay/bin/python"

# Task classes of BiGym 4.x (bigym/envs/*.py) grouped as in the paper. Base motion:
# every demo uses the floating base (pelvis x, y, rz; most tasks also z).
TASK_GROUPS = {
    "articulated": ["DrawerTopOpen", "DrawerTopClose", "DrawersAllOpen", "DrawersAllClose",
                    "WallCupboardOpen", "WallCupboardClose", "CupboardsOpenAll", "CupboardsCloseAll",
                    "DishwasherOpen", "DishwasherClose", "DishwasherOpenTrays", "DishwasherCloseTrays"],
    "pick_place": ["MovePlate", "MoveTwoPlates", "PickBox", "StoreBox", "SaucepanToHob", "StoreKitchenware",
                   "PutCups", "TakeCups", "ToastSandwich", "FlipSandwich", "RemoveSandwich",
                   "DishwasherLoadPlates", "DishwasherLoadCups", "DishwasherLoadCutlery",
                   "DishwasherUnloadPlates", "DishwasherUnloadCups", "DishwasherUnloadCutlery",
                   "DishwasherUnloadPlatesLong", "DishwasherUnloadCupsLong", "DishwasherUnloadCutleryLong",
                   "GroceriesStoreLower", "GroceriesStoreUpper"],
    "manipulation": ["FlipCup", "FlipCutlery", "StackBlocks"],
    "reach": ["ReachTarget", "ReachTargetSingle", "ReachTargetDual"],
}

SIMULATION_ASSUMPTIONS = [
    "states are reconstructed by replaying recorded actions; the release stores no states",
    "recorded with BiGym 4.0.0 (not public); replayed with BiGym 4.1.0 at the pinned commit and "
    "the recorded MuJoCo 3.1.5",
    "the H1 'floating base' is a position-actuated pelvis (slide x/y/z, hinge rz) with animated "
    "legs, not legged locomotion; the base path is the pelvis pose",
    "kinematics are recomputed with this project's MuJoCo from the recorded qpos; the scene MJCF "
    "was exported from the seeded MuJoCo 3.1.5 model and checked against it",
]


# ---------------------------------------------------------------- replay launcher

def replay(zip_path, out_dir, *, python=DEFAULT_REPLAY_PYTHON, tasks=None, members=None, limit=None,
           stride=10, jobs=1, skip_existing=True, check=True):
    """Run :mod:`bigym_replay` in the isolated BiGym interpreter ``python``.

    Explicit and local: reads the downloaded ``demonstrations.zip``, writes replay
    records under ``out_dir`` and returns the parsed per-demo summaries.
    """
    cmd = [str(python), str(REPLAY_SCRIPT), "--zip", str(zip_path), "--out", str(out_dir),
           "--stride", str(stride), "--jobs", str(jobs)]
    if tasks:
        cmd += ["--tasks", ",".join(tasks)]
    if members:
        cmd += ["--members", ",".join(members)]
    if limit:
        cmd += ["--limit", str(limit)]
    if skip_existing:
        cmd.append("--skip-existing")
    res = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return [json.loads(line) for line in res.stdout.splitlines() if line.startswith("{")]


# ---------------------------------------------------------------- records

def read_record(path) -> tuple[dict, dict]:
    """``(meta, arrays)`` of one replay record; ``arrays`` lacks ``meta``/``mjcf``."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        arrays = {k: z[k] for k in z.files if k != "meta"}
    if meta.get("record_version") != RECORD_VERSION:
        raise ValueError(f"{path}: unsupported record version {meta.get('record_version')!r}")
    if "mjcf" in arrays:
        meta["_mjcf"] = bytes(arrays.pop("mjcf")).decode()
    return meta, arrays


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_assets(xml: str, folder: Path | None):
    """Load file assets named in ``xml`` from ``folder``.

    Returns ``(xml, assets, missing)``; with anything missing, file assets are dropped
    and mesh geoms become 1 mm spheres (exact kinematics, no physics).
    """
    root = ET.fromstring(xml)
    assets, missing = {}, []
    for el in root.iter():
        f = el.get("file")
        if not f:
            continue
        p = None if folder is None else Path(folder) / f
        if p is not None and p.is_file():
            assets[f] = p.read_bytes()
        else:
            missing.append(f)
    if missing:
        assets = {}
        asset = root.find("asset")
        dropped = [el for el in (asset if asset is not None else []) if el.get("file")]
        textures = {el.get("name") for el in dropped if el.tag == "texture"}
        meshes = {el.get("name") for el in dropped if el.tag == "mesh"}
        for el in dropped:
            asset.remove(el)
        for mat in root.iter("material"):
            if mat.get("texture") in textures:
                mat.attrib.pop("texture")
        for geom in root.iter("geom"):
            if geom.get("mesh") in meshes or geom.get("type") == "mesh":
                geom.attrib.pop("mesh", None)
                geom.set("type", "sphere")
                geom.set("size", "0.001")
        xml = ET.tostring(root, encoding="unicode")
    return xml, assets, missing


def _compile_tolerant(mujoco, xml: str, assets: dict):
    """Compile; meshes rejected by newer MuJoCo for too small a volume get
    ``inertia="shell"`` (MuJoCo's own suggestion). Returns ``(model, xml, shell_meshes)``."""
    shell = []
    while True:
        try:
            return mujoco.MjModel.from_xml_string(xml, assets), xml, shell
        except ValueError as err:
            mt = re.search(r"mesh volume is too small: (\S+?) \.", str(err))
            if not mt or mt.group(1) in shell or len(shell) > 50:
                raise
            root = ET.fromstring(xml)
            next(el for el in root.iter("mesh") if el.get("name") == mt.group(1)).set("inertia", "shell")
            shell.append(mt.group(1))
            xml = ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------- model geometry

class _Gripper:
    """Robotiq 2F-85 measured from the compiled model (reference = qpos0, fully open)."""

    def __init__(self, model: "_Model", prefix: str):
        mj, m = model.mj, model.m
        self.prefix, self.m = prefix, m
        root = model.body_id(prefix)
        sub = [b for b in range(m.nbody) if model.in_subtree(b, root)]
        drivers = [j for j in range(m.njnt) if m.jnt_bodyid[j] in sub
                   and model.joint_names[j].endswith("driver_joint")]
        pad_bodies = sorted({int(m.geom_bodyid[g]) for g in range(m.ngeom)
                             if m.geom_bodyid[g] in sub and "pad" in model.geom_names[g]
                             and m.geom_type[g] == mj.mjtGeom.mjGEOM_BOX
                             and (m.geom_contype[g] or m.geom_conaffinity[g])},
                            key=lambda b: model.body_names[b])
        if len(drivers) != 2 or len(pad_bodies) != 2:
            raise ValueError(f"{prefix}: expected two driver joints and two pad bodies (2F-85)")
        self.drivers, self.pad_bodies = drivers, pad_bodies
        self.pad_geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == b
                           and (m.geom_contype[g] or m.geom_conaffinity[g])] for b in pad_bodies]
        self.palm = int(m.body_parentid[m.jnt_bodyid[drivers[0]]])
        self.range = m.jnt_range[drivers].copy()
        self.qadr = m.jnt_qposadr[drivers]
        d = mj.MjData(m)
        d.qpos[:] = m.qpos0
        mj.mj_kinematics(m, d)
        R, p = d.xmat[self.palm].reshape(3, 3), d.xpos[self.palm]
        c0, c1 = self._centers(d)
        mid = (c0 + c1) / 2
        z = R.T @ (mid - p)
        z /= np.linalg.norm(z)
        y = R.T @ (c1 - c0)
        y -= z * (y @ z)
        y /= np.linalg.norm(y)
        fix = np.column_stack([np.cross(y, z), y, z])
        snapped = np.round(fix)
        self.R_fix = snapped if np.abs(fix - snapped).max() < 1e-3 and abs(np.linalg.det(snapped) - 1) < 1e-9 else fix
        self.offset = R.T @ (mid - p)
        self.width_open = self._gap(d, R @ self.R_fix[:, 1])

    def _centers(self, d):
        return [d.geom_xpos[gs].mean(0) for gs in self.pad_geoms]

    def _gap(self, d, y) -> float:
        m = self.m
        c0, c1 = self._centers(d)
        half = [max(float(np.abs(d.geom_xmat[g].reshape(3, 3).T @ y) @ m.geom_size[g]) for g in gs)
                for gs in self.pad_geoms]
        return float((c1 - c0) @ y - half[0] - half[1])

    def read(self, d):
        """``(pose 4x4, opening, width)`` from the current MjData."""
        R = d.xmat[self.palm].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R @ self.R_fix
        T[:3, 3] = d.xpos[self.palm] + R @ self.offset
        q = d.qpos[self.qadr]
        frac = np.mean((q - self.range[:, 0]) / (self.range[:, 1] - self.range[:, 0]))
        return T, float(np.clip(1.0 - frac, 0.0, 1.0)), max(0.0, self._gap(d, T[:3, 1]))

    def describe(self, model: "_Model") -> dict:
        return {"palm_body": model.body_names[self.palm],
                "driver_joints": [model.joint_names[j] for j in self.drivers],
                "driver_range_rad": self.range.round(6).tolist(),
                "pad_bodies": [model.body_names[b] for b in self.pad_bodies],
                "pad_geoms": [[model.geom_names[g] for g in gs] for gs in self.pad_geoms],
                "palm_to_contract_rotation": self.R_fix.round(6).tolist(),
                "grasp_center_in_palm_frame_m": self.offset.round(6).tolist(),
                "width_open_m": round(self.width_open, 6)}


class _Model:
    def __init__(self, xml: str, asset_dir):
        import mujoco

        self.mj = mujoco
        self.xml, self.assets, self.missing = resolve_assets(xml, asset_dir)
        self.m, self.xml, self.shell_meshes = _compile_tolerant(mujoco, self.xml, self.assets)
        m = self.m
        self.d = mujoco.MjData(m)
        nm = lambda kind, i: mujoco.mj_id2name(m, kind, i) or ""
        self.body_names = [nm(mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        self.joint_names = [nm(mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        self.geom_names = [nm(mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
        robot_root = self.body_id(ROBOT_PREFIX)
        self.is_robot = np.array([m.body_rootid[b] == robot_root for b in range(m.nbody)])
        free, scalar = int(mujoco.mjtJoint.mjJNT_FREE), (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        self.free = {self.body_names[m.jnt_bodyid[j]].rstrip("/"): j
                     for j in range(m.njnt) if m.jnt_type[j] == free and not self.is_robot[m.jnt_bodyid[j]]}
        # BiGym's Prop.disable() (e.g. unused groceries): no collisions, parked out of reach.
        collides_b = lambda b: any((m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)
                                   if self.in_subtree(int(m.geom_bodyid[g]), b))
        self.inactive = sorted(n for n, j in self.free.items() if not collides_b(int(m.jnt_bodyid[j])))
        self.inactive_bodies = [self.body_names[m.jnt_bodyid[self.free[n]]] for n in self.inactive]
        for n in self.inactive:
            self.free.pop(n)
        self.articulations: dict[str, list[int]] = {}
        for j in range(m.njnt):
            b = m.jnt_bodyid[j]
            if m.jnt_type[j] in scalar and not self.is_robot[b]:
                self.articulations.setdefault(self.body_names[m.body_rootid[b]].rstrip("/"), []).append(j)
        collides = (m.geom_contype | m.geom_conaffinity) != 0
        jointed_roots = set(m.body_rootid[m.jnt_bodyid])
        self.fixtures = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0 and not self.is_robot[b]
                         and b not in jointed_roots and self.body_names[b] != "floor" and collides[m.body_rootid[m.geom_bodyid] == b].any()]
        self.cabinets = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0 and not self.is_robot[b]
                         and self.body_names[b].rstrip("/") in self.articulations]
        self.grippers = {}

    def body_id(self, name: str) -> int:
        i = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"body {name!r} not in the recorded model")
        return i

    def in_subtree(self, b: int, root: int) -> bool:
        while b > 0 and b != root:
            b = self.m.body_parentid[b]
        return b == root

    def body_aabb(self, b: int) -> dict:
        """Collision-geometry AABB of body ``b``'s subtree in ``b``'s frame (empty without assets)."""
        m, d = self.m, self.d
        if self.missing:
            return {}
        pts = []
        for g in np.flatnonzero(((m.geom_contype | m.geom_conaffinity) != 0)):
            if not self.in_subtree(int(m.geom_bodyid[g]), b):
                continue
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            corners = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            world = d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T
            pts.append((world - d.xpos[b]) @ d.xmat[b].reshape(3, 3))
        if not pts:
            return {}
        pts = np.concatenate(pts)
        lo, hi = pts.min(0), pts.max(0)
        return {"kind": "aabb", "center": ((lo + hi) / 2).round(6).tolist(),
                "half_extents": ((hi - lo) / 2).round(6).tolist(), "frame": "body"}


@lru_cache(maxsize=4)
def _compile(xml: str, asset_dir: str | None, gripper_prefixes: tuple) -> _Model:
    model = _Model(xml, None if asset_dir is None else Path(asset_dir))
    for side, prefix in gripper_prefixes:
        model.grippers[side] = _Gripper(model, prefix)
    return model


# ---------------------------------------------------------------- episodes

def _yaw(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _instruction(doc: str | None) -> str | None:
    if not doc:
        return None
    text = re.sub(r"\s+task\.?$", "", doc.strip().rstrip(".")).strip()
    return text[:1].upper() + text[1:] if text else None


def task_group(task: str) -> str | None:
    return next((g for g, ts in TASK_GROUPS.items() if task in ts), None)


def _record_paths(path: Path):
    if path.is_dir():
        return sorted(p for p in path.rglob("*.npz") if p.parent.name != "assets")
    return [path]


def _asset_dir(record: Path, asset_dir):
    if asset_dir is not None:
        return Path(asset_dir)
    for parent in record.parents:
        if (parent / "assets").is_dir():
            return parent / "assets"
    return None


@register(FAMILY)
def read_bigym(path: Path, *, family: str = FAMILY, asset_dir=None, catalog=None, with_scene: bool = True,
               include_failed: bool = True):
    """Yield one :class:`SourceEpisode` per replay record (a file or a folder of them).

    Records whose replay raised before any state was saved are skipped (they stay in
    the replay summary); partial replays are yielded with ``success=None`` unless
    ``include_failed`` is false. ``with_scene=False`` skips loading mesh assets (about
    10x faster compilation; exact kinematics, no object geometry, no scene).
    """
    catalog = catalog if catalog is not None else load_catalog()
    for rec in _record_paths(Path(path)):
        meta, arrays = read_record(rec)
        if "qpos" not in arrays or len(arrays["qpos"]) < 2 or "_mjcf" not in meta:
            continue
        if meta.get("error") and not include_failed:
            continue
        assets = _asset_dir(rec, asset_dir) if with_scene else None
        yield _episode(rec, meta, arrays, assets, catalog, family, with_scene)


def _episode(rec: Path, meta: dict, arrays: dict, asset_dir, catalog, family, with_scene):
    grippers = tuple(sorted((side, g["body"]) for side, g in meta["robot"]["grippers"].items()))
    model = _compile(meta["_mjcf"], None if asset_dir is None else str(asset_dir), grippers)
    m, d = model.m, model.d
    qpos, time = np.asarray(arrays["qpos"], float), np.asarray(arrays["time"], float)
    if qpos.shape[1] != m.nq:
        raise ValueError(f"{rec}: qpos width {qpos.shape[1]} != model nq {m.nq}")
    T = len(qpos)
    pelvis = model.body_id(meta["robot"]["pelvis"])
    poses = {k: np.zeros((T, 4, 4)) for k in model.grippers}
    opening = {k: np.zeros(T) for k in model.grippers}
    width = {k: np.zeros(T) for k in model.grippers}
    base, torso = np.zeros((T, 3)), np.zeros(T)
    obj_pose = {n: np.zeros((T, 7)) for n in model.free}
    art = {n: np.zeros((T, len(js))) for n, js in model.articulations.items()}
    # Source reference depth (SceneRef.reference): contacts of the replayed states. Disabled props
    # have no collision geometry, so every collidable free body is an object of the scene.
    reference = (ObjectEnvironmentDepth(m, {n: int(m.jnt_bodyid[j]) for n, j in model.free.items()}, model.is_robot)
                 if with_scene and not model.missing else None)
    for t in range(T):
        d.qpos[:] = qpos[t]
        model.mj.mj_kinematics(m, d)
        if reference is not None:
            model.mj.mj_collision(m, d)
            reference.update(d, t, time[t])
        for k, g in model.grippers.items():
            poses[k][t], opening[k][t], width[k][t] = g.read(d)
        base[t] = [d.xpos[pelvis][0], d.xpos[pelvis][1], _yaw(d.xmat[pelvis].reshape(3, 3))]
        torso[t] = d.xpos[pelvis][2]
        for n, j in model.free.items():
            b = m.jnt_bodyid[j]
            obj_pose[n][t] = np.r_[d.xpos[b], d.xquat[b]]
        for n, js in model.articulations.items():
            art[n][t] = d.qpos[m.jnt_qposadr[js]]
    base[:, 2] = np.unwrap(base[:, 2])

    # Contract: the floor the robot stands on is z = 0. BiGym's world.xml puts the floor
    # plane at the origin; measure it and translate everything if it ever differs.
    floor_z = _floor_z(model)
    if floor_z != 0.0:
        for k in poses:
            poses[k][:, 2, 3] -= floor_z
        for n in obj_pose:
            obj_pose[n][:, 2] -= floor_z
        torso -= floor_z

    d.qpos[:] = qpos[0]
    model.mj.mj_kinematics(m, d)
    effectors = {k: Effector(pose=poses[k], opening=opening[k], width=width[k], side_hint=k)
                 for k in model.grippers}
    objects = {n: ObjectTrack(pose=obj_pose[n], valid=np.ones(T, bool), role="manipulated",
                              geometry={**model.body_aabb(m.jnt_bodyid[j]), "body": model.body_names[m.jnt_bodyid[j]]})
               for n, j in model.free.items()}
    for b in model.fixtures + model.cabinets:
        name = model.body_names[b].rstrip("/")
        static = np.r_[d.xpos[b] - [0, 0, floor_z], d.xquat[b]]
        role = ("support" if re.search(r"table|counter", name) else
                "receptacle" if re.search(r"drainer|rack", name) else "fixture")
        objects.setdefault(name, ObjectTrack(pose=np.tile(static, (T, 1)),
                                             valid=np.ones(T, bool), role=role,
                                             geometry={**model.body_aabb(b), "body": model.body_names[b]}))
    initial_qpos = {model.joint_names[j]: qpos[0, m.jnt_qposadr[j]:m.jnt_qposadr[j] + 7].tolist()
                    for j in model.free.values()}
    initial_qpos.update({model.joint_names[j]: float(qpos[0, m.jnt_qposadr[j]])
                         for js in model.articulations.values() for j in js})
    scene = None
    if with_scene and not model.missing and floor_z == 0.0:
        scene = SceneRef(mjcf=model.xml, robot_prefixes=[ROBOT_PREFIX], initial_qpos=initial_qpos,
                         assets=model.assets, inactive_bodies=list(model.inactive_bodies),
                         reference=reference.result())

    error = meta.get("error")
    complete = error is None and meta.get("replayed_actions") == meta.get("n_actions")
    success = bool(meta["success_any"]) if complete else None
    zip_entry = catalog.get(DEMO_ZIP_ID)
    code = _code_entry(meta, catalog)
    task = meta["task"]
    provenance = {
        "record": str(rec), "record_sha256": _sha256(rec),
        "zip_member": meta["zip_member"], "member_sha256": meta["member_sha256"],
        "zip_sha256": meta.get("zip_sha256"),
        "zip_catalog_id": zip_entry.id if zip_entry else None,
        "zip_sha256_matches_catalog": bool(zip_entry and zip_entry.sha256 == meta.get("zip_sha256")),
        "url": zip_entry.url if zip_entry else None, "revision": zip_entry.revision if zip_entry else None,
        "code": None if code is None else {"catalog_id": code.id, "url": code.url, "revision": code.revision,
                                           "sha256": code.sha256},
        "seed": meta.get("seed"), "recorded_date": meta.get("recorded_date"),
        "recorded_package_versions": meta.get("recorded_package_versions"),
        "replay_runtime_versions": meta.get("runtime_versions"),
        "version_gap": "recorded with BiGym 4.0.0, replayed with BiGym 4.1.0 (4.0.0 source is not public)",
        "action_mode": meta.get("action_mode"), "action_mode_absolute": meta.get("action_mode_absolute"),
        "floating_dofs": meta.get("floating_dofs"),
        "control_frequency_hz": meta.get("control_frequency_hz"), "sim_timestep_s": meta.get("timestep_s"),
        "stride": meta.get("stride"),
        "time_source": "mjData.time of the replay (reset state, then every stride-th post-action state "
                       "and the last one); the recording stores no timestamps",
        "frame_steps": np.asarray(arrays["step"]).tolist() if "step" in arrays else None,
        "n_actions": meta.get("n_actions"), "replayed_actions": meta.get("replayed_actions"),
        "replay_error": error,
        "actions_clipped": bool(meta.get("actions_clipped")),
        "clip": meta.get("clip"),
        "success_source": "replay: BiGym env success check after any replayed step "
                          "(DemoPlayer.validate_in_env rule)",
        "success_any": meta.get("success_any"), "success_final": meta.get("success_final"),
        "first_success_step": meta.get("first_success_step"),
        "first_success_time_s": meta.get("first_success_time_s"),
        "recorded_termination_count": meta.get("recorded_termination_count"),
        "recorded_first_termination_step": meta.get("recorded_first_termination_step"),
        "recorded_label_note": "termination flags mark success OR failure at recording time; "
                               "the release has no success label",
        "model_export_check": meta.get("model_export"),
        "state_route": "bigym_replay_kinematic_only" if model.missing else "bigym_replay",
        "assets_loaded": asset_dir is not None and not model.missing,
        "missing_assets": model.missing, "mesh_inertia_shell": model.shell_meshes,
        "effectors": {k: g.describe(model) for k, g in model.grippers.items()},
        "opening": "1 - mean normalized 2F-85 driver joint angle (0 = closed, 1 = open)",
        "width": "gap between the inner faces of the two pad collision boxes along +y",
        "floor_z_in_source_m": floor_z,
        "world_offset_m": [0.0, 0.0, 0.0 - floor_z],
        "world_frame": "BiGym world; floor plane (body 'floor') at z = 0, so no translation"
                       if floor_z == 0.0 else "translated so the floor plane is z = 0 (scene omitted)",
        "base_source": f"{meta['robot']['pelvis']} body world pose (x, y, yaw); torso_height = pelvis z",
        "articulation_roots": sorted(model.articulations),
        "inactive_free_bodies": model.inactive,
        "scene_reference": None if scene is None else scene.reference,
        "task_group": task_group(task),
        "instruction_source": "derived from the BiGym task class docstring (recordings carry no language)",
        "env_class": meta.get("env_class"),
        "simulation_assumptions": SIMULATION_ASSUMPTIONS,
    }
    others = [v for v in meta.get("zip_versions_of_recording", []) if v != meta["zip_member"]]
    lineage = {"generated": False, "recording_uuid": meta["uuid"],
               "replay_variant": "clipped_actions" if meta.get("actions_clipped") else "recorded_actions",
               "other_versions": others,
               "other_versions_note": "same recording converted to the other action mode; not an independent demo"}
    return SourceEpisode(
        family=family, dataset=f"bigym/v0.9.0/{task}", episode_id=meta["uuid"], task=task, time=time,
        effectors=effectors, objects=objects, base=base, torso_height=torso,
        articulations={n: Articulation(joint_names=[model.joint_names[j] for j in js], qpos=art[n])
                       for n, js in model.articulations.items()},
        scene=scene, instruction=_instruction(meta.get("env_doc")), success=success,
        regime="mobile_manipulation", license=zip_entry.license if zip_entry else "Apache-2.0",
        provenance=provenance, lineage=lineage)


def _floor_z(model) -> float:
    """World z of the floor plane geom (BiGym: geom ``floor`` of body ``floor``)."""
    mj, m = model.mj, model.m
    jointed = set(m.jnt_bodyid)
    planes = [g for g in range(m.ngeom) if m.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE
              and (m.geom_bodyid[g] == 0 or (m.body_parentid[m.geom_bodyid[g]] == 0
                                             and m.geom_bodyid[g] not in jointed))]
    if not planes:
        raise ValueError("no floor plane in the recorded model")
    d = mj.MjData(m)
    mj.mj_kinematics(m, d)
    zs = {round(float(d.geom_xpos[g][2]), 9) for g in planes}
    if len(zs) != 1:
        raise ValueError(f"ambiguous floor planes at z = {sorted(zs)}")
    return float(zs.pop()) + 0.0


def _code_entry(meta, catalog):
    url = ((meta.get("runtime_versions") or {}).get("bigym_direct_url") or {}).get("url") or ""
    name = os.path.basename(url)
    return next((e for e in catalog.values() if e.id.startswith(CODE_ID_PREFIX) and e.path.endswith(name)
                 and name), None)


# ---------------------------------------------------------------- catalog helpers

def list_recordings(zip_path) -> dict:
    """Per task: unique recording UUIDs and how many files store them (offline, local zip)."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".safetensors") and "/lightweight/" in n]
    out: dict[str, dict] = {}
    for n in names:
        task, mode = n.split("/")[:2]
        t = out.setdefault(task, {"uuids": set(), "files": 0, "modes": set()})
        t["uuids"].add(n.split("/")[-1].split(".")[0])
        t["files"] += 1
        t["modes"].add(mode)
    return out


__all__ = ["DEMO_ZIP_ID", "TASK_GROUPS", "list_recordings", "read_bigym", "read_record", "replay",
           "resolve_assets", "task_group"]
