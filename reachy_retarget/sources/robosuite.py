"""Adapter for robosuite-recorded HDF5 files (robomimic v1.5, MimicGen, and the shared
reader used by the LIBERO and DexMimicGen adapters).

Every demo stores ``states`` rows ``[sim time, qpos, qvel]`` and the complete MJCF it
was recorded with (``model_file`` attribute). The adapter compiles that MJCF with the
MuJoCo bindings, sets ``qpos``/``qvel`` from each row, runs ``mj_forward`` and reads
exact simulator kinematics: grasp-site poses, finger separation, free-object poses,
static fixtures and scene articulations. No robosuite code is imported and nothing is
ever stepped.

Mesh and texture paths in the MJCF are absolute paths on the recording machine. They
are resolved inside pinned asset archives read as zip files: the robosuite wheel of the
recorded ``env_version`` (``robosuite/models/assets/``) plus every catalogued
``kind: assets`` archive whose ``asset_marker`` (or a family alias of it) occurs in the
MJCF, e.g. MimicGen's ``mimicgen_envs/models/robosuite/assets/`` or LIBERO's
``libero/libero/assets/``. When assets are missing, mesh geoms are replaced by
placeholders: kinematics stay exact but the episode gets no physics scene
(``provenance["state_route"] == "mjcf_kinematic_only"``).

Gripper convention: the contract frame (+z approach from palm toward the fingertips,
+y closing) is derived from the model geometry, not assumed. At a reference
configuration with the fingers open, approach = palm (finger parent body origin) ->
midpoint of the two finger pads, closing = first-finger pad -> second-finger pad. The
grasp center is that pad midpoint, a fixed offset from the robosuite grip site.

Stripped ``*.state.hdf5`` copies (see :mod:`reachy_retarget.acquire.strip`) are read
transparently: the catalog entry is found through the original SHA-256 they record.

Family specifics (task tables, instructions, lineage, asset aliases) live in a
:class:`Profile`; ``libero.py`` and ``dexmimicgen.py`` register their own profiles.
"""
from __future__ import annotations

import json
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from ..acquire import CatalogEntry, identify, load_catalog
from ..schema.source import Effector, ObjectTrack, Articulation, SceneRef, SourceEpisode
from .registry import register

ASSET_MARKER = "robosuite/models/assets/"
ROBOT_PREFIX = re.compile(r"^((?:robot|gripper|mount|fixed_mount)\d+_)")
GRIP_SITE = re.compile(r"^(gripper\d+(?:_(?:right|left))?)_grip_site$")
PARALLEL_YAW_TOL = np.deg2rad(30)

# Task-relevant free bodies and roles per robosuite env (object name = body minus
# "_main"/"_root"). Other free bodies are inactive (e.g. hidden Milk/Bread/Cereal in
# PickPlaceCan) and only listed in provenance. Unknown envs keep every free body.
TASK_OBJECTS = {
    "Lift": {"cube": "manipulated"},
    "PickPlaceCan": {"Can": "manipulated"},
    "NutAssemblySquare": {"SquareNut": "manipulated"},
    "ToolHang": {"tool": "manipulated", "frame": "manipulated", "stand": "support"},
    "TwoArmTransport": {"payload": "manipulated", "trash": "manipulated",
                        "transport_start_bin_lid": "manipulated",
                        "transport_start_bin": "receptacle", "transport_target_bin": "receptacle",
                        "transport_trash_bin": "receptacle"},
    "Stack": {"cubeA": "manipulated", "cubeB": "support"},
    "Square": {"SquareNut": "manipulated"},          # MimicGen Square = NutAssemblySquare
    "Threading": {"needle_obj": "manipulated", "tripod_obj": "support"},
    "TwoArmThreading": {"needle_obj": "manipulated", "tripod_obj": "manipulated"},
}


def _env_key(env_name: str) -> str:
    """MimicGen variants share the base task table: ``Stack_D1`` -> ``Stack``."""
    return re.sub(r"_[DO]\d+$", "", env_name)


# ---------------------------------------------------------------- MJCF and assets

class AssetArchive:
    """Model assets read from a zip (wheel or source archive).

    Members after ``marker`` are addressable; recorded file references are matched on
    ``marker`` or any of ``aliases`` (older package names of the same tree, e.g. LIBERO's
    ``chiliocosm/assets/``). Resolved assets are named ``prefix + relative path``.
    """

    def __init__(self, path, marker: str = ASSET_MARKER, aliases=(), prefix: str = ""):
        self.path = Path(path)
        self.marker, self.aliases, self.prefix = marker, tuple(aliases), prefix
        self.zip = zipfile.ZipFile(self.path)
        self.names = {posixpath.normpath(n.split(marker, 1)[1]): n for n in self.zip.namelist()
                      if marker in n and not n.endswith("/")}

    def rel(self, file_attr: str) -> str | None:
        for m in (self.marker, *self.aliases):
            if m in file_attr:
                return posixpath.normpath(file_attr.split(m, 1)[1])
        return None

    def read(self, rel: str) -> bytes | None:
        name = self.names.get(rel)
        return self.zip.read(name) if name else None


def _asset_rel(file_attr: str, marker: str = ASSET_MARKER) -> str | None:
    if marker not in file_attr:
        return None
    return posixpath.normpath(file_attr.split(marker, 1)[1])


def resolve_mjcf(xml: str, archives):
    """Rewrite file references to archive-relative names.

    ``archives`` is ``None``, one :class:`AssetArchive` or a sequence of them (the
    first archive whose marker matches a reference serves it). Returns ``(xml, assets,
    missing)``. ``assets`` maps names to bytes for ``MjModel.from_xml_string``. If
    anything is missing, mesh geoms become small spheres and file textures are dropped,
    so the returned XML still compiles with exact kinematics but must not be used for
    physics.
    """
    if archives is None:
        archives = []
    elif isinstance(archives, AssetArchive):
        archives = [archives]
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
    assets, missing, by_base = {}, [], {}
    for el in root.iter():
        f = el.get("file")
        if not f:
            continue
        data = None
        for a in archives:
            rel = a.rel(f)
            if rel is not None:
                data = a.read(rel)
                break
        if data is None:
            missing.append(f)
            continue
        key = a.prefix + rel
        # MuJoCo's asset dictionary is keyed by base name: identical files share one key,
        # different files with the same base name get a flattened (unique) name.
        other = by_base.get(posixpath.basename(key))
        if other is not None and other != key:
            key = other if assets[other] == data else key.replace("/", "__")
        by_base.setdefault(posixpath.basename(key), key)
        assets[key] = data
        el.set("file", key)
    if missing:
        assets = {}
        asset = root.find("asset")
        dropped = [el for el in (asset if asset is not None else []) if el.get("file")]
        textures = {el.get("name") for el in dropped if el.tag == "texture"}
        for el in dropped:
            asset.remove(el)
        for mat in root.iter("material"):
            if mat.get("texture") in textures:
                mat.attrib.pop("texture")
        for geom in root.iter("geom"):
            if geom.get("mesh") or geom.get("type") == "mesh":
                geom.attrib.pop("mesh", None)
                geom.set("type", "sphere")
                geom.set("size", "0.001")
    return ET.tostring(root, encoding="unicode"), assets, missing


def _compile_tolerant(mujoco, xml: str, assets: dict):
    """Compile ``xml``; meshes rejected by newer MuJoCo versions as having too small a
    volume (thin visual shells such as the Sawyer head) get ``inertia="shell"``, the fix
    MuJoCo itself suggests. Returns ``(model, xml, shell_meshes)``."""
    shell = []
    while True:
        try:
            return mujoco.MjModel.from_xml_string(xml, assets), xml, shell
        except ValueError as err:
            mt = re.search(r"mesh volume is too small: (\S+?) \.", str(err))
            if not mt or mt.group(1) in shell or len(shell) > 50:
                raise
            root = ET.fromstring(xml)
            mesh = next(el for el in root.iter("mesh") if el.get("name") == mt.group(1))
            mesh.set("inertia", "shell")
            shell.append(mt.group(1))
            xml = ET.tostring(root, encoding="unicode")


class _Model:
    """A compiled recorded model plus the geometry derived from it once."""

    def __init__(self, xml: str, archive: AssetArchive | None):
        import mujoco

        self.mj = mujoco
        self.xml, self.assets, self.missing = resolve_mjcf(xml, archive)
        self.m, self.xml, self.shell_meshes = _compile_tolerant(mujoco, self.xml, self.assets)
        self.d = mujoco.MjData(self.m)
        m = self.m
        name = lambda kind, i: mujoco.mj_id2name(m, kind, i) or ""
        self.body_names = [name(mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        self.joint_names = [name(mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        self.robot_prefixes = sorted({mt.group(1) for n in self.body_names if (mt := ROBOT_PREFIX.match(n))})
        is_robot = lambda b: any(self.body_names[m.body_rootid[b]].startswith(p) for p in self.robot_prefixes)
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        self.free = {re.sub(r"_(main|root)$", "", self.body_names[m.jnt_bodyid[j]]): j
                     for j in range(m.njnt) if m.jnt_type[j] == free}
        scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        self.articulations: dict[str, list[int]] = {}
        for j in range(m.njnt):
            b = m.jnt_bodyid[j]
            if m.jnt_type[j] in scalar and not is_robot(b):
                self.articulations.setdefault(self.body_names[m.body_rootid[b]], []).append(j)
        collides = (m.geom_contype | m.geom_conaffinity) != 0
        jointed = set(m.body_rootid[m.jnt_bodyid])
        self.fixtures = [b for b in range(1, m.nbody)
                         if m.body_parentid[b] == 0 and b not in jointed and not is_robot(b)
                         and collides[m.body_rootid[m.geom_bodyid] == b].any()]
        self.grippers = {}
        for s in range(m.nsite):
            mt = GRIP_SITE.match(name(mujoco.mjtObj.mjOBJ_SITE, s))
            if mt:
                self.grippers[mt.group(1)] = _Gripper(self, mt.group(1), s)

    def set_state(self, row: np.ndarray):
        m, d = self.m, self.d
        d.qpos[:] = row[1:1 + m.nq]
        d.qvel[:] = row[1 + m.nq:1 + m.nq + m.nv]
        self.mj.mj_forward(m, d)

    def body_aabb(self, b: int) -> dict:
        """Collision-geometry AABB of the tree under world-child body ``b``, in ``b``'s frame.

        Empty when assets were missing (placeholder geoms carry no real extent).
        """
        m, d = self.m, self.d
        if self.missing:
            return {}
        pts = []
        for g in np.flatnonzero((m.body_rootid[m.geom_bodyid] == b)
                                & ((m.geom_contype | m.geom_conaffinity) != 0)):
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

    def in_subtree(self, b: int, root: int) -> bool:
        while b > 0 and b != root:
            b = self.m.body_parentid[b]
        return b == root


class _Gripper:
    """Two-finger parallel gripper measured from model geometry."""

    def __init__(self, model: _Model, prefix: str, site: int):
        m, d, mj = model.m, model.d, model.mj
        self.site = site
        fingers = sorted(j for j, n in enumerate(model.joint_names)
                         if n.startswith(prefix + "_") and "finger" in n
                         and m.jnt_type[j] in (int(mj.mjtJoint.mjJNT_SLIDE), int(mj.mjtJoint.mjJNT_HINGE)))
        pads = [[g for g in range(m.ngeom) if "pad" in (mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or "")
                 and model.in_subtree(m.geom_bodyid[g], m.jnt_bodyid[j])] for j in fingers]
        if len(fingers) != 2 or any(len(p) != 1 for p in pads):
            raise ValueError(f"{prefix}: expected two finger joints with one pad each (parallel gripper)")
        self.fingers, self.pads = fingers, [p[0] for p in pads]
        palm = m.body_parentid[m.jnt_bodyid[fingers[0]]]
        rng = m.jnt_range[fingers]
        open_q = np.where(np.abs(rng[:, 1]) >= np.abs(rng[:, 0]), rng[:, 1], rng[:, 0])
        closed_q = np.where(np.abs(rng[:, 1]) >= np.abs(rng[:, 0]), rng[:, 0], rng[:, 1])
        adr = m.jnt_qposadr[fingers]

        def measure(q):
            d.qpos[:] = m.qpos0
            d.qpos[adr] = q
            mj.mj_kinematics(m, d)
            R = d.site_xmat[site].reshape(3, 3)
            p1, p2 = d.geom_xpos[self.pads[0]], d.geom_xpos[self.pads[1]]
            return R, d.site_xpos[site].copy(), p1.copy(), p2.copy(), d.xpos[palm].copy()

        R, ps, p1, p2, pp = measure(open_q)
        mid = (p1 + p2) / 2
        z = R.T @ (mid - pp)
        z /= np.linalg.norm(z)
        y = R.T @ (p2 - p1)
        y -= z * (y @ z)
        y /= np.linalg.norm(y)
        fix = np.column_stack([np.cross(y, z), y, z])
        snapped = np.round(fix)
        self.R_fix = snapped if np.abs(fix - snapped).max() < 1e-3 and abs(np.linalg.det(snapped) - 1) < 1e-9 else fix
        self.offset = R.T @ (mid - ps)           # grasp center in grip-site frame
        sep_open = (p2 - p1) @ (R @ self.R_fix[:, 1])
        R, _, p1, p2, _ = measure(closed_q)
        self.sep_closed = (p2 - p1) @ (R @ self.R_fix[:, 1])
        self.width_max = sep_open - self.sep_closed

    def read(self, d) -> tuple[np.ndarray, float]:
        """Contract pose (4x4) and finger width (m) from the current MjData."""
        Rs = d.site_xmat[self.site].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = Rs @ self.R_fix
        T[:3, 3] = d.site_xpos[self.site] + Rs @ self.offset
        sep = (d.geom_xpos[self.pads[1]] - d.geom_xpos[self.pads[0]]) @ T[:3, 1]
        return T, max(0.0, sep - self.sep_closed)

    def describe(self, model: _Model) -> dict:
        mj, m = model.mj, model.m
        return {"site": mj.mj_id2name(m, mj.mjtObj.mjOBJ_SITE, self.site),
                "finger_joints": [model.joint_names[j] for j in self.fingers],
                "pads": [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) for g in self.pads],
                "site_to_contract_rotation": self.R_fix.round(6).tolist(),
                "grasp_center_in_site_frame_m": self.offset.round(6).tolist(),
                "width_max_m": round(float(self.width_max), 6)}




@lru_cache(maxsize=8)
def _compile(xml: str, specs: tuple) -> _Model:
    """``specs``: tuple of ``(path, marker, aliases, prefix)`` archive descriptions."""
    return _Model(xml, [AssetArchive(p, m, a, x) for p, m, a, x in specs])


# ---------------------------------------------------------------- family profiles

def _seed_task(stem: str) -> str:
    """``core/coffee_d1`` / ``robot/square_d0_iiwa`` / ``object/mug_cleanup_o1`` -> task."""
    return re.sub(r"_[do]\d+(_[a-z0-9]+)?$", "", stem)


def _default_lineage(entry: CatalogEntry | None, dataset: str) -> dict:
    if entry is None:
        return {"generated": None}
    parts = entry.dataset.split("/") if entry.dataset else []
    if parts[:1] == ["mimicgen"] and len(parts) == 3 and parts[1] != "source":
        return {"generated": True, "seed": f"mimicgen/source/{_seed_task(parts[2])}",
                "seed_demo": None,  # not recorded per demo in the published files
                "variant_group": parts[1]}
    return {"generated": False}


def _default_success(g, kwargs) -> bool | None:
    """Max sparse reward >= 1; unknown when rewards are missing or shaped."""
    if "rewards" not in g or kwargs.get("reward_shaping"):
        return None
    return bool(np.asarray(g["rewards"][:]).max() >= 1.0)


@dataclass
class Profile:
    """Family-specific metadata hooks of the shared robosuite reader."""

    task_objects: Callable[[str, h5py.File], dict | None] = \
        lambda env_name, f: TASK_OBJECTS.get(_env_key(env_name))
    instruction: Callable[[h5py.File], str | None] = lambda f: None
    lineage: Callable[[CatalogEntry | None, str], dict] = _default_lineage
    extra_provenance: Callable[[h5py.File, dict], dict] = lambda f, archives: {}
    # archive asset_marker -> older recorded markers of the same tree
    marker_aliases: dict = field(default_factory=dict)
    # robosuite wheel used when env_args carries no env_version
    default_env_version: str | None = None
    success: Callable[[h5py.Group, dict], bool | None] = _default_success
    success_rule: str = "max sparse reward >= 1"  # provenance text of the success label


ROBOSUITE = Profile()


# ---------------------------------------------------------------- episodes

def _data_root(path: Path, root):
    if root is not None:
        return Path(root)
    return next((p.parent for p in path.resolve().parents if p.name == "raw"), None)


def _locate_archives(env_version: str | None, xml: str, path: Path, root, catalog, profile: Profile):
    """Local asset archives needed by ``xml``: ``[(spec, entry)]`` plus notes."""
    root = _data_root(path, root)
    out, notes = [], {}
    version = env_version or profile.default_env_version
    if env_version is None and version:
        notes["env_version_assumed"] = version
    wheel = catalog.get(f"robosuite/robosuite-{version}-py3-none-any.whl") if version else None
    if wheel is not None and root is not None and wheel.local_path(root).exists():
        out.append(((str(wheel.local_path(root)), ASSET_MARKER, (), ""), wheel))
    for e in catalog.values():
        if e.kind != "assets" or not e.asset_marker or e.asset_marker == ASSET_MARKER:
            continue
        aliases = tuple(profile.marker_aliases.get(e.asset_marker, ()))
        if not any(m in xml for m in (e.asset_marker, *aliases)):
            continue
        if root is not None and e.local_path(root).exists():
            out.append(((str(e.local_path(root)), e.asset_marker, aliases, e.asset_marker), e))
    # Longer markers first, so a nested tree never shadows a more specific archive.
    out.sort(key=lambda se: -len(se[0][1]))
    return out, notes


def _yaw(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _side_hints(bases: dict[str, list[float]], grippers) -> tuple[dict, str | None]:
    """Left/right from the robot base layout when all arms face the same way.

    Arms facing each other (robomimic Transport) or a single arm get no hint. For
    side-by-side arms, the base with the largest offset along the shared facing
    direction's left normal is ``left``, the smallest ``right``.
    """
    owner = {g: f"robot{re.match(r'gripper(\d+)', g).group(1)}_base" for g in grippers}
    if len(set(owner.values())) < 2 or not all(o in bases for o in owner.values()):
        return {}, None
    yaws = np.array([bases[o][2] for o in set(owner.values())])
    mean = np.arctan2(np.sin(yaws).mean(), np.cos(yaws).mean())
    if np.abs(np.angle(np.exp(1j * (yaws - mean)))).max() > PARALLEL_YAW_TOL:
        return {}, None
    left = np.array([-np.sin(mean), np.cos(mean)])
    lateral = {g: float(np.array(bases[o][:2]) @ left) for g, o in owner.items()}
    lo, hi = min(lateral.values()), max(lateral.values())
    if hi - lo < 1e-3:
        return {}, None
    return ({g: "left" if v == hi else "right" if v == lo else None for g, v in lateral.items()},
            "robot base layout: arms face the same way; lateral offset along their left normal")


def read_robosuite_family(path: Path, *, family: str, profile: Profile = ROBOSUITE, demos=None,
                          root=None, asset_archive=None, catalog=None, with_scene: bool = True):
    """Yield one :class:`SourceEpisode` per demo of a robosuite-recorded HDF5 file.

    ``demos`` selects demo keys (default: all, in numeric order). ``root`` is the data
    root holding ``raw/<family>/...`` asset archives (inferred from ``path`` when it
    lies under a ``raw/`` folder); ``asset_archive`` overrides the robosuite asset zip
    explicitly (then no other archive is used).
    """
    path = Path(path)
    catalog = catalog if catalog is not None else load_catalog()
    entry, ident = identify(path, catalog)
    with h5py.File(path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        env_name, env_version = env_args["env_name"], env_args.get("env_version")
        kwargs = env_args.get("env_kwargs", {})
        control_freq = float(kwargs["control_freq"])
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1])) if demos is None else list(demos)
        masks = {k: {v.decode() for v in f["mask"][k][:]} for k in f["mask"]} if "mask" in f else {}
        dataset = entry.dataset if entry and entry.dataset else f"{family}/{path.name.split('.')[0]}"
        ctx = dict(family=family, dataset=dataset, env_name=env_name, env_version=env_version,
                   control_freq=control_freq, kwargs=kwargs, roles=profile.task_objects(env_name, f),
                   entry=entry, ident=ident, path=path, masks=masks, with_scene=with_scene,
                   instruction=profile.instruction(f), lineage=profile.lineage(entry, dataset),
                   success_fn=profile.success, success_rule=profile.success_rule)
        extra = None
        for key in keys:
            g = f["data"][key]
            xml = g.attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode()
            if asset_archive is not None:
                found, notes = [((str(asset_archive), ASSET_MARKER, (), ""), None)], {}
            else:
                found, notes = _locate_archives(env_version, xml, path, root, catalog, profile)
            if extra is None:
                extra = profile.extra_provenance(f, {"archives": found})
            model = _compile(xml, tuple(spec for spec, _ in found))
            yield _episode(model, g, key, archives=found, archive_notes=notes, extra=extra, **ctx)


@register("robomimic", "mimicgen")
def read_robosuite_hdf5(path: Path, *, family: str, **kw):
    """robomimic / MimicGen files (see :func:`read_robosuite_family`)."""
    return read_robosuite_family(path, family=family, profile=ROBOSUITE, **kw)


def _episode(model, g, key, *, family, dataset, env_name, env_version, control_freq, kwargs, roles,
             entry, ident, path, masks, with_scene, instruction, lineage, success_fn, success_rule,
             archives, archive_notes, extra):
    m, d = model.m, model.d
    states = np.asarray(g["states"][:], float)
    if states.ndim != 2 or states.shape[1] != 1 + m.nq + m.nv + m.na:
        raise ValueError(f"{key}: states width {states.shape} does not match 1+nq+nv={1 + m.nq + m.nv}")
    T = len(states)
    time = states[:, 0] - states[0, 0]
    time_source = "recorded simulator time"
    if not np.all(np.diff(time) > 0):
        time, time_source = np.arange(T) / control_freq, "index / control_freq (recorded time not increasing)"

    poses = {k: np.zeros((T, 4, 4)) for k in model.grippers}
    widths = {k: np.zeros(T) for k in model.grippers}
    obj_names = [n for n in model.free if roles is None or n in roles]
    obj_pose = {n: np.zeros((T, 7)) for n in obj_names}
    art = {n: np.zeros((T, len(js))) for n, js in model.articulations.items()}
    for t in range(T):
        model.set_state(states[t])
        for k, gr in model.grippers.items():
            poses[k][t], widths[k][t] = gr.read(d)
        for n in obj_names:
            b = m.jnt_bodyid[model.free[n]]
            obj_pose[n][t] = np.r_[d.xpos[b], d.xquat[b]]
        for n, js in model.articulations.items():
            art[n][t] = d.qpos[m.jnt_qposadr[js]]

    model.set_state(states[0])
    bases = {n: [float(d.xpos[i][0]), float(d.xpos[i][1]), _yaw(d.xmat[i].reshape(3, 3))]
             for i, n in enumerate(model.body_names) if re.fullmatch(r"robot\d+_base", n)}
    sides, side_source = _side_hints(bases, model.grippers)
    effectors = {k: Effector(pose=poses[k], width=widths[k],
                             opening=np.clip(widths[k] / model.grippers[k].width_max, 0, 1),
                             side_hint=sides.get(k))
                 for k in model.grippers}
    objects = {n: ObjectTrack(pose=obj_pose[n], valid=np.ones(T, bool),
                              role=(roles or {}).get(n, "manipulated"),
                              geometry={**model.body_aabb(m.jnt_bodyid[model.free[n]]),
                                        "body": model.body_names[m.jnt_bodyid[model.free[n]]]})
               for n in obj_names}
    for b in model.fixtures:
        name = model.body_names[b]
        pose = np.r_[d.xpos[b], d.xquat[b]]
        role = "support" if "table" in name else "receptacle" if "bin" in name else "fixture"
        objects.setdefault(name, ObjectTrack(pose=np.tile(pose, (T, 1)), valid=np.ones(T, bool),
                                             role=role, geometry={**model.body_aabb(b), "body": name}))

    initial_qpos = {model.joint_names[j]: d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 7].tolist()
                    for j in model.free.values()}
    initial_qpos.update({model.joint_names[j]: float(d.qpos[m.jnt_qposadr[j]])
                         for js in model.articulations.values() for j in js})
    route = "mjcf_kinematic_only" if model.missing else "mjcf_states"
    scene = None
    if with_scene and not model.missing:
        scene = SceneRef(mjcf=model.xml, robot_prefixes=model.robot_prefixes,
                         initial_qpos=initial_qpos, assets=model.assets)

    success = success_fn(g, kwargs)

    wheel = next(((spec, e) for spec, e in archives if spec[1] == ASSET_MARKER), None)
    provenance = {
        "file": str(path), "sha256": ident["sha256"], "demo_key": key,
        "file_sha256": ident["file_sha256"], "stripped": ident["stripped"] is not None,
        "url": entry.url if entry else None, "revision": entry.revision if entry else None,
        "catalog_id": entry.id if entry else None,
        "env_name": env_name, "env_version": env_version, "control_freq_hz": control_freq,
        "sim_timestep_s": float(m.opt.timestep), "time_source": time_source,
        "state_route": route, "state_layout": "time, qpos, qvel (robosuite sim state)",
        "asset_archive": None if wheel is None else {
            "path": wheel[0][0], "id": wheel[1].id if wheel[1] else None,
            "sha256": wheel[1].sha256 if wheel[1] else None},
        "asset_archives": [{"path": spec[0], "marker": spec[1], "aliases": list(spec[2]),
                            "id": e.id if e else None, "sha256": e.sha256 if e else None,
                            "license": e.license if e else None} for spec, e in archives],
        **archive_notes,
        "missing_assets": model.missing,
        "mesh_inertia_shell": model.shell_meshes,
        "effectors": {k: gr.describe(model) for k, gr in model.grippers.items()},
        "opening": "pad separation along +y minus fully-closed separation, / its open-closed range",
        "side_hint_source": side_source,
        "inactive_free_bodies": sorted(set(model.free) - set(obj_names)),
        "articulation_roots": sorted(model.articulations),
        "robot_bases_xy_yaw": bases,
        "splits": sorted(k for k, v in masks.items() if key in v),
        "success_source": None if success is None else success_rule,
        "robots": kwargs.get("robots"),
        "env_configuration": kwargs.get("env_configuration"),
        **extra,
    }
    first_base = bases.get("robot0_base")
    return SourceEpisode(
        family=family, dataset=dataset, episode_id=key, task=env_name, time=time,
        effectors=effectors, objects=objects, base_hint=None if first_base is None else np.array(first_base),
        articulations={n: Articulation(joint_names=[model.joint_names[j] for j in js], qpos=art[n])
                       for n, js in model.articulations.items()},
        scene=scene, instruction=instruction, success=success, regime="tabletop",
        license=entry.license if entry else "unknown", provenance=provenance, lineage=dict(lineage))


__all__ = ["ASSET_MARKER", "AssetArchive", "Profile", "ROBOSUITE", "TASK_OBJECTS", "read_robosuite_family",
           "read_robosuite_hdf5", "resolve_mjcf"]
