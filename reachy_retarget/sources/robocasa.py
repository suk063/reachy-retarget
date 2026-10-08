"""Adapter for RoboCasa365 LeRobot datasets with simulator extras (family ``robocasa``).

Input is one extracted dataset directory (a ``tar_stream`` catalog entry fetched with
:func:`reachy_retarget.acquire.fetch`, which keeps only non-image members): the folder that
holds ``lerobot/`` (or the ``lerobot`` folder itself) with

* ``extras/episode_<k>/model.xml.gz``: the complete MJCF the episode was recorded with,
* ``extras/episode_<k>/states.npz``: ``states`` rows ``[sim time, qpos, qvel]``,
* ``extras/episode_<k>/ep_meta.json``: language (``lang``), task objects
  (``object_cfgs``), referenced fixtures (``fixture_refs``), layout/style ids,
* ``extras/dataset_meta.json``: env name, RoboCasa/robosuite/MuJoCo versions, controller,
* ``data/chunk-*/episode_<k>.parquet`` (optional): rewards and recorded observations.

The MJCF is compiled with MuJoCo, each state row is set and ``mj_forward`` gives exact
simulator kinematics; nothing is stepped and no RoboCasa or robosuite code is imported.
The replay itself (state rows, Panda grasp-center convention, MJCF asset rewriting,
mesh-inertia fallback) is shared with :mod:`.robosuite`.

Robot: PandaOmron = Panda arm and gripper on the Omron holonomic base with a torso lift.
The base pose is the world pose of body ``mobilebase0_base`` (it moves through the
``mobilebase0_joint_mobile_{forward,side,yaw}`` joints in the frame of ``robot0_base``);
the torso height is the ``mobilebase0_joint_torso_height`` value (0 to 0.34 m).

Assets: references under ``robocasa/models/assets/`` resolve in the catalogued RoboCasa
archives (GitHub source zip, HF ``robocasa/robocasa-assets`` zips, Box lightwheel zips)
and, as a fallback, in ``raw/robocasa/asset_subset/`` (CRC-checked members copied out of
those archives by :func:`reachy_retarget.acquire.assets.fetch_robocasa_asset_subset`); references under
``robosuite/models/assets/`` resolve in the robosuite wheel of the recorded version.
Missing assets give the kinematic-only route (``provenance["state_route"]``).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import posixpath
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..acquire import TAR_MANIFEST, load_catalog
from ..schema.source import Articulation, Effector, ObjectTrack, SceneRef, SourceEpisode
from .registry import register
from .robosuite import ASSET_MARKER as ROBOSUITE_MARKER
from .contact_reference import ObjectEnvironmentDepth
from .robosuite import PARK_DISTANCE_M, AssetArchive, _Model

ASSET_MARKER = "robocasa/models/assets/"
BASE_BODY = "mobilebase0_base"
BASE_JOINTS = ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw")
TORSO_JOINT = "mobilebase0_joint_torso_height"
EXTRA_ROBOT_PREFIXES = ("mobilebase0_",)
BASE_MOVE_TOL_M = 0.01      # planar displacement from the first frame that counts as base motion
BASE_MOVE_TOL_RAD = 0.01
FLOOR_TOL_M = 0.005         # |floor top z| above this is refused (contract: floor at z = 0)
ART_MOVE_TOL = 1e-3          # joint change (rad or m) that counts as an articulation being operated
SUBSET_DIR = "asset_subset"
FLAT_SEP = "__"
LICENSE = "CC-BY-4.0"        # RoboCasa README: assets and datasets CC BY 4.0, code MIT

# Fixture classes (ep_meta "fixtures"[name]["cls"]) and the object role they get when the
# task references them (ep_meta "fixture_refs"). Derived labels, see docs/sources.md.
SUPPORT_CLS = {"Counter", "Stove", "Stovetop", "Island", "Box", "Stool", "Floor"}
RECEPTACLE_CLS = {"Sink", "Microwave", "HingeCabinet", "SingleCabinet", "OpenCabinet", "Drawer", "Oven",
                  "ToasterOven", "Dishwasher", "Toaster", "CoffeeMachine", "HousingCabinet", "Fridge",
                  "FridgeFrenchDoor", "FridgeSideBySide", "FridgeBottomFreezer", "ElectricKettle",
                  "StandMixer", "Blender"}


# ---------------------------------------------------------------- assets

class _Source:
    """One place RoboCasa asset members can be read from.

    ``kind`` "zip": a zip whose member names either contain ``marker`` (source archive)
    or equal the reference text after ``marker`` (asset zips such as ``textures.zip``).
    ``kind`` "dir": a directory laid out like ``robocasa/models/assets/``.
    """

    def __init__(self, kind, path, marker=ASSET_MARKER, id=None, sha256=None):
        self.kind, self.path, self.id, self.sha256 = kind, Path(path), id, sha256
        self.sub = marker[len(ASSET_MARKER):] if marker.startswith(ASSET_MARKER) else ""
        if kind == "zip":
            self.zip = zipfile.ZipFile(self.path)
            names = [n for n in self.zip.namelist() if not n.endswith("/")]
            if any(ASSET_MARKER in n for n in names):
                self.names = {posixpath.normpath(n.split(ASSET_MARKER, 1)[1]): n for n in names if ASSET_MARKER in n}
            else:
                self.names = {posixpath.normpath(self.sub + n): n for n in names}

    def read(self, rel: str) -> bytes | None:
        if self.kind == "dir":
            p = self.path / rel
            return p.read_bytes() if p.is_file() else None
        name = self.names.get(rel)
        return self.zip.read(name) if name else None

    def describe(self) -> dict:
        return {"kind": self.kind, "path": str(self.path), "id": self.id, "sha256": self.sha256,
                "subset": self.kind == "dir"}


class RobocasaAssets:
    """All RoboCasa asset sources behind one marker, duck-typed like ``AssetArchive``
    for :func:`.robosuite.resolve_mjcf` (``rel``, ``read``, ``prefix``)."""

    marker, aliases, prefix = ASSET_MARKER, (), "robocasa/"

    def __init__(self, sources):
        self.sources = list(sources)
        self.used: dict[str, int] = {}   # rel -> index of the source that served it

    def rel(self, file_attr: str) -> str | None:
        """Flattened path (``/`` -> ``__``): MuJoCo's asset dictionary matches base names
        case-insensitively (``Prop.obj`` vs ``prop.obj`` of two different objects), so every
        RoboCasa asset gets its whole relative path as a unique base name."""
        if ASSET_MARKER not in file_attr:
            return None
        return posixpath.normpath(file_attr.split(ASSET_MARKER, 1)[1]).replace("/", FLAT_SEP)

    def read(self, rel: str) -> bytes | None:
        rel = rel.replace(FLAT_SEP, "/")
        for i, s in enumerate(self.sources):
            data = s.read(rel)
            if data is not None:
                self.used[rel] = i
                return data
        return None

    def usage(self) -> list[dict]:
        counts = np.bincount(list(self.used.values()), minlength=len(self.sources)) if self.used else \
            np.zeros(len(self.sources), int)
        return [{**s.describe(), "members_used": int(c)} for s, c in zip(self.sources, counts)]


def asset_sources_from_catalog(catalog) -> list[dict]:
    """Catalogued RoboCasa asset zips: ``{"id", "marker", "member": rel -> zip member name}``.

    Used by :func:`reachy_retarget.acquire.assets.fetch_robocasa_asset_subset` to locate members remotely. Only
    archives whose members are named relative to their marker are listed (the GitHub
    source archive is small and fetched whole).
    """
    out = []
    for e in catalog.values():
        if not _is_assets(e) or not e.asset_marker or not e.asset_marker.startswith(ASSET_MARKER):
            continue
        if e.path.endswith(".zip") and e.path.startswith("assets/"):
            sub = e.asset_marker[len(ASSET_MARKER):]
            out.append({"id": e.id, "kind": "zip", "marker": e.asset_marker,
                        "member": (lambda rel, sub=sub: rel[len(sub):] if rel.startswith(sub) else None)})
    return out


def _data_root(path: Path, root):
    if root is not None:
        return Path(root)
    return next((p.parent for p in path.resolve().parents if p.name == "raw"), None)


def locate_assets(root, catalog, robosuite_version):
    """Local asset archives: ``(RobocasaAssets | None, robosuite AssetArchive | None, notes)``."""
    sources, notes = [], {}
    wheel = None
    if root is not None:
        for e in sorted(catalog.values(), key=lambda e: e.id):
            if _is_assets(e) and e.asset_marker and e.asset_marker.startswith(ASSET_MARKER) \
                    and e.local_path(root).exists():
                sources.append(_Source("zip", e.local_path(root), e.asset_marker, e.id, e.sha256))
        subset = Path(root) / "raw" / "robocasa" / SUBSET_DIR
        if subset.is_dir():
            sources.append(_Source("dir", subset, id=f"robocasa/{SUBSET_DIR}"))
        we = catalog.get(f"robosuite/robosuite-{robosuite_version}-py3-none-any.whl")
        if we is not None and we.local_path(root).exists():
            wheel = (AssetArchive(we.local_path(root)), we)
        else:
            notes["robosuite_wheel_missing"] = f"robosuite/robosuite-{robosuite_version}-py3-none-any.whl"
    return (RobocasaAssets(sources) if sources else None), wheel, notes


@lru_cache(maxsize=4)
def _compile(xml: str, key: tuple, archives_ref) -> _Model:
    return _Model(xml, list(archives_ref[0]))


class _ArchivesRef:
    """Hashable holder so a model cache entry is tied to one set of archive objects."""

    def __init__(self, archives):
        self.archives = archives

    def __getitem__(self, i):
        return self.archives

    def __hash__(self):
        return id(self)


# ---------------------------------------------------------------- dataset layout

def _lerobot_dir(path: Path) -> Path:
    for cand in (path, path / "lerobot"):
        if (cand / "extras").is_dir():
            return cand
    raise FileNotFoundError(f"{path}: no extras/ folder (expected a RoboCasa LeRobot dataset with extras)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_assets(e) -> bool:
    return e.content == "assets" or e.kind == "assets"  # kind "assets": entries built the earlier way


def _find_tar(lerobot: Path, root, tars):
    """The ``tar_stream`` catalog entry the dataset was extracted from, as a flat view
    (catalog fields + family fields + ``box_sha1``), or ``None``."""
    if tars is None:
        tars = {k: e for k, e in load_catalog(tables=False).items() if e.family == "robocasa" and e.transport == "tar_stream"}
    rel = lerobot.parent.as_posix()
    hit = next((e for e in tars.values() if root is not None
                and e.output_dir(root).resolve() == lerobot.parent.resolve()), None) \
        or next((e for e in tars.values() if rel.endswith(posixpath.dirname(e.path))), None)
    if hit is None:
        return None
    return SimpleNamespace(**{"seed": None, "horizon": None, **hit.meta}, id=hit.id, url=hit.url,
                           dataset=hit.dataset, revision=hit.revision, license=hit.license,
                           box_sha1=hit.digests.get("sha1"))


def _read_manifest(dataset_dir: Path) -> dict | None:
    """``tar_members.json`` of an extracted tar directory (or its ``lerobot`` child)."""
    for p in (dataset_dir / TAR_MANIFEST, dataset_dir.parent / TAR_MANIFEST):
        if p.exists():
            return json.loads(p.read_text())
    return None


def _parquet(lerobot: Path, ep: str):
    idx = int(ep.split("_")[-1])
    hits = sorted(lerobot.glob(f"data/chunk-*/episode_{idx:06d}.parquet"))
    if not hits:
        return None, None
    import pyarrow.parquet as pq
    return hits[0], pq.read_table(hits[0]).to_pydict()


def _yaw(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _fixture_role(cls: str | None) -> str:
    if cls in SUPPORT_CLS:
        return "support"
    if cls in RECEPTACLE_CLS or (cls or "").startswith("Fridge"):
        return "receptacle"
    return "fixture"


# ---------------------------------------------------------------- adapter

@register("robocasa", select=True)
def read_robocasa(path: Path, *, family: str = "robocasa", episodes=None, root=None, catalog=None, tars=None,
                  with_scene: bool = True, articulations: str = "all", select=None):
    """Yield one :class:`SourceEpisode` per episode of an extracted RoboCasa dataset.

    ``episodes`` selects ``episode_<k>`` names or integer indices (default: all, sorted).
    ``root`` is the data root holding ``raw/robocasa`` and ``raw/robosuite`` archives
    (inferred from ``path`` under a ``raw/`` folder). ``articulations`` = ``"all"``
    (every scene fixture joint) or ``"task"`` (only fixtures named in ``fixture_refs``).
    """
    path = Path(path)
    lerobot = _lerobot_dir(path)
    root = _data_root(path, root)
    catalog = catalog if catalog is not None else load_catalog(tables=False)
    tar = _find_tar(lerobot, root, tars)
    manifest = _read_manifest(lerobot.parent)
    member_sha = {m["member"]: m["sha256"] for m in (manifest or {}).get("members", [])}
    dmeta_path = lerobot / "extras" / "dataset_meta.json"
    dmeta = json.loads(dmeta_path.read_text()) if dmeta_path.exists() else {}
    env_args = dmeta.get("env_args", {})
    versions = {"robocasa": dmeta.get("robocasa_version") or env_args.get("env_version"),
                "robosuite": dmeta.get("robosuite_version") or env_args.get("robosuite_version"),
                "mujoco": dmeta.get("mujoco_version") or env_args.get("mujoco_version")}
    rc_assets, wheel, notes = locate_assets(root, catalog, versions["robosuite"] or "1.5.2")
    archives = [a for a in (rc_assets, wheel[0] if wheel else None) if a is not None]
    ref = _ArchivesRef(archives)
    names = sorted(p.name for p in (lerobot / "extras").glob("episode_*") if (p / "states.npz").exists())
    if episodes is not None:
        want = {e if isinstance(e, str) else f"episode_{int(e):06d}" for e in episodes}
        names = [n for n in names if n in want]
    ctx = dict(family=family, lerobot=lerobot, tar=tar, manifest=manifest, member_sha=member_sha, dmeta=dmeta,
               versions=versions, rc_assets=rc_assets, wheel=wheel, notes=notes, with_scene=with_scene,
               articulations=articulations)
    for i, ep in enumerate(names):
        if select is not None and not select(i):
            yield None  # another shard's episode: its MJCF is never decompressed or compiled
            continue
        xml = gzip.decompress((lerobot / "extras" / ep / "model.xml.gz").read_bytes()).decode()
        model = _compile(xml, (len(xml), hashlib.sha1(xml.encode()).hexdigest()), ref)
        yield _episode(model, ep, **ctx)


def _verify_members(lerobot: Path, files: dict, member_sha: dict) -> dict:
    """SHA-256 of every file read; compared with the extraction manifest when present."""
    out, bad = {}, []
    for label, p in files.items():
        if p is None:
            continue
        h = _sha256(p)
        out[label] = h
        member = f"lerobot/{p.relative_to(lerobot).as_posix()}"
        if member in member_sha and member_sha[member] != h:
            bad.append(member)
    if bad:
        raise ValueError(f"files differ from the extraction manifest: {bad}")
    return out


def _episode(model, ep, *, family, lerobot, tar, manifest, member_sha, dmeta, versions, rc_assets, wheel, notes,
             with_scene, articulations):
    m, d = model.m, model.d
    ex = lerobot / "extras" / ep
    meta = json.loads((ex / "ep_meta.json").read_text())
    states = np.asarray(np.load(ex / "states.npz")["states"], float)
    if states.ndim != 2 or states.shape[1] != 1 + m.nq + m.nv + m.na:
        raise ValueError(f"{ep}: states width {states.shape} does not match 1+nq+nv={1 + m.nq + m.nv}")
    pq_path, pq = _parquet(lerobot, ep)
    files = _verify_members(lerobot, {"model.xml.gz": ex / "model.xml.gz", "states.npz": ex / "states.npz",
                                      "ep_meta.json": ex / "ep_meta.json", "parquet": pq_path}, member_sha)
    T = len(states)
    fps = float(json.loads((lerobot / "meta" / "info.json").read_text())["fps"]) \
        if (lerobot / "meta" / "info.json").exists() else 20.0
    time = states[:, 0] - states[0, 0]
    time_source = "recorded simulator time"
    if not np.all(np.diff(time) > 0):
        time, time_source = np.arange(T) / fps, "index / fps (recorded time not increasing)"

    jid = {n: j for j, n in enumerate(model.joint_names)}
    bid = {n: b for b, n in enumerate(model.body_names)}
    base_b = bid.get(BASE_BODY)
    torso_adr = m.jnt_qposadr[jid[TORSO_JOINT]] if TORSO_JOINT in jid else None
    cfgs = {o["name"]: o for o in meta.get("object_cfgs", [])}
    task_objs = [n for n in cfgs if not n.startswith("distr") and n in model.free]
    distractors = [n for n in cfgs if n.startswith("distr") and n in model.free]
    tracked = task_objs + distractors
    refs = {k: v for k, v in meta.get("fixture_refs", {}).items() if isinstance(v, str)}
    fixture_cls = {k: (v or {}).get("cls") for k, v in meta.get("fixtures", {}).items()}
    art_roots = {r: js for r, js in model.articulations.items()
                 if articulations == "all" or re.sub(r"_main$", "", r) in refs.values()}

    poses = {k: np.zeros((T, 4, 4)) for k in model.grippers}
    widths = {k: np.zeros(T) for k in model.grippers}
    obj_pose = {n: np.zeros((T, 7)) for n in tracked}
    art = {r: np.zeros((T, len(js))) for r, js in art_roots.items()}
    base = np.zeros((T, 3))
    torso = np.zeros(T)
    center = np.zeros((T, 3)), np.zeros((T, 3, 3))
    site_center = next((s for s in range(m.nsite) if model.mj.mj_id2name(m, model.mj.mjtObj.mjOBJ_SITE, s)
                        == "mobilebase0_center"), None)
    grip_site = {k: gr.site for k, gr in model.grippers.items()}
    grip_pos = {k: np.zeros((T, 3)) for k in model.grippers}
    free_body = {n: int(m.jnt_bodyid[j]) for n, j in model.free.items()}
    prefixes = (*model.robot_prefixes, *EXTRA_ROBOT_PREFIXES)
    robot_body = np.array([model.body_names[m.body_rootid[b]].startswith(prefixes) for b in range(m.nbody)])
    reference = (ObjectEnvironmentDepth(m, free_body, robot_body)
                 if with_scene and not model.missing else None)  # placeholder geoms: no contacts
    untracked = [n for n in model.free if n not in tracked]
    untracked_pos = np.zeros((T, len(untracked), 3))
    for t in range(T):
        model.set_state(states[t])
        if reference is not None:
            reference.update(d, t, time[t])
        for i, n in enumerate(untracked):
            untracked_pos[t, i] = d.xpos[free_body[n]]
        for k, gr in model.grippers.items():
            poses[k][t], widths[k][t] = gr.read(d)
            grip_pos[k][t] = d.site_xpos[grip_site[k]]
        for n in tracked:
            b = m.jnt_bodyid[model.free[n]]
            obj_pose[n][t] = np.r_[d.xpos[b], d.xquat[b]]
        for r, js in art_roots.items():
            art[r][t] = d.qpos[m.jnt_qposadr[js]]
        if base_b is not None:
            base[t] = [d.xpos[base_b][0], d.xpos[base_b][1], _yaw(d.xmat[base_b].reshape(3, 3))]
        if torso_adr is not None:
            torso[t] = d.qpos[torso_adr]
        if site_center is not None:
            center[0][t], center[1][t] = d.site_xpos[site_center], d.site_xmat[site_center].reshape(3, 3)
    base[:, 2] = np.unwrap(base[:, 2])
    # Untracked free bodies that stay farther than PARK_DISTANCE_M from the base path and every
    # tracked object sample are parked out of use (inactive, removed for validation).
    workspace = np.concatenate([np.c_[base[:, :2], np.zeros(T)] if base_b is not None else np.zeros((0, 3)),
                                *[obj_pose[n][:, :3] for n in tracked]])
    far = lambda i: np.linalg.norm(workspace[:, None] - untracked_pos[None, :, i], axis=-1).min() > PARK_DISTANCE_M
    inactive = sorted(n for i, n in enumerate(untracked) if len(workspace) and far(i))

    model.set_state(states[0])
    floor_z = _floor_top(model)
    if floor_z is not None and abs(floor_z) > FLOOR_TOL_M:
        raise ValueError(f"{ep}: kitchen floor top at z={floor_z:.4f} m; the source contract needs the floor at "
                         "z = 0 and this adapter does not translate scenes")
    effectors = {k: Effector(pose=poses[k], width=widths[k],
                             opening=np.clip(widths[k] / model.grippers[k].width_max, 0, 1), side_hint=None)
                 for k in model.grippers}
    objects = {}
    for n in task_objs:
        b = m.jnt_bodyid[model.free[n]]
        cfg = cfgs[n]
        role = "receptacle" if cfg.get("graspable") is False else "manipulated"
        objects[n] = ObjectTrack(pose=obj_pose[n], valid=np.ones(T, bool), role=role,
                                 geometry={**model.body_aabb(b), "body": model.body_names[b],
                                           "category": (cfg.get("info") or {}).get("cat"),
                                           "mjcf_path": (cfg.get("info") or {}).get("mjcf_path")})
    for ref_key, fx in refs.items():
        b = bid.get(f"{fx}_main", bid.get(fx))
        if b is None or fx in objects:
            continue
        pose = np.r_[d.xpos[b], d.xquat[b]]
        objects[fx] = ObjectTrack(pose=np.tile(pose, (T, 1)), valid=np.ones(T, bool),
                                  role=_fixture_role(fixture_cls.get(fx)),
                                  geometry={**model.body_aabb(b), "body": model.body_names[b], "fixture_ref": ref_key,
                                            "fixture_cls": fixture_cls.get(fx)})

    art_out = {re.sub(r"_main$", "", r): Articulation(joint_names=[model.joint_names[j] for j in js], qpos=art[r])
               for r, js in art_roots.items()}
    art_motion = {n: float(np.abs(a.qpos - a.qpos[0]).max()) for n, a in art_out.items()}
    task_art = sorted(n for n in art_out if n in refs.values())

    travel = float(np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1).sum()) if base_b is not None else 0.0
    disp = float(np.linalg.norm(base[:, :2] - base[0, :2], axis=1).max()) if base_b is not None else 0.0
    dyaw = float(np.abs(base[:, 2] - base[0, 2]).max()) if base_b is not None else 0.0
    moved = disp > BASE_MOVE_TOL_M or dyaw > BASE_MOVE_TOL_RAD
    obj_moved = {n: float(np.linalg.norm(obj_pose[n][:, :3] - obj_pose[n][0, :3], axis=1).max()) for n in tracked}
    manipulates = bool(task_objs) or any(art_motion[n] > ART_MOVE_TOL for n in task_art)
    regime = "tabletop" if not moved else ("mobile_manipulation" if manipulates else "navigation")

    initial_qpos = {model.joint_names[j]: d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 7].tolist()
                    for j in model.free.values()}
    initial_qpos.update({model.joint_names[j]: float(d.qpos[m.jnt_qposadr[j]])
                         for js in model.articulations.values() for j in js})
    route = "mjcf_kinematic_only" if model.missing else "mjcf_states"
    scene = None
    if with_scene and not model.missing:
        scene = SceneRef(mjcf=model.xml, robot_prefixes=sorted({*model.robot_prefixes, *EXTRA_ROBOT_PREFIXES}),
                         initial_qpos=initial_qpos, assets=model.assets,
                         inactive_bodies=[model.body_names[free_body[n]] for n in inactive],
                         reference=reference.result(inactive))

    success, success_source, obs_check = None, None, {}
    if pq is not None:
        rew = np.asarray(pq.get("next.reward", []), float)
        if len(rew):
            success, success_source = bool(rew.max() >= 1.0), "max parquet next.reward >= 1"
        obs_check = _check_observations(pq, T, center, grip_pos, model, states)

    lineage = {"generated": None}
    if tar is not None:
        lineage = {"generated": tar.source == "mg"}
        if tar.source == "mg":
            lineage.update(seed=tar.seed, seed_demo=meta.get("source_demo") or None, variant_group=tar.variant)
    dataset = tar.dataset if tar else f"{family}/{lerobot.parent.name}"
    provenance = {
        "dir": str(lerobot), "episode": ep, "files_sha256": files,
        "manifest": None if manifest is None else {"tar_id": manifest.get("tar_id"),
                                                   "box_sha1": manifest.get("box_sha1"),
                                                   "tar_sha1_verified": manifest.get("tar_sha1_verified")},
        "catalog_id": tar.id if tar else None, "url": tar.url if tar else None,
        "revision": {"registry_commit": tar.revision, "box_file_id": tar.box_file_id,
                     "box_file_version": tar.box_file_version, "box_sha1": tar.box_sha1} if tar else None,
        "split": tar.split if tar else None, "task_type": tar.task_type if tar else None,
        "source": tar.source if tar else None, "horizon": tar.horizon if tar else None,
        "env_name": dmeta.get("env"), "versions": versions, "robots": (dmeta.get("env_args", {})
                                                                       .get("env_kwargs", {}).get("robots")),
        "controller": _controller_summary(dmeta),
        "layout_id": meta.get("layout_id"), "style_id": meta.get("style_id"),
        "sim_timestep_s": float(m.opt.timestep), "time_source": time_source, "fps": fps,
        "state_route": route, "state_layout": "time, qpos, qvel (MuJoCo sim state, states.npz)",
        "asset_sources": rc_assets.usage() if rc_assets else [],
        "assets_from_subset": bool(rc_assets and any(rc_assets.sources[i].kind == "dir"
                                                     for i in rc_assets.used.values())),
        "robosuite_wheel": None if wheel is None else {"id": wheel[1].id, "sha256": wheel[1].sha256},
        **notes,
        "missing_assets": model.missing, "mesh_inertia_shell": model.shell_meshes,
        "effectors": {k: gr.describe(model) for k, gr in model.grippers.items()},
        "opening": "pad separation along +y minus fully-closed separation, / its open-closed range",
        "base": {"body": BASE_BODY, "joints": [j for j in BASE_JOINTS if j in jid], "moved": moved,
                 "travel_m": round(travel, 4), "max_displacement_m": round(disp, 4),
                 "max_yaw_change_rad": round(dyaw, 4), "initial_xy_yaw": base[0].round(6).tolist(),
                 "init_robot_base_pos": meta.get("init_robot_base_pos"),
                 "init_robot_base_ori": meta.get("init_robot_base_ori")},
        "torso": {"joint": TORSO_JOINT if torso_adr is not None else None,
                  "range_m": [float(torso.min()), float(torso.max())]},
        "world_frame": {"floor_top_z_m": floor_z, "z_offset_applied_m": 0.0,
                        "note": "RoboCasa world = source world; room floor top at z = 0 (checked per episode), "
                                "so no translation is applied" if floor_z is not None else
                                "no room floor found; RoboCasa kitchens put the floor at z = 0, no translation"},
        "regime_rule": "tabletop if the base never moves (>1 cm or >0.01 rad from frame 0); otherwise "
                       "mobile_manipulation when there are task objects or an operated referenced fixture, "
                       "else navigation",
        "fixture_refs": refs, "task_articulations": task_art,
        "articulation_motion": {n: round(v, 5) for n, v in art_motion.items() if v > ART_MOVE_TOL},
        "object_roles_rule": "object_cfgs: distr* = distractor (not tracked as task object), graspable false = "
                             "receptacle, else manipulated; fixture_refs: role from fixture class",
        "distractors": {n: {"pose0": obj_pose[n][0].round(6).tolist(), "max_displacement_m": round(obj_moved[n], 4)}
                        for n in distractors},
        "object_max_displacement_m": {n: round(obj_moved[n], 4) for n in task_objs},
        "inactive_free_bodies": inactive,
        "untracked_free_bodies_in_scene": sorted(set(untracked) - set(inactive)),
        "scene_reference": None if scene is None else scene.reference,
        "success_source": success_source, "observation_check": obs_check,
    }
    return SourceEpisode(
        family=family, dataset=dataset, episode_id=ep, task=dmeta.get("env") or (tar.task if tar else "unknown"),
        time=time, effectors=effectors, objects=objects,
        base=base if moved else None, base_hint=base[0].copy() if base_b is not None else None,
        torso_height=torso if torso_adr is not None else None, articulations=art_out, scene=scene,
        instruction=meta.get("lang"), success=success, regime=regime,
        license=tar.license if tar else LICENSE, provenance=provenance, lineage=lineage)


def _floor_top(model) -> float | None:
    """Highest top face of the room floor boxes (bodies ``floor*_room_main``, not the
    ``backing`` slabs below them). RoboCasa builds kitchens with this at z = 0."""
    m, d, mj = model.m, model.d, model.mj
    tops = [d.geom_xpos[g][2] + m.geom_size[g][2] for g in range(m.ngeom)
            if m.geom_type[g] == int(mj.mjtGeom.mjGEOM_BOX)
            and re.fullmatch(r"floor\w*_room_main", model.body_names[m.geom_bodyid[g]])
            and "backing" not in model.body_names[m.geom_bodyid[g]]]
    return float(max(tops)) if tops else None


def _controller_summary(dmeta: dict) -> dict | None:
    cc = dmeta.get("env_args", {}).get("env_kwargs", {}).get("controller_configs")
    if not cc:
        return None
    return {"type": cc.get("type"),
            "body_parts": {k: v.get("type") for k, v in cc.get("body_parts", {}).items()}}


def _check_observations(pq: dict, T: int, center, grip_pos, model, states) -> dict:
    """Compare the recorded LeRobot observations with the replayed kinematics.

    ``observation.state`` = base position (mobilebase0_center site), base quaternion,
    end-effector position relative to that site frame, its quaternion, gripper qpos.
    """
    obs = np.asarray(pq.get("observation.state", []), float)
    if obs.ndim != 2 or len(obs) != T or obs.shape[1] < 16 or not grip_pos:
        return {"compared": False, "reason": "observation.state missing or length differs from states"}
    k = next(iter(grip_pos))
    pos, R = center
    rel = np.einsum("tji,tj->ti", R, grip_pos[k] - pos)
    fingers = model.grippers[k].fingers
    adr = model.m.jnt_qposadr[fingers]
    gq = states[:, 1 + adr]
    return {"compared": True,
            "base_position_max_err_m": float(np.abs(obs[:, 0:3] - pos).max()),
            "eef_position_relative_max_err_m": float(np.abs(obs[:, 7:10] - rel).max()),
            "gripper_qpos_max_err": float(np.abs(obs[:, 14:16] - gq).max())}


__all__ = ["read_robocasa", "RobocasaAssets", "asset_sources_from_catalog", "locate_assets", "ASSET_MARKER"]
