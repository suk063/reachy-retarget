"""Tier-P scene: the source MuJoCo scene with its robot replaced by Reachy 2.

`build_scene(scene_ref)` consumes a `schema.source.SceneRef`:

* `mjcf`: complete MJCF of the source scene (expanded: no `<include>`/`<attach>`).
* `robot_prefixes`: name prefixes of every source-robot element, e.g. robosuite
  `["robot0_", "gripper0_", "mount0_"]`. Every body (with its subtree) whose name starts with
  one of them is removed, and so is every actuator, sensor, equality, tendon, contact
  pair/exclude and keyframe that references a removed element or a prefixed name. Mocap
  bodies (e.g. teleoperation/IK targets) are removed as well: nothing kinematically driven
  may remain in a physics validation scene. Objects, fixtures, tables and articulations stay.
* `initial_qpos`: `{source joint name: value}` (a float for hinge/slide, 7 values xyz + wxyz for
  free joints), written once at reset. Joints not listed keep the MJCF default (`qpos0`).
* `assets`: `{file attribute: bytes}` for every asset file the MJCF references. Keys are
  matched against each `file=` attribute: exactly first, then as a path suffix on a `/`
  boundary (so keys may be archive-relative paths such as `objects/meshes/can.stl` or bare
  file names while the MJCF still holds the recording machine's absolute paths; the longest
  matching key wins). The robosuite adapter rewrites `file=` to archive-relative names and
  fills `assets` with exactly those keys.

When `assets` does not have a file, `asset_resolver(file_attr) -> bytes | None` (if given) is
asked, then each directory of `meshdir` (a path or a list of paths) is searched for the path's
suffixes (`a/b/c.stl`, `b/c.stl`, `c.stl`). Only assets still referenced after the robot is
removed are resolved. A missing mesh used by a colliding geom raises `MissingSceneAssets`
(the scene cannot be validated faithfully); a missing texture is dropped with its material
references and a missing mesh of a purely visual geom drops that geom; both are recorded in
`Scene.info["missing_visual_assets"]`.

Reachy is attached with `robot.mjcf.attach_reachy` at the world origin at floor height
`floor_z`, so its base joints are world x, y, yaw (canonical `q[0:3]`); its names carry
`prefix`. Global options are set to `robot.mjcf.OPTIONS` (elliptic cones, impratio 10,
implicitfast, 2 ms); the source values are recorded in `Scene.info["source_option"]`.
The scene is audited: no gravity compensation, mocap, actuator, tendon or equality may act
on any non-Reachy body or joint ("no object assistance").
"""
from __future__ import annotations

import hashlib
import posixpath
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from ..robot import mjcf as rmjcf
from ..schema.source import SceneRef

ASSET_TAGS = ("mesh", "texture", "hfield", "skin")
_SECTIONS = ("actuator", "sensor", "equality", "tendon", "contact")
_FORBIDDEN_TAGS = ("include", "attach", "replicate", "frame", "composite", "flexcomp")


class MissingSceneAssets(FileNotFoundError):
    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__("missing collision assets: " + ", ".join(self.missing))


@dataclass
class Scene:
    """Compiled tier-P scene.

    `model` is what the rollout and the replay use. `xml` (`spec.to_xml()`) + `assets` recompile
    to an equivalent model for inspection; MuJoCo's XML writer rounds values (about 1e-7 m), so it
    is not bit-identical.
    """

    model: mujoco.MjModel
    spec: mujoco.MjSpec
    xml: str
    assets: dict
    prefix: str
    free_bodies: dict[str, str]          # free joint name -> body name (all non-Reachy free bodies)
    initial_qpos: dict                   # applied initial values per joint
    info: dict = field(default_factory=dict)

    def reachy_bodies(self) -> set[int]:
        root = self.model.body(f"{self.prefix}base_link").id
        return subtree(self.model, root)


def subtree(model, root: int) -> set[int]:
    out = {root}
    for b in range(root + 1, model.nbody):
        if int(model.body_parentid[b]) in out:
            out.add(b)
    return out


def _starts(value: str, prefixes) -> bool:
    return any(value.startswith(p) for p in prefixes)


def _suffixes(path: str):
    parts = [p for p in posixpath.normpath(path.replace("\\", "/")).split("/") if p not in ("", ".")]
    return ["/".join(parts[i:]) for i in range(len(parts))]


def _lookup(name: str, assets: dict, resolver, meshdirs) -> tuple[bytes | None, str]:
    if name in assets:
        return assets[name], "assets:exact"
    for suffix in _suffixes(name):
        if suffix in assets:
            return assets[suffix], f"assets:{suffix}"
    norm = name.replace("\\", "/")
    keys = sorted((k for k in assets if norm.endswith("/" + k.lstrip("/"))), key=len, reverse=True)
    if keys:
        return assets[keys[0]], f"assets:{keys[0]}"
    if resolver is not None:
        data = resolver(name)
        if data is not None:
            return bytes(data), "resolver"
    for d in meshdirs:
        for suffix in _suffixes(name):
            p = Path(d) / suffix
            if p.is_file():
                return p.read_bytes(), f"meshdir:{p}"
    return None, ""


def _strip_robot(root: ET.Element, prefixes: list[str]) -> dict:
    """Remove the source robot (and mocap bodies) in place; returns what was removed."""
    world = root.find("worldbody")
    removed_bodies, mocap_bodies, removed_names = [], [], set()

    def collect(el):
        for e in el.iter():
            if e.get("name"):
                removed_names.add(e.get("name"))

    def visit(parent):
        for child in list(parent):
            if child.tag != "body":
                if child.tag in ("geom", "site", "camera", "light") and _starts(child.get("name", ""), prefixes):
                    collect(child)
                    parent.remove(child)
                continue
            name = child.get("name", "")
            if _starts(name, prefixes) or child.get("mocap", "false") == "true":
                collect(child)
                (removed_bodies if _starts(name, prefixes) else mocap_bodies).append(name)
                parent.remove(child)
            else:
                visit(child)

    visit(world)

    def references_removed(el) -> bool:
        for e in el.iter():
            for k, v in e.attrib.items():
                if k == "name" and e is el and not _starts(v, prefixes):
                    continue
                if v in removed_names or _starts(v, prefixes):
                    return True
        return False

    removed_elements = {}
    for section in _SECTIONS:
        for node in root.findall(section):
            for el in list(node):
                if references_removed(el):
                    node.remove(el)
                    removed_elements.setdefault(section, []).append(el.get("name") or el.tag)
    for node in root.findall("keyframe"):  # qpos sizes change: keyframes no longer apply
        removed_elements.setdefault("keyframe", []).extend(k.get("name") or "key" for k in node)
        root.remove(node)
    return {"bodies": removed_bodies, "mocap_bodies": mocap_bodies, "elements": removed_elements}


def _prune_assets(root: ET.Element) -> list[str]:
    """Drop assets no longer referenced (robot meshes, materials, textures)."""
    asset = root.find("asset")
    if asset is None:
        return []
    body_parts = [e for sec in ("worldbody", "default") for n in root.findall(sec) for e in n.iter()]
    meshes = {e.get("mesh") for e in body_parts if e.get("mesh")}
    hfields = {e.get("hfield") for e in body_parts if e.get("hfield")}
    materials = {e.get("material") for e in body_parts if e.get("material")}
    for node in root.findall("tendon") + root.findall("deformable"):
        materials |= {e.get("material") for e in node.iter() if e.get("material")}
    textures = {m.get("texture") for m in asset.findall("material") if m.get("name") in materials and m.get("texture")}
    pruned = []
    for el in list(asset):
        name = el.get("name") or (Path(el.get("file", "")).stem if el.get("file") else None)
        keep = {"mesh": name in meshes, "hfield": name in hfields, "material": name in materials,
                "texture": name in textures or el.get("type") == "skybox"}.get(el.tag, True)
        if not keep:
            asset.remove(el)
            pruned.append(f"{el.tag}:{name}")
    return pruned


def _resolve_assets(root: ET.Element, assets: dict, resolver, meshdirs) -> tuple[dict, list, list]:
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
    vfs, records, missing_visual, missing = {}, [], [], []
    asset = root.find("asset")
    world_geoms = [g for n in root.findall("worldbody") for g in n.iter("geom")]
    for el in list(asset if asset is not None else []):
        f = el.get("file")
        if not f:
            continue
        data, how = _lookup(f, assets, resolver, meshdirs)
        name = el.get("name") or Path(f).stem
        if data is None:
            if el.tag == "mesh":
                users = [g for g in world_geoms if g.get("mesh") == name]
                if any(g.get("contype", "1") != "0" or g.get("conaffinity", "1") != "0" for g in users):
                    missing.append(f)
                    continue
                for g in users:  # purely visual geoms: drop them
                    for parent in root.iter():
                        if g in list(parent):
                            parent.remove(g)
            elif el.tag == "texture":
                for m in asset.findall("material"):
                    if m.get("texture") == name:
                        m.attrib.pop("texture")
            else:
                missing.append(f)
                continue
            asset.remove(el)
            missing_visual.append({"tag": el.tag, "name": name, "file": f})
            continue
        key = f"scene_assets/{len(vfs):04d}_{posixpath.basename(f.replace(chr(92), '/'))}"
        if not el.get("name") and el.tag in ("mesh", "hfield"):
            el.set("name", name)  # the default name derived from the file must survive renaming
        el.set("file", key)
        vfs[key] = data
        records.append({"file": f, "as": key, "found": how, "sha256": hashlib.sha256(data).hexdigest()})
    if missing:
        raise MissingSceneAssets(missing)
    return vfs, records, missing_visual


def _audit_no_assistance(model, reachy: set[int]) -> None:
    """Nothing but Reachy's own servos and mimic couplings may act in the scene."""
    for b in range(1, model.nbody):
        if b in reachy:
            continue
        if model.body_gravcomp[b] != 0 or model.body_mocapid[b] != -1:
            raise ValueError(f"scene body {model.body(b).name!r} has gravity compensation or mocap")
    for a in range(model.nu):
        trn = int(model.actuator_trntype[a])
        if trn != int(mujoco.mjtTrn.mjTRN_JOINT) or int(model.jnt_bodyid[model.actuator_trnid[a, 0]]) not in reachy:
            raise ValueError(f"actuator {model.actuator(a).name!r} does not drive a Reachy joint")
    for e in range(model.neq):
        if int(model.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT):
            raise ValueError(f"equality {model.equality(e).name!r} is not a Reachy joint coupling")
        for j in (model.eq_obj1id[e], model.eq_obj2id[e]):
            if j >= 0 and int(model.jnt_bodyid[j]) not in reachy:
                raise ValueError(f"equality {model.equality(e).name!r} couples a non-Reachy joint")
    if model.ntendon:
        raise ValueError("tendons in the scene are not audited; refusing")


def build_scene(scene_ref: SceneRef, *, prefix: str = "reachy/", floor_z: float = 0.0,
                asset_resolver: Callable[[str], bytes | None] | None = None,
                meshdir: str | Path | list | None = None, options: dict | None = None) -> Scene:
    """Compile the source scene with its robot replaced by Reachy (see module docstring)."""
    root = ET.fromstring(scene_ref.mjcf)
    if root.tag != "mujoco" or root.find("worldbody") is None:
        raise ValueError("SceneRef.mjcf must be a complete MJCF document")
    for tag in _FORBIDDEN_TAGS:
        if root.find(f".//{tag}") is not None:
            raise ValueError(f"unexpanded MJCF element <{tag}> is not supported")
    source_sha = hashlib.sha256(scene_ref.mjcf.encode()).hexdigest()
    prefixes = [p for p in scene_ref.robot_prefixes if p]
    if not prefixes:
        raise ValueError("SceneRef.robot_prefixes is empty: the source robot cannot be identified")
    if any(_starts(prefix, [p]) or _starts(p, [prefix]) for p in prefixes):
        raise ValueError(f"Reachy prefix {prefix!r} collides with source robot prefixes {prefixes}")
    removed = _strip_robot(root, prefixes)
    pruned = _prune_assets(root)
    meshdirs = [] if meshdir is None else [meshdir] if isinstance(meshdir, (str, Path)) else list(meshdir)
    vfs, asset_records, missing_visual = _resolve_assets(root, dict(scene_ref.assets or {}), asset_resolver, meshdirs)
    stripped = ET.tostring(root, encoding="unicode")
    spec = mujoco.MjSpec.from_string(stripped, assets=vfs)
    source_option = {"timestep": float(spec.option.timestep), "integrator": int(spec.option.integrator),
                     "cone": int(spec.option.cone), "impratio": float(spec.option.impratio),
                     "iterations": int(spec.option.iterations)}
    o = {**rmjcf.OPTIONS, **(options or {})}
    spec.option.timestep = o["timestep"]
    spec.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{o['integrator'].upper()}")
    spec.option.cone = getattr(mujoco.mjtCone, f"mjCONE_{o['cone'].upper()}")
    spec.option.impratio = o["impratio"]
    spec.option.iterations = o["iterations"]
    spec.option.gravity = list(o["gravity"])
    rmjcf.attach_reachy(spec, prefix=prefix, pos=(0.0, 0.0, floor_z))
    model = spec.compile()
    reachy = subtree(model, model.body(f"{prefix}base_link").id)
    _audit_no_assistance(model, reachy)

    free_bodies = {model.joint(j).name: model.body(model.jnt_bodyid[j]).name for j in range(model.njnt)
                   if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_bodyid[j] not in reachy}
    initial = {}
    for name, value in (scene_ref.initial_qpos or {}).items():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            raise ValueError(f"initial_qpos joint {name!r} is not in the scene")
        j = model.joint(name)
        if int(j.bodyid[0]) in reachy:
            raise ValueError(f"initial_qpos may not set Reachy joint {name!r}")
        v = np.atleast_1d(np.asarray(value, float))
        width = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(int(j.type[0]), 1)
        if v.shape != (width,) or not np.isfinite(v).all():
            raise ValueError(f"initial_qpos[{name!r}] must hold {width} finite values")
        if width == 7:
            v[3:] /= np.linalg.norm(v[3:])
        initial[name] = v
    xml = spec.to_xml()
    info = {"source_mjcf_sha256": source_sha, "robot_prefixes": prefixes, "removed": removed,
            "pruned_assets": pruned, "assets": asset_records, "missing_visual_assets": missing_visual,
            "source_option": source_option, "option": {k: (list(v) if isinstance(v, tuple) else v)
                                                       for k, v in o.items()},
            "reachy_prefix": prefix, "reachy_frame": {"pos": [0.0, 0.0, floor_z], "quat": [1.0, 0.0, 0.0, 0.0]},
            "joints_without_initial_qpos": sorted(
                model.joint(j).name for j in range(model.njnt)
                if model.jnt_bodyid[j] not in reachy and model.joint(j).name not in initial),
            "scene_sha256": hashlib.sha256(xml.encode()).hexdigest()}
    return Scene(model=model, spec=spec, xml=xml, assets=vfs, prefix=prefix, free_bodies=free_bodies,
                 initial_qpos=initial, info=info)


def reset(scene: Scene, q0, data: mujoco.MjData | None = None) -> mujoco.MjData:
    """Initial state: MJCF defaults, source initial object qpos, Reachy at q0 with ctrl = q0."""
    m = scene.model
    d = data if data is not None else mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for name, v in scene.initial_qpos.items():
        adr = m.joint(name).qposadr[0]
        d.qpos[adr:adr + len(v)] = v
    rmjcf.set_reachy_state(m, d, q0, scene.prefix)
    mujoco.mj_forward(m, d)
    return d

