"""Scene components of a source MuJoCo scene (``schema.source.SceneRef``) with their meshes.

The scene is prepared exactly as for tier P (``validate.scene``): the source robot, mocap bodies
and the bodies the adapter declares inactive (``SceneRef.inactive_bodies``) are removed, unused
assets are pruned and asset files are resolved from ``SceneRef.assets``. It is then compiled
with MuJoCo **without its textures** (they are visual only and dominate compile memory; their
files are taken from the asset bytes and their parameters from the MJCF), and everything is read
from the compiled model:

* **Components** are rigid body groups: a body that carries a joint, or is a child of the world
  body, together with all its descendants that have no joint of their own. Geoms of the world
  body itself form the component ``worldbody``. A group is ``free`` (a free joint),
  ``articulated`` (hinge/slide joints) or ``static``.
* **Parts**: per geom of a component, its MuJoCo type and ``size``, its pose in the component
  frame (``pos``, ``quat`` wxyz; nested bodies composed in), group, contact bits, geom ``rgba``
  and material. *Visual* parts are the visible (effective alpha > 0) geoms that do not collide,
  plus visible colliding geoms in one of the scene's visual render groups (the groups of its
  visible non-colliding geoms: robosuite/LIBERO/MimicGen/RoboCasa 1, BiGym 2; e.g. the robosuite
  floor plane, which is both); a component without any (BiGym floor, primitive scenes) uses its
  visible colliding geoms and says so (``visual_from_collision``). *Collision* parts are the
  colliding geoms.
* **Meshes** are the compiled MuJoCo meshes, so ``scale``, ``refpos``/``refquat`` and MuJoCo's
  re-centring are applied exactly: vertices are in MuJoCo's mesh frame and the part pose is the
  compiled geom pose. Each is written as ``.msh`` with its compiled normals and texture
  coordinates (vertices split per distinct normal/texcoord corner). ``mesh_source`` records the
  source file, its SHA-256, ``scale`` and the compile transform (``mesh_pos``, ``mesh_quat``:
  ``stored = R(mesh_quat)^T (scale * file_vertex - mesh_pos)``). Collision meshes are stored the
  same way; MuJoCo collides with their convex hull (``collision_shape``).
* **Visual primitives** additionally get a generated surface mesh (``generated_mesh``,
  :func:`reachy_retarget.schema.scene_assets.primitive_mesh`); collision primitives keep type
  and size only.
* **Materials** (compiled rgba, specular, shininess, reflectance, emission, metallic, roughness,
  texrepeat, texuniform) and **textures** (every MJCF attribute except the name; ``file*``
  attributes become library references; builtin textures keep their parameters).

Component poses for a sequence of scene-joint positions (columns named like the tier-P rollout:
``<joint>`` or ``<joint>/x .. /qz``) come from MuJoCo forward kinematics (:meth:`body_poses`).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np

from ..schema.scene_assets import encode_msh, primitive_mesh
from ..schema.source import SceneRef

GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}
DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0])
WORLD = "worldbody"


def qpos_columns(model, mujoco, keep=None) -> tuple[list[str], np.ndarray]:
    """Rollout-style qpos column names and qpos addresses of the joints whose body passes ``keep``
    (a callable on the body id; default: all joints)."""
    names, adr = [], []
    for j in range(model.njnt):
        b = int(model.jnt_bodyid[j])
        if keep is not None and not keep(b):
            continue
        n, t, a = model.joint(j).name, int(model.jnt_type[j]), int(model.jnt_qposadr[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE):
            names += [f"{n}/{c}" for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
            adr += list(range(a, a + 7))
        elif t == int(mujoco.mjtJoint.mjJNT_BALL):
            names += [f"{n}/{c}" for c in ("qw", "qx", "qy", "qz")]
            adr += list(range(a, a + 4))
        else:
            names.append(n)
            adr.append(a)
    return names, np.array(adr, int)


def _compile_tolerant(mujoco, xml: str, assets: dict):
    """Compile; meshes rejected as too thin get ``inertia="shell"`` (as the source adapters do)."""
    shell = []
    while True:
        try:
            return mujoco.MjModel.from_xml_string(xml, assets), shell
        except ValueError as err:
            mt = re.search(r"mesh volume is too small: (\S+?) \.", str(err))
            if not mt or mt.group(1) in shell or len(shell) > 200:
                raise
            root = ET.fromstring(xml)
            next(el for el in root.iter("mesh") if el.get("name") == mt.group(1)).set("inertia", "shell")
            shell.append(mt.group(1))
            xml = ET.tostring(root, encoding="unicode")


def _quat(R) -> list[float]:
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return (q if q[0] >= 0 else -q).tolist()


def split_mesh(m, i: int) -> dict:
    """Compiled mesh ``i`` as per-vertex arrays (vertices split per distinct normal/texcoord corner)."""
    va, nv = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
    fa, nf = int(m.mesh_faceadr[i]), int(m.mesh_facenum[i])
    V = np.asarray(m.mesh_vert[va:va + nv], np.float32)
    F = np.asarray(m.mesh_face[fa:fa + nf], np.int64)
    cols = [F.reshape(-1)]
    N = T = None
    if int(m.mesh_normalnum[i]) > 0:
        na = int(m.mesh_normaladr[i])
        N = np.asarray(m.mesh_normal[na:na + int(m.mesh_normalnum[i])], np.float32)
        cols.append(np.asarray(m.mesh_facenormal[fa:fa + nf], np.int64).reshape(-1))
    if int(m.mesh_texcoordnum[i]) > 0 and int(m.mesh_texcoordadr[i]) >= 0:
        ta = int(m.mesh_texcoordadr[i])
        T = np.asarray(m.mesh_texcoord[ta:ta + int(m.mesh_texcoordnum[i])], np.float32)
        cols.append(np.asarray(m.mesh_facetexcoord[fa:fa + nf], np.int64).reshape(-1))
    keys = np.stack(cols, 1)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    out = {"vertices": V[uniq[:, 0]], "faces": inverse.reshape(-1, 3), "normals": None, "uv": None}
    k = 1
    if N is not None:
        out["normals"] = N[uniq[:, k]]
        k += 1
    if T is not None:
        out["uv"] = T[uniq[:, k]]
    return out


class SceneExtraction:
    """The robot-free compiled scene of a ``SceneRef`` and its components (see the module docstring)."""

    def __init__(self, scene_ref: SceneRef, *, drop_bodies=None):
        import mujoco

        from ..validate import scene as vs

        self.mj = mujoco
        root = ET.fromstring(scene_ref.mjcf)
        drop = set(scene_ref.inactive_bodies or []) if drop_bodies is None else set(drop_bodies)
        self.removed = vs._strip_robot(root, [p for p in scene_ref.robot_prefixes if p], drop)
        vs._prune_assets(root)
        # raises validate.scene.MissingSceneAssets when a colliding mesh is missing
        vfs, records, self.missing = vs._resolve_assets(root, dict(scene_ref.assets or {}), None, [])
        self.asset_records = {r["as"]: r for r in records}
        self.vfs = vfs
        asset = root.find("asset")
        self.texture_el, self.material_tex = {}, {}
        for el in list(asset if asset is not None else []):
            if el.tag == "texture":
                f = el.get("file") or next((v for k, v in el.attrib.items() if k.startswith("file")), None)
                name = el.get("name") or (f.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("_", 1)[-1] if f else None)
                if name and el.get("type") != "skybox":
                    self.texture_el[name] = dict(el.attrib)
            elif el.tag == "material" and el.get("name"):
                layers = {l.get("role", "rgb"): l.get("texture") for l in el.findall("layer") if l.get("texture")}
                if el.get("texture"):
                    layers.setdefault("rgb", el.get("texture"))
                self.material_tex[el.get("name")] = layers
        self.mesh_el = {el.get("name"): dict(el.attrib) for el in (asset if asset is not None else [])
                        if el.tag == "mesh" and el.get("name")}
        # Compile without textures: kinematics and geometry never depend on them.
        for el in [e for e in (asset if asset is not None else []) if e.tag == "texture"]:
            asset.remove(el)
        for mat in root.iter("material"):
            mat.attrib.pop("texture", None)
            for layer in mat.findall("layer"):
                mat.remove(layer)
        for sky in root.iter("skybox"):
            sky.attrib.pop("texture", None)
        used = {e.get("file") for e in root.iter() if e.get("file")}
        self.m, self.shell = _compile_tolerant(mujoco, ET.tostring(root, encoding="unicode"),
                                               {k: v for k, v in vfs.items() if k in used})
        self.d = mujoco.MjData(self.m)
        self.initial_qpos = dict(scene_ref.initial_qpos or {})
        self.reset()
        self._groups()

    # ---------------------------------------------------------------- state

    def reset(self):
        """MJCF defaults plus the source initial qpos, forward kinematics."""
        m, d, mj = self.m, self.d, self.mj
        mj.mj_resetData(m, d)
        for name, value in self.initial_qpos.items():
            j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                continue  # a joint of a removed (inactive) body
            v = np.atleast_1d(np.asarray(value, float))
            a = int(m.jnt_qposadr[j])
            d.qpos[a:a + len(v)] = v
        mj.mj_kinematics(m, d)
        self.qpos0 = d.qpos.copy()

    def _groups(self):
        m = self.m
        self.body_names = [m.body(b).name for b in range(m.nbody)]
        root = np.zeros(m.nbody, int)
        for b in range(1, m.nbody):
            r = b
            while m.body_jntnum[r] == 0 and m.body_parentid[r] != 0:
                r = int(m.body_parentid[r])
            root[b] = r
        self.group_of = root
        self.groups: dict[int, list[int]] = {}
        for b in range(m.nbody):
            self.groups.setdefault(int(root[b]), []).append(b)
        free = int(self.mj.mjtJoint.mjJNT_FREE)
        self.kind = {}
        for r in self.groups:
            js = range(int(m.body_jntadr[r]), int(m.body_jntadr[r]) + int(m.body_jntnum[r])) if r else []
            types = {int(m.jnt_type[j]) for j in js}
            self.kind[r] = "static" if not types else "free" if free in types else "articulated"
        self.joint_names = {j: m.joint(j).name for j in range(m.njnt)}
        # render groups of the scene's visual geoms (visible, non-colliding): robosuite family and
        # RoboCasa 1, BiGym 2; a colliding visible geom in such a group (robosuite floor) is visual too
        self.visual_groups = {int(m.geom_group[g]) for g in range(m.ngeom)
                              if not (m.geom_contype[g] or m.geom_conaffinity[g]) and self.effective_rgba(g)[3] > 0}

    def group_joints(self, r: int) -> list[str]:
        m = self.m
        return [self.joint_names[j] for j in range(int(m.body_jntadr[r]), int(m.body_jntadr[r]) + int(m.body_jntnum[r]))]

    def group_name(self, r: int) -> str:
        return WORLD if r == 0 else self.body_names[r]

    def geoms_of(self, r: int) -> np.ndarray:
        m = self.m
        if r == 0:
            return np.flatnonzero(m.geom_bodyid == 0)
        return np.flatnonzero(self.group_of[m.geom_bodyid] == r) if r else np.zeros(0, int)

    def effective_rgba(self, g: int) -> np.ndarray:
        m = self.m
        rgba = np.asarray(m.geom_rgba[g], float)
        k = int(m.geom_matid[g])
        if k >= 0 and np.allclose(rgba, DEFAULT_RGBA):
            return np.asarray(m.mat_rgba[k], float)
        return rgba

    def geom_world_aabb(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        m, d = self.m, self.d
        if int(m.geom_type[g]) == 0:  # plane: unbounded in its own x/y when size is 0
            z = d.geom_xpos[g][2]
            return np.array([-np.inf, -np.inf, z]), np.array([np.inf, np.inf, z])
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        corners = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        w = d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T
        return w.min(0), w.max(0)

    def distance(self, geoms, points) -> float:
        """Smallest distance between the world AABBs (initial state) of ``geoms`` and ``points`` (n, 3)."""
        best = np.inf
        for g in geoms:
            lo, hi = self.geom_world_aabb(int(g))
            gap = np.maximum(np.maximum(lo - points, points - hi), 0.0)
            best = min(best, float(np.linalg.norm(gap, axis=1).min()))
        return best

    # ---------------------------------------------------------------- parts

    def parts(self, r: int, geoms, library, cache: dict) -> tuple[list, list, dict]:
        """(visual parts, collision parts, notes) of the geoms ``geoms`` of group ``r``."""
        m, d = self.m, self.d
        Xr = np.eye(4)
        if r:
            Xr[:3, :3], Xr[:3, 3] = d.xmat[r].reshape(3, 3), d.xpos[r]
        Xr_inv = np.linalg.inv(Xr)
        visual, collision, notes = [], [], {"unsupported_geoms": []}
        candidates = []
        for g in geoms:
            g = int(g)
            kind = GEOM_TYPES.get(int(m.geom_type[g]))
            if kind is None:
                notes["unsupported_geoms"].append({"geom": m.geom(g).name, "type": int(m.geom_type[g])})
                continue
            Xg = np.eye(4)
            Xg[:3, :3], Xg[:3, 3] = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            L = Xr_inv @ Xg
            collides = bool(m.geom_contype[g] or m.geom_conaffinity[g])
            k = int(m.geom_matid[g])
            part = {"geom": m.geom(g).name or f"geom{g}", "type": kind,
                    "size": np.asarray(m.geom_size[g], float).round(9).tolist(),
                    "pos": L[:3, 3].round(9).tolist(), "quat": np.round(_quat(L[:3, :3]), 9).tolist(),
                    "group": int(m.geom_group[g]), "contype": int(m.geom_contype[g]),
                    "conaffinity": int(m.geom_conaffinity[g]),
                    "rgba": np.asarray(m.geom_rgba[g], float).round(6).tolist(),
                    "material": m.material(k).name if k >= 0 else None}
            if kind == "mesh":
                part["mesh"], part["mesh_source"] = self.mesh_ref(int(m.geom_dataid[g]), library, cache)
            candidates.append((g, collides, self.effective_rgba(g)[3] > 0, part))
        vis = [p for g, c, v, p in candidates if v and (not c or int(m.geom_group[g]) in self.visual_groups)]
        if not vis:
            vis = [dict(p) for g, c, v, p in candidates if v and c]
            notes["visual_from_collision"] = bool(vis)
        for p in vis:
            p = dict(p)
            if p["type"] != "mesh":
                p["generated_mesh"] = self.primitive_ref(p["type"], p["size"], library, cache)
            visual.append(p)
        for g, c, v, p in candidates:
            if c:
                p = dict(p)
                if p["type"] == "mesh":
                    p["collision_shape"] = "convex hull of the mesh (MuJoCo)"
                collision.append(p)
        return visual, collision, notes

    def mesh_ref(self, i: int, library, cache: dict) -> tuple[dict, dict]:
        key = ("mesh", i)
        if key not in cache:
            a = split_mesh(self.m, i)
            data = encode_msh(a["vertices"], a["faces"], a["normals"], a["uv"])
            ref = {**library.put(data, "msh"), "vertices": len(a["vertices"]), "faces": len(a["faces"]),
                   "normals": a["normals"] is not None, "uv": a["uv"] is not None}
            name = self.m.mesh(i).name
            el = self.mesh_el.get(name, {})
            rec = self.asset_records.get(el.get("file"), {})
            src = {"name": name, "file": rec.get("file"), "sha256": rec.get("sha256"),
                   "scale": np.asarray(self.m.mesh_scale[i], float).round(9).tolist(),
                   "mesh_pos": np.asarray(self.m.mesh_pos[i], float).round(9).tolist(),
                   "mesh_quat": np.asarray(self.m.mesh_quat[i], float).round(9).tolist(),
                   **({k: el[k] for k in ("refpos", "refquat") if k in el})}
            if el.get("vertex") is not None:
                src["inline_vertices"] = True
            cache[key] = (ref, src)
        return cache[key]

    @staticmethod
    def primitive_ref(kind: str, size, library, cache: dict) -> dict:
        key = ("primitive", kind, tuple(np.round(size, 9)))
        if key not in cache:
            v, n, f = primitive_mesh(kind, size)
            ref = library.put(encode_msh(v, f, n), "msh")
            cache[key] = {**ref, "vertices": len(v), "faces": len(f), "normals": True, "uv": False}
        return cache[key]

    def material(self, name: str) -> dict:
        m = self.m
        k = self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_MATERIAL, name)
        layers = {role: t for role, t in self.material_tex.get(name, {}).items() if t in self.texture_el}
        out = {"rgba": np.asarray(m.mat_rgba[k], float).round(6).tolist(),
               "specular": float(m.mat_specular[k]), "shininess": float(m.mat_shininess[k]),
               "reflectance": float(m.mat_reflectance[k]), "emission": float(m.mat_emission[k]),
               "metallic": float(m.mat_metallic[k]), "roughness": float(m.mat_roughness[k]),
               "texrepeat": np.asarray(m.mat_texrepeat[k], float).round(9).tolist(),
               "texuniform": bool(m.mat_texuniform[k]), "texture": layers.get("rgb")}
        if set(layers) - {"rgb"}:
            out["layers"] = layers
        return out

    def texture(self, name: str, library) -> dict:
        attrs = dict(self.texture_el[name])
        attrs.pop("name", None)
        files = {}
        for key in [k for k in attrs if k.startswith("file")]:
            ref = attrs.pop(key)
            data = self.vfs.get(ref)
            if data is None:
                raise FileNotFoundError(f"texture {name}: {key} {ref} not resolved")
            rec = self.asset_records.get(ref, {})
            files[key] = {**library.put(data, None, ref), "source_file": rec.get("file")}
        attrs.pop("content_type", None)
        return {"attributes": attrs, "files": files, "file": files.get("file"),
                "builtin": attrs.get("builtin")}

    # ---------------------------------------------------------------- poses

    def body_poses(self, columns, rows, bodies) -> np.ndarray:
        """World poses (N, len(bodies), 7) of ``bodies`` for qpos ``rows`` (N, len(columns)).

        Columns not in this scene are ignored; joints without a column keep the initial state.
        Free-joint quaternions are normalized; rows with non-finite values keep the initial value."""
        m, d, mj = self.m, self.d, self.mj
        names, adr = qpos_columns(m, mj)
        where = dict(zip(names, adr))
        pairs = [(i, where[c]) for i, c in enumerate(columns) if c in where]
        src = np.array([i for i, _ in pairs], int)
        dst = np.array([a for _, a in pairs], int)
        quats = [int(m.jnt_qposadr[j]) + 3 for j in range(m.njnt) if int(m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_FREE)]
        rows = np.asarray(rows, float)
        bodies = np.asarray(bodies, int)
        out = np.empty((len(rows), len(bodies), 7))
        for t, row in enumerate(rows):
            d.qpos[:] = self.qpos0
            vals = row[src]
            ok = np.isfinite(vals)
            d.qpos[dst[ok]] = vals[ok]
            for a in quats:
                n = np.linalg.norm(d.qpos[a:a + 4])
                d.qpos[a:a + 4] = d.qpos[a:a + 4] / n if n > 0 else [1, 0, 0, 0]
            mj.mj_kinematics(m, d)
            out[t, :, :3] = d.xpos[bodies]
            out[t, :, 3:] = d.xquat[bodies]
        out[..., 3:] *= np.where(out[..., 3:4] < 0, -1.0, 1.0)
        self.reset()
        return out
