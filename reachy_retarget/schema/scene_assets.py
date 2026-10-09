"""Reader of the episode ``/scene`` section and its asset library (no MuJoCo needed for arrays).

An episode written by ``reachy_retarget.build`` lists its scene *components* (rigid bodies:
task objects, supports, receptacles, fixtures, articulated parts and Reachy links), their
per-frame world poses and, per component, visual and collision *parts* that reference mesh and
texture files of the dataset's asset library ``<out>/assets/<sha256>.<ext>`` (see docs/schema.md).

* :func:`read_scene` -> :class:`~reachy_retarget.schema.episode.SceneComponents` and the library path.
* :func:`load_component_meshes` -> ``{component: [part, ...]}`` with trimesh-compatible arrays
  (``vertices`` (n, 3) float64 in the component frame, ``faces`` (m, 3) int64, optional
  ``normals``/``uv``), the effective colour, material and texture file of every part.
* :func:`scene_mjcf` -> a MuJoCo MJCF string of the components (static bodies at the poses of one
  frame, absolute asset paths) that compiles and renders with ``mujoco.MjModel.from_xml_string``.

Meshes are MuJoCo binary ``.msh`` files: int32 ``nvertex, nnormal, ntexcoord, nface``, then
float32 vertices (nvertex, 3), normals (nnormal, 3), texture coordinates (ntexcoord, 2, MuJoCo
convention: used as stored, no v flip) and int32 faces (nface, 3); ``nnormal`` and ``ntexcoord``
are 0 or ``nvertex``. Primitive parts (box, sphere, capsule, cylinder, ellipsoid, plane) keep
their MuJoCo type and size; :func:`primitive_mesh` generates their surface (also stored in the
library for visual primitives).
"""
from __future__ import annotations

import gzip
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .rotations import quat_to_matrix

SCENE_SCHEMA = "reachy-retarget-scene-v1"
ASSET_DIR = "assets"
MESH_FORMAT = "msh"
# Size-0 MuJoCo planes are infinite; their generated surface is a square of this half size (m).
INFINITE_PLANE_HALF = 10.0
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)  # MuJoCo's internal geom rgba default
_MAGIC = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpg"), (b"\xabKTX", "ktx"), (b"glTF", "glb"))


def asset_format(data: bytes, name: str = "") -> str:
    """File format (extension) of an asset: magic bytes first, then the name's extension."""
    for magic, fmt in _MAGIC:
        if data[:len(magic)] == magic:
            return fmt
    ext = Path(name).suffix.lower().lstrip(".")
    if not ext or not ext.isalnum():
        raise ValueError(f"cannot determine the format of asset {name!r}")
    return "jpg" if ext == "jpeg" else ext


# ------------------------------------------------------------------------------- msh

def encode_msh(vertices, faces, normals=None, uv=None) -> bytes:
    """MuJoCo binary mesh bytes (see the module docstring)."""
    v = np.ascontiguousarray(vertices, np.float32).reshape(-1, 3)
    f = np.ascontiguousarray(faces, np.int32).reshape(-1, 3)
    n = np.zeros((0, 3), np.float32) if normals is None else np.ascontiguousarray(normals, np.float32).reshape(-1, 3)
    t = np.zeros((0, 2), np.float32) if uv is None else np.ascontiguousarray(uv, np.float32).reshape(-1, 2)
    if len(n) not in (0, len(v)) or len(t) not in (0, len(v)):
        raise ValueError("msh normals and texture coordinates must be per vertex")
    if len(f) and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError("msh face index out of range")
    return struct.pack("<4i", len(v), len(n), len(t), len(f)) + v.tobytes() + n.tobytes() + t.tobytes() + f.tobytes()


def decode_msh(data: bytes) -> dict:
    """``{"vertices", "faces", "normals" | None, "uv" | None}`` of MuJoCo ``.msh`` bytes."""
    nv, nn, nt, nf = struct.unpack_from("<4i", data)
    off = 16
    out = {}
    for key, count, width, dtype in (("vertices", nv, 3, np.float32), ("normals", nn, 3, np.float32),
                                      ("uv", nt, 2, np.float32), ("faces", nf, 3, np.int32)):
        size = count * width * 4
        out[key] = np.frombuffer(data, dtype, count * width, off).reshape(count, width) if count else None
        off += size
    if off != len(data):
        raise ValueError(f"msh size mismatch: header implies {off} bytes, got {len(data)}")
    out["vertices"] = out["vertices"].astype(np.float64)
    out["faces"] = np.zeros((0, 3), np.int64) if out["faces"] is None else out["faces"].astype(np.int64)
    return out


# ------------------------------------------------------------------------------- primitives

def _lathe(profile, n: int = 32):
    """Surface of revolution about z. ``profile``: rings ``(radius, z, normal_r, normal_z, joined)``
    from top to bottom; ``joined`` = connect to the previous ring (False starts a new strip, so
    rings at the same place with different normals give sharp edges). Radius 0 = a pole vertex."""
    ph = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    c, s = np.cos(ph), np.sin(ph)
    v, nrm, f, prev = [], [], [], None
    for r, z, nr, nz, joined in profile:
        start = len(v)
        if r == 0:
            v.append([0.0, 0.0, z])
            nrm.append([0.0, 0.0, np.sign(nz) or 1.0])
            ring = [start] * n
        else:
            v += [[r * c[j], r * s[j], z] for j in range(n)]
            nrm += [[nr * c[j], nr * s[j], nz] for j in range(n)]
            ring = list(range(start, start + n))
        if joined and prev is not None:
            for j in range(n):
                j2 = (j + 1) % n
                a, b, a2, b2 = prev[j], prev[j2], ring[j], ring[j2]
                if a != b:
                    f.append([a, a2, b])
                if a2 != b2:
                    f.append([b, a2, b2])
        prev = ring
    return np.array(v, float), np.array(nrm, float), np.array(f, int)


def primitive_mesh(kind: str, size) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(vertices, normals, faces)`` of a MuJoCo primitive geom of ``size`` (MuJoCo semantics:
    box half extents, sphere radius, capsule/cylinder radius and half length along z, ellipsoid
    radii, plane half extents in x/y with 0 = infinite, see :data:`INFINITE_PLANE_HALF`)."""
    s = np.asarray(size, float)
    n = 32
    if kind == "box":
        h = s[:3]
        v, nrm, f = [], [], []
        for axis in range(3):
            u, w = [a for a in range(3) if a != axis]
            for sign in (1.0, -1.0):
                base = len(v)
                for cu, cw in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    p = np.zeros(3)
                    p[axis], p[u], p[w] = sign * h[axis], cu * h[u], cw * h[w]
                    v.append(p)
                    nrm.append(np.eye(3)[axis] * sign)
                # (u, w, axis) is a cyclic permutation of (x, y, z): counter-clockwise about +axis
                tri = [[0, 1, 2], [0, 2, 3]] if sign > 0 else [[0, 2, 1], [0, 3, 2]]
                f += [[base + a for a in t] for t in tri]
        return np.array(v), np.array(nrm), np.array(f)
    if kind in ("sphere", "ellipsoid"):
        th = np.linspace(0, np.pi, 17)
        v, nrm, f = _lathe([(np.sin(t), np.cos(t), np.sin(t), np.cos(t), True) for t in th], n)
        r = np.full(3, s[0]) if kind == "sphere" else s[:3]
        nrm = nrm / r
        return v * r, nrm / np.linalg.norm(nrm, axis=1, keepdims=True), f
    if kind == "cylinder":
        r, h = s[0], s[1]
        return _lathe([(0, h, 0, 1, False), (r, h, 0, 1, True), (r, h, 1, 0, False), (r, -h, 1, 0, True),
                       (r, -h, 0, -1, False), (0, -h, 0, -1, True)], n)
    if kind == "capsule":
        r, h = s[0], s[1]
        top = [(r * np.sin(t), h + r * np.cos(t), np.sin(t), np.cos(t), True) for t in np.linspace(0, np.pi / 2, 9)]
        bottom = [(r * np.sin(t), -h + r * np.cos(t), np.sin(t), np.cos(t), True)
                  for t in np.linspace(np.pi / 2, np.pi, 9)]
        return _lathe(top + bottom, n)
    if kind == "plane":
        hx = s[0] if s[0] > 0 else INFINITE_PLANE_HALF
        hy = s[1] if s[1] > 0 else INFINITE_PLANE_HALF
        v = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]], float)
        return v, np.tile([0.0, 0.0, 1.0], (4, 1)), np.array([[0, 1, 2], [0, 2, 3]])
    raise ValueError(f"unsupported primitive {kind!r}")


# ------------------------------------------------------------------------------- reading

def _json_bytes(data) -> dict:
    raw = bytes(np.asarray(data, np.uint8))
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def read_scene(path):
    """``(SceneComponents, library directory)`` of an episode file; ``(None, None)`` without a scene."""
    import h5py

    from .episode import SceneComponents

    path = Path(path)
    with h5py.File(path, "r") as f:
        if "scene" not in f:
            return None, None
        g = f["scene"]
        if g.attrs.get("schema") != SCENE_SCHEMA:
            raise ValueError(f"{path}: /scene is not {SCENE_SCHEMA}")
        desc = _json_bytes(g["description"][()])
        info = dict(desc.get("info") or {})
        info.setdefault("library", g.attrs.get("library") or None)
        scene = SceneComponents(components=desc["components"], materials=desc.get("materials", {}),
                                textures=desc.get("textures", {}), poses=g["poses"][()], valid=g["valid"][()],
                                physics_poses=g["physics_poses"][()] if "physics_poses" in g else None, info=info)
    return scene, library_path(path, info)


def library_path(episode_path, info: dict | None = None, library=None) -> Path:
    """The asset library of an episode: ``library`` if given, else the relative path recorded in
    ``/scene`` (``info["library"]``), else the first ``assets/`` folder found upwards."""
    if library is not None:
        return Path(library)
    episode_path = Path(episode_path).resolve()
    rel = (info or {}).get("library")
    if rel:
        p = (episode_path.parent / rel).resolve()
        if p.is_dir():
            return p
    for parent in episode_path.parents:
        if (parent / ASSET_DIR).is_dir():
            return parent / ASSET_DIR
    raise FileNotFoundError(f"no asset library found for {episode_path}")


def asset_file(library, ref: dict) -> Path:
    p = Path(library) / f"{ref['sha256']}.{ref['format']}"
    if not p.is_file():
        raise FileNotFoundError(f"asset {p} missing from the library")
    if p.stat().st_size != ref["bytes"]:
        raise ValueError(f"asset {p} has {p.stat().st_size} bytes, the episode records {ref['bytes']}")
    return p


def _pose_matrix(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(np.asarray(quat, float))
    T[:3, 3] = pos
    return T


def effective_rgba(part: dict, materials: dict) -> list[float]:
    """MuJoCo's colour rule: a geom rgba different from the internal default wins over its material."""
    rgba = list(part.get("rgba", DEFAULT_RGBA))
    mat = materials.get(part.get("material")) if part.get("material") else None
    if mat is not None and np.allclose(rgba, DEFAULT_RGBA):
        return list(mat["rgba"])
    return rgba


def load_component_meshes(episode_path, *, library=None, kind: str = "visual", components=None,
                          frame: int | None = None) -> dict[str, list[dict]]:
    """``{component name: [part, ...]}`` for ``kind`` ``"visual"`` or ``"collision"``.

    Each part: ``vertices`` (n, 3) float64 in the component frame (world frame at episode row
    ``frame`` if given; parts of components invalid at that row are skipped), ``faces`` (m, 3),
    ``normals`` (n, 3) or None, ``uv`` (n, 2) or None, ``type``/``size`` (MuJoCo geom), ``rgba``
    (effective colour), ``material`` (dict or None), ``texture`` (``{"path", "format", ...}`` of the
    material's texture file, or the builtin texture parameters, or None), ``source`` (geom name).
    Collision meshes are the convex hulls MuJoCo collides with only after hull computation: the
    stored mesh is the source mesh (``part["collision_shape"]`` says how MuJoCo uses it).
    """
    scene, lib = read_scene(episode_path)
    if scene is None:
        raise ValueError(f"{episode_path}: no /scene section")
    lib = library_path(episode_path, scene.info, library)
    want = None if components is None else set(components)
    cache: dict[str, dict] = {}
    out = {}
    for c, comp in enumerate(scene.components):
        if want is not None and comp["name"] not in want:
            continue
        if frame is not None and not scene.valid[frame, c]:
            continue
        W = np.eye(4) if frame is None else _pose_matrix(scene.poses[frame, c, :3], scene.poses[frame, c, 3:])
        parts = []
        for p in comp.get(kind, []):
            ref = p.get("mesh") or p.get("generated_mesh")
            if ref is not None:
                if ref["sha256"] not in cache:
                    cache[ref["sha256"]] = decode_msh(asset_file(lib, ref).read_bytes())
                m = cache[ref["sha256"]]
                v, faces, nrm, uv = m["vertices"], m["faces"], m["normals"], m["uv"]
            else:
                v, nrm, faces = primitive_mesh(p["type"], p["size"])
                uv = None
            T = W @ _pose_matrix(p["pos"], p["quat"])
            mat = scene.materials.get(p.get("material")) if p.get("material") else None
            tex = None
            if mat is not None and mat.get("texture"):
                t = scene.textures[mat["texture"]]
                tex = {**t, "name": mat["texture"]}
                if t.get("file"):
                    tex["path"] = str(asset_file(lib, t["file"]))
            parts.append({"vertices": v @ T[:3, :3].T + T[:3, 3], "faces": faces,
                          "normals": None if nrm is None else nrm @ T[:3, :3].T, "uv": uv,
                          "type": p["type"], "size": p.get("size"), "rgba": effective_rgba(p, scene.materials),
                          "material": mat, "texture": tex, "source": p.get("geom"),
                          "collision_shape": p.get("collision_shape")})
        out[comp["name"]] = parts
    return out


def surface_area(vertices, faces) -> float:
    tri = np.asarray(vertices)[np.asarray(faces)]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def _num(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.ravel(v))


def scene_mjcf(episode_path, *, frame: int | None = 0, library=None, kinds=("visual", "collision"),
               components=None, physics: bool = False) -> str:
    """MJCF of the scene components as static bodies at their poses of episode row ``frame``
    (``physics=True``: row ``frame`` of the tier-P rollout poses; ``frame=None``: every component
    at the origin, i.e. in its own frame, e.g. to render one component for a surface map with
    ``components=[name]``). Visual geoms keep their
    recorded group (non-colliding), collision geoms go to group 3 with their contact bits.
    Asset paths are absolute library paths; meshes use ``inertia="shell"`` (bodies are static,
    the inertia is never used and thin visual shells would otherwise be rejected)."""
    scene, _ = read_scene(episode_path)
    if scene is None:
        raise ValueError(f"{episode_path}: no /scene section")
    lib = library_path(episode_path, scene.info, library).resolve()
    poses = scene.physics_poses if physics else scene.poses
    if poses is None:
        raise ValueError(f"{episode_path}: no physics poses")
    if frame is None:
        poses = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], float), (1, len(scene.components), 1))
        frame = 0
        valid = np.ones(len(scene.components), bool)
    else:
        valid = np.isfinite(poses[frame]).all(axis=1) if physics else scene.valid[frame]
    want = None if components is None else set(components)
    root = ET.Element("mujoco", model="reachy_retarget_scene")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    meshes, mats, texs = set(), set(), set()

    def texture(name):
        if name in texs:
            return
        texs.add(name)
        t = scene.textures[name]
        attrs = {k: str(v) for k, v in (t.get("attributes") or {}).items()}
        for key, ref in (t.get("files") or {}).items():
            attrs[key] = str(asset_file(lib, ref))
        ET.SubElement(asset, "texture", name=name, **attrs)

    def material(name):
        if name in mats:
            return
        mats.add(name)
        m = scene.materials[name]
        attrs = {k: _num(m[k]) for k in ("rgba", "specular", "shininess", "reflectance", "emission",
                                           "texrepeat") if m.get(k) is not None}
        if m.get("texuniform") is not None:
            attrs["texuniform"] = "true" if m["texuniform"] else "false"
        el = ET.SubElement(asset, "material", name=name, **attrs)
        if m.get("texture"):
            texture(m["texture"])
            el.set("texture", m["texture"])

    for c, comp in enumerate(scene.components):
        if (want is not None and comp["name"] not in want) or not valid[c]:
            continue
        body = ET.SubElement(world, "body", name=comp["name"], pos=_num(poses[frame, c, :3]),
                             quat=_num(poses[frame, c, 3:]))
        for kind in kinds:
            for i, p in enumerate(comp.get(kind, [])):
                attrs = {"name": f"{comp['name']}/{kind}{i}", "pos": _num(p["pos"]), "quat": _num(p["quat"]),
                         "rgba": _num(p.get("rgba", DEFAULT_RGBA))}
                ref = p.get("mesh")
                if ref is not None:
                    mname = f"mesh_{ref['sha256'][:16]}"
                    if mname not in meshes:
                        meshes.add(mname)
                        ET.SubElement(asset, "mesh", name=mname, file=str(asset_file(lib, ref)), inertia="shell")
                    attrs.update(type="mesh", mesh=mname)
                else:
                    attrs.update(type=p["type"], size=_num(p["size"]))
                if kind == "visual":
                    attrs.update(group=str(p.get("group", 1)), contype="0", conaffinity="0")
                    if p.get("material"):
                        material(p["material"])
                        attrs["material"] = p["material"]
                else:
                    attrs.update(group="3", contype=str(p.get("contype", 1)), conaffinity=str(p.get("conaffinity", 1)))
                ET.SubElement(body, "geom", **attrs)
    return ET.tostring(root, encoding="unicode")


__all__ = ["SCENE_SCHEMA", "ASSET_DIR", "MESH_FORMAT", "asset_format", "encode_msh", "decode_msh", "primitive_mesh",
           "read_scene", "library_path", "asset_file", "effective_rgba", "load_component_meshes", "surface_area",
           "scene_mjcf"]
