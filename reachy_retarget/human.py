"""Human state adapters without SMPL/MANO downloads or arbitrary pickle code."""

import io
import json
import pickle
import re
import shutil
import struct
import zipfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import trimesh
from .episodes import write_episode, matrices_to_pose
from .store import sha256


def legacy_numpy_bytes(value, encoding="latin1", errors="strict"):
    if (
        not isinstance(value, str)
        or encoding not in ("latin1", "latin-1", "ascii", "utf-8")
        or errors != "strict"
    ):
        raise pickle.UnpicklingError("Unsupported legacy NumPy byte encoding")
    return value.encode(encoding, errors)


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("_codecs", "encode"):
            return legacy_numpy_bytes
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"),
            ("numpy.core.numeric", "_frombuffer"),
            ("numpy._core.numeric", "_frombuffer"),
            ("collections", "OrderedDict"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported pickle global {module}.{name}")
        return super().find_class(module, name)


def array_pickle(content):
    return ArrayUnpickler(io.BytesIO(content)).load()


def safe_extract(archive, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        total = sum(i.file_size for i in z.infolist())
        if shutil.disk_usage(directory).free - total < 50_000_000_000:
            raise RuntimeError("50 GB disk reserve during extraction")
        for info in z.infolist():
            p = Path(info.filename)
            if (
                p.is_absolute()
                or ".." in p.parts
                or ((info.external_attr >> 16) & 0o170000) == 0o120000
            ):
                raise ValueError("Unsafe archive member")
            dst = directory / p
            if info.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            if dst.exists() and dst.stat().st_size == info.file_size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)


class GLB:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        if raw[:4] != b"glTF":
            raise ValueError("Not a glTF binary")
        off = 12
        while off < len(raw):
            length, kind = struct.unpack_from("<II", raw, off)
            chunk = raw[off + 8 : off + 8 + length]
            if kind == 0x4E4F534A:
                self.doc = json.loads(chunk)
            elif kind == 0x004E4942:
                self.buffer = chunk
            off += 8 + length

    def accessor(self, index):
        a = self.doc["accessors"][index]
        v = self.doc["bufferViews"][a["bufferView"]]
        if "sparse" in a:
            raise ValueError("Sparse accessor unsupported")
        dtype = np.dtype(
            {
                5120: "i1",
                5121: "u1",
                5122: "<i2",
                5123: "<u2",
                5125: "<u4",
                5126: "<f4",
            }[a["componentType"]]
        )
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        offset = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = v.get("byteStride", width * dtype.itemsize)
        x = np.ndarray(
            (a["count"], width),
            dtype=dtype,
            buffer=self.buffer,
            offset=offset,
            strides=(stride, dtype.itemsize),
        ).copy()
        return x[:, 0] if width == 1 else x

    def world_animation(self):
        d = self.doc
        nodes = d["nodes"]
        channels = {}
        all_times = []
        # Separate glTF animation tracks animate the body and interacted objects
        # simultaneously in the official release (not alternative takes).
        for anim in d.get("animations", []):
            for c in anim["channels"]:
                s = anim["samplers"][c["sampler"]]
                t = self.accessor(s["input"]).astype(float)
                v = self.accessor(s["output"])
                key = (c["target"]["node"], c["target"]["path"])
                if key in channels:
                    raise ValueError("Ambiguous overlapping animation tracks")
                channels[key] = (t, v, s.get("interpolation", "LINEAR"))
                all_times.append(t)
        t = np.unique(np.concatenate(all_times))
        n = len(t)
        local = []
        for i, node in enumerate(nodes):
            props = {}
            for key, default in [
                ("translation", [0, 0, 0]),
                ("rotation", [0, 0, 0, 1]),
                ("scale", [1, 1, 1]),
            ]:
                base = node.get(key, default)
                if (i, key) not in channels:
                    props[key] = np.tile(base, (n, 1))
                    continue
                tt, v, method = channels[i, key]
                if method == "STEP":
                    props[key] = v[
                        np.clip(np.searchsorted(tt, t, side="right") - 1, 0, len(v) - 1)
                    ]
                elif method == "LINEAR":
                    props[key] = (
                        Slerp(tt, Rotation.from_quat(v))(
                            np.clip(t, tt[0], tt[-1])
                        ).as_quat()
                        if key == "rotation"
                        else np.stack(
                            [np.interp(t, tt, v[:, j]) for j in range(v.shape[1])],
                            axis=1,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported interpolation {method}")
            m = np.tile(np.eye(4), (n, 1, 1))
            m[:, :3, 3] = props["translation"]
            m[:, :3, :3] = (
                Rotation.from_quat(props["rotation"]).as_matrix()
                * props["scale"][:, None, :]
            )
            if "matrix" in node:
                if any(k[0] == i for k in channels):
                    raise ValueError("Animated matrix node")
                m[:] = np.array(node["matrix"]).reshape(4, 4).T
            local.append(m)
        parent = {
            child: i
            for i, node in enumerate(nodes)
            for child in node.get("children", [])
        }
        world = {}

        def visit(i):
            if i not in world:
                world[i] = visit(parent[i]) @ local[i] if i in parent else local[i]
            return world[i]

        # glTF is meters, right-handed Y-up. Map X->X, Y->Z, Z->-Y.
        basis = np.eye(4)
        basis[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        return t - t[0], np.stack([basis @ visit(i) for i in range(len(nodes))], axis=1)


def rigid_matrices(m):
    out = m.copy()
    u, _, v = np.linalg.svd(out[..., :3, :3])
    out[..., :3, :3] = u @ v
    if np.any(np.linalg.det(out[..., :3, :3]) < 0):
        raise ValueError("Reflected transform")
    return out


def humoto(store, source, limit):
    files = sorted((store.root / "data/raw" / source).rglob("*.glb"))
    out = []
    for p in files[: limit or None]:
        glb = GLB(p)
        t, world = glb.world_animation()
        nodes = glb.doc["nodes"]
        names = [n.get("name", str(i)) for i, n in enumerate(nodes)]
        rigid = rigid_matrices(world)
        arrays = {
            "time_s": t,
            "human/joint_names": names,
            "human/node_matrix": world,
            "human/joint_pose": matrices_to_pose(rigid),
        }
        for side, label in [("left", "Left"), ("right", "Right")]:
            i = names.index("mixamorig:" + label + "Hand")
            arrays["hand/" + side + "_pose"] = matrices_to_pose(rigid[:, i])
        hip = names.index("mixamorig:Hips")
        arrays["human/root_pose"] = matrices_to_pose(rigid[:, hip])
        objects = {}
        for i, node in enumerate(nodes):
            if "mesh" not in node or "skin" in node:
                continue
            oid = re.sub("[^a-zA-Z0-9_-]", "_", names[i])
            arrays[f"objects/{oid}/pose"] = matrices_to_pose(rigid[:, i])
            scale = np.linalg.norm(world[:, i, :3, :3], axis=1)
            if np.max(np.ptp(scale, axis=0)) > 1e-4:
                raise ValueError("Time-varying object scale needs a deformable adapter")
            meshes = []
            for primitive in glb.doc["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("mode", 4) != 4:
                    raise ValueError("Object geometry is not triangles")
                verts = glb.accessor(primitive["attributes"]["POSITION"]) * scale[0]
                faces = glb.accessor(primitive["indices"]).reshape(-1, 3)
                meshes.append(trimesh.Trimesh(verts, faces, process=False))
            asset = store.root / "data/assets/humoto" / p.stem / (oid + ".obj")
            asset.parent.mkdir(parents=True, exist_ok=True)
            trimesh.util.concatenate(meshes).export(asset)
            objects[oid] = {
                "mesh": str(asset.relative_to(store.root)),
                "sha256": sha256(asset),
                "pose_frame": "world",
                "source_node": i,
                "scale_baked": scale[0].tolist(),
            }
        meta = {
            "source_format": "glTF-2.0-HUMOTO",
            "objects": objects,
            "source_coordinates": "right-handed Y-up, meters, glTF xyzw",
            "world_frame": "right-handed Z-up",
            "provenance": [
                {"path": str(p.relative_to(store.root)), "sha256": sha256(p)}
            ],
            "missing": ["contact", "mass", "friction"],
            "hand_orientation": "source anatomical wrist frame, not a robot TCP",
            "time_origin": "first common glTF keyframe; all animation tracks combined",
        }
        out.append(write_episode(store, source, p.stem, arrays, meta))
    if not out:
        raise ValueError("No downloaded HUMOTO GLB files")
    return out


def parahome(store, source, limit):
    root = store.root / "data/raw" / source
    if not (root / "seq.zip").exists():
        raise ValueError("ParaHome seq.zip download not complete")
    assets = store.root / "data/assets/parahome"
    safe_extract(root / "scan.zip", assets)
    out = []
    with zipfile.ZipFile(root / "seq.zip") as z:
        seqs = sorted(
            {
                n.rsplit("/", 1)[0]
                for n in z.namelist()
                if n.endswith("/joint_positions.pkl")
            },
            key=lambda n: int(re.search(r"s(\d+)$", n).group(1)),
        )
        for seq in seqs[: limit or None]:

            def load(name):
                return array_pickle(z.read(seq + "/" + name + ".pkl"))

            joints = load("joint_positions")
            body_6d = load("body_joint_orientations")
            root_T = load("body_global_transform")
            a, b = body_6d[..., :3], body_6d[..., 3:]
            x = a / np.linalg.norm(a, axis=-1, keepdims=True)
            y = b - np.sum(x * b, axis=-1, keepdims=True) * x
            y /= np.linalg.norm(y, axis=-1, keepdims=True)
            body_R = np.stack((x, y, np.cross(x, y)), axis=-2)
            transforms = load("object_transformations")
            states = load("joint_states")
            # Camera-synchronized release: paper Appendix A.2 specifies 30 Hz
            # cameras and 60 Hz IMUs. These are aligned per camera frame by the
            # release, rather than treating raw IMU samples as 30 Hz.
            fps = 30.0
            arrays = {
                "time_s": np.arange(len(joints)) / fps,
                "human/joint_position": joints,
                "human/body_orientation": body_R,
                "human/root_pose": matrices_to_pose(root_T),
                "source/body_orientation_6d": body_6d,
                "source/hand_orientation_6d": load("hand_joint_orientations"),
            }
            for side, idx in [("left", 14), ("right", 10)]:
                m = np.tile(np.eye(4), (len(joints), 1, 1))
                m[:, :3, 3] = joints[:, idx]
                m[:, :3, :3] = root_T[:, :3, :3] @ body_R[:, idx]
                arrays["hand/" + side + "_pose"] = matrices_to_pose(m)
            objects = {}
            in_scene = json.loads(z.read(seq + "/object_in_scene.json"))
            names = sorted(
                {
                    name
                    for frame in transforms.values()
                    for name in frame
                    if name.rsplit("_", 1)[0] in in_scene
                }
            )
            for name in names:
                valid = np.array(
                    [
                        i in transforms and name in transforms[i]
                        for i in range(len(joints))
                    ]
                )
                pose = np.full((len(joints), 7), np.nan)
                pose[valid] = matrices_to_pose(
                    np.array([transforms[i][name] for i in np.flatnonzero(valid)])
                )
                arrays[f"objects/{name}/pose"] = pose
                arrays[f"objects/{name}/valid"] = valid
                obj, part = name.rsplit("_", 1)
                meshes = list(
                    (assets / "scan" / obj / "simplified").glob(part + ".obj")
                )
                objects[name] = {
                    "meshes": [str(p.relative_to(store.root)) for p in meshes],
                    "mesh_sha256": [sha256(p) for p in meshes],
                    "parent_object": obj,
                    "pose_frame": "source-world",
                    "articulation_metadata": str(
                        (root / "joint_info.zip").relative_to(store.root)
                    ),
                }
                if name in states:
                    value = np.array(
                        [states[name].get(i, np.nan) for i in range(len(joints))]
                    )
                    arrays[f"objects/{name}/joint_state"] = value
            annopath = seq + "/text_annotation.json"
            meta = {
                "source_format": "ParaHome-numpy",
                "objects": objects,
                "fps": fps,
                "time_evidence": "https://arxiv.org/html/2401.10232v2#A2; 30 Hz camera-synchronized release, raw IMUs 60 Hz",
                "world_frame": "source camera/world; hand and object transforms remain aligned",
                "provenance": [
                    {
                        "path": str((root / "seq.zip").relative_to(store.root)),
                        "archive_member_prefix": seq,
                    }
                ],
                "missing": ["contact", "mass", "friction"],
                "hand_orientation": "anatomical wrist; indices from official visualize/utils.py",
                "annotations": json.loads(z.read(annopath))
                if annopath in z.namelist()
                else None,
                "object_validity": "NaN pose with explicit valid mask when source frame is absent",
                "source_group": source + "/" + seq,
            }
            out.append(write_episode(store, source, seq, arrays, meta))
    return out
