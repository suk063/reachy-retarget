"""Generic URDF kinematics (numpy only): the joint tree, mimic resolution and batched forward
kinematics / geometric Jacobians of selected links.

Only joints, origins, axes, limits and mimic tags are read (standard-library XML, no meshes).
Units are metres and radians. Nothing here knows about Reachy.
"""
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _floats(text, default="0 0 0"):
    return np.array([float(v) for v in (text or default).split()])


def pose(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    """4x4 transform from a translation and URDF fixed-axis roll-pitch-yaw."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def axis_rotation(axis, angle):
    """Batched Rodrigues rotation: unit axis (3,), angles (...) -> (..., 3, 3)."""
    angle = np.asarray(angle, float)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    R = np.empty(angle.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 0, 2] = c + x * x * C, x * y * C - z * s, x * z * C + y * s
    R[..., 1, 0], R[..., 1, 1], R[..., 1, 2] = y * x * C + z * s, c + y * y * C, y * z * C - x * s
    R[..., 2, 0], R[..., 2, 1], R[..., 2, 2] = z * x * C - y * s, z * y * C + x * s, c + z * z * C
    return R


_EYE4 = np.eye(4)


def _axis_rotation_one(axis, angle):
    """:func:`axis_rotation` of one angle: axis a tuple of Python floats (the same IEEE operations
    in the same order, so bit-identical)."""
    x, y, z = axis
    c, s = float(np.cos(angle)), float(np.sin(angle))
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


class URDF:
    """Joint tree of a URDF file.

    `joints` maps child link -> joint record (name, type, parent, origin T, unit axis);
    `mimic[name] = (source, multiplier, offset)` is resolved to the independent source joint,
    so a mimic of a mimic is a single affine map.
    """

    def __init__(self, path):
        self.path = Path(path)
        root = ET.parse(self.path).getroot()
        self.links = [e.get("name") for e in root.findall("link")]
        self.joints, self.limits, direct = {}, {}, {}
        for j in root.findall("joint"):
            o, a, lim, m = j.find("origin"), j.find("axis"), j.find("limit"), j.find("mimic")
            o = {} if o is None else o.attrib
            axis = _floats(None if a is None else a.get("xyz"), "1 0 0")
            child = j.find("child").get("link")
            self.joints[child] = {
                "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"), "child": child,
                "T": pose(_floats(o.get("xyz")), _floats(o.get("rpy"))), "axis": axis / (np.linalg.norm(axis) or 1.0)}
            if lim is not None and j.get("type") in ("revolute", "prismatic"):
                self.limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
            if m is not None:
                direct[j.get("name")] = (m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0)))
        self.by_name = {j["name"]: j for j in self.joints.values()}
        roots = [link for link in self.links if link not in self.joints]
        if len(roots) != 1:
            raise ValueError(f"expected one root link in {self.path.name}, got {roots}")
        self.root = roots[0]
        self.mimic = {}
        for name in direct:
            src, mult, off = name, 1.0, 0.0
            while src in direct:
                s, m, o = direct[src]
                src, mult, off = s, mult * m, m * off + o
            self.mimic[name] = (src, mult, off)

    def source(self, name):
        """(independent joint, multiplier, offset) that drives joint `name`."""
        return self.mimic.get(name, (name, 1.0, 0.0))

    def motion(self, joint, value):
        """4x4 motion of a joint record at its own value (rad or m)."""
        M = np.eye(4)
        if joint["type"] in ("revolute", "continuous"):
            M[:3, :3] = axis_rotation(joint["axis"], value)
        elif joint["type"] == "prismatic":
            M[:3, 3] = joint["axis"] * value
        return M

    def joint_path(self, root, link):
        """Joint records from `root` down to `link` (root first)."""
        out = []
        while link != root:
            if link not in self.joints:
                raise KeyError(f"link {link!r} is not below {root!r} in {self.path.name}")
            out.append(self.joints[link])
            link = out[-1]["parent"]
        return out[::-1]

    def transform(self, frm, to, q=None):
        """Reference (slow, unbatched) T_frm_to for a joint dict q; missing joints are 0."""
        q = q or {}

        def from_root(link):
            T = np.eye(4)
            for j in self.joint_path(self.root, link):
                src, mult, off = self.source(j["name"])
                T = T @ j["T"] @ self.motion(j, mult * q.get(src, 0.0) + off)
            return T
        return np.linalg.inv(from_root(frm)) @ from_root(to)


class KinematicTree:
    """Batched FK and geometric Jacobians of `targets` relative to link `root`.

    Joint variables are the independent joints `names` (vector order). Joints driven by other
    independent joints (not in `names`) are held at `fixed.get(source, 0)` and folded into constant
    transforms together with all fixed joints, so a pass costs one small matmul per movable joint
    plus one per branch/target. Mimic joints whose source is in `names` move with it; their
    Jacobian columns are scaled by the multiplier and summed into the source column.
    """

    def __init__(self, model, root, targets, names, fixed=None):
        fixed = fixed or {}
        self.names, self.targets = list(names), list(targets)
        col = {n: i for i, n in enumerate(self.names)}
        paths = {t: model.joint_path(root, t) for t in self.targets}
        children = {}  # link -> child links used by some target path (branch detection)
        for p in paths.values():
            for j in p:
                children.setdefault(j["parent"], set()).add(j["child"])
        # nodes: (parent node or -1, F, axis, kind 0 fixed / 1 revolute / 2 prismatic, column, mult, off)
        self.nodes, entry = [], {root: (-1, np.eye(4))}
        for t in self.targets:
            for j in paths[t]:
                child = j["child"]
                if child in entry:
                    continue
                parent, acc = entry[j["parent"]]
                src, mult, off = model.source(j["name"])
                movable = j["type"] in ("revolute", "continuous", "prismatic") and src in col
                if movable:
                    kind = 1 if j["type"] != "prismatic" else 2
                    self.nodes.append((parent, acc @ j["T"], j["axis"], kind, col[src], mult, off))
                    entry[child] = (len(self.nodes) - 1, np.eye(4))
                    continue
                F = acc @ j["T"] @ model.motion(j, mult * fixed.get(src, 0.0) + off)
                if child in self.targets or len(children.get(child, ())) > 1:
                    self.nodes.append((parent, F, None, 0, -1, 0.0, 0.0))
                    entry[child] = (len(self.nodes) - 1, np.eye(4))
                else:
                    entry[child] = (parent, F)
        self.index = {t: entry[t][0] for t in self.targets}
        self.chains = {t: self._ancestors(self.index[t]) for t in self.targets}
        # single-configuration fast path (IK inner loop): the axis as Python floats
        self._axes = [None if axis is None else tuple(float(v) for v in axis) for _, _, axis, *_ in self.nodes]

    def _ancestors(self, i):
        out = []
        while i >= 0:
            if self.nodes[i][3]:
                out.append(i)
            i = self.nodes[i][0]
        return out[::-1]

    def node_poses(self, q):
        """Poses of all nodes for q (..., len(names)) -> list of (..., 4, 4)."""
        q = np.asarray(q, float)
        if q.shape[-1] != len(self.names):
            raise ValueError(f"expected {len(self.names)} joint values, got shape {q.shape}")
        if q.ndim == 1:
            return self._node_poses_one(q)
        batch = q.shape[:-1]
        out = []
        for parent, F, axis, kind, c, mult, off in self.nodes:
            T = F if parent < 0 else out[parent] @ F
            if kind:
                v = mult * q[..., c] + off
                M = np.broadcast_to(np.eye(4), batch + (4, 4)).copy()
                if kind == 1:
                    M[..., :3, :3] = axis_rotation(axis, v)
                else:
                    M[..., :3, 3] = v[..., None] * axis
                T = T @ M
            elif batch and parent < 0:
                T = np.broadcast_to(T, batch + (4, 4))
            out.append(T)
        return out

    def _node_poses_one(self, q):
        """:meth:`node_poses` of one configuration (q (len(names),)): the same floating-point
        operations without the batch broadcasting, whose per-node overhead dominates IK time."""
        out = []
        for (parent, F, axis, kind, c, mult, off), xyz in zip(self.nodes, self._axes):
            T = F if parent < 0 else out[parent] @ F
            if kind:
                v = mult * q[c] + off
                M = _EYE4.copy()
                if kind == 1:
                    M[:3, :3] = _axis_rotation_one(xyz, v)
                else:
                    M[:3, 3] = v * axis
                T = T @ M
            out.append(T)
        return out

    def fk(self, q):
        """{target: T_root_target (..., 4, 4)}."""
        P = self.node_poses(q)
        return {t: P[i] for t, i in self.index.items()}

    def jacobian(self, q, target, poses=None):
        """(T_root_target (4,4), 6 x len(names) geometric Jacobian in the root frame, linear rows
        first) for one configuration q; ``poses`` = ``node_poses(q)`` when already computed."""
        P = self.node_poses(q) if poses is None else poses
        T = P[self.index[target]]
        J = np.zeros((6, len(self.names)))
        for i in self.chains[target]:
            _, _, axis, kind, c, mult, _ = self.nodes[i]
            a = P[i][:3, :3] @ axis
            if kind == 2:
                J[:3, c] += mult * a
            else:
                r = T[:3, 3] - P[i][:3, 3]  # a x r, written out (np.cross dominates IK time)
                J[0, c] += mult * (a[1] * r[2] - a[2] * r[1])
                J[1, c] += mult * (a[2] * r[0] - a[0] * r[2])
                J[2, c] += mult * (a[0] * r[1] - a[1] * r[0])
                J[3:, c] += mult * a
        return T, J
