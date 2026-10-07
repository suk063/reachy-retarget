"""HDF5 storage of :class:`ReachyEpisode` (``reachy-retarget-episode-v2``) and the dataset index.

See ``docs/schema.md`` for the layout. Robot poses are stored as ``(T, 7)`` xyz + wxyz,
metadata as JSON attributes, and ``/actions`` holds precomputed control views that are
always regenerated from the canonical state on write.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .control_modes import compute_modes, MODES
from .episode import JOINTS, SIDES, PhysicsRollout, ReachyEpisode, Reference
from .rotations import pose_to_vec7, vec7_to_pose
from .source import Articulation, ObjectTrack

SCHEMA = "reachy-retarget-episode-v2"
_IMAGE_NAME = re.compile(r"image|img|rgb|depth|camera|pixel|video|segmentation", re.I)
_META = ("uid", "family", "dataset", "episode_id", "task", "instruction", "regime", "license",
         "provenance", "lineage", "variant_of", "body_parts", "retarget_config", "tier", "extra")


def _json(x):
    return json.dumps(x, sort_keys=True)


def check_not_image(name: str, data: np.ndarray) -> None:
    """Refuse datasets that look like images: suggestive names or uint8 arrays with ndim >= 3."""
    if _IMAGE_NAME.search(name):
        raise ValueError(f"refusing to store image-like dataset {name!r}")
    if data.dtype == np.uint8 and data.ndim >= 3:
        raise ValueError(f"refusing to store uint8 array of ndim {data.ndim} at {name!r}")


def _put(group: h5py.Group, name: str, data, **attrs) -> None:
    data = np.asarray(data)
    check_not_image(f"{group.name}/{name}", data)
    opts = {"compression": "gzip", "shuffle": True} if data.ndim and data.size else {}
    ds = group.create_dataset(name, data=data, **opts)
    for k, v in attrs.items():
        ds.attrs[k] = v


def write_episode(path, ep: ReachyEpisode) -> Path:
    """Atomically write ``ep`` and all control views to ``path``."""
    path = Path(path)
    modes = compute_modes(ep)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema"] = SCHEMA
            f.attrs["joint_names"] = _json(list(JOINTS))
            f.attrs["metadata"] = _json({k: getattr(ep, k) for k in _META})
            _put(f, "time", ep.time)
            _put(f, "source_time", ep.source_time)
            st = f.create_group("state")
            _put(st, "q", ep.q, columns=_json(list(JOINTS)))
            _put(st, "qd", ep.qd, columns=_json(list(JOINTS)))
            _put(st, "base_pose_world", ep.base_pose_world)
            world = ep.tcp_world
            for s in SIDES:
                g = st.create_group(f"tcp/{s}")
                _put(g, "world", pose_to_vec7(world[s]))
                _put(g, "base", pose_to_vec7(ep.tcp_base[s]))
            g = st.create_group("head")
            _put(g, "world", pose_to_vec7(ep.head_world))
            _put(g, "base", pose_to_vec7(ep.head_base))
            g = st.create_group("gripper")
            _put(g, "opening", ep.gripper_opening)
            _put(g, "width", ep.gripper_width)
            ref = f.create_group("reference")
            for s, P in ep.reference.tcp.items():
                _put(ref.require_group("tcp"), s, pose_to_vec7(P))
            if ep.reference.head is not None:
                _put(ref, "head", pose_to_vec7(ep.reference.head))
            if ep.reference.base is not None:
                _put(ref, "base", ep.reference.base)
            objs = f.create_group("objects")
            for k, o in ep.objects.items():
                g = objs.create_group(k)
                g.attrs["role"], g.attrs["geometry"] = o.role, _json(o.geometry)
                _put(g, "pose", o.pose)
                _put(g, "valid", o.valid)
            arts = f.create_group("articulations")
            for k, a in ep.articulations.items():
                g = arts.create_group(k)
                g.attrs["joint_names"] = _json(a.joint_names)
                _put(g, "qpos", a.qpos)
            val = f.create_group("validation")
            for k, v in ep.validation.items():
                _put(val, k, v)
            if ep.physics is not None:
                p, g = ep.physics, f.create_group("physics")
                g.attrs["info"] = _json(p.info)
                _put(g, "time", p.time)
                for k in ("qpos", "qvel", "ctrl"):
                    _put(g, k, getattr(p, k), columns=_json(getattr(p, f"{k}_names")))
            act = f.create_group("actions")
            for name, arrays in modes.items():
                spec, g = MODES[name], act.create_group(name)
                g.attrs.update(frame=spec.frame, units=spec.units, semantics=spec.semantics)
                for k, a in arrays.items():
                    _put(g, k, a, columns=_json(list(spec.columns[k])))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_episode(path) -> ReachyEpisode:
    """Read the canonical episode content; ``/actions`` and world-frame views are ignored."""
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} file")
        if json.loads(f.attrs["joint_names"]) != list(JOINTS):
            raise ValueError(f"{path}: joint order differs from JOINTS")
        meta = json.loads(f.attrs["metadata"])
        meta["body_parts"] = tuple(meta["body_parts"])
        ref = f["reference"]
        reference = Reference(
            tcp={s: vec7_to_pose(d[()]) for s, d in ref["tcp"].items()} if "tcp" in ref else {},
            head=vec7_to_pose(ref["head"][()]) if "head" in ref else None,
            base=ref["base"][()] if "base" in ref else None)
        objects = {k: ObjectTrack(g["pose"][()], g["valid"][()], g.attrs["role"], json.loads(g.attrs["geometry"]))
                   for k, g in f["objects"].items()}
        arts = {k: Articulation(json.loads(g.attrs["joint_names"]), g["qpos"][()])
                for k, g in f["articulations"].items()}
        physics = None
        if "physics" in f:
            g = f["physics"]
            physics = PhysicsRollout(
                g["time"][()], g["qpos"][()], g["qvel"][()], g["ctrl"][()],
                *(json.loads(g[k].attrs["columns"]) for k in ("qpos", "qvel", "ctrl")),
                info=json.loads(g.attrs["info"]))
        st = f["state"]
        return ReachyEpisode(
            time=f["time"][()], source_time=f["source_time"][()], q=st["q"][()], qd=st["qd"][()],
            tcp_base={s: vec7_to_pose(st[f"tcp/{s}/base"][()]) for s in SIDES},
            head_base=vec7_to_pose(st["head/base"][()]),
            gripper_opening=st["gripper/opening"][()], gripper_width=st["gripper/width"][()],
            reference=reference, objects=objects, articulations=arts,
            validation={k: d[()] for k, d in f["validation"].items()}, physics=physics, **meta)


def recompute_modes(path) -> Path:
    """Regenerate every ``/actions`` view of an episode file from its canonical state."""
    return write_episode(path, read_episode(path))


INDEX_SCHEMA = pa.schema([
    ("uid", pa.string()), ("file", pa.string()), ("family", pa.string()), ("dataset", pa.string()),
    ("task", pa.string()), ("regime", pa.string()), ("body_parts", pa.list_(pa.string())),
    ("tier_k_passed", pa.bool_()), ("tier_p_passed", pa.bool_()), ("license", pa.string()),
    ("lineage_seed", pa.string()), ("variant_of", pa.string()), ("length", pa.int64()),
    ("duration", pa.float64())])


def index_row(ep: ReachyEpisode, file: str) -> dict:
    """One ``index.parquet`` row; ``file`` is the episode path relative to the index."""
    seed = ep.lineage.get("seed")
    return {"uid": ep.uid, "file": str(file), "family": ep.family, "dataset": ep.dataset,
            "task": ep.task, "regime": ep.regime, "body_parts": list(ep.body_parts),
            "tier_k_passed": ep.tier["K"]["passed"],
            "tier_p_passed": None if ep.tier["P"] is None else ep.tier["P"]["passed"],
            "license": ep.license, "lineage_seed": None if seed is None else str(seed),
            "variant_of": ep.variant_of, "length": ep.length, "duration": ep.duration}


def write_index(directory, rows) -> Path:
    """Atomically write ``<directory>/index.parquet`` from :func:`index_row` dicts."""
    path = Path(directory) / "index.parquet"
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        pq.write_table(pa.Table.from_pylist(list(rows), schema=INDEX_SCHEMA), tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
