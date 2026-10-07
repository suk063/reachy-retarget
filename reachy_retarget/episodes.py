"""Versioned state-only episodes. Missing quantities stay missing."""

import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .store import json_write, sha256

SCHEMA = "reachy-retarget-episode-v1"


def split_for(source_group):
    bucket = int(hashlib.sha256(source_group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def matrices_to_pose(m):
    m = np.asarray(m)
    q = (
        Rotation.from_matrix(m[..., :3, :3].reshape(-1, 3, 3))
        .as_quat()
        .reshape(*m.shape[:-2], 4)
    )
    return np.concatenate((m[..., :3, 3], q[..., [3, 0, 1, 2]]), axis=-1)


def pose_to_matrices(p):
    p = np.asarray(p)
    m = np.broadcast_to(np.eye(4), (*p.shape[:-1], 4, 4)).copy()
    m[..., :3, 3] = p[..., :3]
    m[..., :3, :3] = (
        Rotation.from_quat(p[..., [4, 5, 6, 3]].reshape(-1, 4))
        .as_matrix()
        .reshape(*p.shape[:-1], 3, 3)
    )
    return m


def interpolate_pose(t, p, new_t):
    p = np.asarray(p)
    out = np.empty((len(new_t), 7))
    for j in range(3):
        out[:, j] = np.interp(new_t, t, p[:, j])
    out[:, 3:] = Slerp(t, Rotation.from_quat(p[:, [4, 5, 6, 3]]))(
        np.clip(new_t, t[0], t[-1])
    ).as_quat()[:, [3, 0, 1, 2]]
    return out


def write_episode(store, source, sequence, arrays, metadata):
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    t = arrays["time_s"].astype(float)
    if len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("Episode needs strictly increasing finite timestamps")
    eid = hashlib.sha256((source + "\0" + sequence).encode()).hexdigest()[:24]
    group = metadata.get("source_group", source + "/" + sequence)
    fingerprint = hashlib.sha256()
    for key in sorted(arrays):
        a = arrays[key]
        if a.dtype.kind not in "biuf":
            continue
        fingerprint.update(key.encode())
        fingerprint.update(str(a.shape).encode())
        fingerprint.update(a.tobytes())
    digest = fingerprint.hexdigest()
    split = split_for(group)
    for other in store.rows("episodes"):
        if json.loads(other["details"]).get("content_fingerprint") == digest:
            split = other["split"]
            group = json.loads(other["details"]).get("source_group", group)
            break
    metadata = dict(
        metadata,
        schema=SCHEMA,
        source_id=source,
        source_sequence=sequence,
        episode_id=eid,
        source_group=group,
        split=split,
        content_fingerprint=digest,
        canonical_units={
            "length": "m",
            "angle": "rad",
            "time": "s",
            "quaternion": "wxyz",
        },
        physics_validated=False,
    )
    path = store.root / "data/normalized" / source / (eid + ".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".h5.part")
    with h5py.File(temp, "w") as f:
        f.attrs["schema"] = SCHEMA
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key, a in arrays.items():
            if a.dtype.kind in "USO":
                f.create_dataset(
                    key, data=a.astype(object), dtype=h5py.string_dtype("utf-8")
                )
            else:
                f.create_dataset(
                    key, data=a, compression="gzip" if a.ndim and a.size > 10 else None
                )
    temp.replace(path)
    metadata["hdf5_sha256"] = sha256(path)
    json_write(path.with_suffix(".json"), metadata)
    objects = len(metadata.get("objects", {}))
    store.episode(
        eid,
        source,
        sequence,
        split,
        float(t[-1] - t[0]),
        len(t),
        objects,
        path.relative_to(store.root),
        "normalized",
        metadata,
    )
    return path


def read_episode(path):
    arrays = {}
    with h5py.File(path, "r") as f:
        metadata = json.loads(f.attrs["metadata_json"])

        def read(name, obj):
            if isinstance(obj, h5py.Dataset):
                arrays[name] = obj.asstr()[()] if obj.dtype.kind in "OSU" else obj[()]

        f.visititems(read)
    # The JSON sidecar is authoritative for provenance clarifications added
    # after immutable numeric HDF5 creation (and includes the HDF5 checksum).
    sidecar = Path(path).with_suffix(".json")
    if sidecar.exists():
        metadata.update(json.loads(sidecar.read_text()))
    return arrays, metadata


class StateDataset:
    """Minimal manipulation learning interface; explicit channels and validity."""

    def __init__(self, paths, channels, window=16):
        self.paths = list(map(Path, paths))
        self.channels = tuple(channels)
        self.window = window

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        arrays, meta = read_episode(self.paths[index])
        n = min(self.window, len(arrays["time_s"]))
        data = {key: arrays[key][:n] for key in self.channels if key in arrays}
        return {
            "data": data,
            "available": {k: k in arrays for k in self.channels},
            "metadata": meta,
        }
