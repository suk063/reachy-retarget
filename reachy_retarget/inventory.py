"""Count acquired sequences, never advertised sizes or resampled windows."""

import collections
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pyarrow.parquet as pq
import h5py
from .store import json_write, sha256
from .human import array_pickle, GLB


def fingerprint_array(h, x):
    a = np.asarray(x)
    if a.dtype.kind in "bifu":
        a = np.round(a.astype("<f8"), 6)
        h.update(str(a.shape[1:]).encode())
        h.update(a.tobytes())


def run(store):
    inventory = []
    sources = store.rows("sources")
    downloaded = collections.defaultdict(list)
    for f in store.rows("files", "status='downloaded'"):
        downloaded[f["source_id"]].append(f)
    for source in sources:
        sid = source["id"]
        root = store.root / "data/raw" / sid
        records = {}
        hashes = {}
        files = downloaded[sid]
        cache = store.root / "catalog/inventory_cache" / (sid + ".json")
        signature = hashlib.sha256(
            json.dumps(sorted((f["path"], f["sha256"]) for f in files)).encode()
        ).hexdigest()
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved.get("version") == 3 and saved.get("signature") == signature:
                inventory.extend(saved["records"])
                continue
        before = len(inventory)
        parquets = sorted(
            [
                f
                for f in files
                if f["path"].endswith(".parquet") and "/data/" in "/" + f["path"]
            ],
            key=lambda f: f["path"],
        )
        for file in parquets:
            p = root / file["path"]
            schema = pq.read_schema(p)
            names = schema.names
            if "episode_index" not in names:
                continue
            # Numeric state/actions provide cross-reformat content fingerprints.
            numeric = [
                k
                for k in (
                    "observation.state",
                    "action",
                    "observation.environment_state",
                )
                if k in names
            ]
            if not numeric:
                continue
            columns = (
                ["episode_index"]
                + (["timestamp"] if "timestamp" in names else [])
                + numeric
            )
            table = pq.read_table(p, columns=columns).to_pydict()
            ep = np.asarray(table["episode_index"])
            prefix = file["path"].split("data/", 1)[0]
            # Convert each column once, not once per episode in a large v3 shard.
            times = (
                np.asarray(table["timestamp"], float) if "timestamp" in table else None
            )
            values = {name: np.asarray(table[name]) for name in numeric}
            order = np.argsort(ep, kind="stable")
            boundaries = np.r_[0, np.flatnonzero(np.diff(ep[order])) + 1, len(ep)]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                ix = order[start:end]
                episode = ep[ix[0]]
                sequence = prefix + f"episode_{int(episode):06d}"
                key = (sid, sequence)
                r = records.setdefault(
                    sequence,
                    {
                        "source_id": sid,
                        "sequence": sequence,
                        "frames": 0,
                        "time_min": float("inf"),
                        "time_max": float("-inf"),
                        "format": "parquet",
                        "files": [],
                    },
                )
                r["frames"] += len(ix)
                r["files"].append(file["path"])
                channels = hashes.setdefault(sequence, {})
                if times is not None:
                    t = times[ix]
                    r["time_min"] = min(r["time_min"], float(np.nanmin(t)))
                    r["time_max"] = max(r["time_max"], float(np.nanmax(t)))
                for name in numeric:
                    value = values[name][ix]
                    if name not in channels:
                        channels[name] = hashlib.sha256()
                        channels[name].update(str(value.shape[1:]).encode())
                    channels[name].update(np.round(value.astype("<f8"), 6).tobytes())
        for sequence, r in records.items():
            r["duration_s"] = r.pop("time_max") - r.pop("time_min")
            r["duration_s"] = (
                r["duration_s"]
                if np.isfinite(r["duration_s"]) and r["duration_s"] >= 0
                else None
            )
            digest = hashlib.sha256()
            for name, h in sorted(hashes[sequence].items()):
                digest.update(name.encode())
                digest.update(h.digest())
            r["state_content_sha256"] = digest.hexdigest()
            inventory.append(r)
        hdf5 = [f for f in files if f["path"].endswith((".hdf5", ".h5"))]
        for file in hdf5:
            p = root / file["path"]
            try:
                with h5py.File(p, "r") as f:
                    group = f["data"] if "data" in f else f
                    env = json.loads(group.attrs.get("env_args", "{}"))
                    fps = env.get("env_kwargs", {}).get("control_freq")
                    for name, g in group.items():
                        if not isinstance(g, h5py.Group):
                            continue
                        if "steps" in g:
                            steps = g["steps"]
                            h = hashlib.sha256()
                            channels = []

                            def rlds_hash(key, d):
                                if (
                                    isinstance(d, h5py.Dataset)
                                    and d.dtype.kind in "bifu"
                                    and (
                                        key.startswith("action")
                                        or key.startswith("observation")
                                    )
                                ):
                                    h.update(key.encode())
                                    fingerprint_array(h, d[()])
                                    channels.append(key)

                            steps.visititems(rlds_hash)
                            if channels:
                                n = (
                                    len(steps["is_first"])
                                    if "is_first" in steps
                                    else None
                                )
                                inventory.append(
                                    {
                                        "source_id": sid,
                                        "sequence": file["path"] + "/" + name,
                                        "frames": n,
                                        "duration_s": None,
                                        "state_content_sha256": h.hexdigest(),
                                        "format": "rlds_state_hdf5",
                                        "files": [file["path"]],
                                    }
                                )
                            continue
                        if "actions" not in g:
                            continue
                        n = len(g["actions"])
                        h = hashlib.sha256()
                        fingerprint_array(h, g["actions"][()])
                        if "states" in g:
                            fingerprint_array(h, g["states"][()])
                        elif "env_states" in g:

                            def state_hash(key, d):
                                if isinstance(d, h5py.Dataset):
                                    h.update(key.encode())
                                    fingerprint_array(h, d[()])

                            g["env_states"].visititems(state_hash)
                        inventory.append(
                            {
                                "source_id": sid,
                                "sequence": file["path"] + "/" + name,
                                "frames": n,
                                "duration_s": (n - 1) / fps if fps else None,
                                "state_content_sha256": h.hexdigest(),
                                "format": "hdf5",
                                "files": [file["path"]],
                            }
                        )
                    if (
                        "actions" in f
                        and isinstance(f["actions"], h5py.Dataset)
                        and "terminals" in f
                    ):
                        ends = np.asarray(f["terminals"][()], bool).ravel()
                        if "timeouts" in f:
                            ends |= np.asarray(f["timeouts"][()], bool).ravel()
                        start = 0
                        for i, end in enumerate(np.flatnonzero(ends) + 1):
                            h = hashlib.sha256()
                            for key in ("actions", "observations"):
                                if key in f:
                                    h.update(key.encode())
                                    fingerprint_array(h, f[key][start:end])
                            inventory.append(
                                {
                                    "source_id": sid,
                                    "sequence": file["path"] + f"/episode_{i}",
                                    "frames": int(end - start),
                                    "duration_s": None,
                                    "state_content_sha256": h.hexdigest(),
                                    "format": "flat_hdf5",
                                    "files": [file["path"]],
                                }
                            )
                            start = end
            except (OSError, ValueError, KeyError) as e:
                print("inventory", sid, file["path"], str(e), flush=True)
        print(
            "inventory",
            sid,
            "sequences",
            len(records)
            + sum(r["source_id"] == sid and r["format"] == "hdf5" for r in inventory),
            flush=True,
        )
        json_write(
            cache, {"version": 3, "signature": signature, "records": inventory[before:]}
        )
    # Canonical human sequence identities from the actual downloaded archives.
    archive = store.root / "data/raw/parahome/seq.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not name.endswith("/joint_positions.pkl"):
                    continue
                x = array_pickle(z.read(name))
                h = hashlib.sha256()
                fingerprint_array(h, x)
                inventory.append(
                    {
                        "source_id": "parahome",
                        "sequence": name.rsplit("/", 1)[0],
                        "frames": len(x),
                        "duration_s": (len(x) - 1) / 30,
                        "state_content_sha256": h.hexdigest(),
                        "format": "parahome",
                        "files": ["seq.zip::" + name],
                    }
                )
    for p in sorted((store.root / "data/raw/humoto").rglob("*.glb")):
        glb = GLB(p)
        times = [
            glb.accessor(s["input"])
            for a in glb.doc.get("animations", [])
            for s in a["samplers"]
        ]
        t = np.unique(np.concatenate(times))
        inventory.append(
            {
                "source_id": "humoto",
                "sequence": p.stem,
                "frames": len(t),
                "duration_s": float(t[-1] - t[0]),
                "state_content_sha256": sha256(p),
                "format": "glb",
                "files": [str(p.relative_to(store.root / "data/raw/humoto"))],
            }
        )
    for p in sorted((store.root / "data/raw/lafan1").rglob("*.zip")):
        if not zipfile.is_zipfile(p):
            print("Inventory excludes invalid/LFS-pointer archive:", p, flush=True)
            continue
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if not name.lower().endswith(".bvh"):
                    continue
                text = z.read(name).decode()
                import re

                n = int(re.search(r"Frames:\s*(\d+)", text).group(1))
                dt = float(re.search(r"Frame Time:\s*([\d.]+)", text).group(1))
                inventory.append(
                    {
                        "source_id": "lafan1",
                        "sequence": name,
                        "frames": n,
                        "duration_s": (n - 1) * dt,
                        "state_content_sha256": hashlib.sha256(
                            text.encode()
                        ).hexdigest(),
                        "format": "bvh",
                        "files": [p.name + "::" + name],
                    }
                )
    objects = store.root / "catalog/object_inventory.json"
    if objects.exists():
        for r in json.loads(objects.read_text()).get(
            "additional_verified_sequence_folders", []
        ):
            if "state_content_sha256" in r:
                inventory.append(
                    dict(
                        r,
                        source_id=r["source"],
                        format="linked_object_pose_archive",
                        files=["Object_Poses.zip"],
                    )
                )
    # Byte-identical numerical demonstrations are counted once even across mirrors
    # and raw/lowdim representations. Unknown overlap remains explicitly flagged.
    unique = {}
    duplicates = []
    for r in inventory:
        key = r["state_content_sha256"]
        if key in unique:
            duplicates.append(
                {
                    "source_id": r["source_id"],
                    "sequence": r["sequence"],
                    "same_as": unique[key]["source_id"] + "/" + unique[key]["sequence"],
                }
            )
            r["duplicate_of"] = duplicates[-1]["same_as"]
        else:
            unique[key] = r
    path = store.root / "catalog/sequence_inventory.jsonl"
    with path.open("w") as f:
        for r in inventory:
            f.write(json.dumps(r) + "\n")
    summary = {
        "observed_sequences_before_exact_dedup": len(inventory),
        "unique_state_fingerprints": len(unique),
        "exact_duplicate_sequences": len(duplicates),
        "known_duration_s_after_exact_dedup": sum(
            r["duration_s"] for r in unique.values() if r["duration_s"] is not None
        ),
        "unknown_duration_sequences": sum(
            r["duration_s"] is None for r in unique.values()
        ),
        "scope": "Acquired numeric LeRobot, HDF5, selected RLDS shards, TACO object-pose sequences, ParaHome, public HUMOTO and LAFAN1. Other unparsed archives are excluded. Unknown frame rates are never inferred.",
        "dedup_limits": "Content dedup detects matching numerical arrays. Semantically overlapping crops, re-timed motions and differing schemas may remain; corpus-family lower bound reported separately.",
        "sources": dict(collections.Counter(r["source_id"] for r in unique.values())),
    }
    json_write(store.root / "catalog/inventory_summary.json", summary)
    json_write(store.root / "catalog/duplicate_sequences.json", duplicates)
    return summary
