"""Inventory source-scoped object IDs linked to acquired pose annotations."""

import io
import hashlib
import json
from pathlib import Path
import re
import zipfile
import tarfile
import numpy as np
from .store import json_write


def run(store):
    objects = []
    sequences = []
    poses = store.root / "data/raw/taco/Object_Poses.zip"
    models = store.root / "data/raw/taco/Object_Models.zip"
    if poses.exists() and models.exists():
        with zipfile.ZipFile(models) as z:
            meshes = {
                Path(n).stem.removesuffix("_cm"): n
                for n in z.namelist()
                if n.endswith(".obj")
            }
        used = {}
        records = {}
        with zipfile.ZipFile(poses) as z:
            for name in sorted(z.namelist()):
                if not name.endswith(".npy"):
                    continue
                x = np.load(io.BytesIO(z.read(name)), allow_pickle=False)
                oid = Path(name).stem.rsplit("_", 1)[-1]
                sequence = str(Path(name).parent)
                used.setdefault(oid, []).append(sequence)
                r = records.setdefault(
                    sequence,
                    {"frames": len(x), "objects": set(), "hash": hashlib.sha256()},
                )
                r["objects"].add(oid)
                r["hash"].update(oid.encode())
                r["hash"].update(np.round(x.astype("<f8"), 6).tobytes())
                if not np.isfinite(x).all():
                    raise ValueError("Nonfinite TACO object annotation: " + name)
        for oid, seqs in used.items():
            objects.append(
                {
                    "source": "taco",
                    "object_id": oid,
                    "mesh_archive": "data/raw/taco/Object_Models.zip",
                    "mesh_member": meshes.get(oid),
                    "pose_sequences": len(set(seqs)),
                    "units": "source mesh filenames use _cm; canonical adapter not yet applied",
                }
            )
        for sid, r in records.items():
            sequences.append(
                {
                    "source": "taco",
                    "sequence": sid,
                    "frames": r["frames"],
                    "objects": sorted(r["objects"]),
                    "duration_s": None,
                    "state_content_sha256": r["hash"].hexdigest(),
                    "note": "Unique sequence folders with actual numeric object poses; frame rate not assumed. Fingerprints cover linked object states, not human joint arrays.",
                }
            )
    for sid, pose_dir, mesh_dir in [
        ("hodome", "object", "scaned_object"),
        ("hoim3", "object_with_distortion", "scanned_object"),
    ]:
        root = store.root / "data/raw" / sid
        available = []
        for path in (root / mesh_dir).glob("*"):
            if path.suffix == ".tar":
                available.append(path.stem)
            elif path.is_dir() and any(path.rglob("*.obj")):
                available.append(path.name)
        if sid == "hodome":
            for oid in sorted(available):
                refs = list((root / pose_dir).glob("subject*_" + oid + ".npz"))
                if refs:
                    objects.append(
                        {
                            "source": sid,
                            "object_id": oid,
                            "mesh_container": str(
                                (root / mesh_dir / (oid + ".tar")).relative_to(
                                    store.root
                                )
                            ),
                            "pose_sequences": len(refs),
                            "units": "raw source; canonical adapter pending",
                        }
                    )
        else:
            # Some room arrays map multiple object IDs inside a pickled dict.
            # Geometry count alone is not promoted into an observed-object count.
            pass
    # ParaHome links declared scene objects to acquired scans and transformations.
    archive = store.root / "data/raw/parahome/seq.zip"
    seen = {}
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            names = set(z.namelist())
            for name in names:
                if not name.endswith("/object_in_scene.json"):
                    continue
                sequence = name.rsplit("/", 1)[0]
                if sequence + "/object_transformations.pkl" not in names:
                    continue
                for oid in json.loads(z.read(name)):
                    seen.setdefault(oid, set()).add(sequence)
        for oid, seqs in sorted(seen.items()):
            mesh = store.root / "data/assets/parahome/scan" / oid
            objects.append(
                {
                    "source": "parahome",
                    "object_id": oid,
                    "mesh_directory": str(mesh.relative_to(store.root))
                    if mesh.exists()
                    else None,
                    "pose_sequences": len(seqs),
                    "units": "m; representative transforms verified by canonical adapter",
                }
            )
    # HO-3D object names are the same explicit YCB model identifiers in the
    # official DexYCB model archive. Preserve source poses/contact as raw labels;
    # this establishes an asset link, not a coordinate conversion or physics pass.
    ho3d = store.root / "data/raw/hf__Ronaldo-GOAT__ho3d_transfer/HO3D_v3.zip.state.zip"
    ycb = store.root / "data/raw/dexycb/models.tar.gz"
    if ho3d.exists() and ycb.exists():
        from .human import array_pickle

        mesh_index = store.root / "catalog/evidence/dexycb-model-members.json"
        if mesh_index.exists():
            meshes = json.loads(mesh_index.read_text())
        else:
            meshes = {}
            with tarfile.open(ycb, "r:gz") as t:
                for member in t:
                    if member.name.endswith("/textured.obj"):
                        meshes[Path(member.name).parent.name] = member.name
            json_write(mesh_index, meshes)
        seen = {}
        inspected = set()
        with zipfile.ZipFile(ho3d) as z:
            for name in sorted(z.namelist()):
                if not name.endswith(".pkl") or "/meta/" not in name:
                    continue
                sequence = name.split("/meta/")[0]
                if sequence in inspected:
                    continue
                inspected.add(sequence)
                d = array_pickle(z.read(name))
                oid = d.get("objName")
                if oid and all(
                    k in d and np.isfinite(d[k]).all() for k in ("objRot", "objTrans")
                ):
                    seen.setdefault(oid, []).append(name)
        for oid, refs in sorted(seen.items()):
            if oid in meshes:
                objects.append(
                    {
                        "source": "ho3d",
                        "object_id": oid,
                        "mesh_archive": str(ycb.relative_to(store.root)),
                        "mesh_member": meshes[oid],
                        "pose_archive": str(ho3d.relative_to(store.root)),
                        "verified_pose_members": refs,
                        "pose_sequences": len(refs),
                        "units": "Raw axis-angle and translation retained; canonical HO-3D adapter pending.",
                        "link_evidence": "Exact YCB object-name identifier shared by acquired HO-3D metadata and official DexYCB textured.obj path.",
                    }
                )
    result = {
        "source_scoped_objects_with_pose_links": len(objects),
        "objects": objects,
        "additional_verified_sequence_folders": sequences,
        "scope": "TACO numeric pose arrays, HODome named pose/scan pairs, ParaHome scene-object/transform pairs, and HO-3D pose/YCB mesh-name matches. Source IDs are not asserted to identify unique physical objects across corpora. Multipart meshes and mirror variants are not separate objects.",
        "all_linked_geometry_available": all(
            x.get("mesh_member") or x.get("mesh_container") or x.get("mesh_directory")
            for x in objects
        ),
    }
    json_write(store.root / "catalog/object_inventory.json", result)
    return result
