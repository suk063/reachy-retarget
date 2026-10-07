"""Selective, resumable HDF5 extraction over HTTP Range; never retain pixels."""

import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
import fsspec
import h5py
import numpy as np
import requests
from .store import sha256, json_write
from .disk import available

PIXELS = ("image", "rgb", "depth", "segmentation", "video", "pointcloud", "point_cloud")


@contextmanager
def open_remote(url, block_size=2 * 1024 * 1024, maxblocks=32):
    """Resolve once, then issue byte ranges directly to the returned CDN.

    Signed URLs stay in memory. Provenance records the stable public URL only.
    Supplying the observed total size prevents fsspec from resolving the original
    hub URL again for every dataset/member range.
    """
    with requests.get(
        url,
        headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(30, 120),
    ) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", "").lower():
            raise ValueError(
                "HTML access/confirmation page instead of the selected state archive; no form or account flow submitted"
            )
        resolved = response.url
        content_range = response.headers.get("Content-Range", "")
        size = (
            int(content_range.rsplit("/", 1)[1])
            if "/" in content_range
            else int(response.headers.get("Content-Length", 0))
        )
        if not size:
            raise ValueError("Remote range source has no verifiable content length")
    fs = fsspec.filesystem("http")
    with fs.open(
        resolved,
        "rb",
        size=size,
        block_size=block_size,
        cache_type="blockcache",
        cache_options={"maxblocks": maxblocks},
    ) as stream:
        yield stream


def is_state(name, obj):
    if any(word in name.lower() for word in PIXELS):
        return False
    return not (obj.ndim >= 4 and obj.dtype.kind in "buif")


def hdf5_state(url, target, reserve=50_000_000_000):
    """Resume at dataset boundaries. Remote full-file hash is not claimed verified.

    Store an exact inventory of included/excluded datasets and per-dataset hashes.
    Both source revision and original LFS digest remain in the acquisition ledger.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if target.exists():
        return target.stat().st_size, sha256(target)
    included = []
    excluded = []
    with open_remote(url) as remote:
        with h5py.File(remote, "r") as src, h5py.File(part, "a") as dst:
            if dst.attrs.get("extraction_source_url", url) != url:
                raise ValueError("Resume source revision changed")
            dst.attrs["extraction_source_url"] = url

            def attrs(a, b):
                for k, v in a.attrs.items():
                    b.attrs[k] = v

            attrs(src, dst)

            def copy(name, obj):
                if isinstance(obj, h5py.Group):
                    attrs(obj, dst.require_group(name))
                    return
                if not is_state(name, obj):
                    excluded.append(
                        {"path": name, "shape": obj.shape, "dtype": str(obj.dtype)}
                    )
                    return
                estimated = max(
                    1024, int(np.prod(obj.shape)) * max(obj.dtype.itemsize, 8)
                )
                if available(target.parent) - estimated < reserve:
                    raise RuntimeError("50 GB reserve during selective extraction")
                # A dataset is committed after its entire source content is read.
                if name in dst and dst[name].attrs.get("_extraction_complete", False):
                    included.append(name)
                    return
                if name in dst:
                    del dst[name]
                value = obj[()]
                d = dst.create_dataset(
                    name,
                    data=value,
                    compression="gzip"
                    if obj.ndim and obj.size > 10 and obj.dtype.kind not in "OSU"
                    else None,
                )
                attrs(obj, d)
                d.attrs["_extraction_complete"] = True
                included.append(name)
                if len(included) % 100 == 0:
                    dst.flush()

            src.visititems(copy)
            dst.attrs["extraction_scope"] = (
                "numeric state and metadata; all named pixel/depth/video arrays excluded"
            )
            dst.attrs["source_full_file_sha256_verified"] = False
            dst.flush()
    part.replace(target)
    result = {
        "source_url": url,
        "included_datasets": included,
        "excluded_datasets": excluded,
        "sha256": sha256(target),
        "bytes": target.stat().st_size,
        "method": "HTTP Range HDF5 dataset reads; original server full file was not downloaded or hashed",
        "source_integrity": "Pinned source revision; local extracted payload SHA256; no claim of remote full-file verification",
    }
    json_write(target.with_suffix(target.suffix + ".extraction.json"), result)
    return result["bytes"], result["sha256"]


def parquet_state(url, target, local=None, reserve=50_000_000_000):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import contextlib

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target.stat().st_size, sha256(target)
    with (
        contextlib.nullcontext(str(local))
        if local
        else open_remote(url, block_size=65536)
    ) as src:
        f = pq.ParquetFile(src)
        columns = [
            x.name
            for x in f.schema_arrow
            if not (pa.types.is_struct(x.type) and "bytes" in str(x.type))
            and not any(w in x.name.lower() for w in PIXELS)
        ]
        if not any(
            any(
                w in c.lower()
                for w in (
                    "state",
                    "action",
                    "pose",
                    "position",
                    "motion",
                    "joint",
                    "qpos",
                    "dof_pos",
                    "object_pos",
                    "root_pos",
                )
            )
            for c in columns
        ):
            raise ValueError(
                "No interpretable state channel in Parquet; media-only source"
            )
        table = f.read(columns=columns)
        if available(target.parent) - table.nbytes < reserve:
            raise RuntimeError("50 GB reserve during selective extraction")
        part = target.with_name(target.name + ".part")
        pq.write_table(table, part, compression="zstd")
        part.replace(target)
        json_write(
            target.with_suffix(".extraction.json"),
            {
                "source_url": url,
                "source_rows": f.metadata.num_rows,
                "columns": columns,
                "excluded_columns": [
                    x for x in f.schema_arrow.names if x not in columns
                ],
                "sha256": sha256(target),
                "method": "Parquet column projection, remote HTTP Range or locally already acquired payload",
            },
        )
    return target.stat().st_size, sha256(target)


def archive_state(url, target, kind="zip", reserve=50_000_000_000, local=None):
    """Copy only state/mesh members into a compact ZIP, preserving member names."""
    import tarfile, zipfile, io, contextlib

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target.stat().st_size, sha256(target)
    part = target.with_name(target.name + ".part")
    included = []
    excluded_count = 0

    def selected(name):
        p = Path(name)
        low = name.lower()
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("Unsafe archive member path")
        return p.suffix.lower() in (
            ".pkl",
            ".npz",
            ".npy",
            ".json",
            ".jsonl",
            ".yaml",
            ".yml",
            ".txt",
            ".obj",
            ".ply",
            ".stl",
            ".xml",
            ".urdf",
            ".mtl",
            ".bvh",
            ".amc",
            ".asf",
            ".csv",
        ) and not any(
            w in low
            for w in ("/rgb/", "/images/", "/depth/", "/segmentation/", "hand_crops")
        )

    with Path(local).open("rb") if local else open_remote(url, maxblocks=8) as remote:
        source = (
            zipfile.ZipFile(remote)
            if kind == "zip"
            else tarfile.open(fileobj=remote, mode="r|gz" if kind == "tgz" else "r:")
        )
        with source, contextlib.ExitStack() as stack:

            def destination():
                return stack.enter_context(
                    zipfile.ZipFile(
                        part, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=3
                    )
                )

            dst = destination()
            done = set(dst.namelist())
            members = (
                sorted(source.infolist(), key=lambda x: x.header_offset)
                if kind == "zip"
                else source
            )
            for member in members:
                name = member.filename if kind == "zip" else member.name
                regular = not member.is_dir() if kind == "zip" else member.isfile()
                if not regular or not selected(name):
                    excluded_count += 1
                    continue
                included.append(name)
                if name in done:
                    continue
                size = member.file_size if kind == "zip" else member.size
                if available(target.parent) - size < reserve:
                    raise RuntimeError("50 GB reserve during archive extraction")
                reader = (
                    source.open(member) if kind == "zip" else source.extractfile(member)
                )
                if name.lower().endswith(".npz"):
                    with reader:
                        payload = reader.read()
                    # NPZ frames may bundle pixels and states. Filter inner NPY
                    # members without executing pickles or changing numeric data.
                    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                        result = io.BytesIO()
                        with zipfile.ZipFile(
                            result, "w", compression=zipfile.ZIP_DEFLATED
                        ) as filtered:
                            for entry in inner.infolist():
                                if any(
                                    w in entry.filename.lower() for w in PIXELS
                                ) or Path(entry.filename).stem.lower() in (
                                    "seg",
                                    "mask",
                                    "masks",
                                ):
                                    excluded_count += 1
                                    continue
                                p = Path(entry.filename)
                                if p.is_absolute() or ".." in p.parts:
                                    raise ValueError("Unsafe nested NPZ member path")
                                filtered.writestr(entry.filename, inner.read(entry))
                        dst.writestr(name, result.getvalue())
                else:
                    with reader, dst.open(name, "w", force_zip64=True) as out:
                        shutil.copyfileobj(reader, out, 1024 * 1024)
                done.add(name)
                if len(done) % 500 == 0:
                    # Commit the central directory so an interrupted process can
                    # resume at a verified member boundary.
                    dst.close()
                    dst = destination()
                    print(target.name, "state members", len(done), flush=True)
    part.replace(target)
    json_write(
        target.with_suffix(".extraction.json"),
        {
            "source_url": url,
            "included_members": included,
            "excluded_members_count": excluded_count,
            "sha256": sha256(target),
            "method": f"HTTP Range {kind} state/geometry member selection; pixel members omitted",
            "source_full_file_sha256_verified": False,
        },
    )
    return target.stat().st_size, sha256(target)
