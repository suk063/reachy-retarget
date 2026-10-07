"""Stream TFDS/RLDS records to state-only HDF5 without storing pixel payloads."""

import base64
import json
import os
import shutil
import struct
from pathlib import Path
import numpy as np
import h5py
import requests
import google_crc32c
from .store import json_write, sha256
from .disk import available


def masked_crc(value):
    crc = google_crc32c.value(value)
    return ((crc >> 15) | (crc << 17) & 0xFFFFFFFF) + 0xA282EAD8 & 0xFFFFFFFF


def extract(url, target, reserve=50_000_000_000):
    # Import only the TensorFlow wire schema, not a dataset pipeline or GPU job.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow.core.example.example_pb2 import Example

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target.stat().st_size, sha256(target)
    part = target.with_name(target.name + ".part")
    included = set()
    excluded = set()
    with h5py.File(part, "a") as dst:
        if dst.attrs.get("source_url", url) != url:
            raise ValueError("RLDS resume source mismatch")
        offset = int(dst.attrs.get("input_offset", 0))
        episode = int(dst.attrs.get("episodes", 0))
        dst.attrs["source_url"] = url
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with requests.get(
            url, headers=headers, stream=True, timeout=(30, 180)
        ) as response:
            response.raise_for_status()
            if offset and (
                response.status_code != 206
                or not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                )
            ):
                raise ValueError("RLDS resume offset not honored")
            response.raw.decode_content = False
            while True:
                length = response.raw.read(8)
                if not length:
                    break
                if len(length) != 8:
                    raise ValueError("Truncated TFRecord length")
                length_crc = response.raw.read(4)
                if len(length_crc) != 4 or struct.unpack("<I", length_crc)[
                    0
                ] != masked_crc(length):
                    raise ValueError("TFRecord length CRC32C mismatch")
                n = struct.unpack("<Q", length)[0]
                if n > 4_000_000_000:
                    raise ValueError(
                        "TFRecord exceeds bounded decoder memory; use range-aware adapter"
                    )
                data = response.raw.read(n)
                data_crc = response.raw.read(4)
                if (
                    len(data) != n
                    or len(data_crc) != 4
                    or struct.unpack("<I", data_crc)[0] != masked_crc(data)
                ):
                    raise ValueError("Truncated or corrupt TFRecord payload")
                example = Example.FromString(data)
                del data
                name = f"data/demo_{episode}"
                if name in dst:
                    del dst[name]
                group = dst.create_group(name)
                channels = 0
                for key, feature in example.features.feature.items():
                    low = key.lower()
                    if any(
                        w in low
                        for w in (
                            "image",
                            "rgb",
                            "depth",
                            "segmentation",
                            "video",
                            "pointcloud",
                            "point_cloud",
                        )
                    ):
                        excluded.add(key)
                        continue
                    kind = feature.WhichOneof("kind")
                    if kind == "float_list":
                        value = np.asarray(feature.float_list.value, dtype=np.float32)
                    elif kind == "int64_list":
                        value = np.asarray(feature.int64_list.value, dtype=np.int64)
                    elif kind == "bytes_list":
                        try:
                            strings = [
                                bytes(v).decode("utf-8")
                                for v in feature.bytes_list.value
                            ]
                        except UnicodeDecodeError:
                            excluded.add(key)
                            continue
                        if any("\x00" in v for v in strings):
                            excluded.add(key)
                            continue
                        value = np.asarray(strings, dtype=h5py.string_dtype())
                    else:
                        continue
                    if available(target.parent) - value.nbytes - 1048576 < reserve:
                        raise RuntimeError("50 GB reserve during RLDS extraction")
                    group.create_dataset(
                        key,
                        data=value,
                        compression="gzip"
                        if value.dtype.kind != "O" and value.size
                        else None,
                    )
                    included.add(key)
                    channels += 1
                group.attrs["source_record_index"] = episode
                group.attrs["source_wire_shapes"] = (
                    "TFDS flattened feature vectors; reconstruct using companion features.json; no guessed units"
                )
                group.attrs["source_record_crc32c_verified"] = True
                offset += n + 16
                episode += 1
                dst.attrs["input_offset"] = offset
                dst.attrs["episodes"] = episode
                dst.flush()
                if episode % 25 == 0:
                    print(target.name, "RLDS records", episode, flush=True)
    part.replace(target)
    manifest = {
        "source_url": url,
        "episodes": episode,
        "input_bytes_streamed": offset,
        "included_channels": sorted(included),
        "excluded_channels": sorted(excluded),
        "method": "TFRecord HTTP streaming, per-record CRC32C verified, numeric and UTF-8 metadata retained; pixels never written to disk",
        "scope": "One explicitly selected source shard; not the complete parent corpus",
        "sha256": sha256(target),
        "bytes": target.stat().st_size,
    }
    json_write(target.with_suffix(".extraction.json"), manifest)
    return manifest["bytes"], manifest["sha256"]
