"""Explicit, bounded discovery/fetch of one object-interaction native sample.

No source simulator, auto-downloader or robot SDK is imported. Reading schemas
does not establish a successful source replay or Reachy physical validation.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tarfile
from urllib.parse import quote
import zipfile

import numpy as np

from .acquire import RESERVE, StorageFull
from .disk import available
from .extract import open_remote
from .store import Store, json_write, sha256, now


SOURCES = {
    "robocasa": {
        "url": "https://utexas.box.com/shared/static/myhvbjtfii7at6itjepbivcuo1y8krbp.tar",
        "kind": "tar", "task": "PickPlaceCounterToCabinet",
        "release": "pretrain/atomic/PickPlaceCounterToCabinet/20250819/lerobot.tar",
        "code_revision": "456174f62b89b8fca99eaaf33949c29fec9cfc2a",
        "registry_url": "https://raw.githubusercontent.com/robocasa/robocasa/456174f62b89b8fca99eaaf33949c29fec9cfc2a/robocasa/models/assets/box_links/box_links_ds.json",
        "registry_sha256": "f6544237349efdf2961657200e817f68c3d34139350c90559611ddd926451c8f",
    },
    "bigym": {
        "url": "https://github.com/NeuracoreAI/bigym_data/releases/download/v0.9.0/demonstrations.zip",
        "kind": "zip", "task": "MovePlate", "release": "v0.9.0",
        "code_revision": "14beb30318ad14c5d6723175c2ee2281129792af",
        "registry_url": "https://github.com/NeuracoreAI/bigym_data/releases/tag/v0.9.0",
    },
}


def safe_member(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe archive member path")
    return path


def selected_members(source, entries):
    for entry in entries:
        safe_member(entry["path"])
    if source == "bigym":
        found = [e for e in entries if e["path"].startswith("MovePlate/")
                 and "/lightweight/" in e["path"] and "_absolute/" in e["path"]
                 and e["path"].endswith(".safetensors")]
        return sorted(found,key=lambda e:e["path"])[:1]
    return [e for e in entries if (
        "episode_000000" in e["path"] and "/videos/" not in e["path"]
        or "/meta/" in e["path"] and e["path"].endswith((".json", ".jsonl"))
        or e["path"].endswith("/extras/dataset_meta.json"))]


def discover(root, source):
    store = Store(Path(root).resolve());spec=SOURCES[source]
    entries=[]
    with open_remote(spec["url"],block_size=128*1024,maxblocks=4) as remote:
        archive_size=remote.size
        archive=zipfile.ZipFile(remote) if spec["kind"]=="zip" else tarfile.open(fileobj=remote,mode="r:")
        with archive:
            for item in archive.infolist() if spec["kind"]=="zip" else archive:
                if (item.is_dir() if spec["kind"]=="zip" else not item.isfile()):
                    continue
                entries.append({"path":item.filename if spec["kind"]=="zip" else item.name,
                                "size":item.file_size if spec["kind"]=="zip" else item.size})
    chosen=selected_members(source,entries)
    if not chosen:
        raise ValueError("Archive does not contain the declared object-interaction selection")
    manifest={"created":now(),**spec,"archive_bytes":archive_size,"members":chosen,
              "selected_bytes":sum(x["size"] for x in chosen),"original_capture_count":1,
              "whole_archive_sha256":None,"whole_archive_checksum_verified":False,
              "status":"discovered_not_downloaded"}
    path=store.root/"catalog"/(source+"-pilot.json")
    json_write(path,manifest)
    store.source(source+"_native_pilot","simulation",spec["url"],"discovered",**{k:v for k,v in spec.items() if k != "url"})
    return manifest


def fetch(root, source):
    store=Store(Path(root).resolve());sid=source+"_native_pilot"
    path=store.root/"catalog"/(source+"-pilot.json")
    manifest=json.loads(path.read_text())
    if manifest["selected_bytes"] > 100_000_000:
        raise ValueError("Pilot selection exceeds the explicit 100 MB extraction limit")
    if available(store.root)-manifest["selected_bytes"] < RESERVE:
        raise StorageFull("50 GB reserve before native pilot extraction")
    records=[]
    with open_remote(manifest["url"],block_size=128*1024,maxblocks=4) as remote:
        archive=zipfile.ZipFile(remote) if manifest["kind"]=="zip" else tarfile.open(fileobj=remote,mode="r:")
        with archive:
            for entry in manifest["members"]:
                relative=safe_member(entry["path"])
                target=store.root/"data/raw"/sid/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    if available(store.root)-entry["size"] < RESERVE:
                        raise StorageFull("50 GB reserve during native pilot extraction")
                    reader=archive.open(entry["path"]) if manifest["kind"]=="zip" else archive.extractfile(entry["path"])
                    with reader:
                        payload=reader.read(entry["size"]+1)
                    if len(payload)!=entry["size"]:
                        raise ValueError("Archive entry length changed")
                    target.write_bytes(payload)
                if target.stat().st_size!=entry["size"]:
                    raise ValueError("Cached source member size mismatch")
                digest=sha256(target)
                url=manifest["url"]+"#member="+quote(entry["path"],safe="/")
                store.file(sid,entry["path"],url,entry["size"])
                store.update_file(sid,entry["path"],status="downloaded",bytes=entry["size"],sha256=digest)
                records.append({**entry,"sha256":digest,"source_url":url})
    manifest.update(status="selected_raw_members_downloaded",downloaded=records,finished=now(),
                    retargeted=False,physics_validated=False)
    json_write(path,manifest)
    store.source(sid,"simulation",manifest["url"],"downloaded",selection_manifest=str(path))
    return inspect(root,source)


def inspect(root,source):
    store=Store(Path(root).resolve());folder=store.root/"data/raw"/(source+"_native_pilot")
    report={"source":source,"created":now(),"files":[],"retargeted":False,"physics_validated":False}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():continue
        record={"path":str(path.relative_to(folder)),"bytes":path.stat().st_size,"sha256":sha256(path)}
        if path.suffix==".npz":
            with np.load(path,allow_pickle=False) as arrays:
                record["arrays"]={k:{"shape":list(arrays[k].shape),"dtype":str(arrays[k].dtype)} for k in arrays.files}
        elif path.suffix==".parquet":
            import pyarrow.parquet as pq
            table=pq.read_table(path)
            record.update(rows=table.num_rows,columns=table.column_names)
        elif path.name.endswith(".xml.gz"):
            import xml.etree.ElementTree as ET
            xml=gzip.decompress(path.read_bytes());tree=ET.fromstring(xml)
            record.update(xml_sha256=hashlib.sha256(xml).hexdigest(),
                          joints=[j.attrib for j in tree.findall(".//worldbody//joint")],
                          freejoints=[j.attrib for j in tree.findall(".//worldbody//freejoint")],
                          external_assets=[a.attrib for a in tree.findall("./asset/*") if a.get("file")])
        elif path.suffix==".safetensors":
            raw=path.read_bytes();header_size=struct.unpack("<Q",raw[:8])[0]
            if header_size>len(raw)-8:raise ValueError("Invalid safetensors header size")
            header=json.loads(raw[8:8+header_size]);record["header"]=header
        report["files"].append(record)
    report["missing_for_reachy_physics"]=(
        ["referenced source assets", "source replay and state-to-body verification", "task adapter and Reachy rollout"]
        if source=="robocasa" else
        ["pinned source simulator/assets", "successful source action replay", "reconstructed object/base states", "task adapter and Reachy rollout"])
    json_write(store.root/"catalog"/(source+"-pilot-inspection.json"),report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("discover","fetch","inspect"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--source",choices=tuple(SOURCES),required=True)
    args=parser.parse_args()
    result=globals()[args.operation](args.root,args.source)
    print(json.dumps({k:v for k,v in result.items() if k not in {"files","members","downloaded"}},indent=2))


if __name__=="__main__":
    main()
