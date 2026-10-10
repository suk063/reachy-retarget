"""Fetch job manifests: one JSONL line per cluster job, batching catalog entries by size.

Each line is ``{"id", "argv", "publish": "data", "timeout_s"}`` with ``argv`` =
``-m reachy_retarget.cli fetch <selectors> --root {out} --strip-images [--workers N]``; the
job fetches into its private ``{out}`` and :mod:`cluster.job` merges it no-overwrite into
the shared data root. Selectors are catalog ids or directory prefixes ending in ``/``
(a prefix is used only when every usable catalog entry below it is part of the job), so
argv stays short for families with thousands of small files.

Batching: entries are grouped by catalog path, groups are split until each fits
``max_bytes`` of transfer, and consecutive groups are packed into jobs that move about
``target_bytes`` (1-5 GB) and never need more than ``scratch_bytes`` (40 GB) of local
scratch (upper bound of what the job writes: file size, non-image tar members <= tar size,
inflated package size). A single entry larger than the transfer target is its own job.
Job ids are derived from the selectors (``fetch-<family>-<first selector>-<hash>``), so the
same catalog gives the same ids.

    python -m cluster.manifests behavior --out runs/fetch-w2-behavior-v1.jsonl

Build manifests (see the "build manifests" section below) are planned from PVC file listings:

    python -m cluster.manifests build robocasa --listing runs/listings/pvc-raw-robocasa.txt \
        --listing runs/listings/pvc-raw-robosuite.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reachy_retarget.acquire import CatalogEntry, load_catalog  # noqa: E402

GB = 1_000_000_000
FAMILIES = {  # family: catalog prefixes of the full-scale fetch
    "behavior": ["behavior/2025-challenge-demos/", "behavior/omnigibson-robot-assets/"],
    "mobilemanibench": ["mobilemanibench/"],
    "robocasa": ["robocasa/"],
    "roboverse": ["roboverse/"],
    "molmobot": ["molmobot/"],
}


def transfer_bytes(e: CatalogEntry) -> int:
    """Bytes moved over the network to acquire ``e``."""
    s = e.source
    if e.transport == "range_member":
        return (s.get("stored_size") or e.size or 0) + 512
    if e.transport == "range_package":
        return s["stored_size"]
    return e.size or 0


def scratch_bytes(e: CatalogEntry) -> int:
    """Upper bound of the bytes written to the job's private root for ``e``."""
    if e.transport == "range_package":
        return e.source.get("inflated_size") or 4 * e.source["stored_size"]
    if e.transport == "file_images_embedded":
        return 2 * (e.size or 0)  # original + state-only copy until the original is deleted
    return e.size or 0


def _units(entries, catalog, max_bytes):
    """``[(selector, [entries])]``: directory prefixes whose usable entries are all selected
    and fit ``max_bytes``, else smaller prefixes, else single ids; in id order."""
    chosen = {e.id for e in entries}
    usable_below = {}
    for e in catalog.values():
        if e.usable:
            parts = e.id.split("/")
            for i in range(1, len(parts)):
                usable_below.setdefault("/".join(parts[:i]) + "/", []).append(e.id)

    def split(prefix, members):
        total = sum(transfer_bytes(e) for e in members)
        complete = all(i in chosen for i in usable_below.get(prefix, ()))
        if prefix and complete and total <= max_bytes and len(members) > 1:
            return [(prefix, members)]
        children, leaves = {}, []
        depth = prefix.count("/")
        for e in members:
            parts = e.id.split("/")
            if len(parts) > depth + 1:
                children.setdefault("/".join(parts[:depth + 1]) + "/", []).append(e)
            else:
                leaves.append(e)
        out = [(e.id, [e]) for e in leaves]
        for child, sub in children.items():
            out.extend(split(child, sub))
        return sorted(out, key=lambda u: u[0])

    return split("", sorted(entries, key=lambda e: e.id))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def plan(family: str, *, catalog=None, prefixes=None, target_bytes=3 * GB, max_bytes=5 * GB,
         scratch_limit=40 * GB, max_selectors=200, skip=()) -> list[dict]:
    """Jobs (manifest lines plus ``bytes``/``scratch``/``entries`` stats) for ``family``."""
    catalog = catalog or load_catalog()
    prefixes = prefixes or FAMILIES[family]
    skip = set(skip)
    entries = [e for e in catalog.values() if e.usable and e.id not in skip
               and any(e.id.startswith(p) for p in prefixes)]
    units = _units(entries, catalog, max_bytes)
    jobs, cur = [], []

    def close():
        if cur:
            jobs.append(list(cur))
            cur.clear()

    for unit in units:
        t = sum(transfer_bytes(e) for e in unit[1])
        s = sum(scratch_bytes(e) for e in unit[1])
        ct = sum(transfer_bytes(e) for u in cur for e in u[1])
        cs = sum(scratch_bytes(e) for u in cur for e in u[1])
        if cur and (ct + t > max_bytes or (ct >= target_bytes) or cs + s > scratch_limit or len(cur) >= max_selectors):
            close()
        cur.append(unit)
    close()
    if len(jobs) > 1:  # a small remainder joins the previous job when both fit the limits
        last, prev = jobs[-1], jobs[-2]
        size = lambda js, f: sum(f(e) for u in js for e in u[1])  # noqa: E731
        if size(last, transfer_bytes) < GB and size(prev + last, transfer_bytes) <= max_bytes \
                and size(prev + last, scratch_bytes) <= scratch_limit and len(prev + last) <= max_selectors:
            jobs[-2:] = [prev + last]
    out = []
    for units_ in jobs:
        sel = [u[0] for u in units_]
        ents = [e for u in units_ for e in u[1]]
        t = sum(transfer_bytes(e) for e in ents)
        s = sum(scratch_bytes(e) for e in ents)
        if s > scratch_limit and len(ents) > 1:
            raise ValueError(f"job {sel[0]} needs {s} bytes of scratch")
        requests = sum(1 for e in ents if e.transport in ("range_member", "file", "file_images_embedded"))
        workers = 8 if requests > 50 else 1
        timeout = int(min(12 * 3600, max(3600, 1800 + t / 10e6 + requests * 1.0 / workers)))
        digest = hashlib.sha1("\n".join(sel).encode()).hexdigest()[:8]
        job_id = f"fetch-{family}-{_slug(sel[0].split('/', 1)[1] if '/' in sel[0] else sel[0])}-{digest}"
        argv = ["-m", "reachy_retarget.cli", "fetch", *sel, "--root", "{out}", "--strip-images"]
        if workers > 1:
            argv += ["--workers", str(workers)]
        out.append({"id": job_id, "argv": argv, "publish": "data", "timeout_s": timeout,
                    "_stats": {"entries": len(ents), "transfer_bytes": t, "scratch_bytes": s}})
    ids = [j["id"] for j in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate job ids")
    return out


def write_manifest(family: str, path, **kw) -> dict:
    """Write the manifest of ``family`` to ``path`` (JSONL); return a summary."""
    jobs = plan(family, **kw)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps({k: v for k, v in j.items() if not k.startswith("_")}) + "\n")
    t = [j["_stats"]["transfer_bytes"] for j in jobs]
    s = [j["_stats"]["scratch_bytes"] for j in jobs]
    return {"family": family, "jobs": len(jobs), "entries": sum(j["_stats"]["entries"] for j in jobs),
            "transfer_gb": round(sum(t) / GB, 2), "scratch_gb_upper": round(sum(s) / GB, 2),
            "job_transfer_gb": [round(min(t) / GB, 3), round(sorted(t)[len(t) // 2] / GB, 3), round(max(t) / GB, 3)],
            "job_scratch_gb_max": round(max(s) / GB, 2)}


def build_main(argv):
    ap = argparse.ArgumentParser(prog="python -m cluster.manifests build",
                                 description="Write build-<wave>-<family>-<version>.jsonl from a PVC file listing.")
    ap.add_argument("family", choices=BUILD_FAMILIES)
    ap.add_argument("--listing", required=True, action="append",
                    help="file listing (<size> <path> per line); repeatable (e.g. raw/robocasa + raw/robosuite)")
    ap.add_argument("--out", help="manifest path (default runs/build-<wave>-<family>-<version>.jsonl)")
    ap.add_argument("--wave", default="w2")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--target-episodes", type=int, default=30, help="episodes per job (default 30 = 12.5 min)")
    ap.add_argument("--max-episodes", type=int, default=36, help="largest packed/sharded job (default 36 = 15 min)")
    ap.add_argument("--seconds-per-episode", type=float, default=SECONDS_PER_EPISODE)
    a = ap.parse_args(argv)
    print(json.dumps(write_build_manifest(a.family, a.listing, a.out, wave=a.wave, version=a.version,
                                          target_episodes=a.target_episodes, max_episodes=a.max_episodes,
                                          seconds_per_episode=a.seconds_per_episode), indent=1))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["build"]:
        return build_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("families", nargs="+", choices=sorted(FAMILIES))
    ap.add_argument("--out", help="manifest path (one family) or a directory (default runs/)")
    ap.add_argument("--wave", default="w2", help="manifest name: fetch-<wave>-<family>-<version>.jsonl")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--skip-root", help="data root whose ledger records mark entries as already fetched")
    a = ap.parse_args(argv)
    skip = ()
    if a.skip_root:
        from reachy_retarget.acquire import read_ledger
        skip = set(read_ledger(a.skip_root))
    for fam in a.families:
        out = Path(a.out) if a.out and a.out.endswith(".jsonl") else Path(a.out or ROOT / "runs") \
            / f"fetch-{a.wave}-{fam}-{a.version}.jsonl"
        print(json.dumps({**write_manifest(fam, out, skip=skip), "path": str(out)}))



# ================================================================ build manifests
#
# ``python -m cluster.manifests build <family> --listing <file>`` plans the build jobs of one
# family from a listing of the files on the PVC (``<size> <path>`` per line, as written by
# ``kubectl exec <pod> -- find /mnt/reachy-retarget/data/raw/<family> -type f -printf '%s %p\n'``;
# paths may be absolute, relative to the data root (``raw/...``, ``derived/...``) or relative to
# ``raw/``). Each job runs ``python -m reachy_retarget.build --family F --path P [--path P2 ...]
# --shard k/n [--physics] --out {out}`` and publishes to ``datasets/<wave>-<version>``.
#
# Input units (the ``--path`` an adapter reads) per family:
#
# * behavior: ``raw/behavior/2025-challenge-demos/data/task-XXXX`` (one parquet per episode);
# * mobilemanibench: ``.../<object>/train_N`` folders (one ``state_infos.pkl`` per episode);
# * robocasa: the extracted dataset folder holding ``lerobot/`` (one ``extras/episode_*`` each);
# * roboverse: one trajectory pickle (RLBench: 10 demos, ``close_box`` 100; CALVIN: windows,
#   estimated from the file size with :data:`CALVIN_BYTES_PER_EPISODE`);
# * molmobot: one ``trajectories_batch_*.h5`` (trajectory count from the catalog package members);
# * bigym: one task folder of replay records (``derived/bigym/replay-v1/<Task>``; the clipped-
#   action replays ``replay-v1-clipped`` publish to a separate dataset tree because they reuse the
#   recording uuids as episode ids).
#
# Sizing: about ``target_episodes`` (30 = 12.5 min at 25 s/episode) per job, at most
# ``max_episodes`` (36 = 15 min). A larger unit is split with ``--shard k/n``
# (n = ceil(episodes / target)); smaller units are packed into one job with repeated ``--path``.
# Every adapter of these families honours ``select`` (other shards' episodes are not read).
# ``--physics`` only where the adapter can attach a SceneRef (robocasa, bigym, RoboVerse RLBench).

PVC_DATA = "/mnt/reachy-retarget/data"
SECONDS_PER_EPISODE = 25.0
CALVIN_BYTES_PER_EPISODE = 49_500  # env A: 389 files, 6,027 windows, 298.8 MB; D_val: 299, 1,087, 53.5 MB
ROBOCASA_ROBOSUITE = "1.5.2"  # robosuite_version of all 350 RoboCasa365 dataset_meta.json on the PVC
BUILD_FAMILIES = ("behavior", "mobilemanibench", "robocasa", "roboverse", "molmobot", "bigym")


def read_listing(paths) -> dict[str, int]:
    """``{data-root-relative path: size}`` from one or more listing files (``<size> <path>`` or
    ``<path>`` per line)."""
    files = {}
    paths = [paths] if isinstance(paths, (str, Path)) else paths
    lines = [line for p in paths for line in Path(p).read_text().splitlines()]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        size, _, rest = line.partition(" ")
        if size.isdigit() and rest:
            p, n = rest.strip(), int(size)
        else:
            p, n = line, 0
        files[normalize(p)] = n
    return files


def normalize(p: str) -> str:
    """A listing path relative to the data root."""
    p = p.strip()
    if p.startswith(PVC_DATA + "/"):
        return p[len(PVC_DATA) + 1:]
    if p.startswith("./"):
        p = p[2:]
    if p.startswith(("raw/", "derived/")):
        return p
    if p.startswith("/"):
        raise ValueError(f"{p}: absolute path outside {PVC_DATA}")
    return "raw/" + p


class Unit:
    """One adapter input path with its estimated episode count."""

    def __init__(self, path: str, episodes: int, *, physics: bool, publish_suffix: str = "", exact: bool = True):
        self.path, self.episodes, self.physics = path, episodes, physics
        self.publish_suffix, self.exact = publish_suffix, exact

    def __repr__(self):
        return f"Unit({self.path!r}, {self.episodes})"


def _group(files, pattern, key=None):
    """``{unit path: [matching files]}`` for files matching regex ``pattern`` (group 1 = unit)."""
    rx = re.compile(pattern)
    out = {}
    for f in sorted(files):
        m = rx.fullmatch(f)
        if m:
            out.setdefault(m.group(1), []).append(f)
    return out


def units_behavior(files, catalog=None):
    groups = _group(files, r"(raw/behavior/2025-challenge-demos/data/task-\d+)/episode_\d+\.parquet")
    return [Unit(u, len(fs), physics=False) for u, fs in groups.items()]


def units_mobilemanibench(files, catalog=None):
    groups = _group(files, r"(raw/mobilemanibench/.+?/train_\d+)/trajectories/traj_\d+/episode_\d+/state_infos\.pkl")
    return [Unit(u, len(fs), physics=False) for u, fs in groups.items()]


def units_robocasa(files, catalog=None):
    groups = _group(files, r"(raw/robocasa/.+?)/lerobot/extras/episode_[^/]+/states\.npz")
    return [Unit(u, len(fs), physics=True) for u, fs in groups.items()]


def units_roboverse(files, catalog=None):
    out = []
    for f in sorted(files):
        m = re.fullmatch(r"raw/roboverse/trajs/rlbench/([^/]+)/v2/franka_v2\.pkl\.gz", f)
        if m:
            out.append(Unit(f, 100 if m.group(1) == "close_box" else 10, physics=True))
        elif re.fullmatch(r"raw/roboverse/trajs/calvin/calvin_traj_ann/env_[A-D](_val)?_out/task_\d+_v2\.pkl", f):
            out.append(Unit(f, max(1, round(files[f] / CALVIN_BYTES_PER_EPISODE)), physics=False, exact=False))
    return out


def _molmobot_counts(catalog, files):
    """``{file: (trajectories, exact)}``: member counts of catalogued packages, else the package's
    ``est_trajectories`` split evenly over its listed h5 files (trust-on-first-use packages)."""
    by_pkg = {}
    for f in files:
        by_pkg.setdefault(posixpath.dirname(f), []).append(f)
    counts = {}
    for e in catalog.values():
        if e.family != "molmobot" or e.transport != "range_package":
            continue
        local = _local(e)
        members = e.source.get("members") or []
        for m in members:
            if m.get("trajectories"):
                counts[posixpath.join(posixpath.dirname(local), m["name"])] = (m["trajectories"], True)
        est = (e.meta or {}).get("est_trajectories")
        listed = by_pkg.get(local, [])
        for f in listed:
            if f not in counts and est:
                counts[f] = (max(1, round(est / len(listed))), False)
    return counts


def units_molmobot(files, catalog=None):
    h5 = [f for f in sorted(files) if re.fullmatch(r"raw/molmobot/.+/trajectories_batch_\d+_of_\d+\.h5", f)]
    counts = _molmobot_counts(catalog, h5) if catalog is not None else {}
    return [Unit(f, counts.get(f, (5, False))[0], physics=False, exact=counts.get(f, (5, False))[1]) for f in h5]


def units_bigym(files, catalog=None):
    groups = _group(files, r"(derived/bigym/replay-[A-Za-z0-9-]+/(?!assets/)[^/]+)/[0-9a-f]+\.npz")  # not *.partial-* staging
    return [Unit(u, len(fs), physics=True,
                 publish_suffix="-bigym-clipped" if "/replay-v1-clipped/" in u + "/" else "")
            for u, fs in groups.items()]


UNITS = {"behavior": units_behavior, "mobilemanibench": units_mobilemanibench, "robocasa": units_robocasa,
         "roboverse": units_roboverse, "molmobot": units_molmobot, "bigym": units_bigym}


def _exists(files, dirs, rel):
    return rel in files or rel.rstrip("/") in dirs


def check_inputs(family: str, files: dict, catalog=None) -> list[str]:
    """Problems with the extra inputs the adapter looks up under the data root (robot models,
    asset archives, metadata), plus catalogued usable entries that are not on the PVC."""
    dirs = {posixpath.dirname(f) for f in files}
    dirs |= {d for f in list(dirs) for d in _parents(f)}
    need, problems = [], []
    if family == "behavior":
        need = ["raw/behavior/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf",
                "raw/behavior/2025-challenge-demos/meta/tasks.jsonl"]
        for f in files:
            m = re.fullmatch(r"raw/behavior/2025-challenge-demos/data/(task-\d+)/(episode_\d+)\.parquet", f)
            if m and f"raw/behavior/2025-challenge-demos/meta/episodes/{m[1]}/{m[2]}.json" not in files:
                problems.append(f"{f}: episode metadata missing (the adapter raises; ends its shard)")
        tasks_meta = {m[1] for f in files if (m := re.fullmatch(r"raw/behavior/2025-challenge-demos/meta/episodes/"
                                                                 r"(task-\d+)/episode_\d+\.json", f))}
        tasks_data = {m[1] for f in files if (m := re.fullmatch(r"raw/behavior/2025-challenge-demos/data/"
                                                                 r"(task-\d+)/episode_\d+\.parquet", f))}
        for t in sorted(tasks_meta - tasks_data):
            problems.append(f"raw/behavior/2025-challenge-demos/data/{t}/: no episode parquet files (not fetched)")
        if not any("2025-challenge-rawdata" in f for f in files):
            problems.append("note: no 2025-challenge-rawdata HDF5 (optional): success = None for every episode")
    elif family == "mobilemanibench":
        need = ["raw/mobilemanibench/Assets/g1_robot_rotate/G1_120s.urdf"]
        cats = {m[1] for f in files if (m := re.fullmatch(r"raw/mobilemanibench/[^/]+/[^/]+/([^/]+)/.*state_infos\.pkl",
                                                           f))}
        need += [f"raw/mobilemanibench/code/unimanip/configs/data/analysis_{c}.yaml" for c in sorted(cats)]
        for u in units_mobilemanibench(files):
            if f"{u.path}/params/env.yaml" not in files:
                problems.append(f"{u.path}/params/env.yaml missing (the adapter raises; ends its shard)")
    elif family == "robocasa":
        for u in units_robocasa(files):
            if f"{u.path}/lerobot/extras/dataset_meta.json" not in files:
                problems.append(f"{u.path}/lerobot/extras/dataset_meta.json missing (versions unknown)")
        wheel = f"raw/robosuite/robosuite-{ROBOCASA_ROBOSUITE}-py3-none-any.whl"
        if wheel not in files:
            problems.append(f"{wheel}: missing (every RoboCasa365 dataset_meta records robosuite "
                            f"{ROBOCASA_ROBOSUITE}; without it robosuite assets are missing and episodes take the kinematic-only route, no tier P; include "
                            "raw/robosuite in the listing, fetch robosuite/robosuite-"
                            f"{ROBOCASA_ROBOSUITE}-py3-none-any.whl)")
        if catalog is not None:
            for e in sorted(catalog.values(), key=lambda e: e.id):
                if e.family == "robocasa" and (e.content == "assets" or e.kind == "assets") and e.usable \
                        and not _exists(files, dirs, _local(e)):
                    problems.append(f"{_local(e)}: asset archive missing (meshes from it give placeholder geoms)")
    elif family == "roboverse":
        need = [f"raw/roboverse/{p}" for p in ("robots/franka/urdf/franka_panda.urdf", "robots/franka/mjcf/panda.xml",
                                              "robots/franka_calvin/panda_longer_finger.urdf",
                                              "trajs/calvin/calvin_traj_ann/ann_dict.npy")]
    elif family == "molmobot":
        for name, version in (("rby1m", "20251224"), ("franka_droid", "20260127")):
            d = f"raw/molmobot/robots/{name}/{version}"
            if d not in dirs:
                shard = f"raw/molmobot/molmospaces/mujoco/robots/{name}/{version}/shards/00000.tar"
                how = ("shard present; unpack it with reachy_retarget.sources.molmobot.unpack_robot_assets"
                       if shard in files else f"fetch molmobot/molmospaces/mujoco/robots/{name}/{version}/shards/"
                                              "00000.tar, then unpack_robot_assets")
                problems.append(f"{d}/: robot assets missing; every episode of this robot fails ({how})")
    elif family == "bigym":
        for root in sorted({u.path.rsplit("/", 1)[0] for u in units_bigym(files)}):
            if f"{root}/assets" not in dirs:
                problems.append(f"{root}/assets/: missing (no SceneRef, kinematic-only route)")
    problems = [f"{p}: missing" for p in need if p not in files] + problems
    if catalog is not None and family != "bigym":
        absent = [e.id for e in sorted(catalog.values(), key=lambda e: e.id)
                  if e.family == family and e.usable and not _exists(files, dirs, _local(e))]
        if absent:
            by = {}
            for a in absent:  # grouped by folder
                key = posixpath.dirname(a) + "/"
                by[key] = by.get(key, 0) + 1
            groups = ", ".join(f"{k} x{v}" for k, v in sorted(by.items())[:30])
            problems.append(f"{len(absent)} catalogued usable entries not on the PVC: {groups}"
                            + (" ..." if len(by) > 30 else ""))
    return problems


def _parents(d):
    while "/" in d:
        d = d.rsplit("/", 1)[0]
        yield d


def _local(e) -> str:
    """Data-root-relative local path of a catalog entry (the extraction folder for tar streams)."""
    p = e.output_dir("") if e.transport == "tar_stream" and hasattr(e, "output_dir") else e.local_path("")
    return Path(p).as_posix().lstrip("/")


def build_jobs(family: str, files: dict, *, catalog=None, wave="w2", version="v1", target_episodes=30,
               max_episodes=36, seconds_per_episode=SECONDS_PER_EPISODE) -> list[dict]:
    """Build jobs (manifest lines plus ``_stats``) covering every input unit of ``family`` in ``files``."""
    units = sorted(UNITS[family](files, catalog), key=lambda u: u.path)
    publish = f"datasets/{wave}-{version}"
    jobs = []

    def job(paths, shard, episodes, physics, suffix, exact):
        k, n = shard
        first = paths[0].split("/", 2)[2] if paths[0].count("/") >= 2 else paths[0]
        tail = re.sub(r"[^a-z0-9]+", "-", first.lower()).strip("-")[-48:].strip("-")
        digest = hashlib.sha1("\n".join(paths).encode()).hexdigest()[:8]
        jid = f"build-{wave}-{family}{suffix}-{tail}-{digest}"  # stable: a function of the paths only
        if len(paths) > 1:
            jid += f"-n{len(paths)}"
        if n > 1:
            jid += f"-{k}of{n}"
        argv = ["-m", "reachy_retarget.build", "--family", family]
        for p in paths:
            argv += ["--path", "{pvc}/data/" + p]
        argv += ["--shard", f"{k}/{n}"] + (["--physics"] if physics else []) + ["--out", "{out}"]
        est = episodes * seconds_per_episode
        jobs.append({"id": jid, "argv": argv, "publish": publish + suffix,
                     "timeout_s": int(max(3600, 4 * est)),
                     "_stats": {"units": len(paths), "episodes": episodes, "exact": exact, "est_seconds": est}})

    pack = []

    def flush():
        if pack:
            job([u.path for u in pack], (0, 1), sum(u.episodes for u in pack), pack[0].physics,
                pack[0].publish_suffix, all(u.exact for u in pack))
            pack.clear()

    for u in units:
        if u.episodes > max_episodes:
            flush()
            n = math.ceil(u.episodes / target_episodes)
            for k in range(n):
                job([u.path], (k, n), len(range(k, u.episodes, n)), u.physics, u.publish_suffix, u.exact)
            continue
        if pack and (sum(x.episodes for x in pack) + u.episodes > max_episodes or u.physics != pack[0].physics
                     or u.publish_suffix != pack[0].publish_suffix
                     or sum(x.episodes for x in pack) >= target_episodes):
            flush()
        pack.append(u)
    flush()
    ids = [j["id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate job ids")
    return jobs


def write_build_manifest(family: str, listing, out=None, *, catalog=None, **kw) -> dict:
    """Write ``runs/build-<wave>-<family>-<version>.jsonl``; return a summary with input checks."""
    files = read_listing(listing)
    if catalog is None and family in ("molmobot", "robocasa", "behavior", "mobilemanibench", "roboverse"):
        catalog = load_catalog()
    jobs = build_jobs(family, files, catalog=catalog, **kw)
    units = UNITS[family](files, catalog)
    wave, version = kw.get("wave", "w2"), kw.get("version", "v1")
    out = Path(out) if out else ROOT / "runs" / f"build-{wave}-{family}-{version}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for j in jobs:
            f.write(json.dumps({k: v for k, v in j.items() if not k.startswith("_")}) + "\n")
    eps = [j["_stats"]["episodes"] for j in jobs]
    return {"family": family, "path": str(out), "units": len(units), "episodes_est": sum(eps),
            "episodes_exact": all(j["_stats"]["exact"] for j in jobs), "jobs": len(jobs),
            "job_episodes_min_median_max": [min(eps), sorted(eps)[len(eps) // 2], max(eps)] if eps else None,
            "cpu_hours_est": round(sum(j["_stats"]["est_seconds"] for j in jobs) / 3600, 1),
            "physics_jobs": sum("--physics" in j["argv"] for j in jobs),
            "problems": check_inputs(family, files, catalog)}


if __name__ == "__main__":
    main()
