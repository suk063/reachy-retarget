"""Official registries and public release discovery; no credential solicitation."""

import concurrent.futures
import csv
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import quote
import requests
from .store import json_write, now

SEEDS = [
    ("omomo", "human_object", "https://github.com/lijiaman/omomo_release"),
    ("parahome", "human_object", "https://github.com/snuvclab/ParaHome"),
    ("oakink", "human_object", "https://oakink.net/"),
    ("oakink2", "human_object", "https://github.com/oakink/OakInk2"),
    ("taco", "human_object", "https://github.com/leolyliu/TACO-Instructions"),
    ("dexycb", "human_object", "https://dex-ycb.github.io/"),
    ("hot3d", "human_object", "https://facebookresearch.github.io/hot3d/"),
    ("humoto", "human_object", "https://github.com/adobe-research/humoto"),
    ("hoi4d", "human_object", "https://github.com/leolyliu/HOI4D-Instructions"),
    ("h2o", "human_object", "https://h2odataset.ethz.ch/"),
    ("ho3d", "human_object", "https://github.com/shreyashampali/ho3d"),
    ("grab", "human_object", "https://grab.is.tue.mpg.de/"),
    ("behave", "human_object", "https://virtualhumans.mpi-inf.mpg.de/behave/"),
    ("intercap", "human_object", "https://intercap.is.tue.mpg.de/"),
    ("arctic", "human_object", "https://arctic.is.tue.mpg.de/"),
    ("hodome", "human_object", "https://github.com/Juzezhang/NeuralDome_Toolbox"),
    ("chairs", "human_object", "https://jnnan.github.io/chairs/"),
    ("interact", "human_object", "https://github.com/wzyabcas/InterAct"),
    ("intermimic", "human_object", "https://github.com/Sirui-Xu/InterMimic"),
    ("hoi_retarget", "human_object", "https://github.com/leggedrobotics/hoi-retarget"),
    ("oxe", "robot", "https://github.com/google-deepmind/open_x_embodiment"),
    ("droid", "robot", "https://droid-dataset.github.io/droid/the-droid-dataset"),
    ("bridge", "robot", "https://bridgedata-v2.github.io/"),
    ("aloha", "robot", "https://github.com/tonyzhaozh/act"),
    ("mobile_aloha", "robot", "https://github.com/MarkFzp/mobile-aloha"),
    (
        "umi",
        "robot",
        "https://github.com/real-stanford/universal_manipulation_interface",
    ),
    ("fastumi", "robot", "https://github.com/YdingTeam/FastUMI"),
    ("roboturk", "robot", "https://roboturk.stanford.edu/dataset_real.html"),
    ("robomimic", "simulation", "https://github.com/ARISE-Initiative/robomimic"),
    ("mimicgen", "simulation", "https://github.com/NVlabs/mimicgen"),
    ("robocasa", "simulation", "https://github.com/robocasa/robocasa"),
    ("libero", "simulation", "https://github.com/Lifelong-Robot-Learning/LIBERO"),
    (
        "maniskill",
        "simulation",
        "https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html",
    ),
    ("calvin", "simulation", "https://github.com/mees/calvin"),
    ("furniturebench", "robot", "https://github.com/clvrai/furniture-bench"),
    ("d4rl_kitchen", "simulation", "https://github.com/Farama-Foundation/D4RL"),
    ("cmu", "motion", "https://mocap.cs.cmu.edu/"),
    (
        "lafan1",
        "motion",
        "https://github.com/ubisoft/ubisoft-laforge-animation-dataset",
    ),
    ("amass", "motion", "https://amass.is.tue.mpg.de/"),
]


def get(url, **kwargs):
    error = None
    for attempt in range(3):
        try:
            r = requests.get(
                url,
                timeout=(20, 60),
                headers={"User-Agent": "reachy-retarget/0.1"},
                **kwargs,
            )
            if r.status_code == 429:
                return r  # Caller records provider cooldown; do not retry immediately.
            if r.status_code in (500, 502, 503):
                r.raise_for_status()
            return r
        except requests.RequestException as e:
            error = e
            time.sleep(2**attempt)
    raise error


def hf_id(repo):
    return "hf__" + repo.replace("/", "__")


def state_file(path, mode="lerobot"):
    p = path.lower()
    name = Path(p).name
    if name.startswith(("readme", "license", "notice")):
        return True
    if mode == "humoto":
        return p.endswith((".glb", ".fbx", ".json", ".pdf"))
    if mode == "robomimic":
        return p.endswith(".hdf5") and "image" not in p and not p.startswith("test/")
    if mode == "oakink":
        return (
            p.startswith("anno_preview/")
            or p.startswith("program/")
            or p.startswith("object")
            or p.startswith("meta")
            or name in ("anno.tar", "anno.tar.gz")
        ) and not any(s in p for s in ("video", "image", "frame"))
    if mode == "hot3d":
        return (
            p.endswith((".json", ".pdf"))
            or name in ("hot3d_models.zip", "hot3d_base.zip")
            or p.startswith("object_models/")
            and p.endswith((".ply", ".glb"))
        )
    if mode == "hoi":
        return p.endswith(
            (
                ".parquet",
                ".npz",
                ".npy",
                ".pt",
                ".urdf",
                ".obj",
                ".stl",
                ".ply",
                ".json",
                ".md",
                ".txt",
            )
        ) and not any(s in p for s in ("video", "image"))
    return (
        p.startswith(("data/", "meta/", "meta_data/")) or p.endswith(".json")
    ) and p.endswith((".parquet", ".json", ".jsonl", ".safetensors", ".md", ".txt"))


def hf_discover(store, repo, mode="lerobot", category="robot", parent=None):
    from .acquire import host_paused, pause_host

    sid = hf_id(repo)
    url = f"https://huggingface.co/datasets/{repo}"
    from .scope import EXCLUDED_SOURCES
    if sid in EXCLUDED_SOURCES:
        store.source(sid, category, url, "excluded_objectless", reason=EXCLUDED_SOURCES[sid])
        return
    checkpoint = store.root / "catalog/discovery_checkpoints" / sid
    checkpoint.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint / "cursor.json"
    pages = checkpoint / "selected.jsonl"
    info = {}
    inventory = []
    total = 0
    selected_bytes = 0
    u = None
    if host_paused(store, url):
        print(f"{sid}: provider backoff; discovery queue retained", flush=True)
        return
    try:
        if state_path.exists():
            state = json.loads(state_path.read_text())
            info = state["info"]
            u = state["next_url"]
            total = state["repository_files"]
            if pages.exists():
                # Ignore uncommitted appended lines after an interrupted cursor write.
                with pages.open() as f:
                    for i, line in enumerate(f):
                        if i >= state["selected_files"]:
                            break
                        inventory.append(json.loads(line))
                pages.write_text("".join(json.dumps(x) + "\n" for x in inventory))
            selected_bytes = sum(x.get("size", 0) for x in inventory)
        else:
            r = get(f"https://huggingface.co/api/datasets/{repo}")
            r.raise_for_status()
            info = r.json()
            if info.get("gated"):
                store.source(
                    sid,
                    category,
                    url,
                    "access_required",
                    parent=parent,
                    gated=info["gated"],
                )
                return
            u = f"https://huggingface.co/api/datasets/{repo}/tree/{info['sha']}?recursive=true&limit=1000"
        revision = info["sha"]
        tags = info.get("tags", [])
        while u:
            resp = get(u)
            resp.raise_for_status()
            page_entries = []
            selected_page = []
            for x in resp.json():
                if x["type"] != "file":
                    continue
                total += 1
                if not state_file(x["path"], mode):
                    continue
                selected_bytes += x.get("size", 0)
                inventory.append(x)
                selected_page.append(x)
                page_entries.append(
                    (
                        x["path"],
                        f"{url}/resolve/{revision}/{quote(x['path'])}",
                        x.get("size"),
                        (x.get("lfs") or {}).get("oid"),
                    )
                )
            store.files(sid, page_entries)
            with pages.open("a") as f:
                for x in selected_page:
                    f.write(json.dumps(x) + "\n")
                f.flush()
                import os

                os.fsync(f.fileno())
            u = resp.links.get("next", {}).get("url")
            json_write(
                state_path,
                {
                    "info": info,
                    "next_url": u,
                    "repository_files": total,
                    "selected_files": len(inventory),
                },
            )
        store.source(
            sid,
            category,
            url,
            "queued" if inventory else "no_selected_state_files",
            repo_id=repo,
            revision=revision,
            parent=parent,
            mode=mode,
            license=[t[8:] for t in tags if t.startswith("license:")] or "unspecified",
            selected_files=len(inventory),
            repository_files=total,
            selected_bytes=selected_bytes,
            discovery_complete=True,
            error=None,
            coverage="state-and-metadata subset; videos excluded; episode/object usability not yet verified",
        )
        print(
            f"{sid}: {len(inventory)}/{total} files, {selected_bytes / 1e9:.3f} GB",
            flush=True,
        )
    except Exception as e:
        if "429" in str(e):
            pause_host(store, url)
        store.source(
            sid,
            category,
            url,
            "discovery_failed",
            error=f"{type(e).__name__}: {e}",
            parent=parent,
            revision=info.get("sha"),
            repo_id=repo,
            mode=mode,
            discovery_complete=False,
            coverage="Partial discovery; durable page cursor and selected files retained. Retry resumes the pinned revision.",
        )
        print(f"{sid}: {type(e).__name__}; partial listing retained", flush=True)
    finally:
        if info and inventory:
            json_write(
                store.root / "catalog/evidence" / f"{sid}.json",
                {
                    "info": info,
                    "selected_files": inventory,
                    "discovery_complete": u is None,
                },
            )


def github_release(store, repo, sid, mode="humoto", category="human_object"):
    info = get(f"https://api.github.com/repos/{repo}").json()
    branch = info.get("default_branch", "main")
    commit = get(f"https://api.github.com/repos/{repo}/commits/{branch}").json()["sha"]
    resp = get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1")
    resp.raise_for_status()
    tree = resp.json()
    if tree.get("truncated"):
        raise ValueError(
            "Truncated GitHub tree must not be treated as a complete release"
        )
    selected = []
    for x in tree["tree"]:
        if x["type"] == "blob" and state_file(x["path"], mode):
            u = f"https://raw.githubusercontent.com/{repo}/{commit}/{quote(x['path'])}"
            store.file(sid, x["path"], u, x.get("size"))
            selected.append(x)
    json_write(
        store.root / "catalog/evidence" / f"{sid}.json",
        {"revision": commit, "files": selected},
    )
    store.source(
        sid,
        category,
        f"https://github.com/{repo}",
        "queued",
        revision=commit,
        selected_files=len(selected),
        coverage="Public release only; full HUMOTO requires separate application"
        if sid == "humoto"
        else "selected state files",
    )


def probe_seeds(store):
    def probe(seed):
        sid, category, url = seed
        try:
            evidence_url = url
            if url.startswith("https://github.com/") and len(url.split("/")) == 5:
                api = get(
                    "https://api.github.com/repos/"
                    + "/".join(url.split("/")[-2:])
                    + "/readme"
                )
                if api.ok:
                    import base64

                    text = base64.b64decode(api.json()["content"]).decode()
                    evidence_url = api.json().get("html_url", url)
                    code = api.status_code
                else:
                    text, code = api.text, api.status_code
            else:
                r = get(url)
                text, code = r.text, r.status_code
            path = store.root / "catalog/evidence" / f"{sid}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            links = sorted(set(re.findall(r'https?://[^\s<>"\)\]]+', text)))
            status = "catalogued" if code == 200 else "access_failed"
            if sid in ("grab", "arctic", "amass", "intercap"):
                status = "access_required"
            store.source(
                sid,
                category,
                url,
                status,
                evidence_url=evidence_url,
                http_status=code,
                outgoing_links=links,
                decision="No manual applications or new accounts; inspect official public release routes.",
            )
        except Exception as e:
            store.source(sid, category, url, "access_failed", error=str(e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(probe, SEEDS))


def discover_reachy(store):
    found = {}
    for endpoint in ("search=reachy", "author=pollen-robotics"):
        u = f"https://huggingface.co/api/datasets?{endpoint}&limit=100&full=true"
        while u:
            r = get(u)
            r.raise_for_status()
            found.update({x["id"]: x for x in r.json()})
            u = r.links.get("next", {}).get("url")
    exclude = (
        "mini",
        "benchmark",
        "emotions",
        "sticker",
        "songs",
        "speech",
        "app-",
        "app_",
        "arcade",
        "gripette",
        "grabette",
        "microduck",
        "fleet_usage",
        "personalities",
        "reachytodo",
        "Reachy_RENS",
        "anyskin",
        "fridaf/",
    )
    repos = sorted(k for k in found if not any(w.lower() in k.lower() for w in exclude))
    # These rows are not manipulation corpora despite matching the search text.
    for repo in set(found) - set(repos):
        store.source(
            hf_id(repo),
            "search_exclusion",
            f"https://huggingface.co/datasets/{repo}",
            "out_of_scope",
            reason="Reachy Mini, non-motion content, or unrelated device/benchmark",
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda repo: hf_discover(store, repo, category="reachy"), repos))


def oxe_registry(store):
    url = "https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/export?format=csv&gid=0"
    r = get(url)
    r.raise_for_status()
    if "text/html" in r.headers.get("Content-Type", ""):
        raise ValueError("OXE registry returned HTML")
    path = store.root / "catalog/evidence/oxe_registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    rows = list(csv.reader(io.StringIO(r.text)))
    json_write(store.root / "catalog/evidence/oxe_registry.json", rows)
    store.source(
        "oxe",
        "robot",
        "https://github.com/google-deepmind/open_x_embodiment",
        "registry_downloaded",
        registry_url=url,
        rows=len(rows),
    )
    return rows


def discover_oxe(store):
    rows = oxe_registry(store)
    header = next(i for i, r in enumerate(rows) if "Registered Dataset Name" in r)
    names = {
        r[18]: dict(zip(rows[header], r))
        for r in rows[header + 1 :]
        if len(r) > 18 and r[18].strip() and r[18] != "-"
    }
    repos = []
    u = "https://huggingface.co/api/datasets?author=lerobot&limit=1000"
    while u:
        response = get(u)
        response.raise_for_status()
        repos.extend(x["id"] for x in response.json())
        u = response.links.get("next", {}).get("url")
    selected = {}
    for name, row in names.items():
        sid = "oxe__" + re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        base = name.replace("_converted_externally_to_rlds", "")
        candidates = [
            r for r in repos if r.split("/")[-1] in (name, base, base + "_v2")
        ]
        if name == "bridge":
            candidates = [
                r
                for r in repos
                if r.split("/")[-1] in ("bridge", "bridge_v2", "bridge_orig")
            ]
        store.source(
            sid,
            "robot",
            "https://github.com/google-deepmind/open_x_embodiment",
            "mirror_identified" if candidates else "rlds_route_pending",
            registry=row,
            mirrors=candidates,
            reason="Official OXE constituent; state subset preferred over multi-TB image-containing RLDS. Mirror is not an independent source.",
        )
        for repo in candidates:
            selected[repo] = sid
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda item: hf_discover(store, item[0], parent=item[1]),
                selected.items(),
            )
        )
    # Extra public corpora whose robot state is stored independently of video.
    extras = [
        r
        for r in repos
        if any(
            x in r.split("/")[-1]
            for x in (
                "aloha",
                "droid",
                "calvin",
                "furniture",
                "mimicgen",
                "robocasa",
                "libero",
                "maniskill",
                "metaworld",
                "umi",
            )
        )
        and not any(x in r for x in ("eval_", "diffusion", "test_"))
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda repo: hf_discover(
                    store, repo, parent="extra_public_robot_corpora"
                ),
                extras,
            )
        )


def discover(store, group="core"):
    if group in ("all", "core"):
        # Native Reachy numeric mirrors lack object-state adapters; keep their
        # historical inventory as evidence, outside the active collection scope.
        for repo, mode, category, parent in [
            ("robomimic/robomimic_datasets", "robomimic", "simulation", "robomimic"),
            ("kelvin34501/OakInk-v2", "oakink", "human_object", "oakink2"),
            ("leggedrobotics/hoi-retarget", "hoi", "human_object", "hoi_retarget"),
            ("bop-benchmark/hot3d", "hot3d", "human_object", "hot3d"),
        ]:
            hf_discover(store, repo, mode, category, parent)
        github_release(store, "adobe-research/humoto", "humoto")
        # Official ParaHome state/mesh archives (images are not requested).
        for name, ident in {
            "scan": "1-OuWvVFOFCEhut7J2t1kNbr5jv78QNFP",
            "seq": "10MYSSM2H7f6g2n9nnXta48qmAhZ7r4yd",
            "metadata": "1jPRCsotiep0nElHgyLQNjlkHsWgHbjhi",
            "joint_info": "15fGnZn8o4I2bzQtQF-9MliwxKc2IUdzI",
        }.items():
            store.file(
                "parahome",
                name + ".zip",
                f"https://drive.google.com/uc?export=download&id={ident}",
            )
        store.source(
            "parahome",
            "human_object",
            "https://github.com/snuvclab/ParaHome",
            "queued",
            license="CC-BY-NC-SA-4.0",
            coverage="Official scan, seq, metadata and joint-info archives",
        )
    if group in ("all", "survey"):
        probe_seeds(store)
        oxe_registry(store)
    if group in ("all", "oxe"):
        discover_oxe(store)
