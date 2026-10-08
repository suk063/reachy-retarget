"""Fetch-manifest batching (synthetic catalog; no network, no cluster)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.manifests import GB, plan, write_manifest  # noqa: E402
from reachy_retarget.acquire import CatalogEntry, select  # noqa: E402


def e(path, size, kind="file", usable=True, **kw):
    return CatalogEntry(id=f"fam/{path}", family="fam", path=path, url="u", revision="r", sha256=None, size=size,
                        license="MIT", kind=kind, content="low_dim", usable=usable, **kw)


def catalog():
    entries = [e(f"small/t{t}/ep{i}.bin", 50_000_000) for t in range(4) for i in range(30)]  # 1.5 GB per task
    entries += [e(f"big/{i}.tar", 4 * GB, kind="tar_stream") for i in range(3)]
    entries += [e("big/huge.tar", 12 * GB, kind="tar_stream"), e("big/mg.tar", 9 * GB, kind="tar_stream", usable=False)]
    entries += [e(f"mixed/keep{i}.bin", 100_000_000) for i in range(3)] + [e("mixed/skip.bin", 1)]
    entries += [e(f"pkg/house_{i}", None, kind="range_package",
                  source={"stored_size": 300_000_000, "inflated_size": 320_000_000}) for i in range(20)]
    return {x.id: x for x in entries}


def test_plan_batches_by_size_with_complete_prefixes_and_stable_ids():
    cat = catalog()
    selected = [x for x in cat.values() if x.id != "fam/mixed/skip.bin"]
    jobs = plan("fam", catalog=cat, prefixes=["fam/"], skip={"fam/mixed/skip.bin"})
    sels = [s for j in jobs for s in j["argv"][3:j["argv"].index("--root")]]
    picked = select(cat, sels)
    assert sorted(x.id for x in picked) == sorted(x.id for x in selected if x.usable)  # exact cover, no extras
    assert len({x.id for x in picked}) == len(picked)
    assert "fam/small/t0/" in sels and "fam/mixed/" not in sels  # skip.bin is excluded, so no prefix there
    assert "fam/big/mg.tar" not in sels and "fam/big/" not in sels  # unusable entries are never selected
    for j in jobs:
        st = j["_stats"]
        assert st["scratch_bytes"] <= 40 * GB
        assert st["transfer_bytes"] <= 5 * GB or st["entries"] == 1  # only a single huge entry exceeds 5 GB
        assert j["argv"][:3] == ["-m", "reachy_retarget.cli", "fetch"] and j["publish"] == "data"
        assert j["argv"][j["argv"].index("--root"):][:3] == ["--root", "{out}", "--strip-images"]
        assert j["id"].startswith("fetch-fam-") and j["timeout_s"] >= 3600
    assert [j["id"] for j in plan("fam", catalog=cat, prefixes=["fam/"], skip={"fam/mixed/skip.bin"})] == \
        [j["id"] for j in jobs]


def test_write_manifest(tmp_path):
    out = tmp_path / "m.jsonl"
    summary = write_manifest("fam", out, catalog=catalog(), prefixes=["fam/pkg/"])
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert summary["jobs"] == len(lines) == 2 and summary["entries"] == 20
    assert all(set(x) == {"id", "argv", "publish", "timeout_s"} for x in lines)
    assert summary["transfer_gb"] == 6.0 and summary["scratch_gb_upper"] == 6.4
