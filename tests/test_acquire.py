"""Offline tests of the catalogue and the fetcher (a fake opener replaces HTTP)."""
import hashlib
import importlib
import io
import json
import re

import pytest

from reachy_retarget.acquire import (CatalogEntry, ChecksumMismatch, InsufficientDisk, fetch,
                                     find_entry, load_catalog, read_ledger)
fetch_mod = importlib.import_module("reachy_retarget.acquire.fetch")  # module, not the function

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


def entry(**kw):
    base = dict(id="demo/a/b.bin", family="demo", path="a/b.bin", url="https://example.invalid/b.bin",
                revision="r1", sha256=hashlib.sha256(PAYLOAD).hexdigest(), size=len(PAYLOAD),
                license="MIT", kind="low_dim")
    return CatalogEntry(**{**base, **kw})


class FakeResponse(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.status = status


def opener(log, honour_range=True):
    def open_url(req):
        rng = req.get_header("Range")
        log.append(rng)
        if rng and honour_range:
            start = int(re.match(r"bytes=(\d+)-", rng).group(1))
            return FakeResponse(PAYLOAD[start:], 206)
        return FakeResponse(PAYLOAD, 200)
    return open_url


@pytest.fixture
def plenty_of_disk(monkeypatch):
    monkeypatch.setattr(fetch_mod, "_free", lambda p: 10**15)


def test_catalog_is_pinned_and_complete():
    cat = load_catalog()
    for e in cat.values():
        assert e.id == f"{e.family}/{e.path}"
        assert re.fullmatch(r"[0-9a-f]{64}", e.sha256) and e.size > 0 and e.license
        if "huggingface.co" in e.url:
            assert f"/resolve/{e.revision}/" in e.url and re.fullmatch(r"[0-9a-f]{40}", e.revision)
    datasets = {e.dataset for e in cat.values() if e.family == "robomimic"}
    for task in ["lift", "can", "square", "transport"]:
        assert {f"robomimic/{task}/ph", f"robomimic/{task}/mh"} <= datasets
    assert "robomimic/tool_hang/ph" in datasets
    assert all(e.kind == "images_embedded" for e in cat.values() if e.family == "mimicgen")
    assert {e.kind for e in cat.values() if e.family == "robosuite"} == {"assets"}


def test_fetch_resumes_verifies_and_records(tmp_path, plenty_of_disk):
    e = entry()
    part = e.local_path(tmp_path).with_name("b.bin.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:1000])
    log = []
    (rec,) = fetch([e], tmp_path, opener=opener(log))
    assert log == ["bytes=1000-"]
    assert e.local_path(tmp_path).read_bytes() == PAYLOAD and not part.exists()
    assert rec["sha256_verified"] == e.sha256 and rec["sha256_source"] == "catalog"
    assert read_ledger(tmp_path)[e.id]["bytes"] == len(PAYLOAD)
    # Already present: verified again, no transfer.
    fetch([e], tmp_path, opener=opener(log))
    assert len(log) == 1
    assert find_entry(e.local_path(tmp_path), {e.id: e}) == e


def test_fetch_restarts_when_range_is_ignored(tmp_path, plenty_of_disk):
    e = entry()
    part = e.local_path(tmp_path).with_name("b.bin.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"garbage")
    fetch([e], tmp_path, opener=opener([], honour_range=False))
    assert e.local_path(tmp_path).read_bytes() == PAYLOAD


def test_fetch_rejects_bad_checksum(tmp_path, plenty_of_disk):
    e = entry(sha256="0" * 64)
    with pytest.raises(ChecksumMismatch):
        fetch([e], tmp_path, opener=opener([]))
    assert not e.local_path(tmp_path).exists()
    assert e.local_path(tmp_path).with_name("b.bin.sha256-mismatch").exists()
    assert read_ledger(tmp_path) == {}


def test_fetch_keeps_50gb_reserve(tmp_path, monkeypatch):
    e = entry()
    monkeypatch.setattr(fetch_mod, "_free", lambda p: 50_000_000_000 + len(PAYLOAD) - 1)
    log = []
    with pytest.raises(InsufficientDisk):
        fetch([e], tmp_path, opener=opener(log))
    assert log == []  # refused before any transfer


def test_fetch_refuses_images_by_default(tmp_path, plenty_of_disk):
    with pytest.raises(PermissionError):
        fetch([entry(kind="images_embedded")], tmp_path, opener=opener([]))


def test_ledger_is_json(tmp_path, plenty_of_disk):
    fetch([entry()], tmp_path, opener=opener([]))
    json.loads((tmp_path / "raw" / "ledger.json").read_text())
