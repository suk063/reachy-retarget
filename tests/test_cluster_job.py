"""The pod-side job runner: streaming, parallel, no-overwrite publishing and receipts."""
import json

import pytest

from cluster import job


def test_publisher_streams_complete_files_once_and_never_overwrites(tmp_path):
    out, dest = tmp_path / "out", tmp_path / "dest"
    (out / "episodes" / "d").mkdir(parents=True)
    for i in range(20):
        (out / "episodes" / "d" / f"e{i}.h5").write_bytes(bytes([i]) * 1000)
    (out / "episodes" / "d" / ".e20.h5.tmp-1").write_bytes(b"partial")  # still being written
    publisher = job.Publisher(out, dest, threads=4)
    publisher.stream(["episodes/**/*.h5"])
    assert len(publisher.futures) == 20
    (out / "episodes" / "d" / ".e20.h5.tmp-1").rename(out / "episodes" / "d" / "e20.h5")
    (out / "records.jsonl").write_text("{}\n")
    publisher.stream(["episodes/**/*.h5"])
    records = publisher.finish()
    assert [r["path"] for r in records] == sorted(r["path"] for r in records)
    assert len(records) == 22 and (dest / "records.jsonl").exists()
    assert (dest / "episodes" / "d" / "e7.h5").read_bytes() == bytes([7]) * 1000
    assert not list(dest.rglob(".*"))  # no partial copies left
    (out / "episodes" / "d" / "e7.h5").write_bytes(b"other")
    with pytest.raises(FileExistsError):
        job.publish(out, dest)


def test_job_streams_while_running_and_writes_a_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "PVC", tmp_path / "pvc")
    monkeypatch.setattr(job, "RESERVE", 0)
    script = tmp_path / "work.py"
    script.write_text("import pathlib, sys, time\n"
                      "out = pathlib.Path(sys.argv[1])\n"
                      "(out / 'a.h5').write_text('a')\n"
                      "deadline = time.time() + 30\n"
                      "while not (pathlib.Path(sys.argv[2]) / 'a.h5').exists() and time.time() < deadline:\n"
                      "    time.sleep(.1)\n"
                      "(out / 'log.jsonl').write_text('done')\n")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    spec = {"id": "j1", "batch": "b", "release": "r", "argv": [str(script), "{out}", "{pvc}/pub"],
            "publish": "pub", "stream": ["*.h5"], "publish_threads": 2}
    (job_dir / "spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(job.time, "sleep", lambda s: None)  # poll without waiting
    assert job.main(str(job_dir)) == 0
    receipt = json.loads((tmp_path / "pvc" / "receipts" / "b" / "j1.json").read_text())
    assert receipt["state"] == "succeeded" and receipt["returncode"] == 0
    assert sorted(r["path"] for r in receipt["published"]) == ["a.h5", "log.jsonl"]
