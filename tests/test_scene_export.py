"""Exported source scenes: SceneRef save/load and the export checks against a recorded tier-P scene."""
import json
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
import pytest
from test_validate_physics import BOX0, BOX_OBJ, scene_xml

from reachy_retarget import scene_export, sources
from reachy_retarget.schema.source import SceneRef
from reachy_retarget.validate.scene import build_scene

LEG = "/home/recorder/robosuite/models/assets/objects/meshes/leg.obj"


def _ref():
    return SceneRef(mjcf=scene_xml(leg_mesh_file=LEG), robot_prefixes=["robot0_", "gripper0_"],
                    initial_qpos={"box_joint": [*BOX0, 1, 0, 0, 0]},
                    assets={"objects/meshes/leg.obj": BOX_OBJ.encode(), "robots/panda/link0.stl": b"unused"},
                    inactive_bodies=[], reference={"object_environment_depth_m": 0.001})


def test_scene_ref_round_trip_keeps_bytes_and_checks_hashes(tmp_path):
    ref = _ref()
    doc = ref.save(tmp_path / "s", ["objects/meshes/leg.obj"])
    assert set(doc["assets"]) == {"objects/meshes/leg.obj"}
    back = SceneRef.load(tmp_path / "s")
    assert back.mjcf == ref.mjcf and back.assets == {"objects/meshes/leg.obj": BOX_OBJ.encode()}
    np.testing.assert_array_equal(back.initial_qpos["box_joint"], [*BOX0, 1, 0, 0, 0])
    assert back.robot_prefixes == ref.robot_prefixes and back.reference == ref.reference
    assert build_scene(back).info["scene_sha256"] == build_scene(ref).info["scene_sha256"]
    with pytest.raises(FileExistsError):
        ref.save(tmp_path / "s")
    (tmp_path / "s" / "assets" / doc["assets"]["objects/meshes/leg.obj"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        SceneRef.load(tmp_path / "s")


def _episode(path, info, demo="demo_3"):
    metadata = {"dataset": "mimicgen/core/test", "episode_id": demo, "family": "mimicgen",
                "provenance": {"file": "/mnt/raw/test.hdf5", "demo_key": demo}}
    with h5py.File(path, "w") as f:
        f.attrs["metadata"] = json.dumps(metadata)
        f.create_group("physics").attrs["info"] = json.dumps(
            {"scene": info, "simulator": f"mujoco {mujoco.__version__}"})


def test_export_saves_a_verified_scene_once(tmp_path, monkeypatch):
    ref = _ref()
    recorded = build_scene(ref).info
    _episode(tmp_path / "a.h5", recorded, "demo_3")
    _episode(tmp_path / "b.h5", recorded, "demo_4")
    calls = []

    def demos(family, path, demos):
        calls.append((family, path, demos))
        return iter([SimpleNamespace(scene=ref, provenance={"demo_key": k}) for k in demos])
    monkeypatch.setattr(sources, "iter_episodes", demos)
    records = scene_export.export([tmp_path / "a.h5", tmp_path / "b.h5"], tmp_path / "out", name="job")
    assert [r["status"] for r in records] == ["exported", "exists"]
    assert calls == [("mimicgen", "/mnt/raw/test.hdf5", ["demo_3", "demo_4"])]  # one read per source file
    assert records[0]["checks"] == {"source_mjcf_sha256": True, "assets": True, "parked_bodies": True,
                                    "scene_sha256": True, "passed": True}
    saved = SceneRef.load(tmp_path / "out" / records[0]["scene"])
    assert set(saved.assets) == {"objects/meshes/leg.obj"}  # the source robot's assets are not kept
    assert build_scene(saved).info["scene_sha256"] == recorded["scene_sha256"]
    lines = (tmp_path / "out" / "records" / "job.jsonl").read_text().splitlines()
    assert [json.loads(line)["status"] for line in lines] == ["exported", "exists"]


def test_export_refuses_a_scene_that_differs_from_the_record(tmp_path, monkeypatch):
    ref = _ref()
    recorded = build_scene(ref).info
    _episode(tmp_path / "a.h5", recorded)
    other = _ref()
    other.assets = {"objects/meshes/leg.obj": BOX_OBJ.replace("v 1 1 1", "v 2 2 2").encode()}
    monkeypatch.setattr(sources, "iter_episodes",
                        lambda *a, **k: iter([SimpleNamespace(scene=other, provenance={"demo_key": "demo_3"})]))
    rec = scene_export.export([tmp_path / "a.h5"], tmp_path / "out")[0]
    assert rec["status"] == "mismatch" and rec["checks"]["assets"] is False
    assert not (tmp_path / "out" / "scenes").exists()


def test_asset_keys_need_every_asset_from_the_scene_ref():
    assert scene_export.asset_keys([{"file": "/x/a.obj", "found": "assets:a.obj"},
                                    {"file": "b.png", "found": "assets:exact"}]) == ["a.obj", "b.png"]
    assert scene_export.asset_keys([{"file": "/x/a.obj", "found": "resolver"}]) is None
