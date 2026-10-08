"""Memory-bounding behaviour of the build path: shard selection inside adapters, release of
compiled source models, chunked self-collision checks, the table-free catalog and the
per-episode memory fields of build records. All offline, on the tiny robomimic fixture."""
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from reachy_retarget.acquire import load_catalog
from reachy_retarget.build import build
from reachy_retarget.robot.collision import SelfCollision
from reachy_retarget.retarget.wbik import Clearance
from reachy_retarget.sources import iter_episodes, register
from reachy_retarget.sources import robosuite as rs

FIXTURE = Path(__file__).parent / "fixtures" / "robomimic_can_ph_demo0_sub.hdf5"


@pytest.fixture
def three_demos(tmp_path):
    """The fixture with demo_0 copied to demo_1 and demo_2; demo_2 records a different MJCF."""
    path = tmp_path / "three.hdf5"
    shutil.copy(FIXTURE, path)
    with h5py.File(path, "a") as f:
        for key in ("demo_1", "demo_2"):
            f.copy(f["data/demo_0"], f["data"], name=key)
        f["data/demo_2"].attrs["model_file"] = f["data/demo_2"].attrs["model_file"] + "\n<!-- recorded variant -->"
    return path


def test_select_skips_without_compiling(three_demos, monkeypatch):
    calls = []
    real = rs._compile
    monkeypatch.setattr(rs, "_compile", lambda xml, specs: calls.append(xml) or real(xml, specs))
    out = list(iter_episodes("robomimic", three_demos, select=lambda i: i == 1))
    assert out[0] is None and out[2] is None and out[1].episode_id == "demo_1"
    assert len(calls) == 1
    full = list(iter_episodes("robomimic", three_demos))
    np.testing.assert_array_equal(out[1].effectors["gripper0_right"].pose, full[1].effectors["gripper0_right"].pose)


def test_compiled_model_kept_only_for_the_next_identical_mjcf(three_demos):
    stream = iter_episodes("robomimic", three_demos)
    next(stream)  # demo_1 records the same MJCF: the model is kept for it
    assert len(rs._LAST_MODEL) == 1
    first = rs._LAST_MODEL[0][1]
    next(stream)  # demo_2 records another MJCF: released before the yield
    assert not rs._LAST_MODEL
    next(stream)  # last demo: nothing follows
    assert not rs._LAST_MODEL
    del first


def test_select_fallback_for_adapters_without_it(tmp_path):
    seen = []

    @register("_memtest_plain")
    def _plain(path, *, family, **kw):
        for i in range(4):
            seen.append(i)
            yield i

    assert list(iter_episodes("_memtest_plain", tmp_path, select=lambda i: i % 2 == 1)) == [None, 1, None, 3]
    assert seen == [0, 1, 2, 3]  # built, then dropped


def test_texture_free_kinematic_xml():
    xml = ('<mujoco><asset><texture name="t" type="2d" file="a/t.png"/><texture name="sky" type="skybox" '
           'builtin="gradient" width="8" height="8"/><material name="m" texture="t"/><material name="n" texture="sky"/>'
           '<mesh name="me" file="a/me.stl"/></asset><worldbody/></mujoco>')
    out, assets = rs._without_file_textures(xml, {"a/t.png": b"png", "a/me.stl": b"stl"})
    assert 'file="a/t.png"' not in out and 'name="sky"' in out and 'texture="sky"' in out
    assert 'texture="t"' not in out and assets == {"a/me.stl": b"stl"}
    assert rs._set_shell_inertia(xml, []) == xml
    assert 'inertia="shell"' in rs._set_shell_inertia(xml, ["me"])


def test_chunked_clearance_is_bitwise_identical():
    q = np.random.default_rng(0).uniform(-1, 1, (150, 22))
    sc = SelfCollision.load()
    np.testing.assert_array_equal(sc.min_clearance(q), sc.distances(q).min(axis=-1))
    np.testing.assert_array_equal(sc.min_clearance(q.reshape(3, 50, 22)), sc.distances(q).min(axis=-1).reshape(3, 50))
    assert type(sc.min_clearance(q[0])) is type(sc.distances(q[0]).min(axis=-1))
    cl = Clearance(("left", "right"))
    np.testing.assert_array_equal(cl.minimum(q), cl.distances(q)[0].min(axis=-1))


def test_catalog_without_tables_is_the_inline_subset():
    full, inline = load_catalog(), load_catalog(tables=False)
    assert len(inline) < len(full) and all(full[k] == e for k, e in inline.items())
    assert {e.family for e in inline.values()} >= {"robomimic", "mimicgen", "libero", "robosuite"}
    assert not any(e.content == "assets" and e.asset_marker for k, e in full.items() if k not in inline)


def test_build_reads_only_its_shard_and_records_memory(three_demos, tmp_path, monkeypatch):
    calls = []
    real = rs._compile
    monkeypatch.setattr(rs, "_compile", lambda xml, specs: calls.append(xml) or real(xml, specs))
    counts = build("robomimic", str(three_demos), (1, 3), str(tmp_path / "out"), physics=False)
    assert counts["episodes"] == 1 and len(calls) == 1 and counts["max_rss_mb"] > 0
    (rec_file,) = (tmp_path / "out" / "records" / "robomimic").glob("*.jsonl")
    (rec,) = [json.loads(line) for line in rec_file.read_text().splitlines()]
    assert rec["index"] == 1 and rec["episode_id"] == "demo_1"
    assert rec["rss_mb"] is None or 0 < rec["rss_mb"] <= rec["max_rss_mb"] * 1.5


def test_scene_xml_does_not_depend_on_the_mujoco_asset_cache():
    import mujoco

    from reachy_retarget.schema.source import SceneRef
    from reachy_retarget.validate.scene import build_scene

    obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    ref = SceneRef(mjcf='<mujoco><asset><mesh name="m" file="/rec/objects/m.obj" scale=".05 .05 .05"/></asset>'
                        '<worldbody><body name="o_main" pos="0.5 0 0.5"><freejoint name="o_joint"/>'
                        '<geom type="mesh" mesh="m"/></body><body name="robot0_base"><geom size=".01"/></body>'
                        '</worldbody></mujoco>',
                   robot_prefixes=["robot0_"], assets={"objects/m.obj": obj})
    cache = mujoco.mj_getCache()
    capacity = mujoco.mj_getCacheCapacity(cache)
    try:
        xmls = []
        for size in (0, 64 << 20, 64 << 20):  # decoded, decoded (fills the cache), served from the cache
            mujoco.mj_setCacheCapacity(cache, size)
            scene = build_scene(ref)
            xmls.append((scene.xml, scene.info["scene_sha256"]))
    finally:
        mujoco.mj_setCacheCapacity(cache, capacity)
    assert xmls[0] == xmls[1] == xmls[2] and "content_type" not in xmls[0][0]
