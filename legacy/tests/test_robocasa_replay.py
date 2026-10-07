"""Offline FK checks against a full primitive model and recorded parquet rows."""

import gzip
import hashlib
import json

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.agent_dataset import export_normalized, inspect_archive
from reachy_retarget.mobile_pilot import SOURCES
from reachy_retarget.native_replay.robocasa import compile_kinematic_model, reconstruct, MODALITY_ACTION, MODALITY_STATE


XML = """<mujoco><compiler angle="radian"/>
<worldbody><body name="robot0_base" pos=".1 -.2 0" euler="0 0 .2">
 <body name="mobilebase0_base"><joint name="base_x" type="slide" axis="1 0 0" ref=".2"/>
  <joint name="base_yaw" axis="0 0 1" ref=".1"/>
  <geom type="box" size=".1 .1 .1"/>
  <body name="support" pos="0 0 .7"><site name="mobilebase0_center"/>
   <body name="robot0_right_hand" pos=".3 0 .1"><joint name="wrist" axis="0 1 0" pos=".02 0 0"/>
    <geom type="sphere" size=".04"/>
    <site name="gripper0_right_grip_site" pos=".05 0 0" euler="0 0 1.5707963267948966"/>
    <body name="finger1" pos=".05 .02 0"><joint name="gripper0_right_finger_joint1" type="slide" axis="0 1 0"/>
     <geom type="box" size=".01 .01 .02"/></body>
    <body name="finger2" pos=".05 -.02 0"><joint name="gripper0_right_finger_joint2" type="slide" axis="0 1 0"/>
     <geom type="box" size=".01 .01 .02"/></body>
   </body>
  </body>
 </body>
</body>
<body name="counter_main" pos="1 0 .4"><geom type="box" size=".3 .3 .4"/></body>
<body name="cab_main" pos="1 .8 1"><geom type="box" size=".2 .2 .2"/>
 <body name="left_door" pos="-.2 -.2 0"><joint name="cab_left" axis="0 0 1"/>
  <geom type="box" size=".1 .01 .2"/><body name="handle" pos=".2 -.03 0"/>
 </body>
 <body name="right_door" pos=".2 -.2 0"><joint name="cab_right" axis="0 0 1"/>
  <geom type="box" size=".1 .01 .2"/>
 </body>
</body>
<body name="obj_main" pos="1 0 .85"><freejoint name="obj_free"/><geom type="box" size=".03 .06 .1"/></body>
<body name="distr_counter_main" pos="3 0 .3"><freejoint name="distr_free"/><geom type="sphere" size=".1"/></body>
</worldbody></mujoco>"""


def fixture(tmp_path, *, bad_observation=False):
    folder = tmp_path / "data/raw/robocasa_native_pilot/lerobot"
    episode = folder / "extras/episode_000000"
    episode.mkdir(parents=True)
    (folder / "meta").mkdir()
    (folder / "data/chunk-000").mkdir(parents=True)
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    states, obs = [], []
    for frame, time in enumerate((.5, .55, .65)):
        data.qpos[:] = model.qpos0
        for name, value in {"base_x": .2 + frame * .03, "base_yaw": .1 + frame * .2, "wrist": frame * .1,
                            "cab_left": frame * .2, "cab_right": -frame * .1,
                            "gripper0_right_finger_joint1": .01, "gripper0_right_finger_joint2": -.01}.items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        data.qvel[:] = np.arange(model.nv) * .01
        mujoco.mj_kinematics(model, data)
        bs, es = model.site("mobilebase0_center").id, model.site("gripper0_right_grip_site").id
        br = data.site_xmat[bs].reshape(3, 3)
        hand = data.xmat[model.body("robot0_right_hand").id].reshape(3, 3)
        obs.append(np.r_[data.site_xpos[bs], Rotation.from_matrix(br).as_quat(),
                         br.T @ (data.site_xpos[es] - data.site_xpos[bs]), Rotation.from_matrix(br.T @ hand).as_quat(), .01, -.01])
        states.append(np.r_[time, data.qpos, data.qvel])
    obs = np.asarray(obs)
    if bad_observation:
        obs[1, 7] += .1
    np.savez(episode / "states.npz", states=np.asarray(states))
    (episode / "model.xml.gz").write_bytes(gzip.compress(XML.encode()))
    (episode / "ep_meta.json").write_text(json.dumps({"object_cfgs": [{"name": "obj", "info": {"cat": "cereal"}},
        {"name": "distr_counter", "info": {"cat": "unrelated"}}], "fixture_refs": {"counter": "counter", "cab": "cab"}}))
    (folder / "extras/dataset_meta.json").write_text(json.dumps({"env": "PickPlaceCounterToCabinet",
        "robocasa_version": "0.5.1", "robosuite_version": "1.5.2", "mujoco_version": "3.3.1"}))
    modality = {kind: {name: {"start": start, "end": end, "original_key": key} for name, (start, end) in spec.items()}
                for kind, spec, key in (("state", MODALITY_STATE, "observation.state"), ("action", MODALITY_ACTION, "action"))}
    (folder / "meta/modality.json").write_text(json.dumps(modality))
    pq.write_table(pa.table({"observation.state": obs.tolist(), "action": np.arange(36).reshape(3, 12).tolist(),
        "timestamp": [0., .05, .1], "frame_index": [0, 1, 2], "next.reward": [0., 0., 1.], "next.done": [False, False, True]}),
        folder / "data/chunk-000/episode_000000.parquet")
    rows = [{"path": str(path.relative_to(folder.parent)), "size": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in folder.rglob("*") if path.is_file()]
    (tmp_path / "catalog").mkdir()
    (tmp_path / "catalog/robocasa-pilot.json").write_text(json.dumps({**SOURCES["robocasa"], "downloaded": rows}))
    return np.asarray(states)


def test_projection_matches_complete_primitive_model_at_all_named_frames():
    full = mujoco.MjModel.from_xml_string(XML)
    projected, details = compile_kinematic_model(XML)
    original, scratch = mujoco.MjData(full), mujoco.MjData(projected)
    assert projected.ngeom == 0 and projected.nu == 0
    assert details["physical_model"] is False
    rng = np.random.default_rng(12)
    for _ in range(10):
        original.qpos[:] = full.qpos0
        for i in range(full.njnt):
            if full.jnt_type[i] in (2, 3):
                original.qpos[full.jnt_qposadr[i]] += rng.uniform(-.3, .3)
        scratch.qpos[:] = original.qpos
        mujoco.mj_kinematics(full, original)
        mujoco.mj_kinematics(projected, scratch)
        for kind, count, position, matrix in ((mujoco.mjtObj.mjOBJ_BODY, full.nbody, "xpos", "xmat"),
                                             (mujoco.mjtObj.mjOBJ_SITE, full.nsite, "site_xpos", "site_xmat")):
            for i in range(1 if kind == mujoco.mjtObj.mjOBJ_BODY else 0, count):
                name = mujoco.mj_id2name(full, kind, i)
                j = mujoco.mj_name2id(projected, kind, name)
                np.testing.assert_allclose(getattr(original, position)[i], getattr(scratch, position)[j], atol=1e-14)
                np.testing.assert_allclose(getattr(original, matrix)[i], getattr(scratch, matrix)[j], atol=1e-14)


def test_reconstruct_verifies_observations_preserves_gaps_and_excludes_distractors(tmp_path, monkeypatch):
    original = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Source reconstruction must not step or perform a dynamics forward call")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    record = reconstruct(tmp_path)
    a, m = record["arrays"], record["metadata"]
    assert record["status"] == "source_fk_verified_actions_preserved"
    np.testing.assert_array_equal(a["source/state"], original)
    np.testing.assert_allclose(a["timestamp"], [0., .05, .15])
    np.testing.assert_allclose(a["source/parquet_timestamp"], [0., .05, .1])
    np.testing.assert_array_equal(a["source/action"], np.arange(36).reshape(3, 12))
    assert not any("distr_counter" in key for key in a)
    assert {key.split('/')[1] for key in a if key.startswith('objects/')} == {"obj", "counter", "cab"}
    assert len(m["objects"]["cab"]["part_body_names"]) == 4
    assert m["object_coverage"]["all_task_object_poses_decoded"] is True
    assert max(m["source_kinematic_reconstruction"]["validation_max_errors"].values()) < 1e-14
    legacy = Rotation.from_quat(a["source/right_eef_observable_pose_base"][:, [4, 5, 6, 3]])
    site = Rotation.from_quat(a["source/right_tcp_pose_base"][:, [4, 5, 6, 3]])
    np.testing.assert_allclose((legacy.inv() * site).magnitude(), np.pi / 2)
    path = export_normalized(record, tmp_path / "common", "robocasa", "fk-fixture")
    assert inspect_archive(path)["rows"] == 3
    assert m["physics_validated"] is False


def test_wrong_recorded_eef_observation_fails_closed(tmp_path):
    fixture(tmp_path, bad_observation=True)
    with pytest.raises(ValueError, match="FK disagrees"):
        reconstruct(tmp_path)


@pytest.mark.parametrize("xml", [XML.replace('<worldbody>', '<default><joint damping="1"/></default><worldbody>'),
                                 XML.replace('angle="radian"', 'angle="radian" alignfree="true"')])
def test_unverified_compiler_semantics_are_rejected(xml):
    with pytest.raises(ValueError, match="defaults|Aligned"):
        compile_kinematic_model(xml)
