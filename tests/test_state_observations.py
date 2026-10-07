"""Read real MuJoCo state without forwarding, stepping or touching its buffers."""

import json

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.agent_dataset import assess_state_observations, write_archive
from reachy_retarget.state_observations import StateObserver, StateObservationBuffer, stack_observations


XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody><geom type="plane" size="10 10 .1"/>
 <body name="base_link" pos="0 0 .6"><freejoint name="base_free"/>
  <inertial pos=".03 .04 .05" quat=".7071067811865476 0 0 .7071067811865476" mass="3" diaginertia=".02 .03 .04"/>
  <geom type="box" size=".1 .1 .1" mass="3"/>
  <camera name="torso" pos=".1 0 .1" euler="0 90 0"/>
  <body name="left_arm" pos=".2 .2 .1"><joint name="l_shoulder_pitch" axis="0 1 0"/>
   <geom type="capsule" size=".03" fromto="0 0 0 .2 0 0"/>
   <site name="left_tcp" pos=".2 0 0"/>
  </body>
  <body name="right_arm" pos=".2 -.2 .1"><joint name="r_shoulder_pitch" axis="0 1 0"/>
   <geom type="capsule" size=".03" fromto="0 0 0 .2 0 0"/>
   <site name="right_tcp" pos=".2 0 0"/>
  </body>
  <body name="head" pos="0 0 .3"><joint name="neck_yaw" axis="0 0 1"/>
   <inertial pos=".02 .01 .03" quat=".7071067811865476 0 .7071067811865476 0" mass=".2" diaginertia=".001 .002 .0025"/>
   <geom type="sphere" size=".05"/><camera name="left_head" pos=".05 0 .05"/>
  </body>
 </body>
 <body name="can" pos="1 .4 .3"><freejoint name="can_free"/><geom type="cylinder" size=".04 .08"/></body>
 <body name="bin" pos="1.5 .4 .2"><geom type="box" size=".2 .2 .1"/></body>
 <body name="cabinet" pos="2 -.5 .4"><geom type="box" size=".1 .1 .2"/>
  <body name="door" pos=".11 0 0"><joint name="door_hinge" axis="0 0 1" limited="true" range="0 90"/>
   <geom type="box" size=".01 .1 .2"/>
  </body>
 </body>
 <body name="unrelated_prop" pos="3 4 .5"><freejoint name="prop_free"/><geom type="sphere" size=".1"/>
  <body name="unrelated_flap" pos=".1 0 0"><joint name="unrelated_hinge"/><geom type="box" size=".05 .05 .01"/></body>
 </body>
 <camera name="unrelated_overview" pos="0 0 5"/>
</worldbody>
<actuator><position name="left_motor" joint="l_shoulder_pitch" kp="10"/>
 <position name="right_motor" joint="r_shoulder_pitch" kp="10"/>
 <velocity name="neck_motor" joint="neck_yaw" kv="2"/>
</actuator></mujoco>"""


def setup():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[3:7] = Rotation.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    data.qvel[:6] = [2, 1, 0, 0, 0, .3]
    for name, q, v in (("l_shoulder_pitch", .1, .2), ("r_shoulder_pitch", -.15, -.1), ("neck_yaw", .2, .4), ("door_hinge", .3, -.2)):
        index = model.joint(name).id
        data.qpos[model.jnt_qposadr[index]], data.qvel[model.jnt_dofadr[index]] = q, v
    data.ctrl[:] = [.12, -.14, .3]
    data.time = .2
    mujoco.mj_forward(model, data)
    return model, data


def observer(model, **overrides):
    options = dict(task_objects={"can": "can", "bin": "bin", "cabinet": "cabinet"},
                   robot_body_root="base_link", base_body="base_link", head_body="head",
                   tcp_sites={"left": "left_tcp", "right": "right_tcp"},
                   object_roles={"pickup": "can", "receptacle": "bin"},
                   required_joint_names=("l_shoulder_pitch", "r_shoulder_pitch", "neck_yaw"))
    options.update(overrides)
    return StateObserver(model, **options)


def arrays_snapshot(value):
    # Covers both integration state and caches, not only qpos/qvel.
    return {name: getattr(value, name).copy() for name in dir(value)
            if not name.startswith("_") and isinstance(getattr(value, name), np.ndarray)}


def state_snapshot(data):
    # Inactive sparse-constraint views can address unallocated arena memory in
    # dense mode. Compare defined integration/kinematic/dynamic buffers instead.
    names = ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "act", "act_dot",
             "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "userdata",
             "xpos", "xquat", "xmat", "cvel", "subtree_com", "site_xpos", "site_xmat",
             "cam_xpos", "cam_xmat", "actuator_length", "actuator_velocity", "actuator_force")
    return {name: getattr(data, name).copy() for name in names}


def test_capture_is_read_only_and_returns_detached_real_state(monkeypatch):
    model, data = setup()
    recorder = observer(model)
    before_model, before_data, time = arrays_snapshot(model), state_snapshot(data), data.time
    def forbidden(*args, **kwargs):
        raise AssertionError("Collector must not update the simulator")
    for name in ("mj_forward", "mj_step", "mj_kinematics", "mj_setState", "mj_resetData"):
        monkeypatch.setattr(mujoco, name, forbidden)
    frame = recorder.capture(data)
    for name, value in before_model.items():
        np.testing.assert_array_equal(getattr(model, name), value, err_msg=name)
    for name, value in before_data.items():
        np.testing.assert_array_equal(getattr(data, name), value, err_msg=name)
    assert data.time == time
    arrays = frame["arrays"]
    assert arrays["observation/joint_position"].shape == (4,)
    assert arrays["observation/body_poses"].shape == (8, 7)
    assert arrays["observation/head_pose"].shape == (7,)
    assert arrays["timestamp"].shape == ()
    assert not any(key.endswith("rgb") for key in arrays)
    assert set(frame["metadata"]["cameras"]) == {"torso", "left_head"}
    assert frame["metadata"]["joint_groups"]["neck"] == [2]
    np.testing.assert_array_equal(arrays["observation/actuator_control"], data.ctrl)
    np.testing.assert_array_equal(arrays["observation/joint_position"], data.qpos[recorder.qadr])
    expected = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, expected, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.testing.assert_array_equal(arrays["physics_state"], expected)
    arrays["observation/joint_position"][:] = 999
    np.testing.assert_array_equal(data.qpos, before_data["qpos"])


def test_batched_pose_conversion_matches_every_scalar_channel(monkeypatch):
    import reachy_retarget.state_observations as module
    model, data = setup()
    recorder = observer(model)
    batched = recorder.capture(data)
    counts = []
    def original_scalar_conversion(position, rotation):
        counts.append(len(position))
        return np.stack([np.r_[p, Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]]
                         for p, r in zip(position, rotation)])
    monkeypatch.setattr(module, "_poses", original_scalar_conversion)
    scalar = recorder.capture(data)
    assert counts == [len(recorder.body_ids)+6+len(recorder.camera_ids)]
    assert scalar["metadata"] == batched["metadata"]
    assert scalar["missing_fields"] == batched["missing_fields"]
    assert scalar["arrays"].keys() == batched["arrays"].keys()
    for channel in scalar["arrays"]:
        np.testing.assert_array_equal(scalar["arrays"][channel], batched["arrays"][channel], err_msg=channel)


@pytest.mark.parametrize("cache,name", [("xmat", "head"), ("site_xmat", "left_tcp"), ("cam_xmat", "torso")])
def test_batch_rejects_invalid_body_site_and_camera_rotation_caches(cache, name):
    model, data = setup()
    recorder = observer(model)
    index = (model.body(name) if cache == "xmat" else model.site(name)
             if cache == "site_xmat" else model.camera(name)).id
    getattr(data, cache)[index, 0] = 2.
    with pytest.raises(ValueError, match="Invalid pose cache"):
        recorder.capture(data)


def test_metadata_template_and_all_nested_snapshots_are_independent():
    model, data = setup()
    identity = {"nested": [{"revision": "pinned"}], "tuple": ("preserve", 2)}
    recorder = observer(model, model_identity=identity)
    first = recorder.capture(data)
    public = recorder.metadata
    public["joint_names"].clear()
    public["objects"]["cabinet"]["joints"][0]["name"] = "corrupted"
    identity["nested"][0]["revision"] = "external mutation"
    first["metadata"]["model_identity"]["nested"][0]["revision"] = "frame mutation"
    first["metadata"]["cameras"]["torso"]["fovy_degrees"] = 999
    second = recorder.capture(data)
    assert second["metadata"]["joint_names"] == recorder.joint_names
    assert second["metadata"]["objects"]["cabinet"]["joints"][0]["name"] == "door_hinge"
    assert second["metadata"]["model_identity"] == {"nested": [{"revision": "pinned"}], "tuple": ("preserve", 2)}
    assert second["metadata"]["cameras"]["torso"]["fovy_degrees"] != 999


def test_base_and_ee_twists_use_correct_origins_axes_and_relative_motion():
    model, data = setup()
    arrays = observer(model).capture(data)["arrays"]
    # Free-base translational velocity is [2,1,0] in world; yaw=90 means
    # [1,-2,0] in local/base axes. Spatial vectors are angular then linear.
    np.testing.assert_allclose(arrays["observation/base_twist_world"], [0, 0, .3, 2, 1, 0], atol=1e-12)
    np.testing.assert_allclose(arrays["observation/base_twist"], [1, -2, .3], atol=1e-12)
    base = model.body("base_link").id
    rotation = data.xmat[base].reshape(3, 3)
    world = arrays["observation/left_tcp_twist_world"]
    np.testing.assert_allclose(arrays["observation/left_tcp_twist_base"],
        np.r_[rotation.T @ world[:3], rotation.T @ world[3:]], atol=1e-12)
    # A finite-difference check uses a *separate* state and is outside capture.
    step = 1e-7
    later = mujoco.MjData(model)
    later.qpos[:] = data.qpos
    later.qvel[:] = data.qvel
    mujoco.mj_integratePos(model, later.qpos, later.qvel, step)
    mujoco.mj_forward(model, later)
    site = model.site("left_tcp").id
    fd = (later.site_xpos[site] - data.site_xpos[site])/step
    np.testing.assert_allclose(world[3:], fd, atol=1e-7)
    old_relative = rotation.T @ (data.site_xpos[site]-data.xpos[base])
    new_relative = later.xmat[base].reshape(3, 3).T @ (later.site_xpos[site]-later.xpos[base])
    np.testing.assert_allclose(arrays["observation/left_tcp_twist_relative_base"][3:],
                               (new_relative-old_relative)/step, atol=1e-7)
    np.testing.assert_allclose(arrays["observation/left_tcp_pose_base"][:3], old_relative, atol=1e-12)


def test_camera_is_optical_world_pose_and_only_task_object_subtrees_are_exposed():
    model, data = setup()
    frame = observer(model).capture(data)
    arrays, meta = frame["arrays"], frame["metadata"]
    assert "unrelated_prop" not in meta["body_names"]
    assert "unrelated_hinge" not in meta["joint_names"]
    assert meta["scalar_joint_coverage_complete"] is True
    assert meta["compiled_scalar_joint_coverage_complete"] is False
    assert "door" in meta["body_names"] and "cabinet" in meta["body_names"]
    assert set(meta["objects"]) == {"can", "bin", "cabinet"}
    assert not any("unrelated_prop" in key for key in arrays)
    assert meta["task_object_coverage_complete"] is True
    # Full replay state still retains the unrelated prop rather than editing it.
    assert arrays["observation/qpos"].shape == (model.nq,)
    np.testing.assert_allclose(arrays["observation/objects/cabinet/qpos"], [.3])
    np.testing.assert_allclose(arrays["observation/objects/cabinet/qvel"], [-.2])
    camera = model.camera("torso").id
    value = arrays["observation/cameras/torso/pose"]
    np.testing.assert_allclose(value[:3], data.cam_xpos[camera])
    np.testing.assert_allclose(Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix(),
                              data.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1]), atol=1e-12)


def test_body_twists_follow_pose_origins_not_offset_rotated_inertial_frames():
    model, data = setup()
    frame = observer(model).capture(data)
    arrays = frame["arrays"]
    step = 1e-7
    later = mujoco.MjData(model)
    later.qpos[:] = data.qpos
    later.qvel[:] = data.qvel
    mujoco.mj_integratePos(model, later.qpos, later.qvel, step)
    mujoco.mj_forward(model, later)
    base = model.body("base_link").id
    base_rotation = data.xmat[base].reshape(3, 3)
    for name, channel in (("base_link", "base"), ("head", "head")):
        body = model.body(name).id
        # Both origins and axes differ from the inertial/COM frame in this model.
        assert np.linalg.norm(model.body_ipos[body]) > .01
        assert not np.allclose(model.body_iquat[body], [1, 0, 0, 0])
        velocity = arrays[f"observation/{channel}_twist_world"]
        fd_linear = (later.xpos[body]-data.xpos[body])/step
        rotation = data.xmat[body].reshape(3, 3)
        next_rotation = later.xmat[body].reshape(3, 3)
        fd_angular = Rotation.from_matrix(next_rotation @ rotation.T).as_rotvec()/step
        np.testing.assert_allclose(velocity[:3], fd_angular, atol=1e-7)
        np.testing.assert_allclose(velocity[3:], fd_linear, atol=1e-7)
        body_index = frame["metadata"]["body_ids"].index(body)
        np.testing.assert_allclose(arrays["observation/body_twists_world"][body_index], velocity, atol=1e-12)
        wrong_com_velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, wrong_com_velocity, 0)
        assert not np.allclose(velocity[3:], wrong_com_velocity[3:])
        if channel == "base":
            np.testing.assert_allclose(arrays["observation/base_twist_local"],
                np.r_[rotation.T @ fd_angular, rotation.T @ fd_linear], atol=1e-7)
        else:
            np.testing.assert_allclose(arrays["observation/head_twist_base"],
                np.r_[base_rotation.T @ fd_angular, base_rotation.T @ fd_linear], atol=1e-7)
    head = model.body("head").id
    old_relative = base_rotation.T @ (data.xpos[head]-data.xpos[base])
    new_base_rotation = later.xmat[base].reshape(3, 3)
    new_relative = new_base_rotation.T @ (later.xpos[head]-later.xpos[base])
    np.testing.assert_allclose(arrays["observation/head_twist_relative_base"][3:],
                              (new_relative-old_relative)/step, atol=1e-7)
    np.testing.assert_allclose(arrays["observation/head_twist_relative_base"][:3], [0, 0, .4], atol=1e-12)


def test_undefined_roles_are_missing_and_absent_neck_is_not_silently_complete():
    model, data = setup()
    frame = observer(model, object_roles={}, required_joint_names=("neck_roll", "neck_pitch", "neck_yaw")).capture(data)
    assert "observation/pickup_pose" not in frame["arrays"]
    assert "observation/receptacle_pose" not in frame["arrays"]
    assert set(frame["metadata"]["missing_required_joints"]) == {"neck_roll", "neck_pitch"}
    assert not frame["metadata"]["required_joint_coverage_complete"]
    assessment = assess_state_observations(stack_observations([frame]))
    assert not assessment["state_observations_complete"]
    assert "required_joint/neck_roll" in assessment["missing_fields"]
    assert assessment["rgb_required"] is False


@pytest.mark.parametrize("options", [
    {"task_objects": {}}, {"task_objects": {"missing": "unknown"}},
    {"task_objects": {"wrong": "base_link"}},
    {"task_objects": {"cabinet": "cabinet", "duplicate": "door"}},
    {"object_roles": {"pickup": "unrelated_prop"}},
    {"tcp_sites": {"left": "unknown", "right": "right_tcp"}},
    {"head_body": "can"}, {"cameras": {"torso": "nonexistent"}},
    {"joint_names": ["neck_yaw"]},
])
def test_explicit_mapping_errors_fail_without_fallback(options):
    model, _ = setup()
    with pytest.raises(ValueError):
        observer(model, **options)


def test_state_only_assessment_and_archive_do_not_require_rgb_or_claim_controls(tmp_path):
    model, data = setup()
    recorder = observer(model)
    frames = [recorder.capture(data)]
    data.time += .01
    mujoco.mj_forward(model, data)
    frames.append(recorder.capture(data))
    record = stack_observations(frames)
    assert record["arrays"]["observation/joint_position"].shape == (2, 4)
    assert record["arrays"]["observation/body_poses"].shape == (2, 8, 7)
    assessment = assess_state_observations(record)
    assert assessment["state_observations_complete"] and assessment["table_fields_complete"]
    assert assessment["rgb_required"] is False and assessment["policy_ready"] is False
    assert assessment["control_recording_complete"] is False
    assert not assessment["recorded_command_fields"]
    assert "controller_state_json" in assessment["missing_control_fields"]
    # An actuator snapshot is not automatically labelled as desired EE/joint targets.
    assert "observation/actuator_control" in record["arrays"]
    assert "command/joint_position" not in record["arrays"]
    metadata = dict(record["metadata"], source_sequence="real-toy-model-state", source_group="one-state-trial",
                    source_urls=[], source_revision=None, derived_fields={}, simulation_assumptions=[])
    path = write_archive(tmp_path, "states", "one", record["arrays"], metadata)
    stored = assess_state_observations(path)
    assert stored["state_observations_complete"] and not stored["physics_validated"]
    import h5py
    with h5py.File(path) as handle:
        recorded = json.loads(handle.attrs["metadata_json"])
        assert recorded["storage_profile"] == "state-only"


def test_buffer_matches_every_frame_channel_and_detaches_outputs(tmp_path):
    model, data = setup()
    recorder = observer(model, model_identity={"nested": [{"revision": "pinned"}]})
    buffer = StateObservationBuffer(recorder)
    frames = []
    for _ in range(4):
        frames.append(recorder.capture(data))
        before = state_snapshot(data)
        buffer.capture(data)
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(data, key), value)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    assert len(buffer) == 4
    expected, actual = stack_observations(frames), buffer.finish()
    assert actual.keys() == expected.keys()
    for key in expected["arrays"]:
        np.testing.assert_array_equal(actual["arrays"][key], expected["arrays"][key])
    for key in ("metadata", "missing_fields", "status"):
        assert actual[key] == expected[key]
    # Previously returned frames, public metadata and completed sequences cannot
    # alter the buffer's privately retained metadata or numeric records.
    frames[0]["metadata"]["model_identity"]["nested"][0]["revision"] = "frame mutation"
    exposed = recorder.metadata
    exposed["joint_names"].append("not a real joint")
    actual["metadata"]["model_identity"]["nested"][0]["revision"] = "output mutation"
    actual["arrays"]["observation/qpos"][:] = 999
    second = buffer.finish()
    assert second["metadata"] == expected["metadata"]
    np.testing.assert_array_equal(second["arrays"]["observation/qpos"], expected["arrays"]["observation/qpos"])
    meta = dict(second["metadata"], source_sequence="buffer-check", source_group="one-state-trial",
                source_urls=[], source_revision=None, derived_fields={}, simulation_assumptions=[])
    path = write_archive(tmp_path, "states", "buffer", second["arrays"], meta)
    assert assess_state_observations(path)["state_observations_complete"]


def test_buffer_requires_real_increasing_states():
    model, data = setup()
    buffer = StateObservationBuffer(observer(model))
    with pytest.raises(ValueError, match="At least one"):
        buffer.finish()
    buffer.capture(data)
    buffer.capture(data)
    with pytest.raises(ValueError, match="timestamps must increase"):
        buffer.finish()
        assert not recorded["rgb_required"] and not recorded["rgb_stored"]
        assert "observation/cameras/torso/rgb" not in recorded["missing_fields"]
        assert "observation/cameras/torso/rgb" in recorded["missing_native_v5_fields"]


def test_missing_commands_are_not_recovered_by_copying_observed_state():
    model, data = setup()
    record = stack_observations([observer(model).capture(data)])
    record["arrays"]["command/joint_position"] = record["arrays"]["observation/joint_position"].copy()
    result = assess_state_observations(record)
    assert result["command_fields_without_semantics"] == ["command/joint_position"]
    assert not result["control_recording_complete"]
    assert result["state_observations_complete"]


def test_stacking_rejects_duplicate_time_and_changed_task_contract():
    model, data = setup()
    first = observer(model).capture(data)
    with pytest.raises(ValueError, match="timestamps"):
        stack_observations([first, first])
    second = observer(model, object_roles={}).capture(data)
    with pytest.raises(ValueError, match="contract changed"):
        stack_observations([first, second])


def test_complete_position_servo_record_does_not_require_unissued_control_modes():
    model, data = setup()
    record = stack_observations([observer(model).capture(data)])
    # Explicit test command records, not values invented by the collector.
    record["arrays"]["command/joint_position"] = np.array([[.12, -.14]])
    record["arrays"]["applied_controls"] = np.tile(data.ctrl, (1, 5, 1))
    controller = {"format": "custom-servo-state-v1", "identity": "synthetic-test-controller",
                  "joints": {"left": .1, "right": -.15}, "targets": {"left": .12, "right": -.14},
                  "memory": {"elapsed_s": 0}, "servo_targets": {"left": .12, "right": -.14}}
    record["arrays"]["controller_state_json"] = np.array([json.dumps(controller)])
    record["metadata"]["issued_control_modes"] = ["joint_position"]
    record["metadata"]["command_semantics"] = {"command/joint_position": {
        "source": "actual supplied command", "joint_names": ["l_shoulder_pitch", "r_shoulder_pitch"], "units": "rad"}}
    result = assess_state_observations(record)
    assert result["state_observations_complete"] and result["actual_controls_complete"]
    assert result["complete_recorded_command_profile"]
    assert not result["missing_control_fields"]
    assert result["all_control_modes_available"] is False
    assert "command/joint_velocity" in result["alternative_control_fields_missing"]
    assert result["physics_validated"] is False and result["policy_ready"] is False
    record["metadata"]["issued_control_modes"] = ["joint_position", "joint_velocity"]
    assert not assess_state_observations(record)["actual_controls_complete"]
    record["metadata"]["command_profile"] = {"name": "reviewed-position-servo", "required_fields": ["command/joint_position"]}
    assert not assess_state_observations(record)["actual_controls_complete"]
    record["metadata"]["issued_control_modes"] = ["joint_position"]
    assert assess_state_observations(record)["actual_controls_complete"]


def test_state_assessment_rejects_undeclared_object_channels():
    model, data = setup()
    record = stack_observations([observer(model).capture(data)])
    record["arrays"]["observation/objects/unrelated_prop/pose"] = np.array([[3, 4, .5, 1, 0, 0, 0.]])
    assessment = assess_state_observations(record)
    assert not assessment["state_observations_complete"]
    assert any("Undeclared task object" in error for error in assessment["errors"])
