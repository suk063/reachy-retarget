import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from reachy_retarget.mobile_pilot import SOURCES
from reachy_retarget.native_replay.bigym import action_contract, capture_state, load_pilot, normalize, replay


def pilot(root, **changes):
    metadata = {"uuid": "test-recording", "seed": 783198766,
        "package_versions": {"mujoco": "3.1.5", "bigym": "4.0.0"},
        "environment_data": {"env_name": "MovePlate", "action_mode_name": "JointPositionActionMode",
            "floating_base": True, "action_mode_absolute": True,
            "floating_dofs": ["pelvis_x", "pelvis_y", "pelvis_rz"],
            "observation_config": {"cameras": [], "proprioception": True, "privileged_information": False}}}
    metadata.update(changes)
    values = np.arange(45, dtype="<f8").reshape(3, 15)
    header = {"__metadata__": {k: json.dumps(v) for k, v in metadata.items()},
              "info_demo_action": {"dtype": "F64", "shape": [3, 15], "data_offsets": [0, values.nbytes]}}
    content = json.dumps(header).encode()
    path = root / "data/raw/bigym_native_pilot/MovePlate/pilot.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(struct.pack("<Q", len(content)) + content + values.tobytes())
    catalog = root / "catalog"
    catalog.mkdir()
    (catalog / "bigym-pilot.json").write_text(json.dumps({**SOURCES["bigym"], "downloaded": [{
        "path": "MovePlate/pilot.safetensors", "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}))
    return path, values


def test_pilot_verifies_native_configuration_and_preserves_action_order(tmp_path):
    _, expected = pilot(tmp_path)
    metadata, tensors, records, _ = load_pilot(tmp_path)
    assert metadata["seed"] == 783198766
    np.testing.assert_array_equal(tensors["info_demo_action"], expected)
    assert len(records) == 1


@pytest.mark.parametrize("changes", [{"seed": None}, {"seed": True}, {"package_versions": {"mujoco": "3.15.0"}}])
def test_unverified_clock_reset_version_contract_rejected(tmp_path, changes):
    pilot(tmp_path, **changes)
    with pytest.raises(ValueError):
        load_pilot(tmp_path)


def test_original_checksum_is_required(tmp_path):
    path, _ = pilot(tmp_path)
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_pilot(tmp_path)


def test_mixed_action_contract_distinguishes_base_arm_and_gripper():
    base = [NS(joint=NS(full_identifier=name, type=kind)) for name, kind in (
        ("h1/pelvis_x", "slide"), ("h1/pelvis_y", "slide"), ("h1/pelvis_rz", "hinge"))]
    arms = [NS(joint=NS(full_identifier=f"h1/{side}_{i}")) for side in ("left", "right") for i in range(5)]
    class Side:
        def __init__(self, name):
            self.name = name
    grippers = {Side(name): NS(range=(0, 1), actuators=[NS(full_identifier=name + "_actuator")]) for name in ("LEFT", "RIGHT")}
    env = NS(robot=NS(floating_base=NS(all_actuators=base), limb_actuators=arms, grippers=grippers), action_space=NS(shape=(15,)))
    channels = action_contract(env)
    assert [row["kind"] for row in channels[:3]] == ["base_position_target_increment"] * 3
    assert channels[2]["unit"] == "rad"
    assert all(row["kind"] == "absolute_joint_position_target" for row in channels[3:13])
    assert [row["name"] for row in channels[13:]] == ["left_gripper", "right_gripper"]


def model_and_selection():
    mj = pytest.importorskip("mujoco")
    xml = '''<mujoco><option timestep=".002" gravity="0 0 0"/><worldbody>
    <body name="pelvis"><joint name="pelvis_x" type="slide" axis="1 0 0"/><geom size=".1"/>
      <camera name="head" pos="0 0 .3"/>
      <body name="left_arm" pos="0 .2 0"><joint name="left_joint"/><geom size=".05"/><site name="left_wrist" pos=".2 0 0"/></body>
      <body name="right_arm" pos="0 -.2 0"><joint name="right_joint"/><geom size=".05"/><site name="right_wrist" pos=".2 0 0"/></body>
    </body>
    <body name="plate" pos="1 0 1"><freejoint/><geom size=".05"/></body>
    <body name="rack_start" pos="1 1 0"><geom size=".1"/></body>
    <body name="rack_target" pos="1 -1 0"><geom size=".1"/></body>
    <body name="table" pos="1 0 0"><geom size=".1"/></body>
    <body name="unrelated" pos="10 0 0"><freejoint/><geom size=".1"/></body>
    </worldbody></mujoco>'''
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    selection = {"joint_ids": [0, 1, 2], "robot_body_ids": [1, 2, 3], "base": 1,
        "objects": {name: model.body(name).id for name in ("plate", "rack_start", "rack_target", "table")},
        "sites": {side: model.site(side + "_wrist").id for side in ("left", "right")},
        "cameras": {"head": model.camera("head").id}}
    return mj, model, data, selection


def test_capture_refreshes_post_step_pose_without_changing_state_or_warmstart():
    mj, model, data, selection = model_and_selection()
    data.qvel[0] = .7
    mj.mj_step(model, data)
    original = {key: np.array(getattr(data, key), copy=True) for key in ("qpos", "qvel", "qacc_warmstart", "ctrl")}
    row = capture_state(model, data, selection)
    for key, value in original.items():
        np.testing.assert_array_equal(getattr(data, key), value)
    assert row["timestamp"] == .002
    assert row["source/base_pose"][0] == data.qpos[0]
    assert row["source/left_eef_pose"][0] == pytest.approx(data.qpos[0] + .2)
    assert row["source/joint_velocity"][0] == pytest.approx(.7)
    assert row["source/cameras/head/pose"].shape == (7,)
    assert not any("neck" in key or "head_pose" in key or "rgb" in key for key in row)


def test_only_task_object_pose_channels_are_selected():
    mj, model, data, selection = model_and_selection()
    mj.mj_forward(model, data)
    row = capture_state(model, data, selection)
    object_names = {key.split("/")[1] for key in row if key.startswith("objects/")}
    assert object_names == {"plate", "rack_start", "rack_target", "table"}
    assert "unrelated" not in " ".join(row)
    assert row["source/qpos"].shape == (model.nq,)  # Opaque full replay state retained.
    assert row["source/robot_body_poses"].shape == (3, 7)


def test_failed_attempt_is_immutable_and_saved_for_inspection(tmp_path):
    output = tmp_path / "attempt"
    result = replay(tmp_path / "missing-pilot", output, tmp_path / "missing-source")
    assert result["status"] == "failed_source_replay"
    assert result["metadata"]["source_actions_replayed"] == 0
    assert result["metadata"]["physics_validated"] is False
    assert normalize(output)["metadata"]["source_error"]["type"] == "FileNotFoundError"
    with pytest.raises(FileExistsError):
        replay(tmp_path / "missing-pilot", output, tmp_path / "missing-source")
    with (output / "states.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        normalize(output)
