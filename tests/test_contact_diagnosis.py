import hashlib
import json

import h5py
import mujoco
import numpy as np
import pytest

from reachy_retarget.contact_diagnosis import (contact_category, diagnose, passive_prefix_convergence,
                                              passive_admission, global_compliance_sensitivity)


def _fixture(tmp_path, *, object_height=.06):
    scene = tmp_path / "scene.xml"
    scene.write_text(('''<mujoco><option timestep=".002"/><worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body name="base_link" pos="1 0 .5">
        <body name="r_hand_distal_link"><joint name="r_hand_finger" type="hinge"/>
          <geom name="pad" type="box" size=".02 .01 .04" mass="1"/></body>
        <body name="r_hand_distal_mimic_link" pos="0 .1 0">
          <geom name="pad2" type="box" size=".02 .01 .04"/></body>
      </body>
      <body name="object" pos="0 0 OBJECT_HEIGHT"><freejoint name="object_joint"/>
        <geom name="cube" type="box" size=".02 .02 .02" mass=".1"/></body>
      </worldbody><actuator><position name="r_hand_finger" joint="r_hand_finger" kp="10"/>
      </actuator></mujoco>''').replace('OBJECT_HEIGHT', str(object_height)))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.qvel[model.joint("object_joint").dofadr[0] + 2] = -.1
    initial = {key: getattr(data, key).copy() for key in ("qpos", "qvel", "ctrl")}
    mujoco.mj_forward(model, data)
    records = {key: [] for key in ("time_s", "simulation/qpos", "simulation/qvel",
                                   "simulation/actuator_control", "objects/cube/pose")}
    for _ in range(30):
        data.ctrl[:] = 0.
        for _ in range(5):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        records["time_s"].append(float(data.time))
        for key in ("qpos", "qvel"):
            records["simulation/" + key].append(getattr(data, key).copy())
        records["simulation/actuator_control"].append(data.ctrl.copy())
        body = model.body("object").id
        records["objects/cube/pose"].append(np.r_[data.xpos[body], data.xquat[body]])
    with h5py.File(tmp_path / "replay.h5", "w") as f:
        for key, values in records.items():
            f[key] = values
        for key, values in initial.items():
            f["initial/" + key] = values
        f["metrics/bilateral_contact"] = np.zeros(30)
        f["metrics/grasp_drift_m_rad"] = np.full((30, 2), np.nan)
        f["command/gripper_position"] = np.zeros(30)
    report = {"scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
              "plan": {"object_id": "cube", "source_start_s": 0.},
              "physics": {"objects": {"cube": {"body": "object", "joint": "object_joint"}}, "control_hz": 100},
              "simulated_s": .3, "recorded_frames": 30, "physics_validated": False,
              "gates": {"example": False}, "failure_reasons": ["example"]}
    (tmp_path / "result.json").write_text(json.dumps(report))
    return tmp_path


def test_contact_categories_respect_robot_and_active_subtrees():
    scopes = dict(robot={1, 2, 3}, hands={2, 3}, active={4, 5})
    assert contact_category([2, 5], **scopes) == "hand_object"
    assert contact_category([1, 4], **scopes) == "nonhand_object_robot"
    assert contact_category([0, 4], **scopes) == "object_environment"
    assert contact_category([2, 3], **scopes) == "robot_self"
    assert contact_category([2, 6], **scopes) == "robot_environment"
    assert contact_category([4, 5], **scopes) == "object_self"


def test_diagnosis_replays_real_contacts_and_preserves_all_inputs(tmp_path):
    path = _fixture(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    result = diagnose(path)
    assert result["replay"]["exact_within_1e7"]
    assert result["replay"]["max_qpos_error"] == 0
    assert result["initial"]["qvel_max_abs"] == .1
    assert result["initial"]["first_control_position_errors"]["r_hand_finger"] == 0
    contact = result["contacts"]["object_environment"]
    assert contact["first"]["time_s"] > 0
    assert set(contact["maximum_depth"]["geoms"]) == {"cube", "floor"}
    assert contact["maximum_depth"]["depth_m"] > 0
    assert result["recorded_contact_metrics"]["grasp_drift_available"] is False
    assert result["reported_physics_validated"] is False
    assert result["reported_gates"] == {"example": False}
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    json.dumps(result, allow_nan=False)


def test_diagnosis_rejects_changed_scene(tmp_path):
    path = _fixture(tmp_path)
    with (path / "scene.xml").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        diagnose(path)


def test_numeric_probe_preserves_reset_controls_and_contact_material(tmp_path):
    path = _fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    result = passive_prefix_convergence(path, duration_s=.15, timesteps=(.002, .001))
    assert len(result["variants"]) == 4
    poses = [row["initial_object_pose"] for row in result["variants"]]
    np.testing.assert_array_equal(poses, np.repeat([poses[0]], 4, axis=0))
    assert all(not row["robot_object_contact"] for row in result["variants"])
    assert all(row["max_object_environment_depth_m"] > 0 for row in result["variants"])
    assert all(len(row["object_pose_at_control_boundaries"]) == 15 for row in result["variants"])
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(ValueError, match="divide"):
        passive_prefix_convergence(path, timesteps=(.003,))


def test_passive_admission_is_separate_constant_initial_hold(tmp_path):
    path = _fixture(tmp_path)
    # Admission can extend past a stopped recording using its explicit hold
    # policy; it is never called an actuator replay of those unrecorded rows.
    result = passive_admission(path, duration_s=.4)
    assert result["completed"]
    assert result["isolated_passive_object"]
    assert result["initial_support_gap"]["distance_m"] == pytest.approx(.04)
    assert result["initial_nearest_environment"]["geoms"] == ["cube", "floor"]
    assert result["original_physics_validated"] is False
    assert result["duration_completed_s"] == pytest.approx(.4)


def test_compliance_sensitivity_changes_only_declared_global_contact_fields(tmp_path):
    path = _fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    result = global_compliance_sensitivity(path)
    assert result["source_faithful"] is False
    assert result["validation_result_promoted"] is False
    assert result["contact_override_used"] is False
    for row in result["variants"]:
        assert row["changed_model_array_fields"] == ["geom_solref"]
        assert row["unchanged_model_array_field_count"] > 100
        old = np.asarray(row["solref_changes"]["geom_solref"]["before"])
        new = np.asarray(row["solref_changes"]["geom_solref"]["after"])
        np.testing.assert_array_equal(new[:, 1], old[:, 1])
        np.testing.assert_array_equal(new[:, 0], np.minimum(old[:, 0], .004))
    assert result["variants"][0]["unchanged_model_arrays_sha256"] == result["variants"][1]["unchanged_model_arrays_sha256"]
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}


def test_touching_support_uses_contact_normal_for_zero_length_distance(tmp_path):
    result = passive_admission(_fixture(tmp_path, object_height=.02))
    assert result["initial_support_gap"]["distance_m"] == pytest.approx(0, abs=1e-10)
    assert result["initial_support_gap"]["gravity_alignment"] == pytest.approx(1)
