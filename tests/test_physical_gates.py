import copy
import hashlib

import mujoco
import numpy as np
import pytest

from reachy_retarget.dynamics_audit import bind_scene_assets
from reachy_retarget.physical_gates import GateObserver, verify_actuator_replay


XML = """<mujoco><option timestep=".002" gravity="0 0 -9.81"/>
<worldbody><geom name="floor" type="plane" size="3 3 .1"/>
 <body name="base_link" pos="0 0 .4"><geom name="base" type="sphere" size=".05"/>
  <body name="left" pos=".3 .2 0"><joint name="l_shoulder_pitch" axis="0 1 0" limited="true" range="-57.2957795 57.2957795"/>
   <geom name="left_arm" type="sphere" size=".03"/>
   <site name="l_arm_tip_tcp" pos=".15 0 0"/>
   <body name="l_hand_distal_link" pos=".15 -.022 0"><geom name="pad1" type="box" size=".04 .003 .04"/></body>
   <body name="l_hand_distal_mimic_link" pos=".15 .022 0"><geom name="pad2" type="box" size=".04 .003 .04"/></body>
  </body>
  <body name="right" pos=".3 -.2 0"><joint name="r_shoulder_pitch" axis="0 1 0" limited="true" range="-57.2957795 57.2957795"/>
   <geom name="right_arm" type="sphere" size=".03"/><site name="r_arm_tip_tcp"/>
   <body name="r_hand_distal_link" pos="0 -.05 0"><geom type="sphere" size=".01"/></body>
   <body name="r_hand_distal_mimic_link" pos="0 .05 0"><geom type="sphere" size=".01"/></body>
  </body>
 </body>
 <body name="plate/" pos=".45 .2 .4"><freejoint name="plate/"/>
  <body name="plate/plate"><geom name="plate_collision" type="box" size=".02 .02 .02" mass=".02"/></body>
 </body>
</worldbody><actuator><position name="left" joint="l_shoulder_pitch" kp="10"/>
<position name="right" joint="r_shoulder_pitch" kp="10"/></actuator></mujoco>"""


def sync(model, data):
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    mujoco.mj_fwdActuation(model, data)


def fixture(tmp_path):
    scene = tmp_path/"scene.xml"
    scene.write_text(XML)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return scene, model, data


def clearance(model, data):
    return mujoco.mj_geomDistance(model, data, model.geom("left_arm").id,
                                 model.geom("right_arm").id, 1., None)


def make_observer(scene, model, data, **kwargs):
    return GateObserver(model, data, active_object_body="plate/", grasp_hands=("left",),
        arm_joint_names=("l_shoulder_pitch", "r_shoulder_pitch"),
        scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(), **kwargs)


def test_subtree_contacts_real_state_read_only_and_missing_grasp_never_pass(tmp_path, monkeypatch):
    scene, model, data = fixture(tmp_path)
    observer = make_observer(scene, model, data, initial_self_clearance_m=clearance(model, data))
    mujoco.mj_step(model, data)
    contacts = observer.sample_contacts(data)
    assert contacts["depths"]["hand_object"] > .0009
    assert contacts["bilateral"]["left"]
    sync(model, data)
    before = {key: getattr(data, key).copy() for key in ("qpos", "qvel", "ctrl", "xpos", "xmat", "cvel")}
    def forbidden(*args, **kwargs):
        raise AssertionError("Gate collection must not advance or reset state")
    for name in ("mj_step", "mj_forward", "mj_setState", "mj_resetData"):
        monkeypatch.setattr(mujoco, name, forbidden)
    observer.update(data, contact_sample=contacts, grasp_intent={"left": True},
                    task_satisfied=False, self_clearance_m=clearance(model, data))
    report = observer.finish(expected_steps=1, expected_final_time_s=.002)
    assert not report["physics_validated"]
    assert not report["validation_complete"]
    assert "independent_actuator_replay" in report["missing_fields"]
    assert report["grasp_hands"]["left"]["grasp_translation_m"] is None
    assert not report["gates"]["grasp_translation_3mm"]
    assert report["gates"]["rollout_complete"]
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(data, key), value)


def test_measured_speed_margin_and_absent_clearance_are_real_failed_gates(tmp_path):
    scene, model, data = fixture(tmp_path)
    q = int(model.jnt_qposadr[model.joint("r_shoulder_pitch").id])
    v = int(model.jnt_dofadr[model.joint("r_shoulder_pitch").id])
    data.qpos[q], data.qvel[v] = .99, 2.
    mujoco.mj_forward(model, data)
    observer = make_observer(scene, model, data)
    mujoco.mj_step(model, data)
    contacts = observer.sample_contacts(data)
    sync(model, data)
    observer.update(data, contact_sample=contacts, grasp_intent={"left": False}, task_satisfied=None)
    report = observer.finish(expected_steps=1, expected_final_time_s=.002,
                             actuator_replay={"actuator_replay_pass": True, "scene_sha256": "other"})
    assert not report["gates"]["actual_arm_speed_1rad_s"]
    assert not report["gates"]["joint_margin_25mrad"]
    assert not report["gates"]["self_clearance_9mm"]
    assert {"measured_self_clearance", "native_task_predicate", "matching_collector_replay_scene_identity"} <= set(report["missing_fields"])
    assert not report["physics_validated"]


def test_rejects_skipping_physics_steps(tmp_path):
    scene, model, data = fixture(tmp_path)
    observer = make_observer(scene, model, data)
    mujoco.mj_step(model, data)
    mujoco.mj_step(model, data)
    contacts = observer.sample_contacts(data)
    sync(model, data)
    with pytest.raises(ValueError, match="every physics step"):
        observer.update(data, contact_sample=contacts, grasp_intent={"left": False}, task_satisfied=False)


def test_cannot_omit_inactive_arm_or_use_truthy_predicate_dictionary(tmp_path):
    scene, model, data = fixture(tmp_path)
    with pytest.raises(ValueError, match="Both arms"):
        GateObserver(model, data, active_object_body="plate/", grasp_hands=("left",),
                     arm_joint_names=("l_shoulder_pitch",))
    observer = make_observer(scene, model, data)
    mujoco.mj_step(model, data)
    contacts = observer.sample_contacts(data)
    sync(model, data)
    with pytest.raises(ValueError, match="explicit boolean"):
        observer.update(data, contact_sample=contacts, grasp_intent={"left": False},
                        task_satisfied={"success": False})
    with pytest.raises(ValueError, match="explicit per-hand booleans"):
        observer.update(data, contact_sample=contacts, grasp_intent={"left": "open"},
                        task_satisfied=False)
    assert observer.steps == 0


def recording(model, data, count=30):
    initial = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), timestamp=np.array(data.time))
    arrays = {key: [] for key in ("timestamp", "observation/qpos", "observation/qvel", "physics_state",
                                  "command/joint_position", "control_interval_s", "observation/objects/plate/pose")}
    body = model.body("plate/").id
    for i in range(count):
        data.ctrl[:] = [.001*i, -.002*i]
        sync(model, data)
        state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        arrays["timestamp"].append(data.time)
        arrays["observation/qpos"].append(data.qpos.copy())
        arrays["observation/qvel"].append(data.qvel.copy())
        arrays["physics_state"].append(state)
        arrays["command/joint_position"].append(data.ctrl.copy())
        arrays["control_interval_s"].append(model.opt.timestep)
        arrays["observation/objects/plate/pose"].append(np.r_[data.xpos[body], data.xquat[body]])
        mujoco.mj_step(model, data)
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    sync(model, data)
    arrays.update({"terminal/timestamp": np.array(data.time), "terminal/observation/qpos": data.qpos.copy(),
                   "terminal/observation/qvel": data.qvel.copy(),
                   "terminal/observation/objects/plate/pose": np.r_[data.xpos[body], data.xquat[body]]})
    return initial, arrays


def verify(scene, arrays, initial):
    return verify_actuator_replay(scene, arrays,
        expected_scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        expected_assets=bind_scene_assets(scene.read_text(), scene.parent),
        active_object_body="plate/", active_object_joint="plate/", object_pose_key="observation/objects/plate/pose",
        initial_state=initial)


def test_exact_actuator_replay_checks_every_row_terminal_and_input_immutability(tmp_path):
    scene, model, data = fixture(tmp_path)
    initial, arrays = recording(model, data)
    before = copy.deepcopy(arrays)
    result = verify(scene, arrays, initial)
    assert result["actuator_replay_pass"]
    assert max(result["max_absolute_errors"].values()) == 0
    assert result["physical_steps"] == 30 and result["compared_terminal"]
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])
    arrays["command/joint_position"][5, 0] += .1
    assert not verify(scene, arrays, initial)["actuator_replay_pass"]


def test_replay_rejects_missing_boundary_and_object_assistance(tmp_path):
    scene, model, data = fixture(tmp_path)
    initial, arrays = recording(model, data)
    bad = dict(arrays)
    bad.pop("terminal/timestamp")
    with pytest.raises(ValueError, match="Missing exact replay"):
        verify(scene, bad, initial)
    bad = dict(arrays, control_interval_s=np.full(30, .003))
    with pytest.raises(ValueError, match="intervals"):
        verify(scene, bad, initial)
    scene.write_text(XML.replace('<body name="plate/"', '<body name="plate/" gravcomp="1"'))
    with pytest.raises(ValueError, match="gravity compensation"):
        verify(scene, arrays, initial)
