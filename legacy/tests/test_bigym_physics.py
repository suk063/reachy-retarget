"""Task-scene safety and success semantics without network/native packages."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.native_replay.bigym_physics import _remove_source_robot, task_success, timing_plan, simulation_scene


def test_removal_preserves_free_plate_and_nonrobot_constraints():
    root = ET.fromstring('''<mujoco><worldbody><body name="h1/"><joint name="h1/base"/></body>
        <body name="plate/"><freejoint name="plate/"/></body><body name="table/"/></worldbody>
        <actuator><position name="h1/arm" joint="h1/arm"/></actuator>
        <equality><joint joint1="h1/left" joint2="h1/right"/><joint joint1="fixture_a" joint2="fixture_b"/></equality>
        <contact><exclude body1="h1/one" body2="h1/two"/><exclude body1="fixture_a" body2="fixture_b"/></contact></mujoco>''')
    _remove_source_robot(root)
    assert root.find("worldbody/body[@name='h1/']") is None
    assert root.find("worldbody/body[@name='plate/']/freejoint").get("name") == "plate/"
    assert len(root.find("actuator")) == 0
    assert [e.get("joint1") for e in root.find("equality")] == ["fixture_a"]
    assert [e.get("body1") for e in root.find("contact")] == ["fixture_a"]


def success_fixture():
    rotation = Rotation.from_euler("x", np.pi/2).as_matrix()
    model = SimpleNamespace(body=lambda n: SimpleNamespace(id=0), site=lambda n: SimpleNamespace(id=int(n)))
    data = SimpleNamespace(xpos=np.array([[.2, -.3, 1.1]]), xmat=np.array([rotation.ravel()]),
                           site_xpos=np.array([[.2, -.3, 1.1], [.2, -.4, 1.1]]))
    contact = dict(plate_target_contact=True, plate_table_contact=False, plate_floor_contact=False, plate_pad_contact=False)
    return model, data, contact


def test_native_task_predicate_requires_release_support_orientation_and_position():
    model, data, contact = success_fixture()
    assert task_success(model, data, ["0", "1"], np.eye(4), contact)["success"]
    for key, value in (("plate_target_contact", False), ("plate_table_contact", True), ("plate_pad_contact", True)):
        assert not task_success(model, data, ["0", "1"], np.eye(4), dict(contact, **{key: value}))["success"]
    data.xpos[0, 0] += .051
    assert not task_success(model, data, ["0", "1"], np.eye(4), contact)["success"]
    data.xpos[0, 0] -= .051
    data.xmat[0] = np.eye(3).ravel()
    assert not task_success(model, data, ["0", "1"], np.eye(4), contact)["success"]


def test_task_orientation_predicate_uses_original_world_after_shared_gauge():
    model, data, contact = success_fixture()
    gauge = np.eye(4); gauge[:3, :3] = Rotation.from_euler("z", .9).as_matrix()
    data.xmat[0] = (gauge[:3, :3] @ data.xmat[0].reshape(3, 3)).ravel()
    assert task_success(model, data, ["0", "1"], gauge, contact)["success"]
    assert not task_success(model, data, ["0", "1"], np.eye(4), contact)["success"]


def test_interval_local_timing_preserves_every_source_boundary_and_speed_budget():
    clock = np.array([1., 1.002, 1.004, 1.008])
    q = np.zeros((4, 6)); q[:, 2] = 1
    q[:, 4] = [0., .0001, .0081, .0082]
    original = q.copy()
    plan = timing_plan(clock, q, [4, 5], .002, arm_speed_rad_s=.8)
    np.testing.assert_array_equal(plan["substeps_per_source_interval"], [1, 5, 2])
    np.testing.assert_array_equal(plan["source_clock_at_boundary"][plan["source_boundary_step_index"]], clock)
    assert np.all(np.diff(plan["source_clock_at_boundary"]) > 0)
    target = np.vstack([q[0], plan["target_qpos"]])
    np.testing.assert_allclose(target[plan["source_boundary_step_index"]], q)
    assert np.abs(np.diff(target[:, 4:6], axis=0)/.002).max() <= .80000001
    np.testing.assert_array_equal(q, original)
    assert plan["summary"]["derived_duration_s"] == pytest.approx(.016)


def test_planar_yaw_unwraps_but_bounded_arm_coordinates_do_not():
    q = np.zeros((2, 5)); yaw = np.array([np.pi-.01, -np.pi+.01])
    q[:, 2], q[:, 3] = np.cos(yaw), np.sin(yaw)
    q[:, 4] = [3., -3.]
    plan = timing_plan([0., .002], q, [4], .002, arm_speed_rad_s=1.)
    assert plan["summary"]["derived_duration_s"] == pytest.approx(6.)
    target = np.vstack([q[0], plan["target_qpos"]])
    assert np.abs(np.diff(target[:, 4])/.002).max() <= 1.000001
    assert np.ptp(np.unwrap(np.arctan2(target[:, 3], target[:, 2]))) == pytest.approx(.02)


def test_unretimed_plan_keeps_native_clock_and_rejects_nonphysical_boundaries():
    q = np.zeros((3, 5)); q[:, 2] = 1
    clock = [.002, .004, .006]
    plan = timing_plan(clock, q, [4], .002)
    np.testing.assert_allclose(plan["physical_clock"], clock)
    assert plan["summary"]["enabled"] is False
    with pytest.raises(ValueError, match="boundaries"):
        timing_plan([0., .003, .006], q, [4], .002)
    with pytest.raises(ValueError, match="budget"):
        timing_plan(clock, q, [4], .002, arm_speed_rad_s=1., max_steps=1)


def test_declared_interval_grid_preserves_robot_timing_across_numerical_steps():
    q = np.zeros((3, 5)); q[:, 2] = 1; q[:, 4] = [0., .0038, .0045]
    coarse = timing_plan([0., .002, .004], q, [4], .002, arm_speed_rad_s=.8)
    fine = timing_plan([0., .002, .004], q, [4], .001, arm_speed_rad_s=.8, interval_grid_s=.002)
    assert coarse["summary"]["derived_duration_s"] == fine["summary"]["derived_duration_s"]
    np.testing.assert_allclose(coarse["target_qpos"], fine["target_qpos"][1::2])
    np.testing.assert_array_equal(fine["source_boundary_step_index"], 2*coarse["source_boundary_step_index"])
    with pytest.raises(ValueError, match="grid"):
        timing_plan([0., .002, .004], q, [4], .002, interval_grid_s=.001)


def test_alternative_compliance_retains_original_scene_and_object_physical_arrays(tmp_path):
    import hashlib
    import mujoco
    xml = '''<mujoco><option timestep=".002"/><worldbody>
      <geom name="floor" type="plane" size="2 2 .1" solref=".02 1"/>
      <body name="plate" pos="0 0 1"><freejoint name="plate"/>
      <geom name="plate_geom" type="box" size=".1 .1 .003" mass=".4" friction=".7 .005 .001"/>
      </body></worldbody></mujoco>'''
    scene = tmp_path/"scene.xml"; scene.write_text(xml)
    sha = hashlib.sha256(scene.read_bytes()).hexdigest()
    original = mujoco.MjModel.from_xml_path(str(scene))
    model, manifest = simulation_scene(original, {"scene_path": str(scene), "scene_sha256": sha}, tmp_path, "global-contact-4ms")
    assert (tmp_path/"source-contact-scene.xml").read_text() == xml
    assert manifest["source_contact_scene_sha256"] == sha
    assert manifest["scene_sha256"] != sha
    assert manifest["simulation_profile"]["source_fidelity"] is False
    assert model.opt.timestep == .001
    for field in ("body_mass", "body_inertia", "geom_friction", "geom_size", "geom_pos", "qpos0"):
        np.testing.assert_array_equal(getattr(model, field), getattr(original, field))
    np.testing.assert_allclose(model.geom_solref[:, 0], .004)
    with pytest.raises(ValueError, match="declared"):
        simulation_scene(original, {"scene_path": str(scene)}, tmp_path, "undeclared-contact")


def test_calibrated_gripper_control_preserves_source_intent_and_other_hand():
    from reachy_retarget.native_replay.bigym_physics import gripper_control_policy
    original = {'left':np.array([2.,-.06,-.06,2.]),'right':np.full(4,2.)}
    plan = {'pad_calibration':{'hands':{'left':{'nominal_gripper_rad':.2456865}}}}
    policy = dict(method='cad_contact_minus_margin', closure_margin_rad={'left':.02},
                  evidence={'cad_geometry':'test fixture'})
    targets, report = gripper_control_policy(original,plan,policy)
    np.testing.assert_allclose(targets['left'],[2.,.2256865,.2256865,2.])
    np.testing.assert_array_equal(original['left'],[2.,-.06,-.06,2.])
    np.testing.assert_array_equal(targets['right'],original['right'])
    assert report['source_intent_preserved']
    for margin in (0.,np.nan,.11):
        with pytest.raises(ValueError):
            gripper_control_policy(original,plan,dict(policy,closure_margin_rad={'left':margin}))
    with pytest.raises(ValueError,match='evidence'):
        gripper_control_policy(original,plan,dict(policy,evidence={}))
    with pytest.raises(ValueError,match='grasping hand'):
        gripper_control_policy({'left':np.full(4,2.)},plan,policy)


def test_inactive_hand_guard_reads_actual_solved_forces_without_mutating_state():
    import mujoco
    from reachy_retarget.native_replay.bigym_physics import InactiveHandContactAudit
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/>
      <worldbody><body name="base_link"><body name="l_hand_palm" pos="3 0 .3"><geom type="sphere" size=".08"/></body>
      <body name="r_hand_palm" pos="0 0 .3"><geom type="sphere" size=".08"/></body></body>
      <body name="plate/" pos="0 0 .2"><freejoint/><geom name="plate_geom" type="box" size=".05 .05 .05" mass=".4"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model);audit=InactiveHandContactAudit(model,('left',))
    mujoco.mj_step(model,data)
    before=[v.copy() for v in (data.qpos,data.qvel,data.ctrl)]
    audit.sample(data)
    for value,copy in zip((data.qpos,data.qvel,data.ctrl),before):np.testing.assert_array_equal(value,copy)
    report=audit.finish(1)
    assert report['complete'] and not report['passed']
    assert report['positive_force_contacts']>0
    assert report['events'][0]['side']=='right'
    assert not audit.finish(2)['complete']
    both=InactiveHandContactAudit(model,('left','right'));both.sample(data)
    assert both.finish(1)['passed']


def test_arm_feedforward_bounds_actual_position_commands_and_preserves_references():
    import mujoco
    from reachy_retarget.native_replay.bigym_physics import bounded_arm_feedforward
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><option timestep=".002"/><worldbody>
      <body><joint name="arm" range="-1 1"/><geom type="sphere" size=".05"/></body></worldbody>
      <actuator><position name="arm" joint="arm" kp="180" kv="24"/></actuator></mujoco>''')
    t=np.arange(1,501)*.002; reference=(.5*t)[:,None]; before=reference.copy()
    policy=dict(method='bounded_velocity',scale=1.,max_offset_rad=.2,
                max_command_speed_rad_s=1.,joint_margin_rad=.025,boundary_taper_s=.1,evidence={'measured_lag':'test'})
    commands,report=bounded_arm_feedforward(model,['arm'],reference,t,[0.],policy)
    np.testing.assert_array_equal(reference,before)
    np.testing.assert_allclose(commands[100:300]-reference[100:300],24/180*.5,atol=1e-12)
    assert report['maximum_command_rate_rad_s']<=1.+1e-10
    assert report['maximum_applied_offset_rad']<=.2
    assert np.all(commands>=-1+.025) and np.all(commands<=1-.025)
    assert report['reference_clock_unchanged'] and report['source_gripper_intent_unchanged']
    baseline_data, corrected_data = mujoco.MjData(model), mujoco.MjData(model)
    for index in range(350):
        baseline_data.ctrl[:]=reference[index]
        corrected_data.ctrl[:]=commands[index]
        mujoco.mj_step(model,baseline_data);mujoco.mj_step(model,corrected_data)
    assert abs(corrected_data.qpos[0]-reference[349,0]) < .2*abs(baseline_data.qpos[0]-reference[349,0])
    unchanged,disabled=bounded_arm_feedforward(model,['arm'],reference,t,[0.])
    np.testing.assert_array_equal(unchanged,reference)
    assert not disabled['enabled']
    near_limit=.9+.074*t[:,None]
    clamped,details=bounded_arm_feedforward(model,['arm'],near_limit,t,[.9],policy)
    assert clamped.max()<=.975 and details['position_clamps']>0
    for invalid in (dict(policy,evidence={}),dict(policy,max_command_speed_rad_s=2.),dict(policy,scale=float('nan'))):
        with pytest.raises(ValueError):bounded_arm_feedforward(model,['arm'],reference,t,[0.],invalid)
    with pytest.raises(ValueError,match='margin and rate'):
        bounded_arm_feedforward(model,['arm'],reference*3,t,[0.],policy)


def test_post_acquisition_compensation_tapers_from_zero_and_is_pause_invariant():
    import mujoco
    from reachy_retarget.native_replay.bigym_physics import resumed_arm_feedforward
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body><joint name="arm" range="-2 2"/>
        <geom type="sphere" size=".05"/></body></worldbody>
      <actuator><position name="arm" joint="arm" kp="180" kv="24"/></actuator></mujoco>''')
    t = np.arange(1, 501)*.002
    reference = (.2+.5*t)[:, None]
    saved = reference.copy()
    policy = dict(method='bounded_velocity', activation='after_measured_acquisition',
                  evidence={'post_resume_lag': 'measured fixture'}, scale=.5,
                  boundary_taper_s=.1, max_command_speed_rad_s=1.)
    commands, report = resumed_arm_feedforward(model, ['arm'], reference, t, [.2], policy)
    delayed, _ = resumed_arm_feedforward(model, ['arm'], reference, t+37.5, [.2], policy)
    np.testing.assert_allclose(commands, delayed, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(commands[0], reference[0])
    np.testing.assert_allclose(commands[100:300]-reference[100:300], .5*24/180*.5)
    np.testing.assert_array_equal(reference, saved)
    assert not report['entry_settle_close_compensation']
    assert report['first_resumed_offset_rad'] == [0.]
    with pytest.raises(ValueError, match='post-acquisition'):
        resumed_arm_feedforward(model, ['arm'], reference, t, [.2], dict(policy, activation='always'))


def test_post_acquisition_policy_requires_isolated_measured_rendezvous():
    from reachy_retarget.native_replay.bigym_physics import rollout
    policy = {'activation': 'after_measured_acquisition'}
    with pytest.raises(ValueError, match='isolated measured rendezvous'):
        rollout(None, None, None, None, None, None, post_acquisition_arm_feedforward_policy=policy)
    with pytest.raises(ValueError, match='isolated measured rendezvous'):
        rollout(None, None, None, None, None, None, acquisition_policy={},
                arm_feedforward_policy={}, post_acquisition_arm_feedforward_policy=policy)
