from copy import deepcopy
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget import placement_hand_targets as hand


def fixture(monkeypatch):
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
      <mesh name="cad" vertex="-.04 -.015 -.01 .04 -.015 -.01 .04 .015 -.01 -.04 .015 -.01 -.04 -.015 .01 .04 -.015 .01 .04 .015 .01 -.04 .015 .01"/>
      </asset><worldbody>
      <geom name="near_wall" type="box" pos="-.12 0 .8" size=".01 .3 .15"/>
      <geom name="far_wall" type="box" pos=".12 0 .8" size=".01 .3 .15"/>
      <geom name="visual_distractor" type="box" size="1 1 1" contype="0" conaffinity="0"/>
      <body name="base_link" pos="0 0 .8"><freejoint name="root"/>
       <body name="r_hand_palm_link" quat=".9238795325 0 0 .3826834324">
        <geom name="palm" type="mesh" mesh="cad" pos=".06 0 0" quat=".9659258263 0 .2588190451 0"/>
        <site name="r_arm_tip_tcp" pos=".01 0 .01" quat=".9238795325 .3826834324 0 0"/>
        <body name="finger" pos="0 .02 0"><joint name="finger" axis="0 0 1"/>
         <geom name="finger_geom" type="box" pos=".03 0 0" size=".025 .006 .008"/>
        </body>
       </body>
      </body>
      <body name="object" pos="0 0 .8"><freejoint name="object_joint"/><geom name="object_geom" type="sphere" size=".02"/></body>
      </worldbody></mujoco>''')
    def initialize(model, data, robot, q, mimics, apertures):
        data.qpos[:7] = q
        data.qpos[7] = apertures['r_hand_finger']
        mujoco.mj_forward(model, data)
    monkeypatch.setattr(hand, 'initialize', initialize)
    robot = SimpleNamespace(pack=lambda arms, base: model.qpos0[:7].copy())
    data = mujoco.MjData(model)
    initialize(model, data, robot, model.qpos0[:7], {}, {'r_hand_finger': .8})
    pose = np.eye(4); pose[:3, :3] = data.site_xmat[0].reshape(3, 3); pose[:3, 3] = data.site_xpos[0]
    source = [None]*12; source[0] = model; source[2] = dict(objects={'object':dict(body='object')}, mimics={})
    source[5] = dict(object_id='object', world_placement=np.eye(4),
        task_contract=dict(bin_lower=[-.109,-.2,.7], bin_upper=[.109,.2,.9]),
        pad_alignment=dict(inferred_contact_angle_rad=.9), effective_candidate=dict(gripper_closed_target=.8))
    objects = np.tile(np.eye(4), (4,1,1)); objects[:,2,3] = .8
    source[7:12] = [np.arange(4)*.01, np.zeros((4,17)), np.array([2.,-.06,-.06,2.]), np.tile(pose,(4,1,1)), objects]
    return source, robot, initialize


def test_compiled_mesh_and_articulated_hand_query_matches_real_forward_kinematics(monkeypatch):
    source, robot, initialize = fixture(monkeypatch)
    model = source[0]; original = model.qpos0.copy(); query = hand.CompiledHandFixtureQuery(source, robot, minimum_m=.05)
    direct = mujoco.MjData(model)
    for xyz, angles in [([.07,0,.8],[0,0,0]), ([-.05,.02,.82],[.1,-.2,.3]), ([0,-.03,.85],[-.4,.1,-.6])]:
        for aperture in (.8, 1.1, 1.4):
            quat = Rotation.from_euler('xyz',angles).as_quat()[[3,0,1,2]]
            initialize(model,direct,robot,np.r_[xyz,quat],{}, {'r_hand_finger':aperture})
            goal=np.eye(4);goal[:3,:3]=direct.site_xmat[0].reshape(3,3);goal[:3,3]=direct.site_xpos[0]
            expected=min(mujoco.mj_geomDistance(model,direct,int(a),int(b),.050001,None) for a,b in query.pairs)
            found=query.inspect(goal,[aperture])
            assert found['minimum_signed_distance_or_lower_bound_m'] == pytest.approx(expected,abs=2e-12)
    np.testing.assert_array_equal(model.qpos0,original)
    np.testing.assert_array_equal(query.data.qpos[8:],original[8:])
    assert query.data.time == 0.
    assert all('visual_distractor' != model.geom(b).name for _,b in query.pairs)
    assert all('object_geom' != model.geom(b).name for _,b in query.pairs)


def test_far_wall_is_checked_and_bounded_xy_shift_preserves_source(monkeypatch,tmp_path):
    source, robot, _ = fixture(monkeypatch)
    before=[v.copy() for v in source[7:12]]
    parameters=dict(placement_xy_world=[.075,0.],rotation_xyz_deg=[0.,0.,0.],extra_descent_m=0.,release_aperture_rad=1.4)
    query=hand.CompiledHandFixtureQuery(source,robot)
    failed=hand.screen(source,query,parameters,segment_samples=3)
    assert not failed['rigid_hand_screen_passed']
    assert failed['worst']['closest_geom_pair'][1]=='far_wall'
    candidates=[dict(id=9,parameters=parameters)]
    old=deepcopy(candidates)
    selected,report=hand.propose(source,robot,candidates,tmp_path/'proposals',segment_samples=5,xy_grid_count=3)
    assert selected and report['screened_count']>1
    assert selected[0]['parameters']['placement_xy_world'] != parameters['placement_xy_world']
    assert not report['full_robot_admitted'] and not report['physics_validated']
    assert candidates==old
    for a,b in zip(before,source[7:12]):np.testing.assert_array_equal(a,b)
    with np.load(tmp_path/'proposals/original-inputs.npz') as saved:
        np.testing.assert_array_equal(saved['object_targets'],source[11])
    assert json.loads((tmp_path/'proposals/result.json').read_text())['selected_ids']


def test_template_rejects_accidental_object_or_clock_mutation(monkeypatch):
    source, robot, initialize = fixture(monkeypatch)
    def bad(*args):
        initialize(*args)
        args[1].qpos[-1] += .1
    monkeypatch.setattr(hand,'initialize',bad)
    query=hand.CompiledHandFixtureQuery(source,robot)
    with pytest.raises(ValueError,match='modified an object'):
        query.inspect(source[10][0],[.8])


def test_invalid_cartesian_pose_or_outside_task_region_is_rejected(monkeypatch):
    source, robot, _=fixture(monkeypatch)
    query=hand.CompiledHandFixtureQuery(source,robot)
    reflection=source[10][0].copy();reflection[:3,0]*=-1
    with pytest.raises(ValueError,match='proper rigid'):
        query.inspect(reflection,[.8])
    with pytest.raises(ValueError,match='original task region'):
        hand.waypoints(source,dict(placement_xy_world=[2.,0.]))


def test_cartesian_screen_waypoints_equal_the_actual_placement_proposal(monkeypatch,tmp_path):
    from test_mobile_placement_ik import source_and_robot, fake_solver
    from reachy_retarget import supported_placement as placement, mobile_placement_ik
    source,robot=source_and_robot(monkeypatch)
    source[5].update(world_placement=np.eye(4),task_contract=dict(bin_lower=[-2.,-2.,-2.],bin_upper=[2.,2.,2.]))
    fake_solver(monkeypatch)
    def stop(*args,**kwargs):
        raise RuntimeError('Stop after independently generated Cartesian proposal')
    monkeypatch.setattr(mobile_placement_ik.Solver,'solve',stop)
    parameters=dict(placement_xy_world=[.02,.01],rotation_xyz_deg=[0.,40.,-130.],extra_descent_m=.002,mobile_ik={})
    expected=hand.waypoints(source,parameters)
    with pytest.raises(ValueError,match='IK rejected'):
        placement.prepare(source,robot,tmp_path/'actual',**parameters)
    with np.load(tmp_path/'actual/original-and-proposal.npz') as saved:
        np.testing.assert_array_equal(expected,saved['waypoints'][:3])
