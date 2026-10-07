"""Actual compiled-mesh query parity without changing simulated object state."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.carried_object_clearance import CarriedObjectQuery,measured_attachment_bound


def fixture():
    vertices=' '.join(str(x) for a in (-.03,.03) for b in (-.02,.02) for c in (-.04,.04) for x in (a,b,c))
    xml=f'''<mujoco><asset><mesh name="box" vertex="{vertices}"/></asset><worldbody>
      <geom name="floor" type="box" size="1 1 .01" pos="0 0 -.01"/>
      <geom name="wall" type="box" size=".01 1 .5" pos=".15 0 .5"/>
      <body name="base_link" pos="-2 0 0"><geom type="sphere" size=".1"/></body>
      <body name="object" pos="0 0 .4" quat=".9238795325 0 0 .3826834324"><freejoint/>
        <body pos=".007 .003 0" quat=".984807753 0 .173648178 0">
          <geom name="object_geom" type="mesh" mesh="box" quat=".965925826 .258819045 0 0"/>
        </body>
      </body></worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    root=model.body('object').id;pose=np.eye(4);pose[:3,:3]=data.xmat[root].reshape(3,3);pose[:3,3]=data.xpos[root]
    hands=np.tile(np.eye(4),(3,1,1));objects=np.repeat(pose[None],3,axis=0)
    prepared=(model,xml,{'objects':{'item':{'body':'object'}}},{},{},{'object_id':'item'},None,
              np.arange(3)*.01,np.zeros((3,17)),np.array([2.,-.06,2.]),hands,objects)
    return prepared,data


def test_compiled_rotated_mesh_and_child_transform_match_real_fk_without_state_mutation():
    source,_=fixture();model=source[0];q0=model.qpos0.copy();mesh=model.mesh_vert.copy()
    query=CarriedObjectQuery(source,support_geoms=['floor'])
    goal=np.eye(4);goal[:3,:3]=Rotation.from_euler('xyz',[.1,-.2,.3]).as_matrix();goal[0,3]=.21
    found=query.inspect(goal)
    expected=goal@query.hand_to_object;data=mujoco.MjData(model)
    # Independent offline FK reconstruction used only as the numerical oracle.
    data.qpos[:3]=expected[:3,3];quat=Rotation.from_matrix(expected[:3,:3]).as_quat();data.qpos[3:7]=quat[[3,0,1,2]]
    mujoco.mj_forward(model,data)
    distance=mujoco.mj_geomDistance(model,data,model.geom('object_geom').id,model.geom('wall').id,1.,None)
    assert distance<0 and not found['admitted']
    assert found['minimum_signed_distance_or_lower_bound_m']==pytest.approx(distance,abs=1e-12)
    np.testing.assert_array_equal(query.data.qpos,q0);np.testing.assert_array_equal(query.data.qvel,0.)
    np.testing.assert_array_equal(model.qpos0,q0);np.testing.assert_array_equal(model.mesh_vert,mesh)
    assert query.data.time==0.


def test_only_declared_horizontal_support_is_exempt_and_only_when_explicit():
    source,_=fixture();query=CarriedObjectQuery(source,support_geoms=['floor'])
    goal=np.eye(4);goal[2,3]=-.4
    assert not query.inspect(goal)['admitted']
    assert query.inspect(goal,allow_support=True)['admitted']
    goal[0,3]=.13
    assert not query.inspect(goal,allow_support=True)['admitted']
    with pytest.raises(ValueError,match='horizontal floor'):
        CarriedObjectQuery(source,support_geoms=['wall'])
    with pytest.raises(ValueError,match='Explicit intended support'):
        CarriedObjectQuery(source,support_geoms=[])


def test_measured_attachment_bound_covers_every_surface_point_and_rejects_ineligible_window():
    ideal=np.eye(4);ideal[:3,3]=[.01,.02,.03]
    hands=np.tile(np.eye(4),(2,1,1));hands[1,:3,:3]=Rotation.from_euler('xyz',[.4,-.2,.8]).as_matrix()
    hands[1,:3,3]=[1.,-2.,.5]
    attachments=np.tile(ideal,(2,1,1));attachments[0,:3,3]+=[.01,0.,0.]
    attachments[1,:3,:3]=Rotation.from_euler('z',.15).as_matrix();attachments[1,:3,3]+=[-.003,.002,0.]
    objects=hands@attachments;before=objects.copy()
    eligible={key:np.ones(2,bool) for key in ['bilateral_lifted_grasp','no_fixture_contact','pre_placement']}
    settings=dict(frame_indices=np.array([10,11]),sample_times_s=[.1,.11],object_radius_m=.05,
                  eligibility=eligible,provenance={'archive_sha256':'offline-test'})
    report=measured_attachment_bound(hands,objects,ideal,**settings)
    points=np.random.default_rng(2).normal(size=(100,3));points=.05*points/np.linalg.norm(points,axis=1)[:,None]
    original=points@ideal[:3,:3].T+ideal[:3,3]
    for i,attachment in enumerate(attachments):
        actual=points@attachment[:3,:3].T+attachment[:3,3]
        assert np.linalg.norm(actual-original,axis=1).max()<=report['observed_mesh_displacement_bound_m'][i]+1e-14
    assert report['translation_error_m'][0]==pytest.approx(.01)
    np.testing.assert_allclose(report['measured_hand_to_object'],attachments,atol=1e-15)
    np.testing.assert_array_equal(objects,before)
    eligible['no_fixture_contact'][1]=False
    with pytest.raises(ValueError,match='eligibility conditions'):
        measured_attachment_bound(hands,objects,ideal,**settings)
