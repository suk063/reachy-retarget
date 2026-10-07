"""Reference mapping and failure preservation without simulator/network access."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget import pose_retarget
from reachy_retarget.episodes import matrices_to_pose, pose_to_matrices
from reachy_retarget.agent_dataset import inspect_archive


def record():
    base=np.tile(np.eye(4),(3,1,1));base[:,:3,:3]=Rotation.from_euler('z',.4).as_matrix()
    base[:,:3,3]=[[2,3,.5],[2.1,3,.5],[2.2,3,.5]]
    hand=base.copy();hand[:,:3,3]+=[.3,-.2,.6]
    obj=hand.copy();obj[:,:3,3]+=[.04,0,0]
    return dict(arrays={'timestamp':np.array([0.,.05,.15]),'source/base':matrices_to_pose(base),
                        'source/hand':matrices_to_pose(hand),'objects/item/pose':matrices_to_pose(obj)},
                metadata={'source_sequence':'one','source_group':'original-one','source_urls':['https://example.invalid/pinned'],
                          'source_revision':'fixture','objects':{'item':{}},'simulation_assumptions':[]},missing_fields=[])


def test_one_world_gauge_preserves_object_relative_hand_path_and_clock():
    r=record();home=np.tile(np.eye(4),(2,1,1))
    base,targets,placement,attachments,active=pose_retarget.references(r['arrays'],home,'source/base',{'right':'source/hand'})
    assert active==(1,)
    np.testing.assert_allclose(base[0,:2,3],0,atol=1e-14)
    src=pose_to_matrices(r['arrays']['source/hand'])
    np.testing.assert_allclose(targets[:,1,:3,3],(placement@src)[:,:3,3])
    obj=placement@pose_to_matrices(r['arrays']['objects/item/pose'])
    np.testing.assert_allclose(np.linalg.norm(obj[:,:3,3]-targets[:,1,:3,3],axis=1),.04)
    np.testing.assert_allclose(targets[:,1,:3,:3],(placement@src)[:,:3,:3]@attachments[1])


def test_failed_ik_is_saved_as_derived_output_without_measured_observations(tmp_path,monkeypatch):
    from reachy_retarget import robot
    class FakeRobot:
        def __init__(self,*args):
            self.q=np.zeros(20);self.q[2]=1
            self.arm_v=np.arange(3,17);self.arm_ids=np.arange(4,18);self.identity={'fixture':'offline'}
        def fk(self,q):return np.tile(np.eye(4),(2,1,1))
        def ik(self,targets,q,active_hands,iterations):
            assert active_hands==(1,)
            return q.copy(),.4,.01
    monkeypatch.setattr(robot,'Robot',FakeRobot)
    r=record()
    result=pose_retarget.retarget(r,tmp_path,'fixture','trial',tmp_path/'attempt',clock_key='timestamp',
                                 base_key='source/base',hand_keys={'right':'source/hand'})
    assert result['status']=='kinematic_failed' and not result['physics_validated']
    report=inspect_archive(result['retarget_archive'])
    assert not any(key.startswith('observation/') for key in report['fields'])
    raw=np.load(tmp_path/'attempt/raw-ik.npz')
    np.testing.assert_equal(raw['original_clock_s'],r['arrays']['timestamp'])
    assert (tmp_path/'attempt/raw-ik.json').exists()


def test_invalid_quaternion_cannot_silently_become_a_target():
    r=record();r['arrays']['source/hand'][0,3:]=0
    with pytest.raises(ValueError,match='quaternion'):
        pose_retarget.references(r['arrays'],np.tile(np.eye(4),(2,1,1)),'source/base',{'right':'source/hand'})
