"""A derived start window may not hide source interaction or moving state."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_retarget.dynamics import source_window


def source():
    return {"time_s": np.arange(10)*.05,
            "source/action": np.tile([0.,0.,0.,0.,0.,0.,-1.],(10,1)),
            "objects/cube/pose": np.tile([.1,.2,.8,1.,0.,0.,0.],(10,1))}


def test_stationary_window_keeps_original_poses_and_source_unchanged():
    a=source(); before={k:v.copy() for k,v in a.items()}
    window,start,original_time=source_window(a,{},.2)
    assert start==4 and window['time_s'][0]==0
    np.testing.assert_array_equal(window['objects/cube/pose'],before['objects/cube/pose'][4:])
    np.testing.assert_array_equal(original_time,before['time_s'])
    for k in a: np.testing.assert_array_equal(a[k],before[k])


def test_crop_cannot_omit_closed_gripper():
    a=source();a['source/action'][1,6]=1
    with pytest.raises(ValueError,match='omit demonstrated grasp'):
        source_window(a,{},.2)


def test_crop_requires_stationary_object_translation():
    a=source();a['objects/cube/pose'][4,0]+=.01
    with pytest.raises(ValueError,match='stationary'):
        source_window(a,{},.2)


def test_crop_requires_stationary_object_rotation():
    a=source();q=Rotation.from_euler('z',.2).as_quat();a['objects/cube/pose'][4,3:]=q[[3,0,1,2]]
    with pytest.raises(ValueError,match='rotationally stationary'):
        source_window(a,{},.2)


@pytest.mark.parametrize('start',[-.1,.5,float('nan')])
def test_invalid_source_window_rejected(start):
    with pytest.raises(ValueError,match='Invalid source interval'):
        source_window(source(),{},start)


def test_t_plus_one_crop_is_explicitly_unsupported():
    with pytest.raises(ValueError,match=r'T\+1'):
        source_window(source(),{'source_format':'ManiSkill-HDF5'},.2)
