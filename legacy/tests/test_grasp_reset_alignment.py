import json
import numpy as np
import pytest
import mujoco
from reachy_retarget.grasp_reset_alignment import apply


def prepared():
    objects=np.repeat(np.eye(4)[None],4,axis=0)
    objects[:,0,3]=[0.,.004,.009,.010]
    hands=objects.copy();hands[0,2,3]=.1
    commands=np.array([2.,2.,-.06,-.06])
    pad={'pad':{'midpoint_tcp':[0.,0.,0.]},'section':{'center_object':[0.,0.,0.]}}
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="cube"><freejoint/><geom size=".02"/></body></worldbody></mujoco>')
    return (model,None,{'objects':{'cube':{'body':'cube'}}},None,None,{'pad_alignment':pad,'object_id':'cube'},None,np.arange(4)*.01,
            None,commands,hands,objects)


def test_unrealized_pregrasp_object_motion_corrects_only_robot_path(tmp_path):
    original=prepared();saved=original[10].copy()
    result=apply(original,tmp_path/'alignment')
    # At acquisition the source object has moved9mm but the physical object
    # is still at its reset pose. The translated robot grasps that reset center.
    np.testing.assert_allclose(result[10][2,:3,3],original[11][0,:3,3])
    np.testing.assert_allclose(np.diff(result[10][:,:3,3],axis=0),np.diff(saved[:,:3,3],axis=0))
    np.testing.assert_array_equal(original[10],saved)
    for i in (7,9,11):assert result[i] is original[i]
    assert result[5]['grasp_reset_alignment']['requires_full_ik_and_collision_admission']


def test_excessive_reset_discrepancy_is_saved_and_rejected(tmp_path):
    original=prepared()
    with pytest.raises(ValueError,match='rejected'):
        apply(original,tmp_path/'rejected',max_translation_m=.005)
    report=json.loads((tmp_path/'rejected/result.json').read_text())
    assert report['admitted'] is False
    assert (tmp_path/'rejected/reset-alignment.npz').exists()


def test_source_initial_pose_must_match_the_unchanged_physical_reset(tmp_path):
    original=prepared();original[11][0,0,3]=.01
    with pytest.raises(ValueError,match='rejected'):apply(original,tmp_path/'mismatch')
    assert json.loads((tmp_path/'mismatch/result.json').read_text())['reference_reset_center_error_m']==.01
