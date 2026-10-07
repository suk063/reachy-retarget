import numpy as np
from reachy_retarget.feasible_timing import segment_durations, prepare


def test_diagonal_base_motion_respects_rotation_invariant_speed_norm():
    t=np.arange(301)*.01;q=np.zeros((301,17))
    q[100:201,:2]=np.linspace(0,.4,101)[:,None];q[201:,:2]=.4
    component_caps=np.r_[.55,.55,1.5,np.full(14,.8)]
    old=segment_durations(q,t,component_caps)
    assert np.max(np.linalg.norm(np.diff(q[:,:2],axis=0)/old[:,None],axis=1))>.55
    bounded=segment_durations(q,t,component_caps,planar_speed_m_s=.45)
    assert np.max(np.linalg.norm(np.diff(q[:,:2],axis=0)/bounded[:,None],axis=1))<=.45+1e-12
    np.testing.assert_allclose(bounded[:70],.01)
    np.testing.assert_array_equal(q[-1,:2],[.4,.4])


def test_local_retiming_bounds_speed_and_does_not_stretch_distant_idle_segments():
    t=np.arange(500)*.01;q=np.zeros((500,1));q[200:300,0]=np.linspace(0,.2,100);q[300:,0]=.2
    duration=segment_durations(q,t,np.array([.1]))
    v=np.diff(q,axis=0)/duration[:,None]
    assert np.max(abs(v))<=.1+1e-8
    np.testing.assert_allclose(duration[:160],.01)
    assert duration.sum()<12 # Local slowdown does not triple the entire path.


def test_retiming_retains_full_source_interval_and_task_contact_phase(tmp_path):
    t=np.arange(100)*.01;q=np.zeros((100,17));q[:,3]=t*2
    poses=np.tile(np.eye(4),(100,1,1));poses[:,0,3]=t
    original=[object() for _ in range(12)]
    original[5]={'frozen_factors':['source clock','object geometry/mass/friction']}
    original[7:]=[t,q,np.where(t<.5,2.,-.06),poses,poses]
    changed=prepare(tuple(original),tmp_path/'timing')
    assert changed[7][-1]>t[-1] and changed[5]['robot_control_retiming']['original_interval_complete']
    np.testing.assert_array_equal(changed[8][[0,-1]],q[[0,-1]])
    np.testing.assert_array_equal(changed[11][[0,-1]],poses[[0,-1]])
    assert set(changed[9])=={2.,-.06}
    assert changed[9][0]==2 and changed[9][-1]==-.06
    assert np.max(np.abs(np.diff(changed[8][:,3])/.01))<=.8+1e-8
    assert 'source clock' in original[5]['frozen_factors']
