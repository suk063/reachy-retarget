"""The insertion retains source rows; only measured support permits opening."""
import json

import numpy as np
import pytest

from reachy_retarget import supported_placement as placement
from test_grasp_approach import Robot, fixture
from reachy_retarget import grasp_approach


def prepared(monkeypatch):
    result=list(fixture(monkeypatch))
    monkeypatch.setattr(placement,'initialize',grasp_approach.initialize)
    result[10][-1]=result[10][-2]
    result[8][-1]=result[8][-2]
    return tuple(result)


def test_preserves_every_original_row_and_pauses_source_clock(monkeypatch,tmp_path):
    original=prepared(monkeypatch);robot=Robot();active=robot.active.copy()
    result=placement.prepare(original,robot,tmp_path/'place')
    policy=result[5]['robot_supported_placement']
    with np.load(policy['source_clock_artifact']) as artifact:
        rows=artifact['original_row_indices']
        np.testing.assert_array_equal(artifact['source_reference_time_s'][rows],original[7])
        assert np.all(np.diff(artifact['source_reference_time_s'])>=0)
        assert artifact['source_reference_time_s'][6]==original[7][6]
    for i in (8,9,10,11):np.testing.assert_array_equal(result[i][rows],original[i])
    for i in (0,1,2,3,4,6):assert result[i] is original[i]
    np.testing.assert_array_equal(robot.active,active)
    assert policy['admitted'] and not policy['physics_validated']
    assert 'robot_supported_placement' not in original[5]


def test_lower_acceleration_timing_keeps_source_traverse_and_geometric_knots(monkeypatch,tmp_path):
    source=prepared(monkeypatch);ordinary=placement.prepare(source,Robot(),tmp_path/'ordinary')
    limits=np.r_[1.,1.,2.,np.full(14,3.)]
    limited=placement.prepare(source,Robot(),tmp_path/'limited',lower_acceleration_limits=limits)
    old=ordinary[5]['robot_supported_placement'];new=limited[5]['robot_supported_placement']
    a,b=old['phase_frames']['traverse'];c,d=new['phase_frames']['traverse']
    for field in (8,9,10,11):np.testing.assert_array_equal(ordinary[field][a:b],limited[field][c:d])
    with np.load(new['source_clock_artifact']) as saved:
        indices=saved['original_row_indices']
        for field in (8,9,10,11):np.testing.assert_array_equal(limited[field][indices],source[field])
    np.testing.assert_array_equal(np.load(tmp_path/'limited/ik-knots.npz')['q'],
                                  np.load(tmp_path/'ordinary/ik-knots.npz')['q'])
    a,b=new['phase_frames']['lower'];q=limited[8][a:b+1]
    assert np.all(np.max(abs(np.diff(q,n=2,axis=0))/.01**2,axis=0)<=limits+1e-9)
    assert new['interval_clock_artifacts'][0]['progress_law']=='per-segment quintic'
    assert len(new['interval_clock_artifacts'])==1
    with pytest.raises(ValueError,match='unchanged retreat'):
        placement.prepare(source,Robot(),tmp_path/'bad',lower_acceleration_limits=limits,retreat_speed_factor=2.)


def exercise(result,force):
    control=placement.Controller(result[5],result[8],result[10],result[9])
    rows=[]
    for step in range(len(result[7])):
        rows.append(control.update(step,force(step,control),1.,
                    object_position_world=control.policy['placement_position_world'],grasp_intact=True))
    return control,rows


def test_no_support_never_opens_and_retains_a_failed_attempt(monkeypatch,tmp_path):
    result=placement.prepare(prepared(monkeypatch),Robot(),tmp_path/'place')
    control,rows=exercise(result,lambda step,ctrl:0.)
    start=control.phases['lower'][0]
    assert all(row[2]<0 for row in rows[start:])
    assert control.failed and not control.opened
    assert 'No stable measured support' in control.state['failure_reason']
    np.testing.assert_array_equal(rows[-1][0],rows[-2][0])


def test_measured_support_freezes_descent_then_opens_and_reverses_admitted_prefix(monkeypatch,tmp_path):
    result=placement.prepare(prepared(monkeypatch),Robot(),tmp_path/'place')
    low=result[5]['robot_supported_placement']['phase_frames']['lower']
    contact=low[0]+(low[1]-low[0])//2
    control,rows=exercise(result,lambda step,ctrl:.8 if step>=contact else 0.)
    assert control.supported and control.opened and not control.failed
    assert control.contact_index==contact-1
    open_start=control.phases['open'][0]
    assert all(row[2]<0 for row in rows[low[0]:open_start])
    assert rows[open_start][2]==2.
    for step in range(contact,control.phases['open'][1]):
        np.testing.assert_array_equal(rows[step][0],result[8][contact-1])
    actual=np.array([row[0] for row in rows])
    insertion=slice(control.phases['traverse'][0]-1,control.phases['return'][1]+1)
    assert np.max(abs(np.diff(actual[insertion,10:],axis=0))/.01)<=.800001
    np.testing.assert_array_equal(rows[-1][0],result[8][-1])


def test_transient_support_is_insufficient(monkeypatch,tmp_path):
    result=placement.prepare(prepared(monkeypatch),Robot(),tmp_path/'place')
    lower=result[5]['robot_supported_placement']['phase_frames']['lower'][0]
    control,_=exercise(result,lambda step,ctrl:1. if lower+5<=step<lower+9 else 0.)
    assert control.contact_index is not None
    assert control.failed and not control.opened


def test_explicit_support_wait_allows_late_measured_touchdown_without_changing_debounce(monkeypatch,tmp_path):
    source=prepared(monkeypatch)
    ordinary=placement.prepare(source,Robot(),tmp_path/'ordinary')
    extended=placement.prepare(source,Robot(),tmp_path/'extended',support_wait_s=.85)
    old=ordinary[5]['robot_supported_placement'];new=extended[5]['robot_supported_placement']
    assert new['inserted_frames']-old['inserted_frames']==50
    assert new['support_hold_s']==old['support_hold_s']==.12
    assert new['support_wait_s']==.85
    # The same force arrives 0.4 s after descent, beyond the original 0.35 s window.
    def late(step,ctrl):return .8 if step>=ctrl.phases['lower'][1]+40 else 0.
    rejected,_=exercise(ordinary,late)
    accepted,_=exercise(extended,late)
    assert rejected.failed and not rejected.opened
    assert not accepted.failed and accepted.opening_verified
    with np.load(new['source_clock_artifact']) as saved:
        rows=saved['original_row_indices']
        for i in (8,9,10,11):np.testing.assert_array_equal(extended[i][rows],source[i])
    a,b=new['phase_frames']['hold']
    np.testing.assert_allclose(extended[10][a:b],np.broadcast_to(extended[10][a],extended[10][a:b].shape))
    with pytest.raises(ValueError,match='bounded'):
        placement.prepare(source,Robot(),tmp_path/'unbounded',support_wait_s=2.01)


@pytest.mark.parametrize('failure', ['wrong_location','no_grasp','stale_grasp','missing_position'])
def test_support_from_dropped_object_cannot_authorize_opening(monkeypatch,tmp_path,failure):
    result=placement.prepare(prepared(monkeypatch),Robot(),tmp_path/'place')
    control=placement.Controller(result[5],result[8],result[10],result[9])
    lower=control.phases['lower'][0]
    for step in range(len(result[7])):
        point=np.asarray(control.policy['placement_position_world']).copy()
        intact=failure not in ('no_grasp','stale_grasp') or (failure=='stale_grasp' and step<lower-30)
        if failure=='wrong_location':point[0]+=.5
        if failure=='missing_position':point=None
        control.update(step,1.,1.,object_position_world=point,grasp_intact=intact)
    assert not control.opened and control.failed and not control.opening_verified
    assert control.contact_index is None


def test_support_can_unload_gripper_during_debounce_after_verified_grasp(monkeypatch,tmp_path):
    result=placement.prepare(prepared(monkeypatch),Robot(),tmp_path/'place')
    control=placement.Controller(result[5],result[8],result[10],result[9])
    contact=control.phases['lower'][0]+5
    for step in range(len(result[7])):
        control.update(step,1. if step>=contact else 0.,1.,
                       object_position_world=control.policy['placement_position_world'],grasp_intact=step<contact)
    assert control.opened and control.opening_verified and not control.failed
    assert control.state['support_grasp_verified']


def test_interrupted_ik_preserves_original_proposal_and_partial_knots(monkeypatch,tmp_path):
    original=prepared(monkeypatch);robot=Robot(fail_at=4);active=robot.active.copy()
    with pytest.raises(ValueError,match='IK rejected'):
        placement.prepare(original,robot,tmp_path/'failed')
    report=json.loads((tmp_path/'failed/result.json').read_text())
    assert not report['admitted'] and 'solver stopped' in report['exception']
    with np.load(tmp_path/'failed/ik-knots.npz') as f:assert len(f['q'])==4
    assert (tmp_path/'failed/original-and-proposal.npz').is_file()
    np.testing.assert_array_equal(robot.active,active)


def test_partial_opening_is_recorded_then_full_opening_occurs_above_wall(monkeypatch,tmp_path):
    result=placement.prepare(prepared(monkeypatch),Robot(),tmp_path/'partial',release_aperture_rad=1.4)
    lower=result[5]['robot_supported_placement']['phase_frames']['lower'][0]
    control,rows=exercise(result,lambda step,ctrl:1. if step>=lower+5 else 0.)
    open_start,open_end=control.phases['open']
    assert control.opened and control.supported and not control.failed
    assert all(row[2]==1.4 for row in rows[open_start:control.phases['return'][0]])
    assert all(row[2]==2. for row in rows[control.phases['return'][0]:])
    assert rows[open_end][1][2,3] >= rows[open_start][1][2,3]


def test_requested_new_placement_must_remain_in_original_task_region(monkeypatch,tmp_path):
    source=list(prepared(monkeypatch))
    source[5]=dict(source[5],task_contract=dict(bin_lower=[0,-.1,.3],bin_upper=[.5,.1,.5]),
                   world_placement=np.eye(4).tolist())
    with pytest.raises(ValueError,match='outside the original verified task region'):
        placement.prepare(tuple(source),Robot(),tmp_path/'outside',placement_xy_world=[.6,0])


def test_reverse_path_restores_exact_source_branch_after_base_and_orientation_motion(monkeypatch,tmp_path):
    source=list(prepared(monkeypatch))
    source[5]=dict(source[5],task_contract=dict(bin_lower=[0,-.1,.3],bin_upper=[.5,.1,.5]),
                   world_placement=np.eye(4).tolist())
    result=placement.prepare(tuple(source),Robot(),tmp_path/'return',placement_xy_world=[.25,.02],
                             base_offset_xyyaw=[.1,0,0],rotation_xyz_deg=[0,10,5])
    policy=result[5]['robot_supported_placement'];first=policy['phase_frames']['traverse'][0]
    end=policy['phase_frames']['return'][1]
    np.testing.assert_array_equal(result[8][first],source[8][-1])
    np.testing.assert_array_equal(result[8][end],source[8][-1])
    assert np.max(abs(np.diff(result[8][first:end+1,:3],axis=0))/.01)<=.30001


def test_intermediate_aperture_collision_rejects_even_when_both_endpoints_are_free(monkeypatch,tmp_path):
    source=list(prepared(monkeypatch))
    source[5]=dict(source[5],effective_candidate={'gripper_closed_target':1.12})
    initialize=placement.initialize;depths=placement.collision_depths;observed=[]
    def record_aperture(model,data,robot,q,mimics,grip):
        assert grip['l_hand_finger'] == 2.
        observed.append(grip['r_hand_finger']);initialize(model,data,robot,q,mimics,grip)
    def collision(model,data,bodies,object_body):
        result,pairs=depths(model,data,bodies,object_body)
        if 1.2<observed[-1]<1.35:result['environment']=.004
        return result,pairs
    monkeypatch.setattr(placement,'initialize',record_aperture)
    monkeypatch.setattr(placement,'collision_depths',collision)
    with pytest.raises(ValueError,match='admission rejected'):
        placement.prepare(tuple(source),Robot(),tmp_path/'aperture',release_aperture_rad=1.4)
    report=json.loads((tmp_path/'aperture/result.json').read_text())
    assert not report['admission_checks']['environment']
    assert report['max_collision_depths_m'][0]==.004
    assert 1.12 in observed and 1.4 in observed and 2. in observed


def test_faster_retreat_keeps_every_grasp_support_and_open_row_exact(monkeypatch,tmp_path):
    original=prepared(monkeypatch)
    slow=placement.prepare(original,Robot(),tmp_path/'slow')
    fast=placement.prepare(original,Robot(),tmp_path/'fast',retreat_speed_factor=3.)
    a,b=slow[5]['robot_supported_placement']['phase_frames']['retreat']
    c,d=fast[5]['robot_supported_placement']['phase_frames']['retreat']
    assert a==c and d-c<b-a
    for key in (7,8,9,10,11):np.testing.assert_array_equal(fast[key][:a],slow[key][:a])
    for key in (8,9,10,11):np.testing.assert_array_equal(fast[key][d:],slow[key][b:])
    for result in (slow,fast):
        lower=result[5]['robot_supported_placement']['phase_frames']['lower'][0]
        controller,rows=exercise(result,lambda step,ctrl:1. if step>=lower+5 else 0.)
        refs=np.asarray([row[0]for row in rows]);start=controller.phases['traverse'][0]
        assert np.max(abs(np.diff(refs[start:,10:],axis=0))/.01)<=.800001
        assert controller.opened and not controller.failed


@pytest.mark.parametrize('speed', [.6, .4])
def test_loaded_rotation_dilation_preserves_geometry_source_and_unloaded_return(monkeypatch,tmp_path,speed):
    original=prepared(monkeypatch)
    args=dict(rotation_xyz_deg=[0.,40.,0.])
    baseline=placement.prepare(original,Robot(),tmp_path/'baseline',**args)
    slowed=placement.prepare(original,Robot(),tmp_path/'slowed',loaded_rotation_speed_rad_s=speed,**args)
    baseline_policy=baseline[5]['robot_supported_placement']
    policy=slowed[5]['robot_supported_placement']
    a,b=baseline_policy['phase_frames']['traverse']
    c,d=policy['phase_frames']['traverse']
    assert a==c and d-c>b-a
    for index in (7,8,9,10,11):np.testing.assert_array_equal(slowed[index][:a],baseline[index][:a])
    # Everything after the loaded leg, including return timing, is identical.
    for index in (8,9,10,11):np.testing.assert_array_equal(slowed[index][d:],baseline[index][b:])
    with np.load(tmp_path/'baseline/ik-knots.npz') as original_knots, np.load(tmp_path/'slowed/ik-knots.npz') as slow_knots:
        np.testing.assert_array_equal(original_knots['q'],slow_knots['q'])
    with np.load(policy['source_clock_artifact']) as artifact:
        rows=artifact['original_row_indices']
        for index in (8,9,10,11):np.testing.assert_array_equal(slowed[index][rows],original[index])
        np.testing.assert_array_equal(artifact['source_reference_time_s'][rows],original[7])
    assert policy['loaded_traverse_angular_peak_rad_s']<=speed+1e-7
    assert policy['admission_checks']['loaded_rotation_speed']
    assert policy['completed_frames']==policy['total_frames']==len(slowed[7])
    assert not policy['physics_validated']


@pytest.mark.parametrize('speed', [0., .09, .81, float('nan')])
def test_loaded_rotation_speed_is_bounded(monkeypatch,tmp_path,speed):
    with pytest.raises(ValueError,match='bounded'):
        placement.prepare(prepared(monkeypatch),Robot(),tmp_path/'bad',loaded_rotation_speed_rad_s=speed)


def test_opening_audit_uses_original_step_history_after_measured_acquisition_pause(monkeypatch,tmp_path):
    from copy import deepcopy
    from reachy_retarget.acquisition_clock import SourceIndexedController
    result=placement.prepare(prepared(monkeypatch),Robot(),tmp_path/'place')
    controller=placement.Controller(result[5],result[8],result[10],result[9])
    wrapper=SourceIndexedController(controller)
    indices=[0,1,1,1,1]+list(range(2,len(result[7])))
    states=[];positions=[]
    target=controller.policy['placement_position_world']
    for index in indices:
        intact=index>=2
        wrapper.update(index,1.,1.,object_position_world=target,grasp_intact=intact)
        states.append({'memory':{'supported_placement':deepcopy(controller.state),
                                 'drift_anchor':np.eye(4).tolist() if intact else None}})
        positions.append(list(target)+[1.,0.,0.,0.])
    bilateral=np.ones(len(states))
    proof=placement.audit_opening_guard(states,positions,bilateral,controller.policy,reference_indices=indices)
    assert proof['passed'] and proof['opening_events']
    assert controller.opened and controller.opening_verified
    assert not placement.audit_opening_guard(states,positions,bilateral,controller.policy)['passed']
    corrupted=deepcopy(states)
    corrupted[2]['memory']['supported_placement']['support_force_N']+=1
    with pytest.raises(ValueError,match='unchanged pre-placement'):
        placement.audit_opening_guard(corrupted,positions,bilateral,controller.policy,reference_indices=indices)
    positions[4][0]+=.1
    assert not placement.audit_opening_guard(states,positions,bilateral,controller.policy,
        reference_indices=indices)['passed']
