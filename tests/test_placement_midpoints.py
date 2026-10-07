import json
import numpy as np
import pytest

from reachy_retarget.placement_midpoints import (refine,MidpointAdmissionError,interval_times,
                                               acceleration_limited_times,quintic_knot_progress)


class Robot:
    arm_ids=np.arange(3,17)
    def pack(self,arms,base):return np.r_[base,arms]
    def fk(self,q):
        value=np.tile(np.eye(4),(2,1,1))
        value[1,:3,3]=[q[10],q[11]-q[10]**2,0.]
        return value


class Solver:
    base_yaw=0.
    pose_tolerance_constraints=True
    minimum_motion=False
    orientation_transport=True
    def __init__(self,invalid=False):self.records=[];self.invalid=invalid
    def reference_base(self,q):return q[:3]
    def inspect(self,q,apertures):return dict(admitted=True,minimum_signed_distance_or_lower_bound_m=.004)
    def solve(self,goal,seed,apertures,**kwargs):
        q=seed.copy();q[10]=goal[0,3];q[11]=goal[1,3]+q[10]**2
        self.records.append(dict(q=q.tolist(),ik_errors=[.004 if self.invalid else 0.,0.],fixture=self.inspect(q,apertures)))
        return q


def test_refines_curved_interpolation_without_replacing_original_knots(tmp_path):
    robot=Robot();solver=Solver();refs=np.zeros((2,17));refs[:,10]=[-.1,.1];refs[:,11]=.01
    hands=np.array([robot.fk(q)[1] for q in refs]);original=refs.copy()
    result,goals,weights,report=refine(robot,solver,refs,hands,[.8,1.4],tmp_path/'refine',phase='lower',rounds=3)
    assert len(result)==5 and len(report['events'])==3
    np.testing.assert_array_equal(result[[0,-1]],original)
    np.testing.assert_array_equal(refs,original)
    for a,b,g,h in zip(result[:-1],result[1:],goals[:-1],goals[1:]):
        actual=robot.fk((a+b)/2)[1]
        assert np.linalg.norm(actual[:3,3]-(g[:3,3]+h[:3,3])/2)<.002
    assert report['original_knots_retained'] and not report['full_interpolated_admission']
    assert solver.base_yaw==refs[-1,2]
    with np.load(tmp_path/'refine/knots.npz') as saved:
        assert len(saved['attempted_midpoint_q'])==3


def test_failed_midpoint_is_preserved_without_promoting_partial_path(tmp_path):
    robot=Robot();solver=Solver(invalid=True);refs=np.zeros((2,17));refs[:,10]=[-.1,.1];refs[:,11]=.01
    hands=np.array([robot.fk(q)[1] for q in refs])
    with pytest.raises(MidpointAdmissionError):
        refine(robot,solver,refs,hands,[.8,1.4],tmp_path/'failed',phase='lower',rounds=3)
    report=json.loads((tmp_path/'failed/result.json').read_text())
    assert report['error'] and not report['full_interpolated_admission']
    with np.load(tmp_path/'failed/knots.npz') as saved:
        np.testing.assert_array_equal(saved['original_references'],refs)
        assert len(saved['attempted_midpoint_q'])==1


def test_quarter_probes_find_curvature_that_midpoints_miss(tmp_path):
    class RippleRobot(Robot):
        def fk(self,q):
            poses=np.tile(np.eye(4),(2,1,1))
            poses[1,:3,3]=[q[10],q[11]-.004*np.sin(2*np.pi*q[10]),0.]
            return poses
    class RippleSolver(Solver):
        def solve(self,goal,seed,apertures,**kwargs):
            q=seed.copy();q[10]=goal[0,3];q[11]=goal[1,3]+.004*np.sin(2*np.pi*q[10])
            self.records.append(dict(q=q.tolist(),ik_errors=[0.,0.],fixture=self.inspect(q,apertures)))
            return q
    robot=RippleRobot();refs=np.zeros((2,17));refs[-1,10]=1.
    hands=np.array([robot.fk(q)[1] for q in refs])
    plain,_,_,report=refine(robot,RippleSolver(),refs,hands,[.8,1.4],tmp_path/'plain',phase='lower',rounds=3)
    assert len(plain)==2 and not report['events']
    result,goals,weights,report=refine(robot,RippleSolver(),refs,hands,[.8,1.4],tmp_path/'quarters',
        phase='lower',rounds=3,quarter_probes=True)
    assert report['quarter_probes'] and len(result)>2
    assert {event['sample_fraction'] for event in report['events']}=={.25,.75}
    np.testing.assert_array_equal(result[[0,-1]],refs)
    np.testing.assert_array_equal(goals[[0,-1]],hands)
    samples=np.linspace(0.,1.,1025)
    dense=np.column_stack([np.interp(samples,weights,result[:,j]) for j in range(17)])
    assert max(abs(robot.fk(q)[1][1,3]) for q in dense)<.002
    assert report['original_knots_retained'] and not report['full_interpolated_admission']


def test_unexpected_midpoint_backend_error_is_not_a_checked_rejection(tmp_path):
    robot=Robot();solver=Solver();refs=np.zeros((2,17));refs[:,10]=[-.1,.1];refs[:,11]=.01
    hands=np.array([robot.fk(q)[1] for q in refs])
    def broken(*args,**kwargs):raise RuntimeError('backend unavailable')
    solver.solve=broken
    with pytest.raises(ValueError) as error:
        refine(robot,solver,refs,hands,[.8,1.4],tmp_path/'error',phase='lower',rounds=1)
    assert not isinstance(error.value,MidpointAdmissionError)
    assert 'RuntimeError' in json.loads((tmp_path/'error/result.json').read_text())['error']


def test_local_motion_repair_starts_at_chord_midpoint_and_restores_solver_modes(tmp_path):
    robot=Robot();solver=Solver();refs=np.zeros((2,17));refs[:,10]=[-.1,.1];refs[:,11]=.01
    hands=np.array([robot.fk(q)[1] for q in refs]);seeds=[];original=solver.solve
    def record(goal,seed,apertures,**kwargs):
        assert solver.minimum_motion and not solver.orientation_transport
        seeds.append(seed.copy())
        return original(goal,seed,apertures,**kwargs)
    solver.solve=record
    refine(robot,solver,refs,hands,[.8,1.4],tmp_path/'local',phase='lower',rounds=1,minimum_motion=True)
    np.testing.assert_array_equal(seeds[0],refs.mean(0))
    assert not solver.minimum_motion and solver.orientation_transport


def test_midpoint_reserve_only_applies_to_new_knots_and_checks_stricter_solution(tmp_path):
    robot=Robot();solver=Solver();refs=np.zeros((2,17));refs[:,10]=[-.1,.1];refs[:,11]=.01
    hands=np.array([robot.fk(q)[1] for q in refs]);options=[];original=solver.solve
    def record(goal,seed,apertures,**kwargs):
        options.append(kwargs.copy())
        return original(goal,seed,apertures,**kwargs)
    solver.solve=record
    result,goals,weights,report=refine(robot,solver,refs,hands,[.8,1.4],tmp_path/'reserved',
        phase='lower',rounds=3,minimum_motion=True,position_reserve_m=.0001)
    assert options and all(o['position_interior_reserve_m']==.0001 for o in options)
    assert report['position_reserve_m']==.0001
    assert report['final_admission_tolerances_m_rad']==[.002,.02]
    np.testing.assert_array_equal(result[[0,-1]],refs)
    np.testing.assert_array_equal(goals[[0,-1]],hands)
    def misses_reserve(goal,seed,apertures,**kwargs):
        q=original(goal,seed,apertures,**kwargs)
        solver.records[-1]['ik_errors']=[.00195,0.]
        return q
    solver.solve=misses_reserve
    with pytest.raises(MidpointAdmissionError):
        refine(robot,solver,refs,hands,[.8,1.4],tmp_path/'reserved-failed',phase='lower',rounds=1,
               position_reserve_m=.0001)


def test_interval_clock_does_not_slow_unrelated_knots_and_retains_caps():
    refs=np.zeros((4,17));refs[:,10]=[0.,.01,2.01,2.02]
    hands=np.tile(np.eye(4),(4,1,1));weights=np.linspace(0.,1.,4)
    before=refs.copy();times=interval_times(refs,hands,weights,1.,np.ones(17),angular_speed=.8)
    assert times[-1] < 3. and times[-1] < 3*np.max(abs(np.diff(refs,axis=0)))
    assert np.max(abs(np.diff(refs,axis=0))/np.diff(times)[:,None])<=1.
    np.testing.assert_array_equal(refs,before)


def test_quintic_segment_timing_bounds_velocity_corners_without_moving_knots():
    refs=np.zeros((4,17));refs[:,10]=[0.,.01,-.02,.025];refs[:,0]=[0.,.002,.001,.006]
    hands=np.tile(np.eye(4),(4,1,1));hands[:,0,3]=refs[:,10]
    weights=np.linspace(0.,1.,4);speeds=np.full(17,.65);speeds[:3]=[.3,.3,.8]
    acceleration=np.full(17,3.);acceleration[:3]=[1.,1.,2.]
    before=refs.copy()
    clock=acceleration_limited_times(refs,hands,weights,.12,speeds,acceleration,angular_speed=.8)
    np.testing.assert_allclose(quintic_knot_progress(clock,clock),np.arange(4),atol=1e-15)
    ticks=int(np.ceil(clock[-1]/.001));sample=np.linspace(0.,clock[-1],ticks+1)
    progress=quintic_knot_progress(sample,clock)
    trajectory=np.column_stack([np.interp(progress,np.arange(4),refs[:,j]) for j in range(17)])
    dt=sample[1];velocity=np.diff(trajectory,axis=0)/dt;measured=np.diff(velocity,axis=0)/dt
    assert np.all(abs(velocity).max(0)<=speeds)
    assert np.all(abs(measured).max(0)<=acceleration)
    np.testing.assert_array_equal(refs,before)
    np.testing.assert_array_equal(trajectory[[0,-1]],refs[[0,-1]])
    # Arbitrary interior fractions remain exactly on their original line segment.
    for i in range(3):
        interior=trajectory[(sample>clock[i])&(sample<clock[i+1])]
        ratios=(interior[:,10]-refs[i,10])/(refs[i+1,10]-refs[i,10])
        np.testing.assert_allclose(interior[:,0],refs[i,0]+ratios*(refs[i+1,0]-refs[i,0]),atol=1e-15)
    with pytest.raises(ValueError,match='acceleration caps'):
        acceleration_limited_times(refs,hands,weights,.12,speeds,np.zeros(17),angular_speed=.8)
