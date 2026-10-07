"""Refine invalid interpolation chords while retaining every original IK knot."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .store import json_write, sha256


class MidpointAdmissionError(ValueError):
    """A numerically recorded midpoint failed the ordinary admission limits."""


def refine(robot, solver, references, hands, apertures, output, *, phase, rounds,
           minimum_motion=False, position_reserve_m=0., fixture_reserve_m=0., quarter_probes=False):
    """Insert constrained IK only at failing Cartesian/joint interpolation midpoints.

    This is a bounded repair, not admission: the caller must still inspect every
    final control row. Original endpoints and all original knots stay exact.
    """
    if type(quarter_probes) is not bool:
        raise ValueError('Quarter probes require an explicit boolean')
    if type(rounds) is not int or not 1 <= rounds <= 3:
        raise ValueError('One to three explicit midpoint refinement rounds required')
    if (type(minimum_motion) is not bool or (minimum_motion and not solver.pose_tolerance_constraints)):
        raise ValueError('Local midpoint minimum-motion requires hard pose constraints')
    if (not np.isfinite(position_reserve_m) or not 0 <= position_reserve_m <= .001
            or (position_reserve_m and not solver.pose_tolerance_constraints)):
        raise ValueError('A midpoint position reserve in [0, 1 mm] requires hard pose constraints')
    if not np.isfinite(fixture_reserve_m) or not 0 <= fixture_reserve_m <= .001:
        raise ValueError('A midpoint fixture reserve in [0, 1 mm] is required')
    references, hands = np.asarray(references), np.asarray(hands)
    if references.ndim != 2 or references.shape[1] != 17 or hands.shape != (len(references),4,4):
        raise ValueError('Aligned robot reference and hand knots required')
    original_refs, original_hands = references.copy(), hands.copy()
    weights = np.linspace(0.,1.,len(references))
    original_weights = weights.copy()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    events = []
    failure = None
    checked_rejection = False
    try:
        for iteration in range(rounds):
            rows, poses, values = [references[0]], [hands[0]], [weights[0]]
            added = 0
            for i in range(len(references)-1):
                for fraction in ((.25,.5,.75) if quarter_probes else (.5,)):
                    midref = ((references[i]+references[i+1])/2 if fraction == .5 else
                              (1-fraction)*references[i]+fraction*references[i+1])
                    goal = hands[i].copy()
                    goal[:3,3] = ((hands[i,:3,3]+hands[i+1,:3,3])/2 if fraction == .5 else
                                   (1-fraction)*hands[i,:3,3]+fraction*hands[i+1,:3,3])
                    goal[:3,:3] = Slerp([0.,1.],Rotation.from_matrix(hands[i:i+2,:3,:3]))(fraction).as_matrix()
                    weight = float((weights[i]+weights[i+1])/2 if fraction == .5 else
                                   (1-fraction)*weights[i]+fraction*weights[i+1])
                    q = robot.pack(midref[3:],midref[:3])
                    actual = robot.fk(q)[1]
                    error = [float(np.linalg.norm(goal[:3,3]-actual[:3,3])),
                             float(Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T).magnitude())]
                    fixture = solver.inspect(q,apertures)
                    if np.any(np.asarray(error)>[.002,.02]) or not fixture['admitted']:
                        event=dict(round=iteration,sample_fraction=fraction,left_weight=float(weights[i]),right_weight=float(weights[i+1]),
                                   weight=weight,before_errors=error,before_fixture=fixture)
                        events.append(event)
                        seedref=midref if minimum_motion else references[i]
                        seed=robot.pack(seedref[3:],seedref[:3])
                        solver.base_yaw=float(seedref[2])
                        original_mode=getattr(solver,'minimum_motion',False)
                        original_transport=getattr(solver,'orientation_transport',False)
                        try:
                            if minimum_motion:
                                solver.minimum_motion=True
                                solver.orientation_transport=False
                            options = ({'position_interior_reserve_m': position_reserve_m}
                                       if position_reserve_m else {})
                            if fixture_reserve_m:
                                options['fixture_interior_reserve_m'] = fixture_reserve_m
                            q=solver.solve(goal,seed,apertures,phase=phase+'_midpoint',weight=weight,**options)
                        finally:
                            solver.minimum_motion=original_mode
                            solver.orientation_transport=original_transport
                        checked=solver.records[-1]
                        event['after']=checked
                        if np.any(np.asarray(checked['ik_errors'])>[.002-position_reserve_m,.02]) or not checked['fixture']['admitted']:
                            checked_rejection=True
                            raise ValueError('Constrained midpoint failed independent IK/fixture checks')
                        row=np.r_[solver.reference_base(q),references[i,3:10],q[robot.arm_ids[7:]]]
                        rows.append(row);poses.append(goal);values.append(weight);added+=1
                rows.append(references[i+1]);poses.append(hands[i+1]);values.append(weights[i+1])
            references,hands,weights=np.asarray(rows),np.asarray(poses),np.asarray(values)
            if not added:
                break
    except Exception as error:
        failure=type(error).__name__+': '+str(error)
    finally:
        solver.base_yaw=float(original_refs[-1,2])
    artifact=output/'knots.npz'
    np.savez_compressed(artifact,original_references=original_refs,original_hands=original_hands,
                        original_weights=original_weights,references=references,hands=hands,weights=weights,
                        attempted_midpoint_q=np.asarray([event['after']['q'] for event in events if 'after' in event]))
    report=dict(phase=phase,rounds=rounds,original_knots=len(original_refs),refined_knots=len(references),
                minimum_motion=minimum_motion,quarter_probes=quarter_probes,position_reserve_m=float(position_reserve_m),
                fixture_reserve_m=float(fixture_reserve_m),
                final_admission_tolerances_m_rad=[.002,.02],
                events=events,error=failure,artifact_sha256=sha256(artifact),
                original_knots_retained=all(np.any(weights==w) for w in original_weights),
                full_interpolated_admission=False,physics_validated=False)
    json_write(output/'result.json',report)
    if failure:
        error_type=MidpointAdmissionError if checked_rejection else ValueError
        raise error_type('Mobile midpoint refinement failed; inspect '+str(output/'result.json'))
    return references,hands,weights,report


def interval_times(references, hands, weights, nominal_duration, speeds, *, angular_speed):
    """Allocate time to each unchanged geometric interval under the declared caps."""
    references,hands,weights,speeds=map(np.asarray,(references,hands,weights,speeds))
    if (len(references)<2 or hands.shape!=(len(references),4,4) or weights.shape!=(len(references),)
            or speeds.shape!=(references.shape[1],) or not np.isfinite(references).all()
            or not np.isfinite(weights).all() or not np.isfinite(speeds).all() or np.any(speeds<=0)
            or np.any(np.diff(weights)<=0) or not np.isfinite(nominal_duration) or nominal_duration<=0
            or not np.isfinite(angular_speed) or angular_speed<=0):
        raise ValueError('Finite aligned knots and positive interval timing bounds required')
    rotation_steps=Rotation.from_matrix(hands[1:,:3,:3]@hands[:-1,:3,:3].transpose(0,2,1)).magnitude()
    durations=np.maximum.reduce([nominal_duration*np.diff(weights),
        np.max(abs(np.diff(references,axis=0))/speeds,axis=1),rotation_steps/angular_speed])
    return np.r_[0.,np.cumsum(durations)]


def acceleration_limited_times(references, hands, weights, nominal_duration, speeds,
                               accelerations, *, angular_speed):
    """Time unchanged line segments with zero speed/acceleration at every knot.

    A quintic progress function has analytic derivative maxima 15/8 and
    10/sqrt(3). These bounds apply continuously, not only at recorded ticks.
    """
    references, hands, weights, speeds = map(np.asarray, (references,hands,weights,speeds))
    accelerations = np.asarray(accelerations, float)
    # Reuse the ordinary shape/finite/positive-speed validation.
    interval_times(references,hands,weights,nominal_duration,speeds,angular_speed=angular_speed)
    if (accelerations.shape != speeds.shape or not np.isfinite(accelerations).all()
            or np.any(accelerations<=0)):
        raise ValueError('Finite positive acceleration caps for every reference channel required')
    delta=abs(np.diff(references,axis=0))
    angles=Rotation.from_matrix(hands[1:,:3,:3]@hands[:-1,:3,:3].transpose(0,2,1)).magnitude()
    durations=np.maximum.reduce([nominal_duration*np.diff(weights),
        np.max(1.875*delta/speeds,axis=1),
        np.sqrt(np.max((10/np.sqrt(3))*delta/accelerations,axis=1)),
        1.875*angles/angular_speed])
    return np.r_[0.,np.cumsum(durations)]


def quintic_knot_progress(sample_times, knot_times):
    """Map elapsed time to a continuous knot index on the identical polyline."""
    sample_times,knot_times=map(np.asarray,(sample_times,knot_times))
    if (sample_times.ndim!=1 or knot_times.ndim!=1 or len(knot_times)<2
            or not np.isfinite(sample_times).all() or not np.isfinite(knot_times).all()
            or np.any(np.diff(knot_times)<=0) or np.any(sample_times<knot_times[0])
            or np.any(sample_times>knot_times[-1])):
        raise ValueError('Finite samples within a strictly increasing knot clock required')
    index=np.clip(np.searchsorted(knot_times,sample_times,side='right')-1,0,len(knot_times)-2)
    fraction=(sample_times-knot_times[index])/(knot_times[index+1]-knot_times[index])
    smooth=np.clip(fraction**3*(10-15*fraction+6*fraction*fraction),0.,1.)
    return index+smooth
