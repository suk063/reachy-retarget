"""Insert a bounded, support-gated robot placement phase without object writes.

The original source/control rows remain verbatim in an immutable artifact. The
inserted phase pauses the source clock; observed support can stop descent early.
Only pre-admitted robot references and gripper intent are returned at runtime.
"""
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .feasible_maniskill import collision_depths
from .physics import initialize
from .store import json_write, sha256


def prepare(prepared, robot, output, *, extra_descent_m=.008, support_hold_s=.12,
            arm_speed_rad_s=.65, placement_xy_world=None, base_offset_xyyaw=(0.,0.,0.),
            rotation_xyz_deg=(0.,0.,0.), release_aperture_rad=2., retreat_speed_factor=1.,
            loaded_rotation_speed_rad_s=.8, mobile_ik=None, adaptive_midpoints=0,
            midpoint_minimum_motion=False, interval_knot_timing=False, support_wait_s=.35,
            midpoint_position_reserve_m=0., midpoint_fixture_reserve_m=0.,
            lower_acceleration_limits=None, midpoint_quarter_probes=False,
            mobile_base_axis_speed_m_s=.3):
    """Call after retiming. Add a placement phase at the first source release.

    Geometry, object reset and source data are unchanged. The nominal translation
    moves the release object reference to its final recorded position, preserving
    the grasp orientation. Measured support, not this reference, authorizes opening.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/Path(__file__).name).write_text(Path(__file__).read_text())
    if (type(adaptive_midpoints) is not int or not 0 <= adaptive_midpoints <= 3
            or (adaptive_midpoints and mobile_ik is None)):
        raise ValueError('Adaptive midpoints require mobile IK and zero to three bounded rounds')
    if (type(midpoint_minimum_motion) is not bool or type(interval_knot_timing) is not bool
            or (midpoint_minimum_motion and not adaptive_midpoints)
            or (interval_knot_timing and mobile_ik is None)):
        raise ValueError('Explicit local refinement/timing options require mobile placement')
    if (not np.isfinite(midpoint_position_reserve_m) or not 0 <= midpoint_position_reserve_m <= .001
            or (midpoint_position_reserve_m and not adaptive_midpoints)):
        raise ValueError('An explicit midpoint position reserve in [0, 1 mm] requires adaptive refinement')
    if (not np.isfinite(midpoint_fixture_reserve_m) or not 0 <= midpoint_fixture_reserve_m <= .001
            or (midpoint_fixture_reserve_m and not adaptive_midpoints)):
        raise ValueError('An explicit midpoint fixture reserve in [0, 1 mm] requires adaptive refinement')
    if type(midpoint_quarter_probes) is not bool or (midpoint_quarter_probes and not adaptive_midpoints):
        raise ValueError('Quarter probes require an explicit boolean and adaptive refinement')
    if (isinstance(mobile_base_axis_speed_m_s,bool) or not np.isfinite(mobile_base_axis_speed_m_s)
            or not .05<=mobile_base_axis_speed_m_s<=.3
            or (mobile_base_axis_speed_m_s!=.3 and mobile_ik is None)):
        raise ValueError('Inserted mobile base axis speed must be in [0.05, 0.3] m/s')
    if lower_acceleration_limits is not None:
        lower_acceleration_limits=np.asarray(lower_acceleration_limits,float)
        if (lower_acceleration_limits.shape!=(17,) or not np.isfinite(lower_acceleration_limits).all()
                or np.any(lower_acceleration_limits<=0) or np.any(lower_acceleration_limits>20)
                or retreat_speed_factor!=1.):
            raise ValueError('Lower acceleration timing requires 17 caps in (0,20] and an unchanged retreat speed factor')
    if not (0 <= extra_descent_m <= .02 and .05 <= support_hold_s <= .5
            and .35 <= support_wait_s <= 2.
            and 0 < arm_speed_rad_s <= .8 and np.isfinite(retreat_speed_factor)
            and 1 <= retreat_speed_factor <= 4
            and np.isfinite(loaded_rotation_speed_rad_s)
            and .1 <= loaded_rotation_speed_rad_s <= .8):
        raise ValueError('Placement requires bounded descent, debounce and robot speed')
    times, reference, commands, hands, objects = map(np.asarray, prepared[7:12])
    if len(times) < 3 or not np.allclose(np.diff(times), .01, atol=1e-10, rtol=0):
        raise ValueError('Placement requires a complete 100 Hz robot control clock')
    closed = np.flatnonzero(commands < 0)
    releases = np.flatnonzero((np.arange(len(commands)) > closed[0]) & (commands >= 1.9999)) if len(closed) else []
    if not len(releases):
        raise ValueError('A recorded closed-to-open transition is required')
    release = int(releases[0]); anchor = hands[release].copy()
    base_offset = np.asarray(base_offset_xyyaw, dtype=float)
    rotation_vector = np.asarray(rotation_xyz_deg, dtype=float)
    rotation_limits = np.array([90., 90., 180. if mobile_ik is not None else 90.])
    if (base_offset.shape != (3,) or rotation_vector.shape != (3,)
            or not np.isfinite(base_offset).all() or not np.isfinite(rotation_vector).all()
            or np.linalg.norm(base_offset[:2]) > .4 or abs(base_offset[2]) > np.pi/2
            or np.any(abs(rotation_vector) > rotation_limits) or not 1.2 <= release_aperture_rad <= 2.):
        raise ValueError('Invalid bounded robot placement posture')
    mobile_solver = None
    if mobile_ik is not None:
        if not isinstance(mobile_ik, dict) or np.any(base_offset != 0.):
            raise ValueError('Mobile placement requires explicit options and no prescribed base offset')
        from .mobile_placement_ik import Solver
        from .feasible_maniskill import intent_parameters
        mobile_contact, mobile_closed_target = intent_parameters(prepared[5])
        if 'initial_base_yaw' in mobile_ik:
            raise ValueError('Mobile placement yaw must be anchored to the unchanged source release row')
        mobile_solver = Solver(prepared, robot, initial_base_yaw=float(reference[release, 2]), **mobile_ik)
    final_position = objects[-1,:3,3].copy()
    if placement_xy_world is not None:
        xy=np.asarray(placement_xy_world,dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():raise ValueError('Finite placement XY required')
        final_position[:2]=xy
        contract=prepared[5].get('task_contract',{})
        if 'bin_lower' not in contract:raise ValueError('Placement XY override requires a verified target region')
        frame=np.asarray(prepared[5]['world_placement'])
        local=frame[:3,:3].T@(final_position-frame[:3,3])
        if not (np.all(local>contract['bin_lower']) and np.all(local<contract['bin_upper'])):
            raise ValueError('Placement target lies outside the original verified task region')
    delta=final_position-objects[release,:3,3]
    if delta[2] > .02 or abs(delta[2]) > .3 or np.linalg.norm(delta[:2]) > .3:
        raise ValueError('Recorded placement is outside the bounded local translation policy')
    rotation=Rotation.from_euler('xyz',rotation_vector,degrees=True).as_matrix()
    relative_hand=anchor[:3,3]-objects[release,:3,3]
    bottom=anchor.copy();bottom[:3,:3]=rotation@anchor[:3,:3]
    bottom[:3,3]=final_position+rotation@relative_hand
    bottom[2,3]-=extra_descent_m
    traverse=bottom.copy();traverse[2,3]=max(anchor[2,3],bottom[2,3])
    names=['traverse','lower','hold','open','retreat','return']
    waypoints=[anchor,traverse,bottom,bottom,bottom,traverse,anchor]
    traverse_duration=max(.4,1.875*np.linalg.norm(traverse[:3,3]-anchor[:3,3])/.12,
                          1.875*Rotation.from_matrix(rotation).magnitude()/.8,
                          1.875*np.linalg.norm(base_offset[:2])/.3)
    loaded_traverse_duration=max(traverse_duration,
        1.875*Rotation.from_matrix(rotation).magnitude()/loaded_rotation_speed_rad_s)
    lower_duration=max(.5,1.875*abs(traverse[2,3]-bottom[2,3])/.07)
    durations=[traverse_duration,lower_duration,float(support_wait_s),.8,lower_duration,traverse_duration]
    proposal = output/'original-and-proposal.npz'
    np.savez_compressed(proposal, original_time_s=times, original_reference=reference,
                        original_gripper_intent=commands, original_hand_targets=hands,
                        original_object_targets=objects, waypoints=np.stack(waypoints))
    previous_active = robot.active.copy(); robot.active = robot.arm_v[7:]
    q = robot.pack(reference[release, 3:], reference[release, :3])
    knots, generated = [], []
    midpoint_reports = []
    timing_reports = []
    endpoints = {}
    return_source = None
    effective_retreat_speed_factor = 1.
    failure = None
    rejection_kind = None
    try:
        for name, first, last, duration in zip(names, waypoints[:-1], waypoints[1:], durations):
            if name in ('retreat','return'):
                original_name='lower' if name=='retreat' else 'traverse'
                _,out_refs,out_hands=next(item for item in generated if item[0]==original_name)
                if name=='return' and return_source is not None:
                    out_refs,out_hands=return_source
                end_ref,end_hand=endpoints[original_name]
                refs=np.concatenate([out_refs[1:],end_ref[None]])[::-1].copy()
                hs=np.concatenate([out_hands[1:],end_hand[None]])[::-1].copy()
                if name=='retreat' and retreat_speed_factor>1:
                    full_ref=np.concatenate([refs,out_refs[:1]])
                    full_hand=np.concatenate([hs,out_hands[:1]])
                    speeds=np.r_[mobile_base_axis_speed_m_s,mobile_base_axis_speed_m_s,.8,np.full(14,arm_speed_rad_s)]
                    peak_ratio=float(np.max(abs(np.diff(full_ref,axis=0))/.01/speeds))
                    factor=min(float(retreat_speed_factor),1./max(peak_ratio,1e-12))
                    ticks=max(1,int(np.ceil(len(refs)/max(1.,factor))))
                    effective_retreat_speed_factor=len(refs)/ticks
                    sample=np.arange(ticks)*len(refs)/ticks
                    refs=np.column_stack([np.interp(sample,np.arange(len(full_ref)),full_ref[:,j])for j in range(17)])
                    hs=np.tile(first,(ticks,1,1))
                    hs[:,:3,:3]=Slerp(np.arange(len(full_hand)),Rotation.from_matrix(full_hand[:,:3,:3]))(sample).as_matrix()
                    for axis in range(3):hs[:,axis,3]=np.interp(sample,np.arange(len(full_hand)),full_hand[:,axis,3])
                generated.append((name,refs,hs))
                continue
            count=max(2,int(np.ceil(duration/.04)))
            phase_q,phase_hands=[],[]
            interpolation=Slerp([0.,1.],Rotation.from_matrix(np.stack([first[:3,:3],last[:3,:3]])))
            for weight in np.linspace(0,1,count+1):
                smooth=weight**3*(10-15*weight+6*weight*weight)
                goal=first.copy();goal[:3,3]=(1-smooth)*first[:3,3]+smooth*last[:3,3]
                goal[:3,:3]=interpolation(smooth).as_matrix()
                base=reference[release,:3]+base_offset*(smooth if name=='traverse' else 1.)
                if mobile_solver is None:
                    q=robot.pack(q[robot.arm_ids],base)
                targets=robot.fk(q);targets[1]=goal
                if name=='traverse' and weight==0:
                    q=robot.pack(reference[release,3:],reference[release,:3])
                elif name not in ('hold','open'):
                    if mobile_solver is None:
                        q,_,_=robot.ik(targets,q,active_hands=(1,),iterations=150,joint_margin=.031)
                    else:
                        upper = 2. if name == 'traverse' else max(release_aperture_rad, mobile_contact)
                        q = mobile_solver.solve(goal, q, np.linspace(mobile_closed_target, upper, 5),
                                                phase=name, weight=weight)
                if mobile_solver is not None:
                    base = mobile_solver.reference_base(q)
                phase_q.append(np.r_[base,reference[release,3:10],q[robot.arm_ids[7:]]])
                phase_hands.append(goal);knots.append(q.copy())
                if mobile_solver is not None and mobile_solver.records:
                    checked = mobile_solver.records[-1]
                    if (np.any(np.asarray(checked['ik_errors']) > [.002, .02])
                            or not checked['fixture']['admitted']):
                        rejection_kind = 'mobile_knot_admission'
                        raise ValueError('Mobile placement knot failed before interpolation: '
                                         + name + ' at weight ' + str(float(weight)))
            phase_q,phase_hands=np.asarray(phase_q),np.asarray(phase_hands)
            knot_weights=np.linspace(0,1,count+1)
            if adaptive_midpoints and name in ('traverse','lower'):
                from .placement_midpoints import refine
                upper=2. if name=='traverse' else max(release_aperture_rad,mobile_contact)
                phase_q,phase_hands,knot_weights,refinement=refine(robot,mobile_solver,phase_q,phase_hands,
                    np.linspace(mobile_closed_target,upper,5),output/('midpoint-'+name),
                    phase=name,rounds=adaptive_midpoints,minimum_motion=midpoint_minimum_motion,
                    position_reserve_m=midpoint_position_reserve_m,
                    fixture_reserve_m=midpoint_fixture_reserve_m,quarter_probes=midpoint_quarter_probes)
                midpoint_reports.append(refinement)
            speeds=np.r_[mobile_base_axis_speed_m_s,mobile_base_axis_speed_m_s,.8,np.full(14,arm_speed_rad_s)]
            required=max(duration,float(np.max(np.abs(np.diff(phase_q,axis=0))/speeds)*count)) if not adaptive_midpoints else max(
                duration,float(np.max(np.abs(np.diff(phase_q,axis=0))/speeds/np.diff(knot_weights)[:,None])))
            if interval_knot_timing:
                required=duration
            if lower_acceleration_limits is not None and name=='lower':
                required=duration
            def sample_phase(seconds,*,loaded=False):
                progress=knot_weights
                acceleration_timing=lower_acceleration_limits is not None and name=='lower'
                if interval_knot_timing or acceleration_timing:
                    from .placement_midpoints import interval_times
                    cap=loaded_rotation_speed_rad_s if loaded and name=='traverse' else .8
                    if acceleration_timing:
                        from .placement_midpoints import acceleration_limited_times
                        interval_clock=acceleration_limited_times(phase_q,phase_hands,knot_weights,duration,
                            speeds,lower_acceleration_limits,angular_speed=cap)
                    else:
                        interval_clock=interval_times(phase_q,phase_hands,knot_weights,duration,speeds,angular_speed=cap)
                    seconds=max(seconds,float(interval_clock[-1]))
                    interval_clock*=seconds/interval_clock[-1]
                    progress=interval_clock/interval_clock[-1]
                    timing_file=output/('interval-clock-'+name+('-loaded' if loaded else '-return')+'.npz')
                    np.savez_compressed(timing_file,reference=phase_q,hand_targets=phase_hands,
                                        original_knot_weights=knot_weights,time_s=interval_clock)
                    timing_reports.append(dict(phase=name,loaded=loaded,seconds=seconds,
                        artifact=str(timing_file),artifact_sha256=sha256(timing_file),angular_cap_rad_s=cap,
                        progress_law='per-segment quintic' if acceleration_timing else 'linear',
                        acceleration_limits=None if not acceleration_timing else lower_acceleration_limits.tolist()))
                ticks=max(1,int(np.ceil(seconds/.01)));w=np.arange(ticks)/ticks
                if acceleration_timing:
                    from .placement_midpoints import quintic_knot_progress
                    w=quintic_knot_progress(w*interval_clock[-1],interval_clock)
                    progress=np.arange(len(phase_q),dtype=float)
                sampled_q=np.column_stack([np.interp(w,progress,phase_q[:,j])for j in range(17)])
                sampled_hand=np.tile(first,(ticks,1,1))
                sampled_hand[:,:3,:3]=Slerp(progress,Rotation.from_matrix(phase_hands[:,:3,:3]))(w).as_matrix()
                for axis in range(3):sampled_hand[:,axis,3]=np.interp(w,progress,phase_hands[:,axis,3])
                return sampled_q,sampled_hand
            if name=='traverse':
                # Identical geometric IK knots and unloaded return at every cap.
                # Only the loaded leg receives the additional angular duration.
                return_source=sample_phase(required)
                required=max(required,loaded_traverse_duration)
            sampled_q,sampled_hand=sample_phase(required,loaded=True)
            endpoints[name]=(phase_q[-1].copy(),phase_hands[-1].copy())
            generated.append((name,sampled_q,sampled_hand))
    except Exception as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
        if adaptive_midpoints:
            from .placement_midpoints import MidpointAdmissionError
            if isinstance(exc,MidpointAdmissionError):
                rejection_kind='mobile_knot_admission'
    finally:
        robot.active = previous_active
    np.savez_compressed(output/'ik-knots.npz', q=np.asarray(knots))
    if mobile_solver is not None:
        json_write(output/'mobile-ik-knots.json', dict(mobile_solver.metadata,
                                                     records=mobile_solver.records,
                                                     adaptive_midpoint_reports=midpoint_reports))
    if failure:
        rejection = dict(admitted=False, exception=failure, physics_validated=False,
                         completed_frames=0, total_frames=None)
        if mobile_solver is not None:
            rejection.update(complete=False, completed_knot_count=len(knots), original_source_rows=len(times),
                failed_knot=mobile_solver.records[-1] if mobile_solver.records else None,
                numeric_knots_sha256=sha256(output/'ik-knots.npz'),
                original_proposal_sha256=sha256(proposal),
                mobile_ik=mobile_solver.metadata,
                scope='Retained original arrays, complete proposed waypoints and partial numeric knots; no interpolated-path admission or physics')
            if rejection_kind is not None:
                rejection['rejection_kind'] = rejection_kind
        json_write(output/'result.json', rejection)
        raise ValueError('Supported placement IK rejected; inspect '+str(output/'result.json'))
    phases = {}; cursor = release
    for name, refs, _ in generated:
        phases[name] = [cursor, cursor+len(refs)]; cursor += len(refs)
    added = cursor-release
    inserted_reference = np.concatenate([g[1] for g in generated])
    inserted_hands = np.concatenate([g[2] for g in generated])
    traverse_hands=np.concatenate([generated[0][2],endpoints['traverse'][1][None]])
    angular_peak=float(np.max(Rotation.from_matrix(
        traverse_hands[1:,:3,:3]@traverse_hands[:-1,:3,:3].transpose(0,2,1)).magnitude())/.01)
    changed_reference = np.concatenate([reference[:release], inserted_reference, reference[release:]])
    changed_hands = np.concatenate([hands[:release], inserted_hands, hands[release:]])
    changed_commands = np.concatenate([commands[:release], np.full(added, float(commands[release-1])), commands[release:]])
    # Opening remains runtime-gated. The nominal plan declares its intended phase.
    changed_commands[phases['open'][0]:phases['return'][0]] = release_aperture_rad
    changed_commands[phases['return'][0]:cursor] = 2.
    changed_objects = np.concatenate([objects[:release], np.repeat(objects[release:release+1],added,axis=0), objects[release:]])
    changed_time = times[0]+np.arange(len(changed_reference))*.01
    original_indices = np.r_[np.arange(release), np.arange(release,len(times))+added]
    model, _, manifest = prepared[:3]
    body = model.body(manifest['objects'][prepared[5]['object_id']]['body']).id
    robot_bodies = {model.body('base_link').id}
    for child in range(1,model.nbody):
        if int(model.body_parentid[child]) in robot_bodies: robot_bodies.add(child)
    data = mujoco.MjData(model)
    errors=np.full((len(changed_reference),2),np.nan)
    depths=np.full((len(changed_reference),2),np.nan)
    margins=np.full(len(changed_reference),np.nan); gaps=margins.copy()
    closed_target=float(prepared[5].get('effective_candidate',{}).get('gripper_closed_target',-.06))
    if not np.isfinite(closed_target) or not -.06 <= closed_target <= 2.:
        raise ValueError('Finite bounded closed gripper actuator target required')
    aperture_samples={'inside_bin':np.linspace(closed_target,release_aperture_rad,5).tolist(),
                      'above_bin':np.linspace(closed_target,2.,5).tolist()}
    if mobile_solver is not None:
        aperture_samples['inside_bin'] = np.linspace(min(mobile_closed_target, release_aperture_rad),
            max(mobile_contact, release_aperture_rad), 5).tolist()
    completed_frames = 0
    mobile_fixture_gaps = np.full(len(changed_reference), np.nan) if mobile_solver is not None else None
    try:
        for i, ref in enumerate(changed_reference):
            q=robot.pack(ref[3:],ref[:3]); actual=robot.fk(q)[1]
            errors[i]=[np.linalg.norm(actual[:3,3]-changed_hands[i,:3,3]),
                       Rotation.from_matrix(changed_hands[i,:3,:3]@actual[:3,:3].T).magnitude()]
            apertures=(aperture_samples['inside_bin'] if phases['lower'][0]<=i<phases['return'][0]
                       else aperture_samples['above_bin']) if release<=i<cursor else [2.]
            depths[i]=0.
            for grip in apertures:
                initialize(model,data,robot,q,manifest['mimics'],
                           {'l_hand_finger':2., 'r_hand_finger':float(grip)})
                collision,_=collision_depths(model,data,robot_bodies,body)
                depths[i]=np.maximum(depths[i],[collision['environment'],collision['self']])
            gaps[i],margins[i],_=robot.r.geometry(q)
            if mobile_solver is not None:
                proof = mobile_solver.inspect(q, apertures)
                mobile_fixture_gaps[i] = proof['minimum_signed_distance_or_lower_bound_m']
            completed_frames += 1
    except Exception as exc:
        failure=type(exc).__name__+': '+str(exc)
    complete=failure is None and np.isfinite(errors).all() and np.isfinite(depths).all()
    peak=float(np.max(abs(np.diff(changed_reference[:,10:],axis=0))/.01))
    phase_slice=slice(release-1,cursor+1)
    inserted_peak=float(np.max(abs(np.diff(changed_reference[phase_slice,10:],axis=0))/.01))
    checks=dict(complete=bool(complete),hand_ik=bool(complete and np.all(errors<=[.002,.02])),
                environment=bool(complete and np.max(depths[:,0])<=.002),
                self_collision=bool(complete and np.max(depths[:,1])<=.002),
                self_clearance=bool(complete and np.min(gaps)>=.009),
                joint_margin=bool(complete and np.min(margins)>=.025),
                loaded_rotation_speed=bool(angular_peak<=loaded_rotation_speed_rad_s+1e-7),
                inserted_reference_speed=bool(inserted_peak<=max(.8,arm_speed_rad_s)+1e-7))
    mobile_speed_peak = None
    if mobile_solver is not None:
        # Internal inserted differences use their declared slower limits. Both
        # source boundary intervals remain recorded and are checked against the
        # previously declared source reference limits as well.
        mobile_speed_peak = np.max(np.abs(np.diff(inserted_reference, axis=0))/.01, axis=0)
        limits = np.r_[mobile_base_axis_speed_m_s,mobile_base_axis_speed_m_s,.8,np.full(14,arm_speed_rad_s)]
        boundary_peak = np.max(np.abs(np.diff(changed_reference[phase_slice], axis=0))/.01, axis=0)
        boundary_limits = np.r_[.55, .55, 1.5, np.full(14, .8)]
        checks.update(mobile_positive_fixture_clearance=bool(complete and np.all(
            mobile_fixture_gaps >= mobile_solver.minimum)),
            mobile_inserted_reference_speed=bool(np.all(mobile_speed_peak <= limits + 1e-7)),
            mobile_boundary_reference_speed=bool(np.all(boundary_peak <= boundary_limits + 1e-7)),
            original_references_exact=bool(np.array_equal(changed_reference[original_indices], reference)),
            original_hands_exact=bool(np.array_equal(changed_hands[original_indices], hands)),
            original_objects_exact=bool(np.array_equal(changed_objects[original_indices], objects)))
    inherited=prepared[5].get('robot_control_retiming',{})
    source_time=times.copy()
    if inherited.get('source_clock_artifact'):
        if mobile_solver is not None and sha256(inherited['source_clock_artifact']) != inherited.get('artifact_sha256'):
            raise ValueError('Mobile placement inherited source-clock checksum differs')
        with np.load(inherited['source_clock_artifact']) as saved:
            source_time=saved['source_reference_time_s'].copy()
        if source_time.shape!=times.shape: raise ValueError('Inherited source clock does not align')
        if mobile_solver is not None and (not np.isfinite(source_time).all() or np.any(np.diff(source_time)<0)):
            raise ValueError('Mobile placement requires a finite monotonic inherited source clock')
    mapped=np.r_[source_time[:release],np.full(added,source_time[release]),source_time[release:]]
    artifact=output/'placement-admission.npz'
    np.savez_compressed(artifact,control_time_s=changed_time,source_reference_time_s=mapped,
                        original_time_s=times,original_row_indices=original_indices,
                        reference=changed_reference,hand_targets=changed_hands,gripper_intent=changed_commands,
                        errors=errors,collision_depths=depths,self_clearance_m=gaps,joint_margin_rad=margins)
    report=dict(admitted=all(checks.values()),admission_checks=checks,physics_validated=False,
                completed_frames=completed_frames,total_frames=len(changed_reference),
                exception=failure,original_release_frame=release,inserted_frames=added,phase_frames=phases,
                support_hold_s=float(support_hold_s),support_wait_s=float(support_wait_s),support_min_weight_fraction=.5,
                extra_descent_m=float(extra_descent_m),arm_speed_rad_s=float(arm_speed_rad_s),
                loaded_rotation_speed_rad_s=float(loaded_rotation_speed_rad_s),
                loaded_traverse_angular_peak_rad_s=angular_peak,
                nominal_loaded_traverse_duration_s=float(loaded_traverse_duration),
                nominal_unloaded_return_duration_s=float(traverse_duration),
                loaded_rotation_scope='Time dilation of inserted loaded traverse only; geometric IK knots, original source rows and unloaded return remain unchanged',
                placement_position_world=final_position.tolist(),base_offset_xyyaw=base_offset.tolist(),
                rotation_xyz_deg=rotation_vector.tolist(),release_aperture_rad=float(release_aperture_rad),
                requested_retreat_speed_factor=float(retreat_speed_factor),
                effective_retreat_speed_factor=float(effective_retreat_speed_factor),
                source_clock_artifact=str(artifact),artifact_sha256=sha256(artifact),
                original_artifact=str(proposal),original_artifact_sha256=sha256(proposal),
                inserted_duration_s=added*.01,peak_reference_speed_rad_s=peak,
                adaptive_midpoints=adaptive_midpoints,midpoint_minimum_motion=midpoint_minimum_motion,
                midpoint_position_reserve_m=float(midpoint_position_reserve_m),
                midpoint_fixture_reserve_m=float(midpoint_fixture_reserve_m),
                midpoint_quarter_probes=midpoint_quarter_probes,
                mobile_base_axis_speed_m_s=float(mobile_base_axis_speed_m_s),
                lower_acceleration_limits=None if lower_acceleration_limits is None else lower_acceleration_limits.tolist(),
                lower_acceleration_scope='Optional time-only per-segment quintic in lower and exactly reversed retreat; source and traverse unchanged',
                interval_knot_timing=interval_knot_timing,interval_clock_artifacts=timing_reports,
                original_interval_complete=True,object_state_assignment='none',
                support_guard_version=1,support_position_tolerance_m=.04,
                support_grasp_recency_s=.2,
                admitted_gripper_apertures_rad=aperture_samples,
                scope='Inserted robot-only placement posture, support-gated opening and retreat; every original row retained',
                object_reference_semantics='Inserted object references repeat the paused source release pose; they are neither measured nor prescribed physical object states')
    if complete:
        report.update(max_ik_errors=errors.max(0).tolist(),max_collision_depths_m=depths.max(0).tolist(),
                      min_self_clearance_m=float(gaps.min()),min_joint_margin_rad=float(margins.min()))
    if mobile_solver is not None:
        report['mobile_ik'] = dict(mobile_solver.metadata,
            knot_artifact=str(output/'mobile-ik-knots.json'), knot_sha256=sha256(output/'mobile-ik-knots.json'),
            minimum_signed_distance_or_lower_bound_m=(float(np.nanmin(mobile_fixture_gaps))
                                                     if np.isfinite(mobile_fixture_gaps).any() else None),
            inserted_speed_peak=mobile_speed_peak.tolist() if mobile_speed_peak is not None else None,
            source_prefix_suffix_exact=True, controller='Unchanged measured-support Controller')
        np.save(output/'mobile-fixture-gaps.npy', mobile_fixture_gaps, allow_pickle=False)
        report['mobile_ik']['fixture_gap_artifact_sha256'] = sha256(output/'mobile-fixture-gaps.npy')
    json_write(output/'result.json',report)
    if not report['admitted']: raise ValueError('Supported placement admission rejected; inspect '+str(output/'result.json'))
    result=list(prepared); details=deepcopy(prepared[5]); details['robot_supported_placement']=report
    # The common exporter reads this established clock pointer.
    details['robot_control_retiming']=dict(inherited,source_clock_artifact=str(artifact),
        artifact_sha256=sha256(artifact),original_interval_complete=True,
        inserted_phase='Support-gated robot placement; source clock paused',duration_s=float(changed_time[-1]))
    details['duration_s']=float(changed_time[-1]);result[5]=details
    result[7:12]=[changed_time,changed_reference,changed_commands,changed_hands,changed_objects]
    return tuple(result)


class Controller:
    """Support feedback selects a prefix/reversal of the pre-admitted descent.

    No simulator object or robot state is mutated here. A failed support check
    leaves the grasp closed and freezes the robot, retaining a diagnostic failure.
    """
    def __init__(self, details, reference, goals, commands):
        self.policy=details['robot_supported_placement'];self.phases=self.policy['phase_frames']
        self.reference=np.array(reference,copy=True);self.goals=np.array(goals,copy=True)
        self.commands=np.array(commands,copy=True)
        self.contact_index=None;self.stable_s=0.;self.supported=False;self.failed=False
        self.opened=False;self.last_step=-1;self.state={}
        self.last_intact_grasp_step=None;self.support_grasp_verified=False;self.opening_verified=False

    def update(self, step, measured_support_N, object_weight_N, *, object_position_world=None, grasp_intact=False):
        if step!=self.last_step+1: raise ValueError('Placement controller requires consecutive control steps')
        self.last_step=step
        phase=next((k for k,(a,b) in self.phases.items() if a<=step<b),'source')
        ref=self.reference[step].copy();goal=self.goals[step].copy();command=float(self.commands[step])
        lower_start,lower_end=self.phases['lower']; open_start=self.phases['open'][0]
        valid=np.isfinite(measured_support_N) and np.isfinite(object_weight_N) and object_weight_N>0
        position=np.asarray(object_position_world,dtype=float) if object_position_world is not None else np.full(3,np.nan)
        target=np.asarray(self.policy['placement_position_world'],dtype=float)
        position_error=float(np.linalg.norm(position-target)) if position.shape==(3,) and np.isfinite(position).all() else None
        near=bool(position_error is not None and position_error<=self.policy.get('support_position_tolerance_m',.04))
        if grasp_intact and not self.opened:self.last_intact_grasp_step=step
        age=None if self.last_intact_grasp_step is None else (step-self.last_intact_grasp_step)*.01
        recent=bool(age is not None and age<=self.policy.get('support_grasp_recency_s',.2)+1e-10)
        force_support=bool(valid and measured_support_N>=max(.03,self.policy['support_min_weight_fraction']*object_weight_N))
        support=bool(force_support and near and (recent or self.support_grasp_verified))
        # The force belongs to the preceding physical step. Freezing at that row
        # avoids issuing another downward command after support is observed.
        if phase in ('lower','hold') and not self.supported:
            if support and self.contact_index is None:
                self.contact_index=min(lower_end-1,max(lower_start,step-1));self.support_grasp_verified=recent
            if self.contact_index is not None:
                self.stable_s=self.stable_s+.01 if support else 0.
                self.supported=self.stable_s+1e-10>=self.policy['support_hold_s']
        if not near and not self.opened:self.supported=False;self.stable_s=0.
        if step==open_start and not (self.supported and near and self.support_grasp_verified):self.failed=True
        if self.failed:
            index=self.contact_index if self.contact_index is not None else lower_end-1
            ref=self.reference[index].copy();goal=self.goals[index].copy();command=-.06
        elif self.contact_index is not None and phase in ('lower','hold','open'):
            ref=self.reference[self.contact_index].copy();goal=self.goals[self.contact_index].copy()
            command=self.policy.get('release_aperture_rad',2.) if phase=='open' and self.supported else -.06
        elif self.contact_index is not None and phase=='retreat':
            # Reverse the actual admitted descent prefix. The planned retreat
            # resampling caps the full-path speed; every shorter prefix is bounded
            # by the same cap, without changing any grasp/support/opening row.
            a,b=self.phases['retreat'];w=(step-a)/(b-a)
            path_position=self.contact_index*(1-w)+lower_start*w
            lo=int(np.floor(path_position));hi=min(lo+1,lower_end-1);fraction=path_position-lo
            ref=(1-fraction)*self.reference[lo]+fraction*self.reference[hi]
            goal=self.goals[lo].copy();goal[:3,3]=(1-fraction)*self.goals[lo,:3,3]+fraction*self.goals[hi,:3,3]
            command=self.policy.get('release_aperture_rad',2.)
        elif phase in ('lower','hold'):
            command=-.06
        if phase=='open' and command>=self.policy.get('release_aperture_rad',2.)-1e-8:
            if not self.opened:self.opening_verified=bool(self.supported and near and self.support_grasp_verified)
            self.opened=True
        self.state=dict(phase=phase,support_force_N=float(measured_support_N),support_threshold_N=max(.03,self.policy['support_min_weight_fraction']*object_weight_N),
                        support_contact_frame=self.contact_index,stable_support_s=self.stable_s,
                        support_confirmed=bool(self.supported),opened=bool(self.opened),failed=bool(self.failed),
                        support_guard_version=1,measured_object_position_world=position.tolist() if position.shape==(3,) and np.isfinite(position).all() else None,
                        placement_position_world=target.tolist(),object_position_error_m=position_error,
                        object_near_placement=near,grasp_intact=bool(grasp_intact),last_intact_grasp_step=self.last_intact_grasp_step,
                        grasp_age_s=age,support_grasp_verified=bool(self.support_grasp_verified),opening_guard_verified=bool(self.opening_verified),
                        failure_reason='No stable measured support with object near placement and recent intact grasp before opening deadline' if self.failed else None)
        return ref,goal,command


def audit_opening_guard(states, object_positions, bilateral_contact, policy, *, reference_indices=None):
    """Check location/grasp evidence against recorded previous-interval states.

    This supplements exact actuator replay. Support forces are controller trace
    values; this check does not independently recompute contact forces.
    """
    objects=np.asarray(object_positions);bilateral=np.asarray(bilateral_contact)
    if len(states)!=len(objects) or len(states)!=len(bilateral):
        raise ValueError('Aligned controller and measured object/contact records required')
    indices = np.arange(len(states)) if reference_indices is None else np.asarray(reference_indices)
    if (indices.shape != (len(states),) or indices.dtype.kind not in 'iu' or not len(indices)
            or indices[0] != 0 or np.any((np.diff(indices) < 0) | (np.diff(indices) > 1))):
        raise ValueError('Source-indexed placement audit requires a complete ordered prefix')
    checks=dict(measured_position=True,intact_grasp=True,grasp_history=True,opening_authorized=True)
    target=np.asarray(policy['placement_position_world']);tolerance=policy.get('support_position_tolerance_m',.04)
    recency=policy.get('support_grasp_recency_s',.2);last=None;opened=False;verified=False;events=[]
    for i,state in enumerate(states):
        memory=state['memory'];guard=memory['supported_placement']
        source_step = int(indices[i])
        if i > 0 and source_step == indices[i-1]:
            if (source_step >= policy['phase_frames']['traverse'][0]
                    or guard != states[i-1]['memory']['supported_placement']):
                raise ValueError('Only an unchanged pre-placement controller may pause on a source row')
            continue
        intact=bool(i>0 and bilateral[i-1]>=.95 and memory.get('drift_anchor') is not None)
        checks['intact_grasp'] &= guard.get('grasp_intact') is intact
        if intact and not opened:last=source_step
        checks['grasp_history'] &= guard.get('last_intact_grasp_step')==last
        position=np.asarray(guard.get('measured_object_position_world'),dtype=float)
        if i>0:
            checks['measured_position'] &= bool(position.shape==(3,) and np.allclose(position,objects[i-1,:3],rtol=0,atol=1e-8))
        near=bool(position.shape==(3,) and np.isfinite(position).all() and np.linalg.norm(position-target)<=tolerance)
        if guard.get('support_contact_frame') is not None and not verified:
            verified=bool(last is not None and (source_step-last)*.01<=recency+1e-10 and near)
            checks['grasp_history'] &= verified
        if guard.get('opened') and not opened:
            allowed=bool(near and verified and guard.get('support_grasp_verified')
                         and guard.get('support_confirmed') and guard.get('opening_guard_verified')
                         and guard.get('stable_support_s',0.)+1e-10>=policy['support_hold_s'])
            checks['opening_authorized'] &= allowed
            events.append(dict(frame=i,authorized=allowed,object_position=position.tolist()))
        opened=bool(guard.get('opened'))
    return dict(applicable=True,passed=all(checks.values()),checks=checks,opening_events=events,
                scope='Recorded previous-interval object position, bilateral contact and acquired grasp anchor; contact-force values are not independently recomputed here')
