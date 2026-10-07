"""Locally slow a complete robot path to bounded reference speed.

Original timestamps and all original samples are retained in the time-map
artifact. No source interval is cropped and no object state is prescribed.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d, gaussian_filter1d
from .episodes import interpolate_pose, matrices_to_pose, pose_to_matrices
from .store import json_write, sha256


def segment_durations(reference, seconds, speed, *, planar_speed_m_s=None):
    reference, seconds = np.asarray(reference), np.asarray(seconds)
    delta = np.diff(reference, axis=0)
    original = np.diff(seconds)
    if len(original) < 1 or not np.all(original > 0):
        raise ValueError('Retiming requires an increasing complete clock')
    required=np.maximum(1.,np.max(np.abs(delta)/speed,axis=1)/original)
    if planar_speed_m_s is not None:
        if (reference.shape[1] < 3 or not np.isscalar(planar_speed_m_s)
                or not np.isfinite(planar_speed_m_s) or not 0 < planar_speed_m_s <= .55):
            raise ValueError('A planar base speed norm in (0, .55] m/s is required')
        required=np.maximum(required,np.linalg.norm(delta[:,:2],axis=1)/planar_speed_m_s/original)
    window=max(1,int(np.ceil(.10/np.median(original))))
    envelope=maximum_filter1d(required,size=2*window+1,mode='nearest')
    smooth=gaussian_filter1d(envelope,sigma=max(1,window/2),mode='nearest')
    # Never speed up any segment. Smooth only the dilation envelope; the
    # original geometric reference stays intact. Acceleration is diagnostic,
    # not falsely claimed bounded for an arbitrary piecewise-linear IK path.
    return original*np.maximum(required,smooth)


def prepare(prepared, output, *, arm_speed_rad_s=.8, planar_speed_m_s=None):
    if not np.isfinite(arm_speed_rad_s) or not .1 <= arm_speed_rad_s <= .8:
        raise ValueError('Declared arm reference speed must be in [0.1, 0.8] rad/s')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    original=np.asarray(prepared[7]); reference=np.asarray(prepared[8])
    speed=np.r_[.55,.55,1.5,np.full(14,arm_speed_rad_s)]
    segments=segment_durations(reference,original,speed,planar_speed_m_s=planar_speed_m_s)
    warped=np.r_[original[0],original[0]+np.cumsum(segments)]
    # The last sample may extend the final hold by less than one control tick.
    seconds=original[0]+np.arange(int(np.ceil((warped[-1]-warped[0])/.01))+1)*.01
    source_clock=np.interp(seconds,warped,original)
    changed=np.column_stack([np.interp(source_clock,original,reference[:,i]) for i in range(reference.shape[1])])
    commands=prepared[9][np.clip(np.searchsorted(original,source_clock,side='right')-1,0,len(original)-1)]
    poses=[pose_to_matrices(interpolate_pose(original,matrices_to_pose(prepared[i]),source_clock)) for i in (10,11)]
    artifact=output/'time-map.npz'
    np.savez_compressed(artifact,original_time_s=original,warped_original_time_s=warped,
                        control_time_s=seconds,source_reference_time_s=source_clock,
                        original_reference=reference,retimed_reference=changed)
    report=dict(method='local segment dilation with bounded reference speed and smoothed dilation envelope; linear joints, pose SLERP, zero-order held gripper intent',
                original_duration_s=float(original[-1]-original[0]),duration_s=float(seconds[-1]-seconds[0]),
                duration_ratio=float((seconds[-1]-seconds[0])/(original[-1]-original[0])),
                speed_limits=speed.tolist(),
                planar_speed_norm_limit_m_s=planar_speed_m_s,
                peak_reference_acceleration=np.max(abs(np.diff(changed,n=2,axis=0)/.01**2),axis=0).tolist(),
                original_interval_complete=True,source_clock_artifact=str(artifact),artifact_sha256=sha256(artifact),
                physics_validated=False,note='Reference limits are planning constraints; actual physical speed remains independently gated')
    json_write(output/'result.json',report)
    if report['duration_ratio']>3:
        raise ValueError('Required local retiming exceeds the declared 3x total-duration budget; artifact retained')
    result=list(prepared);details=deepcopy(prepared[5])
    details['robot_control_retiming']=report;details['duration_s']=float(seconds[-1])
    details['frozen_factors']=[s for s in details.get('frozen_factors',[]) if s!='source clock']
    result[5]=details;result[7]=seconds;result[8]=changed;result[9]=commands;result[10:12]=poses
    return tuple(result)
