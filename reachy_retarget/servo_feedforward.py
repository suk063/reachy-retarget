"""Bounded position-command feedforward for the recorded robot reference.

For a position servo kp*(target-q)-kv*qdot, adding kv/kp*qdot_ref
cancels the damping lag at constant reference velocity. The arm and optional
source-prefix or whole-source base helpers change actuator targets only; simulator state,
object forces and reference timing do not.
"""
import numpy as np


def velocity_offsets(model, actuators, reference, seconds, scale):
    result = np.zeros_like(reference)
    if scale == 0:
        return result
    if not 0 < scale <= 1:
        raise ValueError('Feedforward scale must be in (0, 1]')
    ids = np.asarray(actuators)[3:]
    kp = model.actuator_gainprm[ids, 0]
    kv = -model.actuator_biasprm[ids, 2]
    if np.any(kp <= 0) or np.any(kv < 0):
        raise ValueError('Feedforward requires positive position servo gains')
    rates = np.gradient(reference[:, 3:], seconds, axis=0)
    result[:, 3:] = np.clip(scale*kv/kp*rates, -.20, .20)
    return result


def source_prefix_offsets(model, actuators, reference, seconds, scale, stop_index, *, taper_s=.1):
    """Compensate only the source prefix; all inserted/feedback phases stay zero."""
    seconds=np.asarray(seconds)
    if not isinstance(stop_index,(int,np.integer)) or not 2<=stop_index<=len(seconds):
        raise ValueError('Source prefix must contain at least two complete reference rows')
    if not np.isfinite(taper_s) or taper_s<=0:
        raise ValueError('A positive feedforward boundary taper is required')
    result=velocity_offsets(model,actuators,reference,seconds,scale)
    distance=np.minimum(seconds-seconds[0],seconds[stop_index-1]-seconds)
    phase=np.clip(distance/taper_s,0.,1.)
    weight=phase**3*(10.-15.*phase+6.*phase**2)
    result*=weight[:,None]
    result[stop_index:]=0.
    return result


def source_prefix_base_offsets(model, actuators, reference, seconds, scale, stop_index, *, taper_s=.1):
    """Bounded base-servo compensation, zero outside the original source prefix.

    Translation offsets are limited to 50 mm per world axis and yaw to 0.1 rad.
    This changes commands only. Actual base speed and contact remain rollout
    gates; reference limits alone do not establish physical feasibility.
    """
    reference=np.asarray(reference)
    seconds=np.asarray(seconds)
    if (reference.ndim!=2 or reference.shape[1]<3 or len(reference)!=len(seconds)
            or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(seconds))
            or np.any(np.diff(seconds)<=0)):
        raise ValueError('Finite base references on a strictly increasing clock required')
    if not isinstance(stop_index,(int,np.integer)) or not 2<=stop_index<=len(seconds):
        raise ValueError('Source prefix must contain at least two complete reference rows')
    if not np.isfinite(scale) or not 0<=scale<=1 or not np.isfinite(taper_s) or taper_s<=0:
        raise ValueError('Bounded base compensation and a positive taper are required')
    result=np.zeros_like(reference)
    if scale==0:
        return result
    ids=np.asarray(actuators)[:3]
    kp=model.actuator_gainprm[ids,0]
    kv=-model.actuator_biasprm[ids,2]
    if np.any(kp<=0) or np.any(kv<0) or not np.all(np.isfinite(kp+kv)):
        raise ValueError('Base compensation requires finite positive position servo gains')
    rates=np.gradient(reference[:stop_index,:3],seconds[:stop_index],axis=0)
    limits=np.array([.05,.05,.1])
    offsets=np.clip(scale*kv/kp*rates,-limits,limits)
    distance=np.minimum(seconds[:stop_index]-seconds[0],seconds[stop_index-1]-seconds[:stop_index])
    phase=np.clip(distance/taper_s,0.,1.)
    weight=phase**3*(10.-15.*phase+6.*phase**2)
    result[:stop_index,:3]=offsets*weight[:,None]
    return result


def base_velocity_offsets(model, actuators, reference, seconds, scale, *, taper_s=.1):
    """Compensate base damping over the whole immutable reference interval.

    Columns and actuators must be ordered world-X slide, world-Y slide, yaw
    hinge, then the remaining joints. Unit transmission gears make the first
    two actuator targets meters and the third radians. This function changes
    neither the model gains nor any reference/state array. It is unsuitable
    for a feedback-paused reference unless its executed clock is supplied.
    """
    import mujoco

    reference = np.asarray(reference, dtype=float)
    seconds = np.asarray(seconds, dtype=float)
    ids = np.asarray(actuators)
    if (reference.ndim != 2 or reference.shape[1] < 3 or seconds.ndim != 1
            or len(seconds) < 2 or len(reference) != len(seconds)
            or not np.isfinite(reference).all() or not np.isfinite(seconds).all()
            or np.any(np.diff(seconds) <= 0)):
        raise ValueError('Finite aligned base references and a strictly increasing clock required')
    if (not np.isscalar(scale) or not np.isfinite(scale) or not 0 <= scale <= 1
            or not np.isscalar(taper_s) or not np.isfinite(taper_s) or taper_s <= 0):
        raise ValueError('Base compensation scale must be in [0, 1] with a positive taper')
    if (ids.ndim != 1 or len(ids) != reference.shape[1]
            or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= model.nu)
            or len(set(map(int, ids[:3]))) != 3):
        raise ValueError('Aligned actuator IDs and three distinct base actuators required')
    base = ids[:3]
    joint_transmission = int(mujoco.mjtTrn.mjTRN_JOINT)
    if np.any(model.actuator_trntype[base] != joint_transmission):
        raise ValueError('Base actuator units require direct joint transmissions')
    joints = model.actuator_trnid[base, 0]
    types = [int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_SLIDE),
             int(mujoco.mjtJoint.mjJNT_HINGE)]
    if (len(set(map(int, joints))) != 3 or not np.array_equal(model.jnt_type[joints], types)
            or not np.allclose(model.actuator_gear[base], [1., 0., 0., 0., 0., 0.], atol=1e-12, rtol=0)):
        raise ValueError('Base actuator units must be unit-gear meters, meters, radians')
    kp = model.actuator_gainprm[base, 0]
    kv = -model.actuator_biasprm[base, 2]
    if (np.any(model.actuator_dyntype[base] != int(mujoco.mjtDyn.mjDYN_NONE))
            or np.any(model.actuator_gaintype[base] != int(mujoco.mjtGain.mjGAIN_FIXED))
            or np.any(model.actuator_biastype[base] != int(mujoco.mjtBias.mjBIAS_AFFINE))
            or not np.isfinite(kp+kv).all() or np.any(kp <= 0) or np.any(kv < 0)
            or not np.allclose(model.actuator_biasprm[base, 1], -kp, atol=1e-12, rtol=0)):
        raise ValueError('Base compensation requires unchanged calibrated position servos')
    return source_prefix_base_offsets(model, ids, reference, seconds, scale, len(seconds), taper_s=taper_s)


class PostAcquisitionBaseOffsets:
    """Read-only base feedforward after a measured acquisition has resumed.

    The original reference determines velocities. Actual integration time and
    consecutive source progress jointly determine the resume taper. Repeated
    acquisition rows cannot accumulate a command offset while holding still.
    """
    phases = frozenset(('source', 'delayed_open', 'entry', 'settle', 'close', 'exit'))

    def __init__(self, model, actuators, reference, seconds, scale, resume_index, *, taper_s=.1):
        self.offsets = base_velocity_offsets(model, actuators, reference, seconds, scale, taper_s=taper_s)
        self.seconds = np.array(seconds, float, copy=True)
        if (not isinstance(resume_index, (int, np.integer)) or isinstance(resume_index, bool)
                or not 1 <= resume_index < len(self.seconds)):
            raise ValueError('A valid source row after the complete acquisition exit is required')
        self.resume_index = int(resume_index)
        self.scale, self.taper_s = float(scale), float(taper_s)
        self.last_time = self.last_index = self.resume_time = None
        self.state = dict(started=False, resume_index=self.resume_index,
                          actual_resume_time_s=None, source_resume_time_s=float(self.seconds[self.resume_index]),
                          scale=self.scale, taper_s=self.taper_s, weight=0.,
                          offset=np.zeros(self.offsets.shape[1]).tolist())

    def update(self, index, phase, actual_time_s):
        if (not isinstance(index, (int, np.integer)) or isinstance(index, bool)
                or not 0 <= index < len(self.seconds) or phase not in self.phases
                or not np.isscalar(actual_time_s) or not np.isfinite(actual_time_s)
                or actual_time_s < 0):
            raise ValueError('Valid source index, acquisition phase and actual integration time required')
        index, actual_time_s = int(index), float(actual_time_s)
        if ((self.last_time is not None and actual_time_s <= self.last_time)
                or (self.last_index is None and index != 0)
                or (self.last_index is not None and index-self.last_index not in (0, 1))):
            raise ValueError('Actual time must increase and every original source row must remain ordered')
        result = np.zeros(self.offsets.shape[1])
        weight = 0.
        if index >= self.resume_index:
            if phase != 'source':
                raise ValueError('Post-acquisition compensation requires the unchanged resumed source suffix')
            if self.resume_time is None:
                if index != self.resume_index:
                    raise ValueError('The first resumed source row cannot be skipped')
                self.resume_time = actual_time_s
            elif index != self.last_index+1:
                raise ValueError('Resumed feedforward cannot reuse a paused source row')
            actual_elapsed = actual_time_s-self.resume_time
            source_elapsed = float(self.seconds[index]-self.seconds[self.resume_index])
            if not np.isclose(actual_elapsed, source_elapsed, atol=1e-7, rtol=0):
                raise ValueError('Actual resume clock must follow the unchanged source interval')
            progress = float(np.clip(min(actual_elapsed, source_elapsed)/self.taper_s, 0., 1.))
            weight = float(np.clip(progress**3*(10.+progress*(-15.+6.*progress)), 0., 1.))
            result = self.offsets[index]*weight
        self.last_time, self.last_index = actual_time_s, index
        self.state = dict(started=self.resume_time is not None, resume_index=self.resume_index,
            actual_resume_time_s=self.resume_time, source_resume_time_s=float(self.seconds[self.resume_index]),
            source_index=index, source_time_s=float(self.seconds[index]), actual_time_s=actual_time_s,
            phase=phase, scale=self.scale, taper_s=self.taper_s, weight=weight, offset=result.tolist())
        return result
