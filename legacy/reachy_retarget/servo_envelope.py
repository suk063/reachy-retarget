"""Joint-limit envelope for summed robot servo targets and integral memory.

This changes commands only. A target margin does not guarantee a measured
dynamic margin; the independent trajectory validator still checks every step.
"""
import numpy as np


def limits(model, actuators, margin):
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('A finite positive servo margin is required')
    ids = np.asarray(actuators)
    joint = model.actuator_trnid[ids, 0]
    if np.any(model.actuator_trntype[ids] != 0):
        raise ValueError('Servo envelope requires direct joint actuators')
    lower = np.full(len(ids), -np.inf)
    upper = np.full(len(ids), np.inf)
    active = model.jnt_limited[joint].astype(bool)
    lower[active] = model.jnt_range[joint[active], 0]+margin
    upper[active] = model.jnt_range[joint[active], 1]-margin
    controlled = model.actuator_ctrllimited[ids].astype(bool)
    lower[controlled] = np.maximum(lower[controlled], model.actuator_ctrlrange[ids[controlled], 0])
    upper[controlled] = np.minimum(upper[controlled], model.actuator_ctrlrange[ids[controlled], 1])
    if np.any(lower >= upper):
        raise ValueError('Servo margin leaves no admissible target interval')
    return lower, upper


def target(reference, integral, feedforward, bounds):
    return np.clip(reference+integral+feedforward, *bounds)


def integrate(integral, error, reference, feedforward, bounds, gain=.02, cap=.03):
    """Block windup toward saturation; allow immediate error-driven recovery."""
    unconstrained = reference+integral+feedforward
    lower, upper = bounds
    increment = gain*np.asarray(error)
    blocked = ((unconstrained >= upper) & (increment > 0)) | ((unconstrained <= lower) & (increment < 0))
    return np.clip(integral+np.where(blocked, 0., increment), -cap, cap)
