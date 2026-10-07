"""Time scaling and resampling."""
import numpy as np

from reachy_retarget.retarget import timing
from reachy_retarget.retarget.config import RetargetConfig
from reachy_retarget.robot import VELOCITY
from reachy_retarget.schema import DT, ObjectTrack
from reachy_retarget.schema.rotations import so3_exp, so3_log

CFG = RetargetConfig()


def joint_path(T=40, rate=10.0):
    t = np.arange(T) / rate
    q = np.zeros((T, 22))
    q[:, 10] = np.where(t < 2, 0.0, 1.5)          # a 1.5 rad step in one 0.1 s interval
    q[:, 2] = np.linspace(3.0, 3.6, T)            # unwrapped yaw across +pi
    q[:, 0] = 0.05 * t                            # slow base drift
    return t, q


def test_clock_respects_limits_and_never_speeds_up():
    t, q = joint_path()
    c = timing.clock(t, q, CFG)
    assert np.allclose(np.diff(c.time), DT) and c.time[0] == 0
    assert np.all(np.diff(c.source_time) >= 0)
    assert c.source_time[0] == t[0] and c.source_time[-1] == t[-1]
    assert np.all(c.dilation >= 1) and c.dilation.max() > 10
    assert c.time[-1] >= t[-1] - t[0]
    out = timing.linear(c, q)
    speed = np.abs(np.diff(out, axis=0)) / DT
    assert np.all(speed <= VELOCITY * CFG.velocity_scale * (1 + 1e-9))
    qd = timing.velocity(out)
    assert np.all(np.abs(qd) <= VELOCITY * CFG.velocity_scale * (1 + 1e-9))
    # Undilated stretches keep the source speed: the base drifts at 0.05 m/s far from the step.
    early = c.source_time < 1.0
    assert np.allclose(np.diff(out[early, 0]) / DT, 0.05, atol=1e-9)


def test_slow_source_is_not_dilated():
    t = np.arange(20) / 20
    q = np.zeros((20, 22))
    q[:, 3] = 0.1 * t
    c = timing.clock(t, q, CFG)
    assert np.all(c.dilation == 1)
    assert np.allclose(c.source_time[:-1], c.time[:-1])
    assert c.source_time[-1] == t[-1] and c.time[-1] - t[-1] < DT  # final row holds the last frame


def test_object_resampling_respects_validity():
    t = np.arange(6) / 10.0
    pose = np.zeros((6, 7))
    pose[:, 0] = np.arange(6)
    pose[:, 3] = 1
    valid = np.ones(6, bool)
    valid[3] = False
    pose[3] = np.nan
    c = timing.clock(t, np.zeros((6, 22)), CFG)
    out = timing.object_track(c, ObjectTrack(pose, valid, "manipulated", {"kind": "box"}))
    s = c.source_time
    assert not out.valid[(s > 0.2 + 1e-9) & (s < 0.4 - 1e-9)].any()
    assert out.valid[s <= 0.2 + 1e-9].all() and out.valid[s >= 0.4 - 1e-9].all()
    assert np.isnan(out.pose[~out.valid]).all() and np.isfinite(out.pose[out.valid]).all()
    np.testing.assert_allclose(out.pose[out.valid, 0], (s * 10)[out.valid], atol=1e-9)
    assert out.role == "manipulated" and out.geometry == {"kind": "box"}


def test_pose_slerp_and_labels():
    t = np.array([0.0, 0.1])
    X = np.tile(np.eye(4), (2, 1, 1))
    X[1, :3, :3] = so3_exp(np.array([0, 0, 1.0]))
    X[1, :3, 3] = [1, 0, 0]
    c = timing.clock(t, np.zeros((2, 22)), CFG)
    P = timing.poses(c, X)
    mid = np.argmin(np.abs(c.fraction - 0.5))
    np.testing.assert_allclose(so3_log(P[mid, :3, :3]), [0, 0, c.fraction[mid]], atol=1e-12)
    np.testing.assert_allclose(P[:, 0, 3], c.fraction, atol=1e-12)
    lab = timing.labels(c, np.array([2, 2]))
    assert (lab == 2).all()
    lab = timing.labels(c, np.array([-1, 2]))
    assert lab[0] == -1 and lab[-1] == 2 and (lab[1:-1] == -1).all()
