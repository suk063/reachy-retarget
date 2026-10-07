"""Slip auditing must distinguish rigid transport from motion inside the grasp."""

import numpy as np
from scipy.spatial.transform import Rotation

from reachy_retarget.contact import grasp_drift, ramp_closure
from reachy_retarget.episodes import matrices_to_pose


def test_rigid_transport_has_zero_drift_and_release_is_excluded():
    hand = np.tile(np.eye(4), (5, 2, 1, 1))
    hand[:, 1, :3, :3] = Rotation.from_euler("z", np.array([0, 10, 20, 30, 40])[:, None], degrees=True).as_matrix()
    hand[:, 1, 2, 3] = [0, .04, .08, .1, .1]
    attachment = np.eye(4)
    attachment[:3, 3] = [.03, .01, .02]
    obj = hand[:, 1] @ attachment
    obj[-1, 0, 3] += .3
    report, series = grasp_drift(matrices_to_pose(hand), matrices_to_pose(obj),
                                 [-1, -1, -1, -1, 2], [False, True, True, True, False])
    assert report["available"]
    assert report["max_translation_m"] < 1e-12
    assert report["max_rotation_deg"] < 1e-12
    assert np.isnan(series[-1]).all()
    obj[3] = hand[3, 1] @ attachment
    obj[3, :3, 3] += hand[3, 1, :3, :3] @ [.006, 0, 0]
    obj[3, :3, :3] = hand[3, 1, :3, :3] @ Rotation.from_euler("x", 8, degrees=True).as_matrix()
    report, _ = grasp_drift(matrices_to_pose(hand), matrices_to_pose(obj),
                            [-1, -1, -1, -1, 2], [False, True, True, True, False])
    np.testing.assert_allclose(report["max_translation_m"], .006, atol=1e-12)
    np.testing.assert_allclose(report["max_rotation_deg"], 8, atol=1e-12)


def test_no_lifted_bilateral_grasp_is_not_reported_as_zero_slip():
    hand = np.tile([0, 0, 0, 1, 0, 0, 0], (2, 2, 1))
    report, series = grasp_drift(hand, hand[:, 0], [-1, -1], [False, False])
    assert not report["available"]
    assert np.isnan(series).all()


def test_closing_ramp_never_precedes_source_event_or_delays_opening():
    times = np.arange(101) / 100
    commands = np.where((times >= .2) & (times < .8), -.06, 2.)
    original = commands.copy()
    values = ramp_closure(times, commands, .3)
    np.testing.assert_array_equal(commands, original)
    np.testing.assert_array_equal(values[times < .2], 2.)
    assert values[20] == 2.
    assert np.all(np.diff(values[20:51]) <= 0)
    np.testing.assert_allclose(values[(times >= .5) & (times < .8)], -.06)
    np.testing.assert_array_equal(values[times >= .8], 2.)
    np.testing.assert_array_equal(ramp_closure(times, commands, 0), original)
