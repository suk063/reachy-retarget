import json
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_retarget.grasp_closure import apply, plan


def fixture():
    times = np.arange(20)*.01
    hands = np.tile(np.eye(4), (len(times), 1, 1))
    objects = hands.copy()
    hands[:5, 0, 3] = .01
    objects[15:, 2, 3] = .01
    hands[15:, 2, 3] = .01
    commands = np.r_[np.full(2, 2.), np.full(15, -.06), np.full(3, 2.)]
    return times, hands, objects, commands


def test_dwell_and_response_delay_only_intent_preserve_all_inputs():
    times, hands, objects, commands = fixture()
    originals = [x.copy() for x in (times, hands, objects, commands)]
    changed, report, diagnostics = plan(times, hands, objects, commands, [0, 0, 0], [0, 0, 0],
        [0, 0, 1], response_time_s=.03, settle_s=.02, center_tolerance_m=.001)
    assert report["admitted"]
    assert report["derived_closure_frame"] == 7
    np.testing.assert_array_equal(changed[:7], 2.)
    np.testing.assert_array_equal(changed[7:], commands[7:])
    for actual, original in zip((times, hands, objects, commands), originals):
        np.testing.assert_array_equal(actual, original)
    assert diagnostics["central_region_mask"][5:].all()


def test_rejects_response_after_lift_or_release_without_closing_later():
    times, hands, objects, commands = fixture()
    changed, report, _ = plan(times, hands, objects, commands, [0, 0, 0], [0, 0, 0],
        [0, 0, 1], response_time_s=.12, settle_s=.02)
    assert not report["admitted"]
    np.testing.assert_array_equal(changed, commands)


def test_apply_saves_rejected_evidence_and_uses_scene_gravity(tmp_path):
    times, hands, objects, commands = fixture()
    details = {"pad_alignment": {"pad": {"midpoint_tcp": [0, 0, 0]},
                                  "section": {"center_object": [0, 0, 0]}}}
    prepared = (SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0, 0, -9.81]))),
                None, None, None, None, details, None, times, None, commands, hands, objects)
    with pytest.raises(ValueError, match="closure rejected"):
        apply(prepared, tmp_path/"rejected", response_time_s=.12, settle_s=.02)
    report = json.loads((tmp_path/"rejected/result.json").read_text())
    assert report["admitted"] is False
    assert report["world_up"] == [0., 0., 1.]
    with np.load(tmp_path/"rejected/closure-intent.npz") as data:
        np.testing.assert_array_equal(data["original_object_goals"], objects)
        np.testing.assert_array_equal(data["derived_gripper_intent"], commands)


def test_stability_checks_complete_response_window_and_requires_known_intent():
    times, hands, objects, commands = fixture()
    hands[8, 0, 3] = .01
    changed, report, _ = plan(times, hands, objects, commands, [0, 0, 0], [0, 0, 0],
        [0, 0, 1], response_time_s=.01, settle_s=.02)
    assert report["derived_closure_frame"] >= 11
    commands[10] = 2.
    with pytest.raises(ValueError, match="contiguous"):
        plan(times, hands, objects, commands, [0, 0, 0], [0, 0, 0], [0, 0, 1], response_time_s=.01)
