import numpy as np
import pytest

from reachy_retarget.store import sha256
from reachy_retarget.terminal_hold import apply


def example(tmp_path):
    time = np.arange(5)*.01
    joint = np.arange(85, dtype=float).reshape(5, 17)
    path = tmp_path/'original-map.npz'
    np.savez_compressed(path, control_time_s=time, source_reference_time_s=time*.5,
                        original_time_s=time*.5, retimed_reference=joint)
    details = dict(robot_control_retiming=dict(source_clock_artifact=str(path),
        artifact_sha256=sha256(path), original_duration_s=.02, duration_s=.04))
    pose = np.repeat(np.eye(4)[None], 5, axis=0)
    return (None, None, None, None, None, details, None, time, joint,
            np.array([2., 2., -.06, -.06, -.06]), pose, pose.copy())


def test_terminal_hold_keeps_every_prefix_and_original_source_clock(tmp_path):
    original = example(tmp_path)
    held = apply(original, tmp_path/'hold', duration_s=.2)
    for index in (7, 8, 9, 10, 11):
        np.testing.assert_array_equal(held[index][:5], original[index])
    for index in (8, 9, 10, 11):
        np.testing.assert_array_equal(held[index][5:], np.repeat(original[index][-1:], 20, axis=0))
    with np.load(held[5]['robot_control_retiming']['source_clock_artifact']) as mapping:
        np.testing.assert_array_equal(mapping['source_reference_time_s'][:5], original[7]*.5)
        np.testing.assert_array_equal(mapping['source_reference_time_s'][5:], np.repeat(.02, 20))
    assert 'robot_terminal_hold' not in original[5]
    assert held[5]['robot_terminal_hold']['added_control_intervals'] == 20


def test_hold_rejects_missing_or_changed_clock_and_double_extension(tmp_path):
    original = example(tmp_path)
    held = apply(original, tmp_path/'hold', duration_s=.1)
    with pytest.raises(ValueError, match='second terminal'):
        apply(held, tmp_path/'twice', duration_s=.1)
    original[5]['robot_control_retiming']['artifact_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='checksum'):
        apply(original, tmp_path/'changed', duration_s=.1)
    with pytest.raises(ValueError, match='integer number'):
        apply(original, tmp_path/'partial', duration_s=.005)


def test_hold_cannot_silently_extend_a_runtime_placement_controller(tmp_path):
    original = example(tmp_path)
    original[5]['robot_supported_placement'] = {'phase_frames': {'open': [3, 5]}}
    with pytest.raises(ValueError, match='separately verified terminal phase'):
        apply(original, tmp_path/'placement', duration_s=.2)
