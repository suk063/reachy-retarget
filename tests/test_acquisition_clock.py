import pytest

from reachy_retarget.acquisition_clock import AcquisitionClock, complete_reference_coverage


def clock(**kwargs):
    return AcquisitionClock(source_rows=9, first_close_index=2, acquisition_index=4,
                            entry_rows=3, exit_source_indices=[5, 6], **kwargs)


def test_measured_wait_preserves_every_row_and_consumes_exit_rows_once():
    execution = clock()
    frames = []
    for frame in execution.frames():
        frames.append(frame)
        if frame.phase in ('settle', 'close'):
            execution.observe(alignment=True, bilateral_force=frame.phase == 'close')
    indices = [f.reference_index for f in frames]
    assert complete_reference_coverage(indices, 9)
    assert indices == [0, 1, 2, 3]+[4]*23+[5, 6, 7, 8]
    assert [f.phase for f in frames[:4]] == ['source', 'source', 'delayed_open', 'delayed_open']
    assert [(f.reference_index, f.correction_index) for f in frames if f.phase == 'exit'] == [(5, 0), (6, 1)]
    assert execution.complete
    assert execution.report()['inserted_duration_s'] == pytest.approx(.2)


def test_failed_close_keeps_executed_prefix_and_cannot_claim_full_coverage():
    execution = clock(wait_limit_s=.2)
    indices = []
    for frame in execution.frames():
        indices.append(frame.reference_index)
        if frame.phase in ('settle', 'close'):
            execution.observe(alignment=True, bilateral_force=False)
    assert execution.wait.failure == 'close_timeout'
    assert not execution.complete
    assert not complete_reference_coverage(indices, 9)
    assert max(indices) == 4


def test_missing_or_duplicate_guards_and_reusing_clock_fail_closed():
    execution = clock()
    iterator = execution.frames()
    frame = next(iterator)
    while frame.phase != 'settle':
        frame = next(iterator)
    with pytest.raises(ValueError, match='exactly one'):
        next(iterator)
    with pytest.raises(ValueError, match='only once'):
        next(execution.frames())
    second = clock()
    for frame in second.frames():
        if frame.phase == 'settle':
            second.observe(alignment=False, bilateral_force=False)
            with pytest.raises(ValueError, match='single guard'):
                second.observe(alignment=False, bilateral_force=False)
            break


def test_complete_coverage_rejects_crops_skips_reordering_and_noninteger_indices():
    for invalid in ([], [1, 2], [0, 2], [0, 1, 0, 2], [0, 1, 2.], [0, 1]):
        assert not complete_reference_coverage(invalid, 3)
    assert complete_reference_coverage([0, 0, 1, 1, 2], 3)
    with pytest.raises(ValueError, match='consecutive'):
        AcquisitionClock(source_rows=9, first_close_index=2, acquisition_index=4,
                         entry_rows=3, exit_source_indices=[6, 7])


def test_command_policy_changes_only_admitted_robot_rows_and_open_intent():
    import numpy as np
    from reachy_retarget.acquisition_clock import Frame, command
    reference = np.arange(17, dtype=float)
    hand = np.eye(4)
    feedforward = np.ones(17)*.1
    plan = {'arrays': {'entry_reference': np.array([reference, reference+.01]),
                      'entry_hand_goals': np.array([hand, hand]),
                      'exit_reference': np.array([reference+.02]),
                      'exit_hand_goals': np.array([hand])}}
    for phase, correction in [('source', -1), ('delayed_open', -1), ('entry', 1),
                              ('settle', -1), ('close', -1), ('exit', 0)]:
        q, goal, intent, ff = command(Frame(4, phase, correction), reference, hand, -.06, feedforward, plan)
        assert intent == (2. if phase in ('delayed_open', 'entry', 'settle') else -.06)
        assert np.array_equal(ff, feedforward if phase in ('source', 'delayed_open') else np.zeros(17))
        expected = reference + (.02 if phase == 'exit' else .01 if phase in ('entry', 'settle', 'close') else 0.)
        assert np.array_equal(q, expected)
        q[:] = 99.; goal[:] = 99.; ff[:] = 99.
    assert np.array_equal(reference, np.arange(17, dtype=float))
    assert np.array_equal(hand, np.eye(4))
    assert np.array_equal(feedforward, np.ones(17)*.1)
    assert plan['arrays']['entry_reference'][1, 0] == .01


def test_source_indexed_controller_pauses_updates_but_consumes_every_original_row():
    import numpy as np
    from reachy_retarget.acquisition_clock import SourceIndexedController
    class Controller:
        def __init__(self): self.calls = []
        def update(self, index, support):
            assert index == len(self.calls)
            self.calls.append((index, support))
            return np.array([float(index)]), np.eye(4), -.06
    controller = Controller()
    updates = SourceIndexedController(controller)
    execution = clock()
    actual_frames = 0
    for frame in execution.frames():
        reference, _, _ = updates.update(frame.reference_index, actual_frames)
        assert reference[0] == frame.reference_index
        reference[0] = 99.  # Caller cannot corrupt a repeated row's cached target.
        actual_frames += 1
        if frame.phase in ('settle', 'close'):
            execution.observe(alignment=True, bilateral_force=True)
    assert len(controller.calls) == 9 and actual_frames > 9
    assert controller.calls[5][1] > 5
    with pytest.raises(ValueError, match='consecutive'):
        updates.update(10, 0.)
