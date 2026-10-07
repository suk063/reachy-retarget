"""Explicit source-row clock for bounded, measured acquisition waits.

This iterator controls command progression only. It never accesses simulation
state; callers must integrate and record every yielded frame independently.
"""
from dataclasses import dataclass
from copy import deepcopy

from .acquisition_rendezvous import Rendezvous


@dataclass(frozen=True)
class Frame:
    reference_index: int
    phase: str
    correction_index: int = -1


class SourceIndexedController:
    """Advance a source-indexed controller once per original row, never per wait."""
    def __init__(self, controller):
        self.controller, self.index, self.cached = controller, -1, None

    def update(self, index, *args, **kwargs):
        if index == self.index and self.cached is not None:
            return deepcopy(self.cached)
        if index != self.index+1:
            raise ValueError('Source-indexed control must retain consecutive original rows')
        self.cached = deepcopy(self.controller.update(index, *args, **kwargs))
        self.index = index
        return deepcopy(self.cached)


class AcquisitionClock:
    def __init__(self, *, source_rows, first_close_index, acquisition_index,
                 entry_rows, exit_source_indices, timestep=.01,
                 wait_limit_s=2., stable_s=.1):
        values = (source_rows, first_close_index, acquisition_index, entry_rows)
        if (any(type(v) is not int for v in values)
                or not 0 <= first_close_index <= acquisition_index < source_rows
                or entry_rows < 1):
            raise ValueError('Complete source rows and valid integer acquisition bounds required')
        exits = list(exit_source_indices)
        if (not exits or any(type(i) is not int for i in exits)
                or exits != list(range(acquisition_index+1, acquisition_index+1+len(exits)))
                or exits[-1] >= source_rows):
            raise ValueError('Exit blend must consume consecutive original rows after acquisition')
        self.rows, self.first, self.anchor, self.entry_rows = values
        self.exits = exits
        self.wait = Rendezvous(timestep=timestep, wait_limit_s=wait_limit_s, stable_s=stable_s)
        self.started = self.complete = False
        self.current = None
        self.observed = False

    def _yield(self, frame):
        self.current, self.observed = frame, False
        yield frame

    def frames(self):
        if self.started:
            raise ValueError('Acquisition clock can be consumed only once')
        self.started = True
        for index in range(self.rows):
            if index == self.anchor:
                for entry in range(self.entry_rows):
                    yield from self._yield(Frame(index, 'entry', entry))
                self.wait.begin()
                while self.wait.phase in ('settle', 'close'):
                    yield from self._yield(Frame(index, self.wait.phase))
                    if not self.observed:
                        raise ValueError('Every paused frame requires exactly one measured guard update')
                if self.wait.phase == 'failed':
                    return
            elif index in self.exits:
                yield from self._yield(Frame(index, 'exit', index-self.anchor-1))
            else:
                phase = 'delayed_open' if self.first <= index < self.anchor else 'source'
                yield from self._yield(Frame(index, phase))
        self.complete = True

    def observe(self, *, alignment, bilateral_force):
        if (self.current is None or self.current.phase not in ('settle', 'close')
                or self.observed):
            raise ValueError('A single guard update is allowed only after a paused frame')
        self.observed = True
        return self.wait.update(alignment=alignment, bilateral_force=bilateral_force)

    def report(self):
        return dict(self.wait.report(), complete_source_progression=self.complete,
                    original_reference_rows=self.rows, first_close_index=self.first,
                    acquisition_index=self.anchor, entry_rows=self.entry_rows,
                    exit_source_indices=self.exits,
                    clock_scope='Source rows are retained; entry and measured waits repeat the acquisition index. Physical time must advance at every frame.')


def complete_reference_coverage(indices, rows):
    """Check actual recorded indices, not the number of proposed clock frames."""
    return bool(indices and rows > 0 and indices[0] == 0 and indices[-1] == rows-1
                and all(type(i) is int for i in indices)
                and all(b-a in (0, 1) for a, b in zip(indices, indices[1:])))


def command(frame, reference, hand_goal, intent, feedforward, plan):
    """Select admitted robot commands without changing any original array."""
    import numpy as np
    arrays = plan['arrays']
    phase = frame.phase
    if phase == 'entry':
        reference = arrays['entry_reference'][frame.correction_index]
        hand_goal = arrays['entry_hand_goals'][frame.correction_index]
    elif phase in ('settle', 'close'):
        reference = arrays['entry_reference'][-1]
        hand_goal = arrays['entry_hand_goals'][-1]
    elif phase == 'exit':
        reference = arrays['exit_reference'][frame.correction_index]
        hand_goal = arrays['exit_hand_goals'][frame.correction_index]
    elif phase not in ('source', 'delayed_open'):
        raise ValueError('Unknown acquisition phase')
    if phase in ('delayed_open', 'entry', 'settle'):
        intent = 2.
    if phase in ('entry', 'settle', 'close', 'exit'):
        feedforward = np.zeros_like(feedforward)
    return reference.copy(), hand_goal.copy(), float(intent), feedforward.copy()
