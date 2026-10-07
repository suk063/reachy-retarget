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
    subdivisions: int = 1


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
                 wait_limit_s=2., stable_s=.1, dilation=None):
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
        self.dilation = _dilation_rows(dilation, exits[-1], source_rows)
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
            elif self.dilation.get(index, 1) > 1:
                count = self.dilation[index]
                for part in range(count):
                    yield from self._yield(Frame(index, 'dilated', part, count))
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
                    post_acquisition_dilated_rows=len(self.dilation),
                    post_acquisition_added_frames=int(sum(self.dilation.values())-len(self.dilation)),
                    clock_scope='Source rows are retained; entry and measured waits repeat the acquisition index. Physical time must advance at every frame.')


def _dilation_rows(dilation, exit_end, source_rows):
    """Validate an optional post-exit source-row subdivision schedule."""
    if dilation is None:
        return {}
    start, counts = dilation.get('start_index'), dilation.get('frames_per_row')
    if (type(start) is not int or not isinstance(counts, list) or not counts
            or any(type(c) is not int or not 1 <= c <= 20 for c in counts)
            or start <= exit_end or start+len(counts) > source_rows-1):
        raise ValueError('Dilation must subdivide whole original rows strictly after the exit blend and before the last row')
    return {start+i: c for i, c in enumerate(counts) if c > 1}


def dilation_kwargs(metadata):
    """Clock arguments bound to acquisition metadata; empty for the default clock."""
    value = metadata.get('post_acquisition_dilation')
    return {} if value is None else {'dilation': dict(start_index=value['start_index'],
                                                       frames_per_row=list(value['frames_per_row']))}


def tcp_speed_dilation(plan, *, linear_speed_m_s, angular_speed_rad_s, start_index=None,
                       stop_index=None, max_subdivisions=10, smoothing_s=.1):
    """Derive whole-row subdivisions capping the admitted post-exit hand reference speed.

    Only command progression is slowed: every original row is still consumed
    once in order and intermediate commands interpolate consecutive original
    rows. No source sample, object array or physical model is changed.
    """
    import numpy as np
    from scipy.ndimage import maximum_filter1d, gaussian_filter1d
    from scipy.spatial.transform import Rotation
    arrays, metadata = plan['arrays'], plan['metadata']
    times, hands = np.asarray(arrays['original_time_s']), np.asarray(arrays['original_hand_goals'])
    exit_end = int(arrays['exit_source_indices'][-1])
    start = exit_end+1 if start_index is None else int(start_index)
    stop = len(times)-1 if stop_index is None else int(stop_index)
    values = [linear_speed_m_s, angular_speed_rad_s, smoothing_s]
    if (not np.isfinite(values).all() or min(values) <= 0 or type(max_subdivisions) is not int
            or not 1 <= max_subdivisions <= 20 or not exit_end < start < stop <= len(times)-1):
        raise ValueError('Positive speed caps and a post-exit row window are required')
    dt = np.diff(times)[start:stop]
    linear = np.linalg.norm(np.diff(hands[start:stop+1, :3, 3], axis=0), axis=1)/dt
    angular = np.array([Rotation.from_matrix(hands[i+1, :3, :3]@hands[i, :3, :3].T).magnitude()
                        for i in range(start, stop)])/dt
    required = np.maximum(1., np.maximum(linear/linear_speed_m_s, angular/angular_speed_rad_s))
    window = max(1, int(np.ceil(smoothing_s/np.median(dt))))
    smooth = gaussian_filter1d(maximum_filter1d(required, size=2*window+1, mode='nearest'),
                               sigma=max(1, window/2), mode='nearest')
    counts = np.clip(np.ceil(np.maximum(required, smooth)-1e-9), 1, max_subdivisions).astype(int)
    result = dict(start_index=start, frames_per_row=[int(c) for c in counts],
                  linear_speed_cap_m_s=float(linear_speed_m_s), angular_speed_cap_rad_s=float(angular_speed_rad_s),
                  max_subdivisions=max_subdivisions, smoothing_s=float(smoothing_s),
                  added_frames=int(counts.sum()-len(counts)),
                  peak_original_linear_m_s=float(linear.max()), peak_original_angular_rad_s=float(angular.max()),
                  peak_dilated_linear_m_s=float(np.max(linear/counts)), peak_dilated_angular_rad_s=float(np.max(angular/counts)),
                  saturated_rows=int(np.sum(required > max_subdivisions)),
                  method='Post-exit whole-row subdivision; linear joint and SLERP hand interpolation between consecutive original rows; velocity feedforward divided by subdivisions',
                  original_arrays_unchanged=True, physics_validated=False)
    metadata['post_acquisition_dilation'] = result
    return result


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
    elif phase == 'dilated':
        from scipy.spatial.transform import Rotation, Slerp
        index, fraction = frame.reference_index, frame.correction_index/frame.subdivisions
        rows, hands = arrays['original_reference'], arrays['original_hand_goals']
        if not np.array_equal(reference, rows[index]):
            raise ValueError('Dilated command must start from the bound original row')
        reference = (1-fraction)*rows[index]+fraction*rows[index+1]
        hand_goal = np.array(hands[index], dtype=float)
        hand_goal[:3, 3] = (1-fraction)*hands[index][:3, 3]+fraction*hands[index+1][:3, 3]
        hand_goal[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(np.stack([hands[index][:3, :3], hands[index+1][:3, :3]])))(fraction).as_matrix()
        feedforward = np.asarray(feedforward)/frame.subdivisions
    elif phase not in ('source', 'delayed_open'):
        raise ValueError('Unknown acquisition phase')
    if phase in ('delayed_open', 'entry', 'settle'):
        intent = 2.
    if phase in ('entry', 'settle', 'close', 'exit'):
        feedforward = np.zeros_like(feedforward)
    return reference.copy(), hand_goal.copy(), float(intent), feedforward.copy()
