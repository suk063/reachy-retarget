"""Bounded command-clock rendezvous; this module never reads or writes physics.

Geometry adapters provide measured alignment and contact guards. A source clock
may pause, but no source row is discarded and the physical clock never pauses.
"""
import math


class Rendezvous:
    CODES = {'approach': 0, 'settle': 1, 'close': 2, 'carry': 3, 'failed': 4}

    def __init__(self, *, timestep, wait_limit_s=2., stable_s=.1):
        if not all(math.isfinite(x) for x in (timestep, wait_limit_s, stable_s)) or not 0 < timestep <= stable_s <= wait_limit_s <= 2.:
            raise ValueError('Finite positive waits bounded by two seconds are required')
        self.timestep, self.limit, self.stable = timestep, wait_limit_s, stable_s
        self.phase, self.elapsed, self.good = 'approach', 0., 0.
        self.events, self.hold_steps, self.failure = [], 0, None

    def begin(self):
        if self.phase != 'approach':
            raise ValueError('Rendezvous may begin only once')
        self.phase = 'settle'
        self.events.append({'event': 'settle_started', 'hold_step': self.hold_steps})

    def update(self, *, alignment, bilateral_force):
        if self.phase not in ('settle', 'close'):
            raise ValueError('Measured guards apply only to an active wait')
        self.elapsed += self.timestep
        self.hold_steps += 1
        okay = bool(alignment and (self.phase == 'settle' or bilateral_force))
        self.good = self.good+self.timestep if okay else 0.
        if self.good+1e-12 >= self.stable:
            old = self.phase
            self.phase = 'close' if old == 'settle' else 'carry'
            self.events.append({'event': old+'_guard_passed', 'hold_step': self.hold_steps,
                                'elapsed_phase_s': self.elapsed})
            self.elapsed = self.good = 0.
        elif self.elapsed+1e-12 >= self.limit:
            self.failure = self.phase+'_timeout'
            self.phase = 'failed'
            self.events.append({'event': self.failure, 'hold_step': self.hold_steps})
        return self.phase

    def report(self):
        return {'phase': self.phase, 'failure': self.failure, 'inserted_steps': self.hold_steps,
                'inserted_duration_s': self.hold_steps*self.timestep,
                'wait_limit_per_phase_s': self.limit, 'stable_guard_s': self.stable,
                'events': self.events, 'phase_codes': self.CODES,
                'source_clock_semantics': 'Pause command progression during settle/close; never alter the integration clock or source samples'}
