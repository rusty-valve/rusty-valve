"""Keep the approved approach curve clock separate from measured state gates.

Normal sampling uses the original nominal curve starts.  A gate can freeze at
the current curve's end when the physical mechanism has not completed it, then
resume with only that actual waiting time added to later curve starts.  This
module observes no geometry, advances no simulator, and selects no motion.
"""
from __future__ import annotations

import math


class PreservedApproachClock:
    def __init__(self, parameters):
        names = ('down_hold_s', 'lift_s', 'chest_hold_s', 'reach_s', 'approach_s')
        durations = {}
        for name in names:
            value = float(parameters[name])
            if not math.isfinite(value) or value < 0.:
                raise ValueError(f'{name} must be a finite nonnegative duration')
            durations[name] = value
        down = durations['down_hold_s']
        chest = down+durations['lift_s']
        reach = down+durations['lift_s']+durations['chest_hold_s']
        approach = reach+durations['reach_s']
        turn = approach+durations['approach_s']
        self.nominal_starts = dict(initial_down_hold=0., lift_to_chest=down,
            chest_ready=chest, reach=reach, approach=approach, turn=turn)
        self._nominal_ends = dict(initial_down_hold=down, lift_to_chest=chest,
            chest_ready=reach, reach=approach, approach=turn)
        self.virtual_time = 0.
        self.delay_s = 0.
        self.waiting = False
        self._last_actual_time = None

    def start_of(self, phase):
        try:
            return self.nominal_starts[phase]
        except KeyError as exc:
            raise ValueError(f'Unknown preserved approach phase: {phase}') from exc

    def end_of(self, phase):
        try:
            return self._nominal_ends[phase]
        except KeyError as exc:
            raise ValueError(f'No preserved approach endpoint for phase: {phase}') from exc

    def sample(self, actual_t):
        t = float(actual_t)
        if not math.isfinite(t) or t < 0.:
            raise ValueError('Actual time must be finite and nonnegative')
        if self._last_actual_time is not None and t < self._last_actual_time:
            raise ValueError('Actual time cannot move backwards')
        if self.waiting:
            # Keep the frozen curve endpoint exact while actual time moves.
            self.delay_s = t-self.virtual_time
        else:
            self.virtual_time = max(0., t-self.delay_s)
        self._last_actual_time = t
        return self.virtual_time

    def elapsed(self, phase):
        return max(0., self.virtual_time-self.start_of(phase))

    def hold_at_phase_end(self, phase):
        end = self.end_of(phase)
        if self._last_actual_time is None or self.virtual_time < end:
            raise ValueError('Sample at or beyond the curve endpoint before holding')
        # The original boundary may fall between control samples. Account for
        # that overshoot only when the state gate actually requires waiting.
        self.virtual_time = end
        self.delay_s = self._last_actual_time-end
        self.waiting = True

    def resume(self):
        self.waiting = False
