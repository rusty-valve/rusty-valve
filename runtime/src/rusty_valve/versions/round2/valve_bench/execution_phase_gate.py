"""Measured phase transitions; trajectory time alone never advances motion.

Durations shape the existing paths and bound a stuck stage.  Qualification is
owned by the unchanged scorer, not by this execution gate.
"""
from __future__ import annotations

import copy


VERSION = 'measured_G1_phases_v1'


class MeasuredPhaseGate:
    def __init__(self, parameters):
        self.p = parameters
        self.phase = 'initial_down_hold'
        self.started = 0.
        self.reached_since = None
        self.done = False
        self.blocked_reason = None
        self.events = []

    def elapsed(self, time_s):
        return max(0., float(time_s)-self.started)

    def _switch(self, phase, t, reason, measurements):
        old = self.phase
        self.phase, self.started = phase, float(t)
        # Preparation and final-rest dwell begin with the transition's actual
        # reached measurement, not one control sample after reaching it.
        self.reached_since = (float(t) if phase in ('chest_ready','final_down_hold')
                              and measurements.get('reached') else None)
        self.events.append(dict(time_s=float(t), from_phase=old, to_phase=phase,
            reason=reason, measurements=copy.deepcopy(measurements)))

    def advance(self, time_s, *, reached=False, trajectory_complete=False,
                seated=False, in_goal=False, goal_hold_completed=False,
                detached=False, measurements=None):
        if self.done or self.blocked_reason:
            return False
        t, old = float(time_s), self.phase
        evidence = dict(measurements or {})
        evidence.update(reached=bool(reached), trajectory_complete=bool(trajectory_complete),
            seated=bool(seated), in_goal=bool(in_goal),
            goal_hold_completed=bool(goal_hold_completed), detached=bool(detached))
        self.reached_since = (t if self.reached_since is None else self.reached_since) if reached else None
        dwell = 0. if self.reached_since is None else t-self.reached_since
        switch = lambda phase, why: self._switch(phase, t, why, evidence)
        if old == 'initial_down_hold' and reached and dwell >= self.p['down_hold_s']-1e-9:
            switch('lift_to_chest', 'actual_shared_rest_held')
        elif old == 'lift_to_chest' and trajectory_complete and reached:
            switch('chest_ready', 'actual_preparation_pose_reached')
        elif old == 'chest_ready' and reached and dwell >= self.p['chest_hold_s']-1e-9:
            switch('reach', 'actual_preparation_pose_held')
        elif old == 'reach' and trajectory_complete and reached:
            switch('approach', 'actual_selected_precontact_pose_reached')
        elif old == 'approach' and trajectory_complete and seated:
            switch('turn', 'actual_recess_and_correct_working_face_seated')
        elif old in ('turn', 'hold_final'):
            if in_goal and goal_hold_completed:
                switch('release', 'actual_current_goal_hold_completed')
            elif not seated:
                switch('approach', 'contact_or_recess_lost_unload_and_reseat')
            elif old == 'turn' and in_goal:
                switch('hold_final', 'actual_valve_entered_existing_goal_band')
            elif old == 'hold_final' and not in_goal:
                switch('turn', 'actual_valve_left_goal_band')
        elif old == 'release' and trajectory_complete and reached and detached:
            switch('return_chest', 'actual_exit_pose_reached_and_no_handle_contact')
        elif old == 'return_chest' and trajectory_complete and reached:
            switch('lower_to_natural', 'actual_return_preparation_pose_reached')
        elif old == 'lower_to_natural' and trajectory_complete and reached:
            switch('final_down_hold', 'actual_shared_rest_reached')
        elif old == 'final_down_hold' and reached and dwell >= self.p['down_hold_s']-1e-9:
            self.done = True
            self.events.append(dict(time_s=t, from_phase=old, to_phase='completed',
                reason='actual_shared_rest_held', measurements=evidence))
        if self.phase != old or self.done:
            return True
        # A stage guard ends computation; it never forces the next phase or
        # adds a timeout rule to the task scorer. Turning uses the unchanged
        # overall 90 s observation limit instead of a new speed requirement.
        names = dict(initial_down_hold='down_hold_s', lift_to_chest='lift_s',
            chest_ready='chest_hold_s', reach='reach_s', approach='approach_s',
            release='withdraw_s', return_chest='return_s',
            lower_to_natural='lower_s', final_down_hold='down_hold_s')
        if old in names:
            limit = max(8., 3.*float(self.p[names[old]]))
            if self.elapsed(t) > limit:
                self.blocked_reason = dict(phase=old, reason='actual_stage_not_completed',
                    stage_guard_s=limit, time_s=t, measurements=evidence,
                    score_changed=False)
                self.events.append(dict(self.blocked_reason, to_phase='blocked'))
        return False
