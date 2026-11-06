"""S5 contact choices around the passed, state-driven G1 executor.

Keep a usable preferred radius unchanged. Planning can reject an inadmissible
candidate before motion. If actual seating fails, permit one measured retreat
and another allowed radius, never a radial sweep while still touching.
No workstation name, new posture, motor limit or task score is introduced.
"""
from __future__ import annotations

import copy

from .contact_radius_choice import ContactRadiusChoice
from .execution_state_observation import measure_handle_clearance
from .path_obstacles import PathObstacleConflict
from .state_g1_controller import StateG1Controller
from .tool_upright import ToolUprightConflict


class RangeG1Controller(StateG1Controller):
    def __init__(self, env, variant, method, document, parameters):
        self.radius_choice = ContactRadiusChoice(env.station_spec.handle_length_m,
                                                 parameters['r_m'])
        self.radius_events = []
        self.physical_radius_retries = 0
        self._radius_retreat_pending = False
        self._retry_pre_geometry = None
        super().__init__(env,variant,method,document,parameters)
        self.phase_version += '_contact_range_v1'
        self.phase_settings.update(contact_target_selection_changed=True,
            contact_radius_policy='preferred_then_finite_allowed_radius_candidates_v1',
            preferred_radius_m=self.radius_choice.preferred_radius_m,
            maximum_physical_radius_retries=1,
            radius_change_requires_actual_detachment=True)

    def _prepare(self, poses):
        while self.radius_choice.current_radius_m is not None:
            self.p['r_m'] = self.radius_choice.current_radius_m
            try:
                super()._prepare(poses)
            except (PathObstacleConflict,ToolUprightConflict) as exc:
                self.radius_events.append(dict(kind='planning_candidate_rejected',
                    radius_m=self.p['r_m'],reason=type(exc).__name__,
                    observation=copy.deepcopy(exc.observation),physics_started=False))
                if self.radius_choice.advance('preferred_or_alternate_planning_rejected') is None:
                    exc.observation['contact_radius_selection'] = self.radius_summary()
                    raise
            else:
                self.radius_events.append(dict(kind='initial_radius_selected',
                    radius_m=self.p['r_m'],physical_seating_not_yet_proved=True))
                return
        raise ValueError('No allowed contact radius candidate remains')

    def _uses_original_clock(self, phase):
        # Actual retreat/re-entry has its own local clock. The successful
        # preferred initial trajectory keeps the original S4 sampling exactly.
        if self.physical_radius_retries:
            return False
        return super()._uses_original_clock(phase)

    def _handle_blocked_phase(self, t):
        blocked = self.phase_gate.blocked_reason
        candidates = self.radius_choice.candidates
        selected = self.radius_choice.current_radius_m
        remaining = (selected is not None and
                     candidates.index(selected)+1 < len(candidates))
        if (not blocked or blocked.get('phase') != 'approach'
                or self.physical_radius_retries >= 1 or not remaining):
            return
        self.radius_events.append(dict(kind='actual_seating_failed_request_retreat',
            time_s=float(t),radius_m=self.p['r_m'],blocked=copy.deepcopy(blocked)))
        self.physical_radius_retries += 1
        self._radius_retreat_pending = True
        self.integral.fill(0.)
        self.reference = self.opening
        # Keep the OLD radius throughout withdrawal. Another point is not
        # selected or commanded until the measured release gate has passed.
        self.phase_gate.blocked_reason = None
        self.phase_gate.phase = 'release'
        self.phase_gate.started = float(t)
        self.phase_gate.reached_since = None
        self.phase_gate.events.append(dict(time_s=float(t),from_phase='approach',
            to_phase='release',reason='one_allowed_contact_radius_retry_requires_detachment'))

    def _prepare_retry_radius(self, t):
        e,p = self.env,self.p
        separation = measure_handle_clearance(e)
        if separation['any_handle_contact'] or separation['minimum_distance_m'] <= 0.:
            raise RuntimeError('Cannot change radius while the real tool touches the handle')
        old_radius = p['r_m']
        radius = self.radius_choice.advance('actual_seating_failed_after_measured_detachment')
        last_error = None
        while radius is not None:
            p['r_m'] = radius
            try:
                actual = e.data.qpos[e.active_qpos_indices].copy()
                pos,rot = self.target(self.opening,clearance=p['approach_clearance_m'])
                pre,pe = self._prepare_target(pos,rot,
                    [self._approved_contact_seed(self.opening),actual],
                    'contact_radius_retry_precontact')
                points = self._choose_free_path(actual,pre,pos,rot,pre,
                    'reach_path_ranking',pe)
                self._require_upright_path(points,'reach')
            except (PathObstacleConflict,ToolUprightConflict) as exc:
                self.radius_events.append(dict(kind='detached_retry_candidate_rejected',
                    time_s=float(t),radius_m=radius,reason=type(exc).__name__,
                    observation=copy.deepcopy(exc.observation)))
                last_error = exc
                radius = self.radius_choice.advance('same_G1_route_or_direction_rejected')
                p['r_m'] = old_radius
            else:
                self.reach_points = points
                self._retry_pre_geometry = self._static_geometry(pre)
                self.qseed = actual.copy()
                self.radius_events.append(dict(kind='detached_retry_radius_selected',
                    time_s=float(t),from_radius_m=old_radius,to_radius_m=radius,
                    separation=separation,route_side_preserved=True))
                return
        p['r_m'] = old_radius
        if last_error is not None:
            last_error.observation['contact_radius_selection'] = self.radius_summary()
            raise last_error
        raise RuntimeError('No further allowed contact radius is available')

    def _enter(self, old, phase, t):
        if old == 'release' and phase == 'return_chest' and self._radius_retreat_pending:
            self._prepare_retry_radius(t)
            self._radius_retreat_pending = False
            self.reseating = True
            self.phase_gate.phase = 'reach'
            self.phase_gate.started = float(t)
            self.phase_gate.reached_since = None
            self.phase_gate.events.append(dict(time_s=float(t),from_phase='release',
                to_phase='reach',reason='actual_detachment_allows_new_radius_on_same_G1_side'))
            return
        if old == 'reach' and phase == 'approach' and self._retry_pre_geometry is not None:
            self.reseating = True
            self.approach_start_pos = self._retry_pre_geometry['position'].copy()
            self.approach_start_rotation = self._retry_pre_geometry['rotation'].copy()
            self.reference = self.opening
            self.turn_started = None
            self._retry_pre_geometry = None
            return
        super()._enter(old,phase,t)

    def radius_summary(self):
        summary = self.radius_choice.summary()
        summary.update(effective_target_radius_m=self.p['r_m'],
            physical_retry_started=bool(self.physical_radius_retries),
            physical_radius_retries=self.physical_radius_retries,
            maximum_physical_radius_retries=1,
            events=copy.deepcopy(self.radius_events),
            score_changed=False,G1_reference_poses_changed=False,route_side_changed=False)
        return summary

    def command(self):
        action = super().command()
        self.last_diagnostics.update(contact_target_radius_m=self.p['r_m'],
            contact_target_fraction=self.p['r_m']/self.radius_choice.handle_length_m,
            contact_radius_retries=self.physical_radius_retries)
        return action
