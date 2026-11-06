"""S4: execute the existing G1 geometry with measured stage transitions.

No station branch, new contact target, new motor capability or score is added.
The reviewed curves keep their original durations as motion speeds. Reaching
the end of a curve is insufficient: the real mechanism must complete it.
"""
from __future__ import annotations

import math

import numpy as np

from .execution_phase_gate import MeasuredPhaseGate, VERSION
from .execution_state_observation import measure_seating, measure_handle_clearance
from .g2_episode import G2Controller, blend, interpolate_pose
from .preserved_motion_clock import PreservedApproachClock


class StateG1Controller(G2Controller):
    def __init__(self, env, variant, method, document, parameters):
        if isinstance(document, dict):
            document = document['keyposes']
        super().__init__(env, variant, method, document, parameters)
        self.phase_gate = MeasuredPhaseGate(parameters)
        self.motion_clock = PreservedApproachClock(parameters)
        self.phase_version = VERSION+'_preserved_approach_v2'
        self.reseating = False
        self.blocked_reason = None
        self.turn_started = None
        self.release_endpoint_q = None
        self.release_reference = None
        self.release_start_pos = self.release_start_rotation = None
        self.release_goal_pos = self.release_goal_rotation = None
        if parameters.get('release_reference_policy'):
            from .g1_release_reference import G1ReleaseReference, VERSION as RELEASE_VERSION
            if parameters['release_reference_policy'] != RELEASE_VERSION:
                raise ValueError('Unknown G1 release reference policy')
            self.release_reference = G1ReleaseReference(self.approved_poses[5],
                                                        self.approved_poses[6])
            self.phase_version += '_' + RELEASE_VERSION
        self.phase_settings = dict(
            arrival_joint_tolerance_deg=3.,
            arrival_basis='existing successful preparation tracking residual up to 1.81 deg; control arrival only, not a score',
            release_gate='actual original exit target reached and native tool-handle contact absent',
            release_path_clearance_m=parameters['approach_clearance_m'],
            release_distance_is_measured_not_a_new_55mm_minimum=True,
            time_policy='curve parameter and stuck-stage guard, never sufficient phase evidence',
            contact_target_radius_m=parameters['r_m'],
            contact_target_selection_changed=False,
            first_reach_waypoints_preserved=True,
            initial_approach_selected_pre_FK_preserved=True,
            original_approach_curve_clock_preserved=True,
            score_changed=False)
        if self.release_reference is not None:
            self.phase_settings['release_reference'] = self.release_reference.metadata()
            self.phase_settings['release_gate'] = 'actual G1 exit target reached and native tool-handle contact absent'
            self.phase_settings['release_path_clearance_m'] = None
            self.phase_settings['legacy_clearance_target_used'] = False

    def _uses_original_clock(self, phase):
        return phase in ('initial_down_hold','lift_to_chest','chest_ready','reach') or (
            phase == 'approach' and not self.reseating)

    def _motion_elapsed(self, phase, t):
        if self._uses_original_clock(phase):
            return self.motion_clock.elapsed(phase)
        return self.phase_gate.elapsed(t)

    def _arrival(self, target):
        e = self.env
        actual = np.asarray(e.data.qpos[e.active_qpos_indices])
        active_error = float(np.max(np.abs(actual-np.asarray(target))))
        idle_error = float(np.max(np.abs(
            e.data.qpos[e.idle_qpos_indices]-self.natural[self.idle_offsets])))
        reached = max(active_error, idle_error) <= math.radians(
            self.phase_settings['arrival_joint_tolerance_deg'])
        return reached, dict(active_max_joint_error_deg=math.degrees(active_error),
            idle_max_joint_error_deg=math.degrees(idle_error))

    def _actual_start_approach(self):
        self.approach_start_pos = self.env.data.site_xpos[self.ik.site].copy()
        self.approach_start_rotation = self.env.data.site_xmat[self.ik.site].reshape(3, 3).copy()
        self.qseed = self.env.data.qpos[self.env.active_qpos_indices].copy()
        self.reference = self.opening
        self.approach_elapsed_s = 0.

    def _enter(self, old, phase, t):
        e,p = self.env,self.p
        actual = e.data.qpos[e.active_qpos_indices].copy()
        if phase == 'reach':
            # G1 already supplied and S3 checked the complete curve. Replacing
            # its first waypoint changes the route before any state gate acts.
            # Only the permission to enter it is measured here.
            pass
        elif phase == 'approach':
            if old in ('turn','hold_final'):
                # Unload the previous turning servo bias as well as the
                # explicit tangential force. Fresh seating can build only
                # its own local tracking correction from actual joints.
                self.integral.fill(0.)
                self.reseating = True
                self._actual_start_approach()
            else:
                # On FIRST entry retain _prepare's selected-pre FK and the
                # last reached command seed, exactly as the passed generator.
                # Measured starting FK is appropriate only after a real slip.
                self.reseating = False
            self.turn_started = None
        elif phase == 'turn' and old == 'approach':
            feedback_open = bool(p['slow_feedback'] and self.variant != 'close')
            if self.reseating:
                self.reference = self.opening
                self.turn_started = t
            else:
                self.turn_started = (t if feedback_open else
                    self.motion_clock.nominal_starts['turn']+self.motion_clock.delay_s)
        elif phase == 'release':
            self.withdraw_started = t
            if self.release_reference is not None:
                self.release_start_pos = e.data.site_xpos[self.ik.site].copy()
                self.release_start_rotation = e.data.site_xmat[self.ik.site].reshape(3,3).copy()
                radial = e.station_spec.radial_at(e.valve_angle_rad)
                stem = e.stem_axis_world
                frame = np.column_stack((radial, np.cross(stem,radial), stem))
                self.release_goal_pos,self.release_goal_rotation = self.release_reference.goal_from_actual(
                    self.release_start_pos,self.release_start_rotation,frame)
                self.qseed = actual.copy()
                self.phase_settings['release_actual_entry'] = dict(
                    time_s=float(t), opening_deg=self.opening,
                    start_tcp_position_world_m=self.release_start_pos.tolist(),
                    start_tcp_rotation_world=self.release_start_rotation.tolist(),
                    goal_tcp_position_world_m=self.release_goal_pos.tolist(),
                    goal_tcp_rotation_world=self.release_goal_rotation.tolist())
                pos,rot = self.release_goal_pos,self.release_goal_rotation
            else:
                pos,rot = self.target(self.opening,clearance=p['approach_clearance_m'])
            self._sync_ik()
            self._motion_candidate_stage = 'release_candidate'
            q,_,_ = self._solve_contact(pos,rot,actual,p['ik_iterations'],self.opening,
                allow_alternate=self.release_reference is None)
            self.release_endpoint_q = q.copy()
        elif phase == 'return_chest':
            self.return_started = t
            self.qseed = actual.copy()
            # Only actual detached joints/FK can start the return path.
            pos = e.data.site_xpos[self.ik.site].copy()
            rot = e.data.site_xmat[self.ik.site].reshape(3,3).copy()
            self.return_points = self._choose_free_path(actual,
                self.chest[self.active_offsets],pos,rot,actual,'return_path_ranking')
            self._require_upright_path(self.return_points,'return_chest')
        elif phase == 'lower_to_natural':
            self.lower_started = t
            self._require_clear_connection(reverse=True)
            self._lower_path_obstacles_checked = True

    def _observe(self):
        e,p = self.env,self.p
        phase = self.phase_gate.phase
        elapsed = self._motion_elapsed(phase,e.data.time)
        duration_names = dict(lift_to_chest='lift_s',reach='reach_s',
            approach='approach_s',release='withdraw_s',return_chest='return_s',
            lower_to_natural='lower_s')
        duration = p[duration_names[phase]] if phase in duration_names else 0.
        complete = (self.motion_clock.virtual_time >= self.motion_clock.end_of(phase)
            if self._uses_original_clock(phase) else elapsed >= duration-1e-9)
        if phase in ('initial_down_hold','lower_to_natural','final_down_hold'):
            target = self.natural[self.active_offsets]
        elif phase in ('lift_to_chest','chest_ready','return_chest'):
            target = self.chest[self.active_offsets]
        elif phase == 'reach':
            target = self.reach_points[-1]
        elif phase == 'release':
            target = self.release_endpoint_q
        else:
            target = None
        reached, detail = self._arrival(target) if target is not None else (False,{})
        seating = (measure_seating(e,self.method['mode'],p['r_m'])
            if phase in ('approach','turn','hold_final') else None)
        separation = measure_handle_clearance(e) if phase == 'release' else None
        c = e.scorer.config
        in_goal = abs(self.opening-self.goal) <= c.angle_tolerance_deg+1e-9
        # Read the scorer's CURRENT continuous interval, not its sticky first
        # completion_time. An old successful hold cannot authorize new retreat.
        target_start = e.scorer.target_start
        held = bool(in_goal and target_start is not None and
            float(e.data.time)-target_start >= c.final_hold_s-1e-9)
        detail.update(seating=seating,separation=separation,
            current_goal_hold_s=(float(e.data.time)-target_start
                                 if target_start is not None and in_goal else 0.))
        detached = bool(separation is not None and not separation['any_handle_contact']
                        and separation['minimum_distance_m'] > 0.)
        return dict(reached=reached,trajectory_complete=complete,
            seated=bool(seating and seating['ready_to_turn']),
            in_goal=in_goal,goal_hold_completed=held,detached=detached,
            measurements=detail)

    def _handle_blocked_phase(self, t):
        """S4 stops; later allowed policies may request a measured retreat."""
        return

    def command(self):
        e,p = self.env,self.p
        t = float(e.data.time)
        self.motion_clock.sample(t)
        observed = self._observe()
        old = self.phase_gate.phase
        self.phase_gate.advance(t,**observed)
        if self.phase_gate.blocked_reason is not None:
            self._handle_blocked_phase(t)
        phase = self.phase_gate.phase
        if phase != old:
            self.motion_clock.resume()
            self._enter(old,phase,t)
            phase = self.phase_gate.phase
        elif self._uses_original_clock(phase) and observed['trajectory_complete']:
            # Freeze only when actual-state permission is still absent at the
            # ORIGINAL curve endpoint. No fixed one-tick timing patch is used.
            self.motion_clock.hold_at_phase_end(phase)
        self.blocked_reason = self.phase_gate.blocked_reason
        self.done = self.phase_gate.done
        elapsed = self._motion_elapsed(phase,t)
        drive=pe=re=0.
        slow_active=False
        if phase in ('initial_down_hold','final_down_hold'):
            q=self.natural[self.active_offsets].copy()
        elif phase == 'lift_to_chest':
            q=self._free_connection(elapsed)
        elif phase == 'chest_ready':
            q=self.chest[self.active_offsets].copy()
        elif phase == 'reach':
            q=self._free(elapsed,self.reach_points,p['reach_s'])
        elif phase == 'return_chest':
            q=self._free(elapsed,self.return_points,p['return_s'])
            self.integral *= math.exp(-p['control_dt']/.5)
        elif phase == 'lower_to_natural':
            q=self._free_connection(elapsed,reverse=True)
            self.integral *= math.exp(-p['control_dt']/.5)
        else:
            if phase == 'approach':
                # A partial contact or an expired trajectory is insufficient.
                # There is no angular lead or extra tangential force here.
                if self.reseating:
                    self.reference=self.opening
                contact_pos,contact_rot=self.target(self.opening,clearance=0.)
                pos,rot=interpolate_pose(self.approach_start_pos,
                    self.approach_start_rotation,contact_pos,contact_rot,
                    elapsed/p['approach_s'])
            elif phase == 'release':
                if self.release_reference is not None:
                    pos,rot=interpolate_pose(self.release_start_pos,self.release_start_rotation,
                        self.release_goal_pos,self.release_goal_rotation,elapsed/p['withdraw_s'])
                else:
                    pos,rot=self.target(self.opening,
                        clearance=p['approach_clearance_m']*blend(elapsed/p['withdraw_s']))
                self.integral *= math.exp(-p['control_dt']/.5)
            else:
                lead,_,drive=self._pacing(t)
                slow_active=self._slow_feedback_active(phase)
                target_lead=lead*self.direction
                if self.slow_braking:
                    target_lead=self._braking_target_lead()
                    self.reference=self.opening+self.direction*target_lead
                    drive=0.
                elif slow_active:
                    drive *= p['start_force_scale']
                pos,rot=self.target(self.opening,lead=target_lead)
                force_elapsed=max(0.,t-(self.turn_started if self.turn_started is not None else t))
                drive *= min(1.,force_elapsed/p['force_ramp_s'])
            self._sync_ik()
            self._motion_candidate_stage=phase+'_candidate'
            q,pe,re=self._solve_contact(pos,rot,self.qseed,p['ik_iterations'],self.opening,
                allow_alternate=phase!='approach' and not (
                    phase=='release' and self.release_reference is not None))
            if phase == 'release' and elapsed >= p['withdraw_s']:
                self.release_endpoint_q=q.copy()
        action=self._motor_action(q,phase=phase,position_error=pe,
            rotation_error=re,drive=drive,slow_active=slow_active)
        self.last_diagnostics.update(phase_transition_policy=self.phase_version,
            task_stage=dict(initial_down_hold='接近',lift_to_chest='接近',
                chest_ready='接近',reach='接近',approach='入位',turn='转动',
                hold_final='保持',release='脱离',return_chest='收回',
                lower_to_natural='收回',final_down_hold='收回')[phase],
            stage_elapsed_s=elapsed,state_observation=observed['measurements'],
            original_curve_clock=dict(virtual_time_s=self.motion_clock.virtual_time,
                actual_waiting_s=self.motion_clock.delay_s,
                waiting_for_measured_gate=self.motion_clock.waiting),
            seating_state=observed['measurements'].get('seating'),
            blocked_reason=self.blocked_reason)
        return action
