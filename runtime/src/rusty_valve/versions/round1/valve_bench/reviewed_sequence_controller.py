"""Opt-in execution of an explicitly confirmed, complete edited sequence.

References come from saved geometry. Valve angle selects the working pose;
native motors/contact, measured phase gates and the task scorer remain in the
existing executor. No code in this module writes a physical valve joint.
"""
from __future__ import annotations

import copy
from pathlib import Path
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .path_obstacles import PathObstacleConflict
from .range_g1_controller import RangeG1Controller
from .storyboard_inputs import CONTROLLER_ROLES
from .tool_upright import ToolUprightConflict


POLICY = 'user_confirmed_sequence_reference_v1'
WORKING_FIT_POLICY = 'finite_saved_rotation_working_faces_v1'


def _quintic(value):
    value = float(np.clip(value, 0., 1.))
    return value**3*(10.+value*(-15.+6.*value))


def _rotation(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T@matrix, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix), 1., atol=1e-6)):
        raise ValueError('Confirmed sequence contains an invalid rotation')
    return matrix


class ConfirmedSequenceReference:
    """All edited free-motion knots and all angle-indexed working references."""

    def __init__(self, path, env=None):
        self.path = Path(path).resolve()
        self.bundle = json.loads(self.path.read_text(encoding='utf-8'))
        if self.bundle.get('schema') != POLICY:
            raise ValueError('Expected a user-confirmed sequence reference bundle')
        approval = self.bundle.get('approval', {})
        if approval.get('user_confirmed') is not True or not approval.get('user_original_text'):
            raise ValueError('Edited motion requires its explicit user confirmation')
        self.reference_contact_radius_m = float(self.bundle['reference_contact_radius_m'])
        if not np.isfinite(self.reference_contact_radius_m) or self.reference_contact_radius_m <= 0.:
            raise ValueError('Confirmed source contact radius must be finite and positive')
        self.sequence_path = Path(self.bundle['sequence_file'])
        if not self.sequence_path.is_absolute():
            self.sequence_path = (self.path.parent / self.sequence_path).resolve()
        self.sequence = json.loads(self.sequence_path.read_text(encoding='utf-8'))
        self.names = self.sequence['qpos_joint_names']
        self.rows = self.sequence['keyposes']
        self.roles = self.bundle['roles']
        if set(self.roles) != set(CONTROLLER_ROLES):
            raise ValueError('Confirmed sequence needs all eight role indices')
        self.final_index = self.bundle['final_rest_index']
        indices = [self.roles[role] for role in CONTROLLER_ROLES]+[self.final_index]
        if (any(isinstance(i, bool) or not isinstance(i, int) for i in indices)
                or indices != sorted(set(indices)) or indices[0] != 0
                or indices[-1] != len(self.rows)-1):
            raise ValueError('Confirmed role indices must cover the ordered complete sequence')
        if (len(self.names) != len(set(self.names)) or not self.names
                or self.sequence.get('model_revision') != self.bundle.get('model_revision')):
            raise ValueError('Confirmed sequence joint/model identity is inconsistent')
        self.q = np.asarray([row['qpos'] for row in self.rows], dtype=float)
        if self.q.shape != (len(self.rows), len(self.names)) or not np.isfinite(self.q).all():
            raise ValueError('Confirmed sequence has invalid full joint vectors')
        self.side = self.sequence.get('active_side', 'left')
        if self.side not in ('left', 'right'):
            raise ValueError('Confirmed sequence needs an operating arm')
        self.active = np.array([self.names.index(f'openarm_{self.side}_joint{i}')
                               for i in range(1, 8)], dtype=int)
        idle_side = 'right' if self.side == 'left' else 'left'
        self.idle = np.array([self.names.index(f'openarm_{idle_side}_joint{i}')
                             for i in range(1, 8)], dtype=int)
        if not np.allclose(self.q[:, self.idle], self.q[0, self.idle], rtol=0., atol=1e-8):
            raise ValueError('The one-hand executor cannot silently discard edited idle-arm motion')
        # These short stages currently have one saved edge each. Refuse extra
        # knots instead of quietly dropping a future richer approach/release.
        for first, last in (('chest_ready', 'precontact'), ('precontact', 'contact'), ('end', 'release'),
                            ('release', 'chest_return')):
            if self.roles[last] != self.roles[first]+1:
                raise ValueError(f'Additional {first}-to-{last} knots need an explicit stage mapping')
        self.section_indices = dict(
            lift=list(range(self.roles['standby'], self.roles['chest_ready']+1)),
            lower=list(range(self.roles['chest_return'], self.final_index+1)))
        self.source_paths = {key:self.q[value][:, self.active].copy()
                             for key, value in self.section_indices.items()}
        inactive = [i for i in range(len(self.names)) if i not in self.active]
        for section, selected in self.section_indices.items():
            if not np.allclose(self.q[selected][:, inactive], self.q[selected[0], inactive],
                               rtol=0., atol=1e-8):
                raise ValueError(f'Confirmed {section} changes an inactive joint or the valve')
        self.paths = copy.deepcopy(self.source_paths)
        self.actual_starts = {}
        self.return_connection_plan = None
        self.work_indices = list(range(self.roles['contact'], self.roles['end']+1))
        angles = np.asarray([self.rows[i]['opening_deg'] for i in self.work_indices], dtype=float)
        delta = np.diff(angles)
        expected_sign = 1. if self.sequence.get('task') == 'open' else -1.
        if not np.isfinite(angles).all() or not np.all(expected_sign*delta > 0.):
            raise ValueError('Every working frame needs a distinct ordered actual valve angle')
        positions, rotations = [], []
        for index in self.work_indices:
            frame = self.rows[index]['actual_in_handle_frame']
            position = np.asarray(frame['position_m'], dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('Confirmed working TCP position is invalid')
            positions.append(position)
            rotations.append(_rotation(frame['rotation_matrix']))
        order = np.argsort(angles)
        self.angles = angles[order]
        self.positions = np.asarray(positions)[order]
        self.rotations = np.asarray(rotations)[order]
        self.work_q = self.q[self.work_indices][:, self.active][order]
        self.slerp = Slerp(self.angles, Rotation.from_matrix(self.rotations))
        pre = self.rows[self.roles['precontact']]['actual_in_handle_frame']
        contact = self.rows[self.roles['contact']]['actual_in_handle_frame']
        self.approach_delta = np.asarray(pre['position_m'])-np.asarray(contact['position_m'])
        if self.approach_delta.shape != (3,) or not np.isfinite(self.approach_delta).all():
            raise ValueError('Confirmed approach direction is invalid')
        if env is not None:
            self.bind(env)

    def bind(self, env):
        adapter = env.robot_adapter.to_dict()
        for key, adapter_key in (('model_revision', 'model_revision'),
                                 ('robot_adapter_id', 'adapter_id')):
            if not self.bundle.get(key) or self.bundle[key] != adapter.get(adapter_key):
                raise ValueError(f'Confirmed sequence {key} differs from the physical model')
        names = [env.model.joint(i).name for i in range(env.model.njnt)]
        if names != self.names or env.model.nq != len(names):
            raise ValueError('Confirmed sequence joint order differs from the physical model')
        if self.side != env.side or not np.array_equal(self.active, env.active_qpos_indices):
            raise ValueError('Confirmed sequence operating arm differs from the physical model')
        for jid in np.flatnonzero(env.model.jnt_limited):
            address = int(env.model.jnt_qposadr[jid])
            lower, upper = env.model.jnt_range[jid]
            if np.any(self.q[:, address] < lower-1e-7) or np.any(self.q[:, address] > upper+1e-7):
                raise ValueError('Confirmed pose exceeds a native joint limit')
        for aid in np.flatnonzero(env.model.actuator_ctrllimited):
            jid = int(env.model.actuator_trnid[aid, 0])
            if jid < 0:
                continue
            address = int(env.model.jnt_qposadr[jid])
            lower, upper = env.model.actuator_ctrlrange[aid]
            if np.any(self.q[:, address] < lower-1e-7) or np.any(self.q[:, address] > upper+1e-7):
                raise ValueError('Confirmed pose exceeds a native motor command range')

    def validate_roles(self, poses):
        if len(poses) != 8:
            raise ValueError('Confirmed controller needs the eight formal G1 roles')
        for role, pose in zip(CONTROLLER_ROLES, poses):
            if (pose.get('qpos_joint_names') != self.names
                    or not np.allclose(pose['qpos'], self.q[self.roles[role]], rtol=0., atol=1e-8)):
                raise ValueError(f'Formal {role} differs from the confirmed sequence')

    def active_pose(self, role):
        return self.q[self.roles[role], self.active].copy()

    def bind_actual_start(self, section, actual):
        values = np.asarray(actual, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError('Actual free-motion start needs seven finite joints')
        self.paths[section] = self.source_paths[section].copy()
        self.paths[section][0] = values
        self.actual_starts[section] = values.tolist()
        if section == 'lower':
            # A planned connection belongs to its measured start. Do not reuse
            # the preflight curve after actual motor motion changes that start.
            self.return_connection_plan = None

    def evaluate(self, fraction, reverse=False):
        if reverse and self.return_connection_plan is not None:
            return self.return_connection_plan.evaluate(fraction)
        points = self.paths['lower' if reverse else 'lift']
        u = float(np.clip(fraction, 0., 1.))*(len(points)-1)
        index = min(int(u), len(points)-2)
        weight = _quintic(u-index)
        return (1.-weight)*points[index]+weight*points[index+1]

    def working_pose(self, opening):
        angle = float(np.clip(opening, self.angles[0], self.angles[-1]))
        position = np.asarray([np.interp(angle, self.angles, self.positions[:, i]) for i in range(3)])
        return position, self.slerp([angle]).as_matrix()[0]

    def working_seed(self, opening):
        angle = float(np.clip(opening, self.angles[0], self.angles[-1]))
        return np.asarray([np.interp(angle, self.angles, self.work_q[:, i]) for i in range(7)])

    @property
    def summary(self):
        return dict(policy=POLICY, file=str(self.path), sequence_file=str(self.sequence_path),
            model_revision=self.bundle['model_revision'], robot_adapter_id=self.bundle['robot_adapter_id'],
            saved_frame_count=len(self.rows), roles=copy.deepcopy(self.roles),
            final_rest_index=self.final_index, free_section_source_indices=self.section_indices,
            work_source_indices=self.work_indices, work_reference_angles_deg=self.angles.tolist(),
            free_curve=('piecewise_confirmed_knots_with_local_geometry_bends_on_obstructed_lower_edges'
                if self.return_connection_plan is not None else
                'piecewise_all_seven_joint_quintic_through_every_saved_knot'),
            work_curve='actual_valve_angle_linear_TCP_and_Slerp_rotation_in_actual_handle_frame',
            actual_free_starts=copy.deepcopy(self.actual_starts),
            reference_contact_radius_m=self.reference_contact_radius_m,
            return_connection=(self.return_connection_plan.summary
                if self.return_connection_plan is not None else None),
            idle_arm_unchanged=True, score_changed=False, valve_commanded=False)


class _SequenceOrientationReferences:
    def __init__(self, env, reference):
        self.env, self.reference = env, reference
        scratch = mujoco.MjData(env.model)
        scratch.qpos[:] = reference.q[reference.roles['chest_ready']]
        mujoco.mj_kinematics(env.model, scratch)
        self.preparation = scratch.xmat[env.active_hook_body_id].reshape(3, 3).copy()

    def at(self, opening_deg, valve_q, *, scope='work'):
        if scope == 'preparation':
            return self.preparation.copy()
        radial, stem = self.env.station_spec.radial_at(valve_q), self.env.stem_axis_world
        frame = np.column_stack((radial, np.cross(stem, radial), stem))
        return frame@self.reference.working_pose(opening_deg)[1]


class ReviewedSequenceController(RangeG1Controller):
    """Optional complete edited references around the unchanged native servo."""

    def __init__(self, env, variant, method, document, parameters):
        self.contact_preload_m = 0.
        self.contact_preload_updates = 0
        self.sequence_reference = ConfirmedSequenceReference(
            parameters['confirmed_sequence_reference_file'], env)
        self._scratch_reference_valve_q = None
        if self.sequence_reference.sequence.get('task') != variant:
            raise ValueError('Confirmed sequence operation differs from the requested episode')
        super().__init__(env, variant, method, document, parameters)
        self.phase_version += '_confirmed_sequence_v1_geometry_return_v1'
        self.phase_settings.update(confirmed_sequence_reference=self.sequence_reference.summary,
            all_saved_free_waypoints_retained=True, all_saved_work_angles_retained=True,
            return_target='independent_confirmed_return_pose',
            working_target='confirmed_actual_handle_relative_TCP_and_orientation',
            score_changed=False)
        if parameters.get('working_contact_fit_policy') == WORKING_FIT_POLICY:
            self.phase_settings.update(working_target=WORKING_FIT_POLICY,
                actual_correct_face_seating_still_required=True,
                contact_preload_max_m=float(parameters.get('contact_preload_max_m',0.)),
                contact_preload_min_m=float(parameters.get('contact_preload_min_m',0.)),
                contact_preload_rate_m_s=float(parameters.get('contact_preload_rate_m_s',0.)),
                contact_preload_is_control_target_not_force_or_score_limit=True)

    def command(self):
        """Close a measured empty gap without relaxing the real seating gate."""
        e,p = self.env,self.p
        maximum=float(p.get('contact_preload_max_m',0.))
        minimum=min(maximum,float(p.get('contact_preload_min_m',0.)))
        rate=float(p.get('contact_preload_rate_m_s',0.))
        phase=self.phase_gate.phase
        if p.get('working_contact_fit_policy') == WORKING_FIT_POLICY and maximum>0. and rate>0.:
            from .execution_state_observation import measure_seating
            if phase in ('approach','turn','hold_final'):
                near=(phase!='approach' or self._motion_elapsed(phase,e.data.time)>=.8*p['approach_s'])
                if near:
                    seat=measure_seating(e,self.method['mode'],p['r_m'])
                    # Leave an incorrect contact to the existing unloading
                    # and reseating path; never push a wrong face harder.
                    wrong='nonworking' in seat['working_faces']
                    if seat['geometry_seated'] and not wrong and (
                            self.contact_preload_m<minimum or not seat['current_correct_face_contact']):
                        self.contact_preload_m=min(maximum,self.contact_preload_m+rate*p['control_dt'])
                        self.contact_preload_updates+=1
                    elif wrong:
                        self.contact_preload_m=max(0.,self.contact_preload_m-rate*p['control_dt'])
            else:
                self.contact_preload_m=0.
        action=super().command()
        self.last_diagnostics.update(contact_normal_preload_m=self.contact_preload_m,
            contact_normal_preload_updates=self.contact_preload_updates)
        return action

    def _sync_ik(self):
        super()._sync_ik()
        if self._scratch_reference_valve_q is not None:
            self.ik.data.qpos[self.env.valve_qpos_index] = self._scratch_reference_valve_q

    def _require_clear_connection(self, reverse=False):
        if not reverse or self.path_obstacle_guard is None:
            return super()._require_clear_connection(reverse=reverse)
        from .free_return_connection import plan_return_connections

        reference, env = self.sequence_reference, self.env
        active_path = reference.paths['lower']
        # Only the operating-arm connection is planned. The other arm and the
        # measured valve angle are copied into scratch configurations, never
        # assigned to physical data. Preflight deliberately uses the saved end
        # angle, while actual entry uses the measured end angle and arm start.
        full = np.repeat(env.data.qpos[None, :], len(active_path), axis=0)
        full[:, reference.active] = active_path
        if self._scratch_reference_valve_q is not None:
            full[:, env.valve_qpos_index] = self._scratch_reference_valve_q
        reference.return_connection_plan = plan_return_connections(
            env, full, active_indices=reference.active)
        summary = reference.return_connection_plan.summary
        self.preparation.append(dict(name='geometry_aware_lower_connections', **summary))
        if hasattr(self, 'phase_settings'):
            self.phase_settings['return_connection'] = summary
        # Keep the original contact classifier and control-time precheck as an
        # independent final gate. No collision/score tolerance is relaxed.
        return super()._require_clear_connection(reverse=True)

    def _approved_contact_seed(self, opening):
        return self.sequence_reference.working_seed(opening)

    def target(self, measured, *, lead=0., clearance=0.):
        e, p = self.env, self.p
        q = e.q_close_rad+(e.q_open_rad-e.q_close_rad)*float(measured)/90.
        radial, stem = e.station_spec.radial_at(q), e.stem_axis_world
        frame = np.column_stack((radial, np.cross(stem, radial), stem))
        relative, rotation = self.sequence_reference.working_pose(measured)
        if p.get('working_contact_fit_policy') == WORKING_FIT_POLICY:
            from .working_contact_fit import fit_working_contact
            from .execution_state_observation import _handle_section
            width, thickness = _handle_section(e)
            geometry = fit_working_contact(e.valve_pivot_world,radial,stem,
                frame@rotation,self.method['mode'],p['r_m'],width,thickness,
                # Both fixed S flanks have orthogonal normals; convert the
                # existing retreat distance to equal normal-plane clearance.
                clearance=float(clearance)/np.sqrt(2.),
                preload_m=(getattr(self,'contact_preload_m',0.) if not clearance else 0.))
            lead_q = e.q_close_rad+(e.q_open_rad-e.q_close_rad)*(float(measured)+self.direction*lead)/90.
            lead_radial = e.station_spec.radial_at(lead_q)
            return (geometry['tcp']+p['r_m']*(lead_radial-radial),geometry['rotation'])
        relative[0] += p['r_m']-self.sequence_reference.reference_contact_radius_m
        if clearance:
            relative += (float(clearance)/p['approach_clearance_m'])*self.sequence_reference.approach_delta
        lead_q = e.q_close_rad+(e.q_open_rad-e.q_close_rad)*(float(measured)+self.direction*lead)/90.
        lead_radial = e.station_spec.radial_at(lead_q)
        position = e.valve_pivot_world+frame@relative+p['r_m']*(lead_radial-radial)
        return position, frame@rotation

    def _prepare(self, poses):
        # Preserve S5's finite choices and one actual-detachment retry policy.
        while self.radius_choice.current_radius_m is not None:
            self.p['r_m'] = self.radius_choice.current_radius_m
            try:
                self._prepare_confirmed_candidate(poses)
            except (PathObstacleConflict, ToolUprightConflict) as exc:
                self.radius_events.append(dict(kind='planning_candidate_rejected',
                    radius_m=self.p['r_m'], reason=type(exc).__name__,
                    observation=copy.deepcopy(exc.observation), physics_started=False))
                if self.radius_choice.advance('preferred_or_alternate_planning_rejected') is None:
                    exc.observation['contact_radius_selection'] = self.radius_summary()
                    raise
            else:
                self.radius_events.append(dict(kind='initial_radius_selected',
                    radius_m=self.p['r_m'], physical_seating_not_yet_proved=True))
                return

    def _prepare_confirmed_candidate(self, poses):
        e, p, reference = self.env, self.p, self.sequence_reference
        reference.validate_roles(poses)
        if not np.allclose(reference.q[reference.final_index, reference.active],
                           self.natural[self.active_offsets], rtol=0., atol=1e-8):
            raise ValueError('The confirmed final natural rest differs from its shared initial rest')
        if np.any(np.asarray(p['standby_offset_m'], dtype=float)):
            raise ValueError('A standby offset cannot silently replace a confirmed preparation pose')
        self.orientation_references = _SequenceOrientationReferences(e, reference)
        self.reviewed_free_connections = reference
        e.reset(seed=int(p.get('seed', 0)), initial_robot_qpos=self.natural)
        observation = self._obstacle_for_joints(self.natural[self.active_offsets], 'approved_standby')
        if not observation['clear']:
            raise PathObstacleConflict(observation)
        self.observe_tool_orientation(e.data, stage='approved_standby')
        self.preparation.append(dict(name='complete_confirmed_sequence_input', **reference.summary))
        self.preparation.append(dict(name='confirmed_contact_radius_mapping',
            source_reference_contact_radius_m=reference.reference_contact_radius_m,
            preferred_contact_radius_m=self.radius_choice.preferred_radius_m,
            effective_contact_radius_m=p['r_m'],
            handle_radial_translation_m=p['r_m']-reference.reference_contact_radius_m,
            source_joints_and_reference_geometry_changed=False))
        self._require_clear_connection()
        # Lowering must be checked at its saved final valve angle in scratch
        # data, rather than against the episode's initial valve angle.
        self._scratch_reference_valve_q = float(reference.q[reference.roles['chest_return'], e.valve_qpos_index])
        try:
            self._require_clear_connection(reverse=True)
        finally:
            self._scratch_reference_valve_q = None
        pre = reference.active_pose('precontact')
        if p.get('working_contact_fit_policy') == WORKING_FIT_POLICY:
            # The saved pose remains an IK reference. Finish alignment while
            # clear of the handle, then enter through the physical recess;
            # do not sweep an outer metal flank across it while rotating.
            position, rotation = self.target(self.opening,clearance=p['approach_clearance_m'])
            pre, _ = self._prepare_target(position,rotation,
                [pre,reference.active_pose('contact')],'working_faces_aligned_precontact')
            self.preparation.append(dict(name='finite_working_faces_contact_policy',
                policy=WORKING_FIT_POLICY,saved_rotation_preserved=True,
                current_radius_used_at_every_measured_angle=True,
                fitting_is_not_actual_seating=True,preload_m=0.,
                aligned_before_insertion=True,saved_precontact_used_as_ik_seed=True,
                score_changed=False,native_motor_capabilities_changed=False))
        geometry = self._static_geometry(pre)
        self.approach_start_pos = geometry['position'].copy()
        self.approach_start_rotation = geometry['rotation'].copy()
        self.reach_points = self._choose_free_path(self.chest[self.active_offsets], pre,
            geometry['position'], geometry['rotation'], pre, 'reach_path_ranking')
        self._require_upright_path(self.reach_points, 'reach')
        self.last_free_q, self.qseed = pre.copy(), pre.copy()
        self.reference = self.opening

    def _enter(self, old, phase, t):
        e, reference = self.env, self.sequence_reference
        actual = e.data.qpos[e.active_qpos_indices].copy()
        if phase == 'return_chest' and not self._radius_retreat_pending:
            # The inherited gate, return planner and native command must all
            # agree on this independent endpoint, not the lift's ready pose.
            self.chest[self.active_offsets] = reference.active_pose('chest_return')
        if phase == 'lift_to_chest':
            reference.bind_actual_start('lift', actual)
            self._require_clear_connection()
        elif phase == 'lower_to_natural':
            reference.bind_actual_start('lower', actual)
        super()._enter(old, phase, t)
        if phase == 'approach':
            self._actual_start_approach()
        elif phase == 'reach' and self.phase_gate.phase == 'reach':
            # The reference endpoint remains exact; the physical arm supplies
            # the first point of its already chosen same-side corridor.
            self.reach_points[0] = actual
            check = self._scan_obstacle_path(
                lambda u:self._free(u*self.p['reach_s'], self.reach_points, self.p['reach_s']),
                self.p['reach_s'], 'reach_actual_start')
            if not check['clear']:
                raise PathObstacleConflict(check['first_block'])
            self._require_upright_path(self.reach_points, 'reach')
