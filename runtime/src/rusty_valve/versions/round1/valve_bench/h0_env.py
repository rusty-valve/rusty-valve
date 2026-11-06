"""Compose a passive workstation and a separately declared robot on local CPU.

The runner has no workstation-specific or robot-name-specific contact rules.
Only reset assigns positions. Each step applies passive valve friction and
finite robot actuator targets, then obtains contacts from MuJoCo's solve.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .h0_robot import compose_scene, load_robot_adapter, robot_geom_groups
from .h0_scene import environment_geom_groups
from .h0_station import load_workstation


class H0Env:
    def __init__(self, workstation='A1', robot_adapter='openarm_s_hook', *,
                 task='close', control_dt=.005, max_time_s=60.,
                 physics_observer=None, require_endstop=False, final_hold_s=2.8,
                 **kwargs):
        if task not in ('close', 'open', 'slow_open'):
            raise ValueError('Choose close, open or slow_open')
        self.task = 'open' if task == 'slow_open' else task
        self.variant = task
        self.station_spec = load_workstation(workstation)
        self.station = self.station_spec.case
        self.robot_adapter = load_robot_adapter(robot_adapter)
        self.side = self.robot_adapter.side
        # Use the exact full model already compiled and validated by the
        # composer. Recompiling its XML would briefly allocate another model
        # and repeat mesh compilation without adding a validation check.
        self.xml, self.composition_metadata, self.model = compose_scene(
            self.station_spec, self.robot_adapter, return_compiled=True)
        self.data = mujoco.MjData(self.model)
        self.physics_dt = float(self.model.opt.timestep)
        self.control_dt, self.max_time_s = float(control_dt), float(max_time_s)
        self.frame_skip = round(self.control_dt/self.physics_dt)
        if self.frame_skip < 1 or abs(self.frame_skip*self.physics_dt-self.control_dt) > 1e-9:
            raise ValueError('control_dt must be a positive multiple of physics_dt')
        self.physics_observer = physics_observer
        self.require_endstop, self.final_hold_s = require_endstop, final_hold_s
        a, m, s = self.robot_adapter, self.model, self.station_spec
        self.robot_groups = robot_geom_groups(m, a)
        self.environment_groups = environment_geom_groups(m, s)
        self.robot_geom_ids = set(self.robot_groups['robot_geom_ids'])
        self.active_tool_geom_ids = set(self.robot_groups['operation_end'])
        self.idle_geom_ids = set(self.robot_groups['idle_hand'])
        self.valve_geom_ids = set(self.environment_groups['handle_geom_ids'])
        self.fixture_geom_ids = set(self.environment_groups['fixture_geom_ids'])
        self.robot_joint_names = tuple(a.robot_joint_names)
        self.active_joint_names = tuple(a.active_joint_names)
        self.active_actuator_names = tuple(a.active_actuator_names)
        ad = a.to_dict()
        self.camera_parameters = self.composition_metadata.get('cameras', ad.get('cameras', []))
        self.control_mode = ad['control']['mode']
        region = s.to_dict()['allowed_base_region_world']
        base_xy = np.asarray(ad['placement']['base_position_m'][:2])
        if np.any(base_xy < region['xy_min_m']) or np.any(base_xy > region['xy_max_m']):
            raise ValueError('Robot base lies outside the workstation allowed region')
        self.idle_joint_names = tuple(ad.get('control', {}).get('idle_joint_names', ()))
        self.idle_actuator_names = tuple(ad.get('control', {}).get('idle_actuator_names', ()))
        # The example teacher is OpenArm-specific. These bindings come from the
        # adapter; the environment and scorer do not use arm-name templates.
        self.ee_site_name = a.ee_site_name
        self.robot_joint_ids = np.array([m.joint(n).id for n in self.robot_joint_names], int)
        self.robot_qpos_indices = m.jnt_qposadr[self.robot_joint_ids].copy()
        self.robot_dof_indices = m.jnt_dofadr[self.robot_joint_ids].copy()
        self.actuator_ids = np.array([m.actuator(n).id for n in a.actuator_names], int)
        self.active_actuator_ids = np.array([m.actuator(n).id for n in a.active_actuator_names], int)
        self.idle_actuator_ids = np.array([m.actuator(n).id for n in self.idle_actuator_names], int)
        self.active_qpos_indices = np.array([m.joint(n).qposadr[0] for n in a.active_joint_names], int)
        self.active_dof_indices = np.array([m.joint(n).dofadr[0] for n in a.active_joint_names], int)
        self.idle_qpos_indices = np.array([m.joint(n).qposadr[0] for n in self.idle_joint_names], int)
        ee_body = int(m.site(self.ee_site_name).bodyid[0])
        # The S teacher uses the tool root's orientation; the adapter names it.
        self.active_hook_body_id = int(m.body(ad['control']['operation_body_name']).id) if ad.get('control', {}).get('operation_body_name') else ee_body
        self.valve_joint_id = int(m.joint(s.valve_joint_name).id)
        self.valve_qpos_index = int(m.jnt_qposadr[self.valve_joint_id])
        self.valve_dof_index = int(m.jnt_dofadr[self.valve_joint_id])
        self.valve_body_id = int(m.body(s.valve_moving_body_name).id)
        self.valve_mount_body_id = int(m.body(s.valve_mount_body_name).id)
        self.valve_installation_body_id = int(m.body(s.installation_body_name).id)
        self.q_open = self.q_open_rad = s.q_open_rad
        self.q_close = self.q_close_rad = s.q_close_rad
        self.stem_axis_world = s.stem_axis_world
        self.valve_hinge_axis_world = s.closing_axis_world
        self.valve_pivot_world = s.pivot_world_m
        self.valve_radial_zero_world = s.radial_zero_world
        self.target_angle_rad = self.q_open if self.task == 'open' else self.q_close
        self.initial_angle_rad = self.q_close if self.task == 'open' else self.q_open
        self.resistance = s.make_resistance()
        self.resistances = {self.station: self.resistance}
        self._force_caps = m.actuator_forcerange.copy()
        self._force_limited = m.actuator_forcelimited.copy()
        self._renderer = None
        self._started = False

    @property
    def valve_angle_rad(self): return float(self.data.qpos[self.valve_qpos_index])

    @property
    def valve_speed_rad_s(self): return float(self.data.qvel[self.valve_dof_index])

    @property
    def warning_count(self): return sum(int(w.number) for w in self.data.warning)

    def _constraint_force(self, kind, identifier, absolute=False):
        ids = np.flatnonzero((self.data.efc_type == int(kind)) & (self.data.efc_id == int(identifier)))
        forces = self.data.efc_force[ids]
        return float(np.sum(np.abs(forces)) if absolute else np.sum(forces))

    def runtime_guarantees(self):
        m, d = self.model, self.data
        caps_same = (self.composition_metadata['actuator_limits_unchanged']
                     and np.array_equal(m.actuator_forcerange, self._force_caps)
                     and np.array_equal(m.actuator_forcelimited, self._force_limited))
        no_valve_motor = all(int(m.actuator_trnid[i, 0]) != self.valve_joint_id for i in range(m.nu))
        return dict(motor_limits_unchanged=bool(caps_same),
                    valve_contact_only=bool(no_valve_motor and not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)),
                    evidence='Separate robot model actuator limits preserved during composition and run; passive valve has no actuator; runner uses only ctrl targets and passive friction, with no applied external driving force.')

    def runtime_declaration(self):
        return self.runtime_guarantees()

    def reset(self, *, seed=0, initial_robot_qpos=None):
        self.seed = int(seed)
        key = 'h0_closed' if self.task == 'open' else 'h0_open'
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(key).id)
        self.data.qpos[self.valve_qpos_index] = self.initial_angle_rad
        if initial_robot_qpos is not None:
            q = np.asarray(initial_robot_qpos, float)
            if q.shape != self.robot_qpos_indices.shape or not np.isfinite(q).all():
                raise ValueError('Initial robot positions must match the adapter robot joints')
            limits = self.model.jnt_range[self.robot_joint_ids]
            if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
                raise ValueError('Initial robot positions exceed model joint limits')
            self.data.qpos[self.robot_qpos_indices] = q
        if self.control_mode == 'joint_position':
            for aid in self.actuator_ids:
                jid = int(self.model.actuator_trnid[aid, 0])
                self.data.ctrl[aid] = self.data.qpos[self.model.jnt_qposadr[jid]]
        self.resistance.reset(-1 if self.task == 'open' else 1)
        self._resistance_state = {self.station: self.resistance.apply(self.model, self.data)}
        mujoco.mj_forward(self.model, self.data)
        self._started = True
        self._has_fixture_contact = self._has_idle_contact = self._has_self_contact = False
        self._has_invalid_tool_contact = self._has_handle_contact = False
        self._max_penetration = self._max_speed = self._stop_reaction = self._hold_time = 0.
        self._stop_seen = self._success = False
        self._last_action_time_s = 0.
        self._last_action = self.data.ctrl[self.actuator_ids].copy()
        self._observe_contacts(track=False)
        if self.physics_observer: self.physics_observer(self)
        return self._info()

    def _observe_contacts(self, *, track):
        from .h0_records import classify_contacts
        self.current_contacts = classify_contacts(self.model, self.data, self.robot_groups, self.environment_groups)
        if track:
            for c in self.current_contacts:
                if c['force_N'] <= 0.: continue
                kind, part = c['kind'], c['robot_part']
                self._has_handle_contact |= kind == 'handle'
                self._has_fixture_contact |= kind == 'fixture' and c['force_N'] > 5.
                self._has_idle_contact |= part == 'idle_hand' or c.get('other_robot_part') == 'idle_hand'
                self._has_self_contact |= kind == 'self'
                self._has_invalid_tool_contact |= kind == 'handle' and part != 'operation_end'
            for c in self.data.contact:
                if int(c.geom1) in self.robot_geom_ids or int(c.geom2) in self.robot_geom_ids:
                    self._max_penetration = max(self._max_penetration, -float(c.dist))

    def step(self, action):
        if not self._started: raise RuntimeError('Call reset before step')
        action = np.asarray(action, float)
        if action.shape != self.actuator_ids.shape or not np.isfinite(action).all():
            raise ValueError('Action must have one finite position target per declared actuator')
        bounds = self.model.actuator_ctrlrange[self.actuator_ids]
        limited = self.model.actuator_ctrllimited[self.actuator_ids]
        command = np.where(limited, np.clip(action, bounds[:, 0], bounds[:, 1]), action)
        self.data.ctrl[self.actuator_ids] = command
        self._last_action_time_s = float(self.data.time)
        self._last_action = command.copy()
        for _ in range(self.frame_skip):
            self._resistance_state[self.station] = self.resistance.apply(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            self._observe_contacts(track=True)
            self._stop_reaction = self._constraint_force(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT, self.valve_joint_id, absolute=True)
            self._max_speed = max(self._max_speed, abs(self.valve_speed_rad_s))
            near = abs(self.target_angle_rad-self.valve_angle_rad) < math.radians(.5)
            self._stop_seen |= near and self._stop_reaction > .05
            stable = near and abs(self.valve_speed_rad_s) < math.radians(1.)
            self._hold_time = self._hold_time+self.physics_dt if stable else 0.
            self._success = self._hold_time >= self.final_hold_s
            if self.physics_observer: self.physics_observer(self)
        return self._info()

    def _info(self):
        return dict(time_s=float(self.data.time), valve_angle_rad=self.valve_angle_rad,
                    opening_angle_deg=90.*(self.valve_angle_rad-self.q_close)/(self.q_open-self.q_close),
                    has_fixture_contact=self._has_fixture_contact, has_idle_contact=self._has_idle_contact,
                    has_self_contact=self._has_self_contact, has_invalid_tool_contact=self._has_invalid_tool_contact,
                    maximum_robot_contact_penetration_m=self._max_penetration,
                    goal_hold_reached=self._success, contacts=self.current_contacts)

    def close(self):
        if self._renderer is not None: self._renderer.close()
