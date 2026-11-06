"""Small Gymnasium interface for the stock OpenArm valve task.

Actions are the seven official joint-position commands (rad) and the official
finger-position command (m). Observations and info describe the state *after*
the control interval; ``action_time_s`` says when its command was sent. Only
reset sets joint positions. All subsequent movement comes from MuJoCo dynamics.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .resistance import ValveResistance
from .robot import ARM_ACTUATORS, ARM_JOINTS, GRIPPER_ACTUATOR, build_robot_scene


class ValveEnv(gym.Env):
    """Fixed-base reference task, with an unchanged official OpenArm gripper.

    Valve angle zero is open; positive angle closes. Ball tasks use a 90 degree
    stroke. The globe model has a declared, assumed four-turn stroke.
    ``resistance_scale`` changes only the valve's passive friction envelope.
    ``slow_open`` adds an illustrative rule: open 5--10 degrees at no more than
    10 deg/s, stop at <=1 deg/s for one second, then continue. These thresholds
    are task settings, not measured valve characteristics or a plant procedure.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        valve: str = "ball",
        layout: str = "horizontal_top",
        task: str = "close",
        resistance_scale: float = 1.,
        valve_position=(-.01, .125, .70),
        base_position=(0, 0, .55),
        base_quat=(1, 0, 0, 0),
        image_size=(320, 240),
        include_image: bool = True,
        control_dt: float = .02,
        max_time_s: float = 30.,
        close_angle_deg: float = 90.,
        robot_configuration: str = "single",
        shoulder_height: float = 1.4,
        tool: str = "stock",
        require_endstop: bool = False,
    ):
        self.robot_configuration = robot_configuration
        self.shoulder_height = float(shoulder_height)
        self.tool = tool
        self.require_endstop = bool(require_endstop)
        self.valve = valve
        self.layout = layout
        self.task = task
        self.resistance_scale = float(resistance_scale)
        self.valve_position = tuple(valve_position)
        self.base_position = tuple(base_position)
        self.base_quat = tuple(base_quat)
        self.image_size = tuple(image_size)  # width, height
        self.include_image = bool(include_image)
        self.control_dt = float(control_dt)
        self.max_time_s = float(max_time_s)
        self.close_angle_deg = float(close_angle_deg)
        self.render_mode = "rgb_array"
        self._renderer = None
        self._renderer_size = None
        self._episode_started = False
        self._build()

    def _build(self) -> None:
        if self.task not in ("close", "open", "slow_open", "wheel_turn"):
            raise ValueError("task must be close, open, slow_open or wheel_turn")
        if len(self.image_size) != 2 or any(int(v) != v or v <= 0 for v in self.image_size):
            raise ValueError("image_size must be (positive width, positive height)")
        if not math.isfinite(self.max_time_s) or self.max_time_s <= 0:
            raise ValueError("max_time_s must be positive and finite")
        self.close()
        if self.robot_configuration == "bimanual":
            from .bimanual_robot import build_bimanual_scene, robot_names
            self.names = robot_names(tool=self.tool)
            self.xml = build_bimanual_scene(
                valve=self.valve, layout=self.layout, valve_position=self.valve_position,
                shoulder_height=self.shoulder_height, resistance_scale=self.resistance_scale,
                close_angle_deg=self.close_angle_deg, tool=self.tool)
        else:
            self.names = dict(robot_joint_names=(*ARM_JOINTS, "openarm_finger_joint1", "openarm_finger_joint2"),
                              actuator_names=(*ARM_ACTUATORS, GRIPPER_ACTUATOR),
                              active_joint_names=ARM_JOINTS, active_actuator_names=ARM_ACTUATORS,
                              gripper_actuator_name=GRIPPER_ACTUATOR,
                              root_body_names=("openarm_link0",))
            self.xml = build_robot_scene(
                valve=self.valve, layout=self.layout,
                valve_position=self.valve_position, base_position=self.base_position,
                base_quat=self.base_quat, resistance_scale=self.resistance_scale,
                close_angle_deg=self.close_angle_deg)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.physics_dt = float(self.model.opt.timestep)
        if not math.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be positive and finite")
        self.frame_skip = round(self.control_dt / self.physics_dt)
        if (self.frame_skip < 1
                or abs(self.frame_skip * self.physics_dt - self.control_dt) > 1e-9):
            raise ValueError("control_dt must be a positive integer multiple of the physics timestep")
        robot_joints = self.names["robot_joint_names"]
        self.robot_joint_ids = np.array([self.model.joint(n).id for n in robot_joints])
        self.robot_qpos_indices = self.model.jnt_qposadr[self.robot_joint_ids].copy()
        self.robot_dof_indices = self.model.jnt_dofadr[self.robot_joint_ids].copy()
        self.actuator_ids = np.array([
            self.model.actuator(n).id for n in self.names["actuator_names"]
        ])
        self.valve_joint_id = self.model.joint("valve_hinge").id
        self.valve_qpos_index = int(self.model.jnt_qposadr[self.valve_joint_id])
        self.valve_dof_index = int(self.model.jnt_dofadr[self.valve_joint_id])
        self.valve_body_id = self.model.body("valve_moving").id
        self.valve_geom_ids = set(np.flatnonzero(self.model.geom_bodyid == self.valve_body_id))
        self.fixture_geom_ids = set(np.flatnonzero(self.model.geom_bodyid == self.model.body("valve_mount").id))
        robot_bodies = {self.model.body(n).id for n in self.names["root_body_names"]}
        for body in range(1, self.model.nbody):
            if int(self.model.body_parentid[body]) in robot_bodies:
                robot_bodies.add(body)
        # Pipe supports are fixed world geoms; include them in obstruction checks.
        self.fixture_geom_ids.update(g for g in range(self.model.ngeom)
            if (self.model.geom(g).name or '').startswith(
                ('support_', 'neighbor_support_', 'vertical_pipe_')))
        self.robot_geom_ids = set(
            np.flatnonzero(np.isin(self.model.geom_bodyid, list(robot_bodies)))
        )
        # The idle arm must not lend a hand by touching either the moving
        # handle/wheel or its fixed pipework. Include its fingers and base link.
        idle_joint_names = self.names.get('idle_joint_names', ())
        idle_bodies = {int(self.model.jnt_bodyid[self.model.joint(n).id])
                       for n in idle_joint_names}
        if idle_joint_names:
            first_body = int(self.model.jnt_bodyid[self.model.joint(idle_joint_names[0]).id])
            idle_bodies.add(int(self.model.body_parentid[first_body]))
        for body in range(1, self.model.nbody):
            if int(self.model.body_parentid[body]) in idle_bodies:
                idle_bodies.add(body)
        self.idle_geom_ids = set(
            np.flatnonzero(np.isin(self.model.geom_bodyid, list(idle_bodies)))
        )
        self.idle_forbidden_geom_ids = self.fixture_geom_ids | self.valve_geom_ids
        ranges = self.model.actuator_ctrlrange[self.actuator_ids]
        self.action_space = spaces.Box(ranges[:, 0].astype(np.float32),
                                       ranges[:, 1].astype(np.float32))
        observation_spaces = {
            "robot_qpos": spaces.Box(-np.inf, np.inf, (len(robot_joints),), dtype=np.float32),
            "robot_qvel": spaces.Box(-np.inf, np.inf, (len(robot_joints),), dtype=np.float32),
        }
        if self.include_image:
            width, height = map(int, self.image_size)
            observation_spaces["image"] = spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(observation_spaces)
        self.metadata = dict(type(self).metadata, render_fps=round(1 / self.control_dt))
        self.resistance = ValveResistance.from_config(self.valve, scale=self.resistance_scale)
        self._episode_started = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = dict(options or {})
        mutable = (
            "valve", "layout", "task", "resistance_scale", "valve_position",
            "base_position", "base_quat", "image_size", "include_image",
            "control_dt", "max_time_s", "close_angle_deg",
            "robot_configuration", "shoulder_height", "tool", "require_endstop",
        )
        unknown = set(options) - set(mutable) - {"initial_robot_qpos", "initial_ctrl"}
        if unknown:
            raise ValueError(f"Unknown reset options: {sorted(unknown)}")
        if any(k in options for k in mutable):
            for key in mutable:
                if key in options:
                    value = options[key]
                    if key in ("valve_position", "base_position", "base_quat", "image_size"):
                        value = tuple(value)
                    setattr(self, key, value)
            self._build()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        lo, hi = (float(v) for v in self.model.jnt_range[self.valve_joint_id])
        self.initial_angle_rad = hi if self.task in ("open", "slow_open") else lo
        self.target_angle_rad = (
            lo if self.task in ("open", "slow_open") else
            min(lo + math.tau, hi) if self.task == "wheel_turn" else hi
        )
        self.data.qpos[self.valve_qpos_index] = self.initial_angle_rad
        if "initial_robot_qpos" in options:
            q = self._finite_vector(options["initial_robot_qpos"], len(self.robot_joint_ids), "initial_robot_qpos")
            bounds = self.model.jnt_range[self.robot_joint_ids]
            if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
                raise ValueError("initial_robot_qpos is outside the official joint ranges")
            self.data.qpos[self.robot_qpos_indices] = q
        if "initial_ctrl" in options:
            action = self._clip_action(options["initial_ctrl"])
            self.data.ctrl[self.actuator_ids] = action
        elif "initial_robot_qpos" in options:
            # Position inputs match the initial pose rather than snapping back
            # toward home. The second finger is coupled by the official model.
            for aid in self.actuator_ids:
                jid = self.model.actuator_trnid[aid, 0]
                if self.model.actuator_trntype[aid] == int(mujoco.mjtTrn.mjTRN_JOINT):
                    self.data.ctrl[aid] = self.data.qpos[self.model.jnt_qposadr[jid]]
        self.resistance.reset(direction=-1 if self.task in ("open", "slow_open") else 1)
        self._resistance_state = self.resistance.apply(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._has_handle_contact = False
        self._has_fixture_contact = False
        self._fixture_peak_N = 0.
        self._has_idle_contact = False
        self._idle_contact_peak_N = 0.
        self._max_valve_speed_rad_s = 0.
        self._hold_time_s = 0.
        self._stop_reaction_Nm = 0.
        self._stop_seen = False
        self._success = False
        self._slow_pause_complete = False
        self._slow_violation = False
        self._slow_pause_time_s = 0.
        self._slow_peak_speed_rad_s = 0.
        self._episode_started = True
        self._last_action_time_s = 0.
        self._last_action = self.data.ctrl[self.actuator_ids].copy()
        self._contacts = self._read_contacts()
        self._has_handle_contact = bool(self._contacts["handle_contact_count"])
        return self._observation(), self._info()

    @staticmethod
    def _finite_vector(value, size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite vector of length {size}")
        return vector

    def _clip_action(self, action) -> np.ndarray:
        vector = self._finite_vector(action, len(self.actuator_ids), "action")
        ranges = self.model.actuator_ctrlrange[self.actuator_ids]
        return np.clip(vector, ranges[:, 0], ranges[:, 1])

    def step(self, action):
        if not self._episode_started:
            raise RuntimeError("Call reset before step")
        command = self._clip_action(action)
        self._last_action_time_s = float(self.data.time)
        self._last_action = command.copy()
        self.data.ctrl[self.actuator_ids] = command
        previous_error = abs(self.target_angle_rad - self.valve_angle_rad)
        for _ in range(self.frame_skip):
            self._resistance_state = self.resistance.apply(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            self._update_slow_open()
            self._max_valve_speed_rad_s = max(self._max_valve_speed_rad_s,abs(self.valve_speed_rad_s))
            self._sample_fixture_contacts()
            self._stop_reaction_Nm = self._actual_stop_reaction()
            if self.task in ("close", "open", "slow_open"):
                near = abs(self.target_angle_rad-self.valve_angle_rad) < math.radians(.5)
                self._stop_seen |= near and self._stop_reaction_Nm > .05
        # Forces are sampled at the last physics solve of this control interval.
        # They are not a peak or an integral over every intermediate contact.
        self._contacts = self._read_contacts()
        self._has_handle_contact |= bool(self._contacts["handle_contact_count"])
        error = abs(self.target_angle_rad - self.valve_angle_rad)
        stable = error < math.radians(2) and abs(self.valve_speed_rad_s) < math.radians(5)
        if self.require_endstop and self.task in ("close", "open", "slow_open"):
            stable &= self._stop_seen and self._stop_reaction_Nm > .05
        slow_ok = self.task != "slow_open" or (
            self._slow_pause_complete and not self._slow_violation
        )
        if stable and self._has_handle_contact and slow_ok and not self._has_idle_contact:
            self._hold_time_s += self.control_dt
        else:
            self._hold_time_s = 0.
        self._success = self._hold_time_s >= .5 - 1e-9
        finite = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        terminated = self._success or not finite
        truncated = float(self.data.time) >= self.max_time_s - 1e-9 and not terminated
        info = self._info()
        info["termination_reason"] = (
            "success" if self._success else "nonfinite_state" if not finite
            else "time_limit" if truncated else None
        )
        reward = previous_error - error + float(self._success)
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    @property
    def valve_angle_rad(self) -> float:
        return float(self.data.qpos[self.valve_qpos_index])

    @property
    def valve_speed_rad_s(self) -> float:
        return float(self.data.qvel[self.valve_dof_index])

    def _update_slow_open(self) -> None:
        if self.task != "slow_open" or self._slow_pause_complete:
            return
        travel = self.initial_angle_rad - self.valve_angle_rad
        speed = abs(self.valve_speed_rad_s)
        self._slow_peak_speed_rad_s = max(self._slow_peak_speed_rad_s, speed)
        if speed > math.radians(10) + 1e-8 or travel > math.radians(10) + 1e-8:
            self._slow_violation = True
        if math.radians(5) <= travel <= math.radians(10) and speed <= math.radians(1):
            self._slow_pause_time_s += self.physics_dt
            if self._slow_pause_time_s >= 1. - 1e-9:
                self._slow_pause_complete = True
        else:
            self._slow_pause_time_s = 0.

    def _sample_fixture_contacts(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a,b = int(c.geom1),int(c.geom2)
            fixture_pair = ((a in self.robot_geom_ids and b in self.fixture_geom_ids)
                            or (b in self.robot_geom_ids and a in self.fixture_geom_ids))
            idle_pair = ((a in self.idle_geom_ids and b in self.idle_forbidden_geom_ids)
                         or (b in self.idle_geom_ids and a in self.idle_forbidden_geom_ids))
            if not (fixture_pair or idle_pair):
                continue
            force=np.zeros(6)
            mujoco.mj_contactForce(self.model,self.data,i,force)
            if fixture_pair:
                self._fixture_peak_N=max(self._fixture_peak_N,float(force[0]))
                self._has_fixture_contact |= bool(force[0]>1.)
            if idle_pair:
                self._idle_contact_peak_N=max(self._idle_contact_peak_N,float(force[0]))
                self._has_idle_contact |= bool(force[0]>1.)

    def _read_contacts(self) -> dict[str, Any]:
        contacts = []
        total_force = np.zeros(3)
        normal_torque = 0.
        axis_origin = self.data.xpos[self.valve_body_id]
        axis = self.data.xmat[self.valve_body_id].reshape(3, 3)[:, 2]
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self.robot_geom_ids and geom2 in self.valve_geom_ids:
                valve_geom, robot_geom, sign = geom2, geom1, 1.
            elif geom2 in self.robot_geom_ids and geom1 in self.valve_geom_ids:
                valve_geom, robot_geom, sign = geom1, geom2, -1.
            else:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if force[0] <= 0:
                continue
            frame = contact.frame.reshape(3, 3)
            # mj_contactForce's positive force acts on geom2. The first
            # contact-frame row is the geom1->geom2 normal; all sums below are
            # forces applied to the valve, in world coordinates (m, N, N m).
            world_force = sign * frame.T @ force[:3]
            world_normal = sign * force[0] * frame[0]
            moment = float(np.dot(np.cross(contact.pos - axis_origin, world_normal), axis))
            total_force += world_force
            normal_torque += moment
            contacts.append({
                "robot_geom": self.model.geom(robot_geom).name,
                "valve_geom": self.model.geom(valve_geom).name,
                "position_world_m": contact.pos.copy().tolist(),
                "force_on_valve_world_N": world_force.tolist(),
                "normal_force_N": float(force[0]),
                "normal_torque_about_valve_axis_Nm": moment,
            })
        return {
            "handle_contact_count": len(contacts),
            "handle_contact_force_on_valve_world_N": total_force.tolist(),
            "handle_normal_torque_Nm": normal_torque,
            "handle_contacts": contacts,
        }

    def _actual_stop_reaction(self) -> float:
        count = self.data.nefc
        selected = ((self.data.efc_type[:count] == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
                    & (self.data.efc_id[:count] == self.valve_joint_id))
        return float(np.sum(np.abs(self.data.efc_force[:count][selected])))

    def _actual_friction_torque(self) -> float:
        count = self.data.nefc
        selected = (
            (self.data.efc_type[:count] == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
            & (self.data.efc_id[:count] == self.valve_dof_index)
        )
        return float(np.sum(self.data.efc_force[:count][selected]))

    def _info(self) -> dict[str, Any]:
        return {
            "time_s": float(self.data.time),
            "action_time_s": self._last_action_time_s,
            "applied_action": self._last_action.copy(),
            "valve_angle_rad": self.valve_angle_rad,
            "valve_angle_deg": math.degrees(self.valve_angle_rad),
            "valve_speed_rad_s": self.valve_speed_rad_s,
            "target_angle_rad": self.target_angle_rad,
            "target_angle_deg": math.degrees(self.target_angle_rad),
            "angle_error_rad": self.target_angle_rad - self.valve_angle_rad,
            "is_success": self._success,
            "mechanical_endstop_seen": self._stop_seen,
            "mechanical_endstop_reaction_Nm": self._stop_reaction_Nm,
            "stable_hold_time_s": self._hold_time_s,
            "has_handle_contact": self._has_handle_contact,
            "has_fixture_contact": self._has_fixture_contact,
            "fixture_peak_normal_N": self._fixture_peak_N,
            "has_idle_contact": self._has_idle_contact,
            "idle_contact_peak_N": self._idle_contact_peak_N,
            "maximum_valve_speed_deg_s": math.degrees(self._max_valve_speed_rad_s),
            "valve_friction_torque_Nm": self._actual_friction_torque(),
            "resistance": dict(self._resistance_state),
            "actuator_force": self.data.actuator_force[self.actuator_ids].copy(),
            "slow_open_pause_complete": self._slow_pause_complete,
            "slow_open_rule_violated": self._slow_violation,
            "slow_open_pause_time_s": self._slow_pause_time_s,
            "slow_open_initial_peak_speed_rad_s": self._slow_peak_speed_rad_s,
            "contact_sample": "last_physics_solve_in_control_interval",
            "contact_sample_time_s": max(0., float(self.data.time) - self.physics_dt),
            **self._contacts,
        }

    def _observation(self) -> dict[str, np.ndarray]:
        observation = {
            "robot_qpos": self.data.qpos[self.robot_qpos_indices].astype(np.float32),
            "robot_qvel": self.data.qvel[self.robot_dof_indices].astype(np.float32),
        }
        if self.include_image:
            observation["image"] = self.render()
        return observation

    def render(self, camera="robot_overview", width=None, height=None):
        width = int(self.image_size[0] if width is None else width)
        height = int(self.image_size[1] if height is None else height)
        if width <= 0 or height <= 0:
            raise ValueError("Render dimensions must be positive")
        if self._renderer is None or self._renderer_size != (width, height):
            self.close()
            self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
            self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_size = (width, height)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_size = None
