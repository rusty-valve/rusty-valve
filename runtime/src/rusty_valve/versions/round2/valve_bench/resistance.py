"""Passive, angle-dependent valve resistance for the WP1 scenes.

Call ``apply(model, data)`` once before each ``mujoco.mj_step``.  This module
changes only the named valve joint's dry-friction constraint and its solver
regularization.  MuJoCo decides the actual resisting torque, including static
friction below the limit.  Nothing here prescribes a valve angle, moves a body,
adds a robot force, or supplies positive mechanical work.

The ball-valve envelope follows Bray TB1005's typical break/run/end ratios.
Transition angles and friction-state thresholds remain task assumptions, not
an identified, angle-by-angle curve for a particular physical valve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).with_name("configs") / "resistance.json"


def _smoothstep(value: float) -> float:
    x = min(1.0, max(0.0, value))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class ValveResistance:
    """One independent friction state per simulated valve episode.

    Angle zero means open; increasing angle closes the valve.  Angle and speed
    use radians and radians/second.  ``scale`` changes the valve's load, not a
    robot's motor limits.  Use a separate model per environment because the
    current friction limit is stored in ``MjModel``.
    """

    valve: str
    breakaway_Nm: float
    running_ratio: float
    closing_end_ratio: float
    opening_start_fraction: float
    closing_start_fraction: float
    closing_end_fraction: float
    scale: float = 1.0
    motion_enter_rad_s: float = 0.02
    rest_enter_rad_s: float = 0.003
    rest_dwell_s: float = 0.10
    joint_name: str = "valve_hinge"
    solref: tuple[float, float] = (0.002, 1.0)
    solimp: tuple[float, float, float, float, float] = (
        0.9999, 0.9999, 0.001, 0.5, 2.0
    )
    # None preserves profiles without a final rise when opening (e.g. globe).
    opening_end_ratio: float | None = None
    opening_end_fraction: float = 0.22
    _model_identity: int | None = field(default=None, init=False, repr=False)
    _joint_id: int = field(default=-1, init=False, repr=False)
    _dof: int = field(default=-1, init=False, repr=False)
    _qadr: int = field(default=-1, init=False, repr=False)
    _moving: bool = field(default=False, init=False, repr=False)
    _direction: int = field(default=1, init=False, repr=False)
    _below_since: float | None = field(default=None, init=False, repr=False)
    _last_time: float | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        valve: str = "ball",
        scale: float = 1.0,
        path: str | Path | None = None,
        **overrides: Any,
    ) -> "ValveResistance":
        """Read one declared profile; metadata stays in the JSON source file."""
        content = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
        if valve not in content["profiles"]:
            raise ValueError(f"Unknown valve profile: {valve!r}")
        values = dict(content["common"])
        values.update(content["profiles"][valve]["parameters"])
        values.update(overrides)
        values["valve"] = valve
        values["scale"] = scale
        for key in ("solref", "solimp"):
            if key in values:
                values[key] = tuple(values[key])
        instance = cls(**values)
        instance._validate()
        return instance

    def _validate(self) -> None:
        if not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError("Resistance scale must be finite and nonnegative.")
        if not math.isfinite(self.breakaway_Nm) or self.breakaway_Nm <= 0:
            raise ValueError("breakaway_Nm must be positive and finite.")
        if not 0 <= self.running_ratio <= self.closing_end_ratio <= 1:
            raise ValueError("Require 0 <= running_ratio <= closing_end_ratio <= 1.")
        if self.opening_end_ratio is not None and not (
            self.running_ratio <= self.opening_end_ratio <= 1
        ):
            raise ValueError("Require running_ratio <= opening_end_ratio <= 1.")
        for name in (
            "opening_start_fraction", "opening_end_fraction",
            "closing_start_fraction", "closing_end_fraction"
        ):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between zero and one.")
        if not 0 <= self.rest_enter_rad_s < self.motion_enter_rad_s:
            raise ValueError("Rest/motion speed thresholds must provide hysteresis.")
        if self.rest_dwell_s < 0:
            raise ValueError("rest_dwell_s must be nonnegative.")

    def reset(self, direction: int = 1) -> None:
        """Reset friction history when resetting/reusing an episode.

        ``direction`` is just the initial opening/closing envelope hint.  Once
        actual motion occurs, its direction is taken from measured velocity.
        """
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 (opening) or +1 (closing).")
        self._moving = False
        self._direction = direction
        self._below_since = None
        self._last_time = None

    def moving_limit(self, normalized_angle: float, direction: int) -> float:
        """Assumed sliding-friction bound, not a simulated/measured torque.

        The bound is deliberately nonnegative.  It is fed to MuJoCo's passive
        friction constraint rather than applied as a hand-written signed force,
        so passing through zero speed cannot turn it into a drive torque.
        """
        x = min(1.0, max(0.0, normalized_angle))
        run = self.running_ratio
        if direction < 0:
            # x runs from 1 (closed) to 0 (open). Bray's typical ball-valve
            # envelope has a final rise in both directions; legacy profiles
            # may explicitly omit this opening-end rise.
            u = (1.0 - x) / self.opening_start_fraction
            ratio = run + (1.0 - run) * math.exp(-u * u)
            if self.opening_end_ratio is not None:
                rise = _smoothstep(
                    (self.opening_end_fraction - x) / self.opening_end_fraction
                )
                ratio += (self.opening_end_ratio - run) * rise
        else:
            # Closing: initial peak, running resistance, then final seating rise.
            u = x / self.closing_start_fraction
            ratio = run + (1.0 - run) * math.exp(-u * u)
            rise = _smoothstep(
                (x - (1.0 - self.closing_end_fraction))
                / self.closing_end_fraction
            )
            ratio += (self.closing_end_ratio - run) * rise
        # No angle gets a bound above the separately declared breakaway limit.
        return self.scale * self.breakaway_Nm * min(1.0, max(0.0, ratio))

    def _bind(self, model: Any) -> None:
        if self._model_identity == id(model):
            return
        joint = model.joint(self.joint_name)
        self._joint_id = int(joint.id)
        self._dof = int(joint.dofadr[0])
        self._qadr = int(joint.qposadr[0])
        lo, hi = (float(v) for v in model.jnt_range[self._joint_id])
        if not hi > lo:
            raise ValueError("Valve hinge must have a nonempty angular range.")
        # Explicitly avoid the old inherited soft friction-constraint setting.
        model.dof_solref[self._dof] = self.solref
        model.dof_solimp[self._dof] = self.solimp
        self._model_identity = id(model)
        self.reset(self._direction)

    def apply(self, model: Any, data: Any) -> dict[str, float | int | str]:
        """Update the passive bound; return settings useful for the run log.

        Read ``data.qfrc_constraint`` after the physical step to obtain the actual
        solver torque.  This returned bound is configuration, not that torque.
        Near a joint stop, qfrc_constraint can also include a limit reaction, so
        a diagnostic should identify friction constraints or exclude stop loads.
        """
        self._bind(model)
        time = float(data.time)
        if self._last_time is not None and time < self._last_time:
            self.reset(self._direction)
        q = float(data.qpos[self._qadr])
        speed = float(data.qvel[self._dof])
        abs_speed = abs(speed)
        lo, hi = (float(v) for v in model.jnt_range[self._joint_id])
        fraction = min(1.0, max(0.0, (q - lo) / (hi - lo)))

        if abs_speed >= self.motion_enter_rad_s:
            self._moving = True
            self._direction = 1 if speed > 0 else -1
            self._below_since = None
        elif self._moving:
            if abs_speed <= self.rest_enter_rad_s:
                if self._below_since is None:
                    self._below_since = time
                if time - self._below_since >= self.rest_dwell_s:
                    self._moving = False
                    self._below_since = None
            else:
                self._below_since = None
                # Preserve the direction through a brief stop, but adopt a real
                # reversal once its velocity exceeds the lower noise threshold.
                self._direction = 1 if speed > 0 else -1

        sliding = self.moving_limit(fraction, self._direction)
        static = self.scale * self.breakaway_Nm
        limit = sliding if self._moving else static
        # Static and sliding resistance are alternative bounds, not two loads
        # added together.  The constraint supplies only the torque actually
        # needed, up to this bound, to resist the current motion/attempted motion.
        model.dof_frictionloss[self._dof] = limit
        self._last_time = time
        return {
            "friction_limit_Nm": limit,
            "static_breakaway_limit_Nm": static,
            "moving_limit_Nm": sliding,
            "mode": "moving" if self._moving else "static",
            "direction": "closing" if self._direction > 0 else "opening",
            "direction_sign": self._direction,
            "normalized_angle": fraction,
            "angle_rad": q,
            "speed_rad_s": speed,
            "scale": self.scale,
        }
