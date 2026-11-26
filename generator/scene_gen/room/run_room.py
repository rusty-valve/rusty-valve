"""Open a room in MuJoCo and apply the opening and closing resistance of each valve."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_profiles(path, operator):
    parameters = read_json(path)
    resistance = parameters['resistance']
    if operator == 'wheel':
        transmission = parameters['motion']['gear_ratio'] * resistance['efficiency']
        keys = ('stem_open', 'stem_close')
    elif operator == 'lever':
        transmission = 1.
        keys = ('open', 'close')
    else:
        raise ValueError(f'{path}: unsupported valve operator {operator}')
    if not math.isfinite(transmission) or transmission <= 0.:
        raise ValueError(f'{path}: invalid valve transmission')
    profiles = []
    for key in keys:
        curve = np.asarray(resistance[key], dtype=float)
        if (curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2
                or not np.isfinite(curve).all() or curve[0, 0] != 0.
                or not np.all(np.diff(curve[:, 0]) > 0.) or np.any(curve[:, 1] <= 0.)):
            raise ValueError(f'{path}: invalid resistance curve {key}')
        profiles.append((curve[:, 0], curve[:, 1] / transmission))
    open_angles, open_torque = profiles[0]
    # Opening breaks free and stays easy. Closing retains the seating rise.
    profiles[0] = (open_angles, np.minimum.accumulate(open_torque))
    return tuple(profiles)


@dataclass
class ValveResistance:
    dof: int
    stem_position: int
    stem_open: float
    stem_close: float
    close_sign: float
    profiles: tuple
    breakaway: float
    direction: int | None = None
    still_time: float = 0.

    @classmethod
    def bind(cls, model, task, profiles):
        operator = task['operator_motion']
        stem = task['valve_stem_motion']
        operator_joint = model.joint(operator['joint'])
        stem_joint = model.joint(stem['joint'])
        if (operator_joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE
                or stem_joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"{task['id']}: valve joints must be hinges")
        operator_travel = operator['closed_qpos_rad'] - operator['open_qpos_rad']
        stem_travel = abs(math.degrees(stem['closed_qpos_rad'] - stem['open_qpos_rad']))
        if (not math.isfinite(operator_travel) or operator_travel == 0.
                or stem_travel == 0. or not all(
                    math.isclose(angles[-1], stem_travel, abs_tol=1.e-5)
                    for angles, _ in profiles)):
            raise ValueError(f"{task['id']}: resistance curve and valve travel disagree")
        return cls(int(operator_joint.dofadr[0]), int(stem_joint.qposadr[0]),
                   stem['open_qpos_rad'], stem['closed_qpos_rad'], math.copysign(1., operator_travel),
                   profiles, float(max(torque[0] for _, torque in profiles)))

    def reset(self):
        self.direction = None
        self.still_time = 0.

    def bound(self, qpos, qvel, timestep):
        speed = float(qvel[self.dof]) * self.close_sign
        if abs(speed) >= .02:
            self.direction = int(speed > 0.)
            self.still_time = 0.
        elif self.direction is not None:
            self.still_time = self.still_time + timestep if abs(speed) <= .003 else 0.
            if self.still_time >= .10:
                self.reset()
        if self.direction is None:
            return self.breakaway
        origin = self.stem_close if self.direction == 0 else self.stem_open
        travel_sign = math.copysign(1., self.stem_close - self.stem_open)
        progress = math.degrees(qpos[self.stem_position] - origin) * travel_sign
        if self.direction == 0:
            progress = -progress
        angles, torque = self.profiles[self.direction]
        return float(np.interp(progress, angles, torque))


class RoomSimulation:
    """Own one model/data pair. Use step() for every integration step."""

    def __init__(self, folder, model, data):
        folder = Path(folder).resolve()
        self.model = model
        self.data = data
        self.tasks = read_json(folder / 'task_index.json')['valves']
        self.valves = {}
        profiles_by_package = {}
        bound_dofs = set()
        for task in self.tasks:
            relative_model = Path(task['package_model'])
            path = (folder / relative_model).with_name('params.json').resolve()
            if relative_model.is_absolute() or not path.is_relative_to(folder):
                raise ValueError(f"{task['id']}: resistance parameters must be inside the room package")
            key = (path, task['operator'])
            if key not in profiles_by_package:
                profiles_by_package[key] = load_profiles(*key)
            valve = ValveResistance.bind(model, task, profiles_by_package[key])
            if task['id'] in self.valves or valve.dof in bound_dofs:
                raise ValueError(f"{task['id']}: duplicate valve ID or operator joint")
            if not math.isclose(model.dof_frictionloss[valve.dof], valve.breakaway, rel_tol=.01):
                raise ValueError(f"{task['id']}: packaged valve and breakaway torque disagree")
            self.valves[task['id']] = valve
            bound_dofs.add(valve.dof)
        self.reset_resistance()

    def reset_resistance(self):
        for valve in self.valves.values():
            valve.reset()
            self.model.dof_frictionloss[valve.dof] = valve.breakaway

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.reset_resistance()
        mujoco.mj_forward(self.model, self.data)

    def step(self, nstep=1):
        model, data = self.model, self.data
        for _ in range(nstep):
            for valve in self.valves.values():
                model.dof_frictionloss[valve.dof] = valve.bound(data.qpos, data.qvel, model.opt.timestep)
            mujoco.mj_step(model, data)


def load_room(folder=None):
    if folder is None:
        folder = Path(__file__).resolve().parent
    folder = Path(folder).resolve()
    model = mujoco.MjModel.from_xml_path(str(folder / 'room.xml'))
    simulation = RoomSimulation(folder, model, mujoco.MjData(model))
    mujoco.mj_forward(model, simulation.data)
    return simulation


def main(argv=None, *, default_room=Path(__file__).resolve().parent):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--room', type=Path, default=default_room, required=default_room is None,
                        help='Packaged room directory')
    parser.add_argument('--steps', type=int, help='Run this many steps without a window')
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 0:
        parser.error('--steps must be nonnegative')
    simulation = load_room(args.room)
    model, data = simulation.model, simulation.data
    if args.steps is not None:
        simulation.step(args.steps)
        print(json.dumps({'valves_with_curves': len(simulation.valves), 'time_s': float(data.time)}))
        return
    from mujoco import viewer
    with viewer.launch_passive(model, data) as window:
        while window.is_running():
            start = time.monotonic()
            with window.lock():
                simulation.step()
            window.sync()
            time.sleep(max(0., model.opt.timestep - (time.monotonic() - start)))


if __name__ == '__main__':
    main()
