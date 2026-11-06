"""Read-only, restorable state samples from a physical script execution.

Attach after ProcedureRecorder: its every-physics-step callback runs first and
is left intact.  This recorder never calls mj_step, changes controls, or assigns
simulation state.  State vectors use MuJoCo's public mjSTATE_INTEGRATION API;
the passive-resistance history lives outside MjData and is saved separately.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np


class ScriptStateRecorder:
    """Sample a reset SixStationEnv-compatible instance, normally at 50 Hz.

    ``ScriptStateRecorder(env, sample_hz=50.)`` attaches immediately.  Call
    ``write(output_directory)`` before closing the environment; it includes the
    current final state even when it lies between regular sample times.  Call
    ``close()`` to restore the previous physical observer.  Output filenames are
    ``full_states.npz`` and ``full_states.json``; existing files are not replaced.
    """

    def __init__(self, env, sample_hz=50.):
        if not math.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError('sample_hz must be finite and positive')
        if not getattr(env, '_started', False):
            raise ValueError('Reset the environment before attaching the state recorder')
        self.env = env
        self.sample_hz = float(sample_hz)
        self.sample_period_s = 1. / self.sample_hz
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_size = int(mujoco.mj_stateSize(env.model, self.state_spec))
        self._profiles = tuple(sorted(env.resistances))
        self._profile_dofs = np.asarray([
            int(env.model.joint(env.resistances[name].joint_name).dofadr[0])
            for name in self._profiles], dtype=np.int32)
        self._previous_observer = None
        self._attached = False
        self._samples = []
        self._next_sample_s = float(env.data.time)
        self.attach()
        self._capture()

    def attach(self):
        """Append this read-only observer; preserve any existing scorer callback."""
        if self._attached:
            if self.env.physics_observer != self.observe:
                raise RuntimeError('Another caller replaced the attached state observer')
            return self
        self._previous_observer = self.env.physics_observer
        if self._previous_observer == self.observe:
            raise RuntimeError('Cannot chain the state recorder to itself')
        self.env.physics_observer = self.observe
        self._attached = True
        return self

    def observe(self, env):
        """Called by the existing runner after a physical solve (or reset)."""
        if env is not self.env:
            raise ValueError('State recorder received a different environment')
        if self._previous_observer is not None:
            self._previous_observer(env)
        current = float(env.data.time)
        if current == 0.:
            self._samples.clear()
            self._next_sample_s = 0.
        if self._samples and current < self._samples[-1]['time_s']:
            raise ValueError('Physical state times moved backwards without reset')
        if current >= self._next_sample_s - 1e-9:
            self._capture()

    def _capture(self):
        env = self.env
        current = float(env.data.time)
        if self._samples and current == self._samples[-1]['time_s']:
            return
        if self._samples and current < self._samples[-1]['time_s']:
            raise ValueError('Final state precedes the recorded states')
        integration = np.empty(self.state_size, dtype=env.data.qpos.dtype)
        mujoco.mj_getState(env.model, env.data, integration, self.state_spec)
        profiles = [env.resistances[name] for name in self._profiles]
        self._samples.append(dict(
            time_s=current,
            integration_state=integration,
            qpos=env.data.qpos.copy(),
            qvel=env.data.qvel.copy(),
            ctrl=env.data.ctrl.copy(),
            friction_moving=np.asarray([p._moving for p in profiles], dtype=bool),
            friction_direction=np.asarray([p._direction for p in profiles], dtype=np.int8),
            friction_below_since_s=np.asarray([
                -1. if p._below_since is None else p._below_since for p in profiles], dtype=float),
            friction_last_time_s=np.asarray([
                -1. if p._last_time is None else p._last_time for p in profiles], dtype=float),
            frictionloss_Nm=env.model.dof_frictionloss[self._profile_dofs].copy(),
        ))
        # Preserve the original sampling grid after a forced final sample.
        while self._next_sample_s <= current + 1e-9:
            self._next_sample_s += self.sample_period_s

    def write(self, output):
        """Write current samples plus the final state; return artifact paths."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        npz_path, json_path = output/'full_states.npz', output/'full_states.json'
        if npz_path.exists() or json_path.exists():
            raise FileExistsError('Full-state recording already exists in output directory')
        self._capture()
        arrays = {key: np.asarray([row[key] for row in self._samples])
                  for key in self._samples[0]}
        model = self.env.model
        # Body offsets are MjModel fields, not part of mjSTATE_INTEGRATION.
        # Include them so randomized valve placements can be restored too.
        arrays['model_body_pos'] = model.body_pos.copy()
        arrays['model_body_quat'] = model.body_quat.copy()
        metadata = dict(
            schema_version=1,
            data_kind='recorded_simulation',
            source='Read-only samples from new MuJoCo physical script execution',
            mujoco_version=mujoco.__version__,
            sampling=dict(requested_hz=self.sample_hz,
                          physics_dt_s=float(model.opt.timestep),
                          frames=len(self._samples),
                          first_time_s=float(arrays['time_s'][0]),
                          last_time_s=float(arrays['time_s'][-1]),
                          first_and_final_state_preserved=True,
                          note='The first physical callback at/after each sample time is retained; no interpolation.'),
            state=dict(api='mj_getState / mj_setState',
                       specification='mjSTATE_INTEGRATION',
                       specification_value=int(self.state_spec),
                       size=self.state_size,
                       note='Complete MuJoCo integration state in official field order; explicit qpos/qvel/ctrl are convenience copies.'),
            model=dict(nq=int(model.nq), nv=int(model.nv), na=int(model.na),
                       nu=int(model.nu), njnt=int(model.njnt), nbody=int(model.nbody),
                       solver=int(model.opt.solver), iterations=int(model.opt.iterations),
                       ls_iterations=int(model.opt.ls_iterations),
                       physics_dt_s=float(model.opt.timestep),
                       joint_names=[model.joint(i).name for i in range(model.njnt)],
                       joint_qpos_addresses=model.jnt_qposadr.tolist(),
                       joint_dof_addresses=model.jnt_dofadr.tolist(),
                       actuator_names=[model.actuator(i).name for i in range(model.nu)],
                       body_names=[model.body(i).name for i in range(model.nbody)],
                       body_transforms='model_body_pos/model_body_quat are complete current local body transforms; includes episode position randomization.',
                       source_requirement='Use the matching episode scene.xml and model topology; do not apply this state vector to another model or assume cross-version layout compatibility.'),
            passive_resistance=dict(
                names=list(self._profiles),
                joint_names=[self.env.resistances[n].joint_name for n in self._profiles],
                dof_indices=self._profile_dofs.tolist(),
                direction_sign='1 closing, -1 opening',
                missing_time_sentinel=-1.,
                history_fields={'friction_moving': '_moving',
                                'friction_direction': '_direction',
                                'friction_below_since_s': '_below_since',
                                'friction_last_time_s': '_last_time'},
                note='Restore each resistance object history and the matching model dof_frictionloss after binding the profile to the restored model; missing time sentinel means None. The next apply() continues the normal passive-friction update.'),
            arrays={name: dict(shape=list(value.shape), dtype=str(value.dtype),
                               axis_0='model body' if name.startswith('model_body_') else 'sample')
                    for name, value in arrays.items()},
            scoring='ProcedureRecorder retains its separate, unthinned every-physics-step record; this file does not replace rule evidence.',
            limitations=['Physical/controller states outside MuJoCo and the passive-resistance profiles are not implicitly captured. To resume the teacher controller itself, also save its integral, phase, reference or source-command clock.'],
        )
        with npz_path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        with json_path.open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        return dict(states=str(npz_path), metadata=str(json_path), frames=len(self._samples))

    def close(self):
        """Remove this observer while restoring the previous scoring observer."""
        if self._attached:
            if self.env.physics_observer != self.observe:
                raise RuntimeError('Cannot detach: another caller replaced the observer')
            self.env.physics_observer = self._previous_observer
            self._attached = False

