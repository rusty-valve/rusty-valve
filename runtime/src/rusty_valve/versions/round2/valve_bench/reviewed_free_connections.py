"""Execute the reviewed free-arm endpoint curves, not an old lift waypoint.

The saved samples are review evidence.  They must describe two all-active-joint
quintic curves, and execution evaluates those same curves analytically.  The
lowering curve has its own approved starting pose; ``reverse=True`` does not
reverse the lifting curve.  These are input-consistency checks, not score rules.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np


POLICY_VERSION = 'reviewed_joint_endpoint_quintic_v1'
_DATA_ATOL = 1e-8
_NATIVE_ATOL = 1e-7


def _quintic(fraction):
    return fraction**3 * (10. + fraction * (-15. + 6. * fraction))


def _same(actual, expected, context):
    if not np.allclose(actual, expected, rtol=0., atol=_DATA_ATOL):
        raise ValueError(context)


def _source_path(value, directory):
    if not isinstance(value, str) or not value:
        raise ValueError('Reviewed connection needs a source sequence file')
    path = Path(value)
    return (path if path.is_absolute() else directory / path).resolve()


class ReviewedFreeConnections:
    """Two reviewed seven-joint curves bound to one model and G1 source."""

    def __init__(self, path, env, poses):
        self.path = Path(path).resolve()
        document = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(document, dict):
            raise ValueError('Reviewed connection file must be an object')
        if document.get('approval', {}).get('user_confirmed') is not True:
            raise ValueError('Reviewed free connections need explicit user confirmation')
        adapter = env.robot_adapter.to_dict()
        for field in ('model_revision', 'robot_adapter_id'):
            value = document.get(field)
            model_value = adapter.get('adapter_id' if field == 'robot_adapter_id' else field)
            if not isinstance(value, str) or not value or value != model_value:
                raise ValueError(f'Reviewed connection {field} differs from the robot model')
        self.model_revision = document['model_revision']
        self.robot_adapter_id = document['robot_adapter_id']
        self.source_sequence = _source_path(document.get('source_sequence_file'), self.path.parent)

        model = env.model
        names = [model.joint(i).name for i in range(model.njnt)]
        if (document.get('qpos_joint_names') != names or len(names) != model.nq
                or any(int(kind) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                         int(mujoco.mjtJoint.mjJNT_SLIDE))
                       for kind in model.jnt_type)):
            raise ValueError('Reviewed connection joint order differs from the scalar-joint model')
        active_names = list(env.active_joint_names)
        self.active = np.asarray(env.active_qpos_indices, dtype=int).copy()
        if (len(active_names) != 7 or self.active.shape != (7,)
                or len(set(active_names)) != 7 or len(set(self.active.tolist())) != 7
                or any(name not in names for name in active_names)
                or not np.array_equal(self.active, [names.index(name) for name in active_names])):
            raise ValueError('Reviewed connection active seven-joint binding is inconsistent')
        idle = np.asarray([names.index(name) for name in env.idle_joint_names], dtype=int)
        inactive = np.asarray([i for i in range(model.nq) if i not in self.active], dtype=int)

        if len(poses) != 8:
            raise ValueError('Reviewed connection requires the eight resolved G1 roles')
        pose_q = []
        for pose in poses:
            if pose.get('qpos_joint_names') != names:
                raise ValueError('Reviewed connection G1 joint order differs')
            q = np.asarray(pose.get('qpos', pose.get('full_qpos')), dtype=float)
            if q.shape != (model.nq,) or not np.isfinite(q).all():
                raise ValueError('Reviewed connection G1 pose has invalid joints')
            if _source_path(pose.get('reviewed_sequence_file'), self.path.parent) != self.source_sequence:
                raise ValueError('Reviewed connection and G1 use different reviewed sequences')
            if pose.get('model_revision', self.model_revision) != self.model_revision:
                raise ValueError('Reviewed connection and G1 use different model revisions')
            pose_q.append(q)

        samples = document.get('samples')
        if not isinstance(samples, list) or len(samples) != 202:
            raise ValueError('Reviewed connection needs 101 lift and 101 lower samples')
        sections = {'lift': [], 'lower': []}
        for row in samples:
            if not isinstance(row, dict) or row.get('section') not in sections:
                raise ValueError('Reviewed connection sample has an unknown section')
            fraction = row.get('fraction')
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not np.isfinite(fraction):
                raise ValueError('Reviewed connection sample has an invalid fraction')
            q = np.asarray(row.get('qpos'), dtype=float)
            if q.shape != (model.nq,) or not np.isfinite(q).all():
                raise ValueError('Reviewed connection sample has invalid joints')
            sections[row['section']].append((float(fraction), q))

        # Position actuator ranges are real robot limits, not an added margin.
        control = adapter.get('control', {})
        actuator_names = control.get('actuator_names', [])
        controlled_names = control.get('controlled_joint_names', [])
        if len(actuator_names) != len(controlled_names):
            raise ValueError('Reviewed connection actuator bindings are inconsistent')
        actuator_checks = []
        for actuator_name, joint_name in zip(actuator_names, controlled_names):
            if joint_name is None:
                continue
            if joint_name not in names:
                raise ValueError('Reviewed connection controlled joint is absent')
            aid = model.actuator(actuator_name).id
            if int(model.actuator_trnid[aid, 0]) != model.joint(joint_name).id:
                raise ValueError('Reviewed connection actuator drives another joint')
            if model.actuator_ctrllimited[aid]:
                actuator_checks.append((names.index(joint_name), model.actuator_ctrlrange[aid].copy()))

        self.endpoints = {}
        for section, rows in sections.items():
            if len(rows) != 101:
                raise ValueError('Reviewed connection section needs 101 samples')
            fractions = np.asarray([row[0] for row in rows])
            _same(fractions, np.linspace(0., 1., 101), 'Reviewed connection fractions are incomplete or out of order')
            q = np.asarray([row[1] for row in rows])
            expected_start = pose_q[0] if section == 'lift' else pose_q[7]
            expected_end = pose_q[1] if section == 'lift' else pose_q[0].copy()
            if section == 'lower':
                # The valve remains at its final angle while the arm lowers.
                expected_end[inactive] = expected_start[inactive]
            _same(q[0], expected_start, f'Reviewed {section} start differs from its G1 role')
            _same(q[-1], expected_end, f'Reviewed {section} end differs from its G1 role')
            _same(q[:, inactive], np.broadcast_to(q[0, inactive], (101, len(inactive))),
                  f'Reviewed {section} moves an inactive joint or the valve')
            _same(q[:, idle], np.broadcast_to(pose_q[0][idle], (101, len(idle))),
                  'Reviewed connection moves the idle arm')
            weights = _quintic(fractions)[:, None]
            expected = (1.-weights) * q[0, self.active] + weights * q[-1, self.active]
            _same(q[:, self.active], expected, f'Reviewed {section} is not the approved all-joint quintic curve')
            for jid in np.flatnonzero(model.jnt_limited):
                address = int(model.jnt_qposadr[jid])
                lower, upper = model.jnt_range[jid]
                if np.any(q[:, address] < lower-_NATIVE_ATOL) or np.any(q[:, address] > upper+_NATIVE_ATOL):
                    raise ValueError('Reviewed connection exceeds a native joint limit')
            for address, bounds in actuator_checks:
                if np.any(q[:, address] < bounds[0]-_NATIVE_ATOL) or np.any(q[:, address] > bounds[1]+_NATIVE_ATOL):
                    raise ValueError('Reviewed connection exceeds a native actuator range')
            self.endpoints[section] = (q[0, self.active].copy(), q[-1, self.active].copy())

        self.summary = dict(policy=POLICY_VERSION, file=str(self.path),
            source_sequence_file=str(self.source_sequence), model_revision=self.model_revision,
            robot_adapter_id=self.robot_adapter_id, active_joint_names=active_names,
            samples_per_section=101, curve='all_seven_joints_endpoint_quintic',
            lower_policy='independent_approved_lower_endpoints_not_reversed_lift',
            endpoints={section: dict(start_rad=ends[0].tolist(), end_rad=ends[1].tolist())
                       for section, ends in self.endpoints.items()},
            idle_arm_unchanged=True, valve_unchanged_within_each_section=True,
            native_joint_and_actuator_ranges_respected=True, physics_steps_run=0,
            score_changed=False)

    def evaluate(self, fraction, reverse=False):
        """Return active targets; ``reverse=True`` means forward through lower."""
        fraction = float(fraction)
        if not np.isfinite(fraction):
            raise ValueError('Reviewed connection fraction must be finite')
        start, end = self.endpoints['lower' if reverse else 'lift']
        if fraction <= 0.:
            return start.copy()
        if fraction >= 1.:
            return end.copy()
        weight = _quintic(fraction)
        return (1.-weight) * start + weight * end
