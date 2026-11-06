"""Stage and G1 references for orientation, independent from task scoring.

Resting hands have no working-side constraint. Work orientation is compared in
the actual handle frame, so a translated or differently mounted valve does not
inherit an A1/world-height rule. No function advances physics or edits sources.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

POLICY_VERSION = 'g1_stage_aware_orientation_v1'
REST_FILE = Path(__file__).parent / 'configs/reviewed_natural_rest.json'


def orientation_scope(stage, *, active_contact=False):
    """This scopes a direction check; it never switches controller phases."""
    if active_contact:
        return 'work'
    name = str(stage)
    for suffix in ('_candidate', '_actual'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    if name in ('initial', 'approved_standby', 'initial_down_hold', 'final_down_hold'):
        return 'rest'
    if name in ('lift_to_chest', 'lower_to_natural', 'standby_chest_connection',
                'return_chest', 'return_path_ranking', 'release'):
        return 'transition'
    if name == 'chest_ready':
        return 'preparation'
    # Unknown/working phases must not silently lose the direction check.
    return 'work'


def active_tool_handle_contact(env, data):
    """Native tool/handle contact, including zero-force geometric touching.

    Positive-gap native margin contacts count only when they exert force.
    No new force threshold, collision exclusion or success criterion is added.
    """
    tool = set(env.active_tool_geom_ids)
    handle = set(env.valve_geom_ids)
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        a, b = int(contact.geom1), int(contact.geom2)
        if not ((a in tool and b in handle) or (b in tool and a in handle)):
            continue
        if float(contact.dist) <= 0.:
            return True
        if int(contact.efc_address) >= 0:
            mujoco.mj_contactForce(env.model, data, index, force)
            if np.linalg.norm(force[:3]) > 0.:
                return True
    return False


def apply_reviewed_rest(poses, *, side, path=REST_FILE):
    """Copy user arm-only rest into standby and the idle arm, never a valve.

    Active working poses remain unchanged. The controller's final return uses
    the same standby, rather than zeroing joints at the end of a timer.
    """
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if document.get('schema') != 'reviewed_bimanual_natural_rest_v1':
        raise ValueError('Need the user-reviewed bimanual natural rest data')
    if side not in ('left', 'right') or not poses:
        raise ValueError('Need an operating hand and complete G1 poses')
    copied = copy.deepcopy(poses)
    idle = 'right' if side == 'left' else 'left'
    values = {}
    for hand in ('left', 'right'):
        entry = document['arms'][hand]
        q = np.asarray(entry['joint_angles_rad'], dtype=float)
        expected = [f'openarm_{hand}_joint{i}' for i in range(1, 8)]
        if entry['joint_names'] != expected or q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Reviewed rest has invalid arm joints')
        values[hand] = dict(zip(expected, q.tolist()))
    for index, row in enumerate(copied):
        names = row['qpos_joint_names']
        q = np.asarray(row['qpos'], dtype=float).copy()
        for hand in ((side, idle) if index == 0 else (idle,)):
            for name, value in values[hand].items():
                q[names.index(name)] = value
        row['qpos'] = q.tolist()
        row['shared_rest_source'] = str(Path(path).resolve())
    return copied


class G1OrientationReferences:
    """Recover actual S-tool rotations from the named saved G1 joints."""
    def __init__(self, env, poses, tool_body_name):
        self.env = env
        body_id = env.model.body(tool_body_name).id
        scratch = mujoco.MjData(env.model)
        self.preparation_rotation = None
        samples = {}
        for index in (1, 3, 4, 5):
            row = poses[index]
            q = np.asarray(row['qpos'], dtype=float)
            if q.shape != (env.model.nq,) or not np.isfinite(q).all():
                raise ValueError('G1 orientation reference has invalid saved joints')
            scratch.qpos[:] = q
            mujoco.mj_kinematics(env.model, scratch)
            rotation = scratch.xmat[body_id].reshape(3, 3).copy()
            if index == 1:
                self.preparation_rotation = rotation
            else:
                angle = float(row['opening_deg'])
                frame = self.handle_frame(float(q[env.valve_qpos_index]))
                samples[angle] = frame.T @ rotation
        self.angles = np.asarray(sorted(samples), dtype=float)
        if len(self.angles) < 2:
            raise ValueError('Working G1 needs rotations at distinct valve angles')
        self._relative = Slerp(self.angles, Rotation.from_matrix(
            np.asarray([samples[angle] for angle in self.angles])))

    def handle_frame(self, valve_q):
        radial = self.env.station_spec.radial_at(valve_q)
        stem = self.env.stem_axis_world
        return np.column_stack((radial, np.cross(stem, radial), stem))

    def at(self, opening_deg, valve_q, *, scope='work'):
        if scope == 'preparation':
            return self.preparation_rotation.copy()
        angle = float(np.clip(opening_deg, self.angles[0], self.angles[-1]))
        relative = self._relative([angle]).as_matrix()[0]
        return self.handle_frame(valve_q) @ relative
