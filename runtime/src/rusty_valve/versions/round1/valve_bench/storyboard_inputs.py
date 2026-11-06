"""Read the formal workstation and G1 data without changing motion behavior.

This module does not import a simulator or an IK solver. Provenance paths are
metadata only: execution inputs come exclusively from the two formal files.
Approval is a caller/job decision; historical approval flags are preserved.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


CONTROLLER_ROLES = ('standby', 'chest_ready', 'precontact', 'contact',
                    'mid_turn', 'end', 'release', 'chest_return')
_METHOD_FIELDS = ('mode', 'radius_m', 'pitch_deg', 'tilt_deg', 'roll_deg',
                  'push_contact_geometry')


class StoryboardInputError(ValueError):
    """Formal input is missing or internally inconsistent."""


class IncompleteStoryboardError(StoryboardInputError):
    def __init__(self, station: str, operation: str, missing_roles):
        self.station = station
        self.operation = operation
        self.missing_roles = tuple(missing_roles)
        super().__init__(f'{station}/{operation}: missing G1 roles: '
                         + ', '.join(self.missing_roles)
                         + '; no replacement pose was generated')


def _read(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoryboardInputError(f'Cannot read formal input {path}: {exc}') from exc
    if not isinstance(document, dict):
        raise StoryboardInputError(f'Formal input must be an object: {path}')
    return document


def _finite_number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StoryboardInputError(f'{context} must be a finite number')
    return value


def _controller_pose(pose: dict, role: str) -> dict:
    """Add legacy names to a copy; never interpolate or replace saved joints."""
    row = copy.deepcopy(pose)
    actual_opening = _finite_number(row.get('opening_deg'), f'{role}.opening_deg')
    differences = row.get('source_target_differences', {})
    if not isinstance(differences, dict):
        raise StoryboardInputError(f'{role}.source_target_differences must be an object')
    # V1 stored the measured angle plus (original target - measured angle).
    # Recover that original label so this data-only migration does not change
    # the old controller's reference interpolation endpoints.
    source_delta = _finite_number(differences.get('stored_opening_vs_joint_deg', 0.),
                                 f'{role}.stored_opening_vs_joint_deg')
    source_opening = actual_opening + source_delta
    row.update(qpos=copy.deepcopy(row['full_qpos']), opening_deg=source_opening,
               actual_opening_deg=actual_opening,
               source_opening_deg=source_opening, storyboard_role=role,
               opening_label_policy=('original_source_label_recovered_from_saved_delta'
                                     if 'stored_opening_vs_joint_deg' in differences
                                     else 'formal_opening_no_separate_source_label'))
    return row


@dataclass(frozen=True)
class StoryboardInputs:
    storyboard_path: Path
    workstation_path: Path
    station: str
    operation: str
    side: str
    method: dict[str, Any]
    placement: dict[str, Any]
    cameras: dict[str, Any]
    physical_description: dict[str, Any]
    poses_by_role: dict[str, dict | None]
    missing_roles: tuple[str, ...]
    storyboard: dict[str, Any]
    workstation: dict[str, Any]

    @property
    def controller_poses(self) -> list[dict]:
        if self.missing_roles:
            raise IncompleteStoryboardError(self.station, self.operation, self.missing_roles)
        return [_controller_pose(self.poses_by_role[role], role)
                for role in CONTROLLER_ROLES]

    def controller_inputs(self) -> dict:
        """Return old G2Controller-shaped inputs; baseline parameters stay separate."""
        return dict(station=self.station, operation=self.operation, side=self.side,
                    method=copy.deepcopy(self.method), placement=copy.deepcopy(self.placement),
                    cameras=copy.deepcopy(self.cameras), poses=self.controller_poses,
                    physical_description=copy.deepcopy(self.physical_description))


def load_storyboard_inputs(storyboard_path, *, expected_station=None,
                          expected_operation=None, require_complete=True) -> StoryboardInputs:
    """Load only a formal G1 file and its referenced formal workstation file.

    Set require_complete=False for inspection of incomplete converted material.
    controller_poses still refuses to fabricate the missing legacy input roles.
    No tool-orientation, collision, phase, or contact-search rules are added here.
    """
    path = Path(storyboard_path).resolve()
    document = _read(path)
    if document.get('schema') != 'g1_storyboard_v1' or document.get('version') != 1:
        raise StoryboardInputError('Expected g1_storyboard_v1 version 1')
    station, operation = document.get('station'), document.get('operation')
    if not isinstance(station, str) or not station or operation not in ('open', 'close'):
        raise StoryboardInputError('Formal G1 needs a station and open/close operation')
    if expected_station is not None and station != expected_station:
        raise StoryboardInputError(f'G1 station {station} differs from requested {expected_station}')
    if expected_operation is not None and operation != expected_operation:
        raise StoryboardInputError(f'G1 operation {operation} differs from requested {expected_operation}')
    relative = document.get('workstation_file')
    if not isinstance(relative, str) or not relative:
        raise StoryboardInputError('Formal G1 has no workstation_file')
    station_path = Path(relative)
    if not station_path.is_absolute():
        station_path = path.parent/station_path
    station_path = station_path.resolve()
    workstation = _read(station_path)
    if workstation.get('schema') != 'valve_workstation_inputs_v1':
        raise StoryboardInputError('Expected valve_workstation_inputs_v1')
    if workstation.get('station') != station:
        raise StoryboardInputError('G1 and workstation station differ')
    side = workstation.get('side')
    if side not in ('left', 'right'):
        raise StoryboardInputError('Workstation side must be left or right')
    for field in ('placement', 'cameras', 'physical_description'):
        if not isinstance(workstation.get(field), dict) or not workstation[field]:
            raise StoryboardInputError(f'Formal workstation has no {field}')
    physical = workstation['physical_description']
    if physical.get('case') != station:
        raise StoryboardInputError('Embedded physical description belongs to another station')
    formal_method = document.get('method')
    if not isinstance(formal_method, dict):
        raise StoryboardInputError('Formal G1 has no method')
    method = copy.deepcopy(formal_method.get('original_parameters'))
    if not isinstance(method, dict):
        raise StoryboardInputError('Formal G1 has no preserved geometry method')
    missing_method = [field for field in _METHOD_FIELDS if field not in method]
    if missing_method:
        raise StoryboardInputError('Geometry method is missing: '+', '.join(missing_method))
    if method['mode'] not in ('push', 'pull') or method['mode'] != formal_method.get('mode'):
        raise StoryboardInputError('Formal method and preserved push/pull mode differ')
    for key, expected in (('station', station), ('task', operation), ('side', side)):
        if key in method and method[key] != expected:
            raise StoryboardInputError(f'Preserved method {key} differs from formal inputs')
    for field in ('radius_m', 'pitch_deg', 'tilt_deg', 'roll_deg'):
        _finite_number(method[field], f'method.{field}')

    rows = document.get('poses')
    keyframes = document.get('keyframes')
    if not isinstance(rows, list) or not isinstance(keyframes, list):
        raise StoryboardInputError('Formal G1 needs poses and keyframes lists')
    by_id = {}
    joint_names = None
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] in by_id:
            raise StoryboardInputError('G1 pose IDs must be present and unique')
        q, names = row.get('full_qpos'), row.get('qpos_joint_names')
        if (not isinstance(q, list) or not q or not isinstance(names, list)
                or len(q) != len(names) or any(not isinstance(n, str) or not n for n in names)
                or len(names) != len(set(names))):
            raise StoryboardInputError(f'{row["id"]}: invalid saved joints/name mapping')
        for value in q:
            _finite_number(value, f'{row["id"]}.full_qpos')
        if joint_names is not None and names != joint_names:
            raise StoryboardInputError('Saved pose joint ordering is inconsistent')
        joint_names = names
        _finite_number(row.get('opening_deg'), f'{row["id"]}.opening_deg')
        by_id[row['id']] = row
    role_rows = {}
    for keyframe in keyframes:
        if not isinstance(keyframe, dict) or not isinstance(keyframe.get('role'), str):
            raise StoryboardInputError('Keyframe role is missing')
        role = keyframe['role']
        if role in role_rows:
            raise StoryboardInputError(f'Duplicate G1 role: {role}')
        role_rows[role] = keyframe
    resolved, missing = {}, []
    for role in CONTROLLER_ROLES:
        keyframe = role_rows.get(role, {})
        ref = keyframe.get('pose_ref')
        if ref is None:
            if keyframe.get('state') == 'present':
                raise StoryboardInputError(f'{role} claims present but has no pose_ref')
            resolved[role] = None
            missing.append(role)
        else:
            if ref not in by_id or keyframe.get('state') != 'present':
                raise StoryboardInputError(f'{role} has an unresolved/inconsistent pose_ref')
            resolved[role] = copy.deepcopy(by_id[ref])
    loaded = StoryboardInputs(path, station_path, station, operation, side, method,
        copy.deepcopy(workstation['placement']), copy.deepcopy(workstation['cameras']),
        copy.deepcopy(physical), resolved, tuple(missing),
        copy.deepcopy(document), copy.deepcopy(workstation))
    if require_complete and missing:
        raise IncompleteStoryboardError(station, operation, missing)
    return loaded
