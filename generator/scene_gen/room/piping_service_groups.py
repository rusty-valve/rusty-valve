"""Group utility connections by service and floor."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import itertools

import numpy as np

from .evaluation_common import require_environment
from .recipe import CandidateRefused


EPS = 1e-8
VERSION = 'v1.6_frozen_body_position_service_groups'
_CORE = {'vessel', 'horizontal_vessel', 'tank', 'exchanger', 'condenser',
         'fixed_bed', 'tower', 'filter', 'pump'}


class ServiceGroupUnavailable(CandidateRefused):
    def __init__(self, code, detail, **evidence):
        self.code = code
        self.evidence = {'code': code, 'detail': detail, **evidence}
        super().__init__(code+': '+detail)


def frozen_body_positions(frozen):
    """Union of actual frozen primary body/head boxes; no declared tank axis."""
    scene = frozen['scene']
    equipment = frozen.get('contracts', {}).get('layout', {}).get('equipment', [])
    owners = {row['id'] for row in equipment}
    if not owners:
        owners = {uid for uid, unit in frozen.get('graph', {}).get('units', {}).items()
                  if unit.get('kind') in _CORE}
    rows = []
    for owner in owners:
        row=next((r for r in equipment if r['id']==owner),{})
        physical_owner=row.get('integral_to',owner)
        parts = [c for c in scene['components'] if (c.get('owner') == physical_owner or c['id'] == physical_owner)
                 and (c.get('kind') in _CORE|{'head'} or c.get('catalogue_primary_body'))]
        if not parts:
            raise ServiceGroupUnavailable('SERVICE_GROUP_BODY_MISSING',
                'Saved equipment has no verifiable physical body geometry', equipment=owner)
        boxes = np.asarray([c['box'] for c in parts], dtype=float)
        if boxes.shape != (len(parts), 6) or not np.isfinite(boxes).all():
            raise ServiceGroupUnavailable('SERVICE_GROUP_BODY_INVALID', 'Saved body box lacks finite physical dimensions', equipment=owner)
        low = boxes[:, :3].min(axis=0); high = boxes[:, 3:].max(axis=0)
        if np.any(high < low):
            raise ServiceGroupUnavailable('SERVICE_GROUP_BODY_INVALID', 'Saved body box bounds are reversed', equipment=owner)
        rows.append({'equipment': owner, 'center': ((low+high)/2).tolist(),
            'body_rect': [float(low[0]), float(low[1]), float(high[0]), float(high[1])],
            'body_box': [*low.tolist(), *high.tolist()], 'component_ids': [c['id'] for c in parts],
            'source': 'Use the combined center of physical shell and head boxes'})
    return sorted(rows, key=_body_key)


def _body_key(row):
    return tuple(row['center']+row['body_rect'])


def _group_key(rows):
    return tuple(value for row in sorted(rows, key=_body_key) for value in _body_key(row))


def _dominant(delta, footprint):
    if abs(abs(delta[0])-abs(delta[1])) > EPS:
        return int(abs(delta[1]) > abs(delta[0]))
    return int(footprint[1] > footprint[0]+EPS)


def _row_geometry(rows, footprint):
    centres = np.asarray([row['center'][:2] for row in rows])
    spread = centres.max(axis=0)-centres.min(axis=0)
    if max(spread) <= EPS:
        return None
    axis = _dominant(spread, footprint); perpendicular = 1-axis
    lo = max(row['body_rect'][perpendicular] for row in rows)
    hi = min(row['body_rect'][perpendicular+2] for row in rows)
    if lo > hi+EPS or spread[axis]+EPS < spread[perpendicular]:
        return None
    return {'axis_index': axis, 'center_spread_xy_m': spread.tolist(),
            'perpendicular_common_interval_m': [lo, hi],
            'transverse_to_axial_spread_ratio': float(spread[perpendicular]/spread[axis])}


def _single_axis(row, all_bodies, footprint):
    centre = np.asarray(row['center'][:2])
    others = []
    for other in all_bodies:
        if other['equipment'] == row['equipment']:
            continue
        delta = np.asarray(other['center'][:2])-centre
        distance = float(np.linalg.norm(delta))
        if distance > EPS:
            others.append((distance, tuple(other['center']), delta, other))
    if others:
        _, _, delta, other = min(others, key=lambda item: item[:2])
        return _dominant(delta, footprint), {'kind': 'single_owner_nearest_other_frozen_body_center',
            'own_center': row['center'], 'other_center': other['center'], 'delta_xy': delta.tolist(),
            'other_body_components': other['component_ids'], 'tie': 'equal distance -> actual centre xyz, never ID'}
    delta = np.asarray(footprint)/2-centre
    axis = _dominant(delta, footprint)
    return axis, {'kind': 'single_owner_relative_to_actual_shell_center',
        'own_center': row['center'], 'shell_center_xy': (np.asarray(footprint)/2).tolist(),
        'delta_xy': delta.tolist(), 'shell_footprint_m': list(footprint),
        'tie': 'larger actual shell span, then canonical +X when exactly equal; not body axis'}


def service_groups_from_frozen(frozen, requirements):
    """Every original service point appears in one same-level/medium group."""
    require_environment()
    bodies = frozen_body_positions(frozen); body_map = {row['equipment']: row for row in bodies}
    footprint = frozen['scene']['footprint']
    equipment = {row['id']: row for row in frozen.get('contracts', {}).get('layout', {}).get('equipment', [])}
    buckets = {}
    for point in requirements['service_points']:
        owner = point['equipment']
        if owner not in body_map:
            raise ServiceGroupUnavailable('SERVICE_GROUP_BODY_MISSING', 'Source service is missing its saved component', equipment=owner)
        try:
            port = equipment[owner]['ports'][point['port']]
            position = np.asarray(port['pos'], dtype=float)
            normal = np.asarray(port['normal'], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceGroupUnavailable('SERVICE_GROUP_PORT_MISSING', 'Source service is missing its saved physical port', requirement=point['id']) from exc
        if (position.shape != (3,) or normal.shape != (3,) or not np.isfinite(position).all()
                or not np.isfinite(normal).all() or np.max(abs(position-point['position'])) > EPS
                or np.max(abs(normal-point['normal'])) > EPS):
            raise ServiceGroupUnavailable('SERVICE_GROUP_PORT_CHANGED', 'Requested port differs from the saved physical port', requirement=point['id'])
        buckets.setdefault((point['level'], point['medium']), []).append(point)
    groups = []
    for (level, medium), points in sorted(buckets.items()):
        owner_set = {p['equipment'] for p in points}
        clusters = [[deepcopy(body_map[owner])] for owner in owner_set]
        clusters.sort(key=_group_key); merge_log = []
        all_centres=np.asarray([body_map[owner]['center'][:2] for owner in owner_set])
        global_spread=all_centres.max(axis=0)-all_centres.min(axis=0)
        global_axis=_dominant(global_spread,footprint)
        while True:
            choices = []
            for i, j in itertools.combinations(range(len(clusters)), 2):
                merged = clusters[i]+clusters[j]
                geometry = _row_geometry(merged, footprint)
                if geometry is None:
                    continue
                centres = [np.asarray(row['center'][:2]) for row in merged]
                diameter = max(float(np.linalg.norm(a-b)) for a, b in itertools.combinations(centres, 2))
                choices.append((geometry['transverse_to_axial_spread_ratio'],
                                int(geometry['axis_index']!=global_axis),diameter,
                                _group_key(merged), i, j, geometry))
            if not choices:
                break
            _, _, _, _, i, j, geometry = min(choices)
            merged = sorted(clusters[i]+clusters[j], key=_body_key)
            merge_log.append({'body_centres': [r['center'] for r in merged], **geometry})
            clusters = [c for index, c in enumerate(clusters) if index not in (i, j)]+[merged]
            clusters.sort(key=_group_key)
        for ordinal, rows in enumerate(clusters):
            geometry = _row_geometry(rows, footprint)
            if geometry is None:
                axis, basis = _single_axis(rows[0], bodies, footprint)
                geometry = {'axis_index': axis, 'center_spread_xy_m': [0., 0.],
                    'perpendicular_common_interval_m': [rows[0]['body_rect'][1-axis], rows[0]['body_rect'][3-axis]],
                    'axis_basis': basis, 'row_alignment_is_nondirectional': True}
            else:
                axis = geometry['axis_index']
                geometry['axis_basis'] = {'kind': 'frozen_body_center_row',
                    'rule': 'common transverse body interval; dominant centre spread along building X/Y; not cylinder axis'}
            owner_ids = {row['equipment'] for row in rows}
            selected = [p for p in points if p['equipment'] in owner_ids]
            selected.sort(key=lambda p: (p['position'][axis], p['position'][1-axis], p['position'][2], *p['normal'], p['dn']))
            vector = [0., 0., 0.]; vector[axis] = 1.
            groups.append({'id': f'SG_{medium}_L{level}_G{ordinal:02}', 'version': VERSION,
                'medium': medium, 'level': level, 'axis_index': axis, 'axis': vector,
                'point_ids': [p['id'] for p in selected], 'equipment': [row['equipment'] for row in rows],
                'geometry_evidence': {**geometry, 'device_centres': rows, 'merge_log': deepcopy(merge_log),
                    'algorithm': 'finite common-transverse-interval rows; prefer lower transverse ratio, shared dominant centre-spread axis, then geometric diameter',
                    'all_service_body_center_spread_xy_m':global_spread.tolist(),
                    'equal_straightness_preferred_axis_index':global_axis,
                    'no_acceptance_threshold_added': True},
                'source_position_order': [{'requirement': p['id'], 'position': list(p['position'])} for p in selected]})
    counts = Counter(pid for group in groups for pid in group['point_ids'])
    expected = Counter(p['id'] for p in requirements['service_points'])
    if any(value != 1 for value in expected.values()) or counts != expected:
        raise ServiceGroupUnavailable('SERVICE_GROUP_COVERAGE', 'Each source service must be covered exactly once')
    return groups


def building_long_edge_axis(scene, level):
    """Read the actual outer frame; all square frames use canonical +X.

    The frozen shell uses one outer frame on all actual floors. Its footprint
    excludes stairs and platform holes, which do not change that frame.
    """
    footprint = np.asarray(scene['footprint'], dtype=float)
    if footprint.shape != (2,) or not np.isfinite(footprint).all() or np.any(footprint <= 0):
        raise ServiceGroupUnavailable('SERVICE_GROUP_FRAME_MISSING', 'Saved building boundary lacks positive finite length and width')
    axis = int(footprint[1] > footprint[0]+EPS)
    return axis, {'kind': 'frozen_building_platform_outer_frame_long_edge',
        'level': level, 'frame_rect_m': [0., 0., *footprint.tolist()],
        'shell_footprint_m': footprint.tolist(), 'square_tie_axis_index': 0,
        'rule': 'Use the long floor or building boundary axis. Square boundaries use +X.'}


def shared_service_groups_from_frozen(frozen, requirements):
    """Method v1.4: one long-edge header per floor/independent service.

    Supply and return remain independent. The building/platform outer frame
    chooses the common axis; equipment positions only order natural tees.
    """
    require_environment()
    bodies = frozen_body_positions(frozen)
    body_map = {row['equipment']: row for row in bodies}
    equipment = {row['id']: row for row in frozen.get('contracts', {}).get('layout', {}).get('equipment', [])}
    footprint = frozen['scene']['footprint']; buckets = {}
    for point in requirements['service_points']:
        owner = point['equipment']
        if owner not in body_map:
            raise ServiceGroupUnavailable('SERVICE_GROUP_BODY_MISSING', 'Source service is missing its saved component', equipment=owner)
        try:
            port = equipment[owner]['ports'][point['port']]
            position = np.asarray(port['pos'], dtype=float)
            normal = np.asarray(port['normal'], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceGroupUnavailable('SERVICE_GROUP_PORT_MISSING', 'Source service is missing its saved physical port', requirement=point['id']) from exc
        if (position.shape != (3,) or normal.shape != (3,) or not np.isfinite(position).all()
                or not np.isfinite(normal).all() or np.max(abs(position-point['position'])) > EPS
                or np.max(abs(normal-point['normal'])) > EPS):
            raise ServiceGroupUnavailable('SERVICE_GROUP_PORT_CHANGED', 'Requested port differs from the saved physical port', requirement=point['id'])
        buckets.setdefault((point['level'], point['medium']), []).append(point)
    groups = []
    for (level, medium), points in sorted(buckets.items()):
        rows = sorted((deepcopy(body_map[owner]) for owner in {p['equipment'] for p in points}), key=_body_key)
        centres = np.asarray([row['center'][:2] for row in rows])
        spread = centres.max(axis=0)-centres.min(axis=0)
        axis, basis = building_long_edge_axis(frozen['scene'], level)
        perpendicular = 1-axis
        selected = sorted(points, key=lambda p: (p['position'][axis], p['position'][perpendicular],
            p['position'][2], *p['normal'], p['dn']))
        vector = [0., 0., 0.]; vector[axis] = 1.
        groups.append({'id': f'SG_shared_{medium}_L{level}', 'version': 'method_v1.4_shared_service',
            'medium': medium, 'level': level, 'axis_index': axis, 'axis': vector,
            'point_ids': [p['id'] for p in selected], 'equipment': [row['equipment'] for row in rows],
            'geometry_evidence': {'axis_index': axis, 'axis_basis': basis, 'device_centres': rows,
                'center_spread_xy_m': spread.tolist(),
                'perpendicular_common_interval_m': [max(r['body_rect'][perpendicular] for r in rows),
                                                     min(r['body_rect'][perpendicular+2] for r in rows)],
                'perpendicular_interval_is_alignment_evidence_only': True,
                'independent_service_identities': sorted({str(p.get('service_identity', medium)) for p in points}),
                'algorithm': 'One group per physical floor and independent service. Follow the long boundary axis and preserve projected outlet order.',
                'no_equipment_row_split': True, 'no_acceptance_threshold_added': True},
            'source_position_order': [{'requirement': p['id'], 'position': list(p['position'])} for p in selected]})
    expected = Counter(p['id'] for p in requirements['service_points'])
    counts = Counter(pid for group in groups for pid in group['point_ids'])
    if any(value != 1 for value in expected.values()) or counts != expected:
        raise ServiceGroupUnavailable('SERVICE_GROUP_COVERAGE', 'Each source service must be covered exactly once')
    return groups
