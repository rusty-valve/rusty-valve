"""Choose where branch pipes join a shared pipe."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .evaluation_common import require_environment
from .piping_assets import inspect_asset, AssetUnavailable
from .recipe import CandidateRefused


EPS = 1e-8
MAX_CANDIDATES = 24  # Original workstation resource cap; not an install rule.
VERSION = 'v1.5_ordered_header_taps_trial'
FORMULA = {
    'tee_half_length_m': 't_i = max(actual_header_OD, actual_branch_i_OD)',
    'straight_reserve_m': 's_ij = max(header_OD, branch_i_OD, branch_j_OD, native_L_i, native_L_j)',
    'minimum_center_distance_m': 'pitch_ij = t_i + t_j + s_ij',
    'end_clearance_m': 'end_i = t_i + max(header_OD, branch_i_OD, native_L_i)',
    'basis': 'Reserve an approximate tee envelope and one associated valve installation length for assembly clearance',
    'diameter_source': 'Physical OD table in requirements.config.pipe_outer_diameter_mm',
    'length_source': 'Use native install_length_m for the same DN. Connections without a valve explicitly use L equal to zero.',
    'parallel_50mm_used_as_axial_pitch': False,
    'scope': 'Plan first. Verify tees, blinds, flanges and spacing from the emitted XML.',
}


class HeaderTapUnavailable(CandidateRefused):
    def __init__(self, code, detail, **evidence):
        self.code = code
        self.evidence = {'code': code, 'detail': detail, **evidence}
        super().__init__(code+': '+detail)


def _reject(code, detail, **evidence):
    raise HeaderTapUnavailable(code, detail, **evidence)


def _number(value, label, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        _reject('HEADER_TAP_INPUT', label+'Missing finite '+('nonnegative' if zero else 'positive')+'measured dimension')
    return float(value)


def _point(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HeaderTapUnavailable('HEADER_TAP_INPUT', label+'missing finite three-dimensional coordinates') from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        _reject('HEADER_TAP_INPUT', label+'missing finite three-dimensional coordinates')
    return result


def _od(config, dn):
    if isinstance(dn, bool) or not isinstance(dn, int):
        _reject('HEADER_TAP_DN', 'Tap DN must be an integer in the existing OD table', dn=dn)
    try:
        value = config['pipe_outer_diameter_mm'][str(dn)]
    except (KeyError, TypeError) as exc:
        raise HeaderTapUnavailable('HEADER_TAP_DN', 'Tap DN lacks an actual pipe OD entry', dn=dn) from exc
    return _number(value, f'DN{dn} OD')/1000.


def tee_reservation(config, header_dn, branch_dn, install_length_m=0.):
    """Public coarse reservation formula; all lengths in metres."""
    header_od = _od(config, header_dn)
    branch_od = _od(config, branch_dn)
    length = _number(install_length_m, 'native install_length_m', zero=True)
    takeout = max(header_od, branch_od)
    straight = max(header_od, branch_od, length)
    return {'header_dn': int(header_dn), 'branch_dn': int(branch_dn),
        'header_od_m': header_od, 'branch_od_m': branch_od,
        'tee_half_length_m': takeout, 'slot_install_length_m': length,
        'straight_reserve_m': straight, 'end_reserve_m': takeout+straight,
        'basis': FORMULA['basis'], 'is_project_coarse_estimate': True}


def adjacent_reservation(first, second):
    """Centre spacing, without importing the parallel clearance threshold."""
    straight = max(first['header_od_m'], second['header_od_m'],
        first['branch_od_m'], second['branch_od_m'],
        first['slot_install_length_m'], second['slot_install_length_m'])
    halves = [first['tee_half_length_m'], second['tee_half_length_m']]
    return {'required_center_distance_m': sum(halves)+straight,
        'tee_halves_m': halves, 'straight_reserve_m': straight,
        'basis': FORMULA['basis']}


def _canonical(header):
    start = _point(header.get('start'), 'header.start')
    end = _point(header.get('end'), 'header.end')
    delta = end-start
    axes = np.flatnonzero(abs(delta) > EPS)
    if len(axes) != 1 or int(axes[0]) == 2:
        _reject('HEADER_TAP_NON_HORIZONTAL_AXIS', 'Floor header must follow X or Y and have nonzero length')
    axis = int(axes[0]); tangent = np.zeros(3); tangent[axis] = 1.
    origin, terminal = (start, end) if delta[axis] > 0 else (end, start)
    return axis, tangent, origin, terminal, float(terminal[axis]-origin[axis])


def _native_length(dn, supplied=None, *, allow_zero=False):
    if supplied is None:
        try:
            asset = inspect_asset(dn)
        except AssetUnavailable as exc:
            raise HeaderTapUnavailable('HEADER_TAP_NATIVE_UNAVAILABLE', str(exc), dn=dn) from exc
        return float(asset['install_length_m']), {'source': 'same_DN_locked_native_asset',
            'package': asset.get('package'), 'dn': dn}
    source = {'source': 'caller_related_native_install_length', 'dn': dn}
    if isinstance(supplied, dict):
        if 'dn' in supplied and supplied['dn'] != dn:
            _reject('HEADER_TAP_NATIVE_DN', 'Use the valve installation length for the same DN', expected_dn=dn, actual_dn=supplied['dn'])
        length = supplied.get('install_length_m')
        source.update({key: deepcopy(supplied[key]) for key in ('package', 'id') if key in supplied})
    else:
        length = supplied
    length = _number(length, 'associated native installation length', zero=allow_zero)
    if length == 0:
        source['source'] = 'explicit_no_related_native_valve'
    return length, source


def _end_limits(components, origin, axis, span):
    """Projection of the already-emitted real end fitting component boxes."""
    groups = {'low': [], 'high': []}
    for component in components:
        kind = component.get('kind')
        if kind not in ('piping_blind', 'piping_flange', 'blind', 'flange'):
            _reject('HEADER_END_FITTING_KIND', 'End parts must be physical blind plates or flanges', component=component.get('id'))
        try:
            box = np.asarray(component['box'], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise HeaderTapUnavailable('HEADER_END_FITTING_GEOMETRY', 'End part lacks a physical box') from exc
        if box.shape != (6,) or not np.isfinite(box).all() or np.any(box[3:] < box[:3]):
            _reject('HEADER_END_FITTING_GEOMETRY', 'End part lacks a complete finite box')
        projection = [float(box[axis]-origin[axis]), float(box[axis+3]-origin[axis])]
        end = 'low' if sum(projection)/2 <= span/2 else 'high'
        groups[end].append({'id': component.get('id'), 'kind': kind,
            'box': box.tolist(), 'axial_interval_m': projection,
            'source': 'caller_already_emitted_actual_component_box; independent XML check required'})
    for end, rows in groups.items():
        if not any(r['kind'] in ('piping_blind', 'blind') for r in rows) or not any(
                r['kind'] in ('piping_flange', 'flange') for r in rows):
            _reject('HEADER_END_CLOSURE_MISSING', 'Both header ends require a physical flange and blind plate', end=end)
    low = max(0., max(row['axial_interval_m'][1] for row in groups['low']))
    high = min(span, min(row['axial_interval_m'][0] for row in groups['high']))
    if low > high+EPS:
        _reject('HEADER_END_FITTINGS_OVERLAP', 'End flanges and blinds leave no usable header body', limits=[low, high])
    return low, high, groups


def _device_rows(frozen, requirements, header, origin, axis, root_slots):
    equipment = {row['id']: row for row in frozen.get('contracts', {}).get('layout', {}).get('equipment', [])}
    points = [p for p in requirements['service_points']
              if p['medium'] == header['medium'] and p['level'] == header['level']]
    if 'point_ids' in header:
        ids=header['point_ids']
        if len(ids)!=len(set(ids)) or not set(ids)<={p['id'] for p in points}:
            _reject('HEADER_SERVICE_SET','Service group contains duplicate, cross-floor, cross-service or undeclared demands')
        points=[p for p in points if p['id'] in set(ids)]
    if not points:
        _reject('HEADER_NO_SERVICE', 'No physical service users for the proposed header tap')
    if 'point_ids' in header and (len(header['point_ids']) != len(points)
            or set(header['point_ids']) != {p['id'] for p in points}):
        _reject('HEADER_SERVICE_SET', 'Header services must match the saved active-group demands item by item')
    other = 1-axis; rows = []; identifiers = set()
    for point in points:
        if point['id'] in identifiers:
            _reject('HEADER_DUPLICATE_REQUIREMENT', 'Service demand is duplicated')
        identifiers.add(point['id'])
        try:
            physical = equipment[point['equipment']]['ports'][point['port']]
        except KeyError as exc:
            raise HeaderTapUnavailable('HEADER_FROZEN_PORT_MISSING', 'Equipment tap must correspond to the saved physical outlet') from exc
        position = _point(physical.get('pos'), 'saved port.pos')
        normal = _point(physical.get('normal'), 'saved port.normal')
        if (np.max(abs(position-_point(point.get('position'), 'requirement.position'))) > EPS
                or np.max(abs(normal-_point(point.get('normal'), 'requirement.normal'))) > EPS):
            _reject('HEADER_FROZEN_PORT_CHANGED', 'Requested outlet differs from the saved physical outlet anchor', requirement=point['id'])
        scalar = float(position[axis]-origin[axis])
        length, source = _native_length(point['dn'], root_slots.get(point['id']))
        reservation = tee_reservation(requirements['config'], header['dn'], point['dn'], length)
        key = [scalar, float(position[other]), float(position[2]), *normal.tolist(), int(point['dn'])]
        virtual_normal = [0., 0., 0.]; virtual_normal[axis] = 1.
        rows.append({'kind': 'equipment', 'requirement': point['id'], 'endpoint': point['endpoint'],
            'source_position': position.tolist(), 'source_normal': normal.tolist(),
            'source_plane': {'origin': position.tolist(), 'normal': virtual_normal, 'scalar': scalar,
                'anchor': 'frozen_actual_equipment_outlet_point',
                'definition': 'Orthogonal virtual plane anchored at the original outlet and perpendicular to the canonical transverse axis'},
            'preferred_scalar': scalar, 'geometry_key': key, 'dn': point['dn'],
            'reservation': reservation, 'slot_install_length': length,
            'slot_install_length_source': source})
    # Equal projections are grouped using only numerical metre equality.
    rows.sort(key=lambda row: row['preferred_scalar'])
    result = []; ties = []; offset = 0
    while offset < len(rows):
        finish = offset+1
        while finish < len(rows) and abs(rows[finish]['preferred_scalar']-rows[offset]['preferred_scalar']) <= EPS:
            finish += 1
        group = sorted(rows[offset:finish], key=lambda row: tuple(row['geometry_key'][1:]))
        for first, second in zip(group, group[1:]):
            if np.max(abs(np.asarray(first['geometry_key'])-second['geometry_key'])) <= EPS:
                _reject('HEADER_COINCIDENT_SOURCE_GEOMETRY', 'Identical outlet geometry cannot be ordered arbitrarily by ID',
                    requirements=[first['requirement'], second['requirement']])
        if len(group) > 1:
            ties.append({'requirements': [row['requirement'] for row in group],
                'geometry_keys': [row['geometry_key'] for row in group],
                'rule': 'same_projection: other_horizontal_coordinate,z,normal_xyz,DN; no ID/name/seed'})
        result.extend(group); offset = finish
    for index, row in enumerate(result):
        row['device_order_index'] = index
    return result, ties


def _chain_limits(chain, body_limits):
    intervals = [[body_limits[0]+row['reservation']['end_reserve_m'],
                  body_limits[1]-row['reservation']['end_reserve_m']] for row in chain]
    gaps = [adjacent_reservation(a['reservation'], b['reservation'])['required_center_distance_m']
            for a, b in zip(chain, chain[1:])]
    return intervals, gaps


def _place_chain(chain, body_limits, fixed_main=None):
    intervals, gaps = _chain_limits(chain, body_limits)
    for i, row in enumerate(chain):
        if row['kind'] == 'main' and fixed_main is not None:
            intervals[i] = [max(intervals[i][0], fixed_main), min(intervals[i][1], fixed_main)]
    earliest = [row[0] for row in intervals]
    latest = [row[1] for row in intervals]
    for i in range(1, len(chain)):
        earliest[i] = max(earliest[i], earliest[i-1]+gaps[i-1])
    for i in range(len(chain)-2, -1, -1):
        latest[i] = min(latest[i], latest[i+1]-gaps[i])
    if any(low > high+EPS for low, high in zip(earliest, latest)):
        return None
    values = []
    for i, row in enumerate(chain):
        low = earliest[i] if i == 0 else max(earliest[i], values[-1]+gaps[i-1])
        values.append(min(latest[i], max(low, row['preferred_scalar'])))
    return values, intervals, gaps, earliest, latest


def _allocate(frozen, requirements, header, main_source, root_slots, end_fittings, max_candidates):
    axis, tangent, origin, terminal, span = _canonical(header)
    config = requirements['config']
    _od(config, header['dn'])
    low, high, fitting_evidence = _end_limits(end_fittings, origin, axis, span)
    devices, ties = _device_rows(frozen, requirements, header, origin, axis, root_slots)
    if not isinstance(main_source, dict):
        _reject('HEADER_MAIN_SOURCE_MISSING', 'main_source must specify the actual position and header connection DN')
    position = _point(main_source.get('position'), 'main_source.position')
    dn = main_source.get('dn')
    _od(config, dn)
    length_input = main_source.get('install_length_m')
    length, length_source = _native_length(dn, length_input, allow_zero=True)
    main = {'kind': 'main', 'requirement': None, 'source_position': position.tolist(),
        'preferred_scalar': float(position[axis]-origin[axis]), 'dn': dn,
        'reservation': tee_reservation(config, header['dn'], dn, length),
        'slot_install_length': length, 'slot_install_length_source': length_source}
    proposals = []
    insertion_failures = []
    for insertion in range(len(devices)+1):
        chain = [*devices[:insertion], main, *devices[insertion:]]
        placement = _place_chain(chain, [low, high])
        if placement is None:
            insertion_failures.append({'main_insertion_index': insertion, 'code': 'HEADER_TAPS_DO_NOT_FIT'})
            continue
        _, _, gaps, earliest, latest = placement
        lo, hi = earliest[insertion], latest[insertion]
        preferred = main['preferred_scalar']
        values = [min(hi, max(lo, preferred)), lo, hi, (lo+hi)/2]
        if insertion:
            values.append(devices[insertion-1]['preferred_scalar']+gaps[insertion-1])
        if insertion < len(devices):
            values.append(devices[insertion]['preferred_scalar']-gaps[insertion])
        # Main near real source/near the adjacent outlet projections/in gaps.
        unique = []
        for value in values:
            value = min(hi, max(lo, value))
            if not any(abs(value-other) <= EPS for other in unique):
                unique.append(value)
        for value in unique:
            proposals.append((abs(value-preferred), insertion, float(value)))
    proposals.sort()
    candidates = []; attempts = []
    for _, insertion, value in proposals[:max_candidates]:
        chain = [*devices[:insertion], main, *devices[insertion:]]
        placement = _place_chain(chain, [low, high], value)
        if placement is None:
            attempts.append({'main_insertion_index': insertion, 'main_scalar': value,
                             'status': 'rejected', 'code': 'HEADER_TAPS_DO_NOT_FIT'})
            continue
        scalars, intervals, _, _, _ = placement
        taps = []
        for i, (row, scalar) in enumerate(zip(chain, scalars)):
            taps.append({**deepcopy(row), 'scalar': float(scalar),
                'position': (origin+tangent*scalar).tolist(), 'order_index': i,
                'usable_center_interval_m': intervals[i]})
        spacing = []
        for first, second in zip(taps, taps[1:]):
            spacing.append({'first': first['requirement'] if first['kind'] == 'equipment' else '__main__',
                'second': second['requirement'] if second['kind'] == 'equipment' else '__main__',
                'actual_distance_m': second['scalar']-first['scalar'],
                **adjacent_reservation(first['reservation'], second['reservation'])})
        displacement = sum(abs(t['scalar']-t['preferred_scalar']) for t in taps)
        candidate = {'header_id': header['id'], 'canonical_axis': tangent.tolist(), 'axis_index': axis,
            'origin': origin.tolist(), 'terminal': terminal.tolist(), 'length_m': span,
            'limits': {'actual_end_fitting_inner_interval_m': [low, high],
                       'per_tap': [{'kind': t['kind'], 'requirement': t['requirement'],
                                    'interval_m': t['usable_center_interval_m']} for t in taps]},
            'main_insertion_index': insertion, 'main_tap': next(t for t in taps if t['kind'] == 'main'),
            'device_taps': [t for t in taps if t['kind'] == 'equipment'], 'spacing': spacing,
            'total_projection_displacement_m': displacement, 'formula': deepcopy(FORMULA),
            'source_geometry_ties': deepcopy(ties), 'status': 'candidate_not_actual_geometry_pass'}
        candidates.append(candidate)
        attempts.append({'main_insertion_index': insertion, 'main_scalar': value,
                         'status': 'candidate', 'total_projection_displacement_m': displacement})
    # Best only among this finite candidate set; insertion index is geometric.
    candidates.sort(key=lambda c: (c['total_projection_displacement_m'],
        abs(c['main_tap']['scalar']-main['preferred_scalar']), c['main_tap']['scalar']))
    for index, candidate in enumerate(candidates):
        candidate['candidate_index'] = index
    report = {'version': VERSION, 'status': 'ready' if candidates else 'rejected',
        'header_id': header['id'], 'canonical_axis': tangent.tolist(), 'axis_index': axis,
        'origin': origin.tolist(), 'terminal': terminal.tolist(), 'header_length_m': span,
        'header_dn': header['dn'], 'header_od_m': _od(config, header['dn']),
        'end_fittings': fitting_evidence, 'body_usable_interval_m': [low, high],
        'ordered_devices': devices, 'source_geometry_ties': ties,
        'main_source': main, 'formula': deepcopy(FORMULA), 'candidates': candidates,
        'candidate_budget': max_candidates, 'attempted_candidates': len(attempts),
        'attempts': attempts, 'insertion_refusals': insertion_failures,
        'proposals_not_tried': max(0, len(proposals)-max_candidates),
        'global_optimality_claim': False, 'scope': 'Geometry ordering and bounded joint tap reservations'}
    if not candidates:
        report['refusal'] = {'code': 'HEADER_TAPS_DO_NOT_FIT',
            'detail': 'Tried joint candidates cannot fit ordered tees and assembly allowances'}
    return report


def allocate_header_taps(frozen, requirements, header_plan, *, main_source,
                         root_slots=None, end_fittings=(), max_candidates=24):
    """Return JSON-ready allocation candidates or an explicit expected refusal.

    root_slots: requirement ID -> real native install length, or dict with
    install_length_m/dn/package. Missing device slots resolve the same-DN
    locked native asset. main_source is {position,dn,install_length_m}; only
    an explicitly valve-free main connection may supply L=0 (e.g. PG15).
    end_fittings contains the already-emitted flange/blind component dicts
    for both ends of this header. No geometry, package or input is modified.
    """
    require_environment()
    try:
        if (isinstance(max_candidates, bool) or not isinstance(max_candidates, int)
                or not 1 <= max_candidates <= MAX_CANDIDATES):
            _reject('HEADER_TAP_BUDGET', 'Tap candidate limit must be from 1 to the existing limit of 24')
        if root_slots is not None and not isinstance(root_slots, dict):
            _reject('HEADER_TAP_INPUT', 'root_slots must map demand IDs to actual native installation lengths')
        return _allocate(frozen, requirements, header_plan, main_source,
                         root_slots or {}, end_fittings, max_candidates)
    except HeaderTapUnavailable as exc:
        return {'version': VERSION, 'status': 'rejected', 'header_id': header_plan.get('id'),
            'candidates': [], 'refusal': exc.evidence, 'formula': deepcopy(FORMULA),
            'global_optimality_claim': False}
