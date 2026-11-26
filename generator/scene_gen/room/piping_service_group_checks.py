"""Check that shared pipes serve the correct equipment on each floor."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from .building_equipment_checks import EPS


def _err(code, group=None, header=None, component=None, **detail):
    return {'code': code, 'group': group, 'header': header, 'component': component, **detail}


def service_group_contracts(frozen, piping, *, require_groups=True):
    """Independently count original demand; never let a group erase a source."""
    from .piping_header_checks import _sources
    points = _sources(frozen); by_id = {p['id']: p for p in points}
    headers = [r for r in piping.get('routes', []) if r.get('role') == 'floor_header']
    errors, groups = [], []
    legacy = 'service_groups' not in piping
    if legacy:
        if require_groups:
            errors.append(_err('SERVICE_GROUP_COVERAGE', reason='The current check requires service_groups'))
        keys = sorted({(p['medium'], p['level']) for p in points})
        for medium, level in keys:
            selected = [h for h in headers if h.get('medium') == medium and h.get('level') == level]
            groups.append({'id': f'legacy/{medium}/{level}', 'medium': medium, 'level': level,
                'header_id': selected[0]['id'] if len(selected) == 1 else None,
                'point_ids': [p['id'] for p in points if p['medium'] == medium and p['level'] == level]})
    else:
        supplied = piping.get('service_groups')
        if not isinstance(supplied, list):
            return {'legacy': False, 'groups': [], 'points': points,
                    'errors': [_err('SERVICE_GROUP_COVERAGE', reason='service_groups must be a complete list')]}
        ids = [g.get('id') for g in supplied]
        if len(ids) != len(set(ids)):
            errors.append(_err('SERVICE_GROUP_COVERAGE', reason='Duplicate service group ID'))
        for original in supplied:
            group = dict(original); gid = group.get('id'); members = group.get('point_ids', [])
            if not isinstance(gid, str) or not gid:
                errors.append(_err('SERVICE_GROUP_MEMBERSHIP', gid, reason='Each service group requires a distinct nonempty ID'))
            matched = [h for h in headers if h.get('service_group') == gid]
            group['header_id'] = matched[0]['id'] if len(matched) == 1 else None
            if not members or len(members) != len(set(members)):
                errors.append(_err('SERVICE_GROUP_COVERAGE', gid, group['header_id'], reason='Empty group or duplicate demand within a group'))
            for member in members:
                p = by_id.get(member)
                if not p or p['medium'] != group.get('medium') or p['level'] != group.get('level'):
                    errors.append(_err('SERVICE_GROUP_MEMBERSHIP', gid, group['header_id'], requirement=member,
                                       reason='A member differs from the saved floor or service'))
            owners = {by_id[m]['equipment'] for m in members if m in by_id}
            if set(group.get('equipment', [])) != owners:
                errors.append(_err('SERVICE_GROUP_MEMBERSHIP', gid, group['header_id'], reason='Equipment list differs from the saved demand owners'))
            if (len(matched) != 1 or any(h.get('medium') != group.get('medium') or h.get('level') != group.get('level')
                    or set(h.get('point_ids', [])) != set(members) or len(h.get('point_ids', [])) != len(members)
                    for h in matched)):
                errors.append(_err('SERVICE_GROUP_MEMBERSHIP', gid, group['header_id'], reason='Each active group needs exactly one header on the same floor and service with matching members'))
            groups.append(group)
        counts = Counter(m for g in groups for m in g.get('point_ids', []))
        for member in set(counts)|set(by_id):
            if counts[member] != 1 or member not in by_id:
                errors.append(_err('SERVICE_GROUP_COVERAGE', requirement=member, actual_count=counts[member], expected_count=1))
        assigned = [g['header_id'] for g in groups if g.get('header_id')]
        for h in headers:
            if assigned.count(h['id']) != 1:
                errors.append(_err('SERVICE_GROUP_COVERAGE', header=h['id'], reason='A physical header must have exactly one original service group'))
    return {'legacy': legacy, 'groups': groups, 'points': points, 'errors': errors}


def floor_link_evidence(header, main, position, routes, by_route, valve_rows, valve_segments, graph,
                        *, allow_unvalved_legacy=False):
    """Prove one connection's actual in -> verified native -> out chain only."""
    from . import piping_checks as common
    hid = header['id']; mid = main['id']; medium = header.get('medium'); level = header.get('level')
    matching = [r for r in routes if r.get('role') == 'floor_branch' and r.get('distribution_id') == hid
                and r.get('main_id') == mid and r.get('medium') == medium and r.get('level') == level
                and (header.get('service_group') is None or r.get('service_group') == header.get('service_group'))]
    actual = {v['id']: v for v in valve_rows if v['status'] == 'pass'}
    original = {v['id']: v for v in graph.get('valves', [])}
    roots = set(v for r in matching for v in r.get('valves', []))
    main_parts = by_route.get(mid, []); header_parts = by_route.get(hid, [])
    targets = [p for s in main_parts for p in s['ends']]
    row = {'header': hid, 'main': mid, 'medium': medium, 'level': level,
           'route_ids': [r['id'] for r in matching], 'valves': sorted(roots), 'connected': False,
           'in_routes': [], 'out_routes': [], 'actual_main_to_in': False, 'actual_out_to_header': False}
    if medium == 'offgas' or (allow_unvalved_legacy and not graph.get('valves')):
        pieces = [s for r in matching for s in by_route.get(r['id'], [])]
        row.update(connected=bool(not roots and matching and common._connected(position, targets,
                    pieces+header_parts+main_parts)),
            scope='PG15 direct off-gas connection without a root valve' if medium == 'offgas' else 'Legacy local fixture without a valve. Full P1 still requires root valves.')
        return row
    if len(roots) != 1:
        row['reason'] = 'Each active supply branch needs one physical root valve with matching identity'; return row
    root = next(iter(roots)); native = actual.get(root); definition = original.get(root, {})
    strict_identity = header.get('service_group') is not None
    if (not native or definition.get('role') != 'utility_floor_root'
            or definition.get('medium') != medium or definition.get('level') != level
            or definition.get('distribution_id', hid) != hid or definition.get('main_id', mid) != mid
            or definition.get('service_group', header.get('service_group')) != header.get('service_group')):
        row['reason'] = 'Native package is unverified or the valve belongs to another connection group'; return row
    if strict_identity and any(definition.get(key) != value for key,value in (
            ('distribution_id',hid),('main_id',mid),('service_group',header['service_group']))):
        row['reason'] = 'Each root valve needs a complete matching service-group identity'; return row
    own = [r for r in matching if root in r.get('valves', [])]
    if any(r.get('dn') != native['dn'] for r in own):
        row['reason'] = 'The group straight DN differs from its verified package'; return row
    connections = {r.get('connection_id') for r in own}
    if len(connections) != 1 or definition.get('connection_id', next(iter(connections), None)) not in connections:
        row['reason'] = 'The two pipe sides do not share one connection_id'; return row
    inlet, outlet = native['actual_ports']['in'], native['actual_ports']['out']
    ins = [r for r in own if any(common._on_segment(inlet, *s['ends']) for s in by_route.get(r['id'], []))]
    outs = [r for r in own if any(common._on_segment(outlet, *s['ends']) for s in by_route.get(r['id'], []))]
    in_parts = [s for r in ins for s in by_route.get(r['id'], [])]
    out_parts = [s for r in outs for s in by_route.get(r['id'], [])]
    main_in = bool(ins and common._connected(inlet, targets, in_parts+main_parts))
    out_header = bool(outs and common._connected(outlet, [position], out_parts+header_parts))
    disjoint = not ({r['id'] for r in ins}&{r['id'] for r in outs})
    row.update(in_routes=[r['id'] for r in ins], out_routes=[r['id'] for r in outs],
        actual_in_port=inlet, actual_out_port=outlet, actual_main_to_in=main_in,
        actual_out_to_header=out_header, verified_native=root, connection_id=next(iter(connections)),
        connected=bool(main_in and out_header and disjoint),
        scope='Check inlet and outlet against the same allocated pipe, main, floor, service, group, connection and physically verified valve')
    return row


def _body_centres(frozen):
    kinds = {uid: u['kind'] for uid, u in frozen['contracts']['sizing']['graph']['units'].items()}
    result = {}
    for uid, kind in kinds.items():
        boxes = [c['box'] for c in frozen['scene']['components'] if (c.get('owner') == uid or c['id'] == uid)
                 and c.get('kind') in (kind, 'head') and kind not in ('header', 'junction')]
        if boxes:
            boxes = np.asarray(boxes); result[uid] = ((boxes[:, :3].min(axis=0)+boxes[:, 3:].max(axis=0))/2).tolist()
    for unit in frozen['contracts']['layout']['equipment']:
        if unit.get('integral_to') in result:
            result[unit['id']]=list(result[unit['integral_to']])
    return result


def _row_axes(owners, centres, footprint):
    positions = [centres[o] for o in owners if o in centres]
    if len(positions) != len(owners):
        return [], {'reason': 'Missing saved equipment body geometry'}
    if len(positions) > 1:
        span = np.ptp(np.asarray(positions)[:, :2], axis=0)
        return [i for i in (0, 1) if span[i]+EPS >= span[1-i]], {'centres': positions, 'span_xy_m': span.tolist()}
    origin = np.asarray(positions[0][:2]); others = [np.asarray(p[:2]) for u, p in centres.items() if u not in owners]
    if others:
        distance = min(float(np.linalg.norm(p-origin)) for p in others)
        vectors = [p-origin for p in others if abs(float(np.linalg.norm(p-origin))-distance) <= EPS]
    else:
        delta = np.asarray(footprint)/2-origin
        vectors = [delta if np.linalg.norm(delta) > EPS else np.asarray(footprint)]
    axes = sorted({i for v in vectors for i in (0, 1) if abs(v[i])+EPS >= abs(v[1-i])})
    return axes, {'centres': positions, 'single_owner_reference_vectors': [v.tolist() for v in vectors],
                  'scope': 'For a single unit, derive the candidate axis from nearby physical equipment and building geometry'}


def check_service_groups(frozen, graph, scene, piping, model, data, *, require_groups=True):
    from . import piping_checks as common
    from .piping_header_checks import _header_body, _plain
    from .pullout_reservations import check_pullout_reservations
    from .building_equipment_checks import equipment_footprints, read_layout_rules, _side_strips
    from .maintenance_geometry import level_maintenance
    from .process import RULES
    import mujoco
    contract = service_group_contracts(frozen, piping, require_groups=require_groups); errors = list(contract['errors']); rows = []
    if contract['legacy']:
        return {'status': 'fail' if errors else 'pass', 'errors': errors, 'rows': [],
                'legacy': True, 'scope': 'Legacy single-header fixture with no service_groups check'}
    actual = common._measured(model, data); route_map = {r['id']: r for r in piping.get('routes', [])}
    by_route, route_errors, _ = common._route_segments(list(route_map.values()),
        {c['id']: c for c in scene['components']}, actual, piping['config']['pipe_outer_diameter_mm'], model, data)
    native_rows, _, native_segments, _ = common._valves(graph, actual)
    valve_parts = {s['valve']:s for s in native_segments}
    points = {p['id']: p for p in contract['points']}; centres = _body_centres(frozen)
    for group in contract['groups']:
        gid = group['id']; header = route_map.get(group.get('header_id')); members = group.get('point_ids', [])
        if not header or not members or any(m not in points for m in members):
            continue
        body = _header_body(header, by_route)
        if body is None:
            errors.append(_err('SERVICE_GROUP_DIRECTION', gid, header['id'], reason='No continuous physical horizontal header')); continue
        owners = sorted({points[m]['equipment'] for m in members})
        axes, evidence = _row_axes(owners, centres, frozen['scene']['footprint'])
        axis = body['axis']; component = body['parts'][0]['id']
        aligned = axis in axes
        scalars = [points[m]['position'][axis] for m in members]
        covered = min(scalars) >= body['start'][axis]-EPS and max(scalars) <= body['end'][axis]+EPS
        distances = [math.hypot(*(np.asarray(points[m]['position'])-np.asarray(body['start']))[
            [i for i in range(3) if i != axis]]) for m in members]
        row = {'group': gid, 'header': header['id'], 'component': component, 'point_ids': members,
            'actual_axis_index': axis, 'allowed_geometric_axes': axes, 'frozen_arrangement': evidence,
            'actual_header_interval_m': [body['start'][axis], body['end'][axis]],
            'frozen_outlet_interval_m': [min(scalars), max(scalars)], 'aligned': aligned, 'covered': covered,
            'per_outlet_perpendicular_distance_m': distances,
            'distance_difference_m': max(distances)-min(distances), 'equal_length_threshold': None}
        rows.append(row)
        if not aligned:
            errors.append(_err('SERVICE_GROUP_DIRECTION', gid, header['id'], component, detail=row))
        if not covered:
            errors.append(_err('SERVICE_GROUP_SPAN', gid, header['id'], component, detail=row))
        for member in members:
            point = points[member]
            linked = [r for r in route_map.values() if r.get('source_requirement')==member]
            identity = bool(linked) and all(r.get('role')=='equipment_connection'
                and r.get('distribution_id')==header['id'] and r.get('service_group')==gid
                and r.get('level')==group['level'] and r.get('medium')==group['medium'] for r in linked)
            own_parts = [s for r in linked for s in by_route.get(r['id'],[])]
            own_parts += [valve_parts[v] for r in linked for v in r.get('valves',[]) if v in valve_parts]
            touches = common._touch_points(own_parts,body['parts'])
            connected = common._connected(point['position'],touches,own_parts)
            if not identity or not connected:
                errors.append(_err('SERVICE_GROUP_MEMBERSHIP',gid,header['id'],
                    linked[0].get('components',[point['nozzle']])[0] if linked else point['nozzle'],
                    requirement=member, actual_distribution_ids=[r.get('distribution_id') for r in linked],
                    actual_group_identity=identity, actual_source_to_assigned_header=connected))
    pullout = check_pullout_reservations(graph, scene, model)
    if pullout['status'] != 'pass':
        for violation in pullout['violations']:
            errors.append(_err('SERVICE_GROUP_PULLOUT_BLOCKED', component=violation.get('equipment'), original=violation))
    reservations = []
    layout_graph = frozen.get('graph', graph); rules = read_layout_rules()
    for footprint in equipment_footprints(layout_graph, scene, model):
        if not footprint.get('rect'):
            continue
        unit = layout_graph['units'][footprint['id']]
        for level in footprint['occupied_levels']:
            profile = level_maintenance(footprint, unit, scene, level, rules)
            z = scene['floors'][level]
            for side, rect in _side_strips(profile['rect'], profile['width_m']).items():
                reservations.append({'owner':footprint['id'], 'purpose':'original_maintenance_'+side,
                    'box':[rect[0],rect[1],z,rect[2],rect[3],z+RULES['geometry_screen']['walk_height_m']]})
    header_names = {name for r in route_map.values() if r.get('role')=='floor_header' for name in r.get('components',[])}
    original_names = {c['id'] for c in frozen['scene']['components']}
    rack_names = {c['id'] for c in scene['components'] if c['id'] not in original_names
                  and c.get('kind') in ('beam','hanger','equipment_beam') and c.get('piping')}
    maintenance = []; cache = {}
    for name in sorted(header_names|rack_names):
        measured = actual.get(name)
        if measured is None:
            continue
        box = np.asarray(measured['box'])
        for reservation in reservations:
            other = np.asarray(reservation['box'])
            if np.any(np.minimum(box[3:],other[3:])-np.maximum(box[:3],other[:3]) <= EPS):
                continue
            virtual = {'type':int(mujoco.mjtGeom.mjGEOM_BOX),'position':(other[:3]+other[3:])/2,
                       'size':(other[3:]-other[:3])/2,'matrix':np.eye(3),'box':other.tolist()}
            proof = common._separation_proof(measured,virtual,model,cache)
            blocked = not (proof and proof['gap_m'] >= -EPS)
            row = {'component':name,'reservation':reservation,'actual_box':measured['box'],
                   'actual_support_separation':proof,'blocked':blocked}
            maintenance.append(row)
            if blocked:
                errors.append(_err('SERVICE_GROUP_MAINTENANCE_BLOCKED',component=name,detail=row))
    if route_errors:
        errors.append(_err('SERVICE_GROUP_MEMBERSHIP', reason='Base physical pipe or OD is unverified', details=route_errors))
    return _plain({'status': 'fail' if errors else 'pass', 'errors': errors, 'rows': rows,
        'coverage': contract, 'alignment': rows, 'span': rows, 'pullout': pullout, 'maintenance':maintenance,
        'scope': 'Check each source demand once, one header per group, body-derived layout, outlet span and pull-out space. Distances are reported separately.'})
