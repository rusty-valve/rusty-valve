"""Check that installed pipes and valves follow their required connections."""
from collections import Counter
from itertools import pairwise
import math

import numpy as np

from .connection_evidence import EPS, SERVICE_ROLES, direction, line_ends, owner, straight_spans, valve_faces
from .material_rack import MATERIAL, rack_run_evidence
from .valve_roles import position_check
from .vessel_bottom import beyond_side


def finding(subject, passed, **evidence):
    return {'subject': subject, 'pass': None if passed is None else bool(passed), **evidence}


def chain_label(chain):
    return ' → '.join(edge['id'] for edge in chain)


def connected(room, edge):
    return room.connections.get(edge['id'], {}).get('connected')


def chain_connected(room, chain):
    states = [connected(room, edge) for edge in chain]
    return False if False in states else None if None in states else True


def rack_geometry(path, band, destination):
    """Use the same rack bounds and descent criterion as the route search."""
    if not path or band is None or destination is None:
        return None
    return rack_run_evidence(straight_spans(path), band, destination)


def heated_returns(room):
    for edge in room.edges.values():
        source = room.units[owner(edge['a'])]
        vessel = source.get('service_owner') or source.get('owner')
        if (source.get('kind') != 'exchanger' or source.get('service') != 'steam_heater'
                or edge['role'] not in MATERIAL or edge['a'].rsplit('.', 1)[-1] != 'out'):
            continue
        for chain in room.chains(edge, MATERIAL):
            if owner(chain[-1]['b']) == vessel or edge.get('branch_purpose') == 'heated_return':
                yield chain


def check_m1_service_destination(room):
    for circuit in room.process.get('reflux_circuits', ()):
        vessel, condenser = circuit['owner'], circuit['condenser']
        vapor = [e for e in room.outgoing[vessel] if owner(e['b']) == condenser and e['role'] == 'vapor']
        yield finding(condenser, len(vapor) == 1 and circuit.get('return_to_owner') is True,
                      vapor_edges=[e['id'] for e in vapor], vessel=vessel)
    for transfer in room.process.get('transfers', ()):
        if not transfer.get('path_units'):
            continue
        units = transfer['path_units']
        yield finding(transfer['id'], units[0] == transfer['source'] and units[-1] == transfer['target'],
                      expected=[transfer['source'], transfer['target']], actual=units)


def check_m1_horizontal_vessel(room):
    for edge in room.edges.values():
        unit = room.units[owner(edge['a'])]
        port = edge['a'].rsplit('.', 1)[-1]
        if unit['kind'] != 'vessel' or unit.get('orientation') != 'horizontal':
            continue
        if port == 'bottom':
            side = [line for line in room.edge_path(edge['id']) if line['kind'] in ('pipe', 'valve')
                    and beyond_side(unit, line['ends'][1], direction(line['ends']))]
            yield finding(edge['id'], bool(side) if connected(room, edge) is True else None,
                          outside_side_pieces=[line['id'] for line in side])
        elif port in ('feed', 'liquid_return'):
            normal = room.port(edge['a']).get('normal')
            yield finding(edge['id'], normal[2] > .9 if normal else None, normal=normal)
    for edge in room.edges.values():
        unit = room.units[owner(edge['b'])]
        if (unit['kind'] == 'vessel' and unit.get('orientation') == 'horizontal'
                and edge['b'].rsplit('.', 1)[-1] in ('feed', 'liquid_return')):
            normal = room.port(edge['b']).get('normal')
            yield finding(edge['id'], normal[2] > .9 if normal else None, normal=normal)


def check_m2_bottom_root(room):
    for source in room.source_valves.values():
        if source.get('valve_role') != 'bottom_root':
            continue
        endpoint = source['at']
        part = room.units[owner(endpoint)]
        faces = [valve_faces(v) for v in room.installed[source['id']]]
        if not faces or not all(len(value) == 2 for value in faces):
            continue  # missing valves and their geometric seats have their own checks
        points = faces[0]
        axis = direction((points['in'], points['out']))
        if abs(axis[2]) > .9 and part.get('orientation') != 'horizontal':
            okay = True
        else:
            center = np.mean(list(points.values()), axis=0)
            okay = beyond_side(part, center, axis) if part.get('orientation') == 'horizontal' else (
                abs(axis[2]) <= EPS and any(center[i] < part['rect'][i]-EPS
                                           or center[i] > part['rect'][i+2]+EPS for i in (0, 1)))
        yield finding(source['id'], okay, endpoint=endpoint, faces=points)


def pump_groups(room):
    for transfer in room.process.get('transfers', ()):
        if transfer.get('pump'):
            yield transfer, transfer.get('protection_filter'), transfer['pump']


def check_m2_pump_protection(room):
    for transfer, filter_id, pump in pump_groups(room):
        inlet = [e for e in room.incoming[pump] if e['b'] == pump+'.in']
        before = [e for e in room.incoming[filter_id] if e['b'] == filter_id+'.in'] if filter_id else []
        after = [e for e in room.outgoing[filter_id] if e['a'] == filter_id+'.out'] if filter_id else []
        if not all(len(edges) == 1 for edges in (inlet, before, after)):
            yield finding(pump, False, filter=filter_id, transfer=transfer['id'])
            continue
        split, join = owner(before[0]['a']), owner(after[0]['b'])
        bypass = [e for e in room.outgoing[split] if e['role'] == 'bypass' and owner(e['b']) == join]
        upstream = [e for e in room.incoming[split] if e['a'] == transfer['source']+'.bottom']
        okay = (owner(inlet[0]['a']) == join and split != join and len(bypass) == len(upstream) == 1
                and all(room.units[uid]['kind'] == 'junction' for uid in (split, join)))
        chain = [*upstream, *before, *after, *inlet, *bypass]
        yield finding(pump, okay and chain_connected(room, chain), filter=filter_id,
                      edges=[e['id'] for e in chain], split=split, join=join)


def valve_role(valve):
    return valve.get('valve_role') or valve.get('role')


def edge_roles(room, edge):
    return [valve_role(room.valves[step['id']]) for step in room.edge_path(edge['id'])
            if step['kind'] == 'valve' and room.valves[step['id']].get('manual')]


def roles_in_order(observed, expected):
    remaining = iter(observed)
    return all(any(item == role for item in remaining) for role in expected)


def check_m2_pump_valve_order(room):
    for transfer, filter_id, pump in pump_groups(room):
        required_ports = {transfer['source']+'.bottom': 'bottom_root',
                          pump+'.in': 'pump_inlet', pump+'.out': 'pump_outlet'}
        if filter_id:
            required_ports.update({filter_id+'.in': 'filter_inlet', filter_id+'.out': 'filter_outlet'})
        actual = {}
        for endpoint, role in required_ports.items():
            uid = owner(endpoint)
            edges = [e for e in room.outgoing[uid]+room.incoming[uid] if endpoint in (e['a'], e['b'])]
            actual[endpoint] = [item for edge in edges for item in edge_roles(room, edge)]
        bypass = [e for e in room.edges.values() if e['role'] == 'bypass'
                  and any(owner(e['b']) == owner(i['a']) for i in room.incoming[pump])]
        actual['bypass'] = [item for edge in bypass for item in edge_roles(room, edge)]
        expected = {**required_ports, 'bypass': 'bypass'}
        missing = [role for port, role in expected.items() if actual[port].count(role) != 1]
        yield finding(pump, not missing, missing_or_duplicate=missing, observed=actual)


def check_m2_pump_returns(room):
    for transfer, _, pump in pump_groups(room):
        initial = [e for e in room.outgoing[pump] if e['a'] == pump+'.out']
        chains = [c for e in initial for c in room.chains(e, MATERIAL)]
        heated = [c for c in heated_returns(room)
                  if owner(c[-1]['b']) == transfer['source'] and
                  any(owner(prefix[-1]['b']) == owner(c[0]['a']) for prefix in chains)]
        for chain in [*chains, *heated]:
            target = chain[-1]['b']
            if owner(target) != transfer['source']:
                continue
            observed = [role for e in chain for role in edge_roles(room, e)]
            expected = ['recycle_regulation', 'return_root']
            expected.insert(0, 'heater_outlet' if chain in heated else 'pump_outlet')
            if room.units[owner(target)].get('return_manifold'):
                expected.append('common_root')
            yield finding(chain_label(chain), chain_connected(room, chain) and roles_in_order(observed, expected),
                          final_port=target, expected=expected, observed=observed)


def heater_ports(room):
    for uid, unit in room.units.items():
        if unit['kind'] == 'exchanger' and unit.get('service') == 'steam_heater':
            yield uid, unit


def check_m3_heater_circuits(room):
    for uid, unit in heater_ports(room):
        expected = {'in': MATERIAL, 'out': MATERIAL, 'utility_in': {'steam'},
                    'utility_out': {'steam_condensate'}}
        observed = {}
        for port, roles in expected.items():
            endpoint = uid+'.'+port
            incoming = port in ('in', 'utility_in')
            edges = [e for e in (room.incoming[uid] if incoming else room.outgoing[uid])
                     if e['b' if incoming else 'a'] == endpoint]
            observed[port] = [e['role'] for e in edges]
        yield finding(uid, all(len(observed[port]) == 1 and observed[port][0] in roles
                               for port, roles in expected.items()), roles=observed, form=unit.get('orientation'))


def check_m3_heater_nozzles(room):
    for uid, unit in heater_ports(room):
        keys = ('in', 'out', 'utility_in', 'utility_out')
        points = {key: room.point(uid+'.'+key) for key in keys}
        if any(value is None for value in points.values()):
            yield finding(uid, None, points=points)
            continue
        material = (points['in'], points['out'])
        steam, drain = points['utility_in'], points['utility_out']
        okay = material[0][2] < material[1][2]-EPS and steam[2] > drain[2]+EPS
        if unit.get('orientation') == 'vertical':
            inlet = room.port(uid+'.utility_in').get('normal')
            outlet = room.port(uid+'.utility_out').get('normal')
            okay = okay and (float(np.dot(inlet, outlet)) < -.9 if inlet and outlet else False)
        yield finding(uid, okay, points=points)


def check_m3_heater_low_outlet(room):
    checked = {row['heater']: row for report in room.measure.get('confirmed_rules', ())
               if report['rule'] == 'Heater S3 is at the lowest point of the shell side' for row in report['evidence']}
    for uid, _ in heater_ports(room):
        row = checked.get(uid, {})
        yield finding(uid, row.get('pass'), opening=row.get('opening'))


def check_m3_heater_valves(room):
    for uid, _ in heater_ports(room):
        required = ('in', 'out', 'utility_in', 'utility_out')
        actual = {key: [v['id'] for v in room.valves.values()
                        if v.get('manual') and v.get('at') == uid+'.'+key] for key in required}
        yield finding(uid, all(len(ids) == 1 for ids in actual.values()), required_ports=actual)


def check_m3_heated_return_rack(room):
    for chain in heated_returns(room):
        source, final = chain[0]['a'], chain[-1]['b']
        first_level, last_level = room.level(source), room.level(final)
        if first_level is None or last_level is None:
            yield finding(chain_label(chain), None, source=source, final_port=final)
        elif first_level < last_level:
            band = room.level_band(last_level)
            answer = rack_geometry(room.chain_path(chain), band, room.point(final))
            okay = answer['pass'] if answer is not None else (False if chain_connected(room, chain) is False else None)
            yield finding(chain_label(chain), okay, source=source, final_port=final,
                          edges=[e['id'] for e in chain], rack_band_m=band,
                          missing_path=answer is None, geometry=answer)


def trap_routes(room):
    for uid, unit in room.units.items():
        if unit['kind'] != 'automatic_steam_trap':
            continue
        inlet = [e for e in room.incoming[uid] if e['b'] == uid+'.in' and e['role'] == 'steam_condensate']
        outlet = [e for e in room.outgoing[uid] if e['a'] == uid+'.out' and e['role'] == 'steam_condensate']
        yield uid, inlet, outlet


def trap_inlet_geometry(path, source, inlet):
    rising = [i for i, step in enumerate(path) if step['ends'][1][2] > step['ends'][0][2]+EPS]
    okay = source[2] > inlet[2]+EPS and not rising if source is not None and inlet is not None and path else None
    return okay, rising


def trap_discharge_geometry(path, source, end, target, floor):
    axes = [tuple(direction(line).round(6)) for line in straight_spans(path)]
    axes = [axis for i, axis in enumerate(axes) if i == 0 or axis != axes[i-1]]
    okay = (math.dist(end, target) <= EPS and abs(end[2]-floor) <= EPS
            and len(axes) == 2 and axes[0][2] == 0 and axes[1][2] == -1
            if source is not None and end is not None and target is not None and path else None)
    return okay, axes


def check_m3_trap_inlet(room):
    for uid, inlet, _ in trap_routes(room):
        if len(inlet) != 1:
            yield finding(uid, False, inlet_edges=[e['id'] for e in inlet])
            continue
        edge = inlet[0]
        a, b = room.edge_points(edge['id'])
        path = room.edge_path(edge['id'])
        okay, rising = trap_inlet_geometry(path, a, b)
        if connected(room, edge) is False:
            okay = False
        yield finding(edge['id'], okay, trap=uid, source_z=a[2] if a else None,
                      inlet_z=b[2] if b else None, rising_pieces=[path[i]['id'] for i in rising])


def check_m3_trap_discharge(room):
    endpoints = {row['endpoint']: row for row in room.material.get('drain_endpoints', [])}
    for uid, _, outlet in trap_routes(room):
        if len(outlet) != 1:
            yield finding(uid, False, outlet_edges=[e['id'] for e in outlet])
            continue
        edge = outlet[0]
        a, b = room.edge_points(edge['id'])
        target = endpoints.get(edge['b'], {}).get('position')
        path = room.edge_path(edge['id'])
        okay, axes = trap_discharge_geometry(path, a, b, target, room.geometry['floors'][0])
        if connected(room, edge) is False:
            okay = False
        yield finding(edge['id'], okay, trap=uid, measured_end=b, designated_end=target, directions=axes)


def condenser_circuits(room):
    return room.process.get('reflux_circuits', ())


def check_m4_vertical_common(room):
    for circuit in condenser_circuits(room):
        if circuit['form'] != 'vertical':
            continue
        vessel, condenser = circuit['owner'], circuit['condenser']
        shared = [e for e in room.outgoing[vessel] if owner(e['b']) == condenser and e['role'] == 'vapor']
        separate = [e for e in room.outgoing[condenser] if e['role'] == 'condensate']
        path = room.edge_path(shared[0]['id']) if len(shared) == 1 else []
        okay = len(shared) == 1 and not separate and circuit.get('seal_junction') is None
        if okay:
            okay = chain_connected(room, shared) and bool(path)
        yield finding(condenser, okay, shared=[e['id'] for e in shared], separate=[e['id'] for e in separate])


def condenser_return(room, circuit):
    for edge in room.outgoing[circuit['condenser']]:
        if edge['role'] == 'condensate':
            yield from room.chains(edge, {'condensate'})


def check_m4_horizontal_seal(room):
    for circuit in condenser_circuits(room):
        if circuit['form'] != 'horizontal':
            continue
        condenser = circuit['condenser']
        chains = list(condenser_return(room, circuit))
        if len(chains) != 1:
            yield finding(condenser, False, liquid_paths=[chain_label(chain) for chain in chains])
            continue
        chain = chains[0]
        seal = circuit['seal_junction']
        low = room.junctions.get(seal, {}).get('position')
        target = room.point(chain[-1]['b'])
        path = room.chain_path(chain)
        drain = [e for e in room.outgoing[seal] if e['role'] == 'seal_drain']
        okay = chain[-1]['b'] == circuit['return_endpoint'] and len(drain) == 1
        if okay and path and low is not None and target is not None:
            okay = (low[2] < target[2]-EPS and all(p[2] >= low[2]-EPS for step in path for p in step['ends'])
                    and math.dist(room.edge_points(drain[0]['id'])[0], low) <= EPS
                    and connected(room, drain[0]) is True)
        elif okay:
            okay = False if chain_connected(room, chain) is False else None
        yield finding(condenser, okay, chain=chain_label(chain), seal=seal, low=low,
                      drain=[e['id'] for e in drain])


def check_m4_condenser_rack(room):
    for circuit in condenser_circuits(room):
        if circuit['form'] != 'horizontal':
            continue
        for chain in condenser_return(room, circuit):
            a, b = room.level(chain[0]['a']), room.level(chain[-1]['b'])
            if a is None or b is None:
                yield finding(chain_label(chain), None, reason='Missing floor evidence')
            elif a > b:
                seal = circuit['seal_junction']
                upstream = [e for e in chain if owner(e['a']) != seal]
                last = room.junctions.get(seal, {}).get('position')
                band = room.level_band(b)
                answer = rack_geometry(room.chain_path(upstream), band, last)
                okay = answer['pass'] if answer else (False if chain_connected(room, upstream) is False else None)
                yield finding(chain_label(chain), okay, final_port=chain[-1]['b'], seal=seal,
                              rack_band_m=band, geometry=answer)


def check_m4_condenser_coolant(room):
    for circuit in condenser_circuits(room):
        uid = circuit['condenser']
        keys = ('utility_in', 'utility_out', 'vapor')
        points = {key: room.point(uid+'.'+key) for key in keys}
        if circuit['form'] == 'horizontal':
            points['condensate'] = room.point(uid+'.condensate')
        supply = [e for e in room.incoming[uid] if e['b'] == uid+'.utility_in' and e['role'] == 'water']
        returns = [e for e in room.outgoing[uid] if e['a'] == uid+'.utility_out' and e['role'] == 'water_return']
        okay = len(supply) == len(returns) == 1
        if not all(value is not None for value in points.values()):
            okay = None if okay else False
        elif okay:
            cold = np.asarray(points['utility_out'])-points['utility_in']
            okay = cold[2] > EPS and chain_connected(room, [*supply, *returns])
            if circuit['form'] == 'horizontal':
                hot = np.asarray(points['condensate'])-points['vapor']
                okay = bool(okay and hot[:2] @ cold[:2] < -EPS)
        yield finding(uid, okay, points=points, form=circuit['form'])


def check_m4_vapor_inlet_floor(room):
    for row in room.placement.get('rows', ()):
        if row.get('role') != 'condenser_vapor' or not row.get('endpoint'):
            continue
        endpoint = row['endpoint']
        ports = valve_faces(room.valves.get(row['id'], {}))
        if not ports:
            trial = row.get('best_failure') or {}
            if 'in_m' in trial:
                ports = {'in':trial['in_m'], 'out':trial['out_m']}
        expected = room.level(endpoint)
        tested = {max((i for i, floor in enumerate(room.geometry['floors']) if floor <= p[2]+EPS), default=0)
                  for p in ports.values()}
        passed = tested == {expected} if tested and expected is not None else None
        yield finding(row['source_valve'], passed, endpoint=endpoint, expected_floor=expected,
                      tested_floors=sorted(tested), status=row['status'])


def check_m4_condenser_valves(room):
    for circuit in condenser_circuits(room):
        uid, plant = circuit['condenser'], circuit['owner']
        expected = [plant+'.vapor', uid+'.vapor', uid+'.utility_in', uid+'.utility_out', uid+'.offgas']
        if circuit['form'] == 'horizontal':
            expected += [uid+'.condensate', circuit['return_endpoint']]
        missing = [port for port in expected if sum(v.get('at') == port and v.get('manual', False)
                                                   for v in room.valves.values()) != 1]
        if circuit['form'] == 'horizontal':
            chains = list(condenser_return(room, circuit))
            observed = [role for chain in chains for edge in chain for role in edge_roles(room, edge)]
            required = ('condensate_root', 'condensate_regulation', 'condenser_return_root')
            if not roles_in_order(observed, required):
                missing.append('H3 → H4 → H5')
            for role in ('seal_drain',):
                if not any(role in edge_roles(room, edge) for edge in room.outgoing[circuit['seal_junction']]):
                    missing.append(role)
        yield finding(uid, not missing, missing_or_out_of_order=missing)


def offgas_entry_geometry(path):
    end = direction(path[-1]['ends']).tolist() if path else None
    return (end[2] < -.9 if end is not None else None), end


def check_m5_offgas_entry(room):
    for edge in room.edges.values():
        if edge['role'] != 'offgas':
            continue
        path = room.edge_path(edge['id'])
        okay, end = offgas_entry_geometry(path)
        if connected(room, edge) is False:
            okay = False
        yield finding(edge['id'], okay, endpoint=edge['a'], terminal_direction=end,
                      header=room.connections.get(edge['id'], {}).get('assigned_header'))


def check_m5_header_arrangement(room):
    headers = [r for r in room.routes.values() if r.get('role') == 'floor_header']
    groups = Counter((r['level'], r['medium']) for r in headers)
    for (level, medium), count in groups.items():
        yield finding(f'{medium} L{level}', count == 1, count=count)
    for route in headers:
        group = next((g for g in room.piping.get('service_groups', [])
                      if g['id'] == route.get('service_group')), {})
        expected = group.get('geometry_evidence', {}).get('axis_basis', {}).get('axis_index')
        if expected is None:
            decks = [s['box'] for s in room.geometry['surfaces'] if s['level'] == route['level']]
            if decks:
                width, depth = (max(box[i+3] for box in decks)-min(box[i] for box in decks)
                                for i in (0, 1))
                expected = 0 if width+EPS >= depth else 1
        spans = [segment for segment in room.route_segments([route['id']]).values()
                 if segment['kind'] == 'pipe']
        actual = sorted({int(np.argmax(abs(direction(segment['ends'])))) for segment in spans})
        yield finding(route['id'], actual == [expected] if expected is not None and spans else None,
                      expected_axis=expected, axes=actual)


def check_m5_header_bottoms(room):
    levels = {}
    for route in room.routes.values():
        if route.get('role') != 'floor_header':
            continue
        heights = [min(point[2] for point in part['ends'])-room.pieces[part['id']]['radius']
                   for part in room.route_segments([route['id']]).values() if part['kind'] == 'pipe']
        if heights:
            levels.setdefault(route['level'], {})[route['id']] = min(heights)
    for level, bottoms in levels.items():
        if len(bottoms) > 1:
            yield finding(f'L{level}', max(bottoms.values())-min(bottoms.values()) <= EPS, bottoms_m=bottoms)


def check_m5_station_ends(room):
    for row in room.measure.get('header_to_main', ()):
        route_ids = set(row.get('route_ids', ()))
        manual = [v for rid in route_ids for v in room.routes.get(rid, {}).get('valves', ())
                  if v in room.valves]
        medium = row.get('medium')
        if medium in ('offgas', 'water_return'):
            yield finding(row['header'], row.get('connected') is True and not manual
                          and row.get('actual_bend_count') == 0,
                          medium=medium, valves=manual, bends=row.get('actual_bend_count'))
        else:
            direct = row.get('actual_same_vertical_plane')
            okay = row.get('connected') is True and direct is True if direct is not None else None
            yield finding(row['header'], okay, medium=medium, valves=manual, same_vertical_plane=direct)
    temporary = room.measure.get('workstation_temporary_outlets', {})
    if 'actual_verified_native_count' in temporary:
        count = temporary['actual_verified_native_count']
        yield finding('previous temporary outlet', count == 0, observed=count)


def check_m5_cold_utility_approach(room):
    for circuit in condenser_circuits(room):
        uid = circuit['condenser']
        for edge in room.incoming[uid]+room.outgoing[uid]:
            if edge['role'] not in ('water', 'water_return'):
                continue
            path = room.edge_path(edge['id'])
            axis = direction(path[-1]['ends']).tolist() if path else None
            # A selected route alone cannot show whether a shorter or top-down
            # route from the same source was feasible against the same obstacles.
            measured = room.connections.get(edge['id'], {})
            options, selected = measured.get('feasible_approaches'), measured.get('selected_approach')
            okay = None
            if options and selected:
                def preference(option):
                    below_header = not option['drops_from_above'] if option['crosses_above_header'] else False
                    return below_header, option['length_m']
                okay = preference(selected) == min(map(preference, options))
            yield finding(edge['id'], okay, observed_entry=axis,
                          reason='No record of other feasible connections on the same model' if okay is None else '')


def check_m6_connected(room):
    for edge in room.edges.values():
        yield finding(edge['id'], connected(room, edge), source=edge['a'], target=edge['b'])


def check_m6_orthogonal(room):
    for part in room.pieces.values():
        if not part.get('piping') or part.get('kind') != 'pipe':
            continue
        ends = line_ends(part)
        if ends:
            yield finding(part['id'], sum(abs(direction(ends)) > EPS) == 1, ends=ends)
    for elbow in room.elbows:
        angle = elbow.get('angle_degrees')
        yield finding(elbow['id'], abs(angle-90.) <= EPS if angle is not None else None, angle=angle)
    for error in room.measure.get('directions', {}).get('errors', ()):
        if error.get('code') in ('PG17_TURN_WITHOUT_ELBOW', 'PG17_EXPECTED_ELBOW_MISSING'):
            yield finding(error.get('component', 'bend fittings'), False, measurement=error)


def check_m6_pipe_sizes(room):
    # The sizing contract is the record of the project's nominal-size table before piping.
    upstream = {e['id']: e for e in (room.original or {}).get('edges', [])}
    service_diameters = {point['id']: point['dn']
                         for point in room.piping.get('requirements', {}).get('service_points', ())}
    for edge in room.edges.values():
        source = upstream.get(edge['id'])
        routes = [room.routes[name] for name in room.route_names(edge['id']) if name in room.routes]
        main = [r for r in routes if r.get('role') in ('material_connection', 'equipment_connection',
                                                       'vessel_bottom_side')]
        measured = room.connections.get(edge['id'], {})
        if edge['id'] in service_diameters:
            sizes = sorted({r['dn'] for r in main if r.get('dn') is not None})
        else:
            sizes = sorted(set(measured.get('actual_route_dns', [])))
        expected = service_diameters.get(edge['id'], source.get('dn') if source else None)
        yield finding(edge['id'], sizes == [expected] if expected and sizes else None,
                      expected_dn=expected, actual_dns=sizes)


def check_m6_clearance(room):
    for key in ('actual_material_penetration_count', 'geometry_measurement_error_count'):
        count = room.measure.get(key)
        yield finding(key, count == 0 if count is not None else None, count=count)


def check_m6_flange_fit(room):
    for label in room.measure.get('confirmed_rules', ()):
        if label['rule'] == 'Same-port flanges mate at the actual faces and clear bores':
            for row in label['evidence']:
                yield finding(row['endpoint'], row.get('pass'), actual_mating=row)


def check_m6_fixed_layout(room):
    if room.original_layout is None:
        yield finding('original layout', None)
        return
    orig = room.original_layout
    for key in ('floors', 'footprint'):
        yield finding(key, orig.get(key) == room.geometry.get(key))
    for pose in orig['poses']:
        uid = pose['id']
        actual = room.units.get(uid)
        if actual is None:
            yield finding(uid, False)
            continue
        before, after = pose.get('center'), actual.get('center')
        angle, rotation = pose.get('rotation_deg'), actual.get('rotation_deg')
        okay = (math.dist(before, after) <= EPS and pose['level'] == actual.get('level')
                if before is not None and after is not None else None)
        yield finding(uid, okay, saved=after, original=before, saved_rotation=rotation, original_rotation=angle)


def check_k1_frozen_graph(room):
    if room.original is None:
        yield finding('saved process definition', None)
        return
    expected = {e['id']: (e['a'], e['b'], e['role'], e['dn']) for e in room.original['edges']}
    actual = {e['id']: (e['a'], e['b'], e['role'], e['dn']) for e in room.edges.values()}
    yield finding('process piping', expected == actual,
                  changed_edges=sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)))
    for source in room.source_valves.values():
        for actual_valve in room.installed[source['id']]:
            if source.get('placement_rule') == 'at_main_tee':
                continue
            yield finding(source['id'], (source['edge'], valve_role(source)) ==
                          (actual_valve.get('edge') or actual_valve.get('source_requirement'), valve_role(actual_valve)))


def check_k1_main_rank(room):
    plans = room.material.get('network_plan', {}).get('ranking', ())
    ranks = {'recycle': 1, 'bypass': 2}
    for row in plans:
        arms = [room.edges[edge] for edge in row['arms']]
        scores = [(-e['dn'], ranks.get(e['role'], 0)) for e in arms]
        # Saved ranking has no independent reference lengths for the final tie.
        yield finding(row['junction'], scores == sorted(scores) and row['main'] == row['arms'][:2],
                      arms=row['arms'], main=row['main'])


def run_order(room):
    runs = room.material.get('network_runs', ())
    emitted = room.material.get('pipe_emission_order', ())
    return runs, {edge: i for i, edge in enumerate(emitted)}


def check_k1_main_before_branch(room):
    runs, emitted = run_order(room)
    parents = {joint: run for run in runs for joint in run['junctions']}
    for run in runs:
        for end in (run['start'], run['end']):
            if end['kind'] != 'tap':
                continue
            main = parents.get(end['junction'])
            branch_at = emitted.get(run['edges'][0])
            main_at = emitted.get(main['edges'][-1]) if main else None
            yield finding(end['junction']+' / '+run['id'],
                          main_at < branch_at if main_at is not None and branch_at is not None else None,
                          parent=main['id'] if main else None)


def check_k1_scheduler(room):
    runs, emitted = run_order(room)
    jointly_built = {joint: run['id'] for run in runs for joint in run['junctions']}
    for run in runs:
        if run['start']['kind'] == 'reducer_stub':
            jointly_built[run['start']['junction']] = run['id']
        if run['end']['kind'] == 'reducer_stub':
            jointly_built[run['end']['junction']] = run['id']
    dependencies = {run['id']: {jointly_built.get(endpoint['junction'])
                                for endpoint in (run['start'], run['end'])
                                if endpoint['kind'] == 'tap' or endpoint.get('large_edge') == endpoint['edge']}
                    for run in runs}
    seals = {row.get('seal_junction') for row in condenser_circuits(room)}
    order = [run for run in runs if run['edges'][0] in emitted]
    order.sort(key=lambda run: emitted[run['edges'][0]])
    pending = {run['id'] for run in runs}
    for current in order:
        available = [run for run in runs if run['id'] in pending and
                     not dependencies[run['id']] & (pending | {None})]
        unlocks = {next(iter(waiting)) for name in pending if None not in dependencies[name]
                   and len(waiting := dependencies[name] & pending) == 1}

        def rank(run):
            roles = [room.edges[e] for e in run['edges']]
            seal = max((2 if edge['role'] == 'seal_drain' else 1
                        for edge in roles if owner(edge['a']) in seals), default=0)
            stage = 0 if any(s['kind'] == 'tap' for s in (run['start'], run['end'])) else (
                1 if run['id'] in unlocks else 2)
            return seal, stage, -run['dn']

        if available:
            best = min(rank(run) for run in available)
            yield finding(current['id'], current in available and rank(current) == best,
                          selected_rank=rank(current), eligible_rank=best)
        else:
            yield finding(current['id'], None, reason='Incomplete dependency records')
        pending.remove(current['id'])
    if order:
        yield finding('Try shorter runs first within each group and diameter', None, reason='The saved records lack candidate-endpoint length estimates')


def check_k1_tee_positions(room):
    runs, _ = run_order(room)
    for run in runs:
        if not run.get('route'):
            continue
        items = [room.junctions.get(joint, {}).get('position') for joint in run['junctions']]
        for previous, current in pairwise(run['edges']):
            i = run['edges'].index(previous)
            joint = run['junctions'][i] if i < len(items) else None
            point = items[i] if joint else None
            if point is None:
                yield finding(f'{previous} / {current}', None, reason='Missing main tee coordinates')
                continue
            a, b = [room.edge_path(edge) for edge in (previous, current)]
            if run['reversed'][i]:
                a = [{**step, 'ends': step['ends'][::-1]} for step in reversed(a)]
            if run['reversed'][i+1]:
                b = [{**step, 'ends': step['ends'][::-1]} for step in reversed(b)]
            ends_match = bool(a and b and math.dist(a[-1]['ends'][1], point) <= EPS
                              and math.dist(b[0]['ends'][0], point) <= EPS)
            yield finding(joint, ends_match and np.linalg.norm(direction(a[-1]['ends'])-
                          direction(b[0]['ends'])) <= EPS if a and b else None,
                          previous=previous, next=current, point=point)


def check_k1_seal_order(room):
    _, emitted = run_order(room)
    for circuit in condenser_circuits(room):
        uid = circuit.get('seal_junction')
        if not uid:
            continue
        sequence = [*[e for e in room.incoming[uid] if e['role'] == 'condensate'],
                    *[e for e in room.outgoing[uid] if e['role'] == 'condensate'],
                    *[e for e in room.outgoing[uid] if e['role'] == 'seal_drain']]
        positions = [emitted.get(e['id']) for e in sequence]
        yield finding(uid, len(positions) == 3 and positions == sorted(positions)
                      if all(pos is not None for pos in positions) else None,
                      sequence=[e['id'] for e in sequence], emission_positions=positions)


def check_k1_attempts(room):
    for attempt in room.material.get('network_attempts', ()):
        rows = attempt.get('attempts')
        used = attempt.get('used')
        if used is None or rows is None:
            yield finding(attempt['pipe'], None)
            continue
        valid = all(row.get('status') == 'feasible' or
                    (row.get('status') == 'rejected' and row.get('reason')) for row in rows)
        yield finding(attempt['pipe'], used <= 400 and len(rows) == used and valid,
                      attempts=used, logged=len(rows))
    for row in room.piping.get('requirements', {}).get('service_points', ()):
        yield finding(row['id'], None, reason='The previous utility record lacks the full routing attempt log')


def check_k1_candidate_families(room):
    prefixes = ('direct_search', 'lead_', 'destination_rack_guide_', 'pump_front_side_guide',
                'trap_discharge_horizontal_then_down', 'shared_horizontal_plane', 'joint_riser',
                'terminal_', 'blind_', 'fixed_reducer_', 'reducer_', 'small_dn_straight_and_reducer')
    for attempt in room.material.get('network_attempts', ()):
        labels = sorted({row.get('approach', '') for row in attempt.get('attempts', ())})
        unknown = [label for label in labels if not label.startswith(prefixes)]
        yield finding(attempt['pipe'], not unknown, unrecognised=unknown)


def check_k1_candidate_preference(room):
    runs, _ = run_order(room)
    by_run = {run['id']: run for run in runs}
    for log in room.material.get('network_attempts', ()):
        candidates = [row for row in log.get('attempts', ()) if row['status'] == 'feasible']
        if len(candidates) <= 1:
            continue
        chosen = log.get('selected_attempt') or by_run.get(log['pipe'], {}).get('selected_attempt')
        if chosen is None or not all('ends_without_valve_room' in c for c in candidates):
            yield finding(log['pipe'], None, reason='Saved records lack the final selected ID or some candidate valve position scores')
            continue
        def score(row):
            return row['ends_without_valve_room'], row['elbows'], row['length_m']
        yield finding(log['pipe'], chosen == min(candidates, key=score)['attempt'])


def check_k1_reported_breaks(room):
    failures = {row.get('edge') for row in room.material.get('failures', ())
                if row.get('stage') != 'material_valve'}
    for edge_id, row in room.connections.items():
        if row.get('connected') is False:
            yield finding(edge_id, edge_id in failures if row.get('phase') == 'P2' else None,
                          recorded_as_routing_failure=edge_id in failures)


def check_k1_no_cut(room):
    cuts = room.piping.get('collision_breaks', ())
    if not cuts:
        yield finding('saved model', True)
    for cut in cuts:
        yield finding(cut.get('edge', cut.get('component', 'cut segment')), False, record=cut)


def check_k1_blind_ends(room):
    for row in room.measure.get('material', {}).get('topology', {}).get('terminals', ()):
        yield finding(row['terminal'], row.get('pass'), blind=row.get('blind'))
    for row in room.measure.get('workstation_main_ends', ()):
        yield finding(row['main'], row.get('pass'))


def check_k1_tee_clearance(room):
    runs, _ = run_order(room)
    for run in runs:
        for previous, current in pairwise(run['edges']):
            prev, nxt = room.edge_path(previous), room.edge_path(current)
            a, b = room.edges[previous], room.edges[current]
            joint = owner(a['a'] if run['reversed'][run['edges'].index(previous)] else a['b'])
            point = room.junctions.get(joint, {}).get('position')
            if not (prev and nxt and point):
                yield finding(joint, None, reason='Straight sections adjacent to the tee are missing')
                continue
            # Valve faces do not change straight geometry. Fittings and elbows end the run.
            spans = []
            for steps in (prev, nxt):
                from_joint = steps if math.dist(steps[0]['ends'][0], point) <= EPS else [
                    {**step, 'ends': step['ends'][::-1]} for step in reversed(steps)]
                first = next(iter(straight_spans(from_joint)), None)
                spans.append(math.dist(*first) if first and math.dist(first[0], point) <= EPS else 0.)
            radius = max((room.piping['config']['pipe_outer_diameter_mm'][str(e['dn'])]/2000
                          for e in (a, b)))
            minimum = 2*radius+.05
            yield finding(joint, min(spans)+EPS >= minimum, straight_m=spans, required_m=minimum)


def check_k1_single_dn(room):
    for run in room.material.get('network_runs', ()):
        if not run.get('route'):
            continue
        # An equipment adapter remains a different route before its reducer.
        main_names = [r['id'] for r in room.routes.values() if r.get('network_run') == run['id']]
        sizes = sorted({p['dn'] for p in room.route_segments(main_names).values()
                        if p['kind'] == 'pipe' and p.get('dn') is not None})
        applicable = len(run['edges']) > 1
        if applicable:
            yield finding(run['id'], sizes == [run['dn']] if sizes else None, expected_dn=run['dn'], sizes=sizes)
    for reducer in room.piping.get('preplaced_source_reducers', ()):
        endpoint = reducer.get('endpoint')
        if endpoint:
            yield finding(endpoint, None, reason='The reserved reducer-side length was not recorded before rerouting')


def check_k1_post_route_valves(room):
    stages = room.material.get('generation_order', {}).get('stages', ())
    pipe_index = [i for i, stage in enumerate(stages) if stage in ('main_runs_port_to_port', 'branches_onto_built_pipes')]
    valve_index = [i for i, stage in enumerate(stages) if stage in
                   ('fit_all_valves_on_actual_runs', 'reserve_valve_intervals_on_emitted_straights')]
    yield finding('process piping stage', max(pipe_index) < min(valve_index) if pipe_index and valve_index else None,
                  recorded_stages=stages)


def check_k1_takeoff_spacing(room):
    for row in room.measure.get('workstation_takeoff_spacing', {}).get('pairs', ()):
        distance = row.get('separation_m')
        yield finding(' / '.join(row.get('connections', ())), distance+EPS >= .3 if distance is not None else None,
                      distance_m=distance, required_m=.3)


def check_k2_same_floor(room):
    for edge in room.edges.values():
        if edge['role'] not in SERVICE_ROLES:
            continue
        in_header = room.units[owner(edge['a'])]['kind'] == 'header'
        out_header = room.units[owner(edge['b'])]['kind'] == 'header'
        if not (in_header or out_header):
            continue
        measured = room.connections.get(edge['id'], {})
        endpoint = edge['b'] if in_header else edge['a']
        if room.units[owner(endpoint)]['kind'] == 'safety_valve':
            device_side = [e['a'] for e in room.incoming[owner(endpoint)] if e['role'] == 'emergency_discharge']
            if len(device_side) == 1:
                endpoint = device_side[0]
        expected = room.level(endpoint)
        assigned = measured.get('assigned_header', measured.get('header'))
        route = room.routes.get(assigned, {})
        if not route and measured.get('routes'):
            # The automatic safety-valve leg is measured in the material record.
            options = [r for r in room.routes.values() if r.get('role') == 'floor_header'
                       and edge['id'] in r.get('point_ids', ())]
            route = options[0] if len(options) == 1 else {}
        observed = route.get('level')
        status = expected == observed and measured.get('connected') is True if observed is not None else None
        yield finding(edge['id'], status, endpoint=endpoint, level=expected, header_level=observed)


def check_k3_positions(room, placement_rule):
    for row in room.placement.get('rows', ()):
        if row.get('placement_rule') != placement_rule or row.get('status') != 'placed':
            continue
        source = room.source_valves.get(row.get('source_valve'))
        actual = room.valves.get(row['id'])
        faces = valve_faces(actual) if actual else {}
        if not source or len(faces) != 2:
            yield finding(row['id'], None, reason='Missing valve port positions or original valve role')
            continue
        # Physical flange faces may be offset from the logical equipment mouth.
        graph = {**(room.original or room.process), 'units': room.units}
        policy = {'role': row['role'], 'placement_rule': placement_rule,
                  'position_endpoint': row.get('endpoint')}
        if placement_rule == 'free':
            chain = room.edge_path(row['edge']) if row.get('edge') in room.edges else []
            yield finding(row['id'], any(p['id'] == row['id'] for p in chain) if chain else None)
            continue
        if placement_rule == 'after_split' and row.get('edge') in room.edges:
            edge = room.edges[row['edge']]
            device = owner(edge['a'])
            if (room.units[device]['kind'] == 'exchanger' and
                    not any(valve_role(v) == 'heater_outlet' for v in room.valves_on(edge['id']))):
                yield finding(row['id'], None, reason='Heater outlet valve pending. Valve-group checks record its required order.')
                continue
        answer = position_check(graph, source, room.role_context, faces, policy,
                                allow_at_port_elbow=True)
        okay = answer['valid'] if answer.get('reference_progress_m') is not None else None
        yield finding(row['id'], okay, rule=placement_rule, edge=row.get('edge'), between=answer.get('between', []))


def check_k3_common_port(room):
    for uid, unit in room.units.items():
        manifold = unit.get('return_manifold')
        if not manifold:
            continue
        trunk = room.edges.get(manifold['trunk'])
        branches = [room.edges[edge] for edge in manifold['branches']]
        roles = {e['id']: edge_roles(room, e) for e in branches}
        root = edge_roles(room, trunk) if trunk else []
        okay = (bool(trunk) and trunk['b'] == uid+'.liquid_return'
                and root.count('common_root') == 1
                and all(owner(e['b']) == manifold['junction'] for e in branches)
                and all(values.count('return_root') == 1 or values.count('feed_root') == 1
                        for values in roles.values()))
        yield finding(uid, okay, common=root, branches=roles)


def check_k3_prohibited_valves(room):
    for edge in room.edges.values():
        if edge['role'] == 'emergency_discharge':
            valves = [v['id'] for v in room.valves_on(edge['id']) if v.get('manual')]
            yield finding(edge['id'], not valves, manual_valves=valves)
    for route in room.routes.values():
        if route.get('medium') == 'offgas' and route.get('role') != 'equipment_connection':
            valves = [v for v in route.get('valves', ()) if v in room.valves]
            yield finding(route['id'], not valves, manual_valves=valves)


def check_k3_station_valves(room):
    for row in room.piping.get('end_bundle_station', {}).get('connections', []):
        name, medium = row['id'], row['medium']
        if medium in ('offgas', 'water_return'):
            continue
        actual = room.valves.get(name)
        faces = valve_faces(actual) if actual else {}
        if len(faces) != 2:
            yield finding(name, False, reason='Missing root valve between the floor header and the main')
            continue
        tap = row.get('main_tap')
        route_names = [r['id'] for r in room.routes.values()
                       if r.get('connection_id') == name or name in r.get('valves', ())]
        path = room.route_segments(route_names)
        from .piping_material_checks import _actual_path
        approach = min((_actual_path(tap, face, path.values()) for face in faces.values()),
                       key=lambda steps: sum(math.dist(*p['ends']) for p in steps) if steps else math.inf)
        intervening = [p['id'] for p in approach if path[p['id']]['kind'] in ('elbow', 'valve')]
        yield finding(name, not intervening if approach else None, intervening=intervening)


def measured_seat(spans, length, vertical=False):
    """A candidate supplies unbroken same-DN spans cut at actual fittings."""
    available = [math.dist(*ends) for ends in spans
                 if not vertical or abs(direction(ends)[2]) > .99]
    longest = max(available, default=0.)
    return {'available_m': longest, 'required_m': length, 'fits': longest+EPS >= length}


def valve_route_evidence(room, row):
    edge = row.get('edge')
    if edge in room.edges:
        return room.route_names(edge), connected(room, room.edges[edge])
    station = [entry for entry in room.piping.get('end_bundle_station', {}).get('connections', ())
               if entry['id'] in (row['id'], row.get('source_valve'))]
    if len(station) != 1:
        return [], None
    station = station[0]
    names = [r['id'] for r in room.routes.values() if r.get('connection_id') == station['id']]
    measured = [item for item in room.measure.get('header_to_main', ())
                if item.get('header') == station['header_id']]
    return names, measured[0].get('connected') if len(measured) == 1 else None


def valve_seat_findings(room):
    for row in room.placement.get('rows', ()):
        subject = row['id']
        length, dn = row.get('installation_length_m'), row.get('dn')
        vertical = row.get('placement_rule') == 'top_vertical'
        names, connection = valve_route_evidence(room, row)
        parts = room.route_segments(names)
        spans = [part['ends'] for part in parts.values() if part['kind'] == 'pipe' and part['dn'] == dn]
        pending = row.get('status') == 'pending'
        reason = row.get('reason', '')
        no_slot = (row.get('best_failure', {}).get('no_candidate') and
                   any(phrase in reason for phrase in ('valve space on a matching-diameter straight', 'valve space on a vertical straight', 'no main tee reserved on the same floor',
                                                       'saved valve section', 'recorded valve section')))
        if connection is False and pending:
            yield finding(subject, None, attribution='pipe_open', edge=row.get('edge'))
            continue
        if not length or not dn:
            yield finding(subject, None, attribution='insufficient_evidence', edge=row.get('edge'))
            continue
        size = measured_seat(spans, length, vertical)
        if not pending:
            actual = room.valves.get(subject)
            faces = valve_faces(actual) if actual else {}
            if len(faces) != 2:
                yield finding(subject, None, edge=row.get('edge'))
                continue
            ports = (faces['in'], faces['out'])
            size = measured_seat([ports] if actual['dn'] == dn else [], length, vertical)
            yield finding(subject, size['fits'], edge=row.get('edge'), **size)
        elif connection is not True:
            yield finding(subject, None, attribution='insufficient_evidence', edge=row.get('edge'), **size)
        else:
            # The measurement above is an upper bound before applying role and fitting stops.
            # The geometric seat trace retains the actual eligible subset and trial evidence.
            tried = bool(row['attempts'] or any(not item.get('no_position', True)
                                                 for item in row.get('orientation_refusals', ())))
            if no_slot or (spans and not size['fits'] and not tried):
                yield finding(subject, False, attribution='no_seat', edge=row.get('edge'),
                              saved_reason=reason, **size)
            else:
                yield finding(subject, True if tried else None, attribution='other' if tried else 'insufficient_evidence',
                              edge=row.get('edge'), **size)


def check_valve_reservation(room):
    yield from valve_seat_findings(room)


def valve_packing_deficit(total_straight, lengths):
    """A lower bound. Fittings and operating space can only increase the deficit."""
    occupied = 0.
    for count, length in enumerate(sorted(lengths), 1):
        occupied += length
        if occupied > total_straight+EPS:
            return len(lengths)-count+1
    return 0


def check_shared_straight(room):
    groups = {}
    for row in room.placement.get('rows', ()):
        edge = row.get('edge')
        if (row.get('placement_rule') in ('before_join', 'after_split', 'before_seal', 'free')
                and edge in room.edges and row.get('dn') and row.get('installation_length_m')):
            groups.setdefault((edge, row['dn']), []).append(row)
    for (edge, dn), rows in groups.items():
        if len(rows) < 2 or connected(room, room.edges[edge]) is not True:
            continue
        path = room.edge_path(edge)
        length = sum(math.dist(*step['ends']) for step in path
                     if step['dn'] == dn and step['kind'] in ('pipe', 'valve'))
        deficit = valve_packing_deficit(length, [row['installation_length_m'] for row in rows])
        yield finding(edge, deficit == 0 if path else None, available_m=length,
                      required_m=sum(r['installation_length_m'] for r in rows), minimum_deficit=deficit,
                      valves=[r['id'] for r in rows])


VESSEL_PORTS = {
    'small': {'liquid_return': 'c', 'feed': 'c', 'condensate_return': 'd',
              'vapor': 'h', 'offgas': 'h', 'nitrogen': 'g', 'safety': 'f', 'bottom': 'j'},
    'medium': {'liquid_return': 'g1', 'feed': 'g1', 'condensate_return': 'g2',
               'vapor': 'e2', 'offgas': 'e2', 'nitrogen': 'k', 'safety': 'g4', 'bottom': 'a'},
}
VESSEL_PORTS['large'] = VESSEL_PORTS['medium']


def check_k4_vessel_ports(room):
    catalogue = {row['id']: row for row in room.geometry.get('catalogue_devices', [])}
    for uid, unit in room.units.items():
        if unit['kind'] != 'vessel' or not unit.get('reactor') or unit.get('orientation') != 'vertical':
            continue
        tags = VESSEL_PORTS.get(unit.get('size_tier'))
        if not tags:
            continue
        ports = {**unit.get('ports', {}), **catalogue.get(uid, {}).get('ports', {})}
        used = {point.rsplit('.', 1)[-1] for edge in room.incoming[uid]+room.outgoing[uid]
                for point in (edge['a'], edge['b']) if owner(point) == uid}
        for key in used & tags.keys():
            port = ports.get(key, {})
            tag = port.get('source_tag', port.get('catalogue_tag'))
            yield finding(uid+'.'+key, tag.casefold() == tags[key].casefold() if tag else None,
                          expected_tag=tags[key], actual_tag=tag)


def check_k4_vapor_offgas(room):
    for uid, unit in room.units.items():
        if unit['kind'] != 'vessel' or not unit.get('reactor'):
            continue
        condenser = next((r['condenser'] for r in condenser_circuits(room) if r['owner'] == uid), None)
        offgas = [e for e in room.outgoing[uid] if e['role'] == 'offgas']
        if condenser:
            vents = [e for e in room.outgoing[condenser] if e['role'] == 'offgas']
            vapor = [e for e in room.outgoing[uid] if e['role'] == 'vapor' and owner(e['b']) == condenser]
            yield finding(uid, not offgas and len(vapor) == len(vents) == 1,
                          direct_vents=[e['id'] for e in offgas], condenser_vents=[e['id'] for e in vents])
        else:
            yield finding(uid, len(offgas) == 1, vents=[e['id'] for e in offgas])


def check_k4_common_feed_order(room):
    _, emitted = run_order(room)
    mains = {row['junction']: row for row in room.material.get('network_plan', {}).get('ranking', ())}
    for uid, unit in room.units.items():
        manifold = unit.get('return_manifold')
        if not manifold:
            continue
        trunk = manifold['trunk']
        branches = [room.edges[k] for k in manifold['branches']]
        external = [e for e in branches if e['role'] in ('feed', 'external_feed')]
        selected = [e for e in mains.get(manifold['junction'], {}).get('main', []) if e != trunk]
        if len(selected) != 1 or not branches:
            yield finding(uid, None, reason='Missing main selection record for a shared equipment port')
            continue
        picked = room.edges[selected[0]]
        compared = external if external else branches
        okay = picked in compared and picked['dn'] == max(e['dn'] for e in compared)
        times = [emitted.get(e['id']) for e in branches if e['id'] != picked['id']]
        at = emitted.get(trunk)
        yield finding(uid, okay and all(at <= pos for pos in times) if at is not None and
                      all(pos is not None for pos in times) else None, selected=picked['id'], main=trunk)
