"""Check connected process pipes and installed valves in the room model."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

import mujoco
import numpy as np

from . import piping_checks as common
from .piping_assets import inspect_asset
from .building_equipment_checks import EPS


def _error(code, **detail):
    return {'code': code, **detail}


def _section(rows, errors, **detail):
    return {'status': 'fail' if errors else 'pass', 'rows': rows,
            'errors': errors, 'violations': errors, **detail}


def _source_edge(route):
    return route.get('source_edge', route.get('source_requirement'))




def _axis_intersection(point, direction, segment):
    """Return signed metres to a genuine intersection with a finite pipe axis."""
    a, b = map(np.asarray, segment['ends']); vector = b-a
    length = float(np.linalg.norm(vector))
    if length <= EPS:
        return None
    axis = vector/length
    if np.linalg.norm(np.cross(direction, axis)) <= EPS:
        return None
    solution, *_ = np.linalg.lstsq(np.column_stack((direction, -axis)), a-point, rcond=None)
    distance, along = map(float, solution)
    if -EPS <= along <= length+EPS and np.linalg.norm(point+distance*direction-(a+along*axis)) <= EPS:
        return distance
    return None


def _straight_reaches(point, direction, distance, segments):
    """A valve reservation must be a continuous straight pipe/adaptor run."""
    intervals = []
    for segment in segments:
        a, b = map(np.asarray, segment['ends'])
        if (np.linalg.norm(np.cross(a-point, direction)) > EPS
                or np.linalg.norm(np.cross(b-point, direction)) > EPS):
            continue
        values = sorted((float((a-point)@direction), float((b-point)@direction)))
        intervals.append(values)
    reached = 0.
    for low, high in sorted(intervals):
        if low > reached+EPS:
            break
        if high >= -EPS:
            reached = max(reached, high)
    return reached+EPS >= distance, reached


def _installation_spaces(graph, expected, actual, by_route, valve_rows, components):
    rows, errors = [], []
    expected_ids = {v['id'] for v in expected['manual_valves']}
    valid = {v['id']: v for v in valve_rows if v['status'] == 'pass'}
    segments = [s for parts in by_route.values() for s in parts]
    seen_segments = {}; segments = [seen_segments.setdefault(s['id'], s) for s in segments
                                    if s['id'] not in seen_segments]
    ports = expected['ports']
    for valve in graph.get('valves', []):
        if valve.get('source_valve') not in expected_ids:
            continue
        measured = valid.get(valve['id'])
        if measured is None:
            continue  # Native-package failure is already a hard error.
        asset = inspect_asset(valve['dn'], valve.get('operator'))
        length = math.dist(*measured['actual_ports'].values())
        for port_key, origin in measured['actual_ports'].items():
            point = np.asarray(origin); direction = np.asarray(measured['actual_port_normals'][port_key])
            candidates = []
            for endpoint, port in ports.items():
                q = np.asarray(port['position']); along = float((q-point)@direction)
                if along >= -EPS and np.linalg.norm(q-point-along*direction) <= EPS:
                    candidates.append((max(0., along), 'equipment_nozzle', endpoint, q.tolist()))
            # Two native valves can share one straight run. The neighbouring
            # verified native face is its real boundary; searching only for
            # an elbow beyond that second valve would leave no evidence.
            for peer_id, peer in valid.items():
                if peer_id==valve['id']:continue
                for peer_port,position in peer['actual_ports'].items():
                    q=np.asarray(position);along=float((q-point)@direction)
                    normal=np.asarray(peer['actual_port_normals'][peer_port])
                    if (along>EPS and np.linalg.norm(q-point-along*direction)<=EPS
                            and float(normal@direction)<-1+EPS):
                        candidates.append((along,'native_valve_face',peer_id+'.'+peer_port,q.tolist()))
            for segment in segments:
                distance = _axis_intersection(point, direction, segment)
                if distance is None or distance < -EPS:
                    continue
                q = point+distance*direction
                interior = min(math.dist(q, p) for p in segment['ends']) > EPS
                # A tee is measured to the actual main outer wall. An emitted
                # sharp turn is measured to its actual connection end plane;
                # no historical guessed bend radius is introduced here.
                gap = distance-segment.get('radius', 0.) if interior else distance
                candidates.append((max(0., gap), 'main_pipe' if interior else 'elbow_connection',
                                   segment['id'], q.tolist()))
            # Only adjacent objects connected by the real collinear run count;
            # unrelated equipment elsewhere on the same infinite line do not.
            linked = []
            for gap, kind, target, position in candidates:
                connected, straight = _straight_reaches(point, direction, gap, segments)
                if connected:
                    linked.append((gap, kind, target, position, straight))
            if not linked:
                # A genuinely blind-ended terminal has no neighbouring elbow,
                # device or main on its capped side. Record that applicability
                # from an actual blind, not from a missing reference label.
                caps = []
                for name, component in components.items():
                    shape = actual.get(name)
                    cap = common._cylinder(shape) if shape else None
                    if component.get('kind') != 'piping_blind' or cap is None:
                        continue
                    for q in cap['ends']:
                        q = np.asarray(q); along = float((q-point)@direction)
                        if along >= -EPS and np.linalg.norm(q-point-along*direction) <= EPS:
                            straight, run = _straight_reaches(point, direction, along, segments)
                            if straight:
                                caps.append({'component': name, 'position': q.tolist(), 'straight_run_m': run})
                row = {'valve': valve['id'], 'source_valve': valve['source_valve'], 'port': port_key,
                       'actual_face': origin, 'actual_normal': direction.tolist(), 'installation_length_m': length,
                       'package': asset['package'], 'applicable': False if caps else None,
                       'blind_terminal_evidence': caps, 'pass': bool(caps)}
                rows.append(row)
                if not caps:
                    errors.append(_error('P2_VALVE_SPACE_EVIDENCE_MISSING', **row))
                continue
            gap, kind, target, position, straight = min(linked, key=lambda x: x[0])
            okay = gap+EPS >= length
            row = {'valve': valve['id'], 'source_valve': valve['source_valve'], 'port': port_key,
                   'actual_face': origin, 'actual_normal': direction.tolist(), 'installation_length_m': length,
                   'gap_m': gap, 'neighbor_kind': kind, 'neighbor': target, 'neighbor_position': position,
                   'straight_run_m': straight, 'package': asset['package'], 'applicable': True, 'pass': okay}
            rows.append(row)
            if not okay:
                errors.append(_error('P2_ONE_VALVE_SPACE', **row))
    return _section(rows, errors, rule='PG-09',
                    measurement='Measure from both actual valve faces to the neighboring equipment port, turn face or main wall. Use verified native valve length.')




def _records(value):
    if isinstance(value, dict):
        return [dict(record, id=record.get('id', key)) for key, record in value.items()]
    return value if isinstance(value, list) else []


def _terminal_points(expected, material, actual, components):
    """Locate the model's closed boundary, without inventing an external plant."""
    rows, errors, endpoints = [], [], {}
    output = {t['id']: t for t in _records(material.get('terminals', []))}
    for terminal in expected['terminals']:
        saved = output.get(terminal['id'], {})
        name = saved.get('blind')
        if isinstance(name, dict):
            name = name.get('id')
        shape = actual.get(name); cylinder = common._cylinder(shape) if shape else None
        position = saved.get('boundary_position')
        okay = bool(cylinder and components.get(name, {}).get('kind') == 'piping_blind'
                    and position is not None and any(common._near(position, p) for p in cylinder['ends']))
        row = {'phase': 'P2', 'terminal': terminal['id'], 'type': terminal['type'],
               'boundary_endpoint': terminal['boundary_endpoint'], 'blind': name,
               'actual_blind': cylinder, 'boundary_position': position, 'pass': okay}
        rows.append(row)
        if okay:
            endpoints[terminal['boundary_endpoint']] = list(position)
        else:
            errors.append(_error('P2_TERMINAL_REAL_BLIND_MISSING', **row))
    return endpoints, rows, errors


def _endpoint_positions(expected, material, terminal_points):
    points = {key: list(port['position']) for key, port in expected['ports'].items()}
    points.update(terminal_points)
    saved = {j['id']: j for j in _records(material.get('junctions', []))}
    rows, errors = [], []
    for junction in expected['junctions']:
        position = saved.get(junction['id'], {}).get('position')
        okay = (isinstance(position, (list, tuple)) and len(position) == 3
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in position))
        row = {'phase': 'P2', 'junction': junction['id'], 'source_ports': junction['ports'],
               'source_edges': junction['edge_ids'], 'position': position, 'pass': okay}
        rows.append(row)
        if okay:
            for endpoint in junction['ports']:
                points[endpoint] = list(position)
        else:
            errors.append(_error('P2_SHARED_NODE_MISSING', **row))
    return points, rows, errors


def _material_connectivity(frozen, graph, scene, piping, expected, actual, by_route, valve_rows,
                           valve_segments, *, measured_flange_rows=()):
    material = piping.get('material', {})
    components = {c['id']: c for c in scene['components']}
    terminal_points, terminal_rows, errors = _terminal_points(expected, material, actual, components)
    positions, junction_rows, node_errors = _endpoint_positions(expected, material, terminal_points)
    errors.extend(node_errors)
    edges = {e['id']: e for e in expected['edges']}
    routes = defaultdict(list)
    for route in piping.get('routes', []):
        source = _source_edge(route)
        if source in edges:
            routes[source].append(route)
        elif route.get('phase') == 'P2' or str(route.get('id', '')).startswith('P2_'):
            errors.append(_error('P2_UNEXPECTED_SOURCE_EDGE', route=route.get('id'), source_edge=source))
    valid = {v['id']: v for v in valve_rows if v['status'] == 'pass'}
    segments_by_valve = {v['valve']: v for v in valve_segments}
    actual_valves = {v['id']: v for v in graph.get('valves', [])}
    sources = defaultdict(list)
    for valve in graph.get('valves', []):
        source = valve.get('source_valve')
        if source is not None:
            sources[source].append(valve)
    local_parts = {}
    flanged = {r['component']: r for r in measured_flange_rows
               if r.get('kind')=='measured_equipment_flange_pair' and r.get('phase')=='P2' and r.get('pass')}
    catalogued={r['component']:r for r in measured_flange_rows
                if r.get('kind')=='measured_catalogue_flange' and r.get('pass')}
    connections = []
    for edge in expected['edges']:
        linked = routes[edge['id']]
        parts = [part for route in linked for part in by_route.get(route['id'], [])]
        ids = list(dict.fromkeys(vid for route in linked for vid in route.get('valves', [])))
        parts += [segments_by_valve[vid] for vid in ids if vid in segments_by_valve]
        local_parts[edge['id']] = parts
        a, b = positions.get(edge['a']), positions.get(edge['b'])
        choices = edge['dn_candidates']
        declared = bool(linked) and all(route.get('dn') in choices
            and route.get('medium_identity', route.get('medium')) == edge['medium_identity']
            and route.get('a', edge['a']) == edge['a'] and route.get('b', edge['b']) == edge['b']
            for route in linked)
        connected = bool(a is not None and b is not None and declared and common._connected(a, [b], parts))
        row = {'phase': 'P2', 'requirement': edge['id'], 'source_edge': edge['id'],
               'a': edge['a'], 'b': edge['b'], 'role': edge['role'], 'medium': edge['medium_identity'],
               'expected_dn_candidates': choices, 'actual_route_dns': [r.get('dn') for r in linked],
               'actual_endpoints': [a, b], 'routes': [r['id'] for r in linked], 'connected': connected,
               'execution': edge['execution'], 'scope': 'Physical termination with an isolated blind boundary'}
        connections.append(row)
        if not declared:
            errors.append(_error('P2_EDGE_DN_OR_IDENTITY', **row))
        if not connected:
            errors.append(_error('P2_REQUIRED_CONNECTION_NOT_CONNECTED', **row))
        for endpoint, port in expected['ports'].items():
            if endpoint in (edge['a'], edge['b']):
                continue
            if any(common._on_segment(port['position'], *part['ends']) for part in parts):
                errors.append(_error('P2_UNEXPECTED_PHYSICAL_ENDPOINT', requirement=edge['id'],
                                     endpoint=endpoint, actual_position=port['position']))
    manual = []
    expected_valve_ids = {v['id'] for v in expected['manual_valves']}
    for demand in expected['manual_valves']:
        matches = sources.get(demand['id'], [])
        edge = edges[demand['edge']]
        actual_ids = [v['id'] for v in matches]
        good = len(matches) == 1
        detail = []
        for valve in matches:
            measured = valid.get(valve['id']); part = segments_by_valve.get(valve['id'])
            identity = bool(measured and valve.get('at') == demand['at']
                and valve.get('source_edge', valve.get('edge')) == demand['edge']
                and valve.get('dn') in demand['dn_candidates'])
            a, b = positions.get(edge['a']), positions.get(edge['b'])
            parts = local_parts[edge['id']]
            in_series = bool(a is not None and b is not None and part
                and common._connected(a, [b], parts)
                and not common._connected(a, [b], [p for p in parts if p.get('valve') != valve['id']]))
            root_reachable = bool(measured and a is not None and b is not None
                and all(common._connected(a, [position], parts) for position in measured['actual_ports'].values()))
            detail.append({'valve': valve['id'], 'native_identity_valid': identity,
                           'in_series_not_bypassed': in_series, 'actual_ports_connected': root_reachable})
            good = good and identity and in_series and root_reachable
        row = {'phase': 'P2', 'requirement': demand['id'], 'source_valve': demand['id'],
               'source_edge': demand['edge'], 'endpoint': demand['at'], 'role': demand['role'],
               'expected_dn_candidates': demand['dn_candidates'], 'installed': bool(good),
               'valves': actual_ids, 'details': detail}
        manual.append(row)
        if not good:
            errors.append(_error('P2_REQUIRED_MANUAL_VALVE_MISSING_OR_MISMATCH', **row))
    for source, valves in sources.items():
        if source not in expected_valve_ids:
            errors.append(_error('P2_UNEXPECTED_MANUAL_SOURCE', source_valve=source,
                                 valves=[v['id'] for v in valves]))
    # One original common node must really be a common actual network node.
    for row, junction in zip(junction_rows, expected['junctions']):
        position = row['position']; incident = []
        for entry in junction['incident']:
            peer = positions.get(entry['peer_endpoint'])
            okay = bool(position is not None and peer is not None
                and common._connected(position, [peer], local_parts.get(entry['edge'], [])))
            incident.append({'edge': entry['edge'], 'peer': entry['peer_endpoint'], 'connected': okay})
        row.update(incident_actual=incident, pass_=row['pass'] and all(e['connected'] for e in incident))
        if not row['pass_']:
            errors.append(_error('P2_SHARED_NODE_NOT_CONNECTED', **row))
    # Reducers must connect the unchanged equipment neck to the selected line.
    for endpoint, port in expected['ports'].items():
        incident_edges = [e for e in expected['edges'] if endpoint in (e['a'], e['b'])]
        if not incident_edges:
            continue
        for edge in incident_edges:
            point = np.asarray(port['position']); normal = np.asarray(port['normal'])
            departures = []
            for segment in local_parts[edge['id']]:
                if not common._on_segment(point, *segment['ends']):
                    continue
                for end in segment['ends']:
                    vector = np.asarray(end)-point; length = np.linalg.norm(vector)
                    if length > EPS and float(vector@normal/length) >= 1.-EPS:
                        departures.append(segment['id'])
            if not departures:
                errors.append(_error('P2_ORIGINAL_PORT_NORMAL_NOT_CONNECTED', requirement=edge['id'],
                    endpoint=endpoint, original_position=port['position'], original_normal=port['normal']))
            selected = {r['dn'] for r in routes[edge['id']]}
            if len(selected) != 1:
                continue
            dn = next(iter(selected)); pipe_radius = expected['config']['pipe_outer_diameter_mm'][str(dn)]/2000.
            source_radius = port['actual_radius_m'] or port['connection_radius_m']
            mate=catalogued.get(endpoint)
            if mate and mate['pipe_dn']==dn and mate['through_bore'] in departures and any(
                    s['id']==mate['stub'] and abs(s['radius']-pipe_radius)<=EPS for s in local_parts[edge['id']]):
                continue
            reducer_face = port['position']
            flange = flanged.get(endpoint)
            if flange is not None:
                if flange['stub'] not in departures:
                    errors.append(_error('P2_FLANGE_STUB_NOT_ON_EDGE', requirement=edge['id'],
                                         endpoint=endpoint, stub=flange['stub']))
                    continue
                reducer_face = flange['stub_end_m']
                source_radius = flange['stub_radius_m']
                if dn == flange['neck_dn']:
                    # Equal nominal DN may have distinct true tube OD series. The
                    # adjoining line must still terminate at the measured stub back.
                    line_fits = any(s['id'] != flange['stub'] and any(common._near(end,reducer_face)
                        and abs(s.get('radii',[s.get('radius',0.)]*2)[i]-pipe_radius)<=EPS
                        for i,end in enumerate(s['ends'])) for s in local_parts[edge['id']])
                    peer_endpoint = edge['b'] if endpoint==edge['a'] else edge['a']
                    peer = expected['ports'].get(peer_endpoint,{})
                    peer_shape = actual.get(peer.get('nozzle_component'))
                    peer_neck = common._cylinder(peer_shape) if peer_shape else None
                    direct = bool(peer_neck and peer.get('source_dn')==dn and
                        common._near(peer.get('position'),reducer_face) and common._near(peer.get('normal'),-normal)
                        and abs(peer_neck['radius']-pipe_radius)<=EPS and any(
                            common._near(end,reducer_face) and float((np.asarray(peer_neck['ends'][1-i])-end)@normal)>EPS
                            and np.linalg.norm(np.cross(np.asarray(peer_neck['ends'][1-i])-end,normal))<=EPS
                            for i,end in enumerate(peer_neck['ends'])))
                    if not line_fits and not direct:
                        errors.append(_error('P2_FLANGE_STUB_LINE_NOT_CONNECTED', requirement=edge['id'],
                            endpoint=endpoint, stub=flange['stub'], expected_position=reducer_face,
                            selected_dn=dn, target_pipe_radius_m=pipe_radius))
                    continue
            elif abs(source_radius-pipe_radius) <= EPS:
                continue
            candidates = [s for s in local_parts[edge['id']]
                if components.get(s['id'], {}).get('kind') == 'piping_reducer'
                and any(common._near(p, reducer_face) for p in s['ends'])]
            correct = False
            for candidate in candidates:
                index = next((i for i, p in enumerate(candidate['ends']) if common._near(p, reducer_face)), None)
                radii = candidate.get('radii', [])
                correct |= bool(index is not None and len(radii) == 2
                    and abs(radii[index]-source_radius) <= EPS and abs(radii[1-index]-pipe_radius) <= EPS)
                if flange is not None and correct:
                    vector = np.asarray(candidate['ends'][1-index])-reducer_face
                    correct = (float(vector@normal)>EPS and np.linalg.norm(np.cross(vector,normal))<=EPS
                               and components[candidate['id']].get('source_dn')==flange['neck_dn']
                               and components[candidate['id']].get('target_dn')==dn)
                if correct:
                    break
            if not correct:
                errors.append(_error('P2_REQUIRED_REAL_REDUCER', requirement=edge['id'], endpoint=endpoint,
                    source_actual_radius_m=source_radius, selected_dn=dn, target_pipe_radius_m=pipe_radius,
                    expected_reducer_face=reducer_face, actual_reducers=candidates))
    return connections, manual, errors, {'junctions': junction_rows, 'terminals': terminal_rows,
                                        'positions': positions, 'parts': local_parts}


def _terminal_checks(graph, scene, piping, expected, actual, by_route, valve_rows, topology, model):
    rows, errors = [], []
    output = {t['id']: t for t in _records(piping.get('material', {}).get('terminals', []))}
    components = {c['id']: c for c in scene['components']}
    valve_map = {v['id']: v for v in graph.get('valves', [])}
    actual_valves = {v['id']: v for v in valve_rows if v['status'] == 'pass'}
    all_parts = [part for values in by_route.values() for part in values]
    cache = {}
    for terminal in expected['terminals']:
        saved = output.get(terminal['id'], {})
        roots = [v for v in graph.get('valves', []) if v.get('source_valve') == terminal['root_source_valve']]
        root = roots[0] if len(roots) == 1 else None
        native = actual_valves.get(root['id']) if root else None
        cap_id = saved.get('blind'); cap_id = cap_id.get('id') if isinstance(cap_id, dict) else cap_id
        cap = common._cylinder(actual[cap_id]) if cap_id in actual else None
        source = expected['ports'][terminal['physical_endpoint']]['position']
        relevant = topology['parts'].get(terminal['id'], [])+topology['parts'].get(terminal['root_edge'], [])
        if terminal['root_edge'] == terminal['id']:
            relevant = topology['parts'].get(terminal['id'], [])
        end = topology['positions'].get(terminal['boundary_endpoint'])
        connected = bool(native and cap and end is not None and common._connected(source, [end], relevant))
        radius = expected['config']['pipe_outer_diameter_mm'][str(terminal['line_dn'])]/2000.
        cap_size = bool(cap and cap['radius']+EPS >= radius and math.dist(*cap['ends']) > EPS)
        # Real annular flange mesh around this terminal line. No label-only
        # flange, solid decorative box or unrelated room flange can satisfy it.
        flanges = []
        for connection in scene.get('flange_connections', []):
            position = connection.get('position')
            if position is None or not any(common._on_segment(position, *s['ends']) for s in relevant):
                continue
            names = connection.get('components', [])
            if not names or any(name not in actual or components.get(name, {}).get('kind') != 'piping_flange'
                                or actual[name]['type'] != int(mujoco.mjtGeom.mjGEOM_MESH) for name in names):
                continue
            axis = np.asarray(cap['ends'][1])-cap['ends'][0] if cap else None
            if axis is None or np.linalg.norm(axis) <= EPS:
                continue
            axis /= np.linalg.norm(axis); origin = np.asarray(position)
            vertices = np.concatenate([common._mesh_support_data(actual[name], model, cache)[0] for name in names])
            relative = vertices-origin; axial = relative@axis
            radial = np.linalg.norm(relative-np.outer(axial, axis), axis=1)
            if radial.min() >= radius-EPS and radial.max() > radius+EPS and axial.max()-axial.min() > EPS:
                flanges.append({'id': connection['id'], 'components': names,
                                'actual_radial_range_m': [float(radial.min()), float(radial.max())],
                                'actual_axial_range_m': [float(axial.min()), float(axial.max())]})
        free_tails = []
        if cap and native and end is not None:
            # The cap's outward face is the one farther from the root. A pipe
            # beyond that real closed end is a forbidden free extension.
            reference = np.mean(list(native['actual_ports'].values()), axis=0)
            outside = max(cap['ends'], key=lambda p: math.dist(p, reference))
            direction = np.asarray(outside)-reference
            direction /= np.linalg.norm(direction)
            for segment in all_parts:
                if components.get(segment['id'], {}).get('kind') != 'pipe':
                    continue
                for q in segment['ends']:
                    q = np.asarray(q); distance = float((q-np.asarray(outside))@direction)
                    if (distance > EPS and np.linalg.norm(q-outside-distance*direction) <= EPS
                            and any(common._on_segment(face, *segment['ends']) for face in cap['ends'])):
                        free_tails.append(segment['id'])
        okay = connected and cap_size and bool(flanges) and not free_tails
        row = {'phase': 'P2', 'terminal': terminal['id'], 'type': terminal['type'],
               'physical_endpoint': terminal['physical_endpoint'], 'root_source_valve': terminal['root_source_valve'],
               'root': root['id'] if root else None, 'blind': cap_id, 'root_to_blind_connected': connected,
               'actual_blind_size_matches': cap_size, 'actual_flanges': flanges,
               'free_tails': sorted(set(free_tails)), 'pass': okay,
               'scope': terminal['scope']}
        rows.append(row)
        if not okay:
            errors.append(_error('P2_TERMINAL_NOT_ROOT_FLANGE_BLIND', **row))
    return _section(rows, errors, rule='PG-16 / blind termination for unused ports')


def _actual_path(start, finish, segments):
    """An ordered path made only from measured axes and verified native ports."""
    import heapq
    if start is None or finish is None:
        return []
    nodes = []
    def node(point):
        found = next((i for i, value in enumerate(nodes) if common._near(point, value)), None)
        if found is None:
            nodes.append(list(point)); return len(nodes)-1
        return found
    begin, end = node(start), node(finish)
    for segment in segments:
        for point in segment['ends']:
            node(point)
    adjacency = defaultdict(list)
    for segment in segments:
        indices = [i for i, point in enumerate(nodes) if common._on_segment(point, *segment['ends'])]
        indices.sort(key=lambda i: math.dist(nodes[i], segment['ends'][0]))
        for one, two in zip(indices, indices[1:]):
            distance = math.dist(nodes[one], nodes[two])
            if distance <= EPS:
                continue
            adjacency[one].append((two, distance, segment['id']))
            adjacency[two].append((one, distance, segment['id']))
    queue = [(0., begin)]; distances = {begin: 0.}; parents = {}
    while queue:
        distance, current = heapq.heappop(queue)
        if current == end:
            break
        if distance > distances[current]:
            continue
        for other, length, component in adjacency[current]:
            candidate = distance+length
            if candidate < distances.get(other, math.inf):
                distances[other] = candidate; parents[other] = (current, component)
                heapq.heappush(queue, (candidate, other))
    if end not in parents and end != begin:
        return []
    path = []; current = end
    while current != begin:
        previous, component = parents[current]
        path.append({'id': component, 'ends': [nodes[previous], nodes[current]]}); current = previous
    return list(reversed(path))


def _material_hierarchy(frozen, scene, piping, expected, actual, by_route, topology):
    rows, errors = [], []
    floors = frozen['scene']['floors']; config = expected['config']
    regions = common._edge_regions(frozen, piping, actual, by_route, config)
    zones = common._zone_rows(piping); demands = {e['id']: e for e in expected['edges']}
    board_ids = {c['id'] for c in scene['components'] if c['kind'] in ('floor', 'walk_surface')}
    actual_paths = {edge['id']: _actual_path(topology['positions'].get(edge['a']),
        topology['positions'].get(edge['b']), topology['parts'][edge['id']]) for edge in expected['edges']}
    for route in piping.get('routes', []):
        edge = demands.get(_source_edge(route))
        if edge is None:
            continue
        exempt = edge['role'] in ('vapor', 'process_evaporated_vapor') and edge['medium_identity'] == 'process_evaporated_vapor'
        segments = by_route.get(route['id'], [])
        for index, segment in enumerate(segments):
            a, b = map(np.asarray, segment['ends']); delta = b-a
            if abs(delta[2]) <= EPS:
                continue
            for level, z in enumerate(floors[1:], 1):
                if min(a[2], b[2])+EPS >= z or max(a[2], b[2])-EPS <= z:
                    continue
                point = a+delta*(z-a[2])/delta[2]
                radius = max(segment.get('radii', [segment.get('radius', 0.)]))
                # Oblique cylinder/plane intersection is an ellipse, not a
                # radius-sized square. Include the real slab thickness too.
                boards = [shape for name, shape in actual.items()
                    if name in board_ids
                    and shape['box'][2] < z+EPS and shape['box'][5] >= z-EPS]
                bottom = min((s['box'][2] for s in boards), default=z)
                top = max((s['box'][5] for s in boards), default=z)
                axes = [a+delta*(height-a[2])/delta[2] for height in (bottom, top)]
                extents = radius*np.sqrt(1.+(delta[:2]/delta[2])**2)
                rect = [min(p[0] for p in axes)-extents[0], min(p[1] for p in axes)-extents[1],
                        max(p[0] for p in axes)+extents[0], max(p[1] for p in axes)+extents[1]]
                region = regions.get(level)
                at_edge = bool(region and region['declared_matches'] and all(
                    rect[i] >= region['rect'][i]-EPS and rect[i+2] <= region['rect'][i+2]+EPS for i in range(2)))
                openings = [o for o in piping.get('openings', []) if o.get('level') == level]
                registered = [o for o in openings if all(rect[i] >= o['rect'][i]-EPS
                    and rect[i+2] <= o['rect'][i+2]+EPS for i in range(2))]
                from .grating import grating,overlaps
                covers=any(grating(c) and abs(c['box'][5]-z)<EPS and overlaps(rect,c['box']) for c in frozen['scene']['components'])
                okay = not any(overlaps(rect,b['box']) for b in boards) and (bool(registered) or not covers)
                row = {'phase': 'P2', 'requirement': edge['id'], 'route': route['id'],
                       'component': segment['id'], 'level': level, 'actual_crossing': point.tolist(),
                       'actual_outer_rect': [float(v) for v in rect], 'actual_slab_z_band_m': [bottom, top],
                       'edge_region': at_edge, 'registered_openings': registered,
                       'vapor_exception': exempt, 'pass': okay}
                rows.append(row)
                if not okay:
                    errors.append(_error('P2_GRATING_OPENING', **row))
                if not edge.get('requires_edge_crossing') or exempt:
                    continue
                path = actual_paths[edge['id']]
                crossing_step = next((i for i, step in enumerate(path)
                    if step['id'] == segment['id'] and step['ends'][0][2]+EPS < z < step['ends'][1][2]-EPS), None)
                if crossing_step is not None:
                    horizontal = next((s for s in path[crossing_step+1:]
                        if abs(s['ends'][0][2]-s['ends'][1][2]) <= EPS), None)
                    if horizontal is not None:
                        zone = zones.get(level); shape = actual.get(horizontal['id'])
                        inside = bool(zone and shape and all(shape['box'][i] >= zone[i]-EPS
                            and shape['box'][i+3] <= zone[i+3]+EPS for i in range(3)))
                        after = {'phase': 'P2', 'requirement': edge['id'], 'component': horizontal['id'],
                                 'level': level, 'actual_box': shape['box'] if shape else None,
                                 'actual_directed_step': horizontal, 'zone': zone, 'pass': inside}
                        rows.append(after)
                        if not inside:
                            errors.append(_error('P2_UPWARD_PIPE_OUTSIDE_RACK', **after))
    return _section(rows, errors, scope='Verify physical grating holes. Existing openings and positions outside the original panel are allowed.')


def _transfer_checks(frozen, expected, p1_report):
    from .piping_material_requirements import pumpless_condition
    rows, errors = [], []
    graph = frozen['contracts']['sizing']['graph']
    for circuit in expected['circuits']:
        row = pumpless_condition(graph, circuit, expected['ports'])
        if row is None:
            continue
        rows.append({'phase': 'P2', **row})
        if not row['pass']:
            errors.append(_error('P2_PUMPLESS_PATH', transfer=row['transfer'], path_units=row['path_units']))
    return _section(rows, errors, scope='Verify equipment order and zero pumps for pump-free routes. Record elevations and check physical connections and nitrogen separately.')


def _network_separation(piping, expected, by_route):
    rows, errors = [], []
    edge_ids = {e['id'] for e in expected['edges']}
    material = [r for r in piping.get('routes', []) if _source_edge(r) in edge_ids]
    utilities = piping['p1_piping']['routes']
    for route in material:
        for utility in utilities:
            touches = common._touch_points(by_route.get(route['id'], []), by_route.get(utility['id'], []))
            if touches:
                row = {'phase': 'P2', 'requirement': _source_edge(route), 'route': route['id'],
                       'utility_route': utility['id'], 'utility_medium': utility.get('medium'),
                       'actual_external_joints': touches, 'pass': False}
                rows.append(row)
                errors.append(_error('P2_MATERIAL_UTILITY_SHORT', **row))
    return _section(rows, errors, scope='Material and distillation networks must remain separate from P1 utilities outside equipment')


def check_p2(frozen, graph, scene, piping, model, data, operation_report=None):
    """Check inherited P1 plus appended P2 against independently frozen demands."""
    from .piping_material_requirements import requirements_from_frozen
    p1 = deepcopy(piping['p1_piping'])
    expected = requirements_from_frozen(frozen, p1)
    errors = []
    if piping.get('phase') != 'P2':
        errors.append(_error('P2_PHASE_MISMATCH'))
    material_ids = {edge['id'] for edge in expected['edges']}
    original_p1_ids = {route['id'] for route in p1['routes']}
    p1['routes'] = [deepcopy(route) for route in piping.get('routes', [])
        if route['id'] in original_p1_ids or (_source_edge(route) not in material_ids
            and route.get('phase') != 'P2' and not str(route['id']).startswith('P2_'))]
    p1['allowed_contacts'] = deepcopy(piping.get('allowed_contacts', []))
    p1['openings'] = deepcopy(piping.get('openings', []))
    p1['material'] = deepcopy(piping.get('material', {}))
    # Full graph/scene and explicit all-route geometry preserve every P2 solid.
    inherited = common.check_p1(frozen, graph, scene, p1, model, data, operation_report,
                               physical_routes=piping.get('routes', []))
    errors.extend(inherited['errors'])
    actual = common._measured(model, data)
    components = {c['id']: c for c in scene['components']}
    by_route, route_errors, pipe_sizes = common._route_segments(piping.get('routes', []), components, actual,
        expected['config']['pipe_outer_diameter_mm'], model, data)
    errors.extend(route_errors)
    valve_rows, valve_errors, valve_segments, _ = common._valves(graph, actual)
    errors.extend(valve_errors)
    connections, manual, topology_errors, topology = _material_connectivity(
        frozen, graph, scene, piping, expected, actual, by_route, valve_rows, valve_segments,
        measured_flange_rows=inherited['directions']['rows'])
    errors.extend(topology_errors)
    for issue in expected.get('unresolved', []):
        errors.append(_error('P2_REQUIREMENT_UNRESOLVED', requirement=issue))
    terminals = _terminal_checks(graph, scene, piping, expected, actual, by_route, valve_rows, topology, model)
    installation = _installation_spaces(graph, expected, actual, by_route, valve_rows, components)
    spacing = inherited['spacing']
    material_hierarchy = _material_hierarchy(frozen, scene, piping, expected, actual, by_route, topology)
    transfers = _transfer_checks(frozen, expected, inherited)
    separation = _network_separation(piping, expected, by_route)
    for section in (terminals, installation, material_hierarchy, transfers, separation):
        errors.extend(section['errors'])
    p1_connections = [dict(row, phase='P1') for row in inherited['connections']['rows']]
    p1_manual = [dict(row, phase='P1') for row in inherited['manual_valves']['rows']]
    combined_connections = p1_connections+connections
    combined_manual = p1_manual+manual
    hierarchy = deepcopy(inherited['hierarchy'])
    hierarchy['rows'] += material_hierarchy['rows']+terminals['rows']+installation['rows']+transfers['rows']+separation['rows']
    hierarchy['errors'] += material_hierarchy['errors']+terminals['errors']+installation['errors']+transfers['errors']+separation['errors']
    hierarchy['violations'] = hierarchy['errors']; hierarchy['status'] = 'fail' if hierarchy['errors'] else 'pass'
    # Repeated helper checks yield identical errors; do not inflate counts.
    import json
    unique = {}; errors = [unique.setdefault(json.dumps(e, sort_keys=True, ensure_ascii=False), e)
        for e in errors if json.dumps(e, sort_keys=True, ensure_ascii=False) not in unique]
    return {'status': 'fail' if errors else 'pass', 'errors': errors, 'checks_revision': 'PG14_blind_complete',
        'connections': {'expected': len(combined_connections),
            'connected': sum(row['connected'] for row in combined_connections), 'rows': combined_connections,
            'groups': inherited['connections'].get('groups', []), 'junctions': topology['junctions']},
        'manual_valves': {'expected': len(combined_manual),
            'installed': sum(row['installed'] for row in combined_manual), 'rows': combined_manual,
            'native_packages': valve_rows, 'pipe_sizes': pipe_sizes},
        'geometry': inherited['geometry'], 'hierarchy': hierarchy, 'spacing': spacing,
        'operation_report': deepcopy(inherited['operation_report']),
        'material': {'terminals': terminals, 'installation_spaces': installation, 'parallel_clearances': spacing,
            'crossings': material_hierarchy, 'transfer_conditions': transfers, 'network_separation': separation,
            'filter_bypass': {'applicable': any(e['role'] == 'bypass' for e in expected['edges']),
                             'scope': 'Check only filters called for in the process graph'}},
        'source_port_conflicts': inherited['source_port_conflicts'],
        'scope': {'phase': 'P2', 'requirements_source': 'independently rederived from frozen sizing/layout/connections',
            'inherited_p1_complete': True, 'unified_actual_geometry': True, 'operation_ratio_gate': False,
            'coverage_quotas': False, 'p3_robot_or_process_success': 'not claimed',
            'terminal_scope': expected['scope']}}
