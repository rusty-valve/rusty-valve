"""Derive required process connections from the equipment and recipe."""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import math

from .evaluation_common import require_environment
from .piping_requirements import P1_ROLES, load_stage_config, _contracts, equipment_source, port_level
from .material_support import canonical_transfer_mode
from .recipe import InputRejected
from .valve_roles import describe


AUTHORITY = 'PG v1.2 §§3–4；PG05–11、14、16'
MATERIAL_DN = {'small': 50, 'medium': 80, 'large': 100}
VAPOR_DNS = (200, 250, 300)
EPS = 1.e-8  # Existing geometric equality; not a clearance allowance.


def pumpless_condition(graph, transfer, ports):
    if canonical_transfer_mode(transfer['mode']) != 'no_pump':
        return None
    units = graph['units']
    path = transfer['path_units']
    pumps = [uid for uid in path if units.get(uid, {}).get('kind') == 'pump']
    source, target = transfer['source'], transfer['target']
    correct_path = bool(path and path[0] == source and path[-1] == target and not pumps)
    row = {'transfer': transfer['id'], 'mode': 'no_pump', 'path_units': list(path),
           'pumps_in_path': pumps, 'pass': correct_path,
           'status': 'pass' if correct_path else 'fail',
           'code': None if correct_path else 'P2_PUMPLESS_PATH'}
    outlet = ports.get(source + '.bottom')
    inlet = ports.get(target + '.feed')
    if outlet and inlet:
        row.update(actual_source_outlet=outlet['position'], actual_target_inlet=inlet['position'],
                   height_difference_m=outlet['position'][2] - inlet['position'][2])
    return row


class MaterialRequirementRejected(InputRejected):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail
        super().__init__(f'{code}: {detail}')


def _reject(code, detail):
    raise MaterialRequirementRejected(code, detail)


def _path(graph, edges, start, finish):
    """Directed path between two real ports, passing only logical junctions."""
    outgoing = defaultdict(list)
    for edge in edges:
        outgoing[edge['a']].append(edge)
    queue, seen = deque([(start, [])]), {start}
    while queue:
        endpoint, path = queue.popleft()
        if endpoint == finish:
            return path
        uid = endpoint.split('.')[0]
        starts = ([uid+'.'+p for p in graph['units'][uid]['ports']]
                  if graph['units'][uid]['kind'] == 'junction' else [endpoint])
        for ep in starts:
            for edge in outgoing[ep]:
                target = edge['b']
                if target not in seen:
                    seen.add(target); queue.append((target, path+[edge['id']]))
    return None


def _nearest_physical(graph, edges, endpoint, forward, equipment):
    """Resolve a boundary through junctions without crossing a real device."""
    queue, seen, found = deque([endpoint]), set(), set()
    while queue:
        ep = queue.popleft()
        if ep in seen:
            continue
        seen.add(ep); uid = ep.split('.')[0]
        if uid in equipment:
            found.add(ep); continue
        if graph['units'][uid]['kind'] != 'junction':
            continue
        for edge in edges:
            a, b = (edge['a'], edge['b']) if forward else (edge['b'], edge['a'])
            if a.split('.')[0] == uid:
                queue.append(b)
    return sorted(found)


def _pump_face(uid, port, physical, components, connection_radius):
    body = components.get(uid)
    if not body or body.get('kind') != 'pump' or body.get('shape') != 'cylinder':
        _reject('P2_PUMP_PHYSICAL_INTERFACE', f'{uid}.{port}Missing original pump cylinder')
    ends = body.get('ends', []); normal = physical['normal']; position = physical['pos']
    if len(ends) != 2 or not body.get('radius'):
        _reject('P2_PUMP_PHYSICAL_INTERFACE', f'{uid}.{port}Missing physical pump end face')
    length = math.dist(*ends)
    axis = [(ends[1][i]-ends[0][i])/length for i in range(3)]
    alignment = sum(axis[i]*normal[i] for i in range(3))
    if abs(abs(alignment)-1.) > EPS:
        _reject('P2_PUMP_PHYSICAL_INTERFACE', f'{uid}.{port}The original port is not normal to the physical cylinder end')
    face = ends[1] if alignment > 0 else ends[0]
    gap = sum((position[i]-face[i])*normal[i] for i in range(3))
    if gap < -EPS or math.dist(position, [face[i]+normal[i]*gap for i in range(3)]) > EPS:
        _reject('P2_PUMP_PHYSICAL_INTERFACE', f'{uid}.{port}The port is inside the original coaxial pump face')
    coincident_obstacles = []
    for c in components.values():
        if c.get('owner') != uid or c['id'] == uid or c.get('shape') != 'cylinder':
            continue
        if any(math.dist(face, end) <= EPS for end in c.get('ends', [])):
            coincident_obstacles.append(c['id'])
    return {'kind': 'body_face', 'component_ids': [uid], 'position': deepcopy(face),
        'normal': deepcopy(normal), 'radius_m': float(body['radius']),
        'logical_port_position': deepcopy(position), 'adapter_path': [deepcopy(face), deepcopy(position)],
        'gap_m': gap, 'adapter_pipe_radius_m': connection_radius,
        'coincident_obstacles': coincident_obstacles,
        'logical_anchor_required': True,
        'scope': 'No original stub outside the saved port. Add pipe only on the original axis to the physical face and check local mating.'}


def _port_rows(frozen, graph, equipment, config):
    components = {c['id']: c for c in frozen['scene']['components']}
    rows = {}
    for uid, unit in equipment.items():
        for name, physical in unit['ports'].items():
            if physical.get('catalogue_missing'):
                continue
            endpoint = uid+'.'+name; source_dn = physical['dn']
            if str(source_dn) not in config['pipe_outer_diameter_mm']:
                _reject('P2_PORT_DN', f'{endpoint}Unknown pipe OD for the original DN')
            source = equipment_source(unit, name)
            nozzle = components.get(source['source_nozzle_id'])
            connection_radius = config['pipe_outer_diameter_mm'][str(source_dn)]/2000.
            row = {'equipment': uid, 'port': name, 'endpoint': endpoint,
                'position': deepcopy(physical['pos']), 'normal': deepcopy(physical['normal']),
                **source, 'line_dn': physical.get('line_dn', source_dn),
                'equipment_neck_dn': physical.get('equipment_neck_dn'),
                'actual_radius_m': float(nozzle['radius']) if nozzle and 'radius' in nozzle else None,
                'connection_radius_m': connection_radius,
                'nozzle_component': nozzle['id'] if nozzle else None,
                'level': port_level(physical, unit),
                'source_port': deepcopy(physical)}
            if face := source.get('source_catalogue_face'):
                row['actual_radius_m']=face['opening_diameter_m']/2
                row['physical_interface']={'kind':'catalogue_flange_face','component_ids':face['contact_ids'],
                    'position':row['position'],'normal':row['normal'],'radius_m':row['actual_radius_m']}
            elif nozzle:
                row['source_radius_m'] = row['actual_radius_m']
                row['physical_interface'] = {'kind': 'nozzle_face', 'component_ids': [nozzle['id']],
                    'position': deepcopy(physical['pos']), 'normal': deepcopy(physical['normal']),
                    'radius_m': float(nozzle['radius'])}
            elif unit['kind'] == 'pump':
                row['physical_interface'] = _pump_face(uid, name, physical, components, connection_radius)
            else:
                _reject('P2_PORT_PHYSICAL_MISSING', f'{endpoint}Missing saved physical port')
            rows[endpoint] = row
    return rows




def requirements_from_frozen(frozen, p1_piping=None):
    """Return an immutable-source P2 plan contract; unresolved items stay counted.

    ``dn`` is the preferred modelling value. For vapor only, ``dn_candidates``
    permits the builder to choose another declared value consistently across a
    connected shared manifold, valves and reducers. No coordinate is invented
    for a logical junction or an external system.
    """
    require_environment()
    contracts, graph, equipment = _contracts(frozen)
    if p1_piping is not None and (p1_piping.get('phase') != 'P1' or p1_piping.get('stage') != 'piping'):
        _reject('P2_P1_SOURCE_PHASE', 'Use the saved P1 physical result as the starting model')
    config = deepcopy(p1_piping['config']) if p1_piping is not None else load_stage_config()
    ports = _port_rows(frozen, graph, equipment, config)
    original = [deepcopy(e) for e in graph['edges'] if e.get('role') not in P1_ROLES]
    if p1_piping is not None and p1_piping.get('source_p2_edges') != original:
        _reject('P2_P1_SOURCE_EDGES', 'P1 material edges differ from the four source contracts')
    by_id = {e['id']: e for e in original}
    if len(by_id) != len(original): _reject('P2_EDGE_IDS', 'Duplicate source material edge IDs')
    valves = [deepcopy(v) for v in graph['valves'] if v['edge'] in by_id and v.get('manual')]
    votes, memberships = defaultdict(list), defaultdict(list)
    circuits, unresolved, transfer_conditions = [], [], []

    def assign(path, dn, medium, circuit):
        for eid in path:
            votes[eid].append((dn, medium)); memberships[eid].append(circuit)

    for transfer in graph['transfers']:
        tid, source, target = transfer['id'], transfer['source'], transfer['target']
        tier = graph['units'][source].get('size_tier')
        if tier not in MATERIAL_DN: _reject('P2_SOURCE_TIER', f'{tid}Source equipment lacks a valid saved size tier')
        dn = MATERIAL_DN[tier]; mode = canonical_transfer_mode(transfer['mode']); paths = []; extras = []
        if mode == 'distillation':
            condenser = transfer['condenser']
            segments = [(source+'.vapor', condenser+'.vapor', 200, 'process_evaporated_vapor'),
                        (condenser+'.condensate', target+'.feed', 50, 'condenser_material_condensate')]
        else:
            chain = transfer['path_units']; segments = []
            for a, b in zip(chain, chain[1:]):
                start = a+('.bottom' if graph['units'][a]['kind'] == 'vessel' else '.out')
                end = ('X_'+b+'.port' if b.startswith('EXPORT') else
                       b+('.feed' if graph['units'][b]['kind'] == 'vessel' else '.in'))
                segments.append((start, end, dn, 'unknown_material'))
        for start, end, local_dn, medium in segments:
            path = _path(graph, original, start, end)
            if path is None:
                unresolved.append({'code': 'P2_TRANSFER_PATH_MISSING', 'object': tid, 'detail': start+'→'+end}); continue
            assign(path, local_dn, medium, tid); paths.extend(path)
        if mode == 'pump':
            pump = transfer['pump']
            for edge in original:
                if edge['role'] == 'recycle' and edge['b'].split('.')[0] == source:
                    path = _path(graph, original, pump+'.out', edge['b'])
                    if path and edge['id'] in path:
                        assign(path, dn, 'unknown_material', tid); extras.extend(path)
            for aux in transfer['path_units']:
                if graph['units'].get(aux, {}).get('kind') != 'filter': continue
                inlet = next(e for e in original if e['b'] == aux+'.in')
                outlet = next(e for e in original if e['a'] == aux+'.out')
                for edge in original:
                    if edge['role'] == 'bypass' and edge['a'].split('.')[0] == inlet['a'].split('.')[0] and edge['b'].split('.')[0] == outlet['b'].split('.')[0]:
                        assign([edge['id']], dn, 'unknown_material', tid); extras.append(edge['id'])
        condition = pumpless_condition(graph, transfer, ports)
        if condition is not None:
            transfer_conditions.append(condition)
        circuits.append({'id': tid, 'source_transfer': deepcopy(transfer), 'source': source, 'target': target,
            'mode': mode, 'path_units': deepcopy(transfer['path_units']), 'source_size_tier': tier,
            'ordinary_material_dn': dn, 'edge_ids': list(dict.fromkeys(paths)),
            'recycle_bypass_edge_ids': list(dict.fromkeys(extras)),
            'vapor_dn_candidates': list(VAPOR_DNS) if mode == 'distillation' else [],
            'condensate_dn': 50 if mode == 'distillation' else None})

    terminals = []
    for edge in original:
        a, b = edge['a'], edge['b']; au, bu = a.split('.')[0], b.split('.')[0]
        if graph['units'][au]['kind'] != 'boundary' and graph['units'][bu]['kind'] != 'boundary': continue
        upstream = graph['units'][au]['kind'] == 'boundary'
        boundary_ep, network_ep = (a, b) if upstream else (b, a)
        found = _nearest_physical(graph, original, network_ep, upstream, equipment)
        if len(found) != 1:
            unresolved.append({'code': 'P2_BOUNDARY_AMBIGUOUS', 'object': edge['id'], 'detail': found}); continue
        physical_ep = found[0]; uid, port = physical_ep.split('.')
        kind = port if port in ('drain', 'gauge') else ('feed' if upstream else 'product')
        roots = [v for v in valves if v['at'] == physical_ep]
        if len(roots) != 1:
            unresolved.append({'code': 'P2_TERMINAL_ROOT_MISSING', 'object': edge['id'], 'detail': physical_ep}); continue
        if kind in ('drain', 'gauge'): local_dn = 25
        else:
            own_tier = graph['units'][uid].get('size_tier')
            local_dn = max((v[0] for v in votes[roots[0]['edge']]), default=MATERIAL_DN.get(own_tier, edge['dn']))
        terminal = {'id': edge['id'], 'type': kind, 'source_edge': deepcopy(edge), 'a': a, 'b': b,
            'boundary': boundary_ep.split('.')[0], 'boundary_endpoint': boundary_ep, 'network_endpoint': network_ep,
            'physical_endpoint': physical_ep, 'upstream_known': not upstream,
            'external_system_known': False, 'closure_policy': 'root_flange_blind',
            'boundary_policy': 'root_flange_blind', 'root_source_valve': roots[0]['id'],
            'root_edge': roots[0]['edge'], 'source_dn': ports[physical_ep]['source_dn'], 'line_dn': local_dn,
            'reuse_root': kind == 'product', 'status': 'terminal',
            'scope': 'Isolated blind boundary only. External systems and instruments are outside the model.'}
        terminals.append(terminal)
        medium = 'unknown_material' if kind in ('feed', 'product') else 'unknown'
        assign([edge['id'], roots[0]['edge']], local_dn, medium, 'terminal:'+edge['id'])

    terminal_by_edge = {v['id']: v for v in terminals}
    edges = []
    for original_edge in original:
        eid = original_edge['id']; choices = votes[eid]; mediums = {v[1] for v in choices}
        # Distinct incoming streams can share a final segment. This is a
        # conservative modelling diameter, never a flow-capacity assertion.
        local_dn = max((v[0] for v in choices), default=None)
        if not choices:
            unresolved.append({'code': 'P2_EDGE_UNASSIGNED', 'object': eid, 'detail': deepcopy(original_edge)})
        if len(mediums) > 1:
            unresolved.append({'code': 'P2_SHARED_MEDIUM_UNSPECIFIED', 'object': eid, 'detail': sorted(mediums)})
        medium = next(iter(mediums)) if len(mediums) == 1 else 'unknown'
        vapor = medium == 'process_evaporated_vapor'
        row = {**deepcopy(original_edge), 'source_edge': deepcopy(original_edge), 'source_dn': original_edge['dn'],
            'dn': local_dn, 'line_dn': local_dn, 'dn_candidates': list(VAPOR_DNS) if vapor else ([local_dn] if local_dn else []),
            'medium_identity': medium, 'medium': medium, 'circuit_ids': list(dict.fromkeys(memberships[eid])),
            'execution': 'terminal' if eid in terminal_by_edge else ('execute' if choices else 'unresolved'),
            'status': 'ready' if choices else 'unresolved',
            'path_policy': ('terminal_short_spool_only' if eid in terminal_by_edge else
                'PG11_shortest_feasible_then_elbows' if vapor else 'PG10_edge_crossing_then_rack_fewest_elbows_then_length'),
            'requires_edge_crossing': not vapor,
            'dn_source': 'Use DN50, DN80 or DN100 for small, medium or large source equipment. Shared runs use the largest branch DN. These are model sizing conventions.' if medium == 'unknown_material' else
                'Permitted material vapor candidates are DN200, DN250 and DN300' if vapor else
                'PG08 permits condensate DN25 to DN50. Use DN50 below the vapor inlet DN.' if medium == 'condenser_material_condensate' else
                'Keep the original DN15 port and reduce on the pipe side to the minimum available DN25 valve package',
            'required_manual_valves': [v['id'] for v in valves if v['edge'] == eid]}
        edges.append(row)
    edge_rows = {e['id']: e for e in edges}
    manual = []
    for v in valves:
        e = edge_rows[v['edge']]; at = v['at']
        manual.append({'id': v['id'], 'source_valve': deepcopy(v), 'source_valve_id': v['id'],
            'edge': v['edge'], 'at': at, 'source_role': v['role'], **describe(graph,v),
            'dn': e['dn'], 'source_dn': v['dn'], 'dn_candidates': deepcopy(e['dn_candidates']),
            'operator': v.get('operator'), 'manual': True, 'medium': e['medium_identity'],
            'flow_direction': 'along_original_edge' if at == 'branch' else
                ('equipment_to_network' if at == e['a'] else 'network_to_equipment'),
            'equipment': at.split('.')[0] if at != 'branch' else None})
    junctions = []
    for uid, unit in graph['units'].items():
        if unit['kind'] != 'junction': continue
        incident = [e for e in edges if uid in (e['a'].split('.')[0], e['b'].split('.')[0])]
        if not incident: continue
        junctions.append({'id': uid, 'source_unit': deepcopy(unit),
            'ports': [uid+'.'+p for p in unit['ports']], 'edge_ids': [e['id'] for e in incident],
            'incident': [{'edge': e['id'], 'endpoint': e['a'] if e['a'].split('.')[0] == uid else e['b'],
                'peer_endpoint': e['b'] if e['a'].split('.')[0] == uid else e['a']} for e in incident],
            'position': None, 'position_source': 'logical_only_builder_must_materialise_one_shared_node',
            'dn': max((e['dn'] for e in incident if e['dn'] is not None), default=None)})
    return {'phase': 'P2', 'stage': 'piping', 'status': 'unresolved' if unresolved else 'ready',
        'equipment_flange_scope': 'all_material',
        'circuits': circuits, 'edges': edges, 'source_p2_edges': deepcopy(original), 'ports': ports,
        'junctions': junctions, 'terminals': terminals, 'manual_valves': manual,
        'expected_connections': len(edges), 'expected_manual_valves': len(manual),
        'unresolved': unresolved, 'transfer_conditions': transfer_conditions, 'config': config,
        'source': {'authority': AUTHORITY, 'source_dir': frozen.get('source_dir'),
            'contracts': ['sizing.json', 'shell.json', 'layout.json', 'connections.json']},
        'scope': 'Add P2 material geometry while preserving equipment, P1 and the four source contracts. External feeds, drains and instruments end in blind flanges after root valves. The product line reuses the bottom valve. D and G retain DN15 ports with pipe-side transitions to DN25. Shared feed uses the maximum branch DN. Storage size tiers are not volume measurements. Pump-free transfer checks and utility nitrogen connections are separate. Geometry gives no hydraulic or safety validation.'}


def requirements_for_end_bundle(frozen, graph, config, pipe_spec, extra_ports=None):
    """Use explicit process edges, including exchanger-in-recycle/reflux.

    This deliberately does not reinterpret a reflux condenser as a receiver
    transfer. Physical original mouths still come from the accepted layout.
    """
    _, _, equipment = _contracts(frozen)
    ports = _port_rows(frozen, graph, equipment, config)
    for port in ports.values():
        reference = port.get('source_port', {}).get('operation_reference', {})
        if reference.get('status') == 'standing_face_reference':
            port['level'] = int(reference['level'])
            port['operation_reference'] = deepcopy(reference)
    ports.update(deepcopy(extra_ports or {}))
    skipped = set(P1_ROLES)
    original = [deepcopy(e) for e in graph['edges'] if e.get('role') not in skipped]
    valves = [deepcopy(v) for v in graph['valves'] if v.get('manual') and
              v.get('edge') in {e['id'] for e in original}]
    ordinary = pipe_spec['material_by_tier']
    owner = next((u for u in graph['units'].values() if u.get('kind') == 'vessel'), {})
    default_dn = int(ordinary[owner.get('size_tier', 'small')])
    edges, manual, terminals, unresolved, deferred_failures = [], [], [], [], []
    for source in original:
        role = source.get('role', 'material'); dn = int(source.get('dn', default_dn))
        au, bu = (graph['units'][source[k].split('.')[0]] for k in ('a', 'b'))
        vapor = role in ('vapor', 'process_vapor', 'reflux_vapor')
        liquid = role in ('condensate', 'reflux_condensate', 'condenser_condensate', 'condensate_return')
        steam_water = role in ('steam_condensate', 'condensate_drain', 'trap_drain')
        emergency = role in ('emergency_discharge', 'safety_discharge')
        if emergency:
            medium = 'emergency_discharge'
            dn = int(pipe_spec['services']['offgas']['emergency_dn'])
        elif vapor:
            medium = 'process_evaporated_vapor'
            if dn not in pipe_spec['process_vapor_candidates']:
                dn = int(pipe_spec['process_vapor_candidates'][0])
        elif liquid:
            medium = 'condenser_material_condensate'
            if dn not in pipe_spec['condenser_condensate_candidates']:
                dn = int(pipe_spec['condenser_condensate_candidates'][-1])
        elif steam_water:
            medium = 'steam_condensate'; dn = int(pipe_spec['services']['steam_condensate']['device_dn'])
        else:
            medium = 'unknown_material'
            if role not in ('drain', 'gauge', 'seal_drain'):
                dn = int(source['dn']) if source.get('dn_source')=='pipe_spec.json' else default_dn
            else:
                # Local material drain retains its explicit process-line size.
                # Seal drain shares the selected condenser liquid circuit.
                if source.get('dn_source') != 'pipe_spec.json':
                    _reject('LOCAL_DN_SOURCE_MISSING', source['id'])
                dn = int(source['dn'])
                if str(dn) not in pipe_spec['pipe_outer_diameter_mm']:
                    _reject('LOCAL_DN_UNKNOWN', source['id'])
        row = {**deepcopy(source), 'source_edge': deepcopy(source), 'source_dn': source.get('dn'),
            'dn': dn, 'line_dn': dn, 'dn_candidates': [dn], 'medium': medium,
            'medium_identity': medium, 'execution': 'execute', 'status': 'ready',
            'path_policy': 'shortest_feasible' if vapor or liquid or steam_water else 'orthogonal_avoid_solids',
            'requires_edge_crossing': False,
            'required_manual_valves': [v['id'] for v in valves if v['edge'] == source['id']]}
        for endpoint in (row['a'], row['b']):
            uid = endpoint.split('.')[0]
            if endpoint not in ports and graph['units'][uid]['kind'] not in ('junction', 'boundary'):
                missing = equipment.get(uid,{}).get('ports',{}).get(endpoint.split('.')[1],{}).get('catalogue_missing')
                if missing:
                    row.update(execution='deferred_failure', status='missing port')
                    deferred_failures.append({'code':'CATALOGUE_PORT_SHORTAGE','endpoint':endpoint,
                                              'edge':row['id'],'reason':missing})
                else:
                    unresolved.append({'code': 'PHYSICAL_PORT_MISSING', 'endpoint': endpoint, 'edge': row['id']})
        for upstream, endpoint, network in ((True, row['a'], row['b']), (False, row['b'], row['a'])):
            unit = graph['units'][endpoint.split('.')[0]]
            if unit.get('kind') != 'boundary' or endpoint in ports:
                continue
            branch_roots = [v for v in valves if v['edge'] == row['id'] and v['at'] == 'branch']
            found = _nearest_physical(graph, original, network, upstream, equipment)
            physical = found[0] if len(found) == 1 else network
            roots = branch_roots or [v for v in valves if v['at'] == physical]
            if len(roots) != 1:
                unresolved.append({'code': 'BOUNDARY_ROOT_MISSING', 'edge': row['id'], 'endpoint': network})
                continue
            row['execution'] = 'terminal'
            terminals.append({'id': row['id'], 'type': 'feed' if upstream else 'product',
                'a': row['a'], 'b': row['b'], 'boundary_endpoint': endpoint, 'network_endpoint': network,
                'physical_endpoint': physical, 'root_source_valve': roots[0]['id'],
                'boundary_policy': 'root_flange_blind', 'closure_policy': 'root_flange_blind'})
        edges.append(row)
    edge_by_id = {e['id']: e for e in edges}
    for valve in valves:
        edge = edge_by_id[valve['edge']]
        manual.append({'id': valve['id'], 'source_valve': deepcopy(valve), 'source_valve_id': valve['id'],
            'edge': valve['edge'], 'at': valve['at'], 'source_role': valve.get('role'),
            **describe(graph,valve),
            'dn': edge['dn'], 'source_dn': valve.get('dn'),
            'dn_candidates': [edge['dn']],
            'operator': 'wheel' if valve.get('operator') == 'handwheel' else valve.get('operator'), 'manual': True,
            'medium': edge['medium_identity'], 'flow_direction': 'along_original_edge'})
    junctions = []
    for uid, unit in graph['units'].items():
        incident = [e for e in edges if uid in (e['a'].split('.')[0], e['b'].split('.')[0])]
        if unit.get('kind') == 'junction' and incident:
            junctions.append({'id': uid, 'ports': [uid+'.'+p for p in unit['ports']],
                'edge_ids': [e['id'] for e in incident], 'source_unit': deepcopy(unit), 'position': None,
                'dn': max(e['dn'] for e in incident),
                'incident': [{'edge': e['id'], 'endpoint': e['a'] if e['a'].split('.')[0] == uid else e['b'],
                    'peer_endpoint': e['b'] if e['a'].split('.')[0] == uid else e['a']} for e in incident]})
    return {'phase': 'P2', 'stage': 'piping', 'config': deepcopy(config), 'ports': ports,
        'equipment_flange_scope': graph.get('execution_scope', 'all_material'),
        'edges': edges, 'source_p2_edges': original, 'manual_valves': manual, 'junctions': junctions,
        'terminals': terminals, 'unresolved': unresolved, 'deferred_failures': deferred_failures,
        'circuits': deepcopy(graph.get('transfers', [])),
        'transfer_conditions': [row for t in graph.get('transfers', [])
            if (row := pumpless_condition(graph, t, ports)) is not None], 'expected_connections': len(edges), 'expected_manual_valves': len(manual),
        'source': {'authority': pipe_spec.get('authority'), 'template': 'explicit_process_edges_v2'},
        'status': 'unresolved' if unresolved else 'partial' if deferred_failures else 'ready'}
