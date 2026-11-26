"""Determine which pipes and valves the process requires."""
from collections import defaultdict

from .connection_evidence import SERVICE_ROLES, owner
from .material_rack import MATERIAL
from .piping_assets import inspect_asset
from .valve_roles import describe
from .valve_positioning import wheel_policy


def valve_requirement(graph, source, dn=None, *, top_vertical=False):
    info = describe(graph, source)
    actual_dn = source['dn'] if dn is None else dn
    operator, _ = wheel_policy({**source, **info, 'dn': actual_dn}, graph)
    package = inspect_asset(actual_dn, operator)
    if top_vertical:
        info['placement_rule'] = 'top_vertical'
    return {
        'id': source['id'], 'edge': source['edge'], **info,
        'dn': actual_dn, 'operator': operator, 'package_id': package['package_id'],
        'package_version': package['version'], 'length_m': package['install_length_m'],
        'direction': 'vertical' if info['placement_rule'] == 'top_vertical' else 'pipe_axis',
    }


def edge_valves(graph, material_requirements=(), service_points=()):
    """Prefer the live selected DN where the layout contract supplies one."""
    material = {row['id']: row for row in material_requirements}
    services = {row['id']: row for row in service_points}
    grouped = defaultdict(list)
    for source in graph['valves']:
        if not source.get('manual'):
            continue
        policy = describe(graph, source)
        if policy['placement_rule'] == 'at_main_tee':
            continue  # the station demands are issued at the actual main and header
        service = services.get(source['edge'], {})
        selected = material.get(source['id'], {})
        uid, _, port = source['at'].partition('.')
        unit = graph['units'].get(uid, {})
        normal = service.get('normal', ())
        top = bool(normal and normal[2] > .9 and unit.get('kind') != 'pump' and port not in ('bottom', 'drain'))
        demand = valve_requirement(graph, source, service.get('dn', selected.get('dn')), top_vertical=top)
        grouped[source['edge']].append(demand)
    destinations = {edge['id']: edge['b'] for edge in graph['edges']}
    return {edge: sorted(rows, key=lambda row: (row['position_endpoint'] == destinations[edge],
                                               row['sequence_rank'], row['id']))
            for edge, rows in grouped.items()}


def process_paths(graph):
    """Walk every non-junction source to the next physical destination."""
    outgoing = defaultdict(list)
    for edge in graph['edges']:
        outgoing[owner(edge['a'])].append(edge)

    def follow(edge, seen):
        uid = owner(edge['b'])
        if graph['units'][uid]['kind'] != 'junction':
            yield (edge,)
            return
        if uid in seen:
            return
        continuations = outgoing[uid]
        # A low-point drain is its own route. It is not an alternate
        # destination of the condenser's process return.
        if edge['role'] == 'condensate':
            continuations = [nxt for nxt in continuations if nxt['role'] == 'condensate']
        if not continuations:
            yield (edge,)
        for nxt in continuations:
            for tail in follow(nxt, seen | {uid}):
                yield (edge, *tail)

    for edge in graph['edges']:
        kind = graph['units'][owner(edge['a'])]['kind']
        low_drain = kind == 'junction' and edge['role'] == 'seal_drain'
        if kind != 'junction' or low_drain:
            yield from follow(edge, {owner(edge['a'])})


def path_rule(graph, path, levels):
    first, last = path[0], path[-1]
    source = graph['units'][owner(first['a'])]
    a, b = levels.get(first['a']), levels.get(last['b'])
    if first['role'] in MATERIAL and source.get('service') == 'steam_heater' and (
            (source.get('service_owner') or source.get('owner')) == owner(last['b'])
            or first.get('branch_purpose') == 'heated_return'):
        if a is None or b is None or a < b:
            return 'M3d'
    if first['role'] == 'condensate' and source.get('orientation') == 'horizontal':
        if a is None or b is None or a > b:
            return 'M4c'
    if first['role'] == 'offgas':
        return 'M5a'
    if first['role'] == 'steam_condensate':
        if source['kind'] == 'automatic_steam_trap':
            return 'M3f'
        if graph['units'][owner(last['b'])]['kind'] == 'automatic_steam_trap':
            return 'M3e'
    if first['role'] in SERVICE_ROLES:
        return 'K20'
    return None


def route_requirements(graph, edge_demands, levels=None):
    levels = levels or {}
    paths = []
    for path in process_paths(graph):
        first, last = path[0], path[-1]
        paths.append({
            'source': first['a'], 'target': last['b'], 'edges': [edge['id'] for edge in path],
            'junctions': [owner(edge['b']) for edge in path[:-1]],
            'valve_spans': [valve for edge in path for valve in edge_demands.get(edge['id'], ())],
            'routing_rule': path_rule(graph, path, levels),
        })
    return paths


def station_requirements(station):
    paths = []
    for link in station.get('connections', ()):
        valves = []
        if link.get('native_L_m', 0.) > 0.:
            valves.append({'id': link['id'], 'dn': link['dn'], 'package_id': link.get('package'),
                           'length_m': link['native_L_m'], 'direction': 'pipe_axis',
                           'placement_rule': 'at_main_tee'})
        paths.append({'source': link['main_id'], 'target': link['header_id'], 'edges': [link['id']],
                      'junctions': [], 'valve_spans': valves, 'routing_rule': 'M5c'})
    return paths
