"""Read pipe and equipment geometry for connection checks."""
from collections import defaultdict
from functools import cached_property, lru_cache
import math

import numpy as np

from .piping_material_checks import _actual_path
from .valve_roles import position_context


EPS = 1e-6
SERVICE_ROLES = {'nitrogen', 'offgas', 'emergency_discharge', 'steam', 'water', 'water_return'}


def owner(endpoint):
    return endpoint.split('.')[0]


def line_ends(component):
    if 'ends' in component:
        return component['ends']
    ports = component.get('ports', {})
    if 'in' in ports and 'out' in ports:
        return [ports[key]['position'] for key in ('in', 'out')]
    return None


def valve_faces(valve):
    return {name: port['position'] for name, port in valve.get('world_ports', {}).items()
            if name in ('in', 'out')}


class ConnectionEvidence:
    """Plain data adapter. Future generation candidates can supply the same evidence."""

    def __init__(self, process, geometry, piping, measure, placement, *, original=None, original_layout=None):
        self.process = process
        self.geometry = geometry
        self.piping = piping
        self.measure = measure
        self.placement = placement
        self.original = original
        self.original_layout = original_layout
        self.units = process['units']
        self.edges = {edge['id']: edge for edge in process['edges']}
        self.valves = {valve['id']: valve for valve in process['valves']}
        self.source_valves = {valve['id']: valve for valve in (original or process)['valves']
                              if valve.get('manual')}
        self.installed = defaultdict(list)
        for valve in self.valves.values():
            self.installed[valve.get('source_valve') or valve['id']].append(valve)
        self.pieces = {part['id']: part for part in geometry['components']}
        self.elbows = geometry.get('elbows', piping.get('elbows', []))
        self.routes = {route['id']: route for route in piping['routes']}
        self.material = piping.get('material', {})
        self.junctions = {joint['id']: joint for joint in self.material.get('junctions', [])}
        self.ports = {**self.material.get('requirements', {}).get('ports', {})}
        self.connections = {row.get('source_edge', row.get('requirement')): row
                            for scope in (measure, measure.get('material', {}))
                            for row in scope.get('connections', {}).get('rows', [])}
        self.route_ids = defaultdict(list)
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        for edge in self.edges.values():
            self.outgoing[owner(edge['a'])].append(edge)
            self.incoming[owner(edge['b'])].append(edge)
        for route in self.routes.values():
            ids = [route.get(key) for key in ('source_edge', 'source_requirement', 'connection_id')]
            ids += route.get('source_edges', [])
            if len(route.get('point_ids', [])) == 1 and route.get('role') == 'equipment_connection':
                ids += route['point_ids']
            for edge in dict.fromkeys(value for value in ids if value):
                self.route_ids[edge].append(route['id'])
        self._elbows_by_route = defaultdict(list)
        for elbow in self.elbows:
            self._elbows_by_route[elbow['route_id']].append(elbow)

    @cached_property
    def role_context(self):
        return position_context(self.piping, self.valves.values(), components=self.pieces.values())

    def port(self, endpoint):
        unit, port = endpoint.split('.', 1)
        return self.units.get(unit, {}).get('ports', {}).get(port, {})

    def point(self, endpoint):
        if endpoint in self.ports and 'position' in self.ports[endpoint]:
            return self.ports[endpoint]['position']
        return self.port(endpoint).get('pos', self.junctions.get(owner(endpoint), {}).get('position'))

    def level(self, endpoint):
        point = self.point(endpoint)
        if point is None:
            return None
        return max((i for i, z in enumerate(self.geometry['floors']) if z <= point[2]+EPS), default=0)

    def level_band(self, level):
        zones = self.piping.get('rack_zones', {})
        zone = zones.get(level, zones.get(str(level)))
        return (zone['bottom_z_m'], zone['top_z_m']) if zone else None

    def valves_on(self, edge_id):
        return [v for v in self.valves.values()
                if (v.get('source_edge') or v.get('edge') or v.get('source_requirement')) == edge_id]

    def route_names(self, edge_id):
        measured = self.connections.get(edge_id, {}).get('routes', ())
        return list(dict.fromkeys([*self.route_ids[edge_id], *measured]))

    def edge_points(self, edge_id):
        edge = self.edges[edge_id]
        measured = self.connections.get(edge_id, {})
        if measured.get('actual_endpoints'):
            return measured['actual_endpoints']
        start, end = self.point(edge['a']), self.point(edge['b'])
        if measured.get('actual_source_position') is not None:
            start = measured['actual_source_position']
        if measured.get('actual_touch_points'):
            end = measured['actual_touch_points'][0]
        return start, end

    def route_segments(self, route_names):
        segments = {}
        for name in route_names:
            route = self.routes.get(name, {})
            for cid in route.get('components', ()):
                part = self.pieces.get(cid, {})
                if part.get('elbow_id') or part.get('kind') in ('piping_blind', 'piping_flange'):
                    continue
                ends = line_ends(part)
                if ends and math.dist(*ends) > EPS:
                    segments[cid] = {'id': cid, 'ends': ends, 'kind': part.get('kind'), 'dn': part.get('dn')}
            for elbow in self._elbows_by_route[name]:
                cid = elbow['id']
                if all(piece in self.pieces for piece in elbow['components']):
                    segments[cid] = {'id': cid, 'ends': line_ends(elbow), 'kind': 'elbow', 'dn': elbow.get('dn')}
            for vid in route.get('valves', ()):
                valve = self.valves.get(vid, {})
                faces = valve_faces(valve)
                if len(faces) == 2:
                    segments[vid] = {'id': vid, 'ends': [faces['in'], faces['out']],
                                     'kind': 'valve', 'dn': valve.get('dn')}
        return segments

    @lru_cache(maxsize=256)
    def edge_path(self, edge_id):
        segments = self.route_segments(self.route_names(edge_id))
        for valve in self.valves_on(edge_id):
            faces = valve_faces(valve)
            if len(faces) == 2:
                segments[valve['id']] = {'id': valve['id'], 'ends': [faces['in'], faces['out']],
                                         'kind': 'valve', 'dn': valve.get('dn')}
        return [{**segments[step['id']], **step}
                for step in _actual_path(*self.edge_points(edge_id), segments.values())]

    def chains(self, first, roles):
        """Follow only process continuations at junctions, ending at real device ports."""
        def visit(edge, trail, seen):
            if owner(edge['b']) in seen:
                return
            trail = (*trail, edge)
            uid = owner(edge['b'])
            if self.units[uid]['kind'] != 'junction':
                yield trail
                return
            for continuation in self.outgoing[uid]:
                if continuation['role'] in roles:
                    yield from visit(continuation, trail, seen | {uid})

        yield from visit(first, (), {owner(first['a'])})

    def chain_path(self, chain):
        legs = [self.edge_path(edge['id']) for edge in chain]
        return [step for leg in legs for step in leg] if legs and all(legs) else []


def direction(ends):
    one, two = map(np.asarray, ends)
    span = np.linalg.norm(two-one)
    return (two-one)/span if span > EPS else np.zeros(3)


def straight_spans(path):
    """Merge collinear pieces through native valve faces for route geometry."""
    spans = []
    for step in path:
        axis = direction(step['ends'])
        if step['kind'] == 'elbow' or sum(abs(axis) > EPS) != 1:
            spans.append(step)
            continue
        if spans and spans[-1]['kind'] != 'elbow':
            previous = spans[-1]
            if (math.dist(previous['ends'][1], step['ends'][0]) <= EPS and
                    np.linalg.norm(direction(previous['ends'])-axis) <= EPS):
                spans[-1] = {**previous, 'ends': [previous['ends'][0], step['ends'][1]]}
                continue
        spans.append(step)
    return [step['ends'] for step in spans if step['kind'] != 'elbow']
