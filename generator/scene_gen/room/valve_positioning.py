"""Fit and orient valve models on the routed pipes."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import json
import math
import os
from pathlib import Path
import shutil
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import valve_position_rules as rules
from .equipment_catalog import reactor_nozzle_mount
from .equipment_details import REACTOR_PORTS
from .export import nums
from .insulation import equipment_skin_parts, route_scope
from .piping_assets import flow_ports, inspect_asset, place_asset
from .piping_end_bundle import DEMAND_ONLY_TERMINALS, TAKEOFF_LOW_TO_HIGH
from .piping_operations import _grips, _mapping
from .pipeline_status import pipeline_status
from .process import write_json
from .valve_roles import position_check, position_context, position_reference
from .valve_route_trace import RunTrace, related_routes
from .valve_sweep_geometry import BARE_CLEARANCE, Geometry, GripClearance, SKIN_THICKNESS


EPS = rules.EPS
STEP = .02
LIMIT = 400
WHEEL_DNS = {50, 65, 80, 100, 250, 300}
WHEEL_ROLES = {'bottom_root', 'recycle_regulation', 'condensate_regulation', 'utility_floor_root'}
JOINT_STATION_MEDIA = {'steam', 'water'}
RESERVATIONS = {'drive_reservation', 'pullout_reservation', 'maintenance_reservation',
                'catalogue_equipment_envelope'}


def read(path):
    return json.loads(path.read_text())


def vector(value):
    return np.asarray(value, dtype=float)


def unit(value):
    return value/np.linalg.norm(value)


def ports(valve):
    return [vector(valve['world_ports'][key]['position']) for key in ('in', 'out')]


def end_box(ends, radius):
    a, b = map(vector, ends)
    pad = radius*np.sqrt(np.maximum(0., 1.-unit(b-a)**2))
    return np.r_[np.minimum(a,b)-pad, np.maximum(a,b)+pad]


def union(boxes):
    return np.r_[boxes[:,:3].min(axis=0), boxes[:,3:].max(axis=0)]


def straight_coverage(parts, valves, outside_diameters):
    """Canonical covered intervals, including the material occupied by valve ports."""
    lines=defaultdict(list)
    def add(ends,radius):
        a,b=map(vector,ends)
        axes=np.flatnonzero(abs(a-b)>EPS)
        if len(axes)!=1:
            raise ValueError('Saved straight or valve port is not on one axis')
        axis=int(axes[0])
        lateral=tuple(round(float(a[i]),6) for i in range(3) if i!=axis)
        ends=sorted(round(float(p[axis]),6) for p in (a,b))
        lines[(axis,lateral,round(radius,6))].append(ends)
    for part in parts:
        if part['kind']=='pipe' and part['shape']=='cylinder':
            add(part['ends'],part['radius'])
    for valve in valves:
        add(ports(valve),outside_diameters[str(valve['dn'])]/2000.)
    merged={}
    for key, intervals in lines.items():
        covered=[]
        for lo,hi in sorted(intervals):
            if covered and lo<=covered[-1][1]+EPS:
                covered[-1][1]=max(hi,covered[-1][1])
            else:
                covered.append([lo,hi])
        merged[key]=covered
    return merged


@dataclass
class Seat:
    start: np.ndarray
    end: np.ndarray
    axis: np.ndarray
    level: int
    route: dict | None
    parts: tuple
    inlet: np.ndarray | None = None
    outlet: np.ndarray | None = None
    elbows: tuple = ()
    approach_end: np.ndarray | None = None
    approach_distance: float = 0.

    def cuts(self, inlet, outlet):
        ends = ((self.start, inlet), (outlet, self.end))
        return [(part, a, b) for part, (a, b) in zip(self.parts, ends) if part is not None]


@dataclass
class Placement:
    preference: tuple
    seat: Seat
    evidence: dict
    origin: np.ndarray
    operation: np.ndarray
    group: tuple
    facing: np.ndarray
    outward: tuple | None
    shape: tuple


def wheel_policy(valve, graph):
    endpoint = valve.get('at') or ''
    device, _, port = endpoint.partition('.')
    heater_steam = port == 'utility_in' and graph['units'].get(device, {}).get('service') == 'steam_heater'
    designated = valve.get('role') in WHEEL_ROLES or heater_steam or valve['dn'] >= 250
    return ('wheel' if designated and valve['dn'] in WHEEL_DNS else 'lever',
            designated and valve['dn'] not in WHEEL_DNS)


def restore_saved_port_faces(graph, piping):
    """Previously saved process units can retain only the logical DN for an already built port."""
    material = piping['material']['requirements'].get('ports', {})
    services = {r['endpoint']: r for r in piping.get('requirements', {}).get('service_points', ())}
    for endpoint in material.keys() | services.keys():
        uid, _, name = endpoint.partition('.')
        port = graph['units'].get(uid, {}).get('ports', {}).get(name)
        record = material.get(endpoint, services.get(endpoint, {}))
        if port is None or port.get('pos') is not None or record.get('position') is None:
            continue
        port['pos'] = record['position']
        port['normal'] = record['normal']
        source = record.get('source_port', {})
        catalogue = source.get('catalogue_face') or record.get('source_catalogue_face')
        if catalogue:
            port['catalogue_face'] = catalogue
            port['source_tag'] = catalogue.get('source_tag')
        if source.get('nozzle_mount'):
            port['nozzle_mount'] = source['nozzle_mount']


def port_mount(unit, key):
    """Read the port schedule, including its original fields in previously saved rooms."""
    port = unit.get('ports', {}).get(key, {})
    if 'nozzle_mount' in port:
        return port['nozzle_mount']
    if unit.get('kind') == 'jacket':
        return 'jacket'
    if unit.get('kind') != 'vessel' or 'pos' not in port:
        return None
    if unit.get('form', unit.get('orientation')) == 'horizontal':
        row = next((item for item in unit.get('nozzle_schedule', ()) if item['port'] == key), {})
        if 'nozzle_mount' in row:
            return row['nozzle_mount']
        # The archived tank schedule distinguishes shell ports and the f1/f2 heads.
        if row:
            return 'end_head' if row['drawing_tag'] in ('f1', 'f2') else 'shell'
        return None
    tag = port.get('source_tag')
    if tag:
        model = next((model for model in REACTOR_PORTS['mount']
                      if REACTOR_PORTS.get(key, {}).get(model) == tag), None)
        return REACTOR_PORTS['mount'].get(model, {}).get(tag)
    return reactor_nozzle_mount(key)


def equipment_sides(valve, graph):
    at = valve.get('at') or ''
    uid, _, key = at.partition('.')
    equip = graph['units'].get(uid, {})
    port = equip.get('ports', {}).get(key, {})
    if 'pos' not in port:
        return None
    kind = equip.get('kind')
    exchange = kind in ('exchanger','heater','condenser','reflux_condenser')
    if exchange and equip.get('form',equip.get('orientation')) == 'horizontal':
        rect=vector(equip['rect'])
        long_axis=int(np.argmax(rect[2:]-rect[:2]))
        side=np.eye(2)[1-long_axis]
        return (side,-side)
    if not exchange:
        if port_mount(equip, key) not in ('upper_head', 'lower_head'):
            return None
    radial = vector(port['pos'][:2])-vector(equip['center'][:2])
    return rules.outward_quadrants(radial) or None


def rotation(axis, stem):
    return np.column_stack((axis, np.cross(stem, axis), stem))


def stems(axis):
    up = np.array([0.,0.,1.])
    base = up-axis*(axis@up)
    if np.linalg.norm(base) < EPS:
        base = np.array([1.,0.,0.])
    base = unit(base)
    quadrant = np.cross(axis, base)
    return ((base,0),(quadrant,90),(-base,180),(-quadrant,270))


def wheel_face(asset, grip):
    axis = vector(asset['params']['motion']['wheel_axis'])
    return axis*math.copysign(1.,axis@grip)


def operating_directions(shape, axis, stem):
    asset, grip, closed_grip = shape
    matrix=rotation(axis,stem)
    if asset['operator']=='wheel':
        direction=matrix@wheel_face(asset,grip)
        return direction,direction
    plane=np.array([1.,1.,0.])
    facing=closed_grip if abs(axis[2])>.5 else grip
    return matrix@(facing*plane),matrix@(grip*plane)


def station_orientation(shape, axis, stem, low_side, *, upright_lever=False):
    """Keep the lever upright or put the handwheel face toward the low end."""
    if shape[0]['operator']=='lever' and (upright_lever or low_side is not None):
        return [] if stem[2]>1-EPS else ['Utility root valve lever must sit above the pipe']
    facing, _ = operating_directions(shape,axis,stem)
    return [] if low_side is None or facing@low_side>1-EPS else ['Utility root valve wheel face does not point toward the lower return end']


@lru_cache(maxsize=None)
def asset_shape(dn, operator):
    asset = inspect_asset(dn, operator)
    model = mujoco.MjModel.from_xml_path(asset['model_xml'])
    data = mujoco.MjData(model)
    _, _, root, joints, _, _ = _mapping({**asset, 'id': 'asset', 'prefix': ''}, model)
    for joint in joints.values():
        data.qpos[joint['address']] = joint['open']
    mujoco.mj_kinematics(model, data)
    grip = vector(_grips(asset, data, root)[0]).mean(axis=0)
    joint = joints[asset['motion']['operator_joint']]
    initial = data.xmat[root].reshape(3, 3).T
    pivot = initial @ (data.xanchor[joint['id']] - data.xpos[root])
    axis = initial @ data.xaxis[joint['id']]
    angle = joint['closed'] - joint['open']
    delta = grip - pivot
    closed_grip = pivot + delta * math.cos(angle) + np.cross(axis, delta) * math.sin(angle) + axis * (delta @ axis) * (1 - math.cos(angle))
    return asset, grip, closed_grip


class Room:
    def __init__(self, source):
        self.source = source
        self.graph, self.scene, self.piping, self.measured = (read(source/name) for name in
            ('process.json','geometry.json','piping.json','measure.json'))
        restore_saved_port_faces(self.graph, self.piping)
        self.original = read(source.parent/'layout/sizing.json')['graph']
        self.parts = {c['id']:c for c in self.scene['components']}
        self.trap_inlet_edges = {edge['id'] for edge in self.graph['edges']
                                 if self.graph['units'][edge['b'].split('.')[0]]['kind'] == 'automatic_steam_trap'}
        self.saddles = defaultdict(list)
        for part in self.parts.values():
            if part.get('catalogue_source_solid') and '_saddle_' in part['id']:
                self.saddles[part['owner']].append((part['id'], vector(part['box'])))
        self.routes = self.piping['routes']
        connection_rows = [*self.measured['connections']['rows'],
                           *self.measured.get('material', {}).get('connections', {}).get('rows', ())]
        self.disconnected = {row.get('source_edge') or row['requirement']
                             for row in connection_rows if row.get('connected') is False}
        self.component_contacts = defaultdict(list)
        for contact in self.piping.get('allowed_contacts', ()):
            first, second = contact['pair']
            point = vector(contact['point'])
            self.component_contacts[first].append((second, point))
            self.component_contacts[second].append((first, point))
        bundle = self.piping.get('end_bundle_station',{})
        self.station_connections = {(c['medium'],c['level']):c for c in bundle.get('connections',[])}
        mains = {r['medium']:r for r in bundle.get('risers',[])}
        supply = [mains[medium] for medium in TAKEOFF_LOW_TO_HIGH if medium in mains]
        self.demand_only = bundle.get('terminals_policy')==DEMAND_ONLY_TERMINALS
        reordered = (self.demand_only or
                     any(r.get('low_to_high_pass') is True for r in
                         bundle.get('spacing',{}).get('planned_takeoff_spacing',[])))
        self.station_low_side = (unit(np.r_[vector(supply[0]['xy'])-vector(supply[-1]['xy']),0.])
                                 if reordered and len(supply)>1 else None)
        self.hot_edges, self.hot_routes, equipment = route_scope(self.graph,self.piping)
        scoped = self.hot_routes
        self.wrapped = {name for route in self.routes if route['id'] in scoped for name in route.get('valves', ())}
        self.insulated = {cid for r in self.routes if r['id'] in scoped for cid in r['components']
                          if cid in self.parts and self.parts[cid]['kind'] in ('pipe', 'piping_elbow', 'piping_reducer')}
        self.insulated.update(part['id'] for uid in equipment for part in equipment_skin_parts(uid, self.scene))
        self.existing = {v['id']:v for v in self.graph['valves']}
        self.slot_by_valve = {key:[r for r in self.routes if key in r.get('valves', [])]
                              for key in self.existing}
        self.floor_bounds = [[z, min((surface['box'][2] for surface in self.scene['surfaces']
                                      if surface['level']==idx+1),default=self.scene['shell']['roof_z_m'])]
                             for idx,z in enumerate(self.scene['floors'])]
        # When refitting a saved room, keep its assigned floor. Archived ranges
        # supply the assignment only; the search follows the real pipe.
        self.frozen_operating_floors = {
            span['valve']: span['level'] if span.get('level') is not None
            else self.level_at(np.mean(span['section_ends_m'], axis=0))
            for route in self.routes for span in route.get('reserved_valve_spans', ())
        }
        self.current = dict(self.existing)
        self.geometry = None
        self.moves = {}
        self.accepted = {}
        self.seats_used = {}
        self.rows = []
        self.role_context = position_context(self.piping,self.graph['valves'],components=self.scene['components'])
        self.original_valves = {v['id']:v for v in self.original['valves']}
        self.fixed_fittings = [(vector(contact['point']), ' / '.join(contact['pair']))
                               for contact in self.piping.get('allowed_contacts',[])
                               if any(self.parts.get(name,{}).get('kind','') in
                                      ('piping_flange','piping_reducer','piping_blind')
                                      for name in contact['pair'])]
        self.fixed_fittings += [(vector(row['position']),row['id'])
                                for row in self.piping.get('flange_connections',[])]
        self.fixed_fittings += [(vector(row['position']),row['id'])
                                for row in self.piping['material'].get('junctions',[])]
        self.obstacles = {key:self.obstacle(c) for key,c in self.parts.items()
                          if not (c.get('annotation_only') or c.get('visual_only') or
                                  c['kind'] in RESERVATIONS or key.endswith('_mating_envelope'))}

    def load_geometry(self):
        if self.geometry is None:
            keys = {(v['dn'], v['operator']) for v in self.current.values()}
            keys.update((r['dn'], wheel_policy(r, self.graph)[0]) for r, _ in self.worklist() if r['dn'])
            assets = {key: asset_shape(*key)[0] for key in keys}
            self.geometry = Geometry(self.source, assets, self.insulated, self.parts)
        return self.geometry

    def obstacle(self, part):
        name = part['id']
        skin = name in self.insulated
        if skin and part.get('ends') and part['shape']=='cylinder':
            return end_box(part['ends'],part['radius']+SKIN_THICKNESS)
        return vector(part['box'])+np.array([-SKIN_THICKNESS]*3+[SKIN_THICKNESS]*3)*(skin and part['kind'] not in ('piping_flange','piping_blind'))

    def worklist(self):
        installed_sources = {r['source_valve']:r for r in self.measured['valve_roles']['rows']}
        requirements = {r['id']:r for r in self.piping['material']['requirements']['manual_valves']}
        by_installed = {r['actual_valve']:r for r in installed_sources.values() if r.get('actual_valve')}
        actual_sources = {v.get('source_valve') for v in self.existing.values() if v.get('source_valve')}
        services = {p['id']:p for p in self.piping.get('requirements', {}).get('service_points', ())}
        requests = []
        for name,valve in self.existing.items():
            demand = by_installed.get(name) or installed_sources.get(valve.get('source_valve'), {})
            request={**valve}
            if valve.get('role')!='utility_floor_root' and demand:
                request.update(role=demand['role'],source_valve=demand['source_valve'],
                               at=demand.get('source_at',valve.get('at')),
                               edge=demand.get('source_edge',valve.get('edge')))
                request.update({k:demand[k] for k in ('placement_rule','valve_role','role_label',
                                                      'sequence_rank','position_endpoint') if k in demand})
            requests.append((request,valve))
        for key, row in installed_sources.items():
            if row.get('actual_valve') in self.existing or key in actual_sources:
                continue
            if key in self.original_valves:
                original = self.original_valves[key]
                req = requirements.get(key,original)
                edge = row.get('source_edge', req.get('edge'))
                service = services.get(edge, {})
                requests.append(({**req, **row, 'id':'FITTED_'+key, 'source_valve':key,
                                  'at':row.get('source_at',req.get('at')), 'edge':edge,
                                  'dn':service.get('dn', req['dn'])},None))
            elif key.startswith('floor:'):
                _,medium,level = key.split(':')
                connection = self.station_connections.get((medium,int(level)))
                if connection:
                    requests.append(({**row,'id':connection['id'],'dn':connection['dn'],'medium':medium,
                                      'level':int(level),'connection_id':connection['id'],
                                      'main_id':connection['main_id']},None))
                else:
                    requests.append(({**row,'id':key,'dn':None},None))
        regulated_edges = {p.get('edge') for p, _ in requests if p.get('role') == 'recycle_regulation'}

        def order(item):
            p, old = item
            uid, _, port = (p.get('at') or '').partition('.')
            head = port_mount(self.graph['units'].get(uid, {}), port) in ('upper_head', 'lower_head')
            # Put the exchanger outlet or condenser root in first, then the
            # regulator, then the return root reached from the downstream tee.
            stage = 0
            if p.get('role') in ('condensate_regulation', 'recycle_regulation'):
                stage = 1
            elif p.get('placement_rule') == 'before_join' and p.get('edge') in regulated_edges:
                stage = 2
            return stage, not head, wheel_policy(p,self.graph)[0]!='wheel' if p['dn'] else True, -(p['dn'] or 0), p['id']
        return sorted(requests,key=order)

    def level_at(self, point):
        return max((i for i, (floor, _) in enumerate(self.floor_bounds) if floor <= point[2] + EPS),
                   default=0)

    def original_seat(self, old):
        a,b = ports(old)
        axis = unit(b-a)
        pair, extents = [], []
        for point, direction in ((a,-axis),(b,axis)):
            touching = [(c,vector(y)) for c in self.parts.values() if c['kind']=='pipe' and c['shape']=='cylinder'
                        for x,y in (c['ends'],c['ends'][::-1])
                        if np.linalg.norm(vector(x)-point)<EPS and
                        (vector(y)-point)@direction>EPS and np.linalg.norm(np.cross(vector(y)-point,axis))<EPS]
            if len(touching)==1:
                part, other = touching[0]
            else:
                part, other = None,point
            pair.append(part);extents.append(other)
        attached = self.slot_by_valve.get(old['id'],())
        route = next((row for row in attached if row.get('source_edge')==old.get('edge')),None)
        level = self.level_at((a + b) / 2)
        return Seat(*extents,axis,level,route,tuple(pair),a,b)

    def role_start(self, request, old):
        endpoint = request.get('position_endpoint') or request.get('at')
        edge = next((e for e in self.graph['edges'] if e['id'] == request.get('edge')), None)
        rule = request.get('placement_rule')
        source = self.original_valves.get(request.get('source_valve'))
        if request.get('role') == 'utility_floor_root':
            connection = self.station_connections[request['medium'], request['level']]
            return vector(connection['main_tap']), connection['level']
        if rule == 'free' and old is not None:
            point = ports(old)[0]
            return point, self.level_at(point)
        graph = {**self.original, 'units': self.graph['units']}
        point = position_reference(graph, source, self.role_context, request) if source else None
        if rule == 'after_split' and edge is not None:
            root = next((v for v in self.current.values() if v.get('role') == 'heater_outlet'
                         and v.get('edge') == edge['id']), None)
            if root is not None and point is not None:
                point = max(ports(root), key=lambda end: math.dist(end, point))
        if point is None:
            endpoint = endpoint if endpoint and endpoint != 'branch' else edge['a'] if edge else ''
            device, _, port = endpoint.partition('.')
            point = self.graph['units'].get(device, {}).get('ports', {}).get(port, {}).get('pos')
        if point is None:
            return None, None
        return vector(point), self.level_at(point)

    def seats(self, request, old):
        target, level = self.role_start(request, old)
        level = self.frozen_operating_floors.get(request.get('source_valve', request['id']), level)
        if old is not None:
            original = self.original_seat(old)
            if level is not None:
                original.level = level
            yield original
            return
        initial = self.slot_by_valve.get(request['id'], ())
        routes = [r for r in related_routes(self, request, initial) if r['dn'] == request['dn']]
        if target is None or not routes:
            return
        stops = [*self.fixed_fittings,
                 *((p, name) for name, v in self.current.items() if name != request['id'] for p in ports(v))]
        trace = RunTrace(routes, self.parts, self.piping['elbows'], stops)
        local_routes = ()
        if request.get('edge') in self.disconnected:
            endpoint = request.get('position_endpoint')
            local_routes = {r['id'] for r in routes if any(
                (r.get(side) or {}).get('endpoint') == endpoint for side in ('source', 'target'))}
        for distance, approach, _, part, route, elbows in trace.from_reference(target, local_routes):
            start, end = map(vector, part['ends'])
            yield Seat(start, end, unit(end-start), level, route, (part, part),
                       elbows=elbows, approach_end=approach, approach_distance=distance)

    def route_check(self, request, seat, inlet, outlet):
        source = self.original_valves.get(request.get('source_valve'))
        if source is None or request.get('role')=='utility_floor_root':
            return True
        keys = ('role','placement_rule','position_endpoint')
        if not all(request.get(k) for k in keys):
            return seat.inlet is not None and np.linalg.norm(seat.inlet-inlet)<EPS
        policy = {k:request[k] for k in keys}
        if seat.route is None and seat.inlet is not None and np.linalg.norm(seat.inlet-inlet)<EPS:
            return True
        partial = request.get('edge') in self.disconnected
        return position_check({**self.original,'units':self.graph['units']},source,self.role_context,
                              {'in':inlet,'out':outlet},policy,allow_at_port_elbow=True,
                              partial=partial)['valid']

    def station_side(self, request):
        if request.get('role')=='utility_floor_root' and request.get('medium') in TAKEOFF_LOW_TO_HIGH:
            return self.station_low_side
        return None

    def upright_station_handle(self, request):
        return self.demand_only and request.get('role')=='utility_floor_root'

    def joint_station_key(self, request):
        key = request.get('medium'), request.get('level')
        if (key in self.station_connections and key[0] in JOINT_STATION_MEDIA
                and self.station_side(request) is not None):
            return key
        return None

    def jacket(self, valve, matrix, origin):
        asset = asset_shape(valve['dn'], valve['operator'])[0]
        body = np.asarray([g['bbox'] for g in asset['local_geoms'] if not g['moving']
                           and g['name'].startswith(('visual_valve_body', 'visual_welded_body',
                           'visual_center_body', 'visual_left_end', 'visual_right_end',
                           'visual_left_rf_end', 'visual_right_rf_end', 'visual_bonnet', 'collar_'))])
        lo, hi = np.split(union(body), 2)
        span = sorted(port['position'][0] for port in asset['local_ports'].values())
        lo[0], hi[0] = span
        if valve['operator'] == 'wheel':
            return ('box', lo - [0., SKIN_THICKNESS, SKIN_THICKNESS], hi + [0., SKIN_THICKNESS, 0.], matrix, origin)
        ends = [origin + matrix @ [x, 0., 0.] for x in span]
        radius = max(abs(lo[1]), abs(hi[1]), abs(lo[2])) + SKIN_THICKNESS
        return ('pipe', ends, radius, end_box(ends, radius))

    def skin_hits(self, sweep, skin, grip=None):
        if skin is None:
            return False
        kind, *dimensions = skin
        return (self.geometry.box_hit(sweep, *dimensions, grip=grip) if kind == 'box'
                else self.geometry.pipe_hit(sweep, *dimensions, moving_only=True, insulated=True, grip=grip))

    def valve_clash(self, first, second, first_skin=None, second_skin=None, grip=None):
        hit = self.geometry.peer_hit(first, second, grip)
        if hit:
            return hit
        if self.skin_hits(first, second_skin, grip):
            return "This valve's operating space intersects neighboring insulation"
        if self.skin_hits(second, first_skin):
            return "The neighboring valve's operating space intersects this valve's insulation"
        return None

    def candidate_skin(self, request, seat, asset, matrix, origin):
        if (request['id'] in self.wrapped or request.get('edge') in self.hot_edges
                or seat.route and seat.route['id'] in self.hot_routes):
            return self.jacket(asset, matrix, origin)
        return None

    def saddle_preference(self, request, center, facing):
        equipment = (request.get('at') or '').split('.')[0]
        saddles = self.saddles.get(equipment)
        if request.get('edge') not in self.trap_inlet_edges or not saddles:
            return False, None
        name, bounds = min(saddles, key=lambda row: np.linalg.norm(
            np.maximum(0., np.maximum(row[1][:3]-center, center-row[1][3:]))))
        delta = (bounds[:3] + bounds[3:]) / 2 - center
        return bool(facing[:2] @ delta[:2] > EPS), name

    def mating_components(self, request, seat, inlet, outlet, nearby):
        """Exact weld continuations and flanges at the chosen mating faces."""
        owned = {part['id'] for part in seat.parts if part is not None}
        routes = related_routes(self, request, (seat.route,) if seat.route else ())
        line = {name for route in routes for name in route['components']}
        mating = set()
        for name, _ in nearby:
            if name not in line:
                continue
            part = self.parts[name]
            if part['kind'] == 'pipe' and part['shape'] == 'cylinder' and part.get('dn') == request['dn']:
                first, second = map(vector, part['ends'])
                for joint, outward in ((seat.start, -seat.axis), (seat.end, seat.axis)):
                    for end, far in ((first, second), (second, first)):
                        if (math.dist(end, joint) <= EPS and float((far-end) @ outward) > EPS
                                and np.linalg.norm(np.cross(far-end, outward)) <= EPS):
                            mating.add(name)
            elif part['kind'] == 'piping_flange':
                if any(peer in owned and min(math.dist(point, face) for face in (inlet, outlet)) <= EPS
                       for peer, point in self.component_contacts[name]):
                    mating.add(name)
        return mating

    def candidate(self, request, seat, shape, inlet, stem, outward, motion, obstacles, peers, *, reverse=False):
        asset, grip, _ = shape
        axis = seat.axis
        low_side = self.station_side(request)
        mount_axis = -axis if reverse else axis
        station_errors = station_orientation(
            shape, mount_axis, stem, low_side, upright_lever=self.upright_station_handle(request))
        matrix = rotation(mount_axis,stem)
        origin = inlet-matrix@vector(flow_ports(asset,reverse)['in']['position'])
        outlet = inlet+axis*asset['install_length_m']
        geometry = self.load_geometry()
        sweep = geometry.valve(asset, matrix, origin)
        envelope = sweep.span
        floor, upper = self.floor_bounds[seat.level]
        operation = origin+matrix@grip
        _,installation=operating_directions(shape,mount_axis,stem)
        native_sweep = geometry.assets[(asset['dn'], asset['operator'])][0]
        operator_center = origin + matrix @ native_sweep.operator_center
        radial = operator_center - (inlet + outlet) / 2
        facing = radial - axis * (axis @ radial)
        vertical=request.get('placement_rule')!='top_vertical' or abs(axis[2])>1-EPS
        checks = {
            '1':rules.on_straight(inlet,outlet,axis,(seat.start,seat.end),motion[1],motion[0])
                +([] if vertical else ['Vertical top-port valves require a vertical straight']),
            '2':rules.within_floor(envelope,floor,upper),
            '3':[],
            '4':rules.operator_orientation(stem,installation),
            '5': [],
        }
        clearance = GripClearance()
        checks['6'] = []
        for part,a,b in seat.cuts(inlet,outlet):
            if np.linalg.norm(a-b)>EPS:
                name = part['id']
                insulated = name in self.insulated
                radius = part['radius']+(SKIN_THICKNESS if insulated else 0.)
                box = end_box((a,b),radius)
                # Only the moving operator needs clearance from its own mating pipes.
                if geometry.pipe_hit(sweep, (a,b), radius, box, mating=True, insulated=insulated, grip=clearance):
                    checks['6'].append(name)
        names, boxes = obstacles
        limit = max(BARE_CLEARANCE, clearance.distance)
        near = np.all(np.minimum(envelope[3:] + limit, boxes[:,3:]) -
                      np.maximum(envelope[:3] - limit, boxes[:,:3]) > EPS, axis=1)
        fixed_names = [(name, box) for name, box in zip(names[near], boxes[near]) if name in self.parts]
        mating = self.mating_components(request, seat, inlet, outlet, fixed_names)
        stable_names = [(name, box) for name, box in fixed_names if name not in self.moves]
        checks['6'].extend(geometry.obstacles(sweep, (name for name, _ in stable_names),
                                             (box for _, box in stable_names), mating=mating, grip=clearance))
        for name, box in fixed_names:
            if name in self.moves:
                part = self.parts[name]
                insulated = name in self.insulated
                radius = part['radius'] + (SKIN_THICKNESS if insulated else 0.)
                if geometry.pipe_hit(sweep, part['ends'], radius, box, mating=name in mating,
                                     insulated=insulated, grip=clearance):
                    checks['6'].append(name)
        first_skin = self.candidate_skin(request, seat, asset, matrix, origin)
        center = (inlet + outlet) / 2
        spacing = []
        for valve in peers:
            name = valve['id']
            second_skin = (self.jacket(valve, vector(valve['matrix']), vector(valve['origin']))
                           if name in self.wrapped else None)
            second = geometry.peer(valve)
            hit = self.valve_clash(sweep, second, first_skin, second_skin, clearance)
            if hit:
                checks['5'].append(f'{name}: {hit}')
            other_a, other_b = ports(valve)
            other_axis = unit(other_b - other_a)
            if abs(axis @ other_axis) > 1 - EPS:
                delta = (other_a + other_b) / 2 - center
                radial = np.linalg.norm(delta - (axis @ delta) * axis)
                axial = abs(delta @ axis)
                loose_gap = np.linalg.norm(np.maximum(0., np.maximum(
                    envelope[:3] - second.span[3:], second.span[:3] - envelope[3:])))
                overlap = axial < (abs(axis) @ (envelope[3:] - envelope[:3] + second.span[3:] - second.span[:3])) / 2
                if loose_gap < .10 or radial < .50 and overlap:
                    spacing.append({'valve': name, 'axis_distance_m': float(radial),
                                    'bounds_gap_m': float(loose_gap), 'axial_projection_overlap': bool(overlap)})
        score = max(floor+.6-operation[2],0.,operation[2]-(floor+1.6))
        front = operation[:2]-origin[:2]
        if np.linalg.norm(front)<EPS:
            front = stem[:2]
        front = operation[:2]+.6*unit(front) if np.linalg.norm(front)>EPS else operation[:2]
        side_ok = any(row['level']==seat.level and np.all(front>=row['box'][:2])
                      and np.all(front<=row['box'][3:5]) for row in self.scene['surfaces'])
        group = (seat.level, request.get('role'), request.get('medium'))
        heights = [r['operation_m'][2] for r in self.rows if r['status']=='placed' and r['group']==list(group)]
        if seat.inlet is not None:
            move = math.dist(inlet, seat.inlet)
        elif seat.approach_end is not None:
            move = seat.approach_distance + min(math.dist(end, seat.approach_end) for end in (inlet, outlet))
        else:
            move = math.dist(inlet, seat.start)
        direction_rank = rules.direction_preference(facing, outward)
        faces_saddle, saddle = self.saddle_preference(request, center, facing)
        pref = (faces_saddle, direction_rank, bool(station_errors), seat.inlet is not None and move > EPS,
                clearance.tight, score,not side_ok,
                abs(operation[2]-np.mean(heights)) if heights else 0.,
                -clearance.distance,move)
        row = {'in_m':inlet.tolist(),'out_m':outlet.tolist(),'stem':stem.tolist(),'rules':checks,
               'parallel_spacing_hints':spacing,
               'grip_clearance_m':clearance.distance if math.isfinite(clearance.distance) else None,
               'grip_tight':clearance.tight, 'direction_rank':direction_rank,
               'operator_sweep_center_m':operator_center.tolist(), 'operator_side':facing.tolist(),
               'nearest_saddle':saddle, 'faces_saddle':faces_saddle,
               'mating_components':sorted(mating)}
        row['reversed_ends'] = reverse
        if low_side is not None:
            station_face = matrix @ wheel_face(asset, grip) if asset['operator'] == 'wheel' else stem
            row.update(low_side=low_side.tolist(),face_direction=station_face.tolist())
        return pref, row, origin, operation, group, facing

    def locations(self, seat, length, max_points=LIMIT//8):
        if seat.inlet is not None and None in seat.parts:
            yield seat.inlet if seat.parts[0] is None else seat.outlet-seat.axis*length
            return
        span = float((seat.end-seat.start) @ seat.axis)
        low, high = 2*EPS, span-length-2*EPS
        if seat.approach_end is not None:
            approach = float((seat.approach_end-seat.start) @ seat.axis)
            forward = approach < span/2
            for point, _ in self.fixed_fittings:
                delta = point-seat.start
                station = float(delta @ seat.axis)
                if EPS < station < span-EPS and np.linalg.norm(delta-station*seat.axis) < EPS:
                    if forward:
                        high = min(high, station-length-2*EPS)
                    else:
                        low = max(low, station+2*EPS)
        if high < low:
            return
        if seat.inlet is not None:
            reference = float((seat.inlet-seat.start) @ seat.axis)
        else:
            reference = low if seat.approach_end is None or forward else high
        reference = float(np.clip(reference, low, high))
        offsets = {reference, low, high}
        offsets.update(reference+STEP*k for k in range(math.ceil((low-reference)/STEP),
                                                       1+math.floor((high-reference)/STEP)))
        ordered = sorted(offsets, key=lambda v: (abs(v-reference), v))
        if len(ordered) > max_points:
            ordered = [ordered[i] for i in np.linspace(0, len(ordered)-1, max_points, dtype=int)]
        for offset in ordered:
            yield seat.start+seat.axis*offset

    def blockers(self, seat, inlet, outlet):
        axis = seat.axis
        prior = [float((p-seat.start)@axis) for p in (seat.inlet,seat.outlet)] if seat.inlet is not None else []
        lo, hi = ((inlet-seat.start)@axis,(outlet-seat.start)@axis)
        for point,name in self.fixed_fittings:
            delta = point-seat.start
            station = delta@axis
            if np.linalg.norm(delta-station*axis)>EPS:
                continue
            if (lo+EPS < station < hi-EPS or
                prior and (lo < station-EPS < prior[0]-EPS or prior[1]+EPS < station+EPS < hi)):
                # A crossed fitting cannot be skipped by moving past it.
                yield float((lo+hi)/2),name

    def identify(self, request, valve, old, seat):
        stable = ('id','name','prefix','geom_prefix','role','medium','medium_identity','source_edge',
                  'edge','source_requirement','source_valve','at','phase','equipment','valve_role','role_label',
                  'placement_rule','sequence_rank','position_endpoint','main_tee_m','group_id','manual',
                  'main_id','connection_id','end_station_method')
        valve.update({k:request[k] for k in stable if k in request})
        key=request['id']
        valve.update(id=key, prefix=key+'_', geom_prefix=key+'_', level=seat.level,manual=True,
                     operating_floor_z_m=self.floor_bounds[seat.level][0])
        if old is None:
            route=seat.route
            valve.update(name=request.get('source_role',key),phase=route.get('phase','P1'),source_edge=request.get('edge'),
                         medium=route['medium'],medium_identity=route['medium'])
        return valve

    def placements(self, request, old, *, peer=None, peer_parts=()):
        key=request['id']
        output={'id':key,'dn':request['dn'],'role':request.get('role'),'edge':request.get('edge'),
                'endpoint':request.get('at'),'old':bool(old),'status':'pending','attempts':[],
                'source_valve':request.get('source_valve', key),
                'placement_rule':request.get('placement_rule'),
                'disconnected_pipe':request.get('edge') in self.disconnected,
                'orientation_refusals':[]}
        if request['dn'] is None:
            output['reason']='No matching same-floor main tee'
            output['best_failure']={'rules': {'1': [output['reason']]}, 'no_candidate': True}
            return output, []
        operator,missing=wheel_policy(request,self.graph)
        device, _, port = (request.get('at') or '').partition('.')
        equipment = self.graph['units'].get(device, {})
        output.update(operator=operator,wheel_dn_missing=missing, port_mount=port_mount(equipment, port),
                      port_reference_m=equipment.get('ports', {}).get(port, {}).get('pos'))
        shape=asset_shape(request['dn'],operator)
        length=shape[0]['install_length_m']
        output['installation_length_m'] = length
        peers = [valve for name, valve in self.current.items() if name not in (key, peer)]
        available=[]
        seats = list(self.seats(request,old))
        output['straight_search'] = []
        for index, seat in enumerate(seats):
            if len(output['attempts']) >= LIMIT:
                break
            own = {part['id'] for part in seat.parts if part is not None}
            fixed={name:box for name,box in self.obstacles.items()
                   if name not in own and name not in peer_parts}
            obstacles = (np.array(list(fixed)), np.array(list(fixed.values())).reshape(-1, 6))
            outward=equipment_sides(request,self.graph)
            points = max(1, (LIMIT-len(output['attempts'])) // (8*(len(seats)-index)))
            positions = list(self.locations(seat, length, points))
            output['straight_search'].append({'route':seat.route['id'] if seat.route else None,
                'pipe':seat.parts[0]['id'] if seat.parts[0] else None, 'ends_m':[seat.start.tolist(),seat.end.tolist()],
                'length_m':float(math.dist(seat.start,seat.end)), 'level':seat.level,
                'crossed_elbows':list(seat.elbows), 'positions':len(positions)})
            choices = []
            for stem, angle in stems(seat.axis):
                for reverse in (False, True):
                    axis = -seat.axis if reverse else seat.axis
                    _, installation = operating_directions(shape, axis, stem)
                    errors = {'4': rules.operator_orientation(stem, installation)}
                    if any(errors.values()):
                        output['orientation_refusals'].append({'axis': seat.axis.tolist(), 'angle_deg': angle,
                                                               'reversed_ends': reverse, 'rules': errors,
                                                               'no_position': not positions})
                    else:
                        choices.append((stem, angle, reverse))
            if not choices:
                continue
            for inlet in positions:
                outlet=inlet+seat.axis*length
                packing = list(self.blockers(seat,inlet,outlet))
                if seat.outlet is not None and seat.parts[1] is None and np.linalg.norm(outlet-seat.outlet)>EPS:
                    packing.append((float(((inlet+outlet)/2-seat.start)@seat.axis),'original rigid outlet face'))
                role_ok=self.route_check(request,seat,inlet,outlet)
                for stem,angle,reverse in choices:
                    if len(output['attempts'])>=LIMIT:
                        break
                    pref,row,origin,operation,group,facing=self.candidate(
                        request,seat,shape,inlet,stem,outward,(role_ok,packing),obstacles,peers,reverse=reverse)
                    row.update(angle_deg=angle, host=seat.route['id'] if seat.route else None,
                               crossed_elbows=list(seat.elbows))
                    output['attempts'].append(row)
                    if not any(row['rules'].values()):
                        available.append(Placement(pref,seat,row,origin,operation,group,facing,outward,shape))
                if len(output['attempts'])>=LIMIT:
                    break
            if len(output['attempts'])>=LIMIT:
                break
        if not available:
            failures = [row for row in output['attempts'] if any(row['rules'].values())]
            if not failures:
                failures = [row for row in output['orientation_refusals'] if row.get('no_position') is False]
            if failures:
                best = min(failures, key=lambda row: sum(bool(hits) for hits in row['rules'].values()))
            else:
                reason = ('No same-DN straight is reachable from the role origin' if not seats else
                          'Reachable same-DN straights are too short')
                best = {'rules': {'1': [reason]}, 'no_candidate': True}
            output['best_failure'] = best
            output['reason'] = '、'.join(dict.fromkeys(hit for hits in best['rules'].values() for hit in hits))
        return output, sorted(available,key=lambda trial:trial.preference)

    def install(self, request, old, output, placement):
        key=request['id']
        seat,row,shape=placement.seat,placement.evidence,placement.shape
        reverse = row.get('reversed_ends',False)
        valve=self.identify(request,place_asset(shape[0],placement.origin,-seat.axis if reverse else seat.axis,
                                               vector(row['stem']),reversed_ends=reverse),old,seat)
        a,b=ports(valve)
        self.current[key]=valve
        self.accepted[key] = valve
        if self.candidate_skin(request, seat, shape[0], vector(valve['matrix']), placement.origin):
            self.wrapped.add(key)
        host={'bounds_m':[seat.start.tolist(),seat.end.tolist()],'cut_m':[a.tolist(),b.tolist()],
              'route':row['host'],'crossed_elbows':row['crossed_elbows']}
        sides = [side.tolist() for side in placement.outward] if placement.outward is not None else None
        output.update(status='placed', chosen=row,host=host,operation_m=placement.operation.tolist(),
                      group=list(placement.group),facing=placement.facing.tolist(),outward=sides,
                      high=bool(placement.operation[2]-self.floor_bounds[seat.level][0]>2.),
                      level=seat.level,motion_samples=self.load_geometry().assets[(valve['dn'], valve['operator'])][0].bounds.shape[0])
        if old and seat.elbows:
            self.close_original_cut(old,seat.route)
        seat.parts=self.update_pipe(seat,a,b,key)
        self.seats_used[key]=(request,seat)
        self.role_context = position_context(self.piping, self.current.values(), components=self.parts.values())
        return output

    def select(self, request, old):
        output, available = self.placements(request,old)
        return self.install(request,old,output,available[0]) if available else output

    def pair_trial(self, request, placement):
        row, seat = placement.evidence, placement.seat
        inlet, outlet = vector(row['in_m']), vector(row['out_m'])
        pipes = []
        for part, a, b in seat.cuts(inlet, outlet):
            if math.dist(a, b) > EPS:
                radius = part['radius'] + (SKIN_THICKNESS if part['id'] in self.insulated else 0.)
                pipes.append(((a, b), radius, end_box((a, b), radius), part['id'] in self.insulated))
        asset = placement.shape[0]
        axis = -seat.axis if row['reversed_ends'] else seat.axis
        matrix = rotation(axis, vector(row['stem']))
        geometry = self.load_geometry()
        sweep = geometry.valve(asset, matrix, placement.origin)
        skin = self.candidate_skin(request, seat, asset, matrix, placement.origin)
        return placement, sweep, skin, pipes

    def pair_blocker(self, first, second):
        hit = self.valve_clash(first[1], second[1], first[2], second[2])
        if hit:
            return hit
        for trial, peer in ((first, second), (second, first)):
            for ends,radius,box,insulated in peer[3]:
                if self.geometry.pipe_hit(trial[1], ends,radius,box, insulated=insulated):
                    return 'straight or insulation beside neighboring valve'
        return None

    def select_pair(self, group):
        choices,outputs = [],[]
        for i,(request,old) in enumerate(group):
            peer, previous = group[1-i]
            part_ids = {part['id'] for part in self.original_seat(previous).parts if part is not None} if previous else set()
            output, available = self.placements(request,old,peer=peer['id'],peer_parts=part_ids)
            outputs.append(output)
            choices.append([self.pair_trial(request, trial) for trial in available])
        rejected = Counter()
        accepted = []
        for first,second in product(*choices):
            problem = self.pair_blocker(first,second)
            if problem:
                rejected[problem] += 1
            else:
                accepted.append((first[0],second[0]))
        info = {'valves':[request['id'] for request,_ in group],
                'individual_candidates':[len(candidates) for candidates in choices],
                'joint_matches':len(accepted),'joint_rejections':dict(rejected)}
        for output in outputs:
            output['joint_search'] = info
        if accepted:
            pair = min(accepted,key=lambda trials:tuple(sum(values) for values in
                       zip(*(trial.preference for trial in trials))))
            for (request,old),output,placement in zip(group,outputs,pair):
                self.install(request,old,output,placement)
        else:
            # Retain a single qualified valve if the two cannot both fit.
            # Pull a failed original valve out before testing the next one.
            outputs = []
            for request, old in group:
                row = self.select(request, old)
                row['joint_search'] = info
                if old and row['status'] == 'pending':
                    self.leave_pending(row)
                outputs.append(row)
        return outputs

    def save_cut(self, part, a, b):
        if np.linalg.norm(a-b)<EPS:
            raise ValueError('Trimmed straight has zero length: '+part['id'])
        ends=[a.tolist(),b.tolist()]
        if np.linalg.norm(vector(part['ends'][1])-a)<np.linalg.norm(vector(part['ends'][0])-a):
            ends.reverse()
        if np.allclose(part['ends'],ends,atol=EPS,rtol=0) and part['id'] not in self.moves:
            return
        part.update(ends=ends,box=end_box(ends,part['radius']).tolist())
        self.moves.setdefault(part['id'],{'source':part['id']})['ends']=ends
        self.obstacles[part['id']]=self.obstacle(part)

    def close_original_cut(self, old, new_route=None):
        seat=self.original_seat(old)
        if seat.parts[0] is not None:
            part,a,b=seat.parts[0],seat.start,seat.outlet
        else:
            part,a,b=seat.parts[1],seat.inlet,seat.end
        if part is None:
            raise ValueError('No same-DN straight can restore the previous valve position: '+old['id'])
        self.save_cut(part,a,b)
        for route in self.routes:
            if old['id'] not in route.get('valves',[]):
                continue
            if route is new_route:
                continue
            route['valves'].remove(old['id'])
            for key in ('source','target'):
                endpoint=route.get(key) or {}
                if endpoint.get('valve')==old['id']:
                    endpoint.pop('valve')
                    if part['id'] in route.get('components',[]):
                        endpoint['position']=(b if seat.parts[0] is part else a).tolist()
        for record in self.piping['material']['edge_routes']:
            if ((new_route is None or new_route['id'] not in record.get('routes',[]))
                    and old['id'] in record['valves']):
                record['valves'].remove(old['id'])

    def leave_pending(self, row):
        """Remove a failed manual valve only if its frozen pipe span can be restored."""
        name=row['id']
        valve=self.current.get(name)
        if valve is None or not any(part is not None for part in self.original_seat(valve).parts):
            return False
        self.close_original_cut(valve)
        del self.current[name]
        self.accepted.pop(name,None)
        self.seats_used.pop(name,None)
        row.update(status='pending',disposition='Original coaxial pipe restored pending valve placement')
        return True

    def update_pipe(self, seat, inlet, outlet, key):
        cuts=seat.cuts(inlet,outlet)
        if len(cuts)==2 and cuts[0][0]['id']==cuts[1][0]['id']:
            a,b=cuts[1][1:]
            part=deepcopy(cuts[1][0])
            original=part['id']
            tail = original+'_valve_tail'
            name = tail
            number = 2
            while name in self.parts:
                name = f'{tail}_{number}'
                number += 1
            part['id'] = name
            cuts[1]=(part,a,b)
            self.parts[part['id']]=part
            self.scene['components'].append(part)
            if original in self.insulated:
                self.insulated.add(part['id'])
            route=seat.route
            for column in ('components','pipe_components'):
                if column in route:
                    route[column].insert(route[column].index(original)+1,part['id'])
            if key not in route['valves']:
                route['valves'].append(key)
            for row in self.piping['material']['edge_routes']:
                if route['id'] in row.get('routes',[]):
                    row['components'].append(part['id'])
                    if key not in row['valves']:
                        row['valves'].append(key)
            self.moves[part['id']]={'source':original}
        for part,a,b in cuts:
            self.save_cut(part,a,b)
        # This partition of the original straight is the only pipe edit:
        # same two outside endpoints, axis, OD, fittings and route ownership.
        span=math.dist(seat.start,inlet)+math.dist(inlet,outlet)+math.dist(outlet,seat.end)
        if not math.isclose(span,math.dist(seat.start,seat.end),abs_tol=EPS):
            raise ValueError('Straight coverage changed: '+key)
        updated={i:part for i,(part,_,_) in zip((i for i,part in enumerate(seat.parts) if part is not None),cuts)}
        return tuple(updated.get(i) for i in range(2))

    def solve(self):
        worklist = self.worklist()
        per_floor = defaultdict(list)
        for task in worklist:
            key = self.joint_station_key(task[0])
            if key:
                per_floor[key[1]].append(task)
        handled = set()
        for request,old in worklist:
            if request['id'] in handled:
                continue
            key = self.joint_station_key(request)
            group = per_floor[key[1]] if key else []
            if {task[0]['medium'] for task in group} == JOINT_STATION_MEDIA and len(group)==2:
                rows = self.select_pair(group)
            else:
                group,rows = [(request,old)],[self.select(request,old)]
            for (item,prior),row in zip(group,rows):
                if prior and row['status']=='pending':
                    self.leave_pending(row)
                self.rows.append(row)
                handled.add(item['id'])
        self.graph['valves']=list(self.current.values())
        return self.rows

    def audit(self):
        originals = {request['id']:request for request,_ in self.worklist()}
        checks, grip = {}, {}
        for key,valve in self.current.items():
            if key in self.seats_used:
                request,planned = self.seats_used[key]
                # A later valve can split one of this valve's adjacent pipes.
                seat = self.original_seat(valve)
                seat.route,seat.level = planned.route,planned.level
            else:
                request = originals[key]
                seat = next(self.seats(request, valve), None)
                if seat is None:
                    checks[key] = {str(i): ['Original valve lacks its original straight pipe'] if i == 1 else [] for i in range(1, 7)}
                    continue
            a,b = ports(valve)
            cut_ids={part['id'] for part in seat.parts if part is not None}
            fixed = {name:box for name,box in self.obstacles.items() if name not in cut_ids}
            evidence=(self.route_check(request,seat,a,b),list(self.blockers(seat,a,b)))
            near = (np.array(list(fixed)), np.array(list(fixed.values())).reshape(-1, 6))
            peer = [v for name, v in self.current.items() if name != key]
            outward=equipment_sides(request,self.graph)
            shape=asset_shape(valve['dn'],valve['operator'])
            _,row,*_=self.candidate(request,seat,shape,a,vector(valve['stem']),outward,evidence,near,peer,reverse=valve.get('reversed_ends',False))
            checks[key]=row['rules']
            grip[key]={'clearance_m':row['grip_clearance_m'],'tight':row['grip_tight']}
        return checks,grip


def automatic_geometry(root, scene):
    names={part for auto in scene.get('automatic_components',[]) for part in auto.get('components',[])}
    return {geom.get('name'):geom.attrib for geom in root.findall('worldbody/geom') if geom.get('name') in names}


def save_model(room, folder):
    root=ET.parse(room.source/'room.xml').getroot()
    assets,world=root.find('asset'),root.find('worldbody')
    original_auto=automatic_geometry(root,room.scene)
    files={}
    for element in root.iter():
        if 'file' in element.attrib:
            path=(room.source/element.attrib['file']).resolve()
            element.set('file',os.path.relpath(path,folder))
            if element.tag=='model':
                files[path]=element.attrib['name']
    geometries={c.get('name'):c for c in world.findall('geom')}
    for name,move in room.moves.items():
        if name not in geometries:
            part=deepcopy(geometries[move['source']])
            world.append(part)
            part.set('name',name)
            geometries[name]=part
        geometries[name].set('fromto',nums(move['ends'][0]+move['ends'][1]))
    frames={row.find('attach').get('prefix'):row for row in world.findall('frame') if row.find('attach') is not None}
    for name in room.existing.keys()-room.current.keys():
        world.remove(frames[room.existing[name]['prefix']])
    for name in room.accepted:
        valve=room.current[name]
        path=Path(valve['model_xml']).resolve()
        if path not in files:
            files[path]=f'valve_asset_{len(files)}'
            ET.SubElement(assets,'model',name=files[path],file=os.path.relpath(path,folder))
        frame=frames.get(valve['prefix'])
        if frame is None:
            frame=ET.SubElement(world,'frame')
            ET.SubElement(frame,'attach',prefix=valve['prefix'],body=valve['root_body'])
        frame.set('pos',nums(valve['origin']))
        frame.set('quat',nums(valve['quat']))
        frame.find('attach').set('model',files[path])
    unchanged=rules.automatic_fixed(original_auto,automatic_geometry(root,room.scene))
    if unchanged:
        raise ValueError('Automatic component changed: '+'、'.join(unchanged))
    folder.mkdir(parents=True)
    path=folder/'room.xml'
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    return path, len(original_auto)


def verify_saved(room, path, *, valve_ids=None):
    model=mujoco.MjModel.from_xml_path(str(path))
    data=mujoco.MjData(model)
    mujoco.mj_kinematics(model,data)
    station = {}
    for name in room.accepted if valve_ids is None else valve_ids:
        valve=room.current[name]
        root=model.body(valve['prefix']+valve['root_body']).id
        origin, matrix = data.xpos[root], data.xmat[root].reshape(3,3)
        if not (np.allclose(origin,valve['origin'],atol=EPS,rtol=0)
                and np.allclose(matrix,valve['matrix'],atol=EPS,rtol=0)):
            raise ValueError('Saved valve placement differs on readback: '+name)
        low_side = room.station_side(valve)
        if low_side is not None or (valve['operator']=='lever' and room.upright_station_handle(valve)):
            shape = asset_shape(valve['dn'],valve['operator'])
            asset = shape[0]
            local = flow_ports(asset,valve.get('reversed_ends',False))
            for end, port in valve['world_ports'].items():
                measured = origin+matrix@vector(local[end]['position'])
                if not np.allclose(measured,port['position'],atol=EPS,rtol=0):
                    raise ValueError('Saved utility valve port differs on readback: '+name)
            facing = matrix@wheel_face(asset,shape[1]) if valve['operator']=='wheel' else matrix[:,2]
            expected = low_side if valve['operator']=='wheel' else np.array([0.,0.,1.])
            station[name]={'medium':valve['medium'],'level':valve['level'],'operator':valve['operator'],
                           'facing':facing.tolist(),'low_side':None if low_side is None else low_side.tolist(),
                           'preferred_orientation':bool(facing@expected >= 1-EPS)}
    for name, change in room.moves.items():
        gid=model.geom(name).id
        center=data.geom_xpos[gid]
        delta=data.geom_xmat[gid].reshape(3,3)[:,2]*model.geom_size[gid,1]
        actual=[center-delta,center+delta]
        if not any(np.allclose(actual,order,atol=EPS,rtol=0) for order in
                   (change['ends'],change['ends'][::-1])):
            raise ValueError('Saved pipe port differs on readback: '+name)
    return station


def generate(source, output):
    source,output=source.resolve(),output.resolve()
    if output.is_relative_to(source.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('Use a new valve-positioning output directory')
    output.mkdir(parents=True,exist_ok=True)
    try:
        room=Room(source)
        room.piping['valve_placement']='role_trace'
        before=pipeline_status(room.measured,room.piping,model_present=True)
        ods=room.piping['config']['pipe_outer_diameter_mm']
        coverage=straight_coverage(room.parts.values(),room.graph['valves'],ods)
        route_axes=[(r['id'],r.get('control_points',r.get('points'))) for r in room.routes]
        rows=room.solve()
        if coverage!=straight_coverage(room.parts.values(),room.graph['valves'],ods):
            raise ValueError('Straight-pipe and valve-port coverage changed after positioning')
        if route_axes!=[(r['id'],r.get('control_points',r.get('points'))) for r in room.routes]:
            raise ValueError('Valve positioning changed the saved pipe direction')
        final_audit,final_grip=room.audit()
        room.geometry = None
        folder=output/'model'
        path,auto=save_model(room,folder)
        station = verify_saved(room,path)
        added=Counter(room.current[r['id']]['phase'] for r in rows if r['status']=='placed' and not r['old'])
        # Saved utility valves predate the phase field carried by material valves.
        removed=Counter(v.get('phase','P1') for name,v in room.existing.items() if name not in room.current)
        room.measured['manual_valves']['installed']+=added['P1']-removed['P1']
        room.measured['material']['manual_valves']['installed']+=added['P2']-removed['P2']
        after=pipeline_status(room.measured,room.piping,model_present=True)
        for key in ('connections','penetrations','collision_breaks'):
            if before[key]!=after[key]:
                raise ValueError('Saved pipe comparison failed: '+key)
        report={'source':str(source),'before':before,'after':after,'considered':len(rows),
                'newly_installed':sum(added.values()),'fixed_automatic_geometries':auto,
                'on_disconnected_pipe':{'requested':sum(r['disconnected_pipe'] for r in rows),
                                        'installed':sum(r['disconnected_pipe'] and r['status']=='placed' for r in rows)},
                'removed_manual_valves':sum(removed.values()),
                'qualified':len(room.accepted),'pending':sum(r['status']=='pending' for r in rows),
                'retained_unqualified':[r['id'] for r in rows if r['old'] and r['status']=='pending' and r['id'] in room.current],
                'wheels':sum(room.current[key]['operator']=='wheel' for key in room.accepted),
                'high':sum(r.get('high',False) for r in rows),
                'hard_rule_hits_for_selected':{str(i):sum(bool(final_audit[r['id']][str(i)]) for r in rows
                                             if r['status']=='placed') for i in range(1,7)}|{'7':0},
                'hard_rule_hits_in_saved_model':{str(i):sum(bool(check[str(i)]) for check in final_audit.values())
                                                  for i in range(1,7)}|{'7':0},
                'final_audit':final_audit,'final_grip':final_grip,
                'grip_tight':sum(record['tight'] for record in final_grip.values()),
                'station_operators_in_saved_model':station,
                'same_straight_cut_parts':room.moves,'axis_and_span_preserved':True,'rows':rows,
                'collision_scope':'Screen rectangles before measuring each physical part. Sample valve motion in steps.',
                'pipe_scope':'Keep the original pipe axis and fixed adapters. Restore old valve openings, cut new ones and verify ports and valve positions from the saved model.',
                'head_lever_reference':'Project the operator center of mass over its full travel onto the sides of the valve axis'}
        for name,data in (('process.json',room.graph),('geometry.json',room.scene),('piping.json',room.piping),
                          ('measure.json',room.measured),('valve_positioning.json',report)):
            write_json(folder/name,data)
        (output/'layout').symlink_to(source.parent/'layout',target_is_directory=True)
        shutil.copy2(source.parent/'input.json',output/'input.json')
        return {key:report[key] for key in ('qualified','pending','wheels','high','after')}
    except Exception:
        write_json(output/'failure.json',{'traceback':traceback.format_exc()})
        raise
