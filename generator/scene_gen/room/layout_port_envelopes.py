"""Allow room for pipe connections and valves around equipment ports."""
from copy import deepcopy
from functools import lru_cache
from itertools import product

import numpy as np

from .catalogue_flange import adapter_envelope
from .catalogue_geometry import rotation, world_ports
from .equipment_details import REACTOR_PORTS, _model_key, _stagger_heater_inlet, catalogue_manifest, catalogue_pose
from .piping_assets import inspect_asset
from .pipe_spec import load
from .route_candidate_checks import FITTING_CLEARANCE_M
from .valve_roles import describe
from .valve_positioning import wheel_policy
from .vessel_bottom import applies

EPS = 1e-6


@lru_cache(maxsize=1)
def _catalogue():
    return catalogue_manifest()


def _catalogue_faces(unit, prototype, floor):
    catalogue = _catalogue()
    key = _model_key(unit, prototype)
    if key is None:
        return {}, None
    spec = deepcopy(catalogue[key])
    if key == 'exchanger_vertical':
        _stagger_heater_inlet(spec)
    if key in REACTOR_PORTS['yaw_offset_deg']:
        sources = spec['ports']
        remapped = {}
        for role in unit['ports']:
            if role not in REACTOR_PORTS:
                continue
            tag = REACTOR_PORTS[role][key]
            if tag in sources:
                remapped[role] = sources[tag]
            elif tag in REACTOR_PORTS['reopen_from'][key]:
                ref = REACTOR_PORTS['reopen_from'][key][tag]
                remapped[role] = {**sources[ref], 'position': spec['facts']['port_positions'][tag]['position']}
        spec['ports'] = remapped
    target, yaw, anchor = catalogue_pose(unit, key, spec, catalogue, floor)
    matrix = rotation(yaw)
    origin = np.asarray(target)-matrix@np.asarray(spec['anchor'] if anchor is None else anchor)
    bounds = None
    if bb := spec.get('bounds_m'):
        corners = np.array(list(product(*zip(bb['min'], bb['max']))))@matrix.T+origin
        bounds = np.r_[corners.min(axis=0), corners.max(axis=0)]
    return world_ports(spec, origin, matrix), bounds


def _cylinder_bounds(start, end, axis, radius):
    pad = radius*np.sqrt(np.maximum(0., 1.-axis**2))
    return np.r_[np.minimum(start, end)-pad, np.maximum(start, end)+pad]


def _departures(face, normal, radius, reach, flange_radius, length, *, flank=None):
    end = face + normal*reach
    adapter = _cylinder_bounds(face, end, normal, flange_radius)
    reserved = length+FITTING_CLEARANCE_M if length else radius
    straight = _cylinder_bounds(end, end+normal*reserved, normal, radius)
    options = [[adapter, straight]] if flank is None else []
    if normal[2] < -.99 and length:
        turn = end+normal*(3.*radius+.015)
        bend = _cylinder_bounds(end, turn, normal, radius)
        for k in (range(2) if flank is None else [flank[0]]):
            for sign in (-1, 1):
                direction = np.eye(3)[k]*sign
                start = turn+direction*(3.*radius+FITTING_CLEARANCE_M)
                if flank is not None:
                    start[k] = flank[1+(sign > 0)]+sign*FITTING_CLEARANCE_M
                tail = _cylinder_bounds(turn, start+direction*reserved, direction, radius)
                options.append([adapter, bend, tail])
    return options


def requirements(graph):
    dn = {}
    for edge in graph['edges']:
        for endpoint in (edge['a'], edge['b']):
            dn[endpoint] = max(edge['dn'], dn.get(endpoint, 0))
    by_port = {}
    for valve in graph['valves']:
        if valve.get('manual') and valve.get('at') in dn:
            by_port.setdefault(valve['at'], []).append(valve)
    return dn, by_port, load()['pipe_outer_diameter_mm']


def envelopes(graph, prototype, shell, required):
    unit = prototype['unit']
    dn_by_port, by_port, ods = required
    faces, body = _catalogue_faces(unit, prototype, shell['floors'][unit['level']])
    rows = []
    if body is not None:
        # Detail substitution can be wider than the coarse placement proxy.
        rows.append({'equipment': unit['id'], 'port': None, 'options': [[body]], 'horizontal_only': True})
    for device in (unit, *prototype.get('companions', ())):
        for name, port in device['ports'].items():
            endpoint = device['id']+'.'+name
            if endpoint not in dn_by_port:
                continue
            actual = faces.get(name) if device is unit else None
            point = actual['position'] if actual else port.get('pos')
            normal = actual['normal'] if actual else port.get('normal')
            if point is None or normal is None:
                continue
            dn = dn_by_port[endpoint]
            radius = ods[str(dn)]/2000.
            normal = np.asarray(normal, float)
            normal /= np.linalg.norm(normal)
            reach, flange = 2*max(radius, ods.get(str(port['dn']), 2*radius*1000)/2000.), radius
            if actual:
                try:
                    reach, flange = adapter_envelope(actual, radius)
                except ValueError:
                    # Layout reserves the departure space even if the subsequent
                    # piping stage must reject a particular bore combination.
                    flange = max(radius, actual.get('flange_outer_diameter_m', 0.)/2)
                    reach = max(reach, 2*flange)
            lengths = []
            for original in by_port.get(endpoint, ()):
                role = describe(graph, original)
                operator, _ = wheel_policy({**original, **role, 'dn': dn}, graph)
                lengths.append(inspect_asset(dn, operator)['install_length_m'])
            flank = None
            if applies(device, endpoint):
                k = 1 if device.get('rotation_deg', 0) % 180 == 0 else 0
                flank = (k, prototype['rect'][k], prototype['rect'][k+2])
            rows.append({'equipment': device['id'], 'port': name, 'dn': dn,
                         'stub_length_m': reach, 'valve_body_length_m': sum(lengths),
                         'options': _departures(np.asarray(point), normal, radius, reach, flange,
                                                sum(lengths), flank=flank)})
    return rows


def in_room(rows, xy, shell):
    width, depth = shell['footprint_m']
    shift = np.array([*xy, 0.] * 2)
    limit = np.array([0., 0., shell['floors'][0], width, depth, shell['roof_z_m']])
    approved = []
    for row in rows:
        axes = [0, 1] if row.get('horizontal_only') else [0, 1, 2]
        for boxes in row['options']:
            world = np.asarray(boxes)+shift
            if (np.all(world[:, axes] >= limit[axes]-EPS) and
                    np.all(world[:, [i+3 for i in axes]] <= limit[[i+3 for i in axes]]+EPS)):
                approved.append({k: v for k, v in row.items() if k != 'options'}|{'boxes': world.tolist()})
                break
        else:
            return None, f"{row['equipment']}.{row['port'] or 'equipment outline'}"
    return approved, None
