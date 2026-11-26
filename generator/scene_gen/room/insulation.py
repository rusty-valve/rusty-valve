"""Add insulation to pipes, valves and equipment."""
from collections import Counter, defaultdict
import gc
import json
import math
import os
import shutil
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import saved_geometry as geometry
from .export import nums
from .insulation_skin import EPS, HANDLE_GAP, THICKNESS, Tube, equipment_skin, lever_case, valve_case, valve_skin
from .pipeline_status import pipeline_status
from .process import write_json


MATERIAL = 'insulation_aluminium'
COLOR = (.73, .77, .81)
HEAT_SERVICES = {'steam', 'hot_water'}
CONTROL_NAMES = ('lever', 'handle', 'wheel', 'grip')


def read(path):
    return json.loads(path.read_text())


def route_scope(graph, piping):
    """Follow frozen service and device-edge identities, independent of room ID."""
    units = graph['units']
    heaters = {uid for uid, u in units.items() if u['kind'] == 'exchanger' and u['service'] == 'steam_heater'}
    condensers = {uid for uid, u in units.items() if u['kind'] == 'condenser'}
    jackets = {u['owner'] for u in units.values() if u['kind'] == 'jacket'}
    edges = {}
    for edge in graph['edges']:
        start, target = edge['a'].split('.')[0], edge['b'].split('.')[0]
        role = edge['role']
        if role in HEAT_SERVICES:
            edges[edge['id']] = 'utility heating'
        elif role == 'steam_condensate':
            edges[edge['id']] = 'condensate pipe'
        elif start in heaters and edge['a'].endswith('.out'):
            edges[edge['id']] = 'heater material outlet'
        elif target in condensers and edge['b'].endswith('.vapor') and role == 'vapor':
            edges[edge['id']] = 'reactor top to condenser material inlet'
    served_valves = {v for r in piping['routes'] if r['role'] == 'floor_branch'
                     and r['medium'] in HEAT_SERVICES and (r.get('target') or {}).get('header_id')
                     for v in r.get('valves', [])}
    hot_station = {c['id'] for c in piping.get('end_bundle_station', {}).get('connections', [])
                   if c['medium'] in HEAT_SERVICES and c.get('header_id')}
    routes = {}
    for r in piping['routes']:
        edge = r.get('source_edge') or r.get('source_requirement')
        if edge in edges:
            routes[r['id']] = edges[edge]
        elif r['medium'] in HEAT_SERVICES and (r['role'] in ('workstation_main', 'floor_header')
                                              or r.get('connection_id') in hot_station
                                              or served_valves.intersection(r.get('valves', []))):
            routes[r['id']] = 'utility heating'
    devices = {uid: scope for group, scope in ((heaters, 'heater body'), (condensers, 'condenser heads and straight sections'),
               (jackets, 'jacketed vessel shell and heads')) for uid in group}
    return edges, routes, devices


def control_sweeps(graph, model, data, *, actual_operators=False, bounds_only=False):
    """Sample the complete native operator travel using forward kinematics."""
    cache = {}
    sweeps = {}
    original = data.qpos.copy()
    for valve in graph['valves']:
        uid = valve['id']
        prefix = valve.get('prefix', uid+'_')
        control = [(prefix+g['name'], model.geom(prefix+g['name']).id) for g in valve['local_geoms']
                   if g['moving'] and any(word in g['name'] for word in CONTROL_NAMES)]
        if actual_operators:
            from .insulation_motion_check import operator_geometries
            control = operator_geometries(valve, model)
        if not control:
            continue
        pose = valve['motion']['joint_states']
        addressed = [(model.jnt_qposadr[model.joint(prefix+name).id], value, pose['closed'][name])
                     for name, value in pose['open'].items()]
        # Step at most five degrees for a lever and retain the actual swept shape.
        # Full-turn wheel geometry is sampled uniformly up to 73 steps.
        travel = max(abs(b-a) for _, a, b in addressed)
        frames = min(73, max(19, math.ceil(travel/math.radians(5))+1))
        poses = []
        vertices = [geometry.mesh(model, gid, cache)[0] for _, gid in control]
        for fraction in np.linspace(0., 1., frames):
            for addr, start, end in addressed:
                arc = end-start
                if actual_operators:
                    arc = math.copysign(min(abs(arc), math.tau), arc)
                data.qpos[addr] = start+arc*fraction
            mujoco.mj_kinematics(model, data)
            points = np.vstack([points@data.geom_xmat[gid].reshape(3, 3).T+data.geom_xpos[gid]
                                for (_, gid), points in zip(control, vertices)])
            poses.append(np.r_[points.min(axis=0), points.max(axis=0)] if bounds_only else points)
        data.qpos[:] = original
        points = np.vstack(poses)
        if bounds_only:
            sweeps[uid] = {'bounds': np.r_[points[:, :3].min(axis=0), points[:, 3:].max(axis=0)], 'frames': frames}
            continue
        if actual_operators:
            points = points.astype(np.float32)
        sweeps[uid] = {'points': points, 'bounds': np.r_[points.min(axis=0), points.max(axis=0)], 'frames': frames}
    mujoco.mj_kinematics(model, data)
    return sweeps


def nearby_sweeps(sweeps, bounds, reach=.20):
    for uid, sweep in sweeps.items():
        other = sweep['bounds']
        if np.all(other[:3] < bounds[3:]+reach) and np.all(other[3:] > bounds[:3]-reach):
            yield uid, sweep['points']


def pipe_tubes(components, piping, routes, scopes):
    """Yield the saved straight and curved pipes, with no skins on flange hardware."""
    seen = set()
    for route in routes.values():
        rid = route['id']
        for cid in route['components']:
            if cid in seen or cid not in components:
                continue
            c = components[cid]
            if c['kind'] == 'pipe' and c['shape'] == 'cylinder':
                tube = Tube.straight(c['ends'], c['radius'])
            elif c['kind'] == 'piping_reducer' and c.get('radii'):
                tube = Tube.straight(c['ends'], c['radii'])
            else:
                continue
            seen.add(cid)
            yield cid, c['kind'], scopes[rid], rid, tube
    for el in piping.get('elbows', []):
        rid = el['route_id']
        if rid in scopes:
            yield el['id'], 'elbow', scopes[rid], rid, Tube.elbow(el)


def flange_stops(components, piping, routes):
    """Find the pipe-side stop on the saved flange neck or on each bare pipe flange."""
    joints = {j['endpoint']: j for j in piping.get('catalogue_flange_joints', [])}
    for interface in piping.get('material', {}).get('interfaces', []):
        if joint := interface.get('equipment_flange_joint'):
            joints.setdefault(interface['endpoint'], {'position': interface['physical_position'],
                'normal': interface['normal'], **joint})
    for endpoint, joint in joints.items():
        connected = [rid for rid, route in routes.items() if joint.get('pipe_stub') in route['components']
                     or any((route.get(end) or {}).get('endpoint') == endpoint for end in ('source', 'target'))]
        if not connected:
            continue
        pos, axis = (np.asarray(joint[k]) for k in ('position', 'normal'))
        neck = (joint.get('transition') or {}).get('ends')
        stop = np.asarray(neck[-1] if neck else joint['end'])
        source = (joint.get('equipment_plate_components', [])+joint.get('equipment_rf_components', [])
                  or joint.get('source_components', []))
        pipe = [cid for cid in joint['components'] if cid not in source]
        yield {'id': endpoint, 'endpoint': endpoint, 'datum': pos.tolist(), 'axis': axis.tolist(),
               'bare_span_m': [0., float((stop-pos)@axis)], 'routes': connected,
               'equipment_geom_ids': source, 'pipe_flange_geom_ids': pipe}
    for flange in piping.get('flange_connections', []):
        identity = flange['id']
        connected = [rid for rid, route in routes.items() if identity == rid or identity.startswith(rid+'_')
                     or rid.startswith(identity+'_')
                     or any(identity.startswith(v+'_') for v in route.get('valves', []))]
        if not connected:
            continue
        parts = [components[k] for k in flange.get('components', []) if k in components]
        if not parts:
            continue
        pos = np.asarray(flange['position']); axis = np.asarray(flange['axis'])
        boxes = np.asarray([c['box'] for c in parts]).reshape(-1, 2, 3)
        low, high = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
        yield {'id': identity, 'datum': pos.tolist(), 'axis': axis.tolist(), 'routes': connected,
               'bare_span_m': sorted([float((low-pos)@axis), float((high-pos)@axis)]),
               'equipment_geom_ids': [], 'pipe_flange_geom_ids': [p['id'] for p in parts]}


def cut_at_flange(tube, stop):
    """Subtract the axial interval of a coaxial flange from a straight skin."""
    if len(tube.points) != 2:
        return [tube]
    axis, datum = np.asarray(stop['axis']), np.asarray(stop['datum'])
    delta = tube.points-datum
    if np.max(np.linalg.norm(np.cross(delta, axis), axis=1)) > 1.e-6:
        return [tube]
    a, b = delta@axis
    lower, upper = sorted([(stop['bare_span_m'][0]-a)/(b-a), (stop['bare_span_m'][1]-a)/(b-a)])
    if lower >= 1.-EPS or upper <= EPS:
        return [tube]
    step = tube.points[1]-tube.points[0]
    radius_step = tube.radii[1]-tube.radii[0]
    intervals = [(0., min(lower, 1.)), (max(0., upper), 1.)]
    return [Tube.straight(tube.points[0]+np.array([lo, hi])[:, None]*step,
                          tube.radii[0]+np.array([lo, hi])*radius_step)
            for lo, hi in intervals if hi-lo > EPS]


def stop_faces(tube, stops):
    if len(tube.points) != 2:
        return []
    faces = []
    for stop in stops:
        datum, axis = np.asarray(stop['datum']), np.asarray(stop['axis'])
        delta = tube.points-datum
        if np.max(np.linalg.norm(np.cross(delta, axis), axis=1)) > 1.e-6:
            continue
        low, high = sorted(delta@axis)
        a, b = stop['bare_span_m']
        if abs(high-a) < 1.e-6:
            faces.append({'flange': stop['id'], 'position': (datum+axis*a).tolist(), 'toward_pipe': (-axis).tolist()})
        if abs(low-b) < 1.e-6:
            faces.append({'flange': stop['id'], 'position': (datum+axis*b).tolist(), 'toward_pipe': axis.tolist()})
    return faces


def pipe_skins(components, piping, routes, scopes, flange_records):
    by_route = defaultdict(list)
    for flange in flange_records:
        for route in flange['routes']:
            by_route[route].append(flange)
    for name, kind, scope, route, tube in pipe_tubes(components, piping, routes, scopes):
        flanges = by_route[route]
        pieces = [tube]
        for flange in flanges:
            pieces = [piece for candidate in pieces for piece in cut_at_flange(candidate, flange)]
        for piece in pieces:
            yield name, kind, scope, route, piece, stop_faces(piece, flanges)


def equipment_skin_parts(uid, scene):
    selected = [c for c in scene['components'] if c.get('owner') == uid and not c.get('piping')]
    vessel = next((c for c in selected if c['kind'] == 'vessel' and c['shape'] == 'cylinder'), None)
    if vessel is not None:
        return [vessel, *(c for c in selected if c['kind'] == 'head' and c['shape'] == 'ellipsoid')]
    return [c for c in selected if c['id'].endswith(('_HXh_shell', '_left_channel', '_right_channel',
             '_left_head', '_right_head', '_process_shell', '_upper_head', '_lower_head'))]


def equipment_parts(uid, scene):
    selected = equipment_skin_parts(uid, scene)
    bare = next((c for c in selected if c['kind'] == 'vessel'), None)
    if bare is not None:
        a, b = map(np.asarray, bare['ends']); axis = (b-a)/np.linalg.norm(b-a)
        yield bare['id'], a, axis, bare['radius'], math.dist(a, b), False
        for c in selected[1:]:
            center = np.asarray(c['pos'])
            pole = axis if (center-(a+b)/2)@axis > 0 else -axis
            yield c['id'], center, pole, float(c['size'][0]), float(c['size'][2]), True
        return
    shell = next((c for c in selected if c['id'].endswith(('_HXh_shell', '_process_shell'))), None)
    if shell is None:
        return
    bounds = np.asarray(shell['box']).reshape(2, 3)
    axis_index = int(np.argmax(np.diff(bounds, axis=0)))
    axis = np.eye(3)[axis_index]
    midpoint = bounds.mean(axis=0)
    for c in selected:
        name = c['id']
        low, high = np.asarray(c['box']).reshape(2, 3)
        position = (low+high)/2
        extent = high-low
        radius = float(min(extent[abs(axis)<.5])/2)
        head = name.endswith('_head')
        direction = -axis if head and (position-midpoint)@axis < 0 else axis
        position[axis_index] = high[axis_index] if direction[axis_index]<0 else low[axis_index]
        yield name, position, direction, radius, float(extent[axis_index]), head


def apertures(uid, graph):
    for key, u in graph['units'].items():
        if key != uid and not (u['kind'] == 'jacket' and u.get('owner') == uid):
            continue
        for port in u.get('ports', {}).values():
            if not all(k in port for k in ('pos', 'normal')):
                continue
            catalogue = port.get('catalogue_face') or {}
            radius = catalogue.get('opening_diameter_m', 0.)/2
            radius = max(radius, port.get('radius', 0.))
            yield np.asarray(port['pos']), np.asarray(port['normal']), max(float(radius), .045)


def write_skin(folder, scene, specs, name, owner, scope, kind, vertices, faces, thickness, **meta):
    identifier = f'INS_{len(specs)+1:03d}'
    path = folder/'cladding'/(identifier+'.obj')
    vertices = np.asarray(vertices)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    origin = (lower+upper)/2
    with path.open('w') as out:
        for vertex in vertices-origin:
            out.write('v '+nums(vertex)+'\n')
        for face in faces:
            out.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
    component = {'id': identifier, 'owner': owner, 'kind': 'insulation', 'shape': 'mesh', 'color': list(COLOR),
                 'visual_only': True, 'material': MATERIAL, 'pos': origin.tolist(), 'box': np.r_[lower, upper].tolist(),
                 'mesh_path': str(path), 'insulated_piece': name, 'thickness_m': thickness}
    if meta.get('profile') == 'lever_straight_shoulders':
        component['flat_shading'] = True
    scene['components'].append(component)
    specs.append({'id': identifier, 'source': name, 'owner': owner, 'scope': scope,
                  'kind': kind, 'thickness_m': thickness, **meta})


def derive(source, folder):
    from .piping_method import _heights, load_frozen_layout
    from .equipment_details import prepare_full_room
    from .piping_collision_breaks import clear_stair_deck
    from .piping_method_measure import measure_room
    from .piping_material_templates import measure_material_templates

    graph, scene, piping = (read(source/name) for name in ('process.json', 'geometry.json', 'piping.json'))
    base = mujoco.MjModel.from_xml_path(str(source/'room.xml'))
    base_data = mujoco.MjData(base)
    mujoco.mj_forward(base, base_data)
    has_pruned_valves = 'close_valve_pruning' in piping
    sweeps = control_sweeps(graph, base, base_data, actual_operators=has_pruned_valves)
    expected_piece_keys = Counter()
    expected_valves = []
    edge_scopes, scopes, devices = route_scope(graph, piping)
    folder.mkdir(parents=True)
    (folder/'cladding').mkdir()
    records, omitted = [], []
    components = {c['id']: c for c in scene['components']}
    routes = {r['id']: r for r in piping['routes'] if r['id'] in scopes}
    flanges = list(flange_stops(components, piping, routes))
    clearances = defaultdict(list)
    for name, kind, scope, rid, tube, stop_rows in pipe_skins(components, piping, routes, scopes, flanges):
        expected_piece_keys[name,rid,kind] += 1
        close = list(nearby_sweeps(sweeps, tube.bounds(THICKNESS)))
        before = [(uid, tube.distance(points)) for uid, points in close]
        gap = min((d for _, d in before), default=math.inf)
        thickness = min(THICKNESS, max(0., gap-HANDLE_GAP))
        for (uid, prior), (_, points) in zip(before, close):
            after = tube.distance(points, thickness)
            clearances[uid].append({'source': name, 'before_m': prior, 'after_m': after,
                                    'thickness_m': thickness, 'pending': prior < HANDLE_GAP or after < HANDLE_GAP-EPS})
        if thickness <= EPS:
            omitted.append({'source': name, 'kind': kind, 'scope': scope, 'reason': 'Less than 10 mm of handle travel remains'})
            continue
        vertices, faces = tube.skin(thickness)
        write_skin(folder, scene, records, name, rid, scope, kind, vertices, faces, thickness,
                   measure_axis=tube.points[0].tolist(), bare_radius_m=float(tube.radii[0]),
                   ring_count=48, flange_stops=stop_rows)
    for uid, scope in devices.items():
        holes = list(apertures(uid, graph))
        for name, position, axis, radius, length, head in equipment_parts(uid, scene):
            expected_piece_keys[name,uid,'equipment envelope'] += 1
            vertices, faces = equipment_skin(position, axis, radius, length, THICKNESS, head=head, apertures=holes)
            write_skin(folder, scene, records, name, uid, scope, 'equipment envelope', vertices, faces, THICKNESS,
                       measure_axis=position.tolist(), bare_radius_m=radius, ring_count=48)
    wrapped = {v for route in piping['routes'] if route['id'] in scopes for v in route.get('valves', [])}
    valve_labels = []
    valve_mesh_cache = {}
    for valve in graph['valves']:
        uid = valve['id']
        if uid not in wrapped:
            continue
        expected_piece_keys[uid,uid,'valve body'] += 1
        expected_valves.append(uid)
        if has_pruned_valves:
            from .insulation_valves import build_valve_shell
            radius = piping['config']['pipe_outer_diameter_mm'][str(valve['dn'])]/2000
            built = build_valve_shell(valve,base,base_data,valve_mesh_cache,sweeps,flanges,radius)
            if built is None:
                omitted.append({'source':uid,'kind':'valve body','scope':'manual valve on an insulated pipe','reason':'No available clearance near the operator'})
                continue
            vertices, faces, thickness, profile, frame = built
            write_skin(folder,scene,records,uid,uid,'manual valve on an insulated pipe','valve body',vertices,faces,thickness,
                       profile=profile,valve_frame=frame,end_plates=2,opening_lining=True)
            valve_labels.append({'label':uid+' · '+{'threaded_lever_cylinder':'threaded lever valve cylinder',
                'flanged_lever_cylinder':'flanged lever valve cylinder spanning both flanges','handwheel_box':'handwheel valve box'}[profile],
                'position':np.mean(vertices,axis=0).tolist(),'valve':uid,'exposed':[],'cladding':records[-1]['id']})
            continue
        origin, matrix, bare_low, bare_high, shaft = valve_skin(valve)
        nominal, _ = valve_case(origin, matrix, bare_low-[0., THICKNESS, THICKNESS],
                                bare_high+[0., THICKNESS, 0.], shaft)
        bounds = np.r_[nominal.min(axis=0), nominal.max(axis=0)]
        probes = [(peer, (points-origin)@matrix) for peer, points in nearby_sweeps(sweeps, bounds)]

        def box_distance(points, low, high):
            gap = np.maximum(np.maximum(low-points, points-high), 0.)
            return float(np.linalg.norm(gap, axis=1).min())

        before = [(peer, box_distance(points, bare_low, bare_high)) for peer, points in probes]
        active = [points for (_, points), (_, gap) in zip(probes, before) if gap >= HANDLE_GAP]
        thickness = THICKNESS
        if any(box_distance(points, bare_low-[0., thickness, thickness], bare_high+[0., thickness, 0.])
               < HANDLE_GAP for points in active):
            small, large = 0., thickness
            for _ in range(18):
                trial = (small+large)/2
                nearest = min(box_distance(points, bare_low-[0., trial, trial], bare_high+[0., trial, 0.])
                              for points in active)
                if nearest >= HANDLE_GAP:
                    small = trial
                else:
                    large = trial
            thickness = small
        # The cap plane stays at the actual bonnet; only body sides and bottom extend.
        low, high = bare_low-[0., thickness, thickness], bare_high+[0., thickness, 0.]
        lever = valve['operator'] == 'lever'
        if lever:
            end_top = max(g['bbox'][5] for g in valve['local_geoms']
                          if g['name'].startswith(('visual_right_end', 'visual_left_end')))
            vertices, faces = lever_case(origin, matrix, low, high, shaft, thickness, end_top)
        else:
            vertices, faces = valve_case(origin, matrix, low, high, shaft)
        write_skin(folder, scene, records, uid, uid, 'manual valve on an insulated pipe', 'valve body', vertices, faces, thickness,
                   profile='lever_straight_shoulders' if lever else 'plain',
                   valve_frame={'origin': origin.tolist(), 'matrix': matrix.tolist(),
                               'bare_low_m': bare_low.tolist(), 'bare_high_m': bare_high.tolist(),
                               'stem_opening_half_width_m': shaft})
        exposed = [g['name'] for g in valve['local_geoms']
                   if g['moving'] and any(w in g['name'] for w in (*CONTROL_NAMES, 'stem'))]
        cap = origin+matrix[:, 2]*high[2]
        label = ' · straight shell covers the body and leaves the stem and lever exposed' if lever else ' · exposed above the bonnet'
        valve_labels.append({'label': uid+label, 'position': cap.tolist(), 'valve': uid,
                             'exposed': exposed, 'cladding': records[-1]['id']})
        for (peer, points), (_, prior) in zip(probes, before):
            after = box_distance(points, low, high)
            clearances[peer].append({'source': uid+'insulation shell', 'before_m': prior, 'after_m': after,
                                     'thickness_m': thickness, 'pending': after < HANDLE_GAP-EPS})
    baseline_geometry = geometry_snapshot(base, base_data)
    del base, base_data, valve_mesh_cache
    sweeps = {uid: {'bounds': row['bounds'], 'frames': row['frames']} for uid, row in sweeps.items()}
    gc.collect()
    xml = append_model(source, folder, scene)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    measure_skins(model, data, baseline_geometry, records)
    if has_pruned_valves:
        from .insulation_motion_check import audit
        conflicts, sampled_poses = audit(graph,model,data,records,sweeps)
    else:
        conflicts, sampled_poses = [],{}

    checked_stops = defaultdict(list)
    for rec in records:
        for stop in rec.get('flange_stops', []):
            checked_stops[stop['flange']].append({'cladding': rec['id'], **stop})
    boundaries = []
    for flange in flanges:
        hardware = flange['pipe_flange_geom_ids']+flange['equipment_geom_ids']
        for identifier in hardware:
            model.geom(identifier)
        stops = checked_stops[flange['id']]
        if stops and 'endpoint' in flange:
            stop = stops[0]
            boundaries.append({'label': flange['endpoint']+' · both flanges stay bare and insulation ends on the pipe side',
                               'position': stop['position'], 'endpoint': flange['endpoint'],
                               'cladding': stop['cladding']})
        flange['checked_stops'] = stops
    before = read(source/'measure.json')
    if has_pruned_valves:
        # measure_skins compared all original native geoms after reload.
        # The added geometries are explicitly visual-only in the saved XML.
        measured = before
    else:
        frozen = load_frozen_layout(source.parent/'layout')
        prepare_full_room(frozen, folder/'measurement_reference')
        clear_stair_deck(frozen['scene'])
        measured = measure_room(frozen, graph, scene, piping, model, data)
        measured['material'] = measure_material_templates(frozen, graph, scene, piping, model, data, p1_report=measured)
        for key in ('valve_roles', 'confirmed_rules'):
            if key in before:
                measured[key] = before[key]
    summary = {'source': str(source), 'material': MATERIAL, 'revision': 'straight_lever_bare_flanges',
               'source_edges': edge_scopes, 'routes': scopes,
               'devices': devices, 'pieces': records, 'omitted': omitted, 'equipment_boundaries': boundaries,
               'flanges': flanges,
               'valve_labels': valve_labels, 'handle_clearance': [
                   {'valve': uid, 'sampled_poses': sweep['frames'], 'pending': any(r['pending'] for r in clearances[uid]),
                    'closest_after_m': min((r['after_m'] for r in clearances[uid]), default=None),
                    'rows': clearances[uid]} for uid, sweep in sweeps.items()],
               'before_insulation': pipeline_status(before, piping, model_present=True),
               'stage4': pipeline_status(measured, piping, model_present=True),
               'clearance_scope': 'Sample all native valve travel. Use saved straight and elbow centerlines and envelopes. Insulation does not enter the original routing and valve solver.'}
    if has_pruned_valves:
        actual_piece_keys = Counter((record['source'],record['owner'],record['kind']) for record in records)
        summary.update(revision='confirmed_valve_insulation_profiles', collisions=conflicts, operation_samples=sampled_poses,
            original_geometry_verified=len(baseline_geometry),
            pipe_measurement='Keep the installed-valve measurements. Recompile the saved XML and compare each original shape type, size, position and orientation.',
            scope_check={'required_device_ids':sorted(devices),'required_valve_ids':sorted(expected_valves),
                         'missing_devices':sorted(set(devices)-{r['owner'] for r in records if r['kind']=='equipment envelope'}),
                         'expected_pieces':sum(expected_piece_keys.values()),
                         'missing':list((expected_piece_keys-actual_piece_keys).elements()),
                         'extra':list((actual_piece_keys-expected_piece_keys).elements())},
            reduced_thickness=[{'id':rec['id'],'source':rec['source'],'owner':rec['owner'],
                                'measured_thickness_m':rec['measured_thickness_m']}
                               for rec in records if rec['measured_thickness_m']<THICKNESS-.00005])
        for key in ('connections','penetrations','collision_breaks','valves','pending_valves'):
            if summary['before_insulation'][key] != summary['stage4'][key]:
                raise ValueError('Original measured counts differ after insulation: '+key+' '+str(summary['stage4'][key]))
        for filename in ('valve_positioning.json','close_valve_pruning.json'):
            shutil.copy2(source/filename,folder/filename)
    for name, value in (('geometry.json', scene), ('process.json', graph), ('piping.json', piping),
                        ('measure.json', measured), ('insulation.json', summary)):
        write_json(folder/name, value)
    return summary


def append_model(source, folder, scene):
    root = ET.parse(source/'room.xml').getroot()
    for item in root.iter():
        if item.get('file'):
            item.set('file', os.path.relpath((source/item.attrib['file']).resolve(), folder))
    assets, world = root.find('asset'), root.find('worldbody')
    ET.SubElement(assets, 'material', name=MATERIAL, rgba=nums([*COLOR, 1.]), specular='.75', shininess='.65')
    for c in scene['components']:
        if c.get('material') != MATERIAL:
            continue
        name = c['id']
        ET.SubElement(assets, 'mesh', name=name, file=os.path.relpath(c['mesh_path'], folder))
        ET.SubElement(world, 'geom', name=name, type='mesh', mesh=name, pos=nums(c['pos']), material=MATERIAL,
                      contype='0', conaffinity='0', mass='0')
    path = folder/'room.xml'
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return path


def geometry_snapshot(model, data):
    return {model.geom(gid).name: (int(model.geom_type[gid]), model.geom_size[gid].copy(),
            data.geom_xpos[gid].copy(), data.geom_xmat[gid].copy()) for gid in range(model.ngeom)}


def measure_skins(model, data, baseline_geometry, records):
    cache = {}
    for name, (kind, size, position, matrix) in baseline_geometry.items():
        gid = model.geom(name).id
        if kind != model.geom_type[gid] or not all(np.allclose(x, y, atol=1.e-8, rtol=0.) for x, y in (
                (size, model.geom_size[gid]), (position, data.geom_xpos[gid]),
                (matrix, data.geom_xmat[gid]))):
            raise ValueError('Original physical geometry changed after insulation: '+name)
    for rec in records:
        gid = model.geom(rec['id']).id
        mat = model.geom_matid[gid]
        if model.material(mat).name != MATERIAL or model.geom_contype[gid] or model.geom_conaffinity[gid]:
            raise ValueError('Insulation material or collision properties were saved incorrectly: '+rec['id'])
        vertices, _ = geometry.mesh(model, gid, cache)
        actual = vertices@data.geom_xmat[gid].reshape(3, 3).T+data.geom_xpos[gid]
        rec['actual_box_m'] = np.r_[actual.min(axis=0), actual.max(axis=0)].tolist()
        if 'measure_axis' in rec:
            radial = np.linalg.norm(actual[:rec['ring_count']]-rec['measure_axis'], axis=1)
            rec['measured_thickness_m'] = float(np.median(radial)-rec['bare_radius_m'])
            for stop in rec.get('flange_stops', []):
                distance = (actual-np.asarray(stop['position']))@np.asarray(stop['toward_pipe'])
                nearest = float(distance.min())
                if abs(nearest) > .00005:
                    raise ValueError('Insulation fails to terminate at the bare flange edge: '+stop['flange'])
                stop['measured_stop_offset_m'] = nearest
        elif rec.get('profile') in ('threaded_lever_cylinder','flanged_lever_cylinder','handwheel_box'):
            frame = rec['valve_frame']
            local = (actual-frame['origin'])@np.asarray(frame['matrix'])
            lower, upper = local.min(axis=0),local.max(axis=0)
            if not np.allclose([lower[0],upper[0]],frame['ends_m'],atol=.00005,rtol=0.):
                raise ValueError('Insulation closure plate does not align with the valve end: '+rec['id'])
            if rec['profile']=='handwheel_box':
                y0,y1=frame['bare_side_y_m']
                offsets=(y0-lower[1],upper[1]-y1,frame['bare_bottom_z_m']-lower[2])
                if not np.allclose(offsets,rec['thickness_m'],atol=.00005,rtol=0):
                    raise ValueError('Could not read back handwheel valve box thickness: '+rec['id'])
                rec['measured_thickness_m']=float(np.mean(offsets))
            else:
                outer=np.linalg.norm(local[:,1:],axis=1).max()
                rec['measured_thickness_m']=float(outer-frame['bare_radius_m'])
        else:
            frame = rec['valve_frame']
            local = (actual-frame['origin'])@np.asarray(frame['matrix'])
            lower, upper = local.min(axis=0), local.max(axis=0)
            lower_delta = np.asarray(frame['bare_low_m'])-lower
            upper_delta = upper-frame['bare_high_m']
            extension = [lower_delta[1], lower_delta[2], upper_delta[1]]
            if not np.allclose(extension, rec['thickness_m'], atol=.00005, rtol=0.) or abs(upper_delta[2]) > .00005:
                raise ValueError('Valve body shell thickness or bonnet boundary differs on readback: '+rec['id'])
            rec['measured_thickness_m'] = float(np.mean(extension))
            if rec.get('profile') == 'lever_straight_shoulders':
                ends = np.isclose(local[:, 0], lower[0], atol=.00005) | np.isclose(local[:, 0], upper[0], atol=.00005)
                end_width = float(np.ptp(local[ends, 1]))
                middle_width = float(upper[1]-lower[1])
                opening = local[np.isclose(local[:, 2], upper[2], atol=.00005)]
                hole = float(frame['stem_opening_half_width_m'])
                hole_corners = sum(any(np.linalg.norm(p[:2]-[a,b]) < .00005 for p in opening)
                                   for a in (-hole, hole) for b in (-hole, hole))
                if not end_width < middle_width or hole_corners != 4:
                    raise ValueError('Straight shoulder or square stem opening differs on readback: '+rec['id'])
                rec.update(measured_cuff_width_m=end_width, measured_body_width_m=middle_width,
                           measured_stem_opening_corners=hole_corners)
        if not math.isclose(rec['measured_thickness_m'], rec['thickness_m'], abs_tol=.00005):
            raise ValueError(f"{rec['id']} Insulation thickness differs on readback: {rec['measured_thickness_m']:.6f} m")


def generate(source, output, *, threaded_lever_sample=None):
    geometry.geometry_dependencies()
    source, output = source.resolve(), output.resolve()
    if not (source/'room.xml').is_file() or (threaded_lever_sample is None and not (source.parent/'layout').is_dir()):
        raise ValueError('Insulation requires a saved complete room model and its saved layout.')
    if output.is_relative_to(source.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('Choose a new empty insulation output directory.')
    output.mkdir(parents=True, exist_ok=True)
    try:
        if threaded_lever_sample is not None:
            from .insulation_threaded_sample import derive_sample
            return derive_sample(source, output/'model', threaded_lever_sample)
        report = derive(source, output/'model')
        if report['revision']=='confirmed_valve_insulation_profiles':
            (output/'layout').symlink_to((source.parent/'layout').resolve(),target_is_directory=True)
            shutil.copy2(source.parent/'input.json',output/'input.json')
        return report
    except Exception as exc:
        write_json(output/'failure.json', {'type': type(exc).__name__, 'reason': str(exc),
                                         'traceback': traceback.format_exc()})
        raise
