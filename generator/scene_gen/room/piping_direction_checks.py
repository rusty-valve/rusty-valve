"""Measure pipe directions and verify the shape and placement of elbows."""
from __future__ import annotations

from collections import defaultdict
import math

import mujoco
import numpy as np

from .building_equipment_checks import EPS


def _error(code, component=None, route=None, **detail):
    return {'code': code, 'component': component, 'route': route, **detail}


def _near(first, second):
    a, b = np.asarray(first), np.asarray(second)
    return a.shape == b.shape and bool(np.all(np.abs(a-b) <= EPS))


def _axis(ends):
    delta = np.asarray(ends[1])-ends[0]
    changing = np.flatnonzero(np.abs(delta) > EPS).tolist()
    return {'delta_m': delta.tolist(), 'changing_axes': changing,
            'orthogonal': len(changing) == 1, 'length_m': float(np.linalg.norm(delta))}


def _records(value):
    if isinstance(value, dict):
        return [dict(row, id=row.get('id', key)) for key, row in value.items()]
    return value if isinstance(value, list) else []


def _circle_on_plane(points):
    """Fit a planar circular polygon without trusting a declared port plane."""
    if len(points) < 4:
        return None
    origin = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points-origin, full_matrices=False)
    if len(singular) < 3 or singular[1] == 0.:
        return None
    normal = axes[-1]; plane_error = float(np.max(np.abs((points-origin)@normal)))
    if plane_error > EPS:
        return None
    uv = (points-origin)@axes[:2].T
    solution, *_ = np.linalg.lstsq(np.column_stack((2*uv, np.ones(len(uv)))),
                                  np.sum(uv*uv, axis=1), rcond=None)
    center = origin+solution[:2]@axes[:2]
    distances = np.linalg.norm(points-center, axis=1)
    radius = float(distances.mean()); error = float(np.max(np.abs(distances-radius)))
    if radius <= EPS or error > EPS:
        return None
    return {'position': center.tolist(), 'normal': normal.tolist(), 'radius': radius,
            'plane_error_m': plane_error, 'radial_error_m': error}


def _piece_rings(row, model, cache):
    from . import piping_checks as common
    if row['type'] != int(mujoco.mjtGeom.mjGEOM_MESH):
        return None
    vertices, _ = common._mesh_support_data(row, model, cache)
    mesh = int(model.geom_dataid[row['index']]); start = int(model.mesh_faceadr[mesh])
    count = int(model.mesh_facenum[mesh]); faces = np.asarray(model.mesh_face[start:start+count], dtype=int)
    candidates = {}; middle = vertices.mean(axis=0)
    for face in faces:
        a, b, c = vertices[face]; normal = np.cross(b-a, c-a)
        norm = float(np.linalg.norm(normal))
        if norm == 0.:
            continue
        normal /= norm; distances = (vertices-a)@normal
        if distances.min() < -EPS and distances.max() > EPS:
            continue  # It is not an actual supporting end face.
        indices = frozenset(np.flatnonzero(np.abs(distances) <= EPS).tolist())
        if indices in candidates or len(indices) < 4:
            continue
        points = vertices[sorted(indices)]; circle = _circle_on_plane(points)
        if circle is None:
            continue
        outward = np.asarray(circle['normal'])
        if float((np.asarray(circle['position'])-middle)@outward) < 0:
            outward = -outward
        circle.update(normal=outward.tolist(), vertex_indices=sorted(indices),
                      vertices=points.tolist(), vertex_count=len(points))
        candidates[indices] = circle
    all_indices = set(range(len(vertices)))
    for first, one in candidates.items():
        for second, two in candidates.items():
            if first & second or first | second != all_indices:
                continue
            if abs(one['radius']-two['radius']) > EPS:
                continue
            return [one, two]
    return None


def elbow_records_from_components(components):
    """Locate producer-declared groups; measure_elbows verifies their real shape."""
    grouped = defaultdict(list)
    for component in components:
        if component.get('elbow_id') is not None:
            grouped[component['elbow_id']].append(component)
    result = []
    for uid, pieces in grouped.items():
        pieces.sort(key=lambda p: p.get('elbow_piece_index', 0))
        result.append({'id': uid, 'route_id': pieces[0].get('route_id'),
            'components': [p['id'] for p in pieces],
            'ports': {'in': pieces[0].get('ports', {}).get('in', {}),
                      'out': pieces[-1].get('ports', {}).get('out', {})}})
    return result


def measure_elbows(scene, piping, model, data, actual=None):
    """Return actual end rings/arc geometry and flow segments for true bends."""
    from . import piping_checks as common
    actual = actual if actual is not None else common._measured(model, data)
    records = _records(piping.get('elbows', scene.get('elbows', [])))
    rows, errors, by_component = [], [], {}; cache = {}; used = set()
    for record in records:
        uid = record.get('id'); names = record.get('components', [])
        row = {'id': uid, 'route': record.get('route_id'), 'components': names, 'valid': False}
        reason = None; pieces = []
        if not names or len(names) != len(set(names)) or used & set(names):
            reason = 'Physical elbow parts are missing, duplicated or reused by multiple elbows'
        for name in names:
            measured = actual.get(name)
            rings = _piece_rings(measured, model, cache) if measured else None
            if rings is None:
                reason = 'Mesh does not form two circular rims of equal diameter'; break
            pieces.append({'id': name, 'rings': rings})
        if reason is None:
            nodes = []; edges = []; references = defaultdict(list)
            for piece in pieces:
                indices = []
                for ring in piece['rings']:
                    index = next((i for i, existing in enumerate(nodes)
                                  if _near(existing['position'], ring['position'])), None)
                    if index is None:
                        index = len(nodes); nodes.append(ring)
                    else:
                        existing = nodes[index]
                        if abs(existing['radius']-ring['radius']) > EPS:
                            reason = 'Adjacent physical end rims have different diameters'; break
                        a = np.asarray(existing['vertices']); b = np.asarray(ring['vertices'])
                        if max(float(np.min(np.linalg.norm(b-p, axis=1))) for p in a) > EPS:
                            reason = 'Adjacent circular end faces cannot mate'; break
                    indices.append(index); references[index].append(piece['id'])
                if reason:
                    break
                edges.append((indices[0], indices[1], piece['id']))
            terminals = [i for i in range(len(nodes)) if len(references[i]) == 1]
            if reason is None and (len(terminals) != 2 or any(len(r) > 2 for r in references.values())):
                reason = 'Elbow sectors do not form one continuous face chain'
        if reason is None:
            declared_in = record.get('ports', {}).get('in', {}).get('position')
            begin = next((i for i in terminals if declared_in is not None and _near(nodes[i]['position'], declared_in)), None)
            if begin is None:
                reason = 'Declared inlet differs from the exposed face'
        if reason is None:
            current = begin; ordered = [begin]; ordered_pieces = []; previous = set()
            while True:
                options = [edge for edge in edges if current in edge[:2] and edge[2] not in previous]
                if not options:
                    break
                edge = options[0]; previous.add(edge[2]); ordered_pieces.append(edge[2])
                current = edge[1] if edge[0] == current else edge[0]; ordered.append(current)
            if len(previous) != len(edges):
                reason = 'The elbow includes disconnected or branched parts'
        if reason is None:
            start = np.asarray(nodes[ordered[0]]['position']); end = np.asarray(nodes[ordered[-1]]['position'])
            tangent_in = -np.asarray(nodes[ordered[0]]['normal'])
            tangent_out = np.asarray(nodes[ordered[-1]]['normal'])
            normal = np.cross(tangent_in, tangent_out); sine = float(np.linalg.norm(normal))
            dot = float(np.clip(tangent_in@tangent_out, -1., 1.)); angle = math.acos(dot)
            radius = float(np.mean([nodes[i]['radius'] for i in ordered]))
            if sine == 0. or angle == 0.:
                reason = 'The two end tangents do not form a physical turn'
            else:
                normal /= sine
                bend_radius = float(np.linalg.norm(end-start)/(2*math.sin(angle/2)))
                center = start+np.cross(normal, tangent_in)*bend_radius
                factor = piping.get('config', {}).get('pg_rules', {}).get('orthogonal_routing', {}).get(
                    'coarse_bends', {}).get('centreline_radius_outer_diameter_factor', 1.5)
                if abs(bend_radius-factor*2*radius) > EPS:
                    reason = 'Elbow center radius differs from the specified 1.5 pipe OD'
                for index in ordered:
                    point = np.asarray(nodes[index]['position']); radial = point-center
                    tangent = np.cross(normal, radial/bend_radius)
                    vertices = np.asarray(nodes[index]['vertices'])
                    if (abs(np.linalg.norm(radial)-bend_radius) > EPS
                            or abs(float(radial@normal)) > EPS
                            or np.max(np.abs((vertices-point)@tangent)) > EPS):
                        reason = 'End-ring centers, tangents and envelope do not lie on one arc'; break
                ports = record.get('ports', {})
                if not _near(ports.get('out', {}).get('position', []), end):
                    reason = 'Declared outlet differs from the exposed face'
                if not abs(dot) <= EPS:
                    reason = 'Only physical 90-degree elbows are allowed'
        if reason is not None:
            errors.append(_error('PG17_ELBOW_GEOMETRY', names[0] if names else uid, record.get('route_id'),
                                 elbow=uid, reason=reason)); row['reason'] = reason
        else:
            segments = []
            for index, name in enumerate(ordered_pieces):
                segment = {'id': name, 'ends': [nodes[ordered[index]]['position'], nodes[ordered[index+1]]['position']],
                           'radius': radius, 'path_kind': 'actual_elbow_piece', 'elbow': uid}
                segments.append(segment); by_component[name] = segment
            row.update(valid=True, ports={'in': {'position': start.tolist(), 'tangent': tangent_in.tolist()},
                       'out': {'position': end.tolist(), 'tangent': tangent_out.tolist()}},
                       center=center.tolist(), plane_normal=normal.tolist(), pipe_outer_radius_m=radius,
                       centerline_radius_m=bend_radius, angle_degrees=math.degrees(angle),
                       actual_end_rings=[{'position': nodes[i]['position'], 'radius': nodes[i]['radius'],
                           'plane_error_m': nodes[i]['plane_error_m'], 'radial_error_m': nodes[i]['radial_error_m']}
                           for i in ordered], segments=segments)
            used.update(names)
        rows.append(row)
    return {'status': 'fail' if errors else 'pass', 'rows': rows, 'errors': errors,
            'by_component': by_component, 'verified_component_ids': sorted(used)}


def _frozen_parts(frozen):
    contracts = frozen.get('contracts', {})
    graph = contracts.get('sizing', {}).get('graph', {})
    equipment = contracts.get('layout', {}).get('equipment', [])
    return graph, {unit['id']: unit for unit in equipment}


def _source_point(frozen, requirement, actual):
    """A genuine frozen ordinary-offgas source, never an output medium label."""
    from . import piping_checks as common
    graph, equipment = _frozen_parts(frozen)
    edge = next((e for e in graph.get('edges', []) if e['id'] == requirement), None)
    if edge is None or edge.get('role') != 'offgas':
        return None
    candidates = []
    for endpoint in (edge['a'], edge['b']):
        uid, name = endpoint.split('.', 1)
        if uid not in equipment or name not in equipment[uid].get('ports', {}):
            continue
        peer = edge['b'] if endpoint == edge['a'] else edge['a']
        header_unit = graph.get('units', {}).get(peer.split('.')[0], {})
        if header_unit.get('kind') != 'header' or header_unit.get('service') != 'offgas':
            continue
        port = equipment[uid]['ports'][name]
        nozzle_name = uid+'_'+name+'_equipment_nozzle'
        measured = actual.get(nozzle_name); cylinder = common._cylinder(measured) if measured else None
        if not cylinder or not any(_near(p, port['pos']) for p in cylinder['ends']):
            continue
        face_index = next(i for i, p in enumerate(cylinder['ends']) if _near(p, port['pos']))
        neck = np.asarray(cylinder['ends'][face_index])-cylinder['ends'][1-face_index]
        normal = np.asarray(port['normal'], dtype=float); normal_length = float(np.linalg.norm(normal))
        if normal_length == 0.:
            continue
        normal /= normal_length
        if float(neck@normal) <= 0. or np.linalg.norm(neck-float(neck@normal)*normal) > EPS:
            continue
        from .piping_requirements import port_level
        level = port_level(port, equipment[uid])
        candidates.append({'endpoint': endpoint, 'position': port['pos'], 'normal': port['normal'],
                           'level': level, 'source_edge': edge['id'], 'actual_nozzle': nozzle_name})
    return candidates[0] if len(candidates) == 1 else None


def _offgas_entries(frozen, routes, actual, by_route, valve_segments, straights):
    """Measure the frozen offgas source, top-down entry and actual onward path."""
    from . import piping_checks as common
    graph, _ = _frozen_parts(frozen)
    valves = {segment['valve']: segment for segment in valve_segments}
    rows, errors = [], []
    for edge in graph.get('edges', []):
        if edge.get('role') != 'offgas':
            continue
        source = _source_point(frozen, edge['id'], actual)
        branches = [r for r in routes.values() if r.get('source_requirement') == edge['id']]
        branch_parts = [segment for r in branches for segment in by_route.get(r['id'], [])]
        branch_parts.extend(valves[v] for r in branches for v in r.get('valves', []) if v in valves)
        joins = []
        for branch in branches:
            header = routes.get(branch.get('distribution_id'), {})
            main = routes.get(branch.get('main_id'), {})
            header_parts = by_route.get(header.get('id'), [])
            main_parts = by_route.get(main.get('id'), [])
            horizontal_header = bool(header_parts) and all(
                _axis(s['ends'])['changing_axes'] in ([0], [1]) for s in header_parts)
            vertical_main = bool(main_parts) and all(
                _axis(s['ends'])['changing_axes'] == [2] for s in main_parts)
            floor_identity = bool(source and header.get('level') == source['level']
                                  and branch.get('level') == source['level'])
            floors = frozen.get('scene', {}).get('floors', [])
            level = source['level'] if source else None
            floor_height = None
            if isinstance(level, int) and 0 <= level < len(floors):
                upper = floors[level+1] if level+1 < len(floors) else frozen.get('shell', {}).get('roof_z_m')
                floor_height = upper is not None and all(
                    floors[level]-EPS <= p[2] <= upper+EPS for s in header_parts for p in s['ends'])
            genuine_distribution = bool(header.get('role') == 'floor_header' and header.get('medium') == 'offgas'
                and main.get('role') == 'workstation_main' and main.get('medium') == 'offgas'
                and header.get('main_id') == main.get('id') and horizontal_header and vertical_main
                and floor_identity and floor_height)
            for segment in by_route.get(branch['id'], []):
                if segment['id'] not in straights:
                    continue  # The last approach must be a real straight cylinder, not an arc chord.
                for index, q in enumerate(segment['ends']):
                    touched = [s for s in header_parts if common._on_segment(q, *s['ends'])]
                    if not touched:
                        continue
                    p = segment['ends'][1-index]; delta = np.asarray(q)-p
                    downward = bool(abs(delta[0]) <= EPS and abs(delta[1]) <= EPS and delta[2] < -EPS)
                    above = bool(p[2]-segment['radius'] > q[2]+max(s['radius'] for s in touched))
                    source_connected = bool(source and common._connected(source['position'], [q], branch_parts))
                    main_targets = [end for s in main_parts for end in s['ends']]
                    onward_connected = common._connected(q, main_targets, header_parts+main_parts)
                    passed = bool(genuine_distribution and downward and above and source_connected and onward_connected)
                    joins.append({'component': segment['id'], 'route': branch['id'],
                        'header': header.get('id'), 'main': main.get('id'),
                        'actual_approach': p, 'actual_join': q, 'actual_delta_m': delta.tolist(),
                        'actual_top_down': downward, 'above_actual_header': above,
                        'frozen_source_connected': source_connected, 'header_to_main_connected': onward_connected,
                        'same_frozen_floor': floor_identity, 'actual_header_in_floor_band': floor_height,
                        'actual_horizontal_header': horizontal_header, 'actual_vertical_main': vertical_main,
                        'pass': passed})
        row = {'kind': 'offgas_top_vertical_down', 'requirement': edge['id'], 'source': source,
               'actual_joins': joins, 'pass': bool(joins) and all(j['pass'] for j in joins)}
        rows.append(row)
        if not row['pass']:
            component = joins[0]['component'] if joins else source['actual_nozzle'] if source else None
            errors.append(_error('PG13_OFFGAS_TOP_VERTICAL_DOWN', component,
                joins[0]['route'] if joins else branches[0]['id'] if branches else None,
                requirement=edge['id'], detail=row,
                reason="Saved off-gas source must join its level's horizontal collector with a final downward Z segment. The collector must connect to the vertical main."))
    return {'status': 'fail' if errors else 'pass', 'rows': rows, 'errors': errors}




def _through_segments(point, segments):
    from . import piping_checks as common
    directions = []
    for segment in segments:
        if not common._on_segment(point, *segment['ends']):
            continue
        for end in segment['ends']:
            vector = np.asarray(end)-point; length = float(np.linalg.norm(vector))
            if length > EPS:
                directions.append(vector/length)
    return any(np.linalg.norm(a+b) <= EPS for i, a in enumerate(directions) for b in directions[i+1:])


def _through_point(point, route_id, routes, straights):
    return _through_segments(point, [straights[name] for name in routes[route_id].get('components', [])
                                     if name in straights])


def _safety_discharge_tee(point, branch, main, routes, frozen, piping, straights, measured):
    """An original safety discharge, proven by loaded solids and three rays."""
    if measured is None or branch.get('role')!='automatic_safety_connection':return False
    from . import piping_checks as common
    from .pipe_spec import load
    graph,equipment=_frozen_parts(frozen);spec=load();service=spec['services']['offgas']
    edge=next((e for e in graph.get('edges',[]) if e['id']==branch.get('source_edge')),None)
    if not edge or edge.get('role') not in ('emergency_discharge','safety_discharge'):return False
    uid,port=edge['a'].split('.',1);target=edge['b'].split('.')[0]
    unit=graph['units'].get(uid,{});header=graph['units'].get(target,{})
    if (unit.get('kind')!='safety_valve' or port!='out' or header.get('kind')!='header'
            or header.get('service')!='offgas' or branch.get('a')!=edge['a'] or branch.get('b')!=edge['b']
            or branch.get('medium_identity')!=edge.get('medium_identity',edge['role'])
            or edge.get('dn')!=service['emergency_dn'] or branch.get('dn')!=service['emergency_dn']
            or unit.get('ports',{}).get('out',{}).get('dn')!=service['emergency_dn']
            or main.get('role')!='floor_header' or main.get('medium')!='offgas'
            or main.get('dn')!=service['header_dn']):return False
    assigned=piping.get('material',{}).get('requirements',{}).get('ports',{}).get(edge['b'],{})
    if assigned.get('header_id')!=main['id'] or branch.get('distribution_id',main['id'])!=main['id']:return False
    source=unit.get('source_endpoint','').split('.')
    if len(source)!=2 or source[0]!=unit.get('owner'):return False
    original=equipment.get(source[0],{}).get('ports',{}).get(source[1])
    floors=frozen.get('scene',{}).get('floors',[])
    if not original or not floors:return False
    level=max((i for i,z in enumerate(floors) if z<=original['pos'][2]+EPS),default=-1)
    if level<0 or main.get('level')!=level or branch.get('level')!=level:return False
    upper=floors[level+1] if level+1<len(floors) else frozen['shell']['roof_z_m']
    if not floors[level]-EPS<=point[2]<upper-EPS:return False
    records=[r for r in measured['scene'].get('automatic_components',[]) if r.get('id')==uid]
    if len(records)!=1:return False
    record=records[0];body=measured['actual'].get(uid+'_automatic_body')
    nozzle=routes.get('AUTO_'+uid+'_out');out=record.get('ports',{}).get('out',{}).get('pos')
    radius=spec['pipe_outer_diameter_mm'][str(service['emergency_dn'])]/2000
    header_radius=spec['pipe_outer_diameter_mm'][str(service['header_dn'])]/2000
    if out is None:return False
    if body:
        if (body['type']!=int(mujoco.mjtGeom.mjGEOM_BOX)
                or not _near(body['box'],record.get('body_bbox_m',[]))
                or not nozzle or nozzle.get('role')!='automatic_nozzle' or len(nozzle['components'])!=1):return False
        cid=nozzle['components'][0]
        if cid not in record.get('components',[]) or cid not in measured['actual']:return False
        shape=common._cylinder(measured['actual'][cid])
        if not shape or abs(shape['radius']-radius)>EPS:return False
        matches=[i for i,p in enumerate(shape['ends']) if _near(p,out)]
        if len(matches)!=1:return False
        face=np.asarray(shape['ends'][1-matches[0]]);delta=np.asarray(out)-face
        axes=np.flatnonzero(abs(delta)>EPS)
        if len(axes)!=1:return False
        axis=int(axes[0]);sign=1 if delta[axis]>0 else -1;box=body['box']
        if abs(face[axis]-box[axis+(3 if sign>0 else 0)])>EPS:return False
        if any(face[k]-radius<box[k]-EPS or face[k]+radius>box[k+3]+EPS for k in range(3) if k!=axis):return False
    else:
        jointly_loaded=next((j for j in measured['scene'].get('catalogue_flange_joints',[])
                             if j['endpoint']==edge['a'] and _near(j['position'],out)),None)
        if not jointly_loaded:return False
        source_parts=jointly_loaded['source_components']
        if not source_parts or not set(source_parts)<=set(record['components']):return False
        parts=[*source_parts,*jointly_loaded['components'],jointly_loaded['pipe_stub']]
        if not all(name in measured['actual'] for name in parts):return False
        stub=common._cylinder(measured['actual'][jointly_loaded['pipe_stub']])
        if (not stub or abs(stub['radius']-radius)>EPS or
                not any(_near(end,jointly_loaded['end']) for end in stub['ends'])):return False
    if not floors[level]-EPS<=out[2]<upper-EPS:return False
    if not common._connected(out,[point],measured['by_route'].get(branch['id'],[])):return False
    names=set(main.get('components',[]));touched=[s for s in straights.values() if common._on_segment(point,*s['ends'])]
    headers=[s for s in touched if s['id'] in names];incoming=[s for s in touched if s['id'] not in names]
    if (not headers or len(incoming)!=1 or incoming[0]['id'] not in branch['components']
            or abs(incoming[0]['radius']-radius)>EPS
            or any(abs(s['radius']-header_radius)>EPS for s in headers)
            or not _through_point(point,main['id'],routes,straights)):return False
    header_axes={_axis(s['ends'])['changing_axes'][0] for s in headers if _axis(s['ends'])['orthogonal']}
    if len(header_axes)!=1 or not header_axes<={0,1} or _axis(incoming[0]['ends'])['changing_axes']!=[2]:return False
    if not any(_near(point,p) for p in incoming[0]['ends']):return False
    rays=set()
    for segment in touched:
        for end in segment['ends']:
            delta=np.asarray(end)-point;axes=np.flatnonzero(abs(delta)>EPS)
            if len(axes)==1:rays.add((int(axes[0]),1 if delta[axes[0]]>0 else -1))
    haxis=next(iter(header_axes))
    return rays=={(haxis,-1),(haxis,1),(2,1)}


def _legitimate_tee(point, first, second, routes_for, routes, frozen, piping, straights, measured=None):
    """A branch to an actual distributor is not a two-piece direction change."""
    from . import piping_checks as common
    graph, _ = _frozen_parts(frozen)
    for ra in routes_for.get(first, []):
        for rb in routes_for.get(second, []):
            a, b = routes[ra], routes[rb]
            for branch, main in ((a, b), (b, a)):
                if _safety_discharge_tee(point,branch,main,routes,frozen,piping,straights,measured):return True
                if (branch.get('distribution_id') == main['id'] and main.get('role') == 'floor_header'
                        and _through_point(point, main['id'], routes, straights)):
                    edge = next((e for e in graph.get('edges', []) if e['id'] == branch.get('source_requirement')), None)
                    if (edge and edge.get('role') == branch.get('medium') == main.get('medium')
                            and branch.get('level') == main.get('level')):
                        return True
                    if (branch.get('role') == 'floor_branch' and branch.get('medium') == main.get('medium')
                            and branch.get('level') == main.get('level')
                            and any(e.get('role') == branch.get('medium') for e in graph.get('edges', []))):
                        # v1.5 alone adds a floor-link tee on a real header body.
                        # Count all actual incident pipes; two branches are not a tee.
                        header_names = set(main.get('components', []))
                        touched = [s for s in straights.values() if common._on_segment(point, *s['ends'])]
                        incoming = [s for s in touched if s['id'] not in header_names]
                        header_axes = {_axis(s['ends'])['changing_axes'][0] for s in touched
                                       if s['id'] in header_names and _axis(s['ends'])['orthogonal']}
                        rays = set()
                        for segment in touched:
                            for end in segment['ends']:
                                delta = np.asarray(end)-point
                                axes = np.flatnonzero(np.abs(delta) > EPS)
                                if len(axes) == 1:
                                    axis = int(axes[0]); rays.add((axis, 1 if delta[axis] > 0. else -1))
                        if (len(header_axes) == 1 and header_axes <= {0, 1} and len(incoming) == 1 and len(rays) == 3
                                and _axis(incoming[0]['ends'])['orthogonal']
                                and _axis(incoming[0]['ends'])['changing_axes'][0] not in header_axes
                                and any(_near(point, end) for end in incoming[0]['ends'])):
                            return True
                if (branch.get('role') in ('floor_branch', 'temporary_outlet') and main.get('role') == 'workstation_main'
                        and branch.get('main_id') == main['id'] and branch.get('medium') == main.get('medium')
                        and any(e.get('role') == branch.get('medium') for e in graph.get('edges', []))
                        and _through_point(point, main['id'], routes, straights)):
                    return True
                if (branch.get('role') == 'floor_header' and main.get('role') == 'workstation_main'
                        and branch.get('main_id') == main['id'] and branch.get('medium') == main.get('medium')
                        and any(e.get('role') == branch.get('medium') for e in graph.get('edges', []))
                        and _through_point(point, main['id'], routes, straights)):
                    header_parts = [straights[name] for name in branch.get('components', []) if name in straights]
                    main_parts = [straights[name] for name in main.get('components', []) if name in straights]
                    floors = frozen.get('scene', {}).get('floors', []); level = branch.get('level')
                    if isinstance(level, int) and 0 <= level < len(floors):
                        upper = floors[level+1] if level+1 < len(floors) else frozen.get('shell', {}).get('roof_z_m')
                        if (upper is not None and header_parts and main_parts
                                and all(_axis(s['ends'])['changing_axes'] in ([0], [1]) for s in header_parts)
                                and all(_axis(s['ends'])['changing_axes'] == [2] for s in main_parts)
                                and all(floors[level]-EPS <= p[2] <= upper+EPS for s in header_parts for p in s['ends'])):
                            return True
    for junction in _records(piping.get('material', {}).get('junctions', [])):
        if junction.get('position') is not None and _near(point, junction['position']):
            original = graph.get('units', {}).get(junction['id'], {})
            if original.get('kind') == 'junction' and _through_segments(point, straights.values()):
                return True
    return False


def _turn_checks(straights, routes_for, routes, frozen, piping, measured=None):
    from . import piping_checks as common
    errors = []; bends = defaultdict(list)
    parts = list(straights.values())
    for index, first in enumerate(parts):
        a, b = map(np.asarray, first['ends']); u = (b-a)/np.linalg.norm(b-a)
        for second in parts[index+1:]:
            c, d = map(np.asarray, second['ends']); v = (d-c)/np.linalg.norm(d-c)
            if (np.linalg.norm(c-a-float((c-a)@u)*u) <= EPS
                    and np.linalg.norm(d-a-float((d-a)@u)*u) <= EPS):
                continue
            points = []
            for point in common._touch_points([first], [second]):
                if not any(_near(point, previous) for previous in points):
                    points.append(point)
            for point in points:
                if _legitimate_tee(point, first['id'], second['id'], routes_for, routes, frozen, piping, straights,measured):
                    continue
                first_routes, second_routes = routes_for.get(first['id'], []), routes_for.get(second['id'], [])
                common_routes = sorted(set(first_routes) & set(second_routes))
                route = common_routes[0] if common_routes else None
                errors.append(_error('PG17_TURN_WITHOUT_ELBOW', first['id'], route,
                    other_component=second['id'], actual_point=point,
                    first_routes=first_routes, second_routes=second_routes,
                    reason='Two straight pipes turn at an intersection with no physical elbow face or envelope'))
                for segment in (first, second):
                    end = next((i for i, p in enumerate(segment['ends']) if _near(p, point)), None)
                    if end is not None:
                        bends[segment['id']].append((end, 3*segment['radius'], route))
    for name, requirements in bends.items():
        needs = {}
        for endpoint, trim, route in requirements:
            needs[endpoint] = max(trim, needs.get(endpoint, 0.))
        length = math.dist(*straights[name]['ends']); required = sum(needs.values())
        if length+EPS < required:
            errors.append(_error('PG17_DENSE_TURNS', name, requirements[0][2],
                actual_straight_length_m=length, required_existing_bend_trim_m=required,
                reason='The short straights cannot fit the 1.5-OD physical elbow'))
    return errors


def _native_installation_axes(native_rows, actual, graph):
    """Use an actual fixed package geom to recover only the external flow axis."""
    from .piping_assets import inspect_asset
    rows, errors = [], []
    definitions = {valve['id']: valve for valve in graph.get('valves', [])}
    for native in native_rows:
        if native['status'] != 'pass':
            continue  # The original full-package errors already prevent acceptance.
        asset = inspect_asset(native['dn'], definitions[native['id']].get('operator'))
        index = next(i for i, part in enumerate(asset['local_geoms']) if not part['moving'])
        local = asset['local_geoms'][index]; name = native['geom_ids'][index]; measured = actual[name]
        matrix = np.asarray(measured['matrix'])@np.asarray(local['matrix']).T
        origin = np.asarray(measured['position'])-matrix@np.asarray(local['center'])
        ports = {key: (origin+matrix@np.asarray(asset['local_ports'][key]['position'])).tolist()
                 for key in ('in', 'out')}
        ends = [ports['in'], ports['out']]; direction = _axis(ends)
        row = {'kind': 'native_valve_installation_axis', 'valve': native['id'], 'component': name,
               'package': native['package'], 'actual_in_out_ports': ports, 'actual_ends': ends,
               'verified_native_geom_ids': native['geom_ids'], 'actual_pose_reference_geom': name,
               'pose_basis': 'Check fixed native valve pose and XML-verified inlet and outlet. Internal chamfers and lever direction are excluded.',
               **direction, 'pass': direction['orthogonal']}
        rows.append(row)
        if not row['pass']:
            errors.append(_error('PG17_NATIVE_AXIS_NON_ORTHOGONAL', name, None, valve=native['id'],
                actual_ends=ends, actual_delta_m=direction['delta_m'], changing_axes=direction['changing_axes'],
                reason='The installed valve flow axis does not follow the saved building axes'))
    return {'status': 'fail' if errors else 'pass', 'total': len(native_rows), 'checked': len(rows),
            'nonorthogonal_violations': len(errors), 'rows': rows, 'errors': errors}


def _verified_supports(frozen, components, actual):
    """Only real axis-aligned boxes face-connected to frozen frame members."""
    original = {c['id'] for c in frozen.get('scene', {}).get('components', [])
                if c.get('kind') in ('column', 'beam', 'equipment_beam') and c['id'] in actual}
    proven = set(original); proofs = {}; pending = set()
    for name, component in components.items():
        row = actual.get(name)
        if (name not in original and row and row['type'] == int(mujoco.mjtGeom.mjGEOM_BOX)
                and component.get('kind') in ('beam', 'hanger', 'equipment_beam')
                and _near(row['box'], component.get('box', []))
                and np.all(np.abs(np.abs(np.asarray(row['matrix'])).sum(axis=0)-1.) <= EPS)):
            pending.add(name)
    for _ in range(len(pending)+1):
        added = []
        for name in sorted(pending):
            a = actual[name]['box']
            for anchor in sorted(proven):
                b = actual[anchor]['box']; overlap = np.minimum(a[3:], b[3:])-np.maximum(a[:3], b[:3])
                touching = [axis for axis in range(3) if abs(a[axis+3]-b[axis]) <= EPS or abs(b[axis+3]-a[axis]) <= EPS]
                if any(all(overlap[other] > EPS for other in range(3) if other != axis) for axis in touching):
                    proofs[name] = {'actual_box': a, 'actual_anchor': anchor, 'anchor_box': b,
                                    'basis': 'actual box face contact path to unchanged frozen column/beam'}
                    added.append(name); break
        if not added:
            break
        pending.difference_update(added); proven.update(added)
    return proofs


def _remaining_plates(frozen, components, actual):
    originals = [c for c in frozen.get('scene', {}).get('components', [])
                 if c.get('kind') in ('floor', 'walk_surface') and c.get('shape') == 'box']
    proofs = {}
    for name, c in components.items():
        row = actual.get(name)
        if (c.get('kind') not in ('floor', 'walk_surface') or row is None
                or row['type'] != int(mujoco.mjtGeom.mjGEOM_BOX) or not _near(row['box'], c.get('box', []))
                or not np.all(np.abs(np.abs(np.asarray(row['matrix'])).sum(axis=0)-1.) <= EPS)):
            continue
        a = row['box']
        parent = next((p for p in originals if abs(a[2]-p['box'][2]) <= EPS and abs(a[5]-p['box'][5]) <= EPS
            and all(a[i] >= p['box'][i]-EPS and a[i+3] <= p['box'][i+3]+EPS for i in (0, 1))), None)
        if parent:
            proofs[name] = {'actual_box': a, 'original_plate': parent['id'], 'original_box': parent['box'],
                            'basis': 'actual same-height box subset of unchanged frozen plate'}
    return proofs


def _flange_axis(record, components, actual, model, cache, od):
    """Recognise the existing two hollow-flange meshes from real vertices."""
    from . import piping_checks as common
    names = record.get('components', [])
    if not names or any(name not in actual or actual[name]['type'] != int(mujoco.mjtGeom.mjGEOM_MESH) for name in names):
        return None
    vertices = np.concatenate([common._mesh_support_data(actual[name], model, cache)[0] for name in names])
    center = vertices.mean(axis=0); _, _, basis = np.linalg.svd(vertices-center, full_matrices=False)
    axis = basis[-1]; axial = (vertices-center)@axis
    radial = np.linalg.norm(vertices-center-np.outer(axial, axis), axis=1)
    diameter = od.get(str(record.get('dn')))
    if diameter is None:
        return None
    r = diameter/2000.; inner = r/math.cos(math.pi/16); outer = r+.034
    if (np.max(np.minimum(np.abs(radial-inner), np.abs(radial-outer))) > EPS
            or abs(float(axial.max()-axial.min())-.039) > EPS):
        return None
    levels = np.array([0., .018, .021, .039]); z = axial-axial.min()
    if np.max(np.min(np.abs(z[:, None]-levels[None, :]), axis=1)) > EPS:
        return None
    return {'ends': [(center+axis*axial.min()).tolist(), (center+axis*axial.max()).tolist()],
            'radius': outer, 'components': names, 'actual_shape': 'two_existing_hollow_flange_rings'}


def check_directions(frozen, graph, scene, piping, model, data, *, flange_joints=None):
    """v1.4 actual XYZ straight pipes, real 90-degree bends and offgas entries."""
    from . import piping_checks as common
    actual = common._measured(model, data); components = {c['id']: c for c in scene.get('components', [])}
    routes = {r['id']: r for r in piping.get('routes', [])}; routes_for = defaultdict(list)
    for route in routes.values():
        for name in route.get('components', []):
            routes_for[name].append(route['id'])
    original = {c['id'] for c in frozen.get('scene', {}).get('components', [])}
    native_rows, native_errors, valve_segments, _ = common._valves(graph, actual)
    native = {name for row in native_rows if row['status'] == 'pass' for name in row['geom_ids']}
    native_axes = _native_installation_axes(native_rows, actual, graph)
    elbows = measure_elbows(scene, piping, model, data, actual)
    errors = list(elbows['errors']); rows = []
    errors.extend(native_errors)
    errors.extend(native_axes['errors']); rows.extend(native_axes['rows'])
    by_route, route_errors, _ = common._route_segments(list(routes.values()), components, actual,
        piping['config']['pipe_outer_diameter_mm'], model, data, elbow_measurements=elbows)
    errors.extend(route_errors)
    new_ids = set(actual)-original-native
    support_proofs = _verified_supports(frozen, components, actual)
    plate_proofs = _remaining_plates(frozen, components, actual)
    from .piping_equipment_flange_checks import measure_joints
    if flange_joints is None:
        flange_joints = measure_joints(frozen, piping, model, actual)
    errors.extend(flange_joints['errors'])
    rows.extend(flange_joints['rows'])
    automatic_proofs = {}
    for device in scene.get('catalogue_devices',[]):
        registered=device['components']+device['envelopes']
        if all(name in actual and name in components and
               np.allclose(actual[name]['box'],components[name]['box'],atol=EPS,rtol=0)
               for name in registered if components.get(name,{}).get('shape')=='box') and all(
                   name in actual and components.get(name,{}).get('owner')==device['id'] for name in registered):
            for name in registered:
                automatic_proofs[name]={'catalogue_device':device['id'],'catalogue_key':device['catalogue_key']}
    for component in scene.get('automatic_components', []):
        if component.get('kind') not in ('safety_valve','automatic_steam_trap'):continue
        body_id=component['id']+'_automatic_body'
        measured=actual.get(body_id)
        expected_box=component.get('body_bbox_m')
        nozzles=[name for name in component.get('components',[]) if name!=body_id]
        # Recognition uses the loaded box and registered actual nozzle shapes;
        # names/kind alone never exempt a shape from checking.
        if (measured and measured['type']==int(mujoco.mjtGeom.mjGEOM_BOX)
                and expected_box is not None and np.max(abs(np.asarray(measured['box'])-expected_box))<=EPS
                and len(nozzles)==2 and all(common._cylinder(actual[n]) is not None
                    and routes_for.get(n) for n in nozzles if n in actual)
                and all(n in actual for n in nozzles)):
            automatic_proofs[body_id]={'actual_box':measured['box'],'nozzles':nozzles,
                'scope':'actual coarse automatic body and two registered nozzles; no control-performance proof'}
    known_meshes = set(elbows['verified_component_ids']); mesh_axes = {}; cache = {}
    for record in _records(scene.get('flange_connections', piping.get('flange_connections', []))):
        shape = _flange_axis(record, components, actual, model, cache, piping['config']['pipe_outer_diameter_mm'])
        if shape is not None:
            known_meshes.update(shape['components'])
            mesh_axes[record['id']] = shape
    for name in new_ids:
        component = components.get(name)
        if component and component.get('kind') == 'piping_reducer' and actual[name]['type'] == int(mujoco.mjtGeom.mjGEOM_MESH):
            reducer = common._mesh_reducer(component, actual[name], model, data)
            if reducer and reducer['declared_axis_matches']:
                known_meshes.add(name); mesh_axes[name] = reducer
    straights = {}; checked_ids = set(); unregistered = set(); straight_total = 0; nonorthogonal = set()
    for name in sorted(new_ids, key=str):
        row = actual[name]; component = components.get(name); route_ids = routes_for.get(name, [])
        route_id = route_ids[0] if route_ids else None
        if component is None:
            errors.append(_error('PG17_UNREGISTERED_GEOMETRY', name, route_id,
                                 reason='New geometry is unregistered. A collision mask or name does not exempt it.'))
            unregistered.add(name)
        if name in flange_joints['hardware'] and component is not None:
            checked_ids.add(name)
            continue
        cylinder = common._cylinder(row)
        if cylinder is not None:
            cap = False
            if component and component.get('kind') == 'piping_blind':
                dn = piping['config']['pipe_outer_diameter_mm'].get(str(component.get('dn')))
                cap = (dn is not None and abs(cylinder['radius']-(dn/2000.+.034)) <= EPS
                       and abs(math.dist(*cylinder['ends'])-.018) <= EPS)
            if not cap:
                straight_total += 1; straights[name] = {'id': name, **cylinder}
                if not route_ids and name not in flange_joints['stubs']:
                    errors.append(_error('PG17_UNREGISTERED_GEOMETRY', name, None,
                                         reason='An actual straight was omitted from the route and is still checked'))
                    unregistered.add(name)
                if component and (component.get('kind') in ('elbow', 'piping_elbow') or component.get('elbow_id') is not None):
                    errors.append(_error('PG17_ELBOW_GEOMETRY', name, route_id,
                                         reason='The object is physically a straight cylinder'))
            direction = _axis(cylinder['ends'])
            if not direction['orthogonal']:
                errors.append(_error('PG17_NON_ORTHOGONAL', name, route_id,
                    actual_ends=cylinder['ends'], actual_delta_m=direction['delta_m'],
                    reason='All straight runs follow the saved building axes'))
                if not cap:
                    nonorthogonal.add(name)
            for rid in route_ids:
                points = routes[rid].get('points', [])
                if not points or not all(any(common._on_segment(point, a, b) for a, b in zip(points, points[1:]))
                                         for point in cylinder['ends']):
                    errors.append(_error('PG17_ROUTE_MODEL_MISMATCH', name, rid,
                        actual_ends=cylinder['ends'], reason='Route points differ from the parent-transformed physical straight'))
            rows.append({'component': name, 'route': route_id, 'kind': 'blind_axis' if cap else 'actual_straight',
                         'actual_ends': cylinder['ends'], **direction, 'pass': direction['orthogonal']})
            checked_ids.add(name)
        elif name in automatic_proofs:
            rows.append({'component':name,'kind':'verified_automatic_body','pass':True,
                         'automatic_proof':automatic_proofs[name]});checked_ids.add(name)
        elif row['type'] == int(mujoco.mjtGeom.mjGEOM_MESH):
            if name not in known_meshes:
                code = 'PG17_ELBOW_GEOMETRY' if component and component.get('kind') in ('elbow', 'piping_elbow') else 'PG17_UNREGISTERED_GEOMETRY'
                errors.append(_error(code, name, route_id, reason='The mesh is not a validated elbow, coaxial reducer or hollow flange'))
                unregistered.add(name)
            checked_ids.add(name)
        elif name in support_proofs or name in plate_proofs:
            rows.append({'component': name, 'kind': 'verified_actual_support', 'pass': True,
                         'support_proof': support_proofs.get(name, plate_proofs.get(name))}); checked_ids.add(name)
        else:
            errors.append(_error('PG17_UNREGISTERED_GEOMETRY', name, route_id,
                                 reason='New geometry needs verified support or fitting evidence'))
            unregistered.add(name); checked_ids.add(name)
    for uid, shape in mesh_axes.items():
        direction = _axis(shape['ends'])
        if not direction['orthogonal']:
            errors.append(_error('PG17_NON_ORTHOGONAL', shape.get('components', [uid])[0], None,
                actual_ends=shape['ends'], reason='The measured reducer or flange axis is not aligned with a building axis'))
        rows.append({'component': uid, 'kind': 'actual_fitting_axis', **direction})
    for elbow in elbows['rows']:
        if not elbow['valid']:
            continue
        for endpoint in elbow['ports'].values():
            start = np.asarray(endpoint['position']); end = start+np.asarray(endpoint['tangent'])*elbow['centerline_radius_m']
            if not _axis([start, end])['orthogonal']:
                errors.append(_error('PG17_NON_ORTHOGONAL', elbow['components'][0], elbow['route'],
                                     reason='End tangents of the physical 90-degree elbow do not follow the building axes'))
    errors.extend(_turn_checks(straights, routes_for, routes, frozen, piping,
        {'actual':actual,'by_route':by_route,'scene':scene}))
    offgas = _offgas_entries(frozen, routes, actual, by_route, valve_segments, straights)
    rows.extend(offgas['rows']); errors.extend(offgas['errors'])
    import json
    seen = set(); unique = []
    for error in errors:
        key = json.dumps(error, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            unique.append(error); seen.add(key)
    errors = unique
    direction_components = {error['component'] for error in errors if error['code'] == 'PG17_NON_ORTHOGONAL'}
    native_axis_components = {error['component'] for error in errors if error['code'] == 'PG17_NATIVE_AXIS_NON_ORTHOGONAL'}
    return {'status': 'fail' if errors else 'pass', 'errors': errors, 'violations': errors,
        'actual_straight_total': straight_total, 'checked': len(straights),
        'nonorthogonal_violations': len(direction_components | native_axis_components),
        'straight_nonorthogonal_violations': len(nonorthogonal),
        'native_valve_axis_violations': native_axes['nonorthogonal_violations'],
        'fitting_axis_violations': len(direction_components-set(straights)),
        'nonorthogonal_counting_scope': 'Deduplicate actual nonorthogonal objects by ID. List pipes, native valve axes and fitting end directions separately.',
        'native_installation_axes': native_axes,
        'rows': rows, 'elbows': elbows, 'offgas_entries': offgas,
        'rule_version': 'v1.4', 'nonorthogonal_exemptions': False,
        'coverage': {'actual_new_geom_ids': sorted(new_ids, key=str), 'checked_geom_ids': sorted(checked_ids, key=str),
                     'unregistered_geom_ids': sorted(unregistered, key=str),
                     'verified_equipment_flange_geom_ids': sorted(flange_joints['hardware']),
                     'verified_equipment_flange_stub_ids': sorted(flange_joints['stubs']),
                     'verified_support_geom_ids': sorted(support_proofs),
                     'verified_remaining_plate_ids': sorted(plate_proofs)},
        'numerical_equality_eps_m': EPS, 'building_frame': 'immutable frozen world XYZ; output frames ignored',
        'scope': 'New pipes use building axes and 90-degree elbows. Off-gas enters its floor collector vertically downward.'}
