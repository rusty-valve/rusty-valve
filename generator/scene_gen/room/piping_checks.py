"""Check physical pipe connections, spacing and intersections."""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import math

import mujoco
import numpy as np

from .building_equipment_checks import EPS
from .storage_vessel import storage_service_checks
from .checks import geom_bounds
from .piping_assets import flow_ports, inspect_asset, AssetUnavailable
from .piping_requirements import requirements_from_frozen, detect_port_conflicts
from .solid_collision_checks import PENETRATION_LIMIT_M


def _error(code, **detail):
    return {'code': code, **detail}


def _section(rows, errors, **detail):
    return {'status': 'fail' if errors else 'pass', 'rows': rows,
            'errors': errors, 'violations': errors, **detail}


def _near(a, b):
    return np.asarray(a).shape == np.asarray(b).shape and bool(np.allclose(a, b, atol=EPS, rtol=0.))


def _on_segment(point, a, b):
    point, a, b = map(np.asarray, (point, a, b))
    vector = b-a; length = float(np.linalg.norm(vector))
    if length <= EPS:
        return math.dist(point, a) <= EPS
    axis = vector/length
    along = float((point-a)@axis)
    # EPS is metres, never a normalized fraction of this particular pipe.
    closest = a+min(length, max(0., along))*axis
    return -EPS <= along <= length+EPS and math.dist(point, closest) <= EPS


def _measured(model, data):
    boxes = geom_bounds(model, data, list(range(model.ngeom)))
    rows = {}
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if name is None:
            name = f'__unnamed_geom_{index}'
        rows[name] = {'id': name, 'index': index, 'type': int(model.geom_type[index]),
            'box': boxes[index].tolist(), 'position': data.geom_xpos[index].tolist(),
            'matrix': data.geom_xmat[index].reshape(3, 3).tolist(),
            'size': model.geom_size[index].tolist()}
    return rows


def _cylinder(row):
    if row['type'] != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return None
    center = np.asarray(row['position']); axis = np.asarray(row['matrix'])[:, 2]
    half = row['size'][1]
    return {'ends': [(center-axis*half).tolist(), (center+axis*half).tolist()],
            'radius': row['size'][0]}


def _mesh_support_data(row, model, cache):
    if row['id'] in cache:
        return cache[row['id']]
    mesh = int(model.geom_dataid[row['index']]); start = int(model.mesh_vertadr[mesh])
    count = int(model.mesh_vertnum[mesh])
    vertices = np.asarray(model.mesh_vert[start:start+count], dtype=float)
    vertices = vertices@np.asarray(row['matrix']).T+row['position']
    face_start = int(model.mesh_faceadr[mesh]); face_count = int(model.mesh_facenum[mesh])
    faces = np.asarray(model.mesh_face[face_start:face_start+face_count], dtype=int)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    # Reject only exactly degenerate triangle normals. These axes provide
    # candidate directions; actual support intervals always prove the result.
    norms = np.linalg.norm(normals, axis=1)
    normals = normals[norms > 0.]/norms[norms > 0., None]
    cache[row['id']] = (vertices, normals)
    return vertices, normals


def _support_interval(row, axis, model, cache):
    centre = float(np.asarray(row['position'])@axis)
    local = np.asarray(row['matrix']).T@axis; size = np.asarray(row['size'])
    kind = row['type']
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = float(np.abs(local)@size)
    elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half = abs(float(local[2]))*size[1]+size[0]*math.sqrt(max(0., 1.-float(local[2])**2))
    elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half = abs(float(local[2]))*size[1]+size[0]
    elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        half = size[0]
    elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        half = float(np.linalg.norm(local*size))
    elif kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        vertices, _ = _mesh_support_data(row, model, cache)
        projection = vertices@axis
        return [float(projection.min()), float(projection.max())]
    else:
        return None
    return [float(centre-half), float(centre+half)]


def _separation_proof(first, second, model, cache):
    """A real supporting plane, not acceptance of a zero backend distance.

    Axis intervals are measured in metres using all actual mesh vertices and
    analytic primitive support. Any successful axis proves separation or at
    most EPS numerical contact, even when the backend reports deep overlap.
    Lack of a plane proves nothing: normal deep-overlap fallback continues.
    """
    candidates = [*np.eye(3)]
    for row in (first, second):
        if row['type'] == int(mujoco.mjtGeom.mjGEOM_MESH):
            _, normals = _mesh_support_data(row, model, cache)
            candidates.extend(normals)
        elif row['type'] in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                             int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
            candidates.extend(np.asarray(row['matrix']).T)
    # Cross axes also cover ordinary oriented box/cylinder contact directions.
    a_axes, b_axes = np.asarray(first['matrix']).T, np.asarray(second['matrix']).T
    candidates.extend(np.cross(a, b) for a in a_axes for b in b_axes)
    for direction in candidates:
        norm = float(np.linalg.norm(direction))
        if norm == 0.:
            continue
        axis = np.asarray(direction)/norm
        a = _support_interval(first, axis, model, cache)
        b = _support_interval(second, axis, model, cache)
        if a is None or b is None:
            continue
        gap = max(b[0]-a[1], a[0]-b[1])
        if gap >= -EPS:
            return {'axis': axis.tolist(), 'first_support_interval_m': a, 'second_support_interval_m': b,
                    'gap_m': float(gap), 'numerical_contact_eps_m': EPS,
                    'basis': 'actual mesh vertices/face-normal axes + analytic primitive support; unit axis, metre intervals'}
    return None


def _registered_face_proof(first, second, record, segments, model, cache):
    """Prove that both closed solids terminate on opposite sides of the recorded face."""
    if set(record.get('pair', ())) != {first['id'], second['id']}:
        return None
    point = np.asarray(record.get('point'), dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        return None
    directions = []
    if 'normal' in record:
        directions.append(np.asarray(record['normal'], dtype=float))
    else:
        for row in (first, second):
            segment = _cylinder(row) or segments.get(row['id'])
            if not segment:
                continue
            for end, other in ((0, 1), (1, 0)):
                if _near(segment['ends'][end], point):
                    directions.append(np.asarray(segment['ends'][other])-point)
    for direction in directions:
        if direction.shape != (3,) or not np.isfinite(direction).all():
            continue
        length = np.linalg.norm(direction)
        if length <= EPS:
            continue
        axis = direction/length
        plane = float(point@axis)
        a = _support_interval(first, axis, model, cache)
        b = _support_interval(second, axis, model, cache)
        if a is None or b is None:
            continue
        for left, right in ((a, b), (b, a)):
            gap = right[0]-left[1]
            if (abs(left[1]-plane) <= EPS and abs(right[0]-plane) <= EPS and gap >= -EPS
                    and left[0] < plane-EPS and right[1] > plane+EPS):
                return {'axis': axis.tolist(), 'registered_point': point.tolist(),
                        'first_support_interval_m': a, 'second_support_interval_m': b,
                        'gap_m': float(gap), 'numerical_contact_eps_m': EPS,
                        'basis': 'registered face; all actual mesh vertices and analytic primitive bounds occupy opposite closed half-spaces'}
    return None


def _mesh_reducer(component, row, model, data):
    """Measure both actual round end faces, not a tagged cone or its AABB."""
    if row['type'] != int(mujoco.mjtGeom.mjGEOM_MESH):
        return None
    declared = np.asarray(component.get('ends', []), dtype=float)
    if declared.shape != (2, 3) or math.dist(*declared) <= EPS:
        return None
    direction = declared[1]-declared[0]; direction /= np.linalg.norm(direction)
    mesh = int(model.geom_dataid[row['index']]); start = int(model.mesh_vertadr[mesh])
    count = int(model.mesh_vertnum[mesh])
    vertices = np.asarray(model.mesh_vert[start:start+count], dtype=float)
    vertices = vertices@data.geom_xmat[row['index']].reshape(3, 3).T+data.geom_xpos[row['index']]
    coordinates = vertices@direction; lo, hi = float(coordinates.min()), float(coordinates.max())
    faces = []
    for limit in (lo, hi):
        face = vertices[np.abs(coordinates-limit) <= EPS]
        if len(face) < 3:
            return None
        center = face.mean(axis=0)
        radii = np.linalg.norm(face-center, axis=1)
        faces.append({'center': center.tolist(), 'radius': float(radii.max()), 'vertex_count': len(face)})
    aligned = all(math.dist(faces[i]['center'], declared[i]) <= EPS for i in (0, 1))
    transverse = faces[1]['center']-np.asarray(faces[0]['center'])
    aligned = aligned and np.linalg.norm(np.cross(transverse, direction)) <= EPS
    return {'ends': [f['center'] for f in faces], 'radii': [f['radius'] for f in faces],
            'radius': max(f['radius'] for f in faces), 'faces': faces, 'declared_axis_matches': bool(aligned)}


def _valves(graph, actual):
    rows, errors, segments, geom_owner = [], [], [], {}
    for valve in graph.get('valves', []):
        uid = valve['id']; local_errors = []
        try:
            asset = inspect_asset(valve['dn'], valve.get('operator'))
            ordered_ports = flow_ports(asset, valve.get('reversed_ends', False))
        except AssetUnavailable as error:
            errors.append(_error('P1_VALVE_PACKAGE_UNAVAILABLE', valve=uid, detail=str(error)))
            continue
        if valve.get('package') != asset['package'] or valve.get('scale') != [1., 1., 1.]:
            local_errors.append(_error('P1_VALVE_PACKAGE_MISMATCH', valve=uid))
        prefix = valve.get('geom_prefix', valve.get('prefix', uid+'_'))
        mapping = valve.get('geom_ids')
        origin = np.asarray(valve.get('origin', valve.get('center', [])))
        matrix = np.asarray(valve.get('matrix', []))
        if origin.shape != (3,) or matrix.shape != (3, 3):
            local_errors.append(_error('P1_VALVE_POSE_MISSING', valve=uid))
            rows.append({'id': uid, 'status': 'fail', 'errors': local_errors})
            errors.extend(local_errors)
            continue
        names = []; fixed_names = []
        for local in asset['local_geoms']:
            name = (mapping.get(local['name']) if isinstance(mapping, dict) else
                    next((n for n in mapping if n.endswith(local['name'])), None)
                    if isinstance(mapping, list) else prefix+local['name'])
            measured = actual.get(name)
            if measured is None:
                local_errors.append(_error('P1_VALVE_NATIVE_GEOM_MISSING', valve=uid, component=name,
                                          package_geom=local['name']))
                continue
            if name in geom_owner and geom_owner[name] != uid:
                local_errors.append(_error('P1_NATIVE_VALVE_SHARED', valve=uid, other_valve=geom_owner[name], component=name))
            names.append(name); geom_owner[name] = uid
            if not local['moving']:
                fixed_names.append(name)
            expected_position = origin+matrix@np.asarray(local['center'])
            expected_matrix = matrix@np.asarray(local['matrix'])
            if (measured['type'] != local['type'] or not _near(measured['size'], local['size'])
                    or not _near(measured['position'], expected_position)
                    or not _near(measured['matrix'], expected_matrix)):
                local_errors.append(_error('P1_VALVE_NATIVE_GEOMETRY_MISMATCH', valve=uid,
                    component=name, actual=measured,
                    expected={'type': local['type'], 'size': local['size'],
                              'position': expected_position.tolist(), 'matrix': expected_matrix.tolist()}))
        ports = {key: (origin+matrix@np.asarray(port['position'])).tolist()
                 for key, port in ordered_ports.items()}
        port_normals = {key: (matrix@np.asarray(port['normal'])).tolist()
                        for key, port in ordered_ports.items()}
        row = {'id': uid, 'dn': asset['dn'], 'package': asset['package'], 'geom_ids': names,
               'fixed_geom_ids': fixed_names, 'actual_ports': ports, 'actual_port_normals': port_normals,
               'status': 'fail' if local_errors else 'pass', 'errors': local_errors,
               'medium': valve.get('medium'), 'level': valve.get('level'),
               'scope': 'Check geometry type, size and pose of the exact-DN pinned package, including visual-only elements'}
        rows.append(row); errors.extend(local_errors)
        if not local_errors:
            segments.append({'id': uid, 'ends': [ports['in'], ports['out']], 'valve': uid})
    return rows, errors, segments, geom_owner


def _route_segments(routes, components, actual, od, model, data, *, elbow_measurements=None):
    by_route = {}; errors = []; measured_rows = []
    from .piping_direction_checks import measure_elbows, elbow_records_from_components
    if elbow_measurements is None:
        records = elbow_records_from_components(components.values())
        elbow_measurements = measure_elbows({'components': list(components.values())},
            {'elbows': records}, model, data, actual) if records else {'by_component': {}, 'errors': []}
    errors.extend(elbow_measurements['errors'])
    for route in routes:
        segments = []
        for name in route.get('components', []):
            measured = actual.get(name); component = components.get(name)
            if measured is None or component is None:
                errors.append(_error('P1_ROUTE_GEOM_MISSING', route=route['id'], component=name))
                continue
            if name in elbow_measurements['by_component']:
                arc = elbow_measurements['by_component'][name]
                expected = od.get(str(route.get('dn')))
                okay = expected is not None and abs(2*arc['radius']-expected/1000.) <= EPS
                measured_rows.append({'component': name, 'route': route['id'], 'dn': route.get('dn'),
                    'actual_outer_diameter_m': 2*arc['radius'],
                    'expected_outer_diameter_m': expected/1000. if expected is not None else None,
                    'path_kind': 'actual_elbow_piece', 'pass': okay})
                if not okay:
                    errors.append(_error('PG17_ELBOW_GEOMETRY', component=name, route=route['id'],
                                         reason='Measured elbow diameter differs from the nominal diameter of the connected pipe'))
                segments.append(dict(arc)); continue
            cylinder = _cylinder(measured)
            if component.get('kind') == 'piping_reducer':
                cylinder = _mesh_reducer(component, measured, model, data)
                okay = bool(cylinder and cylinder['declared_axis_matches']
                    and len(component.get('radii', [])) == 2
                    and _near(cylinder['radii'], component['radii']))
                if not okay:
                    errors.append(_error('P1_REDUCER_ACTUAL_GEOMETRY', route=route['id'],
                        component=name, actual=cylinder))
            if cylinder is None:
                # Flat flange/adapter boxes need an explicitly physical port
                # adapter in a future contract, rather than invented centerlines.
                if component.get('kind') in ('pipe', 'reducer', 'pipe_reducer', 'piping_reducer'):
                    errors.append(_error('P1_PIPE_SHAPE_UNSUPPORTED', route=route['id'], component=name))
                continue
            if component.get('kind') == 'pipe':
                expected = od.get(str(route.get('dn')))
                okay = expected is not None and abs(2*cylinder['radius']-expected/1000.) <= EPS
                measured_rows.append({'component': name, 'route': route['id'], 'dn': route.get('dn'),
                    'actual_outer_diameter_m': 2*cylinder['radius'],
                    'expected_outer_diameter_m': expected/1000. if expected is not None else None,
                    'pass': okay})
                if not okay:
                    errors.append(_error('P1_PIPE_REAL_DN_MISMATCH', **measured_rows[-1]))
            if component.get('kind') in ('pipe', 'reducer', 'pipe_reducer', 'adapter', 'flange',
                                         'blind', 'blind_flange', 'flange_blind', 'piping_reducer'):
                segments.append({'id': name, **cylinder})
        by_route[route['id']] = segments
    return by_route, errors, measured_rows


def _connected(start, targets, segments):
    """Centerline graph from actual primitives, including genuine T intersections."""
    if not targets or not segments:
        return False
    nodes = [list(start), *[list(p) for p in targets]]
    for segment in segments:
        nodes.extend(segment['ends'])
    edges = defaultdict(set)
    for segment in segments:
        contained = [i for i, p in enumerate(nodes) if _on_segment(p, *segment['ends'])]
        for i in contained:
            edges[i].update(contained)
    visited = {0}; queue = deque([0])
    while queue:
        current = queue.popleft()
        for other in edges[current]-visited:
            visited.add(other); queue.append(other)
    return bool(visited & set(range(1, 1+len(targets))))


def _touch_points(first, second):
    result = []
    for a in first:
        for b in second:
            for point in a['ends']:
                if _on_segment(point, *b['ends']):
                    result.append(point)
            for point in b['ends']:
                if _on_segment(point, *a['ends']):
                    result.append(point)
    return result


def _contact_allowed(first, second, record, actual, segments, native_ports, native_geoms):
    pair = record.get('pair', record.get('components'))
    point = record.get('point')
    if not isinstance(pair, (list, tuple)) or set(pair) != {first, second} or not isinstance(point, (list, tuple)) or len(point) != 3:
        return False
    a = segments.get(first) or _cylinder(actual[first])
    b = segments.get(second) or _cylinder(actual[second])
    if a is None or b is None or first in native_geoms or second in native_geoms:
        for pipe_id, native_id, pipe in ((first, second, a), (second, first, b)):
            if pipe is None or pipe_id in native_geoms or native_id not in native_ports:
                continue
            for port in native_ports[native_id]:
                if math.dist(point, port['position']) > EPS:
                    continue
                end = next((i for i, p in enumerate(pipe['ends']) if math.dist(point, p) <= EPS), None)
                if end is None:
                    continue
                outward = np.asarray(pipe['ends'][1-end])-point
                norm = np.linalg.norm(outward)
                if norm <= EPS or float((outward/norm)@np.asarray(port['normal'])) < 1.-EPS:
                    continue
                radius = pipe.get('radii', [pipe['radius']]*2)[end]
                if all(min(actual[first]['box'][i+3], actual[second]['box'][i+3]) <= point[i]+radius+EPS
                       and max(actual[first]['box'][i], actual[second]['box'][i]) >= point[i]-radius-EPS
                       for i in range(3)):
                    return True
        return False
    # A long inclined cylinder has a loose world AABB. The corners of an
    # intersection of two AABBs are not evidence of remote solid overlap.
    # Prove locality from the actual intersecting axes instead. For any point
    # in both infinite cylinders, |axial distance from joint| is bounded by
    # (r1+r2)/sin(angle). Thus finite cylinders cannot overlap outside the
    # recorded finite radius below. Collinear overlaps remain forbidden.
    if (not _on_segment(point, *a['ends']) or not _on_segment(point, *b['ends'])
            or min(math.dist(point, p) for p in a['ends']+b['ends']) > EPS):
        return False
    u = np.asarray(a['ends'][1])-a['ends'][0]; v = np.asarray(b['ends'][1])-b['ends'][0]
    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    sine = float(np.linalg.norm(np.cross(u, v)))
    if sine <= EPS:
        return False
    axial_bound = (a['radius']+b['radius'])/sine
    return {'type': 'declared_noncollinear_endpoint_joint', 'point': list(point),
            'actual_axes': [u.tolist(), v.tolist()], 'sin_angle': sine,
            'actual_radii_m': [a['radius'], b['radius']],
            'intersection_radius_bound_m': math.hypot(axial_bound, max(a['radius'], b['radius'])),
            'basis': 'actual axes meet at recorded joint; at least one real endpoint; finite-cylinder intersection lies within radius derived from actual radii and axis angle'}


def _verified_original_stringer_joint(part_id, board_id, parents, actual, frozen, model, data, source_cache):
    """Prove unchanged original solid plus a contained actual plate remnant.

    No component type is globally exempt. Both actual models must prove this
    exact original pair already overlapped, and the new box adds no volume.
    """
    from pathlib import Path
    if frozen['contracts']['layout']['checks'].get('status') != 'pass':
        return None
    if 'actual' not in source_cache:
        path = frozen.get('source_evidence', {}).get('model_xml') or str(Path(frozen['source_dir'])/'room.xml')
        old_model = mujoco.MjModel.from_xml_path(str(path)); old_data = mujoco.MjData(old_model)
        mujoco.mj_forward(old_model, old_data)
        source_cache.update(model=old_model, data=old_data, actual=_measured(old_model, old_data), path=str(path))
    old = source_cache['actual']; part = actual[part_id]; old_part = old.get(part_id)
    if old_part is None or part['type'] != old_part['type'] or not all(
            _near(part[key], old_part[key]) for key in ('size', 'position', 'matrix')):
        return None
    current_board = actual[board_id]
    if current_board['type'] != int(mujoco.mjtGeom.mjGEOM_BOX):
        return None
    current_corners = np.asarray([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    current_corners = (current_corners*np.asarray(current_board['size']))@np.asarray(current_board['matrix']).T+current_board['position']
    for parent in parents:
        old_board = old.get(parent['id'])
        if old_board is None or old_board['type'] != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        local = (current_corners-old_board['position'])@np.asarray(old_board['matrix'])
        if np.any(np.abs(local) > np.asarray(old_board['size'])+EPS):
            continue
        native = float(mujoco.mj_geomDistance(source_cache['model'], source_cache['data'],
                    old_part['index'], old_board['index'], .01, None))
        if native >= -PENETRATION_LIMIT_M:
            continue
        return {'type': 'verified_original_stringer_plate_joint_inherited_by_subset',
            'original_source_xml': source_cache['path'], 'original_part': part_id,
            'original_plate': parent['id'], 'actual_remaining_plate': board_id,
            'original_native_distance_m': native, 'unchanged_actual_type': part['type'],
            'unchanged_actual_position': part['position'], 'unchanged_actual_matrix': part['matrix'],
            'unchanged_actual_size': part['size'], 'remaining_box_corners_in_old_box_local': local.tolist(),
            'original_actual_box_half_sizes': old_board['size'], 'actual_remaining_box': current_board['box'],
            'scope': 'only this original accepted pair; unchanged original geometry; actual new BOX wholly inside old BOX, hence no new intersection volume; new pipes never exempt'}
    return None


def _inherited_board_joint(first, second, original, components, actual, records, frozen, model, data, source_cache=None):
    """Only a contained remnant of an original plate inherits an old joint."""
    for board_id, part_id in ((first, second), (second, first)):
        board = components.get(board_id, {})
        if board.get('kind') not in ('floor', 'walk_surface') or part_id not in original:
            continue
        measured = actual[board_id]; part = original[part_id]
        if measured['type'] != int(mujoco.mjtGeom.mjGEOM_BOX) or part['kind'] not in (
                'column', 'beam', 'equipment_beam', 'support', 'lug', 'stringer'):
            continue
        if not _near(actual[part_id]['box'], part['box']):
            continue
        parents = [c for c in original.values() if c['kind'] in ('floor', 'walk_surface')
            and all(measured['box'][i] >= c['box'][i]-EPS and measured['box'][i+3] <= c['box'][i+3]+EPS
                    for i in range(3))
            and all(min(c['box'][i+3], part['box'][i+3])-max(c['box'][i], part['box'][i]) > 0 for i in range(3))]
        if not parents:
            continue
        if part['kind'] == 'stringer':
            joint = _verified_original_stringer_joint(part_id, board_id, parents, actual, frozen, model, data,
                                                      source_cache if source_cache is not None else {})
            if joint is None:
                continue
        elif part['kind'] in ('support', 'lug'):
            from .solid_collision_checks import _local_bearing_joint, _legacy_bearing_joint
            by_name = {r['id']: r for r in records}
            joint = (_local_bearing_joint(by_name[part_id], by_name[board_id], records, model, data, frozen['graph'], EPS)
                     or _legacy_bearing_joint(by_name[part_id], by_name[board_id], records, model, data, frozen['graph'], EPS))
            if joint is None:
                continue
        else:
            joint = {'type': 'inherited_original_column_beam_plate_joint'}
        source_parent = next(c for c in parents if c['id'] == joint.get('original_plate', parents[0]['id']))
        return {**joint, 'original_plate': source_parent['id'], 'actual_remaining_plate': board_id,
                'actual_remaining_box': measured['box'], 'source_plate_box': source_parent['box'],
                'scope': 'Only the measured remainder of the original slab and original stationary parts receive the preexisting contact exception'}
    return None


def _rack_column_joint(first, second, original, components, actual):
    for beam_id, column_id in ((first, second), (second, first)):
        beam = components.get(beam_id, {}); column = original.get(column_id, {})
        if not beam.get('piping') or beam.get('kind') != 'beam' or column.get('kind') != 'column':
            continue
        a, b = actual[beam_id], actual[column_id]
        if (a['type'] != int(mujoco.mjtGeom.mjGEOM_BOX) or b['type'] != int(mujoco.mjtGeom.mjGEOM_BOX)
                or not _near(b['box'], column['box'])):
            continue
        axis = int(np.argmax(np.asarray(a['box'][3:])-a['box'][:3]))
        centre = (np.asarray(a['box'][:3])+a['box'][3:])/2
        ends = []
        for side in (axis, axis+3):
            p = centre.copy(); p[axis] = a['box'][side]; ends.append(p.tolist())
        if any(all(b['box'][i]-EPS <= p[i] <= b['box'][i+3]+EPS for i in range(3)) for p in ends):
            return {'type': 'actual_new_rack_beam_end_in_frozen_column', 'beam': beam_id, 'column': column_id,
                    'actual_beam_ends': ends, 'actual_column_box': b['box'],
                    'scope': 'Local original beam end in an original column section. Interior beam penetration still fails.'}
    return None


def _geometry(frozen, scene, piping, model, data, actual, valve_geoms, by_route, valve_rows):
    original_components = {c['id']: c for c in frozen['scene']['components']}
    original = set(original_components)
    components = {c['id']: c for c in scene['components']}
    new = set(actual)-original
    errors = []; rows = []; private_model = private_data = None
    support_cache = {}; support_proofs = 0; fallback_rechecks = 0; original_inheritance_cache = {}
    for name in new:
        if name not in components and name not in valve_geoms:
            errors.append(_error('P1_NEW_GEOMETRY_UNREGISTERED', component=name))
    contacts = piping.get('allowed_contacts', [])
    contacts_by_pair = defaultdict(list)
    for record in contacts:
        pair = record.get('pair', record.get('components', []))
        if len(pair) == 2:
            contacts_by_pair[frozenset(pair)].append(record)
    segments = {segment['id']: segment for values in by_route.values() for segment in values}
    native_ports = {name: [{'position': position, 'normal': row['actual_port_normals'][key]}
                          for key, position in row['actual_ports'].items()]
                    for row in valve_rows if row['status'] == 'pass' for name in row['fixed_geom_ids']}
    from .solid_collision_checks import STRUCTURAL_KINDS, EQUIPMENT_KINDS
    records = []
    for name, measured in actual.items():
        component = components.get(name, {'id': name, 'kind': 'native_valve_part'})
        kind = component['kind']; owner = component.get('owner')
        records.append({'id': name, 'kind': kind, 'owner': owner,
            'role': 'structure' if kind in STRUCTURAL_KINDS else 'equipment' if owner or kind in EQUIPMENT_KINDS else 'other',
            'component': component, 'geom_index': measured['index'], 'geom_type': measured['type'], 'box': measured['box']})
    names = [name for name in actual if not components.get(name,{}).get('visual_only')]
    bounds=np.asarray([actual[name]['box'] for name in names]).reshape(-1,6)
    for index, first in enumerate(names):
        following=bounds[index+1:]
        hits=np.all(np.minimum(bounds[index,3:],following[:,3:])>
                    np.maximum(bounds[index,:3],following[:,:3]),axis=1)
        for offset in np.flatnonzero(hits):
            second=names[index+1+offset]
            if first not in new and second not in new:
                continue
            if first in valve_geoms and valve_geoms.get(first) == valve_geoms.get(second):
                continue  # Only verified internal parts of one native package.
            one,two=components.get(first,{}),components.get(second,{})
            if one.get('owner')==two.get('owner') and any(
                    one.get(key) and two.get(key) for key in ('catalogue_source_solid','catalogue_mate_solid')):
                # Fixed catalogue parts or the pieces of one fabricated mate.
                continue
            a = actual[first]; b = actual[second]
            native = float(mujoco.mj_geomDistance(model, data, a['index'], b['index'], .01, None))
            distance = native
            if abs(native) <= EPS:
                proof = _separation_proof(a, b, model, support_cache)
                if proof is not None:
                    support_proofs += 1
                    rows.append({'first': first, 'second': second, 'native_distance_m': native,
                        'distance_m': proof['gap_m'], 'distance_is_support_bound': True,
                        'status': 'proved_separated_or_numerical_contact', 'support_plane': proof,
                        'fallback_excluded_by_geometry': True,
                        'actual_boxes': [a['box'], b['box']]})
                    continue
                if private_model is None:
                    private_model = deepcopy(model)
                    private_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
                    private_data = mujoco.MjData(private_model)
                    mujoco.mj_copyData(private_data, private_model, data)
                distance = min(native, float(mujoco.mj_geomDistance(private_model, private_data,
                                                                  a['index'], b['index'], .01, None)))
                fallback_rechecks += 1
            if distance >= -PENETRATION_LIMIT_M:
                continue
            pair_contacts = contacts_by_pair.get(frozenset((first, second)), ())
            face_proof = next((proof for record in pair_contacts
                if (proof := _registered_face_proof(a, b, record, segments, model, support_cache))), None)
            if face_proof is not None:
                support_proofs += 1
                rows.append({'first': first, 'second': second, 'native_distance_m': native,
                    'distance_m': face_proof['gap_m'], 'distance_is_support_bound': True,
                    'status': 'proved_registered_face_contact', 'support_plane': face_proof,
                    'actual_boxes': [a['box'], b['box']]})
                continue
            contact_proof = next((proof for record in pair_contacts
                if (proof := _contact_allowed(first, second, record, actual, segments, native_ports, valve_geoms))), None)
            structural_joint = (_inherited_board_joint(first, second, original_components, components, actual,
                                                       records, frozen, model, data, original_inheritance_cache)
                                or _rack_column_joint(first, second, original_components, components, actual))
            permitted = bool(contact_proof) or structural_joint is not None
            row = {'first': first, 'second': second, 'distance_m': distance,
                   'native_distance_m': native, 'penetration_m': -distance,
                   'local_endpoint_joint': permitted, 'structural_joint': structural_joint,
                   'local_endpoint_evidence': contact_proof,
                   'actual_boxes': [a['box'], b['box']]}
            rows.append(row)
            if not permitted:
                errors.append(_error('P1_SOLID_COLLISION', **row))
    return _section(rows, errors, penetration_limit_m=PENETRATION_LIMIT_M,
                    support_plane_proofs=support_proofs, zero_native_fallback_rechecks=fallback_rechecks,
                    collision_masks_used=False, owner_exemption=False,
                    scope='Check all new solids against original and new geometry. Exceptions are limited to verified native assemblies and local mating pairs.')


def _zone_rows(piping):
    raw = piping.get('rack_zones', [])
    if isinstance(raw, dict):
        return {int(k): v['box'] if isinstance(v, dict) else v for k, v in raw.items()}
    return {int(r['level']): r['box'] for r in raw}


def _edge_regions(frozen, piping, actual, by_route, config):
    """Recompute the station strip from original frame faces and actual OD."""
    w, d = frozen['scene']['footprint']
    station = piping.get('station', {}); edge = station.get('edge')
    frames = {'x_min': ([0., 0.], [1., 0.], [0., 1.], d),
              'x_max': ([w, 0.], [-1., 0.], [0., 1.], d),
              'y_min': ([0., 0.], [0., 1.], [1., 0.], w),
              'y_max': ([0., d], [0., -1.], [1., 0.], w)}
    if edge not in frames:
        return {}
    origin, normal, tangent, length = frames[edge]
    origin, normal, tangent = map(np.asarray, (origin, normal, tangent))
    inner_faces, frame_ids = [], []
    for c in frozen['scene']['components']:
        if c['kind'] not in ('column', 'beam') or c['id'] not in actual:
            continue
        box = actual[c['id']]['box']
        corners = [np.array([x, y])-origin for x in (box[0], box[3]) for y in (box[1], box[4])]
        ns = [float(p@normal) for p in corners]; ts = [float(p@tangent) for p in corners]
        if min(ns)-EPS <= 0. <= max(ns)+EPS and (c['kind'] == 'column' or max(ts)-min(ts) >= length-EPS):
            inner_faces.append(max(ns)); frame_ids.append(c['id'])
    radii = [s['radius'] for r in piping.get('routes', []) if r.get('role') == 'workstation_main'
             for s in by_route.get(r['id'], [])]
    if not inner_faces or not radii:
        return {}
    fitting_radii = list(radii)
    for route in piping.get('routes', []):
        if route.get('role') != 'workstation_main':
            continue
        tips = [p for s in by_route.get(route['id'], []) for p in s['ends']]
        if not tips:
            continue
        top = max(tips, key=lambda p: p[2])
        for cap in piping.get('blinds', []):
            measured = actual.get(cap.get('id'))
            cylinder = _cylinder(measured) if measured else None
            if (cap.get('role') == 'main_end' and cap.get('dn') == route.get('dn') and cylinder
                    and min(math.dist(top, p) for p in cylinder['ends']) <= EPS
                    and _near(cylinder['ends'][0][:2], cylinder['ends'][1][:2])):
                fitting_radii.append(cylinder['radius'])
    wall = config['pg_rules']['parallel_pipe_spacing']['outermost_pipe_or_insulation_to_column_or_wall_min_mm']/1000.
    band = max(inner_faces)+wall+max(fitting_radii)+max(radii)
    expected = {'x_min': [0., 0., band, d], 'x_max': [w-band, 0., w, d],
                'y_min': [0., 0., w, band], 'y_max': [0., d-band, w, d]}[edge]
    return {int(r['level']): {'rect': expected, 'declared_matches': _near(r.get('rect', []), expected),
            'frame_component_ids': frame_ids, 'actual_frame_inner_face_m': max(inner_faces),
            'actual_max_main_od_m': 2*max(radii), 'actual_max_main_fitting_radius_m': max(fitting_radii),
            'pipe_to_column_clearance_m': wall}
            for r in station.get('crossing_strips', []) if r.get('edge') == edge}


def _hierarchy(frozen, piping, actual, by_route, config, scene):
    rows, errors = [], []
    zones = _zone_rows(piping); floors = frozen['scene']['floors']
    w, d = frozen['scene']['footprint']; original = {c['id']: c for c in frozen['scene']['components']}
    cfg = config['pg_rules']['distribution']['rack_zone']
    edge_regions = _edge_regions(frozen, piping, actual, by_route, config)
    for route in piping.get('routes', []):
        level = route.get('level'); medium = route.get('medium')
        if route.get('role') == 'floor_header':
            zone = zones.get(level); valid_zone = zone is not None and len(zone) == 6
            if level == len(floors)-1:
                expected_z = [floors[level]+cfg['top_floor_bottom_above_floor_m'],
                              floors[level]+cfg['top_floor_bottom_above_floor_m']+cfg['thickness_m']]
            elif isinstance(level, int) and 0 <= level < len(floors)-1:
                beam_bottoms = [actual[name]['box'][2] for name, c in original.items()
                    if c['kind'] in ('beam', 'equipment_beam') and name in actual
                    and floors[level] < actual[name]['box'][2] < floors[level+1]
                    and actual[name]['box'][5] <= floors[level+1]+EPS]
                expected_z = [min(beam_bottoms)-cfg['thickness_m'], min(beam_bottoms)] if beam_bottoms else None
            else:
                expected_z = None
            if (not valid_zone or expected_z is None or not _near([zone[2], zone[5]], expected_z)
                    or zone[0] < -EPS or zone[1] < -EPS or zone[3] > w+EPS or zone[4] > d+EPS):
                errors.append(_error('P1_RACK_ZONE_NOT_FROM_STRUCTURE', route=route['id'], zone=zone,
                                     expected_z=expected_z))
            for segment in by_route.get(route['id'], []):
                shape = actual[segment['id']]
                horizontal = abs(segment['ends'][0][2]-segment['ends'][1][2]) <= EPS
                inside = bool(valid_zone and all(shape['box'][i] >= zone[i]-EPS
                              and shape['box'][i+3] <= zone[i+3]+EPS for i in range(3)))
                row = {'route': route['id'], 'component': segment['id'], 'level': level,
                       'medium': medium, 'actual_box': shape['box'], 'zone': zone, 'horizontal': horizontal,
                       'inside': inside}
                rows.append(row)
                if horizontal and not inside:
                    errors.append(_error('P1_HEADER_OUTSIDE_RACK', **row))
        for segment in by_route.get(route['id'], []):
            a, b = map(np.asarray, segment['ends'])
            if abs(a[2]-b[2]) <= EPS:
                continue
            for lev, z in enumerate(floors[1:], 1):
                if min(a[2], b[2])+EPS >= z or max(a[2], b[2])-EPS <= z:
                    continue
                point = a+(b-a)*(z-a[2])/(b[2]-a[2]); radius = segment['radius']
                rect = [point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius]
                # A permitted corridor must be derived from actual outer-frame
                # geometry by the station candidate, not an invented strip.
                region = edge_regions.get(lev)
                edge = bool(region and region['declared_matches'] and all(
                    rect[i] >= region['rect'][i]-EPS and rect[i+2] <= region['rect'][i+2]+EPS for i in range(2)))
                openings = [o for o in piping.get('openings', []) if o.get('level') == lev]
                registered = any(all(rect[i] >= o['rect'][i]-EPS and rect[i+2] <= o['rect'][i+2]+EPS
                                     for i in range(2)) for o in openings)
                from .grating import grating,overlaps
                covers=any(grating(c) and abs(c['box'][5]-z)<EPS and overlaps(rect,c['box']) for c in frozen['scene']['components'])
                boards=[actual[c['id']] for c in scene['components'] if grating(c) and c['id'] in actual
                        and abs(c['box'][5]-z)<EPS and overlaps(rect,c['box'])]
                row = {'component': segment['id'], 'level': lev, 'actual_crossing': point.tolist(),
                       'outer_rect': rect, 'edge_region': edge, 'registered_opening': registered,
                       'actual_edge_region_evidence': region,
                       'actual_board_ids_at_level': [v['id'] for v in boards]}
                rows.append(row)
                if boards or (covers and not registered):
                    errors.append(_error('P1_GRATING_OPENING', **row))
    return _section(rows, errors,
        scope='Measure cross-branch edges, rack space under beams and real floor crossings against openings and clearances')


def _utility_details(frozen, scene, piping, expected, actual, by_route, route_parts, valid_valves, group_contract,
                     replaced_flange_ids=()):
    rows, errors = [], list(group_contract['errors'])
    routes = piping.get('routes', []); config = piping['config']
    components = {c['id']: c for c in scene['components']}
    for c in frozen['scene']['components']:
        # Locally cut decks are separately measured by platform coverage.
        if c['kind'] in ('floor', 'walk_surface') or c['id'] in replaced_flange_ids:
            continue
        measured = actual.get(c['id'])
        if measured is None or not _near(measured['box'], c['box']):
            errors.append(_error('P1_FROZEN_SOLID_CHANGED', component=c['id'], actual=measured))
    for medium in expected['active_services']:
        template = expected['service_templates'][medium]
        mains = [r for r in routes if r.get('medium') == medium and r.get('role') == 'workstation_main']
        legal_dns = ([template['main_dn']] if template.get('main_dn') is not None
                     else template.get('main_dn_candidates', []))
        if len(mains) != 1 or any(r.get('dn') not in legal_dns for r in mains):
            errors.append(_error('P1_MAIN_COUNT_OR_DN', medium=medium, actual_mains=[r['id'] for r in mains], candidates=legal_dns))
        for main in mains:
            parts = by_route.get(main['id'], [])
            vertical = bool(parts) and all(_near(s['ends'][0][:2], s['ends'][1][:2]) for s in parts)
            if not vertical:
                errors.append(_error('P1_MAIN_NOT_VERTICAL', route=main['id']))
        for route in routes:
            if route.get('medium') != medium:
                continue
            kind = route.get('role')
            if kind in ('floor_header', 'floor_branch') and route.get('dn') != template['branch_dn']:
                errors.append(_error('P1_FLOOR_BRANCH_DN', route=route['id'], expected_dn=template['branch_dn']))
            if kind == 'temporary_outlet' and route.get('dn') != template['temporary_dn']:
                errors.append(_error('P1_TEMPORARY_DN', route=route['id'], expected_dn=template['temporary_dn']))
        for level in range(len(frozen['scene']['floors'])):
            users = [p for p in expected['service_points'] if p['medium'] == medium and p['level'] == level]
            headers = [r for r in routes if r.get('medium') == medium and r.get('level') == level and r.get('role') == 'floor_header']
            service_groups = [g for g in group_contract['groups'] if g['medium'] == medium and g['level'] == level]
            expected_headers = {g['header_id'] for g in service_groups if g.get('header_id')}
            if (len(headers) != len(service_groups) or not group_contract['legacy']
                    and {h['id'] for h in headers} != expected_headers):
                errors.append(_error('P1_ACTIVE_HEADER_COUNT', medium=medium, level=level,
                                     expected=len(service_groups), actual=len(headers), expected_headers=sorted(expected_headers)))
        # Derive mandatory caps from the frozen service/floor demand, not from
        # whether an emitted route retained a convenient target.blind label.
        slots = []
        for main in mains:
            parts = by_route.get(main['id'], [])
            targets = sorted([p for s in parts for p in s['ends']], key=lambda p: p[2], reverse=True)[:1]
            slots.append(('main_end', len(frozen['scene']['floors'])-1, main['dn'], parts, targets, []))
        for level in range(len(frozen['scene']['floors'])):
            users = [p for p in expected['service_points'] if p['medium'] == medium and p['level'] == level]
            for role in ('temporary_outlet', 'floor_branch'):
                if role == 'floor_branch' and (medium == 'offgas' or users):
                    continue
                matching = [r for r in routes if r.get('medium') == medium and r.get('level') == level and r.get('role') == role]
                parts = [s for r in matching for s in by_route.get(r['id'], [])]
                roots = list(dict.fromkeys(v for r in matching for v in r.get('valves', []) if v in valid_valves))
                targets = [valid_valves[v]['actual_ports']['out'] for v in roots]
                slots.append(('temporary_outlet' if role == 'temporary_outlet' else 'unused_floor_branch',
                              level, template['temporary_dn'] if role == 'temporary_outlet' else template['branch_dn'],
                              parts, None, targets))
        for role, level, dn, parts, exact_targets, roots in slots:
            candidates = []
            for cap in piping.get('blinds', []):
                c = components.get(cap.get('id'), {}); shape = actual.get(cap.get('id'))
                cylinder = _cylinder(shape) if shape else None
                if (cap.get('role') != role or cap.get('level') != level or c.get('medium') != medium
                        or cap.get('dn') != dn or c.get('kind') != 'piping_blind' or not cylinder):
                    continue
                radius = config['pipe_outer_diameter_mm'][str(dn)]/2000.
                face_points = [p for p in cylinder['ends'] if any(_on_segment(p, *s['ends']) for s in parts)]
                if (cylinder['radius']+EPS >= radius and math.dist(*cylinder['ends']) > EPS
                        and any((exact_targets is not None and any(_near(p, q) for q in exact_targets))
                                or (exact_targets is None and any(_connected(q, [p], parts) for q in roots))
                                for p in face_points)):
                    candidates.append(cap['id'])
            row = {'mandatory_blind': f'{medium}/{level}/{role}', 'medium': medium, 'level': level,
                   'role': role, 'dn': dn, 'actual_blinds': candidates, 'pass': len(candidates) == 1}
            rows.append(row)
            if not row['pass']:
                errors.append(_error('P1_REQUIRED_BLIND_MISSING', **row))
    for route in routes:
        target = route.get('target', {})
        if not isinstance(target, dict) or not target.get('blind'):
            continue
        position = target.get('position')
        candidates = []
        for record in piping.get('blinds', []):
            component = components.get(record.get('id')); shape = actual.get(record.get('id'))
            if not component or not shape or component.get('kind') != 'piping_blind':
                continue
            cylinder = _cylinder(shape)
            radius = config['pipe_outer_diameter_mm'].get(str(route.get('dn')), float('inf'))/2000.
            if (cylinder and position is not None and min(math.dist(position, p) for p in cylinder['ends']) <= EPS
                    and cylinder['radius']+EPS >= radius and math.dist(*cylinder['ends']) > EPS
                    and any(_on_segment(position, *s['ends']) for s in by_route.get(route['id'], []))):
                candidates.append(record['id'])
        row = {'route': route['id'], 'role': route['role'], 'position': position,
               'actual_blinds': candidates, 'pass': len(candidates) == 1}
        rows.append(row)
        if not row['pass']:
            errors.append(_error('P1_REQUIRED_BLIND_MISSING', **row))
    return _section(rows, errors, scope='Keep equipment geometry. Check actual main and branch DN, inactive floor branches and blinding. Measure vertical off-gas entry separately.')


def check_p1(frozen, graph, scene, piping, model, data, operation_report=None, *, physical_routes=None):
    """Check actual P1 output. No eligibility-ratio gate is applied in P1."""
    config = piping['config']
    expected = requirements_from_frozen(frozen, config)
    from .piping_service_group_checks import service_group_contracts, floor_link_evidence
    group_contract = service_group_contracts(frozen, piping)
    errors = []
    if piping.get('phase') != 'P1':
        errors.append(_error('P1_PHASE_MISMATCH'))
    actual = _measured(model, data)
    from .piping_equipment_flange_checks import measure_joints
    flange_joints = measure_joints(frozen, piping, model, actual)
    measured_sources = {row['component']:row for row in flange_joints['rows'] if row['phase']=='P1'}
    storage_rows = storage_service_checks(graph, actual)
    errors.extend(_error('HORIZONTAL_STORAGE_SERVICES', **row) for row in storage_rows if row['status'] == 'fail')
    components = {c['id']: c for c in scene['components']}
    routes = piping.get('routes', [])
    route_map = {r['id']: r for r in routes}
    if len(route_map) != len(routes):
        errors.append(_error('P1_ROUTE_ID_DUPLICATED'))
    by_route, route_errors, pipe_sizes = _route_segments(routes, components, actual,
                                                        config['pipe_outer_diameter_mm'], model, data)
    errors.extend(route_errors)
    valve_rows, valve_errors, valve_segments, valve_geoms = _valves(graph, actual)
    errors.extend(valve_errors)
    valves = {v['id']: v for v in graph.get('valves', [])}
    valid_valves = {v['id']: v for v in valve_rows if v['status'] == 'pass'}
    valve_edges = {v['valve']: v for v in valve_segments}
    def route_parts(route):
        return by_route.get(route['id'], [])+[valve_edges[v] for v in route.get('valves', []) if v in valve_edges]
    point_ids = {p['id'] for p in expected['service_points']}
    for route in routes:
        source = route.get('source_requirement')
        if ((source is not None and source not in point_ids)
                or route.get('role') == 'equipment_connection' and source is None):
            errors.append(_error('P1_UNEXPECTED_SOURCE_REQUIREMENT', route=route['id'], source_requirement=source))
        if route.get('medium') not in expected['active_services']:
            errors.append(_error('P1_SERVICE_WITHOUT_SOURCE', route=route['id'], medium=route.get('medium')))
    connections, manual = [], []
    for point in expected['service_points']:
        linked = [r for r in routes if r.get('source_requirement') == point['id']]
        connected = False; detail = []
        if not linked:
            errors.append(_error('P1_REQUIRED_CONNECTION_COUNT', requirement=point['id'], actual_routes=[r['id'] for r in linked]))
        for route in linked[:1]:
            local = [s for piece in linked for s in route_parts(piece)]
            header = route_map.get(route.get('distribution_id')); main = route_map.get(route.get('main_id'))
            correct_header = bool(header and header.get('role') == 'floor_header'
                and header.get('level') == point['level'] and header.get('medium') == point['medium'])
            correct_main = bool(main and main.get('role') == 'workstation_main' and main.get('medium') == point['medium'])
            header_parts = route_parts(header) if header else []
            main_parts = route_parts(main) if main else []
            touches_header = _touch_points(local, header_parts)
            own_to_header = _connected(point['position'], touches_header, local)
            direct_main = bool(_touch_points(local, main_parts))
            floor_proof = (floor_link_evidence(header, main, touches_header[0], routes, by_route,
                valve_rows, valve_segments, graph) if header and main and touches_header else None)
            header_to_main = bool(floor_proof and floor_proof['connected'])
            consistent_pieces = all(piece.get('role') == 'equipment_connection'
                and piece.get('level') == point['level'] and piece.get('medium') == point['medium']
                and piece.get('distribution_id') == route.get('distribution_id')
                and piece.get('main_id') == route.get('main_id') and piece.get('dn') == point['dn'] for piece in linked)
            connected = bool(consistent_pieces and correct_header and correct_main and own_to_header and header_to_main
                             and not direct_main and route.get('dn') == point['dn'])
            detail.append({'route': route['id'], 'actual_port_to_header': own_to_header,
                           'header_to_main': header_to_main, 'direct_to_main': direct_main,
                           'correct_floor_header': correct_header, 'correct_main': correct_main, 'floor_link_evidence': floor_proof})
            roots = list(dict.fromkeys(vid for piece in linked for vid in piece.get('valves', []) if vid in valid_valves
                     and valves[vid].get('dn') == point['dn']
                     and valves[vid].get('equipment') == point['equipment']))
            root_ok = consistent_pieces and len(roots) == 1 and bool(local) and _connected(point['position'],
                [p for vid in roots for p in valid_valves[vid]['actual_ports'].values()], local)
            manual.append({'requirement': point['id'], 'endpoint': point['endpoint'], 'role': point['role'],
                           'expected_dn': point['dn'], 'installed': root_ok, 'valves': roots})
            if not root_ok:
                errors.append(_error('P1_ROOT_VALVE_MISSING_OR_MISMATCH', **manual[-1]))
            if direct_main:
                errors.append(_error('P1_EQUIPMENT_DIRECT_TO_MAIN', requirement=point['id'], route=route['id']))
            if point['reducer']['required']:
                flange = measured_sources.get(point['endpoint'])
                reducer_face = flange['stub_end_m'] if flange else point['position']
                candidates = [s for piece in linked for s in by_route.get(piece['id'], [])
                              if components[s['id']].get('kind') == 'piping_reducer'
                              and _near(s['ends'][0], reducer_face)]
                source_nozzle = actual.get(point['equipment']+'_'+point['port']+'_equipment_nozzle')
                source_radius = _cylinder(source_nozzle)['radius'] if source_nozzle and _cylinder(source_nozzle) else None
                target_radius = config['pipe_outer_diameter_mm'][str(point['dn'])]/2000.
                good = any(source_radius is not None and _near(s.get('radii', []), [source_radius, target_radius])
                           and components[s['id']].get('source_dn') == point['source_dn']
                           and components[s['id']].get('target_dn') == point['dn'] for s in candidates)
                if not good:
                    errors.append(_error('P1_REQUIRED_REAL_REDUCER', requirement=point['id'],
                        endpoint=point['endpoint'], source_actual_radius_m=source_radius,
                        target_pipe_radius_m=target_radius, measured_reducers=candidates))
        if not linked:
            manual.append({'requirement': point['id'], 'endpoint': point['endpoint'],
                           'role': point['role'], 'expected_dn': point['dn'], 'installed': False, 'valves': []})
        connections.append({'requirement': point['id'], 'endpoint': point['endpoint'],
                            'medium': point['medium'], 'level': point['level'], 'connected': connected, 'details': detail})
        if not connected:
            errors.append(_error('P1_REQUIRED_CONNECTION_NOT_CONNECTED', **connections[-1]))
    group_rows = []
    all_parts = [s for route in routes for s in route_parts(route)]
    for edge in expected['source_p2_edges']:
        if edge.get('role') not in ('vapor', 'process_evaporated_vapor'):
            continue
        for endpoint in (edge['a'], edge['b']):
            uid, port = endpoint.split('.', 1)
            placed = next((u for u in frozen['contracts']['layout']['equipment'] if u['id'] == uid), None)
            if placed and port in placed['ports'] and any(
                    _on_segment(placed['ports'][port]['pos'], *s['ends']) for s in all_parts):
                errors.append(_error('P1_PROCESS_VAPOR_IN_UTILITY_GROUP', endpoint=endpoint, source_edge=edge['id']))
    point_map = {p['id']: p for p in expected['service_points']}
    for group in expected['groups']:
        incoming = point_map[group['in_requirement']]; outgoing = point_map[group['out_requirement']]
        selected = [row for row in manual if row['requirement'] in (incoming['id'], outgoing['id'])]
        root_ids = [vid for row in selected for vid in row['valves']]
        short = _connected(incoming['position'], [outgoing['position']], all_parts)
        declared = [g for g in piping.get('groups', []) if g.get('id') == group['id'] and g.get('owner') == group['owner']]
        group_valves = [v['id'] for v in graph.get('valves', []) if v.get('group_id') == group['id']]
        good = (len(root_ids) == 2 and len(set(root_ids)) == 2 and not short and len(declared) == 1
                and set(group_valves) == set(root_ids))
        row = {'group': group['id'], 'owner': group['owner'], 'requirements': [incoming['id'], outgoing['id']],
               'independent_root_valves': root_ids, 'external_short_circuit': short, 'pass': good}
        group_rows.append(row)
        if not good:
            errors.append(_error('P1_GAS_GROUP_TOPOLOGY', **row))
    # v1.6: one fixed root per real supply group; still one unused fixed
    # outlet on a no-service floor and one temporary outlet per service/floor.
    for medium in expected['active_services']:
        for level in range(len(frozen['scene']['floors'])):
            groups = [g for g in group_contract['groups'] if g['medium']==medium and g['level']==level]
            obligations = [('temporary_outlet', None)]
            if medium != 'offgas':
                obligations += [('floor_branch', g) for g in groups] if groups else [('floor_branch', None)]
            for kind, service_group in obligations:
                hid = service_group.get('header_id') if service_group else None
                matching = [r for r in routes if r.get('medium')==medium and r.get('level')==level
                    and r.get('role')==kind and (service_group is None or r.get('distribution_id')==hid)]
                roots = list(dict.fromkeys(vid for r in matching for vid in r.get('valves', [])
                    if vid in valid_valves and (kind=='temporary_outlet' or valves[vid].get('role')=='utility_floor_root')
                    and (service_group is None or group_contract['legacy'] or
                         valves[vid].get('distribution_id')==hid and valves[vid].get('service_group')==service_group['id'])))
                installed = bool(matching) and len(roots)==1
                row = {'requirement':f'{medium}/{level}/{kind}'+('/'+str(hid) if service_group else ''),
                    'medium':medium,'level':level,'role':kind,'header':hid,'installed':installed,'valves':roots}
                manual.append(row)
                if not installed:
                    errors.append(_error('P1_STATION_REQUIRED_VALVE', **row))
                if service_group:
                    header = route_map.get(hid)
                    mains = [r for r in routes if r.get('role')=='workstation_main' and r.get('medium')==medium]
                    positions = [p for segment in by_route.get(hid,[]) for p in segment['ends']]
                    proof = (floor_link_evidence(header,mains[0],positions[0],routes,by_route,valve_rows,valve_segments,graph)
                             if header and len(mains)==1 and positions else None)
                    if not proof or not proof['connected']:
                        errors.append(_error('P1_STATION_CONNECTION_NOT_CONNECTED', header=hid, medium=medium,
                                             level=level, floor_link_evidence=proof))
            fixed = [r for r in routes if r.get('medium')==medium and r.get('level')==level and r.get('role')=='floor_branch']
            if medium=='offgas' and any(r.get('valves') for r in fixed):
                errors.append(_error('P1_OFFGAS_MAIN_ROOT_FORBIDDEN', medium=medium, level=level))
    # A later stage can retain this exact P1 network while adding real process
    # pipes. Its explicit physical_routes contain all of those solids for the
    # same geometry/DN checks; the original P1 call and policy remain unchanged.
    geometry_routes = by_route
    if physical_routes is not None:
        geometry_routes, physical_errors, _ = _route_segments(physical_routes, components, actual,
            config['pipe_outer_diameter_mm'], model, data)
        errors.extend(physical_errors)
    geometry = _geometry(frozen, scene, piping, model, data, actual, valve_geoms, geometry_routes, valve_rows)
    hierarchy = _hierarchy(frozen, piping, actual, by_route, config, scene)
    replaced_flange_ids = {name for row in flange_joints['rows'] for name in row['replaced_components']}
    utility_details = _utility_details(frozen, scene, piping, expected, actual, by_route, route_parts, valid_valves,
                                       group_contract, replaced_flange_ids)
    hierarchy['rows'].extend(utility_details['rows']); hierarchy['errors'].extend(utility_details['errors'])
    from .piping_spacing import check_spacing
    spacing = check_spacing(scene, piping, model, data, actual=actual,
                            by_route=geometry_routes, valve_rows=valve_rows)
    hierarchy['rows'].extend(spacing['rows']); hierarchy['errors'].extend(spacing['errors'])
    from .piping_direction_checks import check_directions
    physical_piping = dict(piping, routes=physical_routes) if physical_routes is not None else piping
    directions = check_directions(frozen, graph, scene, physical_piping, model, data, flange_joints=flange_joints)
    hierarchy['rows'].extend(directions['rows']); hierarchy['errors'].extend(directions['errors'])
    hierarchy['violations'] = hierarchy['errors']
    hierarchy['status'] = 'fail' if hierarchy['errors'] else 'pass'
    from .platform_coverage import check_platform_coverage
    board_coverage = (check_platform_coverage(scene, model) if len(scene['floors']) > 1 else
                      {'status': 'pass', 'levels': [], 'errors': [],
                       'method': 'No platform coverage check when no platform exists. See the operation report for ground and valve standing areas.'})
    geometry['platform_coverage'] = board_coverage
    if board_coverage['status'] != 'pass':
        geometry['errors'].append(_error('P1_PLATFORM_COVERAGE', detail=board_coverage))
        geometry['status'] = 'fail'
    errors.extend(geometry['errors']); errors.extend(hierarchy['errors'])
    if operation_report is None:
        operation_report = {'status': 'not_run', 'total': len(graph.get('valves', [])), 'rows': [],
                            'eligible': None, 'by_role': {}, 'scope': 'P1 individual valve availability has not yet been measured'}
    return {'status': 'fail' if errors else 'pass', 'errors': errors, 'checks_revision': 'PG17_orthogonal_only_v1_4',
        'connections': {'expected': len(expected['service_points']), 'connected': sum(r['connected'] for r in connections),
                        'rows': connections, 'groups': group_rows},
        'manual_valves': {'expected': len(manual), 'installed': sum(r['installed'] for r in manual),
                          'rows': manual, 'native_packages': valve_rows, 'pipe_sizes': pipe_sizes},
        'horizontal_storage': storage_rows,
        'geometry': geometry, 'hierarchy': hierarchy, 'spacing': spacing, 'directions': directions,
        'operation_report': deepcopy(operation_report),
        'source_port_conflicts': detect_port_conflicts(frozen, expected),
        'scope': {'phase': 'P1', 'requirements_source': 'independently rederived from frozen sizing/layout',
                  'operation_ratio_gate': False, 'p2_material_connections': 'preserved, not executed',
                  'p3_robot_or_process_success': 'not claimed', 'coverage_quotas': False,
                  'source_port_conflicts': 'report only; no port movement, vent omission or single-port tee workaround'}}
