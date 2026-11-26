"""Find accessible operating space around equipment."""
from __future__ import annotations

import math

import numpy as np

from .building_equipment_checks import (
    EPS, _area, _overlap, _own_holes, _reachability, _side_strips,
    _structural_obstacles, _touches_reachable, _uncovered,
    equipment_footprints, maintenance_width_for, read_layout_rules,
    standing_board_rectangles,
)
from .process import RULES
from .rule_identity import band
from .recipe import CandidateRefused


KINDS = {'vessel', 'fixed_bed', 'exchanger', 'condenser'}
SIDES = {'x_min': (0, -1), 'x_max': (0, 1),
         'y_min': (1, -1), 'y_max': (1, 1)}
SKIRT_PENDING = 'Check the skirt outlet path during piping'


def _components(scene):
    return [c for c in scene['components']
            if not c.get('annotation_only')
            and (not c.get('visual_only') or c['kind'] == 'drive_reservation')]


def _box_hits(a, b):
    return all(min(a[i+3], b[i+3]) > max(a[i], b[i])+EPS for i in range(3))


def _contains(rect, point):
    return (rect[0]-EPS <= point[0] <= rect[2]+EPS
            and rect[1]-EPS <= point[1] <= rect[3]+EPS)


def _compact_path(points):
    result = []
    for point in points:
        point = [float(x) for x in point]
        if not result or math.dist(result[-1], point) > EPS:
            result.append(point)
    return result


def _segment_hits(a, b, box, radius=0.):
    """Conservative cylindrical corridor, with flat ends rather than end spheres."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    delta = b-a
    length = float(np.linalg.norm(delta))
    if length <= EPS:
        return all(box[i]+EPS < a[i] < box[i+3]-EPS for i in range(3))
    direction = delta/length
    transverse = radius*np.sqrt(np.maximum(0., 1.-direction*direction))
    lower = np.asarray(box[:3])-transverse+EPS
    upper = np.asarray(box[3:])+transverse-EPS
    lo, hi = 0., 1.
    for axis in range(3):
        if abs(delta[axis]) <= EPS:
            if not lower[axis] < a[axis] < upper[axis]:
                return False
            continue
        first, last = sorted(((lower[axis]-a[axis])/delta[axis],
                              (upper[axis]-a[axis])/delta[axis]))
        lo, hi = max(lo, first), min(hi, last)
        if hi-lo <= EPS:
            return False
    return hi-lo > EPS


def _path_hits(path, components, radius):
    # There is no owner-wide exemption: the path must also avoid the equipment's
    # own head, supports and drive.  An endpoint merely touching a face is allowed.
    return sorted({c['id'] for a, b in zip(path, path[1:]) for c in components
                   if _segment_hits(a, b, c['box'], radius)})


def _navigation(scene, rows, components, graph=None):
    from .maintenance_geometry import level_maintenance
    result = {}
    rules=read_layout_rules()
    units=graph['units'] if graph is not None else {}
    for level, z in enumerate(scene['floors']):
        boards = standing_board_rectangles(scene, level)
        structures = _structural_obstacles(components, scene['floors'], level)
        profiles={row['id']:level_maintenance(row,units.get(row['id'],{'kind':row['kind']}),
            scene,level,rules,planning=True) for row in rows if row['rect']}
        equipment = [{'id': row['id'], 'rect': profiles[row['id']]['rect']} for row in rows
                     if row['rect'] and level in row['occupied_levels']]
        holes = [{'id': f'opening_{level}_{i}', 'rect': rect}
                 for i, rect in enumerate(scene['holes'].get(level, scene['holes'].get(str(level), [])))]
        entry = scene.get('navigation_entries', {}).get(str(level))
        if entry is None:
            entry = scene.get('navigation_entries', {}).get(level)
        result[level] = {'boards': boards, 'structural': structures,'maintenance_profiles':profiles,
                         'nav': _reachability(boards, structures+equipment+holes, entry)}
    return result


def _maintenance_available(row, level, width, rows, scene, context, rules):
    rect = row['rect']
    outer = [rect[0]-width, rect[1]-width, rect[2]+width, rect[3]+width]
    own_holes, _, hole_errors = _own_holes(row, scene, level, rules)
    if hole_errors:
        return False, 'The registered opening does not match the equipment cross-section'
    if _area(_uncovered(outer, context['boards']+own_holes+[rect])) > EPS:
        return False, 'Maintenance clearance extends beyond the deck or is cut by an opening'
    others = [r for r in rows if r['id'] != row['id'] and r['rect']
              and level in r['occupied_levels']]
    profiles=context.get('maintenance_profiles',{})
    if any(_overlap(outer,profiles.get(r['id'],r)['rect']) for r in others):
        return False, 'Another unit occupies the maintenance clearance'
    if any(_overlap(outer, c['rect']) for c in context['structural']):
        return False, 'A column or stair occupies the maintenance clearance'
    return True, None


def _stances(row, level, width, context, components, floors, port):
    screen = RULES['geometry_screen']
    half_x, half_y = screen['station_width_m']/2, screen['station_depth_m']/2
    z, height = floors[level], screen['station_height_m']
    strips = _side_strips(row['rect'], width)
    xs, ys, reachable = context['nav']
    if not len(xs):
        return []
    candidates = []
    for ix, iy in zip(*np.where(reachable)):
        x, y = float(xs[ix]), float(ys[iy])
        box = [x-half_x, y-half_y, z, x+half_x, y+half_y, z+height]
        foot = [box[0], box[1], box[3], box[4]]
        if _area(_uncovered(foot, context['boards'])) > EPS:
            continue
        if any(_box_hits(box, c['box']) for c in components):
            continue
        if not _touches_reachable(box, context['nav']):
            continue
        for side, strip in strips.items():
            if (_contains(strip, [x, y]) and foot[0] >= strip[0]-EPS
                    and foot[1] >= strip[1]-EPS and foot[2] <= strip[2]+EPS
                    and foot[3] <= strip[3]+EPS):
                candidates.append((math.dist([x, y], port[:2]), side, box))
    candidates.sort(key=lambda c: (c[0], c[1], c[2][0], c[2][1]))
    return [(side, box) for _, side, box in candidates]


def _outside_position(row, side, stance, port, z, radius):
    axis, sign = SIDES[side]
    tangent = 1-axis
    rect = row['rect']
    point = [float(port[0]), float(port[1]), float(z)]
    point[axis] = rect[axis if sign < 0 else axis+2]+sign*radius
    # First keep the point nearest its real interface; then try the stance's
    # tangential coordinate. Both remain in the same equipment maintenance strip.
    tangent_range = [rect[tangent], rect[tangent+2]]
    source_at = min(tangent_range[1], max(tangent_range[0], point[tangent]))
    stance_at = min(tangent_range[1], max(tangent_range[0],
                         (stance[tangent]+stance[tangent+3])/2))
    for at in dict.fromkeys((source_at, stance_at)):
        candidate = point.copy()
        candidate[tangent] = at
        yield candidate


def _paths(start, normal, position, mode, radius):
    """A few explicit elbow alternatives, not a route solver or a length cutoff."""
    lead = (np.asarray(start)+np.asarray(normal)*radius).tolist()
    side = [position[0], position[1], lead[2]]
    down = [lead[0], lead[1], position[2]]
    options = ([start, lead, down, position], [start, lead, side, position])
    if mode == 'upper':
        options = options[::-1]
    for points in options:
        yield _compact_path(points)
    # Orthogonal alternatives can leave a side nozzle without cutting through
    # its own shell or a drive reservation on the opposite side.
    for first_axis in (0, 1):
        elbow = lead.copy()
        elbow[first_axis] = position[first_axis]
        yield _compact_path([start, lead, elbow, side, position])


def _skirt_at(row, port):
    return next((c for c in row['parts'] if c['kind'] == 'skirt'
                 and all(c['box'][i]-EPS <= port[i] <= c['box'][i+3]+EPS
                         for i in range(3))), None)


def _lower_mode(row, port, normal, floor_z):
    low, high = band()
    if port[2]-floor_z > high+EPS:
        return 'downward'
    shells = [c for c in row['parts'] if c['kind'] in KINDS or c['kind'] == 'head']
    under_body = any(c['box'][0]+EPS < port[0] < c['box'][3]-EPS
                     and c['box'][1]+EPS < port[1] < c['box'][4]-EPS
                     and c['box'][5] > port[2]+EPS for c in shells)
    if port[2]-floor_z < low-EPS or normal[2] < -EPS or under_body:
        return 'horizontal'
    return 'direct'


def upper_operation_levels(floors, interface_z):
    """Fix nearest upper-operation floors from geometry before seeing obstacles.

    Distance is to each existing reference height band, not a new acceptance
    limit. Equally near floors may both be tried. Blockage cannot make a remote
    ground floor become an upper operating surface.
    """
    low, high = band()
    distances = [max(z+low-interface_z, interface_z-z-high, 0.) for z in floors]
    nearest = min(distances)
    return [level for level, distance in enumerate(distances)
            if abs(distance-nearest) <= EPS]


def _potential_floors(scene):
    """Include permitted but unbuilt floors so omission cannot justify ground."""
    shell = scene.get('shell', {})
    limit = shell.get('platform_limit', shell.get('platforms', len(scene['floors'])-1))
    height = shell.get('storey_height_m')
    if height is None:
        # Fixtures without a shell still expose their actual floor choices;
        # production shell records always provide the permitted uniform height.
        return list(scene['floors'])
    return [scene['floors'][0]+i*height for i in range(limit+1)]


def _find_end(unit, row, upper, rows, scene, contexts, components, rules):
    ports = [(name, p) for name, p in unit['ports'].items() if 'pos' in p]
    if not ports:
        raise CandidateRefused(f'BL5 {unit["id"]}A physical port is required to reserve operation space')
    end_z = (max if upper else min)(p['pos'][2] for _, p in ports)
    ports = sorted((name, p) for name, p in ports if abs(p['pos'][2]-end_z) <= EPS)
    from .rule_identity import identity
    low, high = band()
    floors = scene['floors']
    if upper and identity('equipment_top_access') == 'record':
        name,port=ports[0]
        level=min(range(len(floors)),key=lambda i:abs(port['pos'][2]-floors[i]))
        return {'port':name,'related_ports':[n for n,_ in ports], 'interface':{'position':port['pos'],'normal':port.get('normal')},
                'position':port['pos'],'level':level,'status':'record','rule_identity':'record',
                'reason':'Top equipment port. Record operating conditions with the installed valve.','stance_box':None,'external_path':[]}

    levels = upper_operation_levels(_potential_floors(scene), end_z) if upper else list(range(len(floors)))
    from .maintenance_geometry import level_maintenance
    whole_row=row
    failures = []
    for level in levels:
        if level >= len(floors):
            failures.append({'level': level, 'reason': 'The nearest upper operating level is not built. Revise the building layout.'})
            continue
        context = contexts[level]
        profile=level_maintenance(whole_row,unit,scene,level,rules,planning=True)
        row={**whole_row,'rect':profile['rect']}
        width=profile['width_m']
        valid, cause = _maintenance_available(row, level, width, rows, scene, context, rules)
        if not valid:
            failures.append({'level': level, 'reason': cause})
            continue
        for name, port in ports:
            source = [float(v) for v in port['pos']]
            raw_normal = [float(v) for v in port.get('normal', [0., 0., 0.])]
            norm = math.sqrt(sum(v*v for v in raw_normal))
            if norm <= EPS:
                raise CandidateRefused(f'BL5 {unit["id"]}.{name}No physical port direction')
            normal = [v/norm for v in raw_normal]
            radius = float(port.get('dn', 50))/2000
            mode = 'upper' if upper else _lower_mode(row, source, normal, floors[level])
            skirt = None if upper else _skirt_at(row, source)
            if skirt is not None:
                mode = 'horizontal'
            # Upper operation is reserved at a conventional standing height;
            # its bare nozzle stays at the original position and must connect by
            # an actually clear path. Only lower operations retain low-head use.
            target_z = min(source[2], floors[level]+high)
            if upper:
                target_z = max(floors[level]+low, target_z)
            elif target_z < floors[level]-EPS:
                target_z = floors[level]+low
            candidates = _stances(row, level, width, context, components, floors, source)
            for side, stance in candidates:
                for position in _outside_position(row, side, stance, source, target_z, radius):
                    centre = [(stance[0]+stance[3])/2, (stance[1]+stance[4])/2]
                    reach = math.dist(centre, position[:2])
                    if reach > RULES['construction_choices']['stance_reach_proxy_m']+EPS:
                        continue
                    if not _contains(_side_strips(row['rect'], width)[side], position):
                        continue
                    start = source.copy()
                    path_normal = normal
                    if skirt is not None:
                        axis, sign = SIDES[side]
                        start[axis] = skirt['box'][axis if sign < 0 else axis+3]
                        path_normal = [0., 0., 0.]
                        path_normal[axis] = float(sign)
                    for path in _paths(start, path_normal, position, mode, radius):
                        if _path_hits(path, components, radius):
                            continue
                        result = {
                            'port': name,
                            'interface': {'position': source, 'normal': raw_normal},
                            'related_ports': [n for n, _ in ports],
                            'level': level, 'position': position,
                            'maintenance_geometry':profile,
                            'geometric_candidate_levels': list(levels),
                            'stance_box': stance, 'side': side, 'external_path': path,
                            'corridor_radius_m': radius,
                            'horizontal_reach_m': reach,
                            'interface_height_above_floor_m': source[2]-floors[level],
                            'operation_height_above_floor_m': position[2]-floors[level],
                            'operation_reference_height_m': [low, high],
                            'low_head_operation': position[2]-floors[level] < low-EPS,
                            'path_length_m': sum(math.dist(a, b) for a, b in zip(path, path[1:])),
                            'scope': 'Reserve equipment operation and outlet space. Verify again during piping.',
                        }
                        if not upper:
                            pending = ({'required': True, 'status': 'pending_pipe_stage',
                                        'reason': SKIRT_PENDING} if skirt else None)
                            result.update(mode=mode, skirt_internal_status=pending,
                                          internal_path_status=SKIRT_PENDING if skirt else None)
                            if skirt:
                                result['external_path_start'] = 'Outer boundary of the physical solid skirt. The internal route is unverified.'
                                result['internal_exit_boundary'] = start
                                result['skirt_component'] = skirt['id']
                        return result
            failures.append({'level': level, 'port': name,
                             'reason': 'No reachable standing area on the deck, or the port path is blocked by equipment, its drive or framing'})
    end = 'upper operating surface' if upper else 'lower external outlet space'
    detail = '; '.join(f' level{r["level"]}: {r["reason"]}' for r in failures)
    fixed = f'Geometry fixes the nearest upper candidate level{levels}. A more distant level cannot substitute. ' if upper else ''
    name,port=ports[0]
    level=min(range(len(floors)),key=lambda i:abs(port['pos'][2]-floors[i]))
    return {'port':name,'related_ports':[n for n,_ in ports], 'interface':{'position':port['pos'],'normal':port.get('normal')},
            'position':port['pos'],'level':level,'status':'challenge','rule_identity':identity('stance'),
            'reason':f'No direct lower standing area. Check the physical connection during piping: {detail}','stance_box':None,'external_path':[]}



def lower_hint(scene,equipment,port):
    records=scene.get('operation_reservations',[])
    for row in records.values() if isinstance(records,dict) else records:
        hint=row.get('lower') or {}
        if row.get('equipment',row.get('id'))==equipment and hint.get('stance_box') and (
                port==hint.get('port') or port in hint.get('related_ports',[])):
            return hint
    return None


def plan_operations(graph, scene):
    """Return per-device reservations from completed physical building geometry.

    The function mutates neither graph nor scene. Failure is an explicit candidate
    refusal so that the caller may try another complete room placement/shell.
    """
    rules = read_layout_rules()
    components = _components(scene)
    rows = equipment_footprints(graph, scene)
    by_id = {row['id']: row for row in rows}
    contexts = _navigation(scene, rows, components, graph)
    result = []
    for uid, unit in graph['units'].items():
        if unit['kind'] not in KINDS or unit.get('form', unit.get('orientation', 'vertical')) != 'vertical':
            continue
        row = by_id.get(uid)
        if row is None or row['rect'] is None:
            raise CandidateRefused(f'BL5 {uid}Missing measurable equipment body')
        upper = _find_end(unit, row, True, rows, scene, contexts, components, rules)
        lower = _find_end(unit, row, False, rows, scene, contexts, components, rules)
        result.append({'equipment': uid, 'upper': upper, 'lower': lower})
    return result
