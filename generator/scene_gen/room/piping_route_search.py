"""Find a pipe route through the available space."""
from __future__ import annotations

import math

import numpy as np

from .evaluation_common import require_environment
from .platform_coverage import _uncovered
from .piping_port_contacts import flat_pipe_clears, outward_pipe_bounds


EPS = 1e-8  # Numerical equality, not a construction allowance.
_SLABS = {'floor', 'ground', 'walk_surface'}


class RouteUnavailable(ValueError):
    """Explicit invalid geometry, blocked endpoint, no route or search limit."""

    def __init__(self, code, detail, **evidence):
        self.code = code
        self.evidence = {'code': code, 'detail': detail, **evidence}
        super().__init__(f'{code}: {detail}')


def _point(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RouteUnavailable('INVALID_INPUT', label+'must be a finite three-dimensional point') from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise RouteUnavailable('INVALID_INPUT', label+'must be a finite three-dimensional point')
    return result


def _positive(value, label, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise RouteUnavailable('INVALID_INPUT', label+'must be finite and '+('zero or positive' if zero else 'strictly positive'))
    return float(value)


def _box(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RouteUnavailable('INVALID_INPUT', label+'incomplete or nonfinite bounding box') from exc
    if result.shape == (2, 3):
        result = result.ravel()
    if result.shape != (6,) or not np.isfinite(result).all() or np.any(result[3:] < result[:3]):
        raise RouteUnavailable('INVALID_INPUT', label+'incomplete or nonfinite bounding box')
    return result


def _segments_distance(a, b, p, q):
    """Exact closest distance of finite centreline segments, including points."""
    u, v, w = b-a, q-p, a-p
    aa, bb, cc, dd, ee = float(u@u), float(u@v), float(v@v), float(u@w), float(v@w)
    if aa <= 1e-24 and cc <= 1e-24:
        return float(np.linalg.norm(a-p))
    if aa <= 1e-24:
        return float(np.linalg.norm(a-p-np.clip(ee/cc, 0, 1)*v))
    if cc <= 1e-24:
        return float(np.linalg.norm(a+np.clip(-dd/aa, 0, 1)*u-p))
    candidates = [(0., np.clip(ee/cc, 0, 1)), (1., np.clip((ee+bb)/cc, 0, 1)),
                  (np.clip(-dd/aa, 0, 1), 0.), (np.clip((bb-dd)/aa, 0, 1), 1.)]
    denominator = aa*cc-bb*bb
    if denominator > 1e-15*aa*cc:
        t, k = (bb*ee-cc*dd)/denominator, (aa*ee-bb*dd)/denominator
        if 0 <= t <= 1 and 0 <= k <= 1:
            candidates.append((t, k))
    return min(float(np.linalg.norm(a+t*u-p-k*v)) for t, k in candidates)


def _box_hits(a, b, boxes, pad):
    """Continuous segment vs expanded actual boxes, not just grid nodes."""
    if not len(boxes):
        return np.zeros(0, dtype=bool)
    lo, hi = boxes[:, :3]-pad, boxes[:, 3:]+pad
    lower, upper = np.zeros(len(boxes)), np.ones(len(boxes))
    possible = np.ones(len(boxes), dtype=bool)
    delta = b-a
    for axis in range(3):
        if abs(delta[axis]) <= 1e-14:
            possible &= (a[axis] > lo[:, axis]+EPS) & (a[axis] < hi[:, axis]-EPS)
        else:
            first = (lo[:, axis]-a[axis])/delta[axis]
            second = (hi[:, axis]-a[axis])/delta[axis]
            lower = np.maximum(lower, np.minimum(first, second))
            upper = np.minimum(upper, np.maximum(first, second))
    return possible & (upper >= lower) & (upper > EPS) & (lower < 1-EPS)


def _opening_table(openings, floors):
    result = {i: [] for i in range(len(floors))}
    if isinstance(openings, dict):
        records = [{'level': level, 'rect': rect} for level, rects in openings.items() for rect in rects]
    else:
        records = list(openings)
    for record in records:
        if not isinstance(record, dict):
            raise RouteUnavailable('INVALID_OPENING', 'An opening needs a level or elevation and a rectangle')
        level = record.get('level')
        if level is None and 'z' in record:
            try:
                level = next(i for i, z in enumerate(floors) if abs(z-float(record['z'])) <= EPS)
            except (StopIteration, ValueError, TypeError) as exc:
                raise RouteUnavailable('INVALID_OPENING', 'Opening z does not match a physical floor_z') from exc
        try:
            converted = int(level)
        except (ValueError, TypeError, OverflowError) as exc:
            raise RouteUnavailable('INVALID_OPENING', 'Opening lacks a floor index') from exc
        if isinstance(level, bool) or str(converted) != str(level) or converted not in result:
            raise RouteUnavailable('INVALID_OPENING', 'Opening level differs from physical floor_z')
        try:
            rect = np.asarray(record['rect'], dtype=float)
        except (KeyError, ValueError, TypeError) as exc:
            raise RouteUnavailable('INVALID_OPENING', 'Opening lacks a complete rectangle') from exc
        if rect.shape != (4,) or not np.isfinite(rect).all() or np.any(rect[2:] <= rect[:2]):
            raise RouteUnavailable('INVALID_OPENING', 'Opening rect must be a finite positive-area rectangle')
        result[converted].append(rect.tolist())
    return result


def _in_holes(a, b, z_low, z_high, radius, holes, vertical_pad=None):
    """Tube's outer capsule footprint through an entire real slab interval."""
    if not holes:
        return False
    extent = radius if vertical_pad is None else vertical_pad
    delta = b-a
    if abs(delta[2]) <= 1e-14:
        if a[2] < z_low-extent-EPS or a[2] > z_high+extent+EPS:
            return True
        one, two = a, b
    else:
        limits = sorted(((z_low-extent-a[2])/delta[2], (z_high+extent-a[2])/delta[2]))
        low, high = max(0., limits[0]), min(1., limits[1])
        if high < low-EPS:
            return True
        one, two = a+low*delta, a+high*delta
    rect = [min(one[0], two[0])-radius, min(one[1], two[1])-radius,
            max(one[0], two[0])+radius, max(one[1], two[1])+radius]
    missing = _uncovered(rect, holes)
    return not any(part[2]-part[0] > EPS and part[3]-part[1] > EPS for part in missing)


def _prepare(components, floors):
    result = []
    for index, source in enumerate(components):
        if not isinstance(source, dict):
            raise RouteUnavailable('INVALID_COMPONENT', f'Obstacle {index} is not a physical obstacle record')
        name = source.get('id', f'component_{index}')
        box = _box(source.get('box'), str(name))
        item = {'id': name, 'box': box, 'kind': source.get('kind'), 'level': None, 'round': None,
                'planes': np.asarray(source['collision_planes']) if source.get('collision_planes') else None}
        item['physical_box'] = _box(source['physical_box'],str(name)) if 'physical_box' in source else box
        if source.get('kind') in _SLABS:
            matches = [i for i, z in enumerate(floors) if abs(box[5]-z) <= EPS]
            if len(matches) == 1:
                item['level'] = matches[0]
        if source.get('shape') in ('cylinder', 'capsule'):
            try:
                ends = source['ends']
                if len(ends) != 2:
                    raise ValueError('ends')
                first, second = _point(ends[0], str(name)+'.ends[0]'), _point(ends[1], str(name)+'.ends[1]')
                radius = _positive(source['radius'], str(name)+'.radius')
            except (KeyError, TypeError, ValueError) as exc:
                if isinstance(exc, RouteUnavailable):
                    raise
                raise RouteUnavailable('INVALID_COMPONENT', str(name)+'Cylinder lacks physical ends or radius') from exc
            item['round'] = (first, second, radius)
        result.append(item)
    return result


def _simplify(points):
    from .piping_elbows import is_axis_aligned
    result = []
    for point in points:
        point = np.asarray(point, dtype=float)
        if result and np.max(abs(point-result[-1])) <= EPS:
            continue
        while len(result) >= 2:
            one, two = result[-1]-result[-2], point-result[-1]
            if (is_axis_aligned(result[-2],result[-1]) and is_axis_aligned(result[-1],point)
                and is_axis_aligned(result[-2],point) and one@two>=0):
                result.pop()
            else:
                break
        result.append(point)
    return result


def simple_route(a, b, components, radius, bounds, floor_z, openings=(), clearance=.05, max_nodes=60000, straight_only=False,
                 priority='bends_then_length',start_direction=None,end_direction=None,
                 start_straight_m=0.,end_straight_m=0.,diagnostics=None,complete_bend_planes=False,side_discharge=False,
                 port_contacts=(), guide=None):
    """Return checked points, or raise ``RouteUnavailable`` with evidence.

    ``bounds`` is [xmin,ymin,zmin,xmax,ymax,zmax] (or [lo3,hi3]) and
    contains the tube's full outer envelope. ``floor_z`` is the actual floor
    elevation list. Openings are [{level,rect}] / [{z,rect}] or {level:[rect]}.
    Registered holes use the actual pipe radius, not an invented extra edge
    allowance. Other obstacles require ``radius + clearance``. A registered
    hole is permission to plan, not proof the final real slab has been cut.

    Callers may remove the exact adjacent endpoint fitting component before
    this call and independently check that connection. This function removes
    nothing by owner, appearance/contact flag or DN. Registered equipment
    port contacts only remove duplicate clearance where the physical flat
    straight is proven separate from a named piece of that exact joint.
    """
    require_environment()
    from .piping_elbows import plan_polyline,is_axis_aligned,ElbowUnavailable
    a, b = _point(a, 'a'), _point(b, 'b')
    radius = _positive(radius, 'radius')
    clearance = _positive(clearance, 'clearance', zero=True)
    start_straight_m = _positive(start_straight_m, 'start_straight_m', zero=True)
    end_straight_m = _positive(end_straight_m, 'end_straight_m', zero=True)
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise RouteUnavailable('INVALID_INPUT', 'diagnostics must be a writable dictionary or None')
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.clear()
    diagnostics.update(feasibility_rejections={}, obstruction_examples=[])
    if priority not in ('bends_then_length','length_then_bends'):
        raise RouteUnavailable('INVALID_PRIORITY','Choose fewest bends or shortest route. Both require orthogonal straights and installable elbows.')
    def constraint(value,label):
        if value is None:return None
        direction=_point(value,label);norm=float(np.linalg.norm(direction))
        if norm<=EPS or np.count_nonzero(abs(direction/norm)>EPS)!=1:
            raise RouteUnavailable('PG17_UNSUPPORTED_INTERFACE_DIRECTION',label+'must align with an existing building axis')
        return direction/norm
    start_direction=constraint(start_direction,'start_direction');end_direction=constraint(end_direction,'end_direction')
    endpoint_contacts=[]
    for contact in port_contacts:
        point=_point(contact['position'],'mating face')
        if any(np.max(abs(point-end))<=EPS for end in (a,b)):
            axis=constraint(contact['normal'],'outward mating face normal')
            endpoint_contacts.append((point,axis,set(contact['contact_ids'])))
    boundary = _box(bounds, 'bounds')
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise RouteUnavailable('INVALID_INPUT', 'max_nodes must be a positive integer')
    try:
        floors = [float(z) for z in floor_z]
    except (ValueError, TypeError) as exc:
        raise RouteUnavailable('INVALID_INPUT', 'floor_z must list finite physical floor elevations') from exc
    if not all(math.isfinite(z) for z in floors) or len(set(floors)) != len(floors) or floors != sorted(floors):
        raise RouteUnavailable('INVALID_INPUT', 'floor_z must be finite and strictly increasing')
    holes = _opening_table(openings, floors)
    from .grating import grating
    obstacles = _prepare([c for c in components if not grating(c)], floors)
    boxes = np.asarray([c['box'] for c in obstacles]).reshape(-1, 6)
    low, high = boundary[:3]+radius, boundary[3:]-radius
    if np.any(high < low-EPS):
        raise RouteUnavailable('BOUNDARY_TOO_SMALL', 'The permitted region cannot fit the physical pipe OD')
    for name, point in (('a', a), ('b', b)):
        if np.any(point < low-EPS) or np.any(point > high+EPS):
            raise RouteUnavailable('ENDPOINT_OUTSIDE_BOUNDS', name+' pipe outside surface exceeds the dynamic permitted region', endpoint=point.tolist(), bounds=boundary.tolist(), radius_m=radius)

    def obstruction(first, second,tube_radius=None, *, flat_run=True,point_probe=False,joined_arcs=()):
        tube_radius=radius if tube_radius is None else tube_radius
        contact_bounds=list(joined_arcs)
        if flat_run:
            for point,axis,ids in endpoint_contacts:
                extent=outward_pipe_bounds(first,second,tube_radius,point,axis,point_probe=point_probe)
                if extent is None and point_probe:
                    # The tentative far tip of a direct axial lead can lie in
                    # the same redundant endpoint buffer. The whole proposed
                    # flat pipe must be separate before this tip is let through.
                    extent=outward_pipe_bounds(a,b,tube_radius,point,axis)
                contact_bounds.append((extent,ids))
        for level, z in enumerate(floors[:1]):
            if min(first[2], second[2])-tube_radius < z-EPS and max(first[2], second[2])+tube_radius > z+EPS:
                if not _in_holes(first, second, z, z, tube_radius, holes[level]):
                    return f'unregistered_or_too_small_opening_level_{level}'
        for index in np.flatnonzero(_box_hits(first, second, boxes, tube_radius+clearance)):
            obstacle = obstacles[index]
            if any(obstacle['id'] in ids and flat_pipe_clears(bounds,obstacle['physical_box'])
                   for bounds,ids in contact_bounds):
                continue
            if obstacle['level'] is not None and _in_holes(first, second, obstacle['box'][2], obstacle['box'][5], tube_radius, holes[obstacle['level']], tube_radius+clearance):
                continue
            rounded = obstacle['round']
            if rounded is not None and _segments_distance(first, second, rounded[0], rounded[1]) >= tube_radius+rounded[2]+clearance-EPS:
                continue
            planes=obstacle['planes']
            if planes is not None:
                normals,offset=planes[:,:3],planes[:,3]
                distances=np.minimum(normals@first,normals@second)+offset
                if np.any(distances>=tube_radius+clearance-EPS):
                    continue
            return str(obstacle['id'])
        return None

    for name, point in (('a', a), ('b', b)):
        blocked = obstruction(point, point,point_probe=True)
        if blocked is not None:
            raise RouteUnavailable('ENDPOINT_BLOCKED', name+'blocked by a physical obstacle or an unregistered floor section', endpoint=point.tolist(), obstacle=blocked)

    def direction_code(vector):
        axis=int(np.argmax(abs(vector)));return 2*axis+int(vector[axis]>0)

    def arc_clear(elbow, first_corner, last_corner):
        pts=[np.asarray(p) for p in elbow['centerline_points']]
        angle=math.radians(elbow['angle_degrees'])
        half_angle=angle/(2*(len(pts)-1))
        sagitta=elbow['centerline_radius_m']*(1-math.cos(half_angle))
        linked=[]
        for point,axis,ids in endpoint_contacts:
            for corner,tangent,face in ((first_corner,np.asarray(elbow['tangent_in']),elbow['start']),
                                        (last_corner,-np.asarray(elbow['tangent_out']),elbow['end'])):
                if (elbow['corner_index']==corner and np.max(abs(tangent-axis))<=EPS
                        and outward_pipe_bounds(point,face,radius,point,axis,point_probe=True) is not None):
                    linked.append(ids)
        # The mesh sector is the convex hull of its two circular end rings.
        # Use their true transverse extents only for the adjoining equipment
        # joint. Every other obstacle keeps the ordinary swept-envelope check.
        radial=np.asarray(elbow['radial_start']);incoming=np.asarray(elbow['tangent_in'])
        if linked:
            tangents=[-radial*math.sin(t)+incoming*math.cos(t) for t in np.linspace(0.,angle,len(pts))]
            extents=[radius*np.sqrt(np.maximum(0.,1.-t*t)) for t in tangents]
        for i,(first,second) in enumerate(zip(pts,pts[1:])):
            joined=[]
            if linked:
                bounds=(np.minimum(first-extents[i],second-extents[i+1]),
                        np.maximum(first+extents[i],second+extents[i+1]))
                joined=[(bounds,ids) for ids in linked]
            if obstruction(first,second,radius+sagitta,flat_run=False,joined_arcs=joined) is not None:
                return False
        return True

    def reject(code, **evidence):
        counts=diagnostics['feasibility_rejections']
        counts[code]=counts.get(code,0)+1
        if evidence and len(diagnostics['obstruction_examples'])<8:
            diagnostics['obstruction_examples'].append({'code':code,**evidence})
        return None

    def feasible(points):
        points=_simplify(points)
        if len(points)<2:return reject('EMPTY_CANDIDATE')
        if not all(is_axis_aligned(x,y) for x,y in zip(points,points[1:])):return reject('NON_ORTHOGONAL_CANDIDATE')
        if start_direction is not None and direction_code(points[1]-points[0])!=direction_code(start_direction):return reject('START_DIRECTION')
        if end_direction is not None and direction_code(points[-1]-points[-2])!=direction_code(end_direction):return reject('END_DIRECTION')
        try:plan=plan_polyline(points,radius)
        except ElbowUnavailable as exc:return reject(exc.code)
        first=next((r for r in plan['runs'] if r['segment_index']==0),None)
        final=next((r for r in reversed(plan['runs']) if r['segment_index']==len(plan['points'])-2),None)
        first_length=math.dist(first['start'],first['end']) if first else 0.
        final_length=math.dist(final['start'],final['end']) if final else 0.
        if first_length+EPS<start_straight_m:return reject('START_STRAIGHT_RESERVATION')
        if final_length+EPS<end_straight_m:return reject('END_STRAIGHT_RESERVATION')
        for run in plan['runs']:
            endpoint_run=run['segment_index'] in (0,len(plan['points'])-2)
            blocked=obstruction(np.asarray(run['start']),np.asarray(run['end']),flat_run=endpoint_run)
            if blocked is not None:return reject('RUN_OBSTRUCTED',obstacle=blocked,ends=[run['start'],run['end']])
        if plan['elbows']:
            first_corner=plan['original_indices'][1];last_corner=plan['original_indices'][-2]
            if not all(arc_clear(e,first_corner,last_corner) for e in plan['elbows']):
                return reject('ELBOW_OBSTRUCTED')
        length=sum(math.dist(r['start'],r['end']) for r in plan['runs'])+sum(
            e['centerline_radius_m']*math.radians(e['angle_degrees']) for e in plan['elbows'])
        bends=len(plan['elbows'])
        cost=(bends,length) if priority=='bends_then_length' else (length,bends)
        return cost,points

    from .piping_plane_search import search_whole_segments, METHOD
    if guide is not None:
        guided=[_point(point,'specified route point') for point in guide]
        if (len(guided)<2 or not np.allclose(guided[0],a,atol=EPS,rtol=0.) or
                not np.allclose(guided[-1],b,atol=EPS,rtol=0.) or
                any(np.any(point < low-EPS) or np.any(point > high+EPS) for point in guided)):
            raise RouteUnavailable('GUIDE_ENDPOINT_OR_BOUNDS','Supplied route misses a real endpoint or leaves the permitted region')
        checked=feasible(guided)
        if checked is None:
            raise RouteUnavailable('GUIDE_OBSTRUCTED','Supplied route failed physical straight, elbow or clearance checks',**diagnostics)
        diagnostics.update(method=METHOD,selected={'family':'checked_guide'})
        return [p.tolist() for p in checked[1]]
    if np.max(abs(a-b)) <= EPS:
        if start_straight_m>EPS or end_straight_m>EPS:
            raise RouteUnavailable('EMPTY_STRAIGHT_RESERVATION','A shared endpoint cannot fit the required straight length')
        diagnostics.update(method=METHOD,generated_candidates=0,tested_candidates=0,
                           selected={'family':'same_endpoint','bends':0,'length_m':0.})
        return [a.tolist()]
    direct=feasible([a,b]) if is_axis_aligned(a,b) else None
    if direct:
        diagnostics.update(method=METHOD,discrete_grid_search=False,global_optimality_claim=False,
            generated_candidates=1,tested_candidates=1,feasible_candidates=1,
            candidate_budget=min(max_nodes,2048),original_max_nodes=max_nodes,
            selected={'family':'whole_straight','planes':[],'bends':0,'length_m':math.dist(a,b),
                      'guide':[a.tolist(),b.tolist()], 'endpoint_straights_m':{'start':math.dist(a,b),'end':math.dist(a,b)},
                      'priority':priority})
        return [p.tolist() for p in direct[1]]
    if straight_only:
        code='PG17_NON_ORTHOGONAL_STRAIGHT' if not is_axis_aligned(a,b) else 'STRAIGHT_BLOCKED'
        raise RouteUnavailable(code,'The specified straight must follow a building axis and pass solid and end-direction checks',
                               start=a.tolist(),end=b.tolist(),obstacle=obstruction(a,b))
    return search_whole_segments(a,b,low,high,obstacles,holes,radius,clearance,feasible,
        priority=priority,max_nodes=max_nodes,start_direction=start_direction,end_direction=end_direction,
        start_straight_m=start_straight_m,end_straight_m=end_straight_m,diagnostics=diagnostics,
        complete_bend_planes=complete_bend_planes,side_discharge=side_discharge)
