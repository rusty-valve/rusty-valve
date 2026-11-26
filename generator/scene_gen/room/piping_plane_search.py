"""Try alternate pipe paths and bend positions to avoid collisions."""
from __future__ import annotations

from collections import Counter
import itertools
import math
import time

import numpy as np


EPS = 1e-8
MAX_COMPLETE_CANDIDATES = 2048  # Algorithm resource budget, not a rule value.
MAX_PLANES_PER_AXIS = 12
MAX_ENDPOINT_LEADS = 3
METHOD = 'whole_segment_moving_planes_v1'


def _path_between(first, second, order):
    current = first.copy()
    result = [current]
    for axis in order:
        current = current.copy()
        current[axis] = second[axis]
        result.append(current)
    return result


def _plane_values(a, b, low, high, obstacles, holes, radius, clearance, R, complete_bend_planes=False):
    """Move whole turning planes to exact supplied faces, not 0.15m nodes."""
    values = []
    pad = radius + clearance
    for axis in range(3):
        records = [(a[axis], 'endpoint_a'), (b[axis], 'endpoint_b'),
                   ((a[axis]+b[axis])/2, 'endpoint_meeting'),
                   (low[axis], 'boundary_low'), (high[axis], 'boundary_high')]
        if axis < 2:
            for level, rectangles in holes.items():
                for rect in rectangles:
                    records.append(((rect[axis]+rect[axis+2])/2,
                                    f'registered_opening_level_{level}'))
        for obstacle in obstacles:
            box = obstacle['box']
            for extra in (pad, pad+R):
                records.extend(((box[axis]-extra, 'obstacle_low:'+str(obstacle['id'])),
                                (box[axis+3]+extra, 'obstacle_high:'+str(obstacle['id']))))
        # Prefer nearby moving planes.  Preserve distinct physical coordinates;
        # numerical equality is the existing metre EPS, not a spatial grid.
        ordered = sorted(records, key=lambda row: (
            min(abs(row[0]-a[axis]), abs(row[0]-b[axis])),
            row[0], row[1]))
        # Preserve bend-sized alternatives before nearby sampled mesh faces
        # fill the finite list. Nearly equal endpoint heights still need a
        # complete up/across/down route with two real bend radii.
        span=2*R+2*radius+clearance
        required=[(a[axis],'endpoint_a'),(b[axis],'endpoint_b'),
                  ((a[axis]+b[axis])/2,'endpoint_meeting'),
                  (min(a[axis],b[axis])-span,'complete_bend_below_endpoints'),
                  (max(a[axis],b[axis])+span,'complete_bend_above_endpoints')]
        if complete_bend_planes:ordered=required+ordered
        kept = []
        for value, source in ordered:
            value = float(value)
            if value < low[axis]-EPS or value > high[axis]+EPS:
                continue
            if any(abs(value-row['coordinate_m']) <= EPS for row in kept):
                continue
            kept.append({'axis': axis, 'coordinate_m': value, 'source': source})
            if len(kept) == MAX_PLANES_PER_AXIS:
                break
        values.append(kept)
    return values


def _anchors(point, direction, reserve, planes, low, high, R, end=False):
    if direction is None:
        return [point.copy()]
    outward = -direction if end else direction
    axis = int(np.argmax(abs(outward)))
    minimum = reserve + R
    distances = [minimum, reserve+2*R]
    distances.extend((row['coordinate_m']-point[axis])/outward[axis]
                     for row in planes[axis])
    # A full lead may grow to a plane without changing the frozen endpoint.
    distances = sorted(d for d in distances if d >= minimum-EPS)
    result = []
    for distance in distances:
        p = point+outward*distance
        if np.any(p < low-EPS) or np.any(p > high+EPS):
            continue
        if any(np.max(abs(p-q)) <= EPS for q in result):
            continue
        result.append(p)
        if len(result) == MAX_ENDPOINT_LEADS:
            break
    return result


def _round_robin(generators):
    active = [iter(g) for g in generators]
    while active:
        remaining = []
        for generator in active:
            try:
                yield next(generator)
                remaining.append(generator)
            except StopIteration:
                pass
        active = remaining


def whole_segment_candidates(a, b, low, high, obstacles, holes, radius,
                             clearance, start_direction=None, end_direction=None,
                             start_straight_m=0., end_straight_m=0.,complete_bend_planes=False):
    """Yield complete sharp orthogonal guides plus explainable plane records.

    Families: complete axis-order connection; one moving turning plane;
    two perpendicular moving planes; an actual registered vertical corridor.
    Endpoint lead variants are interleaved, never tried by room or seed name.
    """
    from .piping_elbows import coarse_radius
    R = coarse_radius(radius)
    planes = _plane_values(a, b, low, high, obstacles, holes, radius, clearance, R,complete_bend_planes)
    starts = _anchors(a, start_direction, start_straight_m, planes, low, high, R)
    ends = _anchors(b, end_direction, end_straight_m, planes, low, high, R, end=True)
    anchors = list(itertools.product(starts, ends))

    def full(first, middle, last):
        return [a, first, *middle, last, b]

    def direct_orders():
        # Also keep original endpoints: a direct compatible run needs no elbow
        # and must not be ruled out merely because a lead anchor is out of bounds.
        pairs = [(a, b), *anchors]
        for first, last in pairs:
            for order in itertools.permutations(range(3)):
                yield full(first, _path_between(first, last, order), last), {
                    'family': 'whole_axis_order', 'planes': []}

    def single_planes():
        for rank in range(MAX_PLANES_PER_AXIS):
            for first, last in anchors:
                for axis in range(3):
                    if rank >= len(planes[axis]):
                        continue
                    plane = planes[axis][rank]
                    p, q = first.copy(), last.copy()
                    p[axis] = q[axis] = plane['coordinate_m']
                    for order in itertools.permutations([i for i in range(3) if i != axis]):
                        yield full(first, [p, *_path_between(p, q, order), q], last), {
                            'family': 'moving_one_plane', 'planes': [plane]}

    def double_planes():
        # Nearby face pairs first; moving a plane need not add a bend.
        for rank_sum in range(2*MAX_PLANES_PER_AXIS-1):
            for first, last in anchors:
                for i, j in itertools.permutations(range(3), 2):
                    k = 3-i-j
                    for ri in range(MAX_PLANES_PER_AXIS):
                        rj = rank_sum-ri
                        if not (0 <= rj < len(planes[j]) and ri < len(planes[i])):
                            continue
                        pi, pj = planes[i][ri], planes[j][rj]
                        p = first.copy(); p[i] = pi['coordinate_m']
                        q = p.copy(); q[j] = pj['coordinate_m']
                        r = q.copy(); r[k] = last[k]
                        for order in ((i, j), (j, i)):
                            yield full(first, [p, q, r, *_path_between(r, last, order)], last), {
                                'family': 'moving_two_planes', 'planes': [pi, pj]}

    def opening_corridors():
        centres = sorted(set(((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)
                             for rectangles in holes.values() for rect in rectangles))
        for x, y in centres:
            for first, last in anchors:
                p = np.asarray([x, y, first[2]])
                q = np.asarray([x, y, last[2]])
                for one in ((0, 1), (1, 0)):
                    for two in ((0, 1), (1, 0)):
                        yield full(first, [*_path_between(first, p, one), q,
                                           *_path_between(q, last, two)], last), {
                            'family': 'registered_vertical_corridor',
                            'planes': [{'axis': 0, 'coordinate_m': x, 'source': 'opening_centre'},
                                       {'axis': 1, 'coordinate_m': y, 'source': 'opening_centre'}]}

    yield from _round_robin((direct_orders(), single_planes(), double_planes(), opening_corridors()))


def search_whole_segments(a, b, low, high, obstacles, holes, radius, clearance,
                          feasible, *, priority, max_nodes, start_direction=None,
                          end_direction=None, start_straight_m=0., end_straight_m=0.,
                          diagnostics=None,complete_bend_planes=False,side_discharge=False):
    """Evaluate at most min(original budget,2048) full candidates, no A*."""
    from .piping_elbows import plan_polyline, ElbowUnavailable
    from .piping_route_search import _simplify, RouteUnavailable
    started = time.monotonic()
    report = diagnostics if diagnostics is not None else {}
    budget = min(max_nodes, MAX_COMPLETE_CANDIDATES)
    report.update(method=METHOD, candidate_budget=budget, original_max_nodes=max_nodes,
                  discrete_grid_search=False, global_optimality_claim=False,
                  generated_candidates=0, tested_candidates=0, feasible_candidates=0,
                  installation_rejections={}, family_counts={}, selected=None)
    candidates = []; seen = set(); rejected = Counter(); families = Counter()
    def side_drop_candidates():
        if not side_discharge or b[2]>=a[2] or start_direction is None or end_direction is None:return
        if abs(start_direction[2])>.01 or end_direction[2]>-.99:return
        from .piping_elbows import coarse_radius
        R=coarse_radius(radius);side=np.asarray(start_direction)
        k=int(np.argmax(abs(side)));cross=1-k
        # Exit the valve horizontally. Lower the line at the flank, then enter
        # the receiver's straight inlet vertically from above.
        for z in (b[2]+R+.015,b[2]+R+end_straight_m+.01,(a[2]+b[2])/2):
            for step in (R+start_straight_m+.015,3*R+.05):
                turn=a+side*step;lower=turn.copy();lower[2]=z
                for sign in (-1,1):
                    lane=(max(a[cross],b[cross]) if sign==1 else min(a[cross],b[cross]))+sign*(3*R+.05)
                    p=lower.copy();p[cross]=lane
                    q=p.copy();q[k]=b[k]
                    finish=b.copy();finish[2]=z
                    yield [a,turn,lower,p,q,finish,b],{'family':'vessel_side_then_drop','planes':[]}
    general=whole_segment_candidates(a,b,low,high,obstacles,holes,radius,clearance,
        start_direction,end_direction,start_straight_m,end_straight_m,complete_bend_planes)
    for raw,metadata in itertools.chain(side_drop_candidates(),general):
        points = _simplify(raw)
        key = tuple(tuple(float(x) for x in point) for point in points)
        if key in seen:
            continue
        seen.add(key); families[metadata['family']] += 1
        report['generated_candidates'] += 1
        try:
            plan = plan_polyline(points, radius)
        except ElbowUnavailable as exc:
            rejected[exc.code] += 1
        else:
            length = sum(math.dist(r['start'], r['end']) for r in plan['runs'])+sum(
                e['centerline_radius_m']*math.radians(e['angle_degrees']) for e in plan['elbows'])
            bends = len(plan['elbows'])
            cost = (bends, length) if priority == 'bends_then_length' else (length, bends)
            candidates.append((cost, len(candidates), points, metadata, plan))
        if report['generated_candidates'] >= budget:
            break
    report['installation_rejections'] = dict(rejected)
    report['family_counts'] = dict(families)
    report['candidate_set_truncated'] = report['generated_candidates'] >= budget
    for _, _, points, metadata, plan in sorted(candidates, key=lambda row: (row[0], row[1])):
        report['tested_candidates'] += 1
        checked = feasible(points)
        if checked is None:
            continue
        cost, points = checked
        report['feasible_candidates'] += 1
        first = next((r for r in plan['runs'] if r['segment_index'] == 0), None)
        final = next((r for r in reversed(plan['runs']) if r['segment_index'] == len(plan['points'])-2), None)
        length = sum(math.dist(r['start'], r['end']) for r in plan['runs'])+sum(
            e['centerline_radius_m']*math.radians(e['angle_degrees']) for e in plan['elbows'])
        report['selected'] = {**metadata, 'bends': len(plan['elbows']), 'length_m': length,
            'guide': [p.tolist() for p in points],
            'endpoint_straights_m': {'start': math.dist(first['start'], first['end']) if first else 0.,
                                      'end': math.dist(final['start'], final['end']) if final else 0.},
            'priority': priority}
        report['elapsed_s'] = time.monotonic()-started
        return [p.tolist() for p in points]
    report['elapsed_s'] = time.monotonic()-started
    raise RouteUnavailable('PLANE_CANDIDATES_EXHAUSTED',
        'No bounded route or turn-plane candidate passed', **dict(report))
