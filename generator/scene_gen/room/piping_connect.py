"""Route process pipes between equipment and other pipe connections."""
from copy import deepcopy
from itertools import combinations
import math

import numpy as np

from . import piping_material_build as b
from .piping_elbows import plan_polyline
from .piping_build import _root_pipe_obstacle, _nearby_contacts

EPS = 1e-8
ATTEMPT_BUDGET = 400         # route attempts per pipe, the same for every pipe
TAP_POINTS = 12              # candidate tee points considered on a built pipe
KEEP_FEASIBLE = 3            # stop after this many feasible routes with valve room; keep the best
KEEP_ANY = 6                 # or after this many feasible routes of any kind
ROLE_RANK = {'bypass': 2, 'recycle': 1}
LEAD_M = (.4, .8, 1.2)       # fixed straight leads out of a mouth before searching the rest


def _crossed_straights(plan, radius):
    """A straight run must not loop back through its own emitted cylinders."""
    runs = []
    for item in plan['runs']:
        first, last = np.asarray(item['start']), np.asarray(item['end'])
        axis = int(np.argmax(abs(last-first)))
        runs.append((axis, np.minimum(first, last), np.maximum(first, last)))
    for (axis, lo, hi), (other, low, high) in combinations(runs, 2):
        if axis == other:
            cross = [i for i in range(3) if i != axis]
            if min(hi[axis], high[axis])-max(lo[axis], low[axis]) > EPS and (
                    math.dist(lo[cross], low[cross]) < 2*radius-EPS):
                return True
        else:
            third = 3-axis-other
            if (lo[axis]+EPS < low[axis] < hi[axis]-EPS
                    and low[other]+EPS < lo[other] < high[other]-EPS
                    and abs(lo[third]-low[third]) < 2*radius-EPS):
                return True
    return False


class Attempts:
    """Uniform budget, recording and choice for every pipe."""

    def __init__(self, a, label):
        self.a, self.label = a, label
        self.used, self.records, self.feasible = 0, [], []
        self.want = (0., 0.)   # straight wanted at each end for its own valve (preference only)
        self.bypass_straight = 0.  # a free bypass valve can use any real straight
        self.check_candidate = None
        self.rack_constraint = ()
        self.tee_pairs_without_valve_room = 0

    limit = ATTEMPT_BUDGET   # a share of the budget for one end pairing

    @property
    def open(self):
        good = sum(1 for row in self.feasible if not row[0][0])
        return (self.used < min(ATTEMPT_BUDGET, self.limit) and good < KEEP_FEASIBLE
                and len(self.feasible) < KEEP_ANY)

    def missing(self, plan):
        """How many ends lack room for their own valve (0, 1 or 2): ranking only."""
        want = self.want
        self.want = (want[0], 0.)
        a_ok = self.valve_room(plan)
        self.want = (0., want[1])
        b_ok = self.valve_room(plan)
        self.want = want
        bypass_ok = not self.bypass_straight or any(
            math.dist(run['start'], run['end']) >= self.bypass_straight-EPS
            for run in plan['runs'])
        return (not a_ok) + (not b_ok) + (not bypass_ok)

    def valve_room(self, plan):
        """The straight touching each end; an elbow right at the end leaves none."""
        runs = plan['runs']
        if not runs:
            return not any(self.want)
        ends = plan['points'][0], plan['points'][-1]
        first = math.dist(runs[0]['start'], runs[0]['end']) if math.dist(runs[0]['start'], ends[0]) <= 1e-6 else 0.
        last = math.dist(runs[-1]['start'], runs[-1]['end']) if math.dist(runs[-1]['end'], ends[1]) <= 1e-6 else 0.
        return first >= self.want[0]-EPS and last >= self.want[1]-EPS

    def run(self, approach, fn):
        if not self.open:
            return None
        self.used += 1
        try:
            points, extra = fn()
            plan = plan_polyline([np.asarray(p).tolist() for p in points], extra['radius'])
            if _crossed_straights(plan, extra['radius']):
                raise b.PipingRefused('PIPE_SELF_CROSSING: straight segments of the same pipe intersect')
            if self.check_candidate is not None:
                extra.update(self.check_candidate(plan))
        except b.KNOWN_REFUSALS as exc:
            self.records.append({'attempt': self.used, 'approach': approach, 'status': 'rejected',
                                 'reason': b._why(exc), 'evidence': getattr(exc, 'evidence', None)})
            return None
        length = sum(math.dist(x, y) for x, y in zip(plan['points'], plan['points'][1:]))
        room = self.missing(plan)
        self.records.append({'attempt': self.used, 'approach': approach, 'status': 'feasible',
                             'elbows': len(plan['elbows']), 'length_m': length, 'missing_valve_spaces': room})
        # Valve room ranks geometrically valid routes; a shorter lead can still connect.
        self.feasible.append(((room, len(plan['elbows']), length, len(self.feasible)), points,
                              {**extra, 'attempt_id': self.used}))
        return points

    def best(self):
        if not self.feasible:
            return None
        return min(self.feasible, key=lambda row: row[0])

    def report(self):
        best = self.best()
        return {'pipe': self.label, 'budget': ATTEMPT_BUDGET, 'used': self.used,
                'feasible': len(self.feasible), 'attempts': self.records,
                'tee_pairs_without_valve_room': self.tee_pairs_without_valve_room,
                'selected_attempt': best[2].get('attempt_id') if best is not None else None}


def _node(endpoint):
    return endpoint.split('.')[0]


def _level(a, z):
    return max(i for i, floor in enumerate(a.floors) if floor <= z + 1e-6)


# --------------------------------------------------------------------------
# Network plan: main runs and branches


def _resolve_far(a, units, edges, endpoint, seen):
    """A real coordinate reached from this endpoint along non-branch arms."""
    if endpoint in a.anchors:
        return np.asarray(a.anchors[endpoint]['point'], float)
    uid = _node(endpoint)
    if uid in seen or units.get(uid, {}).get('kind') != 'junction':
        return None
    seen.add(uid)
    for edge in edges:
        if edge.get('role') in ROLE_RANK:
            continue
        for mine, other in ((edge['a'], edge['b']), (edge['b'], edge['a'])):
            if _node(mine) == uid and _node(other) != uid:
                point = _resolve_far(a, units, edges, other, seen)
                if point is not None:
                    return point
    return None


def plan_network(a, edges, terminals, free):
    """Rank arms and contract main pairs into runs; nothing is positioned."""
    units = a.frozen['graph']['units']
    arms = {uid: [e for e in edges if uid in (_node(e['a']), _node(e['b']))] for uid in free}
    main = {}
    transitions = {}
    ranking = []
    for uid, incident in arms.items():
        def far(edge):
            other = edge['b'] if _node(edge['a']) == uid else edge['a']
            if edge['id'] in terminals and other == terminals[edge['id']]['boundary_endpoint']:
                return None
            return _resolve_far(a, units, edges, other, {uid})
        points = {e['id']: far(e) for e in incident}
        reference = next((points[e['id']] for e in sorted(incident, key=lambda e: (-e['dn'], ROLE_RANK.get(e['role'], 0), e['id']))
                          if points[e['id']] is not None), None)

        def key(edge):
            point = points[edge['id']]
            length = 0. if point is None or reference is None else float(np.abs(point - reference).sum())
            return (-edge['dn'], ROLE_RANK.get(edge['role'], 0), length, edge['id'])
        ordered = sorted(incident, key=key)
        main[uid] = [e['id'] for e in ordered[:2]]
        if len(ordered) >= 2 and ordered[0]['dn'] != ordered[1]['dn']:
            transitions[uid] = {'large_edge': ordered[0]['id'], 'large_dn': ordered[0]['dn'],
                                'small_edge': ordered[1]['id']}
        ranking.append({'junction': uid, 'arms': [e['id'] for e in ordered], 'main': main[uid],
                        'branches': [e['id'] for e in ordered[2:]],
                        'rule': 'DN descending; main process before recycle before bypass; shorter first'})
    by_id = {e['id']: e for e in edges}
    assigned, runs = set(), []

    def continuation(edge_id, endpoint):
        uid = _node(endpoint)
        if uid in free and edge_id in main[uid]:
            other = next((x for x in main[uid] if x != edge_id
                          and by_id[x]['dn'] == by_id[edge_id]['dn']), None)
            return uid, other
        return None, None

    for edge in edges:
        if edge['id'] in assigned:
            continue
        chain = [(edge['id'], False)]  # (edge, reversed)
        assigned.add(edge['id'])
        # extend forward from b
        tail_edge, tail_end = edge['id'], edge['b']
        while True:
            uid, nxt = continuation(tail_edge, tail_end)
            if not nxt or nxt in assigned:
                break
            e = by_id[nxt]
            rev = _node(e['b']) == uid
            chain.append((nxt, rev))
            assigned.add(nxt)
            tail_edge, tail_end = nxt, (e['a'] if rev else e['b'])
        head_edge, head_end = edge['id'], edge['a']
        while True:
            uid, nxt = continuation(head_edge, head_end)
            if not nxt or nxt in assigned:
                break
            e = by_id[nxt]
            rev = _node(e['a']) == uid
            chain.insert(0, (nxt, rev))
            assigned.add(nxt)
            head_edge, head_end = nxt, (e['b'] if rev else e['a'])
        def end_of(edge_id, endpoint):
            uid = _node(endpoint)
            transition = transitions.get(uid)
            if transition and edge_id in main[uid]:
                kind = 'reducer_stub' if edge_id == transition['small_edge'] else 'fixed'
                return {'kind': kind, 'endpoint': endpoint, 'edge': edge_id,
                        'junction': uid, **transition}
            if edge_id in terminals and endpoint == terminals[edge_id]['boundary_endpoint']:
                return {'kind': 'terminal', 'endpoint': endpoint, 'edge': edge_id}
            if uid in free:
                return {'kind': 'tap', 'junction': uid, 'endpoint': endpoint, 'edge': edge_id}
            return {'kind': 'fixed', 'endpoint': endpoint, 'edge': edge_id}
        first_id, first_rev = chain[0]
        last_id, last_rev = chain[-1]
        start_ep = by_id[first_id]['b'] if first_rev else by_id[first_id]['a']
        end_ep = by_id[last_id]['a'] if last_rev else by_id[last_id]['b']
        inner = []
        for (e1, r1), (e2, r2) in zip(chain, chain[1:]):
            ep = by_id[e1]['a'] if r1 else by_id[e1]['b']
            inner.append(_node(ep))
        run = {'id': 'run_'+first_id, 'edges': [x for x, _ in chain],
               'reversed': [r for _, r in chain], 'dn': edge['dn'],
               'start': end_of(first_id, start_ep), 'end': end_of(last_id, end_ep),
               'junctions': inner}
        if run['start']['kind'] == 'terminal':
            run.update(edges=list(reversed(run['edges'])), reversed=[not x for x in reversed(run['reversed'])],
                       junctions=list(reversed(run['junctions'])), start=run['end'], end=run['start'])
        runs.append(run)
    return runs, ranking


# --------------------------------------------------------------------------
# Built runs: arc positions and tee points


def _run_components(a, run):
    parts = {c['id']: c for c in a.scene['components']}
    route = next(r for r in a.routes if r['id'] == run['route'])
    return [parts[cid] for cid in route['components'] if cid in parts], route


def _arc_table(a, run):
    """Ordered straight pipes of a built run with cumulative arc positions."""
    components, route = _run_components(a, run)
    elbows = {e['id']: e for e in a.scene.get('elbows', []) if e.get('route_id') == route['id']}
    sector_length = {}
    for record in elbows.values():
        length = record['centerline_radius_m']*math.radians(record['angle_degrees'])
        for cid in record['components']:
            sector_length[cid] = length/len(record['components'])
    table, s, cursor = [], 0., np.asarray(route['points'][0], float)
    for c in components:
        if c.get('kind') == 'pipe' and c.get('shape') == 'cylinder' and c.get('ends'):
            one, two = map(np.asarray, c['ends'])
            if math.dist(two, cursor) < math.dist(one, cursor):
                one, two = two, one
            length = math.dist(one, two)
            table.append({'component': c['id'], 'from': one, 'to': two, 's0': s, 'length': length,
                          'radius': float(c['radius'])})
            s += length
            cursor = two
        else:
            s += sector_length.get(c['id'], 0.)
            if c.get('ports'):
                ends = [np.asarray(p['position']) for p in c['ports'].values()]
                cursor = max(ends, key=lambda p: math.dist(p, cursor))
    return table


def _arc_of(table, point):
    for row in table:
        axis = row['to']-row['from']
        if row['length'] <= EPS:
            continue
        t = float((np.asarray(point)-row['from'])@axis)/row['length']**2
        if -EPS <= t <= 1+EPS and np.linalg.norm(row['from']+axis*t-point) <= 1e-6:
            return row['s0']+t*row['length']
    return None


def _tap_candidates(a, run, junction, branch_r, source, *, toward_final=None):
    """Points on the run's real straights between its neighbouring tees."""
    table = _arc_table(a, run)
    junctions = _owned_junctions(run)
    order = junctions.index(junction)
    low, high = 0., sum(row['length'] for row in table) + 10.
    resolved = run['resolved']
    for index, uid in enumerate(junctions):
        if uid not in resolved or uid == junction:
            continue
        s = _arc_of(table, resolved[uid])
        if s is None:
            continue
        gap = 2*(branch_r+run['radius'])+.05
        if index < order:
            low = max(low, s+gap)
        else:
            high = min(high, s-gap)
    # When two neighbouring tees bound a valved leg, allow space for the
    # valve bodies when selecting the second tee. Keep shorter connections
    # as fallbacks, just as for valve space at an equipment mouth.
    between_valves = []
    internal = run['junctions']
    if junction in internal and not hasattr(a, 'path_requirements'):
        index = internal.index(junction)
        for neighbor in (index-1, index+1):
            if not 0 <= neighbor < len(internal) or internal[neighbor] not in resolved:
                continue
            edge = run['edges'][max(index, neighbor)]
            valves = [v for v in a.deferred_valves if v['edge'] == edge and
                      v['placement_rule'] in ('after_split', 'before_join')]
            neighbor_arc = _arc_of(table, resolved[internal[neighbor]])
            if len(valves) < 2 or neighbor_arc is None:
                continue
            from .piping_assets import inspect_asset
            length = sum(inspect_asset(v['dn'], v.get('operator'))['install_length_m'] for v in valves)
            between_valves.append((neighbor_arc, length))

    def keeps_between_valve_room(s):
        for other, length in between_valves:
            lo, hi = sorted((s, other))
            if not any(min(hi, row['s0']+row['length'])-max(lo, row['s0']) >= length-EPS
                       for row in table):
                return False
        return True

    total = sum(row['length'] for row in table)
    keep = run.get('want', (0., 0.))   # mouth straights kept for the run's own valves (ranking only)
    points = []
    for row in table:
        margin = max(2*branch_r, 2*row['radius'])+.05
        start, stop = max(row['s0']+margin, low), min(row['s0']+row['length']-margin, high)
        if stop < start-EPS:
            continue
        axis = (row['to']-row['from'])/row['length']
        offsets = {start, stop, (start+stop)/2}
        projection = row['s0']+float((np.asarray(source)-row['from'])@axis)
        offsets.add(min(stop, max(start, projection)))
        if toward_final is not None:
            projection = row['s0']+float((np.asarray(toward_final)-row['from'])@axis)
            offsets.add(min(stop, max(start, projection)))
        step = .25
        value = start
        while value < stop:
            offsets.add(value)
            value += step
        for s in offsets:
            point = row['from']+axis*(s-row['s0'])
            clear = s >= keep[0]+2*row['radius']+.1 and s <= total-keep[1]-2*row['radius']-.1
            points.append({'point': point, 'axis': axis, 'component': row['component'], 's': s,
                           'main_radius': row['radius'], 'keeps_valve_room': clear,
                           'keeps_between_valve_room': keeps_between_valve_room(s)})
    points.sort(key=lambda p: (not p['keeps_between_valve_room'], not p['keeps_valve_room'],
                               math.dist(p['point'], toward_final) if toward_final is not None else 0.,
                               float(np.abs(p['point']-np.asarray(source)).sum()), p['s']))
    unique = []
    for p in points:
        if all(math.dist(p['point'], q['point']) > .05 for q in unique):
            unique.append(p)
    return unique[:TAP_POINTS]


def _arrivals(axis, toward):
    choices = []
    for k in range(3):
        for sign in (1., -1.):
            d = np.zeros(3)
            d[k] = sign
            if abs(float(d@axis)) > EPS:
                continue
            choices.append(d)
    return sorted(choices, key=lambda d: -float(d@(np.asarray(toward))))


def _tap_obstacles(a, taps, dn, clearance=.05):
    """Retain every part of a target straight outside the local join windows."""
    points_by_part = {}
    for tap in taps:
        points_by_part.setdefault(tap['component'], []).append(tap['point'])
    bare = a.config['pg_rules']['parallel_pipe_spacing']['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000
    extra = []
    for cid, points in points_by_part.items():
        part = next(c for c in a.scene['components'] if c['id'] == cid)
        one, two = map(np.asarray, part['ends'])
        axis = (two-one)/np.linalg.norm(two-one)
        trim = a.radius(dn)+float(part['radius'])+max(clearance, bare)+1e-6
        cursor = one
        for index, p in enumerate(sorted(points, key=lambda p: float((p-one)@axis))):
            stop = p-axis*trim
            if float((stop-cursor)@axis) > EPS:
                suffix = '_before_tee' if not index else '_between_tees'
                extra.append(_root_pipe_obstacle(cid+suffix, cursor, stop, part['radius']))
            cursor = p+axis*trim
        if float((two-cursor)@axis) > EPS:
            extra.append(_root_pipe_obstacle(cid+'_after_tee', cursor, two, part['radius']))
    return extra


def _split_pipe(a, route, cid, point):
    """Replace one straight by two collinear pieces meeting at a tee point."""
    part = next(c for c in a.scene['components'] if c['id'] == cid)
    one, two = map(np.asarray, part['ends'])
    if min(math.dist(one, point), math.dist(two, point)) <= 1e-6:
        return cid, cid
    first = a.pipe(route['id'], one, point, route['dn'], route['medium'], route['id'])
    second = a.pipe(route['id'], point, two, route['dn'], route['medium'], route['id'])
    for piece in (first, second):
        piece.update({k: part[k] for k in ('level', 'phase', 'straight_segment_index') if k in part})
    a.scene['components'] = [c for c in a.scene['components'] if c['id'] != cid]
    for key in ('components', 'pipe_components'):
        if cid in route[key]:
            i = route[key].index(cid)
            route[key][i:i+1] = [first['id'], second['id']]
    contacts = [c for c in a.allowed_contacts if cid in c['pair']]
    a.allowed_contacts = [c for c in a.allowed_contacts if cid not in c['pair']]
    for contact in contacts:
        q = np.asarray(contact['point'])
        new = first['id'] if math.dist(q, one) <= math.dist(q, two) else second['id']
        a.allowed_contacts.append({**contact, 'pair': [new if x == cid else x for x in contact['pair']]})
    for hole in a.openings:
        if hole.get('component') == cid:
            hole['component'] = first['id'] if hole['rect'] and one[2] <= two[2] else second['id']
    a.allow(first['id'], second['id'], point)
    return first['id'], second['id']


# --------------------------------------------------------------------------
# Approaches: every attempt is one call of the common search


def _endpoint_options(a, start, end, extra_points=()):
    from .piping_material_network import _endpoint_search_options
    options = _endpoint_search_options(a, start, end)
    a._network_contact_points = [*a._network_contact_points, *map(np.asarray, extra_points)]
    # Pieces already joined at a shared fixed node (e.g. the seal low point)
    # are the joint itself, not an obstacle for the next arm into that node.
    shared = set()
    for anchor in (start, end):
        if 'adapter_end' in anchor:
            continue
        p = np.asarray(anchor['point'], float)
        for c in a.scene['components']:
            ends = list(c.get('ends') or []) + [q['position'] for q in (c.get('ports') or {}).values()]
            if any(math.dist(e, p) <= 1e-6 for e in ends) and str(c.get('route_id', '')).startswith('P2_'):
                shared.add(c['id'])
    options['skip'] = tuple(sorted(set(options['skip']) | shared))
    return options


def _first_edge(by_id, run, at_end):
    return by_id[run['edges'][-1 if at_end else 0]]


def _drain_guide(first, last):
    return [first, np.r_[last[:2], first[2]], last]


def _pending_drain_straights(a, by_id, plan):
    """Find a fixed drain this candidate would obstruct before it is built."""
    from .piping_route_search import _segments_distance

    built = a.material.get('pipe_emission_order', ())
    radius = plan['pipe_outer_radius_m']
    runs = [(np.asarray(p['start']), np.asarray(p['end'])) for p in plan['runs']]
    obstacles = []
    for edge in by_id.values():
        if edge.get('boundary_kind') != 'ground_drain_point' or edge['id'] in built:
            continue
        start, end = (a.anchors.get(edge[key]) for key in ('a', 'b'))
        if start is None or end is None:
            continue
        drain_radius = a.radius(edge['dn'])
        first, last = np.asarray(start['point']), np.asarray(end['point'])
        guide = _drain_guide(first, last)
        if abs(first[2]-last[2]) <= EPS or np.count_nonzero(abs(guide[1]-first)>EPS) != 1:
            continue
        legs = list(zip(guide, guide[1:]))
        if any(_segments_distance(p, q, x, y) < radius+drain_radius+.05-EPS
               for p, q in runs for x, y in legs):
            obstacles.extend(_root_pipe_obstacle(f"{edge['id']}_planned_drain_{i}", x, y, drain_radius)
                             for i, (x, y) in enumerate(legs))
    return obstacles


def _point_approaches(a, run, by_id, start, end, attempts, *, extra=(), tap_points=(), skip_extra=()):
    """Every approach for one start/end pair: same budget, same recording."""
    from .material_rack import connect_in_rack, destination_rule
    from .piping_material_network import _riser_candidates, _horizontal_lift_candidates
    first, last = np.asarray(start['point'], float), np.asarray(end['point'], float)
    dn = run['dn']
    r = a.radius(dn)
    edge = _first_edge(by_id, run, True)
    head = _first_edge(by_id, run, False)
    units = a.frozen['graph']['units']
    radius = {'radius': r}
    straight = [start.get('straight', 0.), end.get('straight', 0.)]
    want = list(attempts.want)
    full = (max(straight[0], want[0]), max(straight[1], want[1]))
    variants = list(dict.fromkeys([full, (full[0], straight[1]), (straight[0], full[1]), tuple(straight)]))

    def options():
        result = _endpoint_options(a, start, end, tap_points)
        result['skip'] = tuple(sorted(set(result['skip']) | set(skip_extra)))
        return result

    # Steam trap discharge: horizontal, one elbow, down to the ground point.
    if edge['role'] == 'steam_condensate' and units[_node(head['a'])]['kind'] == 'automatic_steam_trap':
        def drain():
            if not np.allclose(last[2], a.floors[0]) or last[2] >= first[2]:
                raise b.PipingRefused('The condensate-trap outlet connects at ground level and routes downward')
            return a.search(first, last, dn, **options(), extra=extra, clearance=.05, start_direction=start['normal'],
                            end_direction=[0, 0, -1], guide=_drain_guide(first, last)), radius
        attempts.run('trap_discharge_horizontal_then_down', drain)
        return
    before_trap = (edge['role'] == 'steam_condensate' and units[_node(edge['b'])]['kind'] == 'automatic_steam_trap')
    levels = {row['id']: row['level'] for row in a.frozen['contracts']['layout']['equipment']}
    rule = destination_rule(a.frozen['graph'], levels, edge, int(start['level']), int(end['level']))
    if attempts.rack_constraint:
        _, band, _ = attempts.rack_constraint
        level = next(k for k, zone in a.plan['rack_zones'].items()
                     if (zone['bottom_z_m'], zone['top_z_m']) == band)
        rule = {'kind': 'ascending_material', 'level': level}
    if rule:
        oriented = {**edge, 'a': head['a']}
        connect_in_rack(a, oriented, start, end, rule, options(), straight[0], straight[1],
                        attempts=attempts, extra=extra,
                        full_rack=attempts.rack_constraint)
        return

    def prefer_drain_clear(label, search):
        first_candidate = len(attempts.feasible)
        points = attempts.run(label, search)
        if points is None or tap_points or head['role'] not in ('material', 'recycle'):
            return False
        obstacles = _pending_drain_straights(a, by_id, plan_polyline(points, r))
        if not obstacles:
            return False
        original = attempts.feasible[first_candidate]
        attempts.records[-1]['obstructed_planned_drains'] = [obstacle['id'] for obstacle in obstacles]
        alternate = attempts.run(label+'_keep_drain_clear', lambda: search(obstacles))
        if (alternate is None or attempts.feasible[-1][0][0] > original[0][0] or
                _pending_drain_straights(a, by_id, plan_polyline(alternate, r))):
            return False
        attempts.feasible[:] = [candidate for candidate in attempts.feasible
                                if candidate[0][0] < original[0][0] or not _pending_drain_straights(
                                    a, by_id, plan_polyline(candidate[1], r))]
        return True

    def plain(guide=None, chosen=None):
        chosen = chosen or tuple(straight)

        def search(keep_clear=()):
            bounds = a.bounds
            try:
                if before_trap:
                    if last[2] >= first[2]-EPS:
                        raise b.PipingRefused('The trap inlet must be below the equipment condensate outlet')
                    a.bounds = np.asarray(a.bounds).copy()
                    a.bounds[5] = min(a.bounds[5], first[2]+r)
                points = a.search(first, last, dn, **options(), extra=(*extra, *keep_clear), clearance=.05,
                                  start_direction=start['normal'], end_direction=-np.asarray(end['normal']),
                                  start_straight_m=chosen[0], end_straight_m=chosen[1],
                                  priority='bends_then_length', complete_bend_planes=True,
                                  side_discharge=start.get('bottom_discharge_after_valve', False),
                                  **({'guide': guide} if guide is not None else {}))
                if before_trap:
                    plan = plan_polyline(points, r)
                    if any(y[2] > x[2]+EPS for x, y in zip(plan['points'], plan['points'][1:])):
                        raise b.PipingRefused('The pipe upstream of the trap cannot rise')
                return points, radius
            finally:
                a.bounds = bounds
        return search
    for index, chosen in enumerate(variants):
        label = 'direct_search'+('_valve_room' if index == 0 and len(variants) > 1 else '')
        if prefer_drain_clear(label, plain(chosen=chosen)):
            return
    # Lead first: leave each mouth straight for a fixed length, then search
    # the rest. The same family for every pipe; each one is one attempt.
    n_start = np.asarray(start['normal'], float)
    n_end = np.asarray(end['normal'], float)
    for length in LEAD_M:
        for use_start, use_end in ((True, False), (False, True), (True, True)):
            if not attempts.open:
                break

            def lead(keep_clear=(), length=length, use_start=use_start, use_end=use_end):
                p = first+n_start*(length+straight[0]) if use_start else first
                q = last+n_end*(length+straight[1]) if use_end else last
                opts = options()
                obstacles = (*extra, *keep_clear)
                # The straight out of a mouth is part of that local joint: real
                # geometry, the same 0.01 m check as the mouth adapter itself.
                joint = .01

                def straight_leg(x, y, anchor, direction):
                    leg = a.search(x, y, dn, **opts, extra=obstacles, clearance=joint if 'adapter_end' in anchor else .05,
                                   start_direction=direction, end_direction=direction)
                    if len(leg) != 2:
                        raise b.PipingRefused('The segment leaving the port must be straight')
                if use_start:
                    straight_leg(first, p, start, n_start)
                if use_end:
                    straight_leg(q, last, end, -n_end)
                middle = a.search(p, q, dn, **opts, extra=obstacles, clearance=.05,
                                  start_direction=n_start, end_direction=-n_end,
                                  priority='bends_then_length', complete_bend_planes=True)
                points = ([first] if use_start else [])+[np.asarray(x) for x in middle]+([last] if use_end else [])
                clean = [np.asarray(points[0], float)]
                for x in points[1:]:
                    if np.linalg.norm(np.asarray(x)-clean[-1]) > 1e-9:
                        clean.append(np.asarray(x, float))
                return clean, radius
            side = 'both' if use_start and use_end else 'start' if use_start else 'end'
            if prefer_drain_clear(f'lead_{length:.1f}_{side}', lead):
                return
    # The approved filter-to-pump front-side guide is one more approach.
    if units[_node(edge['b'])]['kind'] == 'pump' and abs(first[2]-last[2]) <= EPS:
        n1, n2 = np.asarray(start['normal']), np.asarray(end['normal'])
        if abs(n1@n2) <= EPS and abs(n1[2])+abs(n2[2]) <= EPS:
            from .piping_elbows import coarse_radius
            turn = coarse_radius(r)
            one = first+n1*(turn+straight[0]+.02)
            two = last+n2*(turn+.02)
            middle = one+n2*((two-one)@n2)
            attempts.run('pump_front_side_guide', plain([first, one, middle, two, last]))
    crossed = [z for z in a.floors[1:] if min(first[2], last[2]) < z < max(first[2], last[2])]
    if crossed and np.allclose(start['normal'], [0, 0, 1]) and last[2] > first[2]:
        pseudo = {'id': edge['id'], 'dn': dn}
        for label, generator in (('shared_horizontal_plane', _horizontal_lift_candidates),
                                 ('joint_riser', _riser_candidates)):
            if not attempts.open:
                break
            failures = []
            results = generator(a, pseudo, start, end, failures)
            for message in failures:
                if not attempts.open:
                    break
                attempts.used += 1
                attempts.records.append({'attempt': attempts.used, 'approach': label, 'status': 'rejected',
                                         'reason': message})
            for bends, length, points, state in results:
                if not attempts.open:
                    break
                attempts.run(label, lambda: (points, {**radius, 'state': state}))


# --------------------------------------------------------------------------
# Emission


def _departures(a, run, start, end, points):
    merged = [np.asarray(p).tolist() for p in points]
    removed = []
    for side, anchor in enumerate((start, end)):
        for route in list(a.routes):
            if (route.get('source_edge') in run['edges'] and route['id'].endswith('_normal_departure')
                    and math.dist(route['points'][-1], anchor['point']) <= EPS):
                tail = route['control_points']
                merged = tail+merged[1:] if side == 0 else merged+list(reversed(tail))[1:]
                removed.append(route)
    from .piping_material_network import _remove_routes
    _remove_routes(a, removed)
    return merged


def _emit(a, run, by_id, start, end, points):
    edge = by_id[run['edges'][0]]
    merged = _departures(a, run, start, end, points)
    route = a.route('P2_'+run['id'], merged, run['dn'], b._medium(edge), 'material_connection', start['level'],
                    source={'endpoint': run['start']['endpoint'], 'position': merged[0]},
                    target={'endpoint': run['end']['endpoint'], 'position': merged[-1]}, **b._meta(edge))
    route['network_run'] = run['id']
    for point in (merged[0], merged[-1]):
        b._join_contacts(a, route, point)
    run['route'] = route['id']
    return route


def _anchor(a, end_spec, dn=None):
    anchor = deepcopy(a.anchors[end_spec['endpoint']])
    anchor['point'] = np.asarray(anchor['point'], float)
    # A fixed node shared by several pipes (seal low point) keeps a short
    # straight for its fittings: 2r+0.05, the same fitting rule as a tee arm.
    anchor['straight'] = 0. if 'adapter_end' in anchor or dn is None else 2*a.radius(dn)+.05
    return anchor


def _stub_length(a, run, by_id):
    """A blind end only needs room for its own valves and tees."""
    from .piping_assets import inspect_asset
    valves = [v for v in a.deferred_valves if v['edge'] in run['edges']]
    r = a.radius(run['dn'])
    return (sum(3*inspect_asset(v['dn'])['install_length_m'] for v in valves)
            + len(run['junctions'])*(4*r+.1) + 2*r+.05)


def _build_terminal(a, run, by_id, terminals, origin, attempts, *, extra=(), skip_extra=()):
    """A run that ends in a blind flange: straight, or one elbow then straight."""
    dn = run['dn']
    r = a.radius(dn)
    length = _stub_length(a, run, by_id)
    p = np.asarray(origin['point'], float)
    n = np.asarray(origin['normal'], float)
    shapes = [('straight', [p, p+n*(length+origin.get('straight', 0.))])]
    for k in range(3):
        for sign in (1., -1.):
            d = np.zeros(3)
            d[k] = sign
            if abs(float(d@n)) > EPS:
                continue
            bend = p+n*(3*r+.05+origin.get('straight', 0.))
            shapes.append(('elbow_'+'xyz'[k]+('+' if sign > 0 else '-'), [p, bend, bend+d*length]))
    closure = .02+.039+.03   # terminal stub, flange and blind beyond the run end
    for label, guide in shapes:
        def search(guide=guide):
            tail = (guide[-1]-guide[-2])/np.linalg.norm(guide[-1]-guide[-2])
            checked = [*guide[:-1], guide[-1]+tail*closure]
            pts = a.search(checked[0], checked[-1], dn, skip=tuple(b._adjacent_parts(a, origin, p))+tuple(skip_extra),
                           extra=extra, clearance=.01, guide=checked, start_direction=n, end_direction=tail,
                           port_contacts=b._departure_contacts(origin))
            pts = [np.asarray(q, float) for q in pts]
            pts[-1] = pts[-1]-tail*closure
            # A flange and blind are wider than the pipe hole cut in a deck.
            ends = np.array([pts[-1], pts[-1]+tail*closure])
            box = np.r_[ends.min(axis=0)-(r+.04), ends.max(axis=0)+(r+.04)]
            for c in a.scene['components']:
                if c.get('kind') == 'floor' and all(min(box[k+3], c['box'][k+3])-max(box[k], c['box'][k]) > 1e-6
                                                    for k in range(3)):
                    raise b.PipingRefused('Blind flange and plate would intersect the floor: '+c['id'])
            return pts, {'radius': r}
        attempts.run('blind_end_'+label, search)


def _finish_terminal(a, run, by_id, terminals, points):
    terminal = terminals[run['end']['edge']]
    edge = by_id[run['end']['edge']]
    last, before = np.asarray(points[-1], float), np.asarray(points[-2], float)
    normal = (last-before)/np.linalg.norm(last-before)
    saved = a.anchors.get(terminal['network_endpoint'])
    a.anchors[terminal['network_endpoint']] = {'point': last, 'normal': normal, 'level': _level(a, last[2])}
    try:
        b._terminal(a, terminal, edge, pending_root=True, stub_length=.02)
    finally:
        if saved is None:
            a.anchors.pop(terminal['network_endpoint'], None)
        else:
            a.anchors[terminal['network_endpoint']] = saved


# --------------------------------------------------------------------------
# Building one run


def _valve_straight(a, run, spec, anchor):
    """Straight wanted next to this end for its own valve; a preference, never required."""
    from .piping_assets import inspect_asset
    if anchor.get('bottom_side_run'):
        return 0.
    edge = spec['edge']
    if spec['kind'] == 'fixed':
        valves = [v for v in a.deferred_valves if v['edge'] == edge and v['at'] == spec['endpoint']]
    elif spec['kind'] == 'tap':
        valves = [v for v in a.deferred_valves if v['edge'] == edge and v.get('placement_rule') == 'before_join']
    else:
        valves = []
    return max((3*inspect_asset(v['dn'])['install_length_m']+.02 for v in valves), default=0.)


def _record_failure(a, run, attempts, reason):
    report = attempts.report() if attempts else {'pipe': run['id'], 'attempts': []}
    a.material.setdefault('network_attempts', []).append({**report, 'status': 'break', 'reason': reason})
    for edge_id in run['edges']:
        a.material.setdefault('network_failures', []).append({
            'code': 'NETWORK_ROUTE_FAILED', 'stage': 'material_pipe', 'edge': edge_id, 'run': run['id'],
            'reason': reason, 'attempts_used': report.get('used'), 'budget': ATTEMPT_BUDGET})


def _owned_junctions(run):
    start = [run['start']['junction']] if run['start']['kind'] == 'reducer_stub' else []
    end = [run['end']['junction']] if run['end']['kind'] == 'reducer_stub' else []
    return [*start, *run['junctions'], *end]


def _tap_targets(a, runs, run, spec):
    return next((other for other in runs if spec['junction'] in _owned_junctions(other)), None)


def _build_reducer_stub(a, run, ends, by_id, attempts):
    """A checked same-DN equipment lead, then a reducer for the larger run."""
    index = next(i for i, spec in enumerate(ends) if spec['kind'] == 'reducer_stub')
    stub, fixed = ends[index], ends[1-index]
    if fixed['kind'] != 'fixed' or fixed['endpoint'] not in a.requirements['ports']:
        _record_failure(a, run, attempts, 'The reducer small end has no straight equipment outlet')
        return False
    anchor = _anchor(a, fixed, run['dn'])
    origin, direction = anchor['point'], np.asarray(anchor['normal'])
    branches = [row['dn'] for row in by_id.values()
                if stub['junction'] in (_node(row['a']), _node(row['b']))
                and row['id'] not in (stub['small_edge'], stub['large_edge'])]
    tee_room = 2*(2*a.radius(max(run['dn'], *branches))+.05) if branches else 0.
    neck = origin + direction * (_stub_length(a, run, by_id) + tee_room)
    large_dn = stub['large_dn']
    outer = neck + direction*(2*a.radius(large_dn))
    edge = by_id[stub['large_edge']]
    junction = stub['junction']
    mate = next(end for end in (edge['a'], edge['b']) if _node(end) == junction)

    def checked_stub():
        opts = _endpoint_options(a, anchor, anchor)
        a.search(origin, neck, run['dn'], **opts, clearance=.01, straight=True)
        a._network_contact_points = []
        a.search(neck, outer, large_dn, clearance=.01, straight=True)
        return [origin, neck], {'radius': run['radius']}

    try:
        attempts.run('small_dn_straight_and_reducer', checked_stub)
    finally:
        a._network_contact_points = []
    best = attempts.best()
    if best is None:
        _record_failure(a, run, attempts, 'The small-bore stub or reducer intersects an existing solid')
        return False
    if 'routed_valve_spans' in best[2]:
        run['routed_valve_spans'] = best[2]['routed_valve_spans']

    transition = {'point': neck, 'normal': -direction, 'level': _level(a, neck[2])}
    if index:
        route = _emit(a, run, by_id, anchor, transition, [origin, neck])
    else:
        route = _emit(a, run, by_id, transition, anchor, [neck, origin])
    a._network_contact_points = []
    medium = b._medium(edge)
    name = 'P2_'+edge['id']+'_network_reducer'
    _, body = a.reducer(name, neck, direction, run['dn'], large_dn, medium)
    body.update(phase='P2', route_id=name, source_edge=edge['id'])
    reducer = {'id': name, 'points': [neck.tolist(), outer.tolist()], 'components': [body['id']],
               'valves': [], 'dn': large_dn, 'medium': medium, 'role': 'network_reducer',
               'phase': 'P2', 'level': transition['level'], **b._meta(edge)}
    a.routes.append(reducer)
    b._join_contacts(a, reducer, neck)
    a.anchors[mate] = {'point': outer, 'adapter_end': outer, 'normal': direction,
                       'level': _level(a, outer[2]), 'components': [body['id']],
                       'departure_contact': {'position': outer.tolist(), 'normal': direction.tolist(),
                                             'contact_ids': [body['id']]}}
    a.material['junctions'].append({'id': junction, 'position': neck.tolist(),
        'incidents': [e['id'] for e in a.requirements['edges'] if junction in (_node(e['a']), _node(e['b']))],
        'scope': 'The small-bore stub mates with the actual reducer small face'})
    run['reducer_route'] = name
    run['want'] = (0., _valve_straight(a, run, fixed, anchor)) if not index else (
        _valve_straight(a, run, fixed, anchor), 0.)
    a.material.setdefault('network_attempts', []).append({**attempts.report(), 'status': 'connected',
                                                          'route': route['id']})
    return True


def _finish_reducer_tap(a, run, route, tap, pieces):
    """The tee selects the start of the real small-end spool on the reducer."""
    reducer = next(r for r in a.routes if r['id'] == run['reducer_route'])
    point = tap['point'].tolist()
    neck = reducer['points'][0]
    tail = next(c for c in a.scene['components'] if c['id'] in pieces
                and any(math.dist(end, neck)<=EPS for end in c['ends']))
    part = tail['id']
    for key in ('components', 'pipe_components'):
        route[key].remove(part)
    end = 0 if run['start']['kind'] == 'reducer_stub' else -1
    route['points'][end] = point
    route['control_points'][end] = point
    route['source' if end == 0 else 'target']['position'] = point
    tail['owner'] = reducer['id']
    tail['route_id'] = reducer['id']
    reducer['components'].insert(0, part)
    reducer['points'].insert(0, point)
    junction = next(row for row in a.material['junctions'] if row['id'] == tap['junction'])
    junction.update(position=point, scope='Pick a physical tee on the small-end straight. Keep the reducer coaxial.')


def build_run(a, runs, run, by_id, terminals):
    attempts = Attempts(a, run['id'])
    has_full_route_rules = hasattr(a, 'path_requirements')
    if has_full_route_rules:
        from .connection_rule_checks import trap_discharge_geometry, trap_inlet_geometry
        from .route_candidate_checks import branch_rack_constraint, candidate_path, rack_candidate_ok, reverse_path
        trap_rule = next((row['routing_rule'] for row in a.path_requirements if row['edges'] == run['edges']
                          and row['routing_rule'] in ('M3e', 'M3f')), None)

        def guard(plan):
            path = candidate_path(plan, run['dn'])
            if run['start']['kind'] == 'reducer_stub':
                path = reverse_path(path)
            if trap_rule is not None:
                directed = reverse_path(path) if run['reversed'][0] else path
                start, end = directed[0]['ends'][0], directed[-1]['ends'][1]
                if trap_rule == 'M3e':
                    okay, _ = trap_inlet_geometry(directed, start, end)
                else:
                    okay, _ = trap_discharge_geometry(directed, start, end, end, a.floors[0])
                if not okay:
                    raise b.PipingRefused('FULL_ROUTE: condensate routing does not satisfy '+trap_rule)
            return {}

        attempts.check_candidate = guard
    if not has_full_route_rules and len(run['edges']) == 1 and by_id[run['edges'][0]]['role'] == 'bypass':
        from .piping_assets import inspect_asset
        attempts.bypass_straight = max((
            3*inspect_asset(v['dn'], v.get('operator'))['install_length_m']+.02
            for v in a.deferred_valves if v['edge'] == run['edges'][0] and v['placement_rule'] == 'free'
        ), default=0.)
    run['radius'] = a.radius(run['dn'])
    ends = [run['start'], run['end']]
    kinds = [e['kind'] for e in ends]
    for spec in ends:
        if spec['kind'] == 'fixed' and spec['endpoint'] not in a.anchors:
            _record_failure(a, run, attempts, 'Port not formed: '+spec['endpoint'])
            return False
    if 'reducer_stub' in kinds:
        return _build_reducer_stub(a, run, ends, by_id, attempts)
    owners = []
    for spec in ends:
        if spec['kind'] == 'tap':
            owner = _tap_targets(a, runs, run, spec)
            if owner is None or not owner.get('route'):
                _record_failure(a, run, attempts, 'The receiving main was not built: '+spec['junction'])
                return False
            owners.append((spec, owner))
    taps = [spec for spec in ends if spec['kind'] == 'tap']
    if 'terminal' in kinds and kinds.count('terminal') == 2:
        _record_failure(a, run, attempts, 'Both ends are blind and have no physical port')
        return False
    heated = next((r for r in a.path_requirements if r['routing_rule'] == 'M3d'
                   and r['edges'][0] in run['edges']), None) if has_full_route_rules else None
    true_mouth = a.anchors.get(heated['target'], {}).get('point') if heated else None
    # Gather (start, end, tap_info) combinations; a fixed end is one option,
    # a tap end has candidate points on its built pipe.
    def options_for(spec, toward, *, toward_final=None):
        if spec['kind'] == 'fixed':
            return [(_anchor(a, spec, run['dn']), None)]
        if spec['kind'] == 'terminal':
            return [(None, None)]
        owner = next(o for s, o in owners if s is spec)
        result = []
        for tap in _tap_candidates(a, owner, spec['junction'], run['radius'], toward,
                                   toward_final=toward_final):
            for direction in _arrivals(tap['axis'], np.asarray(toward)-tap['point']):
                anchor = {'point': tap['point'], 'normal': direction, 'level': _level(a, tap['point'][2]),
                          'straight': tap['main_radius']+run['radius']+.05}
                result.append((anchor, {**tap, 'owner': owner['id'], 'junction': spec['junction']}))
        return result

    def reference(spec):
        if spec['kind'] == 'fixed':
            return np.asarray(a.anchors[spec['endpoint']]['point'], float)
        if spec['kind'] == 'tap':
            owner = next(o for s, o in owners if s is spec)
            route = next(r for r in a.routes if r['id'] == owner['route'])
            pts = np.asarray(route['points'])
            return pts[len(pts)//2]
        return None
    ref = [reference(ends[0]), reference(ends[1])]
    if 'terminal' in kinds:
        toward = ref[0] if ref[0] is not None else np.zeros(3)
        for anchor, tap in options_for(ends[0], toward+np.array([0., 0., 1.])):
            if not attempts.open:
                break
            extra = _tap_obstacles(a, [tap], run['dn']) if tap else ()
            before = len(attempts.feasible)
            _build_terminal(a, run, by_id, terminals, anchor, attempts, extra=extra,
                            skip_extra=(tap['component'],) if tap else ())
            for row in attempts.feasible[before:]:
                row[2].update(start_tap=tap, end_tap=None, start=anchor)
    else:
        starts = options_for(ends[0], ref[1] if ref[1] is not None else ref[0])
        toward = ref[0] if ref[0] is not None else ref[1]
        finals = options_for(ends[1], toward)
        def compatible(start_tap, end_tap):
            if not start_tap or not end_tap or start_tap['owner'] != end_tap['owner']:
                return True
            owner = next(o for _, o in owners if o['id'] == start_tap['owner'])
            junctions = _owned_junctions(owner)
            sign = 1 if junctions.index(start_tap['junction']) < junctions.index(end_tap['junction']) else -1
            return sign*(end_tap['s']-start_tap['s']) >= 2*(run['radius']+owner['radius'])+.05-EPS

        def priority(pair):
            return (sum(not tap['keeps_between_valve_room'] for _, tap in pair if tap),
                    sum(not tap['keeps_valve_room'] for _, tap in pair if tap),
                    float(np.abs(np.asarray(pair[0][0]['point'])-np.asarray(pair[1][0]['point'])).sum()))

        pairs = sorted(((s, e) for s in starts for e in finals if compatible(s[1], e[1])), key=priority)
        if true_mouth is not None and ends[1]['kind'] == 'tap':
            nearest = options_for(ends[1], toward, toward_final=true_mouth)
            near_pairs = sorted(((s, e) for s in starts for e in nearest if compatible(s[1], e[1])),
                                key=lambda pair: (math.dist(pair[1][0]['point'], true_mouth), priority(pair)))
            pairs, seen = [*pairs[:4], *near_pairs, *pairs[4:]], set()
            distinct = []
            for pair in pairs:
                key = tuple((tuple(anchor['point']), tuple(anchor['normal'])) for anchor, _ in pair)
                if key not in seen:
                    seen.add(key)
                    distinct.append(pair)
            pairs = distinct
        share = max(6, ATTEMPT_BUDGET//max(1, min(len(pairs), 8)))
        for (start, start_tap), (end, end_tap) in pairs:
            attempts.limit = attempts.used+share
            if not attempts.open:
                if attempts.used >= ATTEMPT_BUDGET or attempts.best() and not attempts.best()[0][0]:
                    break
                continue
            selected_taps = [tap for tap in (start_tap, end_tap) if tap]
            if has_full_route_rules:
                rack = branch_rack_constraint(a, run, runs, start, end, end_tap)
                attempts.rack_constraint = rack or ()

                def pair_guard(plan):
                    result = guard(plan)
                    if rack is None:
                        raise b.PipingRefused('FULL_ROUTE: could not trace the final vessel port')
                    if rack and not rack_candidate_ok(candidate_path(plan, run['dn']), *rack):
                        raise b.PipingRefused("FULL_ROUTE: the path to the final vessel port does not follow its level's rack")
                    return result

                attempts.check_candidate = pair_guard
            extra = _tap_obstacles(a, selected_taps, run['dn'])
            tap_points = [tap['point'] for tap in selected_taps]
            skipped = [tap['component'] for tap in selected_taps]
            before = len(attempts.feasible)
            attempts.want = (_valve_straight(a, run, ends[0], start), _valve_straight(a, run, ends[1], end))
            _point_approaches(a, run, by_id, start, end, attempts, extra=extra, tap_points=tap_points,
                              skip_extra=skipped)
            a._network_contact_points = []
            for row in attempts.feasible[before:]:
                row[2].update(start_tap=start_tap, end_tap=end_tap, start=start, end=end)
        attempts.limit = ATTEMPT_BUDGET
    best = attempts.best()
    if best is None:
        _record_failure(a, run, attempts, 'No candidate passed collision and complete-route checks' if has_full_route_rules
                        else 'No attempted orthogonal route was free of penetration')
        return False
    _, points, extra = best
    if extra.get('state') is not None:
        a.restore(extra['state'])
    run['want'] = tuple(_valve_straight(a, run, spec, extra.get(key) or {})
                        for spec, key in ((ends[0], 'start'), (ends[1], 'end')))
    # Split the built pipes at the chosen tee points, then emit this run.
    split_descendants = {}
    for key in ('start_tap', 'end_tap'):
        tap = extra.get(key)
        if not tap:
            continue
        owner = next(r for r in runs if r['id'] == tap['owner'])
        route = next(r for r in a.routes if r['id'] == owner['route'])
        original = tap['component']
        component = original
        if original in split_descendants:
            point = tap['point']
            candidate_ids = set(split_descendants[original]) & set(route['pipe_components'])
            component = next(c['id'] for c in a.scene['components'] if c['id'] in candidate_ids
                             and abs(sum(math.dist(end, point) for end in c['ends'])
                                     -math.dist(*c['ends'])) <= 1e-6)
        pieces = _split_pipe(a, route, component, tap['point'])
        split_descendants[original] = pieces
        if tap['junction'] not in owner['junctions']:
            _finish_reducer_tap(a, owner, route, tap, pieces)
        owner['resolved'][tap['junction']] = np.asarray(tap['point'])
    if 'terminal' in kinds:
        start = extra['start']
        end = {'point': np.asarray(points[-1]), 'level': _level(a, points[-1][2])}
        route = _emit(a, run, by_id, start, end, points)
        _finish_terminal(a, run, by_id, terminals, points)
    else:
        route = _emit(a, run, by_id, extra['start'], extra['end'], points)
    for key in ('start_tap', 'end_tap'):
        tap = extra.get(key)
        if tap:
            _nearby_contacts(a, route['components'][:1] if key == 'start_tap' else route['components'][-1:],
                             tap['point'])
            if not any(j['id'] == tap['junction'] for j in a.material['junctions']):
                a.material['junctions'].append({'id': tap['junction'], 'position': np.asarray(tap['point']).tolist(),
                    'incidents': [e['id'] for e in a.requirements['edges']
                                  if tap['junction'] in (_node(e['a']), _node(e['b']))],
                    'scope': 'The branch joins a physical straight on the existing main. Its route determines the tee position.'})
    a.material.setdefault('network_attempts', []).append({**attempts.report(), 'status': 'connected',
                                                          'route': route['id']})
    return True


# --------------------------------------------------------------------------
# Unresolved junctions and the per-edge split


def _default_positions(a, run):
    """A pass-through or failed-branch junction still needs one point on the run."""
    if not run.get('route'):
        return
    for index, uid in enumerate(run['junctions']):
        if uid in run['resolved']:
            continue
        table = _arc_table(a, run)   # re-read: an earlier cut replaced a straight
        low, high = 0., math.inf
        for j, other in enumerate(run['junctions']):
            if other not in run['resolved']:
                continue
            s = _arc_of(table, run['resolved'][other])
            if s is None:
                continue
            if j < index:
                low = max(low, s+.1)
            else:
                high = min(high, s-.1)
        total = sum(row['length'] for row in table)
        keep = run.get('want', (0., 0.))
        rows = []
        for lo, hi in ((max(low, keep[0]+.1), min(high, total-keep[1]-.1)), (low, high)):
            rows = [row for row in table if row['s0']+row['length'] > lo and row['s0'] < hi]
            if rows:
                low, high = lo, hi
                break
        if not rows:
            continue
        rows.sort(key=lambda row: min(row['s0']+row['length'], high)-max(row['s0'], low), reverse=True)
        row = rows[0]
        lo, hi = max(row['s0'], low), min(row['s0']+row['length'], high)
        point = row['from']+(row['to']-row['from'])*(((lo+hi)/2-row['s0'])/row['length'])
        route = next(r for r in a.routes if r['id'] == run['route'])
        _split_pipe(a, route, row['component'], point)
        run['resolved'][uid] = point
        a.material['junctions'].append({'id': uid, 'position': point.tolist(),
            'incidents': [e['id'] for e in a.requirements['edges'] if uid in (_node(e['a']), _node(e['b']))],
            'scope': "Boundary on the main. Without a branch, use the longest straight's midpoint."})


def split_runs(a, runs, by_id):
    """Give each original edge its own route between its two end points."""
    elbows = {e['id']: e for e in a.scene.get('elbows', [])}
    parts = {c['id']: c for c in a.scene['components']}
    for run in runs:
        if not run.get('route'):
            continue
        route = next(r for r in a.routes if r['id'] == run['route'])
        cuts = [np.asarray(run['resolved'][uid]) for uid in run['junctions'] if uid in run['resolved']]
        if len(run['edges']) == 1 or len(cuts) != len(run['junctions']):
            if len(run['edges']) == 1:
                route.update(b._meta(by_id[run['edges'][0]]))
            continue
        groups, current, k = [], [], 0
        for cid in route['components']:
            current.append(cid)
            c = parts[cid]
            if k < len(cuts) and c.get('kind') == 'pipe' and c.get('ends') and any(
                    math.dist(end, cuts[k]) <= 1e-6 for end in c['ends']) and len(current):
                # close when this pipe ends at the next tee point (in path order)
                other = [end for end in c['ends'] if math.dist(end, cuts[k]) > 1e-6]
                if other:
                    groups.append(current)
                    current, k = [], k+1
        groups.append(current)
        if len(groups) != len(run['edges']):
            a.material.setdefault('network_failures', []).append({
                'code': 'NETWORK_SPLIT_FAILED', 'run': run['id'], 'groups': len(groups), 'edges': len(run['edges'])})
            continue
        a.routes.remove(route)
        for edge_id, group in zip(run['edges'], groups):
            edge = by_id[edge_id]
            piece = {k: deepcopy(v) for k, v in route.items()
                     if k not in ('components', 'pipe_components', 'elbows', 'points', 'control_points', 'valves')}
            piece.update(b._meta(edge), id=route['id']+'_'+edge_id, components=group,
                         pipe_components=[c for c in group if parts[c].get('kind') == 'pipe'],
                         elbows=[e for e, rec in elbows.items() if rec.get('route_id') == route['id']
                                 and set(rec['components']) <= set(group)], valves=[])
            pts = []
            for cid in group:
                c = parts[cid]
                if c.get('kind') == 'pipe' and c.get('ends'):
                    ends = [list(map(float, e)) for e in c['ends']]
                    if pts and math.dist(ends[1], pts[-1]) < math.dist(ends[0], pts[-1]):
                        ends.reverse()
                    for e in ends:
                        if not pts or math.dist(e, pts[-1]) > 1e-8:
                            pts.append(e)
            for e in piece['elbows']:
                elbows[e]['route_id'] = piece['id']
            for cid in group:
                parts[cid]['route_id'] = piece['id']
            piece['points'] = pts
            piece['control_points'] = deepcopy(pts)
            piece['bend_count'] = len(piece['elbows'])
            piece['actual_planned_centerline_length_m'] = sum(math.dist(x, y) for x, y in zip(pts, pts[1:]))
            a.routes.append(piece)


# --------------------------------------------------------------------------
# Entry


def fixed_seals(a, edges):
    """A liquid seal low point is a confirmed bend shape, fixed by its return mouth."""
    from .piping_material_templates import _junction_position
    units = a.frozen['graph']['units']
    for uid, unit in units.items():
        if unit.get('kind') != 'junction' or unit.get('template_role', unit.get('service')) != 'liquid_seal':
            continue
        incidents = [e for e in edges if uid in (_node(e['a']), _node(e['b']))]
        if not incidents:
            continue
        try:
            shared = _junction_position(a, uid, unit, incidents, None)
        except (ValueError, *b.KNOWN_REFUSALS) as exc:
            a.material.setdefault('network_failures', []).append({'stage': 'junction', 'endpoint': uid,
                                                                  'reason': str(exc)})
            continue
        for name in unit['ports']:
            anchor = deepcopy(shared)
            if name in shared.get('port_normals', {}):
                anchor['normal'] = np.asarray(shared['port_normals'][name])
            a.anchors[uid+'.'+name] = anchor
        a.joints[uid] = shared
        a.material['junctions'].append({'id': uid, 'position': shared['point'].tolist(),
            'incidents': [e['id'] for e in incidents], 'scope': 'Lowest point of liquid trap, based on the return port and bend'})


def connect_network(a, edges, terminals):
    from .generation_progress import checkpoint
    units = a.frozen['graph']['units']
    fixed_seals(a, edges)
    free = {uid for uid, u in units.items() if u.get('kind') == 'junction'
            and u.get('template_role', u.get('service')) != 'liquid_seal'
            and any(uid in (_node(e['a']), _node(e['b'])) for e in edges)}
    runs, ranking = plan_network(a, edges, terminals, free)
    for run in runs:
        run['resolved'] = {}
    a.material['network_plan'] = {'ranking': ranking, 'runs': deepcopy(runs),
        'order': 'seal order; ready branches, mains that unlock branches, then other mains; DN descending, shorter first',
        'attempt_budget': ATTEMPT_BUDGET, 'tap_points_per_branch': TAP_POINTS}
    by_id = {e['id']: e for e in edges}

    def estimate(run):
        ends = []
        for spec in (run['start'], run['end']):
            if spec['kind'] == 'fixed' and spec['endpoint'] in a.anchors:
                ends.append((np.asarray(a.anchors[spec['endpoint']]['point']),))
            elif spec['kind'] == 'tap':
                owner = _tap_targets(a, runs, run, spec)
                if owner and owner.get('route'):
                    _, route = _run_components(a, owner)
                    ends.append(tuple(map(np.asarray, route['points'])))
        if len(ends) == 2:
            return min(float(np.abs(first-last).sum()) for first in ends[0] for last in ends[1])
        return 0.
    seals = {uid for uid, u in units.items() if u.get('kind') == 'junction'
             and u.get('template_role', u.get('service')) == 'liquid_seal'}

    def seal_order(run):
        """At a fixed seal: inflow first, then outflow, then the drain (main before branch)."""
        rank = 0
        for edge_id in run['edges']:
            edge = by_id[edge_id]
            if _node(edge['b']) in seals:
                rank = max(rank, 0)
            if _node(edge['a']) in seals:
                rank = max(rank, 2 if edge.get('role') == 'seal_drain' else 1)
        return rank
    owner_by_junction = {junction: run['id'] for run in runs for junction in _owned_junctions(run)}
    dependencies = {
        run['id']: {owner_by_junction.get(spec['junction']) for spec in (run['start'], run['end'])
                    if spec['kind'] == 'tap' or spec.get('large_edge') == spec['edge']}
        for run in runs
    }
    pending = list(runs)
    while pending:
        pending_ids = {run['id'] for run in pending}
        available = [run for run in pending if not dependencies[run['id']] & (pending_ids | {None})] or pending[:1]
        unlocks = {next(iter(waiting)) for run in pending
                   if None not in dependencies[run['id']] and
                   len(waiting := dependencies[run['id']] & pending_ids) == 1}

        def priority(run):
            identifier = run['id']
            branch = any(spec['kind'] == 'tap' for spec in (run['start'], run['end']))
            stage = 0 if branch else 1 if identifier in unlocks else 2
            return seal_order(run), stage, -run['dn'], estimate(run), identifier

        available.sort(key=priority)
        run = available[0]
        pending.remove(run)
        checkpoint(a, {'stage': 'material_pipe', 'run': run['id'], 'edges': run['edges']})
        from .piping_material_network import drain_bounds
        edge = by_id[run['edges'][0]]
        with drain_bounds(a, edge):
            try:
                build_run(a, runs, run, by_id, terminals)
            except b.KNOWN_REFUSALS as exc:
                _record_failure(a, run, None, b._why(exc))
        a.material.setdefault('pipe_emission_order', []).extend(run['edges'])
    for run in runs:
        try:
            _default_positions(a, run)
        except (StopIteration, ValueError, *b.KNOWN_REFUSALS) as exc:
            a.material.setdefault('network_failures', []).append({
                'code': 'NETWORK_JUNCTION_POSITION_FAILED', 'run': run['id'], 'reason': f'{type(exc).__name__}: {exc}'})
    split_runs(a, runs, by_id)
    a.material['network_runs'] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in run.items()
                                   if k != 'resolved'} | {'resolved': {u: np.asarray(p).tolist()
                                   for u, p in run['resolved'].items()}} for run in runs]
    return runs
