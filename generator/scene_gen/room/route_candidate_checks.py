"""Check proposed connections and find usable pipe sections for valves."""
from collections import defaultdict
import math

import numpy as np

from .connection_evidence import direction
from .connection_rule_checks import measured_seat, rack_geometry

EPS = 1e-6
FITTING_CLEARANCE_M = .100
PIPE_JOINT_M = 10*EPS


def candidate_path(plan, dn):
    """The function consumes real tangent-to-tangent straights and elbows."""
    path = []
    for item in plan['ordered']:
        row = plan['runs' if item['kind'] == 'run' else 'elbows'][item['index']]
        path.append({'kind': 'pipe' if item['kind'] == 'run' else 'elbow',
                     'ends': [row['start'], row['end']], 'dn': dn,
                     'exit_axis': row.get('tangent_out'), 'entry_axis': row.get('tangent_in')})
    return path


def _distances(path):
    total = 0.
    for step in path:
        length = math.dist(*step['ends'])
        if length > EPS:
            yield total, total+length, step
            total += length


def pipe_length(path):
    return sum(math.dist(*step['ends']) for step in path)


def offset_of(path, point):
    for low, _, step in _distances(path):
        one, two = map(np.asarray, step['ends'])
        axis = direction(step['ends'])
        delta = np.asarray(point)-one
        t = float(delta @ axis)
        if -EPS <= t <= math.dist(one, two)+EPS and np.linalg.norm(delta-axis*t) <= EPS:
            return low+t
    return None


def trim_path(path, low, high):
    result = []
    for start, end, step in _distances(path):
        a, b = max(start, low), min(end, high)
        if b-a <= EPS:
            continue
        if abs(a-start) <= EPS and abs(b-end) <= EPS:
            result.append(step)
        else:
            origin = np.asarray(step['ends'][0])
            axis = direction(step['ends'])
            result.append({**step, 'ends': [(origin+axis*(t-start)).tolist() for t in (a, b)]})
    return result


def reverse_path(path):
    result = []
    for step in reversed(path):
        row = {**step, 'ends': step['ends'][::-1]}
        if step.get('entry_axis') is not None:
            row.update(entry_axis=[-x for x in step['exit_axis']],
                       exit_axis=[-x for x in step['entry_axis']])
        result.append(row)
    return result


def floor_window(ends, floors, level):
    """Usable axial distances on this straight, within the device port's floor."""
    start, end = map(np.asarray, ends)
    length = math.dist(start, end)
    if level is None or not floors:
        return 0., length
    low = floors[level]
    high = floors[level+1]-PIPE_JOINT_M if level+1 < len(floors) else math.inf
    dz = end[2]-start[2]
    if abs(dz) <= EPS:
        return (0., length) if low-EPS <= start[2] <= high else (1., 0.)
    a, b = sorted(((low-start[2])*length/dz, (high-start[2])*length/dz))
    return max(0., a), min(length, b)


def port_backups(path, reserved, source, destination, floors, levels):
    """Keep physical, ordered after-elbow options short of any other reserved valve."""
    if not floors:
        return
    steps = list(_distances(path))
    for index, span in enumerate(reserved):
        near = span.get('position_endpoint')
        if span['placement_rule'] != 'at_port' or near is None or near not in (source, destination):
            continue
        toward_end = near == source
        floor = levels.get(near)
        primary = next(i for i, (_, _, step) in enumerate(steps) if step['ends'] is span['section_ends_m'])
        if toward_end:
            stop = offset_of(path, reserved[index+1]['reserved_ends_m'][0]) if index+1 < len(reserved) else None
            following = range(primary+1, len(steps))
        else:
            stop = offset_of(path, reserved[index-1]['reserved_ends_m'][1]) if index else None
            following = range(primary-1, -1, -1)
        crossed = []
        span['alternatives'] = []
        for i in following:
            low, high, step = steps[i]
            if stop is not None and (low >= stop-EPS if toward_end else high <= stop+EPS):
                break
            if step['kind'] == 'elbow':
                crossed.append(step['ends'])
                continue
            if step['kind'] != 'pipe' or step['dn'] != span['dn']:
                break
            if not crossed:
                continue
            ends = step['ends']
            left, right = floor_window(ends, floors, floor)
            before = FITTING_CLEARANCE_M if i and steps[i-1][2]['kind'] == 'elbow' else PIPE_JOINT_M
            after = FITTING_CLEARANCE_M if i+1 < len(steps) and steps[i+1][2]['kind'] == 'elbow' else PIPE_JOINT_M
            left, right = max(left, before), min(right, high-low-after)
            if stop is not None:
                if toward_end:
                    right = min(right, stop-low)
                else:
                    left = max(left, stop-low)
            if right-left < span['length_m']-EPS:
                continue
            start = np.asarray(ends[0])
            axis = direction(ends)
            span['alternatives'].append({
                'valve':span['valve'], 'edge':span['edge'], 'dn':span['dn'],
                'length_m':span['length_m'], 'level':floor,
                'section_ends_m':ends, 'section_length_m':high-low,
                'reserved_ends_m':[(start+axis*t).tolist() for t in (left, right)],
                'inlet_range_m':[(start+axis*t).tolist() for t in (left, right-span['length_m'])],
                'crossed_elbow_ends_m':list(crossed),
            })


def assign_valve_spans(path, valves, source=None, destination=None, *, start_tee=False, end_tee=False,
                       floors=(), levels=None):
    """Reserve separate sliding intervals and keep the required clearance at fittings."""
    if not valves:
        return []
    levels = levels or {}
    by_dn = defaultdict(list)
    for i, (low, high, step) in enumerate(_distances(path)):
        if step['kind'] != 'pipe':
            continue
        left = (start_tee if i == 0 else path[i-1]['kind'] == 'elbow')
        right = (end_tee if i == len(path)-1 else path[i+1]['kind'] == 'elbow')
        before = FITTING_CLEARANCE_M if left else PIPE_JOINT_M
        after = FITTING_CLEARANCE_M if right else PIPE_JOINT_M
        by_dn[step['dn']].append((low, high, step, before, after))
    end_offset = pipe_length(path)
    cursor = 0.
    assigned = defaultdict(list)
    for valve in valves:
        vertical = valve['direction'] == 'vertical'
        near = valve['position_endpoint']
        candidates = by_dn[valve['dn']]
        anchored = valve.get('role') == 'condenser_vapor' and bool(floors)
        if anchored and (near is None or near not in (source, destination) or near not in levels):
            return None
        level = levels[near] if anchored else None
        if anchored:
            steps = list(_distances(path))
            allowed = set()
            for low, _, step in (reversed(steps) if near == destination else steps):
                if step['kind'] == 'elbow':
                    continue
                if step['kind'] != 'pipe' or step['dn'] != valve['dn']:
                    break
                allowed.add(low)
            candidates = [row for row in candidates if row[0] in allowed]
            if near == destination:
                candidates.reverse()
        if near == source and vertical:
            candidates = [row for row in candidates if row[0] <= EPS]
        elif near == destination and vertical:
            candidates = [row for row in candidates if row[1] >= end_offset-EPS]
        if valve['placement_rule'] == 'before_seal':
            down = [low for low, _, step in _distances(path) if step['kind'] == 'elbow'
                    and step.get('entry_axis') is not None and abs(step['entry_axis'][2]) < EPS
                    and step['exit_axis'][2] < -.9]
            if not down:
                return None
            candidates = [row for row in candidates if row[1] <= down[0]+EPS]
        for low, high, part, before, after in candidates:
            left, right = floor_window(part['ends'], floors, level)
            left, right = max(left, before), min(right, high-low-after)
            begin = max(cursor, low+left)
            if low+right-begin < valve['length_m']-EPS:
                continue
            size = measured_seat([part['ends']], valve['length_m'], vertical)
            if size['fits']:
                assigned[low].append((valve, high, part, before, after, level, left, right))
                cursor = begin + valve['length_m']
                break
        else:
            return None
    seats = []
    for low, rows in sorted(assigned.items()):
        _, high, part, before, after, *_ = rows[0]
        slack = max(0., (high-low-before-after-sum(row[0]['length_m'] for row in rows))/len(rows))
        origin = np.asarray(part['ends'][0])
        axis = direction(part['ends'])
        start = before
        windows = None
        if any(row[5] is not None for row in rows):
            # Split the straight without extending any reservation across a floor.
            earliest, latest = [], []
            end = before
            for valve, _, _, _, _, _, left, _ in rows:
                begin = max(end, left)
                earliest.append(begin)
                end = begin + valve['length_m']
            begin = high-low-after
            for valve, _, _, _, _, _, _, right in reversed(rows):
                end = min(begin, right)
                begin = end-valve['length_m']
                latest.append(begin)
            latest.reverse()
            if any(a > b+EPS for a, b in zip(earliest, latest)):
                return None
            centers = [(a+b)/2 for a, b in zip(earliest,latest)]
            limits = [before, *((t+row[0]['length_m']+u)/2
                       for t,u,row in zip(centers,centers[1:],rows)), high-low-after]
            windows = [(max(row[6], a), min(row[7], b)) for row,a,b in zip(rows,limits,limits[1:])]
        for i, (valve, _, _, _, _, level, _, _) in enumerate(rows):
            end = start+valve['length_m']+slack
            lower, upper = (start, end) if windows is None else windows[i]
            minimum, maximum = lower, upper-valve['length_m']
            left = min(before, lower) if i == 0 else 0.
            right = min(after, high-low-upper) if i == len(rows)-1 else 0.
            seats.append({'valve': valve['id'], 'edge': valve['edge'], 'dn': valve['dn'],
                          'length_m': valve['length_m'], 'direction': valve['direction'],
                          'placement_rule': valve['placement_rule'], 'operator': valve['operator'],
                          'position_endpoint':valve.get('position_endpoint'), 'level':level,
                          'section_ends_m': part['ends'], 'ordered_floor_m': minimum,
                          'section_length_m': high-low, 'end_clearances_m': [left, right],
                          'reserved_ends_m': [(origin+axis*x).tolist() for x in (minimum-left, upper+right)],
                          'inlet_range_m': [(origin+axis*x).tolist() for x in (minimum, maximum)]})
            start = end
    port_backups(path, seats, source, destination, floors, levels or {})
    return seats


def _edge_valves(demands, edge_id, reverse=False):
    values = demands.get(edge_id, ())
    return list(reversed(values)) if reverse else list(values)


def run_valve_spans(path, run, demands, junction_positions, *, floors=(), levels=None):
    """Apply tee results as cuts, keeping all valves on their own side of a tee."""
    cuts = [(0, 0.)]
    for i, joint in enumerate(run.get('junctions', []), 1):
        at = junction_positions.get(joint)
        if at is None:
            continue
        offset = offset_of(path, at)
        if offset is None or offset+EPS < cuts[-1][1]:
            return None
        cuts.append((i, offset))
    cuts.append((len(run['edges']), pipe_length(path)))
    reserved = []
    for (a, low), (b, high) in zip(cuts, cuts[1:]):
        required = [v for i in range(a, b) for v in _edge_valves(demands, run['edges'][i], run['reversed'][i])]
        path_slice = trim_path(path, low, high)
        source = run['start']['endpoint'] if a == 0 else None
        dest = run['end']['endpoint'] if b == len(run['edges']) else None
        slots = assign_valve_spans(path_slice, required, source, dest,
                                   start_tee=a != 0 or run['start']['kind'] == 'tap',
                                   end_tee=b != len(run['edges']) or run['end']['kind'] == 'tap',
                                   floors=floors, levels=levels)
        if slots is None:
            return None
        reserved.extend(slots)
    return reserved


def rack_candidate_ok(first_path, suffix, band, final):
    path = [*first_path, *suffix]
    if any(math.dist(a['ends'][1], b['ends'][0]) > EPS for a, b in zip(path, path[1:])):
        return False
    evidence = rack_geometry(path, band, final)
    return evidence is not None and evidence['pass']


def built_path(a, run):
    route = next(route for route in a.routes if route['id'] == run['route'])
    from .connection_evidence import line_ends
    by_id = {part['id']: part for part in a.scene['components']}
    elbow_sections = {}
    for elbow in a.scene.get('elbows', ()):
        if elbow['route_id'] == route['id']:
            elbow_sections.update(dict.fromkeys(elbow['components'], elbow))
    result, seen = [], set()
    cursor = route['points'][0]
    for identifier in route['components']:
        piece = elbow_sections.get(identifier, by_id.get(identifier, {}))
        part_id = piece.get('id')
        if part_id in seen or not piece:
            continue
        ends = line_ends(piece)
        if ends is None or math.dist(*ends) <= EPS:
            continue
        seen.add(part_id)
        kind = 'elbow' if identifier in elbow_sections else piece['kind']
        row = {'kind': kind, 'ends': ends, 'dn': piece.get('dn', run['dn'])}
        if kind == 'elbow':
            row.update(entry_axis=piece['ports']['in']['tangent'], exit_axis=piece['ports']['out']['tangent'])
        if math.dist(cursor, ends[1]) < math.dist(cursor, ends[0]):
            row, = reverse_path([row])
        result.append(row)
        cursor = row['ends'][1]
    return result


def branch_rack_constraint(a, run, runs, start, end, end_tap):
    heated = [row for row in a.path_requirements if row['routing_rule'] == 'M3d' and row['edges'][0] in run['edges']]
    for row in heated:
        final = row['target']
        target = a.anchors.get(final)
        if target is None:
            return None
        level = target['level']
        if start['level'] >= level:
            continue
        zone = a.plan['rack_zones'][level]
        band = (zone['bottom_z_m'], zone['top_z_m'])
        if end_tap:
            parent = next(other for other in runs if other['id'] == end_tap['owner'])
            path = built_path(a, parent)
            if parent['end']['endpoint'] == final:
                pass
            elif parent['start']['endpoint'] == final:
                path = reverse_path(path)
            else:
                return None
            tap_offset = offset_of(path, end['point'])
            if tap_offset is None:
                return None
            suffix = trim_path(path, tap_offset, pipe_length(path))
        elif run['end']['endpoint'] == final:
            suffix = []
        else:
            return None
        mouth = a.requirements['ports'].get(final, {}).get('position', target['point'])
        return suffix, band, mouth
    return ()


def valve_spans_for_run(a, path, run, positions):
    demands = dict(a.edge_valve_demands)
    separate = []
    for edge in run['edges']:
        earlier = [r for r in a.routes if r.get('source_edge') == edge and r.get('role') == 'vessel_bottom_side']
        roots = [v for v in demands.get(edge, ()) if v['role'] == 'bottom_root']
        if not (earlier and roots):
            continue
        side_path = built_path(a, {'route': earlier[0]['id'], 'dn': earlier[0]['dn']})
        slots = assign_valve_spans(side_path, roots)
        if slots is None:
            return None
        separate.extend(slots)
        demands[edge] = [v for v in demands[edge] if v not in roots]
    levels = {endpoint: record['level'] for endpoint, record in a.anchors.items() if 'level' in record}
    core = run_valve_spans(path, run, demands, positions, floors=a.floors, levels=levels)
    return [*separate, *core] if core is not None else None


def parent_tee_spans(a, runs, taps):
    proposed = defaultdict(dict)
    for tap in taps:
        if tap is not None:
            proposed[tap['owner']][tap['junction']] = tap['point']
    updated = {}
    for identifier, positions in proposed.items():
        parent = next(r for r in runs if r['id'] == identifier)
        path = built_path(a, parent)
        reserved = valve_spans_for_run(a, path, parent, {**parent['resolved'], **positions})
        if reserved is None:
            return None
        updated[identifier] = reserved
    return updated


def contains_span(ends, interval):
    start, end = map(np.asarray, ends)
    axis = direction(ends)
    delta = np.asarray(interval)-start
    offsets = delta@axis
    return bool(np.all(offsets >= -EPS) and np.all(offsets <= math.dist(start, end)+EPS)
                and np.all(np.linalg.norm(delta-offsets[:, None]*axis, axis=1) <= EPS))


def publish_valve_spans(a, spans, *, route=None):
    """Bind the accepted coordinates to the emitted straight, after tee splits."""
    parts = {part['id']: part for part in a.scene['components'] if part['kind'] == 'pipe'}
    elbow_ids = {frozenset(tuple(np.round(e['ports'][side]['position'], 6)) for side in ('in','out')): e['id']
                 for e in a.scene.get('elbows', ())}
    failures = []

    def locate(span):
        routes = [route] if route else [r for r in a.routes if r.get('source_edge') == span['edge']
                                        or span['edge'] in r.get('source_edges', ())]
        hosts = [(r, cid) for r in routes for cid in r['components'] if cid in parts
                 and parts[cid].get('dn') == span['dn']
                 and contains_span(parts[cid]['ends'], span['reserved_ends_m'])]
        return hosts[0] if len(hosts) == 1 else (None, None)

    for span in spans:
        host, cid = locate(span)
        if host is None:
            failures.append({'code':'RESERVED_PIPE_MISSING', 'edge':span['edge'],
                             'valve':span['valve'], 'reason':'Reserved interval does not map to a unique emitted straight'})
            continue
        options = []
        for alternative in span.get('alternatives', ()):
            other, name = locate(alternative)
            elbows = [elbow_ids.get(frozenset(tuple(np.round(p, 6)) for p in ends))
                      for ends in alternative['crossed_elbow_ends_m']]
            if other is None or None in elbows:
                failures.append({'code':'ALTERNATE_PIPE_MISSING', 'edge':span['edge'],
                                 'valve':span['valve'], 'reason':'Alternative section does not map to emitted straights or elbows'})
                continue
            options.append({**alternative, 'pipe':name, 'route':other['id'], 'crossed_elbows':elbows})
        host.setdefault('reserved_valve_spans', []).append({**span, 'pipe':cid, 'route':host['id'], 'alternatives':options})
    return failures
