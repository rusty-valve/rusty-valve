"""Place shared utility pipes along the building and platform edges."""
from collections import Counter
from copy import deepcopy

import numpy as np

from .piping_elbows import coarse_radius
from .piping_service_groups import building_long_edge_axis
from .piping_station import _header_candidate_hits, _header_end_fittings, PipingStationRejected


EPS = 1e-8
MAX_CROSS_CANDIDATES = 96
MAX_LEVEL_ATTEMPTS = 4096  # Finite planning budgets, not installation limits.


def _required_top_rises(header, requirements, config, heights):
    """Other +Z service mouths must retain their original straight-up space."""
    own = set(header['point_ids'])
    excluded = {'bottom', 'bottom_root', 'drain', 'drain_root', 'pump_inlet', 'pump_outlet'}
    level = header['level']; zone = heights[level]
    for point in requirements['service_points']:
        if ((point['id'] in own and point['medium']!='offgas') or point['level'] != level
                or point.get('role') in excluded or point.get('port') in ('bottom', 'drain')
                or point.get('equipment_kind') == 'pump'):
            continue
        normal = np.asarray(point['normal'], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm <= EPS or np.max(abs(normal/norm-np.array([0., 0., 1.]))) > EPS:
            continue
        radius = config['pipe_outer_diameter_mm'][str(point['dn'])]/2000.
        yield {'requirement': point['id'], 'endpoint': point['endpoint'],
            'position': list(point['position']), 'pipe_dn': point['dn'],
            'source_dn': point['source_dn'], 'outer_radius_m': radius,
            'z_interval_m': [point['position'][2], max(point['position'][2],
                min(float(zone['top_z_m']), float(header['start'][2])))],
            'scope': 'Reserved required +Z rise from the original port'}


def _header_parts(header, config):
    spacing = config['pg_rules']['parallel_pipe_spacing']
    yield {'id': header['id']+'.body', 'ends': [header['start'], header['end']],
        'radius': header['outer_radius_m'], 'required_clearance_m':
            spacing['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.}
    for part in _header_end_fittings([header]):
        yield {**part, 'required_clearance_m': spacing['protrusions']['min_clearance_mm']/1000.}


def _rise_part_clearance(header, rise, part):
    """Distance from a vertical pipe to a finite horizontal pipe/end fitting.

    When the vertical straight covers the header centre plane, the closest
    points lie in that plane. The horizontal cylinder projects to an axial
    interval and its real OD; the vertical pipe projects to its real circle.
    Keeping the finite axial interval avoids treating flange ends as spheres.
    """
    axis = header['axis_index']; other = 1-axis; position = rise['position']
    ends = part['ends']; centre_z = float(ends[0][2]); radius = part['radius']
    z0, z1 = rise['z_interval_m']
    if z0 > centre_z+radius+rise['outer_radius_m']+part['required_clearance_m']:
        return None
    low, high = sorted((float(ends[0][axis]), float(ends[1][axis])))
    axial_gap = max(low-position[axis], position[axis]-high, 0.)
    cross_gap = max(abs(position[other]-ends[0][other])-radius, 0.)
    z_gap = max(z0-(centre_z+radius), (centre_z-radius)-z1, 0.)
    return float(np.linalg.norm([axial_gap, cross_gap, z_gap])-rise['outer_radius_m'])


def required_rise_hits(header, requirements, config, heights):
    """Public API for both initial headers and later main-T/end extensions.

    Own assigned tee mouths are normal joints. Other equipment/use identities
    retain original +Z pipe space with the existing 50/30 mm clearances.
    """
    hits = []
    for rise in _required_top_rises(header, requirements, config, heights):
        for part in _header_parts(header, config):
            clearance = _rise_part_clearance(header, rise, part)
            if clearance is None or clearance >= part['required_clearance_m']-EPS:
                continue
            hits.append({**rise, 'header': header['id'], 'header_part': part['id'],
                'planned_clearance_m': clearance, 'required_clearance_m': part['required_clearance_m'],
                'header_part_radius_m': part['radius'],
                'basis': 'Planned clearance between original +Z risers and branches or end flanges. Each matching tap has local contact allowance.'})
    return hits


def _rise_cross_candidates(header, requirements, config, heights):
    axis = header['axis_index']; other = 1-axis
    values = set()
    for rise in _required_top_rises(header, requirements, config, heights):
        for part in _header_parts(header, config):
            position = rise['position']
            low, high = sorted((part['ends'][0][axis], part['ends'][1][axis]))
            axial_gap = max(low-position[axis], position[axis]-high, 0.)
            bare = config['pg_rules']['parallel_pipe_spacing']['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.
            preferred = max(bare, part['required_clearance_m'])+coarse_radius(rise['outer_radius_m'])+rise['outer_radius_m']
            # The second tangent is only a first-turn-space preference. The
            # actual rejection threshold remains the original 50/30 mm.
            for clearance in (part['required_clearance_m'], preferred):
                clearance_radius = rise['outer_radius_m']+clearance
                if axial_gap >= clearance_radius:
                    continue
                cross = part['radius']+float(np.sqrt(max(0., clearance_radius**2-axial_gap**2)))
                values.update((position[other]-cross, position[other]+cross))
    return values


def _pitch(first, second, bare, protrusion):
    return max(first['outer_radius_m']+second['outer_radius_m']+bare,
               first['fitting_outer_radius_m']+second['fitting_outer_radius_m']+protrusion)


def _coincident_supply_tail_pairs(headers, requirements, config):
    """Find physically overlapping axial projections, without device names."""
    if not headers:
        return []
    axis = headers[0]['axis_index']; level = headers[0]['level']
    active = {pid: h for h in headers for pid in h['point_ids']}
    bare = config['pg_rules']['parallel_pipe_spacing']['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.
    points = [p for p in requirements['service_points'] if p['level'] == level and p['id'] in active
              and np.max(abs(np.asarray(p['normal'])-np.array([0., 0., 1.]))) <= EPS]
    supplies = [p for p in points if p['medium'] != 'offgas']
    tails = [p for p in points if p['medium'] == 'offgas']
    pairs = []
    for supply in supplies:
        for tail in tails:
            if supply['equipment'] != tail['equipment']:
                continue
            radii = [config['pipe_outer_diameter_mm'][str(p['dn'])]/2000. for p in (supply, tail)]
            axial_distance = abs(supply['position'][axis]-tail['position'][axis])
            if axial_distance > sum(radii)+bare+EPS:
                continue
            pairs.append({'supply': supply, 'tail': tail,
                'supply_header': active[supply['id']]['id'],
                'tail_header': active[tail['id']]['id'],
                'axial_distance_m': axial_distance,
                'axial_pipe_and_clearance_span_m': sum(radii)+bare})
    return pairs


def _nested_tail_preference(header, cross, selected, requirements, config, other, bare, protrusion):
    """Prefer an outer tail row after selecting a direct-side supply row.

    This orders existing candidates only. Both mouth coordinates and all
    installation thresholds stay fixed, and fallback candidates remain.
    """
    if header['medium'] != 'offgas':
        return 0, 0.
    members = [*selected, header]
    selected_map = {h['id']: h for h in selected}
    shortfalls = []
    for pair in _coincident_supply_tail_pairs(members, requirements, config):
        if pair['tail_header'] != header['id'] or pair['supply_header'] not in selected_map:
            continue
        supply = selected_map[pair['supply_header']]
        delta = supply['start'][other]-pair['supply']['position'][other]
        if abs(delta) <= EPS:
            continue
        direction = 1. if delta > 0 else -1.
        # The confirmed nested geometry has the tail mouth behind the supply
        # mouth and its descending join beyond the supply's horizontal end.
        if direction*(pair['tail']['position'][other]-pair['supply']['position'][other]) > EPS:
            continue
        outward = direction*(cross-supply['start'][other])
        shortfalls.append(max(0., _pitch(header, supply, bare, protrusion)-outward))
    return sum(v > EPS for v in shortfalls), sum(shortfalls)


def _distance_score(header, cross, points, other, bare, requirements, config, heights):
    """Prefer room for a direct first bend; never reject on this preference."""
    radius = max(header['method_branch_radii_m'])
    side_arm = coarse_radius(radius)+header['outer_radius_m']+3*radius+bare
    distances = [abs(cross-points[pid]['position'][other]) for pid in header['point_ids']]
    shortfalls = [max(0., side_arm-distance) for distance in distances]
    # Keep some of the existing wider candidates around another mouth's
    # first turn. This is a preference, not a changed clearance threshold.
    candidate = deepcopy(header)
    candidate['start'][other] = candidate['end'][other] = float(cross)
    rise_shortfalls = []
    for rise in _required_top_rises(candidate, requirements, config, heights):
        turn_room = coarse_radius(rise['outer_radius_m'])+rise['outer_radius_m']
        for part in _header_parts(candidate, config):
            clearance = _rise_part_clearance(candidate, rise, part)
            if clearance is not None:
                preferred = max(bare, part['required_clearance_m'])+turn_room
                rise_shortfalls.append(max(0., preferred-clearance))
    return (sum(v > EPS for v in shortfalls), sum(shortfalls),
            sum(v > EPS for v in rise_shortfalls), sum(rise_shortfalls),
            sum(distances), max(distances), cross)


def _cross_candidates(header, selected, points, scene, protected, other, bare, protrusion, wall,
                      requirements, config, heights):
    """Independent source/obstacle tangencies plus both actual frame limits."""
    fitting = header['fitting_outer_radius_m']
    low, high = fitting, float(scene['footprint'][other])-fitting
    if low > high+EPS:
        return []
    radius = max(header['method_branch_radii_m'])
    side_arm = coarse_radius(radius)+header['outer_radius_m']+3*radius+bare
    candidates = {low, high, (low+high)/2, float(header['start'][other])}
    candidates.update(_rise_cross_candidates(header, requirements, config, heights))
    for pid in header['point_ids']:
        cross = float(points[pid]['position'][other])
        candidates.update((cross, cross-side_arm, cross+side_arm))
    for previous in selected:
        pitch = _pitch(header, previous, bare, protrusion)
        cross = float(previous['start'][other])
        candidates.update((cross-pitch, cross+pitch))
    for _, box in protected:
        candidates.update((box[other]-fitting, box[other+3]+fitting))
    for component in scene['components']:
        if component.get('annotation_only'):
            continue
        box = component['box']
        gap = wall if component['kind'] in ('wall', 'column') else 0.
        candidates.update((box[other]-fitting-gap, box[other+3]+fitting+gap))
    def score(cross):
        nested = _nested_tail_preference(header, cross, selected, requirements, config, other, bare, protrusion)
        return (*nested, *_distance_score(header, cross, points, other, bare, requirements, config, heights))
    ranked = sorted((v for v in candidates if low-EPS <= v <= high+EPS), key=score)
    # Near-device scoring must not cut the two real outer-frame possibilities
    # from the budget. Many obstructed centre choices cannot consume them.
    boundaries = list(dict.fromkeys((low, high)))
    chosen = ranked[:max(0, MAX_CROSS_CANDIDATES-len(boundaries))]
    chosen.extend(v for v in boundaries if v not in chosen)
    return sorted(chosen, key=score)


def _place_level(headers, z, points, scene, protected, others, other, config, attempts,
                 requirements, heights):
    spacing = config['pg_rules']['parallel_pipe_spacing']
    bare = spacing['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.
    protrusion = spacing['protrusions']['min_clearance_mm']/1000.
    wall = spacing['outermost_pipe_or_insulation_to_column_or_wall_min_mm']/1000.
    # This is only a search order. Final transverse order is unrestricted.
    pairs = _coincident_supply_tail_pairs(headers, requirements, config)
    supply_first = {p['supply_header'] for p in pairs}
    ordered = sorted(headers, key=lambda h: (h['id'] not in supply_first, -h['dn'], h['medium']=='offgas',
        -len(h['point_ids']), str(h['medium'])))
    calls = 0
    backtracks = 0
    def place(index, selected):
        nonlocal calls, backtracks
        if index == len(ordered):
            return selected
        header = deepcopy(ordered[index])
        header['start'][2] = header['end'][2] = float(z)
        crosses = _cross_candidates(header, selected, points, scene, protected,
                                    other, bare, protrusion, wall, requirements, config, heights)
        for cross in crosses:
            if calls >= MAX_LEVEL_ATTEMPTS:
                return None
            calls += 1
            candidate = deepcopy(header)
            candidate['start'][other] = candidate['end'][other] = float(cross)
            candidate['start'][2] = candidate['end'][2] = float(z)
            hits = [previous['id']+'.transverse_clearance' for previous in selected
                    if abs(cross-previous['start'][other]) < _pitch(candidate, previous, bare, protrusion)-EPS]
            hits.extend(_header_candidate_hits(candidate, scene, protected, [*others, *selected], config))
            rise_hits = required_rise_hits(candidate, requirements, config, heights)
            hits.extend(h['requirement']+'.required_positive_z_rise' for h in rise_hits)
            attempts.append({'header': candidate['id'], 'medium': candidate['medium'],
                'centre_z_m': z, 'cross_coordinate_m': cross,
                'selected_media': [h['medium'] for h in selected], 'hits': list(dict.fromkeys(hits)),
                'required_rise_conflicts': rise_hits})
            if hits:
                continue
            found = place(index+1, [*selected, candidate])
            if found is not None:
                return found
            backtracks += 1
        return None
    selected = place(0, [])
    return selected, {'attempts': calls, 'backtracks': backtracks,
        'budget': MAX_LEVEL_ATTEMPTS, 'placement_order': [h['medium'] for h in ordered],
        'coincident_supply_tail_pairs': [{'supply_requirement': p['supply']['id'],
            'tail_requirement': p['tail']['id'], 'axial_distance_m': p['axial_distance_m'],
            'axial_pipe_and_clearance_span_m': p['axial_pipe_and_clearance_span_m']} for p in pairs],
        'supply_first_is_candidate_preference_only': True}


def arrange_parallel_gas_headers(layout, frozen, requirements, config, heights):
    """Compatibility API; applies to every public medium including supply/return."""
    result = deepcopy(layout)
    points = {p['id']: p for p in requirements['service_points']}
    scene = frozen['scene']
    ceiling = max(z['top_z_m'] for z in heights.values())
    protected = [(r['id'], [*r['rect'][:2], r['base_z'], *r['rect'][2:], ceiling])
                 for r in scene.get('pullout_reservations', []) if r.get('open_top')]
    bare = config['pg_rules']['parallel_pipe_spacing']['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.
    for level in sorted({h['level'] for h in result['headers']}):
        headers = [h for h in result['headers'] if h['level'] == level]
        identities = Counter(h['medium'] for h in headers)
        if any(count != 1 for count in identities.values()):
            raise PipingStationRejected('METHOD_DUPLICATE_PUBLIC_SERVICE',
                'Use one common branch per independent service and floor', {'level': level, 'counts': dict(identities)})
        axis, frame_basis = building_long_edge_axis(scene, level)
        other = 1-axis
        for header in headers:
            if header['axis_index'] != axis:
                raise PipingStationRejected('METHOD_HEADER_FRAME_AXIS',
                    "Use the building or platform boundary's long axis for the common branch",
                    {'header': header['id'], 'expected_axis_index': axis,
                     'actual_axis_index': header['axis_index']})
            header['method_branch_radii_m'] = [
                config['pipe_outer_diameter_mm'][str(points[pid]['dn'])]/2000.
                for pid in header['point_ids']]
        zone = heights[level]
        bottom, top = float(zone['bottom_z_m']), float(zone['top_z_m'])
        thickness = top-bottom
        radii = [r for h in headers for r in h['method_branch_radii_m']]
        turn_room = max(coarse_radius(r)+r for r in radii)
        fitting = max(h['fitting_outer_radius_m'] for h in headers)
        overhead = max(turn_room, fitting+bare+2*max(radii))
        low_z, high_z = bottom+max(turn_room, fitting), top-overhead
        if low_z > high_z+EPS:
            raise PipingStationRejected('METHOD_HEADER_TURN_BAND_CAPACITY',
                'Rack depth cannot fit the same-height branch, elbow and crossover stub',
                {'level': level, 'actual_rack_bottom_z_m': bottom,
                 'actual_rack_top_z_m': top, 'first_turn_room_m': turn_room,
                 'overhead_crossing_room_m': overhead})
        z_values = list(dict.fromkeys(float(np.clip(z, low_z, high_z)) for z in
            (bottom+thickness/4, bottom+thickness/2, low_z, high_z)))
        others = [h for h in result['headers'] if h['level'] != level]
        selected = None
        attempted = []
        search_rows = []
        for z in z_values:
            selected, search = _place_level(headers, z, points, scene, protected,
                                            others, other, config, attempted, requirements, heights)
            search_rows.append({'centre_z_m': z, **search})
            if selected is not None:
                break
        if selected is None:
            raise PipingStationRejected('METHOD_PARALLEL_SERVICE_ROWS_UNAVAILABLE',
                'No bounded lateral candidate fit same-height common branches',
                {'level': level, 'axis_index': axis, 'candidates': attempted,
                 'search': search_rows, 'physical_unsatisfiable_claim': False})
        transverse = sorted(selected, key=lambda h: h['start'][other])
        shared = {'axis_index': axis, 'frame_basis': frame_basis,
            'centre_z_m': selected[0]['start'][2],
            'media_order': [h['medium'] for h in transverse],
            'cross_coordinates_m': [h['start'][other] for h in transverse],
            'centre_spacings_m': [b['start'][other]-a['start'][other]
                                  for a,b in zip(transverse, transverse[1:])],
            'actual_rack_bottom_z_m': bottom, 'actual_rack_top_z_m': top,
            'first_turn_room_m': turn_room, 'overhead_crossing_room_m': overhead,
            'basis': 'Along the long boundary axis at the same floor and center height. Select each service offset by physical OD and flange clearance. Preserve outlet order.',
            'independent_transverse_positions': True, 'rigid_bundle': False,
            'all_public_services_included': True, 'fixed_equipment_coordinates': False,
            'equipment_and_ports_unchanged': True, 'search': search_rows,
            'candidate_count': len(attempted), 'candidates': attempted}
        for header in selected:
            header['method_first_turn_room_m'] = turn_room
            header['method_fixed_device_projections'] = True
            header['natural_device_taps'] = {}
            for pid in sorted(header['point_ids'], key=lambda pid: (
                    points[pid]['position'][axis], points[pid]['position'][other],
                    points[pid]['position'][2], *points[pid]['normal'], points[pid]['dn'])):
                position = list(header['start']); position[axis] = points[pid]['position'][axis]
                scalar = position[axis]-header['start'][axis]
                header['natural_device_taps'][pid] = {'position': position,
                    'source_projection_m': scalar, 'selected_projection_m': scalar}
            header['parallel_method_layout'] = deepcopy(shared)
            header['height_selection'] = {'pipe_bottom_m': header['start'][2]-header['outer_radius_m'],
                'common_center_z_m': header['start'][2], 'reason': 'method_v1.4_same_floor_common_center_z'}
        selected_map = {h['id']: h for h in selected}
        result['headers'] = [selected_map.get(h['id'], h) for h in result['headers']]
    return result
