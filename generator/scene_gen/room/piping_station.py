"""Find space for utility risers and their pipe supports."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import product
import math
import random

import numpy as np

from .evaluation_common import require_environment
from .recipe import CandidateRefused, InputRejected


class PipingStationRejected(CandidateRefused):
    def __init__(self, code, detail, evidence=None):
        self.code = code
        self.evidence = evidence or {}
        super().__init__(f'{code}: {detail}')


# Same explicitly labelled coarse flange/blind outline as the P1 builder and
# frozen utilities.blind. It is geometry, not a new installation clearance.
FITTING_RADIAL_EXTRA_M = .034
FITTING_SOURCE = 'Approximate 34 mm radial extension from the existing blind and P1 flange geometry'
# Existing build_skeleton geometry: 2 * 18mm flange plates + 3mm gap,
# followed by its 18mm blind. These are fitting dimensions, not clearances.
HEADER_FLANGE_AXIAL_LENGTH_M = .039
HEADER_BLIND_THICKNESS_M = .018


def _overlap(a, b, eps=1e-8):
    return all(min(a[i+3], b[i+3])-max(a[i], b[i]) > eps for i in range(3))


def _frame(edge, width, depth):
    origin, tangent, inward, length, normal_length = {
        'y_min': ([0., 0.], [1., 0.], [0., 1.], width, depth),
        'y_max': ([0., depth], [1., 0.], [0., -1.], width, depth),
        'x_min': ([0., 0.], [0., 1.], [1., 0.], depth, width),
        'x_max': ([width, 0.], [0., 1.], [-1., 0.], depth, width),
    }[edge]
    return {'edge': edge, 'origin': np.asarray(origin), 'tangent': np.asarray(tangent),
            'inward': np.asarray(inward), 'length': length, 'normal_length': normal_length}


def _world(frame, t, n, z):
    xy = frame['origin']+frame['tangent']*t+frame['inward']*n
    return [float(xy[0]), float(xy[1]), float(z)]


def _local(frame, point):
    delta = np.asarray(point[:2])-frame['origin']
    return float(delta@frame['tangent']), float(delta@frame['inward'])


def _box(frame, t0, n0, z0, t1, n1, z1):
    corners = np.asarray([_world(frame, t, n, z) for t, n, z in
                          product((t0, t1), (n0, n1), (z0, z1))])
    return [*corners.min(axis=0).tolist(), *corners.max(axis=0).tolist()]


def _local_box(frame, box):
    points = [_local(frame, [x, y]) for x, y in product((box[0], box[3]), (box[1], box[4]))]
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def _edge_frame_parts(frame, components):
    parts = []
    for c in components:
        if c['kind'] not in ('column', 'beam'):
            continue
        local = _local_box(frame, c['box'])
        if not local[1]-1e-8 <= 0. <= local[3]+1e-8:
            continue
        if c['kind'] == 'column' or local[2]-local[0] >= frame['length']-1e-8:
            parts.append((c, local))
    return parts


def _radius(config, dn):
    try:
        value = float(config['pipe_outer_diameter_mm'][str(dn)])/2000
    except (KeyError, TypeError, ValueError) as error:
        raise InputRejected(f'P1_PIPE_OD_MISSING: DN{dn}No declared physical pipe OD') from error
    if not math.isfinite(value) or value <= 0:
        raise InputRejected(f'P1_PIPE_OD_INVALID: DN{dn}Invalid physical outside diameter')
    return value


def _heights(scene, pg):
    cfg = pg['distribution']['rack_zone']; floors = scene['floors']
    levels = {}
    for level, base in enumerate(floors):
        if level == len(floors)-1:
            bottom = base+cfg['top_floor_bottom_above_floor_m']
            top = bottom+cfg['thickness_m']; beam_ids = []
            source = 'PG-04: top rack relative to the actual floor and rack depth'
        else:
            above = floors[level+1]
            beams = [c for c in scene['components'] if c['kind'] in ('beam', 'equipment_beam')
                     and base < c['box'][2] < above and c['box'][5] <= above+1e-8]
            if not beams:
                raise PipingStationRejected('P1_RACK_BEAM_UNAVAILABLE', f'{level} has no verifiable upper beam underside')
            top = min(c['box'][2] for c in beams)
            bottom = top-cfg['thickness_m']; beam_ids = [c['id'] for c in beams if abs(c['box'][2]-top) <= 1e-8]
            source = 'Use the lowest actual face of the upper beam'
        if bottom <= base:
            raise PipingStationRejected('P1_RACK_HEIGHT_UNAVAILABLE', f'{level} has no physical room at rack elevation')
        levels[level] = {'bottom_z_m': float(bottom), 'top_z_m': float(top),
                         'beam_ids': beam_ids, 'source': source}
    return levels


def _protected_boxes(scene, ceiling):
    from .process import RULES
    policy = RULES['geometry_screen']; boxes = []
    for level, base in enumerate(scene['floors']):
        entry = scene.get('navigation_entries', {}).get(str(level))
        if entry:
            hw, hd = policy['walk_width_m']/2, policy['walk_depth_m']/2
            boxes.append((f'floor{level}.stair_entry',
                          [entry[0]-hw, entry[1]-hd, base,
                           entry[0]+hw, entry[1]+hd, base+policy['walk_height_m']]))
    for row in scene.get('pullout_reservations', []):
        if row.get('open_top') and row['base_z'] < ceiling:
            rect = row['rect']
            boxes.append((row['id'], [rect[0], rect[1], row['base_z'], rect[2], rect[3], ceiling]))
    return boxes


def _pipe_box(a, b, radius):
    # Here every station/header line is aligned to one world axis.
    axis = np.asarray(b)-a; length = float(np.linalg.norm(axis))
    if length <= 1e-8:
        return [a[i]-radius for i in range(3)]+[a[i]+radius for i in range(3)]
    extent = radius*np.sqrt(np.maximum(0., 1.-(axis/length)**2))
    return [*(np.minimum(a, b)-extent).tolist(), *(np.maximum(a, b)+extent).tolist()]


def _solid_hits(box, components, protected, ignore_floors=False):
    hits = [c['id'] for c in components if not c.get('annotation_only')
            and not (ignore_floors and c['kind'] == 'floor') and _overlap(box, c['box'])]
    hits.extend(name for name, obstacle in protected if _overlap(box, obstacle))
    return hits


def _main_options(requirements, config):
    services = requirements['active_services']; selected = requirements.get('service_templates', {})
    options, templates = [], {}
    for medium in services:
        template = deepcopy(selected.get(medium, config['templates']['services'][medium]))
        templates[medium] = template
        main = template.get('main_dn')
        dns = [main] if main is not None else list(template.get('main_dn_candidates', []))
        if not dns:
            raise InputRejected(f'P1_MAIN_DN_MISSING: {medium}No main DN candidates')
        for dn in dns:
            _radius(config, dn)
        _radius(config, template['branch_dn']); _radius(config, template['temporary_dn'])
        # Large water mains require the already-declared shared real demand.
        if medium in ('water', 'water_return') and any(dn in (200, 250, 300) for dn in dns):
            owners = {p['equipment'] for p in requirements['service_points'] if p['medium'] == medium}
            if len(owners) < 2:
                raise InputRejected(f'P1_LARGE_MAIN_WITHOUT_USERS: {medium}Large main has fewer than two physical service users')
        options.append(dns)
    return templates, [dict(zip(services, values)) for values in product(*options)]


def _anchors(frame, services, radii, points, components, direction, wall_gap, bare_gap, protrusion_gap):
    offsets = {}; previous = None
    for medium in services:
        fit = radii[medium]+FITTING_RADIAL_EXTRA_M
        if previous is None:
            offsets[medium] = fit
        else:
            pitch = max(radii[previous]+radii[medium]+bare_gap,
                        radii[previous]+FITTING_RADIAL_EXTRA_M+fit+protrusion_gap)
            offsets[medium] = offsets[previous]+pitch
        previous = medium
    span = offsets[services[-1]]+radii[services[-1]]+FITTING_RADIAL_EXTRA_M
    columns = [_local_box(frame, c['box']) for c in components if c['kind'] == 'column']
    near = max((b[2] for b in columns if b[0] <= 0. <= b[2]), default=0.)
    far = min((b[0] for b in columns if b[0] <= frame['length'] <= b[2]), default=frame['length'])
    low, high = near+wall_gap, far-wall_gap-span
    if low > high+1e-8:
        return [], offsets, span
    values = {low, high, (low+high)/2}
    projections = [_local(frame, p['position'])[0] for p in points]
    if projections:
        values.add(float(np.median(projections))-span/2)
    for c in components:
        box = _local_box(frame, c['box'])
        values.update((box[0]-wall_gap-span, box[2]+wall_gap))
    values = [float(value) for value in values if low-1e-8 <= value <= high+1e-8]
    values.sort(reverse=direction < 0)
    return values, offsets, span


def _member_segments(frame, axis, low, high, other_low, other_high, z0, z1, columns):
    """Split an axis-aligned rack member at actual intersected column faces."""
    def bounds(a, b):
        return (_box(frame, a, other_low, z0, b, other_high, z1) if axis == 0 else
                _box(frame, other_low, a, z0, other_high, b, z1))
    full = bounds(low, high); intervals = [(low, high)]
    for column in columns:
        if not _overlap(full, column['box']):
            continue
        local = _local_box(frame, column['box']); start, end = local[axis], local[axis+2]
        pieces = []
        for a, b in intervals:
            if end <= a or start >= b:
                pieces.append((a, b))
            else:
                if start-a > 1e-8: pieces.append((a, start))
                if b-end > 1e-8: pieces.append((end, b))
        intervals = pieces
    return [(a, b, bounds(a, b)) for a, b in intervals]


def _column_faces(box, columns):
    contacts = []
    for column in columns:
        other = column['box']
        for axis in range(3):
            if not all(min(box[j+3], other[j+3])-max(box[j], other[j]) > 1e-8
                       for j in range(3) if j != axis):
                continue
            for face, opposing in ((box[axis], other[axis+3]), (box[axis+3], other[axis])):
                if abs(face-opposing) <= 1e-8:
                    contacts.append({'column': column['id'], 'axis': axis, 'coordinate_m': float(face)})
    return contacts


def _header_end_fittings(headers):
    """Real header-end flange/cap envelopes, including their axial extent."""
    fittings = []
    for header in headers:
        start = np.asarray(header['start'], dtype=float)
        end = np.asarray(header['end'], dtype=float)
        length = float(np.linalg.norm(end-start))
        if length <= 1e-8:
            continue
        axis = (end-start)/length
        radius = header.get('fitting_outer_radius_m', header['outer_radius_m']+FITTING_RADIAL_EXTRA_M)
        for suffix, first, last in (
                ('end', end-axis*HEADER_FLANGE_AXIAL_LENGTH_M, end+axis*HEADER_BLIND_THICKNESS_M),
                ('start', start-axis*HEADER_BLIND_THICKNESS_M, start+axis*HEADER_FLANGE_AXIAL_LENGTH_M)):
            fittings.append({'id': header['id']+'_'+suffix+'_fitting_envelope',
            'kind': 'planned_header_end_fitting', 'shape': 'cylinder',
            'ends': [first.tolist(), last.tolist()], 'radius': radius,
            'box': _pipe_box(first, last, radius), 'header': header['id'], 'dn': header['dn'],
            'flange_axial_length_m': HEADER_FLANGE_AXIAL_LENGTH_M,
            'blind_thickness_m': HEADER_BLIND_THICKNESS_M,
            'source': 'Allow 39 mm for the physical flange and 18 mm for the blind at both ends'})
    return fittings


def _shared_supports(frame, level, zone, headers, risers, scene, pg, protected, all_headers=None):
    """One shared frame per active level, using existing edge columns as posts."""
    from .layout import add_box
    cfg = pg['distribution']['rack_support']; active = [h for h in headers if h['active']]
    if not active:
        return [], {'level': level, 'status': 'not_needed', 'active_headers': 0}
    half = cfg['column_width_m']/2; bw = cfg['crossbeam_width_m']/2
    top = zone['bottom_z_m']-cfg['seat_min_height_m']; bottom = top-cfg['crossbeam_depth_m']
    columns = [c for c in scene['components'] if c['kind'] == 'column']
    # A second building-axis rack direction meets existing shared members at
    # actual cut faces; it never relies on rack-owner collision exemptions.
    joins = columns+[c for c in scene['components'] if c.get('owner') == 'P1_shared_rack' and c['kind'] == 'beam']
    join_ids = {c['id'] for c in joins}
    axis_index = int(np.argmax(abs(frame['tangent'])))
    prefix = f'P1_rack_L{level}_A{axis_index}'
    attachments = []
    for side in (0., frame['normal_length']):
        matches = [c['id'] for c in columns if c['box'][2] <= bottom and c['box'][5] >= top
                   and _local_box(frame, c['box'])[1]-1e-8 <= side <= _local_box(frame, c['box'])[3]+1e-8]
        if len(matches) < 2:
            raise PipingStationRejected('P1_RACK_SUPPORT_UNAVAILABLE', f'{level} lacks opposing physical columns for support')
        attachments.append({'edge_local_n_m': side, 'existing_columns': matches})
    proposed = {'components': []}; steel = (.24, .31, .36)
    all_headers = active if all_headers is None else [h for h in all_headers if h['active']]
    fittings = _header_end_fittings(all_headers)
    pipe_obstacles = [{'id':h['id']+'_planned_body','kind':'planned_header_body',
        'box':_pipe_box(h['start'],h['end'],h['outer_radius_m'])} for h in all_headers]
    source_obstacles = [c for c in scene['components'] if c['kind'] != 'floor']+fittings+pipe_obstacles
    non_column_obstacles = [c for c in source_obstacles if c['id'] not in join_ids]
    for index, side in enumerate((0., frame['normal_length'])):
        pieces = _member_segments(frame, 0, 0., frame['length'], side-half, side+half,
                                 bottom-cfg['edge_girder_depth_m'], bottom, joins)
        if not pieces:
            raise PipingStationRejected('P1_RACK_GIRDER_OBSTRUCTED', f'{level} has no physical span between columns for the shared edge beam')
        for segment, (_, _, box) in enumerate(pieces):
            if _solid_hits(box, source_obstacles, protected):
                raise PipingStationRejected('P1_RACK_GIRDER_OBSTRUCTED', f'{level} shared edge beam segments intersect a solid or reserved envelope')
            member = add_box(proposed, f'{prefix}_edge{index}_segment{segment}', box, 'beam', steel, 'P1_shared_rack')
            member['column_face_contacts'] = _column_faces(box, columns)
    spans = {h['id']: sorted((_local(frame, h['start'])[0], _local(frame, h['end'])[0])) for h in active}
    choices = set(scene['shell']['x_axes_m'] if abs(frame['tangent'][0]) > .5 else scene['shell']['y_axes_m'])
    for lo, hi in spans.values():
        choices.update((lo+(hi-lo)/3, lo+2*(hi-lo)/3, (lo+hi)/2))
    for c in source_obstacles:
        b = _local_box(frame, c['box']); choices.update((b[0]-bw, b[2]+bw))
    # Retain the frozen spatial.build_shared_rack short-span rule, rather
    # than demanding two distinct supports on every tiny terminal span.
    required = {name: min(cfg['crossbeam_count'], max(1, int(
        (hi-lo-2*cfg['crossbeam_width_m'])/.4)+1)) for name, (lo, hi) in spans.items()}
    counts = {h['id']: 0 for h in active}
    selected = []; riser_boxes = [_pipe_box([*r['xy'], r['bottom_z']], [*r['xy'], r['top_z']], r['outer_radius_m']) for r in risers]
    while any(count < required[name] for name, count in counts.items()):
        candidates = []
        for t in sorted(choices):
            if not bw < t < frame['length']-bw or any(abs(t-v) <= 2*bw for v in selected):
                continue
            box = _box(frame, t-bw, -half, bottom, t+bw, frame['normal_length']+half, top)
            if _solid_hits(box, non_column_obstacles, protected) or any(_overlap(box, b) for b in riser_boxes):
                continue
            pieces = _member_segments(frame, 1, -half, frame['normal_length']+half,
                                     t-bw, t+bw, bottom, top, joins)
            if not pieces or any(_solid_hits(piece, source_obstacles, protected) for _, _, piece in pieces):
                continue
            covered = [name for name, (lo, hi) in spans.items() if lo+bw < t < hi-bw]
            # Check the same seats before selecting a crossbeam. A beam can
            # clear a fixed rising pipe below it while its later seat would
            # enter that pipe's final elbow or endpoint clearance envelope.
            seats_blocked=False
            for name in covered:
                header=next(h for h in active if h['id']==name)
                n=_local(frame,header['start'])[1]
                seat=_box(frame,t-bw,n-bw,top,t+bw,n+bw,
                          header['start'][2]-header['outer_radius_m'])
                if _solid_hits(seat,source_obstacles,protected):
                    seats_blocked=True;break
            if seats_blocked:continue
            score = sum(counts[name] < required[name] for name in covered)
            if score:
                candidates.append((score, min((abs(t-v) for v in selected), default=frame['length']), t, pieces, covered))
        if not candidates:
            raise PipingStationRejected('P1_RACK_CROSSBEAM_UNAVAILABLE',
                f'{level} shared beam cannot support all branches while clearing real flange ends',
                {'header_fitting_envelopes': fittings, 'header_support_targets': required,
                 'header_support_counts': counts})
        _, _, t, pieces, covered = max(candidates, key=lambda x: x[:3]); selected.append(t)
        beams = []
        for segment, (n0, n1, box) in enumerate(pieces):
            beam = add_box(proposed, f'{prefix}_cross{len(selected)-1}_segment{segment}', box, 'beam', steel, 'P1_shared_rack')
            beam['column_face_contacts'] = _column_faces(box, columns)
            beams.append((n0, n1, beam))
        for name in covered:
            counts[name] += 1
            header = next(h for h in active if h['id'] == name)
            n = _local(frame, header['start'])[1]
            support = next((beam for n0, n1, beam in beams if n0 <= n-bw and n+bw <= n1), None)
            if support is None:
                p = _world(frame,t,n,top)
                support = next((c for c in joins if c['kind']=='beam' and abs(c['box'][5]-top)<=1e-8
                    and c['box'][0]-1e-8<=p[0]-bw and p[0]+bw<=c['box'][3]+1e-8
                    and c['box'][1]-1e-8<=p[1]-bw and p[1]+bw<=c['box'][4]+1e-8),None)
            if support is None:
                raise PipingStationRejected('P1_RACK_SEAT_OBSTRUCTED', f'{level} pipe shoes do not contact the cross-beam between columns')
            actual_pipe_bottom=header['start'][2]-header['outer_radius_m']
            seat = _box(frame, t-bw, n-bw, top, t+bw, n+bw, actual_pipe_bottom)
            if _solid_hits(seat, source_obstacles, protected):
                raise PipingStationRejected('P1_RACK_SEAT_OBSTRUCTED', f'{level} shared pipe shoes intersect a solid or reserved space')
            add_box(proposed, f'{support["id"]}_{name}_seat', seat, 'hanger', steel, 'P1_shared_rack')
    return proposed['components'], {'level': level, 'status': 'candidate', 'attachments': attachments,
        'crossbeam_count': len(selected), 'header_support_counts': counts, 'header_support_targets': required,
        'header_end_fitting_exclusions': fittings,
        'source': 'PG-04 shared rack section spans between existing physical columns'}


def _layout_protected(frozen, ceiling):
    """Original stance, extraction and actual saved maintenance-height bands."""
    from .process import RULES
    scene=frozen['scene'];protected=_protected_boxes(scene,ceiling)
    checks=frozen.get('contracts',{}).get('layout',{}).get('checks',{})
    for row in checks.get('maintenance',{}).get('rows',[]):
        rect=row.get('maintenance_rect');level=row.get('level')
        if rect is None or not isinstance(level,int) or not 0<=level<len(scene['floors']):continue
        profile=row.get('maintenance_geometry',{})
        band=profile.get('z_band_m') or [scene['floors'][level],scene['floors'][level]+RULES['geometry_screen']['walk_height_m']]
        protected.append((f"{row['id']}.L{level}.maintenance",[rect[0],rect[1],band[0],rect[2],rect[3],band[1]]))
    return protected


def _header_candidate_hits(header, scene, protected, existing, config):
    """All real bodies, end fittings, extraction and unchanged PG14 values."""
    spacing=config['pg_rules']['parallel_pipe_spacing']
    wall=spacing['outermost_pipe_or_insulation_to_column_or_wall_min_mm']/1000.
    bare=spacing['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.
    protrusion=spacing['protrusions']['min_clearance_mm']/1000.
    boxes=[('body',_pipe_box(header['start'],header['end'],header['outer_radius_m']))]
    boxes.extend((f['id'],f['box']) for f in _header_end_fittings([header]))
    hits=[]
    for part,box in boxes:
        for c in scene['components']:
            if c.get('annotation_only'):continue
            gap=wall if c['kind'] in ('column','wall') else 0.
            obstacle=(np.asarray(c['box'])+np.r_[-np.full(3,gap),np.full(3,gap)]).tolist()
            if _overlap(box,obstacle):hits.append(c['id'])
        hits.extend(name for name,obstacle in protected if _overlap(box,obstacle))
        for other in existing:
            gap=bare if part=='body' else protrusion
            obstacle=np.asarray(_pipe_box(other['start'],other['end'],other['outer_radius_m']))
            if _overlap(box,(obstacle+np.r_[-np.full(3,gap),np.full(3,gap)]).tolist()):hits.append(other['id'])
            for fitting in _header_end_fittings([other]):
                obstacle=np.asarray(fitting['box'])+np.r_[-np.full(3,protrusion),np.full(3,protrusion)]
                if _overlap(box,obstacle):hits.append(fitting['id'])
    width,depth=scene['footprint']
    if any(box[0]<-1e-8 or box[1]<-1e-8 or box[3]>width+1e-8 or box[4]>depth+1e-8 for _,box in boxes):
        hits.append('actual_shell_boundary')
    return list(dict.fromkeys(hits))


def plan_service_headers(frozen,requirements,config,heights,groups=None,variant=0):
    """Place finite nearby/balanced row headers independently of station edge.

    heights uses the existing {level:{bottom_z_m,top_z_m,...}} PG04 schema.
    This plans pipes/end-fitting envelopes, not actual installed connections.
    Shared full-span steel members are added by the station planner below.
    """
    from .piping_service_groups import service_groups_from_frozen
    from .piping_header_taps import tee_reservation,adjacent_reservation
    from .piping_assets import inspect_asset
    require_environment()
    if isinstance(variant,bool) or not isinstance(variant,int) or variant<0:
        raise ValueError('header variant must be a nonnegative integer')
    groups=service_groups_from_frozen(frozen,requirements) if groups is None else deepcopy(groups)
    scene=frozen['scene'];width,depth=scene['footprint'];footprint=[width,depth]
    ceiling=max(h['top_z_m'] for h in heights.values());protected=_layout_protected(frozen,ceiling)
    point_map={p['id']:p for p in requirements['service_points']}
    templates=requirements.get('service_templates',{})
    selected=[];evidence=[]
    for group in groups:
        axis=group['axis_index'];other=1-axis;medium=group['medium'];level=group['level']
        points=[point_map[pid] for pid in group['point_ids']]
        if not points:raise PipingStationRejected('P1_EMPTY_SERVICE_GROUP','Cannot plan an empty service group')
        template=templates.get(medium,config['templates']['services'][medium]);dn=template['branch_dn']
        radius=_radius(config,dn);fitting_radius=radius+FITTING_RADIAL_EXTRA_M
        reservations=[tee_reservation(config,dn,p['dn'],inspect_asset(p['dn'])['install_length_m']) for p in points]
        main_length=0. if medium=='offgas' else inspect_asset(dn)['install_length_m']
        reservations.append(tee_reservation(config,dn,dn,main_length))
        end_reserve=max(r['end_reserve_m'] for r in reservations)
        pitch=max(adjacent_reservation(a,b)['required_center_distance_m'] for a,b in product(reservations,reservations))
        coordinates=[p['position'][axis] for p in points];cross=[p['position'][other] for p in points]
        length=max(max(coordinates)-min(coordinates)+2*(HEADER_FLANGE_AXIAL_LENGTH_M+end_reserve),
                   2*(HEADER_FLANGE_AXIAL_LENGTH_M+end_reserve)+(len(reservations)-1)*pitch)
        extent=footprint[axis]-2*HEADER_BLIND_THICKNESS_M
        if length>extent+1e-8:
            raise PipingStationRejected('P1_GROUP_HEADER_TOO_LONG','Physical tee and closure reservations exceed saved building-axis space',
                {'service_group':group['id'],'required_length_m':length,'available_m':extent})
        lo=max(HEADER_BLIND_THICKNESS_M,min((min(coordinates)+max(coordinates)-length)/2,
                                         footprint[axis]-HEADER_BLIND_THICKNESS_M-length))
        hi=lo+length
        if lo>min(coordinates)+1e-8 or hi<max(coordinates)-1e-8:
            raise PipingStationRejected('P1_GROUP_HEADER_SPAN','Permitted end closures do not cover the original service outlet projections')
        values={float(np.mean(cross)),float(np.median(cross)),float((min(cross)+max(cross))/2)}
        for body in group['geometry_evidence']['device_centres']:
            rect=body['body_rect'];values.update((rect[other]-fitting_radius,rect[other+2]+fitting_radius))
        for _,box in protected:
            values.update((box[other]-fitting_radius,box[other+3]+fitting_radius))
        for c in scene['components']:
            if c['kind']=='column':
                wall=config['pg_rules']['parallel_pipe_spacing']['outermost_pipe_or_insulation_to_column_or_wall_min_mm']/1000.
                values.update((c['box'][other]-fitting_radius-wall,c['box'][other+3]+fitting_radius+wall))
        for existing in selected:
            if existing['axis_index']==axis and existing['level']==level:
                gap=config['pg_rules']['parallel_pipe_spacing']['protrusions']['min_clearance_mm']/1000.
                lane=existing['start'][other];pitch_r=existing['fitting_outer_radius_m']+fitting_radius+gap
                values.update((lane-pitch_r,lane+pitch_r))
        values=[v for v in values if fitting_radius-1e-8<=v<=footprint[other]-fitting_radius+1e-8]
        values.sort(key=lambda v:(max(abs(v-c) for c in cross),sum(abs(v-c) for c in cross),v))
        # This finite resource cap is independent of the original install rules.
        values=values[:24];zone=heights[level]
        z_values=sorted(set((zone['bottom_z_m']+radius,zone['top_z_m']-radius,
                             (zone['bottom_z_m']+zone['top_z_m'])/2)))
        candidates=[];refusals=Counter()
        for v,z in product(values,z_values):
            if z-radius<zone['bottom_z_m']-1e-8 or z+radius>zone['top_z_m']+1e-8:continue
            start=np.zeros(3);end=np.zeros(3);start[axis]=lo;end[axis]=hi
            start[other]=end[other]=v;start[2]=end[2]=z
            distances=[abs(p['position'][other]-v)+abs(p['position'][2]-z) for p in points]
            header={'id':'P1_header_'+group['id'],'medium':medium,'level':level,'dn':dn,
                'start':start.tolist(),'end':end.tolist(),'active':True,'point_ids':list(group['point_ids']),
                'service_group':group['id'],'service_group_evidence':deepcopy(group),'axis_index':axis,
                'canonical_axis':list(group['axis']),'outer_radius_m':radius,'fitting_outer_radius_m':fitting_radius,
                'fitting_source':FITTING_SOURCE,'flow_toward_station':None,'flow_direction':None,
                'distance_report':{'point_to_axis_manhattan_m':distances,'minimum_m':min(distances),
                    'maximum_m':max(distances),'mean_m':sum(distances)/len(distances),'range_m':max(distances)-min(distances),
                    'equal_length_threshold':None,'scope':'Geometric estimate from original port to header axis'},
                'endpoint_reservation':{'coarse_tap_formula':'v1.5 tee_reservation/adjacent_reservation',
                    'maximum_end_reserve_m':end_reserve,'maximum_adjacent_pitch_m':pitch,
                    'flange_axial_length_m':HEADER_FLANGE_AXIAL_LENGTH_M,'blind_thickness_m':HEADER_BLIND_THICKNESS_M,
                    'required_length_m':length,'source_projection_interval_m':[min(coordinates),max(coordinates)]}}
            hits=_header_candidate_hits(header,scene,protected,selected,config)
            if hits:
                refusals.update(hits);continue
            # Preserve PG04's common pipe-bottom plane first. An upper plane
            # is available only when no lower-plane geometric candidate fits,
            # rather than choosing a different bottom merely for another DN.
            candidates.append((z_values.index(z),max(distances),sum(distances)/len(distances),
                               max(distances)-min(distances),v,z,header))
        if not candidates:
            raise PipingStationRejected('P1_GROUP_HEADER_UNAVAILABLE','No nearby header plane clears physical geometry, service spaces, columns and existing headers',
                {'service_group':group['id'],'axis_index':axis,'cross_coordinates_considered':values,
                 'height_candidates_m':z_values,'refused_obstacles':dict(refusals),'physical_unsatisfiable_claim':False})
        candidates.sort(key=lambda row:row[:-1]);height_rank=candidates[0][0]
        common_plane_candidates=[row for row in candidates if row[0]==height_rank]
        chosen=common_plane_candidates[min(variant,len(common_plane_candidates)-1)][-1]
        chosen['height_selection']={'rank':height_rank,'pipe_bottom_m':chosen['start'][2]-radius,
            'preferred_common_pipe_bottom_m':zone['bottom_z_m'],
            'reason':'common_pipe_bottom' if height_rank==0 else 'lower_plane_candidates_blocked_by_actual_geometry'}
        selected.append(chosen)
        evidence.append({'service_group':group['id'],'axis_index':axis,'candidate_count':len(values)*len(z_values),
            'feasible_candidates':len(candidates),'selected_variant':min(variant,len(common_plane_candidates)-1),
            'height_selection':deepcopy(chosen['height_selection']),
            'distance_report':chosen['distance_report'],'refused_obstacles':dict(refusals),
            'station_edge_used_for_axis':False,'global_optimality_claim':False})
    return {'headers':selected,'service_groups':groups,'selection_evidence':evidence}


def _plan(frame, order, main_dns, templates, anchor, offsets, heights, frozen, requirements, config, header_layout=None):
    from .building_equipment_checks import standing_board_rectangles, _uncovered
    from .process import RULES
    scene = frozen['scene']; pg = config['pg_rules']; floors = scene['floors']
    width,depth=scene['footprint']
    spacing = pg['parallel_pipe_spacing']; bare = spacing['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000
    wall = spacing['outermost_pipe_or_insulation_to_column_or_wall_min_mm']/1000
    points = requirements['service_points']; pad = pg['distribution']['rack_zone']['plan_edge_padding_m']
    radii = {m: _radius(config, main_dns[m]) for m in order}; rmax = max(radii.values())
    fitting_rmax = rmax+FITTING_RADIAL_EXTRA_M
    frame_parts = _edge_frame_parts(frame, scene['components'])
    frame_inner_face = max((local[3] for _, local in frame_parts), default=0.)
    normal = frame_inner_face+wall+fitting_rmax
    top = max(z['top_z_m'] for z in heights.values()); protected = _protected_boxes(scene, top)
    risers, openings = [], []
    for medium in order:
        t = anchor+offsets[medium]; xy = _world(frame, t, normal, 0.)[:2]; radius = radii[medium]
        r = {'id': 'P1_main_'+medium, 'medium': medium, 'dn': main_dns[medium],
             'branch_dn': templates[medium]['branch_dn'], 'temporary_dn': templates[medium]['temporary_dn'],
             'xy': xy, 'bottom_z': float(floors[0]), 'top_z': top, 'served_levels': list(range(len(floors))),
             'outer_radius_m': radius, 'main_actual_radius': radius,
             'fitting_outer_radius_m': radius+FITTING_RADIAL_EXTRA_M, 'fitting_source': FITTING_SOURCE,
             'service_identity': templates[medium].get('service_identity'),
             'service_equipment': sorted({p['equipment'] for p in points if p['medium'] == medium})}
        box = _pipe_box([*xy, r['bottom_z']], [*xy, top], radius)
        hits = _solid_hits(box, scene['components'], protected, ignore_floors=True)
        if hits:
            raise PipingStationRejected('P1_STATION_RISER_OBSTRUCTED', 'Main riser conflicts with a saved solid or operation reservation', {'hits': hits})
        for column in (c for c in scene['components'] if c['kind'] == 'column'):
            b = column['box']
            if min(top, b[5])-max(r['bottom_z'], b[2]) <= 1e-8:
                continue
            dx = max(b[0]-xy[0], xy[0]-b[3], 0.)
            dy = max(b[1]-xy[1], xy[1]-b[4], 0.)
            if math.hypot(dx, dy)-r['fitting_outer_radius_m'] < wall-1e-8:
                raise PipingStationRejected('P1_STATION_FITTING_COLUMN_CLEARANCE', 'Main flange-to-column clearance is below PG-14')
        for level in range(1, len(floors)):
            rect = [xy[0]-radius, xy[1]-radius, xy[0]+radius, xy[1]+radius]
            if _uncovered(rect, standing_board_rectangles(scene, level)):
                raise PipingStationRejected('P1_STATION_OPENING_UNAVAILABLE', 'The riser lacks a complete original deck surface or uses an equipment opening')
            openings.append({'level': level, 'rect': rect, 'owner': r['id'], 'kind': 'pipe',
                'source': 'Local slab crossing follows the actual pipe OD without unspecified construction clearance'})
        risers.append(r)
    # A preliminary common access patch, using existing station dimensions.
    policy = RULES['geometry_screen']; station_t = sum(anchor+offsets[m] for m in order)/len(order)
    stance_n = normal+fitting_rmax+policy['station_depth_m']/2
    access = []
    for level, base in enumerate(floors):
        box = _box(frame, station_t-policy['station_width_m']/2, stance_n-policy['station_depth_m']/2, base,
                   station_t+policy['station_width_m']/2, stance_n+policy['station_depth_m']/2, base+policy['station_height_m'])
        rect = [box[0], box[1], box[3], box[4]]
        if _uncovered(rect, standing_board_rectangles(scene, level)) or _solid_hits(box,
                [c for c in scene['components'] if c['kind'] != 'floor'], protected):
            raise PipingStationRejected('P1_STATION_ACCESS_UNAVAILABLE', 'No clear station-side standing area on the original deck')
        access.append({'level': level, 'box': box, 'status': 'preliminary_empty_patch',
                       'scope': 'Use the existing stance dimensions. Check placed valves, access and sweep after piping.'})
    # Headers follow actual equipment rows, not this workstation edge.
    header_layout=(plan_service_headers(frozen,requirements,config,heights,
        variant=0 if frame['direction']>0 else 1) if header_layout is None else deepcopy(header_layout))
    headers=header_layout['headers'];zones={};support_components=[];support_rows=[]
    protected=_layout_protected(frozen,top)
    for level,h in heights.items():
        active=[row for row in headers if row['level']==level and row['active']]
        for medium in order:
            if any(row['medium']==medium for row in active):continue
            dn=templates[medium]['branch_dn'];r=_radius(config,dn)
            main=next(row for row in risers if row['medium']==medium)
            point=[*main['xy'],h['bottom_z_m']+r]
            headers.append({'id':f'P1_inactive_{medium}_L{level}','medium':medium,'level':level,'dn':dn,
                'start':point,'end':point,'active':False,'point_ids':[],'service_group':None,
                'axis_index':None,'canonical_axis':None,'outer_radius_m':r,
                'fitting_outer_radius_m':r+FITTING_RADIAL_EXTRA_M,'fitting_source':FITTING_SOURCE,
                'inactive_policy':'omit_fixed_outlet' if medium=='offgas' else 'blind_at_root_valve'})
        envelopes=[_pipe_box(row['start'],row['end'],row['outer_radius_m']) for row in active]
        envelopes.extend(row['box'] for row in _header_end_fittings(active))
        box=([max(0.,min(v[0] for v in envelopes)-pad),max(0.,min(v[1] for v in envelopes)-pad),h['bottom_z_m'],
              min(width,max(v[3] for v in envelopes)+pad),min(depth,max(v[4] for v in envelopes)+pad),h['top_z_m']]
             if envelopes else [0.,0.,h['bottom_z_m'],width,depth,h['top_z_m']])
        axes=sorted({row['axis_index'] for row in active})
        zones[level]={**h,'level':level,'box':box,'axes':axes,
            'axis':([1.,0.,0.] if axes==[0] else [0.,1.,0.] if axes==[1] else None),
            'edge':None,'thickness_m':h['top_z_m']-h['bottom_z_m'],
            'plane_source':'actual service-group header and both fitting envelopes + original PG04 padding'}
        for axis in axes:
            members=[row for row in active if row['axis_index']==axis]
            support_scene={**scene,'components':scene['components']+support_components}
            components,proof=_shared_supports(_frame('y_min' if axis==0 else 'x_min',width,depth),
                level,h,members,risers,support_scene,pg,protected,all_headers=active)
            support_components.extend(components);support_rows.append({**proof,'axis_index':axis,
                'joint_policy':'actual cut-face joints with previous shared beams; no rack owner exemption'})
    strip_box = _box(frame, 0., 0., 0., frame['length'], normal+rmax, 0.)
    crossing_strips = [{'level': level, 'edge': frame['edge'],
        'rect': [strip_box[0], strip_box[1], strip_box[3], strip_box[4]],
        'derivation': {'frame_component_ids': [c['id'] for c, _ in frame_parts],
            'frame_inner_face_m': frame_inner_face, 'pipe_to_column_clearance_m': wall,
            'max_main_outer_diameter_m': 2*rmax, 'band_width_m': normal+rmax},
        'source': 'Use inner framing faces, PG-14 clearance, maximum flange radius and actual pipe radius. The hole matches the pipe OD.'}
        for level in range(1, len(floors))]
    for strip in crossing_strips:
        strip['derivation'].update(max_main_fitting_radius_m=fitting_rmax,
            main_actual_radius_m=rmax, fitting_radial_extra_m=FITTING_RADIAL_EXTRA_M,
            fitting_source=FITTING_SOURCE)
    return {'station': {'edge': frame['edge'], 'inward': [*frame['inward'].tolist(), 0.],
                'tangent': [*frame['tangent'].tolist(), 0.],
                'distribution_direction': [float(frame['direction']*v) for v in [*frame['tangent'], 0.]],
                'risers': risers, 'access_candidates': access, 'crossing_strips': crossing_strips,
                'max_main_fitting_radius_m': fitting_rmax, 'main_actual_radius': rmax,
                'boundary_source': 'Actual slab edge and inner column faces'},
            'rack_zones': zones, 'headers': headers, 'service_groups':header_layout['service_groups'],
            'rack_support_components': support_components,
            'openings': openings, 'search_evidence': {'status': 'candidate', 'support_candidates': support_rows,
                'main_dns': deepcopy(main_dns), 'service_order': list(order), 'anchor_t_m': anchor,
                'header_selection':header_layout['selection_evidence'],
                'service_group_version':'v1.6_frozen_body_position_service_groups',
                'station_inset_m': normal, 'pipe_spacing_source': 'Use the greater of bare pipe plus 50 mm and flange envelope plus 30 mm. The outermost flange stays 100 mm from columns and walls.',
                'scope': 'Precheck station, branch and shared rack candidates. Verify connections, fittings, reductions and sweep after piping.'}}


def station_candidates(frozen, requirements, config, seed=None, max_candidates=24):
    """Return bounded, interleaved candidates from all four real shell edges."""
    require_environment()
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 24:
        raise ValueError('max_candidates must be an integer from 1 to 24')
    services = requirements.get('active_services', []); points = requirements.get('service_points', [])
    if not services or not points:
        raise PipingStationRejected('P1_NO_ACTIVE_SERVICES', 'No physical utility users for a station')
    scene = frozen['scene']; width, depth = scene['footprint']; pg = config['pg_rules']
    if any(p['level'] not in range(len(scene['floors'])) for p in points):
        raise InputRejected('P1_SERVICE_LEVEL_INVALID: the actual service level is outside the saved building')
    templates, main_options = _main_options(requirements, config)
    heights = _heights(scene, pg); spacing = pg['parallel_pipe_spacing']
    # These choices do not depend on a workstation anchor. Reuse their pure
    # frozen-geometry result rather than regrouping on every station trial.
    from .piping_service_groups import service_groups_from_frozen
    groups=service_groups_from_frozen(frozen,requirements);header_layouts={};header_failures={}
    for variant in (0,1):
        try:header_layouts[variant]=plan_service_headers(frozen,requirements,config,heights,groups,variant)
        except PipingStationRejected as exc:header_failures[variant]=exc
    if not header_layouts:
        first=header_failures[0]
        raise PipingStationRejected(first.code,str(first),{'header_layout_refusals':{
            str(k):{'code':v.code,'detail':str(v),'evidence':v.evidence} for k,v in header_failures.items()}})
    bare = spacing['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000
    wall = spacing['outermost_pipe_or_insulation_to_column_or_wall_min_mm']/1000
    protrusion = spacing['protrusions']['min_clearance_mm']/1000
    rng = random.Random(frozen['graph'].get('seed') if seed is None else seed)
    edges = ['y_min', 'y_max', 'x_min', 'x_max']; rng.shuffle(edges)
    orders = []
    for order in [services, list(reversed(services)), *[services[i:]+services[:i] for i in range(len(services))]]:
        if order not in orders:
            orders.append(list(order))
    quota = math.ceil(max_candidates/len(edges)); buckets = []; refusals = Counter(); examined = 0
    for edge in edges:
        found = []; frame = _frame(edge, width, depth)
        for main_dns, order, direction in product(main_options, orders, (1, -1)):
            variant=0 if direction>0 else 1
            if variant not in header_layouts:
                refusals[header_failures[variant].code]+=1;continue
            radii = {m: _radius(config, main_dns[m]) for m in order}
            anchors, offsets, _ = _anchors(frame, order, radii, points, scene['components'], direction, wall, bare, protrusion)
            for anchor in anchors:
                examined += 1
                try:
                    plan = _plan({**frame, 'direction': direction}, order, main_dns, templates,
                                 anchor, offsets, heights, frozen, requirements, config,header_layouts[variant])
                except PipingStationRejected as error:
                    refusals[error.code] += 1
                    continue
                found.append(plan)
                if len(found) >= quota:
                    break
            if len(found) >= quota:
                break
        buckets.append(found)
    plans = []
    for index in range(max((len(bucket) for bucket in buckets), default=0)):
        for bucket in buckets:
            if index < len(bucket) and len(plans) < max_candidates:
                plans.append(bucket[index])
    evidence = {'examined_station_candidates': examined, 'returned_candidates': len(plans),
                'refused_candidate_causes': dict(refusals), 'edges_considered': edges,
                'maximum_returned_candidates': max_candidates}
    if not plans:
        raise PipingStationRejected('P1_STATION_UNAVAILABLE', 'No station and shared rack candidate on any side passed the physical, deck and stance prechecks', evidence)
    for index, plan in enumerate(plans):
        plan['search_evidence'].update(candidate_index=index, search=deepcopy(evidence))
    return plans
