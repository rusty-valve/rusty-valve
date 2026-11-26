"""Choose positions for shared pipes and their connections to the main risers."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import numpy as np

from .piping_assets import inspect_asset
from .piping_elbows import coarse_radius, plan_polyline
from .piping_header_taps import tee_reservation, adjacent_reservation
from .piping_lane_connect import _obstacles, _hits, _clear_plan
from .piping_method_headers import required_rise_hits
from .piping_requirements import required_header_keys
from .piping_service_groups import building_long_edge_axis, frozen_body_positions
from .piping_station import (PipingStationRejected, _radius, _frame,
    _edge_frame_parts, _header_candidate_hits, _header_end_fittings,
    _pipe_box, _layout_protected,
    FITTING_RADIAL_EXTRA_M, FITTING_SOURCE, HEADER_FLANGE_AXIAL_LENGTH_M,
    HEADER_BLIND_THICKNESS_M)


EPS = 1e-8
MAX_BUNDLE_TRANSLATIONS = 16
TAKEOFF_STAGGER_M = .30
TAKEOFF_LOW_TO_HIGH = ('water_return', 'water', 'steam', 'nitrogen')
DEMAND_ONLY_TERMINALS = 'demand_only_direct_water_return'
DIRECT_CONNECTION_KINDS = frozenset(('offgas','water_return'))
_TIE_ORDER = ('nitrogen', 'offgas', 'steam', 'water', 'water_return')


def station_root_keys(requirements, bundle, floor_count):
    """Derive required station valves from original use points."""
    if (bundle or {}).get('terminals_policy') == DEMAND_ONLY_TERMINALS:
        return [(medium,level) for medium,level in sorted(required_header_keys(requirements))
                if medium not in DIRECT_CONNECTION_KINDS]
    return [(medium,level) for medium in requirements['active_services']
            if medium!='offgas' for level in range(floor_count)]


def _rejected(code, detail, **evidence):
    raise PipingStationRejected(code, detail, evidence)


def _point(axis, axial, cross, transverse, z):
    p = [0., 0., float(z)]
    p[axis] = float(axial); p[cross] = float(transverse)
    return p


def _spacing(config):
    pg = config['pg_rules']['parallel_pipe_spacing']
    return (pg['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.,
            pg['protrusions']['min_clearance_mm']/1000.,
            pg['outermost_pipe_or_insulation_to_column_or_wall_min_mm']/1000.)


def _level_heights(headers, points, config, heights):
    """Reuse the existing same-level turn-band calculation and quarter plane."""
    bare, _, _ = _spacing(config); result = {}
    by_id = {p['id']: p for p in points}
    for level in sorted({h['level'] for h in headers}):
        rows = [h for h in headers if h['level'] == level]
        radii = [_radius(config, by_id[pid]['dn']) for h in rows for pid in h['point_ids']]
        turn = max(coarse_radius(r)+r for r in radii)
        fitting = max(h['fitting_outer_radius_m'] for h in rows)
        bottom = float(heights[level]['bottom_z_m']); top = float(heights[level]['top_z_m'])
        header_radii = [float(h['outer_radius_m']) for h in rows]
        low = bottom+max(turn, fitting)-min(header_radii)
        high = top-max(turn, fitting+bare+2*max(radii))-max(header_radii)
        if low > high+EPS:
            _rejected('END_BUNDLE_HEIGHT', 'Rack depth cannot fit the first elbow and overhead pipe approach',
                      level=level, low_z_m=low, high_z_m=high)
        # This scalar is the common physical pipe bottom, not a shared axis.
        result[level] = float(np.clip(bottom+(top-bottom)/4, low, high))
    return result


def _header_span(header, points, config, near, axis, end_sign, *, minimum_only=False):
    """Reserve body tees and the existing real axial flange/end distances."""
    rows = sorted((p for p in points if p['id'] in header['point_ids']),
                  key=lambda p: (p['position'][axis], p['position'][1-axis], p['dn']))
    reservations = [tee_reservation(config, header['dn'], p['dn'],
                    0. if p.get('header_only') else inspect_asset(p['dn'])['install_length_m']) for p in rows]
    edges = [HEADER_FLANGE_AXIAL_LENGTH_M+r['end_reserve_m'] for r in reservations]
    gaps = [adjacent_reservation(a, b)['required_center_distance_m']
            for a, b in zip(reservations, reservations[1:])]
    minimum = edges[0]+edges[-1]+sum(gaps)
    if minimum_only:
        return float(near-end_sign*minimum), {}
    preferences=header.get('end_bundle_device_tap_preferences',{})
    projection = [float(preferences.get(p['id'],{}).get('axis_coordinate_m',p['position'][axis]))
                  for p in rows]
    if end_sign > 0:
        far = min(min(p-e for p, e in zip(projection, edges)), near-minimum)
        if near < max(p+e for p, e in zip(projection, edges))-EPS:
            return None
    else:
        far = max(max(p+e for p, e in zip(projection, edges)), near+minimum)
        if near > min(p-e for p, e in zip(projection, edges))+EPS:
            return None
    return float(far), {'minimum_body_tee_span_m':minimum,
        'source_projection_interval_m':[min(p['position'][axis] for p in rows),
                                        max(p['position'][axis] for p in rows)],
        'preferred_tee_projection_interval_m':[min(projection),max(projection)],
        'max_end_reserve_m':max(r['end_reserve_m'] for r in reservations),
        'maximum_adjacent_pitch_m':max(gaps, default=0.),
        'scope':'Use equipment-pipe tees near the first elbow tangent or main intersection'}


def _subtract_interval(intervals, blocked):
    remaining=[]
    for low,high in intervals:
        if blocked[1]<=low+EPS or blocked[0]>=high-EPS:
            remaining.append((low,high));continue
        if blocked[0]>=low-EPS:remaining.append((low,min(high,blocked[0])))
        if blocked[1]<=high+EPS:remaining.append((max(low,blocked[1]),high))
    return remaining


def _device_tap_preferences(header, points, scene, config, axis, cross, transverse, z, ceiling,
                            axial_span=None):
    """Choose ordered body-tee preferences outside actual transverse blockers.

    A changed preference frees the existing lane connector to use axis-first
    travel.  This does not prescribe or emit a route, and cannot turn a later
    failed physical connection into a pass.
    """
    bare,_,wall=_spacing(config)
    physical,_=_obstacles([c for c in scene['components']
        if not c.get('annotation_only') and c.get('kind') not in ('floor','ground')])
    for pullout in scene.get('pullout_reservations',[]):
        if not pullout.get('open_top'):continue
        rect=pullout['rect']
        physical.append({'id':pullout['id'],'kind':'pullout_reservation',
            'box':np.asarray([rect[0],rect[1],pullout['base_z'],rect[2],rect[3],ceiling])})
    rows=[]
    for point in sorted((p for p in points if p['id'] in header['point_ids']),
                        key=lambda p:(p['position'][axis],p['position'][cross],p['position'][2],p['dn'])):
        source=float(point['position'][axis]);r=_radius(config,point['dn']);R=coarse_radius(r)
        cross_low,cross_high=sorted((float(point['position'][cross]),float(transverse)))
        blockers=[];direct=[]
        for obstacle in physical:
            box=obstacle['box'];gap=wall if obstacle['kind'] in ('wall','column') else 0. if obstacle['kind']=='pullout_reservation' else bare
            if not box[2]-r-gap-EPS<z<box[5]+r+gap+EPS:continue
            if min(cross_high,box[cross+3]+r+gap)-max(cross_low,box[cross]-r-gap)<=EPS:continue
            blockers.append((obstacle,gap))
            if box[axis]-r-gap+EPS<source<box[axis+3]+r+gap-EPS:direct.append(obstacle['id'])
        # Retreat by the true centreline radius when moving onto an obstacle
        # side plane; the original PG gap itself is unchanged.
        retreat=R if direct else 0.
        reserve=tee_reservation(config,header['dn'],point['dn'],
            0. if point.get('header_only') else inspect_asset(point['dn'])['install_length_m'])
        low,high=r,float(scene['footprint'][axis])-r
        if axial_span is not None:
            end_reserve=HEADER_FLANGE_AXIAL_LENGTH_M+reserve['end_reserve_m']
            low=max(low,axial_span[0]+end_reserve)
            high=min(high,axial_span[1]-end_reserve)
        intervals=[(low,high)] if low<=high else []
        for obstacle,gap in blockers:
            b=obstacle['box']
            intervals=_subtract_interval(intervals,(b[axis]-r-gap-retreat,b[axis+3]+r+gap+retreat))
        if not intervals:return None
        candidates=[float(np.clip(source,low,high)) for low,high in intervals]
        preferred=min(candidates,key=lambda value:(abs(value-source),value))
        rows.append({'point':point,'source':source,'preferred':preferred,'intervals':intervals,
            'reservation':reserve,'evidence':{'axis_coordinate_m':preferred,
                'source_axis_coordinate_m':source,'obstacles':direct,
                'horizontal_center_z_m':float(z),'pipe_outer_radius_m':r,
                'centerline_R_m':R,'turn_retreat_m':retreat,
                'obstacle_side_planes_m':[{'id':o['id'],'box':o['box'].tolist(),
                    'required_gap_m':gap,'lower_axis_m':float(o['box'][axis]-r-gap-retreat),
                    'upper_axis_m':float(o['box'][axis+3]+r+gap+retreat)} for o,gap in blockers],
                'scope':'Keep equipment ports fixed. Apply tee-axis preference separately from lane routing and physical measurements.'}})
    # The same real tee pitches used by the final allocator preserve the
    # original source order. Both monotone sweeps are finite 1D allocation.
    assignments=[]
    for reverse in (False,True):
        values={};previous=None;valid=True
        for row in reversed(rows) if reverse else rows:
            pitch=adjacent_reservation(row['reservation'],previous['reservation'])['required_center_distance_m'] if previous else 0.
            options=[]
            for low,high in row['intervals']:
                if previous:
                    if reverse:high=min(high,values[previous['point']['id']]-pitch)
                    else:low=max(low,values[previous['point']['id']]+pitch)
                if low<=high+EPS:options.append(float(np.clip(row['preferred'],low,high)))
            if not options:valid=False;break
            values[row['point']['id']]=min(options,key=lambda v:(abs(v-row['preferred']),v))
            previous=row
        if valid:assignments.append((sum(abs(values[r['point']['id']]-r['source']) for r in rows),values))
    if not assignments:return None
    _,values=min(assignments,key=lambda row:row[0]);result={}
    for row in rows:
        evidence=row['evidence'];evidence['axis_coordinate_m']=values[row['point']['id']]
        evidence['source_order_preserved']=True
        result[row['point']['id']]=evidence
    return result


def _connection(main, header, level, floor, z, axis, cross, direction, config):
    medium = main['medium']; dn = main['branch_dn']
    d = np.asarray(direction)
    direct = medium in DIRECT_CONNECTION_KINDS
    elevation = z if direct else floor+1.10+main['takeoff_offset_m']
    pos = [*main['xy'], float(elevation)]
    name = f'P1_end_{medium}_L{level}'
    if header is not None:
        name += '_'+header['id']
    row = {'id':name, 'kind':medium if direct else 'supply',
        'header_id':header['id'] if header else None, 'main_id':main['id'],
        'medium':medium, 'level':level, 'dn':dn, 'plane_axis_index':cross,
        'plane_coordinate_m':float(main['xy'][cross]), 'direction':list(direction),
        'main_tap':pos, 'valve_in':None, 'valve_out':None, 'native_L_m':0.,
        'native_L':0., 'package':None, 'R_m':0., 'lower_corner':None,
        'upper_corner':None, 'header_near_tangent':None, 'far_point':None,
        'lead_control_points':[], 'tail_control_points':[], 'elbow_count':0}
    if direct:
        row['header_near_tangent'] = list(pos)
        return row
    asset = inspect_asset(dn); L = float(asset['install_length_m'])
    radius = _radius(config, dn); R = coarse_radius(radius); bare, _, _ = _spacing(config)
    # One real native length on each side; post-spool also accommodates the
    # existing 39 mm flange before the lower elbow's actual tangent.
    lead = max(L, main['outer_radius_m']+radius+bare+L)
    inlet = np.asarray(pos)+d*lead; outlet = inlet+d*L
    row.update(valve_in=inlet.tolist(), valve_out=outlet.tolist(),
        native_L_m=L, native_L=L, package=asset['package'], R_m=R,
        lead_control_points=[list(pos), inlet.tolist()])
    lower = outlet+d*(L+HEADER_FLANGE_AXIAL_LENGTH_M+R)
    upper = lower.copy(); upper[2] = float(z)
    near = upper+d*R
    row.update(lower_corner=lower.tolist(), upper_corner=upper.tolist(),
        header_near_tangent=near.tolist(), elbow_count=2,
        tail_control_points=[outlet.tolist(),lower.tolist(),upper.tolist(),near.tolist()])
    return row


def takeoff_spacing(risers, connections):
    """Check axial separation on genuinely adjacent workstation risers."""
    by_main={}
    for connection in connections:
        by_main.setdefault(connection['main_id'],[]).append(connection)
    result=[]
    for left,right in zip(risers,risers[1:]):
        for first in by_main.get(left['id'],()):
            for second in by_main.get(right['id'],()):
                a,b=first.get('main_tap'),second.get('main_tap')
                delta=float(a[2])-float(b[2]) if a is not None and b is not None else None
                separation=abs(delta) if delta is not None else None
                if first['level']!=second['level'] and (separation is None or separation>=TAKEOFF_STAGGER_M-EPS):
                    continue
                ordered=(first['level']==second['level'] and left['medium'] in TAKEOFF_LOW_TO_HIGH
                         and right['medium'] in TAKEOFF_LOW_TO_HIGH
                         and all(c.get('kind') not in DIRECT_CONNECTION_KINDS for c in (first,second)))
                order_ok=(delta is not None and delta*(TAKEOFF_LOW_TO_HIGH.index(left['medium'])
                           -TAKEOFF_LOW_TO_HIGH.index(right['medium']))>0) if ordered else None
                result.append({'connections':[first['id'],second['id']],
                               'mains':[left['id'],right['id']], 'levels':[first['level'],second['level']],
                               'tap_centers_m':[a,b], 'separation_m':separation,
                               'required_m':TAKEOFF_STAGGER_M, 'low_to_high_pass':order_ok,
                               'pass':separation is not None and separation>=TAKEOFF_STAGGER_M-EPS
                                      and order_ok is not False})
    return result


def _screen_candidate(scene, protected, headers, mains, connections, config, requirements, heights):
    """Existing physical screen only; native valve poses remain emitter work."""
    bare, _, wall = _spacing(config); selected = []
    for header in headers:
        hits = _header_candidate_hits(header, scene, protected, selected, config)
        rises = required_rise_hits(header, requirements, config, heights)
        if hits or rises:
            return {'stage':'header', 'id':header['id'], 'obstacles':hits, 'required_rises':rises}
        selected.append(header)
    parts = [c for c in scene['components'] if c.get('kind') not in ('floor','ground')
             and not c.get('annotation_only')]
    parts += [{'id':name,'kind':'pullout_reservation','box':box} for name, box in protected
              if any(r.get('id')==name for r in scene.get('pullout_reservations', []))]
    obstacles, _ = _obstacles(parts)
    width, depth = scene['footprint']; ceiling = max(m['top_z'] for m in mains)
    for main in mains:
        a = np.asarray([*main['xy'],main['bottom_z']]); b = np.asarray([*main['xy'],main['top_z']])
        for obstacle in obstacles:
            gap = wall if obstacle['kind'] in ('column','wall') else bare
            if _hits(a,b,obstacle,main['outer_radius_m'],gap):
                return {'stage':'main', 'id':main['id'], 'obstacles':[obstacle['id']]}
    bounds = np.array([0.,0.,float(scene['floors'][0]),width,depth,ceiling+1.])
    for connection in connections:
        if connection['kind'] in DIRECT_CONNECTION_KINDS:
            continue
        r = _radius(config,connection['dn']); direction = np.asarray(connection['direction'])
        for kind, points in (('lead',connection['lead_control_points']),
                             ('tail',connection['tail_control_points'])):
            plan = plan_polyline(points,r)
            end_direction = np.asarray(points[-1])-points[-2]
            end_direction /= np.linalg.norm(end_direction)
            hit = _clear_plan(plan,obstacles,r,bare,bounds,
                {'position':points[0]}, {'position':points[-1]},direction,end_direction,
                is_lead=kind=='lead',is_tail=kind=='tail')
            if hit:
                return {'stage':kind, 'id':connection['id'], 'obstacles':[hit]}
    return None


def _header_free_intervals(headers, points, scene, protected, main_info, offsets,
                           bundle_side, axial, direction, sign, z_by_level,
                           axis, cross, config, low, high, *, minimum_only=False):
    """Intersect whole-bundle slots at each header's actual Z and axial span.

    This only allocates the existing finite translation budget.  Its AABB
    subtraction is the same body/end-fitting screen used below; every selected
    candidate still passes all original rise, main, elbow and physical checks.
    """
    _, _, wall = _spacing(config); intervals = [(low,high)]
    for original in headers:
        h=deepcopy(original); medium=h['medium'];level=h['level'];z=z_by_level[level]+h['outer_radius_m']
        transverse=bundle_side*offsets[medium]
        main={'id':'P1_end_main_'+medium,'medium':medium,**main_info[medium],
              'xy':_point(axis,axial,cross,transverse,0.)[:2]}
        c=_connection(main,h,level,scene['floors'][level],z,axis,cross,direction,config)
        near=c['header_near_tangent']
        span=_header_span(h,points,config,near[axis],axis,sign,minimum_only=minimum_only)
        if span is None:return []
        far=_point(axis,span[0],cross,transverse,z)
        h['start'],h['end']=(far,near) if sign>0 else (near,far)
        boxes=[_pipe_box(h['start'],h['end'],h['outer_radius_m'])]
        boxes.extend(f['box'] for f in _header_end_fittings([h]))
        obstacles=[]
        for part in scene['components']:
            if part.get('annotation_only'):continue
            gap=wall if part['kind'] in ('column','wall') else 0.
            b=part['box'];obstacles.append([v-gap for v in b[:3]]+[v+gap for v in b[3:]])
        obstacles.extend(box for _,box in protected)
        # Offgas rises above its own header before entering downward. Its
        # upward leg must not pass through that same horizontal cylinder.
        bare=_spacing(config)[0]
        for point in points:
            if point['level']!=level or point['medium']!='offgas' or point['normal'][2]<1-EPS:continue
            r=_radius(config,point['dn'])+bare
            x,y,base=point['position']
            obstacles.append([x-r,y-r,base,x+r,y+r,z+2*r])

        for box in boxes:
            for obstacle in obstacles:
                if any(min(box[i+3],obstacle[i+3])-max(box[i],obstacle[i])<=EPS
                       for i in (axis,2)):continue
                blocked=(obstacle[cross]-box[cross+3],obstacle[cross+3]-box[cross])
                remaining=[]
                for one,two in intervals:
                    if blocked[1]<=one+EPS or blocked[0]>=two-EPS:
                        remaining.append((one,two));continue
                    if blocked[0]>=one-EPS:remaining.append((one,min(two,blocked[0])))
                    if blocked[1]<=two+EPS:remaining.append((max(one,blocked[1]),two))
                intervals=remaining
                if not intervals:return []
    return intervals


def _plan_end_bundle_variant(layout, frozen, requirements, config, heights, spacing_mode, translation_budget,
                             *, allocate_within_span=False):
    """Return a detached layout with a short-edge ``end_bundle_station``.

    Candidates move all transverse planes together.  Their media order and
    adjacent pitches remain unchanged across every real floor and main riser.
    No geometry is emitted, no source/contract is changed and no model is run.
    """
    result = deepcopy(layout); scene = frozen['scene']; points = requirements['service_points']
    if not points or not result.get('headers'):
        _rejected('END_BUNDLE_EMPTY', 'Termination plan lacks a physical service port or shared header')
    axis, axis_evidence = building_long_edge_axis(scene, points[0]['level']); cross = 1-axis
    footprint = list(map(float,scene['footprint'])); bare, protrusion, wall = _spacing(config)
    counts = Counter(p['medium'] for p in points)
    media = sorted(requirements['active_services'],key=lambda m:(-counts[m],
        _TIE_ORDER.index(m) if m in _TIE_ORDER else len(_TIE_ORDER),m))
    ordered = iter(m for m in reversed(TAKEOFF_LOW_TO_HIGH) if m in media)
    media = [next(ordered) if m in TAKEOFF_LOW_TO_HIGH else m for m in media]
    templates = {}; main_info = {}
    for index,medium in enumerate(media):
        t = deepcopy(requirements.get('service_templates',{}).get(medium,
                      config['templates']['services'][medium])); templates[medium] = t
        dns = [t['main_dn']] if t.get('main_dn') is not None else t.get('main_dn_candidates',[])
        if not dns:
            _rejected('END_BUNDLE_MAIN_DN', 'No declared main DN for this medium',medium=medium)
        dn = int(dns[0]); branch_dn = int(t['branch_dn'])
        radius = _radius(config,dn)
        spacing_rank = (TAKEOFF_LOW_TO_HIGH.index(medium)-1.5 if medium in TAKEOFF_LOW_TO_HIGH
                        else (-.5 if index%2==0 else .5))
        main_info[medium] = {'dn':dn,'branch_dn':branch_dn,'outer_radius_m':radius,
            'takeoff_offset_m':spacing_rank*TAKEOFF_STAGGER_M,
            'fitting_outer_radius_m':max(radius,_radius(config,branch_dn))+FITTING_RADIAL_EXTRA_M}
    offsets = {media[0]:0.}; pitches = []
    for first, second in zip(media,media[1:]):
        a, b = main_info[first],main_info[second]
        minimum = max(a['outer_radius_m']+b['outer_radius_m']+bare,
                      a['fitting_outer_radius_m']+b['fitting_outer_radius_m']+protrusion)
        # A 50 mm layout grid plus one grid of spare room is a placement
        # preference, never a replacement for the original measured clearance.
        pitch = (math.ceil((minimum-EPS)/.05)*.05+.05 if spacing_mode=='roomy'
                 else minimum+1e-6)
        offsets[second] = offsets[first]+pitch
        pitches.append({'first':first,'second':second,'center_distance_m':pitch,
                        'minimum_from_actual_OD_and_fittings_m':minimum,
                        'spacing_mode':spacing_mode,
                        'numerical_margin_m':1e-6 if spacing_mode=='compact' else 0.})
    first = media[0]; first_points = [p for p in points if p['medium']==first]
    reference=float(np.mean([p['position'][cross] for p in first_points]))
    bodies = {r['equipment']:r for r in frozen_body_positions(frozen)}
    deltas = [p['position'][cross]-bodies[p['equipment']]['center'][cross]
              for p in first_points if p['equipment'] in bodies]
    reference_evidence={'medium':first,'cross_axis_index':cross,
        'source_port_transverse_center_m':reference,
        'body_transverse_center_m':float(np.mean([bodies[p['equipment']]['center'][cross]
            for p in first_points if p['equipment'] in bodies])),
        'source_minus_body_sum_m':float(sum(deltas)),
        'points':[{'requirement':p['id'],'equipment':p['equipment'],
            'source_transverse_coordinate_m':p['position'][cross],
            'body_transverse_center_m':bodies[p['equipment']]['center'][cross]}
            for p in first_points if p['equipment'] in bodies],
        'constraint':'bundle_side * (first_plane - source_port_transverse_center) >= 0'}
    side = 1. if sum(deltas)>=0. else -1.
    source_mean = float(np.mean([p['position'][axis] for p in points]))
    end_sign = 1. if source_mean>=footprint[axis]/2 else -1.
    z_by_level = _level_heights(result['headers'],points,config,heights)
    ceiling = max(float(h['top_z_m']) for h in heights.values())
    protected = _layout_protected(frozen,ceiling)
    branch_r = max(_radius(config,p['dn']) for p in first_points)
    first_header_r = _radius(config,main_info[first]['branch_dn'])
    arm = coarse_radius(branch_r)+first_header_r+3*branch_r+bare
    diagnostics = []; attempt = 0
    for bundle_side in (side,-side):
        desired = (max(p['position'][cross] for p in first_points)+arm if bundle_side>0
                   else min(p['position'][cross] for p in first_points)-arm)
        desired = (math.ceil((desired-EPS)/.05)*.05 if bundle_side>0
                   else math.floor((desired+EPS)/.05)*.05)
        # An outer-frame boundary is not an invented solid wall.  Preserve the
        # real fitting outline here; actual walls/columns retain the original
        # 100 mm requirement in the unchanged physical screen.
        low = max(main_info[m]['fitting_outer_radius_m']-bundle_side*offsets[m] for m in media)
        high = min(footprint[cross]-main_info[m]['fitting_outer_radius_m']-bundle_side*offsets[m]
                   for m in media)
        # The first lane is the device-facing one. Reversing media order may
        # not place an otherwise south-facing bundle north of the real users.
        if bundle_side>0:low=max(low,reference)
        else:high=min(high,reference)
        if low>high+EPS:
            diagnostics.append({'stage':'bundle_width','side':bundle_side,'low':low,'high':high,
                'spacing_mode':spacing_mode,'device_side_reference_m':reference});continue
        candidates = [float(np.clip(desired,low,high)),low,high]
        pitch = max((p['center_distance_m'] for p in pitches),default=.05)
        candidates += [desired+bundle_side*pitch,desired-bundle_side*pitch]
        for c in scene['components']:
            if c.get('annotation_only'):continue
            for medium in media:
                f = main_info[medium]['fitting_outer_radius_m']
                gap = wall if c['kind'] in ('column','wall') else 0.
                candidates += [c['box'][cross]-f-gap-bundle_side*offsets[medium],
                               c['box'][cross+3]+f+gap-bundle_side*offsets[medium]]
        candidates = list(dict.fromkeys(round(float(v),9) for v in candidates if low-EPS<=v<=high+EPS))
        candidates = sorted(candidates,key=lambda v:(abs(v-desired),v))
        for sign in (end_sign,-end_sign):
            edge = ('x' if axis==0 else 'y')+('_max' if sign>0 else '_min')
            frame = _frame(edge,*footprint)
            edge_parts = _edge_frame_parts(frame,scene['components'])
            inner = max((local[3] for _,local in edge_parts),default=0.)
            inset = inner+wall+max(v['fitting_outer_radius_m'] for v in main_info.values())
            axial = footprint[axis]-inset if sign>0 else inset
            direction = [0.,0.,0.];direction[axis] = -sign
            intervals=_header_free_intervals(result['headers'],points,scene,protected,
                main_info,offsets,bundle_side,axial,direction,sign,z_by_level,
                axis,cross,config,low,high)
            if allocate_within_span and not intervals:
                # Source projections can be outside the available run before
                # tees are allocated. Screen the minimum physical span to
                # prioritize translations; the full span is still checked below.
                intervals=_header_free_intervals(result['headers'],points,scene,protected,
                    main_info,offsets,bundle_side,axial,direction,sign,z_by_level,
                    axis,cross,config,low,high,minimum_only=True)
            ordered_intervals=sorted(intervals,key=lambda v:(abs(float(np.clip(desired,*v))-desired),v))
            # One sample from every available interval precedes extra samples
            # in any one interval; its two edges precede its midpoint.
            interval_values=[float(np.clip(desired,one,two)) for one,two in ordered_intervals]
            interval_values.extend(value for one,two in ordered_intervals for value in (one,two))
            interval_values.extend((one+two)/2 for one,two in ordered_intervals)
            # Real boundary slots cannot disappear behind dense near-device
            # tangencies. Keep the same 16 translations per endpoint.
            critical=list(dict.fromkeys([*interval_values,low,high]))
            chosen=critical[:translation_budget-2]
            chosen.extend(v for v in (low,high) if v not in chosen)
            chosen.extend(v for v in candidates if v not in chosen)
            for base in chosen[:translation_budget]:
                attempt += 1
                planes = {m:float(base+bundle_side*offsets[m]) for m in media}
                mains = []
                for medium in media:
                    info = main_info[medium];xy = _point(axis,axial,cross,planes[medium],0.)[:2]
                    mains.append({'id':'P1_end_main_'+medium,'medium':medium,**info,'xy':xy,
                        'bottom_z':float(scene['floors'][0]),'top_z':ceiling,
                        'served_levels':list(range(len(scene['floors'])))})
                main_map = {m['medium']:m for m in mains};headers=[];connections=[];failure=None
                for original in result['headers']:
                    h = deepcopy(original);medium=h['medium'];level=h['level'];z=z_by_level[level]+h['outer_radius_m']
                    h.update(axis_index=axis,canonical_axis=[1. if i==axis else 0. for i in range(3)])
                    preferences=_device_tap_preferences(h,points,scene,config,axis,cross,
                        planes[medium],z,ceiling)
                    if preferences is None:
                        failure={'stage':'ordered_device_tee_preference','id':h['id']};break
                    h['end_bundle_device_tap_preferences']=preferences
                    c = _connection(main_map[medium],h,level,scene['floors'][level],z,axis,cross,direction,config)
                    near = c['header_near_tangent'];span=_header_span(h,points,config,near[axis],axis,sign)
                    far_limits=(HEADER_BLIND_THICKNESS_M,footprint[axis]-HEADER_BLIND_THICKNESS_M)
                    if allocate_within_span and (span is None or not far_limits[0]-EPS<=span[0]<=far_limits[1]+EPS):
                        usable=(far_limits[0],near[axis]) if sign>0 else (near[axis],far_limits[1])
                        bounded=_device_tap_preferences(h,points,scene,config,axis,cross,
                            planes[medium],z,ceiling,axial_span=usable)
                        if bounded is not None:
                            h['end_bundle_device_tap_preferences']=bounded
                            span=_header_span(h,points,config,near[axis],axis,sign)
                    if span is None:
                        failure={'stage':'device_tee_end_room','id':h['id']};break
                    far, reserve = span
                    if not HEADER_BLIND_THICKNESS_M-EPS<=far<=footprint[axis]-HEADER_BLIND_THICKNESS_M+EPS:
                        failure={'stage':'header_span_boundary','id':h['id'],'far_m':far};break
                    far_point = _point(axis,far,cross,planes[medium],z)
                    h['start'],h['end'] = (far_point,list(near)) if sign>0 else (list(near),far_point)
                    h['endpoint_reservation'] = reserve
                    h['end_bundle']={'main_id':c['main_id'],'connection_id':c['id'],
                        'near_side':'end' if sign>0 else 'start','near_point':list(near),
                        'far_side':'start' if sign>0 else 'end','far_point':far_point,
                        'near_has_blind':False,'far_has_blind':True}
                    h['flow_toward_station']=c['kind'] in DIRECT_CONNECTION_KINDS
                    flow_sign=sign if h['flow_toward_station'] else -sign
                    h['flow_direction']=[flow_sign if i==axis else 0. for i in range(3)]
                    c['far_point']=far_point;headers.append(h);connections.append(c)
                if failure is None:
                    stagger=takeoff_spacing(mains,connections)
                    invalid=[row for row in stagger if not row['pass']]
                    failure=({'stage':'workstation_takeoff_spacing','pairs':invalid} if invalid else
                             _screen_candidate(scene,protected,headers,mains,connections,config,requirements,heights))
                evidence={'candidate':attempt,'edge':edge,'bundle_side':bundle_side,
                          'plane_coordinates_m':planes,'status':'rejected' if failure else 'selected',
                          'header_free_base_intervals_m':[list(v) for v in intervals],
                          'spacing_mode':spacing_mode,'device_side_reference_m':reference,
                          'allocate_tees_within_span':allocate_within_span,
                          'signed_device_side_distance_m':bundle_side*(planes[first]-reference)}
                if failure:evidence['failure']=failure
                diagnostics.append(evidence)
                if failure:continue
                openings=[]
                for main in mains:
                    x,y=main['xy'];r=main['outer_radius_m']
                    for level in range(1,len(scene['floors'])):
                        openings.append({'level':level,'rect':[x-r,y-r,x+r,y+r],
                            'owner':main['id'],'kind':'pipe','source':'end_bundle_main_actual_OD'})
                bundle={'version':'end_bundle_v1','mode':'short_edge','axis_index':axis,
                    'cross_axis_index':cross,'edge':edge,'inward':direction,'media_order':media,
                    'plane_coordinates_m':planes,'risers':mains,'openings':openings,'connections':connections,
                    'terminals_policy':DEMAND_ONLY_TERMINALS,
                    'spacing':{'adjacent':pitches,'same_order_and_spacing_all_floors':True,
                               'planned_takeoff_spacing':stagger},
                    'spacing_mode':spacing_mode,'bundle_side':bundle_side,
                    'device_side_reference_m':reference,'device_side_reference':reference_evidence,
                    'source_point_counts':dict(counts),'axis_evidence':axis_evidence,
                    'header_pipe_bottom_z_by_level_m':z_by_level,
                    'header_center_z_by_id_m':{h['id']:h['start'][2] for h in headers},
                    'candidate_diagnostics':diagnostics,
                    'fitting_source':FITTING_SOURCE,'actual_measurement_pending':True}
                result['headers']=headers;result['end_bundle_station']=bundle
                result['end_bundle_selection_evidence']={'selected_candidate':attempt,
                    'candidate_count':len(diagnostics),'geometry_source':'Keep physical equipment, ports, shell, framing and pull-out spaces fixed',
                    'scope':'Bounded bundle offsets and end positions on both short sides. Physical valve assembly and model measurements follow separately.'}
                return result
    _rejected('END_BUNDLE_UNAVAILABLE','All tried positions and short-side terminations conflict with physical space',
              candidates=diagnostics,spacing_mode=spacing_mode,
              corner_implemented=False,physical_unsatisfiable_claim=False)


def plan_end_bundle(layout, frozen, requirements, config, heights):
    """Try roomy spacing, then real-minimum compact spacing if it is refused.

    Both modes share the original sixteen translations per side/endpoint.
    Compact removes discretionary spare room only: its pitches are the actual
    OD/fitting envelopes plus the unchanged original gaps and 1 micrometre of
    positive numerical margin. Physical acceptance remains unchanged.
    """
    missing=required_header_keys(requirements)-{(h['medium'],h['level']) for h in layout.get('headers',[])}
    if missing:
        _rejected('END_BUNDLE_REQUIRED_HEADER_MISSING', 'A physical equipment user lacks its floor header',
                  required_same_floor_headers=sorted(missing))
    diagnostics=[]
    strategies = ((bounded, mode) for bounded in (False, True) for mode in ('roomy', 'compact'))
    for bounded, mode in strategies:
        try:
            result=_plan_end_bundle_variant(layout,frozen,requirements,config,heights,
                mode,MAX_BUNDLE_TRANSLATIONS//2,allocate_within_span=bounded)
        except PipingStationRejected as exc:
            if exc.code!='END_BUNDLE_UNAVAILABLE':raise
            diagnostics.extend(exc.evidence.get('candidates',[]))
            continue
        diagnostics.extend(result['end_bundle_station']['candidate_diagnostics'])
        for index,row in enumerate(diagnostics,1):
            row['candidate_within_spacing_mode']=row.get('candidate')
            row['candidate']=index
        result['end_bundle_station']['candidate_diagnostics']=diagnostics
        result['end_bundle_station']['translation_budget_per_side_endpoint']=MAX_BUNDLE_TRANSLATIONS*(1+bounded)
        result['end_bundle_selection_evidence'].update(selected_candidate=len(diagnostics),
            candidate_count=len(diagnostics),spacing_mode=mode,allocate_tees_within_span=bounded,
            translation_budget_per_mode=MAX_BUNDLE_TRANSLATIONS//2)
        return result
    _rejected('END_BUNDLE_UNAVAILABLE','No short-side candidates passed with either relaxed or compact bundle spacing',
        candidates=diagnostics,spacing_modes=['roomy','compact'],
        translation_budget_per_side_endpoint=2*MAX_BUNDLE_TRANSLATIONS,
        corner_implemented=False,physical_unsatisfiable_claim=False)
