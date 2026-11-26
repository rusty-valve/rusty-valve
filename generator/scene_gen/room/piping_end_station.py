"""Build the connections between floor pipes and their main risers."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .evaluation_common import require_environment
from .piping_assets import inspect_asset, AssetUnavailable
from .piping_build import cut_pipe_openings, _native_contacts, _nearby_contacts, _overlap
from .piping_elbows import coarse_radius, plan_polyline
from .piping_end_bundle import DEMAND_ONLY_TERMINALS, DIRECT_CONNECTION_KINDS
from .piping_lane_connect import _obstacles, _clear_plan
from .piping_station import _pipe_box
from .piping_valve_placement import (physical_valve_obstacles, real_valve_candidates,
                                     forward_pipe_positions)
from .process import write_json
from .recipe import CandidateRefused

EPS = 1e-8


class EndStationRefused(CandidateRefused):
    def __init__(self, code, detail, **evidence):
        self.code = code
        self.evidence = {'code': code, 'detail': detail, **evidence}
        super().__init__(code+': '+detail)


def _point(value, label):
    p = np.asarray(value, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise EndStationRefused('END_STATION_POINT', label+'missing finite three-dimensional coordinates')
    return p


def _same(first, second):
    return np.max(abs(np.asarray(first)-second)) <= EPS


def _direction(connection):
    direction = _point(connection['direction'], 'direction')
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        raise EndStationRefused('END_STATION_DIRECTION', 'Termination direction cannot be zero')
    direction /= norm
    if np.count_nonzero(abs(direction)>EPS) != 1 or abs(direction[2])>EPS:
        raise EndStationRefused('END_STATION_DIRECTION', 'Termination must follow a horizontal building axis')
    return direction


def _identity(connection, header):
    return {'main_id': connection['main_id'], 'distribution_id': connection.get('header_id'),
            'service_group': header.get('service_group') if header else None,
            'connection_id': connection['id'], 'end_station_method': 'end_bundle_v1'}


def _header(a, connection):
    identifier = connection.get('header_id')
    if identifier is None:
        return None
    header = next((r for r in a.routes if r['id']==identifier and r.get('role')=='floor_header'), None)
    if header is None:
        raise EndStationRefused('END_STATION_HEADER_MISSING', 'Required header geometry was not emitted', header=identifier)
    if header['medium'] != connection['medium'] or header['level'] != connection['level'] or header['dn'] != connection['dn']:
        raise EndStationRefused('END_STATION_HEADER_IDENTITY', 'Header service, floor or DN differs from its termination contract', header=identifier)
    near = _point(connection['header_near_tangent'], 'header_near_tangent')
    endpoints = [header['control_points'][0], header['control_points'][-1]]
    if not any(_same(near, p) for p in endpoints):
        raise EndStationRefused('END_STATION_HEADER_ENDPOINT', 'The header must end at the physical tangent point',
                                header=identifier, near=near.tolist(), actual_endpoints=endpoints)
    if any(math.dist(b['position'], near.tolist())<=EPS for b in a.blinds):
        raise EndStationRefused('END_STATION_NEAR_CAP', 'The near end of the header is still blinded', header=identifier)
    return header


def _check_contract(a, full_requirements, bundle):
    if bundle.get('version') != 'end_bundle_v1':
        raise EndStationRefused('END_STATION_VERSION', 'Missing explicit end_bundle_v1 contract')
    if bundle.get('mode') != 'short_edge':
        raise EndStationRefused('END_STATION_MODE_UNSUPPORTED',
            'Only specified straight or two-bend short-side terminations are emitted', mode=bundle.get('mode'))
    axis = bundle['axis_index']; cross = bundle['cross_axis_index']
    if axis not in (0, 1) or cross != 1-axis:
        raise EndStationRefused('END_STATION_AXES', 'Longitudinal and vertical-plane transverse axes are incompatible')
    risers = bundle['risers']; connections = bundle['connections']
    by_medium = {r['medium']: r for r in risers}
    if len(by_medium) != len(risers) or set(by_medium) != set(full_requirements['active_services']):
        raise EndStationRefused('END_STATION_SERVICES', 'Mains must cover exactly the physical services')
    ids = [c['id'] for c in connections]
    if len(set(ids)) != len(ids):
        raise EndStationRefused('END_STATION_DUPLICATE_CONNECTION', 'Duplicate termination identifier')
    expected = set()
    demand_only = bundle.get('terminals_policy')==DEMAND_ONLY_TERMINALS
    for medium in full_requirements['active_services']:
        for level in range(len(a.floors)):
            headers = [r for r in a.routes if r.get('role')=='floor_header'
                       and r['medium']==medium and r['level']==level]
            users = [p for p in full_requirements['service_points']+full_requirements.get('header_demands',[])
                     if p['medium']==medium and p['level']==level]
            if users and not headers:
                raise EndStationRefused('END_STATION_REQUIRED_HEADER_MISSING', 'The level has equipment users but no physical header', medium=medium, level=level)
            if demand_only and headers and not users:
                raise EndStationRefused('END_STATION_UNREQUIRED_HEADER', 'A level without equipment users must have no header', medium=medium, level=level)
            if headers:
                expected.update((medium, level, h['id']) for h in headers)
            elif not demand_only and medium != 'offgas':
                expected.add((medium, level, None))
    actual = [(c['medium'], c['level'], c.get('header_id')) for c in connections]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise EndStationRefused('END_STATION_CONNECTION_COVERAGE', 'Termination or spare root-valve list has missing, duplicate or extra services',
                                expected=sorted(expected, key=str), actual=actual)
    if 'planned_takeoff_spacing' in bundle.get('spacing',{}):
        from .piping_end_bundle import takeoff_spacing
        invalid=[row for row in takeoff_spacing(risers,connections) if not row['pass']]
        if invalid:
            raise EndStationRefused('END_STATION_TAKEOFF_SPACING','Adjacent main branches are staggered by less than 300 mm',pairs=invalid)
    for c in connections:
        direction = _direction(c)
        if int(np.argmax(abs(direction))) != axis:
            raise EndStationRefused('END_STATION_WRONG_AXIS', 'Termination direction differs from the header longitudinal axis', connection=c['id'])
        riser = by_medium[c['medium']]
        if c['main_id'] != riser['id'] or c['dn'] != riser['branch_dn']:
            raise EndStationRefused('END_STATION_MAIN_IDENTITY', 'Termination main and branch diameters differ', connection=c['id'])
        plane = bundle['plane_coordinates_m'][c['medium']]
        if (c['plane_axis_index'] != cross or abs(riser['xy'][cross]-plane)>EPS
                or abs(c['plane_coordinate_m']-plane)>EPS):
            raise EndStationRefused('END_STATION_PLANE', 'The main and header lie on different service planes', connection=c['id'])
        points = [c['main_tap']]
        direct = c['medium']=='offgas' or demand_only and c['medium']=='water_return'
        if direct:
            if c['kind']!=c['medium'] or not c.get('header_id') or c.get('valve_in') or c.get('valve_out'):
                raise EndStationRefused('END_STATION_KIND', 'Direct merge requires a header with no root valve', connection=c['id'])
            points.append(c['header_near_tangent'])
        elif c['kind'] in ('supply', 'standby'):
            if c['medium'] == 'offgas' or bool(c.get('header_id')) != (c['kind']=='supply'):
                raise EndStationRefused('END_STATION_KIND', 'Invalid supply or spare branch identity', connection=c['id'])
            asset = inspect_asset(c['dn'])
            if abs(c['native_L_m']-asset['install_length_m'])>EPS or (c.get('package') and c['package']!=asset['package']):
                raise EndStationRefused('END_STATION_NATIVE', 'Contract length or valve package differs from the actual DN package', connection=c['id'])
            if not _same(_point(c['valve_out'], 'valve_out')-_point(c['valve_in'], 'valve_in'), direction*asset['install_length_m']):
                raise EndStationRefused('END_STATION_NATIVE_LENGTH', 'Root valve face spacing or axis differs', connection=c['id'])
            points += [c['valve_in'], c['valve_out']]
            if c['kind']=='supply':
                R = coarse_radius(a.radius(c['dn']))
                if abs(c['R_m']-R)>EPS:
                    raise EndStationRefused('END_STATION_ELBOW_R', 'Use the original physical elbow tangent radius R equal to 3r', connection=c['id'], required_R_m=R)
                lower = _point(c['lower_corner'], 'lower_corner'); upper = _point(c['upper_corner'], 'upper_corner')
                if not _same(upper-lower, [0., 0., upper[2]-lower[2]]) or upper[2]<=lower[2]:
                    raise EndStationRefused('END_STATION_RISE', 'Supply must rise on its fixed service-plane axis', connection=c['id'])
                if not _same(c['header_near_tangent'], upper+direction*R):
                    raise EndStationRefused('END_STATION_TANGENT', 'The header near end is not the physical upper elbow tangent', connection=c['id'])
                points += [c['lower_corner'], c['upper_corner'], c['header_near_tangent']]
            else:
                points.append(c['blind_point'])
        else:
            raise EndStationRefused('END_STATION_KIND', 'Unknown termination kind', connection=c['id'])
        if any(abs(_point(p, 'connection point')[cross]-plane)>EPS for p in points):
            raise EndStationRefused('END_STATION_PLANE', 'Termination leaves the service plane', connection=c['id'])
        if not _same(c['main_tap'][:2], riser['xy']):
            raise EndStationRefused('END_STATION_MAIN_TAP', 'The side outlet is off the specified main centerline', connection=c['id'])


def _opening_key(opening):
    return int(opening['level']), tuple(float(v) for v in opening['rect']), opening.get('owner')


def _emit_mains(a, bundle, result):
    known = {_opening_key(o) for o in a.openings}
    added = [deepcopy(o) for o in bundle.get('openings', []) if _opening_key(o) not in known]
    if added:
        cut_pipe_openings(a.scene, added)
        a.openings.extend(added)
    a.scene['piping_openings'] = deepcopy(a.openings)
    a.plan['station'] = {k: deepcopy(bundle[k]) for k in ('edge', 'inward', 'risers', 'openings') if k in bundle}
    a.plan['station'].update(tangent=[1. if i==bundle['cross_axis_index'] else 0. for i in range(3)],
        mode=bundle['mode'], version=bundle['version'], spacing={'plane_coordinates_m':deepcopy(bundle['plane_coordinates_m']),
        'media_order':deepcopy(bundle['media_order']), 'same_as_header_planes':True})
    mains = {}
    for riser in bundle['risers']:
        medium = riser['medium']; top = [*riser['xy'], riser['top_z']]
        route = a.route(riser['id'], [[*riser['xy'], riser['bottom_z']], top], riser['dn'], medium,
            'workstation_main', 0, source={'position':[*riser['xy'], riser['bottom_z']], 'boundary':True},
            target={'position':top, 'blind':True}, served_levels=riser['served_levels'], end_station_method='end_bundle_v1')
        cap = a.blind(riser['id'], top, [0., 0., 1.], riser['dn'], medium, len(a.floors)-1, 'main_end')
        flange = a.flange_interface(riser['id'], np.asarray(top)-[0., 0., .039], [0., 0., 1.],
                                    riser['dn'], medium, len(a.floors)-1)
        a.allow(route['components'][-1], cap['id'], top)
        mains[medium] = route
        result['mains'].append({'route':route['id'], 'blind':cap['id'], 'flange':flange['id']})
    return mains


def _flat_contact_ids(a, header, point):
    """Only the actual terminal pieces containing this constructed joint."""
    selected = set(header['components'])
    result = []
    p = np.asarray(point)
    for part in a.scene['components']:
        if part['id'] not in selected:
            continue
        box = np.asarray(part['box'])
        if np.all(p>=box[:3]-EPS) and np.all(p<=box[3:]+EPS):
            result.append(part['id'])
    return result


def _tail_screen(a, plan, header, near, radius, direction, *, tee_source=None):
    physical = physical_valve_obstacles(a)
    # This path approaches an open flat pipe end from its outside half-space.
    # Only the actual terminal piece can touch there, not all header owners.
    terminal = set(_flat_contact_ids(a, header, near)) if header else set()
    if header and not terminal:
        raise EndStationRefused('END_STATION_NO_TERMINAL', 'No real header face at the termination tangent', header=header['id'])
    if terminal:
        near = np.asarray(near); axis = int(np.argmax(abs(direction)))
        sign = float(direction[axis]); boundary = float(near[axis]*sign)
        for run in plan['runs']:
            first=np.asarray(run['start']);last=np.asarray(run['end']);u=(last-first)/np.linalg.norm(last-first)
            extent=radius*np.sqrt(np.maximum(0., 1-u*u))
            if max((first[axis]+sign*extent[axis])*sign, (last[axis]+sign*extent[axis])*sign)>boundary+EPS:
                raise EndStationRefused('END_STATION_HEADER_OVERLAP', 'Termination straight penetrates the inside of the header')
        for elbow in plan['elbows']:
            radial=np.asarray(elbow['radial_start']);u=np.asarray(elbow['tangent_in'])
            for theta,p in zip(np.linspace(0.,math.pi/2,len(elbow['centerline_points'])), elbow['centerline_points']):
                tangent=-radial*math.sin(theta)+u*math.cos(theta)
                extent=radius*math.sqrt(max(0.,1-tangent[axis]**2))
                if p[axis]*sign+extent>boundary+EPS:
                    raise EndStationRefused('END_STATION_HEADER_OVERLAP', 'Termination elbow penetrates the inside of the header')
        physical = [p for p in physical if p['id'] not in terminal]
    obstacles,_ = _obstacles(physical)
    gap = a.config['pg_rules']['parallel_pipe_spacing']['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.
    source = {'position':plan['points'][0]} if tee_source is None else tee_source
    end_direction = np.asarray(plan['points'][-1])-plan['points'][-2]
    end_direction /= np.linalg.norm(end_direction)
    hit = _clear_plan(plan, obstacles, radius, gap if tee_source is None else 0., np.asarray(a.bounds),
        source, {'position':plan['points'][-1]}, direction, end_direction,
        is_lead=tee_source is not None,is_tail=True)
    if hit:
        if tee_source is not None:
            raise EndStationRefused('END_STATION_PIPE_BLOCKED','Branch meets a solid',obstacle=hit)
        raise EndStationRefused('END_STATION_FIXED_ROUTE_BLOCKED', 'A solid blocks the fixed termination route', obstacle=hit)


def _flange_box(a, outlet, direction, dn, length):
    first = np.asarray(outlet)+direction*(length/2)
    last = first+direction*.039
    return first, _pipe_box(first, last, a.radius(dn)+.034)


def _valve_slot(a, connection, candidate, main, header):
    direction=_direction(connection);dn=connection['dn'];radius=a.radius(dn)
    length=float(candidate['valve']['install_length_m'])
    start=_point(connection['main_tap'],'main_tap');inlet=np.asarray(candidate['in_position']);outlet=np.asarray(candidate['out_position'])
    if float((inlet-start)@direction)<length-EPS:
        raise EndStationRefused('END_STATION_BEFORE_VALVE_ROOM', 'Pre-valve straight is shorter than one physical valve installation length')
    if connection['kind']=='supply':
        tail_points=[outlet,connection['lower_corner'],connection['upper_corner'],connection['header_near_tangent']]
        tail_plan=plan_polyline(tail_points,radius)
        if len(tail_plan['elbows'])!=2:
            raise EndStationRefused('END_STATION_SUPPLY_BENDS', 'Supply termination requires two physical 90-degree elbows')
        tangent=np.asarray(tail_plan['runs'][0]['end'])
    else:
        tail_points=[outlet,connection['blind_point']]
        tail_plan=plan_polyline(tail_points,radius)
        if tail_plan['elbows']:
            raise EndStationRefused('END_STATION_STANDBY_BENDS', 'The spare branch ends at a blind after a short post-valve straight')
        tangent=_point(connection['blind_point'],'blind_point')
    post=float((tangent-outlet)@direction)
    if post<length-EPS:
        raise EndStationRefused('END_STATION_AFTER_VALVE_ROOM', 'Post-valve straight to the elbow tangent or blind is shorter than one valve length', actual_m=post, required_m=length)
    if header:
        flange_point,flange_box=_flange_box(a,outlet,direction,dn,length)
    else:
        end=_point(connection['blind_point'],'blind_point')
        flange_point=end-direction*.039
        flange_box=_pipe_box(flange_point,end+direction*.018,a.radius(dn)+.034)
    if (header and length/2+.039>post+EPS) or (not header and post<.039-EPS):
        raise EndStationRefused('END_STATION_FLANGE_ROOM', 'The 39 mm physical flange interface cannot fit in the post-valve straight')
    physical=physical_valve_obstacles(a)
    pg=a.config['pg_rules']['parallel_pipe_spacing'];protrusion=pg['protrusions']['min_clearance_mm']/1000.
    wall=pg['outermost_pipe_or_insulation_to_column_or_wall_min_mm']/1000.
    flange_hits=[]
    for part in [*physical,*({'id':candidate['valve']['prefix']+g['name'],'kind':'own_valve','box':g['box']} for g in candidate['valve']['world_geoms'])]:
        gap=wall if part['kind'] in ('wall','column') else protrusion if part['kind'] in ('pipe','piping_flange','piping_blind','valve') else 0.
        box=np.asarray(part['box'])+np.r_[-np.full(3,gap),np.full(3,gap)]
        if _overlap(flange_box,box):flange_hits.append(part['id'])
    if flange_hits:
        raise EndStationRefused('END_STATION_FLANGE_BLOCKED', 'Occupied space at the post-valve flange', obstacles=flange_hits)
    lead_plan=plan_polyline([inlet,start],radius)
    obstacles,_=_obstacles(physical)
    bare=pg['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.
    hit=_clear_plan(lead_plan,obstacles,radius,bare,np.asarray(a.bounds),
        {'position':inlet.tolist()}, {'position':start.tolist(),'contact_ids':main['components']},
        -direction,-direction,is_tail=True)
    if hit:
        raise EndStationRefused('END_STATION_LEAD_BLOCKED', 'A solid blocks the fixed side outlet before the root valve', obstacle=hit)
    _tail_screen(a,tail_plan,header,connection.get('header_near_tangent'),radius,direction)
    return tail_points,flange_point,post


def _supply(a, connection, main, header):
    direction=_direction(connection);dn=connection['dn'];length=inspect_asset(dn)['install_length_m']
    nominal=_point(connection['valve_in'],'valve_in');start=_point(connection['main_tap'],'main_tap')
    positions=forward_pipe_positions(a,dn,nominal,direction)
    # Include the ends of the permissible low straight, without moving its
    # corners or prescribing a different valve row for each service.
    stop=(_point(connection['lower_corner'],'lower_corner')-direction*coarse_radius(a.radius(dn))
          if header else _point(connection['blind_point'],'blind_point'))
    defer_valves = hasattr(a, 'path_requirements')
    for point in (start+direction*length, stop-direction*(2*length)):
        if not any(_same(point,p) for p in positions):positions.append(point.tolist())
    identity=_identity(connection,header);attempts=[];chosen=None
    candidates=() if defer_valves else real_valve_candidates(a,connection['id'],dn,nominal,direction,[0.,0.,1.],
        connection['medium'],'utility_floor_root',connection['level'],in_positions=positions,metadata=identity)
    for candidate in candidates:
        if not candidate['physical_fit']:
            attempts.append({'code':'END_STATION_VALVE_POSE_BLOCKED',
                'physical_hits':candidate['physical_hits'],'axial_shift_m':candidate['axial_shift_m'],
                'stem_rotation_quarters':candidate['stem_rotation_quarters']});continue
        try:
            tail_points,flange_point,post=_valve_slot(a,connection,candidate,main,header)
        except CandidateRefused as exc:
            attempts.append({'code':getattr(exc,'code',type(exc).__name__),'reason':str(exc),
                'evidence':getattr(exc,'evidence',None),'axial_shift_m':candidate['axial_shift_m'],
                'stem_rotation_quarters':candidate['stem_rotation_quarters']});continue
        chosen=candidate;break
    if chosen is None:
        radius=a.radius(dn)
        points=[start,connection['lower_corner'],connection['upper_corner'],connection['header_near_tangent']] if header else [start,connection['blind_point']]
        plan=plan_polyline(points,radius)
        tee={'position':start.tolist(),'contact_ids':main['components'],'tee':True}
        _tail_screen(a,plan,header,connection.get('header_near_tangent'),radius,direction,tee_source=tee)
        target={'position':points[-1], **({'header_id':header['id']} if header else {'blind':True})}
        pipe=a.route(connection['id']+'_continuous',points,dn,connection['medium'],'floor_branch',
            connection['level'],**identity,source={'position':start.tolist(),'main_id':main['id']},
            target=target,root_valve_pending=True)
        _nearby_contacts(a,pipe['components'][:1],start)
        _nearby_contacts(a,pipe['components'][-1:],points[-1])
        if not header:
            end = _point(connection['blind_point'], 'blind_point')
            cap = a.blind(connection['id']+'_out', end, direction, dn, connection['medium'],
                          connection['level'], 'unused_floor_branch')
            a.allow(pipe['components'][-1], cap['id'], end)
            a.flange_interface(connection['id']+'_out', end-direction*.039, direction, dn,
                               connection['medium'], connection['level'])
        return {'valve':None,'actual_plan':{'connection_id':connection['id'],'route_ids':[pipe['id']],
                'valve_state':'pending','kind':connection['kind']}}
    valve=chosen['valve'];inlet=valve['world_ports']['in']['position'];outlet=valve['world_ports']['out']['position']
    name=connection['id'];level=connection['level'];medium=connection['medium']
    lead=a.route(name+'_in',[start.tolist(),inlet],dn,medium,'floor_branch',level,**identity,
        source={'position':start.tolist(),'main_id':main['id']},target={'position':inlet,'valve':name},route_first=True)

    _nearby_contacts(a,lead['components'][:1],start)
    target=({'position':connection['header_near_tangent'],'header_id':header['id']} if header
            else {'position':connection['blind_point'],'blind':True})
    tail=a.route(name+'_out',tail_points,dn,medium,'floor_branch',level,**identity,
        source={'position':outlet,'valve':name},target=target,route_first=True)
    from .pipe_slots import spool
    spool(a,name,[inlet,outlet],dn,medium,level,'utility_floor_root',identity,direction,[0.,0.,1.],lead,tail,identity)
    a.routes[-1]['role']='floor_branch'
    flange=a.flange_interface(name+'_active_interface',flange_point,direction,dn,medium,level) if header else None
    closure={}
    if header:
        _nearby_contacts(a,tail['components'][-1:],connection['header_near_tangent'])
    else:
        end=_point(connection['blind_point'],'blind_point')
        cap=a.blind(name+'_out',end,direction,dn,medium,level,'unused_floor_branch')
        a.allow(tail['components'][-1],cap['id'],end)
        flange=a.flange_interface(name+'_out',end-direction*.039,direction,dn,medium,level)
        closure={'blind':cap['id'],'flange':flange['id']}
    actual_plan={'connection_id':name,'kind':connection['kind'],'medium':medium,'level':level,
        'header_id':header['id'] if header else None,'main_id':main['id'],'route_ids':[lead['id'],tail['id']],
        'valve_id':name,'valve_state':'reserved_pending_native_install','native_package':valve['package'],
        'native_L_m':valve['install_length_m'],'valve_in':deepcopy(inlet),'valve_out':deepcopy(outlet),
        'plane_axis_index':connection['plane_axis_index'],'plane_coordinate_m':connection['plane_coordinate_m'],
        'lead_control_points':deepcopy(lead['control_points']),'tail_control_points':deepcopy(tail['control_points']),
        'emitted_elbow_ids':deepcopy(tail['elbows']),'emitted_bend_count':tail['bend_count'],
        'method_valve_placement':deepcopy(valve['method_valve_placement']),
        'post_valve_straight_to_tangent_m':post,'actual_measurement_pending':True}
    return {'valve':name,'in_route':lead['id'],'out_route':tail['id'],
        'header':header['id'] if header else None,'unused_floor':header is None,
        'active_interface':flange['id'],'actual_plan':actual_plan,**closure}


def _direct_link(a, connection, main, header):
    near=_point(connection['header_near_tangent'],'header_near_tangent')
    if not _same(near,connection['main_tap']):
        raise EndStationRefused('END_STATION_DIRECT_TAP', 'Direct-merge header end does not coincide with the main tee center')
    if header.get('elbows'):
        raise EndStationRefused('END_STATION_DIRECT_HEADER_BENT', 'The common header for a direct merge must be straight')
    endpoints=[header['control_points'][0],header['control_points'][-1]]
    endpoint='source' if _same(endpoints[0],near) else 'target'
    header[endpoint]={'position':near.tolist(),'main_id':main['id']}
    header.update(main_id=main['id'],end_connection_id=connection['id'],end_station_method='end_bundle_v1')
    terminal=_flat_contact_ids(a,header,near)
    if not terminal:
        raise EndStationRefused('END_STATION_DIRECT_NO_TERMINAL', 'Direct-merge header end has no physical geometry')
    _nearby_contacts(a,terminal,near)
    return {'header':header['id'],'main':main['id'],'route_ids':[header['id']],
        'actual_plan':{'connection_id':connection['id'],'kind':connection['kind'],'main_id':main['id'],
        'header_id':header['id'],'medium':connection['medium'],'level':connection['level'],'route_ids':[header['id']],
        'native_valve_ids':[],'emitted_bend_count':0,'plane_axis_index':connection['plane_axis_index'],
        'plane_coordinate_m':connection['plane_coordinate_m'],'join_position':near.tolist(),
        'direct_main_touch_pending_actual_measurement':True}}


def add_end_station(a, full_requirements):
    """Emit one explicit end station; reserve native valves for caller install."""
    require_environment()
    result={'status':'partial','mains':[],'fixed':[],'direct_links':[],'failures':[],
        'completion':{'required_mains':0,'mains_emitted':0,'required_supply_branches':0,
            'supply_branches_complete':0,'required_supply_roots':0,'supply_roots_reserved':0,
        'supply_roots_installed':0,'required_offgas_links':0,'offgas_links_complete':0,
        'required_water_return_links':0,'water_return_links_complete':0}}
    bundle=a.plan.get('end_bundle_station')
    if not isinstance(bundle,dict):
        result['failures'].append({'code':'END_STATION_PLAN_MISSING','reason':'Missing end_bundle_station. No station was emitted.'})
        return result
    try:
        _check_contract(a,full_requirements,bundle)
    except (CandidateRefused,AssetUnavailable,KeyError,TypeError,ValueError) as exc:
        result['failures'].append({'stage':'contract','code':getattr(exc,'code',type(exc).__name__),
            'reason':str(exc),'evidence':getattr(exc,'evidence',None)})
        return result
    a.requirements=deepcopy(full_requirements)
    result['completion'].update(required_mains=len(bundle['risers']),
        required_supply_branches=sum(c['kind'] not in DIRECT_CONNECTION_KINDS for c in bundle['connections']),
        required_supply_roots=sum(c['kind'] not in DIRECT_CONNECTION_KINDS for c in bundle['connections']),
        required_offgas_links=sum(c['kind']=='offgas' for c in bundle['connections']),
        required_water_return_links=sum(c['kind']=='water_return' for c in bundle['connections']))
    mains=_emit_mains(a,bundle,result)
    failed_geometry=[];actual_plans=[]
    for connection in bundle['connections']:
        snapshot=deepcopy(a)
        try:
            header=_header(a,connection)
            main=mains[connection['medium']]
            direct = connection['kind'] in DIRECT_CONNECTION_KINDS
            item=(_direct_link(a,connection,main,header) if direct
                  else _supply(a,connection,main,header))
            result['direct_links' if direct else 'fixed'].append(item)
            actual_plans.append(item['actual_plan'])
        except (CandidateRefused,AssetUnavailable,ValueError) as exc:
            old_ids={c['id'] for c in snapshot.scene['components']}
            failed_geometry.append({'connection':connection['id'],
                'components':[deepcopy(c) for c in a.scene['components'] if c['id'] not in old_ids],
                'routes':deepcopy(a.routes[len(snapshot.routes):]),
                'valves':[deepcopy(v) for v in a.all_valves() if v['id'] not in {q['id'] for q in snapshot.all_valves()}]})
            result['failures'].append({'connection_id':connection['id'],'medium':connection['medium'],
                'level':connection['level'],'header_id':connection.get('header_id'),
                'reason':str(exc),'code':getattr(exc,'code',type(exc).__name__),'evidence':getattr(exc,'evidence',None)})
            a.__dict__.clear();a.__dict__.update(snapshot.__dict__)
    if failed_geometry:
        write_json(a.folder/'end_station_failed_connection_geometry.json',failed_geometry)
    a.plan['end_bundle_station']['actual_connection_plans']=deepcopy(actual_plans)
    ids={r['valve'] for r in result['fixed']}
    result['completion'].update(mains_emitted=len(result['mains']),
        supply_branches_complete=len(result['fixed']),
        supply_roots_reserved=sum(v['id'] in ids for v in a.planned_valves),
        supply_roots_installed=sum(v['id'] in ids for v in a.graph['valves']),
        offgas_links_complete=sum(r['actual_plan']['kind']=='offgas' for r in result['direct_links']),
        water_return_links_complete=sum(r['actual_plan']['kind']=='water_return' for r in result['direct_links']))
    if not result['failures']:
        result['status']='routed_pending_native_install_and_actual_measurement'
    return result
