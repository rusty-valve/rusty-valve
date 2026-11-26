"""Build utility pipes and connect them to equipment on each floor."""
from __future__ import annotations
from .root_orientation import planning_geoms
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import time
import numpy as np

from .evaluation_common import require_environment
from .piping_inputs import load_frozen_layout,assert_frozen_inputs
from .piping_requirements import requirements_from_frozen,load_stage_config
from .piping_station import (_heights,plan_service_headers,_frame,_shared_supports,
                             _protected_boxes,PipingStationRejected)
from .piping_header_taps import tee_reservation,adjacent_reservation
from .piping_build import Assembly,_unit,_nearby_contacts,_native_contacts
from .piping_assets import inspect_asset
from .piping_service_groups import shared_service_groups_from_frozen
from .piping_elbows import coarse_radius,plan_polyline,SECTOR_DEGREES
from .process import write_json
from .recipe import InputRejected,CandidateRefused


def _tap_table(a,h):
    """Use the pre-emission joint table, never consume its future main interval."""
    joint=h.get('method_joint_taps')
    if not joint:raise CandidateRefused('Reserve the future main tee together with equipment tees before generating the header')
    expected=set(h['point_ids'])
    if set(joint['device_taps'])!=expected:raise CandidateRefused('Joint tee list differs from the original equipment demands')
    a.header_tap_plans[h['id']]=deepcopy(joint)
    return deepcopy(joint['device_taps'])


def _end_device_taps(header,requirements,config):
    """Keep device body tees; the new main connection consumes an end only."""
    axis=int(header['axis_index']);origin=np.asarray(header['start'],float)
    end=np.asarray(header['end'],float)
    if end[axis]<origin[axis]:origin,end=end,origin
    length=float(end[axis]-origin[axis]);rows=[]
    for point in sorted((p for p in requirements['service_points'] if p['id'] in header['point_ids']),
                        key=lambda p:(p['position'][axis],p['position'][1-axis],p['id'])):
        reserve=tee_reservation(config,header['dn'],point['dn'],0. if point.get('header_only') else inspect_asset(point['dn'])['install_length_m'])
        edge=.039+reserve['end_reserve_m']
        preference=header.get('end_bundle_device_tap_preferences',{}).get(point['id'],{})
        preferred=preference.get('axis_coordinate_m',point['position'][axis])-origin[axis]
        rows.append({'point':point,'reservation':reserve,'preferred':preferred,
                     'original_projection':point['position'][axis]-origin[axis],
                     'low':edge,'high':length-edge})
    for first,second in zip(rows,rows[1:]):
        gap=adjacent_reservation(first['reservation'],second['reservation'])['required_center_distance_m']
        second['low']=max(second['low'],first['low']+gap)
    for first,second in zip(reversed(rows[:-1]),reversed(rows[1:])):
        gap=adjacent_reservation(first['reservation'],second['reservation'])['required_center_distance_m']
        first['high']=min(first['high'],second['high']-gap)
    if any(row['low']>row['high']+1e-8 for row in rows):
        raise PipingStationRejected('END_BUNDLE_DEVICE_T_SPACE','Termination header cannot fit the required spacing between real equipment tees',
                                    {'header':header['id'],'header_length_m':length})
    taps={};previous=None;displacement=0.
    for row in rows:
        low=row['low']
        if previous is not None:
            low=max(low,previous['value']+adjacent_reservation(previous['reservation'],row['reservation'])['required_center_distance_m'])
        value=min(row['high'],max(low,row['preferred']));row['value']=value
        position=origin.copy();position[axis]+=value
        taps[row['point']['id']]={'position':position.tolist(),'source_projection_m':row['original_projection'],
            'selected_projection_m':value,'reservation':deepcopy(row['reservation'])}
        displacement+=abs(value-row['original_projection']);previous=row
    endpoint=header['end_bundle']
    return {'device_taps':taps,'end_connection':{'position':deepcopy(endpoint['near_point']),
                'dn':header['dn'],'kind':'connected_header_end_not_body_tee'},
            'canonical_axis':[1. if i==axis else 0. for i in range(3)],'origin':origin.tolist(),
            'header_length_m':length,'device_projection_displacement_m':displacement,
            'scope':'Equipment tees use the header body. The main connects at the near end and the far end is blinded.'}


def _physical_obstacles(a,current_point=None):
    # The native installed-pose pieces are physical. Standing and motion
    # envelopes remain reporting data; they are never routing obstacles.
    components=[c for c in a.scene['components'] if not c.get('annotation_only') and not c.get('visual_only')]
    components=deepcopy(components)
    # Only installed valves are solids; planned valves never steer a pipe.
    for v in a.graph['valves']:
        components.extend({'id':v['prefix']+g['name'],'kind':'valve','shape':'box','box':g['box']}
                          for g in planning_geoms(v))
    for r in a.scene.get('pullout_reservations',[]):
        components.append({'id':r['id'],'kind':'pullout_reservation','shape':'box',
            'box':[r['rect'][0],r['rect'][1],r['base_z'],r['rect'][2],r['rect'][3],a.bounds[5]]})
    if current_point is not None and a.plan.get('end_bundle_station'):
        completed={r.get('source_requirement') for r in a.routes
                   if r.get('role')=='equipment_connection' and r.get('target',{}).get('header_id')}
        gap=a.config['pg_rules']['parallel_pipe_spacing']['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.
        for point in a.requirements['service_points']:
            if (point['id']==current_point['id'] or point['id'] in completed
                    or not _vertical_top_root(a,point)):continue
            radius=a.radius(point['dn']);R=coarse_radius(radius)
            zone=a.plan['rack_zones'][point['level']]
            header=next((h for h in a.plan['headers'] if h['level']==point['level']
                         and h['medium']==point['medium']),None)
            if header is None:continue
            source_point=np.asarray(point['position'],float)
            reducer_end=np.asarray(a.method_reducers[point['id']]['end'],float)
            # Reserve only the earliest mandatory straight, never the whole
            # corridor height. Exhaust must reach above its real header before
            # turning down; ordinary supply may turn as soon as the rack starts.
            first_tangent=zone['bottom_z_m']+radius
            if point['medium']=='offgas':
                first_tangent=max(first_tangent,header['start'][2]+max(
                    R+gap,header['outer_radius_m']+2*radius+gap))
            # The straight rise into the rack is a pipe rule; no valve length here.
            top=max(float(source_point[2]),float(reducer_end[2]),float(first_tangent))
            upper=source_point.copy();upper[2]=top
            extent=np.array([radius,radius,0.])
            components.append({'id':'planned_required_device_rise_'+point['id'],
                'kind':'planned_required_device_rise','shape':'cylinder',
                'ends':[source_point.tolist(),upper.tolist()],'radius':radius,
                'box':np.r_[source_point-extent,upper+extent].tolist(),
                'purpose':'Plan the minimum rise from the original +Z port'})
    return components


def _stem_order(a,point,axis):
    from .piping_build import _operation_side
    op=_operation_side(a.frozen,point) or {}
    side={'x_min':[-1.,0.,0.],'x_max':[1.,0.,0.],
          'y_min':[0.,-1.,0.],'y_max':[0.,1.,0.]}.get(op.get('side'),[1.,0.,0.])
    choices=[side,[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]]
    result=[]
    for v in choices:
        if abs(float(np.asarray(v)@axis))<1e-8 and v not in result:result.append(v)
    return result


def _source_kind(a,point):
    """Read the frozen kind rather than infer it from an equipment name."""
    units=a.frozen['contracts']['sizing']['graph']['units']
    return units[point['equipment']]['kind']


def _vertical_top_root(a,point):
    normal=_unit(point['normal'])
    local_roles={'bottom','bottom_root','drain','drain_root','pump_inlet','pump_outlet'}
    return (_source_kind(a,point)!='pump' and point.get('role') not in local_roles
            and point.get('port') not in ('bottom','drain')
            and np.max(abs(normal-np.array([0.,0.,1.])))<=1e-8)


def _root_position_evidence(a,point,red_end,asset):
    """Record the unchanged real valve space even when no route is emitted."""
    normal=_unit(point['normal']);length=float(asset['install_length_m'])
    inlet=np.asarray(red_end,float)+normal*length;outlet=inlet+normal*length
    return {'method_version':'v1.3','vertical_top_root_required':_vertical_top_root(a,point),
        'original_endpoint':point['endpoint'],'original_equipment_kind':_source_kind(a,point),
        'original_role':point.get('role'),'original_source_position':deepcopy(point['position']),
        'original_source_normal':deepcopy(point['normal']),'reducer_end':np.asarray(red_end).tolist(),
        'source_dn':point['source_dn'],'dn':point['dn'],'actual_native_package':asset['package'],
        'actual_install_length_m':length,'planned_vertical_slot':{
            'in':inlet.tolist(),'out':outlet.tolist(),'axis':normal.tolist(),
            'one_valve_space_each_side_m':length,'status':'reserved_geometry_not_installed'},
        'original_pullout_reservations':deepcopy([r for r in a.frozen['scene'].get('pullout_reservations',[])
            if r.get('owner',r.get('equipment'))==point['equipment']]),
        'physical_unsatisfiable_claim':False}


def _target_directions(point,header,red_end,join):
    """Exhaust must enter from above; other services may do so after real obstruction."""
    if point['medium']=='offgas':return [np.array([0.,0.,-1.])]
    perpendicular=1-int(header['axis_index']);direction=np.zeros(3)
    direction[perpendicular]=1. if join[perpendicular]>=red_end[perpendicular] else -1.
    return [direction,-direction,np.array([0.,0.,1.]),np.array([0.,0.,-1.])]


def _cross_floor_source_trial(a,point,red_end,valve_axis):
    return _physical_obstacles(a,point),{},[]


def _prepare_source_adapter(a,point):
    face=np.asarray(point['position']);normal=_unit(point['normal'])
    name='P1_root_'+point['id'];source=face
    contact_ids=list(point['source_contact_ids'])
    via=[];joint=None
    if point.get('source_catalogue_face'):
        from .catalogue_flange import emit_catalogue_adapter

        joint,component=emit_catalogue_adapter(a,point,name,point['medium'],point['dn'])
        end=np.asarray(joint['end'])
        return {'end':joint['end'],'component':component,
                'flow_components':[joint['through_bore'],component['id']],
                'flow_points':[face.tolist(),component['ends'][0]],
                'contact_ids':joint['departure_contact_ids']}
    if point.get('source_flange'):
        from .piping_equipment_flange import emit_flanged_stub

        joint=emit_flanged_stub(a,point,name)
        source=np.asarray(joint['end'])
        contact_ids=[*joint['source_components'],*joint['components']]
        via.append(joint['pipe_stub']);contact_ids.extend(via)
    end,component=a.reducer(name+'_reducer',source,normal,point['source_dn'],point['dn'],
                            point['medium'],source_radius=point['source_radius_m'])
    a.allow(via[-1] if via else point['source_nozzle_id'],component['id'],source,normal=normal)
    evidence={'source_requirement':point['id'],'endpoint':point['endpoint'],
              'level':point['level'],'preplaced_required_source_adapter':True}
    if joint is not None:
        evidence['equipment_flange_joint']=joint
    component.update(evidence);a.reducers[-1].update(evidence)
    return {'end':end.tolist(),'component':component,'flow_components':[*via,component['id']],
            'flow_points':[face.tolist(),source.tolist()] if via else [source.tolist()],
            'contact_ids':[*contact_ids,component['id']]}


def _attach_source_adapter(a,source_adapter,route):
    a.allow(source_adapter['component']['id'],route['components'][0],source_adapter['end'])
    route['components'][:0]=source_adapter['flow_components']
    route['points'][:0]=source_adapter['flow_points']
    points=[*source_adapter['flow_points'],source_adapter['end']]
    route['actual_planned_centerline_length_m']+=sum(math.dist(x,y) for x,y in zip(points,points[1:]))


def _connect_service(a,point,h,tap,headers,heights,top_entry_extra_m=0.):
    """Connect one equipment service mouth to its header; its root valve is fitted later.

    Same attempt budget, recording and choice as material pipes (piping_connect).
    """
    from .piping_lane_connect import plan_lane_connection,LaneUnavailable
    from .piping_connect import Attempts
    join=tap['position'];normal=_unit(point['normal'])
    cached_reducer=a.method_reducers[point['id']]
    red_end=np.asarray(cached_reducer['end'])
    asset=inspect_asset(point['dn'])
    root_evidence=_root_position_evidence(a,point,red_end,asset)
    contact_ids=cached_reducer['contact_ids']
    attempts=Attempts(a,point['id'])
    defer_valves = hasattr(a, 'path_requirements')
    if defer_valves and point['medium'] == 'offgas':
        from .connection_rule_checks import offgas_entry_geometry
        from .route_candidate_checks import candidate_path

        def service_guard(plan):
            path = candidate_path(plan, point['dn'])
            if not offgas_entry_geometry(path)[0]:
                raise CandidateRefused('FULL_ROUTE: off-gas must enter the header from above')
            return {}

        attempts.check_candidate = service_guard
    want = 3*float(asset['install_length_m'])+.02
    attempts.want=(want,0.)
    side_obstacles=[]
    for arrival in _target_directions(point,h,red_end,join):
        downward_supply=point['medium']!='offgas' and arrival[2]<-1e-8
        if downward_supply and not side_obstacles:continue
        for straight in dict.fromkeys((want, 0.)):
            if not attempts.open:break
            def one(arrival=arrival,straight=straight):
                obstacles,extra,holes=_cross_floor_source_trial(a,point,red_end,normal)
                try:
                    guide=plan_lane_connection({'position':red_end.tolist(),'normal':normal.tolist(),
                        'contact_ids':contact_ids,'method_v1_3':True,'medium':point['medium'],
                        'original_endpoint':point['endpoint'],'minimum_straight_m':straight,
                        'equipment_kind':_source_kind(a,point),'role':point.get('role'),**extra},
                        {'position':join,'normal':(-arrival).tolist(),'radius':a.radius(h['dn']),
                         'contact_ids':headers[h['id']]['components']},
                        obstacles,a.radius(point['dn']),a.bounds,a.floors[point['level']],
                        heights[point['level']]['top_z_m'],h['axis_index'],asset=None,
                        target_direction=arrival,clearance=.05,
                        rack_bottom_z=heights[point['level']]['bottom_z_m'],
                        exception='condenser_short' if _source_kind(a,point)=='condenser' else None,
                        top_entry_extra_m=top_entry_extra_m)
                except LaneUnavailable as exc:
                    evidence=deepcopy(getattr(exc,'evidence',None)) or {}
                    if abs(arrival[2])<1e-8 and evidence.get('rejections',{}).get('ACTUAL_OBSTACLE',0):
                        side_obstacles.extend(deepcopy(evidence.get('examples',[])))
                    raise CandidateRefused(str(exc)) from exc
                guide['source_floor_openings']=holes
                guide['service_entry_selection']={'target_direction':arrival.tolist(),
                    'supply_top_entry_after_side_obstacles':downward_supply,'minimum_straight_m':straight}
                return [np.asarray(q) for q in guide['tail']],{'radius':a.radius(point['dn']),'guide':guide}
            attempts.run('header_lane_'+('valve_room' if straight else 'plain')+'_'+
                         ''.join('xyz'[k]+('+' if arrival[k]>0 else '-') for k in range(3) if abs(arrival[k])>.5),one)
    best=attempts.best()
    a.__dict__.setdefault('service_attempts',[]).append(attempts.report())
    if best is None:
        last=next((r for r in reversed(attempts.records) if r['status']=='rejected'),{})
        return None,{'requirement':point['id'],'endpoint':point['endpoint'],
                     'reason':last.get('reason','No available approach direction'),'evidence':last.get('evidence'),
                     'root_position_evidence':root_evidence,'attempts':attempts.report()}
    guide=best[2]['guide']
    holes=guide.get('source_floor_openings')
    if holes:
        from .piping_build import cut_pipe_openings
        cut_pipe_openings(a.scene,holes);a.openings.extend(holes)
        a.scene['piping_openings']=deepcopy(a.openings)
    common={'source_requirement':point['id'],'distribution_id':h['id'],
            'service_group':h.get('service_group'),'equipment':point['equipment'],
            'group_id':point.get('group_id'),'route_first':True,'connection_method':'one_connection_logic_v1'}
    route=a.route('P1_root_'+point['id']+'_connection',guide['tail'],point['dn'],point['medium'],
                  'equipment_connection',point['level'],**common,
                  source={'endpoint':point['endpoint'],'position':point['position']},
                  target={'header_id':h['id'],'position':join},reducer=cached_reducer['component']['id'],
                  service_entry_selection=guide.get('service_entry_selection'),
                  offgas_join={'point':join,'direction':[0.,0.,-1.]} if point['medium']=='offgas' else None)
    _attach_source_adapter(a,cached_reducer,route)
    _nearby_contacts(a,route['components'][-1:],join)
    axis=normal
    a.__dict__.setdefault('post_route_roots',[]).append({
        'id':'P1_root_'+point['id'],'route':route['id'],'dn':point['dn'],'medium':point['medium'],
        'level':point['level'],'port_end':red_end.tolist(),'top_vertical':_vertical_top_root(a,point),
        'metadata':{'source_requirement':point['id'],'equipment':point['equipment'],
            'group_id':point.get('group_id'),'service_group':h.get('service_group'),
            'at':point['endpoint'],'edge':point['id'],'piping_level':point['level'],
            'equipment_top':point.get('equipment_top',False),'deferred_operator':False},
        'stems':_stem_order(a,point,axis)})
    return {'requirement':point['id'],**guide,'diagnostics':{**guide.get('diagnostics',{}),
            'attempts':attempts.report()}},None


def generate_platform(source,output,*,storage=None,include_material=False):
    require_environment();started=time.perf_counter();output=Path(output).resolve()
    from .external_storage import require_external_output
    if storage is None:storage=require_external_output(output,5.)
    if output.exists() and any(output.iterdir()):raise InputRejected('Choose a new empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    from .generation_progress import atomic_json
    atomic_json(output/'progress.json',{'stage':'piping_inputs'})
    frozen=load_frozen_layout(source);config=load_stage_config()
    original_process=deepcopy(frozen['contracts']['sizing']['graph'])
    if include_material:
        from .equipment_details import prepare_full_room
        from .piping_collision_breaks import clear_stair_deck
        prepare_full_room(frozen,output/'parts')
        clear_stair_deck(frozen['scene'])
    all_req=requirements_from_frozen(frozen,config);req=deepcopy(all_req)
    planning_req=deepcopy(req)
    planning_req['service_points'].extend(planning_req.get('header_demands',[]))
    req['active_services']=[m for m in req['active_services'] if any(p['medium']==m for p in planning_req['service_points'])]
    if not planning_req['service_points']:raise InputRejected('No utility or relief header demands')
    levels=sorted({p['level'] for p in planning_req['service_points']})
    view=deepcopy(frozen);view['scene']['operation_reservations']=[]
    atomic_json(output/'progress.json',{'stage':'header_planning'})
    heights=_heights(frozen['scene'],config['pg_rules'])
    failures=[{**row,'stage':'catalogue_port_assignment'} for row in all_req.get('missing_service_points',())]
    try:
        groups=shared_service_groups_from_frozen(view,planning_req)
        layout=plan_service_headers(view,planning_req,config,heights,groups=groups)
        from .piping_end_bundle import DIRECT_CONNECTION_KINDS, plan_end_bundle
        layout=plan_end_bundle(layout,view,planning_req,config,heights)
        for header in layout['headers']:
            header['method_joint_taps']=_end_device_taps(header,planning_req,config)
    except PipingStationRejected as exc:
        layout={'headers':[],'service_groups':[]}
        failures.append({'stage':'headers','reason':str(exc),'evidence':getattr(exc,'evidence',None)})
    level_headers=layout['headers'];supports=[];support_note=[]
    w,d=frozen['scene']['footprint']
    for level in levels:
        local_headers=[h for h in level_headers if h['level']==level]
        if not local_headers:continue
        axis=local_headers[0]['axis_index'];frame=_frame('y_min' if axis==0 else 'x_min',w,d)
        # Place shared supports around the already registered real tee mouths.
        # A support beneath the header must not consume an arriving branch's
        # normal joint and original clearance. These boxes are planning-only
        # physical pipe-joint volumes, never personal standing reservations.
        gap=config['pg_rules']['parallel_pipe_spacing']['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.
        # Mandatory +Z roots need the same straight-up pipe space whether
        # the optional tee-aware support arrangement succeeds or is retried.
        # These proxies exist only in support planning, never in a.scene.
        required_support_scene=deepcopy(view['scene'])
        bundle=layout.get('end_bundle_station',{})
        for connection in bundle.get('connections',[]):
            if connection['level']!=level or connection['kind'] in DIRECT_CONNECTION_KINDS:continue
            radius=config['pipe_outer_diameter_mm'][str(connection['dn'])]/2000.
            for arm in ('lead_control_points','tail_control_points'):
                path=plan_polyline(connection[arm],radius)
                # Match the unchanged sampled-arc collision screen. Its
                # chord bound includes the arc sagitta; a support placed
                # at the plain radius+gap tangent would still be refused.
                sagitta=coarse_radius(radius)*(1-math.cos(math.radians(SECTOR_DEGREES)/2))
                ext=radius+gap+sagitta+1e-6
                envelopes=[np.r_[np.minimum(run['start'],run['end'])-ext,
                                  np.maximum(run['start'],run['end'])+ext] for run in path['runs']]
                for elbow in path['elbows']:
                    samples=np.asarray(elbow['centerline_points'])
                    envelopes.append(np.r_[samples.min(axis=0)-ext,samples.max(axis=0)+ext])
                for index,box in enumerate(envelopes):
                    required_support_scene['components'].append({
                        'id':connection['id']+'_planned_'+arm+'_'+str(index),
                        'kind':'planned_end_connection','shape':'box','box':box.tolist(),
                        'purpose':'Plan fixed physical straights, bends and tangents with the existing clearances'})
        source_context=SimpleNamespace(frozen=frozen)
        rise_top=min(heights[level]['top_z_m'],max(h['start'][2] for h in local_headers))
        for point in req['service_points']:
            if point['level']!=level or not _vertical_top_root(source_context,point):continue
            rise_source=np.asarray(point['position'],dtype=float)
            radius=config['pipe_outer_diameter_mm'][str(point['dn'])]/2000.
            extent=radius+gap;top=max(float(rise_source[2]),float(rise_top))
            required_support_scene['components'].append({
                'id':'planned_required_source_rise_'+point['id'],
                'kind':'planned_required_source_rise','shape':'box',
                'box':[rise_source[0]-extent,rise_source[1]-extent,rise_source[2]-gap,
                       rise_source[0]+extent,rise_source[1]+extent,top+gap],
                'purpose':'required original +Z pipe and existing clearance; support planning only'})
        support_scene=deepcopy(required_support_scene)
        for header in local_headers:
            joint=header['method_joint_taps']
            mouths=[(pid,tap, next(p['dn'] for p in planning_req['service_points'] if p['id']==pid))
                    for pid,tap in joint['device_taps'].items()]
            main_joint=joint.get('end_connection',joint.get('main_tap'))
            mouths.append(('main',main_joint,main_joint['dn']))
            for pid,tap,dn in mouths:
                radius=config['pipe_outer_diameter_mm'][str(dn)]/2000.
                point=np.asarray(tap['position']);extent=radius+gap
                support_scene['components'].append({'id':header['id']+'_future_joint_'+pid,
                    'kind':'planned_piping_joint','shape':'box','box':np.r_[point-extent,point+extent].tolist(),
                    'purpose':'temporary physical joint obstacle in support planning only'})
        try:
            parts,note=_shared_supports(frame,level,heights[level],local_headers,[],
                support_scene,config['pg_rules'],_protected_boxes(view['scene'],heights[level]['top_z_m']))
            supports.extend(parts);support_note.append({'level':level,'result':note})
        except PipingStationRejected as exc:
            # Retry without optional tee-mouth boxes, retaining every
            # mandatory source rise. A retry must not place a new beam
            # through an unchanged top mouth's only permitted first leg.
            try:
                parts,note=_shared_supports(frame,level,heights[level],local_headers,[],
                    required_support_scene,config['pg_rules'],_protected_boxes(view['scene'],heights[level]['top_z_m']))
                supports.extend(parts)
                support_note.append({'level':level,'result':note,'joint_aware_refusal':str(exc)})
            except PipingStationRejected as original_exc:
                support_note.append({'level':level,'status':'unavailable','reason':str(original_exc),
                                     'joint_aware_refusal':str(exc)})
    zones={lev:{**h,'box':[0.,0.,h['bottom_z_m'],w,d,h['top_z_m']]} for lev,h in heights.items()}
    plan={'station':{},'rack_zones':zones,'headers':level_headers,'service_groups':layout['service_groups'],
          'openings':[],'rack_support_components':supports}
    plan['end_bundle_station']=deepcopy(layout.get('end_bundle_station',{'version':'end_bundle_v1','status':'layout_rejected'}))
    a=Assembly(frozen,req,plan,output/'parts')
    if include_material:
        from .route_requirements import edge_valves, route_requirements, station_requirements
        valves = edge_valves(original_process, service_points=req['service_points'])
        a.path_requirements=route_requirements(original_process, valves,
                                    {p['endpoint']:p['level'] for p in req['service_points']})
        a.station_route_requirements=station_requirements(plan['end_bundle_station'])
        a.frozen['pending_service_points']=req['service_points']
    a.progress_folder=output
    from .generation_progress import checkpoint
    checkpoint(a,{'stage':'service_headers'})
    a.scene['method_stage']='room_level3'
    # Every frozen source adapter is immutable and real, including sources not
    # reached by the later finite route attempts. Earlier tails must see it.
    a.method_reducers={point['id']:_prepare_source_adapter(a,point) for point in req['service_points']}
    headers={};taps={};guides=[]
    for h in sorted(level_headers,key=lambda h:-h['dn']):
        route=a.route(h['id'],[h['start'],h['end']],h['dn'],h['medium'],'floor_header',h['level'],
            service_group=h['service_group'],point_ids=h['point_ids'],axis_index=h['axis_index'],
            **({'end_bundle':deepcopy(h['end_bundle'])} if h.get('end_bundle') else {}))
        direction=_unit(np.asarray(h['end'])-h['start'])
        for suffix,position,outward in (('_start',h['start'],-direction),('_end',h['end'],direction)):
            if h.get('end_bundle') and suffix[1:]==h['end_bundle']['near_side']:continue
            cap=a.blind(h['id']+suffix,position,outward,h['dn'],h['medium'],h['level'],'header_end')
            a.flange_interface(h['id']+suffix,np.asarray(position)-outward*.039,outward,h['dn'],h['medium'],h['level'])
            a.allow(route['components'][0],cap['id'],position)
        headers[h['id']]=route
        try:taps.update({pid:(h,t) for pid,t in _tap_table(a,h).items()})
        except CandidateRefused as exc:
            failures.append({'stage':'header_taps','header':h['id'],'reason':str(exc)})
    station_result=None
    if layout.get('end_bundle_station'):
        from .piping_end_station import add_end_station
        station_result=add_end_station(a,all_req)
    else:
        station_result={'status':'partial','mains':[],'fixed':[],'direct_links':[],
                        'failures':[{'code':'END_BUNDLE_LAYOUT_REJECTED','reason':'No complete termination layout. Keep the original demands.'}]}
    from .piping_lane_connect import plan_lane_connection,LaneUnavailable
    for point in req['service_points']:
        if point['id'] not in taps:
            failures.append({'requirement':point['id'],'endpoint':point['endpoint'],
                'reason':'No usable tap on the same floor. Keep the original equipment and port.',
                'root_position_evidence':_root_position_evidence(a,point,
                    a.method_reducers[point['id']]['end'],inspect_asset(point['dn']))})
    work=sorted((p for p in req['service_points'] if p['id'] in taps),key=lambda p:(-p['dn'],
        math.dist(p['position'],taps[p['id']][1]['position']),*p['position']))
    from .piping_service_connections import connect_groups
    group_guides,group_failures=connect_groups(a,work,taps,headers,heights)
    guides.extend(group_guides);failures.extend(group_failures)
    checkpoint(a,{'stage':'material_templates' if include_material else 'service_drain'})
    material_summary=None
    utility_drain_graph=None
    if not include_material:
        from .utility_drain_scope import logical_scope
        source_logical=deepcopy(frozen['graph'])
        source_logical['valves']=deepcopy(frozen['contracts']['sizing']['graph']['valves'])
        utility_drain_graph=logical_scope(source_logical)
    if include_material or utility_drain_graph is not None:
        from .pipe_spec import load as load_pipe_spec
        from .piping_material_templates import add_material_templates,planned_ground_drain
        width,depth=a.scene['footprint']
        offset=3*a.radius(load_pipe_spec()['services']['steam_condensate']['device_dn'])
        logical=deepcopy(frozen['graph']) if include_material else utility_drain_graph
        if include_material:logical['valves']=deepcopy(frozen['contracts']['sizing']['graph']['valves'])
        material_summary=add_material_templates(a,logical,pipe_spec=load_pipe_spec())
    from gc import collect
    from .generation_progress import step
    collect()
    checkpoint(a,{'stage':'pipe_slot_attachment'})
    from .root_orientation import finish as finish_root_operators
    from .pipe_slots import fit,fit_post_route_roots
    failures.extend(fit(a))
    step(a,{'stage':'post_route_root_placement'})
    if not include_material:
        try:
            failures.extend(fit_post_route_roots(a))
        except Exception as exc:   # a valve problem never discards the finished pipes
            failures.append({'code':'ROOT_VALVE_STAGE_ERROR','reason':f'{type(exc).__name__}: {exc}','pipe_preserved':True})
    step(a,{'stage':'root_operator_orientation'})
    failures.extend(finish_root_operators(a))
    # Attach every required retained package only after physical pipe guides.
    for valve in list(a.planned_valves):a.install_valve(valve)
    piping=a.result()
    if a.scene.get('catalogue_flange_joints'):
        piping['catalogue_flange_joints']=deepcopy(a.scene['catalogue_flange_joints'])
    piping.update(platform_level=None,platform_levels=levels,
        equipment_flange_scope_levels=levels,
        preplaced_source_reducers=[{'requirement':pid,'component':record['component']['id'],'end':record['end']}
                                  for pid,record in a.method_reducers.items()],
        method='pipe_connection_end_bundle_v1',
        method_failures=failures,lane_guides=guides,support_note=support_note,
        scope='Utility service floors from the source layout')
    piping['station_result']=station_result
    piping['end_bundle_station']=deepcopy(a.plan['end_bundle_station'])
    if include_material or utility_drain_graph is not None:
        piping['material']=deepcopy(a.material)
        piping['material_template_summary']=deepcopy(material_summary)
        piping['phase']='P1+P2'
        piping['scope']=('Physical result of coordinate-free recipe, equipment layout, end-bundle utility piping and material valve groups' if include_material else 'Utility piping including condensate to the ground connection')
        if not include_material:piping.update(phase='P1',extension_scope='service_steam_condensate_only')
    if include_material:
        piping['route_requirements']=deepcopy(a.material.get('material_route_requirements', a.path_requirements)+a.station_route_requirements)
        piping['valve_placement'] = 'role_trace'
    write_json(output/'process.json',a.graph);write_json(output/'geometry.json',a.scene);write_json(output/'piping.json',piping)
    from .piping_export import export_piping_model
    model,data=export_piping_model(a.graph,a.scene,output/'room.xml')
    checkpoint(a,{'stage':'actual_model_measurement'})
    from .piping_method_measure import measure_room
    measured=measure_room(frozen,a.graph,a.scene,piping,model,data)
    if include_material:
        from .piping_collision_breaks import omit_intersections
        original_collisions=[e for e in measured['geometry']['errors'] if e['code']=='P1_SOLID_COLLISION']
        removed=omit_intersections(a.graph,a.scene,piping,original_collisions)
        if removed:
            (output/'room.xml').rename(output/'before_omitting_intersections.xml')
            piping['collision_breaks']=removed
            write_json(output/'collision_breaks.json',{'intersections':original_collisions,'breaks':removed})
            for name,value in [('process',a.graph),('geometry',a.scene),('piping',piping)]:
                write_json(output/(name+'.json'),value)
            model,data=export_piping_model(a.graph,a.scene,output/'room.xml')
            measured=measure_room(frozen,a.graph,a.scene,piping,model,data)
    if include_material or utility_drain_graph is not None:
        from .piping_material_templates import measure_material_templates
        measured['material']=measure_material_templates(frozen,a.graph,a.scene,piping,model,data,p1_report=measured)
        if not include_material:measured['material']['scope']='service_steam_condensate_only'
    measured.update(source_dir=str(Path(source).resolve()),method_failures=failures,
                    elapsed_seconds=time.perf_counter()-started,scope=piping['scope'])
    from .piping_requirements import required_header_keys
    actual_headers={(r['medium'],int(r['level'])) for r in piping['routes']
                    if r['role']=='floor_header' and r.get('components')}
    coverage=[{'medium':medium,'level':level,'present':(medium,level) in actual_headers}
              for medium,level in sorted(required_header_keys(all_req))]
    measured['header_coverage']=coverage
    missing=[{'code':'REQUIRED_FLOOR_HEADER_MISSING',**row} for row in coverage if not row['present']]
    measured.setdefault('errors',[]).extend(missing)
    if missing:measured['status']='fail'
    if include_material:
        def semantics(graph):
            return {'edges':[(e['id'],e['a'],e['b'],e['role'],e['dn']) for e in graph['edges']],
                    'valves':[(v['id'],v['edge'],v['at'],v['role'],v.get('valve_role')) for v in graph['valves']]}
        measured['process_unchanged_during_piping']=(semantics(original_process)==semantics(frozen['contracts']['sizing']['graph']))
        if not measured['process_unchanged_during_piping']:
            measured['errors'].append({'code':'PROCESS_CHANGED_DURING_PIPING'})
            measured['status']='fail'
        from .valve_roles import check as check_roles
        measured['valve_roles']=check_roles(original_process,a.graph,a.scene,piping,model,data,
                                            requirements=all_req)
        write_json(output/'valve_roles.json',measured['valve_roles'])
        if measured['valve_roles']['errors']:measured['status']='fail'
        from .rule_checks import check
        measured['confirmed_rules']=check(frozen,a.scene,piping,model,data)
        write_json(output/'confirmed_rules.json',measured['confirmed_rules'])
    write_json(output/'measure.json',measured)
    assert_frozen_inputs(frozen['frozen_inputs'])
    return measured
