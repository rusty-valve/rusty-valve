"""Build common piping arrangements for equipment and process services."""
from __future__ import annotations
from .root_orientation import planning_geoms

from copy import copy,deepcopy
import math

import numpy as np

from .layout import add_box, add_pipe
from .pipe_spec import load as load_pipe_spec
from .piping_assets import inspect_asset
from .piping_build import Assembly, PipingRefused, _overlap, _unit
from .piping_material_build import MaterialAssembly
from .piping_material_requirements import requirements_for_end_bundle
from .piping_requirements import _contracts


EPS = 1.e-8


def planned_ground_drain(a,dn,near=None,direction=None):
    """Choose an unoccupied ground connection from actual frame faces."""
    r=a.radius(dn)
    start=np.asarray(near,float)
    w,d=a.scene['footprint']
    axis=int(np.argmax(np.abs(direction[:2])))
    sign=1 if direction[axis]>0 else -1
    point=start.copy()
    boundary=(w,d)[axis] if sign>0 else 0.
    point[axis]=sign*max(sign*boundary+8*r+.1,sign*start[axis]+12*r+.2)
    point[2]=a.floors[0]
    return {'id':'ground_drain_point','position':point.tolist(),'normal':[0.,0.,1.],
            'source':'first drain-channel connection outside the shell in the trap outlet direction'}


class _LiveMaterial(MaterialAssembly):
    """Borrow existing material primitives without loading a saved P1 batch."""

    def __init__(self, base, logical, specification):
        self.__dict__.update(base.__dict__)
        self.frozen = dict(base.frozen)
        self.frozen['graph'] = deepcopy(logical)
        self._p1_pending = list(getattr(base, 'planned_valves', []))
        self.config = deepcopy(base.config)
        self.config['pipe_outer_diameter_mm'] = deepcopy(specification['pipe_outer_diameter_mm'])
        self.od = self.config['pipe_outer_diameter_mm']
        for key, color in specification['colors'].items():
            self.config['pg_rules']['appearance']['color_map'][key] = {'rgb': list(color)}
        self.specification = specification
        self.p2_route_start = len(self.routes)
        self.p2_valve_start = len(self.graph['valves'])
        self._serial = 0
        self.material = {'requirements': {}, 'edge_routes': [], 'junctions': [], 'terminals': [],
            'interfaces': [], 'new_openings': [], 'automatic_components': [], 'deferred': [],
            'scope': 'Generic process topology; one physical attempt; no fluid/thermal/robot proof'}
        self.contexts, self.anchors, self.joints = {}, {}, {}
        self._pending = []; self._active_valve = None

    def obstacles(self, skip=(), extra=()):
        # Valves not yet installed are not routing obstacles; all valves are
        # fitted after every pipe exists (pipe stage: valves never steer pipes).
        return super().obstacles(skip, extra)


def _grip_points(asset):
    geometry = asset['params']['geometry']
    if asset['operator'] == 'lever':
        length = (geometry['handle_grip_start_mm']+geometry['handle_grip_end_mm'])/2000.
        height = geometry['handle_center_z_mm']/1000.
        return [np.array([length, 0., height]), np.array([0., -length, height])]
    motion = asset['params']['motion']
    center = np.array(motion['wheel_center_m'], dtype=float)
    axis = _unit(motion['wheel_axis'])
    helper = np.eye(3)[np.argmin(abs(axis))]
    first = _unit(np.cross(axis, helper)); second = np.cross(axis, first)
    radius = (geometry['wheel_outer_diameter_mm']-geometry['wheel_rim_diameter_mm'])/2000.
    return [center+radius*(math.cos(t)*first+math.sin(t)*second)
            for t in np.linspace(0., 2*math.pi, 16, endpoint=False)]


def _port_row(endpoint, uid, point, normal, dn, component, level, a):
    r = a.radius(dn)
    return {'equipment': uid, 'port': endpoint.split('.', 1)[1], 'endpoint': endpoint,
        'position': list(map(float, point)), 'normal': list(map(float, normal)),
        'source_dn': int(dn), 'line_dn': int(dn), 'actual_radius_m': r,
        'connection_radius_m': r, 'nozzle_component': component, 'level': int(level),
        'physical_interface': {'kind': 'nozzle_face', 'component_ids': [component],
            'position': list(map(float, point)), 'normal': list(map(float, normal)), 'radius_m': r}}


def _catalogue_automatic(a,uid,owner,kind,target,direction,level,anchor=None):
    from .equipment_details import place_asset, read_catalogue

    spec=read_catalogue(a.frozen['equipment_catalogue_dir'])[kind]
    if anchor=='in':
        anchor=spec['ports']['in']['position']
    yaw=math.degrees(math.atan2(direction[1],direction[0]))
    device=place_asset(a.scene,a.folder,uid,spec,target,yaw,anchor=anchor)
    device.update(kind=kind,catalogue_key=kind,level=level)
    a.scene['catalogue_devices'].append(device)
    extra={}; ports={}
    for name, face in device['ports'].items():
        endpoint=uid+'.'+name
        face={**face,'endpoint':endpoint}
        row=_port_row(endpoint,uid,face['position'],face['normal'],face['dn'],
                      face['contact_ids'][0],level,a)
        row['source_catalogue_face']=face
        row['actual_radius_m']=face['opening_diameter_m']/2
        row['physical_interface'].update(kind='catalogue_flange_face',
            component_ids=face['contact_ids'],radius_m=row['actual_radius_m'])
        extra[endpoint]=row
        ports[name]={'pos':row['position'],'normal':row['normal'],'dn':face['dn']}
    record={'id':uid,'owner':owner,'kind':kind,'automatic':True,
            'components':device['components']+device['envelopes'],'ports':ports,
            'body_bbox_m':device['install_bounds_m'],'internal_connection':['in','out'],
            'scope':'catalogue original and measured external mating face'}
    return record,extra


def _safety_component(a, graph, uid, unit, equipment):
    """Coarse automatic safety valve; independent body, never a hand valve."""
    owner, name = unit['source_endpoint'].split('.')
    physical = equipment[owner]['ports'][name]
    p = np.asarray(physical['pos'], float)
    dn = int(a.specification['services']['offgas']['emergency_dn']); r=a.radius(dn)
    level=max(i for i,z in enumerate(a.floors) if z<=p[2])
    headers=[h for h in a.plan['headers'] if h['medium']=='offgas' and h['level']==level]
    if len(headers)!=1:
        raise ValueError(f'{uid} requires exactly one same-floor offgas header at level {level}; found {len(headers)}')
    h=headers[0]
    from .piping_route_search import RouteUnavailable
    beams = [part for part in a.scene['components']
             if part['kind'] == 'beam' and not part.get('annotation_only') and not part.get('visual_only')]

    def beam_hits(boxes):
        return sorted({part['id'] for part in beams for box in boxes if _overlap(part['box'], box)})

    if a.frozen.get('equipment_catalogue_dir') and physical.get('catalogue_face'):
        cross=1-h['axis_index']
        direction=np.eye(3)[cross]*(1. if h['start'][cross]>p[cross] else -1.)
        transverse=np.eye(3)[h['axis_index']]
        component_count=len(a.scene['components'])
        hits=set()
        for candidate in (direction,-direction,transverse,-transverse):
            record,extra=_catalogue_automatic(a,uid,owner,'safety_valve',p+[0.,0.,.245],
                                              candidate,level,anchor='in')
            blocked=beam_hits([part['box'] for part in a.scene['components'][component_count:]
                               if not part.get('annotation_only') and not part.get('visual_only')])
            if not blocked:
                break
            hits.update(blocked)
            del a.scene['components'][component_count:]
            a.scene['catalogue_devices'].pop()
        else:
            raise PipingRefused(uid+' Safety relief valve intersects a beam: '+', '.join(sorted(hits)))
    else:
        # Envelope is explicitly an estimate derived from the connection OD.
        half=3*r; height=8*r; neck=4*r
        center=p+np.array([0.,0.,neck+height/2+2*r])
        box=np.r_[center-[half,half,height/2],center+[half,half,height/2]].tolist()
        blocked=beam_hits([box])
        if blocked:
            raise PipingRefused(uid+' Safety relief valve intersects a beam: '+', '.join(blocked))
        body=add_box(a.scene,uid+'_automatic_body',box,'safety_valve',[.68,.72,.73],uid)
        body.update(piping=True,automatic=True,count_as_hand_valve=False,
                    coarse_estimate='6OD/2 wide and 8OD/2 tall; geometric estimate, not a catalogue or relief-capacity model')
        direction=np.array([0.,0.,0.]);cross=1-h['axis_index']
        direction[cross]=1. if h['start'][cross]>center[cross] else -1.
        # Rotate the coarse automatic outlet around its vertical inlet when its
        # first horizontal escape would enter the owner's actual drive envelope.
        # Device mouth, inlet and reserved maintenance space stay unchanged.
        directions=[direction,-direction,np.eye(3)[h['axis_index']],-np.eye(3)[h['axis_index']]]
        from .piping_build import _root_pipe_obstacle
        future_roots=[]
        for valve in graph.get('valves',[]):
            endpoint=valve.get('at','')
            if '.' not in endpoint:continue
            owner_id,port_name=endpoint.split('.',1)
            port=equipment.get(owner_id,{}).get('ports',{}).get(port_name)
            if not port or not np.allclose(port.get('normal'),[0.,0.,1.],atol=EPS):continue
            root_dn=int(valve['dn']);asset=inspect_asset(root_dn,operator=valve.get('operator'))
            root_r=a.radius(root_dn);root_p=np.asarray(port['pos'])
            future_roots.append(_root_pipe_obstacle('future_root_'+endpoint,root_p,
                root_p+[0,0,5*asset['install_length_m']+4*root_r+.05],root_r))
        for candidate in directions:
            face=center+candidate*half
            end=face+candidate*(neck+6*r)
            try:
                a.search(face,end,dn,skip=(body['id'],),clearance=.05,straight=True,extra=future_roots)
                zone=a.plan['rack_zones'][level]
                top=end.copy();top[2]=max(end[2],zone['bottom_z_m']+4*r)
                if top[2]>end[2]+EPS:
                    a.search(end,top,dn,skip=(body['id'],),clearance=.05,straight=True,extra=future_roots)
            except RouteUnavailable:
                continue
            direction=candidate
            break
        faces={'in':(center-np.array([0.,0.,height/2]),np.array([0.,0.,-1.])),
               'out':(center+direction*half,direction)}
        record={'id':uid,'owner':owner,'kind':'safety_valve','automatic':True,
                'body_bbox_m':box,'components':[body['id']],'ports':{},'internal_connection':['in','out'],
                'scope':'coarse automatic body; no hand isolation and no pressure-relief performance claim'}
        extra={}
        for name,(face,normal) in faces.items():
            point=face+normal*neck
            route=Assembly.route(a,'AUTO_'+uid+'_'+name,[face,point],dn,'offgas','automatic_nozzle',level,phase='automatic')
            cid=route['components'][0];a.allow(body['id'],cid,face)
            record['components'].append(cid)
            extra[uid+'.'+name]=_port_row(uid+'.'+name,uid,point,normal,dn,cid,level,a)
            record['ports'][name]={'pos':point.tolist(),'normal':normal.tolist(),'dn':dn}
    a.scene.setdefault('automatic_components',[]).append(record)
    a.material['automatic_components'].append(deepcopy(record))
    a.graph['units'].setdefault(uid,deepcopy(unit))['ports']=deepcopy(record['ports'])
    # Separate physical header tap for this distinct safety discharge.
    edge=next(e for e in graph['edges'] if e['a']==uid+'.out')
    target=np.asarray(h['start'],float).copy();axis=h['axis_index']
    planned=h.get('method_joint_taps',{}).get('device_taps',{}).get(edge['id'],{}).get('position')
    margin=6*r+.05
    preferred=p[axis] if planned is None else planned[axis]
    target[axis]=np.clip(preferred,min(h['start'][axis],h['end'][axis])+margin,
                         max(h['start'][axis],h['end'][axis])-margin)
    header_route=next(row for row in a.routes if row['id']==h['id'])
    # Choose a real unoccupied tee position on the existing header. Its ends
    # and supports remain fixed; a clamped source projection can land beside
    # a rack seat even when the header itself is clear.
    lo=min(h['start'][axis],h['end'][axis])+margin
    hi=max(h['start'][axis],h['end'][axis])-margin
    candidates=[target[axis],*np.linspace(lo,hi,17)]
    for obstacle in a.obstacles():
        for face in (obstacle['box'][axis],obstacle['box'][axis+3]):
            candidates.extend((face-r-.05,face+r+.05))
    taps=[]
    from .piping_lane_connect import _clear_plan,_obstacles
    from .piping_elbows import plan_polyline
    obstacles,_=_obstacles(a.obstacles())
    for coordinate in sorted(set(float(c) for c in candidates if lo<=c<=hi),key=lambda c:abs(c-preferred)):
        if any(abs(coordinate-old[axis])<6*r+2*r+.05 for old in taps):continue
        probe=target.copy();probe[axis]=coordinate
        top=probe+np.array([0.,0.,3*r+a.radius(h['dn'])+2*r+.05])
        if _clear_plan(plan_polyline([top,probe],r),obstacles,r,.05,np.asarray(a.bounds),
            {'position':top,'contact_ids':[]},{'position':probe,'contact_ids':header_route['components']},
            np.array([0.,0.,-1.]),np.array([0.,0.,-1.]),is_tail=True):continue
        taps.append(probe.tolist())
        if len(taps)>=8:break
    if taps:target=np.asarray(taps[0])
    ep=edge['b']
    extra[ep]=_port_row(ep,ep.split('.')[0],target,[0,0,1],dn,header_route['components'][0],level,a)
    extra[ep].update(header_id=h['id'],header_radius=a.radius(h['dn']),network_boundary=True,
                     tap_candidates=taps or [target.tolist()])
    a.material.setdefault('safety_header_selection',[]).append({'automatic':uid,
        'source_endpoint':unit['source_endpoint'],'source_position_m':p.tolist(),
        'source_level':level,'header_id':h['id'],'header_level':h['level'],
        'header_start_m':deepcopy(h['start']),'header_end_m':deepcopy(h['end']),
        'planned_tap_m':deepcopy(planned),'selected_tap_m':target.tolist(),
        'discharge_edge':edge['id'],'selection_rule':'same_floor_only'})
    return extra


def _flanged_safety_inlet(a, edge, one, two):
    """Fill only the measured gap between the tank face and the fixed valve inlet."""
    from .piping_equipment_flange import emit_flanged_stub
    from .piping_material_build import _meta

    p, q = map(np.asarray, (one['position'], two['position']))
    axis = np.asarray(one['normal'])
    gap = float((q-p) @ axis)
    if not np.allclose(two['normal'], -axis, atol=EPS, rtol=0) or np.linalg.norm(q-p-gap*axis) > EPS:
        raise ValueError(edge['id']+' Equipment port and fixed safety relief inlet are not coaxial or facing each other')
    dn = edge['dn']
    source_dn = one['source_dn']
    dims = one['source_flange']
    reach = (dims['rf_height_mm']+dims['plate_thickness_mm'])/1000
    reducer_length = 2*max(a.radius(source_dn),a.radius(dn)) if source_dn != dn else 0.
    if gap < reach+reducer_length-EPS:
        raise ValueError(f"{edge['id']} fixed separation {gap:.6f} m, occupied by mating flange and reducer: {reach+reducer_length:.6f}m")

    medium = 'emergency_discharge'
    name = 'P2_'+edge['id']+'_safety'
    joint = emit_flanged_stub(a, {**one,'medium':medium,'source_edge':edge['id']}, name, phase='P2')
    face = np.asarray(joint['end'])
    end = face
    flow = [joint['pipe_stub']]
    points = [p.tolist(),face.tolist()]
    if reducer_length:
        end, reducer = a.reducer(name+'_reducer',face,axis,source_dn,dn,medium,one['source_radius_m'])
        reducer.update(phase='P2',source_edge=edge['id'],equipment_endpoint=one['endpoint'])
        a.allow(flow[-1],reducer['id'],face,normal=axis)
        flow.append(reducer['id'])
        points.append(end.tolist())
    adapter = flow[-1]
    departure = {'position':end.tolist(),'normal':one['normal'],
                 'contact_ids':[*joint['source_components'],*joint['components'],*flow]}
    inlet = {'position':two['position'],'normal':two['normal'],'contact_ids':[two['nozzle_component']]}
    remainder = float((q-end) @ axis)
    if remainder < -EPS:
        raise ValueError(edge['id']+' An installed fitting extends beyond the fixed safety relief inlet')
    if remainder > EPS:
        a.search(end,q,dn,clearance=0.,straight=True,port_contacts=(departure,inlet))
        pipe = a.pipe(name,end,q,dn,medium,name)
        pipe.update(level=one['level'],phase='P2',source_edge=edge['id'])
        a.allow(flow[-1],pipe['id'],end,normal=axis)
        flow.append(pipe['id'])
        points.append(q.tolist())
    a.allow(flow[-1],two['nozzle_component'],q,normal=two['normal'])
    route = {'id':name,'phase':'P2','dn':dn,'medium':medium,'role':'automatic_safety_connection',
             'level':one['level'],'source':{'endpoint':edge['a'],'position':one['position']},
             'target':{'endpoint':edge['b'],'position':two['position']},'points':points,
             'control_points':[p.tolist(),q.tolist()],'components':flow,
             'pipe_components':flow[-1:] if remainder > EPS else [],'elbows':[],'valves':[],
             'bend_count':0,'actual_planned_centerline_length_m':gap,**_meta(edge)}
    from .grating import cut_for_parts
    flow_ids = set(flow)
    cut_for_parts(a,route,[c for c in a.scene['components'] if c['id'] in flow_ids])
    a.routes.append(route)
    a.material['interfaces'].append({'endpoint':one['endpoint'],'source_edge':edge['id'],
        'original_logical_position':one['position'],'physical_position':one['position'],
        'normal':one['normal'],'actual_radius_m':one['actual_radius_m'],'connection_radius_m':one['source_radius_m'],
        'physical_interface':deepcopy(one['physical_interface']),'adapter':adapter,'equipment_flange_joint':joint,
        'departure_contact':departure,'fixed_inlet_position':two['position'],'fixed_gap_m':gap,'remaining_pipe_m':max(0.,remainder)})
    return route


def _catalogue_safety_route(a,edge,one,two):
    from .catalogue_flange import emit_catalogue_adapter
    from .piping_material_build import _meta

    dn=edge['dn']
    start, _ = emit_catalogue_adapter(a,one,'P2_'+edge['id']+'_safety_source','emergency_discharge',dn)
    joints=[start]
    if two.get('source_catalogue_face'):
        finish, _ = emit_catalogue_adapter(a,two,'P2_'+edge['id']+'_safety_inlet','emergency_discharge',dn)
        joints.append(finish)
    contacts=[{'position':joint['end'],'normal':port['normal'],
               'contact_ids':joint['departure_contact_ids']} for joint,port in zip(joints,(one,two))]
    tail=joints[1]['end'] if len(joints)==2 else two['position'];routing_evidence=None
    if two.get('header_id'):
        from .piping_lane_connect import _clear_plan,_obstacles
        from .piping_elbows import plan_polyline
        from .piping_route_search import RouteUnavailable
        r=a.radius(dn);normal=np.asarray(two['normal']);obstacles,_=_obstacles(a.obstacles())
        search=copy(a);search.bounds=list(a.bounds);level=one['level']
        search.bounds[2]=max(search.bounds[2],a.floors[level])
        if level+1<len(a.floors):search.bounds[5]=min(search.bounds[5],a.floors[level+1])
        target_obstacles=[o for o in obstacles if o['id']==two['nozzle_component']];rejected=[]
        for tap in two.get('tap_candidates',[tail]):
            try:
                top=np.asarray(tap)+normal*(5*r+two['header_radius']+.05)
                points=search.search(start['end'],top,dn,port_contacts=contacts,
                    start_direction=one['normal'],end_direction=-normal,complete_bend_planes=True)
                points=[*points,list(tap)];target={'position':tap,'contact_ids':[two['nozzle_component']]}
                hit=_clear_plan(plan_polyline([top,tap],r),obstacles,r,.05,np.asarray(search.bounds),
                    {'position':top},target,-normal,-normal,is_tail=True)
                hit=hit or _clear_plan(plan_polyline(points,r),target_obstacles,r,.05,np.asarray(search.bounds),
                    {'position':start['end']},target,np.asarray(one['normal']),-normal,is_tail=True)
                if hit:raise RouteUnavailable('SAFETY_ROUTE_OBSTACLE','Complete relief route candidate intersects a solid',obstacle=hit)
            except RouteUnavailable as error:
                last_error=error;rejected.append({'tap_m':list(tap),'reason':str(error)});continue
            tail=list(tap);break
        else:raise last_error
        routing_evidence={'same_floor_bounds_m':search.bounds,'terminal_start_m':top.tolist(),
            'target_header':two['header_id'],'target_contact_component':two['nozzle_component'],
            'prefix_all_solids_checked':True,'terminal_all_solids_checked':True,
            'whole_target_header_checked':True,'rejected_candidates':rejected}
        two['position']=two['physical_interface']['position']=tail
        a.material['requirements']['ports'][edge['b']]=deepcopy(two)
        for selected in a.material['safety_header_selection']:
            if selected['discharge_edge']==edge['id']:selected['selected_tap_m']=tail
    else:
        points=a.search(start['end'],tail,dn,port_contacts=contacts,
            start_direction=one['normal'],end_direction=-np.asarray(two['normal']),complete_bend_planes=True)
    route=a.route('P2_'+edge['id']+'_safety',points,dn,'emergency_discharge',
                  'automatic_safety_connection',one['level'],distribution_id=two.get('header_id'),
                  source={'endpoint':edge['a'],'position':one['position']},
                  target={'endpoint':edge['b'],'position':two['position']},**_meta(edge))
    if routing_evidence is not None:route['safety_routing_evidence']=routing_evidence
    a.allow(start['pipe_stub'],route['components'][0],start['end'])
    route['components'][:0]=[start['through_bore'],start['pipe_stub']]
    route['points'][:0]=[one['position'],start['transition']['ends'][1]]
    if len(joints)==2:
        a.allow(joints[1]['pipe_stub'],route['components'][-1],tail)
        route['components'].extend([joints[1]['pipe_stub'],joints[1]['through_bore']])
        route['points'].extend([joints[1]['transition']['ends'][1],two['position']])
    else:
        a.allow(two['nozzle_component'],route['components'][-1],tail)
    return route


def _safety_connections(a):
    from .piping_lane_connect import plan_lane_connection
    from .piping_material_build import _meta
    for edge in a.requirements['edges']:
        if edge['role'] not in ('emergency_discharge','safety_discharge'):continue
        saved = None
        try:
            one=a.requirements['ports'][edge['a']];two=a.requirements['ports'][edge['b']]
            dn=edge['dn'];r=a.radius(dn);level=one['level']
            if one.get('source_catalogue_face') and (two.get('source_catalogue_face') or two.get('header_id')):
                saved=a.snapshot()
                route=_catalogue_safety_route(a,edge,one,two)
                edge['execution']='pre_emitted'
                a.material['edge_routes'].append({'source_edge':edge['id'],'a':edge['a'],'b':edge['b'],
                    'routes':[route['id']],'components':route['components'],'valves':[],'dn':dn,'phase':'P2'})
                continue
            if edge['b'].endswith('.in') and one.get('source_flange'):
                saved = a.snapshot()
                route = _flanged_safety_inlet(a,edge,one,two)
                edge['execution']='pre_emitted'
                a.material['edge_routes'].append({'source_edge':edge['id'],'a':edge['a'],'b':edge['b'],
                    'routes':[route['id']],'components':route['components'],'valves':[],'dn':dn,'phase':'P2'})
                continue
            if edge['b'].endswith('.in'):
                # The upward owner mouth and automatic inlet are coaxial.
                points=[one['position'],two['position']]
                if math.dist(*points)<=EPS:
                    # Co-located connection has no independent pipe length; attach
                    # the actual inlet nozzle route as its physical edge evidence.
                    record=next(row for row in a.routes if row['id']=='AUTO_'+two['equipment']+'_in')
                    edge['execution']='pre_emitted'
                    # Use owner-to-inlet lead of one real OD for explicit evidence.
                    raise ValueError(edge['id']+' automatic inlet lead has zero length')
                contacts = [{'position':port['position'],'normal':port['normal'],'contact_ids':[port['nozzle_component']]}
                            for port in (one,two)]
                a.search(*points,dn,clearance=0.,straight=True,port_contacts=contacts)
            else:
                header=next(h for h in a.plan['headers'] if h['id']==two['header_id'])
                zone=a.plan['rack_zones'][level]
                from .piping_lane_connect import LaneUnavailable
                last_error=None;selected=None
                uid=one['equipment'];record=next(row for row in a.material['automatic_components'] if row['id']==uid)
                center=(np.asarray(record['body_bbox_m'][:3])+record['body_bbox_m'][3:])/2
                old=np.asarray(one['normal']);directions=[old,*[np.asarray(v,float) for v in ([1,0,0],[-1,0,0],[0,1,0],[0,-1,0]) if not np.allclose(v,old)]]
                for direction in directions:
                    face=center+direction*(3*r);position=face+direction*(4*r)
                    try:
                        a.search(face,position,dn,skip=(record['components'][0],one['nozzle_component']),straight=True)
                    except Exception as error:
                        last_error=error;continue
                    for tap in two.get('tap_candidates',[two['position']]):
                        try:
                            guide=plan_lane_connection(
                                {'position':position.tolist(),'normal':direction.tolist(),'contact_ids':[],
                                 'method_v1_3':True,'medium':'offgas','minimum_straight_m':2*r},
                                {'position':tap,'normal':[0,0,1],'radius':two['header_radius'],
                                 'contact_ids':[two['nozzle_component']]},a.obstacles(skip=(one['nozzle_component'],)),r,a.bounds,
                                a.floors[level],zone['top_z_m'],header['axis_index'],asset=None,
                                target_direction=[0,0,-1],rack_bottom_z=zone['bottom_z_m'])
                        except LaneUnavailable as error:
                            last_error=error;continue
                        selected=(direction,face,position,tap);break
                    if selected:break
                if selected is None:raise last_error
                direction,face,position,tap=selected
                if not np.allclose(direction,old):
                    from .piping_material_network import _remove_routes
                    old_cid=one['nozzle_component']
                    old_route=next(row for row in a.routes if row['id']=='AUTO_'+uid+'_out')
                    _remove_routes(a,[old_route])
                    nozzle=Assembly.route(a,'AUTO_'+uid+'_out',[face,position],dn,'offgas','automatic_nozzle',level,phase='automatic')
                    cid=nozzle['components'][0];a.allow(record['components'][0],cid,face)
                    one.update(_port_row(edge['a'],uid,position,direction,dn,cid,level,a))
                    for row in [record,*[x for x in a.scene['automatic_components'] if x['id']==uid]]:
                        row['components']=[cid if c==old_cid else c for c in row['components']]
                        row['ports']['out']={'pos':position.tolist(),'normal':direction.tolist(),'dn':dn}
                    a.graph['units'][uid]['ports']['out']=deepcopy(record['ports']['out'])
                    a.material['requirements']['ports'][edge['a']]=deepcopy(one)
                two['position']=tap;two['physical_interface']['position']=tap
                a.material['requirements']['ports'][edge['b']]=deepcopy(two)
                points=guide['tail']
            route=a.route('P2_'+edge['id']+'_safety',points,dn,'emergency_discharge',
                          'automatic_safety_connection',level,distribution_id=two.get('header_id'),
                          source={'endpoint':edge['a'],'position':one['position']},
                          target={'endpoint':edge['b'],'position':two['position']},**_meta(edge))
            for port in (one,two):
                for cid in route['components']:
                    a.allow(port['nozzle_component'],cid,port['position'])
            edge['execution']='pre_emitted'
            a.material['edge_routes'].append({'source_edge':edge['id'],'a':edge['a'],'b':edge['b'],
                'routes':[route['id']],'components':route['components'],'valves':[],'dn':dn,'phase':'P2'})
        except Exception as error:
            if saved is not None:
                a.restore(saved)
            edge['execution']='emission_failed'
            a.material.setdefault('safety_failures',[]).append({'code':'SAFETY_CONNECTION_FAILED',
                'edge':edge['id'],'reason':str(error),'evidence':getattr(error,'evidence',None)})


def _inline_components(a, graph, base, extra, uid, unit):
    """Place automatic traps from owner mouths; no drainage-site solids."""
    _, _, equipment = _contracts(a.frozen)
    inline = {row['id']: row for row in graph.get('inline_components', [])}
    if unit.get('kind') == 'safety_valve':
        extra.update(_safety_component(a, graph, uid, unit, equipment))
        return extra
    if unit.get('kind') != 'automatic_steam_trap':
        return extra
    definition = {**unit, **inline.get(uid, {})}
    source = definition.get('source_endpoint')
    if not source:
        incoming = [e for e in graph['edges'] if e['b'].split('.')[0] == uid]
        source = incoming[0]['a'] if len(incoming) == 1 else None
    if not source or source.split('.')[0] not in equipment:
        raise ValueError('Automatic trap has no physical owner outlet: '+uid)
    owner, name = source.split('.', 1); physical = equipment[owner]['ports'][name]
    dn = int(a.specification['services']['steam_condensate']['device_dn'])
    from .steam_trap_layout import trap_layout
    p=np.asarray(physical['pos'],float)
    rect=equipment[owner].get('rect',[p[0]-.5,p[1]-.5,p[0]+.5,p[1]+.5])
    placement=trap_layout(physical,rect,a.floors,dn)
    center,direction=placement['center'],placement['direction']
    length,width,height=(placement[k] for k in ('length','width','height'))
    level,r,bbox=placement['level'],placement['radius'],placement['body_box']
    if a.frozen.get('equipment_catalogue_dir'):
        from .equipment_details import read_catalogue

        spec=read_catalogue(a.frozen['equipment_catalogue_dir'])['automatic_steam_trap']
        sizes=np.asarray(spec['bounds_m']['max'])-spec['bounds_m']['min']
        margins=np.array([sizes[0],sizes[1]])/2+.08
        if abs(direction[1])>.5:
            margins=margins[::-1]
        xy=np.clip(center[:2],margins,np.asarray(a.scene['footprint'])-margins)
        record,actual=_catalogue_automatic(a,uid,owner,'automatic_steam_trap',
                                           [*xy,a.floors[level]],direction,level,anchor=[0.,0.,0.])
        extra.update(actual)
    else:
        body = add_box(a.scene, uid+'_automatic_body', bbox, 'automatic_steam_trap',
                       [.42, .45, .46], uid)
        body.update(piping=True, automatic=True, count_as_hand_valve=False,
                    coarse_estimate='automatic trap envelope; no catalogue model/thermal control simulation')
        record = {'id': uid, 'owner': owner, 'kind': 'automatic_steam_trap', 'automatic': True,
            'components': [body['id']], 'body_bbox_m': bbox, 'ports': {}, 'internal_connection': ['in', 'out'],
            'scope': 'coarse physical envelope and two real external nozzle faces; not fluid/control validation'}
        for pname, sign in (('in', -1), ('out', 1)):
            normal = direction*sign; body_face = center+normal*length/2
            point = body_face+normal*max(2*r, .12)
            route = Assembly.route(a, 'AUTO_'+uid+'_'+pname, [body_face, point], dn,
                'steam_condensate', 'automatic_nozzle', level, phase='automatic')
            nozzle = next(c for c in a.scene['components'] if c['id']==route['components'][0])
            nozzle.update(automatic=True)
            a.allow(body['id'], nozzle['id'], body_face)
            record['components'].append(nozzle['id'])
            endpoint = uid+'.'+pname
            extra[endpoint] = _port_row(endpoint, uid, point, normal, dn, nozzle['id'], level, a)
            record['ports'][pname] = {'pos': point.tolist(), 'normal': normal.tolist(), 'dn': dn}
    a.scene.setdefault('automatic_components', []).append(record)
    a.material['automatic_components'].append(deepcopy(record))
    a.graph['units'].setdefault(uid, deepcopy(unit))['ports'] = deepcopy(record['ports'])
    # This is an explicit network endpoint, not a fictitious drain model.
    downstream = [e for e in graph['edges'] if e['a'].split('.')[0] == uid
                  and graph['units'][e['b'].split('.')[0]].get('kind') == 'boundary']
    for edge in downstream:
        endpoint = edge['b']; boundary_uid = endpoint.split('.')[0]
        supplied = getattr(base, 'material_drain_endpoint', None)
        supplied = supplied.get(endpoint, supplied) if isinstance(supplied, dict) else supplied
        if isinstance(supplied, dict):
            point = np.asarray(supplied.get('position', supplied.get('pos')), float)
            normal = _unit(supplied.get('normal', -direction))
        elif supplied is not None:
            point = np.asarray(supplied, float); normal = -direction
        else:
            planned=planned_ground_drain(a,dn,record['ports']['out']['pos'],direction)
            point=np.asarray(planned['position']);normal=np.asarray(planned['normal'])
        extra[endpoint] = {'equipment': boundary_uid, 'port': endpoint.split('.', 1)[1],
            'endpoint': endpoint, 'position': point.tolist(), 'normal': normal.tolist(),
            'source_dn': dn, 'line_dn': dn, 'actual_radius_m': r, 'connection_radius_m': r,
            'nozzle_component': None, 'level': 0, 'network_boundary': True}
        a.anchors[endpoint] = {'point': point, 'normal': normal, 'level': 0,
                              'physical_port': deepcopy(extra[endpoint])}
        a.material.setdefault('drain_endpoints', []).append({'endpoint': endpoint,
            'position': point.tolist(), 'normal': normal.tolist(), 'dn': dn,
            'purpose': 'ground-floor predefined hot condensate drainage connection',
            'geometry': None, 'scope': 'predetermined network endpoint; no ditch/box/grating emitted'})
    return extra


def _junction_position(a, uid, unit, incidents, shared):
    if unit.get('template_role', unit.get('service')) != 'liquid_seal':
        graph=a.frozen['graph']
        outlets=[edge for edge in incidents if edge['a'].split('.')[0]==uid and
                 graph['units'][edge['b'].split('.')[0]]['kind']=='vessel' and edge['b'] in a.anchors]
        if len(outlets)==1:
            anchor=a.anchors[outlets[0]['b']]
            return {k:deepcopy(anchor[k]) for k in ('point','normal','level')}
        return shared
    condenser = a.frozen['graph']['units'].get(unit.get('owner'), {})
    owner = condenser.get('owner', condenser.get('service_owner'))
    returns = [p for p in a.requirements['ports'].values()
               if p['equipment'] == owner and p['port'] == 'condensate_return']
    if len(returns) != 1:
        raise ValueError('Liquid seal has no single owner condensate-return mouth: '+uid)
    target = returns[0]; dn = max(e['dn'] for e in incidents); r = a.radius(dn)
    L = inspect_asset(dn)['install_length_m']; R = 3*r
    level = int(target['level']); floor = a.floors[level]
    # The seal belongs locally to the return mouth, after the overhead run.
    point = np.asarray(target['position'], float).copy()
    _, _, equipment = _contracts(a.frozen)
    rect = equipment[owner].get('rect')
    width, depth = a.scene['footprint']
    if rect:
        margin=R+3*r+.1
        candidates = [(rect[0]-margin, 0), (rect[2]+margin, 0),
                      (rect[1]-margin, 1), (rect[3]+margin, 1)]
        valid = [(abs(coordinate-point[k]), coordinate, k) for coordinate, k in candidates
                 if r+.1 < coordinate < (width if k == 0 else depth)-r-.1]
        if not valid:
            raise ValueError('Liquid seal has no finite side clearance around owner')
        _, coordinate, k = min(valid); point[k] = coordinate
    available = float(target['position'][2])-floor
    depth_m = min(3*L+2*R, available-(R+r+.05))
    if depth_m <= 2*R+EPS:
        raise ValueError('Liquid seal depth/90-degree fittings do not fit the real return level')
    point[2] = float(target['position'][2])-depth_m
    a.material.setdefault('liquid_seals', []).append({'junction': uid, 'owner': owner,
        'condenser': unit.get('owner'), 'bottom_point_m': point.tolist(),
        'return_port_z_m': target['position'][2], 'geometric_depth_m': depth_m,
        'dn': dn, 'scope': 'geometric descend/low drain/return; no seal-pressure calculation'})
    delta=np.asarray(target['position'])-point
    axis=int(np.argmax(abs(delta[:2])));return_normal=np.zeros(3)
    return_normal[axis]=1. if delta[axis]>=0 else -1.
    port_normals={}
    for edge in incidents:
        endpoint=edge['a'] if edge['a'].split('.')[0]==uid else edge['b']
        port_normals[endpoint.split('.')[1]]=([0.,0.,-1.] if edge['role']=='seal_drain'
            else return_normal.tolist() if edge['a'].split('.')[0]==uid else (-return_normal).tolist())
    return {'point': point, 'normal': np.array([0., 0., -1.]), 'level': level,
            'port_normals':port_normals}


def add_material_templates(a, full_graph, pipe_spec=None):
    """One live-assembly extension; return evidence even if its first route fails."""
    specification = deepcopy(pipe_spec) if pipe_spec is not None else load_pipe_spec()
    work = _LiveMaterial(a, full_graph, specification)
    from .piping_material_build import KNOWN_REFUSALS
    failures = []
    try:
        extra = {}
        for uid,unit in full_graph['units'].items():
            if unit.get('kind') not in ('automatic_steam_trap','safety_valve'):continue
            try:
                _inline_components(work,full_graph,a,extra,uid,unit)
            except (ValueError, *KNOWN_REFUSALS) as exc:
                failures.append({'stage':'inline_component','endpoint':uid,'reason':str(exc)})
        work.requirements = requirements_for_end_bundle(work.frozen, full_graph, work.config, specification, extra)
        work.material['requirements'] = deepcopy(work.requirements)
        work.before_material_ports=_safety_connections
        from .piping_material_network import extend_network
        extend_network(work)
    except Exception as exc:
        if not isinstance(exc,KNOWN_REFUSALS):
            import traceback
            work.material['program_error']=f'{type(exc).__name__}: {exc}'
            work.material['program_traceback']=traceback.format_exc()[-3000:]
        failures.append({**getattr(work,'current_step',{}),'code': getattr(exc, 'code', type(exc).__name__), 'reason': str(exc),
                         'evidence': getattr(exc, 'evidence', None)})
        if not work.material['requirements']:
            try:
                work.requirements = requirements_for_end_bundle(work.frozen, full_graph, work.config,
                                                                 specification, locals().get('extra', {}))
                work.material['requirements'] = deepcopy(work.requirements)
            except Exception as requirement_error:
                work.material['requirements_error'] = str(requirement_error)
    finally:
        failures.extend(deepcopy(work.requirements.get('deferred_failures', [])))
        failures.extend(deepcopy(work.material.get('safety_failures', [])))
        failures.extend(deepcopy(work.material.get('network_failures', [])))
        for key in ('scene', 'graph', 'routes', 'allowed_contacts', 'blinds', 'reducers', 'openings', '_serial'):
            setattr(a, key, getattr(work, key))
        a.material = work.material
        a.material.update(template_version='material_templates_v1', failures=failures,
                          generation_status='partial' if failures else 'emitted_pending_actual_measure')
    summary = {'status': 'partial' if failures else 'emitted_pending_actual_measure',
        'failures': failures, 'material': deepcopy(a.material),
        'route_ids': [r['id'] for r in a.routes[work.p2_route_start:]],
        'valve_ids': [v['id'] for v in a.graph['valves'][work.p2_valve_start:]],
        'expected_connections': work.requirements.get('expected_connections', 0),
        'expected_manual_valves': work.requirements.get('expected_manual_valves', 0),
        'scope': 'one first attempt; actual counts require measure_material_templates after native export'}
    a.material_summary = summary
    return summary


def measure_material_templates(frozen, graph, scene, piping, model, data, p1_report=None):
    """Use the original actual-XML segment/native-valve connectivity machinery."""
    from . import piping_checks as common
    from .piping_direction_checks import measure_elbows
    from .piping_material_checks import _material_connectivity, _installation_spaces
    material = piping.get('material', {})
    expected = material.get('requirements', {})
    if not expected.get('edges'):
        return {'status': 'fail', 'errors': [{'code': 'MATERIAL_REQUIREMENTS_MISSING'}],
                'connections': {'expected': 0, 'connected': 0, 'rows': []},
                'manual_valves': {'expected': 0, 'installed': 0, 'rows': []}}
    actual = common._measured(model, data)
    components = {c['id']: c for c in scene['components']}
    elbows = measure_elbows(scene, piping, model, data, actual)
    by_route, route_errors, _ = common._route_segments(piping['routes'], components, actual,
        expected['config']['pipe_outer_diameter_mm'], model, data, elbow_measurements=elbows)
    native, native_errors, segments, _ = common._valves(graph, actual)
    material_graph = {**graph, 'valves': [v for v in graph['valves'] if v.get('phase') == 'P2']}
    directions = (p1_report or {}).get('directions')
    if directions is None:
        from .piping_equipment_flange_checks import measure_joints
        joints = measure_joints(frozen, piping, model, actual)
    else:
        joints = {'rows': directions['rows'], 'errors': directions['errors']}
    connections, manual, errors, topology = _material_connectivity(
        frozen, material_graph, scene, piping, expected, actual, by_route, native, segments,
        measured_flange_rows=joints['rows'])
    errors.extend(e for e in joints['errors'] if e.get('code')=='PG17_EQUIPMENT_FLANGE_GEOMETRY')
    spaces = _installation_spaces(material_graph, expected, actual, by_route, native, components)
    from .piping_material_checks import _transfer_checks
    transfers=_transfer_checks(frozen,expected,p1_report or {'connections':{'rows':[]}})
    errors += route_errors+native_errors+elbows['errors']+spaces['errors']+transfers['errors']
    errors += [{'code': 'MATERIAL_EMISSION_PARTIAL', **row} for row in material.get('failures', [])]
    return {'status': 'fail' if errors else 'pass', 'errors': errors,
        'connections': {'expected': len(connections), 'connected': sum(r['connected'] for r in connections), 'rows': connections},
        'manual_valves': {'expected': len(manual), 'installed': sum(r['installed'] for r in manual), 'rows': manual},
        'transfer_conditions':transfers, 'installation_spaces': spaces, 'topology': {k:v for k,v in topology.items() if k != 'parts'},
        'scope': 'actual exported pipe/elbow/native-package endpoints; no planned coordinates counted as connection proof'}
