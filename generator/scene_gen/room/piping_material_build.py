"""Build the pipes used to transfer material between equipment units."""
from __future__ import annotations
from .root_orientation import planning_geoms

from copy import deepcopy
import json
import math
import pickle
from pathlib import Path

import numpy as np

from .evaluation_common import require_environment
from .piping_assets import inspect_asset, AssetUnavailable
from .piping_build import (Assembly, PipingRefused, _unit,
    _root_spool_points as _p1_spool_points,
    _root_pipe_obstacle, _native_contacts,
    cut_pipe_openings)
from .piping_route_search import RouteUnavailable, simple_route
from .recipe import CandidateRefused
from .piping_elbows import ElbowUnavailable,plan_polyline


SIDES={'x_min':np.array([-1.,0.,0.]),'x_max':np.array([1.,0.,0.]),
       'y_min':np.array([0.,-1.,0.]),'y_max':np.array([0.,1.,0.])}
KNOWN_REFUSALS=(CandidateRefused,AssetUnavailable,RouteUnavailable,ElbowUnavailable)
EPS=1.e-8


def _read(path):return json.loads(Path(path).read_text())


def _why(exc):
    return str(exc)+(' '+json.dumps(exc.evidence,ensure_ascii=False,default=str) if getattr(exc,'evidence',None) else '')


def _fixed_refusal(code,detail,**evidence):
    exc=PipingRefused(code+': '+detail)
    exc.code=code;exc.evidence={'code':code,'fixed_input':True,'detail':detail,**evidence}
    return exc


def _root_spool_points(a,valve):
    points=_p1_spool_points(a,valve);axis=np.asarray(valve['axis']);r=a.radius(valve['dn'])
    gap=a.config['pg_rules']['parallel_pipe_spacing']['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000
    points['pin_far']=points['pin']-axis*(2*r+gap)
    points['pout_far']=points['pout']+axis*(2*r+gap)
    return points


class MaterialAssembly(Assembly):
    """The Assembly primitives with all accepted P1 data retained verbatim."""
    def __init__(self,frozen,p1_folder,requirements,folder):
        require_environment()
        self.frozen=frozen;self.requirements=deepcopy(requirements)
        self.p1_folder=Path(p1_folder).resolve();self.folder=Path(folder).resolve()
        if self.folder.is_relative_to(self.p1_folder) or self.p1_folder.is_relative_to(self.folder):
            raise PipingRefused('Use a separate output for process piping, outside the accepted utility model')
        self.p1_piping=_read(self.p1_folder/'piping.json')
        saved=_read(self.p1_folder/'piping_checks.json')
        if self.p1_piping.get('phase')!='P1' or saved.get('status')!='pass':
            raise PipingRefused('Process piping requires a saved utility model that passed physical checks')
        self.graph=_read(self.p1_folder/'process.json')
        self.scene=_read(self.p1_folder/'geometry.json')
        if self.scene.get('stage')!='piping':raise PipingRefused('The source model is not from the utility piping stage')
        for name in ('sizing.json','shell.json','layout.json','connections.json'):
            record=frozen['frozen_inputs']['files'][name]
            if (self.p1_folder/name).read_bytes()!=record['bytes']:
                raise PipingRefused('Utility model and the four saved layout files differ: '+name)
        self._p1_scene=deepcopy(self.scene);self._p1_graph=deepcopy(self.graph)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.config=deepcopy(self.p1_piping['config']);self.od=self.config['pipe_outer_diameter_mm']
        self.routes=deepcopy(self.p1_piping['routes'])
        self.allowed_contacts=deepcopy(self.p1_piping['allowed_contacts'])
        self.blinds=deepcopy(self.p1_piping['blinds']);self.reducers=deepcopy(self.p1_piping['reducers'])
        self.openings=deepcopy(self.p1_piping['openings'])
        self.plan={'station':deepcopy(self.p1_piping['station']),
            'rack_zones':deepcopy(self.p1_piping['rack_zones']),'openings':self.openings}
        self.floors=deepcopy(self.scene['floors']);self.shell=deepcopy(frozen['shell'])
        w,d=self.shell['footprint_m'];top=max([self.shell['roof_z_m']]+[c['box'][5] for c in self.scene['components']])
        self.bounds=[0.,0.,self.floors[0],w,d,top+.6]
        self._serial=0;self._joins=[];self.original_ids={c['id'] for c in self.scene['components']}
        self.p2_route_start=len(self.routes);self.p2_valve_start=len(self.graph['valves'])
        self.material={'requirements':deepcopy(requirements),'edge_routes':[],
            'junctions':[],'terminals':[],'interfaces':[],'new_openings':[],
            'source_p1_dir':str(self.p1_folder),'scope':'P2 physical connections'}
        self.contexts={};self.anchors={};self.joints={};self._pending=[];self._active_valve=None

    def _id(self,base):
        self._serial+=1
        return 'P2_'+base.removeprefix('P2_')+f'_{self._serial:05}'

    def result(self):
        result=deepcopy(self.p1_piping)
        result.update(phase='P2',p1_piping=deepcopy(self.p1_piping),routes=deepcopy(self.routes),
            allowed_contacts=deepcopy(self.allowed_contacts),blinds=deepcopy(self.blinds),
            reducers=deepcopy(self.reducers),openings=deepcopy(self.openings),material=deepcopy(self.material),
            flange_connections=deepcopy(self.scene.get('flange_connections',[])),
            scope='Retain passed P1 utility piping and add P2 process solids. Operation geometry is reported separately.')
        return result

    def obstacles(self,skip=(),extra=()):
        skip=set(skip)
        rows=[c for c in self.scene['components'] if c['id'] not in skip and not c.get('visual_only')]
        for v in self.graph['valves']:
            if v['id'] in skip:continue
            rows.extend({'id':v['prefix']+g['name'],'kind':'native_valve_part',
                'shape':'box','box':g['box']} for g in planning_geoms(v))
        rows.extend({'id':r['id'],'kind':'pullout_reservation','shape':'box',
            'box':[r['rect'][0],r['rect'][1],r['base_z'],r['rect'][2],r['rect'][3],getattr(self,'_reservation_ceiling',self.bounds[5])]}
            for r in self.scene.get('pullout_reservations',[]))
        for context in self._pending:
            v=context['valve'];port=context['port']
            terminal_ids={t.get('root_source_valve') for t in self.requirements.get('terminals',[])}
            terminal_roots={v['edge'] for v in self.requirements['manual_valves'] if v['id'] in terminal_ids}
            if (not context.get('routed') and context['requirement']['edge'] not in terminal_roots and v['id'] not in skip
                    and v['id']!=self._active_valve and np.allclose(port['normal'],[0,0,1])
                    and np.allclose(v['axis'],[0,0,1])):
                p=np.asarray(context.get('network_tail_end',_root_spool_points(self,v)['pout_far']));level=int(port['level'])
                zone=self.plan['rack_zones'].get(level,self.plan['rack_zones'].get(str(level)))
                z=max(p[2],zone['bottom_z_m']+4*self.radius(v['dn'])) if zone else p[2]
                if z>p[2]+EPS and not any(np.linalg.norm(p-q)<=EPS for q in getattr(self,'_bridge_endpoints',[])):
                    rows.append(_root_pipe_obstacle(v['id']+'_future_top_rise',p,[p[0],p[1],z],self.radius(v['dn'])))
            if context.get('complete'):continue
            if v['id'] in skip or v['id']==self._active_valve:continue
            p=_root_spool_points(self,v)
            rows.extend(_root_pipe_obstacle(v['id']+'_planned_'+side,x,y,self.radius(v['dn']))
                for side,x,y in (('in',p['pin_far'],p['in']),('out',p['out'],p['pout_far'])))
        return rows+list(extra)

    def search(self,a,b,dn,skip=(),clearance=.05,extra=(),straight=False,**constraints):
        pg=self.config['pg_rules']['parallel_pipe_spacing']
        bare=pg['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000
        wall=pg['outermost_pipe_or_insulation_to_column_or_wall_min_mm']/1000
        protrusion=pg['protrusions']['min_clearance_mm']/1000
        obstacles=[]
        port_contacts=constraints.get('port_contacts', ())
        contact_ids={cid for contact in port_contacts for cid in contact['contact_ids']}
        skip=set(skip)-contact_ids
        # A shared endpoint permits contact only locally. Restore the rest of
        # every skipped adjacent long pipe so a later branch cannot cross it.
        contacts=[np.asarray(a),np.asarray(b),*getattr(self,'_network_contact_points',[])]
        restored=[]
        for part in self.scene['components']:
            if part['id'] not in skip or not part.get('ends') or part.get('shape')!='cylinder':continue
            one,two=map(np.asarray,part['ends']);length=float(np.linalg.norm(two-one))
            trim=self.radius(dn)+float(part.get('radius',self.radius(dn)))+max(clearance,bare)+1e-6
            if length<=2*trim:continue
            axis=(two-one)/length;changed=False
            if any(np.linalg.norm(one-p)<=EPS for p in contacts):one=one+axis*trim;changed=True
            if any(np.linalg.norm(two-p)<=EPS for p in contacts):two=two-axis*trim;changed=True
            if changed and float((two-one)@axis)>EPS:
                restored.append(_root_pipe_obstacle(part['id']+'_beyond_contact',one,two,part['radius']))
        for original in self.obstacles(skip,extra)+restored:
            c=deepcopy(original)
            if c['id'] in contact_ids:
                c['physical_box']=c['box']
            if c.get('kind')=='pipe' and c.get('shape')=='cylinder':
                padding=max(0.,bare-clearance)
                c.update(_root_pipe_obstacle(c['id'],*map(np.asarray,c['ends']),c['radius']+padding))
            elif c.get('kind') in ('column','wall','piping_flange'):
                padding=max(0.,(protrusion if c['kind']=='piping_flange' else wall)-clearance)
                c['box']=(np.asarray(c['box'])+np.r_[-np.full(3,padding),np.full(3,padding)]).tolist()
            obstacles.append(c)
        cache=getattr(self,'_exact_search_cache',None)
        diagnostic=constraints.get('diagnostics')
        parameters=(a,b,obstacles,self.radius(dn),self.bounds,self.floors,self.openings,
                    clearance,straight,{k:v for k,v in constraints.items() if k!='diagnostics'})
        key=pickle.dumps(parameters,protocol=5) if cache is not None else None
        if cache is not None and key in cache:
            self._exact_search_counts['hits']+=1
            result,error,saved=cache[key]
            if diagnostic is not None:
                diagnostic.clear();diagnostic.update(deepcopy(saved))
                diagnostic['reused_identical_search']=True
            if error:raise RouteUnavailable(**deepcopy(error))
            return deepcopy(result)
        if cache is not None:self._exact_search_counts['misses']+=1
        result=None;error=None
        try:
            result=simple_route(a,b,obstacles,self.radius(dn),self.bounds,self.floors,
                openings=self.openings,clearance=clearance,straight_only=straight,**constraints)
            return result
        except RouteUnavailable as exc:
            error=deepcopy(exc.evidence)
            raise
        finally:
            if cache is not None and (result is not None or error is not None):
                if len(cache)>=512:cache.pop(next(iter(cache)))
                cache[key]=(deepcopy(result),error,deepcopy(diagnostic or {}))

    def snapshot(self):
        return deepcopy((self.scene,self.routes,self.allowed_contacts,self.blinds,self.reducers,self.openings,self.material))

    def restore(self,state):
        (self.scene,self.routes,self.allowed_contacts,self.blinds,self.reducers,self.openings,self.material)=state

    def route(self,name,points,dn,medium,role,level,**metadata):
        return super().route(name,points,dn,medium,role,level,phase='P2',**metadata)


def _medium(edge):return edge.get('medium_identity',edge.get('medium','material'))


def _meta(edge):
    return {'source_edge':edge['id'],'a':edge['a'],'b':edge['b'],
        'path_policy':edge.get('path_policy','ordinary_edge_then_rack'),
        'medium_identity':_medium(edge)}


def _joint_components(a,position):
    """Only actual adjacent endpoint pieces, never an equipment/route owner."""
    return [c['id'] for c in a.scene['components'] if c.get('ends') and
        any(math.dist(position,p)<=EPS for p in c['ends'])]


def _departure_contacts(*anchors):
    return tuple(anchor['departure_contact'] for anchor in anchors if anchor.get('departure_contact'))


def _adjacent_parts(a,anchor,position):
    joint=anchor.get('departure_contact')
    return [] if joint and math.dist(position,joint['position'])<=EPS else _joint_components(a,position)


def _join_contacts(a,route,point):
    own=set(route['components'])
    for cid in _joint_components(a,point):
        if cid in own:continue
        for other in route['components']:
            c=next(c for c in a.scene['components'] if c['id']==other)
            if any(math.dist(point,p)<=EPS for p in c.get('ends',[])):a.allow(cid,other,point)


def _pump_stub(a,port,face,logical,dn):
    """Exact outward flat-face joint, with every noncontact solid retained."""
    face=np.asarray(face,float);logical=np.asarray(logical,float)
    normal=_unit(port['normal']);physical=port['physical_interface']
    skip=[port['equipment']]
    for cid in physical.get('coincident_obstacles',[]):
        c=next(c for c in a.scene['components'] if c['id']==cid)
        ends=np.asarray(c['ends']);axis=_unit(ends[1]-ends[0])
        # No volume may lie beyond the actual source face into this stub.
        if (c.get('shape')=='cylinder' and abs(abs(axis@normal)-1)<EPS
                and max((ends-face)@normal)<=EPS
                and np.linalg.norm(np.cross(logical-face,normal))<=EPS
                and (logical-face)@normal>EPS):
            skip.append(cid)
    return a.search(face,logical,dn,skip=tuple(skip),clearance=0.,straight=True)


def _adapter(a,port,edge,exit_direction=None):
    """A real original-radius reducer, or an explicitly measured pump face."""
    endpoint=port['endpoint'];normal=_unit(port['normal']);logical=np.asarray(port['position'],float)
    physical=port.get('physical_interface',{});source=logical.copy();pump_bridge=None;joint=None
    dn=int(edge['dn']);medium=_medium(edge);nozzle=port.get('nozzle_component')
    if isinstance(nozzle,dict):nozzle=nozzle['id']
    if port.get('actual_radius_m') is None:
        if physical.get('kind')!='body_face' or not physical.get('component_ids'):
            raise PipingRefused(endpoint+'A physical stub or measured pump face is required')
        # The old logical interface remains the required coordinate.  A pipe
        # may bridge its original axis to the measured pump face, but may not
        # pass through the motor/base or reinterpret that point as a new port.
        radius=float(port['connection_radius_m']);source_dn=int(port['source_dn'])
        face=np.asarray(physical['position'],float);uid=port['equipment']
        if math.dist(face,logical)>EPS:
            try:
                _pump_stub(a,port,face,logical,source_dn)
            except RouteUnavailable as exc:
                raise _fixed_refusal('P2_PUMP_INTERFACE_UNAVAILABLE',endpoint+'The frozen port-to-pump path intersects the motor, base or another solid: '+str(exc),
                    endpoint=endpoint,logical_position=logical.tolist(),actual_face=face.tolist(),route_evidence=exc.evidence) from exc
            pump_bridge=a.route('P2_'+edge['id']+'_pump_face', [face,logical],source_dn,medium,
                'pump_original_interface_bridge',port['level'],source={'position':face.tolist(),'equipment':uid},
                target={'position':logical.tolist(),'endpoint':endpoint},**_meta(edge))
            a.allow(uid,pump_bridge['components'][0],face)
    else:radius=float(port['actual_radius_m']);source_dn=int(port['source_dn'])
    name='interface_'+endpoint.replace('.','_')
    if port.get('source_catalogue_face'):
        from .catalogue_flange import emit_catalogue_adapter

        joint,c=emit_catalogue_adapter(a,port,'P2_'+edge['id']+'_'+name,medium,dn)
    elif port.get('source_flange'):
        from .piping_equipment_flange import emit_flanged_stub

        point={**port,'medium':medium,'source_edge':edge['id']}
        joint=emit_flanged_stub(a,point,name,phase='P2')
        source=np.asarray(joint['end'])
    r=a.radius(dn);floor=max((z for z in a.floors if z<=source[2]+EPS),default=a.bounds[2])
    straight_available=source[2]-floor-3*r-r-.05-.01
    if joint is not None and joint.get('catalogue_joint'):
        end=np.asarray(joint['end'])
    elif joint is not None and source_dn==dn:
        # The flange sleeve already leaves the equipment on the actual DN.
        # Nominally identical OD variants fit the same pipe size.
        end=source
        c=next(part for part in a.scene['components'] if part['id']==joint['pipe_stub'])
    elif joint is None and normal[2]<-.5 and source_dn==dn and abs(radius-r)<=EPS and straight_available<2*r:
        # Equal bores need no reducer. Keep a real straight at the original
        # mouth, sized from the available drop, then use the unchanged elbow.
        # The floor clearance and elbow radius are not reduced.
        if straight_available<=EPS:raise PipingRefused(endpoint+' has no space for the original outlet and a real elbow above the floor')
        end=source+normal*straight_available
        c=a.pipe(name,source,end,dn,medium,name)
        c.update(equal_bore_straight=True,source_dn=source_dn,target_dn=dn,
                 basis='actual floor drop minus real elbow R, pipe radius and unchanged clearance')
    else:
        end,c=a.reducer(name,source,normal,source_dn,dn,medium,radius)
    c.update(phase='P2',source_edge=edge['id'],equipment_endpoint=endpoint)
    components=[c['id']];lead_points=[source.tolist(),end.tolist()]
    if joint:
        if joint['pipe_stub']!=c['id']:
            a.allow(joint['pipe_stub'],c['id'],source,normal=normal)
            components.insert(0,joint['pipe_stub'])
        if joint.get('through_bore'):
            components.insert(0,joint['through_bore'])
            lead_points=[logical.tolist(),c['ends'][0],end.tolist()]
        else:
            lead_points=[logical.tolist(),source.tolist()]
            if math.dist(source,end)>EPS:lead_points.append(end.tolist())
    elif nozzle:a.allow(nozzle,c['id'],source)
    elif physical and math.dist(source,physical['position'])<=EPS:
        # Only the actual pump cylinder face, never a coincident motor/base.
        a.allow(port['equipment'],c['id'],source)
    interface={'endpoint':endpoint,'source_edge':edge['id'],
        'original_logical_position':logical.tolist(),'physical_position':logical.tolist(),
        'normal':normal.tolist(),'actual_radius_m':port.get('actual_radius_m'),
        'connection_radius_m':radius,'adapter':c['id'],'physical_interface':deepcopy(physical)}
    if joint:
        interface.update(equipment_flange_joint=joint,departure_contact={
            'position':end.tolist(),'normal':normal.tolist(),
            'contact_ids':joint.get('departure_contact_ids',list(dict.fromkeys([*joint['source_components'],*joint['components'],*components])))})
    a.material['interfaces'].append(interface)
    failures=[];directions=[normal]
    if exit_direction is not None:directions.append(_unit(exit_direction))
    directions.extend(SIDES.values())
    for direction in directions:
        direction=_unit(direction)
        # Low mouths may turn horizontally immediately after the real
        # reducer, while sufficiently high mouths keep their downward lead.
        # The adapter's real source face/normal and dimensions are unchanged.
        distance=a.radius(dn)+.01 if float(direction@normal)>1.-EPS else max(radius,a.radius(dn))+a.radius(dn)+.01
        far=end+direction*distance
        try:
            a.search(end,far,dn,skip=(c['id'],),clearance=.01,straight=True,
                     port_contacts=_departure_contacts(interface))
            break
        except RouteUnavailable as exc:failures.append(_why(exc))
    else:raise PipingRefused(endpoint+'No legal straight in the original direction or at right angles after the reducer: '+'；'.join(failures[:2]))
    # Record the actual reducer alone.  Its short normal continuation is
    # emitted with the following lead, after search.  Emitting it first and
    # then adding another capsule-end escape would consume valid space under
    # the unchanged pullout lower plane twice.
    lead={'id':'P2_'+edge['id']+'_'+name,
        'points':lead_points,'components':components,'valves':[],
        'dn':dn,'medium':medium,'role':'material_interface','level':int(port['level']),'phase':'P2',
        'source':{'position':logical.tolist(),'endpoint':endpoint},'target':{'position':end.tolist()},**_meta(edge)}
    a.routes.append(lead)
    if pump_bridge:_join_contacts(a,lead,logical)
    return {'point':far,'adapter_end':end,'normal':direction,'level':int(port['level']),'routes':[lead['id']],
            'components':lead['components'].copy(),'physical_port':deepcopy(port),
            'departure_contact':interface.get('departure_contact')}


def _opening(a,point,level,dn,tag,rect=None):
    # Route candidates propose positions, only emitted geometry cuts a grating.
    if hasattr(a,'deferred_valves'):return
    r=a.radius(dn);p=np.asarray(point);rect=rect or [p[0]-r,p[1]-r,p[0]+r,p[1]+r]
    row={'id':tag,'level':int(level),'rect':list(map(float,rect)),'phase':'P2','dn':int(dn),
        'source':'Use the actual planned pipe exterior. Preserve existing utility openings and equipment holes.'}
    cut_pipe_openings(a.scene,[row]);a.openings.append(row)
    a.scene['piping_openings']=deepcopy(a.openings);a.material['new_openings'].append(row)


def _bridge_with_root_tails(a,edge,start,end,points):
    """Include adjacent native outlet spools in the same real elbow plan."""
    merged=[np.asarray(p).tolist() for p in points];tails=[]
    valves={v['id']:v for v in a.graph['valves']}
    for which,anchor in enumerate((start,end)):
        found=[]
        for route in a.routes:
            ids=route.get('valves',[])
            if route.get('source_edge')!=edge['id'] or len(ids)!=1:continue
            valve=valves[ids[0]]
            if (math.dist(route['points'][-1],anchor['point'])<=EPS
                    and math.dist(route['points'][0],valve['world_ports']['out']['position'])<=EPS):found.append(route)
        if len(found)>1:raise PipingRefused('Ambiguous native outlet spool for '+edge['id'])
        if found:
            route=found[0];tail=route.get('control_points',route['points']);tails.append((which,route))
            merged=(tail+merged[1:] if which==0 else merged+list(reversed(tail))[1:])
    return merged,tails


def _terminal(a,terminal,edge,*,pending_root=False,stub_length=None):
    endpoint=terminal['boundary_endpoint'];other=terminal['network_endpoint']
    if other not in a.anchors:raise PipingRefused(edge['id']+'Blind termination lacks the physical root valve or joint')
    start=a.anchors[other];point=np.asarray(start['point']);normal=_unit(start['normal']);dn=int(edge['dn']);r=a.radius(dn)
    # Only straight adapters/flanges needed to close the actual unused end.
    end=point+normal*(2*r+.05 if stub_length is None else stub_length)
    a.search(point,end,dn,skip=_joint_components(a,point),clearance=.01,straight=True)
    route=a.route('P2_'+edge['id']+'_terminal',[point,end],dn,_medium(edge),'material_blind_terminal',start['level'],
        source={'endpoint':other,'position':point.tolist()},target={'endpoint':endpoint,'position':end.tolist()},**_meta(edge))
    _join_contacts(a,route,point)
    flange=a.flange_interface('P2_'+edge['id']+'_terminal_flange',end,normal,dn,_medium(edge),start['level'])
    cap_position=end+normal*(.018*2+.003)
    cap=a.blind('P2_'+edge['id'],cap_position,normal,dn,_medium(edge),start['level'],'material_blind_terminal')
    cap.update(phase='P2',source_edge=edge['id'],endpoint=endpoint)
    # Same-OD stub passes through the hollow flange bore, to the blind face.
    extension=a.route('P2_'+edge['id']+'_flange_spool',[end,cap_position],dn,_medium(edge),'material_blind_terminal',start['level'],**_meta(edge))
    _join_contacts(a,extension,end)
    if extension['components']:a.allow(extension['components'][-1],cap['id'],cap_position)
    a.anchors[endpoint]={'point':cap_position,'normal':normal,'level':start['level']}
    source_root=terminal['root_source_valve']
    actual_root=next((v['id'] for v in a.graph['valves'] if v.get('source_valve')==source_root),None)
    if actual_root is None and not pending_root:raise PipingRefused(edge['id']+'Termination lacks the required root valve: '+source_root)
    record={**deepcopy(terminal),'source_edge':edge['id'],'boundary_position':cap_position.tolist(),
        'actual_blinds':[cap['id']],'actual_flanges':flange['components'],'routes':[route['id'],extension['id']],
        'blind':cap['id'],'root':actual_root,'flange_components':flange['components'],
        'root_source_valve':source_root,
        'scope':'No specified external destination. End at the root valve and short blinded flange.'}
    a.material['terminals'].append(record)
