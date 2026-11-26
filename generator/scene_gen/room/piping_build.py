"""Build pipes and fittings from the selected routes."""
from __future__ import annotations

from copy import deepcopy
import itertools
import math
from pathlib import Path

import numpy as np

from .layout import add_pipe, add_box
from .piping_assets import inspect_asset, place_asset, AssetUnavailable
from .piping_requirements import load_stage_config
from .recipe import CandidateRefused
from .piping_route_search import RouteUnavailable


class PipingRefused(CandidateRefused):
    pass


def _overlap(a, b, eps=1.e-5):
    return all(min(a[i+3], b[i+3])-max(a[i], b[i]) > eps for i in range(3))


def _unit(v):
    v=np.asarray(v,dtype=float); n=float(np.linalg.norm(v))
    if n < 1.e-10: raise PipingRefused('Cannot install pipe along a zero-length direction')
    return v/n


def _pg14_limits(assembly):
    pg=assembly.config['pg_rules']['parallel_pipe_spacing']
    return {'bare':pg['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000.,
            'protrusion':pg['protrusions']['min_clearance_mm']/1000.,
            'wall':pg['outermost_pipe_or_insulation_to_column_or_wall_min_mm']/1000.}


def _pg14_planning_obstacles(assembly,components,clearance,*,moving_protrusion=False):
    """Add the missing part of existing PG14 clearance in search only.

    Local nozzle/reducer attachment can retain its 10 mm search allowance.
    That allowance never reduces the 50 mm kept from other pipe solids, the
    30 mm at native protrusions/flanges, or 100 mm at actual columns/walls.
    No expanded shape is emitted or used to replace actual XML geometry.
    """
    limits=_pg14_limits(assembly);result=[]
    for original in components:
        c=deepcopy(original);kind=c.get('kind');gap=None
        if kind in ('column','wall'):gap=limits['wall']
        elif kind=='pipe':gap=limits['protrusion' if moving_protrusion else 'bare']
        elif kind in ('valve','piping_flange','piping_blind','piping_reducer','piping_elbow','flange','blind'):gap=limits['protrusion']
        padding=max(0.,gap-clearance) if gap is not None else 0.
        if padding:
            if c.get('shape') in ('cylinder','capsule'):
                c['radius']+=padding
                axis=_unit(np.asarray(c['ends'][1])-c['ends'][0])
                extent=c['radius']*np.sqrt(np.maximum(0.,1-axis**2))
                if c['shape']=='capsule':extent=np.full(3,c['radius'])
                elif kind in ('piping_blind','blind'):
                    # A blind protrudes at its end face too. Keep that same
                    # PG14 padding in broad phase, not only radially.
                    extent+=abs(axis)*padding
                c['box']=np.r_[np.minimum(*map(np.asarray,c['ends']))-extent,
                                np.maximum(*map(np.asarray,c['ends']))+extent].tolist()
            else:c['box']=(np.asarray(c['box'])+np.r_[-np.full(3,padding),np.full(3,padding)]).tolist()
        result.append(c)
    return result


def _planned_parallel_check(assembly,routes=None):
    """Early reject real emitted cylinder lanes, including same-route returns.

    This is a planning precheck. The independent checker reloads all actual XML
    and remains authoritative for pipes, protrusions and columns/walls.
    """
    bare=_pg14_limits(assembly)['bare'];eps=1.e-8
    names=None if routes is None else {cid for r in routes for cid in r['components']}
    parts=[c for c in assembly.scene['components'] if c.get('kind')=='pipe' and c.get('shape')=='cylinder']
    pairs=[]
    for i,first in enumerate(parts):
        for second in parts[i+1:]:
            if names is not None and first['id'] not in names and second['id'] not in names:continue
            p,q=map(np.asarray,first['ends']);a,b=map(np.asarray,second['ends'])
            axis=_unit(q-p);other=_unit(b-a)
            if np.linalg.norm(np.cross(axis,other))>eps:continue
            one=sorted((float(p@axis),float(q@axis)));two=sorted((float(a@axis),float(b@axis)))
            overlap=min(one[1],two[1])-max(one[0],two[0])
            if overlap<=eps:continue
            delta=a-p;distance=float(np.linalg.norm(delta-(delta@axis)*axis))
            if distance<=eps:continue  # Collinear overlap is never excused; original solid checker handles it.
            gap=distance-first['radius']-second['radius']
            if gap+eps<bare:
                pairs.append({'first':first['id'],'second':second['id'],'actual_planned_gap_m':gap,
                    'minimum_m':bare,'axial_overlap_m':overlap,
                    'same_route':first.get('route_id')==second.get('route_id'),
                    'route_ids':[first.get('route_id'),second.get('route_id')],
                    'ends':[deepcopy(first['ends']),deepcopy(second['ends'])]})
    if pairs:
        exc=PipingRefused('PG14: insufficient measured clearance between emitted parallel straight pipes, including same-route pairs: '+str(pairs[:2]))
        exc.evidence={'code':'PG14_PLANNED_PARALLEL_PIPE_CLEARANCE','pairs':pairs,
            'scope':'Precheck of cylinders to be emitted. Keep the independent XML check and all owner and route checks.'}
        raise exc
    return {'status':'pass','scope':'Precheck of actual parallel cylinders. Coaxial overlap is checked on the physical solids.'}


def _subtract_rect(rect, cut):
    x0,y0,x1,y1=rect; a,b,c,d=cut
    if min(x1,c)<=max(x0,a) or min(y1,d)<=max(y0,b):return [rect]
    a=max(a,x0);b=max(b,y0);c=min(c,x1);d=min(d,y1)
    return [r for r in ([x0,y0,a,y1],[c,y0,x1,y1],[a,y0,c,b],[a,d,c,y1])
            if r[2]-r[0]>1.e-8 and r[3]-r[1]>1.e-8]


def cut_pipe_openings(scene, openings):
    """Cut only copied floor plates, preserving frozen inputs and floor levels."""
    from .grating import grating
    new=[]; surfaces=[]
    for c in scene['components']:
        if not grating(c) or c['box'][5] <= scene['floors'][0]+1.e-6:
            new.append(c);continue
        level=min(range(len(scene['floors'])),key=lambda i:abs(scene['floors'][i]-c['box'][5]))
        rects=[[c['box'][0],c['box'][1],c['box'][3],c['box'][4]]]
        for opening in openings:
            if int(opening['level'])!=level:continue
            rects=[part for r in rects for part in _subtract_rect(r,opening['rect'])]
        original_rect=[c['box'][0],c['box'][1],c['box'][3],c['box'][4]]
        if rects==[original_rect]:
            new.append(c);continue
        for i,r in enumerate(rects):
            clone=deepcopy(c);clone['id']=c['id']+f'_piping_plate{i}'
            clone['box']=[r[0],r[1],c['box'][2],r[2],r[3],c['box'][5]]
            clone['piping_floor_source']=c['id'];new.append(clone)
    for surface in scene.get('surfaces',[]):
        if not surface.get('level'):
            surfaces.append(surface);continue
        box=surface['box'];rects=[[box[0],box[1],box[3],box[4]]]
        for opening in openings:
            if int(opening['level'])==surface['level']:
                rects=[part for r in rects for part in _subtract_rect(r,opening['rect'])]
        surfaces.extend({**deepcopy(surface),'box':[r[0],r[1],box[2],r[2],r[3],box[5]]} for r in rects)
    scene['components']=new;scene['surfaces']=surfaces
    scene['piping_openings']=deepcopy(openings)
    for o in openings:
        scene.setdefault('holes',{}).setdefault(str(o['level']),[]).append(deepcopy(o['rect']))


class Assembly:
    def __init__(self,frozen,requirements,plan,folder):
        self.frozen=frozen;self.requirements=requirements;self.plan=deepcopy(plan)
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.graph=deepcopy(frozen['graph']);self.graph['valves']=[]
        self.scene=deepcopy(frozen['scene']);self.scene['stage']='piping'
        self.config=requirements['config'];self.od=self.config['pipe_outer_diameter_mm']
        self.routes=[];self.allowed_contacts=[];self.blinds=[];self.reducers=[]
        self.planned_valves=[];self.valve_reservations=[];self.search_diagnostics=[]
        self.header_tap_plans={};self.header_tap_options={}
        self._serial=0;self._joins=[]
        self.openings=deepcopy(plan.get('openings',[]))
        cut_pipe_openings(self.scene,self.openings)
        self.original_ids={c['id'] for c in self.scene['components']}
        self.scene['components'].extend(deepcopy(plan.get('rack_support_components',[])))
        for c in self.scene['components']:
            if c['id'] not in self.original_ids:c['piping']=True
        self.floors=self.scene['floors'];self.shell=frozen['shell']
        width,depth=self.shell['footprint_m'];roof=self.shell['roof_z_m']
        self.bounds=[0.,0.,0.,width,depth,max(roof,max(z['box'][5] for z in plan['rack_zones'].values())+.6)]

    def radius(self,dn):return float(self.od[str(int(dn))])/2000.

    def color(self,medium):
        return list(self.config['pg_rules']['appearance']['color_map'].get(medium,
                    self.config['pg_rules']['appearance']['color_map']['unknown'])['rgb'])

    def _id(self,base):
        self._serial+=1;return f'{base}_{self._serial:04}'

    def pipe(self,name,a,b,dn,medium,route_id,kind='pipe'):
        from .piping_elbows import is_axis_aligned
        if not is_axis_aligned(a,b):raise PipingRefused('PG17_NON_ORTHOGONAL_ROUTE: new straight pipes must follow the building axes')
        c=add_pipe(self.scene,self._id(name),a,b,self.radius(dn),kind,self.color(medium),route_id)
        if c:c.update(dn=int(dn),medium=medium,piping=True,route_id=route_id)
        return c

    def route(self,name,points,dn,medium,role,level,**metadata):
        from .piping_elbows import plan_polyline,emit_elbow,ElbowUnavailable
        pts=[list(map(float,p)) for p in points]
        retired=set(metadata)&{'pg13_local','allowed_45_corners','exceptions','exception_registry','direction_exception'}
        if retired:
            raise ElbowUnavailable('PG17_EXCEPTIONS_DISABLED','Direction exemptions are unsupported',fields=sorted(retired))
        plan=plan_polyline(pts,self.radius(dn))
        route={'id':name,'points':[],'control_points':pts,'dn':int(dn),'medium':medium,'role':role,
               'level':int(level),'components':[],'pipe_components':[],'elbows':[],'valves':[],**deepcopy(metadata)}
        parts=[]
        def append_point(p):
            p=list(map(float,p))
            if not route['points'] or np.max(abs(np.asarray(route['points'][-1])-p))>1.e-8:route['points'].append(p)
        for item in plan['ordered']:
            if item['kind']=='run':
                run=plan['runs'][item['index']]
                c=self.pipe(name,run['start'],run['end'],dn,medium,name)
                if c is None:raise RuntimeError('No physical straight remained after elbow trimming')
                c.update(level=int(level),straight_segment_index=run['segment_index'])
                parts.append((c,run['start'],run['end']));route['pipe_components'].append(c['id'])
                append_point(run['start']);append_point(run['end'])
            else:
                elbow=plan['elbows'][item['index']];identifier=self._id(name+'_elbow')
                components,record=emit_elbow(identifier,elbow,self.folder,self.color(medium),dn,name)
                record.update(medium=medium,level=int(level))
                for c in components:
                    c.update(medium=medium,level=int(level))
                    parts.append((c,c['ports']['in']['position'],c['ports']['out']['position']))
                self.scene['components'].extend(components);self.scene.setdefault('elbows',[]).append(record)
                route['elbows'].append(record['id'])
                for p in record['centerline_points']:append_point(p)
        for c,_,_ in parts:route['components'].append(c['id'])
        for (a,_,end),(b,start,_) in zip(parts,parts[1:]):
            if math.dist(end,start)>1.e-8:raise RuntimeError('Physical elbow and straight faces are disconnected')
            self.allow(a['id'],b['id'],end)
        route['bend_count']=len(plan['elbows'])
        route['actual_planned_centerline_length_m']=sum(math.dist(r['start'],r['end']) for r in plan['runs'])+sum(
            e['centerline_radius_m']*math.radians(e['angle_degrees']) for e in plan['elbows'])
        from .grating import cut_for_parts
        cut_for_parts(self,route,[c for c,_,_ in parts])
        self.routes.append(route);return route

    def allow(self,a,b,p,*,normal=None):
        record={'pair':[a,b],'point':list(map(float,p))}
        if normal is not None:
            record['normal']=list(map(float,normal))
        self.allowed_contacts.append(record)

    def blind(self,name,position,axis,dn,medium,level,role):
        # Simple solid flange/cap; actual bolt pressure rating is not modelled.
        p=np.asarray(position);n=_unit(axis);radius=self.radius(dn)+.034
        if np.count_nonzero(abs(n)>1.e-8)!=1:raise PipingRefused('PG17_NON_ORTHOGONAL_FITTING: blind plate connection axis must be orthogonal')
        c=add_pipe(self.scene,self._id(name+'_blind'),p,p+n*.018,radius,
                   'piping_blind',self.color(medium),name)
        c.update(piping=True,dn=int(dn),medium=medium,level=level,
                 coarse_estimate='Coarse model uses a 34 mm flange extension and an 18 mm blind plate')
        record={'id':c['id'],'position':p.tolist(),'axis':n.tolist(),'dn':dn,
                'role':role,'level':level,'components':[c['id']]}
        self.blinds.append(record);return record

    def flange_interface(self,name,position,axis,dn,medium,level):
        """Two hollow coarse flanges around a real pipe; no solid disk in bore."""
        import mujoco
        p=np.asarray(position);n=_unit(axis);r=self.radius(dn);outer=r+.034
        if np.count_nonzero(abs(n)>1.e-8)!=1:raise PipingRefused('PG17_NON_ORTHOGONAL_FITTING: flange connection axis must be orthogonal')
        helper=np.array([0.,0.,1.]) if abs(n[2])<.9 else np.array([0.,1.,0.])
        x=_unit(np.cross(n,helper));y=np.cross(n,x)
        matrix=np.column_stack((x,y,n));quat=np.empty(4);mujoco.mju_mat2Quat(quat,matrix.ravel())
        ids=[];sectors=16;inner=r/math.cos(math.pi/sectors)
        for side in (0,1):
            start=p+n*(side*(.018+.003));end=start+n*.018
            for index in range(sectors):
                angles=[2*math.pi*index/sectors,2*math.pi*(index+1)/sectors]
                vertices=[[rr*math.cos(theta),rr*math.sin(theta),z]
                          for z in (0.,.018) for rr in (inner,outer) for theta in angles]
                faces=[(1,2,4),(1,4,3),(5,7,8),(5,8,6),(1,5,6),(1,6,2),
                       (3,4,8),(3,8,7),(1,3,7),(1,7,5),(2,6,8),(2,8,4)]
                cid=self._id(name+f'_flange{side}_sector{index}')
                path=self.folder/(cid+'.obj')
                path.write_text('\n'.join('v '+' '.join(f'{v:.17g}' for v in row) for row in vertices)+'\n'+
                                '\n'.join('f '+' '.join(map(str,row)) for row in faces)+'\n')
                points=np.asarray(vertices)@matrix.T+start
                self.scene['components'].append({'id':cid,'kind':'piping_flange','shape':'mesh',
                    'mesh_path':str(path.resolve()),'collision_mesh':True,'pos':start.tolist(),'quat':quat.tolist(),
                    'box':np.r_[points.min(axis=0),points.max(axis=0)].tolist(),'color':self.color(medium),
                    'owner':name,'dn':dn,'medium':medium,'level':level,'piping':True,
                    'coarse_estimate':'Hollow flange with 16 sectors, 18 mm per plate, 3 mm between plates, and 34 mm radial extension'})
                ids.append(cid)
        record={'id':name,'kind':'thread_adapter_flanged_interface','position':p.tolist(),
                'axis':n.tolist(),'dn':dn,'level':level,'components':ids,
                'adapter':'Same-OD straight stub between the original valve and flange. Thread profile is omitted.'}
        self.scene.setdefault('flange_connections',[]).append(record)
        return record

    def reducer(self,name,start,axis,source_dn,target_dn,medium,source_radius=None):
        p=np.asarray(start,dtype=float);n=_unit(axis)
        if np.count_nonzero(abs(n)>1.e-8)!=1:raise PipingRefused('PG17_NON_ORTHOGONAL_FITTING: reducers need an orthogonal axis')
        length=max(self.radius(source_dn),self.radius(target_dn))*2
        end=p+n*length
        r0=float(source_radius) if source_radius else self.radius(source_dn);r1=self.radius(target_dn)
        helper=np.array([0.,0.,1.]) if abs(n[2])<.9 else np.array([0.,1.,0.])
        x=_unit(np.cross(n,helper));y=np.cross(n,x);count=24
        vertices=[(q+r*(math.cos(2*math.pi*i/count)*x+math.sin(2*math.pi*i/count)*y)).tolist()
                  for q,r in ((p,r0),(end,r1)) for i in range(count)]
        vertices.extend([p.tolist(),end.tolist()]);faces=[]
        for i in range(count):
            j=(i+1)%count;faces.extend([(i+1,j+1,j+1+count),(i+1,j+1+count,i+1+count),
                                        (2*count+1,j+1,i+1),(2*count+2,i+count+1,j+count+1)])
        mesh=self.folder/(self._id(name)+'.obj')
        # Keep mesh vertices close to the origin: MuJoCo stores them as float32.
        # World-sized OBJ vertices would introduce false endpoint gaps at 1e-8m.
        local_vertices=(np.asarray(vertices)-p).tolist()
        mesh.write_text('\n'.join('v '+' '.join(f'{v:.17g}' for v in row) for row in local_vertices)+'\n'+
                        '\n'.join('f '+' '.join(str(v) for v in f) for f in faces)+'\n')
        vv=np.asarray(vertices);cid=mesh.stem
        c={'id':cid,'kind':'piping_reducer','shape':'mesh','mesh_path':str(mesh.resolve()),
           'collision_mesh':True,'pos':p.tolist(),'ends':[p.tolist(),end.tolist()],'radii':[r0,r1],
           'box':np.r_[vv.min(axis=0),vv.max(axis=0)].tolist(),'color':self.color(medium),
           'owner':name,'source_dn':int(source_dn),'target_dn':int(target_dn),'dn':int(target_dn),
           'medium':medium,'piping':True,'coarse_estimate':'24-sided solid frustum with length equal to the large-end OD'}
        self.scene['components'].append(c);self.reducers.append(deepcopy(c))
        return end,c

    def plan_valve(self,name,dn,in_position,axis,stem,medium,role,level,**metadata):
        asset=inspect_asset(int(dn));direction=_unit(axis);stem=_unit(stem)
        if np.count_nonzero(abs(direction)>1.e-8)!=1:raise PipingRefused('PG17_NON_ORTHOGONAL_FITTING: new valve pipe axis must be orthogonal. Keep native package geometry.')
        # Place original valve_base so its real, possibly asymmetric in-face aligns.
        initial=place_asset(asset,[0,0,0],direction,stem)
        center=np.asarray(in_position)-np.asarray(initial['world_ports']['in']['position'])
        v=place_asset(asset,center,direction,stem)
        v.update(id=name,name=name,role=role,dn=int(dn),level=int(level),medium=medium,
                 geom_prefix=name+'_',prefix=name+'_',**deepcopy(metadata))
        v.update(root_axial_envelope=_root_axial_envelope(v),root_pipe_radius_m=self.radius(dn))
        return self.reserve_valve(v)

    def reserve_valve(self,valve):
        """Keep the original package/axial/full-travel slot, without emitting it."""
        if 'root_axial_envelope' not in valve:
            valve['root_axial_envelope']=_root_axial_envelope(valve)
        valve.setdefault('root_pipe_radius_m',self.radius(valve['dn']))
        envelope=valve['root_axial_envelope'];lo=envelope['min_original_axes_m'];hi=envelope['max_original_axes_m']
        local=np.asarray(list(itertools.product(*zip(lo,hi))))
        origin=np.asarray(valve.get('origin',valve.get('center')))
        world=local@np.asarray(valve['matrix']).T+origin
        valve['reserved_operation_box']=np.r_[world.min(axis=0),world.max(axis=0)].tolist()
        self.planned_valves[:]=[v for v in self.planned_valves if v['id']!=valve['id']]
        self.planned_valves.append(valve)
        return valve

    def install_valve(self,valve):
        """Attach the unchanged real package only after its two pipe paths exist."""
        if any(v['id']==valve['id'] for v in self.graph['valves']):
            raise ValueError('A physical valve cannot be installed twice: '+valve['id'])
        points=_root_spool_points(self,valve)
        self.valve_reservations.append({'id':valve['id'],'package':valve['package'],
            'install_length_m':valve['install_length_m'],'native_ports':deepcopy(valve['world_ports']),
            'axial_spool_points':{k:np.asarray(p).tolist() for k,p in points.items()},
            'full_travel_box':deepcopy(valve['reserved_operation_box']),
            'frozen_stance_box':deepcopy(valve.get('reserved_stance_box')),
            'routes_before_install':len(self.routes),'stage':'real_package_attached_after_route_plan',
            'physics_steps':0})
        self.planned_valves[:]=[v for v in self.planned_valves if v['id']!=valve['id']]
        self.graph['valves'].append(valve)
        return valve

    def valve(self,name,dn,in_position,axis,stem,medium,role,level,**metadata):
        # Compatibility for existing fixed controls; production uses plan/install.
        return self.install_valve(self.plan_valve(name,dn,in_position,axis,stem,medium,role,level,**metadata))

    def all_valves(self):
        return self.graph['valves']+self.planned_valves

    def obstacles(self,skip=()):
        # Only installed valves are solids. Planned valves, operation boxes and
        # standing areas never steer a pipe (valves are fitted last).
        return [c for c in self.scene['components'] if c['id'] not in set(skip) and not c.get('visual_only')] + [
            {'id':v['id']+'_native_bbox','kind':'valve','shape':'box','box':v['world_bbox']}
            for v in self.graph['valves'] if v['id'] not in set(skip)] + [
            {'id':r['id'],'kind':'pullout_reservation','shape':'box',
             'box':[r['rect'][0],r['rect'][1],r['base_z'],r['rect'][2],r['rect'][3],self.bounds[5]]}
            for r in self.scene.get('pullout_reservations',[])]

    def search(self,a,b,dn,skip=(),clearance=.05,*,priority='bends_then_length',start_direction=None,end_direction=None):
        from .piping_route_search import simple_route
        obstacles=_pg14_planning_obstacles(self,self.obstacles(skip),clearance)
        diagnostic={}
        try:
            return simple_route(a,b,obstacles,self.radius(dn),self.bounds,self.floors,
                                openings=self.openings,clearance=clearance,priority=priority,
                                start_direction=start_direction,end_direction=end_direction,diagnostics=diagnostic)
        finally:self.search_diagnostics.append(diagnostic)

    def result(self):
        return {'phase':'P1','stage':'piping','config':self.config,'requirements':deepcopy(self.requirements),
            'routes':self.routes,'groups':deepcopy(self.requirements['groups']),
            'station':deepcopy(self.plan['station']),'rack_zones':deepcopy(self.plan['rack_zones']),
            'openings':deepcopy(self.openings),'allowed_contacts':self.allowed_contacts,
            'blinds':self.blinds,'reducers':self.reducers,
            'flange_connections':deepcopy(self.scene.get('flange_connections',[])),
            'elbows':deepcopy(self.scene.get('elbows',[])),
            'generation_method':'whole_segment_moving_planes_route_then_native_valves',
            'valve_reservations':deepcopy(self.valve_reservations),
            'pending_valve_reservations':[{'id':v['id'],'package':v['package'],
                'install_length_m':v['install_length_m'],'full_travel_box':v.get('reserved_operation_box')}
                for v in self.planned_valves],
            'route_search_diagnostics':deepcopy(self.search_diagnostics),
            'header_tap_plans':deepcopy(self.header_tap_plans),
            **({'service_groups':deepcopy(self.plan['service_groups'])} if 'service_groups' in self.plan else {}),
            'source_p2_edges':deepcopy(self.requirements['source_p2_edges']),
            'scope':'P1 geometric connectivity. Valve availability is reported separately from fluid and safety performance.'}


def _nearby_contacts(assembly, component_ids, position, skip=()):
    """Record explicit physical neighbour pairs at one constructed junction."""
    p=np.asarray(position);component_ids=list(component_ids)
    target_ids=set()
    for route in assembly.routes:
        if not set(component_ids).intersection(route['components']):continue
        for end in ('source','target'):
            endpoint=route.get(end,{})
            if not isinstance(endpoint,dict) or math.dist(endpoint.get('position',[math.inf]*3),p)>1.e-8:continue
            for key in ('header_id','main_id'):
                peer=next((r for r in assembly.routes if r['id']==endpoint.get(key)),None)
                if peer:target_ids.update(peer['components'])
    for c in assembly.scene['components']:
        if c['id'] not in target_ids or c['id'] in set(component_ids)|set(skip):continue
        b=np.asarray(c['box']);gap=np.maximum(np.maximum(b[:3]-p,p-b[3:]),0.)
        if np.linalg.norm(gap)<1.e-7:
            for cid in component_ids:assembly.allow(cid,c['id'],p)


def _native_contacts(assembly, valve, pipe_id, endpoint):
    # Explicit name pairs remain subject to the checker's local endpoint test.
    for geom in valve['world_geoms']:
        assembly.allow(pipe_id,valve['geom_prefix']+geom['name'],endpoint)


def _pipe_to_valve(assembly,name,a,v,side,dn,medium,role,level,**metadata):
    face=v['world_ports'][side]['position']
    r=assembly.route(name,[a,face],dn,medium,role,level,**metadata)
    if r['components']:_native_contacts(assembly,v,r['components'][-1],face)
    r['valves']=[v['id']];return r


def _prepare_header_taps(a,h,r):
    """Coordinate every device/main tee before any branch is routed."""
    from .piping_header_taps import allocate_header_taps
    names={h['id'],h['id']+'_start'}
    fittings=[c for c in a.scene['components'] if c.get('owner') in names
              and c['kind'] in ('piping_flange','piping_blind')]
    dn=h['dn'] if h['medium']=='offgas' else r['branch_dn']
    length=0. if h['medium']=='offgas' else inspect_asset(dn)['install_length_m']
    slots={p['id']:{'dn':p['dn'],'install_length_m':inspect_asset(p['dn'])['install_length_m']}
           for p in a.requirements['service_points'] if p['medium']==h['medium'] and p['level']==h['level']}
    result=allocate_header_taps(a.frozen,a.requirements,h,
        main_source={'position':[*r['xy'],h['start'][2]],'dn':dn,'install_length_m':length},
        root_slots=slots,end_fittings=fittings,max_candidates=24)
    if result['status']!='ready':
        exc=PipingRefused('Floor header joint tap assignment rejected: '+str(result.get('refusal')))
        exc.evidence=result;raise exc
    a.header_tap_options[h['id']]=result['candidates']
    a.header_tap_plans[h['id']]=deepcopy(result['candidates'][0])


def _header_body_link(a,h,header,main,r,*,valve=None,spool=None):
    """Finite body-T choices; every whole guide precedes native attachment.

    The last/first short tee arm alone may meet its actual through-cylinder.
    All caps, flanges and unrelated geometry stay protected in that search.
    """
    from .piping_elbows import coarse_radius,plan_polyline
    axis=_unit(np.asarray(h['end'])-h['start'])
    cross=np.cross(axis,[0.,0.,1.]);vertical=np.array([0.,0.,1.])
    dn=h['dn'] if valve is None else valve['dn'];radius=a.radius(dn);R=coarse_radius(radius)
    stub=R+a.radius(h['dn'])+radius+_pg14_limits(a)['bare']
    directions=(vertical,-vertical,cross,-cross)
    failures=[]
    for candidate in a.header_tap_options[h['id']]:
        join=np.asarray(candidate['main_tap']['position'])
        for outward in directions:
            approach=join+outward*stub
            contacts=_local_header_contacts(a,header['components'],join,approach)
            try:
                _root_search(a,join,approach,dn,contacts,current=valve['id'] if valve else None,
                             clearance=.03,straight=True)
            except (AssetUnavailable,CandidateRefused,RouteUnavailable) as exc:
                failures.append(_root_error(exc));continue
            if valve is not None:
                try:
                    path=_root_search(a,spool,approach,dn,current=valve['id'],
                        start_direction=np.asarray(valve['axis']),end_direction=-outward)
                    points=[np.asarray(valve['world_ports']['out']['position']),spool]+path[1:]+[join]
                    plan_polyline(points,radius)
                except (AssetUnavailable,CandidateRefused,RouteUnavailable) as exc:
                    failures.append(_root_error(exc));continue
                a.header_tap_plans[h['id']]=deepcopy(candidate)
                return points,join,None
            # PG15 collector-to-main stays overhead and has no root valve.
            # Lift options are derived from actual stub/R dimensions; they are
            # planning candidates, not a changed headroom or clearance rule.
            main_stub=R+a.radius(r['dn'])+radius+_pg14_limits(a)['bare']
            for lift in (0.,stub,2*stub+R):
                target=np.array([*r['xy'],join[2]+lift])
                if target[2]+radius>a.plan['rack_zones'][h['level']]['box'][5]:continue
                for main_out in (np.asarray(a.plan['station']['inward']),
                                 np.asarray(a.plan['station']['tangent']),
                                 -np.asarray(a.plan['station']['tangent'])):
                    before=target+main_out*main_stub
                    main_contacts=_local_header_contacts(a,main['components'],target,before)
                    try:
                        _root_search(a,before,target,dn,main_contacts,clearance=.03,straight=True)
                        path=_root_search(a,approach,before,dn,clearance=.03,
                                          start_direction=outward,end_direction=-main_out)
                        points=[join,approach]+path[1:]+[target]
                        plan_polyline(points,radius)
                    except (AssetUnavailable,CandidateRefused,RouteUnavailable) as exc:
                        failures.append(_root_error(exc));continue
                    a.header_tap_plans[h['id']]=deepcopy(candidate)
                    return points,join,target
    exc=PipingRefused('No candidate tee along the floor header passed: '+h['id']+'；'+str(failures[-4:]))
    exc.evidence={'code':'HEADER_BODY_LINK_UNAVAILABLE','header':h['id'],
        'joint_tap_candidates':len(a.header_tap_options[h['id']]),'failed_guides':len(failures),
        'last_failures':failures[-4:],'physical_unsatisfiable_claim':False}
    raise exc


def build_skeleton(assembly):
    """Build continuous mains, real per-floor outlets and common rack headers."""
    a=assembly;station=a.plan['station'];inward=np.asarray(station['inward'])
    mains={}
    for r in station['risers']:
        route=a.route(r['id'],[[*r['xy'],r['bottom_z']],[*r['xy'],r['top_z']]],
            r['dn'],r['medium'],'workstation_main',0,
            source={'position':[*r['xy'],r['bottom_z']],'boundary':True},
            target={'position':[*r['xy'],r['top_z']],'blind':True},served_levels=r['served_levels'])
        cap=a.blind(r['id'],[*r['xy'],r['top_z']],[0,0,1],r['dn'],r['medium'],len(a.floors)-1,'main_end')
        a.flange_interface(r['id'],[*r['xy'],r['top_z']-.039],[0,0,1],r['dn'],r['medium'],len(a.floors)-1)
        a.allow(route['components'][-1],cap['id'],[*r['xy'],r['top_z']]);mains[r['medium']]=route
    headers={}
    for h in a.plan['headers']:
        if not h['active']:continue
        route=a.route(h['id'],[h['start'],h['end']],h['dn'],h['medium'],'floor_header',h['level'],
            source={'position':h['start'],'blind':True},
            target={'position':h['end'],'blind':True},
            main_id='P1_main_'+h['medium'],flow_toward_station=h['flow_toward_station'],
            service_group=h.get('service_group'),point_ids=deepcopy(h.get('point_ids',[])),
            axis_index=h.get('axis_index'),service_group_evidence=deepcopy(h.get('service_group_evidence')))
        direction=_unit(np.asarray(h['end'])-h['start'])
        cap=a.blind(h['id'],h['end'],direction,h['dn'],h['medium'],h['level'],'header_end')
        a.flange_interface(h['id'],np.asarray(h['end'])-direction*.039,direction,h['dn'],h['medium'],h['level'])
        a.allow(route['components'][-1],cap['id'],h['end'])
        start_cap=a.blind(h['id']+'_start',h['start'],-direction,h['dn'],h['medium'],h['level'],'header_end')
        a.flange_interface(h['id']+'_start',np.asarray(h['start'])+direction*.039,-direction,h['dn'],h['medium'],h['level'])
        a.allow(route['components'][0],start_cap['id'],h['start'])
        headers[h['id']]=route
        _prepare_header_taps(a,h,next(r for r in station['risers'] if r['medium']==h['medium']))
    for mi,r in enumerate(station['risers']):
        medium=r['medium'];main=mains[medium]
        for level,base in enumerate(a.floors):
            # Stagger neighbouring short outlets by actual valve face length.
            temp_dn=r['temporary_dn'];asset=inspect_asset(temp_dn);length=asset['install_length_m']
            z=base+.8+mi*(max(length,.1)+.12)
            if z+length>=a.plan['rack_zones'][level]['box'][2]:
                raise PipingRefused('Temporary station valve does not fit the floor clearance')
            start=np.array([*r['xy'],z]);face=start+inward*(a.radius(r['dn'])+length)
            v=a.plan_valve(f'P1_temp_{medium}_L{level}',temp_dn,face,inward,[0,0,1],medium,'temporary_outlet',level,
                      edge=f'P1_temp_{medium}_L{level}',equipment=None,group_id=None,at=f'P1_main_{medium}.temporary_{level}')
            v.update(root_axial_envelope=_root_axial_envelope(v),root_pipe_radius_m=a.radius(temp_dn))
            t=a.route(v['id']+'_in',[start,face],temp_dn,medium,'temporary_outlet',level,
                main_id=main['id'],source={'position':start.tolist(),'main_id':main['id']},
                target={'position':face.tolist(),'valve':v['id']})
            t['valves']=[v['id']];_native_contacts(a,v,t['components'][-1],face)
            _nearby_contacts(a,t['components'][:1],start)
            end=np.array(v['world_ports']['out']['position']);cap_pos=_root_spools(v)[3]
            tail=a.route(v['id']+'_out',[end,cap_pos],temp_dn,medium,'temporary_outlet',level,
                main_id=main['id'],source={'position':end.tolist(),'valve':v['id']},target={'position':cap_pos.tolist(),'blind':True})
            tail['valves']=[v['id']];_native_contacts(a,v,tail['components'][0],end)
            cap=a.blind(v['id'],cap_pos,inward,temp_dn,medium,level,'temporary_outlet')
            a.allow(tail['components'][-1],cap['id'],cap_pos)
            a.flange_interface(v['id'],cap_pos-inward*.039,inward,temp_dn,medium,level)
            a.install_valve(v)
            floor_headers=[h for h in a.plan['headers'] if h['medium']==medium and h['level']==level and h['active']]
            if medium=='offgas' and not floor_headers:continue
            if not floor_headers:
                floor_headers=[{'id':f'P1_unused_{medium}_L{level}','medium':medium,'level':level,'active':False,'service_group':None}]
            previous_top=None
            for group_index,h in enumerate(floor_headers):
                previous_top=_build_floor_group_link(a,h,headers,main,r,mi,group_index,previous_top)
    _planned_parallel_check(a)
    return headers


def _build_floor_group_link(a,h,headers,main,r,medium_index,group_index,previous_top):
    """One actual root per active supply group; one blind root on unused floors."""
    medium=h['medium'];level=int(h['level']);base=a.floors[level]
    inward=np.asarray(a.plan['station']['inward']);group=h.get('service_group')
    connection=f'P1_floor_{medium}_L{level}_G{group_index}'
    if medium=='offgas':
        points,join,target=_header_body_link(a,h,headers[h['id']],main,r)
        link=a.route(connection,points,h['dn'],medium,'floor_branch',level,
            main_id=main['id'],distribution_id=h['id'],service_group=group,connection_id=connection,
            source={'position':join.tolist(),'header_id':h['id']},
            target={'position':target.tolist(),'main_id':main['id']},route_first=True)
        _nearby_contacts(a,link['components'][:1],join);_nearby_contacts(a,link['components'][-1:],target)
        return previous_top
    dn=r['branch_dn'];length=inspect_asset(dn)['install_length_m']
    name=f'P1_fixed_{medium}_L{level}_G{group_index}'
    start=np.array([*r['xy'],base+1.4+medium_index*.13])
    def reserve():
        face=start+inward*(a.radius(r['dn'])+length)
        valve=a.plan_valve(name,dn,face,inward,[0,0,1],medium,'utility_floor_root',level,
            edge=name,equipment=None,group_id=None,at=main['id']+f'.floor_{level}',
            distribution_id=h['id'],service_group=group,connection_id=connection,main_id=main['id'])
        return face,valve
    face,v=reserve()
    if previous_top is not None:
        shift=max(0.,previous_top+_pg14_limits(a)['protrusion']-v['reserved_operation_box'][2])
        if shift:start[2]+=shift;face,v=reserve()
    end=np.array(v['world_ports']['out']['position']);spool=_root_spools(v)[3]
    if h['active']:
        spool=end+inward*max(math.dist(end,spool),2*length+.039,
            math.dist(end,_root_spool_points(a,v)['pout_far']))
        tail_points,join,_=_header_body_link(a,h,headers[h['id']],main,r,valve=v,spool=spool)
    else:tail_points=[end,spool]
    identity={'main_id':main['id'],'distribution_id':h['id'],'service_group':group,'connection_id':connection}
    lead=a.route(v['id']+'_in',[start,face],dn,medium,'floor_branch',level,**identity,
        source={'position':start.tolist(),'main_id':main['id']},target={'position':face.tolist(),'valve':v['id']},route_first=True)
    lead['valves']=[v['id']];_native_contacts(a,v,lead['components'][-1],face);_nearby_contacts(a,lead['components'][:1],start)
    if not h['active']:
        tail=a.route(v['id']+'_out',tail_points,dn,medium,'floor_branch',level,**identity,
            source={'position':end.tolist(),'valve':v['id']},target={'position':spool.tolist(),'blind':True},active=False)
        cap=a.blind(v['id'],spool,inward,dn,medium,level,'unused_floor_branch');a.allow(tail['components'][-1],cap['id'],spool)
    else:
        tail=a.route(v['id']+'_out',tail_points,dn,medium,'floor_branch',level,**identity,
            source={'position':end.tolist(),'valve':v['id']},target={'position':join.tolist(),'header_id':h['id']},active=True)
        _nearby_contacts(a,tail['components'][-1:],join)
    tail['valves']=[v['id']];_native_contacts(a,v,tail['components'][0],end)
    a.flange_interface(v['id'],spool-inward*(length+.039 if h['active'] else .039),inward,dn,medium,level)
    v['route_first']=True;a.install_valve(v)
    return v['reserved_operation_box'][5]


def _operation_side(frozen,point):
    from .operation_reservations import lower_hint
    return lower_hint(frozen['scene'],point['equipment'],point['port'])


def _root_axial_envelope(valve):
    """Read-only native full-travel projection on all three original axes."""
    import mujoco
    cache=getattr(_root_axial_envelope,'cache',{})
    key=(valve['package'],valve['version'])
    if key not in cache:
        model=mujoco.MjModel.from_xml_path(valve['model_xml']);data=mujoco.MjData(model)
        mujoco.mj_kinematics(model,data)
        root=model.body(valve['root_body']).id;origin=data.xpos[root].copy()
        basis=data.xmat[root].reshape(3,3).copy()
        opened=valve['motion']['joint_states']['open'];closed=valve['motion']['joint_states']['closed']
        ids={name:int(model.joint(name).qposadr[0]) for name in opened}
        samples=int(math.ceil(max(abs(closed[n]-opened[n]) for n in opened)/math.radians(5)))+1
        moving=[model.geom(g['name']).id for g in valve['local_geoms'] if g['moving']]
        def projection(ids):
            lows=[];highs=[]
            for gid in ids:
                rotation=basis.T@data.geom_xmat[gid].reshape(3,3)
                center=basis.T@(data.geom_xpos[gid]-origin);size=model.geom_size[gid]
                kind=int(model.geom_type[gid])
                if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
                    mesh=int(model.geom_dataid[gid]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
                    points=np.asarray(model.mesh_vert[start:start+count],float)@rotation.T+center
                    lows.append(points.min(axis=0));highs.append(points.max(axis=0));continue
                if kind==int(mujoco.mjtGeom.mjGEOM_BOX):extent=abs(rotation)@size
                elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):extent=abs(rotation[:,2])*size[1]+size[0]*np.sqrt(np.maximum(0.,1-rotation[:,2]**2))
                elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):extent=abs(rotation[:,2])*size[1]+size[0]
                elif kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):extent=np.full(3,size[0])
                elif kind==int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):extent=np.linalg.norm(rotation*size,axis=1)
                else:raise AssetUnavailable('Package contains a geom type whose projection cannot be computed: '+str(kind))
                lows.append(center-extent);highs.append(center+extent)
            return np.min(lows,axis=0),np.max(highs,axis=0)
        low,high=projection(range(model.ngeom))
        for fraction in np.linspace(0.,1.,samples):
            for name,address in ids.items():data.qpos[address]=opened[name]+fraction*(closed[name]-opened[name])
            mujoco.mj_kinematics(model,data);one,two=projection(moving)
            low=np.minimum(low,one);high=np.maximum(high,two)
        cache[key]={'min_pipe_axis_m':float(low[0]),'max_pipe_axis_m':float(high[0]),
                    'min_original_axes_m':low.tolist(),'max_original_axes_m':high.tolist(),
                    'samples':samples,'physics_steps':0,
                    'scope':'Full native geometry, sampling travel at up to five degrees. Use actual mesh vertices and exact primitive projections, including zero contact masks.'}
        _root_axial_envelope.cache=cache
    return deepcopy(cache[key])


def _root_spools(valve):
    """Retain at least original L, then escape the actual full-travel outline."""
    axis=np.asarray(valve['axis']);length=valve['install_length_m']
    inlet=np.asarray(valve['world_ports']['in']['position'])
    outlet=np.asarray(valve['world_ports']['out']['position'])
    envelope=valve['root_axial_envelope'];pad=valve['root_pipe_radius_m']+.05
    local_in=valve['local_ports']['in']['position'][0];local_out=valve['local_ports']['out']['position'][0]
    extension_in=max(length,local_in-envelope['min_pipe_axis_m']+pad)
    extension_out=max(length,envelope['max_pipe_axis_m']-local_out+pad)
    return inlet-axis*extension_in,inlet,outlet,outlet+axis*extension_out


def _root_error(exc):
    evidence=getattr(exc,'evidence',None)
    return str(exc)+(('. Routing evidence: '+str(evidence)) if evidence else '')


def _root_native_spool_clear(a,valve,first,second):
    """No self-owner exemption for the actual axial pipe outside its face."""
    direction=_unit(np.asarray(second)-first);radius=a.radius(valve['dn'])
    extent=radius*np.sqrt(np.maximum(0.,1-direction**2))
    box=np.r_[np.minimum(first,second)-extent,np.maximum(first,second)+extent].tolist()
    from .root_orientation import planning_geoms
    hits=[valve['prefix']+g['name'] for g in planning_geoms(valve) if _overlap(box,g['box'])]
    if hits:raise PipingRefused('An axial straight penetrates a physical valve part: '+','.join(hits))


def _root_pipe_obstacle(name,first,second,radius):
    first=np.asarray(first);second=np.asarray(second);axis=_unit(second-first)
    extent=radius*np.sqrt(np.maximum(0.,1-axis**2))
    return {'id':name,'kind':'pipe','shape':'cylinder','ends':[first.tolist(),second.tolist()],
            'radius':radius,'box':np.r_[np.minimum(first,second)-extent,np.maximum(first,second)+extent].tolist(),
            'scope':'Plan fixed end sections using candidate pipe diameters and real obstacles'}


def _root_spool_points(a,valve):
    from .piping_elbows import coarse_radius
    pin,face_in,face_out,pout=_root_spools(valve);axis=np.asarray(valve['axis']);radius=a.radius(valve['dn'])
    gap=_pg14_limits(a)['bare']
    return {'pin':pin,'in':face_in,'out':face_out,'pout':pout,
            'pin_far':pin-axis*(coarse_radius(radius)+gap),'pout_far':pout+axis*(coarse_radius(radius)+gap)}


def _root_source_points(a,context):
    from .piping_elbows import coarse_radius
    point=context['point'];source=np.asarray(point['position']);normal=_unit(point['normal']);radius=a.radius(point['dn'])
    red_end=source+normal*(2*max(a.radius(point['source_dn']),radius))
    departure=red_end+normal*max(coarse_radius(radius),radius+_pg14_limits(a)['protrusion'])
    # The real reducer's expanded end face already protects this normal lead.
    # Adding another capsule-end offset would consume valid space below a
    # pullout reservation without increasing the actual pipe clearance rule.
    return source,red_end,departure,departure


def _root_source_obstacles(a,context):
    point=context['point'];name='P1_root_'+point['id'];source,red_end,departure,far=_root_source_points(a,context)
    return [_root_pipe_obstacle(name+'_reserved_source_adapter',source,red_end,max(context['nozzle']['radius'],a.radius(point['dn']))),
            _root_pipe_obstacle(name+'_reserved_source_departure',red_end,far,a.radius(point['dn']))]


def _root_obstacles(a,skip_components=(),skip_valves=(),current=None,reserve=True):
    """All real solids; native valve parts replace their coarse union boxes."""
    skip_components=set(skip_components);skip_valves=set(skip_valves)
    result=[c for c in a.scene['components'] if c['id'] not in skip_components]
    for valve in a.all_valves():
        if valve['id'] not in skip_valves:
            result.extend({'id':valve['prefix']+g['name'],'kind':'valve','shape':'box',
                           'box':g['box']} for g in valve['world_geoms'])
            if valve.get('reserved_operation_box'):
                result.append({'id':valve['id']+'_reserved_operation','kind':'valve','shape':'box',
                    'box':valve['reserved_operation_box']})
            if valve.get('reserved_stance_box'):
                result.append({'id':valve['id']+'_reserved_standing','kind':'operation_stance','shape':'box',
                    'box':valve['reserved_stance_box']})
        if reserve and valve.get('role')=='equipment_root' and valve['id']!=current:
            points=_root_spool_points(a,valve)
            for side,first,second in (('in',points['pin_far'],points['in']),('out',points['out'],points['pout_far'])):
                result.append(_root_pipe_obstacle(valve['id']+'_reserved_'+side,first,second,a.radius(valve['dn'])))
    result.extend({'id':r['id'],'kind':'pullout_reservation','shape':'box',
        'box':[r['rect'][0],r['rect'][1],r['base_z'],r['rect'][2],r['rect'][3],a.bounds[5]]}
        for r in a.scene.get('pullout_reservations',[]))
    if reserve:
        for context in getattr(a,'_root_connection_contexts',[]):
            point=context['point'];name='P1_root_'+point['id']
            if context.get('connected') or name==current:continue
            result.extend(_root_source_obstacles(a,context))
            first=_root_header_approaches(a,context)[0];second=context['join']
            from .piping_elbows import coarse_radius
            radius=a.radius(point['dn']);direction=_unit(second-first)
            result.append(_root_pipe_obstacle(name+'_reserved_header_approach',
                first-direction*(coarse_radius(radius)+_pg14_limits(a)['bare']),second,radius))
    return result


def _root_search(a,first,second,dn,skip_components=(),skip_valves=(),current=None,clearance=.05,straight=False,extra_obstacles=(),
                 priority='bends_then_length',start_direction=None,end_direction=None):
    from .piping_route_search import simple_route
    obstacles=_pg14_planning_obstacles(a,_root_obstacles(a,skip_components,skip_valves,current)+list(extra_obstacles),clearance)
    diagnostic={}
    try:
        points=simple_route(first,second,obstacles,
            a.radius(dn),a.bounds,a.floors,openings=a.openings,clearance=clearance,straight_only=straight,
            priority=priority,start_direction=start_direction,end_direction=end_direction,diagnostics=diagnostic)
    finally:a.search_diagnostics.append(diagnostic)
    if straight and len(points)>2:
        raise PipingRefused('A solid blocks the axial root-valve pipe or specified merge straight: '+str([np.asarray(first).tolist(),np.asarray(second).tolist()]))
    return points


def _local_header_contacts(a,component_ids,join,approach):
    """Only the single actual non-collinear tee at this endpoint may attach.

    A segmented header is not removed as a whole route. Every unrelated
    component, cap and flange stays in this short straight segment's search.
    The emitted joint is still independently checked against real XML.
    """
    names=set(component_ids);point=np.asarray(join);incoming=_unit(point-approach);result=[]
    for c in a.scene['components']:
        if c['id'] not in names or c.get('shape')!='cylinder':continue
        first,second=map(np.asarray,c['ends']);axis=_unit(second-first)
        projection=float((point-first)@axis);length=math.dist(first,second)
        if (projection < -1.e-8 or projection > length+1.e-8
            or np.linalg.norm(point-first-projection*axis)>1.e-8
            or np.linalg.norm(np.cross(axis,incoming))<=1.e-8):continue
        result.append(c['id'])
    return result


def _root_candidates(a,point,ordinal,sides):
    from .process import RULES
    directions={'x_min':[-1,0,0],'x_max':[1,0,0],'y_min':[0,-1,0],'y_max':[0,1,0]}
    asset=inspect_asset(point['dn']);length=asset['install_length_m']
    normal=_unit(point['normal']);source=np.asarray(point['position'])
    op=_operation_side(a.frozen,point);identifier='P1_root_'+point['id']
    for number,(side,offset,rise) in enumerate(itertools.product(sides,(.10,.25,.40),(.03,.08,.14)),1):
        outward=np.asarray(directions[side],dtype=float);tangent=np.array([-outward[1],outward[0],0.])
        axis=np.array([0.,0.,1.]) if abs(normal[2])>.5 else normal
        stem=outward if abs(np.dot(outward,axis))<.1 else np.array([0.,0.,1.])
        initial=place_asset(asset,[0,0,0],axis,stem)
        envelope=_root_axial_envelope(initial)
        local_tangent=np.asarray(initial['matrix']).T@tangent
        width=float(abs(local_tangent)@(np.asarray(envelope['max_original_axes_m'])-envelope['min_original_axes_m']))
        operation_gap=RULES['geometry_screen']['sweep_margin_m']
        pipe_gap=a.config['pg_rules']['parallel_pipe_spacing']['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000
        protrusion_gap=_pg14_limits(a)['protrusion']
        pitch=max(width+max(operation_gap,protrusion_gap),2*a.radius(point['dn'])+pipe_gap)
        target=np.asarray(op['position']) if op else source+outward*(length+.20)
        target=target+outward*offset+tangent*((ordinal-1)*pitch)
        target[2]=source[2]+rise
        center=target-np.asarray(initial['world_ports']['in']['position'])
        valve=place_asset(asset,center,axis,stem)
        valve.update(id=identifier,name=identifier,role='equipment_root',dn=int(point['dn']),
            level=int(point['level']),medium=point['medium'],prefix=identifier+'_',geom_prefix=identifier+'_',
            edge=point['id'],source_requirement=point['id'],equipment=point['equipment'],
            group_id=point['group_id'],at=point['endpoint'],operating_side=side,operation_side=side,
            root_candidate=number,root_candidate_rule='Try valve offsets .10, .25 and .40 and rises .03, .08 and .14 on the fixed operating side. Reserve actual dimensions before adding the connected assembly.')
        valve['root_axial_envelope']=_root_axial_envelope(valve)
        valve['root_pipe_radius_m']=a.radius(point['dn'])
        if op and op.get('stance_box'):valve['reserved_stance_box']=deepcopy(op['stance_box'])
        valve['root_row_spacing']={'tangent_world':tangent.tolist(),'tangent_local':local_tangent.tolist(),
            'full_travel_projection_width_m':width,'operation_gap_m':operation_gap,'bare_pipe_gap_m':pipe_gap,
            'protrusion_gap_m':protrusion_gap,
            'pitch_m':pitch,'scope':'Actual tangential projection in the current pose. Use the conservative three-axis envelope for arbitrary directions.'}
        yield valve


def _root_fits(a,valve,nozzle):
    from .piping_route_search import _box_hits
    obstacles=_pg14_planning_obstacles(a,_root_obstacles(a,(nozzle['id'],),(valve['id'],),current=valve['id']),
        0.,moving_protrusion=True)
    broad=[c for c in obstacles if _overlap(valve['world_bbox'],c['box'])]
    hits=[]
    for component in broad:
        for geom in valve['world_geoms']:
            if not _overlap(geom['box'],component['box']):continue
            if component.get('shape') in ('cylinder','capsule'):
                first,second=map(np.asarray,component['ends'])
                if not _box_hits(first,second,np.asarray(geom['box'])[None,:],component['radius'])[0]:continue
            hits.append(component['id']);break
    if not hits:hits.extend(_root_fixed_fits(a,valve))
    return hits


def _root_pipe_conflict(pipe,component):
    from .piping_route_search import _box_hits,_segments_distance
    if not _overlap(pipe['box'],component['box']):return False
    first,second=map(np.asarray,pipe['ends'])
    if component.get('shape') in ('cylinder','capsule'):
        one,two=map(np.asarray,component['ends'])
        return _segments_distance(first,second,one,two)<pipe['radius']+component['radius']-1.e-5
    return bool(_box_hits(first,second,np.asarray(component['box'])[None,:],pipe['radius'])[0])


def _root_fixed_fits(a,valve):
    """Preplace real fixed spools as well as the native valve body."""
    points=_root_spool_points(a,valve);name=valve['id'];radius=a.radius(valve['dn'])
    obstacles=_root_obstacles(a,skip_valves=(name,),current=name)
    own=next((c for c in getattr(a,'_root_connection_contexts',[]) if 'P1_root_'+c['point']['id']==name),None)
    if own:obstacles+=_root_source_obstacles(a,own)
    obstacles=_pg14_planning_obstacles(a,obstacles,0.)
    hits=[]
    for side,first,second in (('in',points['pin_far'],points['in']),('out',points['out'],points['pout_far'])):
        try:_root_native_spool_clear(a,valve,first,second)
        except CandidateRefused as exc:
            hits.append(str(exc));continue
        pipe=_root_pipe_obstacle(name+'_proposed_'+side,first,second,radius)
        hits.extend(component['id'] for component in obstacles if _root_pipe_conflict(pipe,component))
    return sorted(set(hits))


def _offgas_vertical_dimensions(a,point,plan):
    from .piping_elbows import coarse_radius,BASIS
    radius=a.radius(point['dn']);header_radius=a.radius(plan['dn']);R=coarse_radius(radius)
    bare=_pg14_limits(a)['bare'];drop=header_radius+radius+R+bare
    return {'header_outer_radius_m':header_radius,'branch_outer_radius_m':radius,
        'centerline_radius_m':R,'bare_clearance_m':bare,
        'protrusion_clearance_m':_pg14_limits(a)['protrusion'],
        'vertical_approach_m':drop,'basis':dict(BASIS),
        'formula':'h=actual_headerR+actual_branchr+R90+bare_gap; R90=3r',
        'scope':'Axial approach above the floor collector, with a final downward Z connection'}


def _root_header_approaches(a,context):
    point=context['point'];join=context['join'];plan=context['plan']
    span=max(.20,a.radius(point['dn'])*4)
    if point['medium']=='offgas':
        calculation=_offgas_vertical_dimensions(a,point,plan)
        context['offgas_vertical_dimensions']=calculation
        return [join+np.array([0.,0.,calculation['vertical_approach_m']])]
    along=_unit(np.asarray(plan['end'])-plan['start']);cross=np.cross(along,[0.,0.,1.])
    return [join+direction*span for direction in (np.array([0.,0.,-1.]),np.array([0.,0.,1.]),cross,-cross)]


def _root_allocate_header_taps(a,contexts):
    """Use the already jointly chosen main/device table, never reclamp alone."""
    for context in contexts:
        header=context['header']['id'];point=context['point']
        chosen=a.header_tap_plans.get(header)
        matches=[] if chosen is None else [t for t in chosen['device_taps'] if t['requirement']==point['id']]
        if len(matches)!=1:raise PipingRefused('Joint assignment lacks a unique equipment tap: '+point['id'])
        tap=matches[0]
        if abs(tap['slot_install_length']-inspect_asset(point['dn'])['install_length_m'])>1.e-8:
            raise PipingRefused('Joint assignment valve length differs from the native package: '+point['id'])
        context['join']=np.asarray(tap['position'])
        context['tap_selection']={**deepcopy(tap),'header_id':header,
            'main_tap':deepcopy(chosen['main_tap']),'formula':deepcopy(chosen['formula']),
            'scope':'Allocate main and equipment tees together in canonical outlet-face order. Verify the actual XML independently.'}


def _root_contexts(a,headers):
    ordinals={};contexts=[];groups={g['id']:g for g in a.requirements['groups']}
    for point in a.requirements['service_points']:
        owner=point['equipment'];medium=point['medium'];level=int(point['level'])
        nozzle=next((c for c in a.scene['components'] if c.get('owner')==owner and
                    c['id']==owner+'_'+point['port']+'_equipment_nozzle'),None)
        if nozzle is None or nozzle.get('shape')!='cylinder' or not nozzle.get('radius'):
            raise PipingRefused('Saved port lacks the physical cylindrical neck or measured radius: '+point['endpoint'])
        matched=[h for h in a.plan['headers'] if h['active'] and h['medium']==medium
                 and h['level']==level and point['id'] in h.get('point_ids',[])]
        if len(matched)!=1:raise PipingRefused('Source service edge lacks a unique same-floor group for its use: '+point['id'])
        plan=matched[0];header=headers[plan['id']]
        axis=_unit(np.asarray(plan['end'])-plan['start']);source=np.asarray(point['position'])
        span=math.dist(plan['start'],plan['end']);along=float((source-np.asarray(plan['start']))@axis)
        along=min(span-.08,max(.08,along));join=np.asarray(plan['start'])+axis*along
        op=_operation_side(a.frozen,point);group=groups.get(point.get('group_id'),{})
        declared=op.get('side') if op else None
        preference=declared or group.get('operating_side')
        sides=sorted(('x_min','x_max','y_min','y_max'),key=lambda side:side!=preference)
        ordinal=ordinals.get(owner,0);ordinals[owner]=ordinal+1
        contexts.append({'point':point,'nozzle':nozzle,'header':header,'plan':plan,'join':join,
                         'candidates':list(_root_candidates(a,point,ordinal,sides))})
    _root_allocate_header_taps(a,contexts)
    return contexts


def _root_connection_self_check(a,routes):
    """Compile only these emitted real pieces and reuse the unchanged ruler.

    This is a bounded connection-local precheck, not a new room checker or
    owner exemption. Normal elbows use the existing verified endpoint pairs;
    non-neighbour self-crossings and reverse overlaps remain errors.
    """
    import xml.etree.ElementTree as ET
    import mujoco
    from .export import nums
    from .piping_checks import _measured,_route_segments,_geometry
    names={name for route in routes for name in route['components']}
    parts=[c for c in a.scene['components'] if c['id'] in names]
    root=ET.Element('mujoco',model='connection_geometry_precheck');ET.SubElement(root,'compiler',angle='radian')
    assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    for component in parts:
        kwargs={'name':component['id'],'type':component['shape']}
        if component['shape']=='cylinder':
            kwargs.update(fromto=nums(component['ends'][0]+component['ends'][1]),size=str(component['radius']))
        elif component['shape']=='mesh':
            mesh='mesh_'+component['id'];ET.SubElement(assets,'mesh',name=mesh,file=str(Path(component['mesh_path']).resolve()))
            kwargs.update(mesh=mesh,pos=nums(component.get('pos',[0,0,0])))
            if component.get('quat'):kwargs['quat']=nums(component['quat'])
        else:raise ValueError('Connection self-check does not support the actual part shape: '+component['shape'])
        ET.SubElement(world,'geom',**kwargs)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data);actual=_measured(model,data)
    scene={'components':parts,'elbows':[deepcopy(e) for e in a.scene.get('elbows',[]) if set(e['components'])<=names]}
    components={c['id']:c for c in parts}
    by_route,errors,_=_route_segments(routes,components,actual,a.od,model,data)
    if errors:raise ValueError('Actual dimensions of the constructed connection failed recheck: '+str(errors))
    checked=_geometry(a.frozen,scene,{'allowed_contacts':a.allowed_contacts,'elbows':scene['elbows']},model,data,actual,{},by_route,[])
    if checked['errors']:raise PipingRefused('Physical sections of the completed connection intersect: '+str(checked['errors']))
    return {'status':'pass','method':'Compile emitted parts in memory and use the P1 geometry and local termination checks',
            'component_count':len(parts),'physics_steps':0}


def _root_connect_one(a,context,valve):
    """Materialise and check one connection; caller rolls back failed trials."""
    from .piping_route_search import RouteUnavailable
    point=context['point'];nozzle=context['nozzle'];plan=context['plan'];join=context['join']
    dn=point['dn'];medium=point['medium'];level=int(point['level']);name=valve['id']
    source=np.asarray(point['position']);normal=_unit(point['normal'])
    points=_root_spool_points(a,valve)
    pin,face_in,face_out,pout=[points[key] for key in ('pin','in','out','pout')]
    pin_far,pout_far=points['pin_far'],points['pout_far']
    axis=np.asarray(valve['axis']);length=valve['install_length_m']
    nominal_in=face_in-axis*length;nominal_out=face_out+axis*length
    _root_native_spool_clear(a,valve,pin_far,face_in)
    _root_native_spool_clear(a,valve,face_out,pout_far)
    radius=a.radius(dn)
    approaches=_root_header_approaches(a,context)
    failures=[]
    for before in approaches:
        saved=(len(a.scene['components']),len(a.routes),len(a.allowed_contacts),len(a.reducers),len(a.scene.get('elbows',[])),len(a.graph['valves']),len(a.valve_reservations))
        try:
            # Plan the original reducer size first; no native entity or pipe
            # is emitted until the entire lead/slot/tail has a complete guide.
            red_end=source+normal*(2*max(a.radius(point['source_dn']),radius))
            from .piping_elbows import coarse_radius
            departure=red_end+normal*max(coarse_radius(radius),radius+_pg14_limits(a)['protrusion']);source_far=departure
            direction=_unit(join-before)
            approach_far=before-direction*(coarse_radius(radius)+_pg14_limits(a)['bare'])
            final_direction=direction
            fixed=[_root_pipe_obstacle(name+'_current_in',pin,face_in,radius),
                   _root_pipe_obstacle(name+'_current_out',face_out,pout,radius),
                   _root_pipe_obstacle(name+'_current_header',before,join,radius)]
            _root_search(a,red_end,source_far,dn,current=name,clearance=.01,straight=True)
            lead_path=_root_search(a,source_far,pin_far,dn,current=name,clearance=.01,extra_obstacles=fixed,end_direction=axis)
            _root_search(a,pin_far,face_in,dn,skip_valves=(name,),current=name,clearance=.01,straight=True)
            lead_points=[red_end.tolist(),departure.tolist(),source_far.tolist()]+lead_path[1:]+[pin.tolist(),nominal_in.tolist(),face_in.tolist()]
            _root_search(a,face_out,pout_far,dn,skip_valves=(name,),current=name,clearance=.01,straight=True)
            lead_obstacles=[_root_pipe_obstacle(name+'_planned_lead_'+str(i),first,second,radius)
                for i,(first,second) in enumerate(zip(lead_points,lead_points[1:])) if math.dist(first,second)>1.e-8]
            adapter_obstacles=[_root_source_obstacles(a,context)[0]]
            route_out=_root_search(a,pout_far,approach_far,dn,current=name,extra_obstacles=fixed+lead_obstacles+adapter_obstacles,
                start_direction=axis,end_direction=final_direction)
            tail_points=[face_out.tolist(),nominal_out.tolist(),pout.tolist(),pout_far.tolist()]+route_out[1:]+[before.tolist(),join.tolist()]
            _root_search(a,approach_far,join,dn,_local_header_contacts(a,context['header']['components'],join,before),
                current=name,clearance=.01,straight=True)
            material_end,red=a.reducer(name+'_reducer',source,normal,point['source_dn'],dn,medium,nozzle['radius'])
            if math.dist(material_end,red_end)>1.e-8:raise ValueError('Reserved and measured reducer dimensions disagree')
            lead=a.route(name+'_from_equipment',lead_points,dn,medium,'equipment_connection',level,
                source_requirement=point['id'],distribution_id=plan['id'],main_id='P1_main_'+medium,
                service_group=plan.get('service_group'),
                source={'position':source.tolist(),'endpoint':point['endpoint']},target={'position':face_in.tolist(),'valve':name},
                reducer=red['id'],equipment=point['equipment'],group_id=point['group_id'],
                root_axial_spool={'from':pin_far.tolist(),'to':face_in.tolist(),'pin':pin.tolist(),
                    'length_m':math.dist(pin_far,face_in),'minimum_original_install_length_m':length,
                    'original_L_point':nominal_in.tolist(),'full_travel_envelope':valve['root_axial_envelope']},
                source_departure={'original_port':source.tolist(),'normal':normal.tolist(),'adapter_end':red_end.tolist(),
                    'normal_straight_end':departure.tolist(),'far_start':source_far.tolist(),'original_nozzle_always_obstacle':True})
            lead['components'].insert(0,red['id']);lead['valves']=[name]
            a.allow(nozzle['id'],red['id'],source);a.allow(red['id'],lead['components'][1],red_end)
            _native_contacts(a,valve,lead['components'][-1],face_in)
            tail=a.route(name+'_to_header',tail_points,dn,medium,'equipment_connection',level,
                source_requirement=point['id'],distribution_id=plan['id'],main_id='P1_main_'+medium,
                service_group=plan.get('service_group'),
                source={'position':face_out.tolist(),'valve':name},target={'position':join.tolist(),'header_id':plan['id']},
                equipment=point['equipment'],group_id=point['group_id'],
                root_axial_spool={'from':face_out.tolist(),'to':pout_far.tolist(),'pout':pout.tolist(),
                    'length_m':math.dist(face_out,pout_far),'minimum_original_install_length_m':length,
                    'original_L_point':nominal_out.tolist(),'full_travel_envelope':valve['root_axial_envelope']},
                header_tap_selection=context['tap_selection'],header_approach_far=approach_far.tolist(),
                offgas_join=({'point':join.tolist(),'approach':before.tolist(),'header_id':plan['id'],
                    'level':level,'equipment':point['equipment'],'direction':[0.,0.,-1.],
                    'construction_basis':deepcopy(context['offgas_vertical_dimensions'])}
                            if medium=='offgas' else None))
            tail['valves']=[name];_native_contacts(a,valve,tail['components'][0],face_out)
            _nearby_contacts(a,tail['components'][-1:],join)
            valve['route_first']=True
            valve['whole_connection_plan']={'lead_points':lead_points,'tail_points':tail_points,
                'valve_gap_ports':[face_in.tolist(),face_out.tolist()],
                'scope':'Reserve the real slot and add the native valve after both leads are complete'}
            a.install_valve(valve)
            checked=_root_connection_self_check(a,[lead,tail])
            checked['parallel_planning']=_planned_parallel_check(a,[lead,tail])
            lead['connection_self_geometry']=checked;tail['connection_self_geometry']=deepcopy(checked)
            return
        except (CandidateRefused,AssetUnavailable,RouteUnavailable) as exc:
            for rows,size in zip((a.scene['components'],a.routes,a.allowed_contacts,a.reducers,a.scene.setdefault('elbows',[]),a.graph['valves'],a.valve_reservations),saved):del rows[size:]
            a.reserve_valve(valve)
            failures.append(_root_error(exc))
    raise PipingRefused('No complete connection candidate has a usable approach: '+'；'.join(failures[:2]))


def connect_equipment(assembly,headers):
    """Reserve real slots, plan whole connections, then attach native packages."""
    a=assembly;contexts=_root_contexts(a,headers);group_sides={}
    a._root_connection_contexts=contexts
    # Offline slots protect future real valve/axial/operation volumes. They
    # are not graph valves and never substitute for the final native package.
    for context in contexts:
        point=context['point'];group=point.get('group_id');failures=[]
        for valve in context['candidates']:
            if group in group_sides and valve['operating_side']!=group_sides[group]:continue
            hits=_root_fits(a,valve,context['nozzle'])
            if hits:
                failures.append(','.join(hits));continue
            context['selected']=valve;a.reserve_valve(valve)
            if group:group_sides[group]=valve['operating_side']
            break
        else:
            raise PipingRefused(point['endpoint']+'No feasible off-line valve placement or straight reservation: '+'；'.join(failures[:2]))
    for context in contexts:
        point=context['point'];group=point.get('group_id');failures=[];selected=context['selected']
        candidates=[selected]+[v for v in context['candidates'] if v['root_candidate']!=selected['root_candidate']]
        for valve in candidates:
            if group in group_sides and valve['operating_side']!=group_sides[group]:continue
            a.reserve_valve(valve)
            hits=_root_fits(a,valve,context['nozzle'])
            if hits:
                failures.append({'candidate':valve['root_candidate'],'reason':'Physical valve placement conflicts with the reserved space: '+','.join(hits)});continue
            saved=(len(a.scene['components']),len(a.routes),len(a.allowed_contacts),len(a.reducers),
                   len(a.scene.get('elbows',[])),len(a.graph['valves']),len(a.valve_reservations))
            try:
                _root_connect_one(a,context,valve)
                context['connected']=True
                valve['root_route_trials']=deepcopy(failures)
                valve['root_reserved_candidate']=selected['root_candidate']
                break
            except (CandidateRefused,AssetUnavailable,RouteUnavailable) as exc:
                for rows,size in zip((a.scene['components'],a.routes,a.allowed_contacts,a.reducers,
                                     a.scene.setdefault('elbows',[]),a.graph['valves'],a.valve_reservations),saved):del rows[size:]
                failures.append({'candidate':valve['root_candidate'],'reason':_root_error(exc)})
        else:
            a.reserve_valve(selected);selected['root_route_trials']=deepcopy(failures)
            raise PipingRefused(point['endpoint']+'Limited merge and valve-placement candidates found no compliant connection: '+'；'.join(r['reason'] for r in failures[:2]))
    return a
