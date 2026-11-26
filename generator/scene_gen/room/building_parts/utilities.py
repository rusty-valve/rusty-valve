"""Connect equipment to utility pipes on each floor."""
import itertools
import json
import math
from pathlib import Path
import numpy as np
from ..layout import add_box, add_pipe, quat_from_axes
from ..routing import ROOT, package, simplify, line_hits

RULES = json.loads((Path(__file__).resolve().parent.parent/'piping_rules.json').read_text())
STEEL=(.24,.31,.36)
SILVER=(.65,.70,.71)

def radius(dn):
    return RULES['diameter_matching']['pipe_outer_diameter_mm'][str(dn)]/2000

def segment_distance(a,b,p,q):
    """Closest finite centerline points; independent of MuJoCo contact tolerance."""
    a,b,p,q=map(lambda x:np.asarray(x,float),(a,b,p,q));u=b-a;v=q-p;w=a-p
    aa=u@u;bb=u@v;cc=v@v;dd=u@w;ee=v@w;den=aa*cc-bb*bb
    candidates=[(0,np.clip(ee/cc,0,1)),(1,np.clip((ee+bb)/cc,0,1)),(np.clip(-dd/aa,0,1),0),(np.clip((bb-dd)/aa,0,1),1)]
    if den>1e-15:
        t=(bb*ee-cc*dd)/den;k=(aa*ee-bb*dd)/den
        if 0<=t<=1 and 0<=k<=1:candidates.append((t,k))
    return min((float(np.linalg.norm(a+t*u-p-k*v)),float(t),float(k)) for t,k in candidates)

def rack_return_route(s,e,a,b,xy,level):
    """Choose a collision-free rack elevation from pipe envelopes, not a fixed lift."""
    rx,ry=xy;zone=s['rack_zones'][level]['box'];r=radius(e['dn'])
    clearance=RULES['routing']['unconnected_crossing_clearance_m']
    others=[c for c in s['components'] if c['kind']=='pipe' and c.get('owner')!=e['id']]
    heights={zone[2]+r}
    for c in others:
        z=max(p[2] for p in c['ends'])+c['radius']+r+clearance
        if zone[2]+r<=z<=zone[5]-r:heights.add(math.ceil(z*1000)/1000)
    trials=[]
    for z in sorted(heights):
        points=simplify([a,[rx,a[1],a[2]],[rx,ry,a[2]],[rx,ry,z],[b[0],ry,z],[b[0],b[1],z],b])
        hits=[]
        for u,v in zip(points,points[1:]):
            for c in others:
                if any(math.dist(p,q)<1e-6 for p in (u,v) for q in c['ends']):continue
                distance,t,k=segment_distance(u,v,*c['ends']);gap=distance-r-c['radius']
                crossed=1e-6<t<1-1e-6 and 1e-6<k<1-1e-6
                if gap<(-1e-6 if not crossed else clearance-1e-6):hits.append(c['id'])
        trials.append({'center_z_m':z,'obstacles':sorted(set(hits))})
        if not hits:
            s.setdefault('rack_route_choices',[]).append({'edge':e['id'],'chosen_z_m':z,'trials':trials})
            return points,z
    raise ValueError(f'{e["id"]}No collision-free route in the rack: {trials}')

def color(medium):
    return RULES['appearance']['color_map'].get(medium,RULES['appearance']['color_map']['unknown'])['rgb']

def valve_space(length):
    return RULES['valve_installation_spacing']['minimum_gap_in_valve_lengths']*length

def offgas_horizontal_run(drop):
    angle=RULES['offgas_collection']['equipment_to_floor_collector_join']['axis_angle_degrees']
    return drop/math.tan(math.radians(angle))

def floor_connection_policy(medium):
    """Resolve one effective rule; medium exceptions override the common case."""
    common=RULES['workstation']['fixed_branch_connection']
    return {**common,**common.get('exceptions',{}).get(medium,{})}

def pipe(s,name,a,b,dn,medium,owner,kind='pipe'):
    c=add_pipe(s,name,a,b,radius(dn),kind,color(medium),owner)
    if c: c.update(dn=dn,medium=medium)
    return c

def graph_network(b,spec):
    cfg=spec['scene_choices']['utility_station']
    for service in cfg['services']:
        medium=service['medium']; dn=service['main_dn'] if medium=='offgas' else service['branch_dn']; main='U_'+medium
        policy=floor_connection_policy(medium)
        service['branch_dn']=dn
        equipment_dn=next(p['pipe_dn'] for p in spec['drawing']['ports'] if p.get('process_port')==medium)
        b.unit(main,'junction',service=medium,utility_tier='workstation_main',main_dn=service['main_dn'])
        for lev in (0,1):
            dist=f'D_{medium}_{lev}'; tmp=f'T_{medium}_{lev}'
            users=[u for u in b.g['units'].values() if u['kind']=='vessel' and u['level']==lev]
            active=bool(users)
            if active or policy['without_floor_source']!='omit_fixed_outlet':
                b.unit(dist,'junction',service=medium,utility_tier='floor_distribution',level=lev)
                a=b.port(main,f'floor{lev}',dn); z=b.port(dist,'in',dn)
                start,finish=(z,a) if medium=='offgas' else (a,z)
                b.edge(start,finish,dn,medium,manual=f'{medium} floor branch isolation valve' if policy['root_manual_valve_required'] else None)
                if policy['root_manual_valve_required']:
                    b.g['valves'][-1].update(role='floor utility branch isolation',name=f'Nitrogen · {'ground level' if lev==0 else 'platform'}branch isolation valve',utility_use='floor_branch')
                e=b.g['edges'][-1];e.update(utility_tier='floor_branch',service=medium,level=lev,main=main,distribution=dist)
            b.unit(tmp,'boundary',dn=cfg['temporary_dn'],utility_tier='temporary_outlet',level=lev,service=medium)
            b.edge(b.port(main,f'temporary{lev}',cfg['temporary_dn']),tmp+'.port',cfg['temporary_dn'],medium,manual='temporary connection ball valve')
            b.g['edges'][-1].update(utility_tier='temporary_outlet',service=medium,level=lev,main=main)
            b.g['valves'][-1].update(role='temporary station connection',name=f'{'nitrogen' if medium=="nitrogen" else 'off-gas'} · {'ground level' if lev==0 else 'platform'} temporary outlet ball valve',utility_use='temporary_outlet')
            if active:
                for user in users:
                    a=b.port(dist,'out' if len(users)==1 else 'out_'+user['id'],equipment_dn);z=user['id']+'.'+medium
                    b.edge(z,a,equipment_dn,medium) if medium=='offgas' else b.edge(a,z,equipment_dn,medium)
                    b.g['edges'][-1].update(utility_tier='equipment_connection',service=medium,level=lev,distribution=dist)
            elif policy['without_floor_source']!='omit_fixed_outlet':
                # Keep the planned root outlet, but terminate it at its flange.
                fixed=next(e for e in b.g['edges'] if e.get('distribution')==dist)
                fixed['unused']=True
                v=next(v for v in b.g['valves'] if v['edge']==fixed['id'])
                v.update(name=v['name'].replace('branch isolation valve','spare root valve'),role='spare root at unused floor')


def compact_header_lanes(g,spec):
    """Pack parallel radial envelopes; root valve/elbow space sets the rear limit.

    The conservative flange envelope is retained along each lane. It avoids
    relying on a particular axial flange stagger to obtain a small pitch.
    """
    cfg=spec['scene_choices']['utility_station'];rules=RULES['parallel_pipe_spacing']
    bare=rules['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000
    flange=rules['protrusions']['min_clearance_mm']/1000
    rear=None;rows=[]
    for service in reversed(cfg['services']):
        medium=service['medium'];r=radius(service['branch_dn'])
        v=next((v for v in g['valves'] if any(e['id']==v['edge'] and e.get('service')==medium and e.get('utility_tier')=='floor_branch' for e in g['edges'])),None)
        if v:
            _,p=package(v);length=p['geometry'].get('face_to_face_mm',p['geometry'].get('body_length_mm'))/1000
            # Main surface -> one valve space -> valve -> one valve space -> elbow tangent.
            limit=cfg['main_y_m']-radius(service['main_dn'])-2*valve_space(length)-length-3*r
            # The turn-up pipe must also clear the full lever sweep, not just body length.
            center_y=cfg['main_y_m']-radius(service['main_dn'])-valve_space(length)-length/2
            operator=p['geometry'].get('handle_length_mm',0)/1000
            limit=min(limit,center_y-operator-.05-r)
        else:
            # Direct overhead collection: only the elbow envelope sets this limit.
            limit=cfg['main_y_m']-radius(service['main_dn'])-3*r
        lane=limit
        if rear:
            rr=radius(rear['branch_dn']);dz=r-rr
            distance=max(r+rr+bare,(r+.035)+(rr+.035)+flange)
            pitch=math.sqrt(max(0,distance*distance-dz*dz))
            lane=min(lane,rear['branch_lane_y_m']-pitch)
        lane=math.floor((lane+1e-9)*1000)/1000
        service['branch_lane_y_m']=lane;rear=service
        rows.append({'service':medium,'lane_y_m':lane,'root_elbow_rear_limit_m':limit,'radial_envelope_m':r+.035})
    # Tighten the rear lanes toward the most restricted front lane. Removing a
    # large offgas valve must not leave an arbitrary gap between overhead pipes.
    for front,back in zip(cfg['services'],cfg['services'][1:]):
        r,rr=radius(front['branch_dn']),radius(back['branch_dn'])
        distance=max(r+rr+bare,(r+.035)+(rr+.035)+flange)
        pitch=math.sqrt(max(0,distance*distance-(r-rr)**2))
        back['branch_lane_y_m']=min(back['branch_lane_y_m'],math.ceil((front['branch_lane_y_m']+pitch)*1000)/1000)
    for row in rows:row['lane_y_m']=next(x['branch_lane_y_m'] for x in cfg['services'] if x['medium']==row['service'])
    return {'rule':'PG-14','lanes':rows,'packing':'bare 50mm and conservative flange envelopes 30mm; 1mm placement resolution, not clearance tolerance'}

def place_valve(v,center,axis,stem,level):
    pack,params=package(v); geo=params['geometry']
    length=geo.get('face_to_face_mm',geo.get('body_length_mm'))/1000
    q,mat=quat_from_axes(axis,stem)
    v.update(center=list(center),axis=list(axis),quat=q,matrix=mat.tolist(),length=length,package=str(pack.relative_to(ROOT)),level=level)
    return np.array(center)-np.array(axis)*length/2,np.array(center)+np.array(axis)*length/2

def blind(s,name,p,axis,dn,owner):
    p=np.array(p);axis=np.array(axis);r=radius(dn)+.034
    flange=add_pipe(s,name+'_flange',p-axis*.025,p,r,'nozzle',SILVER,owner)
    cap=add_pipe(s,name+'_blind',p,p+axis*.018,r,'nozzle',SILVER,owner)
    cap.update(blind=True,dn=dn)
    # Bolt silhouettes are simple cylinders, not CAD fastener meshes.
    normal=np.array([0,0,1]) if abs(axis[2])<.5 else np.array([1,0,0])
    cross=np.cross(axis,normal)
    for i in range(4):
        a=math.tau*i/4;offset=(r-.012)*(normal*math.cos(a)+cross*math.sin(a))
        add_pipe(s,name+f'_bolt{i}',p+offset-axis*.03,p+offset+axis*.025,.006,'nozzle',STEEL,owner)
    return cap

def flange_joint(s,name,valve_end,axis,dn,owner,unused=False,straight_m=0):
    """Simple threaded adapter and bolted flange pair; does not alter valve ports."""
    p=np.asarray(valve_end,float);axis=np.asarray(axis,float);r=radius(dn)+.035
    names=[]
    def part(suffix,a,b,rad):
        c=add_pipe(s,name+suffix,p+axis*a,p+axis*b,rad,'nozzle',SILVER,owner)
        names.append(c['id']);return c
    part('_thread_adapter',0,.045+straight_m,radius(dn)+.005)
    part('_flange',.045+straight_m,.063+straight_m,r)
    part('_blind' if unused else '_mating_flange',.066+straight_m,.084+straight_m,r)
    normal=np.array([0,0,1.]) if abs(axis[2])<.5 else np.array([1.,0,0]);cross=np.cross(axis,normal)
    for i in range(4):
        a=math.tau*i/4;off=(r-.010)*(normal*math.cos(a)+cross*math.sin(a))
        c=add_pipe(s,name+f'_bolt{i}',p+off+axis*(.040+straight_m),p+off+axis*(.090+straight_m),.005,'nozzle',STEEL,owner)
        names.append(c['id'])
    s.setdefault('flange_connections',[]).append({'id':name,'owner':owner,'dn':dn,'unused':unused,
        'components':names,'valve_end':p.tolist(),'axis':axis.tolist(),'straight_m':straight_m,
        'scope':'Thread-to-flange envelope and bolt outline'})
    return (p+axis*(.084+straight_m)).tolist()

def run(s,e,points,valve=None):
    points=simplify(points);pieces=[]
    for i,(a,b) in enumerate(zip(points,points[1:])):
        segments=[(a,b)]
        if valve:
            c=np.array(valve['center']);axis=(np.array(b)-a)/math.dist(a,b)
            t=np.dot(c-np.array(a),axis)
            if np.linalg.norm(c-(np.array(a)+t*axis))<1e-7 and 0<t<math.dist(a,b):
                half=valve['length']/2
                segments=[(a,c-axis*half),(c+axis*half,b)]
        j=0
        for start,end in segments:
            u=(np.asarray(end)-start)/math.dist(start,end);length=math.dist(start,end)
            cuts=[(0,list(start)),(length,list(end))]
            for p in e.get('junction_points',[]):
                t=float(np.dot(np.asarray(p)-start,u))
                if 1e-7<t<length-1e-7 and np.linalg.norm(np.asarray(p)-start-t*u)<1e-7:cuts.append((t,p))
            cuts=sorted(cuts,key=lambda x:x[0])
            for (_,p),(_,q) in zip(cuts,cuts[1:]):
                part=pipe(s,f'{e["id"]}_seg{i}_{j}',p,q,e['dn'],e.get('service',e['role']),e['id']);j+=1
                if part:pieces.append(part['id'])
    s['routing'].append({'edge':e['id'],'points':points,'components':pieces,'elbows':max(0,len(points)-2),'constraints':e.get('utility_tier','equipment_and_support_corridor')})

def build_station(g,s,spec):
    cfg=spec['scene_choices']['utility_station']; H=spec['platform_z_m']; tips={}
    top=s['rack_zones'][-1]['box'][5]
    s['utility_stations']=[];s['station_connections']=[];s['header_supports']=[]
    s['header_layout']=compact_header_lanes(g,spec)
    for service in cfg['services']:
        medium=service['medium'];x=service['main_x_m'];y=cfg['main_y_m'];dn=service['main_dn'];uid='U_'+medium
        lane=service['branch_lane_y_m'];bdn=service['branch_dn']
        zs=sorted({cfg['main_start_z_m'],top,*[base+h for base in (0,H) for h in (cfg['temporary_valve_height_m'],cfg['floor_branch_valve_height_m'])]})
        if floor_connection_policy(medium)['route_to_main']=='direct_overhead':
            for lev in (0,1):
                if any(e.get('service')==medium and e.get('level')==lev and e.get('utility_tier')=='floor_branch' for e in g['edges']):
                    zs.append(s['rack_zones'][lev]['box'][2]+radius(bdn))
            zs=sorted(set(zs))  # Split the real main at the overhead tee.
        for i,(z0,z1) in enumerate(zip(zs,zs[1:])):pipe(s,uid+f'_riser{i}',[x,y,z0],[x,y,z1],dn,medium,uid)
        blind(s,uid+'_end',[x,y,top],[0,0,1],dn,uid)
        add_pipe(s,uid+'_boundary',[x,y,zs[0]-.02],[x,y,zs[0]+.02],radius(dn)+.035,'nozzle',SILVER,uid)
        g['units'][uid].update(center=[x,y,top/2],level=0)
        s['holes'][1].append([x-radius(dn)-.025,y-radius(dn)-.025,x+radius(dn)+.025,y+radius(dn)+.025])
        for lev,base in enumerate((0,H)):
            dist=f'D_{medium}_{lev}';tmp=f'T_{medium}_{lev}'
            active=any(e.get('utility_tier')=='equipment_connection' and e.get('service')==medium and e.get('level')==lev for e in g['edges'])
            bottom=s['rack_zones'][lev]['box'][2]
            overhead=bottom+radius(bdn)
            connections=[e for e in g['edges'] if e.get('utility_tier')=='equipment_connection' and e.get('service')==medium and e.get('level')==lev]
            taps={}
            for e in connections:
                equipment=next(ep for ep in (e['a'],e['b']) if g['units'][ep.split('.')[0]]['kind']=='vessel')
                peer=next(ep for ep in (e['a'],e['b']) if ep.split('.')[0]==dist)
                uu,pp=equipment.split('.');tapx=g['units'][uu]['ports'][pp]['pos'][0]
                if medium=='offgas':tapx+=offgas_horizontal_run(cfg['offgas_join_drop_m'])+cfg['offgas_join_shift_m']
                taps[peer]=[tapx,lane,overhead]
            px=min((p[0] for p in taps.values()),default=4.10)
            end=[px,lane,overhead]
            for kind,height in (('floor_branch',cfg['floor_branch_valve_height_m']),('temporary_outlet',cfg['temporary_valve_height_m'])):
                e=next((e for e in g['edges'] if e.get('utility_tier')==kind and e.get('service')==medium and e.get('level')==lev),None)
                if e is None:continue
                direct=kind=='floor_branch' and floor_connection_policy(medium)['route_to_main']=='direct_overhead'
                z=overhead if direct else base+height
                if direct:
                    v=None;joint=None
                    row={'edge':e['id'],'valve':None,'main':uid,'kind':kind,'level':lev,'joint':None,'unused':False,'direct_overhead':True}
                else:
                    v=next(v for v in g['valves'] if v['edge']==e['id'])
                    _,params=package(v);length=params['geometry'].get('face_to_face_mm',params['geometry'].get('body_length_mm'))/1000
                    center=[x,y-radius(dn)-valve_space(length)-length/2,z]
                    a,b=place_valve(v,center,[0,-1,0],[0,0,1],lev)
                    v['installation_reference']={'kind':'main_surface','main':uid,'surface':[x,y-radius(dn),z]}
                    unused=kind=='temporary_outlet' or not active
                    joint=flange_joint(s,e['id']+'_joint',b,[0,-1,0],v['dn'],e['id'],unused=unused)
                    row={'edge':e['id'],'valve':v['id'],'main':uid,'kind':kind,'level':lev,'joint':e['id']+'_joint','unused':unused}
                s['station_connections'].append(row)
                if kind=='floor_branch' and active:
                    e['junction_points']=list(taps.values())
                    points=[[x,y,z],[x,lane,z],[x,lane,overhead],end]
                    run(s,e,points,v)
                    row.update(header_start=[x,lane,overhead],header_join=end,header_taps=taps,pipe_bottom_z=bottom)
                    g['units'][uid]['ports'][f'floor{lev}'].update(pos=points[0],normal=[0,-1,0],level=lev)
                    g['units'][dist].update(center=end)
                    g['units'][dist]['ports']['in'].update(pos=end,level=lev)
                    if lev==1:
                        for endpoint,point in taps.items():
                            g['units'][dist]['ports'][endpoint.split('.')[1]].update(pos=point,level=lev)
                            tips[endpoint]=point
                        # Preserve the tee as two actual pipe segments, with an unused capped extension.
                        far=[px-.65,lane,overhead]
                        c=pipe(s,e['id']+'_reserve',end,far,bdn,medium,e['id'])
                        s['routing'][-1]['components'].append(c['id'])
                        blind(s,dist+'_reserve',far,[-1,0,0],bdn,dist)
                    else:blind(s,dist+'_reserve',end,[-1,0,0],bdn,dist)
                elif kind=='floor_branch':
                    run(s,e,[[x,y,z],joint],v)
                    g['units'][uid]['ports'][f'floor{lev}'].update(pos=[x,y,z],normal=[0,-1,0],level=lev)
                    g['units'][dist].update(center=joint)
                    g['units'][dist]['ports']['in'].update(pos=joint,level=lev)
                    row['termination']=joint
                else:
                    run(s,e,[[x,y,z],joint],v)
                    g['units'][tmp].update(center=joint);g['units'][tmp]['ports']['port'].update(pos=joint,level=lev)
                    g['units'][uid]['ports'][f'temporary{lev}'].update(pos=[x,y,z],level=lev)
            if lev==0 and active:
                arm=f'{dist}_support'
                add_box(s,arm,[4.06,lane-.035,bottom-.03,5.30,lane+.035,bottom],'hanger',STEEL)
                add_box(s,arm+'_hanger',[4.92,lane-.035,bottom-.03,4.98,lane+.035,H-.04],'hanger',STEEL)
        for k,zz in enumerate((H-.14,top-.18)):
            add_box(s,uid+f'_bracket{k}',[x-.03,y,zz-.02,x+.03,5.98,zz+.02],'hanger',STEEL)
            add_pipe(s,uid+f'_clamp{k}',[x,y,zz-.025],[x,y,zz+.025],radius(dn)+.006,'hanger',STEEL)
        s['utility_stations'].append({'service':medium,'main':uid,'main_dn':dn,'branch_dn':bdn,'temporary_dn':cfg['temporary_dn'],'xyz0':[x,y,zs[0]],'xyz1':[x,y,top],'served_levels':[0,1]})
    return tips


def offgas_route(a,b,scene,dn,drop):
    """The final 45-degree leg follows the collector towards its workstation."""
    # Collector runs towards +x in this room; derive it from the built station.
    row=next(x for x in scene['station_connections'] if x['main']=='U_offgas' and x['kind']=='floor_branch' and x['level']==1)
    forward=np.array(row['header_start'])-np.array(row['header_join']);forward[2]=0;forward/=np.linalg.norm(forward)
    approach=np.array(b)-forward*offgas_horizontal_run(drop)+np.array([0,0,drop])
    from ..process import RULES as PROCESS_RULES
    clear_z=scene['floors'][row['level']]+PROCESS_RULES['geometry_screen']['walk_height_m']+radius(dn)+.05
    waypoints=[[a[0],a[1],clear_z]] if a[2]<clear_z<approach[2] else []
    pts,count=least_elbow_route(a,approach.tolist(),scene,dn,extra_waypoints=waypoints,required_first_axis=2)
    pts.append(list(b))
    return pts,count


def route_cost(points,medium=None):
    length=sum(math.dist(x,y) for x,y in zip(points,points[1:]));bends=max(0,len(points)-2)
    return (length,bends) if medium in ('vapor','process_evaporated_vapor') else (bends,length)


def least_elbow_route(a,b,scene,dn,extra_waypoints=(),medium=None,required_first_axis=None):
    """Compare orthogonal candidates by elbows then length, rejecting solid hits.

    Candidate set is explicit and bounded; this does not claim a global optimum.
    Endpoint boxes are excluded only at the attached equipment connection.
    """
    obstacles=list(scene.get('valve_route_reservations',[]))
    for c in scene['components']:
        if c.get('visual_only'):continue
        if c['kind'] in ('vessel','head','motor','column','beam','walk_surface','pump','nozzle'):
            obstacles.append(c['box'])
        elif c['kind']=='pipe' and not any(math.dist(p,q)<1e-6 for p in (a,b) for q in c['ends']):
            obstacles.append((np.array(c['box'])+np.array([-.04,-.04,-.04,.04,.04,.04])).tolist())
    candidates=[]
    waypoints=list(extra_waypoints)
    for distance in (.20,.35,.50):
        for axis in (0,1):
            for sign in (-1,1):
                p=list(a);p[axis]+=sign*distance;waypoints.append(p)
    for via in [None,*waypoints]:
        for order in itertools.permutations(range(3)):
            points=[list(a)]
            for target in ([b] if via is None else [via,b]):
                for axis in order:
                    p=points[-1].copy();p[axis]=target[axis];points.append(p)
            points=simplify(points)
            if required_first_axis is not None and len(points)>1 and np.argmax(np.abs(np.array(points[1])-points[0]))!=required_first_axis:continue
            if all(not line_hits(x,y,box,radius(dn)+.01) for x,y in zip(points,points[1:]) for box in obstacles):
                candidates.append(points)
    if not candidates:raise ValueError(f'No short route meets the support corridor: {a} -> {b}')
    candidates.sort(key=lambda p:route_cost(p,medium))
    return candidates[0],len(candidates)

def check(g,s,spec):
    errors=[];units=g['units'];cfg=spec['scene_choices']['utility_station'];byid={c['id']:c for c in s['components']}
    def require(ok,code,detail):
        if not ok:errors.append({'code':code,'detail':detail})
    for row in s['utility_stations']:
        uid=row['main'];medium=row['service'];allowed=RULES['workstation']['main_dn']['allowed_values'].get(medium)
        require(allowed is None or row['main_dn'] in allowed,'MAIN_DN',uid)
        require(row['temporary_dn'] in RULES['workstation']['temporary_connection_dn']['allowed_values'],'TEMP_DN',uid)
        require(row['branch_dn']==row['main_dn'] if medium=='offgas' else row['branch_dn']<row['main_dn'],'MAIN_BRANCH_DN',uid)
        require(0<row['xyz0'][0]<6 and 0<row['xyz0'][1]<6 and row['xyz0'][:2]==row['xyz1'][:2],'RISER_POSITION',uid)
        require(abs(row['xyz1'][2]-s['rack_zones'][-1]['box'][5])<1e-6,'RISER_HEIGHT',uid)
        require(uid+'_end_blind' in byid,'MAIN_BLIND',uid)
        for lev in range(len(s['floors'])):
            active=any(e.get('utility_tier')=='equipment_connection' and e.get('service')==medium and e.get('level')==lev for e in g['edges'])
            for kind in ('floor_branch','temporary_outlet'):
                edges=[e for e in g['edges'] if e.get('utility_tier')==kind and e.get('main')==uid and e.get('level')==lev]
                policy=floor_connection_policy(medium)
                direct=kind=='floor_branch' and not policy['root_manual_valve_required']
                omitted=kind=='floor_branch' and not active and policy['without_floor_source']=='omit_fixed_outlet'
                require(len(edges)==(0 if omitted else 1),'FLOOR_CONNECTION',f'{uid}/{lev}/{kind}')
                if edges:
                    valves=[v for v in g['valves'] if v['edge']==edges[0]['id']]
                    require(not valves if direct else any(v['manual'] for v in valves),'OFFGAS_NO_ROOT_VALVE' if direct else 'FLOOR_VALVE',edges[0]['id'])
            tmp=next(e for e in g['edges'] if e.get('utility_tier')=='temporary_outlet' and e.get('main')==uid and e.get('level')==lev)
            require(tmp['id']+'_joint_blind' in byid,'TEMP_BLIND',f'{uid}/{lev}')
            fixed=next((e for e in g['edges'] if e.get('utility_tier')=='floor_branch' and e.get('main')==uid and e.get('level')==lev),None)
            if fixed is None:continue
            cap=f'D_{medium}_{lev}_reserve_blind' if active else fixed['id']+'_joint_blind'
            require(cap in byid,'RESERVE_BLIND',f'{uid}/{lev}')
        for e in g['edges']:
            if e.get('utility_tier')=='equipment_connection' and e.get('service')==medium:
                peers=[ep.split('.')[0] for ep in (e['a'],e['b'])]
                require(any(units[u].get('utility_tier')=='floor_distribution' for u in peers),'DIRECT_TO_MAIN',e['id'])
    for u in units.values():
        if u['kind']!='vessel':continue
        n=u['ports']['nitrogen']['dn'];o=u['ports']['offgas']['dn']
        require(n<=RULES['diameter_defaults']['nitrogen']['max_dn'] and n==o,'NITROGEN_OFFGAS_DN',f'{u["id"]}: {n}/{o}')
    tap_checks=[]
    for e in g['edges']:
        if e.get('utility_tier')!='equipment_connection':continue
        endpoint=next(ep for ep in (e['a'],e['b']) if units[ep.split('.')[0]].get('utility_tier')=='floor_distribution')
        uid,port=endpoint.split('.');point=units[uid]['ports'][port]['pos']
        header=next(h for h in g['edges'] if h.get('utility_tier')=='floor_branch' and h.get('distribution')==uid)
        parts=[c for c in s['components'] if c['kind']=='pipe'];touches=[]
        for owner in (e['id'],header['id']):
            touches.append(any(c.get('owner')==owner and any(math.dist(p,point)<1e-6 for p in c['ends']) for c in parts))
        require(all(touches),'EQUIPMENT_HEADER_TEE',f'{endpoint} has no physical tee connection to the shared branch')
        tap_checks.append({'equipment_edge':e['id'],'medium':e['service'],'tee':endpoint,'point':point,'connected':all(touches)})
    require(bool(s.get('shared_rack')),'TOP_RACK','Top floor branch lacks a full-span rack supported at both sides')
    for c in s['components']:
        if 'dn' in c and 'medium' in c:
            require(abs(c['radius']-radius(c['dn']))<1e-9,'PIPE_OUTER_DIAMETER',c['id'])
            require(list(c['color'])==list(color(c['medium'])),'MEDIUM_COLOR',c['id'])
    return {'status':'fail' if errors else 'pass','errors':errors,'workstations':s['utility_stations'],'equipment_header_tees':tap_checks,
            'routing':[{'edge':r['edge'],'elbows':r.get('elbows',len(r['points'])-2),'candidate_count':r.get('candidate_count'),'constraints':r.get('constraints')} for r in s['routing']],
            'scope':'Check mains, branches, floor valves and blinds, DN, OD, color and supports. Bend counts compare the specified orthogonal candidates.'}
