"""Check pipe supports, valve clearances and utility connections."""
import math
import numpy as np
from ..routing import package, simplify
from .utilities import radius, RULES, valve_space

EPS=1e-6

def result(rows,errors,**extra):
    return {'status':'fail' if errors else 'pass','checked':len(rows),'measurements':rows,'errors':errors,**extra}

def valve_spaces(g,s):
    rows=[];errors=[];route={r['edge']:r for r in s['routing']}
    def measure(v,label,gap,reference):
        _,p=package(v);length=p['geometry'].get('face_to_face_mm',p['geometry'].get('body_length_mm'))/1000
        ok=gap>=valve_space(length)-EPS
        row={'valve':v['id'],'side':label,'gap_mm':round(gap*1000,3),'one_valve_space_mm':length*1000,'pass':ok,'reference':reference}
        rows.append(row)
        if not ok:errors.append({'code':'ONE_VALVE_SPACE',**row})
    for v in g['valves']:
        c=np.array(v['center']);axis=np.array(v['axis']);half=v['length']/2
        ref=v.get('installation_reference',{})
        pts=route[v['edge']]['points']
        if ref.get('kind')=='equipment_port':
            uid,port=ref['endpoint'].split('.');p=g['units'][uid]['ports'][port]
            # The original nozzle face includes its necessary reducer. The
            # generator additionally leaves a valve-length spool after the reducer.
            origin=np.array(p['pos'])
            if p.get('nozzle'):
                origin=np.array(next(n['face_center_m'] for n in g['nozzle_schedule'] if n['tag']==p['nozzle'] and n.get('unit',uid)==uid))
            measure(v,'equipment port side',float(np.dot(c-origin,axis)-half),origin.tolist())
            edge=next(e for e in g['edges'] if e['id']==v['edge'])
            ordered=pts if edge['a']==v['at'] else list(reversed(pts))
            path=simplify([(c+axis*half).tolist(),*ordered])
            if len(path)>2:
                # A simple elbow is reserved by its nominal centreline radius.
                gap=math.dist(path[0],path[1])-3*radius(v['dn'])
                measure(v,'first elbow tangent side',gap,path[1])
        else:
            if ref.get('kind')=='main_surface':
                main=next(x for x in s['utility_stations'] if x['main']==ref['main'])
                origin=np.array([*main['xyz0'][:2],c[2]])+axis*radius(main['main_dn'])
                measure(v,'main pipe outer wall side',float(np.dot(c-origin,axis)-half),origin.tolist())
            for i,(a,b) in enumerate(zip(pts,pts[1:])):
                a=np.array(a);b=np.array(b);d=b-a;n=np.linalg.norm(d);u=d/n;t=np.dot(c-a,u)
                if np.linalg.norm(c-a-t*u)>EPS or t<0 or t>n:continue
                if i>0:measure(v,'upstream elbow tangent side',float(t-half-3*radius(v['dn'])),a.tolist())
                if i+1<len(pts)-1:measure(v,'downstream elbow tangent side',float(n-t-half-3*radius(v['dn'])),b.tolist())
    return result(rows,errors,valve_count=len({x['valve'] for x in rows}),
        method='Use installed valve lengths and a 1.5 pipe-OD elbow centerline radius. Measure to the tangent point. Branch fittings use simplified geometry.')

def station_flanges(g,s):
    rows=[];errors=[];byid={c['id']:c for c in s['components']};vs={v['id']:v for v in g['valves']}
    for e in g['edges']:
        if e.get('utility_tier') not in ('floor_branch','temporary_outlet'):continue
        if e.get('service')=='offgas' and e['utility_tier']=='floor_branch':continue  # PG-15: direct collection, no valve joint.
        row=next((r for r in s.get('station_connections',[]) if r['edge']==e['id']),None)
        v=vs.get(row.get('valve')) if row else None
        active=any(x.get('utility_tier')=='equipment_connection' and x.get('service')==e.get('service') and x.get('level')==e.get('level') for x in g['edges'])
        name=e['id']+'_joint';unused=e['utility_tier']=='temporary_outlet' or not active
        needed=[name+'_thread_adapter',name+'_flange',name+('_blind' if unused else '_mating_flange')]
        missing=[n for n in needed if n not in byid]
        ok=v is not None and v['manual'] and v['edge']==e['id'] and not missing
        if v is not None and not missing:
            # Adapter starts on the actual downstream valve end, not merely near it.
            end=np.array(v['center'])+np.array(v['axis'])*v['length']/2
            ok=ok and math.dist(end,byid[needed[0]]['ends'][0])<EPS
        record={'edge':e['id'],'kind':e['utility_tier'],'root_valve':v['id'] if v else None,'unused':unused,'missing':missing,'pass':bool(ok)}
        rows.append(record)
        if not ok:errors.append({'code':'ROOT_FLANGE_BLIND',**record})
    return result(rows,errors,scope='Check manual root valves, physical adapters, flanges and blind plates. Pressure rating and bolting tool clearance require separate verification.')

def floor_crossings(g,s,spec):
    rows=[];errors=[];w,d=s['footprint'];cfg=spec['scene_choices']['utility_station']
    # Scene-specific corridors, derived from the outer steel bay and workstation.
    # These are not universal platform-edge distances.
    byid={c['id']:c for c in s['components']}
    right_inner=max(c['box'][3] for c in s['components'] if c['id'].startswith('joist') and c['shape']=='box')
    back_inner=cfg['main_y_m']-max(radius(x['main_dn']) for x in cfg['services'])-.10
    rail_x=byid['right_bar0']['ends'][0][0]-byid['right_bar0']['radius']
    rail_y=byid['back_bar0']['ends'][0][1]-byid['back_bar0']['radius']
    edges={e['id']:e for e in g['edges']}
    for c in s['components']:
        if c['kind']!='pipe' or c['shape']!='cylinder':continue
        a,b=map(np.array,c['ends']);r=c['radius']
        for lev,z in enumerate(s['floors'][1:],1):
            if min(a[2],b[2])>=z-EPS or max(a[2],b[2])<=z+EPS:continue
            p=a+(b-a)*(z-a[2])/(b[2]-a[2])
            inside=r<p[0]<rail_x-r and r<p[1]<rail_y-r
            edge_zone=(p[0]-r>=right_inner or p[1]-r>=back_inner)
            hole=any(h[0]<=p[0]-r and h[2]>=p[0]+r and h[1]<=p[1]-r and h[3]>=p[1]+r for h in s['holes'].get(lev,s['holes'].get(str(lev),[])))
            medium=edges.get(c.get('owner'),{}).get('role',c.get('medium'))
            exempt=medium in ('vapor','process_evaporated_vapor')
            ok=inside and hole and (edge_zone or exempt)
            row={'pipe':c['id'],'floor':lev,'crossing':p.tolist(),'inside_guardrail':bool(inside),'edge_zone':bool(edge_zone),'opening':bool(hole),'vapor_exception':exempt,'pass':bool(ok)}
            rows.append(row)
            if not ok:errors.append({'code':'FLOOR_EDGE_CROSSING',**row})
    return result(rows,errors,scene_corridors={'right_inner_x':right_inner,'back_inner_y':back_inner},
        scope='Check each slab crossing using the pipe outer edge, rail inner edge and physical opening. Elbow and beam collisions have separate checks.')

def shared_headers(s):
    rows=[];errors=[];byid={c['id']:c for c in s['components']}
    headers=[r for r in s['station_connections'] if r['kind']=='floor_branch' and r['level']==1]
    bottoms=[]
    for h in headers:
        segments=[c for c in s['components'] if c.get('owner')==h['edge'] and c['kind']=='pipe' and abs(c['ends'][0][2]-c['ends'][1][2])<EPS and abs(c['ends'][0][0]-c['ends'][1][0])>EPS]
        if not segments:
            errors.append({'code':'TOP_HEADER_MISSING','edge':h['edge']});continue
        c=segments[0];bottom=c['ends'][0][2]-c['radius'];bottoms.append(bottom)
        service=h['main'].removeprefix('U_');supports=[]
        for row in s.get('header_supports',[]):
            if row['service']!=service:continue
            arm=byid.get(row['arm']);seat=byid.get(row['seat'])
            if arm is None or seat is None:continue
            seatx=(seat['box'][0]+seat['box'][3])/2
            on_pipe=any(min(p['ends'][0][0],p['ends'][1][0])-EPS<=seatx<=max(p['ends'][0][0],p['ends'][1][0])+EPS and seat['box'][1]<=p['ends'][0][1]<=seat['box'][4] for p in segments)
            touching=abs(seat['box'][5]-bottom)<EPS and abs(seat['box'][2]-arm['box'][5])<EPS
            if on_pipe and touching:supports.append(row['arm'])
        record={'main':h['main'],'pipe_bottom_z':bottom,'shared_arms':supports,'pass':bool(supports)};rows.append(record)
        if not supports:errors.append({'code':'HEADER_SUPPORT_CONTACT',**record})
    if len(bottoms)!=len(headers) or not bottoms or max(bottoms)-min(bottoms)>EPS:errors.append({'code':'TOP_HEADER_PLANE','bottoms':bottoms})
    if rows and not set.intersection(*(set(r['shared_arms']) for r in rows)):errors.append({'code':'NO_COMMON_HEADER_SUPPORT'})
    return result(rows,errors,scope='Align pipe bottoms. Pipe supports physically contact the pipe and shared beam. Structural loads require separate verification.')

def offgas_join(g,s):
    rows=[];errors=[];byid={c['id']:c for c in s['components']}
    if not any(x['service']=='offgas' for x in s['utility_stations']):return result(rows,[{'code':'OFFGAS_MAIN_MISSING'}])
    for e in g['edges']:
        if e.get('utility_tier')!='equipment_connection' or e.get('service')!='offgas':continue
        r=next(x for x in s['routing'] if x['edge']==e['id']);p,q=map(np.array,r['points'][-2:])
        lev=e['level'];collector=next((x for x in s['station_connections'] if x['main']=='U_offgas' and x['kind']=='floor_branch' and x['level']==lev),None)
        if collector is None:errors.append({'code':'OFFGAS_FLOOR_COLLECTOR','edge':e['id']});continue
        towards=np.array(collector['header_start'])-np.array(collector['header_join']);towards/=np.linalg.norm(towards)
        vector=q-p;length=np.linalg.norm(vector);angle=math.degrees(math.acos(np.clip(np.dot(vector,towards)/length,-1,1))) if length else 0
        # Verify the end of the emitted cylinder actually meets collector cylinders.
        pipes=[byid[x] for x in r['components'] if x in byid]
        actual=any(math.dist(c['ends'][1],q)<EPS and math.dist(c['ends'][0],p)<EPS for c in pipes)
        header_ends=[p for c in s['components'] if c['kind']=='pipe' and c.get('owner')==collector['edge'] for p in c['ends']]
        uid,port=e['b'].split('.');target=g['units'][uid]['ports'][port]['pos']
        connected=math.dist(q,target)<EPS and any(math.dist(q,p)<EPS for p in header_ends)
        header_radius=radius(next(x['branch_dn'] for x in s['utility_stations'] if x['service']=='offgas'))
        above=p[2]-radius(e['dn'])>q[2]+header_radius
        descending=vector[2]<0;forward=np.dot(vector[:2],towards[:2])>0
        required=RULES['offgas_collection']['equipment_to_floor_collector_join']['axis_angle_degrees']
        ok=actual and connected and above and descending and forward and abs(angle-required)<1e-4
        row={'edge':e['id'],'angle_deg':round(angle,6),'approach_above_header':bool(above),'descending':bool(descending),'toward_workstation':bool(forward),'actual_connected_segment':bool(actual and connected),'pass':bool(ok)}
        rows.append(row)
        if not ok:errors.append({'code':'OFFGAS_TOP_45_FORWARD',**row})
    if not rows:errors.append({'code':'OFFGAS_SOURCE_JOIN_MISSING'})
    return result(rows,errors,scope='Check connection, 45-degree slope from above, descent and flow direction. Condensate dynamics and backflow pressures require separate checks.')


def unused_floor_outlets(g,s):
    rows=[];errors=[];byid={c['id']:c for c in s['components']}
    for e in g['edges']:
        if e.get('utility_tier')!='floor_branch':continue
        if e.get('service')=='offgas':continue  # No unused outlet at all; checked below.
        active=any(x.get('utility_tier')=='equipment_connection' and x.get('service')==e['service'] and x.get('level')==e['level'] for x in g['edges'])
        if active:continue
        v=next(v for v in g['valves'] if v['edge']==e['id']);axis=np.array(v['axis']);center=np.array(v['center'])
        cap=byid.get(e['id']+'_joint_blind');extension=[]
        if cap:
            limit=max(float(np.dot(np.array(p)-center,axis)) for p in cap['ends'])
            for c in s['components']:
                if c.get('owner')==e['id'] and c['kind']=='pipe':
                    for p in map(np.array,c['ends']):
                        delta=p-center;t=float(np.dot(delta,axis))
                        if np.linalg.norm(delta-t*axis)>EPS or t>limit+EPS:extension.append(c['id']);break
                if c['id'].startswith(e['distribution']+'_reserve'):extension.append(c['id'])
        else:extension.append('root_outlet_blind_missing')
        row={'edge':e['id'],'service':e['service'],'level':e['level'],'root_valve':v['id'],'root_blind':cap['id'] if cap else None,'extended_components':extension,'pass':not extension}
        rows.append(row)
        if extension:errors.append({'code':'UNUSED_FLOOR_EXTENSION',**row})
    return result(rows,errors,scope='Determine which levels have real equipment users. An unused connection keeps its root valve and blind flange without a pipe extension.')


def offgas_diameters(g,s):
    rows=[];errors=[]
    for e in g['edges']:
        if e.get('utility_tier')!='floor_branch' or e.get('service')!='offgas':continue
        main=g['units'][e['main']]['main_dn']
        pipes=[c for c in s['components'] if c.get('owner') in (e['id'],e['main']) and c['kind']=='pipe']
        okay=e['dn']==main and bool(pipes) and all(c['dn']==main and abs(c['radius']-radius(main))<EPS for c in pipes)
        row={'edge':e['id'],'main_dn':main,'branch_dn':e['dn'],'actual_pipe_count':len(pipes),'pass':okay};rows.append(row)
        if not okay:errors.append({'code':'OFFGAS_HEADER_DN',**row})
    return result(rows,errors,scope='The floor off-gas collector matches the main diameter and enters without a root valve. Equipment and temporary ports are sized separately.')


def offgas_main_connections(g,s):
    rows=[];errors=[]
    for lev in range(len(s['floors'])):
        active=any(e.get('utility_tier')=='equipment_connection' and e.get('service')=='offgas' and e.get('level')==lev for e in g['edges'])
        fixed=[e for e in g['edges'] if e.get('utility_tier')=='floor_branch' and e.get('service')=='offgas' and e.get('level')==lev]
        temp=[e for e in g['edges'] if e.get('utility_tier')=='temporary_outlet' and e.get('service')=='offgas' and e.get('level')==lev]
        if len(fixed)!=(1 if active else 0) or len(temp)!=1:
            errors.append({'code':'OFFGAS_FLOOR_CONNECTIONS','level':lev,'active':active})
        for e in fixed:
            valves=[v['id'] for v in g['valves'] if v['edge']==e['id']]
            if valves:errors.append({'code':'OFFGAS_NO_ROOT_VALVE','level':lev,'valves':valves})
            row=next((r for r in s['station_connections'] if r['edge']==e['id']),{})
            pipes=[c for c in s['components'] if c.get('owner')==e['id'] and c['kind']=='pipe']
            main=next(x for x in s['utility_stations'] if x['service']=='offgas')
            ends=[p for c in pipes for p in c['ends']]
            z=row.get('header_join',[0,0,-999])[2]
            overhead=bool(ends) and all(abs(p[2]-z)<EPS for p in ends)
            connected=any(math.dist(p[:2],main['xyz0'][:2])<EPS for p in ends)
            if not overhead or not connected:errors.append({'code':'OFFGAS_DIRECT_OVERHEAD','level':lev,'overhead':overhead,'main_connected':connected})
            rows.append({'level':lev,'active':active,'root_valves':valves,'direct_overhead':overhead and connected})
        if not active:
            # Also reject stale visible branch components even if graph tags were removed.
            extra=[r for r in s['station_connections'] if r['main']=='U_offgas' and r['level']==lev and r['kind']!='temporary_outlet']
            if extra:errors.append({'code':'OFFGAS_FLOOR_CONNECTIONS','level':lev,'extra':extra})
            rows.append({'level':lev,'active':False,'temporary_only':not fixed and not extra and len(temp)==1})
    return result(rows,errors,scope='Active floors join the overhead main directly without a root valve or a dip. Inactive floors have a temporary outlet only. Temporary valves and blinds are checked separately. Equipment off-gas root valves remain.')


def parallel_headers(s):
    """Measure actual parallel cylinders, including end flange/blind silhouettes."""
    rows=[];errors=[];cfg=RULES['parallel_pipe_spacing']
    bare=cfg['bare_unflanged_parallel_runs']['general_min_clearance_mm']/1000
    flange=cfg['protrusions']['min_clearance_mm']/1000
    headers=[r for r in s['station_connections'] if r['kind']=='floor_branch' and not r.get('unused')]
    groups=[]
    for h in headers:
        dist=h['main'].replace('U_','D_')+'_'+str(h['level'])
        cs=[]
        for c in s['components']:
            if c['shape']!='cylinder' or not (c.get('owner')==h['edge'] or c['id'].startswith(dist+'_reserve')):continue
            a,b=map(np.array,c['ends'])
            if abs(a[1]-b[1])<EPS and abs(a[2]-b[2])<EPS and abs(a[0]-b[0])>EPS and abs(a[2]-h['header_join'][2])<.15:cs.append(c)
        groups.append((h,cs))
    for i,(h,cs) in enumerate(groups):
        for hh,other in groups[i+1:]:
            if h['level']!=hh['level']:continue
            for c in cs:
                for o in other:
                    ca,cb=map(np.array,c['ends']);oa,ob=map(np.array,o['ends'])
                    if min(max(ca[0],cb[0]),max(oa[0],ob[0]))<max(min(ca[0],cb[0]),min(oa[0],ob[0]))-EPS:continue
                    gap=float(np.linalg.norm(ca[1:]-oa[1:])-c['radius']-o['radius'])
                    is_bare=c['kind']=='pipe' and o['kind']=='pipe';minimum=bare if is_bare else flange
                    row={'pipe':c['id'],'other':o['id'],'kind':'bare_pipe' if is_bare else 'protrusion','gap_mm':round(gap*1000,3),'minimum_mm':minimum*1000,'pass':gap>=minimum-EPS};rows.append(row)
                    if not row['pass']:errors.append({'code':'PARALLEL_HEADER_CLEARANCE',**row})
    if len(headers)>1 and not rows:errors.append({'code':'PARALLEL_HEADERS_NOT_MEASURED'})
    minima={kind:min((r['gap_mm'] for r in rows if r['kind']==kind),default=None) for kind in ('bare_pipe','protrusion')}
    return result(rows,errors,minimum_measured_mm=minima,scope='This single reactor has bare pipes. Measure clearance between branches and actual flange or blind edges. Check lever sweep separately.')


def check(g,s,spec):
    groups={'valve_spaces':valve_spaces(g,s),'station_interfaces':station_flanges(g,s),
            'floor_crossings':floor_crossings(g,s,spec),'shared_top_headers':shared_headers(s),'offgas':offgas_join(g,s),
            'unused_floor_outlets':unused_floor_outlets(g,s),'offgas_diameters':offgas_diameters(g,s),'parallel_headers':parallel_headers(s)}
    groups['offgas_main_connections']=offgas_main_connections(g,s)
    return {'status':'pass' if all(x['status']=='pass' for x in groups.values()) else 'fail',**groups}
