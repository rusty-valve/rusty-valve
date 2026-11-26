"""Identify each valve's purpose and where it belongs on a pipe."""
from copy import deepcopy
import math
import numpy as np

EPS=1e-7
# Position names implement the seven valve-position rules in section 3.
TABLE={
 'bottom_root':('vessel bottom root valve','at_port',0),
 'filter_inlet':('filter inlet valve','at_port',0),
 'filter_outlet':('filter outlet valve','at_port',0),
 'bypass':('bypass valve','free',1),
 'pump_inlet':('pump suction valve','at_port',0),
 'pump_outlet':('pump discharge valve','at_port',0),
 'recycle_regulation':('recirculation control valve','after_split',1),
 'return_root':('reactor recirculation branch root valve','at_port',2),
 'condenser_return_root':('horizontal condenser reactor return root valve','at_port',2),
 'common_root':('shared-port root valve','at_port',3),
 'heater_inlet':('heater material inlet valve','at_port',0),
 'heater_outlet':('heater material outlet valve','at_port',0),
 'condenser_vapor':('condenser vapor root valve','at_port',0),
 'vapor_root':('reactor top vapor root valve','top_vertical',0),
 'condensate_root':('condensate outlet root valve','at_port',0),
 'condensate_regulation':('condensate return control valve','before_seal',1),
 'seal_drain':('liquid-trap low-point drain valve','low_point',0),
 'feed_root':('feed root valve','at_port',0),
 'utility_root':('equipment utility root valve','at_port',0),
 'material_inlet':('equipment material inlet valve','at_port',0),
 'material_outlet':('equipment material outlet valve','at_port',0),
 'drain_root':('drain root valve','at_port',0),
 'gauge_root':('level root valve','at_port',0),
 'utility_floor_root':('floor header-to-main root valve','at_main_tee',0),
}


def describe(graph,valve):
    """Infer semantics from the source process, never from a generated demand list."""
    edge=next(e for e in graph['edges'] if e['id']==valve['edge'])
    at=valve['at'];original=valve.get('original_at',valve.get('source_endpoint',at))
    if at=='branch' and original=='branch':
        key={'bypass':'bypass','seal_drain':'seal_drain','condensate':'condensate_regulation',
             'recycle':'recycle_regulation'}[edge['role']]
    else:
        uid,port=original.split('.');unit=graph['units'][uid];kind=unit['kind']
        if kind=='header':key='utility_floor_root'
        elif valve.get('common_port_root') or (at!='branch' and port=='liquid_return'):key='common_root'
        elif port=='condensate_return' and at!='branch':key='condenser_return_root'
        elif port.startswith(('recycle_','condensate_return')):key='return_root'
        elif kind=='pump':key='pump_inlet' if port=='in' else 'pump_outlet'
        elif kind in ('filter','protective_filter'):key='filter_inlet' if port=='in' else 'filter_outlet'
        elif kind=='exchanger' and port in ('in','out'):key='heater_inlet' if port=='in' else 'heater_outlet'
        elif port=='vapor':key='condenser_vapor' if kind=='condenser' else 'vapor_root'
        elif port=='condensate':key='condensate_root'
        else:key={'bottom':'bottom_root','feed':'feed_root','drain':'drain_root','gauge':'gauge_root',
                  'in':'material_inlet','out':'material_outlet'}.get(port,'utility_root')
    label,position,order=TABLE[key]
    if key=='return_root' and at=='branch':position='before_join'
    if key=='utility_root' and original!='branch':
        uid,port=original.split('.')
        if graph['units'][uid]['kind']=='vessel' and port in ('nitrogen','offgas'):position='top_vertical'
    endpoint=(at if at!='branch' else edge['b'] if position in ('before_join','before_seal') else edge['a'])
    return {'role':key,'valve_role':key,'role_label':label,'placement_rule':position,
            'sequence_rank':order,'position_endpoint':endpoint}


def annotate_graph(graph):
    for valve in graph['valves']:
        if valve.get('manual'):
            valve.update({k:v for k,v in describe(graph,valve).items() if k!='role'})
    return graph


def position_context(piping,valves,actual_routes=None,natives=None,components=None):
    if actual_routes is None:
        geometry={c['id']:c for c in components or ()};actual_routes={}
        for route in piping['routes']:
            actual_routes[route['id']]=[
                {'ends':part['ends']} if 'ends' in part else
                {'ends':[part['ports'][side]['position'] for side in ('in','out')]}
                for name in route.get('components',[]) if (part:=geometry.get(name))
                and ('ends' in part or part.get('kind')=='piping_elbow')]
    routes=[
        {**route,'id':route['id']+':'+str(i),'points':part.get('centerline_points',part['ends']),
         'control_points':part.get('centerline_points',part['ends'])}
        for route in piping['routes'] for i,part in enumerate(actual_routes[route['id']])]
    observed=[]
    for valve in valves:
        ports=(natives or {}).get(valve['id'],{}).get('actual_ports')
        ports=ports or {k:v['position'] for k,v in valve.get('world_ports',{}).items()}
        if not ports:continue
        observed.append({**valve,'measured_ports':ports})
        edge=valve.get('edge',valve.get('source_requirement',valve.get('connection_id')))
        routes=[*routes,{'id':'native:'+valve['id'],'source_edge':edge,'points':list(ports.values())}]
    routes=[{**r,'source_edge':r.get('source_edge',r.get('source_requirement',r.get('connection_id')))} for r in routes]
    return {**piping,'routes':routes,'observed_valves':observed}


def placement_key(a,requirement,inlet,outlet,fraction,context=None):
    context=context or position_context({'routes':a.routes,'elbows':a.scene.get('elbows',[]),
        'material':a.material},a.all_valves(),components=a.scene['components'])
    result=position_check(a.frozen['graph'],requirement['source_valve'],context,
                          {'in':inlet,'out':outlet},requirement)
    return result['score'] if result['valid'] else math.inf


def _endpoint(graph,piping,endpoint):
    uid,port=endpoint.split('.')
    physical=graph['units'][uid]['ports'].get(port,{})
    if 'pos' in physical:return np.asarray(physical['pos'])
    joint=next((j for j in piping.get('material',{}).get('junctions',[]) if j['id']==uid),None)
    return np.asarray(joint['position']) if joint else None


def _progress(piping,edge,center,origin=None):
    """Cache an oriented edge metric; every candidate/marker uses that same metric."""
    import heapq
    cache=piping.setdefault('_edge_metrics',{});token=(edge,None if origin is None else tuple(origin))
    if token not in cache:
        lines=[(np.asarray(a),np.asarray(b),r['id']) for r in piping['routes']
            if r.get('source_edge',r.get('source_requirement'))==edge
            for a,b in zip(r.get('control_points',r.get('points',[])),r.get('control_points',r.get('points',[]))[1:])
            if math.dist(a,b)>EPS]
        if not lines:return math.inf,None,None
        points=[p for a,b,_ in lines for p in (a,b)];key=lambda p:tuple(np.round(p,7))
        start=min(points,key=lambda p:math.dist(p,origin)) if origin is not None else points[0]
        adjacency={key(p):[] for p in points}
        for one,two,_ in lines:
            delta=two-one;length=float(np.linalg.norm(delta));cuts=[]
            for p in points:
                t=float((p-one)@delta)/(length*length)
                if -EPS<=t<=1+EPS and np.linalg.norm(p-one-t*delta)<EPS:cuts.append((t,key(p)))
            cuts=sorted(set(cuts))
            for (a,u),(b,v) in zip(cuts,cuts[1:]):
                adjacency[u].append((v,(b-a)*length));adjacency[v].append((u,(b-a)*length))
        distances={key(start):math.dist(start,origin) if origin is not None else 0.};queue=[(distances[key(start)],key(start))]
        while queue:
            distance,node=heapq.heappop(queue)
            if distance!=distances[node]:continue
            for peer,length in adjacency[node]:
                new=distance+length
                if new<distances.get(peer,math.inf):distances[peer]=new;heapq.heappush(queue,(new,peer))
        cache[token]=(lines,distances,{})
    lines,distances,positions=cache[token];location=tuple(center)
    if location in positions:return positions[location]
    choices=[]
    for one,two,route in lines:
        delta=two-one;length=float(np.linalg.norm(delta));t=float((center-one)@delta)/(length*length)
        if -EPS<=t<=1+EPS:
            error=float(np.linalg.norm(center-one-t*delta))
            value=min(distances.get(tuple(np.round(one,7)),math.inf)+t*length,
                      distances.get(tuple(np.round(two,7)),math.inf)+(1-t)*length)
            choices.append((error,value if math.isfinite(value) else None,route))
    result=min(choices,key=lambda x:x[0]) if choices else (math.inf,None,None)
    positions[location]=result if result[0]<=EPS else (result[0],None,result[2])
    return positions[location]


def _downward_entry(piping, edge_id, origin):
    route_ids = {r['id'].split(':')[0] for r in piping['routes']
                 if r.get('source_edge', r.get('source_requirement')) == edge_id}
    entries = []
    for elbow in piping.get('elbows', []):
        ports = elbow['ports']
        if (elbow['route_id'] in route_ids and ports['out']['tangent'][2] < -.9
                and abs(ports['in']['tangent'][2]) < EPS):
            point = ports['in']['position']
            _, progress, _ = _progress(piping, edge_id, np.asarray(point), origin)
            if progress is not None:
                entries.append((progress, elbow['id'], point))
    return min(entries, default=(None, None, None))


def position_reference(graph, source, piping, policy):
    rule = policy.get('placement_rule')
    edge_id = source['edge']
    edge = next((e for e in graph['edges'] if e['id'] == edge_id), None)
    if rule in ('at_port', 'top_vertical'):
        if policy['role'] == 'bottom_root':
            side = next((r for r in piping['routes'] if r.get('source_edge') == edge_id
                         and r.get('role') == 'vessel_bottom_side'), None)
            if side:
                elbow = next(e for e in piping['elbows'] if e['route_id'] == side['id'].split(':')[0])
                return elbow['ports']['out']['position']
        return _endpoint(graph, piping, source['at'])
    if edge and rule == 'before_seal':
        return _downward_entry(piping, edge_id, _endpoint(graph, piping, edge['a']))[2]
    if edge and rule in ('before_join', 'after_split', 'low_point', 'free'):
        return _endpoint(graph, piping, edge['b' if rule == 'before_join' else 'a'])
    return policy.get('main_tee_m') if rule == 'at_main_tee' else None


def position_check(graph,source,piping,ports,policy=None,*,allow_at_port_elbow=False,partial=False):
    """v2.1: test intervening fittings, tees and valves, without a distance cutoff."""
    policy=policy or describe(graph,source);rule=policy['placement_rule'];edge_id=source['edge']
    edge=next((e for e in graph['edges'] if e['id']==edge_id),None)
    origin = _endpoint(graph, piping, edge['a']) if edge else None
    reverse_progress = False
    if partial and edge and policy.get('position_endpoint') == edge['b'] and rule in ('at_port', 'top_vertical', 'before_join'):
        origin = _endpoint(graph, piping, edge['b'])
        reverse_progress = True
    center=np.mean(list(ports.values()),axis=0);_,point,_=_progress(piping,edge_id,center,origin)
    evidence={'valid':False,'score':None,'edge_progress_m':point}
    if point is None:return evidence
    values=[_progress(piping,edge_id,np.asarray(p),origin)[1] for p in ports.values()]
    evidence={'valid':False,'rule':rule,'edge_progress_m':point,'score':math.inf,'between':[]}
    if point is None or any(v is None for v in values):return evidence
    events=[];tees=piping.get('material',{}).get('junctions',[])
    for joint in tees:
        _,value,_=_progress(piping,edge_id,np.asarray(joint['position']),origin)
        if value is not None:events.append((value,'tee',joint['id']))
    route_ids={r['id'].split(':')[0] for r in piping['routes'] if r.get('source_edge',r.get('source_requirement'))==edge_id}
    for elbow in piping.get('elbows',[]):
        if elbow['route_id'] not in route_ids:continue
        for port in elbow['ports'].values():
            _,value,_=_progress(piping,edge_id,np.asarray(port['position']),origin)
            if value is not None:events.append((value,'elbow',elbow['id']))
    for valve in piping.get('observed_valves',[]):
        if valve.get('source_valve',valve['id'])==source['id'] or valve['id']==source['id']:continue
        q=np.mean(list(valve['measured_ports'].values()),axis=0);_,value,_=_progress(piping,edge_id,q,origin)
        if value is not None:events.append((value,'valve',valve.get('source_valve',valve['id'])))
    target = center if rule == 'free' else position_reference(graph, source, piping, policy)
    _,reference,_=_progress(piping,edge_id,np.asarray(target),origin) if target is not None else (None,None,None)
    if reference is None:return evidence
    low,high=sorted((point,reference))
    between=[(v,kind,name) for v,kind,name in events if low+EPS<v<high-EPS]
    evidence.update(reference_m=np.asarray(target).tolist(),reference_progress_m=reference,
                    between=[{'type':k,'id':n,'progress_m':v} for v,k,n in between])
    forbidden={'valve','tee'}|({'elbow'} if rule in ('at_port','top_vertical','at_main_tee') else set())
    if rule in ('at_port', 'top_vertical', 'at_main_tee') and allow_at_port_elbow:
        forbidden.discard('elbow')
    permitted=set()
    heater=(rule=='after_split' and edge and graph['units'][edge['a'].split('.')[0]]['kind']=='exchanger')
    if heater:permitted={v['id'] for v in graph.get('valves',[]) if v.get('edge')==edge_id and describe(graph,v)['role']=='heater_outlet'}
    valid=not any(kind in forbidden and name not in permitted for _,kind,name in between)
    if rule=='top_vertical':valid=valid and abs(np.asarray(ports['out'])[2]-ports['in'][2])>math.dist(*ports.values())*.99
    if rule in ('before_join','before_seal'):
        valid = valid and (min(values) >= reference-EPS if reverse_progress else max(values) <= reference+EPS)
    if rule=='after_split':valid=valid and min(values)>=reference-EPS and (not heater or any(n in permitted for _,_,n in between))
    if rule=='before_seal':evidence['first_downward_elbow']=_downward_entry(piping, edge_id, origin)[1]
    if rule=='low_point':
        seal=edge['a'].split('.')[0];main_edges={e['id'] for e in graph['edges'] if e['role']!='seal_drain'
            and seal in (e['a'].split('.')[0],e['b'].split('.')[0])}
        heights=[p[2] for r in piping['routes'] if r.get('source_edge') in main_edges for p in r.get('control_points',r.get('points',[]))]
        valid=valid and bool(heights) and abs(target[2]-min(heights))<EPS
        evidence['lowest_main_z_m']=min(heights) if heights else None
    evidence.update(valid=bool(valid),score=abs(point-reference))
    return evidence


def check(original_graph,graph,scene,piping,model,data,*,requirements=None):
    """Expected valves come from the original graph; poses come from compiled XML."""
    from . import piping_checks as common
    actual=common._measured(model,data)
    native,_,_,_=common._valves(graph,actual)
    from .pipe_spec import load
    components={c['id']:c for c in scene['components']}
    actual_routes,route_errors,_=common._route_segments(piping['routes'],components,actual,
        load()['pipe_outer_diameter_mm'],model,data)
    natives={r['id']:r for r in native if r['status']=='pass'}
    def edge_segments(edge):
        parts=[s for route in piping['routes'] if route.get('source_edge',route.get('source_requirement'))==edge
               for s in actual_routes[route['id']]]
        parts += [{'ends':list(natives[v['id']]['actual_ports'].values())} for v in graph['valves']
                  if v['id'] in natives and v.get('edge',v.get('source_requirement'))==edge]
        return parts
    review_piping=position_context(piping,graph['valves'],actual_routes,natives)
    originals=[v for v in original_graph['valves'] if v.get('manual') and
               not (v['at']!='branch' and original_graph['units'][v['at'].split('.')[0]]['kind']=='header')]
    rows=[];errors=[];centers={};expected={v['id']:describe(original_graph,v) for v in originals}
    for source in originals:
        policy=expected[source['id']]
        edge=next(e for e in original_graph['edges'] if e['id']==source['edge'])
        origin=_endpoint(graph,piping,edge['a'])
        found=[v for v in graph['valves'] if v.get('source_valve')==source['id']]
        if not found:found=[v for v in graph['valves'] if not v.get('source_valve') and
            v.get('at')==source['at'] and v.get('edge',v.get('source_requirement'))==source['edge']]
        row={'source_valve':source['id'],'source_edge':source['edge'],'source_at':source['at'],
             'source_role':source.get('role'),**policy,'actual_valve':None,'role_status':'fail',
             'position_status':'fail','order_status':'pass','errors':[]}
        if len(found)!=1 or found[0]['id'] not in natives:
            row['errors'].append('Missing a unique native valve package verified against the actual XML')
        else:
            valve=found[0];measured=natives[valve['id']];row['actual_valve']=valve['id']
            row['actual_ports']=deepcopy(measured['actual_ports'])
            center=np.mean(list(measured['actual_ports'].values()),axis=0);centers[source['id']]=center
            row['actual_center_m']=center.tolist();row['actual_role']=valve.get('valve_role',valve.get('role'))
            role_ok=(row['actual_role']==policy['role'] and valve.get('edge',valve.get('source_requirement'))==source['edge'])
            row['role_status']='pass' if role_ok else 'fail'
            if not role_ok:row['errors'].append('Valve role or process edge differs from the source graph')
            position=position_check({**original_graph,'units':graph['units']},source,review_piping,
                                    measured['actual_ports'],policy,
                                    allow_at_port_elbow=piping.get('valve_placement')=='role_trace')
            row['position_evidence']=position
            row['edge_progress_m']=position['edge_progress_m']
            position_ok=position['valid']
            row['position_status']='pass' if position_ok else 'fail'
            if not position_ok:row['errors'].append('Measured valve position violates its role rule')
        rows.append(row)
    by_source={r['source_valve']:r for r in rows}
    # Rank on each actual edge, oriented from its original a endpoint. A root
    # on the receiving device is last, irrespective of native package axis.
    edge_orders={}
    for edge in original_graph['edges']:
        local=[v for v in originals if v['edge']==edge['id']]
        def rank(v):
            return (4 if v['at']==edge['b'] else expected[v['id']]['sequence_rank'],v['id'])
        local.sort(key=rank);edge_orders[edge['id']]=local
        projected={v['id']:_progress(review_piping,edge['id'],centers[v['id']],_endpoint(graph,piping,edge['a']))
            if v['id'] in centers else (math.inf,None,None) for v in local}
        for one,two in zip(local,local[1:]):
            a,b=projected[one['id']],projected[two['id']]
            ok=a[1] is not None and b[1] is not None and a[1]<b[1]-EPS
            proof={'before':one['id'],'after':two['id'],'from_process_endpoint':edge['a'],
                   'before_distance_m':a[1],'after_distance_m':b[1],'status':'pass' if ok else 'fail'}
            for source in (one,two):
                row=by_source[source['id']];row.setdefault('order_evidence',[]).append(proof)
                if not ok:row['order_status']='fail';row['errors'].append('Actual valve order differs from the required order')
    chains=[]
    def downstream(path):
        edge=path[-1]
        if original_graph['units'][edge['b'].split('.')[0]]['kind']=='vessel':return path
        target=edge['b'].split('.')[0]
        if original_graph['units'][target]['kind']!='junction':return None
        for next_edge in original_graph['edges']:
            if (next_edge['a'].split('.')[0]==target and next_edge['role']!='seal_drain'
                    and next_edge['id'] not in {e['id'] for e in path}):
                found=downstream([*path,next_edge])
                if found:return found
    for source in originals:
        if expected[source['id']]['role'] not in ('heater_outlet','condensate_root'):continue
        start=next(e for e in original_graph['edges'] if e['id']==source['edge'])
        path=downstream([start])
        if not path:continue
        ordered=[v['id'] for edge in path for v in edge_orders.get(edge['id'],[])]
        proof=[{'source_valve':name,'actual_valve':by_source[name]['actual_valve'],
                'actual_ports':by_source[name].get('actual_ports'),
                'edge':by_source[name]['source_edge'],'edge_progress_m':by_source[name].get('edge_progress_m'),
                'position_status':by_source[name]['position_status'],'order_status':by_source[name]['order_status']}
               for name in ordered]
        connected=[]
        for edge in path:
            first,last=(_endpoint(graph,piping,edge[side]) for side in ('a','b'))
            segments=edge_segments(edge['id'])
            ok=first is not None and last is not None and common._connected(first,[last],segments)
            connected.append({'edge':edge['id'],'from':edge['a'],'to':edge['b'],
                'actual_segment_count':len(segments),'status':'pass' if ok else 'fail'})
        joints=[]
        for one,two in zip(path,path[1:]):
            p1=_endpoint(graph,piping,one['b']);p2=_endpoint(graph,piping,two['a'])
            joints.append({'from_edge':one['id'],'to_edge':two['id'],
                'junction':one['b'].split('.')[0],'point_m':p1.tolist() if p1 is not None else None,
                'status':'pass' if p1 is not None and p2 is not None and math.dist(p1,p2)<EPS else 'fail'})
        chains.append({'source':source['at'],'expected_valves':ordered,'process_edges':[e['id'] for e in path],
            'observed_valves':proof,'junctions':joints,'actual_edge_connections':connected,
            'status':'pass' if all(all(by_source[name][key]=='pass' for key in ('role_status','position_status','order_status'))
                for name in ordered) and all(j['status']=='pass' for j in joints+connected) else 'fail'})
    from .piping_end_bundle import DEMAND_ONLY_TERMINALS, station_root_keys
    bundle = piping.get('end_bundle_station',{})
    demand_only = bundle.get('terminals_policy')==DEMAND_ONLY_TERMINALS
    if demand_only:
        frozen_requirements = requirements if requirements is not None else piping['requirements']
        floor_expected = set(station_root_keys(frozen_requirements,bundle,len(scene['floors'])))
    else:
        services={u['service'] for u in original_graph['units'].values()
                  if u.get('kind')=='header' and u.get('service')!='offgas'}
        floor_expected={(service,level) for service in services for level in range(len(scene['floors']))}
    floors=[v for v in graph['valves'] if v.get('role')=='utility_floor_root']
    floor_counts={key:sum(v.get('medium')==key[0] and v.get('level')==key[1] and v['id'] in natives for v in floors) for key in floor_expected}
    for key,count in sorted(floor_counts.items()):
        own=[v for v in floors if (v.get('medium'),v.get('level'))==key]
        row={'source_valve':f'floor:{key[0]}:{key[1]}','role':'utility_floor_root','role_label':TABLE['utility_floor_root'][0],
             'placement_rule':'at_main_tee','actual_valve':own[0]['id'] if len(own)==1 else None,
             'actual_role':own[0].get('valve_role',own[0].get('role')) if len(own)==1 else None,
             'role_status':'pass' if count==1 else 'fail','position_status':'pass' if count==1 else 'fail',
             'order_status':'pass','errors':[] if count==1 else ['The source service and station level require one root valve']}
        if len(own)==1 and own[0]['id'] in natives:
            valve=own[0];ports=natives[valve['id']]['actual_ports'];row['actual_ports']=deepcopy(ports)
            connection=next((c for c in piping.get('end_bundle_station',{}).get('connections',[])
                             if c['id']==valve.get('connection_id')),None)
            inlet=np.asarray(ports['in']);outlet=np.asarray(ports['out']);axis=(outlet-inlet)/np.linalg.norm(outlet-inlet)
            if connection:
                tap=np.asarray(connection['main_tap']);row['main_tee_m']=tap.tolist()
                row['tee_to_valve_m']=float(np.linalg.norm(inlet-tap))
                main=actual_routes.get(valve.get('main_id'),[])
                lead=[s for route in piping['routes'] if route.get('connection_id')==connection['id']
                      and route.get('target',{}).get('valve')==valve['id'] for s in actual_routes[route['id']]]
                on_main=any(common._on_segment(tap,*segment['ends']) for segment in main)
                connected=common._connected(tap,[inlet],lead)
                collinear=np.linalg.norm(np.cross(inlet-tap,axis))<EPS and float((inlet-tap)@axis)>=-EPS
                row['position_evidence']={'tee_on_actual_main':on_main,'actual_pipe_reaches_valve_in':connected,
                    'valve_collinear_after_tee':bool(collinear),'actual_lead_segments':len(lead)}
                row['position_status']='pass' if on_main and connected and collinear else 'fail'
            else:row['position_status']='fail'
            shared=position_check(original_graph,{'id':valve['id'],'edge':valve.get('connection_id')},review_piping,ports,
                {'placement_rule':'at_main_tee','main_tee_m':row.get('main_tee_m')})
            row['shared_position_rule']=shared
            if not shared['valid']:row['position_status']='fail'
            if row['position_status']=='fail':row['errors'].append('Measured pipe axis, main tee or valve straight does not establish placement at the tee')
        rows.append(row)
    for row in rows:
        row['status']='pass' if all(row[k]=='pass' for k in ('role_status','position_status','order_status')) else 'fail'
        errors.extend({'code':'VALVE_ROLE_POSITION_ORDER','valve':row['source_valve'],'detail':message} for message in row['errors'])
    errors.extend({'code':'VALVE_CHAIN_ORDER','source':c['source']} for c in chains if c['status']=='fail')
    extra=[v['id'] for v in graph['valves'] if v['id'] not in {r.get('actual_valve') for r in rows}]
    errors.extend({'code':'UNEXPECTED_MANUAL_VALVE','valve':name} for name in extra)
    return {'status':'fail' if errors else 'pass','expected':len(rows),'verified':sum(r['status']=='pass' for r in rows),
            'rows':rows,'chains':chains,'errors':errors,'unexpected_valves':extra,
            'expectation_source':('Use the original graph, valve roles and physical service users on each floor. Direct merge branches have no main root valve.' if demand_only
                                  else 'original process graph + valve_roles TABLE + original utility services × station floors'),
            'position_source':'native valve ports measured from compiled XML; physical nozzle/routed-junction references'}
