"""Route the process network and fit its required valves."""
from copy import deepcopy
import math
import numpy as np

from . import piping_material_build as b
from .piping_assets import inspect_asset, place_asset
from .piping_elbows import plan_polyline
from .piping_build import _native_contacts
from .piping_valve_placement import obstacle_boxes, overlapping_obstacle_ids
from .valve_roles import placement_key,position_context

EPS = 1e-8


def _ports(a, edges):
    from .generation_progress import checkpoint
    a.deferred_valves = list(a.requirements['manual_valves'])
    if getattr(a, 'before_material_ports', None):
        a.before_material_ports(a)
    for edge in edges:
        with drain_bounds(a, edge):
            for endpoint in (edge['a'], edge['b']):
                if endpoint not in a.requirements['ports'] or endpoint in a.anchors:
                    continue
                port = {**a.requirements['ports'][endpoint], 'endpoint': endpoint}
                checkpoint(a, {'stage':'material_interface','edge':edge['id'],'endpoint':endpoint})
                try:
                    anchor = b._adapter(a, port, edge)
                    anchor['point'] = np.asarray(anchor['adapter_end'])
                    from .vessel_bottom import applies,emit
                    unit=a.frozen['graph']['units'][port['equipment']]
                    if applies(unit,endpoint):
                        anchor=emit(a,unit,port,anchor,edge)
                except b.KNOWN_REFUSALS as exc:
                    a.material.setdefault('network_failures', []).append({
                        'stage':'material_interface','edge':edge['id'],'endpoint':endpoint,'reason':str(exc)})
                    continue
                a.anchors[endpoint] = anchor


from contextlib import contextmanager


@contextmanager
def drain_bounds(a, edge):
    original = a.bounds
    ditch_opening=None
    if edge.get('role') == 'steam_condensate':
        points = [a.requirements['ports'][ep]['position'] for ep in (edge['a'],edge['b'])
                  if ep in a.requirements.get('ports', {})]
        if points:
            radius=a.radius(edge['dn'])
            a.bounds = expanded_drain_bounds(original, points, radius)
            target=a.requirements['ports'].get(edge['b'],{})
            if target.get('network_boundary') and target.get('normal')==[0.,0.,1.]:
                x,y,z=target['position']
                a.bounds[2]=min(a.bounds[2],z-radius)
                # The external open ditch is outside the actual ground slab.
                # Give only that point the clearance for a flat-ended drop.
                reach=radius+.05
                ditch_opening={'level':0,'rect':[x-reach,y-reach,x+reach,y+reach],
                               'external_ditch_endpoint':edge['b']}
                a.openings.append(ditch_opening)
    try:
        yield
    finally:
        a.bounds = original
        if ditch_opening is not None and ditch_opening in a.openings:
            a.openings.remove(ditch_opening)


def expanded_drain_bounds(bounds, points, radius):
    extended = np.array(bounds, dtype=float)
    margin = 10*radius + .1
    extended[:2] = np.minimum(extended[:2], np.min(points,axis=0)[:2]-margin)
    extended[3:5] = np.maximum(extended[3:5], np.max(points,axis=0)[:2]+margin)
    return extended

def _remove_routes(a, routes):
    ids = {r['id'] for r in routes}; parts = {c for r in routes for c in r['components']}
    a.routes = [r for r in a.routes if r['id'] not in ids]
    a.scene['components'] = [c for c in a.scene['components'] if c['id'] not in parts]
    a.scene['elbows'] = [r for r in a.scene.get('elbows', []) if r['route_id'] not in ids]
    a.allowed_contacts = [r for r in a.allowed_contacts if not parts.intersection(r['pair'])]


def _planned_points(a, edge, start, end, points):
    merged=[np.asarray(p).tolist() for p in points]
    for side,anchor in enumerate((start,end)):
        for route in a.routes:
            if (route.get('source_edge')==edge['id'] and route['id'].endswith('_normal_departure')
                    and math.dist(route['points'][-1],anchor['point'])<=EPS):
                tail=route['control_points']
                merged=tail+merged[1:] if side==0 else merged+list(reversed(tail))[1:]
    return b._bridge_with_root_tails(a,edge,start,end,merged)[0]


def _endpoint_search_options(a,start,end):
    a._network_contact_points=[np.asarray(v['point']) for v in (start,end)]
    ids=set()
    for anchor in (start,end):
        ids.update(b._adjacent_parts(a,anchor,anchor['point']))
        if 'adapter_end' in anchor and not anchor.get('departure_contact'):
            # The short nozzle/pump stub is already joined to this reducer's
            # inlet. End-to-end attachment is not parallel-pipe clearance.
            # Exclude only its exact contact pieces, never the equipment body.
            ids.update(b._joint_components(a,anchor['physical_port']['position']))
    return {'skip':tuple(sorted(ids)), 'port_contacts':b._departure_contacts(start,end)}


def _riser_candidates(a,edge,start,end,failures):
    """Jointly test a vertical leg and both end approaches around real solids."""
    first,last=np.asarray(start['point']),np.asarray(end['point']);r=a.radius(edge['dn']);R=3*r
    low=first[2]+R+.05;high=last[2]-R-.05
    if high<=low:return []
    coordinates=[]
    for c in a.scene['components']:
        box=c['box']
        if c['kind'] not in ('drive_reservation','maintenance_reservation') and 'drive_reservation' not in c['id']:continue
        if box[2]>last[2] or box[5]<first[2]:continue
        for y in (box[1]-r-.05,box[4]+r+.05):
            for x in (first[0],last[0]):coordinates.append((x,y))
    coordinates += [(x,y) for x in (first[0],last[0]) for y in (min(first[1],last[1])-6*r-.1,max(first[1],last[1])+6*r+.1)]
    planes=set()
    for obstacle in a.obstacles():
        box=obstacle['box']
        if box[5]<first[2] or box[2]>last[2] or box[3]<min(first[0],last[0])-r or box[0]>max(first[0],last[0])+r:continue
        for y in (box[1]-r-.05,box[4]+r+.05):
            if r+.1<y<a.bounds[4]-r-.1:planes.add(float(y))
    separated=[]
    for y in sorted(planes,key=lambda y:abs(y-first[1])+abs(y-last[1])):
        if all(abs(y-other)>2*r+.05 for other in separated):separated.append(y)
    coordinates += [(x,y) for y in separated[:20] for x in (first[0],last[0])]
    results=[];search_options=_endpoint_search_options(a,start,end)
    for xy in list(dict.fromkeys(coordinates))[:48]:
        state=a.snapshot();bounds=a.bounds
        try:
            one=np.r_[xy,low];two=np.r_[xy,high]
            for level,z in enumerate(a.floors[1:],1):
                if low<z<high:b._opening(a,np.r_[xy,z],level,edge['dn'],'P2_'+edge['id']+'_joint_riser_'+str(level))
            a.search(one,two,edge['dn'],straight=True)
            a.bounds=np.asarray(bounds).copy();a.bounds[2]=first[2]-r;a.bounds[5]=low+r
            left=a.search(first,one,edge['dn'],**search_options,start_direction=start['normal'],complete_bend_planes=True)
            a.bounds=np.asarray(bounds).copy();a.bounds[2]=high-r;a.bounds[5]=last[2]+r
            right=a.search(two,last,edge['dn'],**search_options,end_direction=-np.asarray(end['normal']),complete_bend_planes=True)
            points=left+right;plan=plan_polyline(_planned_points(a,edge,start,end,points),r)
            length=sum(math.dist(x,y) for x,y in zip(plan['points'],plan['points'][1:]))
            results.append((len(plan['elbows']),length,points,a.snapshot()))
        except b.KNOWN_REFUSALS as exc:failures.append(b._why(exc))
        finally:a.bounds=bounds;a.restore(state)
    return results


def _horizontal_lift_candidates(a,edge,start,end,failures):
    """A shared horizontal plane plus both real vertical approaches, as one plan."""
    first,last=np.asarray(start['point']),np.asarray(end['point']);r=a.radius(edge['dn']);R=3*r
    search_options=_endpoint_search_options(a,start,end);results=[]
    # A transverse U on the shared plane resolves almost-collinear XY mouths
    # without fabricating a tiny elbow or introducing an up/down arch.
    axis=1-int(np.argmax(abs(last[:2]-first[:2])))
    for height in (first[2]+R+.05,last[2]-R-.05):
      for sign in (1,-1):
        state=a.snapshot();bounds=a.bounds
        try:
            one=first.copy();one[2]=height;two=last.copy();two[2]=height
            for point,lo,hi in ((first,min(first[2],height),max(first[2],height)),(last,min(last[2],height),max(last[2],height))):
                for level,z in enumerate(a.floors[1:],1):
                    if lo<z<hi:b._opening(a,np.r_[point[:2],z],level,edge['dn'],'P2_'+edge['id']+'_plane_lift_'+str(level))
            a.search(first,one,edge['dn'],**search_options,straight=True)
            a.search(two,last,edge['dn'],**search_options,straight=True)
            a.bounds=np.asarray(bounds).copy();a.bounds[2]=height-r;a.bounds[5]=height+r
            direction=np.zeros(3);direction[axis]=sign
            horizontal=a.search(one,two,edge['dn'],**search_options,start_direction=direction,end_direction=-direction,
                start_straight_m=R+.05,end_straight_m=R+.05,complete_bend_planes=True)
            points=[first.tolist()]+horizontal+[last.tolist()]
            plan=plan_polyline(_planned_points(a,edge,start,end,points),r)
            length=sum(math.dist(x,y) for x,y in zip(plan['points'],plan['points'][1:]))
            results.append((len(plan['elbows']),length,points,a.snapshot()))
        except b.KNOWN_REFUSALS as exc:failures.append('shared horizontal plane '+str(height)+': '+b._why(exc))
        finally:a.bounds=bounds;a.restore(state)
    return results


def _fit_on_network(a, req):
    """Split a real straight run around the original native package; no bypass."""
    asset=inspect_asset(req['dn'],req.get('operator')); L=asset['install_length_m']; r=a.radius(req['dn'])
    root=req['at'] in a.requirements['ports']; port=a.requirements['ports'].get(req['at'])
    linked=[route for route in a.routes if route.get('source_edge')==req['edge']]
    components={c['id']:c for c in a.scene['components']}
    candidates=[]
    context=position_context({'routes':a.routes,'elbows':a.scene.get('elbows',[]),'material':a.material},a.all_valves(),components=a.scene['components'])
    from .vessel_bottom import applies,beyond_side
    unit=a.frozen['graph']['units'].get(port['equipment'],{}) if root else {}
    side_root=root and applies(unit,req['at'])
    for route in linked:
        if side_root and route['role']!='vessel_bottom_side':continue
        for cid in route.get('pipe_components',[]):
            part=components[cid]
            one,two=map(np.asarray,part['ends']); length=math.dist(one,two)
            if length+EPS<3*L+.02:continue
            axis=b._unit(two-one)
            if root and math.dist(one,port['position'])>math.dist(two,port['position']):
                one,two=two,one;axis=-axis
            if root and port['normal'][2]>.9:
                delta = one-np.asarray(port['position'])
                if abs(axis[2])<.9 or np.linalg.norm(delta[:2])>EPS or delta[2]<-EPS:
                    continue
            obstacle_ids, boxes = obstacle_boxes(a.obstacles(skip=(cid,)))
            for fraction in (0.,.25,.5,.75,1.):
                gap=L+.01+(length-3*L-.02)*fraction; inlet=one+axis*gap
                if side_root and not beyond_side(unit,inlet,axis):continue
                score=placement_key(a,req,inlet,inlet+axis*L,fraction,context)
                if not math.isfinite(score):continue
                for stem in ([0,0,1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]):
                    if abs(float(axis@stem))>EPS:continue
                    initial=place_asset(asset,[0,0,0],axis,stem)
                    origin=inlet-initial['world_ports']['in']['position']; valve=place_asset(asset,origin,axis,stem)
                    identifier='P2_'+req['id']
                    valve.update(id=identifier,name=req.get('source_role',req['id']),prefix=identifier+'_',geom_prefix=identifier+'_',
                        role=req['role'],dn=req['dn'],medium=req['medium'],medium_identity=req['medium'],
                        level=route['level'],equipment=port['equipment'] if port else None,edge=req['edge'],source_edge=req['edge'],
                        source_valve=req['id'],at=req['at'],phase='P2',manual=True,
                        **{key:req[key] for key in ('valve_role','role_label','placement_rule','sequence_rank','position_endpoint')})
                    from .rule_identity import annotate_valve
                    preference=annotate_valve(a,valve,port)
                    # This very pipe is replaced by the real valve gap; all
                    # other pipes, packages and structures remain obstacles.
                    if overlapping_obstacle_ids(valve['world_geoms'], obstacle_ids, boxes):continue
                    candidates.append(((score,*preference),len(candidates),valve,route,part))
    if not candidates:raise b.PipingRefused(req['id']+' no actual straight with native L and end installation spaces')
    _,_,valve,route,part=min(candidates,key=lambda x:x[:2]);cid=part['id'];axis=np.asarray(valve['axis'])
    vin=np.asarray(valve['world_ports']['in']['position']);vout=np.asarray(valve['world_ports']['out']['position'])
    one,two=map(np.asarray,part['ends'])
    if float((two-one)@axis)<0:one,two=two,one
    a.scene['components'].remove(part)
    old_contacts=[c for c in a.allowed_contacts if cid in c['pair']]
    a.allowed_contacts=[c for c in a.allowed_contacts if cid not in c['pair']]
    pieces=[]
    for side,x,y in (('in',one,vin),('out',vout,two)):
        piece=a.pipe(route['id']+'_'+req['id']+'_valve_'+side,x,y,req['dn'],route['medium'],route['id'])
        piece.update(level=valve['level']);pieces.append(piece)
    for key in ('components','pipe_components'):
        index=route[key].index(cid);route[key][index:index+1]=[p['id'] for p in pieces]
    for contact in old_contacts:
        point=np.asarray(contact['point']);new=pieces[0] if np.linalg.norm(point-one)<=np.linalg.norm(point-two) else pieces[1]
        a.allowed_contacts.append({**contact,'pair':[new['id'] if c==cid else c for c in contact['pair']]})
    a.graph['valves'].append(valve);route['valves'].append(valve['id'])
    for piece,point in zip(pieces,(vin,vout)):_native_contacts(a,valve,piece['id'],point)
    valve['network_placement']={'stage':'after_all_material_routes','replaced_pipe':cid,
        'edge':req['edge'],'source_run_ends':[one.tolist(),two.tolist()],
        'installation_length_m':L,'ordinary_valve_was_not_routing_anchor':not root}
    a.material.setdefault('post_route_valves',[]).append({'valve':valve['id'],**valve['network_placement']})


def extend_network(a):
    """Material pipes by the one connection logic, then valves on the built straights."""
    a._reservation_ceiling=float(a.bounds[5])
    edges=deepcopy([e for e in a.requirements['edges'] if e.get('execution') not in ('deferred_failure','pre_emitted','emission_failed')
        and e['role'] not in ('emergency_discharge','safety_discharge')])
    a.material['generation_order']={'version':'one_connection_logic_v1','all_material_rebuilt':True,
        'stages':['prepare_port_adapters','main_runs_port_to_port','branches_onto_built_pipes','fit_all_valves_on_actual_runs'],
        'scope':'tees are selected on built pipes; valve space ranks routes; valves are fitted afterwards'}
    from .generation_progress import checkpoint
    from .piping_connect import connect_network
    if hasattr(a, 'path_requirements'):
        from .route_requirements import edge_valves, route_requirements
        services = a.frozen.get('pending_service_points', ())
        live_valves = edge_valves(a.frozen['graph'], a.requirements['manual_valves'], services)
        levels = {p['endpoint']: p['level'] for p in services}
        levels.update({name: record['level'] for name, record in a.requirements['ports'].items()})
        a.path_requirements = route_requirements(a.frozen['graph'], live_valves, levels)
        a.material['material_route_requirements'] = deepcopy(a.path_requirements)
        a.material['generation_order']['scope'] = 'run and tee topology built before valve placement'
    _ports(a,edges)
    terminals={r['id']:r for r in a.requirements['terminals']}
    runs=connect_network(a,edges,terminals)
    if hasattr(a, 'path_requirements'):
        a.material['generation_order']['stages'][-1] = 'defer_valves_to_role_based_search'
    deferred = [] if hasattr(a, 'path_requirements') else a.deferred_valves
    for req in sorted(deferred,key=lambda v:(v['at'] not in a.requirements['ports'],
            0 if v['placement_rule']=='before_join' else 1,v['id'])):
        checkpoint(a,{'stage':'material_valve','valve':req['id'],'edge':req['edge']})
        try:
            _fit_on_network(a,req)
        except b.KNOWN_REFUSALS as exc:
            a.material.setdefault('network_failures',[]).append({'code':'NETWORK_VALVE_FAILED','stage':'material_valve','edge':req['edge'],'valve':req['id'],'reason':str(exc)})
        if getattr(a,'execution_progress',None):a.execution_progress({'stage':'valve_after_routes','valve':req['id']})
    for terminal in a.material['terminals']:
        terminal['root']=next((v['id'] for v in a.graph['valves'] if v.get('source_valve')==terminal['root_source_valve']),None)
    for edge in edges:
        linked=[r for r in a.routes[a.p2_route_start:] if r.get('source_edge')==edge['id']]
        a.material['edge_routes'].append({'source_edge':edge['id'],'a':edge['a'],'b':edge['b'],
            'routes':[r['id'] for r in linked],'components':[c for r in linked for c in r['components']],
            'valves':list(dict.fromkeys(v for r in linked for v in r.get('valves',[]))),
            'dn':edge['dn'],'path_policy':edge.get('path_policy'),'phase':'P2'})
    a.material['exact_search_reuse_total']=deepcopy(getattr(a,'_exact_search_totals',{}))
    return a
