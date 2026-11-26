"""Find straight pipe sections where an installed valve can fit."""
import math

import numpy as np

from .piping_build import _native_contacts
from copy import deepcopy
from .piping_valve_placement import real_valve_candidates,physical_valve_obstacles

EPS_V = 1e-6


def spool(a,name,ends,dn,medium,level,role,metadata,axis,stem,lead,tail,route_fields):
    route=a.route(name+'_slot',ends,dn,medium,role,level,**route_fields)
    cid=route['components'][0]
    a.allow(lead['components'][-1],cid,ends[0])
    a.allow(tail['components'][0],cid,ends[1])
    a.__dict__.setdefault('pipe_slots',[]).append({
        'id':name,'ends':ends,'dn':dn,'medium':medium,'level':level,'role':role,
        'metadata':metadata,'axis':axis,'stem':stem,'lead':lead['id'],'tail':tail['id'],'slot':route['id']})


def _equipment_root_connections(valve, ends, components):
    """Prove collinear same-DN male pipe ends for the package's female NPT ports."""
    connection = valve['params']['product'].get('end_connection', '')
    if not connection.endswith(' inch female NPT'):
        return None
    pipe_connection = connection.replace('female NPT', 'male NPT')
    joints = []
    for side, point, component in zip(('in', 'out'), ends, components):
        port = valve['world_ports'][side]
        if (component.get('kind') != 'pipe' or component.get('dn') != valve['dn']
                or math.dist(point, port['position']) > 1e-8):
            return None
        index = next((i for i, end in enumerate(component['ends']) if math.dist(end, point) <= 1e-8), None)
        if index is None:
            return None
        axis = np.asarray(component['ends'][1-index])-point
        if float((axis/np.linalg.norm(axis))@port['normal']) < 1.-1e-8:
            return None
        joints.append({'port':side,'position':point,'dn':valve['dn'],'pipe':component['id'],
                       'valve_end':connection,'pipe_end':pipe_connection,
                       'geometry':'Equal-diameter coaxial faces mate. Thread shape and engagement depth are simplified.'})
    return joints


def fit(a):
    routes={r['id']:r for r in a.routes}
    failures=[]
    for slot in getattr(a,'pipe_slots',[]):
        lead,tail,pipe=(routes[slot[key]] for key in ('lead','tail','slot'))
        cid=pipe['components'][0]
        adjacent=[cid,lead['components'][-1],tail['components'][0]]
        equipment_root=slot['role']=='equipment_root'
        if equipment_root:
            from .valve_roles import describe,position_check,position_context
            source_graph=a.frozen['contracts']['sizing']['graph']
            original=next(v for v in source_graph['valves'] if v['at']==slot['metadata']['at']
                          and v['edge']==slot['metadata']['edge'])
            slot['metadata'].update({key:value for key,value in describe(source_graph,original).items() if key!='role'})
            slot['metadata'].update(source_valve=original['id'],source_role=original['role'])
            context=position_context({'routes':a.routes,'elbows':a.scene.get('elbows',[]),'material':getattr(a,'material',{})},
                                     a.all_valves(),components=a.scene['components'])
        candidates=real_valve_candidates(a,slot['id'],slot['dn'],slot['ends'][0],slot['axis'],slot['stem'],
            slot['medium'],slot['role'],slot['level'],metadata=slot['metadata'],
            obstacles=physical_valve_obstacles(a),contact_ids=[cid] if equipment_root else adjacent)
        valve=None;positions=[]
        for candidate in candidates:
            if not candidate['physical_fit']:continue
            if equipment_root:
                position=position_check(a.frozen['graph'],original,context,
                    {key:port['position'] for key,port in candidate['valve']['world_ports'].items()})
                positions.append(position)
                if not position['valid']:continue
            valve=candidate['valve'];break
        if valve is None:
            if equipment_root and positions:
                failures.append({'code':'ROOT_VALVE_POSITION_PENDING','valve':slot['id'],
                                 'positions':positions,'pipe_preserved':True})
            elif equipment_root:
                failures.append({'code':'ROOT_VALVE_PHYSICAL_FIT_PENDING','valve':slot['id'],
                    'edge':slot['metadata']['edge'],'pipe_preserved':True,
                    'trials':[{'quarter':c['stem_rotation_quarters'],'obstacles':c['physical_hits']}
                              for c in candidates], 'reason':'After routing, all four root valve orientations intersect solid geometry'})
            continue
        if equipment_root:
            parts={c['id']:c for c in a.scene['components']}
            joints=_equipment_root_connections(valve,slot['ends'],[parts[name] for name in adjacent[1:]])
            if joints is None:
                failures.append({'code':'ROOT_VALVE_MATCHING_ENDS_PENDING','valve':slot['id'],
                    'edge':slot['metadata']['edge'],'pipe_preserved':True,
                    'reason':'Missing fittings for both actual valve ends'})
                continue
            valve['installation_connections']=joints
            for joint in joints:
                parts[joint['pipe']].setdefault('prepared_ends',[]).append(joint)
        a.routes.remove(pipe)
        a.scene['components']=[c for c in a.scene['components'] if c['id']!=cid]
        a.allowed_contacts=[p for p in a.allowed_contacts if cid not in p['pair']]
        a.reserve_valve(valve)
        for route,component,position in zip((lead,tail),adjacent[1:],slot['ends']):
            route['valves']=[valve['id']]
            _native_contacts(a,valve,component,position)
    return failures


def _route_straights_from_port(a, route, port_end):
    """Straight pipes of one route, ordered from its equipment end."""
    parts = {c['id']: c for c in a.scene['components']}
    rows, cursor = [], np.asarray(port_end, float)
    for cid in route['components']:
        c = parts.get(cid)
        if not c:
            continue
        if c.get('kind') == 'pipe' and c.get('shape') == 'cylinder' and c.get('ends'):
            one, two = map(np.asarray, c['ends'])
            if math.dist(two, cursor) < math.dist(one, cursor):
                one, two = two, one
            if c.get('route_id') == route['id']:   # the mouth adapter's own stub is not a valve straight
                rows.append((cid, one, two))
            cursor = two
        elif c.get('ports'):
            ends = [np.asarray(p['position']) for p in c['ports'].values()]
            cursor = max(ends, key=lambda p: math.dist(p, cursor))
    return rows


def fit_post_route_roots(a):
    """Equipment service roots, fitted on the finished pipes; a miss is a pending valve.

    The valve slides along the straights nearest its equipment mouth. A top
    mouth keeps a vertical valve on the first straight above the mouth.
    """
    from .piping_assets import inspect_asset
    from .valve_roles import describe, position_check, position_context
    failures = []
    source_graph = a.frozen['contracts']['sizing']['graph']
    for demand in getattr(a, 'post_route_roots', []):
        route = next((r for r in a.routes if r['id'] == demand['route']), None)
        if route is None:
            failures.append({'code': 'ROOT_VALVE_ROUTE_MISSING', 'valve': demand['id'], 'pipe_preserved': False})
            continue
        dn = demand['dn']
        length = float(inspect_asset(dn)['install_length_m'])
        gap = .02
        original = next(v for v in source_graph['valves'] if v['at'] == demand['metadata']['at']
                        and v['edge'] == demand['metadata']['edge'])
        metadata = {**demand['metadata'], **{k: v for k, v in describe(source_graph, original).items() if k != 'role'},
                    'source_valve': original['id'], 'source_role': original['role']}
        straights = _route_straights_from_port(a, route, demand['port_end'])
        if demand['top_vertical']:
            straights = straights[:1] if straights and abs(straights[0][2][2]-straights[0][1][2]) > EPS_V else []
        trials, chosen = [], None
        for cid, one, two in straights[:3]:
            run = math.dist(one, two)
            if run < length+2*gap:
                trials.append({'pipe': cid, 'reason': 'Straight section is shorter than the valve and both connecting stubs', 'length_m': run})
                continue
            axis = (two-one)/run
            offsets = [gap+k*.1 for k in range(int((run-length-2*gap)/.1)+1)][:8]
            for offset in offsets:
                inlet = one+axis*offset
                outlet = inlet+axis*length
                state = (list(a.scene['components']), deepcopy(route), list(a.allowed_contacts))
                part = next(c for c in a.scene['components'] if c['id'] == cid)
                pieces = []
                for name, x, y in (('in', one, inlet), ('slot', inlet, outlet), ('out', outlet, two)):
                    piece = a.pipe(route['id']+'_root_'+name, x, y, dn, route['medium'], route['id'])
                    piece.update({k: part[k] for k in ('level', 'phase') if k in part})
                    pieces.append(piece)
                a.scene['components'] = [c for c in a.scene['components'] if c['id'] != cid]
                for key in ('components', 'pipe_components'):
                    if cid in route[key]:
                        i = route[key].index(cid)
                        route[key][i:i+1] = [p['id'] for p in pieces]
                moved = [c for c in a.allowed_contacts if cid in c['pair']]
                a.allowed_contacts = [c for c in a.allowed_contacts if cid not in c['pair']]
                for contact in moved:
                    q = np.asarray(contact['point'])
                    new = pieces[0]['id'] if math.dist(q, one) <= math.dist(q, two) else pieces[2]['id']
                    a.allowed_contacts.append({**contact, 'pair': [new if x == cid else x for x in contact['pair']]})
                slot = pieces[1]['id']
                context = position_context({'routes': a.routes, 'elbows': a.scene.get('elbows', []),
                                            'material': getattr(a, 'material', {})}, a.all_valves(),
                                           components=a.scene['components'])
                valve = None
                # Handle directions perpendicular to this straight: the
                # operating side first, then the remaining building axes.
                stems = [v for v in [*demand['stems'], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                                     [-1., 0., 0.], [0., -1., 0.]] if abs(float(np.asarray(v)@axis)) < 1e-8]
                stems = [list(v) for v in dict.fromkeys(tuple(v) for v in stems)]
                obstacles = physical_valve_obstacles(a)
                for stem in stems:
                    try:
                        candidates = real_valve_candidates(a, demand['id'], dn, inlet, axis, stem, demand['medium'],
                                                           'equipment_root', demand['level'], metadata=metadata,
                                                           obstacles=obstacles, contact_ids=[slot])
                    except ValueError as exc:
                        trials.append({'pipe': cid, 'offset_m': offset, 'reason': str(exc)})
                        continue
                    for candidate in candidates:
                        if not candidate['physical_fit']:
                            continue
                        position = position_check(a.frozen['graph'], original, context,
                                                  {k: p['position'] for k, p in candidate['valve']['world_ports'].items()})
                        if not position['valid']:
                            continue
                        joints = _equipment_root_connections(candidate['valve'], [inlet.tolist(), outlet.tolist()],
                                                             [pieces[0], pieces[2]])
                        if joints is None:
                            continue
                        valve = candidate['valve']
                        valve['installation_connections'] = joints
                        break
                    if valve:
                        break
                if valve is None:
                    a.scene['components'], restored, a.allowed_contacts = state
                    route.clear()
                    route.update(restored)
                    trials.append({'pipe': cid, 'offset_m': offset, 'reason': 'This position intersects a solid or is too far from the port'})
                    continue
                a.scene['components'] = [c for c in a.scene['components'] if c['id'] != slot]
                route['components'].remove(slot)
                route['pipe_components'].remove(slot)
                a.reserve_valve(valve)
                route['valves'] = [*route.get('valves', []), valve['id']]
                for piece, point in ((pieces[0], inlet), (pieces[2], outlet)):
                    piece.setdefault('prepared_ends', []).extend(j for j in valve['installation_connections']
                                                                 if j['pipe'] == piece['id'])
                    _native_contacts(a, valve, piece['id'], point)
                valve['post_route_placement'] = {'pipe': cid, 'offset_from_mouth_side_m': offset,
                                                 'stage': 'after_all_pipes'}
                chosen = valve
                break
            if chosen:
                break
        if chosen is None:
            failures.append({'code': 'ROOT_VALVE_PENDING', 'valve': demand['id'], 'edge': demand['metadata']['edge'],
                             'pipe_preserved': True, 'trials': trials[:12],
                             'reason': 'Pipe connected. The port-side straight has no room for the valve. Leave it for the later valve-placement stage.'})
    return failures
