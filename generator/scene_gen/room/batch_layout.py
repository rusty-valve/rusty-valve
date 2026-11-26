"""Find candidate equipment positions that fit the building."""
from __future__ import annotations

from collections import defaultdict
import math
import random

import numpy as np

from .layout import add_box, add_pipe, box_of, quat_from_axes
from .process import RULES
from .routing import package, reserved_box
from .recipe import CandidateRefused


def _levels(graph, shell):
    units = graph['units']
    platforms = len(shell['floors']) - 1
    for unit in units.values():
        unit['level'] = 1 if unit['kind'] == 'vessel' else 0
    for unit in units.values():
        if unit['kind'] == 'condenser':
            unit['level'] = units[unit['owner']]['level'] + 1
        elif unit['kind'] == 'jacket':
            unit['level'] = units[unit['owner']]['level']
        if unit['level'] > platforms:
            raise CandidateRefused(f'The process needs {unit["level"]} platform levels, but the shell has only {platforms} levels. The process does not fit.')


def _intersects(a, b, margin=0.):
    return all(min(a[i + 3], b[i + 3]) - max(a[i], b[i]) > -margin
               for i in range(3))


def _vessel_positions(vessels, shell, rng, attempt):
    """Space openings evenly, retaining a passage between every opening."""
    width, depth = shell['footprint_m']
    minimum_gap = RULES['geometry_screen']['walk_width_m']
    positions = {}
    by_level = defaultdict(list)
    for vessel in vessels:
        by_level[vessel['level']].append(vessel)
    for level, same_level in by_level.items():
        order = list(same_level)
        rng.shuffle(order)
        openings = [2 * (.64 * unit.get('size', 1.) + .1) for unit in order]
        gap = (width - sum(openings)) / (len(order) + 1)
        if gap < minimum_gap:
            raise CandidateRefused('The shell cannot fit equipment openings while retaining the required aisle width.')
        # Moving the row and changing which item occupies each slot supplies
        # alternatives without device-name or seed-specific branches.
        row_y = depth * (.52 + rng.uniform(-.035, .055))
        cursor = gap
        for unit, opening in zip(order, openings):
            radius = opening / 2
            x = cursor + radius
            jitter = min(.10, max(0., gap - minimum_gap) / 3)
            x += rng.uniform(-jitter, jitter)
            y = row_y + rng.uniform(-.06, .06)
            if not radius + minimum_gap <= y <= depth - radius - minimum_gap:
                raise CandidateRefused('Equipment openings leave insufficient aisle width on both sides.')
            positions[unit['id']] = (x, y)
            cursor += opening + gap
    # Upper-level vessels must leave room for lower equipment's head and motor.
    for i, unit in enumerate(vessels):
        for other in vessels[:i]:
            if unit['level'] == other['level']:
                continue
            separation = .64 * (unit.get('size', 1.) + other.get('size', 1.)) + .35
            if math.dist(positions[unit['id']], positions[other['id']]) < separation:
                # One general alternate row is enough here; the caller tries
                # another whole candidate when this still cannot fit.
                x, y = positions[unit['id']]
                y = depth - y + (depth * .20 if unit['level'] % 2 else -depth * .20)
                margin = .64 * unit.get('size', 1.) + .1 + minimum_gap
                y = min(depth - margin, max(margin, y))
                positions[unit['id']] = (x, y)
                if math.dist(positions[unit['id']], positions[other['id']]) < separation:
                    raise CandidateRefused('Reactor tops and drives overlap between levels. Try another layout.')
    return positions


def _vessel(unit, scene, shell, xy, rng):
    x, y = xy
    z = scene['floors'][unit['level']]
    scale = unit.get('size', 1.)
    radius = .64 * scale
    # A vertical coarse shell is sufficient here; detailed AE 1000 meshes and
    # the sample-room appearance layer intentionally remain frozen.
    unit['orientation'] = 'vertical'
    angle = rng.choice((0., math.pi / 2, math.pi, 3 * math.pi / 2))
    cosine, sine = math.cos(angle), math.sin(angle)

    def vector(dx, dy, dz=0.):
        return [cosine * dx - sine * dy, sine * dx + cosine * dy, dz]

    def point(dx, dy, height):
        a, b, _ = vector(dx, dy)
        return [x + a, y + b, height]

    low, high = z - 1.9, z + .9
    add_pipe(scene, unit['id'], [x, y, low + radius * .3],
             [x, y, high - radius * .3], radius, 'vessel', (.69, .74, .75), unit['id'])
    bounds = [x - radius, y - radius, low, x + radius, y + radius, high]
    for label, height in (('lower', low + radius * .3), ('upper', high - radius * .3)):
        scene['components'].append({'id': unit['id'] + '_' + label, 'kind': 'head',
            'shape': 'ellipsoid', 'pos': [x, y, height], 'size': [radius, radius, radius * .3],
            'box': [x-radius, y-radius, height-radius*.3, x+radius, y+radius, height+radius*.3],
            'color': (.69, .74, .75), 'owner': unit['id']})
    scene['holes'][unit['level']].append([x-radius-.1, y-radius-.1, x+radius+.1, y+radius+.1])
    for side in (-1, 1):
        beam_x = x + side * (radius + .23)
        add_box(scene, unit['id'] + '_lug' + str(side),
                [x+side*(radius+.14)-.2, y-.14, z-.15,
                 x+side*(radius+.14)+.2, y+.14, z+.05],
                'lug', (.38, .46, .48), unit['id'])
        add_box(scene, unit['id'] + '_beam' + str(side),
                [beam_x-.09, 0, z-.32, beam_x+.09, shell['footprint_m'][1], z-.15],
                'equipment_beam', (.25, .37, .45), unit['id'])
    if unit.get('reactor'):
        add_pipe(scene, unit['id'] + '_motor', [x, y, high], [x, y, high+.52],
                 .20, 'motor', (.21, .34, .43), unit['id'])
    unit.update(bottom=low, top=high, center=[x, y, (low+high)/2], radius=radius,
                box=bounds, rotation_deg=math.degrees(angle))
    top_ports = [p for p in unit['ports'] if p not in ('bottom', 'drain')]
    ring = [(-.40,-.37), (0.,-.49), (.40,-.37), (.49,0.),
            (.40,.37), (0.,.49), (-.40,.37), (-.49,0.), (-.25,-.30)]
    for index, port_name in enumerate(top_ports):
        dx, dy = ring[index % len(ring)]
        dx *= scale
        dy *= scale
        pos = point(dx, dy, high)
        surface_z = high-radius*.3 + radius*.3*math.sqrt(max(0., 1-(dx*dx+dy*dy)/(radius*radius)))
        add_pipe(scene, unit['id']+'_'+port_name+'_shellneck', point(dx,dy,surface_z),
                 pos, unit['ports'][port_name]['dn']/2000, 'nozzle', (.4,.6,.62), unit['id'])
        unit['ports'][port_name].update(pos=pos, normal=[0,0,1],
            stem=vector(0, -1 if dy <= 0 else 1))
    for index, port_name in enumerate(p for p in ('bottom','drain') if p in unit['ports']):
        dx = 0. if port_name == 'bottom' else .28
        start = point(dx,0,low)
        elbow = point(dx,0,low-.15-index*.18)
        end = point(dx,-radius-.32,elbow[2])
        nozzle_radius = .028 if port_name == 'bottom' else .012
        add_pipe(scene, unit['id']+'_'+port_name+'_neck1', start, elbow, nozzle_radius,
                 'nozzle', (.4,.6,.62), unit['id'])
        add_pipe(scene, unit['id']+'_'+port_name+'_neck2', elbow, end, nozzle_radius,
                 'nozzle', (.4,.6,.62), unit['id'])
        unit['ports'][port_name].update(pos=end, normal=vector(0,-1), stem=[0,0,1])
    add_pipe(scene, unit['id']+'_safety', point(.25,.18,high), point(.25,.18,high+.3),
             .065, 'safety', (.73,.44,.21), unit['id'])


def _port_envelopes(unit, graph, ports=None):
    """Reserve the same real package and short tail that the router installs.

    These are layout reservations, not extra solids in the exported model.
    Package dimensions and the router's established neck/tail construction are
    reused; no operation-screen threshold is changed here.
    """
    ports = unit['ports'] if ports is None else ports
    result = []
    for valve in graph['valves']:
        if valve['at'] == 'branch' or valve['at'].split('.')[0] != unit['id']:
            continue
        port_name = valve['at'].split('.')[1]
        port = ports.get(port_name)
        if not port or 'pos' not in port:
            continue
        _, params = package(valve)
        geometry = params['geometry']
        length = geometry.get('face_to_face_mm', geometry.get('body_length_mm')) / 1000
        offset = .07 if unit['kind'] == 'header' else .16
        operator_length = .20 if valve['operator'] == 'wheel' else geometry['handle_length_mm'] / 1000
        pos, normal = np.asarray(port['pos']), np.asarray(port['normal'])
        center = pos + normal * (offset + length / 2)
        tip = pos + normal * max(.40 + length, offset + length/2 + operator_length + .12)
        _, matrix = quat_from_axes(normal, port['stem'])
        proxy = {**valve, 'center':center.tolist(), 'matrix':matrix.tolist(), 'length':length}
        package_box = reserved_box(proxy)
        short_pipe = box_of(pos.tolist(), tip.tolist(), valve['dn']/2000)
        box = [min(package_box[i], short_pipe[i]) for i in range(3)] + [
               max(package_box[i+3], short_pipe[i+3]) for i in range(3)]
        result.append({'owner':unit['id'],'endpoint':valve['at'],
                       'box':box,'tip':tip.tolist(),'dn':valve['dn']})
    return result


def _register_ports(unit, graph, scene):
    scene.setdefault('equipment_port_reservations', []).extend(_port_envelopes(unit, graph))


def _free_position(scene, shell, half, z_low, z_high, rng, preferred=None,
                   unit=None, graph=None, relative_ports=None):
    width, depth = shell['footprint_m']
    hx, hy = half
    front = max(RULES['geometry_screen']['walk_width_m'], depth * .225)
    xs = np.linspace(hx+.30, width-hx-.30, max(3, int(width/.65)))
    ys = np.linspace(front+hy+.12, depth-hy-.40, max(3, int(depth/.65)))
    candidates = [(float(x),float(y)) for y in ys for x in xs]
    rng.shuffle(candidates)
    if preferred:
        candidates.sort(key=lambda xy: math.dist(xy, preferred) + rng.random()*.45)
    blockers = [c['box'] for c in scene['components']
                if c['kind'] in ('vessel','head','pump','plinth','filter','exchanger','condenser','saddle')]
    reserved = [row['box'] for row in scene.get('equipment_port_reservations', [])]
    clearance = RULES['geometry_screen']['sweep_margin_m']
    for x, y in candidates:
        box = [x-hx,y-hy,z_low,x+hx,y+hy,z_high]
        if any(_intersects(box, other, .13) for other in blockers):
            continue
        # A new body cannot fill another unit's bottom nozzle / root valve /
        # tail space, even when it fits below the vessel shell itself.
        if any(_intersects(box, other, clearance) for other in reserved):
            continue
        ports = {name:{**port,'pos':[port['pos'][0]+x,port['pos'][1]+y,port['pos'][2]]}
                 for name,port in (relative_ports or {}).items()}
        envelopes = _port_envelopes(unit,graph,ports) if unit is not None else []
        valid = True
        for row in envelopes:
            envelope = row['box']
            # Use the same front and rear routing exclusions and bare-pipe
            # expansion used by batch_routing; a body-only fit is insufficient.
            route_margin = .04 + row['dn']/2000
            if (envelope[0] < route_margin or envelope[3] > width-route_margin
                    or envelope[1] < front+route_margin
                    or envelope[4] > depth-.18-route_margin):
                valid = False
                break
            if any(_intersects(envelope, other, clearance) for other in blockers + reserved):
                valid = False
                break
        if valid:
            return x,y
    raise CandidateRefused('Equipment, root valves and pipe stubs occupy an aisle or rear edge, or overlap existing equipment. Try another layout.')


def _pump(unit, graph, scene, shell, rng):
    source = graph['units'][unit['source_vessel']]
    z = .55
    ports = {'in':{'pos':[-.45,0,z],'normal':[-1,0,0],'stem':[0,0,1]},
             'out':{'pos':[.45,0,z],'normal':[1,0,0],'stem':[0,0,1]}}
    x,y = _free_position(scene,shell,(.56,.36),0.,.86,rng,source['center'][:2],
                         unit,graph,ports)
    add_box(scene,unit['id']+'_base',[x-.55,y-.34,0,x+.55,y+.34,.22],
            'plinth',(.43,.44,.43),unit['id'])
    add_pipe(scene,unit['id'],[x-.30,y,z],[x+.35,y,z],.22,'pump',(.24,.47,.57),unit['id'])
    add_box(scene,unit['id']+'_foot',[x-.2,y-.15,.22,x+.2,y+.15,.4],
            'support',(.3,.4,.44),unit['id'])
    unit.update(center=[x,y,z],box=[x-.45,y-.3,.22,x+.45,y+.3,.85])
    unit['ports']['in'].update(pos=[x-.45,y,z],normal=[-1,0,0],stem=[0,0,1])
    unit['ports']['out'].update(pos=[x+.45,y,z],normal=[1,0,0],stem=[0,0,1])
    add_box(scene,unit['id']+'_check',[x+.72,y-.07,z-.07,x+.84,y+.07,z+.07],
            'check_valve',(.7,.49,.25),unit['id'])
    _register_ports(unit,graph,scene)


def _auxiliary(unit, graph, scene, shell, rng):
    z = scene['floors'][unit['level']]
    low, high = z+.5,z+1.8
    owner = graph['units'].get(unit.get('owner'))
    preferred = owner['center'][:2] if owner else None
    radius = .25
    ports = {}
    for name in unit['ports']:
        side = 1 if name.startswith('utility') else -1
        height = low+.2 if name in ('in','condensate','utility_in') else high-.2
        ports[name] = {'pos':[0,side*radius,height],'normal':[0,side,0],'stem':[0,0,1]}
    x,y = _free_position(scene,shell,(.34,.34),low,high,rng,preferred,
                         unit,graph,ports)
    add_pipe(scene,unit['id'],[x,y,low],[x,y,high],radius,unit['kind'],(.72,.62,.40),unit['id'])
    unit.update(orientation='vertical',center=[x,y,(low+high)/2],
                box=[x-radius,y-radius,low,x+radius,y+radius,high],bottom=low,top=high)
    for side in (-1,1):
        add_box(scene,unit['id']+'_leg'+str(side),[x+side*.2-.04,y-.05,z,x+side*.2+.04,y+.05,low+.3],
                'support',(.27,.38,.44),unit['id'])
    for port_name,port in unit['ports'].items():
        side = 1 if port_name.startswith('utility') else -1
        height = low+.2 if port_name in ('in','condensate','utility_in') else high-.2
        port.update(pos=[x,y+side*radius,height],normal=[0,side,0],stem=[0,0,1])
    _register_ports(unit,graph,scene)


def equipment(graph, shell, attempt=0):
    """Create one deterministic candidate, mutating only this graph instance."""
    _levels(graph,shell)
    rng = random.Random(graph['seed'] + attempt * 104729)
    scene = {'id':graph['id'],'components':[],'holes':defaultdict(list),
             'surfaces':[],'stairs':[],'floors':list(shell['floors']),
             'footprint':list(shell['footprint_m']), 'shell':dict(shell)}
    units = graph['units']
    vessels = [u for u in units.values() if u['kind']=='vessel']
    positions = _vessel_positions(vessels,shell,rng,attempt)
    for unit in vessels:
        _vessel(unit,scene,shell,positions[unit['id']],rng)
        _register_ports(unit,graph,scene)
    for unit in units.values():
        if unit['kind']=='pump':
            _pump(unit,graph,scene,shell,rng)
        elif unit['kind'] in ('filter','exchanger','condenser'):
            _auxiliary(unit,graph,scene,shell,rng)
    for unit in units.values():
        if unit['kind']!='jacket':
            continue
        vessel=units[unit['owner']]
        x,y,_=vessel['center'];z=scene['floors'][vessel['level']];r=vessel['radius']
        add_pipe(scene,unit['id'],[x,y,vessel['bottom']+.2],[x,y,z-.25],
                 r+.045,'jacket_shell',(.56,.68,.72),unit['id'])
        for index,(port_name,port) in enumerate(unit['ports'].items()):
            sign=-1 if index==0 else 1
            pos=[x+sign*(r+.1),y,z-1.45+index*.6]
            start=[x+sign*(r+.045),y,pos[2]]
            add_pipe(scene,unit['id']+'_'+port_name+'_neck',start,pos,port['dn']/2000,
                     'nozzle',(.4,.6,.62),unit['id'])
            port.update(pos=pos,normal=[sign,0,0],stem=[0,0,1])
        unit.update(center=list(vessel['center']),box=list(vessel['box']))
    width,depth=shell['footprint_m']
    headers=[u for u in units.values() if u['kind']=='header']
    header_top=scene['floors'][-1]+1.1
    for index,unit in enumerate(headers):
        y=depth-.35;z=header_top+index*.28
        unit.update(center=[width/2,y,z],level=len(scene['floors'])-1)
        add_pipe(scene,unit['id'],[.4,y,z],[width-.4,y,z],.04,'header',(.3,.52,.58),unit['id'])
        for port_index,port in enumerate(unit['ports'].values()):
            x=.7+(port_index+.3)*(width-1.4)/max(1,len(unit['ports']))
            port.update(pos=[x,y,z],normal=[0,-1,0],stem=[0,0,1])
        for side,x in enumerate((.4,width-.4)):
            add_box(scene,unit['id']+'_support'+str(side),[x-.04,y-.04,0,x+.04,y+.04,z],
                    'support',(.32,.4,.44),unit['id'])
    # Junctions are anchored by actual incident unit ports; no naming convention
    # is interpreted. The router replaces these initial positions with real tees.
    for unit in units.values():
        if unit['kind']!='junction':
            continue
        adjacent=[]
        for edge in graph['edges']:
            endpoints=(edge['a'],edge['b'])
            for index,endpoint in enumerate(endpoints):
                if endpoint.split('.')[0]!=unit['id']:
                    continue
                other_name,other_port=endpoints[1-index].split('.')
                port=units[other_name]['ports'][other_port]
                if 'pos' in port and units[other_name]['kind']!='junction':
                    adjacent.append(port['pos'])
        center=np.mean(adjacent,axis=0).tolist() if adjacent else [width/2,depth/2,.8]
        unit['center']=center
        for port in unit['ports'].values():
            port.update(pos=list(center),normal=[0,0,0],stem=[0,0,1])
    boundaries=[u for u in units.values() if u['kind']=='boundary']
    for index,unit in enumerate(boundaries):
        edge=next(e for e in graph['edges'] if any(ep.split('.')[0]==unit['id'] for ep in (e['a'],e['b'])))
        endpoint=next(ep for ep in (edge['a'],edge['b']) if ep.split('.')[0]!=unit['id'])
        uid,port_name=endpoint.split('.')
        target=units[uid]['ports'][port_name]['pos']
        if port_name=='gauge':
            pos=[target[0]+.3,target[1]+.3,target[2]+.8]
            add_pipe(scene,unit['id']+'_dial',pos,[pos[0],pos[1]-.08,pos[2]],
                     .07,'gauge',(.85,.87,.80),unit['id'])
        else:
            pos=[.35 if index%2==0 else width-.35,
                 depth*.32+(index*.43)%(depth*.48),.35+(index%5)*.16]
        unit['center']=pos
        unit['ports']['port'].update(pos=pos,normal=[0,0,0],stem=[0,0,1])
    scene['equipment_layout']={'attempt':attempt,'method':'Space equipment openings evenly. Search rotations and lower-level vacancies, then score the completed room using the same checks.',
        'units':[{'id':u['id'],'kind':u['kind'],'center':u.get('center'),
                  'rotation_deg':u.get('rotation_deg',0),'level':u['level']}
                 for u in units.values() if u['kind'] in ('vessel','pump','filter','exchanger','condenser')]}
    return scene
