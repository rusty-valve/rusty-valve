"""Keep clear sections of a pipe route and record why a connection broke."""
from collections import defaultdict

import numpy as np


def clear_stair_deck(scene):
    """Cut a local deck notch where an existing stair stringer enters its slab."""
    from .grating import grating, overlaps
    from .piping_build import cut_pipe_openings

    boards=defaultdict(list)
    for part in scene['components']:
        if grating(part):
            boards[part['box'][5]].append(part['box'])
    notches=[]
    for part in scene['components']:
        if part['kind']!='stringer' or part['shape']!='cylinder':
            continue
        first,last=map(np.asarray,part['ends'])
        if first[2]>last[2]:
            first,last=last,first
        delta=last-first
        if delta[2]<=1e-8:
            continue
        axis=delta/np.linalg.norm(delta)
        extent=part['radius']*np.sqrt(np.maximum(0.,1.-axis**2))
        for level,z in enumerate(scene['floors'][1:],1):
            if not boards[z]:
                continue
            bottom=min(box[2] for box in boards[z])
            low=max(0.,(bottom-extent[2]-first[2])/delta[2])
            high=min(1.,(z+extent[2]-first[2])/delta[2])
            if high<=low:
                continue
            one,two=(first[:2]+fraction*delta[:2] for fraction in (low,high))
            reach=extent[:2]+.002
            rect=np.r_[np.minimum(one,two)-reach,np.maximum(one,two)+reach].tolist()
            if any(overlaps(rect,box) for box in boards[z]):
                notches.append({'id':part['id']+'_deck_clearance','level':level,'rect':rect,
                                'stringer':part['id'],'source':'Local opening where the stair stringer enters the landing slab'})
    if notches:
        cut_pipe_openings(scene,notches)
    scene['stair_deck_clearances']=notches


def omit_intersections(graph, scene, piping, errors):
    components={part['id']:part for part in scene['components']}
    routes=defaultdict(list)
    for route in piping['routes']:
        for name in route['components']:
            routes[name].append(route)
    joints={}
    for joint in scene.get('catalogue_flange_joints',[]):
        for name in joint['components']:
            joints[name]=joint
    elbows={name:record for record in scene.get('elbows',[]) for name in record['components']}
    valve_parts={valve['prefix']+g['name']:valve for valve in graph['valves']
                 for g in valve['world_geoms']}
    removed=set()
    withdrawn=set()
    breaks=[]
    for error in errors:
        if error['code']!='P1_SOLID_COLLISION':
            continue
        pair=(error['first'],error['second'])
        if any(n in removed or valve_parts.get(n,{}).get('id') in withdrawn for n in pair):
            continue
        removable=[n for n in pair if n in routes or n in joints or n in elbows or n in valve_parts]
        if not removable:
            continue
        # Keep the host line when a new branch hits it. Otherwise withdraw
        # the installed valve before discarding an independent pipe.
        def priority(name):
            row=components.get(name,{})
            if name in valve_parts:
                return 4
            if any(r.get('role')=='floor_header' for r in routes.get(name,())):
                return 0
            return 3 if row.get('source_edge') or name in elbows else 2
        name=max(removable,key=priority)
        valve=valve_parts.get(name)
        joint=joints.get(name)
        elbow=elbows.get(name)
        losing=set()
        if valve:
            withdrawn.add(valve['id'])
        elif joint:
            losing.update([*joint['components'],joint['pipe_stub'],joint['through_bore']])
        elif elbow:
            losing.update(elbow['components'])
        else:
            losing.add(name)
        removed.update(losing)
        related={r['id']:r for n in losing for r in routes.get(n,())}
        if valve:
            related.update((r['id'],r) for r in piping['routes'] if valve['id'] in r.get('valves',()))
        if not related:
            related={None:{}}
        one,two=error['actual_boxes']
        position=[(max(one[i],two[i])+min(one[i+3],two[i+3]))/2 for i in range(3)]
        for route_id,route in related.items():
            breaks.append({'edge':route.get('source_edge') or route.get('source_requirement')
                                  or (valve or {}).get('source_edge'),
                'valve':(valve or {}).get('source_valve') or (valve or {}).get('id'),
                'route':route_id,'components':sorted(losing),'position':position,'pair':list(pair),
                'penetration_m':error['penetration_m'],
                'reason':f"Solids {pair[0]} and {pair[1]} penetrate by {1000*error['penetration_m']:.1f} mm. Interfering pipe or valve parts were removed."})
    if not breaks:
        return []
    scene['components'][:]=[c for c in scene['components'] if c['id'] not in removed]
    graph['valves'][:]=[v for v in graph['valves'] if v['id'] not in withdrawn]
    omitted_elbows={elbows[n]['id'] for n in removed if n in elbows}
    for route in piping['routes']:
        for key in ('components','pipe_components'):
            if key in route:
                route[key]=[n for n in route[key] if n not in removed]
        route['elbows']=[n for n in route.get('elbows',[]) if n not in omitted_elbows]
        route['valves']=[n for n in route.get('valves',[]) if n not in withdrawn]
    for data in (scene,piping):
        for key in ('elbows','catalogue_flange_joints','flange_connections'):
            if key in data:
                data[key]=[record for record in data[key] if not removed.intersection(record['components'])]
    piping['allowed_contacts']=[row for row in piping.get('allowed_contacts',[])
                                if not removed.intersection(row['pair'])]
    for route in piping.get('material',{}).get('edge_routes',[]):
        route['components']=[n for n in route['components'] if n not in removed]
        route['valves']=[n for n in route.get('valves',[]) if n not in withdrawn]
    return breaks
