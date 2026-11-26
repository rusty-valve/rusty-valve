"""Route process pipes along racks at the destination floor."""
from collections import deque
from itertools import islice, zip_longest
import math

import numpy as np


MATERIAL = {'material','recycle','feed','transfer','pumped_transfer'}
LIQUID = {'condensate','reflux_condensate','condenser_condensate','condensate_return'}


def downstream_vessel(graph, endpoint):
    seen=set()
    pending=deque([endpoint])
    vessels=set()
    while pending:
        uid=pending.popleft().split('.')[0]
        if uid in seen:
            continue
        seen.add(uid)
        unit=graph['units'][uid]
        if unit['kind']=='vessel':
            vessels.add(uid)
        elif unit['kind']=='junction':
            pending.extend(e['b'] for e in graph['edges'] if e['a'].split('.')[0]==uid
                           and e['role'] in MATERIAL|LIQUID)
    return next(iter(vessels)) if len(vessels)==1 else None


def destination_rule(graph, equipment_levels, edge, source_level, target_level):
    vessel=downstream_vessel(graph,edge['b'])
    end_level=equipment_levels.get(vessel)
    if end_level is None or target_level!=end_level:
        return None
    source=graph['units'][edge['a'].split('.')[0]]
    target=graph['units'][edge['b'].split('.')[0]]
    if edge['role'] in MATERIAL and end_level>source_level:
        return {'kind':'ascending_material','level':end_level,'vessel':vessel}
    if (edge['role'] in LIQUID and source['kind']=='condenser'
            and source.get('orientation')=='horizontal'
            and target.get('template_role',target.get('service'))=='liquid_seal'
            and source_level>end_level):
        return {'kind':'horizontal_condenser_to_seal','level':end_level,'vessel':vessel}
    return None


LEAD_LENGTHS_M = (.4, .8)   # fixed extra lead lengths, independent of any valve


def _riser_starts(point, direction, rise, reach):
    if abs(direction[2]) < .5:
        for length in sorted({reach,2*reach,3*reach,*(max(reach,x) for x in LEAD_LENGTHS_M)}):
            lead=point+direction*length
            yield [point,lead],lead
            side=np.cross([0,0,1],direction)
            for sign in (1,-1):
                jog=lead+side*sign*2*reach
                yield [point,lead,jog],jog
    elif direction[2]*rise<0:
        # A mouth facing away from the rack (e.g. a bottom outlet going up):
        # leave it, turn horizontal, then rise beside it.
        lead=point+direction*max(2*reach,.25)
        for axis in (0,1):
            for sign in (1,-1):
                for length in sorted({2*reach,4*reach,*LEAD_LENGTHS_M}):
                    jog=lead+np.eye(3)[axis]*sign*length
                    yield [point,lead,jog],jog
    elif direction[2]*rise>0:
        yield [point],point
        for up in sorted({max(2*reach,.25),*(max(2*reach,x) for x in LEAD_LENGTHS_M)}):
            lead=point+direction*up
            for axis in (0,1):
                for sign in (1,-1):
                    for length in sorted({2*reach,4*reach,6*reach,*LEAD_LENGTHS_M}):
                        jog=lead+np.eye(3)[axis]*sign*length
                        yield [point,lead,jog],jog


def _riser_ends(point, direction, height, reach, local_leads=(), lateral=False):
    if lateral:
        if abs(direction[2])<.5:
            cross=np.cross([0.,0.,1.],direction)
            for lead in local_leads:
                turn=point+direction*lead
                for distance in LEAD_LENGTHS_M:
                    for sign in (-1,1):
                        drop=turn+sign*cross*distance
                        yield drop,[drop,turn,point]
        return
    if abs(direction[2])>=.5 and (height-point[2])*direction[2]<0:
        lead=point+direction*max(2*reach,.25)
        for axis in (0,1):
            for sign in (1,-1):
                for length in sorted({2*reach,4*reach,*LEAD_LENGTHS_M}):
                    turn=lead+np.eye(3)[axis]*sign*length
                    yield turn,[turn,lead,point]
        return
    if abs(direction[2])<.5:
        for length in sorted({reach,2*reach,3*reach,*(max(reach,x) for x in LEAD_LENGTHS_M),*local_leads}):
            turn=point+direction*length
            yield turn,[turn,point]
    elif (height-point[2])*direction[2]>0:
        yield point,[point]
        for up in sorted({max(2*reach,.25),*(max(2*reach,x) for x in LEAD_LENGTHS_M)}):
            lead=point+direction*up
            for axis in (0,1):
                for sign in (1,-1):
                    for length in sorted({2*reach,4*reach,*LEAD_LENGTHS_M}):
                        turn=lead+np.eye(3)[axis]*sign*length
                        yield turn,[turn,lead,point]


def _rack_guides(first,last,start_direction,end_direction,heights,reach,local_leads=(),lateral=False):
    """All heights together, simplest first, without duplicates."""
    seen=set();rows=[]
    for score,points in _rack_guides_by_height(first,last,start_direction,end_direction,heights,reach,local_leads,lateral):
        key=tuple(round(float(v),4) for p in points for v in p)
        if key in seen:continue
        seen.add(key);rows.append((score,points))
    yield from sorted(rows,key=lambda row:row[0])


def _rack_guides_by_height(first,last,start_direction,end_direction,heights,reach,local_leads=(),lateral=False):
    for z in heights:
        source=list(_riser_starts(first,start_direction,z-first[2],reach))
        target=list(_riser_ends(last,end_direction,z,reach,local_leads,lateral))
        guides=[]
        for head,low in source:
            for high,tail in target:
                one=np.r_[low[:2],z];two=np.r_[high[:2],z]
                if abs(one[2]-low[2])<reach or abs(two[2]-high[2])<reach:
                    continue
                for axis in (0,1):
                    corner=one.copy();corner[axis]=two[axis]
                    guide=[*head,one,corner,two,*tail]
                    points=[guide[0]]
                    for p in guide[1:]:
                        if not np.allclose(p,points[-1],atol=1e-8):
                            points.append(p)
                    score=(len(points),sum(math.dist(a,b) for a,b in zip(points,points[1:])))
                    guides.append((score,points))
        yield from sorted(guides,key=lambda candidate:candidate[0])


def rack_run_evidence(runs, band, destination):
    lower,upper=band
    horizontal=[]
    local_drop=False
    for first,last in runs:
        if abs(first[2]-last[2])<1e-6:
            horizontal.append({'z_m':first[2],'length_m':math.dist(first,last)})
        elif destination is not None:
            local_drop |= (math.dist(first[:2],destination[:2])<=.9 and
                           max(first[2],last[2])>=lower-1e-6 and min(first[2],last[2])<lower)
    longest=max(horizontal,key=lambda row:row['length_m'],default=None)
    in_rack=bool(longest and lower-1e-6<=longest['z_m']<=upper+1e-6)
    return {'horizontal_runs':horizontal,'local_descent':bool(local_drop),'pass':in_rack and local_drop}


def _seal_guides(standard, lateral):
    yield from islice(standard, 48)
    for ordinary, sideways in zip_longest(standard, lateral):
        if ordinary is not None:
            yield ordinary
        if sideways is not None:
            yield sideways


def connect_in_rack(a,edge,start,end,rule,options,start_reserve,end_reserve,*,attempts,extra=(),variants=None,
                    full_rack=()):
    """Destination-rack guides, each one attempt of the common budget; no relaxation."""
    from .piping_elbows import ElbowUnavailable, plan_polyline
    from .piping_route_search import RouteUnavailable

    zone=a.plan['rack_zones'][rule['level']]
    lower,upper=zone['bottom_z_m'],zone['top_z_m']
    r=a.radius(edge['dn'])
    heights=((lower+upper)/2,lower+r+.05,upper-r-.05)
    first,last=(np.asarray(p['point']) for p in (start,end))
    normal=np.asarray(start['normal']); target=np.asarray(end['normal'])
    variants=variants or [(start_reserve,end_reserve)]
    reach=3*r+.06
    local_leads=()
    if rule['kind']=='horizontal_condenser_to_seal' and end_reserve>0:
        # Keep the actual seal inlet straight after subtracting the elbow radius.
        local_leads=(end_reserve+3*r,end_reserve+4*r,end_reserve+5*r)
    guides=_rack_guides(first,last,normal,target,heights,reach,local_leads)
    if local_leads:
        lateral=_rack_guides(first,last,normal,target,heights,reach,local_leads,lateral=True)

        def valid_seal_shapes(candidates):
            for score, guide in candidates:
                try:
                    plan=plan_polyline(guide,r)
                except ElbowUnavailable:
                    continue
                runs=[(run['start'],run['end']) for run in plan['runs']]
                if (runs and math.dist(*runs[-1])+1e-8>=end_reserve
                        and rack_run_evidence(runs,(lower,upper),last)['pass']):
                    yield score,guide

        guides=_seal_guides(valid_seal_shapes(guides),valid_seal_shapes(lateral))
    for guide_order,(_,guide) in enumerate(guides,1):
      for variant,(start_reserve,end_reserve) in enumerate(variants):
        if not attempts.open:
            break
        def one(guide=guide,start_reserve=start_reserve,end_reserve=end_reserve):
            try:
                points=a.search(first,last,edge['dn'],**options,extra=extra,clearance=.05,guide=guide,
                                start_direction=normal,end_direction=-target,
                                start_straight_m=start_reserve,end_straight_m=end_reserve)
            except RouteUnavailable as error:
                raise error
            runs=[(run['start'],run['end']) for run in plan_polyline(points,r)['runs']]
            if not full_rack:
                evidence=rack_run_evidence(runs,(lower,upper),last)
                if not evidence['pass']:
                    raise RouteUnavailable('RACK_LONG_RUN_OR_LOCAL_DESCENT','Long horizontal run misses the destination-level rack or does not drop at the vessel',evidence=evidence)
            return points,{'radius':r,'rack':{**rule,'band_m':[lower,upper]}}
        attempts.run('destination_rack_guide_'+str(guide_order)+('_valve_room' if variant==0 and len(variants)>1 else ''),one)
