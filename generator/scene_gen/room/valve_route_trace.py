"""Follow the pipe connected to an installed valve through its elbows."""
from collections import defaultdict
import heapq
import itertools
import math

import numpy as np


EPS = 1.e-6


def _key(point):
    return tuple(round(float(x),6) for x in point)


def _on_line(point, one, two):
    axis=two-one
    length=np.linalg.norm(axis)
    t=float((point-one)@axis)/length
    return EPS<t<length-EPS and np.linalg.norm(np.cross(point-one,axis))<=EPS*length


def related_routes(room, request, initial):
    edge=request.get('edge')
    identity=request.get('connection_id')
    service_ids = {p['id'] for p in room.piping.get('requirements', {}).get('service_points', ())
                   if p['endpoint'] == request.get('at')}
    return [r for r in room.routes if r in initial or (edge and (
        edge in (r.get('source_edge'),r.get('source_requirement')) or edge in r.get('source_edges',())))
        or r.get('source_requirement') in service_ids
        or (identity and r.get('connection_id')==identity)]


class RunTrace:
    def __init__(self, routes, parts, elbows, stops):
        self.stops={_key(point):name for point,name in stops}
        self.nodes=defaultdict(list)
        self.straights=[]
        self.elbows={}
        by_route={route['id']:route for route in routes}
        seen = set()
        for route in routes:
            for cid in dict.fromkeys(route.get('pipe_components') or route.get('components',())):
                part=parts.get(cid)
                if cid in seen or not part or part['kind']!='pipe' or part['shape']!='cylinder':
                    continue
                a,b=map(np.asarray,part['ends'])
                if np.linalg.norm(b-a)<=EPS:
                    continue
                seen.add(cid)
                self.straights.append((a,b,part,route))
                self.nodes[_key(a)].append((b,part,route))
                self.nodes[_key(b)].append((a,part,route))
        for elbow in elbows:
            if elbow['route_id'] not in by_route:
                continue
            a,b=(np.asarray(elbow['ports'][side]['position']) for side in ('in','out'))
            name=elbow['id']
            self.elbows[name]=float(elbow['centerline_radius_m'])*math.radians(elbow.get('angle_degrees',90.))
            route=by_route[elbow['route_id']]
            self.nodes[_key(a)].append((b,name,route))
            self.nodes[_key(b)].append((a,name,route))

    def from_reference(self, point, local_routes=()):
        if not self.straights:
            return
        # The nozzle face can be separated from the first weldable pipe by
        # its own adapter. Start on the closest pipe end after that adapter.
        # At a tee or an in-line union, both actual approaches can start here.
        ends = [p for a, b, _, route in self.straights
                if not local_routes or route['id'] in local_routes for p in (a, b)]
        if not ends:
            return
        start = min(ends, key=lambda end: math.dist(end, point))
        yield from self.later(start)

    def later(self, start, excluded=()):
        """Walk without crossing a fitting, branch or another valve.

        Each yielded straight carries the real elbows crossed from the start.
        Once a pipe meets an internal stop, its approach may still be tried.
        """
        uid=itertools.count()
        pending=[(0.,next(uid),np.asarray(start),None,())]
        visited=set()
        while pending:
            distance,_,point,prior,elbows=heapq.heappop(pending)
            key=_key(point)
            if key in visited or (prior is not None and key in self.stops):
                continue
            visited.add(key)
            available=[(end,part,route) for end,part,route in self.nodes.get(key,())
                       if (part if isinstance(part,str) else part['id']) not in excluded
                       and (part if isinstance(part,str) else part['id'])!=prior]
            if prior is not None and len(available) != 1:
                continue
            for end,part,route in available:
                if isinstance(part,str):
                    heapq.heappush(pending,(distance+self.elbows[part],next(uid),end,part,(*elbows,part)))
                    continue
                yield distance,point,end,part,route,elbows
                if any(_on_line(np.asarray(stop),point,end) for stop in self.stops):
                    continue
                heapq.heappush(pending,(distance+math.dist(point,end),next(uid),end,part['id'],elbows))
