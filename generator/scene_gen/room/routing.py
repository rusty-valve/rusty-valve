"""Find a pipe route made of straight sections aligned to the room axes."""
from __future__ import annotations
import heapq
import itertools
import math
import os
from pathlib import Path
import json
import numpy as np
from .layout import add_pipe, add_box, box_of, quat_from_axes, H
from .runtime_paths import RUNTIME

ROOT=Path(__file__).resolve().parents[2]
PACKAGES=RUNTIME/'valve_library/packages'
GRID=.15
SHAPE=(81,41,68)
COLORS={'material':(.27,.59,.65),'recycle':(.71,.48,.26),'bypass':(.78,.61,.25),'vapor':(.76,.38,.29),'condensate':(.30,.65,.55),'nitrogen':(.43,.62,.79),'offgas':(.52,.50,.66),'steam':(.76,.41,.31),'steam_return':(.65,.47,.37),'water':(.30,.57,.76),'water_return':(.35,.58,.64),'boundary':(.44,.58,.58),'feed':(.28,.62,.64)}

def package(v):
    model=f'bray-7100-dn{v["dn"]:03}-npt-ftg42l' if v['operator']=='wheel' else f'bray-7000-dn{v["dn"]:03}-npt-lever'
    version='v0.3' if v['operator']=='wheel' else 'v0.2'
    p=PACKAGES/model/version
    params=json.loads((p/'params.json').read_text())
    return p,params

def reserved_box(v):
    _,p=package(v);geo=p['geometry'];length=v['length']
    if v['operator']=='wheel':lo=[-length/2,-.24,-.05];hi=[.20,.09,.30]
    else:
        arm=geo['handle_length_mm']/1000;z=geo['overall_height_mm']/1000
        lo=[-length/2,-arm-.025,-.04];hi=[max(length/2,arm)+.025,.04,z+.035]
    points=np.array([np.array(v['center'])+np.array(v['matrix'])@np.array(p) for p in itertools.product(*zip(lo,hi))])
    return [*points.min(axis=0).tolist(),*points.max(axis=0).tolist()]

def line_hits(a,b,box,margin=0):
    lo=0.;hi=1.
    for i in range(3):
        mn=box[i]-margin;mx=box[i+3]+margin;d=b[i]-a[i]
        if abs(d)<1e-10:
            if a[i]<mn or a[i]>mx: return False
        else:
            p=(mn-a[i])/d;q=(mx-a[i])/d
            lo=max(lo,min(p,q));hi=min(hi,max(p,q))
            if hi<lo: return False
    return True

def simplify(points):
    out=[]
    for p in points:
        p=[float(x) for x in p]
        if out and math.dist(out[-1],p)<1e-7: continue
        if len(out)>1:
            d=np.array(out[-1])-out[-2];e=np.array(p)-out[-1]
            if np.linalg.norm(np.cross(d,e))<1e-8 and np.dot(d,e)>0: out[-1]=p;continue
        out.append(p)
    return out

def grid_box(blocked,box,pad=0,value=True):
    starts=[max(0,math.ceil((box[i]-pad)/GRID)) for i in range(3)]
    ends=[min(SHAPE[i],math.floor((box[i+3]+pad)/GRID)+1) for i in range(3)]
    blocked[tuple(slice(a,b) for a,b in zip(starts,ends))]=value

def connector(a,b,boxes,margin=.025):
    for order in itertools.permutations(range(3)):
        points=[list(a)];q=list(a)
        for ax in order:
            q=q.copy();q[ax]=b[ax];points.append(q)
        points=simplify(points)
        if any(abs(x[2]-y[2])<1e-7 and any(abs(x[2]-level*H)<.14 for level in (1,2)) for x,y in zip(points,points[1:])):continue
        if all(not line_hits(x,y,box,margin) for x,y in zip(points,points[1:]) for box in boxes):return points
    return None

def search(a,b,base,boxes,margin=.025):
    direct=connector(a,b,boxes,margin)
    if direct: return direct
    blocked=base.copy()
    nodes=[];connectors=[]
    for pos in (a,b):
        center=np.rint(np.array(pos)/GRID).astype(int)
        candidates=[]
        for off in itertools.product(range(-3,4),repeat=3):
            idx=tuple(center+off)
            if all(0<=idx[j]<SHAPE[j] for j in range(3)) and not blocked[idx]:
                candidates.append((math.dist(pos,np.array(idx)*GRID),idx))
        candidates.sort()
        for _,idx in candidates:
            con=connector(pos,np.array(idx)*GRID,boxes,margin)
            if con is not None: nodes.append(idx);connectors.append(con);break
        else: raise RuntimeError(f'No routing cell can leave the port: {pos}; blocking={[box for box in boxes if line_hits(pos,pos,box,margin)]}')
    start,goal=nodes; q=[(0,0,start)]; previous={}; cost={start:0}; count=0
    # Nodes alone miss a thin pipe between two grid points. Reserve the complete grid edges too.
    edge_blocked=[np.zeros(SHAPE,bool) for _ in range(3)]
    for box in boxes:
        for axis in range(3):
            starts=[max(0,math.ceil((box[j]-margin)/GRID-(1 if j==axis else 0))) for j in range(3)]
            ends=[min(SHAPE[j],math.floor((box[j+3]+margin)/GRID)+1) for j in range(3)]
            edge_blocked[axis][tuple(slice(x,y) for x,y in zip(starts,ends))]=True
    while q:
        _,c,u=heapq.heappop(q)
        if c!=cost[u]:continue
        if u==goal: break
        count+=1
        for axis in range(3):
            for step in (-1,1):
                v=list(u);v[axis]+=step;v=tuple(v)
                if not all(0<=v[j]<SHAPE[j] for j in range(3)) or blocked[v]:continue
                if edge_blocked[axis][u if step==1 else v]:continue
                if axis!=2 and any(abs(u[2]*GRID-level*H)<.14 for level in (1,2)):continue
                # Prefer short routes, then service-side/overhead lanes; no random bends.
                previous_axis=next((j for j in range(3) if u in previous and previous[u][j]!=u[j]),axis)
                nc=c+1+(0.02 if axis==2 else 0)+(.35 if previous_axis!=axis else 0)
                if nc<cost.get(v,1e30):
                    cost[v]=nc;previous[v]=u;h=sum(abs(v[j]-goal[j]) for j in range(3))
                    heapq.heappush(q,(nc+h,nc,v))
    else: raise RuntimeError(f'No route for {a} → {b}')
    path=[goal]
    while path[-1]!=start:path.append(previous[path[-1]])
    path.reverse()
    points=connectors[0]+[np.array(p)*GRID for p in path]+list(reversed(connectors[1]))
    return simplify(points)

def route(g,scene):
    valves=g['valves'];units=g['units'];ends={};fixed=[]
    # Leave paths and structural columns free before routing. Floor penetrations are cut later.
    for c in scene['components']:
        if c['kind'] not in ('nozzle','header','gauge','check_valve'):
            fixed.append(c['box'])
    fixed += [[-.15,-.15,0,.15,.15,10],[5.85,-.15,0,6.15,.15,10],[11.85,-.15,0,12.15,.15,10],
              [-.15,5.85,0,.15,6.15,10],[5.85,5.85,0,6.15,6.15,10],[11.85,5.85,0,12.15,6.15,10]]
    fixed.append([0,0,0,12,1.35,10])
    fixed.append([0,0,-.2,12,6,.12])
    fixed.append([0,5.82,0,12,6.2,10])
    for z in (H,2*H):
        for x in (0,6,12):fixed.append([x-.13,0,z-.33,x+.13,6,z-.1])
    for x in (0,6,12):fixed.append([x-.13,0,9.47,x+.13,6,9.68])
    by_edge={e['id']:[v for v in valves if v['edge']==e['id']] for e in g['edges']}
    for e in g['edges']:
        for ep in (e['a'],e['b']):
            uid,p=ep.split('.');port=units[uid]['ports'][p];pos=np.array(port['pos'],float);normal=np.array(port['normal'],float)
            v=next((v for v in by_edge[e['id']] if v['at']==ep),None)
            if v:
                pack,params=package(v);length=params['geometry'].get('face_to_face_mm',params['geometry'].get('body_length_mm'))/1000
                offset=.07 if units[uid]['kind']=='header' else .16
                tail=.40
                operator_length=.20 if v['operator']=='wheel' else params['geometry']['handle_length_mm']/1000
                center=pos+normal*(offset+length/2)
                tip=pos+normal*max(tail+length,offset+length/2+operator_length+.12)
                q,mat=quat_from_axes(normal,port['stem'])
                v.update(center=center.tolist(),quat=q,matrix=mat.tolist(),package=os.path.relpath(pack, ROOT),length=length,
                         axis=normal.tolist(),level=max(0,min(2,math.floor((center[2]+.05)/H))))
                add_pipe(scene,e['id']+'_'+v['id']+'_neck',pos,center-normal*length/2,e['dn']/2000,'nozzle',COLORS[e['role']],uid)
                add_pipe(scene,e['id']+'_'+v['id']+'_tail',center+normal*length/2,tip,e['dn']/2000,'pipe',COLORS[e['role']],e['id'])
                ends[ep]=tip.tolist()
            else: ends[ep]=pos.tolist()
    # Put manifold tees beyond the actual package tails, with room for separate elbows.
    # Using a guessed fixed offset here can put a new branch just millimetres from its inlet.
    for u in units.values():
        if u['kind']!='junction':continue
        uid=u['id']
        if uid.endswith('_bottom'):
            anchor=ends[uid.removesuffix('_bottom')+'.bottom'];delta=[-1.0,.4,0]
        elif uid.endswith('_feed'):
            owner=units[uid.removesuffix('_feed')];anchor=ends[owner['id']+'.feed'];delta=[-.9,0,0 if owner['orientation']=='vertical' else .4]
        else:
            owner=units[uid.removesuffix('_out')];anchor=ends[owner['id']+'.out'];delta=[.25,.4,0] if owner['kind']=='pump' else [.8,0,0]
        u['center']=(np.array(anchor)+delta).tolist()
        toward=np.array(anchor)-u['center'];axis=int(np.argmax(abs(toward)));direction=np.zeros(3);direction[axis]=1 if toward[axis]>0 else -1
        side=np.array([0,1 if u['center'][1]<3 else -1,0]) if axis==0 else np.array([-1,0,0])
        directions=[direction,-direction,side,-side,np.array([0,0,1]),np.array([0,0,-1])]
        for j,(p,port) in enumerate(u['ports'].items()):
            dock=(np.array(u['center'])+.30*directions[j]).tolist();port['pos']=dock;ends[uid+'.'+p]=dock
            edge=next(e for e in g['edges'] if uid+'.'+p in (e['a'],e['b']))
            arm=add_pipe(scene,f'J_{uid}_{p}',u['center'],dock,edge['dn']/2000,'pipe',COLORS[edge['role']],edge['id']);arm['junction']=uid
    routed=[];scene['routing']=[]
    # Short material routes first, utilities last. Stubs excluded only at their own endpoints.
    ordered=sorted(g['edges'],key=lambda e:(e['role'] not in ('material','recycle','bypass','vapor','condensate'),e['id']))
    for e in ordered:
        a,b=ends[e['a']],ends[e['b']]
        connected={e['a'].split('.')[0],e['b'].split('.')[0]}
        boxes=list(fixed)+[reserved_box(v) for v in valves if 'center' in v]
        for c in scene['components']:
            if c['kind']=='pipe' and c.get('owner')!=e['id'] and ('_tail' in c['id'] or c.get('junction')):
                boxes.append([x-.025 if j<3 else x+.025 for j,x in enumerate(c['box'])])
        for prior,box in routed:
            # Only a small neighborhood of a shared tee is exempt, never the whole prior pipe.
            shared=connected & {prior['a'].split('.')[0],prior['b'].split('.')[0]}
            exceptions=[units[u]['center'] for u in shared if units[u]['kind']=='junction']
            pieces=[box]
            for node in exceptions:
                updated=[]
                for piece in pieces:
                    if not line_hits(node,node,piece,.03):updated.append(piece);continue
                    axis=max(range(3),key=lambda k:piece[k+3]-piece[k])
                    if piece[axis]<node[axis]-.18:
                        part=list(piece);part[axis+3]=node[axis]-.18;updated.append(part)
                    if piece[axis+3]>node[axis]+.18:
                        part=list(piece);part[axis]=node[axis]+.18;updated.append(part)
                pieces=updated
            boxes.extend(pieces)
        blocked=np.zeros(SHAPE,bool)
        for box in boxes:grid_box(blocked,box,.04+e['dn']/2000)
        try:
            points=None
            if e['role'] in ('nitrogen','offgas','water','water_return','steam','steam_return'):
                # Utilities travel above the occupied zone, then drop at the nozzle.
                # This is one rule for every room, never a figure-specific route.
                height=max(a[2],b[2],9.7)
                aa=[a[0],a[1],height];bb=[b[0],b[1],height]
                first=connector(a,aa,boxes,e['dn']/2000+.025);last=connector(bb,b,boxes,e['dn']/2000+.025)
                if first is not None and last is not None:
                    middle=search(aa,bb,blocked,boxes,e['dn']/2000+.025);points=simplify(first+middle+last)
            if points is None:points=search(a,b,blocked,boxes,e['dn']/2000+.025)
        except RuntimeError as ex:
            # Keep explicit failure for reviewer; never silently emit a crossing fallback.
            raise RuntimeError(f'{g["id"]} {e["id"]} {e["a"]} → {e["b"]}: {ex}') from ex
        branch=next((v for v in by_edge[e['id']] if v['at']=='branch'),None)
        if branch:
            def valve_slot(i):
                midpoint=(np.array(points[i])+points[i+1])/2;height=midpoint[2]%H
                return (not .55<=height<=1.6,abs(height-1.1),-math.dist(points[i],points[i+1]))
            candidates=sorted(range(len(points)-1),key=valve_slot)
            idx=next(i for i in candidates if math.dist(points[i],points[i+1])>.5)
            a1=np.array(points[idx]);b1=np.array(points[idx+1]);normal=(b1-a1)/np.linalg.norm(b1-a1)
            center=(a1+b1)/2;stem=[0,0,1] if abs(normal[2])<.5 else [0,-1,0]
            pack,params=package(branch);length=params['geometry'].get('face_to_face_mm',params['geometry'].get('body_length_mm'))/1000;q,mat=quat_from_axes(normal,stem)
            branch.update(center=center.tolist(),quat=q,matrix=mat.tolist(),package=os.path.relpath(pack, ROOT),length=length,axis=normal.tolist(),
                level=max(0,min(2,math.floor((center[2]+.05)/H))))
        for i,(a1,b1) in enumerate(zip(points,points[1:])):
            pieces=[(a1,b1)]
            if branch and i==idx: pieces=[(a1,(center-normal*length/2).tolist()),((center+normal*length/2).tolist(),b1)]
            for j,(aa,bb) in enumerate(pieces):
                add_pipe(scene,f'{e["id"]}_{i}_{j}',aa,bb,e['dn']/2000,'pipe',COLORS[e['role']],e['id'])
            routed.append((e,box_of(a1,b1,e['dn']/2000+.025)))
        scene['routing'].append({'edge':e['id'],'points':points,'end_a':e['a'],'end_b':e['b']})
    # Pipe holes are local; no entire floor is removed to hide pipe/deck collisions.
    for c in scene['components']:
        if c['kind'] not in ('pipe','nozzle'): continue
        a,b=c['ends'];r=c['radius']+.025
        for level in (1,2):
            if min(a[2],b[2])-r < level*H and max(a[2],b[2])+r > level*H-.1:
                scene['holes'][level].append([min(a[0],b[0])-r,min(a[1],b[1])-r,max(a[0],b[0])+r,max(a[1],b[1])+r])
    # Hangers to a beam on the next floor/roof. The short nozzle rests on its equipment.
    for i,c in enumerate(list(scene['components'])):
        if c['kind']!='pipe' or math.dist(*c['ends'])<1.0:continue
        a,b=c['ends'];p=(np.array(a)+b)/2
        if abs(a[2]-b[2])>.01:continue
        ceiling=next((z-.32 for z in (H,2*H,9.6) if z-.32>p[2]+.2),9.6)
        # Put the hanger clear of the equipment body; other intersections are checked separately.
        if any(line_hits(p,[p[0],p[1],ceiling],u['box'],.02) for u in units.values() if u['kind'] in ('vessel','pump','exchanger','filter','condenser')):continue
        support_x=min((0,6,12),key=lambda x:abs(x-p[0]))
        beam=[min(support_x,p[0])-.04,p[1]-.035,ceiling,max(support_x,p[0])+.04,p[1]+.035,ceiling+.08]
        from ..checks import box_overlaps
        if any(box_overlaps(beam,o['box']) for o in scene['components'] if o['kind'] in ('vessel','head','pump','motor','exchanger','filter','condenser','pipe')):continue
        if any(box_overlaps(beam,reserved_box(v)) or line_hits(p,[p[0],p[1],ceiling],reserved_box(v),.025) for v in valves):continue
        add_pipe(scene,f'hanger_{i}',p,[p[0],p[1],ceiling],.012,'hanger',(.35,.40,.44),c['owner'])
        add_box(scene,f'hanger_beam_{i}',beam,'beam',(.29,.38,.43))
    scene['holes']=dict(scene['holes'])
