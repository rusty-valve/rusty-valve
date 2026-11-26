"""Check room geometry and access to operating positions."""
from __future__ import annotations
from collections import deque
import math
import numpy as np
import mujoco
from ..checks import box_overlaps
from .process import RULES,validate
from .rule_identity import band
from .layout import H
from .routing import package
from .building import walking_height

POLICY=RULES['geometry_screen']; STEP=.10

def geom_bounds(model,data,ids):
    ids=np.asarray(ids,dtype=int)
    mats=data.geom_xmat[ids].reshape(-1,3,3)
    centers=data.geom_xpos[ids]+np.einsum('nij,nj->ni',mats,model.geom_aabb[ids,:3])
    half=np.einsum('nij,nj->ni',abs(mats),model.geom_aabb[ids,3:])
    return np.concatenate([centers-half,centers+half],axis=1)

def intersects(box,boxes,margin=0):
    if len(boxes)==0:return np.zeros(0,bool)
    return np.all(np.minimum(np.array(box)[3:]+margin,boxes[:,3:])-np.maximum(np.array(box)[:3]-margin,boxes[:,:3])>1e-7,axis=1)

def union_box(boxes):return [*boxes[:,:3].min(axis=0).tolist(),*boxes[:,3:].max(axis=0).tolist()]

def navigation(scene, obstacles,level, *, seeds=None):
    z=scene.get('floors',[0,H,2*H])[level];width,depth=scene.get('footprint',[12,6])
    nx,ny=int(round(width/STEP))+1,int(round(depth/STEP))+1; free=np.ones((nx,ny),bool)
    hw=POLICY['walk_width_m']/2;hd=POLICY['walk_depth_m']/2
    x=np.arange(nx)*STEP;y=np.arange(ny)*STEP
    terrain=np.array([[walking_height(scene,level,xx,yy,hw,hd) for yy in y] for xx in x]) if scene.get('building') else np.full((nx,ny),z)
    # A continuous deck with local penetrations. Footprint must stay clear of each hole.
    free[(x<hw)|(x>width-hw),:]=False;free[:,(y<hd)|(y>depth-hd)]=False
    for hole in scene['holes'].get(level,scene['holes'].get(str(level),[])):
        maskx=(x>hole[0]-hw)&(x<hole[2]+hw);masky=(y>hole[1]-hd)&(y<hole[3]+hd)
        free[np.ix_(maskx,masky)]=False
    for box in obstacles:
        maskx=(x>box[0]-hw)&(x<box[3]+hw);masky=(y>box[1]-hd)&(y<box[4]+hd)
        horizontal=maskx[:,None]&masky[None,:]
        free[horizontal & (box[5]>terrain+1e-6) & (box[2]<terrain+POLICY['walk_height_m']-1e-6)]=False
    # Entry is the front passage; staircase connection is checked separately.
    if seeds is None:
        entry=scene.get('navigation_entries',{}).get(str(level))
        seeds=([(int(round(entry[0]/STEP)),int(round(entry[1]/STEP)))] if entry else [(i,6) for i in range(108,116)])
    seeds=[p for p in seeds if 0<=p[0]<nx and 0<=p[1]<ny and free[p]]
    reachable=np.zeros_like(free);q=deque(seeds)
    for p in seeds:reachable[p]=True
    while q:
        a,b=q.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            p=(a+dx,b+dy)
            if 0<=p[0]<nx and 0<=p[1]<ny and free[p] and not reachable[p]:reachable[p]=True;q.append(p)
    return reachable,free

def valve_checks(g,scene,model,data):
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(model.ngeom)]
    valve_ids={v['id']:[i for i,n in enumerate(names) if n.startswith(v['id']+'_')] for v in g['valves']}
    checked_components=[c for c in scene['components'] if not c.get('visual_only',False)]
    env=[c['box'] for c in checked_components]
    bounds={v['id']:union_box(geom_bounds(model,data,valve_ids[v['id']])) for v in g['valves']}
    navobs=[c['box'] for c in checked_components if 'walk_level' not in c]+list(bounds.values());nav={lev:navigation(scene,navobs,lev) for lev in range(len(scene.get('floors',[0,H,2*H])))}
    width,depth=scene.get('footprint',[12,6]);floors=scene.get('floors',[0,H,2*H])
    results=[]
    for v in g['valves']:
        prefix=v['id']+'_'; _,params=package(v);geom=params['geometry'];mat=np.array(v['matrix']);center=np.array(v['center'])
        if v['operator']=='wheel':
            point=np.array(params['motion']['wheel_center_m']); movingbody=model.body(prefix+'handwheel').id
            joint=model.joint(prefix+'wheel_hinge'); ball=model.joint(prefix+'ball_hinge');travel=math.tau
        else:
            point=np.array([geom['handle_length_mm']*.75/1000,0,geom['handle_center_z_mm']/1000]);movingbody=model.body(prefix+'valve_rotor').id
            joint=model.joint(prefix+'valve_hinge');ball=None;travel=math.pi/2
        grip=center+mat@point;v['grip']=grip.tolist();ids=[i for i in valve_ids[v['id']] if model.geom_bodyid[i]==movingbody and model.geom_group[i]==3]
        if not ids:raise ValueError(v['id']+'Missing package motion collision geometry')
        other_names=[c['id'] for c in checked_components]+[k for k in bounds if k!=v['id']]
        obstacles=np.array(env+[b for k,b in bounds.items() if k!=v['id']])
        collisions=set()
        for angle in np.linspace(0,travel,int(math.degrees(travel)/5)+1):
            data.qpos[joint.qposadr]=angle
            if ball is not None:data.qpos[ball.qposadr]=angle/params['motion']['gear_ratio']
            mujoco.mj_kinematics(model,data)
            for box in geom_bounds(model,data,ids):
                hit=intersects(box,obstacles,POLICY['sweep_margin_m'])
                collisions.update(other_names[i] for i in np.where(hit)[0])
        data.qpos[joint.qposadr]=0
        if ball is not None:data.qpos[ball.qposadr]=0
        mujoco.mj_kinematics(model,data)
        v['sweep_clear']=not collisions; v['sweep_obstacles']=sorted(collisions)
        lev=v['level'];z=floors[lev];height=grip[2]-z;low,high=band()
        v['needs_elevation']=height>high;v['operator_height_m']=float(height)
        candidates=[]
        for distance in (.55,.65,.75):
            for angle in np.arange(0,2*math.pi,math.pi/8):
                p=[grip[0]+distance*math.cos(angle),grip[1]+distance*math.sin(angle),z]
                a=int(round(p[0]/STEP));b=int(round(p[1]/STEP))
                if not (0<=a<nav[lev][0].shape[0] and 0<=b<nav[lev][0].shape[1]):continue
                # Empty-stance envelope, plus independently computed connected walking space.
                hw=POLICY['station_width_m']/2;hd=POLICY['station_depth_m']/2
                base=walking_height(scene,lev,p[0],p[1],hw,hd) if scene.get('building') else z
                p[2]=base
                station=[p[0]-hw,p[1]-hd,base+.005,p[0]+hw,p[1]+hd,base+POLICY['station_height_m']]
                if station[0]<0 or station[1]<0 or station[3]>width or station[4]>depth:continue
                holes=scene['holes'].get(lev,scene['holes'].get(str(lev),[]))
                if any(station[0]<h[2] and station[3]>h[0] and station[1]<h[3] and station[4]>h[1] for h in holes):continue
                if not any(box_overlaps(station,box) for box in navobs):
                    candidates.append((bool(nav[lev][0][a,b]),p))
        v['stance_free']=bool(candidates);connected=[p for ok,p in candidates if ok]
        v['reachable_from_passage']=bool(connected)
        v['stance']=connected[0] if connected else (candidates[0][1] if candidates else None)
        if v['stance'] is not None:
            height=float(grip[2]-v['stance'][2]);v['operator_height_m']=height;v['needs_elevation']=height>high
        reasons=[]
        if not v['stance_free']:reasons.append('Standing volume is occupied')
        elif not connected:reasons.append('Standing area is disconnected from the aisle')
        if v['needs_elevation']:reasons.append('Requires elevated access')
        if height<low:reasons.append('Low operating posture requires verification')
        if collisions:reasons.append('Insufficient sweep clearance')
        v['eligible']=not reasons;v['reason']='；'.join(reasons) or 'Geometric screening passed'
        results.append({k:v[k] for k in ('id','stance_free','reachable_from_passage','needs_elevation','operator_height_m','sweep_clear','sweep_obstacles','eligible','reason')})
    return {'status':'pass' if all(v['eligible'] for v in g['valves']) else 'partial', 'eligible':sum(v['eligible'] for v in g['valves']),
        'all':len(g['valves']),'valves':results,'method':'Check standing and clearance dimensions, a 0.1 m connectivity grid, and package collision geometry every 5 degrees with 25 mm clearance. Robot IK and force need separate tests.'}

def geometry_checks(g,s,model,data):
    # All geometry remains inspectable even on failure. No hidden collision disabling.
    components=s['components']; names={c['id']:i for i,c in enumerate(components)}
    pipe_ids=[i for i,c in enumerate(components) if c['kind']=='pipe']
    obstacles=[i for i,c in enumerate(components) if c['kind'] in ('vessel','head','motor','column','beam','equipment_beam','rail','floor','pump','filter','exchanger','condenser')]
    boxes=np.array([c['box'] for c in components]); collisions=[]
    for i in pipe_ids:
        c=components[i];possible=np.where(intersects(c['box'],boxes[obstacles]))[0]
        for j in possible:
            k=obstacles[j];o=components[k]
            gid1=model.geom(c['id']).id;gid2=model.geom(o['id']).id
            distance=mujoco.mj_geomDistance(model,data,gid1,gid2,.1,None)
            if distance < -1e-5:collisions.append({'pipe':c['id'],'obstacle':o['id'],'penetration_m':float(-distance)})
    # Reuse the numeric bare-pipe separation rules. Connected elbows/tees are explicit exceptions.
    clearance=[];edge_map={e['id']:e for e in g['edges']}
    def joined(c,o):
        if c['owner']==o['owner']:return True
        e=edge_map.get(c['owner']);f=edge_map.get(o['owner'])
        if not e or not f:return False
        common={e['a'].split('.')[0],e['b'].split('.')[0]} & {f['a'].split('.')[0],f['b'].split('.')[0]}
        return any(g['units'][u]['kind']=='junction' and any(math.dist(p,q)<.06 for p in c['ends'] for q in o['ends']) for u in common)
    for a,i in enumerate(pipe_ids):
        c=components[i];axis=int(np.argmax(abs(np.array(c['ends'][1])-c['ends'][0])))
        for k in pipe_ids[a+1:]:
            o=components[k]
            if joined(c,o):continue
            other_axis=int(np.argmax(abs(np.array(o['ends'][1])-o['ends'][0])))
            if axis!=other_axis:
                if box_overlaps(c['box'],o['box']):
                    dist=mujoco.mj_geomDistance(model,data,model.geom(c['id']).id,model.geom(o['id']).id,.01,None)
                    if dist<-.00001:clearance.append({'pair':[c['id'],o['id']],'kind':'pipe_crossing','gap_m':float(dist)})
                continue
            if min(c['box'][axis+3],o['box'][axis+3])-max(c['box'][axis],o['box'][axis])<=1e-6:continue
            gap=math.sqrt(sum((c['ends'][0][k]-o['ends'][0][k])**2 for k in range(3) if k!=axis))-c['radius']-o['radius']
            if gap<POLICY['parallel_pipe_clearance_m']-1e-7:clearance.append({'pair':[c['id'],o['id']],'kind':'parallel','gap_m':gap})
        for o in components:
            if o['kind']!='column':continue
            gap=math.sqrt(sum(max(c['box'][k]-o['box'][k+3],o['box'][k]-c['box'][k+3],0)**2 for k in range(3)))
            if gap<POLICY['pipe_post_clearance_m']-1e-7:clearance.append({'pair':[c['id'],o['id']],'kind':'pipe_post','gap_m':gap})
    # Measured primitive contacts form a support path to ground. A label alone cannot pass.
    selected=[i for i,c in enumerate(components) if c['kind'] not in ('pipe','nozzle','rail','curb','safety','gauge','check_valve','hanger')]
    adjacency={i:[] for i in selected};contacts=0
    selected_boxes=boxes[selected]
    for a,i in enumerate(selected):
        for j0 in np.where(intersects(boxes[i],selected_boxes[a+1:],.003))[0]:
            j=selected[a+1+j0]
            if mujoco.mj_geomDistance(model,data,model.geom(components[i]['id']).id,model.geom(components[j]['id']).id,.01,None)<=.003:
                adjacency[i].append(j);adjacency[j].append(i);contacts+=1
    ground=names['ground'];anchored={ground};q=deque([ground])
    while q:
        for j in adjacency[q.popleft()]:
            if j not in anchored:anchored.add(j);q.append(j)
    support=[u['id'] for u in g['units'].values() if u['kind'] in ('vessel','pump','filter','exchanger','condenser','header') and names[u['id']] not in anchored]
    floating_decks=[components[i]['id'] for i in selected if components[i]['kind'] in ('floor','landing','step') and i not in anchored]
    support+=floating_decks
    stair_block=[]
    for step in components:
        if step['kind']!='step':continue
        b=step['box'];x=(b[0]+b[3])/2;y=(b[1]+b[4])/2;z=b[5]
        volume=[x-.04,y-.4,z+.01,x+.04,y+.4,z+POLICY['walk_height_m']]
        for o in components:
            if o['id']==step['id'] or o['kind']=='rail':continue
            if box_overlaps(volume,o['box']):stair_block.append({'step':step['id'],'obstacle':o['id']})
    dimensions=all(.55<=flight['run']+2*flight['rise']<=.70 and flight['width']>=.8 for flight in s['stairs'])
    return {'status':'fail' if collisions or support or clearance or stair_block or not dimensions else 'pass','pipe_solid_collisions':collisions,'clearance_violations':clearance,
            'missing_equipment_support':support,'support_contacts':contacts,'support_scope':'MuJoCo shape distances check contact, then support chains from equipment, platforms and stairs to the ground. Structural loads require a separate calculation.',
            'stair_headroom_obstacles':stair_block,'stair_dimensions_pass':dimensions,
            'pipe_route_axis_aligned':all(sum(abs(a[i]-b[i])>1e-7 for i in range(3))==1 for r in s['routing'] for a,b in zip(r['points'],r['points'][1:]))}

def check_room(g,s,model,data):
    report={'process':validate(g,True),'geometry':geometry_checks(g,s,model,data),'operation':valve_checks(g,s,model,data),
        'mujoco':{'status':'pass','nq':model.nq,'geoms':model.ngeom,'scope':'Physical model load and forward kinematics. Robot motion and forces require separate tests.'}}
    if report['geometry']['status']!='pass':
        for v in g['valves']:v['eligible']=False;v['reason']+='. Room geometry verification failed'
        report['operation']['eligible']=0;report['operation']['status']='room_rejected'
    return report
