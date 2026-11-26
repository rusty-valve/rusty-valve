"""Check whether insulation obstructs a moving valve handle or handwheel."""
import math

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from . import saved_geometry as geometry

TOLERANCE = 1.e-6


def cuts_convex_surface(triangles, planes):
    """Clip likely crossing triangles against the solid operator's inner half-spaces."""
    lengths = triangles @ planes[:, :3].T + planes[:, 3]
    possible = (lengths < -TOLERANCE).any(axis=1).all(axis=1)
    for tri, offsets in zip(triangles[possible], lengths[possible]):
        vertices = tri
        for plane in planes[(offsets >= -TOLERANCE).any(axis=0)]:
            values = vertices@plane[:3]+plane[3]+TOLERANCE
            inside = values < 0.
            following = np.roll(vertices,-1,axis=0)
            after = np.roll(values,-1)
            polygon = []
            for i,(a,b) in enumerate(zip(vertices,following)):
                if inside[i]:
                    polygon.append(a)
                if inside[i] != (after[i]<0.):
                    polygon.append(a+(b-a)*values[i]/(values[i]-after[i]))
            if len(polygon)<3:
                break
            vertices = np.asarray(polygon)
        else:
            if np.linalg.norm(np.cross(vertices[1]-vertices[0],vertices[2:]-vertices[0]),axis=1).max()>1.e-12:
                return True
    return False


def operator_geometries(valve, model):
    controls = []
    for item in valve['local_geoms']:
        name=item['name']
        if not item['moving'] or not item.get('rgba',[0,0,0,1])[3]:
            continue
        if any(word in name for word in ('ball','seat')):
            continue
        # Rim and spokes are the real handwheel pieces when present.
        if name=='visual_handwheel' and any(c['name'].startswith('rim_') for c in valve['local_geoms']):
            continue
        controls.append((name,model.geom(valve['prefix']+name).id))
    return controls


def audit(graph, model, data, pieces, sweeps):
    mesh_cache, convex, skins = {}, {}, {}
    for piece in pieces:
        gid=model.geom(piece['id']).id
        vertices,faces=geometry.mesh(model,gid,mesh_cache)
        world=vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
        tris=world[faces]
        skins[piece['id']]=(tris,tris.min(axis=1),tris.max(axis=1))
    conflicts, quant = [], {}
    for valve in graph['valves']:
        uid=valve['id']
        controls=operator_geometries(valve,model)
        if not controls:
            raise ValueError('Operator geometry was not found: '+uid)
        states=valve['motion']['joint_states']
        travel=[]
        for name,start in states['open'].items():
            end=states['closed'][name]
            arc=math.copysign(min(abs(end-start),math.tau),end-start)
            travel.append((model.joint(valve['prefix']+name).qposadr[0],start,arc))
        frames=math.ceil(max(abs(a) for _,_,a in travel)/math.radians(5))+1
        quant[uid]=frames
        sweep=sweeps.get(uid)
        if sweep is None:
            raise ValueError('No full-travel record was generated: '+uid)
        lo,hi=np.split(sweep['bounds'],2)
        nearby=[piece for piece in pieces if np.all(np.asarray(piece['actual_box_m'][3:])+ .05 > lo)
                                             and np.all(np.asarray(piece['actual_box_m'][:3])- .05 < hi)]
        if not nearby:
            continue
        hits=set()
        for i in range(frames):
            for address,start,arc in travel:
                data.qpos[address]=start+arc*i/max(1,frames-1)
            mujoco.mj_kinematics(model,data)
            for name,gid in controls:
                if gid not in convex:
                    v,_=geometry.mesh(model,gid,mesh_cache)
                    planes = np.unique(ConvexHull(v).equations.round(9), axis=0)
                    convex[gid]=(v,planes)
                v,planes=convex[gid]
                matrix=data.geom_xmat[gid].reshape(3,3)
                position=data.geom_xpos[gid]
                world=v@matrix.T+position
                lower,upper=world.min(axis=0),world.max(axis=0)
                for piece in nearby:
                    key=piece['id']
                    if key in hits:
                        continue
                    all_tri,tl,th=skins[key]
                    close=np.all((tl < upper-TOLERANCE) & (th > lower+TOLERANCE),axis=1)
                    if close.any() and cuts_convex_surface((all_tri[close]-position)@matrix,planes):
                        hits.add(key)
                        conflicts.append({'valve':uid,'source_valve':valve.get('source_valve',uid),
                                          'insulation':key,'source':piece['source'],'operator_part':name,
                                          'fraction':i/max(1,frames-1)})
        data.qpos[:]=model.qpos0
    mujoco.mj_kinematics(model,data)
    return conflicts,quant
