"""Build open stair treads and the supports underneath them."""
import numpy as np

from . import building
from .layout import add_box
from .building_parts.basic import Mesh
from .site_grating import EPS
from .site_stairs import flights, process_obstacles


def tread(scene, mesh, name, rect, z, direction):
    building.grating(mesh, rect, lambda _:z, bearing_axis=1)
    part = add_box(scene,name,[*rect[:2],z-.03,*rect[2:],z],'step',building.SILVER)
    part['alpha'] = 0.
    edge = list(rect)
    if direction > 0:
        edge[2] = edge[0]+.023
    else:
        edge[0] = edge[2]-.023
    add_box(scene,name+'_nosing',[*edge[:2],z-.03,*edge[2:],z],
            'step_edge',(.42,.46,.46))


def build(scene, folder, seed, model, data, replaced):
    if not scene['stairs']:
        return [], []
    mesh = Mesh()
    changes, support = [], []
    original = {p['id']:p for p in scene['components']}
    rng = np.random.default_rng(np.random.SeedSequence([14, *seed]))
    for flight, existing in flights(scene):
        level, number = flight['level'], flight['flight']
        first, last = existing[0]['box'], existing[-1]['box']
        direction = -1 if first[0] > last[0] else 1
        start,end = (first[3],last[0]) if direction<0 else (first[0],last[3])
        a,b = first[5]-.06,last[5]-.06
        def stringer_top(x):
            return a+(b-a)*(x-start)/(end-start)
        x0,x1 = sorted((start,end))
        sides = first[1], first[4]
        stringers = []
        for side,y in enumerate(sides):
            name = f'stringer_{level}_{number}_{2*side-1}'
            building.solid(scene,folder,Mesh,name,[x0,y-.026,x1,y+.026],
                           stringer_top(x0),stringer_top(x1),.23,'stringer',building.STEEL,axis=0)
            stringers.append(name)
            if level==0 and number==0:
                add_box(scene,name+'_foot',[start-.09,y-.047,scene['floors'][0]-.025,
                                            start+.04,y+.047,max(scene['floors'][0]+.04,a-.10)],
                        'column',building.STEEL)
        choices = rng.choice(range(1,len(existing)-1),size=int(rng.integers(0,min(2,len(existing)-2)+1)),replace=False)
        offsets = {i:float(rng.choice((-1,1))*rng.uniform(.005,.010)) for i in choices}
        for i,old in enumerate(existing):
            name,box = old['id'],old['box']
            rect = [*box[:2],*box[3:5]]
            lap = 0.
            if i+1 < len(existing):
                lap = float(rng.uniform(.010,.035))
                rect[0 if direction<0 else 2] += direction*lap
                changes.append({'part':name,'kind':'tread_overlap','mm':round(1000*lap,2)})
            z=box[5]+offsets.get(i,0.)
            if i in offsets:
                changes.append({'part':name,'kind':'tread_step','mm':round(1000*offsets[i],2)})
            tread(scene,mesh,name,rect,z,direction)
            ends = []
            lower = min(stringer_top(rect[0]),stringer_top(rect[2]))-.032
            for side,y in enumerate(sides):
                end_plate = f'{name}_end{side}'
                add_box(scene,end_plate,[rect[0],y-.016,lower,rect[2],y+.016,z],
                        'step_edge',building.STEEL)
                ends.append(end_plate)
            support.append({'part':name,'ends':ends,'stringers':stringers,'overlap_m':lap})
    legs = {}
    for level in range(len(scene['floors'])-1):
        for side in ('turn','top'):
            name = f'{side}_landing_{level}'
            x0,y0,_,x1,y1,z = original[name]['box']
            rect = [x0,y0,x1,y1]
            building.grating(mesh,rect,lambda _:z)
            deck = add_box(scene,name,[x0,y0,z-.03,x1,y1,z],'landing',building.SILVER)
            deck['alpha']=0.
            for i,x in enumerate((x0+.026,x1-.026)):
                ends=[x-.038,y0,z-.24,x+.038,y1,z-.03]
                list(building.i_member(scene,f'{name}_edge{i}',ends,1))
            for i,y in enumerate((y0+.026,y1-.026)):
                ends=[x0,y-.038,z-.24,x1,y+.038,z-.03]
                list(building.i_member(scene,f'{name}_cross{i}',ends,0))
                for x in (x0+.026,x1-.026):
                    highest = legs.get((x,y))
                    if highest is None or z-.24 > highest[0]:
                        legs[x,y] = (z-.24,x0,x1)
    _, obstacles, _ = process_obstacles(scene,model,data,replaced)
    lower = scene['floors'][0]-.025
    for i,((x,y),(z,x0,x1)) in enumerate(legs.items()):
        # Keep the leg below the same landing beam and select the nearest
        # position that clears the frozen room and insulation.
        alternatives = [x,*sorted(np.arange(x0+.066,x1-.065,.035),key=lambda value:abs(value-x))]
        for center in alternatives:
            box = np.asarray([center-.033,y-.033,lower,center+.033,y+.033,z])
            if not np.any(np.all((obstacles[:,3:]>box[:3]+EPS)
                                 & (obstacles[:,:3]<box[3:]-EPS),axis=1)):
                break
        else:
            raise ValueError(f'Landing support cannot clear saved physical geometry at position {x:.3f}, {y:.3f}')
        list(building.i_member(scene,f'site_stair_column_{i}',box,2,'column'))
    mesh.save(scene,folder,'site_stair_gratings','grating',building.SILVER)
    return changes,support
