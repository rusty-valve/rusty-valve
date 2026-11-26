"""Build drainage channels and their grating covers."""
import math

import mujoco
import numpy as np

from . import building
from .layout import add_box, add_pipe
from .building_parts.basic import Mesh
from .site_grating import EPS, contains, world_mesh
from .site_ground import BUND_WIDTH


WIDTH = .30
DEPTH = .20
GRADE = .005
WALL = .04
FALL_SPAN = .75


def plan(scene, boundary, drainage):
    datum = scene['floors'][0]
    points = sorted((row['position'] for row in drainage),key=lambda p:p[1])
    groups = []
    for point in points or [[(boundary[0]+boundary[2])/2, boundary[3]-.53, datum]]:
        if not groups or point[1]-groups[-1][0][1] > .16:
            groups.append([])
        groups[-1].append(point)
    channels = []
    for group in groups:
        y = sum(p[1] for p in group)/len(group)
        lo,hi = y-WIDTH/2-WALL,y+WIDTH/2+WALL
        x0,x1=boundary[0]+BUND_WIDTH+.08,boundary[2]-BUND_WIDTH-.08
        obstacles = []
        for part in scene['components']:
            if part['kind'] not in ('column','foundation','support','adapter','pipe','piping_flange'):
                continue
            b=part['box']
            if b[2]<datum-.002 and b[5]>datum-DEPTH and b[1]<hi and b[4]>lo:
                obstacles.append([b[0]-.018,lo,b[3]+.018,hi])
        spans=building.subtract([x0,lo,x1,hi],obstacles,min_size=.04)
        options=[r for r in spans if r[2]-r[0]>.40 and (not points or (
            r[0]+WALL < min(p[0] for p in group) and max(p[0] for p in group) < r[2]-WALL))]
        if not options:
            raise ValueError('No continuous drain channel near the connection can clear existing foundations')
        envelope=max(options,key=lambda r:r[2]-r[0])
        if not contains(boundary,envelope):
            raise ValueError('The drainage channel extends beyond the plinth')
        channels.append({'rect':envelope,'center_y':y,'inlets':group if points else [],
                         'clear_width_m':WIDTH,'datum_depth_m':DEPTH,'grade':GRADE})
    breaks = {boundary[1],boundary[3]}
    edges = []
    for channel in channels:
        a,b=channel['rect'][1],channel['rect'][3]
        edges.extend((a,b))
        breaks.update(p for p in (a,b,a-FALL_SPAN,b+FALL_SPAN) if boundary[1]<p<boundary[3])
    for a,b in zip(channels,channels[1:]):
        breaks.add((a['rect'][3]+b['rect'][1])/2)

    def surface(y):
        distance=min(abs(y-edge) for edge in edges)
        return datum-GRADE*max(0.,FALL_SPAN-distance)

    levels=sorted(breaks)
    cut=[c['rect'] for c in channels]
    rectangles=[r for a,b in zip(levels,levels[1:]) for r in building.subtract(
        [boundary[0],a,boundary[2],b],cut,min_size=.002)]
    return {'channels':channels,'floor_rectangles':rectangles,'surface':surface}


def build(scene, folder, plan_record, seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed[0],79,*seed[1:]]))
    profiles,changes=[],[]
    for index, row in enumerate(plan_record['channels']):
        x0,outer0,x1,outer1=row['rect']
        y0,y1=outer0+WALL,outer1-WALL
        datum=min(plan_record['surface'](outer0),plan_record['surface'](outer1))
        prefix=f'site_drain_{index}'
        downstream=datum-DEPTH-GRADE*(x1-x0)
        bed=building.solid(scene,folder,Mesh,prefix+'_bed',[x0,outer0,x1,outer1],datum-DEPTH,
                           downstream,.052,'drain',(.35,.38,.37),axis=0)
        walls=[]
        for side, (a,b) in enumerate(((outer0,y0),(y1,outer1))):
            wall=add_box(scene,f'{prefix}_side{side}',[x0,a,downstream-.03,x1,b,datum-.058],
                         'drain',building.CONCRETE)
            walls.append(wall['id'])
        for i,(a,b) in enumerate(((x0,x0+WALL),(x1-WALL,x1))):
            add_box(scene,f'{prefix}_end{i}',[a,y0,downstream-.02,b,y1,datum-.035],'drain',building.CONCRETE)
        add_pipe(scene,prefix+'_closed_outlet',[x1-.075,row['center_y'],downstream+.055],
                 [x1-WALL,row['center_y'],downstream+.055],.035,'drain',building.STEEL)
        for i,(x,y,z) in enumerate(row['inlets']):
            add_pipe(scene,f'{prefix}_inlet{i}',[x,y,z-.026],[x,y,z],.058,'drain',building.STEEL)
        length=x1-x0
        nominal=.50 if length<4. else 1.
        edges=np.linspace(x0,x1,max(1,math.ceil(length/nominal))+1)
        eligible=[i for i,(a,b) in enumerate(zip(edges,edges[1:]))
                  if all(not a-.13 < pos[0] < b+.13 for pos in row['inlets'])]
        count=min(int(rng.integers(0,3)),len(eligible))
        variants=dict(zip(rng.choice(eligible,size=count,replace=False),range(count))) if count else {}
        covers=[]
        visible=Mesh()
        for i,(a,b) in enumerate(zip(edges,edges[1:])):
            rect=[float(a)+.006,outer0,float(b)-.006,outer1]
            z0=z1=datum-.004
            name=f'{prefix}_cover_{i}'
            if i in variants and variants[i]%2==0:
                drop=float(rng.uniform(.010,.020))
                z0=z1=datum-drop
                changes.append({'part':name,'kind':'drain_sink','mm':round(drop*1000,2)})
            elif i in variants:
                lift=float(rng.uniform(.005,.010))
                z1=datum+lift
                changes.append({'part':name,'kind':'drain_lift','mm':round(lift*1000,2)})
            top=lambda x:z0+(z1-z0)*(x-rect[0])/(rect[2]-rect[0])
            building.grating(visible,rect,top,axis=0)
            cover=building.solid(scene,folder,Mesh,name,rect,z0,z1,.03,'drain_cover',building.SILVER,
                                 axis=0,hidden=True,walk_level=0)
            cover['alpha']=0.
            seats=[]
            for side,(low,high) in enumerate(((outer0,y0),(y1,outer1))):
                p=building.solid(scene,folder,Mesh,f'{name}_seat{side}',[rect[0],low,rect[2],high],
                                 z0-.0298,z1-.0298,.04,'drain',building.STEEL,axis=0)
                seats.append(p['id'])
            covers.append({'id':name,'seats':seats,'top_m':[z0,z1],'rect':rect})
        visible.save(scene,folder,prefix+'_gratings','grating',(.43,.47,.46))
        profiles.append({**row,'bed':bed['id'],'walls':walls,'covers':covers,'cover_datum_z_m':datum,
                         'bed_z_m':[datum-DEPTH,downstream],'outlet_z_m':downstream+.055})
    return profiles,changes


def audit(scene, model, data, channels):
    from .site_stairs import collisions
    errors=[]
    names=[]
    cache={}
    def points(name):
        return world_mesh(model,data,model.geom(name).id,cache)[0]
    for channel in channels:
        bed=points(channel['bed'])
        extent=np.ptp(bed[:,0])
        left=bed[np.isclose(bed[:,0],bed[:,0].min(),atol=2.e-6),2].max()
        right=bed[np.isclose(bed[:,0],bed[:,0].max(),atol=2.e-6),2].max()
        if abs((left-right)/extent-GRADE)>2.e-6:
            errors.append({'code':'DRAIN_GRADE','part':channel['bed']})
        for cover in channel['covers']:
            names.append(cover['id'])
            part=points(cover['id'])
            for seat,wall in zip(cover['seats'],channel['walls']):
                support=points(seat)
                for x in (part[:,0].min(),part[:,0].max()):
                    low=part[np.isclose(part[:,0],x,atol=2.e-6),2].min()
                    high=support[np.isclose(support[:,0],x,atol=2.e-6),2].max()
                    if abs(high-low)>.001:
                        errors.append({'code':'DRAIN_COVER_SUPPORT','part':cover['id'],'seat':seat})
                        break
                if mujoco.mj_geomDistance(model,data,model.geom(seat).id,model.geom(wall).id,.02,None)>.001:
                    errors.append({'code':'DRAIN_SEAT_SUPPORT','part':seat,'wall':wall})
    errors.extend({'code':'DRAIN_COVER_COLLISION',**h} for h in collisions(scene,model,data,names))
    return errors
