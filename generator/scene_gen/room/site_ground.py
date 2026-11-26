"""Build the floor, containment curb and entrance ramps."""
import numpy as np

from . import building
from .checks import geom_bounds
from .layout import add_box
from .building_parts.basic import Mesh
from .site_grating import EPS, contains


SLAB_DEPTH = .20
BUND_HEIGHT = .30
BUND_WIDTH = .15
RAMP_RATIO = 6.
RAMP_WIDTH = 1.20
CURB = (.60, .61, .57)


def base_rect(scene, prior, drain_points):
    boundary = [*prior[:2],*prior[3:5]]
    for part in scene['components']:
        if part['kind'] in ('step','landing','stringer'):
            x0,y0,_,x1,y1,_ = part['box']
            boundary = [min(boundary[0],x0-.35),min(boundary[1],y0-.80),
                        max(boundary[2],x1+.35),max(boundary[3],y1+.35)]
    for x,y,_ in drain_points:
        boundary = [min(boundary[0],x-.48),min(boundary[1],y-.48),
                    max(boundary[2],x+.48),max(boundary[3],y+.48)]
    return boundary


def entrance_x(scene, boundary, rise):
    x0,y0,x1,_ = boundary
    z = scene['floors'][0]
    start,stop = x0+BUND_WIDTH+.12, x1-BUND_WIDTH-.12-RAMP_WIDTH
    structures = [p['box'] for p in scene['components'] if p['id']!='ground'
                  and not p.get('visual_only') and not p.get('annotation_only')]
    boxes = np.asarray(structures)
    near = boxes[(boxes[:,2] < z+2.) & (boxes[:,5] > z+EPS) & (boxes[:,1] < y0+BUND_WIDTH+rise*RAMP_RATIO)
                 & (boxes[:,4] > y0-rise*RAMP_RATIO)]
    first = sorted(np.arange(start,stop+EPS,.15),key=lambda x:abs(x+RAMP_WIDTH/2-1.10))
    for x in first:
        if not np.any((near[:,3]>x+EPS)&(near[:,0]<x+RAMP_WIDTH-EPS)):
            return float(x)
    raise ValueError('No usable two-ramp route connects the containment curb entrance to the floor')


def build(scene, folder, boundary, height=BUND_HEIGHT, *, drain=None):
    x0,y0,x1,y1 = boundary
    datum = scene['floors'][0]
    start = entrance_x(scene,boundary,height)
    stop = start+RAMP_WIDTH
    run = height*RAMP_RATIO
    rects = [boundary] if drain is None else drain['floor_rectangles']
    slab = []
    for i,r in enumerate(rects):
        name = f'site_ground_{i}'
        levels = [datum,datum] if drain is None else [drain['surface'](y) for y in (r[1],r[3])]
        building.solid(scene,folder,Mesh,name,r,*levels,SLAB_DEPTH,'ground',building.CONCRETE,walk_level=0)
        slab.append(name)
    extension = [x0,y0-run-.30,x1,y0]
    ext = add_box(scene,'site_entry_extension',[*extension[:2],datum-SLAB_DEPTH,*extension[2:],datum],
                  'entry_base',building.CONCRETE)
    ext.update(walk_level=0,top={'rect':extension,'z0':datum,'z1':datum,'axis':1})
    walls = {'front':[x0,y0,x1,y0+BUND_WIDTH], 'rear':[x0,y1-BUND_WIDTH,x1,y1],
             'left':[x0,y0+BUND_WIDTH,x0+BUND_WIDTH,y1-BUND_WIDTH],
             'right':[x1-BUND_WIDTH,y0+BUND_WIDTH,x1,y1-BUND_WIDTH]}
    for side,r in walls.items():
        add_box(scene,'site_bund_'+side,[*r[:2],datum,*r[2:],datum+height],'bund',CURB)
    ramp_rects = [[start,y0-run,stop,y0], [start,y0+BUND_WIDTH,stop,y0+BUND_WIDTH+run]]
    for side,rect,(lo,hi) in zip(('outside','inside'),ramp_rects,((datum,datum+height),(datum+height,datum))):
        building.solid(scene,folder,Mesh,'site_entry_ramp_'+side,rect,lo,hi,.08,'ramp',CURB,
                       walk_level=0,flat_bottom=datum-.02)
    return {'slab':slab,'slab_rect':list(boundary),'datum_z_m':datum,'slab_depth_m':SLAB_DEPTH,
            'bund_rect':list(boundary),'bund_height_m':height,'walls':walls,'extension':extension,
            'entry_ramps':ramp_rects,'ramp_grade':1./RAMP_RATIO}


def audit(scene, model, data, ground):
    from .site_grating import world_mesh
    violations=[]
    datum,height = ground['datum_z_m'],ground['bund_height_m']
    box = geom_bounds(model,data,[model.geom('site_bund_'+side).id for side in ground['walls']])
    if not all(np.allclose(bounds,[*r[:2],datum,*r[2:],datum+height],atol=EPS)
               for bounds,r in zip(box,ground['walls'].values())):
        violations.append({'code':'BUND_DATUM'})
    wall=ground['bund_rect'];inside=[wall[0]+BUND_WIDTH,wall[1]+BUND_WIDTH,wall[2]-BUND_WIDTH,wall[3]-BUND_WIDTH]
    for part in scene['components']:
        if part['kind'] not in ('step','landing','stringer'):
            continue
        box = part['box']
        if not contains(inside,[*box[:2],*box[3:5]]):
            violations.append({'code':'STAIR_OUTSIDE_BUND','part':part['id']})
    ramps=[]
    for side in ('outside','inside'):
        points,_=world_mesh(model,data,model.geom('site_entry_ramp_'+side).id,{})
        ramps.append(np.r_[points.min(axis=0),points.max(axis=0)])
    if any(abs(ramp[5]-datum-height)>2.e-6 or abs(ramp[4]-ramp[1]-height*RAMP_RATIO)>2.e-6 for ramp in ramps):
        violations.append({'code':'BUND_ENTRY_CREST'})
    ext=geom_bounds(model,data,[model.geom('site_entry_extension').id])[0]
    if not np.allclose(ext,[*ground['extension'][:2],datum-SLAB_DEPTH,*ground['extension'][2:],datum],atol=EPS):
        violations.append({'code':'BUND_ENTRY_SUPPORT'})
    return violations
