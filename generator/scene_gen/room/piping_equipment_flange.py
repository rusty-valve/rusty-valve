"""Build mating flanges and bolts for equipment ports."""
from collections import Counter, defaultdict
from functools import lru_cache
import math

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import mujoco

from .layout import add_pipe
from .piping_build import PipingRefused
from .storage_vessel_model import FLANGE_NUT_MM, PL_BORE_MM


def _area(pts):
    return .5*abs(np.sum(pts[:,0]*np.roll(pts[:,1],-1)-pts[:,1]*np.roll(pts[:,0],-1)))


def _circle(angles,r,x=0.):
    return np.column_stack((x+r*np.cos(angles),r*np.sin(angles)))


@lru_cache(maxsize=32)
def _sections(bore, outer, pitch, hole, count):
    """Convex partition of one angular sector, centred on one bolt hole."""
    arc_steps=max(2,24//count)
    angles=np.linspace(-math.pi/count,math.pi/count,arc_steps+1)
    # The polygonal bore is circumscribed, with no material inside B/2.
    inside=_circle(angles, bore/math.cos(math.pi/(count*arc_steps)))
    outside=_circle(angles,outer)
    if hole==0:
        return [np.array([inside[i],outside[i],outside[i+1],inside[i+1]]) for i in range(arc_steps)]
    boundary=np.concatenate((outside,inside[::-1]))
    # A circumscribed octagonal opening contains the nominal round hole; it
    # stays wholly inside the bolt's sector and clear of both circular edges.
    hole_pts=_circle(np.arange(8)*math.tau/8+math.pi/8,
                     hole/math.cos(math.pi/8),pitch)
    pts=np.concatenate((boundary,hole_pts));nh=len(boundary)
    hull=Delaunay(pts)
    # Centroid tests only classify faces after the edge checks below prove
    # that none of these triangles crosses a boundary of the drilled profile.
    inner_equations=ConvexHull(inside).equations if len(inside)>=3 else None
    hole_equations=ConvexHull(hole_pts).equations
    cells=[]
    for tri in hull.simplices:
        mid=pts[tri].mean(axis=0)
        if inner_equations is not None and np.max(inner_equations[:,:2]@mid+inner_equations[:,2])<=1e-10:continue
        if np.max(hole_equations[:,:2]@mid+hole_equations[:,2])<=1e-10:continue
        cells.append(list(tri))
    edges=Counter(tuple(sorted((a,b))) for cell in cells for a,b in zip(cell,cell[1:]+cell[:1]))
    wanted=set()
    for ring in (list(range(nh)),list(range(nh,len(pts)))):
        wanted.update(tuple(sorted((a,b))) for a,b in zip(ring,ring[1:]+ring[:1]))
    if {e for e,n in edges.items() if n==1}!=wanted:
        raise PipingRefused('Flange bore, bolt holes or outer edge do not form a closed section')
    # Merge neighbouring triangles only when the union is convex; MuJoCo
    # then sees the same solid for rendering and collision.
    i=0
    while i<len(cells):
        for j in range(i+1,len(cells)):
            if len(set(cells[i]).intersection(cells[j]))<2:continue
            ids=list(dict.fromkeys(cells[i]+cells[j]))
            h=ConvexHull(pts[ids])
            if abs(h.volume-_area(pts[cells[i]])-_area(pts[cells[j]]))<1e-12:
                cells[i]=[ids[k] for k in h.vertices];del cells[j]
                i=-1
                break
        i+=1
    return [pts[c] for c in cells]


def _mesh(path,polygon,height):
    verts=np.vstack((np.column_stack((polygon,np.zeros(len(polygon)))),
                     np.column_stack((polygon,np.full(len(polygon),height)))))
    if not path.exists():
        n=len(polygon)
        faces=[f for k in range(1,n-1) for f in ((0,k+1,k),(n,k+n,k+1+n))]
        for k in range(n):
            nxt=(k+1)%n
            faces.extend(((k,nxt,n+nxt),(k,n+nxt,n+k)))
        path.write_text(''.join('v %.15g %.15g %.15g\n'%tuple(v) for v in verts)+
                        ''.join('f %d %d %d\n'%tuple(i+1 for i in f) for f in faces))
    return verts


def emit_mating_flange(a,point,*,phase=None):
    """Replace this mouth's temporary cap hardware, leaving its neck in place."""
    dims=point['source_flange'];dn=int(point['source_dn'])
    p=np.asarray(point['position'],float);axis=np.asarray(point['normal'],float)
    axis=axis/np.linalg.norm(axis)
    radius=lambda key:float(dims[key])/2000
    t=dims['plate_thickness_mm']/1000;rf=dims['rf_height_mm']/1000
    reach=t+rf;n=int(dims['bolt_count']);pitch=radius('bolt_circle_mm')
    hole=radius('hole_diameter_mm');thread=int(dims['bolt_thread'][1:])
    try:
        bore=PL_BORE_MM[dn]/2000
    except KeyError as exc:
        raise PipingRefused(f'Equipment port DN{dn}Missing series A bore for the mating flange') from exc
    if bore<point['source_radius_m'] or thread/2000>=hole:
        raise PipingRefused('Neck or stud dimensions do not match the mating flange openings')
    source=point['equipment']+'_'+point['port']
    prefix=(phase or 'P1')+'_'+source+'_mating'
    ordered=point['source_contact_ids'];active=set(ordered)
    old={c['id']:c for c in a.scene['components'] if c['id'] in active}
    if set(old)!=active or len(ordered)!=len(active):
        raise PipingRefused('Missing or duplicate original equipment-port parts: '+source)
    groups=defaultdict(list)
    for cid in ordered:
        groups[old[cid]['kind']].append(old[cid])
    expected={'equipment_nozzle':1,'nozzle_flange':1,'raised_face':1,'flange_bolt':n,'bolt_head':n}
    if {key:len(parts) for key,parts in groups.items()}!=expected:
        raise PipingRefused('Incorrect original neck, flange or fastener count: '+source)
    neck_id=point['source_nozzle_id']
    if groups['equipment_nozzle'][0]['id']!=neck_id:
        raise PipingRefused('Equipment schedule neck identifier differs: '+source)
    def original_cylinder(part,r):
        ends=np.asarray(part.get('ends',[]),dtype=float)
        if (part.get('shape')!='cylinder' or ends.shape!=(2,3) or
                abs(part['radius']-r)>1e-8 or np.linalg.norm(np.cross(ends[1]-ends[0],axis))>1e-8):
            raise PipingRefused('Original nozzle cylinder dimensions or axis differ: '+part['id'])
        offsets=(ends-p)@axis
        transverse=(ends[0]-p)-offsets[0]*axis
        return transverse, sorted(offsets)
    replace=active-{neck_id}
    for kind,r,height in [('nozzle_flange',radius('outer_diameter_mm'),[-reach,-rf]),
                          ('raised_face',radius('rf_diameter_mm'),[-rf,0.])]:
        offset,limits=original_cylinder(groups[kind][0],r)
        if np.linalg.norm(offset)>1e-8 or not np.allclose(limits,height,atol=1e-8,rtol=0.):
            raise PipingRefused('Original sealing face or flange plate position differs: '+source)
    offset,limits=original_cylinder(groups['equipment_nozzle'][0],point['source_radius_m'])
    if np.linalg.norm(offset)>1e-8 or abs(limits[1])>1e-8 or limits[0]>=-1e-8:
        raise PipingRefused('The original neck does not end at the sealing face along the port normal: '+source)
    flats,nut_height=FLANGE_NUT_MM[thread]
    shank_r=thread/2000;flat=flats/2000;nut=nut_height/1000
    centres=[]
    for stud in groups['flange_bolt']:
        offset,limits=original_cylinder(stud,shank_r)
        if not np.allclose(limits,[-reach-nut,nut],atol=1e-8,rtol=0.):
            raise PipingRefused('Original short stud does not pass through the plate: '+stud['id'])
        centres.append(p+offset)
    remaining=set(range(n))
    for head in groups['bolt_head']:
        offset,limits=original_cylinder(head,flat)
        at=next((i for i in remaining if np.linalg.norm(p+offset-centres[i])<=1e-8),None)
        if at is None or not np.allclose(limits,[0.,nut],atol=1e-8,rtol=0.):
            raise PipingRefused('Original nut and short stud positions differ: '+head['id'])
        remaining.remove(at)
    # Use measured placement, including a 90-degree equipment rotation.
    x=(centres[0]-p)/np.linalg.norm(centres[0]-p);y=np.cross(axis,x)
    bearings=[math.atan2((c-p)@y,(c-p)@x) for c in centres]
    for k,theta in enumerate(bearings):
        if (abs(np.linalg.norm(centres[k]-p)-pitch)>1e-7 or
                abs(math.remainder(theta-k*math.tau/n,math.tau))>1e-7):
            raise PipingRefused('Equipment studs do not match the specified pitch or circle: '+source)
    meshes=a.folder/'equipment_flange_parts';meshes.mkdir(exist_ok=True)
    new=[];contacts=[];meta={'piping':True,'owner':prefix,'dn':dn,
                            'medium':point['medium'],'level':point['level']}
    if phase is not None:
        meta['phase']=phase
    layers={};flat_edges=defaultdict(list)
    for kind,outer,hole_radius,h,z0 in (('plate',radius('outer_diameter_mm'),hole,t,rf),
                                        ('rf',radius('rf_diameter_mm'),0,rf,0.)):
        polys=_sections(bore,outer,pitch,hole_radius,n)
        for side in ('equipment','pipe'):
            z=z0 if side=='pipe' else -z0-h
            layer=layers[side,kind]=[]
            for k,theta in enumerate(bearings):
                u=x*math.cos(theta)+y*math.sin(theta);v=np.cross(axis,u)
                rot=np.column_stack((u,v,axis));q=np.empty(4);mujoco.mju_mat2Quat(q,rot.ravel())
                origin=p+z*axis
                for j,poly in enumerate(polys):
                    path=meshes/f'DN{dn}_{kind}_{j:02}.obj'
                    verts=_mesh(path,poly,h);world=verts@rot.T+origin
                    cid=prefix+f'_{side}_{kind}_{k:02}_{j:02}'
                    part={**meta,'id':cid,'kind':'piping_flange','shape':'mesh','mesh_path':str(path.resolve()),
                          'collision_mesh':True,'pos':origin.tolist(),'quat':q.tolist(),
                          'box':np.r_[world.min(axis=0),world.max(axis=0)].tolist(),
                          'color':[.60,.66,.69] if kind=='plate' else [.47,.53,.56]}
                    new.append(part);layer.append(part)
                    face=world[:len(poly),:]
                    for a1,a2 in zip(face,np.roll(face,-1,axis=0)):
                        key=(side,kind,tuple(sorted((tuple(np.round(a1,9)),tuple(np.round(a2,9))))))
                        flat_edges[key].append((cid,(a1+a2+h*axis)/2,np.cross(a2-a1,axis)))
    for group in flat_edges.values():
        if len(group)==2:
            cid,m,norm=group[0];contacts.append((cid,group[1][0],m,norm))
    def register_plane(first,second,z):
        for left in first:
            for right in second:
                overlap=np.minimum(left['box'][3:],right['box'][3:])-np.maximum(left['box'][:3],right['box'][:3])
                if np.all(overlap>=-1e-8):contacts.append((left['id'],right['id'],p+z*axis,axis))
    register_plane(layers['pipe','rf'],layers['equipment','rf'],0.)
    for side,sign in (('equipment',-1),('pipe',1)):
        register_plane(layers[side,'plate'],layers[side,'rf'],sign*rf)
    bolts=[]
    stage={'components':new}
    for k,c in enumerate(centres):
        def boltpart(label,start,end,r,kind):
            obj=add_pipe(stage,prefix+f'_{label}{k+1:02}',(c+axis*start).tolist(),
                         (c+axis*end).tolist(),r,kind,(.30,.34,.36),prefix)
            obj.update(meta);return obj
        shank=boltpart('stud',-reach,reach,shank_r,'flange_bolt')
        lock=boltpart('nut',reach,reach+nut,flat,'bolt_nut')
        head=boltpart('head',-reach-nut,-reach,flat,'bolt_head')
        for obj,side,sign in ((head,'equipment',-1),(lock,'pipe',1)):
            contacts.append((shank['id'],obj['id'],c+axis*(sign*reach),axis))
            register_plane([obj],layers[side,'plate'],sign*reach)
        bolts.append({'shank':shank['id'],'nut':lock['id'],'head':head['id'],'hole_center':c.tolist(),
                      'shank_radius_m':shank_r,'hole_radius_m':hole,'pipe_back_face':(c+axis*reach).tolist()})
    # Publish only after both plates, all holes and hardware were constructed.
    a.scene['components'][:]=[c for c in a.scene['components'] if c['id'] not in replace]+new
    for left,right,pt,normal in contacts:
        a.allow(left,right,pt,normal=normal)
    ids=lambda parts:[r['id'] for r in parts]
    return {'end':(p+axis*reach).tolist(),'components':ids(new),
            'source_components':sorted(active-replace),'replaced_source_components':sorted(replace),
            'rf_components':ids(layers['pipe','rf']),'plate_components':ids(layers['pipe','plate']),
            'equipment_rf_components':ids(layers['equipment','rf']),
            'equipment_plate_components':ids(layers['equipment','plate']),
            'bore_radius_m':bore,'bolts':bolts}


def emit_flanged_stub(a,point,name,*,phase=None):
    """Fit the registered port's drilled flange pair and one axial pipe-side stub."""
    face=np.asarray(point['position'])
    joint=emit_mating_flange(a,point,phase=phase)
    stub=add_pipe(a.scene,a._id(name+'_flanged_stub'),face,np.asarray(joint['end']),
                  point['source_radius_m'],'adapter',a.color(point['medium']),name)
    stub.update(piping=True,dn=point['source_dn'],medium=point['medium'],level=point['level'])
    if phase=='P2':
        stub.update(phase='P2',source_edge=point.get('source_edge',point.get('id')),
                    equipment_endpoint=point['endpoint'])
    else:
        stub['source_requirement']=point['id']
    a.allow(point['source_nozzle_id'],stub['id'],face,normal=point['normal'])
    joint['pipe_stub']=stub['id']
    return joint
