"""Build steel platforms, access stairs and drainage."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from .layout import add_box, add_pipe

RULES = json.loads(Path(__file__).with_name('building_rules.json').read_text())
STEEL=(.24,.31,.36); SILVER=(.53,.58,.59); CONCRETE=(.68,.69,.66)


def i_member(scene, name, box, axis, kind='beam', *, flange=.015, web=.010, placement=None):
    """Build an I section as two flanges and a continuous web."""
    lo, hi = np.asarray(box, dtype=float).reshape(2, 3)
    depth_axis = 1 if axis == 2 else 2
    width_axis = next(a for a in range(3) if a not in (axis, depth_axis))
    flange = min(flange, (hi[depth_axis]-lo[depth_axis])/4)
    web = min(web, (hi[width_axis]-lo[width_axis])/2)
    for suffix, start, stop in (
        ('flange0', lo[depth_axis], lo[depth_axis]+flange),
        ('flange1', hi[depth_axis]-flange, hi[depth_axis]),
        ('web', lo[depth_axis]+flange, hi[depth_axis]-flange),
    ):
        low, high = lo.copy(), hi.copy()
        low[depth_axis], high[depth_axis] = start, stop
        if suffix == 'web':
            middle = (lo[width_axis]+hi[width_axis])/2
            low[width_axis], high[width_axis] = middle-web/2, middle+web/2
        if placement is None:
            yield add_box(scene, name+'_'+suffix, np.r_[low, high].tolist(), kind, STEEL)
        else:
            origin, basis = placement
            center = origin+basis@((low+high)/2)
            half = (high-low)/2
            extent = np.abs(basis)@half
            part = add_box(scene, name+'_'+suffix, np.r_[center-extent,center+extent].tolist(), kind, STEEL)
            part['pose'] = {'pos':center.tolist(),'size':half.tolist(),
                            'xyaxes':np.r_[basis[:,0],basis[:,1]].tolist()}
            yield part


def prism(mesh, rect, z0, z1, depth, axis=1):
    """A convex strip, upper surface linear along x or y."""
    start=len(mesh.v);mesh.box([rect[0],rect[1],-depth],[rect[2],rect[3],0])
    for v in mesh.v[start:]:
        t=(v[axis]-rect[axis])/(rect[axis+2]-rect[axis])
        v[2]+=z0+(z1-z0)*t


def solid(s, folder, Mesh, name, rect, z0, z1, depth, kind, color, axis=1, hidden=False, walk_level=None, flat_bottom=None):
    m=Mesh();prism(m,rect,z0,z1,depth,axis)
    if flat_bottom is not None:
        for point in m.v[:4]:point[2]=flat_bottom
    c=m.save(s,folder,name,kind,color);c['collision_mesh']=True
    if hidden:c['group']=3
    if walk_level is not None:
        c['walk_level']=walk_level
        c['top']={'rect':list(rect),'z0':z0,'z1':z1,'axis':axis}
    return c


def grating(mesh, rect, top, depth=.03, axis=1, bearing_axis=1, holes=()):
    """20 mm pitch. Bars, edge frame and cross bars; no tiny weld/serration meshes."""
    x0,y0,x1,y1=rect
    def bar(r,h):
        for part in subtract(r,holes,min_size=.00001):
            prism(mesh,part,top(part[axis]),top(part[axis+2]),h,axis)
    if bearing_axis==1:
        for x in np.arange(x0,x1-.002,.02):bar([x,y0,min(x+.003,x1),y1],depth)
        for y in np.arange(y0,y1-.002,.1):bar([x0,y,x1,min(y+.005,y1)],.005)
    else:
        for y in np.arange(y0,y1-.002,.02):bar([x0,y,x1,min(y+.003,y1)],depth)
        for x in np.arange(x0,x1-.002,.1):bar([x,y0,min(x+.005,x1),y1],.005)
    for r in ([x0,y0,x0+.004,y1],[x1-.004,y0,x1,y1],[x0,y0,x1,y0+.004],[x0,y1-.004,x1,y1]):bar(r,depth)


def subtract(rect, holes, min_size=.004):
    rects=[rect]
    for hole in holes:
        out=[]
        for a,b,d,e in rects:
            if a<hole[2] and d>hole[0] and b<hole[3] and e>hole[1]:
                x0,x1=max(a,hole[0]),min(d,hole[2]);y0,y1=max(b,hole[1]),min(e,hole[3])
                out.extend([[a,b,x0,e],[x1,b,d,e],[x0,b,x1,y0],[x0,y1,x1,e]])
            else:out.append([a,b,d,e])
        rects=[r for r in out if r[2]-r[0]>min_size and r[3]-r[1]>min_size]
    return rects


def floor_z(y):
    return -.005*min(y,9.9-y)


def knee_targets(s):
    """Beam ends at supporting columns, independent of room size or bay count."""
    cfg=RULES['frame'];targets=[]
    for span in s['frame_spans']:
        ends=[np.array(p,float) for p in span['column_tops']]
        for i,(at,other) in enumerate((ends,ends[::-1])):
            direction=other-at;direction[2]=0;length=np.linalg.norm(direction);direction/=length
            top=at-np.array([0,0,span['beam_depth_m']])
            targets.append((f"knee_{span['id']}_{i}",top-np.array([0,0,cfg['knee_vertical_m']]),top+direction*min(cfg['knee_horizontal_m'],length/4)))
    return targets


def knee_frame(a, b):
    direction = (b-a)/np.linalg.norm(b-a)
    side = np.cross(direction, [0.,0.,1.])
    side /= np.linalg.norm(side)
    return np.column_stack((side, np.cross(direction, side), direction))


def i_knee(scene, name, a, b):
    half_width = RULES['frame']['knee_section_m']/2
    half_length = np.linalg.norm(b-a)/2
    half = np.array([half_width,half_width,half_length])
    for part in i_member(scene,name,np.r_[-half,half],2,flange=.008,web=.006,
                         placement=((a+b)/2,knee_frame(a,b))):
        part.update(frame_brace='knee',ends=[a.tolist(),b.tolist()])


def knee(s, folder, Mesh, name, a, b):
    width = RULES['frame']['knee_section_m']
    mat = knee_frame(a,b)
    half = np.linalg.norm(b-a)/2
    mesh = Mesh()
    mesh.box([-width/2,-width/2,-half], [width/2,width/2,half])
    mesh.v = [(mat@np.array(v)+(a+b)/2).tolist() for v in mesh.v]
    part = mesh.save(s, folder, name, 'beam', STEEL)
    part.update(collision_mesh=True, frame_brace='knee', ends=[a.tolist(),b.tolist()])
    return part


def upgrade_frame(s, folder, spec, Mesh):
    """Keep previous I columns/major beams/rails, replace decks, stairs and base."""
    s['components']=[c for c in s['components'] if c['kind'] not in ('ground','walk_surface','step','landing','grating')
                     and not c['id'].startswith(('brace','stringer'))]
    s['building']={'rules_version':RULES['version'],'panels':[],'stair_treads':[], 'drain_covers':[], 'defects':[],
                   'frame':'Short corner braces leave each bay face open for adjacent rooms. The model represents geometry only.'}
    for name,a,b in knee_targets(s):
        knee(s, folder, Mesh, name, a, b)
    for i,(x,y) in enumerate(((0,0),(6,0),(0,6),(6,6))):
        for zz in (2.75,2.95):
            add_box(s,f'node_stiffener{i}_{zz}',[x-.11,y-.11,zz,x+.11,y+.11,zz+.012],'beam',STEEL)
    # 17 open treads plus the landing = 18 equal rises. The old duplicated
    # last tread/landing is removed. The 30 mm underside/nosing is collidable.
    platform_z=spec['platform_z_m'];risers=RULES['stairs']['risers']
    rise=platform_z/risers;run=RULES['stairs']['going_m'];depth=RULES['stairs']['depth_m'];landing_start=(risers-1)*run
    stairmesh=Mesh()
    for i in range(risers-1):
        y=i*run;z=(i+1)*rise;r=[-1.30,y,-.20,y+depth]
        grating(stairmesh,r,lambda _,z=z:z,bearing_axis=0)
        c=add_box(s,f'stair_collision{i}',[r[0],r[1],z-.03,r[2],r[3],z],'step',STEEL);c['group']=3
        add_box(s,f'nosing{i}',[-1.30,y,z-.03,-.20,y+.025,z+.001],'step_edge',(.42,.46,.46))
        # End plates and bolted cleats connect each board to both stringers.
        for side,x in enumerate((-1.32,-.20)):
            add_box(s,f'tread_end{i}_{side}',[x,y,z-.10,x+.02,y+depth,z],'beam',STEEL)
        s['building']['stair_treads'].append({'id':c['id'],'rect':r,'top':z,'thickness':.03})
    for i,r in enumerate(([-1.30,4.76,-.655,5.96],[-.645,4.76,0,5.96])):
        grating(stairmesh,r,lambda _:3.0)
        c=add_box(s,f'landing_collision{i}',[r[0],r[1],2.97,r[2],r[3],3],'landing',STEEL);c['group']=3
    stairmesh.save(s,folder,'stair_grating','grating',SILVER)
    for i,x in enumerate((-1.34,-.18)):
        # Continuous sloping rectangular stringer; actual top touches end plates.
        inset=RULES['stairs']['stringer_top_below_tread_m']
        solid(s,folder,Mesh,f'stringer{i}',[x,0,x+.025,landing_start],rise-inset,platform_z-inset,.23,'beam',STEEL)
        add_box(s,f'stair_base{i}',[x-.09,-.08,-.02,x+.11,.14,.05],'column',CONCRETE)
    s['stairs']=[{'rise':rise,'run':run,'width':1.1,'steps':18,'open_treads':17,'landing':[-1.3,4.76,0,5.96]}]
    s['navigation_entries']={'0':[1.1,1.85],'1':[.5,5.35]}
    for component in s['components']:
        if component['kind']=='rail' or (component['kind']=='curb' and component['id'].endswith('_toe')):
            component['color']=RULES['rails']['color_rgb']
    ground(s,folder,Mesh)


def ground(s,folder,Mesh):
    # Raised slab; the channel really is absent from both visible/collision floor.
    rects=[[-1.8,-1.8,0,6.5],[0,-1.8,6.5,0],[6,0,6.5,6.5],[0,6,6,6.5],
           [0,0,6,4.76],[0,5.14,6,6],[0,4.76,.25,5.14],[5.75,4.76,6,5.14]]
    for i,r in enumerate(rects):
        # Outside the containment area stays level; interior drains to the channel.
        z0,z1=(floor_z(r[1]),floor_z(r[3])) if i>=4 else (0,0)
        solid(s,folder,Mesh,f'ground_{i}',r,z0,z1,.20,'ground',CONCRETE,walk_level=0)
    # Equipment/column foundations extend into the slab, not floating above a slope.
    add_box(s,'pump_foundation',[4.25,1.92,-.04,5.15,3.05,.001],'foundation',CONCRETE)
    add_box(s,'suction_support_foot',[3.92,1.12,-.02,4.08,1.28,.04],'foundation',CONCRETE)
    for i,x in enumerate((0,6)):
        add_box(s,f'rear_foundation{i}',[x-.2,5.8,-.05,x+.2,6.2,.001],'foundation',CONCRETE)
    # Channel bed falls towards x=5.75. Ledges carry independently seated covers.
    solid(s,folder,Mesh,'drain_bed',[.25,4.8,5.75,5.1],-.22,-.2475,.05,'drain',(.35,.38,.37),axis=0)
    for j,y in enumerate((4.76,5.1)):
        add_box(s,f'drain_wall{j}',[.21,y,-.27,5.79,y+.04,-.06],'drain',CONCRETE)
    drainmesh=Mesh()
    for i in range(11):
        x=.25+i*.5;r=[x+.006,4.76,x+.494,5.14];base=floor_z(4.8)-.004
        z0=z1=base;defect=None
        if i==3:z0=z1=floor_z(4.8)-.015;defect='Drain cover settlement of 15 mm'
        if i==7:z0=base;z1=floor_z(4.8)+.006;defect='Drain cover edge lifted 6 mm'
        top=lambda a,z0=z0,z1=z1,r=r:z0+(z1-z0)*(a-r[0])/(r[2]-r[0])
        # Cover bars span the channel (y). Small end rise represented by x slope.
        grating(drainmesh,r,top,axis=0)
        c=solid(s,folder,Mesh,f'drain_cover{i}',r,z0,z1,.03,'drain_cover',SILVER,axis=0,hidden=True,walk_level=0)
        for j,y in enumerate((4.76,5.105)):
            solid(s,folder,Mesh,f'drain_seat{i}_{j}',[r[0],y,r[2],y+.035],z0-.03,z1-.03,.04,'drain',STEEL,axis=0)
        s['building']['drain_covers'].append({'id':c['id'],'rect':r,'z':[z0,z1],'defect':defect})
        if defect:s['building']['defects'].append({'id':c['id'],'kind':defect,'source':'recorded site defect for simulation, outside the installation acceptance criteria'})
    drainmesh.save(s,folder,'drain_grating','grating',(.43,.47,.46))
    add_box(s,'drain_end_left',[.21,4.76,-.27,.25,5.14,-.024],'drain',CONCRETE)
    add_box(s,'drain_end_right',[5.75,4.76,-.27,5.79,5.14,-.024],'drain',CONCRETE)
    # Closed sump connection stays within the containment, no open discharge cut.
    add_pipe(s,'drain_closed_outlet',[5.71,4.95,-.20],[5.75,4.95,-.20],.025,'drain',SILVER)
    # The boundary is the complete raised slab, including the stair/access apron.
    # The projecting access ramp is separate and does not enlarge the slab boundary.
    slab=[c['box'] for c in s['components'] if c['kind']=='ground']
    x0,y0=min(c[0] for c in slab),min(c[1] for c in slab)
    x1,y1=max(c[3] for c in slab),max(c[4] for c in slab)
    spec=RULES['bund'];t=spec['thickness_m'];h=spec['height_m'];cx=1.1
    ex0,ex1=cx-spec['ramp_width_m']/2,cx+spec['ramp_width_m']/2
    for name,r in [('left',[x0,y0,x0+t,y1]),('right',[x1-t,y0,x1,y1]),('back',[x0+t,y1-t,x1-t,y1]),
                   ('front_left',[x0+t,y0,ex0,y0+t]),('front_right',[ex1,y0,x1-t,y0+t])]:
        add_box(s,'bund_'+name,[r[0],r[1],-.05,r[2],r[3],h],'bund',(.61,.61,.54))
    entry=spec['entry_ramp'];exterior_z=0.0
    run=h*entry['run_per_rise'];outer_run=inner_run=run
    toe=entry['approach_length_m']
    base=add_box(s,'base_entry_extension',[x0,y0-run-toe,-.20,x1,y0,0],'entry_base',CONCRETE)
    base['walk_level']=0;base['top']={'rect':[x0,y0-run-toe,x1,y0],'z0':0,'z1':0,'axis':1}
    # Two identical wedges, mirrored about the centre of the wall crest. A flat
    # underside bears on the continuous base; no detached plank or sloping underside.
    for name,rect,za,zb in [('ramp_out',[ex0,y0-run,ex1,y0],0,h),('ramp_in',[ex0,y0+t,ex1,y0+t+run],h,0)]:
        solid(s,folder,Mesh,name,rect,za,zb,h+.002,'ramp',(.61,.63,.59),walk_level=0,flat_bottom=-.002)
    solid(s,folder,Mesh,'ramp_crest',[ex0,y0,ex1,y0+t],h,h,h+.002,'ramp',(.61,.63,.59),walk_level=0,flat_bottom=-.002)
    s['building']['drain']={'rect':[.25,4.8,5.75,5.1],'bed_z':[-.22,-.2475],'outlet':'controlled closed connection with no external stormwater tie-in'}
    s['building']['bund']={'height_m':h,'thickness_m':t,'boundary':[x0,y0,x1,y1],'entry':'The plinth extends along the entrance edge. Identical mirrored ramps cross the curb at the boundary.',
        'entry_center':[cx,y0],'ramp_path_y':[y0-outer_run-.25,1.85],'ramp_slope':1/entry['run_per_rise'],'exterior_ground_z':exterior_z,'ramp_total_length_m':outer_run+t+inner_run}



def panel(s, folder, Mesh, visible, name, rect, top, splits, holes, level):
    """Render the grating bars and save the matching split contact surface."""
    collision = []
    for k, (y0,y1) in enumerate(zip(splits, splits[1:])):
        span = [rect[0],y0,rect[2],y1]
        grating(visible, span, top, holes=holes)
        for j, piece in enumerate(subtract(span, holes)):
            part = solid(s, folder, Mesh, name+f'_contact{k}_{j}', piece,
                         top(piece[1]), top(piece[3]), .03, 'walk_surface', SILVER,
                         hidden=True, walk_level=level)
            collision.append(part['id'])
    return collision


def deck(s,folder,spec,Mesh):
    rng=np.random.default_rng(RULES['seed']);z=spec['platform_z_m']
    xs=[0,.575,1.15,1.725,2.3,3.1,3.9,4.425,4.95,5.475,6]
    ys=[0,1.2,2.4,3.2,4,5,6]
    # Shared seam lines; a single draw determines both neighbours, no overlaps.
    gx=[.012]+rng.uniform(*RULES['panel']['seam_range_m'],len(xs)-2).tolist()+[.012]
    gy=[.012]+rng.uniform(*RULES['panel']['seam_range_m'],len(ys)-2).tolist()+[.012]
    mesh=Mesh();fixtures=Mesh();panel_count=0
    holes=s['holes'][1];equipment_count=s.get('equipment_opening_count',1)
    for ix,(a,b) in enumerate(zip(xs,xs[1:])):
        for iy,(c,d) in enumerate(zip(ys,ys[1:])):
            raw=[a+gx[ix]/2,c+gy[iy]/2,b-gx[ix+1]/2,d-gy[iy+1]/2]
            for ip,r in enumerate(subtract(raw,holes[:equipment_count])):
                r=[v+(.005 if k<2 else -.005) if abs(v-raw[k])>1e-7 else v for k,v in enumerate(r)]
                if r[2]-r[0]<.025 or r[3]-r[1]<.025:continue
                ident=f'panel_{ix}_{iy}_{ip}';raised=RULES['panel']['raised_edge_m'] if RULES['panel']['defect_count']>=1 and (ix,iy,ip)==(1,1,0) else 0
                sag=RULES['panel']['sag_m'] if RULES['panel']['defect_count']>=2 and (ix,iy,ip)==(2,3,0) else 0
                # Slight in-plane irregularity derives from varying seam widths;
                # the two selected defects are explicitly marked, not hidden.
                split=[r[1],r[1]+.035,(r[1]+r[3])/2,r[3]-.035,r[3]] if sag else [r[1],r[3]]
                top=lambda y,r=r,raised=raised,sag=sag:z+raised-sag*max(0,1-abs(2*(y-r[1]-.035)/(r[3]-r[1]-.07)-1))
                collision=panel(s,folder,Mesh,mesh,ident,r,top,split,holes[equipment_count:],1)
                supports=[]
                # Full cell-width cross ledgers tie into longitudinal framing;
                # 35 mm bearing inside each panel, including trimmed hole panels.
                for k,y in enumerate((r[1],r[3]-.035)):
                    seat=add_box(s,ident+f'_seat{k}',[r[0],y,top(y if k==0 else r[3])-.07,r[2],y+.035,top(y if k==0 else r[3])-.03],'beam',STEEL)
                    supports.append(seat['id'])
                    # 4 simplified retaining clips per panel, batched for display.
                    for x in (r[0]+min(.025,(r[2]-r[0])/4),r[2]-min(.025,(r[2]-r[0])/4)):
                        fixtures.box([x-.009,y+.008,top(r[1] if k==0 else r[3])],[x+.009,y+.026,top(r[1] if k==0 else r[3])+.002])
                # Two longitudinal edge bearers connect seats back to the main grid;
                # extend to cell ends but never across the vessel/pipe openings.
                for k,x in enumerate((r[0],r[2]-.025)):
                    for j,rr in enumerate(subtract([x,c,x+.025,d],holes)):
                        add_box(s,ident+f'_edge{k}_{j}',[rr[0],rr[1],z-.11,rr[2],rr[3],z-.07+raised],'beam',STEEL)
                # Transverse ties reach the retained I joists, with opening cuts.
                panel_count+=1
                row={'id':ident,'rect':r,'top_ends':[top(r[1]),top(r[3])],'sag_m':sag,'raised_m':raised,'supports':supports,'collision':collision,'bearing_axis':'y'}
                s['building']['panels'].append(row)
                if raised or sag:s['building']['defects'].append({'id':ident,'kind':f'raised panel edge{raised*1000:g}mm' if raised else f'center depression{sag*1000:g}mm','source':'recorded site defect for simulation, outside the installation acceptance criteria'})
    # Main secondary beams span x, carry y-spanning boards; split around openings.
    for i,y in enumerate(ys):
        for j,r in enumerate(subtract([0,max(0,y-.04),6,min(6,y+.04)],holes)):
            add_box(s,f'secondary_cross{i}_{j}',[r[0],r[1],z-.16,r[2],r[3],z-.07],'beam',STEEL)
    mesh.save(s,folder,'actual_grating','grating',SILVER)
    fixtures.save(s,folder,'grating_clips','fastener',(.33,.37,.38))['visual_only']=True
    s['building']['panel_grid']={'x':xs,'y':ys,'x_seams':gx,'y_seams':gy}


def walking_height(scene, level, x, y, half_x=0, half_y=0):
    """Highest actual walking surface under a footprint; used only for body clearance.

    Small seams/steps are recorded separately, never certified walkable by this map.
    """
    heights=[]
    for c in scene['components']:
        if c.get('walk_level')!=level:continue
        t=c['top'];r=t['rect']
        if x+half_x<r[0] or x-half_x>r[2] or y+half_y<r[1] or y-half_y>r[3]:continue
        axis=t['axis'];center=(x,y)[axis];half=(half_x,half_y)[axis]
        for p in (max(r[axis],center-half),min(r[axis+2],center+half)):
            heights.append(t['z0']+(t['z1']-t['z0'])*(p-r[axis])/(r[axis+2]-r[axis]))
    return max(heights) if heights else scene['floors'][level]


def check(s,model,data):
    import mujoco
    from collections import deque
    b=s['building'];cs={c['id']:c for c in s['components']};errors=[];deviations=[]
    panels=b['panels'];bearings=[];seams=[]
    # Convex separating-axis test on compiled MuJoCo vertices. mj_geomDistance
    # reports inconsistent positive distances for touching mesh/box pairs on the
    # installed version (coincident closest points); do not loosen the tolerance.
    polycache={}
    def poly(name):
        if name in polycache:return polycache[name]
        gid=model.geom(name).id
        if model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
            mid=model.geom_dataid[gid];va=model.mesh_vertadr[mid];vn=model.mesh_vertnum[mid]
            fa=model.mesh_faceadr[mid];fn=model.mesh_facenum[mid]
            points=model.mesh_vert[va:va+vn];faces=model.mesh_face[fa:fa+fn]
        else:
            h=model.geom_size[gid]
            points=np.array([[x,y,z] for x in (-h[0],h[0]) for y in (-h[1],h[1]) for z in (-h[2],h[2])])
            faces=np.array([[0,1,3],[0,2,3],[4,5,7],[4,6,7],[0,1,5],[0,4,5],[2,3,7],[2,6,7],[0,2,6],[0,4,6],[1,3,7],[1,5,7]])
        points=np.asarray(points)@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
        edges=np.concatenate([points[faces[:,j]]-points[faces[:,(j+1)%3]] for j in range(3)])
        normals=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
        def unique(v):
            length=np.linalg.norm(v,axis=1);v=v[length>1e-9]/length[length>1e-9,None]
            return np.unique(np.round(v,7),axis=0)
        result=(points,unique(normals),unique(edges));polycache[name]=result;return result
    def distance(a,d):
        ap,an,ae=poly(a);bp,bn,be=poly(d)
        cross=np.cross(ae[:,None,:],be[None,:,:]).reshape(-1,3);length=np.linalg.norm(cross,axis=1)
        cross=cross[length>1e-7]/length[length>1e-7,None]
        axes=np.concatenate((an,bn,cross));pa=ap@axes.T;pb=bp@axes.T
        return float(np.max(np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0))))
    landing_intrusions=[]
    for stair in s['stairs']:
        r=stair['landing'];z=s['floors'][1]
        # Inspect compiled structural geometry in the entrance strip, including
        # both stringer ends. Railings are intentional boundaries, not beams.
        for c in s['components']:
            if c['kind'] not in ('beam','column') or c.get('visual_only'):continue
            points=poly(c['id'])[0];lo=points.min(0);hi=points.max(0)
            if hi[0]>r[0]-.06 and lo[0]<r[2] and hi[1]>r[1]-.001 and lo[1]<r[1]+.20 and lo[2]<z+.2 and hi[2]>z+.004:
                landing_intrusions.append({'part':c['id'],'above_floor_m':float(hi[2]-z)})
    if landing_intrusions:errors.append({'kind':'stair_landing_protrusion','parts':landing_intrusions})
    for p in panels:
        r=p['rect']
        for j,seatid in enumerate(p['supports']):
            seat=cs[seatid]['box'];contact_distance=min(distance(c,seatid) for c in p['collision'])
            bearing=max(0,min(r[3],seat[4])-max(r[1],seat[1]));bearings.append(bearing)
            if bearing<.025-1e-6 or contact_distance>.0001:errors.append({'kind':'panel_bearing','panel':p['id'],'seat':seatid,'bearing_m':bearing})
    # Independent support connectivity: actual compiled geom contact, panel ->
    # ledgers/secondary -> existing major beams/columns -> foundations/slab.
    solid=[c for c in s['components'] if c['kind'] in ('beam','column','foundation','ground','walk_surface','step','landing') and not c.get('visual_only') and c['shape']!='cylinder']
    boxes=np.array([c['box'] for c in solid]);adj=[[] for _ in solid]
    for i,a in enumerate(solid):
        candidates=np.where(np.all(np.minimum(boxes[i,3:]+.001,boxes[i+1:,3:])-np.maximum(boxes[i,:3]-.001,boxes[i+1:,:3])>=0,axis=1))[0]+i+1
        for j in candidates:
            if distance(a['id'],solid[j]['id'])<=.001:adj[i].append(j);adj[j].append(i)
    anchored={i for i,c in enumerate(solid) if c['kind']=='ground'};queue=deque(anchored)
    while queue:
        for j in adj[queue.popleft()]:
            if j not in anchored:anchored.add(j);queue.append(j)
    floating=[c['id'] for i,c in enumerate(solid) if c['kind'] in ('walk_surface','step','landing') and i not in anchored]
    if floating:errors.append({'kind':'missing_support_path','parts':floating})
    for i,p in enumerate(panels):
        for q in panels[i+1:]:
            a=p['rect'];d=q['rect']
            for axis in (0,1):
                other=1-axis;overlap=min(a[other+2],d[other+2])-max(a[other],d[other])
                if overlap<.05:continue
                gap=max(a[axis],d[axis])-min(a[axis+2],d[axis+2])
                if -.00001<=gap<=.02001:
                    seams.append(gap)
                    if gap<(.010 if axis==1 else .006)-1e-6 or gap>.017+1e-6:errors.append({'kind':'seam','panels':[p['id'],q['id']],'gap_m':gap})
                    # Sample the entire shared edge including sag centre.
                    for y in (max(a[1],d[1]),(max(a[1],d[1])+min(a[3],d[3]))/2,min(a[3],d[3])) if axis==0 else (a[3] if a[1]<d[1] else a[1],):
                        def height(panel,y):
                            r=panel['rect'];t=np.clip((y-r[1])/(r[3]-r[1]),0,1)
                            return panel['top_ends'][0]-panel['sag_m']*(1-abs(2*t-1))
                        delta=abs(height(p,y)-height(q,y))
                        if delta>.004+1e-6:
                            known=bool(p['sag_m'] or p['raised_m'] or q['sag_m'] or q['raised_m'])
                            row={'panels':[p['id'],q['id']],'edge_step_mm':round(delta*1000,3),'declared_defect':known}
                            deviations.append(row)
                            if not known:errors.append({'kind':'unexpected_panel_step',**row})
    # Probe actual collision surfaces at the two feature centres. Only group 3
    # includes hidden walking proxies: catches accidental flat fill under a sag.
    probes=[]
    for p in panels:
        if not (p['sag_m'] or p['raised_m']):continue
        r=p['rect'];xy=[(r[0]+r[2])/2,(r[1]+r[3])/2];gid=np.array([-1],dtype=np.int32)
        hit=mujoco.mj_ray(model,data,np.array([*xy,3.3]),np.array([0.,0.,-1.]),np.array([0,0,0,1,0,0],dtype=np.uint8),1,-1,gid)
        actual=3.3-hit if hit>=0 else None;expected=p['top_ends'][0]-p['sag_m']
        probes.append({'id':p['id'],'expected_z':expected,'collision_z':actual})
        if actual is None or abs(actual-expected)>.0001:errors.append({'kind':'flattened_panel_collision','id':p['id']})
    stairs=b['stair_treads'];overlaps=[];openings=[]
    for i,t in enumerate(stairs):
        nxt=stairs[i+1] if i+1<len(stairs) else {'rect':[-1.3,4.76,0,5.96],'top':3.,'thickness':.03}
        overlap=t['rect'][3]-nxt['rect'][1];opening=nxt['top']-nxt['thickness']-t['top']
        overlaps.append(overlap);openings.append(opening)
        if not .01-1e-6<=overlap<=.035+1e-6 or opening<=0:errors.append({'kind':'stair_overlap_or_opening','id':t['id']})
    # Channel must have no coincident slab, and covers must bear on both ledges.
    drain_probe=[]
    for p in b['drain_covers']:
        r=p['rect'];x=(r[0]+r[2])/2;y=4.95
        for c in s['components']:
            if c['kind']=='ground' and c['box'][0]<x<c['box'][3] and c['box'][1]<y<c['box'][4]:errors.append({'kind':'filled_drain','cover':p['id']})
        for j in range(2):
            if distance(p['id'],p['id'].replace('cover','seat')+f'_{j}')>.0001:errors.append({'kind':'drain_cover_bearing','id':p['id']})
        drain_probe.append({'id':p['id'],'edge_recess_mm':[round((floor_z(4.8)-z)*1000,3) for z in p['z']],'defect':p['defect']})
    # Sample the entire containment boundary. Ramp crest is part of the wall.
    slab=[c['box'] for c in s['components'] if c['kind']=='ground']
    bx0,by0=min(c[0] for c in slab),min(c[1] for c in slab)
    bx1,by1=max(c[3] for c in slab),max(c[4] for c in slab)
    inset=RULES['bund']['thickness_m']/2
    boundary=[(x,y) for x in np.linspace(bx0+inset,bx1-inset,120) for y in (by0+inset,by1-inset)]+[(x,y) for x in (bx0+inset,bx1-inset) for y in np.linspace(by0+inset,by1-inset,120)]
    containment=[c for c in s['components'] if c['kind']=='bund' or c['id']=='ramp_crest'];heights=[]
    for x,y in boundary:
        tops=[c['box'][5] for c in containment if c['box'][0]-1e-6<=x<=c['box'][3]+1e-6 and c['box'][1]-1e-6<=y<=c['box'][4]+1e-6]
        if not tops:errors.append({'kind':'bund_gap','at':[x,y]});break
        heights.append(max(tops))
    if heights and min(heights)<.15-1e-6:errors.append({'kind':'bund_too_low'})
    # Check compiled shape and actual ramp extent, not just a parameter label.
    wrong_walls=[c['id'] for c in containment if c['kind']=='bund' and (c['shape']!='box' or int(model.geom_type[model.geom(c['id']).id])!=int(mujoco.mjtGeom.mjGEOM_BOX))]
    if wrong_walls:errors.append({'kind':'bund_not_rectangular','parts':wrong_walls})
    ratio=RULES['bund']['entry_ramp']['run_per_rise'];grade=RULES['bund']['entry_ramp']['outer_walkway_offset_m'];wall_height=RULES['bund']['height_m']
    outer_points=poly('ramp_out')[0];inner_points=poly('ramp_in')[0];crest_points=poly('ramp_crest')[0]
    outer_run=float(np.ptp(outer_points[:,1]));inner_run=float(np.ptp(inner_points[:,1]));crest_run=float(np.ptp(crest_points[:,1]))
    if abs(outer_run-(wall_height-grade)*ratio)>.0001 or abs(inner_run-wall_height*ratio)>.0001:errors.append({'kind':'ramp_not_local_threshold','runs_m':[outer_run,inner_run]})
    # The two compiled wedges must be geometric mirror images. The outside one
    # must be fully supported by a slab extension touching the original base.
    mirror_y=(crest_points[:,1].min()+crest_points[:,1].max())/2
    reflected=outer_points.copy();reflected[:,1]=2*mirror_y-reflected[:,1]
    def ordered(points):
        points=np.round(points,6)
        return points[np.lexsort((points[:,2],points[:,1],points[:,0]))]
    symmetric=reflected.shape==inner_points.shape and np.allclose(ordered(reflected),ordered(inner_points),atol=2e-6)
    if not symmetric:errors.append({'kind':'ramp_asymmetry'})
    extension=cs.get('base_entry_extension');extension_ok=False
    if extension is not None:
        eb=extension['box'];opmin=outer_points.min(0);opmax=outer_points.max(0)
        extension_ok=eb[0]<=opmin[0]+1e-6 and eb[3]>=opmax[0]-1e-6 and eb[1]<=opmin[1]+1e-6 and abs(eb[4]-by0)<1e-6 and abs(eb[5])<1e-6 and eb[2]<=-.20+1e-6
    if not extension_ok:errors.append({'kind':'missing_ramp_base_extension'})
    full_width=extension is not None and abs(extension['box'][0]-bx0)<1e-6 and abs(extension['box'][3]-bx1)<1e-6
    if not full_width:errors.append({'kind':'base_extension_not_full_width'})
    # A body-sized route over the ramp; only sloped walking pieces are exempt.
    blocked=[]
    for y in np.linspace(*b['bund']['ramp_path_y'],60):
        z=walking_height(s,0,1.1,y,.30,.25)
        volume=np.array([.8,y-.25,z+.01,1.4,y+.25,z+1.8])
        for c in s['components']:
            if c.get('visual_only') or 'walk_level' in c:continue
            cb=np.array(c['box'])
            if np.all(np.minimum(volume[3:],cb[3:])-np.maximum(volume[:3],cb[:3])>1e-6):blocked.append(c['id'])
    if blocked:errors.append({'kind':'ramp_passage_blocked','obstacles':sorted(set(blocked))})
    pipe_holes=s['holes'].get(1,s['holes'].get('1'))[s.get('equipment_opening_count',1):];frames=[]
    for hole in pipe_holes:
        near=np.array([hole[0]-.03,hole[1]-.03,2.86,hole[2]+.03,hole[3]+.03,2.975])
        for c in s['components']:
            if c['kind']=='beam' and c['id'].startswith('panel_'):
                cb=np.array(c['box'])
                if np.all(np.minimum(near[3:],cb[3:])-np.maximum(near[:3],cb[:3])>1e-6):frames.append(c['id'])
    if frames:errors.append({'kind':'extra_pipe_hole_frame','parts':frames})
    rails=[c for c in s['components'] if c['kind']=='rail' or (c['kind']=='curb' and c['id'].endswith('_toe'))]
    wrong_colors=[c['id'] for c in rails if not np.allclose(c['color'],RULES['rails']['color_rgb'])]
    if wrong_colors:errors.append({'kind':'rail_color','parts':wrong_colors})
    knees=[c for c in s['components'] if c.get('frame_brace')=='knee']
    bad_braces=[]
    for c in s['components']:
        if c['kind']!='beam' or 'ends' not in c:continue
        a,bb=map(np.array,c['ends']);span=np.linalg.norm((bb-a)[:2]);rise=abs(bb[2]-a[2])
        if span>1e-4 and rise>1e-4 and (span>RULES['frame']['knee_horizontal_m']+.01 or c['box'][2]<RULES['frame']['min_clear_height_m']):bad_braces.append(c['id'])
    expected={name for name,_,_ in knee_targets(s)}
    if {c['id'] for c in knees}!=expected:errors.append({'kind':'missing_knee_brace','expected':sorted(expected),'actual':[c['id'] for c in knees]})
    if bad_braces:errors.append({'kind':'full_bay_brace_blocks_passage','parts':bad_braces})
    return {'status':'pass' if not errors else 'fail','errors':errors,'installation_status':'Modeled site defects remain and are recorded separately from installation acceptance.',
        'short_knee_braces':len(knees),'full_bay_cross_braces':len(bad_braces),'knee_min_z_m':round(min(c['box'][2] for c in knees),3) if knees else None,
        'platform_panels':len(panels),'panel_seam_mm':[round(min(seams)*1000,3),round(max(seams)*1000,3)] if seams else [],
        'min_bearing_mm':round(min(bearings)*1000,3),'unanchored_walking_parts':floating,'edge_steps_over_4mm':deviations,'collision_probes':probes,
        'stair_landing_intrusions':landing_intrusions,'open_stair_treads':len(stairs),'stair_overlap_mm':[round(min(overlaps)*1000,3),round(max(overlaps)*1000,3)],'stair_clear_opening_mm':round(min(openings)*1000,3),
        'base_extension_full_width':bool(full_width),'base_extension_width_m':round(bx1-bx0,3) if full_width else 0,'ramp_symmetric':bool(symmetric),'ramp_base_extension_connected':bool(extension_ok),'bund_rectangular_walls':not wrong_walls,'local_ramp_runs_mm':[round(outer_run*1000,1),round(inner_run*1000,1)],'local_ramp_total_mm':round((outer_run+inner_run+crest_run)*1000,1),'pipe_holes_without_extra_frame':len(pipe_holes) if not frames else 0,'yellow_rail_parts':len(rails)-len(wrong_colors),'bund_matches_slab_boundary':not any(e['kind']=='bund_gap' for e in errors),'bund_boundary_m':[bx0,by0,bx1,by1],'drain_covers':drain_probe,'bund_min_height_mm':round(min(heights)*1000,3),'ramp_passage_clear':not blocked,'defects':b['defects'],
        'collision_parts':sum('walk_level' in c or c['kind'] in ('step','landing') for c in s['components']),
        'load_seconds':round(s['model_load_seconds'],3),'scope':'Check compiled convex contact vertices, ray-test the depression, and measure ramp clearance. Millimeter-scale imperfections are retained. Robot gait, toe clearance and structural loads require separate checks.'}
