"""Convex, disjoint pieces around real through holes; SI units throughout.

Each piece supplies both rendering and collision and contributes its own
volume/inertia. There is no enclosing convex hull across the opening.
"""
from itertools import combinations
import numpy as np
from scipy.spatial import ConvexHull
import xml.etree.ElementTree as ET


def clipped_vertices(vertices, extra):
    planes=np.unique(np.round(ConvexHull(vertices).equations,12),axis=0)
    planes=np.vstack([planes,extra]);out=[]
    for ids in combinations(range(len(planes)),3):
        a=planes[list(ids),:3];b=-planes[list(ids),3]
        if abs(np.linalg.det(a))<1e-9:continue
        v=np.linalg.solve(a,b)
        if np.max(planes[:,:3]@v+planes[:,3])<1e-9:out.append(v)
    if not out:return None
    out=np.unique(np.round(out,11),axis=0)
    if len(out)<4 or ConvexHull(out).volume<1e-13:return None
    return out


def perforate(vertices,center,u,v,boundary):
    # Boundary is CCW in the bore plane. Fan wedges partition its exterior.
    center,u,v=map(np.asarray,(center,u,v));pieces=[]
    for a,b in zip(boundary,np.roll(boundary,-1,axis=0)):
        # cross(a,p)>=0; cross(b,p)<=0; cross(b-a,p-a)<=0.
        delta=b-a
        constraints=[(np.array([a[1],-a[0]]),0.),
                     (np.array([-b[1],b[0]]),0.),
                     (np.array([-delta[1],delta[0]]),delta[1]*a[0]-delta[0]*a[1])]
        extra=[]
        for n,k in constraints:
            normal=n[0]*u+n[1]*v
            extra.append([*normal,k-normal@center])
        cell=clipped_vertices(vertices,extra)
        if cell is not None:pieces.append(cell)
    return pieces


def bore_boundary(kind):
    if kind=='round':
        a=np.linspace(0,2*np.pi,33)[:-1]
        return np.column_stack([.004*np.cos(a),.004*np.sin(a)])
    # Long dimension 16mm along v, flats 10mm apart along u.
    a=np.arcsin(5/8)
    long_short=np.array([(8*np.cos(t),8*np.sin(t)) for t in
       np.r_[np.linspace(-a,a,13),np.linspace(np.pi-a,np.pi+a,13)]])*.001
    return long_short[:,::-1][::-1]


def local_round_perforation(vertices, center, u, v, boundary):
    """Keep the flat plate outside the bore's bounding square in large cells.

    The former radial fan continued all the way to the plate edge, multiplying
    coplanar contact patches. Only the small square surrounding the same bore
    needs curved-wall subdivision. Cells have disjoint interiors; the polygonal
    hole, outside surface, volume and density are unchanged.
    """
    center, u, v = map(np.asarray, (center, u, v))
    lo = np.min(boundary, axis=0)
    hi = np.max(boundary, axis=0)
    def plane(axis, sign, bound):
        normal = sign * axis
        return [*normal, -sign * bound - normal @ center]
    # Left/right full strips; top/bottom restricted to the middle strip.
    middle = [plane(u, -1, lo[0]), plane(u, 1, hi[0])]
    regions = [[plane(u, 1, lo[0])], [plane(u, -1, hi[0])],
               middle + [plane(v, 1, lo[1])],
               middle + [plane(v, -1, hi[1])]]
    pieces = []
    for region in regions:
        cell = clipped_vertices(vertices, region)
        if cell is not None:
            pieces.append(cell)
    inner = clipped_vertices(vertices, middle +
                             [plane(v, -1, lo[1]), plane(v, 1, hi[1])])
    if inner is not None:
        pieces.extend(perforate(inner, center, u, v, boundary))
    return pieces


def add_cell(asset,body,name,vertices,density,material='s_hook_metal',collision=True):
    from .hook_tool import _s
    hull=ConvexHull(vertices)
    faces=hull.simplices.copy()
    for i,face in enumerate(faces):
        a,b,c=vertices[face]
        if np.dot(np.cross(b-a,c-a),hull.equations[i,:3])<0:faces[i]=face[::-1]
    ET.SubElement(asset,'mesh',name=name,vertex=_s(vertices.ravel()),
                  face=' '.join(str(int(i)) for i in faces.ravel()))
    ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,
                  **{'class':'collision','group':'0','density':str(density),
                     'material':material,'contype':'1' if collision else '0',
                     'conaffinity':'1' if collision else '0'})


def paint_plane(asset,body,name,vertices,normal,point,material):
    from .hook_tool import _s
    normal=np.asarray(normal);point=np.asarray(point)
    plane=vertices[abs((vertices-point)@normal)<1e-8]
    if len(plane)<3:return
    c=plane.mean(axis=0);u=plane[0]-c;u/=np.linalg.norm(u);v=np.cross(normal,u)
    plane=plane[np.argsort(np.arctan2((plane-c)@v,(plane-c)@u))]+normal*.000015
    faces=[[0,i,i+1] for i in range(1,len(plane)-1)]
    if len(plane)==3:
        # MuJoCo mesh assets require four vertices even for a thin shell.
        plane=np.vstack([plane,plane.mean(axis=0)])
        faces=[[0,1,3],[1,2,3],[2,0,3]]
    ET.SubElement(asset,'mesh',name=name,vertex=_s(plane.ravel()),inertia='shell',
                  face=' '.join(str(i) for f in faces for i in f))
    ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,material=material,
                  contype='0',conaffinity='0',density='0',group='0')


def add_holed_parts(asset,body,side,cfg,points,plus,minus,quads):
    from .hook_tool import _vertices,_paint_face
    holes=cfg['through_holes']
    if (holes['shaft_flat_mm'] != [16,10] or holes['upper_diameter_mm'] != 8
            or holes['shaft_center_behind_junction_mm'] != 14):
        raise ValueError('This approved tool construction requires 16x10, diameter8, and shaft offset14mm')
    width=cfg['out_of_plane_width_mm']*.001
    half=cfg['plate_thickness_mm']*.0005;j=points[1,1]
    # Trim the shaft against the two sloping faces: no overlapping mass.
    for sgn in (-1,1):
        q=np.array([[0,0],[sgn*half,0],[sgn*half,j-2*np.sqrt(2)*half+half],
                    [0,j-2*np.sqrt(2)*half]])
        verts=_vertices(q,width)
        for k,cell in enumerate(perforate(verts,[0,0,j-.014],[1,0,0],[0,0,1],bore_boundary('flat'))):
            add_cell(asset,body,f'{side}_hook_shaft_{sgn}_{k}',cell,cfg['density_kg_m3'])
    tangent=points[1]-points[0];tangent/=np.linalg.norm(tangent)
    normal=np.r_[0,-tangent[1],tangent[0]]
    center=np.r_[0,(points[0]+points[1])/2]
    for k,cell in enumerate(local_round_perforation(_vertices(quads[0],width),center,[1,0,0],np.r_[0,tangent],bore_boundary('round'))):
        name=f'{side}_hook_bend_0_{k}'
        # Only the approved outward working surface is orange. The back,
        # plate thickness and bore walls stay metal; the hole stays open.
        add_cell(asset,body,name,cell,cfg['density_kg_m3'])
        paint_plane(asset,body,name+'_orange',cell,-normal,np.r_[0,minus[0]],'s_hook_push')
    for i in (1,2):
        add_cell(asset,body,f'{side}_hook_bend_{i}',_vertices(quads[i],width),cfg['density_kg_m3'])
    for label,key in [('push','push_flanks'),('pull','pull_flanks')]:
        for i,edge in cfg[key]:
            if i==0:continue  # solid paint would cover the through bore
            curve=plus if edge=='left' else minus
            _paint_face(body,side,label,i,curve[i],curve[i+1],width,
                        coverage=1.)
