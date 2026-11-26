"""Build the F10000 reactor shell and its support lugs."""
import math

import numpy as np

from .layout import add_pipe


SECTION = 'Taiji F10000 to F20000 manufacturer section drawing'
PLAN = 'Taiji F10000 to F20000 manufacturer plan drawing'
CATALOG = 'Taiji F8000 to F20000 catalogue table, F10000 column'
STEEL = (.64, .71, .74)
JACKET = (.35, .52, .58)
SUPPORT = (.32, .41, .45)

DIMENSIONS = {
    'inner_diameter_m': 2.200,
    'jacket_outer_diameter_m': 2.400,
    'lug_pitch_circle_m': 2.964,
    'lug_hole_diameter_m': .036,
    'high_neck_diameter_m': .800,
    'lower_reference_z_m': -2.233,
    'upper_reference_z_m': 1.300,
    'upper_jacket_port_z_m': .400,
    'drive_mount_z_m': 1.510,
    'drive_mount_dn': 200,
    'shaft_diameter_m': .110,
    'drive_reference_height_m': 1.960,
    'catalog_reference_overall_m': 5.890,
    'bottom_jacket_nozzle_offset_m': .470,
}

# Trace of the outer outline. The catalogue gives d1 as an inside diameter
# and does not give the exterior process-shell thickness or head profiles.
SHELL_RADIUS = 1.118
UPPER_TANGENT_Z = .620
CROWN_Z = 1.390
JACKET_LOWER_TANGENT_Z = -1.670
JACKET_LOWER_CROWN_Z = -2.290


def add_mesh(scene, folder, uid, kind, vertices, faces, color, source, status, note, owner):
    world = np.asarray(vertices)
    origin = (world.min(axis=0)+world.max(axis=0))/2
    filename = folder/(uid+'.obj')
    filename.write_text(''.join('v %.12g %.12g %.12g\n' % tuple(v) for v in world-origin)
                        +''.join('f %d %d %d\n' % tuple(i+1 for i in face) for face in faces))
    part = {'id':uid,'owner':owner,'kind':kind,'shape':'mesh','mesh_path':str(filename),
            'pos':origin.tolist(),'box':np.r_[world.min(axis=0),world.max(axis=0)].tolist(),
            'color':list(color),'collision_mesh':True,'source':source,'evidence_status':status,'note':note}
    scene['components'].append(part)
    return part


def _prism(scene, folder, uid, kind, lower, lift, color, source, status, note, owner):
    lower = np.asarray(lower)
    if float(np.cross(lower[1]-lower[0],lower[2]-lower[0])@lift)<0:
        lower=lower[::-1]
    n = len(lower)
    faces = [face for i in range(1,n-1) for face in ((0,i+1,i),(n,n+i,n+i+1))]
    for i in range(n):
        j = (i+1)%n
        faces.extend(((i,j,n+j),(i,n+j,n+i)))
    return add_mesh(scene,folder,uid,kind,np.vstack((lower,lower+lift)),faces,color,source,status,note,owner)


def _revolve(scene, folder, owner, suffix, profile, kind, color, source, note):
    sides=96
    angles=np.arange(sides)*math.tau/sides
    vertices=[[r*math.cos(a),r*math.sin(a),z] for z,r in profile for a in angles]
    faces=[]
    last=(len(profile)-1)*sides
    for i in range(1,sides-1):
        faces.extend(((0,i+1,i),(last,last+i,last+i+1)))
    for band in range(len(profile)-1):
        low=band*sides;high=low+sides
        for i in range(sides):
            j=(i+1)%sides
            faces.extend(((low+i,low+j,high+j),(low+i,high+j,high+i)))
    return add_mesh(scene,folder,owner+'_'+suffix,kind,vertices,faces,color,source,'estimated from the drawing',note,owner)


def _dish(tangent_z, crown_z, radius):
    # Keep a small tip ring so the exported closed mesh has no coincident vertices.
    result=[(tangent_z+(crown_z-tangent_z)*math.sin(a),radius*math.cos(a))
            for a in np.linspace(0.,math.radians(89),18)]
    result.append((crown_z,.002))
    return sorted(result)


def _lug(scene, folder, uid, bearing):
    angle=math.radians(bearing)
    outward=np.array([math.sin(angle),math.cos(angle),0.])
    tangent=np.array([-outward[1],outward[0],0.]);up=np.array([0.,0.,1.])
    base=DIMENSIONS['lug_pitch_circle_m']/2
    hole=DIMENSIONS['lug_hole_diameter_m']/2
    # Plate outline and thickness by the visible plate on plans 17/19.
    min_u,max_u=1.198-base,1.632-base
    half_v=.155
    angles=sorted(set(np.arange(24)*math.tau/24) |
                  {math.atan2(y,x)%math.tau for x in (min_u,max_u) for y in (-half_v,half_v)})
    angles += [angles[0]+math.tau]
    radius=hole/math.cos(max(b-a for a,b in zip(angles,angles[1:]))/2)
    name=f'{uid}_lug_{bearing:03}'

    def point(x,y,z=0.):
        return (base+x)*outward+y*tangent+z*up

    def ring(a):
        c,s=math.cos(a),math.sin(a)
        limits=[(max_u if c>0 else min_u)/c] if abs(c)>1e-10 else []
        if abs(s)>1e-10:limits.append(half_v/abs(s))
        extent=min(limits)
        return point(radius*c,radius*s),point(extent*c,extent*s)

    for i,(a,b) in enumerate(zip(angles,angles[1:])):
        inner_a,outer_a=ring(a);inner_b,outer_b=ring(b)
        part=_prism(scene,folder,f'{name}_pad_{i:02}','support_pad',
            [inner_a,outer_a,outer_b,inner_b],.024*up,SUPPORT,PLAN+'；'+SECTION+'；'+CATALOG,
            'drawing dimensions and estimates','Bearing surface at z=0. Hole centers and diameters use the table. Shoe outline and thickness follow the drawing.',uid)
        part.update(bearing_deg=bearing,hole_center_m=point(0,0).tolist(),hole_diameter_m=2*hole)
    for i,y in enumerate((-.114,.084)):
        _prism(scene,folder,f'{name}_web_{i+1}','support_web',
            [point(1.190-base,y,.024),point(1.610-base,y,.024),
             point(1.190-base,y,.416)],.027*tangent,SUPPORT,SECTION+'；'+PLAN,
            'estimated from the drawing','Triangular lug web. Thickness and rear details are estimated from the drawing.',uid)


def build_shell(scene, folder, uid):
    """Build only the jacket, exposed head, high neck and support ears."""
    _revolve(scene,folder,uid,'head',_dish(UPPER_TANGENT_Z,CROWN_Z,SHELL_RADIUS),'vessel_head',STEEL,
             SECTION,'Top head shell is fitted as an ellipse. ID follows catalogue d1.')
    body=add_pipe(scene,uid+'_jacket',[0.,0.,JACKET_LOWER_TANGENT_Z],[0.,0.,.520],
                  DIMENSIONS['jacket_outer_diameter_m']/2,'jacket_shell',JACKET,uid)
    body.update(source=CATALOG+'；'+SECTION,evidence_status='drawing dimensions and estimates',
                note='Jacket OD is 2.400 m. Tangent elevations are scaled from the section.')
    _revolve(scene,folder,uid,'jacket_shoulder',[(.520,1.200),(UPPER_TANGENT_Z,SHELL_RADIUS)],
             'jacket_shoulder',JACKET,SECTION,'Full-jacket upper shoulder elevation is scaled from the drawing.')
    _revolve(scene,folder,uid,'jacket_base',_dish(JACKET_LOWER_TANGENT_Z,JACKET_LOWER_CROWN_Z,1.200),
             'jacket_dish',JACKET,SECTION,'Jacket bottom head envelope follows the drawing.')
    neck=add_pipe(scene,uid+'_high_neck',[0.,0.,1.318],[0.,0.,DIMENSIONS['drive_mount_z_m']],
                   DIMENSIONS['high_neck_diameter_m']/2,'mount_neck',STEEL,uid)
    neck.update(source=CATALOG+'；'+SECTION,evidence_status='drawing dimensions and estimates',
                note='Tall neck diameter uses d0. The top mates to the d agitator mount. Root shape follows the drawing.')
    for bearing in (45,135,225,315):
        _lug(scene,folder,uid,bearing)
