"""Pure geometry/JSON S-face identification, matching chest_contact_evidence.

Distances and normal tolerances identify a physical face; they are not wrist,
force, speed, motion or face-share acceptance thresholds. The optional v2.5
policy also identifies the working-side lip of a real through hole. Other
ambiguous edges stay unknown. No physics or paint/name inference is performed.
"""
from __future__ import annotations
from functools import lru_cache
import json
import math
from pathlib import Path


def vec(value):
    if not isinstance(value,(list,tuple)) or len(value)!=3:raise ValueError('Finite 3-vector required')
    result=tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in result):raise ValueError('Finite 3-vector required')
    return result
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def scale(a,k):return tuple(x*k for x in a)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def norm(a):return math.sqrt(dot(a,a))
def unit(a):
    length=norm(a)
    if length<=0:raise ValueError('Nonzero direction required')
    return scale(a,1/length)


@lru_cache(maxsize=1)
def geometry():
    # Same miter construction and source dimensions as hook_tool.tool_geometry.
    cfg=json.loads((Path(__file__).parent/'configs/s_hook.json').read_text())
    points=[(0.,float(y),float(z)) for y,z in cfg['profile_yz_mm']]
    tangents=[unit(sub(b,a)) for a,b in zip(points,points[1:])]
    normals=[(0.,-t[2],t[1]) for t in tangents];offsets=[];half=cfg['plate_thickness_mm']/2
    for i in range(len(points)):
        normal=normals[0] if i==0 else normals[-1] if i==len(points)-1 else unit(add(normals[i-1],normals[i]))
        offsets.append(scale(normal,half if i in (0,len(points)-1) else half/dot(normal,normals[i])))
    shift=cfg['total_forward_extent_mm']-max(p[2]+abs(o[2]) for p,o in zip(points,offsets))
    points=[(0.,p[1],p[2]+shift) for p in points]
    plus=[scale(add(p,o),.001) for p,o in zip(points,offsets)]
    minus=[scale(sub(p,o),.001) for p,o in zip(points,offsets)]
    working={(int(i),side):face for key,face in [('push_flanks','orange_push'),('pull_flanks','blue_pull')] for i,side in cfg[key]}
    planes=[]
    for side,curve in [('left',plus),('right',minus)]:
        for i,(a,b) in enumerate(zip(curve,curve[1:])):
            tangent=unit(sub(b,a));normal=(0.,-tangent[2],tangent[1])
            if side=='right':normal=scale(normal,-1)
            planes.append(dict(a=a,b=b,tangent=tangent,normal=normal,length=norm(sub(b,a)),
                segment=i,side=side,face=working.get((i,side),'nonworking')))
    quads=[[plus[i],plus[i+1],minus[i+1],minus[i]] for i in range(len(plus)-1)]
    hole_rims=[]
    holes=cfg.get('through_holes',{})
    if holes.get('upper_diameter_mm'):
        # Same upper bore as holed_tool.add_holed_parts: a 32-sided opening
        # normal to profile segment 0, centred between its centreline ends.
        centre=scale(add(points[0],points[1]),.0005)
        radius=float(holes['upper_diameter_mm'])*.0005
        for plane in planes:
            if plane['segment']==0 and plane['face']!='nonworking':
                hole_rims.append(dict(face=plane['face'],segment=0,side=plane['side'],
                    centre=add(centre,scale(plane['normal'],half*.001)),
                    normal=plane['normal'],u=(1.,0.,0.),v=plane['tangent'],radius=radius,
                    facets=32))
    return dict(version=cfg['version'],planes=planes,quads=quads,hole_rims=hole_rims,
        half_width=cfg['out_of_plane_width_mm']*.0005,half_thickness=cfg['plate_thickness_mm']*.0005,
        shaft_end=points[cfg['shaft_junction_index']][2]*.001,
        plane_tolerance_m=.0015,edge_margin_m=.0003,normal_cos=math.cos(math.radians(30.)))


def _inside_yz(point,polygon):
    signs=[]
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        value=(b[1]-a[1])*(point[2]-a[2])-(b[2]-a[2])*(point[1]-a[1])
        if abs(value)>1e-14:signs.append(value>0)
    return not signs or all(signs) or not any(signs)


def _working_hole_rim_matches(point,normal,g):
    """Identify the approved working-side mouth, not the bore interior/back.

    Reuse the existing physical edge-identification band, without a new score
    threshold or a flat-face angular requirement for an actual sharp lip.
    The sharp lip can have a native edge normal rather than a flat-face
    normal. Its position must be on the real polygonal hole boundary as well
    as at the working-side mouth; its normal must point toward the working
    side and into the opening. Unknown contacts elsewhere are unaffected.
    """
    matches=[]
    for rim in g['hole_rims']:
        delta=sub(point,rim['centre'])
        plane_distance=dot(delta,rim['normal'])
        if abs(plane_distance)>g['edge_margin_m']+1e-9:continue
        xy=(dot(delta,rim['u']),dot(delta,rim['v']))
        distance=math.inf;wall=None
        count=rim['facets'];radius=rim['radius']
        radial=math.hypot(*xy)
        if not (radius*math.cos(math.pi/count)-g['edge_margin_m']-1e-9 <= radial
                <= radius+g['edge_margin_m']+1e-9):continue
        for index in range(count):
            a=(radius*math.cos(2*math.pi*index/count),radius*math.sin(2*math.pi*index/count))
            b=(radius*math.cos(2*math.pi*(index+1)/count),radius*math.sin(2*math.pi*(index+1)/count))
            edge=(b[0]-a[0],b[1]-a[1])
            fraction=max(0.,min(1.,((xy[0]-a[0])*edge[0]+(xy[1]-a[1])*edge[1])/
                (edge[0]*edge[0]+edge[1]*edge[1])))
            closest=(a[0]+fraction*edge[0],a[1]+fraction*edge[1])
            gap=math.hypot(xy[0]-closest[0],xy[1]-closest[1])
            if gap<distance:
                distance=gap
                # CCW hole boundary: the exposed wall points into the hole.
                wall=unit(add(scale(rim['u'],-edge[1]),scale(rim['v'],edge[0])))
        if distance>g['edge_margin_m']+1e-9:continue
        outer_component=dot(normal,rim['normal']);wall_component=dot(normal,wall)
        if outer_component < -1e-9 or wall_component < -1e-9:continue
        if max(outer_component,wall_component)<=1e-12:continue
        matches.append(dict(face=rim['face'],region='working_hole_rim',
            segment=rim['segment'],side=rim['side'],
            distance_to_hole_boundary_m=distance,working_mouth_plane_distance_m=plane_distance,
            membership_basis='physical working-side mouth, polygonal bore boundary and outward working/bore normal'))
    return matches


def classify_s_face(local_point,local_outward_normal,*,allow_working_hole_rims=False):
    p=vec(local_point);n=unit(vec(local_outward_normal));g=geometry();matches=[]
    if abs(p[0])>g['half_width']+1e-9:return dict(face='unknown',reason='outside_tool_extrusion')
    for plane in g['planes']:
        d=sub(p,plane['a']);along=dot(d,plane['tangent']);distance=dot(d,plane['normal'])
        alignment=dot(n,plane['normal'])
        if alignment<g['normal_cos']:continue
        if (g['edge_margin_m']<along<plane['length']-g['edge_margin_m']
                and abs(distance)<=g['plane_tolerance_m']):
            matches.append(dict(face=plane['face'],region='flat_face',segment=plane['segment'],side=plane['side']))
        elif plane['face']!='nonworking':
            # The existing approved physical edge band remains allowed. A
            # normal between incompatible faces is unknown, never a wrong face.
            if min(norm((0.,p[1]-end[1],p[2]-end[2])) for end in (plane['a'],plane['b']))<=g['edge_margin_m']+1e-9:
                matches.append(dict(face=plane['face'],region='working_edge',segment=plane['segment'],side=plane['side']))
    # Explicit broad metal extrusion sides and shaft outer surfaces provide
    # positive nonworking evidence. Failure to match a colour alone does not.
    if abs(abs(p[0])-g['half_width'])<=g['plane_tolerance_m'] and abs(n[0])>=g['normal_cos']:
        if any(_inside_yz(p,q) for q in g['quads']) or (abs(p[1])<=g['half_thickness'] and 0.<=p[2]<=g['shaft_end']):
            matches.append(dict(face='nonworking',region='extrusion_side'))
    for sign in (-1.,1.):
        if (abs(p[1]-sign*g['half_thickness'])<=g['plane_tolerance_m'] and 0.<p[2]<g['shaft_end']
                and sign*n[1]>=g['normal_cos']):
            matches.append(dict(face='nonworking',region='shaft_outer'))
    if allow_working_hole_rims:
        matches=_working_hole_rim_matches(p,n,g)+matches
    faces={m['face'] for m in matches}
    if len(faces)==1:return dict(matches[0],geometry_version=g['version'])
    return dict(face='unknown',reason='ambiguous_physical_faces' if faces else 'unidentified_edge_or_normal')


def assess_contact_drive(evidence,*,allow_working_hole_rims=False):
    if not isinstance(evidence,dict) or evidence.get('basis')!='solved_contact_point_normal_wrench_v1':
        return dict(evidence_status='unknown',face='unknown',driving=None,reason='missing_solved_contact_geometry_and_direction')
    try:
        label=classify_s_face(evidence['tool_surface_position_local_m'],evidence['tool_outward_normal_local'],
            allow_working_hole_rims=allow_working_hole_rims)
        lever=sub(vec(evidence['contact_position_world_m']),vec(evidence['valve_axis_point_world_m']))
        force=vec(evidence['force_on_handle_world_N']);moment=vec(evidence['moment_on_handle_world_Nm'])
        axis=unit(vec(evidence['requested_axis_world']));omega=float(evidence['angular_speed_about_requested_axis_rad_s'])
        if not math.isfinite(omega):raise ValueError('Finite speed required')
        torque=dot(add(cross(lever,force),moment),axis);power=torque*omega
    except (ValueError,TypeError,KeyError,OverflowError):
        return dict(evidence_status='unknown',face='unknown',driving=None,reason='incomplete_or_invalid_contact_vectors')
    return dict(evidence_status='observed',face=label['face'],geometry=label,
        requested_torque_Nm=torque,angular_speed_about_requested_axis_rad_s=omega,power_W=power,
        driving=power>0.,driving_toward_target=power>0. and torque>0.)
