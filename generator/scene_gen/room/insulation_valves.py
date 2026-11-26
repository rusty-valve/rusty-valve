"""Fit insulation around installed valves and their connecting pipes."""
import math

import numpy as np

from . import saved_geometry as geometry
from .insulation_skin import EPS, HANDLE_GAP, THICKNESS
from .insulation_threaded_sample import shell_mesh

BODY_NAMES = ('visual_center_body', 'visual_valve_body', 'visual_welded_body', 'visual_left_end',
              'visual_right_end', 'visual_left_rf_end', 'visual_right_rf_end', 'collar_')


def valve_body(valve, model, data, cache):
    origin = np.asarray(valve['origin'])
    matrix = np.asarray(valve['matrix'])
    body = []
    for item in valve['local_geoms']:
        if not item['moving'] and item['name'].startswith(BODY_NAMES):
            gid = model.geom(valve['prefix']+item['name']).id
            vertices, _ = geometry.mesh(model, gid, cache)
            body.append((vertices @ data.geom_xmat[gid].reshape(3, 3).T+data.geom_xpos[gid]-origin)@matrix)
    if not body:
        raise ValueError('Missing valve body envelope for insulation: '+valve['id'])
    return origin, matrix, np.vstack(body)


def ends_at_flanges(valve, origin, matrix, stops):
    ends = sorted(float((np.asarray(p['position'])-origin)@matrix[:,0])
                  for p in valve['world_ports'].values())
    for stop in stops:
        if 'endpoint' in stop or abs(np.dot(stop['axis'], matrix[:,0])) < 1-1.e-6:
            continue
        datum = np.asarray(stop['datum'])
        if np.linalg.norm(np.cross(datum-origin,matrix[:,0])) > 1.e-6:
            continue
        a, b = sorted(float((datum+np.asarray(stop['axis'])*size-origin)@matrix[:,0])
                      for size in stop['bare_span_m'])
        if a <= ends[0]+1.e-6 and abs(b-ends[0]) < 1.e-6:
            ends[0] = a
        if b >= ends[1]-1.e-6 and abs(a-ends[1]) < 1.e-6:
            ends[1] = b
    return ends


def pole_limits(valve, body):
    parts = {g['name']:g['bbox'] for g in valve['local_geoms']}
    bonnet = parts.get('bonnet')
    if bonnet is not None:
        return float(bonnet[5]), max(abs(np.asarray(bonnet)[[0,1,3,4]]))+.004
    stem_sizes = [min(g['bbox'][3]-g['bbox'][0],g['bbox'][4]-g['bbox'][1])/2
                  for g in valve['local_geoms'] if g['name'] in ('visual_stem','stem','stem_shaft')]
    return float(body[:,2].max()), max(stem_sizes,default=.020)+.005


def wheel_shell(ends, lower, upper, pipe_radius, shaft, recess):
    """Square body, two annular end plates and a lined stem opening."""
    x0,x1 = ends
    yl,zl = lower
    yh,zh = upper
    corners = ((yl,zl),(yh,zl),(yh,zh),(yl,zh))
    outer_xy = ((x0,yl),(x1,yl),(x1,yh),(x0,yh))
    hole_xy = ((-shaft,-shaft),(shaft,-shaft),(shaft,shaft),(-shaft,shaft))
    vertices, faces = [], []

    def quad(points):
        start = len(vertices)
        vertices.extend(points)
        faces.extend(((start,start+1,start+2),(start,start+2,start+3)))

    for i in (0,1,3):
        j=(i+1)%4
        quad([(x0,*corners[i]),(x1,*corners[i]),(x1,*corners[j]),(x0,*corners[j])])
    for i in range(4):
        j=(i+1)%4
        quad([(*outer_xy[i],zh),(*outer_xy[j],zh),(*hole_xy[j],zh),(*hole_xy[i],zh)])
        quad([(*hole_xy[i],zh),(*hole_xy[j],zh),(*hole_xy[j],recess),(*hole_xy[i],recess)])
    angles = np.unique(np.r_[np.linspace(-math.pi,math.pi,49),
                             [math.atan2(z,y) for y,z in corners]])
    for x in (x0,x1):
        for a,b in zip(angles,angles[1:]):
            corners_on_rim, hole = [], []
            for angle in (a,b):
                dy,dz = math.cos(angle),math.sin(angle)
                dy_bound = (yh if dy>0 else yl)/dy if abs(dy)>EPS else math.inf
                dz_bound = (zh if dz>0 else zl)/dz if abs(dz)>EPS else math.inf
                stretch = min(dy_bound,dz_bound)
                corners_on_rim.append((x,dy*stretch,dz*stretch))
                hole.append((x,dy*pipe_radius,dz*pipe_radius))
            quad([*corners_on_rim,hole[1],hole[0]])
    return np.asarray(vertices),np.asarray(faces)


def recessed_cylinder(ends, outer_radius, inner_radius, pipe_radius, cap, shaft):
    """Cylindrical wall and end seals with a recessed, lined operator opening."""
    x0, x1 = ends
    arc = [math.acos(min(cap/r, 1.)) for r in (outer_radius, inner_radius)]
    angles = np.unique(np.r_[np.linspace(0., math.tau, 97), arc, math.tau-np.asarray(arc)])
    rays = np.c_[np.sin(angles), np.cos(angles)]

    def section(radius):
        reaches = np.full(len(angles), radius)
        top = rays[:,1] > 0.
        reaches[top] = np.minimum(radius, cap/rays[top,1])
        return rays*reaches[:,None]

    outer, inner, pipe = section(outer_radius), section(inner_radius), section(pipe_radius)
    vertices, faces = [], []

    def quad(points):
        offset = len(vertices)
        vertices.extend(points)
        faces.extend(((offset, offset+1, offset+2), (offset, offset+2, offset+3)))

    for i in range(len(angles)-1):
        for ring in (outer, inner):
            a, b = ring[i:i+2]
            if not (abs(a[1]-cap)<EPS and abs(b[1]-cap)<EPS):
                quad([(x0,*a),(x1,*a),(x1,*b),(x0,*b)])
        for x in (x0, x1):
            quad([(x,*outer[i]),(x,*outer[i+1]),(x,*pipe[i+1]),(x,*pipe[i])])

    width = math.sqrt(outer_radius**2-cap**2)
    top = [(x0,-width),(x1,-width),(x1,width),(x0,width)]
    hole = [(-shaft,-shaft),(shaft,-shaft),(shaft,shaft),(-shaft,shaft)]
    for i in range(4):
        j = (i+1)%4
        quad([(*top[i],cap),(*top[j],cap),(*hole[j],cap),(*hole[i],cap)])
        quad([(*hole[i],cap),(*hole[j],cap),(*hole[j],cap-.001),(*hole[i],cap-.001)])
    return np.asarray(vertices), np.asarray(faces)


def near_valve_clearance(sweeps, uid, origin, matrix, ends, body, radius, lever, shaft, top):
    closest = math.inf
    bare_lo, bare_hi = body.min(axis=0),body.max(axis=0)
    for other, motion in sweeps.items():
        if np.any(motion['bounds'][:3] > origin+radius+.35) or np.any(motion['bounds'][3:] < origin-radius-.35):
            continue
        points = (motion['points']-origin)@matrix
        if lever:
            axial=np.maximum.reduce((ends[0]-points[:,0], points[:,0]-ends[1], np.zeros(len(points))))
            radial=np.linalg.norm(points[:,1:],axis=1)-radius
            gaps=np.hypot(np.maximum(radial,0.),axial)
            # The valve's own stem and handle use the lined top opening.
            if other==uid:
                allowed=(abs(points[:,0]-np.mean(ends)) < .19*(ends[1]-ends[0])) & (
                         abs(points[:,1]) < 1.65*shaft) & (points[:,2]>top-.005)
                gaps=gaps[(~allowed) & (radial > 0.)]
        elif other != uid:
            outside=np.maximum(0.,np.maximum(bare_lo-points,points-bare_hi))
            gaps=np.linalg.norm(outside,axis=1)
        else:
            continue
        if len(gaps):
            closest=min(closest,float(gaps.min()))
    return closest


def build_valve_shell(valve, model, data, cache, sweeps, flange_records, pipe_radius):
    origin, matrix, body = valve_body(valve,model,data,cache)
    lever=valve['operator']=='lever'
    threaded='npt' in valve['package'].lower()
    original_ends=ends_at_flanges(valve,origin,matrix,())
    ends=original_ends if threaded or not lever else ends_at_flanges(valve,origin,matrix,flange_records)
    pipe_radius+=.0004
    top,shaft=pole_limits(valve,body)
    inner=max(pipe_radius+.004,float(np.linalg.norm(body[:,1:],axis=1).max())+.002)
    gap=near_valve_clearance(sweeps,valve['id'],origin,matrix,ends,body,inner,lever,shaft,top)
    # A bare envelope already touched by an operator is recorded by the
    # saved-mesh travel check. It creates no fictitious positive clearance.
    thickness=min(THICKNESS,max(0.,gap-HANDLE_GAP))
    if not math.isfinite(gap):
        thickness=THICKNESS
    recessed = False
    if thickness<=EPS and lever:
        other_sweeps = {key: value for key, value in sweeps.items() if key!=valve['id']}
        gap = near_valve_clearance(other_sweeps,valve['id'],origin,matrix,ends,body,inner,lever,shaft,top)
        thickness=min(THICKNESS,max(0.,gap-HANDLE_GAP))
        recessed = True
    if thickness<=EPS:
        return None
    if lever:
        profile='threaded_lever_cylinder' if threaded else 'flanged_lever_cylinder'
        length=ends[1]-ends[0]
        opening_axial=min(length-.004,max(length*.38,shaft*2+.010))
        opening_width=min(inner*1.7,max(inner*.95,shaft*3.3))
        if recessed:
            operating = [item['bbox'][2] for item in valve['local_geoms'] if item['moving'] and
                         item['name'] in ('visual_lever','visual_bypassed_latch','visual_grip')]
            cap=min(top,min(operating,default=top+HANDLE_GAP)-HANDLE_GAP)
            if cap<=pipe_radius+EPS:
                return None
            v,f=recessed_cylinder(ends,inner+thickness,inner,pipe_radius,cap,shaft)
        else:
            v,f,_,_=shell_mesh(ends,(pipe_radius,pipe_radius),top,
                                outer_radius=inner+thickness,inner_radius=inner,opening_axial=opening_axial,
                                opening_width=opening_width,shaft_radius=shaft)
        envelope={'bare_radius_m':inner, 'recessed_top_lining':recessed}
    else:
        profile='handwheel_box'
        lo=body[:,1:].min(axis=0)
        hi=body[:,1:].max(axis=0)
        # Keep every side of the fixed reduction gear outside the casing.
        gearbox=[g['bbox'][2] for g in valve['local_geoms']
                 if not g['moving'] and g['name'] in ('gearbox','visual_gearbox','visual_gear_housing')]
        cap=min(top+thickness,min(gearbox,default=math.inf)-.002)
        lo=np.minimum(lo,[-pipe_radius-.002,-pipe_radius-.002])
        hi=np.maximum(hi,[pipe_radius+.002,pipe_radius+.002])
        low=lo-thickness
        high=(hi[0]+thickness,max(cap,pipe_radius+.002))
        shaft=min(shaft,.85*min(ends[1],-ends[0],high[0],-low[0]))
        v,f=wheel_shell(ends,low,high,pipe_radius,shaft,max(top,high[1]-.017))
        envelope={'bare_side_y_m':[float(lo[0]),float(hi[0])], 'bare_bottom_z_m':float(lo[1]),
                  'cap_z_m':float(high[1])}
    frame={'origin':origin.tolist(),'matrix':matrix.tolist(),'ends_m':ends,**envelope}
    return v@matrix.T+origin,f,thickness,profile,frame
