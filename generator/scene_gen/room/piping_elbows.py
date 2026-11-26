"""Build elbows that match the pipe size and bend radius."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .recipe import CandidateRefused

EPS=1.e-8
SECTOR_DEGREES=15.  # Geometry tessellation only, not an angle exception.
RING_VERTICES=24
BASIS={'source':'scene_gen/room/installation_checks.py:54',
       'centerline_radius_policy':'3 * actual pipe outer radius = 1.5 * actual OD',
       'scope':'Approximate project elbow envelope'}


class ElbowUnavailable(CandidateRefused):
    def __init__(self,code,detail,**evidence):
        self.code=code;self.evidence={'code':code,'detail':detail,**evidence}
        super().__init__(code+': '+detail)


def _unit(vector):
    vector=np.asarray(vector,dtype=float);length=float(np.linalg.norm(vector))
    if vector.shape!=(3,) or not np.isfinite(vector).all() or length<=EPS:
        raise ElbowUnavailable('ELBOW_INVALID_INPUT','Direction must be a finite nonzero 3D vector')
    return vector/length


def is_axis_aligned(first,second):
    delta=np.asarray(second,float)-np.asarray(first,float)
    return int(np.count_nonzero(abs(delta)>EPS))<=1


def coarse_radius(pipe_radius):
    if not np.isfinite(pipe_radius) or pipe_radius<=0:
        raise ElbowUnavailable('ELBOW_INVALID_INPUT','Actual outside pipe radius must be finite and positive')
    return 3.*float(pipe_radius)


def _clean(points):
    clean=[]
    for index,point in enumerate(points):
        p=np.asarray(point,float)
        if p.shape!=(3,) or not np.isfinite(p).all():
            raise ElbowUnavailable('ELBOW_INVALID_INPUT','Route points must be finite 3D points')
        if clean and np.max(abs(clean[-1][0]-p))<=EPS:continue
        while len(clean)>=2:
            a=_unit(clean[-1][0]-clean[-2][0]);b=_unit(p-clean[-1][0])
            if (is_axis_aligned(clean[-2][0],clean[-1][0]) and is_axis_aligned(clean[-1][0],p)
                and is_axis_aligned(clean[-2][0],p) and float(a@b)>0):clean.pop()
            else:break
        clean.append((p,index))
    return clean


def plan_polyline(points,pipe_radius,**unsupported):
    """Trim axis-only runs at real 90-degree elbow tangent planes or refuse."""
    if unsupported:
        raise ElbowUnavailable('PG17_EXCEPTIONS_DISABLED','Direction exemptions are unsupported',fields=sorted(unsupported))
    R=coarse_radius(pipe_radius);r=float(pipe_radius)
    clean=_clean(points);pts=[p for p,_ in clean];original=[i for _,i in clean]
    if len(pts)<2:raise ElbowUnavailable('ELBOW_EMPTY_ROUTE','A route needs two distinct endpoints')
    directions=[_unit(b-a) for a,b in zip(pts,pts[1:])]
    for i,(a,b) in enumerate(zip(pts,pts[1:])):
        if not is_axis_aligned(a,b):
            raise ElbowUnavailable('PG17_NON_ORTHOGONAL_ROUTE','Every straight follows a building axis',segment=i,ends=[a.tolist(),b.tolist()])
    corners={}
    for i in range(1,len(pts)-1):
        u,v=directions[i-1],directions[i];dot=float(np.clip(u@v,-1.,1.))
        if abs(dot)<=EPS:angle=90.
        else:raise ElbowUnavailable('PG17_UNSUPPORTED_TURN','Only 90-degree turns are supported',corner=original[i],cosine=dot)
        theta=math.radians(angle);trim=R*math.tan(theta/2)
        inward=_unit(v-u*dot);normal=_unit(np.cross(u,v))
        start=pts[i]-u*trim;end=pts[i]+v*trim;center=start+inward*R
        count=int(round(angle/SECTOR_DEGREES))
        samples=[center+R*(-inward*math.cos(t)+u*math.sin(t)) for t in np.linspace(0.,theta,count+1)]
        corners[i]={'corner_index':original[i],'corner':pts[i].tolist(),'center':center.tolist(),
            'start':start.tolist(),'end':end.tolist(),'tangent_in':u.tolist(),'tangent_out':v.tolist(),
            'radial_start':(-inward).tolist(),'plane_normal':normal.tolist(),
            'centerline_radius_m':R,'pipe_outer_radius_m':r,'angle_degrees':angle,
            'trim_m':trim,'centerline_points':[p.tolist() for p in samples],
            'basis':dict(BASIS),'coarse_geometry':'Closed convex 15-degree sectors with 24-point outer end rims. Interior is omitted.'}
    runs=[];elbows=[];ordered=[]
    for i,(a,b) in enumerate(zip(pts,pts[1:])):
        left=corners[i]['trim_m'] if i in corners else 0.
        right=corners[i+1]['trim_m'] if i+1 in corners else 0.
        length=math.dist(a,b)
        if length+EPS<left+right:
            raise ElbowUnavailable('PG17_DENSE_TURNS','Adjacent straights cannot fit the physical elbow with R equal to 3r',
                segment=i,available_m=length,required_m=left+right,pipe_radius_m=r,centerline_radius_m=R)
        start=a+directions[i]*left;end=b-directions[i]*right
        if math.dist(start,end)>EPS:
            runs.append({'start':start.tolist(),'end':end.tolist(),'segment_index':i,
                         'original_segment_start':original[i]})
            ordered.append({'kind':'run','index':len(runs)-1})
        if i+1 in corners:
            elbows.append(corners[i+1]);ordered.append({'kind':'elbow','index':len(elbows)-1})
    return {'points':[p.tolist() for p in pts],'runs':runs,'elbows':elbows,'ordered':ordered,
        'original_indices':original,'pipe_outer_radius_m':r,'centerline_radius_m':R,'basis':dict(BASIS)}


def emit_elbow(name,plan,folder,color,dn,route_id):
    """Return physical component dicts plus a checkable elbow record."""
    if abs(plan['angle_degrees']-90.)>EPS:
        raise ElbowUnavailable('PG17_NON_90_ELBOW','Generate physical 90-degree elbows only')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    # Snapshots can rewind geometry IDs. Each emission owns its actual files,
    # so a later rejected candidate cannot overwrite the selected elbow.
    import tempfile
    folder=Path(tempfile.mkdtemp(prefix=name+'_',dir=folder))
    center=np.asarray(plan['center']);radial=np.asarray(plan['radial_start'])
    u=np.asarray(plan['tangent_in']);normal=np.asarray(plan['plane_normal'])
    R=plan['centerline_radius_m'];r=plan['pipe_outer_radius_m'];angle=math.radians(plan['angle_degrees'])
    count=int(round(plan['angle_degrees']/SECTOR_DEGREES));components=[];pieces=[]
    for index in range(count):
        t0,t1=angle*index/count,angle*(index+1)/count
        rings=[];ring_centers=[];tangents=[]
        for theta in (t0,t1):
            n=radial*math.cos(theta)+u*math.sin(theta)
            tangent=-radial*math.sin(theta)+u*math.cos(theta)
            p=center+R*n;ring_centers.append(p);tangents.append(tangent)
            rings.extend(p+r*(math.cos(phi)*n+math.sin(phi)*normal)
                for phi in np.linspace(0.,2*math.pi,RING_VERTICES,endpoint=False))
        world=np.asarray(rings);origin=(ring_centers[0]+ring_centers[1])/2;vertices=world-origin
        faces=[];N=RING_VERTICES
        for j in range(N):
            k=(j+1)%N;faces.extend(((j,N+k,k),(j,N+j,N+k)))
        for j in range(1,N-1):faces.extend(((0,j,j+1),(N,N+j+1,N+j)))
        identifier=name+f'_sector_{index:02}';path=folder/(identifier+'.obj')
        path.write_text('\n'.join('v '+' '.join(f'{x:.17g}' for x in row) for row in vertices)+'\n'+
            '\n'.join('f '+' '.join(str(j+1) for j in row) for row in faces)+'\n')
        ports={'in':{'position':ring_centers[0].tolist(),'tangent':tangents[0].tolist(),'normal':(-tangents[0]).tolist()},
               'out':{'position':ring_centers[1].tolist(),'tangent':tangents[1].tolist(),'normal':tangents[1].tolist()}}
        component={'id':identifier,'kind':'piping_elbow','shape':'mesh','mesh_path':str(path.resolve()),
            'collision_mesh':True,'pos':origin.tolist(),'box':np.r_[world.min(axis=0),world.max(axis=0)].tolist(),
            'color':list(color),'owner':route_id,'route_id':route_id,'piping':True,'dn':int(dn),
            'elbow_id':name,'elbow_piece_index':index,'pipe_outer_radius_m':r,
            'centerline_radius_m':R,'angle_degrees':math.degrees(t1-t0),
            'ports':ports,'coarse_estimate':plan['coarse_geometry'],'basis':dict(BASIS)}
        components.append(component);pieces.append({'id':identifier,'ports':ports,
            'angle_range_degrees':[math.degrees(t0),math.degrees(t1)]})
    ports={'in':{'position':plan['start'],'tangent':plan['tangent_in'],'normal':(-np.asarray(plan['tangent_in'])).tolist()},
           'out':{'position':plan['end'],'tangent':plan['tangent_out'],'normal':plan['tangent_out']}}
    record={k:v for k,v in plan.items() if k not in ('start','end','tangent_in','tangent_out','radial_start')}
    record.update(id=name,route_id=route_id,dn=int(dn),components=[c['id'] for c in components],
        ports=ports,pieces=pieces,mesh_tessellation={'sector_degrees':SECTOR_DEGREES,'ring_vertices':RING_VERTICES},
        actual_geometry_scope='Closed native mesh with collision sectors enabled')
    return components,record
