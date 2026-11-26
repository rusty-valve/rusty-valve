"""Fit a connecting flange to an equipment port."""
from collections import Counter
import math

import numpy as np
from scipy.spatial import Delaunay

from .reactor_f_series_fittings import _frame, _ring, _round_sections
from .reactor_f_series_shell import add_mesh

EPS = 1e-8


def direction(value):
    value = np.asarray(value, float)
    return value / np.linalg.norm(value)


def _mate_dimensions(port, pipe_radius):
    f = port.get('flange_reference') or port.get('flange') or {}
    od = f.get('outer_diameter_mm')
    bore = f.get('fit_bore_mm') or f.get('bore_diameter_mm')
    outer = float(od) / 2000 if od else port.get('flange_outer_diameter_m', 0.) / 2
    inner = float(bore) / 2000 if bore else port.get('opening_diameter_m', 0.) / 2
    if outer <= 0 or inner <= 0 or inner >= outer:
        raise ValueError('Face is missing a valid outside diameter or clear bore')
    fit_radius = max(pipe_radius, inner + .0025)
    if fit_radius >= outer - .004:
        raise ValueError('Measured flange diameter cannot fit the pipe OD and transition neck')
    return f, outer, inner, fit_radius


def _mate_reach(port, f, outer, inner, fit, radius):
    thickness = port.get('flange_thickness_m', float(f.get('plate_thickness_mm', 18))/1000)
    raised = float(f.get('rf_height_mm', 0))/1000
    raised_outer = float(f.get('raised_face_diameter_mm', f.get('rf_diameter_mm', 0)))/2000
    has_rf = raised > 0 and inner < raised_outer < outer
    reach = thickness+(raised if has_rf else 0.)
    neck = max(.015, 2*abs(fit-radius), port.get('minimum_connector_departure_m', 0.)-reach)
    return thickness, raised, raised_outer, has_rf, reach, neck


def adapter_envelope(port, radius):
    """Maximum section radius and the same departure reach used to emit the adapter."""
    f, outer, inner, fit = _mate_dimensions(port, radius)
    *_, reach, neck = _mate_reach(port, f, outer, inner, fit, radius)
    stub = max(.02, radius, port.get('minimum_departure_stub_m', 0.))
    return reach+neck+stub, outer


def _drilled_profile(frame, origin, inner, outer, holes):
    x, y, _ = frame
    phase = np.arange(96)*math.tau/96
    circle = np.column_stack((np.cos(phase),np.sin(phase)))
    regions = [(np.zeros(2),inner,96)]
    for h in holes:
        offset = np.asarray(h['center_world_m'])-origin
        r = h['diameter_m']/2
        uv = np.array([offset@x,offset@y])
        if np.linalg.norm(uv)+r >= outer or np.linalg.norm(uv)-r <= inner:
            raise ValueError('Measured bolt holes overlap the inner or outer face boundary')
        regions.append((uv,r,16))
    loops = [circle*outer]+[uv+circle[::96//n]*radius for uv,radius,n in regions]
    uv = np.vstack(loops)
    triangles = Delaunay(uv).simplices
    mid = uv[triangles].mean(axis=1)
    keep = np.ones(len(triangles),dtype=bool)
    for center,r,n in regions:
        keep &= np.linalg.norm(mid-center,axis=1) >= r*math.cos(math.pi/n)-EPS
    triangles = triangles[keep]
    counts = Counter(tuple(sorted((int(a),int(b)))) for t in triangles
                     for a,b in zip(t,np.roll(t,1)))
    expected=set()
    offset=0
    for loop in loops:
        ring=range(offset,offset+len(loop))
        expected.update(tuple(sorted((int(i),int(j)))) for i,j in zip(ring,np.roll(list(ring),1)))
        offset+=len(loop)
    if {edge for edge,n in counts.items() if n==1} != expected:
        raise ValueError('Measured bolt-hole layout leaves no intact boundary')
    return uv @ np.vstack((x,y)),triangles


def extruded_profile(p, n, offsets, tris, thickness):
    tris = np.asarray(tris,int).copy()
    section = offsets[tris]
    reverse = np.cross(section[:,1]-section[:,0],section[:,2]-section[:,0])@n < 0.
    tris[reverse] = tris[reverse,::-1]
    directed = [(int(t[i-1]),int(t[i])) for t in tris for i in range(3)]
    counts = Counter(tuple(sorted(pair)) for pair in directed)
    npts = len(offsets)
    sides = [(i,j) for i,j in directed if counts[tuple(sorted((i,j)))]==1]
    walls = np.asarray([(i,j,k) for u,v in sides
                        for i,j,k in ((u,v,v+npts),(u,v+npts,u+npts))])
    return np.vstack((p+offsets,p+offsets+n*thickness)), np.vstack((tris[:,::-1],tris+npts,walls))


def _extruded_plate(a, folder, uid, p, n, offsets, tris, thickness, color):
    vertices, faces = extruded_profile(p, n, offsets, tris, thickness)
    element = add_mesh(a.scene,folder,uid,'piping_flange',vertices,faces,color,
                       'measured original face opening','mating geometry','The pipe-side plate retains the measured bore and bolt-hole positions.',uid)
    element['collision_mesh']=False
    return element['id']


def _make_mate(a, uid, port, dn, medium):
    """Create an opening in the pipe-side face and a hollow reducing neck."""
    p, n = (np.asarray(port[k], float) for k in ('position', 'normal'))
    r = a.radius(dn)
    f, outer, inner, fit = _mate_dimensions(port, r)
    frame = _frame(n)
    if port.get('bolt_frame_world'):
        x = np.asarray(port['bolt_frame_world'], float)
        frame = (x, np.cross(n, x), n)
    pitch = float(f.get('bolt_circle_mm', 0)) / 2000
    hole = float(f.get('hole_diameter_mm', 0)) / 2000
    count = int(f.get('bolt_count', 0))
    phase = port.get('bolt_phase_rad', math.pi / count if count else 0.)
    if port.get('bolt_offsets_world_m') and count:
        first=np.asarray(port['bolt_offsets_world_m'][0])
        x=direction(first-n*(first@n))
        frame=(x,np.cross(n,x),n)
        phase=0.
    use_pattern = count > 0 and hole > 0 and inner < pitch-hole < pitch+hole < outer
    thickness, raised, raised_outer, has_rf, reach, neck_length = _mate_reach(port, f, outer, inner, fit, r)
    color = (.55, .61, .64)
    folder = a.folder / 'equipment_flanges'
    folder.mkdir(exist_ok=True)
    if port.get('mating_profile_world_offsets_m'):
        offsets=np.asarray(port['mating_profile_world_offsets_m'])
        triangles=port['mating_profile']['triangles']
    elif port.get('bolt_holes') or use_pattern:
        x,y,_=frame
        holes=port.get('bolt_holes') or [
            {'center_world_m':p+pitch*(math.cos(angle)*x+math.sin(angle)*y),'diameter_m':2*hole}
            for angle in phase+np.arange(count)*math.tau/count
        ]
        try:
            offsets,triangles=_drilled_profile(frame,p,inner,outer,holes)
        except ValueError:
            if not use_pattern:
                raise
            offsets=None
    else:
        offsets=None
    if offsets is not None:
        parts=[_extruded_plate(a,folder,uid+'_plate',p+(raised if has_rf else 0.)*n,
                               n,offsets,triangles,thickness,color)]
        use_pattern=True
    else:
        plate=_round_sections(inner,outer,pitch if use_pattern else 0.,
                              hole if use_pattern else 0.,count if use_pattern else 8,phase)
        parts=_ring(a.scene,folder,uid,uid+'_plate','piping_flange',p,frame,
                    thickness,reach-thickness,plate,color,'measured equipment catalogue face','mating geometry',
                    'Coaxial faces retain the central opening.')
    if has_rf:
        parts += _ring(a.scene, folder, uid, uid+'_rf', 'piping_flange', p, frame,
                       raised, 0., _round_sections(inner, raised_outer, 0., 0., 8),
                       color, 'measured equipment catalogue face','mating geometry','The small raised face mates with the original.')
    x, y, _ = frame
    # Radial sectors leave the central bore open through the whole transition.
    end = p + n * (reach + neck_length)
    bore_end = min(inner, r * .875)
    throat=[]
    for i in range(24):
        angles = np.arange(i, i+2) * math.tau / 24
        v = np.cos(angles)[:, None] * x + np.sin(angles)[:, None] * y
        first = p + n*reach + np.vstack((v[0]*inner, v[0]*fit, v[1]*fit, v[1]*inner))
        last = end + np.vstack((v[0]*bore_end, v[0]*r, v[1]*r, v[1]*bore_end))
        # Tapered sides use their two real profiles, with no disk in the bore.
        faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7)]
        for j in range(4):
            k=(j+1)%4
            faces.extend(((j,k,k+4),(j,k+4,j+4)))
        elem = add_mesh(a.scene, folder, uid+f'_neck_{i:02}', 'piping_reducer',
                        np.vstack((first,last)), faces, a.color(medium),
                        'equipment catalogue face transition', 'mating geometry', 'Transition from the measured port bore to the recipe pipe diameter.', uid)
        elem['collision_mesh'] = False
        parts.append(elem['id']);throat.append(elem['id'])
    return end, parts, {'endpoint': port['endpoint'], 'position': p.tolist(), 'normal': n.tolist(),
                        'pipe_dn': dn, 'source_dn': port.get('dn'), 'opening_m': 2*inner,
                        'outer_m': 2*outer, 'bolt_pattern_modelled': use_pattern,
                        'transition': {'ends':[(p+n*reach).tolist(),end.tolist()],
                                       'outer_radii_m':[fit,r],'parts':throat},
                        'status': 'matched using measured or reference plate geometry' if use_pattern else 'Faces centered. Fasteners require separate verification.'}


def emit_catalogue_adapter(assembly, point, name, medium, dn):
    """Emit the open mating plate and register the exact pipe-side outlet."""
    from .piping_route_search import RouteUnavailable

    port = point['source_catalogue_face']
    face=np.asarray(port['position'],float)
    normal=np.asarray(port['normal'],float)
    end, pieces, evidence = _make_mate(assembly, name, port, dn, medium)
    radius=assembly.radius(dn)
    stub_end=end+normal*max(.02,radius,port.get('minimum_departure_stub_m',0.))
    stub=assembly.pipe(name+'_out',end,stub_end,dn,medium,name)
    from .layout import add_pipe
    passage=add_pipe(assembly.scene,name+'_bore_trace',face,end,min(radius*.875,evidence['opening_m']/2),
                     'adapter',(.4,.5,.6),name)
    passage.update(alpha=0.,group=5,annotation_only=True,visual_only=True,
                   verified_by_catalogue_mating_mesh=pieces)
    allowed=[*port['contact_ids'],*port.get('departure_contact_ids',()),*pieces,stub['id']]
    components={c['id']:c for c in assembly.scene['components']}
    for piece in pieces:
        components[piece].update(piping=True,catalogue_mate_solid=True)
        for original in port['contact_ids']:
            assembly.allow(original,piece,face,normal=normal)
        assembly.allow(piece,stub['id'],end,normal=normal)
    joint={**evidence,'id':name,'end':stub_end.tolist(),'pipe_stub':stub['id'],
           'source_components':port['contact_ids'],'components':pieces,
           'departure_contact_ids':list(dict.fromkeys(allowed)),
           'through_bore':passage['id'],'catalogue_joint':True,'mate_bore_open_modelled':True}
    assembly.scene.setdefault('catalogue_flange_joints',[]).append(joint)
    return joint,stub
