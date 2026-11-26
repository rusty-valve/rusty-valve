"""Place equipment models and locate their physical pipe ports."""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import math
import re

import mujoco
import numpy as np

from scipy.spatial import ConvexHull
from .catalogue_surfaces import flange_surfaces
from .reactor_f_series_fittings import _frame
from .reactor_f_series_shell import add_mesh


@lru_cache(maxsize=12)
def _load(xml):
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def rotation(degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    matrix[np.abs(matrix) < 1e-10] = 0.
    return matrix


def mesh(model, geom, cache):
    kind=int(model.geom_type[geom])
    size=np.asarray(model.geom_size[geom])
    t=mujoco.mjtGeom
    is_mesh=kind==t.mjGEOM_MESH
    ident=int(model.geom_dataid[geom])
    key=(kind,ident) if is_mesh else (kind,*size)
    if key not in cache:
        if is_mesh:
            va,vn=model.mesh_vertadr[ident],model.mesh_vertnum[ident]
            fa,fn=model.mesh_faceadr[ident],model.mesh_facenum[ident]
            vertices=np.asarray(model.mesh_vert[va:va+vn],float)
            faces=np.asarray(model.mesh_face[fa:fa+fn],int)
        else:
            theta=np.linspace(0.,math.tau,48,endpoint=False)
            circle=np.column_stack((np.cos(theta),np.sin(theta)))
            if kind==t.mjGEOM_BOX:
                vertices=np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*size
            elif kind==t.mjGEOM_CYLINDER:
                vertices=np.vstack([np.c_[circle*size[0],np.full(48,z*size[1])] for z in (-1,1)])
            elif kind in (t.mjGEOM_SPHERE,t.mjGEOM_ELLIPSOID,t.mjGEOM_CAPSULE):
                phi=np.linspace(-math.pi/2,math.pi/2,25)[1:-1]
                vertices=np.vstack([np.c_[circle*math.cos(a),np.full(48,math.sin(a))] for a in phi]+
                                   [np.array([[0.,0.,-1.],[0.,0.,1.]])])
                radii=size if kind==t.mjGEOM_ELLIPSOID else size[0]
                vertices*=radii
                if kind==t.mjGEOM_CAPSULE:
                    vertices[:,2]+=np.sign(vertices[:,2])*size[1]
            else:
                raise ValueError(f'Catalogue primitive is unsupported: {model.geom(geom).name}')
            hull=ConvexHull(vertices)
            faces=hull.simplices.copy()
            tri=vertices[faces]
            reverse=np.einsum('ij,ij->i',np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),hull.equations[:,:3])<0
            faces[reverse]=faces[reverse,::-1]
        cache[key]=vertices,faces
    return cache[key]


def world_ports(spec, origin, matrix):
    result = {}
    for role, port in spec['ports'].items():
        row = deepcopy(port)
        row['position'] = (matrix @ port['position'] + origin).tolist()
        normal = matrix @ port['normal']
        normal[np.abs(normal) < 1e-8] = 0.
        row['normal'] = normal.tolist()
        result[role] = row
    return result


def _part_family(name):
    # Segment rings are collision candidates as one ring. Unrelated nozzles,
    # heads, legs and drives keep their own envelopes.
    return re.sub(r'_(?:sector_)?\d{2,4}(?:_(?:front|back|top|inner|outer)(?:_nut)?)?$', '', name)


def _source_collision_surface(assembly, folder, name, owner, parts, primary):
    """Compile a native convex solid from the catalogue part's actual outer vertices."""
    vertices=np.unique(np.round(np.concatenate(parts),8),axis=0)
    if len(vertices)<4 or np.linalg.matrix_rank(vertices-vertices[0])<3:
        return None
    hull=ConvexHull(vertices)
    indices=hull.simplices.copy()
    triangles=vertices[indices]
    flipped=np.einsum('ij,ij->i',np.cross(triangles[:,1]-triangles[:,0],
                       triangles[:,2]-triangles[:,0]),hull.equations[:,:3])<0
    indices[flipped]=indices[flipped,::-1]
    used,compact=np.unique(indices,return_inverse=True)
    c=add_mesh(assembly.scene,folder,name,'catalogue_equipment_solid',vertices[used],
               compact.reshape(indices.shape),[.55,.60,.63],'finished equipment exterior','physical collision geometry',
               'Convex parts from original exterior vertices for physical collision checks',owner)
    assembly.scene['components'].pop()
    # Keep the strongest support planes for each coarse direction for route
    # broad-phase refinement. The SAVED collider keeps every native hull face.
    normals=hull.equations[:,:3]
    rounded=np.round(normals*8).astype(int)
    _,keep=np.unique(rounded,axis=0,return_index=True)
    c.update(alpha=0.,group=5,catalogue_primary_body=primary,catalogue_source_solid=True,
             collision_planes=np.round(hull.equations[keep],9).tolist())
    return c


def insert_asset(assembly, uid, spec, origin, matrix):
    model, data = _load(spec['xml'])
    prefix = f'CAT_{uid}_'
    selected = spec.get('include_geoms', ())
    hidden = set(spec.get('hide_geoms', ())) | set(spec.get('remove_geoms', ()))
    visible = set(spec.get('show_geoms', ()))
    material_groups = defaultdict(list)
    families = defaultdict(list)
    names = {}
    offsets = spec.get('geom_offsets', {})
    cache = {}
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name in hidden or (selected and not any(name.startswith(n) for n in selected)):
            continue
        rgba = model.geom_rgba[i].copy()
        if model.geom_matid[i] >= 0:
            rgba = model.mat_rgba[model.geom_matid[i]].copy()
        keep_collider = name.startswith('collision_')
        if rgba[3] <= 1e-6 and name not in visible and not keep_collider:
            continue
        v, f = mesh(model, i, cache)
        world = (v @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]) @ matrix.T + origin
        if name in offsets:
            world += matrix @ np.asarray(offsets[name])
        for stem,extension in spec.get('extended_nozzles',{}).items():
            if name.startswith(stem):
                axis=matrix@np.asarray(extension['axis'])
                axial=world@axis
                fraction=(axial-axial.min())/np.ptp(axial)
                world+=np.outer(fraction*extension['length'],axis)
        family = name if name.startswith('original_') else _part_family(name)
        names[name] = family
        if not keep_collider and name!='flow_arrow':
            families[family].append(world)
        if keep_collider and rgba[3] <= 1e-6 and name not in visible:
            continue
        if name in visible:
            rgba[3] = 1.
        material_groups[tuple(map(float, rgba))].append((world, f, i))
    device_dir = assembly.folder / 'equipment'
    device_dir.mkdir(exist_ok=True)
    ids = []
    source_mesh_groups = {}
    surfaces, flange_evidence = flange_surfaces(model, material_groups, spec, origin, matrix, prefix)
    for name, vertices, faces, rgba in surfaces:
        part = add_mesh(assembly.scene, device_dir, name, 'catalogue_equipment_display',
                        vertices, faces, rgba[:3], str(spec['xml']), 'assembled scene',
                        'Original flange hole locations are retained. Face segments form a continuous surface.', uid)
        part.update(collision_mesh=False, alpha=rgba[3], catalogue_device=uid,
                    visual_only=True, annotation_only=True)
        ids.append(part['id'])
    for i, (rgba, parts) in enumerate(material_groups.items()):
        if not parts:
            continue
        vertices, faces, offset = [], [], 0
        for world, f, _ in parts:
            vertices.append(world)
            faces.append(f + offset)
            offset += len(world)
        part = add_mesh(assembly.scene, device_dir, prefix + f'asset_{i:02}',
                        'catalogue_equipment_display', np.concatenate(vertices), np.concatenate(faces),
                        rgba[:3], str(spec['xml']), 'assembled scene', 'Original display geometry under a rigid transform', uid)
        part.update(collision_mesh=False, alpha=rgba[3], catalogue_device=uid, visual_only=True, annotation_only=True)
        ids.append(part['id'])
        source_mesh_groups[part['id']]=[[int(gid),int(len(faces))] for _,faces,gid in parts]
    if not ids:
        raise ValueError(f'{uid} The assembled model has no visible original part')
    proxy_ids = {}
    for family, solids in families.items():
        pid = prefix + family
        primary = family in spec.get('primary_families', ()) or family in ('F5000_process_shell','F5000_head','F5000_process_base',
            'HXh_shell','HXh_left_head','HXh_right_head','housing') or family.startswith('original_mesh_') and family not in {v.get('source_geom') for v in spec['ports'].values()}
        component=_source_collision_surface(assembly,device_dir,pid,uid,solids,primary)
        if component:
            proxy_ids[family]=pid
            assembly.device_obstacles.append(component)
    ports = world_ports(spec, origin, matrix)
    for role, port in ports.items():
        source_names = list(port.get('component_ids', ())) + list(port.get('contact_geoms', ()))
        raw_disk = port.get('face_geom') or port.get('source_geom')
        if raw_disk:
            source_names.append(raw_disk)
        port['contact_ids'] = sorted({proxy_ids[names[n]] for n in source_names
                                     if n in names and names[n] in proxy_ids})
        route_family = {names[n].split('_plate')[0].split('_rf')[0] for n in source_names if n in names}
        path_necks = [proxy_ids[stem+'_neck'] for stem in route_family if stem+'_neck' in proxy_ids]
        path_necks += [proxy_ids[names[n]] for n in port.get('neck_geoms',[])
                       if n in names and names[n] in proxy_ids]
        port['departure_contact_ids'] = sorted(set(port['contact_ids']+path_necks))
        if port.get('bolt_offsets_m'):
            port['bolt_offsets_world_m'] = [(matrix @ v).tolist() for v in port['bolt_offsets_m']]
        if port.get('bolt_holes'):
            for bolt in port['bolt_holes']:
                bolt['center_world_m'] = (matrix @ bolt['center_m'] + origin).tolist()
        native_port = spec['ports'][role]
        if port.get('mating_profile'):
            points = np.asarray(port['mating_profile']['points_m'])
            x, y, _ = _frame(native_port['normal'])
            plane = points @ np.vstack((x, y))
            port['mating_profile_world_offsets_m'] = (plane @ matrix.T).tolist()
    bounds = np.asarray([c['box'] for c in assembly.scene['components'] if c['id'] in ids])
    return {'id': uid, 'prefix': prefix, 'source': str(spec['xml']), 'components': ids,
            'envelopes': [p['id'] for p in assembly.device_obstacles if p['owner']==uid],
            'origin': origin.tolist(), 'rotation_matrix': matrix.tolist(), 'ports': ports,
            'install_bounds_m': np.r_[bounds[:, :3].min(axis=0), bounds[:, 3:].max(axis=0)].tolist(),
            'missing_ports': spec.get('missing_ports', {}), 'facts': spec.get('facts', {}),
            'source_mesh_groups':source_mesh_groups,'flange_surfaces':flange_evidence}
