"""Build and check an insulation shell for a threaded lever valve."""
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import saved_geometry as geometry
from .export import nums
from .insulation import COLOR, CONTROL_NAMES, HANDLE_GAP, MATERIAL, read
from .pipeline_status import pipeline_status
from .process import write_json


PROFILE = 'threaded_lever_cylinder'
OUTER_RADIUS, INNER_RADIUS = .124, .112
OPENING_AXIAL, OPENING_WIDTH = .076, .120
END_PLATE_THICKNESS = .001
STEM_COLLAR_RADIUS = .028
ARC_SECTIONS = (16, 80)


def connected_pipes(components, valve, ends):
    origin, matrix = np.asarray(valve['origin']), np.asarray(valve['matrix'])
    candidates = [c for c in components if c['kind'] == 'pipe' and c['shape'] == 'cylinder']
    pipes = []
    for index, end in enumerate(ends):
        face = origin+end*matrix[:, 0]
        direction = 2*index-1
        pipe, = (c for c in candidates if any(
            np.linalg.norm(np.asarray(a)-face) < 1e-6 and (np.asarray(b)-a)@matrix[:, 0]*direction > 1e-6
            for a, b in (c['ends'], c['ends'][::-1])))
        pipes.append(pipe)
    return pipes


def shell_mesh(ends, pipe_radii, bonnet_top, *, outer_radius=OUTER_RADIUS, inner_radius=INNER_RADIUS,
               opening_axial=OPENING_AXIAL, opening_width=OPENING_WIDTH, shaft_radius=STEM_COLLAR_RADIUS):
    """Join the cylindrical wall and end plates, then line the recessed operator opening."""
    def section(radius):
        edge = math.asin(opening_width/(2*radius))
        top = np.linspace(-edge, edge, ARC_SECTIONS[0], endpoint=False)
        wall = np.linspace(edge, 2*np.pi-edge, ARC_SECTIONS[1], endpoint=False)
        angle = np.r_[top, wall]
        return np.c_[radius*np.sin(angle), radius*np.cos(angle)]

    mid = np.mean(ends)
    stations = (ends[0], ends[0]+END_PLATE_THICKNESS, mid-opening_axial/2,
                mid+opening_axial/2, ends[1]-END_PLATE_THICKNESS, ends[1])
    rings = (section(outer_radius), section(inner_radius))
    count = sum(ARC_SECTIONS)
    vertices, surface, cells = [], {}, []

    def add_ring(x, section):
        first = len(vertices)
        vertices.extend((x, y, z) for y, z in section)
        return np.arange(first, first+count)

    wall = np.asarray([[add_ring(x, ring) for ring in rings] for x in stations])
    caps = [[add_ring(stations[i], rings[1]*radius/inner_radius) for i in pair]
            for radius, pair in zip(pipe_radii, ((0, 1), (4, 5)))]

    def add_cell(indices, faces):
        cells.append(indices)
        for face in faces:
            key = tuple(sorted(face))
            if key in surface:
                del surface[key]
            else:
                surface[key] = face

    def annular_cell(slab, sector, outer, inner):
        nxt = (sector+1)%count
        a,b,c,d = outer[0][sector], outer[0][nxt], inner[0][nxt], inner[0][sector]
        e,f,g,h = outer[1][sector], outer[1][nxt], inner[1][nxt], inner[1][sector]
        add_cell((a,b,c,d,e,f,g,h),
                 ((a,b,c,d), (e,h,g,f), (a,e,f,b), (d,c,g,h), (a,d,h,e), (b,f,g,c)))

    for slab in range(len(stations)-1):
        outer, inner = wall[slab:slab+2].swapaxes(0, 1)
        for sector in range(count):
            if slab == 2 and sector < ARC_SECTIONS[0]:
                continue
            annular_cell(slab, sector, outer, inner)
    for slab, cap in zip((0, 4), caps):
        for sector in range(count):
            annular_cell(slab, sector, wall[slab:slab+2, 1], cap)

    # The opening retains its outer dimensions. A recessed lining meets the
    # bonnet and covers the cavity up to the circular shaft clearance.
    slot = slice(0, ARC_SECTIONS[0]+1)
    rim = np.r_[wall[2, 1, slot][::-1], wall[3, 1, slot]]
    rim = np.asarray(vertices)[rim]
    left, right = np.split(rim, 2)
    rim = np.vstack((left, np.linspace(left[-1], right[0], 13)[1:-1],
                     right, np.linspace(right[-1], left[0], 13)[1:-1]))
    rim[:, 0] = mid+1.04*(rim[:, 0]-mid)
    rim[:, 1] *= 1.04
    rim[:, 2] += .0007
    shaft = rim.copy()
    shaft[:, :2] *= (shaft_radius/np.linalg.norm(rim[:, :2], axis=1))[:, None]
    shaft[:, 2] = bonnet_top
    size = len(rim)
    first = len(vertices)
    vertices.extend(np.vstack((rim, shaft, rim-[0., 0., END_PLATE_THICKNESS],
                               shaft-[0., 0., END_PLATE_THICKNESS])).tolist())
    for i in range(size):
        j = (i+1)%size
        a, b, c, d = (first+k for k in (i, j, j+size, i+size))
        for u,v,w in ((a,b,c), (a,c,d)):
            x,y,z = (n+2*size for n in (u,v,w))
            add_cell((u,v,w,x,y,z), ((u,v,w), (x,z,y), (u,x,y,v), (v,y,z,w), (w,z,x,u)))

    faces = [triangle for face in surface.values()
             for triangle in ((face[0], face[i], face[i+1]) for i in range(1, len(face)-1))]
    opening = wall[2:4, :, [0, ARC_SECTIONS[0]]].ravel()
    groups = {'wall': wall.ravel(), 'opening': opening,
              'end_holes': [caps[0][0], caps[1][1]], 'stem_hole': np.arange(first+size, first+2*size)}
    return np.asarray(vertices), np.asarray(faces), cells, groups


def rebase_assets(root, source, output):
    for item in root.iter():
        if 'file' in item.attrib:
            item.set('file', os.path.relpath((source/item.attrib['file']).resolve(), output))


def save_replacement(source, folder, valve, old_geom, vertices, faces):
    room = ET.parse(source/'room.xml').getroot()
    assets, world = room.find('asset'), room.find('worldbody')
    attachment, = (el for el in world.iter('attach') if el.get('prefix') == valve['prefix'])
    valve_asset, = (el for el in assets.findall('model') if el.get('name') == attachment.get('model'))
    valve_file = (source/valve_asset.attrib['file']).resolve()
    rebase_assets(room, source, folder)
    for parent in (assets, world):
        old, = (el for el in parent if el.get('name') == old_geom)
        parent.remove(old)

    folder.mkdir(parents=True)
    cladding = folder/'cladding'
    cladding.mkdir()
    obj = cladding/'threaded_lever_shell.obj'
    with obj.open('w') as out:
        for vertex in vertices:
            out.write('v '+nums(vertex)+'\n')
        for face in faces:
            out.write('f '+' '.join(str(i+1) for i in face)+'\n')
    package = ET.parse(valve_file).getroot()
    rebase_assets(package, valve_file.parent, cladding)
    body, = (b for b in package.iter('body') if b.get('name') == attachment.get('body'))
    name = 'threaded_lever_shell'
    ET.SubElement(package.find('asset'), 'mesh', name=name, file=obj.name)
    ET.SubElement(package.find('asset'), 'material', name=MATERIAL, rgba=nums([*COLOR, 1.]),
                  specular='.75', shininess='.65')
    ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name, material=MATERIAL,
                  contype='0', conaffinity='0', mass='0', group='0')
    package_path = cladding/'threaded_lever_valve.xml'
    ET.ElementTree(package).write(package_path, encoding='utf-8')
    cloned_asset = valve_asset.attrib['name']+'_insulation_sample'
    ET.SubElement(assets, 'model', name=cloned_asset, file=os.path.relpath(package_path, folder))
    attachment.set('model', cloned_asset)
    ET.indent(room)
    ET.ElementTree(room).write(folder/'room.xml', encoding='utf-8', xml_declaration=True)
    return valve['prefix']+name, obj


def measure_travel(folder, valve, peers, cells):
    """Measure against the union of convex cells, preserving the actual top opening."""
    saved = ET.parse(folder/'room.xml').getroot()
    root = ET.Element('mujoco')
    assets, world = ET.SubElement(root, 'asset'), ET.SubElement(root, 'worldbody')
    prefixes = {v['prefix'] for v in (valve, *peers)}
    models = set()
    for frame in saved.findall('worldbody/frame'):
        attach = frame.find('attach')
        if attach is not None and attach.get('prefix') in prefixes:
            models.add(attach.attrib['model'])
            world.append(deepcopy(frame))
    for asset in saved.findall('asset/model'):
        if asset.get('name') in models:
            assets.append(deepcopy(asset))
    with tempfile.TemporaryDirectory(dir=folder) as temporary:
        probe = Path(temporary)/'clearance.xml'
        rebase_assets(root, folder, probe.parent)
        ET.ElementTree(root).write(probe, encoding='utf-8')
        spec = mujoco.MjSpec.from_file(str(probe))
        body = spec.body(valve['prefix']+'valve_base')
        for index, points in enumerate(cells):
            name = f'cladding_cell_{index}'
            spec.add_mesh(name=name, uservert=points.ravel())
            body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, contype=0,
                          conaffinity=0, mass=0.)
        model = spec.compile()
    data = mujoco.MjData(model)
    cell_ids = [model.geom(f'cladding_cell_{i}').id for i in range(len(cells))]
    return {other['id']: measure_operator(model, data, other, cell_ids) for other in (valve, *peers)}


def measure_operator(model, data, valve, cell_ids):
    prefix = valve['prefix']
    controls = [(g['name'], model.geom(prefix+g['name']).id) for g in valve['local_geoms']
                if g['moving'] and g['rgba'][3] > 0 and any(word in g['name'] for word in (*CONTROL_NAMES, 'stem'))]
    states = valve['motion']['joint_states']
    travel = [(model.joint(prefix+name).qposadr[0], value, states['closed'][name])
              for name, value in states['open'].items()]
    # 0.25° intervals include both ends. A point at 0.4 m from the axis can
    # travel at most 0.88 mm from the nearest sample.
    fractions = np.linspace(0., 1., 361)
    best = (.2, '', 0.)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    axis = data.xaxis[model.joint(prefix+valve['motion']['operator_joint']).id]
    anchor = data.xanchor[model.joint(prefix+valve['motion']['operator_joint']).id]
    radius = float(np.linalg.norm(np.cross(open_control_points(model, data, controls)-anchor, axis), axis=1).max())
    for fraction in fractions:
        for address, start, end in travel:
            data.qpos[address] = start+(end-start)*fraction
        mujoco.mj_kinematics(model, data)
        for name, gid in controls:
            for cell in cell_ids:
                gap = mujoco.mj_geomDistance(model, data, cell, gid, max(best[0], 1.e-6), None)
                if gap < best[0]:
                    best = (float(gap), name, float(fraction))
    gap, nearest, fraction = best
    between_samples = radius*max(abs(end-start) for _, start, end in travel)/(2*(len(fractions)-1))
    if gap-between_samples < HANDLE_GAP:
        raise ValueError(f"Measured lever or stem clearance is insufficient: {valve['id']} {gap*1000:.3f} mm, part {nearest}")
    return {'min_m': gap, 'nearest_part': nearest, 'travel_deg': fraction*90,
            'frames': len(fractions), 'lower_bound_m': gap-between_samples,
            'scope': 'Distance from saved aluminum skin mesh sections to the visible handle and stem'}


def open_control_points(model, data, controls):
    cache = {}
    return np.vstack([geometry.mesh(model, gid, cache)[0]@data.geom_xmat[gid].reshape(3, 3).T+
                      data.geom_xpos[gid] for _, gid in controls])


def measure_saved(folder, source, name, removed, valve, peers, ends, vertices, groups, cells, pipes):
    model = mujoco.MjModel.from_xml_path(str(folder/'room.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    gid = model.geom(name).id
    body = model.body(valve['prefix']+'valve_base').id
    if model.geom_bodyid[gid] != body or model.geom_contype[gid] or model.geom_conaffinity[gid]:
        raise ValueError('Insulation belongs to the fixed valve body and is visual only')
    local, faces = geometry.mesh(model, gid, {})
    absolute = local@data.geom_xmat[gid].reshape(3, 3).T+data.geom_xpos[gid]
    recovered = (absolute-data.xpos[body])@data.xmat[body].reshape(3, 3)
    if not np.allclose(recovered, vertices, atol=2e-6, rtol=0):
        raise ValueError('Insulation mesh coordinates differ on readback')
    r = np.linalg.norm(recovered[groups['wall'], 1:], axis=1)
    inner, outer = float(r.min()), float(r.max())
    axial = (float(recovered[:,0].min()), float(recovered[:,0].max()))
    triangles = recovered[faces]
    end_caps = []
    for end, hole, pipe in zip(ends, groups['end_holes'], pipes):
        cap = triangles[np.all(np.isclose(triangles[:,:,0], end, atol=2e-6, rtol=0.), axis=1)]
        cap_area = float(.5*np.abs(np.cross(cap[:,1]-cap[:,0], cap[:,2]-cap[:,0])[:,0]).sum())
        measured_radius = np.linalg.norm(recovered[hole,1:], axis=1)
        if not (np.allclose(measured_radius, pipe['radius'], atol=2e-6, rtol=0.)
                and math.isclose(cap_area, math.pi*(OUTER_RADIUS**2-pipe['radius']**2), rel_tol=.005)):
            raise ValueError('End closure does not span from cylinder rim to pipe OD: '+pipe['id'])
        end_caps.append({'pipe': pipe['id'], 'opening_diameter_m': 2*float(np.mean(measured_radius)),
                         'covered_area_m2': cap_area, 'station_m': float(end)})
    stem_hole = 2*float(np.mean(np.linalg.norm(recovered[groups['stem_hole'],:2], axis=1)))
    body_pos, body_quat = (array[body].copy() for array in (model.body_pos, model.body_quat))
    shell_pos, shell_mat = data.geom_xpos[gid].copy(), data.geom_xmat[gid].copy()
    hinge = model.joint(valve['prefix']+valve['motion']['operator_joint']).qposadr[0]
    control = model.body(valve['prefix']+'valve_rotor').id
    rotor_open = data.xmat[control].copy()
    data.qpos[hinge] = valve['motion']['valve_range_rad'][1]
    mujoco.mj_kinematics(model, data)
    closed_shift = float(np.linalg.norm(data.geom_xpos[gid]-shell_pos))
    closed_rotation_error = float(np.linalg.norm(data.geom_xmat[gid]-shell_mat))
    rotor_change = float(np.linalg.norm(data.xmat[control]-rotor_open))
    tilt = np.empty(4)
    mujoco.mju_axisAngle2Quat(tilt, np.array([0., 1., 0.]), .4)
    mujoco.mju_mulQuat(model.body_quat[body], tilt, body_quat)
    shift = np.array([.15, -.11, .08])
    model.body_pos[body] += shift
    mujoco.mj_kinematics(model, data)
    basis = data.xmat[body].reshape(3, 3)
    moved = (local@data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]-data.xpos[body])@basis
    rigid_error = float(np.max(np.linalg.norm(moved-recovered, axis=1)))
    if max(rigid_error, closed_shift, closed_rotation_error) > 2e-6 or rotor_change < 1.:
        raise ValueError('Valve-body motion or independent handle rotation verification failed')
    model.body_pos[body], model.body_quat[body] = body_pos, body_quat
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    baseline = mujoco.MjModel.from_xml_path(str(source/'room.xml'))
    before = mujoco.MjData(baseline)
    mujoco.mj_kinematics(baseline, before)
    if baseline.ngeom != model.ngeom:
        raise ValueError('The sample may replace only one piece of geometry')
    for old in range(baseline.ngeom):
        geom_name = baseline.geom(old).name
        if geom_name == removed:
            continue
        new = model.geom(geom_name).id
        for a, b in ((before.geom_xpos[old], data.geom_xpos[new]), (before.geom_xmat[old], data.geom_xmat[new]),
                     (baseline.geom_size[old], model.geom_size[new])):
            if not np.allclose(a, b, atol=1e-8, rtol=0.):
                raise ValueError('Other geometry changed in the comparison: '+geom_name)
    checked = baseline.ngeom-1
    del baseline, before, model, data
    return {'length_m': axial[1]-axial[0], 'outer_diameter_m': 2*outer, 'inner_diameter_m': 2*inner,
            'thickness_m': outer-inner, 'opening_axial_m': float(np.ptp(recovered[groups['opening'],0])),
            'opening_width_m': float(np.ptp(recovered[groups['opening'],1])),
            'end_errors_m': (np.asarray(axial)-ends).tolist(),
            'end_caps': end_caps, 'stem_collar_opening_m': stem_hole,
            'travel_clearances': measure_travel(folder, valve, peers, [recovered[list(cell)] for cell in cells]),
            'motion': {'fixed_body': valve['prefix']+'valve_base', 'test_translation_m': shift.tolist(),
                       'test_rotation_rad': .4, 'world_transform_error_m': rigid_error,
                       'shell_shift_during_closing_m': closed_shift,
                       'shell_rotation_error_during_closing': closed_rotation_error,
                       'rotor_matrix_change': rotor_change},
            'other_geometries_verified': checked,
            'actual_box_m': np.r_[absolute.min(axis=0), absolute.max(axis=0)].tolist()}


def derive_sample(source, folder, valve_id):
    graph, scene, piping, report, baseline = (read(source/name) for name in
        ('process.json', 'geometry.json', 'piping.json', 'insulation.json', 'measure.json'))
    valve, = (v for v in graph['valves'] if v['id'] == valve_id)
    if valve['operator'] != 'lever' or valve['dn'] != 80 or 'NPT' not in valve['params']['product']['end_connection']:
        raise ValueError('The sample requires an insulated DN80 NPT lever valve')
    record, = (r for r in report['pieces'] if r.get('owner') == valve_id and r['profile'] == 'lever_straight_shoulders')
    label, = (row for row in report['valve_labels'] if row['valve'] == valve_id)
    old = record['id']
    component, = (c for c in scene['components'] if c['id'] == old)
    origin, matrix = np.asarray(valve['origin']), np.asarray(valve['matrix'])
    ends = np.sort([(np.asarray(p['position'])-origin)@matrix[:,0] for p in valve['world_ports'].values()])
    if not math.isclose(float(np.ptp(ends)), .202, abs_tol=1e-6):
        raise ValueError('Distance between thread faces differs from the specified 202 mm')
    pipes = connected_pipes(scene['components'], valve, ends)
    bonnet_top = next(g['bbox'][5] for g in valve['local_geoms'] if g['name'] == 'bonnet')
    vertices, faces, cells, groups = shell_mesh(ends, [p['radius'] for p in pipes], bonnet_top)
    source_label = valve_id+'insulation shell'
    nearby = {r['valve'] for r in report['handle_clearance'] if any(row['source'] == source_label for row in r['rows'])}
    peers = [v for v in graph['valves'] if v['id'] in nearby and v['id'] != valve_id]
    name, obj = save_replacement(source, folder, valve, old, vertices, faces)
    measured = measure_saved(folder, source, name, old, valve, peers, ends, vertices, groups, cells, pipes)
    component.pop('flat_shading', None)
    component.update(id=name, pos=origin.tolist(), box=measured['actual_box_m'], mesh_path=str(obj),
                     thickness_m=OUTER_RADIUS-INNER_RADIUS)
    record.clear()
    record.update(id=name, source=valve_id, owner=valve_id, scope='manual valve on an insulated pipe', kind='valve body',
                  profile=PROFILE, thickness_m=OUTER_RADIUS-INNER_RADIUS,
                  measured_thickness_m=measured['thickness_m'], actual_box_m=measured['actual_box_m'])
    label.update(cladding=name, position=(origin+matrix[:,2]*OUTER_RADIUS).tolist(),
                 label='DN80 threaded lever valve with pipe-end closures and a lined operator opening')
    for r in report['handle_clearance']:
        if r['valve'] not in measured['travel_clearances']:
            continue
        clearance = measured['travel_clearances'][r['valve']]
        for row in r['rows']:
            if row['source'] == source_label:
                row.update(after_m=clearance['min_m'], thickness_m=record['thickness_m'], pending=False,
                           sampled_poses=clearance['frames'])
        r['closest_after_m'] = min(row['after_m'] for row in r['rows'])
        r['pending'] = any(row['pending'] for row in r['rows'])
    report.update(threaded_lever_sample={'valve': valve_id, 'reference': str(source), **measured},
                  stage4=pipeline_status(baseline, piping, model_present=True))
    for filename in ('piping.json', 'process.json', 'measure.json'):
        shutil.copy2(source/filename, folder/filename)
    write_json(folder/'geometry.json', scene)
    write_json(folder/'insulation.json', report)
    return report
