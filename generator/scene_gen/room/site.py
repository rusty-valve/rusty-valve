"""Add the final floor, platforms, stairs and access structures."""
import gc
import os
from pathlib import Path
import shutil
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import (
    saved_geometry, site_access, site_drain, site_frame, site_grating, site_ground,
    site_rails, site_stairs, site_treads,
)
from .export import nums
from .insulation import geometry_snapshot, read
from .process import write_json
from .shell import _frame_parts


def load(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return model, data


def append_geometry(world, assets, parts, folder):
    for part in parts:
        name = part['id']
        attributes = {'name': name, 'type': part['shape'], 'rgba': nums([*part['color'], part.get('alpha', 1.)])}
        if part['shape'] == 'mesh':
            ET.SubElement(assets, 'mesh', name=name, file=os.path.relpath(part['mesh_path'], folder))
            attributes.update(mesh=name, mass='0')
            if not part.get('collision_mesh'):
                attributes.update(contype='0', conaffinity='0')
        elif part['shape'] == 'cylinder':
            attributes.update(fromto=nums(np.asarray(part['ends']).ravel()), size=str(part['radius']))
        elif 'pose' in part:
            attributes.update({k: nums(v) for k, v in part['pose'].items()})
        else:
            low, high = np.asarray(part['box']).reshape(2, 3)
            attributes.update(pos=nums((low+high)/2), size=nums((high-low)/2))
        ET.SubElement(world, 'geom', **attributes)


def append_model(source, folder, removed, new_parts):
    root = ET.parse(source/'room.xml').getroot()
    world, assets = root.find('worldbody'), root.find('asset')
    for node in world.findall('geom'):
        if node.get('name') in removed:
            world.remove(node)
    for node in root.iter():
        if 'file' in node.attrib:
            node.set('file', os.path.relpath((source/node.attrib['file']).resolve(), folder))
    append_geometry(world, assets, new_parts, folder)
    path = folder/'room.xml'
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return path


def assert_preserved(baseline, model, data):
    for name, (kind, size, position, matrix) in baseline.items():
        gid = model.geom(name).id
        if kind != model.geom_type[gid] or not all(np.allclose(a, b, atol=1.e-8, rtol=0.) for a, b in (
                (size, model.geom_size[gid]), (position, data.geom_xpos[gid]), (matrix, data.geom_xmat[gid]))):
            raise ValueError('Site assembly changed original geometry: '+name)


def review_views(scene, floors, ground, changes):
    def aim(target, offset, note):
        return {'target':list(target),'eye':(np.asarray(target)+offset).tolist(),'note':note}

    by_id={v['part']:v['kind'] for v in changes}
    base=ground['entry_ramps'][0]
    channel=ground['drains'][0]
    cover=next((c for c in channel['covers'] if c['id'] in by_id),channel['covers'][len(channel['covers'])//2])
    views={
        'Containment curb and entrance':aim([(base[0]+base[2])/2,base[3],ground['datum_z_m']+.13],[4.2,-5.,3.2],
                         'Full-width plinth, mirrored curb ramps and walking area inside the wall.'),
        'drainage channel':aim([(cover['rect'][0]+cover['rect'][2])/2,channel['center_y'],channel['cover_datum_z_m']],
                     [1.4,-1.55,1.3],'Continuous channel, individual grating covers, joints and elevation changes.'),
    }
    boards=[(f['z_m'],b) for f in floors for b in f['boards']]
    if boards:
        level,board=next((item for item in boards if item[1]['id'] in by_id),boards[0])
        x0,y0,x1,y1=board['rect']
        views['grating joints']=aim([(x0+x1)/2,(y0+y1)/2,level],[1.30,-1.65,1.25],'Individual grating panels, physical joints and under-deck supports.')
    landing=next((p['box'] for p in scene['components'] if p['id']=='turn_landing_0'),None)
    if landing:
        views['stairway side']=aim([landing[3]+.9,landing[1]+.45,landing[5]-.42],[-1.,-3.1,.62],
                             'Open treads, front edges, stair stringers and yellow rails.')
    hatch=next(((f['z_m'],c['rect']) for f in floors for c in f['covers']),None)
    if hatch:
        z,(x0,y0,x1,y1)=hatch
        views['Reactor top opening cover']=aim([(x0+x1)/2,(y0+y1)/2,z],[1.0,-.85,1.65],'Removable grating cover over the agitator extraction opening.')
    return views


def generate(source, output, *, bund_height=site_ground.BUND_HEIGHT, pattern_seed_salt=None):
    saved_geometry.geometry_dependencies()
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('Site generation requires a new empty output directory')
    folder = output/'model'
    folder.mkdir(parents=True)
    try:
        insulation = read(source/'insulation.json')
        if insulation['revision'] != 'confirmed_valve_insulation_profiles':
            raise ValueError('Site generation requires a saved insulated model')
        scene = read(source/'geometry.json')
        frame = {name for name, _, _ in _frame_parts(scene['shell'])}
        removed = frame | {part['id'] for part in scene['components']
                           if part['id'].startswith('deck_') or part['id'] == 'ground'
                           or part['kind'] in site_stairs.STAIR_KINDS}
        old_ground = next(part['box'] for part in scene['components'] if part['id'] == 'ground')
        model, data = load(source/'room.xml')
        baseline = geometry_snapshot(model, data)
        first = len(scene['components'])
        omitted_knees = site_frame.build(scene, model, data, removed)
        graph = read(source/'process.json')
        # Earlier layout records can share the literal room ID "layout".
        # Use the frozen process ID so physical variations differ across rooms.
        salt = graph['id'].encode() if pattern_seed_salt is None else pattern_seed_salt
        seed = (graph['seed'], *salt)
        floors, violations, changes = site_grating.lay(scene, model, data, removed, folder, seed)
        tread_changes, support = site_treads.build(scene,folder,seed,model,data,removed)
        changes.extend(tread_changes)
        rails = site_rails.layout(scene, floors)
        piping = read(source/'piping.json')
        drainage = piping.get('material',{}).get('drain_endpoints',())
        boundary = site_ground.base_rect(scene,old_ground,[point['position'] for point in drainage])
        drainage_plan = site_drain.plan(scene,boundary,drainage)
        ground = site_ground.build(scene, folder, boundary, bund_height, drain=drainage_plan)
        ground['drains'], drain_changes = site_drain.build(scene,folder,drainage_plan,seed)
        changes.extend(drain_changes)
        scene['holes'] = {row['level']: [opening['rect'] for opening in row['openings']] for row in floors}
        scene['surfaces'] = [s for s in scene['surfaces'] if s['level'] == 0]
        scene['surfaces'] += [{'level': row['level'], 'box': [*rect[:2], row['z_m']-site_grating.DEPTH,
                                                            *rect[2:], row['z_m']]}
                              for row in floors for rect in row['panels']]
        del model, data
        gc.collect()
        new_parts = scene['components'][first:]
        scene['components'] = [part for part in scene['components'][:first] if part['id'] not in removed]+new_parts
        path = append_model(source, folder, removed, new_parts)
        model, data = load(path)
        assert_preserved({name: geom for name, geom in baseline.items() if name not in removed}, model, data)
        stairs, stair_violations = site_stairs.audit(scene, model, data, support)
        violations.extend(stair_violations)
        violations.extend(site_rails.audit(scene, model, data, rails))
        violations.extend(site_ground.audit(scene, model, data, ground))
        violations.extend(site_drain.audit(scene,model,data,ground['drains']))
        violations.extend({'code': 'FRAME_SOLID_COLLISION', **hit} for hit in site_stairs.collisions(
            scene, model, data, (p['id'] for p in new_parts if p['kind'] in ('beam','column'))))
        violations.extend(site_grating.audit(scene, model, data, floors))
        violations.extend({'code':'DECK_SOLID_COLLISION',**hit} for hit in site_stairs.collisions(
            scene,model,data,(p['id'] for p in new_parts if p['kind']=='walk_surface')))
        for name in ('piping.json', 'process.json', 'measure.json', 'insulation.json',
                     'valve_positioning.json', 'close_valve_pruning.json'):
            shutil.copy2(source/name, folder/name)
        (output/'layout').symlink_to((source.parent/'layout').resolve(), target_is_directory=True)
        shutil.copy2(source.parent/'input.json', output/'input.json')
        valve_access, operation_hits = site_access.audit(scene, graph, model, data,
                                                        read(folder/'valve_positioning.json'), ground)
        report = {'source': str(source), 'floors': floors, 'stair_flights': stairs, 'rails': rails,
                  'ground': ground, 'valve_access': valve_access, 'operation_conflicts': operation_hits,
                  'hard_violations': violations, 'blocked_knees': omitted_knees, 'variations': changes,
                  'review_views': review_views(scene,floors,ground,changes),
                  'frozen_geometry_verified': len(baseline)-len(removed),
                  'baseline': {**insulation['stage4'], 'insulation_pieces': len(insulation['pieces'])}}
        write_json(folder/'geometry.json', scene)
        write_json(folder/'site.json', report)
        return report
    except Exception as exc:
        write_json(output/'failure.json', {'type': type(exc).__name__, 'reason': str(exc),
                                         'traceback': traceback.format_exc()})
        raise
