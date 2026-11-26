"""Choose detailed equipment models and find their pipe connections."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
from xml.etree import ElementTree

import numpy as np

from .catalogue_geometry import insert_asset, rotation
from .runtime_paths import RUNTIME
from .equipment_catalog import CATALOG
from . import equipment_models


REACTOR_PORTS = json.loads(Path(__file__).with_name("reactor_ports.json").read_text())


def catalogue_archive():
    return RUNTIME / CATALOG['detailed_models']['archive']


def catalogue_manifest():
    with ZipFile(catalogue_archive()) as package:
        original = json.loads(package.read('catalogue.json'))
    original.update(equipment_models.specifications())
    return original


def extract_catalogue(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with ZipFile(catalogue_archive()) as package:
        package.extractall(folder, (name for name in package.namelist()
                             if name.split('/')[0] not in (*equipment_models.ASSETS, 'catalogue.json')))
    manifest = catalogue_manifest()
    for key in equipment_models.ASSETS:
        equipment_models.write_asset(folder/key, key, manifest[key])
    (folder/'catalogue.json').write_text(json.dumps(manifest, ensure_ascii=False))
    return folder


def read_catalogue(folder):
    folder = Path(folder)
    result = json.loads((folder/'catalogue.json').read_text())
    for spec in result.values():
        spec['xml'] = folder/spec['xml']
    return result


def _model_key(unit, row):
    kind = unit['kind']
    models = CATALOG['detailed_models']['selection']
    if kind == 'vessel' and not unit.get('jacket'):
        return models.get(row.get('dimension_spec', {}).get('catalog_key'))
    if kind in ('condenser','exchanger'):
        return models.get(kind+'.'+unit['orientation'])
    return models.get(kind)


def _assign_reactor(graph, uid, spec, model):
    unit = graph['units'][uid]
    source_ports = spec['ports']
    assigned = {role: REACTOR_PORTS[role][model] for role in unit['ports'] if role in REACTOR_PORTS}
    if len(assigned) != len(set(assigned.values())):
        raise ValueError(f'{uid} assigns multiple logical ports to the same physical nozzle')
    for tag, reference in REACTOR_PORTS['reopen_from'][model].items():
        if tag not in assigned.values() or tag in source_ports:
            continue
        template = deepcopy(source_ports[reference])
        prefix = f'{model}_port_{tag}_'
        original = spec['facts']['port_positions'][tag]
        template.update(position=original['position'], source_tag=tag, manufacturer_use=original['manufacturer_use'],
                        component_ids=[name.replace(f'_port_{reference}_', f'_port_{tag}_')
                                       for name in template['component_ids']],
                        description='Remove the original thermowell assembly. Keep its stub and flange as a process port.')
        source_ports[tag] = template
        names = (g.get('name', '') for g in ElementTree.parse(spec['xml']).iter('geom'))
        spec['cap_parts'][tag] = [name for name in names if name.startswith(
            (prefix+'blank_ring_', prefix+'cover_center', prefix+'bolt_'))]
    spec['ports'] = {role: {**source_ports[tag], 'nozzle_mount': REACTOR_PORTS['mount'][model][tag]}
                     for role, tag in assigned.items() if tag in source_ports}
    spec['remove_geoms'] = [name for tag in assigned.values() for name in spec['cap_parts'].get(tag, ())]
    spec['missing_ports'] = {role: 'No port assigned because the equipment catalogue has no matching physical face' for role in unit['ports'] if role not in spec['ports']}
    spec['facts'].update(port_assignments=assigned, return_manifold=unit.get('return_manifold', {}),
                        port_shortage=spec['missing_ports'])
    return assigned


def physical_port(port, uid, role, level, logical):
    source=deepcopy(port)
    source['endpoint']=uid+'.'+role
    pos=source['position']
    old=deepcopy(logical)
    old.update(pos=pos,normal=source['normal'],source_tag=source['source_tag'],
               catalogue_face=source,component_ids=source['contact_ids'],
               nozzle_component_id=next(iter(source['contact_ids']),None),
               equipment_neck_dn=source['dn'] or logical['dn'],
               actual_neck_outer_diameter_m=source['opening_diameter_m'])
    if 'nozzle_mount' in source:
        old['nozzle_mount'] = source['nozzle_mount']
    if old.get('operation_reference'):
        ref=old['operation_reference']
        ref.update(endpoint=source['endpoint'],interface_height_m=pos[2]-ref['standing_surface_z_m'])
    return old


def _stagger_heater_inlet(spec):
    inlet,drain=(spec['ports'][role] for role in ('in','utility_out'))
    normal=np.asarray(inlet['normal'])
    delta=np.asarray(inlet['position'])-drain['position']
    radius_sum=(inlet['flange_outer_diameter_m']+drain['flange_outer_diameter_m'])/2
    if (not np.allclose(normal,drain['normal']) or abs(delta@normal)>1e-6
            or np.linalg.norm(delta)>=radius_sum):
        return
    shoulders=sum((p['flange_reference']['plate_thickness_mm']+p['flange_reference']['rf_height_mm'])/1000
                  for p in (inlet,drain))
    extension=shoulders+.02
    shift=normal*extension
    for name in inlet['component_ids']:
        offsets=spec.setdefault('geom_offsets',{})
        offsets[name]=(np.asarray(offsets.get(name,[0.,0.,0.]))+shift).tolist()
    stem=inlet['component_ids'][0].split('_plate_')[0]
    spec['extended_nozzles']={stem+'_neck_':{'axis':normal.tolist(),'length':extension}}
    inlet['position']=(np.asarray(inlet['position'])+shift).tolist()
    spec['facts']['s3_flange_clearance']={'moved_port':inlet['source_tag'],'axial_extension_m':extension,
        'reason':'Coplanar flange outside diameters overlap. Extend the original material inlet while keeping the condensate port elevation.',
        'source':'Scene stub approximation based on the two original flange thicknesses and diameters'}


def place_asset(scene, folder, uid, spec, target, yaw, *, anchor=None):
    assembly=SimpleNamespace(scene=scene,folder=Path(folder),device_obstacles=[])
    matrix=rotation(yaw)
    offset=np.asarray(spec.get('anchor',(0.,0.,0.)) if anchor is None else anchor)
    device=insert_asset(assembly,uid,spec,np.asarray(target)-matrix@offset,matrix)
    scene['components'].extend(assembly.device_obstacles)
    return device


def catalogue_pose(row, key, spec, catalogue, level_z):
    yaw = row.get('rotation_deg', 0)+REACTOR_PORTS['yaw_offset_deg'].get(key, 0)
    target = [*row['center'][:2], level_z]
    anchor = None
    if key == 'filter':
        pump = catalogue['pump']
        target[2] += max(pump['ports']['in']['position'][2]-pump['anchor'][2],
                         spec['anchor'][2]-spec['bounds_m']['min'][2]+.035)
        anchor = (np.asarray(spec['ports']['in']['position'])+spec['ports']['out']['position'])/2
    return target, yaw, anchor


def prepare_full_room(frozen, folder):
    """Put measured components into the accepted layout before planning any pipes."""
    layout=frozen['contracts']['layout']
    graph=frozen['contracts']['sizing']['graph']
    scene=frozen['scene']
    rows=layout['equipment']
    selected={r['id']:key for r in rows if (key:=_model_key(graph['units'][r['id']],r))}
    if not selected:
        return
    catalogue_dir=extract_catalogue(folder/'catalogue_sources')
    catalogue=read_catalogue(catalogue_dir)
    frozen['equipment_catalogue_dir']=str(catalogue_dir)
    devices=[];supports=[]
    for row in rows:
        uid=row['id']
        if uid not in selected:
            continue
        key=selected[uid]
        unit=graph['units'][uid]
        spec=deepcopy(catalogue[key])
        if key=='exchanger_vertical':
            _stagger_heater_inlet(spec)
        old_ports=row['ports']
        if key in REACTOR_PORTS['yaw_offset_deg']:
            _assign_reactor(graph,uid,spec,key)
        level=row['level']
        level_z=scene['floors'][level]
        target,yaw,anchor=catalogue_pose(row,key,spec,catalogue,level_z)
        keep={c['id'] for c in scene['components'] if c['id'].startswith(uid+'_lug_shoe')}
        removed=set(row['component_ids'])-keep
        scene['components']=[c for c in scene['components'] if c['id'] not in removed]
        device=place_asset(scene,folder,uid,spec,target,yaw,anchor=anchor)
        device.update(kind=unit['kind'],catalogue_key=key,level=level,yaw_deg=yaw)
        if key in REACTOR_PORTS['yaw_offset_deg']:
            from .catalogue_support import install_reactor_supports
            assembly=SimpleNamespace(scene=scene,folder=Path(folder))
            supports.extend(install_reactor_supports(assembly,device))
        actual={}
        for role, logical in unit['ports'].items():
            if role in device['ports']:
                old={**logical,**old_ports.get(role,{})}
                port=device['ports'][role]
                actual[role]=physical_port(port,uid,role,level,old)
            else:
                actual[role]={**logical,'catalogue_missing':device['missing_ports'].get(role,'Equipment catalogue has no matching physical port face')}
            # The sizing graph retains design line DN; equipment hardware adds its own DN.
            merged={**logical,**actual[role]}
            unit['ports'][role]=merged
        for port in actual.values():
            if 'pos' in port:
                port['piping_level']=max(i for i,z in enumerate(scene['floors']) if z<=port['pos'][2]+1e-8)
        row.update(ports=actual,component_ids=device['components']+device['envelopes'],
                   z_range_m=device['install_bounds_m'][2::3],rotation_deg=yaw,
                   catalogue_key=key, catalogued_detail=True)
        devices.append(device)
    scene['catalogue_devices']=devices
    scene['catalogue_supports']=supports
    for instance in scene.get('equipment_instances',[]):
        row=next((r for r in rows if r['id']==instance.get('id') and r.get('catalogued_detail')),None)
        if row:
            instance.update(ports=deepcopy(row['ports']),component_ids=row['component_ids'],
                            catalogue_key=row['catalogue_key'],z_range_m=row['z_range_m'])
    frozen['graph']['edges']=deepcopy(graph['edges'])
    # Keep the layout fields (rect, rotation, centre...) that load_frozen_layout
    # merged into every unit; only catalogued devices take their new pose.
    units=deepcopy(graph['units'])
    by_id={r['id']:r for r in rows}
    for uid,unit in units.items():
        for key,value in frozen['graph']['units'].get(uid,{}).items():
            unit.setdefault(key,deepcopy(value))
        if by_id.get(uid,{}).get('catalogued_detail'):
            unit['rotation_deg']=by_id[uid]['rotation_deg']
    frozen['graph']['units']=units
    frozen['graph']['process_connection_changes']=deepcopy(graph.get('process_connection_changes',[]))
    frozen['graph']['sizing']=deepcopy({k:v for k,v in frozen['contracts']['sizing'].items() if k!='graph'})
