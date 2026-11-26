"""Package a generated room with its meshes, valves and operating instructions."""
from __future__ import annotations

from collections import Counter
import json
import multiprocessing
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import threading
import time
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from . import batch_memory
from .task_index import build_task_index
from .runtime_paths import RUNTIME


VALVE_ROOT = RUNTIME / 'valve_library' / 'packages'
COUNTERS = ('connections', 'breakpoints', 'penetrations', 'collision_breaks',
            'valves', 'pending_valves', 'discarded_close_valves', 'insulation_pieces')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8')


def xml_files(root):
    # These saved MJCFs use file paths relative to the XML containing them.
    for compiler in root.iter('compiler'):
        if compiler.attrib.keys() & {'assetdir', 'meshdir', 'texturedir', 'strippath'}:
            raise ValueError('Unsupported compiler path settings in the frozen MJCF')
    return (node for node in root.iter() if 'file' in node.attrib)


def apply_room_paths(source, destination):
    tree = ET.parse(source)
    root = tree.getroot()
    attached_models = {node.attrib['model'] for node in root.iter('attach')}
    for assets in root.findall('asset'):
        for node in assets.findall('model'):
            if node.attrib['name'] not in attached_models:
                assets.remove(node)
    target_meshes = destination / 'meshes'
    target_meshes.mkdir()
    mesh_sources = {}
    packages = {}
    resolved_paths = {}
    for node in xml_files(root):
        path = node.attrib['file']
        if path not in resolved_paths:
            resolved_paths[path] = (source.parent / path).resolve(strict=True)
        original = resolved_paths[path]
        if node.tag == 'mesh':
            copied = target_meshes / original.name
            if copied not in mesh_sources:
                shutil.copyfile(original, copied)
                mesh_sources[copied] = original
            elif mesh_sources[copied] != original:
                raise ValueError(f'Mesh filenames collide: {mesh_sources[copied]} and {original}')
        elif node.tag == 'model':
            key = original.parent.relative_to(VALVE_ROOT.resolve()).parts
            if len(key) != 2 or original.name != 'model.xml':
                raise ValueError(f'Unversioned valve model: {original}')
            if key not in packages:
                package = destination / 'valves' / Path(*key)
                shutil.copytree(original.parent, package, copy_function=shutil.copyfile)
                packages[key] = original.parent
            copied = destination / 'valves' / Path(*key) / original.name
        else:
            raise ValueError(f'Unexpected room MJCF file reference: <{node.tag}>')
        node.set('file', copied.relative_to(destination).as_posix())

    tree.write(destination / 'room.xml', encoding='utf-8', xml_declaration=True)
    return root, mesh_sources, packages


def check_local_references(folder):
    folder = folder.resolve()
    if any(path.is_symlink() for path in folder.rglob('*')):
        raise ValueError(f'Package contains a symbolic link: {folder}')
    for xml in folder.rglob('*.xml'):
        for node in xml_files(ET.parse(xml).getroot()):
            relative = Path(node.attrib['file'])
            target = (xml.parent / relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(xml.parent) or not target.is_file():
                raise ValueError(f'{xml}: file reference is missing or leaves its model directory: {relative}')


def attachments(root):
    assets = {node.attrib['name']: node.attrib['file'] for node in root.iter('model')}
    instances = {}
    for attach in root.iter('attach'):
        prefix = attach.attrib['prefix']
        if prefix in instances or not prefix.endswith('_'):
            raise ValueError(f'Unexpected attachment prefix: {prefix}')
        instances[prefix] = (attach.attrib['body'], assets[attach.attrib['model']])
    return instances


BUILD_STEP_NAMES = {
    'full': 'equipment and pipes',
    'valves': 'place valves',
    'prune-valves': 'check valve spacing',
    'insulation': 'add insulation',
    'site': 'build the floor, platforms and access',
}
INPUT_FIELDS = ('recipe', 'shell', 'seed', 'site', 'equipment_reference')


def split_room_inputs(record):
    recipe = {field: record[field] for field in INPUT_FIELDS if field in record}
    process = recipe['recipe'].copy()
    if 'description' in process:
        process['description'] = re.sub(
            r' (?:The recipe contains no coordinates|No coordinates)\.$', '', process['description'])
    recipe['recipe'] = process

    version = record['scene_gen']
    expected = record['expected']
    build = {
        'room_id': record['room_id'],
        'generator_version': version['tag'].removeprefix('v'),
        'commit': version['commit'],
        'building': record['resolved_shell'],
        'pipe_routing': 'Shared headers with end connections'
            if record.get('piping_method') == 'end-bundle' else record.get('piping_method'),
        'build_stage': BUILD_STEP_NAMES[record['stage']],
        'build_steps': [BUILD_STEP_NAMES[step] for step in record['stages']],
        'pipes_connected': expected['connections']['connected'],
        'pipes_requested': expected['connections']['expected'],
        'unfinished_pipe_ends': expected['breakpoints'],
        'solid_intersections': expected['penetrations'],
        'pipe_sections_removed': expected['collision_breaks'],
        'valves_installed': expected['valves']['installed'],
        'valves_requested': expected['valves']['expected'],
        'valves_waiting_for_placement': expected['pending_valves'],
        'valves_removed_for_spacing': expected['discarded_close_valves'],
        'insulation_pieces': expected['insulation_pieces'],
    }
    if update := record.get('redrawn_equipment_export'):
        build['equipment_model_update'] = {'commit': update['commit'],
                                           'description': 'Equipment models replaced in place'}
    if record.get('equipment_geometry'):
        build['equipment_modeling'] = 'Dimension-based equipment models, revision 1'
    return recipe, build


def write_room_inputs(folder, record):
    recipe, build = split_room_inputs(record)
    write_json(folder / 'recipe.json', recipe)
    write_json(folder / 'build_info.json', build)


def write_recipe(source, destination, room, site, *, scene_version, equipment_reference=None):
    from .equipment_models import MARK

    input_record = read_json(source / 'input.json')
    geometry = read_json(source / 'model' / 'geometry.json')
    shell = geometry['shell']
    input_record.update({
        'room_id': room,
        'scene_gen': scene_version,
        'equipment_geometry': MARK,
        'resolved_shell': {key: shell[key] for key in input_record['shell']},
        'stages': ['full', 'valves', 'prune-valves', 'insulation', 'site'],
        'site': {'bund_height_mm': round(site['ground']['bund_height_m'] * 1000)},
        'expected': {key: site['baseline'][key] for key in COUNTERS},
    })
    reference = equipment_reference
    if reference is not None:
        layout = read_json(reference / 'layout.json')
        reference_dir = destination / 'recipe_reference'
        reference_dir.mkdir()
        # Retain the fields the generator reads when it locks reference poses.
        equipment = [{key: row[key] for key in ('id', 'kind', 'level', 'center')}
                     for row in layout['equipment']]
        for row, saved in zip(equipment, layout['equipment']):
            if 'rotation_deg' in saved:
                row['rotation_deg'] = saved['rotation_deg']
            spec = saved.get('dimension_spec', {})
            row['dimension_spec'] = {key: spec[key] for key in ('catalog_key', 'leg_clearance_m') if key in spec}
        write_json(reference_dir / 'layout.json', {'status': layout['status'], 'equipment': equipment})
        saved_shell = read_json(reference / 'shell.json')
        trials = [{key: value for key, value in trial.items() if key not in ('inherited_from', 'executed_this_run')}
                  for trial in saved_shell.get('trials', ())]
        write_json(reference_dir / 'shell.json', {'shell': saved_shell['shell'], 'trials': trials})
        input_record['equipment_reference'] = reference_dir.name
    write_room_inputs(destination, input_record)
    return reference


def load_model(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return model, data


def model_snapshot(model, data):
    return {
        'geom_names': [model.geom(i).name for i in range(model.ngeom)],
        'joint_names': [model.joint(i).name for i in range(model.njnt)],
        'joint_types': model.jnt_type.copy(),
        'geom_positions': data.geom_xpos.copy(),
        'geom_orientations': data.geom_xmat.copy(),
        'geom_types': model.geom_type.copy(),
        'geom_sizes': model.geom_size.copy(),
        'joint_axes': data.xaxis.copy(),
        'joint_anchors': data.xanchor.copy(),
        'joint_ranges': model.jnt_range.copy(),
        'joint_couplings': model.eq_data.copy(),
    }


def watch_memory(stop):
    while not stop.wait(batch_memory.POLL_SECONDS):
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin':
            peak_rss *= 1024
        if max(batch_memory.process_memory(os.getpid(), peak_rss)) > batch_memory.ROOM_BYTES:
            print('Room export exceeded 2 GB. Intermediate files were retained on the output volume.', file=sys.stderr, flush=True)
            os._exit(1)


def wait_for_memory(room):
    deadline = time.monotonic() + 120
    while True:
        memory = batch_memory.system_memory()
        if memory['pressure_normal'] and memory['available_bytes'] >= batch_memory.AVAILABLE_BYTES:
            return
        if time.monotonic() >= deadline:
            raise ValueError(f'{room}: Memory pressure or available memory has not recovered')
        time.sleep(1)


def compile_worker(connection, path, task_context):
    """Release MuJoCo's native allocations before the other version is loaded."""
    stop = threading.Event()
    monitor = threading.Thread(target=watch_memory, args=(stop,), daemon=True)
    monitor.start()
    try:
        with connection:
            model, data = load_model(path)
            tasks = build_task_index(*task_context, model, data) if task_context else None
            connection.send((model_snapshot(model, data), tasks))
    finally:
        stop.set()
        monitor.join()


def compiled_material(path, task_context=None):
    wait_for_memory(path.parent.name)
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=compile_worker, args=(sender, path, task_context))
    with receiver, sender:
        worker.start()
        sender.close()
        try:
            result = receiver.recv()
        except EOFError as error:
            raise ValueError(f'MuJoCo validation process stopped: {path}') from error
        finally:
            receiver.close()
            worker.join()
            worker.close()
    return result


def verify_model(source, destination, task_context):
    original, _ = compiled_material(source)
    packaged, tasks = compiled_material(destination, task_context)
    for field in original:
        if not np.array_equal(original[field], packaged[field]):
            raise ValueError(f'{destination.parent.name}: compiled {field} differs from the original room')
    return packaged, tasks


def write_room_runner(destination):
    folder = Path(__file__).parent
    for filename in ('run_room.py', 'ROOM_SIMULATION.md'):
        shutil.copyfile(folder / filename, destination / filename)


def export_room(row, destination, *, scene_version, equipment_reference=None):
    source = Path(row['folder'])
    source_model = source / 'model'
    xml = source_model / 'room.xml'
    destination.mkdir()
    root, mesh_sources, packages = apply_room_paths(xml, destination)
    site = read_json(source_model / 'site.json')
    reference = write_recipe(source, destination, row['id'], site, scene_version=scene_version,
                             equipment_reference=equipment_reference)
    check_local_references(destination)
    task_context = (row['id'], source_model, site, attachments(root))
    model, task_index = verify_model(xml, destination / 'room.xml', task_context)
    write_json(destination / 'task_index.json', task_index)
    write_room_runner(destination)

    return {
        'room': row['id'], 'geoms': len(model['geom_names']), 'joints': len(model['joint_names']),
        'installed_valves': len(task_index['valves']),
        'mesh_files': len(mesh_sources), 'valve_packages': len(packages),
        'exclusive_bytes': sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
        'room_xml_source': str(xml), 'recipe_source': str(source / 'input.json'),
        'recipe_reference_source': str(reference) if reference else None,
        'mesh_origin_directory_counts': dict(Counter(str(path.parent) for path in mesh_sources.values())),
        'valve_package_sources': {'/'.join(key): str(path) for key, path in packages.items()},
        **{key: site['baseline'][key] for key in COUNTERS},
    }


PUBLIC_ID = re.compile(r'V[0-9]{3}')


def name_valves(tasks):
    """Reserve existing unique V-numbers, then number remaining source valves."""
    sources = Counter(task.get('source_valve_id') for task in tasks)
    reserved = {name for name in sources if isinstance(name, str) and PUBLIC_ID.fullmatch(name)}
    next_number = max((int(name[1:]) for name in reserved), default=0) + 1
    names = {}
    for task in sorted(tasks, key=lambda row: (row.get('source_valve_id') or '', row['id'])):
        original = task['id']
        source = task.get('source_valve_id')
        if source in reserved and sources[source] == 1:
            names[original] = source
        else:
            if next_number > 999:
                raise ValueError('A room cannot have more than 999 public valve numbers')
            names[original] = f'V{next_number:03d}'
            next_number += 1
    if len(set(names.values())) != len(tasks):
        raise ValueError('Valve identifiers must be unique within the room')
    return names


def remap_name(name, names):
    for old, new in names.items():
        if name and name.startswith(old + '_'):
            return new + name[len(old):]
    return name


def apply_public_valve_names(room, room_id):
    index_path = room / 'task_index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    tasks = index['valves']
    names = name_valves(tasks)
    path = room / 'room.xml'
    tree = ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    attachments = list(root.iter('attach'))
    if len(attachments) != len(names) or {row.get('prefix') for row in attachments} != {
            name + '_' for name in names}:
        raise ValueError(f'{room_id}: attachments do not match the valve task index')
    root.set('model', room_id)
    for row in attachments:
        row.set('prefix', names[row.attrib['prefix'][:-1]] + '_')
    # External MJCF relationships may name a body, joint, site or geom of an attachment.
    for row in root.iter():
        for field, value in row.attrib.items():
            if re.fullmatch(r'(?:body|joint|site|geom|objname|refname|tendon|actuator)[12]?', field):
                row.set(field, remap_name(value, names))
    index['room_id'] = room_id
    index['access_source'] = 'site.json valve_access'
    index['motion_basis'] = 'Installed valve records and world axes from the compiled room'
    for task in tasks:
        old = task['id']
        task['id'] = names[old]
        task['body'] = remap_name(task['body'], names)
        for motion in (task['operator_motion'], task['valve_stem_motion']):
            motion['joint'] = remap_name(motion['joint'], names)
    tasks.sort(key=lambda task: task['id'])
    tree.write(path, encoding='utf-8', xml_declaration=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return names
