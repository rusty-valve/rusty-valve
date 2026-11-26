"""Load a saved equipment layout and verify that its geometry matches its records."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from .evaluation_common import require_environment
from .recipe import InputRejected


CONTRACT_FILES = ('sizing.json', 'shell.json', 'layout.json', 'connections.json')


class FrozenLayoutRejected(InputRejected):
    """An explicit input refusal, rather than a new layout-search request."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__(f'{code}: {detail}')


def _reject(code, detail):
    raise FrozenLayoutRejected(code, detail)


def _read_json_bytes(raw, path):
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        _reject('FROZEN_JSON_INVALID', f'{path}: {error}')
    if not isinstance(value, dict):
        _reject('FROZEN_JSON_INVALID', f'{path}: top-level JSON must be an object')
    return value


def _read_file(path):
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as error:
        _reject('FROZEN_SOURCE_MISSING', f'{path}: {error}')
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        _reject('FROZEN_SOURCE_CHANGED', f'File changed during reading: {path}')
    return {'path': str(path), 'bytes': raw, 'size_bytes': after.st_size,
            'mtime_ns': after.st_mtime_ns}


def snapshot_inputs(source_dir):
    """Snapshot exactly the four input contracts without writing their directory."""
    require_environment()
    source = Path(source_dir).resolve()
    if not source.is_dir():
        _reject('FROZEN_SOURCE_MISSING', f'Layout directory does not exist: {source}')
    protected = source.parent if (source.parent/'summary.json').is_file() else source
    return {'source_dir': str(source), 'protected_root': str(protected),
            'files': {name: _read_file(source/name) for name in CONTRACT_FILES}}


def assert_frozen_inputs(snapshot):
    """Require original bytes, sizes and modification times to remain unchanged."""
    require_environment()
    records = snapshot.get('files', {})
    if set(records) != set(CONTRACT_FILES):
        _reject('FROZEN_SNAPSHOT_INCOMPLETE', 'Provide snapshots of all four source records')
    for name in CONTRACT_FILES:
        original = records[name]
        current = _read_file(Path(original['path']))
        if any(current[key] != original[key] for key in ('bytes', 'size_bytes', 'mtime_ns')):
            _reject('FROZEN_INPUT_CHANGED', f'Original layout file changed: {original["path"]}')
    return True


def _output_directory(snapshot, output_dir):
    output = Path(output_dir).resolve()
    source = Path(snapshot['source_dir'])
    protected = Path(snapshot['protected_root'])
    if output.is_relative_to(protected) or source.is_relative_to(output):
        _reject('FROZEN_OUTPUT_OVERLAPS_SOURCE', f'Output cannot overlap or overwrite the source layout: {output}')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        _reject('FROZEN_OUTPUT_NOT_EMPTY', f'Output must be a new empty directory: {output}')
    return output


def copy_frozen_inputs(snapshot, output_dir):
    """Copy original contract bytes to a fresh output; never reserialize them."""
    require_environment()
    assert_frozen_inputs(snapshot)
    output = _output_directory(snapshot, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in CONTRACT_FILES:
        path = output/name
        with path.open('xb') as stream:
            stream.write(snapshot['files'][name]['bytes'])
        if path.read_bytes() != snapshot['files'][name]['bytes']:
            _reject('FROZEN_COPY_MISMATCH', f'Failed to copy original bytes: {path}')
        copied[name] = str(path)
    assert_frozen_inputs(snapshot)
    return copied


def discover_layouts(source_dir):
    """Accept one layout or successful rows of its existing layout-stage batch."""
    require_environment()
    source = Path(source_dir).resolve()
    if all((source/name).is_file() for name in CONTRACT_FILES):
        return [source]
    saved = _read_json_bytes(_read_file(source/'summary.json')['bytes'], source/'summary.json')
    if saved.get('stage') != 'layout' or not isinstance(saved.get('rooms'), list):
        _reject('FROZEN_BATCH_STAGE', f'Not a saved layout-stage batch: {source}')
    rooms = []
    for row in saved['rooms']:
        if not isinstance(row, dict):
            _reject('FROZEN_BATCH_INVALID', f'Batch room record is not an object: {source}')
        if row.get('generation') != 'success':
            continue
        name = row.get('room')
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            _reject('FROZEN_BATCH_INVALID', f'Invalid room folder name: {name!r}')
        room = (source/name).resolve()
        if not room.is_relative_to(source) or room in rooms:
            _reject('FROZEN_BATCH_INVALID', f'Room directory is outside the batch or duplicated: {name}')
        rooms.append(room)
    if not rooms:
        _reject('FROZEN_BATCH_NO_SUCCESS', f'The batch contains no successful layout: {source}')
    return rooms


def _contracts(snapshot):
    values = {Path(name).stem: _read_json_bytes(record['bytes'], record['path'])
              for name, record in snapshot['files'].items()}
    for name, stage in (('sizing', 'sizing'), ('shell', 'structure'), ('layout', 'layout')):
        item = values[name]
        if item.get('stage') != stage or item.get('status') != 'pass':
            _reject('FROZEN_CONTRACT_NOT_PASSED', f'{name}.json is not a passing {stage} stage record')
    # connections.json has no status field in the accepted producer contract.
    if values['connections'].get('stage') != 'building_equipment':
        _reject('FROZEN_CONNECTION_STAGE', 'connections.json is not a building and equipment connection contract')
    check = values['layout'].get('checks', {})
    if check.get('status') != 'pass' or check.get('errors') != []:
        _reject('FROZEN_LAYOUT_CHECKS', 'Original layout check failed or its error list is missing')
    return values


def _same(a, b, eps):
    import numpy as np
    try:
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return x.shape == y.shape and bool(np.isfinite(x).all()) and bool(np.isfinite(y).all()) and bool(np.max(np.abs(x-y), initial=0.) <= eps)
    except (TypeError, ValueError):
        return False


def _components_match_xml(scene, model, data, eps):
    """Compare every saved primitive to its compiled, actual XML shape and pose."""
    import mujoco
    import numpy as np
    components = scene.get('components')
    if not isinstance(components, list) or not components:
        _reject('FROZEN_GEOMETRY_MISSING', 'geometry.json lacks physical components')
    names = [c.get('id') for c in components]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        _reject('FROZEN_GEOMETRY_IDS', 'Physical component ID is missing or duplicated')
    actual_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}
    if actual_names != set(names):
        _reject('FROZEN_XML_COMPONENTS', f'XML and geometry inventory differ. Missing: {sorted(set(names)-actual_names)}. Extra: {sorted(str(n) for n in actual_names-set(names))}')
    measured = {}
    for c in components:
        geom = model.geom(c['id']); index = geom.id
        center = data.geom_xpos[index]; matrix = data.geom_xmat[index].reshape(3, 3)
        size = model.geom_size[index]; shape = c.get('shape')
        if shape == 'box':
            lo, hi = np.asarray(c['box'][:3]), np.asarray(c['box'][3:])
            okay = (int(geom.type[0]) == int(mujoco.mjtGeom.mjGEOM_BOX)
                    and _same(center, (lo+hi)/2, eps) and _same(size, (hi-lo)/2, eps)
                    and _same(matrix, np.eye(3), eps))
            bounds = c['box']
        elif shape == 'ellipsoid':
            okay = (int(geom.type[0]) == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
                    and _same(center, c['pos'], eps) and _same(size, c['size'], eps)
                    and _same(matrix, np.eye(3), eps))
            bounds = [*(center-size).tolist(), *(center+size).tolist()]
        elif shape == 'cylinder':
            ends = [center-matrix[:, 2]*size[1], center+matrix[:, 2]*size[1]]
            recorded = c['ends']
            okay = (int(geom.type[0]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
                    and abs(float(size[0])-c['radius']) <= eps
                    and ((_same(ends[0], recorded[0], eps) and _same(ends[1], recorded[1], eps))
                         or (_same(ends[0], recorded[1], eps) and _same(ends[1], recorded[0], eps))))
            extent = size[0]*np.sqrt(np.maximum(0., 1.-matrix[:, 2]**2))
            bounds = [*(np.minimum(*ends)-extent).tolist(), *(np.maximum(*ends)+extent).tolist()]
        else:
            _reject('FROZEN_SHAPE_UNSUPPORTED', f'Cannot identify the source shape: {c["id"]} / {shape}')
        if not okay or not _same(c.get('box'), bounds, eps):
            _reject('FROZEN_XML_GEOMETRY_MISMATCH', f'Original XML and geometry dimensions or positions differ: {c["id"]}')
        annotation = bool(c.get('annotation_only'))
        if (int(geom.contype[0]), int(geom.conaffinity[0])) != ((0, 0) if annotation else (1, 1)):
            _reject('FROZEN_XML_COLLISION_FLAGS', f'Original XML collision flags differ: {c["id"]}')
        measured[c['id']] = {'component': c, 'center': center, 'matrix': matrix, 'size': size}
    return measured


def _merge_equipment(contracts, scene, measured, eps):
    import numpy as np
    sizing, layout, connections = (contracts[k] for k in ('sizing', 'layout', 'connections'))
    graph = deepcopy(sizing.get('graph', {}))
    if not isinstance(graph.get('units'), dict) or not isinstance(graph.get('edges'), list) or not isinstance(graph.get('transfers'), list):
        _reject('FROZEN_PROCESS_MISSING', 'sizing.json lacks the original process graph')
    equipment = layout.get('equipment')
    original_instances = scene.get('equipment_instances')
    if not isinstance(equipment, list) or not isinstance(original_instances, list):
        _reject('FROZEN_EQUIPMENT_MISSING', 'Missing original placement or component ownership inventory')
    instances = {item['id']: item for item in original_instances}
    names = [item.get('id') for item in equipment]
    integral={row['id']:row for row in equipment if row.get('kind')=='jacket'}
    primary_names=set(names)-set(integral)
    expected_integral={uid for uid,u in graph['units'].items() if u.get('kind')=='jacket'}
    if set(integral)!=expected_integral:
        _reject('FROZEN_INTEGRAL_EQUIPMENT_IDS','Jacket demands differ from the actual auxiliary inventory')
    by_id={row['id']:row for row in equipment}
    for uid,row in integral.items():
        owner=graph['units'][uid].get('owner')
        if owner not in instances or row.get('integral_to')!=owner or row.get('owner')!=owner:
            _reject('FROZEN_INTEGRAL_OWNER','Jacket must belong to the placed reactor specified in the source process: '+uid)
        actual_ids={name for name,v in measured.items() if v['component'].get('attached_jacket')==uid}
        if set(row.get('component_ids',[]))!=actual_ids or len(actual_ids)!=len(row.get('ports',{})):
            _reject('FROZEN_INTEGRAL_COMPONENTS','Jacket neck geometry differs from the complete port list: '+uid)
        if not _same(row.get('center'),by_id[owner].get('center'),eps) or row.get('level')!=by_id[owner].get('level'):
            _reject('FROZEN_INTEGRAL_POSE','Jacket center or floor differs from its reactor: '+uid)
    if len(set(names)) != len(names) or len(instances) != len(original_instances) or primary_names != set(instances):
        _reject('FROZEN_EQUIPMENT_IDS', 'Layout and physical equipment IDs differ or are duplicated')
    if primary_names != {item.get('id') for item in connections.get('equipment', [])} or primary_names != {item.get('id') for item in sizing.get('equipment', [])}:
        _reject('FROZEN_CONNECTION_EQUIPMENT', 'connections.json equipment differs from the placement contract')
    port_rows = []
    for row in equipment:
        uid = row['id']; unit = graph['units'].get(uid)
        if not unit or unit.get('kind') != row.get('kind'):
            _reject('FROZEN_EQUIPMENT_KIND', f'Equipment is absent from the original process graph or has a different type: {uid}')
        if uid not in integral and any(row.get(key) != value for key, value in instances[uid].items()):
            _reject('FROZEN_EQUIPMENT_RECORD', f'Source equipment record differs between layout and geometry: {uid}')
        if not isinstance(row.get('ports'), dict) or set(row['ports']) != set(unit.get('ports', {})):
            _reject('FROZEN_PORT_SET', f'Equipment port differs from the original process graph: {uid}')
        if not isinstance(row.get('level'), int) or not 0 <= row['level'] < len(scene['floors']):
            _reject('FROZEN_EQUIPMENT_LEVEL', f'Invalid equipment level: {uid}')
        component_ids = row.get('component_ids', [])
        if not component_ids or any(name not in measured for name in component_ids):
            _reject('FROZEN_EQUIPMENT_COMPONENTS', f'Equipment component lacks its original XML geom: {uid}')
        physical_owner=row['integral_to'] if uid in integral else uid
        if any(measured[name]['component'].get('owner') != physical_owner for name in component_ids):
            _reject('FROZEN_EQUIPMENT_COMPONENTS', f'Equipment component belongs to a different unit: {uid}')
        if uid not in integral and (uid not in measured or not _same(row.get('center'), measured[uid]['center'], eps)):
            # Pump center is the shaft reference and base center; its body
            # cylinder is axially offset within that unchanged assembly.
            bases = [measured[name] for name in component_ids if measured[name]['component']['kind'] == 'plinth']
            center = ([*bases[0]['center'][:2], measured[uid]['center'][2]]
                      if len(bases) == 1 and uid in measured else [])
            if row['kind'] != 'pump' or not _same(row.get('center'), center, eps):
                _reject('FROZEN_EQUIPMENT_CENTER', f'Saved equipment center differs from the physical source: {uid}')
        angle = row.get('rotation_deg',by_id[physical_owner].get('rotation_deg'))
        if not isinstance(angle, (int, float)) or not math.isfinite(angle):
            _reject('FROZEN_EQUIPMENT_ROTATION', f'Missing saved equipment orientation: {uid}')
        if row.get('form') == 'horizontal':
            axis = measured[uid]['matrix'][:, 2]
            expected = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.])
            if abs(abs(float(axis@expected))-1.) > eps:
                _reject('FROZEN_EQUIPMENT_ROTATION', f'Saved axis differs from the source XML: {uid}')
        for name, port in row['ports'].items():
            if port.get('dn') != unit['ports'][name].get('dn'):
                _reject('FROZEN_PORT_DN', f'Saved port DN differs from the equipment selection graph: {uid}.{name}')
            if not _same(port.get('pos'), port.get('pos'), eps) or not _same(port.get('normal'), port.get('normal'), eps):
                _reject('FROZEN_PORT_POSE', f'Invalid port position or direction: {uid}.{name}')
            if len(port['pos']) != 3 or len(port['normal']) != 3 or abs(float(np.linalg.norm(port['normal']))-1.) > eps:
                _reject('FROZEN_PORT_POSE', f'Port position or direction is not a valid 3D vector: {uid}.{name}')
            nozzle = measured.get(uid+'_'+name+'_equipment_nozzle')
            if nozzle:
                center, axis, half = nozzle['center'], nozzle['matrix'][:, 2], nozzle['size'][1]
                ends = [center-axis*half, center+axis*half]
                match = next((i for i in (0, 1) if _same(port['pos'], ends[i], eps)), None)
                if match is None or not _same(port['normal'], (ends[match]-ends[1-match])/(2*half), eps):
                    _reject('FROZEN_PORT_XML_MISMATCH', f'Source XML stub endpoint or direction differs from the saved port: {uid}.{name}')
            elif row['kind'] != 'pump':
                _reject('FROZEN_PORT_XML_MISSING', f'No original equipment stub for the saved port: {uid}.{name}')
            port_rows.append({'equipment': uid, 'port': name, 'position': deepcopy(port['pos']),
                              'xml_nozzle': uid+'_'+name+'_equipment_nozzle' if nozzle else None,
                              'evidence': 'actual_xml_nozzle_end' if nozzle else 'frozen_pump_interface_on_matched_assembly'})
        unit.update(deepcopy(row))
    original_transfers = {item['id']: item for item in graph['transfers']}
    if set(original_transfers) != {item.get('id') for item in connections.get('transfers', [])}:
        _reject('FROZEN_TRANSFERS', 'Original transfer IDs differ')
    for item in connections['transfers']:
        if any(original_transfers[item['id']].get(key) != value for key, value in item.items()):
            _reject('FROZEN_TRANSFERS', f'Original transfer connection changed: {item["id"]}')
    graph['valves'] = []  # Logical P2 edges/transfers remain unchanged.
    graph['sizing'] = deepcopy({key: value for key, value in sizing.items() if key != 'graph'})
    return graph, port_rows


def _load_frozen_layout(source_dir):
    """Load one passed layout without moving or rebuilding any physical component."""
    require_environment()
    import mujoco
    from .building_equipment_checks import EPS
    snapshot = snapshot_inputs(source_dir)
    source = Path(snapshot['source_dir']); contracts = _contracts(snapshot)
    geometry_file = _read_file(source/'geometry.json')
    xml_file = _read_file(source/'room.xml')
    scene = _read_json_bytes(geometry_file['bytes'], source/'geometry.json')
    shell = contracts['shell'].get('shell')
    if scene.get('stage') != 'layout' or scene.get('routing') != []:
        _reject('FROZEN_GEOMETRY_STAGE', 'Source physical model is not a pre-piping layout')
    if not isinstance(shell, dict) or scene.get('shell') != shell or scene.get('floors') != shell.get('floors') or scene.get('footprint') != shell.get('footprint_m'):
        _reject('FROZEN_SHELL_MISMATCH', 'Source building differs from the floor or shell contract')
    if scene.get('operation_reservations') != contracts['layout'].get('operation_reservations'):
        _reject('FROZEN_OPERATION_RESERVATIONS', 'Equipment operation reservations differ from the placement contract')
    try:
        model = mujoco.MjModel.from_xml_path(str(source/'room.xml'))
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    except Exception as error:
        _reject('FROZEN_XML_LOAD', f'Source XML load failed: {error}')
    measured = _components_match_xml(scene, model, data, EPS)
    graph, ports = _merge_equipment(contracts, scene, measured, EPS)
    assert_frozen_inputs(snapshot)
    for original in (geometry_file, xml_file):
        current = _read_file(Path(original['path']))
        if any(current[key] != original[key] for key in ('bytes', 'size_bytes', 'mtime_ns')):
            _reject('FROZEN_EVIDENCE_CHANGED', f'Original physical evidence changed during reading: {original["path"]}')
    return {'graph': graph, 'scene': deepcopy(scene), 'shell': deepcopy(shell),
            'contracts': deepcopy(contracts), 'source_dir': str(source), 'frozen_inputs': snapshot,
            'source_evidence': {'status': 'pass', 'geometry_json': str(source/'geometry.json'),
                'model_xml': str(source/'room.xml'), 'component_count': len(measured),
                'equipment_count': len(contracts['layout']['equipment']), 'ports': ports,
                'files': {Path(record['path']).name: {key: record[key] for key in
                    ('path', 'size_bytes', 'mtime_ns')} for record in (geometry_file, xml_file)},
                'method': 'Compare every original component and physical XML dimension, position, collision flag and nozzle endpoint'}}


def load_frozen_layout(source_dir):
    """Return independent in-memory graph/scene copies of one frozen source."""
    require_environment()
    try:
        return _load_frozen_layout(source_dir)
    except FrozenLayoutRejected:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        _reject('FROZEN_CONTRACT_INVALID', f'Saved layout records are missing or invalid: {error}')


def main():
    import argparse
    environment = require_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New empty directory with byte copies of the four contracts and a read validation report')
    args = parser.parse_args()
    rooms = discover_layouts(args.source)
    if args.output:
        # Check the complete source batch before any output is created.
        for room in rooms:
            _output_directory(snapshot_inputs(room), args.output)
        args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'pass', 'environment': environment, 'source': str(args.source.resolve()),
              'scope': 'Check original layout consistency and exact copies of the four records', 'rooms': []}
    for room in rooms:
        try:
            result = load_frozen_layout(room)
            row = {'room': room.name, **result['source_evidence']}
            snapshot = result['frozen_inputs']
            row['frozen_input_metadata'] = {name: {key: record[key] for key in
                ('path', 'size_bytes', 'mtime_ns')} for name, record in snapshot['files'].items()}
            if args.output:
                row['copied_contracts'] = copy_frozen_inputs(snapshot, args.output/room.name)
            try:
                _output_directory(snapshot, room)
            except FrozenLayoutRejected as error:
                row['source_output_overlap_rejected'] = error.code == 'FROZEN_OUTPUT_OVERLAPS_SOURCE'
            else:
                row['source_output_overlap_rejected'] = False
            assert_frozen_inputs(snapshot)
            row['original_four_files_unchanged'] = True
        except FrozenLayoutRejected as error:
            row = {'room': room.name, 'status': 'rejected', 'code': error.code, 'reason': str(error)}
            report['status'] = 'fail'
        report['rooms'].append(row)
    report['total'] = len(report['rooms'])
    report['passed'] = sum(row['status'] == 'pass' and row.get('source_output_overlap_rejected', False) for row in report['rooms'])
    if report['passed'] != report['total']:
        report['status'] = 'fail'
    if args.output:
        (args.output/'read_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
