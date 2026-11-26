"""Derive utility pipe requirements from the equipment and its connections."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random

from .recipe import InputRejected


P1_ROLES = ('nitrogen', 'offgas', 'water', 'water_return', 'steam', 'steam_return')
GAS_DNS = (25, 32, 40, 50)
AUTHORITY = 'PG v1.6 §§2.1、3.1–3.6'
IMPLEMENTATION_SOURCE = 'P1 main and branch templates for scene modeling'
OD_SOURCE = {
    'url': 'https://tonkinsteel.com/wp-content/uploads/sites/15/2024/05/Tonkin_Steel_Fittings-Data-Charts_April24.pdf',
    'pdf_page': 18,
    'local_evidence': 'Utility header reference figure',
    'note': 'Use the same Tonkin series. DN250 has OD 273.05 mm and DN300 has OD 323.85 mm.'}
SERVICE_IDENTITIES = {'nitrogen': 'nitrogen', 'offgas': 'ordinary_offgas',
    'water': 'cooling_water', 'water_return': 'cooling_water_return',
    'steam': 'utility_steam', 'steam_return': 'utility_steam_condensate'}


class PipingRequirementRejected(InputRejected):
    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__(f'{code}: {detail}')


def _reject(code, detail):
    raise PipingRequirementRejected(code, detail)


def _keys(value, allowed, where):
    if not isinstance(value, dict):
        _reject('PIPING_CONFIG_INVALID', f'{where}must be an object')
    unknown = set(value) - set(allowed)
    if unknown:
        _reject('PIPING_CONFIG_UNKNOWN_FIELD', f'{where}contains unsupported fields: {sorted(unknown)}. Coordinates and orientation are outside the schema.')


def _dn(value, od, where):
    if isinstance(value, bool) or not isinstance(value, int) or str(value) not in od:
        _reject('PIPING_CONFIG_DN', f'{where}must be an integer DN in the physical OD table. Found {value!r}')
    return value


def _defaults():
    from .pipe_spec import p1_services
    return p1_services()


def load_stage_config(config_path=None):
    """Read all runtime DN/colour choices exclusively from pipe_spec.json."""
    from .pipe_spec import load, p1_services, PATH
    spec = load()
    pg_path = Path(__file__).with_name('piping_rules.json')
    pg = json.loads(pg_path.read_text())
    pg['appearance']['color_map'] = {key: {'rgb': list(rgb),
        'label': key, 'basis': spec['authority']} for key, rgb in spec['colors'].items()}
    pg['appearance']['color_map_note'] = 'Runtime colours from pipe_spec.json only'
    raw = {} if config_path is None else json.loads(Path(config_path).read_text())
    if raw:
        _reject('PIPING_SPEC_OVERRIDE_RETIRED',
                'Pipe diameters and colors come only from pipe_spec.json')
    services = p1_services()
    return {'stage': 'P1', 'rule_version': 'v1.6', 'pg_rules': pg,
        'pipe_outer_diameter_mm': deepcopy(spec['pipe_outer_diameter_mm']),
        'templates': {'gas_group_dn_values': deepcopy(spec['services']['nitrogen']['device_candidates']),
                      'services': services},
        'overrides': {},
        'sources': {'pg_rules': str(pg_path.resolve()), 'authority': spec['authority'],
                    'diameter_templates': str(PATH.resolve()), 'added_outer_diameters': OD_SOURCE,
                    'config_path': str(PATH.resolve())},
        'scope': 'One sizing source. Steam uses 150/80/50, cold supply 100/80/50 and off-gas 100. No temporary ports.'}

def _vector(value, where):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        _reject('P1_PORT_POSE_MISSING', f'{where}Missing saved physical 3D vector')
    return deepcopy(list(value))


def _contracts(frozen):
    contracts = frozen.get('contracts')
    if not isinstance(contracts, dict) or 'sizing' not in contracts or 'layout' not in contracts:
        _reject('P1_FROZEN_CONTRACT_MISSING', 'Use the source contracts from load_frozen_layout')
    graph = contracts['sizing'].get('graph', {})
    rows = contracts['layout'].get('equipment', [])
    if not isinstance(graph.get('edges'), list) or not isinstance(graph.get('units'), dict):
        _reject('P1_FROZEN_GRAPH_MISSING', 'sizing.graph lacks original edges or units')
    equipment = {row['id']: row for row in rows}
    if len(equipment) != len(rows):
        _reject('P1_FROZEN_EQUIPMENT_IDS', 'Duplicate saved equipment IDs')
    return contracts, graph, equipment


def port_level(physical, unit):
    return int(physical.get('piping_level',
        physical.get('operation_reference', {}).get('level', unit['level'])))


def required_header_keys(requirements):
    """Same-floor obligations, including relief-only users, without exclusions."""
    return {(p['medium'], p['level']) for p in
            requirements['service_points']+requirements.get('header_demands', [])}


def equipment_source(unit, name):
    """Original process DN and physical nozzle/face, shared by both pipe stages."""
    physical = unit['ports'][name]
    neck = physical.get('nozzle_component_id', f"{unit['id']}_{name}_equipment_nozzle")
    diameter = physical.get('actual_neck_outer_diameter_m')
    row = next((r for r in unit.get('nozzle_schedule', [])
                if r['port'] == name and r.get('connected_to_process')), {})
    source = {'source_dn': physical.get('equipment_neck_dn', physical['dn']),
            'source_radius_m': None if diameter is None else diameter/2,
            'design_line_dn': physical['dn'], 'source_nozzle_id': neck,
            'source_contact_ids': list(row.get('component_ids', [neck])),
            'source_flange': deepcopy(row.get('flange'))}
    if face := physical.get('catalogue_face'):
        source.update(source_catalogue_face=deepcopy(face), source_flange=None,
                      source_nozzle_id=physical['nozzle_component_id'],
                      source_contact_ids=face['departure_contact_ids'])
    return source


def requirements_from_frozen(frozen, config=None):
    """Derive P1 points; retain every non-P1 source edge without executing it."""
    from .rule_identity import equipment_top_port
    from .pipe_spec import load
    current_spec = load()
    cfg = deepcopy(config) if config is not None else load_stage_config()
    _keys(cfg, ('stage', 'rule_version', 'pg_rules', 'pipe_outer_diameter_mm',
                'templates', 'overrides', 'sources', 'scope'), 'normalized piping configuration')
    if cfg.get('stage') != 'P1' or cfg.get('rule_version') not in ('v1.1', 'v1.2', 'v1.3', 'v1.4', 'v1.5', 'v1.6'):
        _reject('PIPING_CONFIG_INVALID', 'config must be the utility configuration returned by load_stage_config')
    contracts, graph, equipment = _contracts(frozen)
    for uid, unit in graph['units'].items():
        if unit.get('kind') == 'vessel' and {'offgas', 'vent'} <= set(unit.get('ports', {})):
            _reject('P1_DUPLICATE_ORDINARY_VENT',
                    f'{uid}Off-gas also serves as venting. Resolve duplicate vent requirements on an authorized copy.')
    rng = random.Random(graph.get('seed', 0))
    points, deferred, owner_gas_dn, missing = [], [], {}, []
    for edge in graph['edges']:
        role = edge.get('role')
        if role not in P1_ROLES:
            deferred.append(deepcopy(edge))
            continue
        matches = []
        for endpoint in (edge['a'], edge['b']):
            uid, port = endpoint.split('.', 1)
            if uid in equipment and port in equipment[uid].get('ports', {}):
                matches.append((uid, port, endpoint))
        if len(matches) != 1:
            _reject('P1_PHYSICAL_ENDPOINT_MISSING' if not matches else 'P1_PHYSICAL_ENDPOINT_AMBIGUOUS',
                    f'{edge["id"]}requires one saved physical equipment endpoint. Found {matches}. Preserve the demand and existing interfaces.')
        uid, port, endpoint = matches[0]
        placed = equipment[uid]; physical = placed['ports'][port]
        if physical.get('catalogue_missing'):
            missing.append({'id':edge['id'],'endpoint':endpoint,'medium':role,'level':placed['level'],
                            'reason':physical['catalogue_missing']})
            continue
        line_dn = physical.get('dn')
        _dn(line_dn, cfg['pipe_outer_diameter_mm'], endpoint+'saved piping DN')
        if graph['units'][uid]['ports'][port].get('dn') != line_dn or edge.get('dn') != line_dn:
            _reject('P1_FROZEN_DN_MISMATCH', f'{edge["id"]}/{endpoint}DN differs between the saved edge, graph and physical port')
        source = equipment_source(placed, port)
        source_dn = source['source_dn']
        _dn(source_dn, cfg['pipe_outer_diameter_mm'], endpoint+'equipment neck DN')
        level = port_level(physical, placed)
        if role == 'nitrogen' or (role == 'offgas' and graph['units'][uid]['kind'] == 'vessel'):
            if uid not in owner_gas_dn:
                owner_gas_dn[uid] = rng.choice(cfg['templates']['gas_group_dn_values'])
            dn = owner_gas_dn[uid]
        elif role == 'offgas':
            dn = current_spec['services']['offgas']['condenser_dn']
        elif role == 'steam':
            dn = current_spec['services']['steam']['device_dn']
        elif role in ('water','water_return'):
            dn = current_spec['services']['cold_source']['device_dn']
        elif role == 'steam_return':
            _reject('P1_OBSOLETE_STEAM_RETURN', 'Steam condensate needs a separate trap to the level-one connection')
        else:
            dn = line_dn
        points.append({'id': edge['id'], 'equipment': uid, 'owner': uid, 'port': port, 'endpoint': endpoint,
            'medium': role, 'service_identity': SERVICE_IDENTITIES[role], 'dn': dn, **source,
            'position': _vector(physical.get('pos'), endpoint+'.pos'),
            'normal': _vector(physical.get('normal'), endpoint+'.normal'), 'equipment_top':equipment_top_port(physical),
            'level': level, 'group_id': None, 'role': role,
            'reducer': {'required': source_dn != dn, 'source_dn': source_dn, 'target_dn': dn,
                        'scope': 'Physical pipe-side reduction demand with the equipment stub and original face preserved'},
            'source_edge': deepcopy(edge), 'source': 'saved sizing.graph.edges and layout.equipment.ports',
            'level_source': 'Use the physical floor for the port or equipment installation floor for legacy interfaces',
            'dn_source': AUTHORITY if role in ('nitrogen', 'offgas', 'steam') else 'Keep the saved utility port DN. Size the main from the template.',
            'separate_vent': role == 'offgas' and port == 'vent'})
    groups = []
    for uid in owner_gas_dn:
        incoming = [p for p in points if p['equipment'] == uid and p['role'] == 'nitrogen' and p['port'] == 'nitrogen']
        outgoing = [p for p in points if p['equipment'] == uid and p['role'] == 'offgas' and p['port'] == 'offgas']
        if not outgoing and graph['units'][uid].get('ordinary_offgas_via_condenser'):
            continue
        if not incoming and not outgoing:
            continue
        if len(incoming) != 1 or len(outgoing) != 1:
            if any(p['endpoint'] in (uid+'.nitrogen',uid+'.offgas') for p in missing):
                continue
            _reject('P1_GAS_GROUP_INCOMPLETE', f'{uid}The head needs its separate original gas inlet and ordinary outlet. A vent cannot substitute.')
        inlet, outlet = incoming[0], outgoing[0]
        group_id = uid+'_head_gas_group'
        inlet['group_id'] = outlet['group_id'] = group_id
        groups.append({'id': group_id, 'owner': uid, 'in_requirement': inlet['id'], 'out_requirement': outlet['id'],
            'source_endpoints': [inlet['endpoint'], outlet['endpoint']], 'dn': inlet['dn'],
            'operating_side': None, 'operating_level': None, 'level': inlet['level'],
            'operating_side_source': 'Select side from the physical equipment port and pipes',
            'scope': 'Use independent manual isolators for nitrogen and off-gas. Keep supply and exhaust separate outside the vessel.'})
    # Automatic relief is a real same-floor header demand, but its two edges
    # are emitted by the safety component rather than a manual P1 root.
    header_demands = []
    for automatic, unit in graph['units'].items():
        if unit.get('kind') != 'safety_valve':
            continue
        endpoint = unit['source_endpoint']; uid, port = endpoint.split('.', 1)
        placed = equipment[uid]; physical = placed['ports'][port]
        edge = next(e for e in graph['edges'] if e['a'] == automatic+'.out')
        level = max(i for i, z in enumerate(frozen['scene']['floors']) if z <= physical['pos'][2])
        if physical.get('catalogue_missing'):
            missing.append({'id': edge['id'], 'endpoint': endpoint, 'medium': 'offgas',
                            'level': level, 'reason': physical['catalogue_missing']})
            continue
        dn = int(current_spec['services']['offgas']['emergency_dn'])
        header_demands.append({'id': edge['id'], 'equipment': uid, 'owner': uid,
            'port': port, 'endpoint': endpoint, 'automatic': automatic, 'header_only': True,
            'medium': 'offgas', 'service_identity': 'emergency_discharge', 'role': 'emergency_discharge',
            'dn': dn, **equipment_source(placed, port), 'position': _vector(physical['pos'], endpoint+'.pos'),
            'normal': _vector(physical['normal'], endpoint+'.normal'), 'level': level,
            'group_id': None, 'equipment_top': equipment_top_port(physical),
            'source_edge': deepcopy(edge), 'source': 'The original relief port adds a header demand only',
            'level_source': 'floor containing the physical relief port'})
    all_points = points+header_demands
    active = [role for role in P1_ROLES if any(p['medium'] == role for p in all_points)]
    selected_templates = {}
    for role in active:
        service_points = [p for p in all_points if p['medium'] == role]
        template = deepcopy(cfg['templates']['services'][role])
        owners = list(dict.fromkeys(p['equipment'] for p in service_points))
        needed_dn = max(p['dn'] for p in service_points)
        shared_water = role in ('water', 'water_return') and len(owners) > 1
        if role == 'nitrogen':
            template['branch_dn'] = needed_dn
            eligible = [dn for dn in current_spec['services']['nitrogen']['main_candidates']
                        if dn > needed_dn]
            template['main_dn_candidates'] = eligible
            template['main_dn'] = rng.choice(eligible)
        if template.get('main_dn') is not None:
            candidates = [template['main_dn']]
        else:
            candidates = template.get('main_dn_candidates', [])
        if template['branch_dn'] < needed_dn or not candidates or any(
                main < template['branch_dn'] or (role != 'offgas' and main == template['branch_dn']) for main in candidates):
            _reject('P1_SERVICE_TEMPLATE_TOO_SMALL', f'{role}The main or branch template cannot fit the connected equipment DN{needed_dn}. Keep the original port and DN.')
        template.update(service=role, service_identity=SERVICE_IDENTITIES[role], shared=shared_water,
            service_equipment=owners, service_requirements=[p['id'] for p in service_points],
            minimum_equipment_dn=needed_dn, source=IMPLEMENTATION_SOURCE,
            candidate_scope='Modeling candidates for real demand. The planner selects within physical space.')
        selected_templates[role] = template
    return {'service_points': points, 'header_demands': header_demands,
        'missing_service_points': missing, 'groups': groups, 'active_services': active,
        'service_templates': selected_templates, 'source_p2_edges': deferred, 'config': cfg,
        'source': {'seed': graph.get('seed'), 'sizing_graph': 'saved contracts.sizing.graph',
                   'physical_ports': 'saved contracts.layout.equipment[*].ports',
                   'source_dir': frozen.get('source_dir'), 'authority': AUTHORITY},
        'scope': 'List P1 demand only. Preserve source_p2_edges, physical interfaces and equipment positions.'}


def detect_port_conflicts(frozen, requirements):
    """Measure original nozzle overlaps, including P1 against non-P1 ports.

    A report only: no port is dropped, moved, merged or repaired. Frozen layout
    sources were already checked by piping_inputs; actual XML is read-only.
    """
    from .evaluation_common import require_environment
    from .building_equipment_checks import EPS
    import mujoco
    require_environment()
    _, _, equipment = _contracts(frozen)
    path = frozen.get('source_evidence', {}).get('model_xml')
    if not path:
        _reject('P1_PORT_XML_MISSING', 'Port clash checks require the same source physical XML')
    model = mujoco.MjModel.from_xml_path(str(path)); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = {p['endpoint']: p for p in requirements['service_points']}
    rows, violations, checked = [], [], set()
    alternative_model = alternative_data = None
    for endpoint, point in points.items():
        uid, port = endpoint.split('.', 1)
        for other_port, other in equipment[uid]['ports'].items():
            other_endpoint = uid+'.'+other_port
            pair = tuple(sorted((endpoint, other_endpoint)))
            if endpoint == other_endpoint or pair in checked:
                continue
            checked.add(pair)
            names = [uid+'_'+name+'_equipment_nozzle' for name in (port, other_port)]
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
            if min(ids) < 0:
                violations.append({'code': 'P1_PORT_NOZZLE_MISSING', 'equipment': uid,
                    'endpoints': list(pair), 'components': names, 'message': 'The physical port lacks a corresponding XML equipment stub'})
                continue
            coincident = math.dist(point['position'], other['pos']) <= EPS
            native = float(mujoco.mj_geomDistance(model, data, ids[0], ids[1], .01, None))
            distance = native
            algorithm = 'MuJoCo actual XML geom distance'
            if abs(native) <= EPS:
                if alternative_model is None:
                    alternative_model = deepcopy(model)
                    alternative_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
                    alternative_data = mujoco.MjData(alternative_model)
                    mujoco.mj_copyData(alternative_data, alternative_model, data)
                distance = min(native, float(mujoco.mj_geomDistance(alternative_model, alternative_data,
                                                                  ids[0], ids[1], .01, None)))
                algorithm += '; native zero rechecked by bundled libccd on private copy'
            overlap = distance < -EPS
            row = {'equipment': uid, 'endpoints': list(pair), 'components': names,
                'p1_requirements': [points[ep]['id'] for ep in pair if ep in points],
                'positions': [deepcopy(equipment[uid]['ports'][ep.split('.', 1)[1]]['pos']) for ep in pair],
                'coincident': coincident, 'overlap': overlap, 'distance_m': distance,
                'native_distance_m': native, 'penetration_m': max(0., -distance),
                'measurement_source': algorithm,
                'actual_geometries': [{'id': name, 'type': int(model.geom_type[index]),
                    'position': data.geom_xpos[index].tolist(), 'size': model.geom_size[index].tolist(),
                    'matrix': data.geom_xmat[index].tolist()} for name, index in zip(names, ids)]}
            if coincident or overlap:
                rows.append(row)
                violations.append({'code': 'P1_ORIGINAL_PORT_CONFLICT', **row,
                    'message': 'Record coincident ports or overlapping stubs explicitly. Preserve distinct ports and obligations.'})
    return {'status': 'fail' if violations else 'pass', 'rows': rows, 'violations': violations,
        'checked_pairs': len(checked), 'conflicting_equipment': sorted({v['equipment'] for v in violations}),
        'source_xml': str(path), 'eps_m': EPS,
        'scope': 'P1 port availability checks include all original port pairs, including pairs on the same equipment'}
