"""Check the Python environment and summarize valve access results."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ENV = ROOT / 'valve_library' / '.venv'
EXPECTED_MUJOCO = '3.16.0'
PORTABLE_GENERATION = ContextVar('portable_generation', default=False)


@contextmanager
def portable_generation():
    token = PORTABLE_GENERATION.set(True)
    try:
        yield
    finally:
        PORTABLE_GENERATION.reset(token)


def require_environment():
    """Stop before generation when either the interpreter or MuJoCo differs."""
    if not PORTABLE_GENERATION.get() and Path(sys.prefix).resolve() != EXPECTED_ENV.resolve():
        raise SystemExit(f'Unexpected Python environment. Use {EXPECTED_ENV}/bin/python -B. Current interpreter: {sys.executable}')
    import mujoco
    if mujoco.__version__ != EXPECTED_MUJOCO:
        raise SystemExit(f'Unexpected MuJoCo version. Required: {EXPECTED_MUJOCO}; installed: {mujoco.__version__}')
    return {'interpreter': sys.executable, 'prefix': sys.prefix,
            'mujoco': mujoco.__version__, 'environment_read_only': True}


ROLE_GROUPS = ('vessel root', 'pump suction and discharge', 'recirculation', 'bypass', 'utilities', 'other process equipment')


def role_group(valve, graph):
    edge = next(e for e in graph['edges'] if e['id'] == valve['edge'])
    at = valve['at']
    port = at.split('.', 1)[1] if '.' in at else ''
    unit = graph['units'].get(at.split('.')[0], {})
    if edge['role'] == 'recycle' or port.startswith('recycle_'):
        return 'recirculation'
    if edge['role'] == 'bypass':
        return 'bypass'
    if unit.get('kind') == 'pump' and port in ('in', 'out'):
        return 'pump suction and discharge'
    if unit.get('kind') in ('header', 'jacket') or port.startswith('utility_') or 'utility_tier' in edge:
        return 'utilities'
    if unit.get('kind') == 'vessel':
        return 'vessel root'
    return 'other process equipment'


def classify_valves(graph, scene, report, room_pass):
    """Keep native decisions; annotate only the existing §8 pending category."""
    from .process import RULES
    floor_heights = scene.get('floors', [0])
    rows = []
    for valve in graph['valves']:
        at = valve['at']
        unit = graph['units'].get(at.split('.')[0], {})
        box = unit.get('box')
        point = valve.get('grip', valve.get('center'))
        ground = floor_heights[valve['level']]
        access_failed = not valve.get('stance_free', False) or not valve.get('reachable_from_passage', False)
        pending = bool(at.endswith('.bottom') and unit.get('kind') == 'vessel' and box and point
                       and box[0] <= point[0] <= box[3] and box[1] <= point[1] <= box[4]
                       and point[2] < box[2]
                       and box[2] - ground < RULES['geometry_screen']['walk_height_m']
                       and access_failed)
        usable = bool(room_pass and valve['eligible'] and not pending)
        rows.append({'id': valve['id'], 'name': valve['name'], 'role': valve['role'],
                     'role_group': role_group(valve, graph), 'at': at,
                     'native_eligible': bool(valve['eligible']), 'usable': usable,
                     'low_head_pending': pending,
                     'layout_failure': bool(room_pass and not usable and not pending),
                     'reason': valve['reason'],
                     'low_head_evidence': {'body_underside_m': box[2], 'walk_level_m': ground,
                         'clearance_m': box[2] - ground,
                         'required_walk_height_m': RULES['geometry_screen']['walk_height_m'],
                         'point_m': point} if pending else None})
    return rows
