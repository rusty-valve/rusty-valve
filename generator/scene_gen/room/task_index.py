"""Record the installed valves, their operating positions and directions of travel."""
from collections import defaultdict
import json

from .piping_operations import _mapping


def records(path, key):
    return json.loads(path.read_text(encoding='utf-8'))[key]


def joint_pose(data, prefix, name, joint):
    travel = joint['closed'] - joint['open']
    return {
        'joint': prefix + name,
        'pivot_world_m': data.xanchor[joint['id']].tolist(),
        'positive_axis_world': data.xaxis[joint['id']].tolist(),
        'open_qpos_rad': joint['open'],
        'closed_qpos_rad': joint['closed'],
        'rotate_to_close_rad': travel,
        'rotate_to_open_rad': -travel,
    }


def build_task_index(room, source, site, attached, model, data):
    placed = {row['id']: row for row in records(source / 'valve_positioning.json', 'rows')
              if row['status'] == 'placed'}
    graph = {valve['id']: valve for valve in records(source / 'process.json', 'valves')}
    access = {row['valve']: row for row in site['valve_access']}
    routes = defaultdict(set)
    for route in records(source / 'piping.json', 'routes'):
        for valve_id in route['valves']:
            routes[valve_id].add(route['id'])

    if not set(placed) == set(graph) == set(access) == set(routes):
        raise ValueError(f'{room}: installed valve IDs differ across saved records')
    expected = {valve['prefix']: (valve['root_body'],
                f"valves/{valve['package_id']}/{valve['version']}/model.xml") for valve in graph.values()}
    if len(expected) != len(graph) or attached != expected:
        raise ValueError(f'{room}: valve attachment IDs or versions differ from saved installation records')

    tasks = []
    for valve_id, positioned in sorted(placed.items()):
        valve, stance = graph[valve_id], access[valve_id]
        if (positioned['dn'] != valve['dn'] or positioned['operator'] != valve['operator']
                or positioned['level'] != stance['level']):
            raise ValueError(f'{room}/{valve_id}: valve attributes differ across saved records')
        _, prefix, _, joints, _, _ = _mapping(valve, model)
        operator_joint = valve['motion']['operator_joint']
        valve_joint = valve['motion']['valve_joint']
        tasks.append({
            'id': valve_id,
            'source_valve_id': positioned.get('source_valve'),
            'model': valve['package_id'],
            'version': valve['version'],
            'package_model': expected[prefix][1],
            'dn': valve['dn'],
            'operator': valve['operator'],
            'body': prefix + valve['root_body'],
            'pipe_routes': sorted(routes[valve_id]),
            'floor_level': stance['level'],
            'floor_z_m': valve['operating_floor_z_m'],
            'operator_point_world_m': positioned['operation_m'],
            'operator_motion': joint_pose(data, prefix, operator_joint, joints[operator_joint]),
            'valve_stem_motion': joint_pose(data, prefix, valve_joint, joints[valve_joint]),
            'access': {
                'can_stand': stance['can_stand'],
                'closest_stance_world_m': stance['closest_stance_m'],
                'reachable_on_foot': stance['stair_access'] or stance.get('entry_access', False),
                'stair_access': stance['stair_access'],
                'entry_access': stance.get('entry_access'),
                'high_valve': stance['high'],
            },
        })
    return {
        'schema_version': 1,
        'room_id': room,
        'room_model': 'room.xml',
        'access_basis': 'geometric_check',
        'access_source': 'site.json valve_access',
        'motion_basis': 'joint states and world axes from the compiled room',
        'rotation_convention': 'Positive is a right-hand rotation about positive_axis_world.',
        'valves': tasks,
    }
