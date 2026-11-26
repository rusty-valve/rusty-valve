"""Add guardrails to open platform and stair edges."""
import math

import numpy as np

from .checks import geom_bounds
from .layout import add_box, add_pipe
from .site_grating import EPS
from .site_stairs import collisions, flights


HEIGHT = 1.05
TOE_HEIGHT = .10
RAIL_RADIUS = .022
TOE_WIDTH = .026
COLOR = (.94, .70, .12)


def beam_box(scene, name, a, b, width, vertical, kind):
    start, end = np.asarray(a), np.asarray(b)
    axis = end-start
    run = float(np.linalg.norm(axis))
    x = axis/run
    y = np.cross([0., 0., 1.], x)
    level = float(np.linalg.norm(y))
    y /= level
    z = np.cross(x, y)
    size = np.asarray([run/2, width/2, vertical*level/2])
    middle = (start+end)/2
    extent = np.abs(np.column_stack([x, y, z]))@size
    part = add_box(scene, name, np.r_[middle-extent, middle+extent], kind, COLOR)
    part['pose'] = {'pos': middle.tolist(), 'size': size.tolist(), 'xyaxes': np.r_[x, y].tolist()}


def guard(scene, name, a, b):
    length = math.dist(a[:2], b[:2])
    if length <= .04:
        return None
    vertical_radius = RAIL_RADIUS*math.hypot(length, b[2]-a[2])/length
    for suffix, height in (('top', HEIGHT), ('mid', HEIGHT/2)):
        add_pipe(scene, f'{name}_{suffix}', [*a[:2], a[2]+height-vertical_radius],
                 [*b[:2], b[2]+height-vertical_radius], RAIL_RADIUS, 'rail', COLOR)
    beam_box(scene, name+'_toe', [*a[:2], a[2]+TOE_HEIGHT/2], [*b[:2], b[2]+TOE_HEIGHT/2],
             TOE_WIDTH, TOE_HEIGHT, 'toe_board')
    for number, base in enumerate(np.linspace(a, b, max(2, math.ceil(length/1.15)+1))):
        add_pipe(scene, f'{name}_post_{number}', base, base+[0., 0., HEIGHT-RAIL_RADIUS], .018, 'rail', COLOR)
    return {'id': name, 'a': list(a), 'b': list(b), 'length_m': length}


def edge_parts(start, stop, openings):
    pieces = [(start, stop)]
    for low, high in openings:
        outside = []
        for a, b in pieces:
            if low >= b or high <= a:
                outside.append((a, b))
            else:
                outside.extend(((a, min(b, low)), (max(a, high), b)))
        pieces = [(a, b) for a, b in outside if b-a > .04]
    return pieces


def perimeter(scene, prefix, rect, level, openings=None):
    x0, y0, x1, y1 = rect
    for edge, start, stop, fixed, axis in (('front', x0, x1, y0, 0), ('rear', x0, x1, y1, 0),
                                            ('left', y0, y1, x0, 1), ('right', y0, y1, x1, 1)):
        for i, (a, b) in enumerate(edge_parts(start, stop, (openings or {}).get(edge, ()))):
            start_xyz, end_xyz = ([a, fixed, level], [b, fixed, level]) if axis == 0 else (
                [fixed, a, level], [fixed, b, level])
            yield guard(scene, f'{prefix}_{edge}_{i}', start_xyz, end_xyz)


def layout(scene, floors):
    records, by_level = [], {}
    width, depth = scene['footprint']
    parts = {part['id']: part for part in scene['components']}
    for row in floors:
        level, z = row['level'], row['z_m']
        landing = parts[f'top_landing_{level-1}']['box']
        records.extend(perimeter(scene, f'site_rail_deck_{level}', [0., 0., width, depth], z,
                                 {'front': [(landing[0], landing[3])]}))
        for index, opening in enumerate(o for o in row['openings'] if o['kind'] == 'pullout'):
            records.extend(perimeter(scene, f'site_rail_pullout_{level}_{index}', opening['rect'], z))
    for flight, treads in flights(scene):
        level, number = flight['level'], flight['flight']
        first, last = treads[0]['box'], treads[-1]['box']
        descending_x = first[0] > last[0]
        start, end = (first[3], last[0]) if descending_x else (first[0], last[3])
        by_level.setdefault(level, {})[number] = (first[1], first[4])
        for side, y in enumerate((first[1], first[4])):
            records.append(guard(scene, f'site_stair_{level}_{number}_side_{side}',
                                 [start, y, first[5]-flight['rise']], [end, y, last[5]]))
    for level in sorted(by_level):
        near, far = by_level[level][0], by_level[level][1]
        turn = parts[f'turn_landing_{level}']['box']
        top = parts[f'top_landing_{level}']['box']
        records.extend(perimeter(scene, f'site_stair_turn_{level}', turn[:2]+turn[3:5], turn[5],
                                 {'right': [near, far]}))
        left_openings = [far] + ([by_level[level+1][0]] if level+1 in by_level else [])
        records.extend(perimeter(scene, f'site_stair_top_{level}', top[:2]+top[3:5], top[5],
                                 {'left': left_openings, 'rear': [(top[0], top[3])]}))
    return [row for row in records if row is not None]


def audit(scene, model, data, rows):
    violations = []
    for row in rows:
        a, b = row['a'], row['b']
        top, middle, toe = [model.geom(row['id']+suffix).id for suffix in ('_top', '_mid', '_toe')]
        box = geom_bounds(model, data, [top, middle, toe])
        if abs(a[2]-b[2]) <= EPS:
            top_height, mid_height, toe_height = box[:, 5]-a[2]
            if not np.allclose([top_height, mid_height, toe_height], [HEIGHT, HEIGHT/2, TOE_HEIGHT], atol=EPS):
                violations.append({'code': 'GUARD_DIMENSIONS', 'part': row['id']})
        else:
            means = data.geom_xpos[[top, middle, toe], 2]-(a[2]+b[2])/2
            axes = data.geom_xmat[[top, middle]].reshape(-1, 3, 3)[:, :2, 2]
            means[:2] += model.geom_size[[top, middle], 0]/np.linalg.norm(axes, axis=1)
            if not np.allclose(means, [HEIGHT, HEIGHT/2, TOE_HEIGHT/2], atol=EPS):
                violations.append({'code': 'STAIR_GUARD_DIMENSIONS', 'part': row['id']})
    names = [p['id'] for p in scene['components'] if p['kind'] in ('rail', 'toe_board')]
    violations.extend({'code': 'GUARD_SOLID_COLLISION', **hit} for hit in collisions(scene, model, data, names))
    return violations
