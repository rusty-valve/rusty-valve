"""Set building dimensions and construct basic steel framing and levels."""
from __future__ import annotations

import math

from .process import RULES
from .layout import add_box, add_pipe

STEEL = (.24, .35, .43)
BOARD = (.53, .58, .60)
RAIL = (.95, .72, .06)


def resolve_shell(params):
    if not isinstance(params, dict):
        raise ValueError('The shell must be a JSON object')
    allowed = {'bays', 'platforms', 'storey_height_m'}
    extra = set(params) - allowed
    if extra:
        raise ValueError('The shell accepts bays, platforms, and storey_height_m. Unexpected fields: ' + ', '.join(sorted(extra)))
    rules = RULES['floor_rules']
    bays = params.get('bays', rules['bays'])
    platforms = params.get('platforms', rules['ground_plus_platforms'])
    height = params.get('storey_height_m', RULES['construction_choices']['storey_height_m'])
    for label, value in (('Bay count', bays), ('Platform count', platforms)):
        if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
            raise ValueError(f'{label} must be 1 or 2')
    low, high = rules['storey_height_range_m']
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height) or not low <= height <= high:
        raise ValueError(f'Storey height must be between {low} and {high} m')
    bay_x, bay_y = map(float, rules['bay_size_m'])
    width, depth = bays * bay_x, bay_y
    return {'bays': bays, 'platforms': platforms, 'storey_height_m': float(height),
            'bay_size_m': [bay_x, bay_y], 'footprint_m': [width, depth],
            'width_m': width, 'depth_m': depth,
            'floors': [i * float(height) for i in range(platforms + 1)],
            'x_axes_m': [i * bay_x for i in range(bays + 1)],
            'y_axes_m': [0., depth], 'roof_z_m': (platforms + 1) * float(height)}


def _frame_parts(shell):
    """One source for actual structural solids and routing reservations."""
    width, depth = shell['footprint_m']
    roof = shell['roof_z_m']
    for ix, x in enumerate(shell['x_axes_m']):
        for iy, y in enumerate(shell['y_axes_m']):
            yield f'column_{ix}_{iy}', [x - .10, y - .10, 0., x + .10, y + .10, roof], 'column'
    for level, z in enumerate(shell['floors'][1:] + [roof], start=1):
        for iy, y in enumerate(shell['y_axes_m']):
            yield f'beam_{level}_{iy}', [0., y - .10, z - .30, width, y + .10, z - .10], 'beam'
        for ix, x in enumerate(shell['x_axes_m']):
            yield f'crossbeam_{level}_{ix}', [x - .10, 0., z - .30, x + .10, depth, z - .10], 'beam'


def routing_obstacles(shell):
    """Return the exact column/beam boxes, before boards are added."""
    return [box for _, box, _ in _frame_parts(shell)]


def _subtract(rect, holes):
    pieces = [rect]
    for hole in holes:
        next_pieces = []
        for x0, y0, x1, y1 in pieces:
            a, b = max(x0, hole[0]), max(y0, hole[1])
            c, d = min(x1, hole[2]), min(y1, hole[3])
            if a < c and b < d:
                next_pieces.extend(([x0, y0, a, y1], [c, y0, x1, y1],
                                    [a, y0, c, b], [a, d, c, y1]))
            else:
                next_pieces.append([x0, y0, x1, y1])
        pieces = [r for r in next_pieces if r[2] - r[0] > 1e-8 and r[3] - r[1] > 1e-8]
    return pieces


def _stairs(scene, shell):
    choices = RULES['construction_choices']
    width, _ = shell['footprint_m']
    stair_width = choices['stair_width_m']
    height = shell['storey_height_m']
    # An even number of rises makes two equal flights at every supported height.
    rises = int(math.ceil(height / choices['stair_rise_m'] / 2)) * 2
    per_flight = rises // 2
    rise = height / rises
    run = choices['stair_run_m']
    right = width - stair_width
    left = right - per_flight * run
    near_y = -stair_width / 2 - .10
    far_y = -stair_width * 1.5 - .30
    ymin = far_y - stair_width / 2
    # The apron is part of the same physical ground slab, not a separate support.
    add_box(scene, 'ground', [-.40, ymin - .40, -.16, width + .40,
                             shell['depth_m'] + .40, 0.], 'floor', (.67, .69, .68))
    scene['surfaces'].append({'box': [0., 0., -.16, *shell['footprint_m'], 0.], 'level': 0})
    passage = choices['passage_width_m']
    scene['navigation_entries'] = {str(i): [width - stair_width / 2, passage / 2]
                                   for i in range(len(shell['floors']))}
    scene['stairs'] = []
    for level, base in enumerate(shell['floors'][:-1]):
        mid, top = base + height / 2, base + height
        for flight, (start, direction, y, z0) in enumerate(
                ((right, -1, near_y, base), (left, 1, far_y, mid))):
            for i in range(per_flight):
                a = start + direction * i * run
                b = start + direction * (i + 1) * run
                z = z0 + (i + 1) * rise
                add_box(scene, f'stair_{level}_{flight}_{i}',
                        [min(a, b), y - stair_width / 2, z - .06,
                         max(a, b), y + stair_width / 2, z], 'step', BOARD)
            for side in (-1, 1):
                sy = y + side * (stair_width / 2 + .05)
                a = [start, sy, z0 - .10]
                b = [start + direction * per_flight * run, sy, z0 + height / 2 - .10]
                add_pipe(scene, f'stringer_{level}_{flight}_{side}', a, b, .12, 'stringer', STEEL)
                if not shell.get('layout_only'):
                    add_pipe(scene, f'stair_rail_{level}_{flight}_{side}',
                             [a[0], sy, z0 + 1.10], [b[0], sy, z0 + height / 2 + 1.10],
                             .025, 'rail', RAIL)
            scene['stairs'].append({'rise': rise, 'run': run, 'width': stair_width,
                                    'steps': per_flight, 'level': level, 'flight': flight})
        add_box(scene, f'turn_landing_{level}',
                [left - stair_width, ymin, mid - .10, left, -.10, mid], 'landing', BOARD)
        add_box(scene, f'top_landing_{level}',
                [right, ymin, top - .10, width, 0., top], 'landing', BOARD)
        # Simple outer guards; the front edge is open where it meets the deck.
        for name, x in (() if shell.get('layout_only') else (('left', right), ('right', width))):
            add_pipe(scene, f'landing_rail_{level}_{name}',
                     [x, ymin, top + 1.10], [x, -.10, top + 1.10], .025, 'rail', RAIL)
        if not shell.get('layout_only'):
            add_pipe(scene, f'turn_rail_{level}', [left - stair_width, ymin, mid + 1.10],
                     [left - stair_width, -.10, mid + 1.10], .025, 'rail', RAIL)


def add_structure(scene, graph, shell):
    """Add the same steel frame reserved for routing, then solid cut boards."""
    scene['footprint'] = list(shell['footprint_m'])
    scene['floors'] = list(shell['floors'])
    scene['shell'] = dict(shell)
    _stairs(scene, shell)
    for name, box, kind in _frame_parts(shell):
        add_box(scene, name, box, kind, STEEL)
    width, depth = shell['footprint_m']
    doorway_x = width - RULES['construction_choices']['stair_width_m']
    for level, z in enumerate(shell['floors'][1:], start=1):
        holes = scene['holes'].get(level, scene['holes'].get(str(level), []))
        for i, rect in enumerate(_subtract([0., 0., width, depth], holes)):
            x0, y0, x1, y1 = rect
            box = [x0, y0, z - .10, x1, y1, z]
            add_box(scene, f'deck_{level}_{i}', box, 'floor', BOARD)
            scene['surfaces'].append({'box': box, 'level': level})
        for zz in (() if shell.get('layout_only') else (.55, 1.10)):
            for side, x in enumerate((0., width)):
                add_pipe(scene, f'rail_side_{level}_{side}_{zz}',
                         [x, 0., z + zz], [x, depth, z + zz], .025, 'rail', RAIL)
            add_pipe(scene, f'rail_front_{level}_{zz}',
                     [0., 0., z + zz], [doorway_x, 0., z + zz], .025, 'rail', RAIL)
            add_pipe(scene, f'rail_rear_{level}_{zz}',
                     [0., depth, z + zz], [width, depth, z + zz], .025, 'rail', RAIL)
