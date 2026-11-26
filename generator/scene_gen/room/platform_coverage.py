"""Check that floor and platform panels cover the intended area."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EPS = 1e-8  # Floating point comparisons only, not a construction tolerance.


def _clip(rect, bounds):
    r = [max(rect[0], bounds[0]), max(rect[1], bounds[1]),
         min(rect[2], bounds[2]), min(rect[3], bounds[3])]
    return r if r[2] - r[0] > EPS and r[3] - r[1] > EPS else None


def _area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def _uncovered(bounds, covers):
    """Exact axis-aligned rectangle-union complement, without a sampling grid."""
    clipped = [r for cover in covers if (r := _clip(cover, bounds))]
    xs = sorted({bounds[0], bounds[2], *(v for r in clipped for v in (r[0], r[2]))})
    result = []
    for xa, xb in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in clipped
                           if r[0] < xb - EPS and r[2] > xa + EPS)
        cursor = bounds[1]
        for ya, yb in intervals + [(bounds[3], bounds[3])]:
            if ya > cursor + EPS:
                result.append([xa, cursor, xb, ya])
            cursor = max(cursor, yb)
    return result


def _registered_seams(scene, level):
    """Reproduce only already-registered narrow construction gaps.

    panel_grid stores every seam centre and its actual width.  Equipment opening
    edges are additionally set back exactly 5 mm in existing building.deck;
    this known cut-edge seam is listed separately, not treated as an extra hole.
    Pipe openings do not receive that setback and are not enlarged here.
    """
    building = scene.get('building', {})
    grid = building.get('panel_grid')
    if level != 1 or not grid:
        return [], []
    xs, ys = grid['x'], grid['y']
    extent = [xs[0], ys[0], xs[-1], ys[-1]]
    seams = []
    sources = []
    for axis, values, widths in ((0, xs, grid['x_seams']), (1, ys, grid['y_seams'])):
        for centre, width in zip(values, widths):
            r = list(extent)
            r[axis], r[axis + 2] = centre - width / 2, centre + width / 2
            if clipped := _clip(r, extent):
                seams.append(clipped)
        sources.append({'source': f'geometry.building.panel_grid.{"x" if axis == 0 else "y"}_seams',
                        'widths_m': list(widths), 'limited_to_panel_grid_rect_m': extent})
    holes = scene['holes'].get(level, scene['holes'].get(str(level), []))
    equipment_count = scene.get('equipment_opening_count', 1)
    # Existing building.deck: r=[v+(.005 if k<2 else -.005) ...].
    # Its rectangular split extends the cut edge to the current board-cell edge.
    # Reuse that unchanged split, then exclude only the exact 5 mm edge strips.
    # Merely registering the panel cell never supplies its missing interior.
    from .building import subtract
    setback = .005
    gx, gy = grid['x_seams'], grid['y_seams']
    for ix, (a, b) in enumerate(zip(xs, xs[1:])):
        for iy, (c, d) in enumerate(zip(ys, ys[1:])):
            raw = [a + gx[ix] / 2, c + gy[iy] / 2,
                   b - gx[ix + 1] / 2, d - gy[iy + 1] / 2]
            for piece in subtract(raw, holes[:equipment_count]):
                trimmed = [v + (setback if k < 2 else -setback)
                           if abs(v - raw[k]) > 1e-7 else v
                           for k, v in enumerate(piece)]
                # Do not classify a dropped narrow panel as an edge seam.
                if trimmed[2] - trimmed[0] < .025 or trimmed[3] - trimmed[1] < .025:
                    continue
                seams.extend(_uncovered(piece, [trimmed]))
    sources.append({'source': 'scene_gen/room/building.py:deck equipment-opening panel cut-edge setback',
                    'setback_m': setback, 'equipment_openings': equipment_count,
                    'note': 'Use recorded panel subdivisions and exclude only existing 5 mm trims. Preserve registered openings.'})
    return seams, sources


def _platform_footprint(scene):
    if 'footprint' in scene:
        return [0., 0., *map(float, scene['footprint'])], 'geometry.footprint'
    # The frozen batch generator registers the in-room ground surface separately
    # from its external stair apron, so no building dimension is invented here.
    ground = [s['box'] for s in scene.get('surfaces', []) if s.get('level') == 0]
    if not ground:
        raise ValueError('Platform coverage requires footprint or level-zero surfaces')
    return [min(b[0] for b in ground), min(b[1] for b in ground),
            max(b[3] for b in ground), max(b[4] for b in ground)], 'geometry.surfaces[level=0]'


def check_platform_coverage(scene, model=None):
    """Return pass/fail plus exact uncovered rectangles for every platform level.

    ``scene`` is the exported geometry dictionary. If supplied, ``model`` also
    verifies that each counted board exists and has collision enabled. Existing
    room checks still decide support, headroom, navigation and valve operation.
    """
    bounds, footprint_source = _platform_footprint(scene)
    floors = scene.get('floors')
    if not floors or len(floors) < 2:
        raise ValueError('Platform coverage requires floor elevations')
    rows, errors = [], []
    for level, z in enumerate(floors[1:], start=1):
        boards, ids, ignored = [], [], []
        for component in scene['components']:
            if component.get('visual_only') or component.get('annotation_only'):
                continue
            kind = component['kind']
            if kind == 'walk_surface':
                if component.get('walk_level') != level:
                    continue
            elif kind == 'floor':
                if abs(component['box'][5] - z) > EPS:
                    continue
            else:
                continue
            # export_model disables collision on mesh components unless this
            # existing flag is set. Keep the offline check consistent with it.
            if component.get('shape') == 'mesh' and not component.get('collision_mesh'):
                ignored.append(component['id'])
                continue
            if model is not None:
                try:
                    geom = model.geom(component['id'])
                    collidable = bool(geom.contype[0] or geom.conaffinity[0])
                except (KeyError, IndexError):
                    collidable = False
                if not collidable:
                    ignored.append(component['id'])
                    continue
            b = component['box']
            boards.append([b[0], b[1], b[3], b[4]])
            ids.append(component['id'])
        holes = scene.get('holes', {}).get(level, scene.get('holes', {}).get(str(level), []))
        holes = [r for h in holes if (r := _clip(h, bounds))]
        seams, sources = _registered_seams(scene, level)
        seams = [r for seam in seams if (r := _clip(seam, bounds))]
        required_area = sum(_area(r) for r in _uncovered(bounds, holes))
        unseamed = _uncovered(bounds, boards + holes)
        missing = _uncovered(bounds, boards + holes + seams)
        uncovered_area = sum(map(_area, missing))
        seam_area = sum(map(_area, unseamed)) - uncovered_area
        row = {'level': level, 'platform_z_m': z,
               'status': 'fail' if missing else 'pass',
               'platform_rect_m': bounds, 'registered_openings': holes,
               'required_board_area_m2': required_area,
               'actual_collision_board_count': len(boards),
               'board_ids': ids, 'ignored_noncollidable_boards': ignored,
               'registered_seam_sources': sources,
               'excluded_existing_seam_area_m2': max(0., seam_area),
               'uncovered_area_m2': uncovered_area,
               'uncovered_rectangles_m': missing}
        rows.append(row)
        if missing:
            errors.append({'code': 'PLATFORM_BOARD_MISSING', 'level': level,
                           'uncovered_area_m2': uncovered_area,
                           'message': f'platform{level}Decking is missing beyond registered openings and existing panel gaps{uncovered_area:.6g}m²',
                           'witness_xy_m': [(missing[0][0] + missing[0][2]) / 2,
                                            (missing[0][1] + missing[0][3]) / 2]})
    return {'status': 'fail' if errors else 'pass', 'levels': rows, 'errors': errors,
            'footprint_source': footprint_source,
            'method': 'Compute the exact rectangular area after removing registered openings, physical collision plates and recorded joints. Beams, visual grating and metadata do not count as plates.'}


def main():
    from .evaluation_common import require_environment
    require_environment()
    import mujoco
    parser = argparse.ArgumentParser()
    parser.add_argument('geometry', type=Path)
    parser.add_argument('--xml', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    scene = json.loads(args.geometry.read_text())
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve())) if args.xml else None
    report = check_platform_coverage(scene, model)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
