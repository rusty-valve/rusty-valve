"""Check standing areas and stair access around installed valves."""
import math

import numpy as np

from .checks import POLICY, STEP, geom_bounds, navigation
from .insulation import control_sweeps
from .insulation_motion_check import audit as audit_motion
from .site_grating import EPS, geometry_index
from .site_stairs import STAIR_KINDS


def site_navigation(scene, model, data, ground):
    parts = {p['id']:p for p in scene['components']}
    ids, bounds, names = geometry_index(scene, model, data)
    exempt = {'grating', 'ground', 'drain', 'drain_cover', 'entry_base', 'walk_surface', 'ramp'}
    obstacles = {names[gid]:list(box) for gid, box in zip(ids,bounds)
                 if parts.get(names[gid], {}).get('kind') not in exempt}
    nav = {}
    hx, hy = POLICY['walk_width_m']/2, POLICY['walk_depth_m']/2
    for level in range(len(scene['floors'])):
        local = obstacles.copy()
        if level == 0 and not scene['stairs']:
            _,inside=ground['entry_ramps']
            x0,x1,y=inside[0],inside[2],max(hy,inside[3]+hy)
        elif level == 0:
            # The lower staircase opens onto the contained apron outside the
            # room. Enter the indoor walking grid from that front passage.
            landing = parts['top_landing_0']['box']
            x0, x1, y = landing[0], landing[3], hy
        else:
            landing = parts[f'top_landing_{level-1}']['box']
            x0, x1, y = landing[0], landing[3], hy
        start_y = math.ceil((y-EPS)/STEP)
        seeds = [(i, start_y) for i in range(math.ceil((x0+hx-EPS)/STEP), math.floor((x1-hx+EPS)/STEP)+1)]
        reached, standing = navigation(scene, list(local.values()), level, seeds=seeds)
        if level == 0:
            x = np.arange(standing.shape[0])*STEP
            y = np.arange(standing.shape[1])*STEP
            for x0, y0, x1, y1 in ground['entry_ramps']:
                standing &= ~((x[:, None] > x0-hx) & (x[:, None] < x1+hx)
                              & (y[None, :] > y0-hy) & (y[None, :] < y1+hy))
            reached &= standing
        nav[level] = reached, standing
    return nav


def audit(scene, graph, model, data, positioning, ground):
    nav = site_navigation(scene, model, data, ground)
    saved = {row['id']: row for row in positioning['rows']}
    valves = [valve for valve in graph['valves'] if valve.get('manual', True)]
    sweeps = control_sweeps({'valves': valves}, model, data, actual_operators=True, bounds_only=True)
    parts = [p for p in scene['components']
             if p['kind'] in (*STAIR_KINDS, 'rail', 'toe_board', 'walk_surface', 'drain_cover')]
    ids = [model.geom(p['id']).id for p in parts]
    bounds = geom_bounds(model, data, ids)
    pieces = [{'id': p['id'], 'source': p['kind'], 'actual_box_m': list(box)} for p, box in zip(parts,bounds)]
    hits, samples = audit_motion({'valves': valves}, model, data, pieces, sweeps)
    hits = [{'valve': h['valve'], 'source_valve': saved[h['valve']].get('source_valve'),
             'part': h['insulation'], 'part_kind': h['source'],
             'operator_part': h['operator_part'], 'fraction': h['fraction']} for h in hits]
    records = []
    for valve in valves:
        name = valve['id']
        evidence = saved[name]
        level = evidence['level']
        operation = np.asarray(evidence['operation_m'])
        facing = np.asarray(evidence.get('facing') or valve['stem'])[:2]
        reached, standing = nav[level]
        cells = np.argwhere(standing)*STEP
        delta = cells-operation[:2]
        nearby = np.linalg.norm(delta,axis=1) <= 1.+EPS
        if np.linalg.norm(facing) > EPS:
            nearby &= delta@facing >= -EPS
        available = cells[nearby]
        accessible = [xy for xy in available if reached[tuple(np.rint(xy/STEP).astype(int))]]
        options = accessible if len(accessible) else available
        closest = min(options,key=lambda xy: np.linalg.norm(xy-operation[:2]),default=None)
        floor = valve['operating_floor_z_m']
        records.append({'valve': name, 'source_valve': evidence.get('source_valve'), 'level': level,
                        'high': bool(evidence.get('high')), 'operator_height_m': float(operation[2]-floor),
                        'can_stand': bool(len(available)), 'stair_access': bool(scene['stairs'] and len(accessible)),
                        **({'entry_access':bool(len(accessible))} if not scene['stairs'] else {}),
                        'closest_stance_m': [*closest, floor] if closest is not None else None,
                        'valve_below_floor': valve['world_bbox'][2] < floor-EPS,
                        'operator_below_floor': bool(sweeps[name]['bounds'][2] < floor-EPS),
                        'operation_samples': samples[name]})
    return records, hits
