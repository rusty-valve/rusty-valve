"""Remove valves that would sit too close together and reconnect the pipe."""
from collections import Counter
import math
import re
import shutil
import traceback

import numpy as np

from .pipeline_status import pipeline_status
from .process import write_json
from .valve_positioning import Room, ports, read, save_model, straight_coverage, verify_saved
from .valve_sweep_geometry import GripClearance, GRIP_PREFERRED

CENTER_LIMIT = .30
DISPOSITION = 'removed for insufficient valve separation'


def priority(valve, row):
    root = row.get('placement_rule') in ('at_port', 'top_vertical')
    name = row['source_valve']
    number = re.split(r'(\d+)', name)
    natural = tuple(int(value) if i % 2 else value for i, value in enumerate(number))
    return not root, valve['operator'] != 'wheel', -valve['dn'], natural


def choose(room, positioning):
    rows = {row['id']: row for row in positioning['rows'] if row['status'] == 'placed'}
    active = [v for v in room.current.values() if v.get('manual')]
    if any(v['id'] not in rows for v in active):
        raise ValueError('A saved manual valve is missing its installation record')
    active.sort(key=lambda valve: priority(valve, rows[valve['id']]))
    geometry = room.load_geometry()
    centers = {v['id']: np.mean(ports(v), axis=0) for v in active}
    kept, removed = [], []
    for valve in active:
        key = valve['id']
        for peer in kept:
            name = peer['id']
            center = math.dist(centers[key], centers[name])
            # If the center test wins, neither sweep distance can affect the result.
            distances = []
            if center >= CENTER_LIMIT:
                a, b = geometry.peer(valve), geometry.peer(peer)
                for grip_side, other_side in ((a, b), (b, a)):
                    grip = GripClearance(GRIP_PREFERRED)
                    geometry.peer_hit(grip_side, other_side, grip, check_hard=False)
                    distances.append(grip.distance if grip.distance < GRIP_PREFERRED else None)
                if all(distance is None for distance in distances):
                    continue
            row, winner = rows[key], rows[name]
            removed.append({'id': key, 'source_valve': row['source_valve'], 'edge': row.get('edge'),
                'dn': valve['dn'], 'operator': valve['operator'], 'placement_rule': row.get('placement_rule'),
                'near': name, 'near_source_valve': winner['source_valve'],
                'near_dn': peer['dn'], 'near_operator': peer['operator'],
                'near_placement_rule': winner.get('placement_rule'), 'center_distance_m': center,
                'grip_to_peer_m': distances[0] if distances else None,
                'peer_grip_to_valve_m': distances[1] if distances else None, 'disposition': DISPOSITION})
            break
        else:
            kept.append(valve)
    return removed


def generate(source, output):
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('Valve separation processing requires a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    try:
        room = Room(source)
        position = read(source/'valve_positioning.json')
        before = pipeline_status(room.measured, room.piping, model_present=True)
        removed = choose(room, position)
        room.geometry = None
        ods = room.piping['config']['pipe_outer_diameter_mm']
        coverage = straight_coverage(room.parts.values(), room.current.values(), ods)
        axes = [(r['id'], r.get('control_points', r.get('points'))) for r in room.routes]
        removed_names = {entry['id'] for entry in removed}
        counts = Counter()
        for entry in removed:
            valve = room.current[entry['id']]
            room.close_original_cut(valve)
            del room.current[entry['id']]
            counts[valve.get('phase', 'P1')] += 1
        room.graph['valves'] = list(room.current.values())
        if coverage != straight_coverage(room.parts.values(), room.current.values(), ods):
            raise ValueError('Replacement pipe does not cover the same axial span as the valve and original pipe')
        if axes != [(r['id'], r.get('control_points', r.get('points'))) for r in room.routes]:
            raise ValueError('Replacement pipe changed the route direction')
        for phase, container in [('P1', room.measured), ('P2', room.measured['material'])]:
            container['manual_valves']['installed'] -= counts[phase]
        room.piping['close_valve_pruning'] = removed
        after = pipeline_status(room.measured, room.piping, model_present=True)
        for key in ('connections', 'penetrations', 'collision_breaks', 'pending_valves'):
            if after[key] != before[key]:
                raise ValueError('Saved model comparison failed: '+key)
        folder = output/'model'
        xml, auto = save_model(room, folder)
        verify_saved(room, xml, valve_ids=room.current)
        for row in position['rows']:
            if row['id'] in removed_names:
                row.update(status='discarded_close', disposition=DISPOSITION)
        for key in ('final_audit', 'final_grip', 'station_operators_in_saved_model'):
            position[key] = {name: row for name, row in position[key].items() if name not in removed_names}
        position.update(source=str(source), before=before, after=after, fixed_automatic_geometries=auto,
                        qualified=after['valves']['installed'], discarded_close=len(removed),
                        wheels=sum(v['operator']=='wheel' for v in room.current.values() if v.get('manual')),
                        high=sum(r.get('high', False) for r in position['rows'] if r['status']=='placed'),
                        grip_tight=sum(r['tight'] for r in position['final_grip'].values()))
        report = {'source': str(source), 'before': before, 'after': after, 'removed': removed,
                  'restored_straights': room.moves, 'fixed_automatic_geometries': auto,
                  'axis_and_span_preserved': True}
        for name, data in [('process', room.graph), ('geometry', room.scene), ('piping', room.piping),
                           ('measure', room.measured), ('valve_positioning', position), ('close_valve_pruning', report)]:
            write_json(folder/(name+'.json'), data)
        (output/'layout').symlink_to((source.parent/'layout').resolve(), target_is_directory=True)
        shutil.copy2(source.parent/'input.json', output/'input.json')
        return report
    except Exception:
        write_json(output/'failure.json', {'traceback': traceback.format_exc()})
        raise
