"""Check that stairs reach the finished platforms."""
import math

import mujoco
import numpy as np

from .checks import geom_bounds
from .site_grating import EPS, geometry_index, overlap, slab_section, world_mesh


MIN_WIDTH = .70
HEADROOM = 2.
STAIR_KINDS = {'step', 'landing', 'stringer', 'step_edge'}


def flights(scene):
    parts = {part['id']: part for part in scene['components']}
    for flight in scene['stairs']:
        level, number = flight['level'], flight['flight']
        prefix = f'stair_{level}_{number}_'
        yield flight, [parts[f'{prefix}{i}'] for i in range(flight['steps'])]


def process_obstacles(scene, model, data, excluded=()):
    ids, bounds, names = geometry_index(scene, model, data, excluded)
    forbidden = {p['id'] for p in scene['components'] if p['kind'] in (
        *STAIR_KINDS, 'floor', 'walk_surface', 'grating', 'beam', 'equipment_beam', 'column',
        'rail', 'toe_board', 'ground', 'bund', 'drain', 'drain_cover', 'entry_base', 'ramp')}
    target = np.asarray([names[gid] not in forbidden for gid in ids])
    return ids[target], bounds[target], names


def collisions(scene, model, data, subjects):
    """Screen using actual geom bounds, then check signed distance in the saved model."""
    ids, bounds, names = process_obstacles(scene,model,data)
    subjects = list(subjects)
    subject_ids = [model.geom(name).id for name in subjects]
    hits = []
    for name, gid, bound in zip(subjects, subject_ids, geom_bounds(model, data, subject_ids)):
        close = np.all((bounds[:, 3:] > bound[:3]+EPS) & (bounds[:, :3] < bound[3:]-EPS), axis=1)
        for other in ids[close]:
            separation = float(mujoco.mj_geomDistance(model, data, gid, int(other), .01, None))
            if separation < -EPS:
                hits.append({'part': name, 'obstacle': names[other], 'penetration_m': -separation})
    return hits


def headroom(scene, model, data, tread_boxes):
    ids, bounds, names = geometry_index(scene, model, data)
    cache = {}
    blocked = []
    for name, box, width_axis in tread_boxes:
        lower, upper = box[5], box[5]+HEADROOM
        rect = list(box[:2]+box[3:5])
        # The required walking strip is centered, leaving room for handrails and posts.
        edge = max(0., (rect[width_axis+2]-rect[width_axis]-MIN_WIDTH)/2.)
        rect[width_axis] += edge
        rect[width_axis+2] -= edge
        near = ((bounds[:, 2] < upper-EPS) & (bounds[:, 5] > lower+EPS)
                & (bounds[:, 3] > rect[0]+EPS) & (bounds[:, 0] < rect[2]-EPS)
                & (bounds[:, 4] > rect[1]+EPS) & (bounds[:, 1] < rect[3]-EPS))
        for other in ids[near]:
            section = slab_section(*world_mesh(model, data, other, cache), lower, upper)
            if section is not None and overlap(rect, section):
                blocked.append({'code': 'STAIR_HEADROOM', 'part': name, 'obstacle': names[other]})
    return blocked


def audit(scene, model, data, supports=()):
    records, violations, tread_boxes = [], [], []
    parts = {p['id']: p for p in scene['components']}
    for flight, treads in flights(scene):
        level, number = flight['level'], flight['flight']
        boxes = geom_bounds(model, data, [model.geom(p['id']).id for p in treads])
        forward = boxes[-1,0] > boxes[0,0]
        fronts = boxes[:,0] if forward else boxes[:,3]
        rise, run = np.diff(boxes[:, 5]), abs(np.diff(fronts))
        angle = math.degrees(math.atan2(float(rise.max()), float(run.min())))
        width = float((boxes[:, 4]-boxes[:, 1]).min())
        if angle > 45.+EPS or width < MIN_WIDTH-EPS:
            violations.append({'code': 'STAIR_DIMENSIONS', 'level': level, 'flight': number})
        if not (np.all(abs(rise-flight['rise']) <= .020+EPS) and np.allclose(run, flight['run'], atol=EPS)):
            violations.append({'code': 'STAIR_STEP_CONNECTION', 'level': level, 'flight': number})
        lap = boxes[:-1,3]-boxes[1:,0] if forward else boxes[1:,3]-boxes[:-1,0]
        if supports and np.any((lap<.010-EPS)|(lap>.035+EPS)):
            violations.append({'code':'STAIR_OVERLAP','level':level,'flight':number})
        for p, box in zip(treads,boxes):
            walk = list(box)
            if forward:
                walk[3] = min(box[3],box[0]+flight['run'])
            else:
                walk[0] = max(box[0],box[3]-flight['run'])
            tread_boxes.append((p['id'],walk,1))
        records.append({'from_level': level, 'flight': number, 'steps': len(treads), 'angle_deg': angle,
                        'clear_width_m': width, 'first': list(boxes[0]), 'last': list(boxes[-1])})
    for level in range(len(scene['floors'])-1):
        pair = [r for r in records if r['from_level'] == level]
        turn, top = (parts[f'{kind}_landing_{level}'] for kind in ('turn', 'top'))
        if len(pair) == 2:
            near, far = pair
            far_y = far['first'][1]+(far['clear_width_m']-MIN_WIDTH)/2
            near_y = near['first'][4]-(near['clear_width_m']-MIN_WIDTH)/2
            for landing in (turn, top):
                box = landing['box'].copy()
                box[1] = max(box[1], far_y)
                if landing is turn:
                    box[4] = min(box[4], near_y)
                tread_boxes.append((landing['id'], box, 0))
        if (len(pair) != 2 or abs(pair[0]['last'][5]-turn['box'][5]) > EPS
                or abs(pair[1]['last'][5]-top['box'][5]) > EPS
                or abs(top['box'][5]-scene['floors'][level+1]) > EPS):
            violations.append({'code': 'STAIR_LEVEL_CONNECTION', 'level': level})
    violations.extend(headroom(scene, model, data, tread_boxes))
    for support in supports:
        tread = model.geom(support['part']).id
        for plate,beam in zip(support['ends'],support['stringers']):
            a,b = model.geom(plate).id,model.geom(beam).id
            if any(mujoco.mj_geomDistance(model,data,i,j,.02,None) > .001
                   for i,j in ((tread,a),(a,b))):
                violations.append({'code':'TREAD_SUPPORT','part':support['part'],'end':plate})
    violations.extend({'code': 'STAIR_SOLID_COLLISION', **hit} for hit in collisions(
        scene, model, data, (p['id'] for p in parts.values() if p['kind'] in STAIR_KINDS)))
    return records, violations
