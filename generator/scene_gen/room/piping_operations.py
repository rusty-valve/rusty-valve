"""Find where an operator can stand and check valve travel for obstructions."""
from __future__ import annotations

from copy import deepcopy
import math

import mujoco
import numpy as np

from .building_equipment_checks import (
    EPS, _reachability, _standing_boards, _touches_reachable, _uncovered,
)
from .evaluation_common import require_environment
from .process import RULES


POLICY = RULES['geometry_screen']
STEP_RAD = math.radians(5.)


class OperationUnavailable(ValueError):
    """An actual package/model motion or operating interface is missing."""


def _union(boxes):
    values = np.asarray(boxes, dtype=float)
    return np.r_[values[:, :3].min(axis=0), values[:, 3:].max(axis=0)].tolist()


def _bounds(model, data, ids):
    """Actual loaded geom bounds, including visual meshes and zero masks."""
    ids = np.asarray(ids, dtype=int)
    if not len(ids):
        return np.empty((0, 6))
    rotations = data.geom_xmat[ids].reshape(-1, 3, 3)
    centers = data.geom_xpos[ids] + np.einsum('nij,nj->ni', rotations, model.geom_aabb[ids, :3])
    half = np.einsum('nij,nj->ni', abs(rotations), model.geom_aabb[ids, 3:])
    return np.c_[centers-half, centers+half]


def _hits(box, boxes, margin=0.):
    if not len(boxes):
        return np.zeros(0, dtype=bool)
    box = np.asarray(box)
    return np.all(np.minimum(box[3:]+margin, boxes[:, 3:]) -
                  np.maximum(box[:3]-margin, boxes[:, :3]) > EPS, axis=1)


def _body_descends(model, body, root):
    while body:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return False


def _asset(valve):
    result = valve.get('asset', valve)
    if not isinstance(result, dict) or not result.get('motion', {}).get('available'):
        raise OperationUnavailable('Missing usable native motion data')
    return result


def _mapping(valve, model):
    asset = _asset(valve)
    prefix = valve.get('prefix', valve['id']+'_')
    root_name = prefix+asset['root_body']
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root < 0:
        raise OperationUnavailable('Actual model lacks the native attachment root: '+root_name)
    original = asset['motion']
    states = original.get('joint_states', {})
    opened, closed = states.get('open'), states.get('closed')
    if not isinstance(opened, dict) or not opened or not isinstance(closed, dict) or set(opened) != set(closed):
        raise OperationUnavailable('Package lacks complete and consistent joint states for fully open and closed positions')
    records = {j['name']: j for j in original.get('joints', [])}
    mapping = {}
    for name in opened:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix+name)
        if jid < 0 or name not in records:
            raise OperationUnavailable('Actual model lacks the original joint: '+prefix+name)
        row = records[name]
        values = [float(opened[name]), float(closed[name])]
        if (not all(math.isfinite(v) for v in values) or
                int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE) or
                not _body_descends(model, int(model.jnt_bodyid[jid]), root) or
                not np.allclose(model.jnt_axis[jid], row['native_body_axis'], atol=EPS, rtol=0) or
                bool(model.jnt_limited[jid]) != row['limited'] or
                not np.allclose(model.jnt_range[jid], row['range_rad'], atol=EPS, rtol=0)):
            raise OperationUnavailable('Actual joint type, axis or range differs from package: '+prefix+name)
        if model.jnt_limited[jid] and any(v < model.jnt_range[jid, 0]-EPS or v > model.jnt_range[jid, 1]+EPS for v in values):
            raise OperationUnavailable('Open or closed state exceeds physical joint travel: '+prefix+name)
        mapping[name] = {'id': jid, 'address': int(model.jnt_qposadr[jid]),
                         'open': values[0], 'closed': values[1]}
    ids, moving = [], []
    for geom in asset.get('local_geoms', []):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix+geom['name'])
        if gid < 0 or not _body_descends(model, int(model.geom_bodyid[gid]), root):
            raise OperationUnavailable('Native package geometry is missing: '+prefix+geom['name'])
        actual_joints = []
        body = int(model.geom_bodyid[gid])
        while body != root:
            actual_joints.extend(range(int(model.body_jntadr[body]),
                                       int(model.body_jntadr[body]+model.body_jntnum[body])))
            body = int(model.body_parentid[body])
        if bool(actual_joints) != bool(geom['moving']):
            raise OperationUnavailable('Actual moving parts differ from package records: '+prefix+geom['name'])
        ids.append(gid)
        if actual_joints:
            if any(model.joint(j).name not in {prefix+n for n in mapping} for j in actual_joints):
                raise OperationUnavailable('Package has moving joints with missing open or closed states')
            moving.append(gid)
    actual_ids = [i for i in range(model.ngeom) if _body_descends(model, int(model.geom_bodyid[i]), root)]
    if set(actual_ids) != set(ids) or not moving:
        raise OperationUnavailable('Actual package geometry is incomplete or lacks moving parts')
    # Forward kinematics does not enforce equality constraints. Verify the
    # original active linear mapping, then explicitly drive both real joints.
    for equation in original.get('equalities', []):
        if not equation.get('active_at_start') or equation.get('type') != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        one, two = equation.get('joint1'), equation.get('joint2')
        if one not in mapping or two not in mapping:
            continue
        candidates = [i for i in range(model.neq)
                      if int(model.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT)
                      and int(model.eq_obj1id[i]) == mapping[one]['id']
                      and int(model.eq_obj2id[i]) == mapping[two]['id']]
        # Some original packages leave their equality unnamed. Identify its
        # actual joint pair instead of inventing an attached equality name.
        eqid = candidates[0] if len(candidates) == 1 else -1
        if eqid < 0 or not bool(model.eq_active0[eqid]):
            raise OperationUnavailable('The loaded model lacks an active native coupling equality')
        if (model.joint(int(model.eq_obj1id[eqid])).name != prefix+one or
                model.joint(int(model.eq_obj2id[eqid])).name != prefix+two or
                not np.allclose(model.eq_data[eqid], equation['data'], atol=EPS, rtol=0)):
            raise OperationUnavailable('The actual joint coupling differs from the package')
        coefficients = np.asarray(equation['data'][:5])
        if any(abs(c) > EPS for c in coefficients[2:]):
            raise OperationUnavailable('Linear interpolation requires linear native coupling')
        for state in ('open', 'closed'):
            expected = coefficients[0]+coefficients[1]*mapping[two][state]
            if abs(mapping[one][state]-expected) > EPS:
                raise OperationUnavailable('Package open or closed states violate the actual coupling')
    return asset, prefix, root, mapping, ids, moving


def _actual_ports(asset, data, root):
    rotation = data.xmat[root].reshape(3, 3)
    return {key: {'position': (data.xpos[root]+rotation @ np.asarray(port['position'])).tolist(),
                  'normal': (rotation @ np.asarray(port['normal'])).tolist()}
            for key, port in asset['local_ports'].items()}


def _motion_envelope(model, data, root, moving):
    """Conservative all-angle broad phase from actual rigid-link lengths.

    Every hinge can add at most twice its anchor's local distance. Triangle
    bounds through the full body chain therefore enclose all actual samples;
    this only prunes distant obstacles, never replaces moving geometry.
    """
    radii = []
    for gid in moving:
        radius = float(np.linalg.norm(model.geom_pos[gid]) +
                       np.linalg.norm(model.geom_aabb[gid, :3]) +
                       np.linalg.norm(model.geom_aabb[gid, 3:]))
        body = int(model.geom_bodyid[gid])
        while body != root:
            radius += float(np.linalg.norm(model.body_pos[body]))
            for joint in range(int(model.body_jntadr[body]), int(model.body_jntadr[body]+model.body_jntnum[body])):
                radius += 2*float(np.linalg.norm(model.jnt_pos[joint]))
            body = int(model.body_parentid[body])
        radii.append(radius)
    radius = max(radii)+POLICY['sweep_margin_m']
    origin = data.xpos[root]
    return np.r_[origin-radius, origin+radius]


def _end_contact(box, obstacle, model, data, ports, margin):
    """Allow only a margin-only touch at a measured coaxial pipe endpoint.

    There is no owner-wide exclusion, and positive actual AABB intrusion is
    never excused by this endpoint rule.
    """
    if obstacle.get('kind') not in ('pipe', 'reducer', 'pipe_reducer'):
        return None
    gid = obstacle['geom_id']
    if int(model.geom_type[gid]) not in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        return None
    actual = np.asarray(obstacle['box'])
    if _hits(box, actual[None, :])[0]:
        return None
    axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    center = data.geom_xpos[gid]
    half_length, radius = float(model.geom_size[gid, 1]), float(model.geom_size[gid, 0])
    for key, port in ports.items():
        position, normal = np.asarray(port['position']), np.asarray(port['normal'])
        if abs(abs(float(axis @ normal))-1) > 1e-7:
            continue
        ends = [center-axis*half_length, center+axis*half_length]
        if min(np.linalg.norm(end-position) for end in ends) > 1e-7 or (center-position) @ normal < -EPS:
            continue
        intersection = np.r_[np.maximum(np.asarray(box)[:3]-margin, actual[:3]),
                              np.minimum(np.asarray(box)[3:]+margin, actual[3:])]
        corners = np.asarray([[intersection[i+3 if flag & (1 << i) else i] for i in range(3)] for flag in range(8)])
        relative = corners-position
        axial = relative @ normal
        radial = relative-axial[:, None]*normal
        if np.max(np.abs(axial)) <= margin+EPS and np.max(np.linalg.norm(radial, axis=1)) <= math.sqrt(2)*radius+margin+EPS:
            return {'port': key, 'component': obstacle['id'], 'actual_intrusion': False,
                    'scope': 'Local coaxial pipe end contact allows only overlap in the 25 mm sweep margin'}
    return None


def _grips(asset, data, root):
    """Original grip segment / real wheel rim points, never its empty hub."""
    geometry, motion = asset['params']['geometry'], asset['params']['motion']
    if asset['operator'] == 'lever':
        try:
            point = [(float(geometry['handle_grip_start_mm'])+float(geometry['handle_grip_end_mm']))/2000,
                     0., float(geometry['handle_center_z_mm'])/1000]
        except (KeyError, TypeError, ValueError) as exc:
            raise OperationUnavailable('Package lacks physical lever grip dimensions') from exc
        local = [point]
        basis = 'Midpoint of handle_grip_start/end and handle_center_z approximate the grasp point'
    else:
        try:
            center = np.asarray(motion['wheel_center_m'], dtype=float)
            axis = np.asarray(motion['wheel_axis'], dtype=float)
            radius = (float(geometry['wheel_outer_diameter_mm'])-float(geometry['wheel_rim_diameter_mm']))/2000
        except (KeyError, TypeError, ValueError) as exc:
            raise OperationUnavailable('Package lacks physical wheel center or rim dimensions') from exc
        if center.shape != (3,) or axis.shape != (3,) or not np.isfinite(np.r_[center, axis, radius]).all() or radius <= 0:
            raise OperationUnavailable('Invalid native rim dimensions')
        axis /= np.linalg.norm(axis)
        reference = np.eye(3)[np.argmin(abs(axis))]
        first = np.cross(axis, reference); first /= np.linalg.norm(first)
        second = np.cross(axis, first)
        local = [center+radius*(math.cos(a)*first+math.sin(a)*second) for a in np.arange(16)*math.tau/16]
        basis = 'Native center and axis define 16 points on the actual wheel rim centerline'
    rotation = data.xmat[root].reshape(3, 3)
    return [(data.xpos[root]+rotation @ p).tolist() for p in local], basis


def _environment(scene, model, data):
    boxes = _bounds(model, data, range(model.ngeom))
    source = {c['id']: c for c in scene['components']}
    obstacles, measured = [], []
    for gid, box in enumerate(boxes):
        name = model.geom(gid).name or f'geom_{gid}'
        meta = source.get(name, {})
        item = {'id': name, 'geom_id': gid, 'kind': meta.get('kind', 'model_geom'),
                'owner': meta.get('owner'), 'box': box.tolist(),
                'no_contact_geometry': not bool(model.geom_contype[gid] or model.geom_conaffinity[gid])}
        obstacles.append(item)
        if name in source:
            measured.append({**meta, 'box': box.tolist()})
    return obstacles, measured


def _navigation(scene, measured, obstacles, valves):
    """Reuse layout's rectangle configuration space with actual board/entry."""
    nav, boards = {}, {}
    real_equipment = {row['id'] for row in scene.get('equipment_instances', [])}
    physical_parts = {'vessel', 'tank', 'head', 'jacket_shell', 'pump', 'motor', 'filter',
                      'exchanger', 'condenser', 'fixed_bed', 'tower', 'drive_reservation',
                      'equipment_nozzle', 'nozzle', 'support', 'lug', 'saddle', 'skirt', 'plinth'}
    valve_geom_groups = {}
    for valve in valves:
        prefix = valve.get('prefix', valve['id']+'_')
        for obstacle in obstacles:
            if obstacle['id'].startswith(prefix):
                valve_geom_groups[obstacle['id']] = ('valve', valve['id'])
    for level, floor in enumerate(scene['floors']):
        boards[level] = _standing_boards(measured, level, floor)
        groups = {}
        for obstacle in obstacles:
            box = obstacle['box']
            if box[5] <= floor+EPS or box[2] >= floor+POLICY['walk_height_m']-EPS:
                continue
            # Equipment envelopes match the existing layout method. Individual
            # pipe segments never merge just because their owners are equal.
            group = valve_geom_groups.get(obstacle['id'])
            if group is None:
                group = (('equipment', obstacle['owner'])
                         if obstacle['owner'] in real_equipment and obstacle['kind'] in physical_parts
                         else ('geom', obstacle['id']))
            groups.setdefault(group, []).append(box)
        rects = []
        for key, values in groups.items():
            box = _union(values)
            rects.append({'id': str(key), 'rect': [box[0], box[1], box[3], box[4]]})
        entries = scene.get('navigation_entries', {})
        entry = entries.get(str(level), entries.get(level))
        nav[level] = _reachability(boards[level], rects, entry)
    return nav, boards


def _actual_distance(model, work, first, second, first_box, second_box, margin, backend, pose_cache, pose_index):
    """Actual geometry query; zero-result fallback cannot invent deep overlap."""
    extent = np.maximum(first_box[3:], second_box[3:])-np.minimum(first_box[:3], second_box[:3])
    cutoff = float(np.linalg.norm(extent))+margin
    native = float(mujoco.mj_geomDistance(model, work, first, second, cutoff, None))
    result = {'distance_m': native, 'native_distance_m': native, 'distance_method': 'actual XML mj_geomDistance',
              'distance_cutoff_m': cutoff, 'support_plane': None, 'fallback_rechecked': False}
    if abs(native) > EPS:
        return result
    from .piping_checks import _separation_proof
    def measured(gid, box):
        return {'id': model.geom(gid).name, 'index': int(gid), 'type': int(model.geom_type[gid]),
                'position': work.geom_xpos[gid].tolist(), 'matrix': work.geom_xmat[gid].reshape(3, 3).tolist(),
                'size': model.geom_size[gid].tolist(), 'box': np.asarray(box).tolist()}
    proof = _separation_proof(measured(first, first_box), measured(second, second_box), model, pose_cache)
    if proof is not None:
        # A support bound near zero still fails the original 25mm clearance;
        # it only prevents a backend from fabricating a large penetration.
        result.update(distance_m=proof['gap_m'], support_plane=proof,
                      distance_method='actual support-plane distance lower bound for native numerical zero')
        return result
    if backend.get('model') is None:
        backend['model'] = deepcopy(model)
        backend['model'].opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
        backend['data'] = mujoco.MjData(backend['model'])
    if backend.get('pose_index') != pose_index:
        mujoco.mj_copyData(backend['data'], backend['model'], work)
        mujoco.mj_kinematics(backend['model'], backend['data'])
        backend['pose_index'] = pose_index
    other = float(mujoco.mj_geomDistance(backend['model'], backend['data'], first, second, cutoff, None))
    result.update(distance_m=min(native, other), fallback_distance_m=other, fallback_rechecked=True,
                  distance_method='actual XML native zero rechecked with bundled libccd on private model')
    return result


def _frozen_hint(valve, scene, stance):
    records = scene.get('operation_reservations', [])
    if isinstance(records, dict):
        records = list(records.values())
    result = []
    for record in records:
        if record.get('equipment', record.get('id')) != valve.get('equipment'):
            continue
        for end in ('upper', 'lower'):
            reservation = record.get(end)
            if not isinstance(reservation, dict):
                continue
            box = reservation.get('stance_box')
            result.append({'end': end, 'side': reservation.get('side'), 'level': reservation.get('level'),
                           'same_operation_side': (valve.get('operation_side') == reservation.get('side')) if valve.get('operation_side') else None,
                           'stance_inside_frozen_reservation': (box[0]-EPS <= stance[0] <= box[3]+EPS and box[1]-EPS <= stance[1] <= box[4]+EPS) if box and stance else None,
                           'scope': 'Original operating-side relation is context for physical deck, access and sweep checks'})
    return result


def _stance(valve, grips, scene, obstacles, nav, boards):
    level = valve.get('level')
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level < len(scene['floors']):
        raise OperationUnavailable('Valve lacks an actual operating level index')
    floor = scene['floors'][level]
    from .rule_identity import band,height_preference,in_band
    low, high = band()
    preference=height_preference(grips,scene['floors'],[level],valve.get('equipment_top',False))
    reach = RULES['construction_choices']['stance_reach_proxy_m']
    hx, hy = POLICY['station_width_m']/2, POLICY['station_depth_m']/2
    candidates, connected = [], []
    obstacle_boxes = np.asarray([o['box'] for o in obstacles]).reshape(-1, 6)
    checked = 0
    for grip in grips:
        height = grip[2]-floor
        if not in_band(height):
            continue
        for distance in (.55, .65, .75):
            if distance > reach+EPS:
                continue
            for angle in np.arange(16)*math.tau/16:
                point = [grip[0]+distance*math.cos(angle), grip[1]+distance*math.sin(angle), floor]
                box = [point[0]-hx, point[1]-hy, floor, point[0]+hx, point[1]+hy, floor+POLICY['station_height_m']]
                checked += 1
                if _uncovered([box[0], box[1], box[3], box[4]], boards[level]):
                    continue
                raised = np.array(box); raised[2] += EPS
                if _hits(raised, obstacle_boxes).any():
                    continue
                candidate = {'point': point, 'box': box, 'grip': grip, 'height': height,
                             'connected': bool(_touches_reachable(box, nav[level]))}
                candidates.append(candidate)
                if candidate['connected']:
                    connected.append(candidate)
    selected = (connected or candidates or [None])[0]
    heights = [p[2]-floor for p in grips]
    return {'operation_preference':preference,'rule_identity':preference['identity'],'stance': selected['point'] if selected else None, 'stance_box': selected['box'] if selected else None,
            'grip': selected['grip'] if selected else grips[0], 'stance_free': bool(candidates),
            'reachable_from_passage': bool(connected), 'needs_elevation': all(h > high+EPS for h in heights),
            'below_floor': all(h < -EPS for h in heights),
            'operator_height_m': selected['height'] if selected else heights[0],
            'posture': 'low/stooped allowed' if selected and selected['height'] < low else 'upright reference' if selected else None,
            'candidate_samples': checked, 'free_candidates': len(candidates), 'connected_candidates': len(connected),
            'board_rectangles': len(boards[level]), 'navigation_has_entry': bool(len(nav[level][0])),
            'scope': 'Use the standing envelope and actual deck configuration space. Crouching is allowed.'}


def _check_one(valve, scene, model, work, original_qpos, obstacles, nav, boards):
    asset, prefix, root, joints, own_ids, moving = _mapping(valve, model)
    own = set(own_ids)
    all_other = [o for o in obstacles if o['geom_id'] not in own]
    envelope = _motion_envelope(model, work, root, moving)
    other = [o for o in all_other if _hits(envelope, np.asarray(o['box'])[None, :])[0]]
    other_boxes = np.asarray([o['box'] for o in other]).reshape(-1, 6)
    count = int(math.ceil(max(abs(j['closed']-j['open']) for j in joints.values())/STEP_RAD))+1
    if count < 2:
        raise OperationUnavailable('The native operator has no nonzero travel')
    work.qpos[:] = original_qpos
    hits, touches, measurements = {}, {}, {}
    backend = {}; distance_queries = 0; fallback_rechecks = 0
    swept_per_geom = None
    actual_samples = 0
    for index, fraction in enumerate(np.linspace(0., 1., count)):
        for joint in joints.values():
            work.qpos[joint['address']] = joint['open']+fraction*(joint['closed']-joint['open'])
        mujoco.mj_kinematics(model, work)
        ports = _actual_ports(asset, work, root)
        moving_boxes = _bounds(model, work, moving)
        if swept_per_geom is None:
            swept_per_geom = moving_boxes.copy()
        else:
            swept_per_geom[:, :3] = np.minimum(swept_per_geom[:, :3], moving_boxes[:, :3])
            swept_per_geom[:, 3:] = np.maximum(swept_per_geom[:, 3:], moving_boxes[:, 3:])
        actual_samples += 1
        pose_cache = {}  # Moving mesh support data cannot leak across poses.
        for gid, box in zip(moving, moving_boxes):
            for obstacle_index in np.flatnonzero(_hits(box, other_boxes, POLICY['sweep_margin_m'])):
                obstacle = other[obstacle_index]
                distance = _actual_distance(model, work, int(gid), obstacle['geom_id'], box,
                    np.asarray(obstacle['box']), POLICY['sweep_margin_m'], backend, pose_cache, index)
                distance_queries += 1
                fallback_rechecks += int(distance['fallback_rechecked'])
                key = (int(gid), obstacle['geom_id'])
                observation = {'moving_geom': model.geom(gid).name, 'obstacle': obstacle['id'],
                               'kind': obstacle['kind'], 'owner': obstacle['owner'],
                               'sample': index, 'fraction': float(fraction), **distance}
                if key not in measurements or distance['distance_m'] < measurements[key]['distance_m']:
                    measurements[key] = observation
                if distance['distance_m'] >= POLICY['sweep_margin_m']-EPS:
                    continue
                contact = _end_contact(box, obstacle, model, work, ports, POLICY['sweep_margin_m'])
                if contact:
                    touches[(contact['port'], contact['component'])] = contact
                    continue
                if key not in hits or distance['distance_m'] < hits[key]['distance_m']:
                    hits[key] = {**observation, 'moving_box': box.tolist(),
                                 'obstacle_box': obstacle['box'],
                                 'no_contact_geometry': bool(obstacle['no_contact_geometry']),
                                 'moving_no_contact_geometry': not bool(model.geom_contype[gid] or model.geom_conaffinity[gid])}
    # Grip positions are read in the package's real fully-open pose.
    for joint in joints.values():
        work.qpos[joint['address']] = joint['open']
    mujoco.mj_kinematics(model, work)
    grips, grip_basis = _grips(asset, work, root)
    stance_obstacles = obstacles + [{'id': 'sweep_'+model.geom(gid).name, 'box': box.tolist()}
                                    for gid, box in zip(moving, swept_per_geom)]
    stance = _stance(valve, grips, scene, stance_obstacles, nav, boards)
    reasons = []
    if not stance['stance_free']:
        reasons.append('No obstacle-free standing area on a complete deck surface')
    elif not stance['reachable_from_passage']:
        reasons.append('Standing area is disconnected from the physical entrance')
    if stance['needs_elevation']:
        reasons.append('Grip exceeds the permitted operating height and needs elevated access')
    if stance['below_floor']:
        reasons.append('Grip is below the operating deck surface')
    if hits:
        reasons.append('The sampled moving parts have less than 25 mm minimum clearance through travel')
    return {'id': valve['id'], 'name': valve.get('name', valve['id']), 'role': valve.get('role', 'unspecified'),
            'medium': valve.get('medium'), 'level': valve.get('level'), 'group_id': valve.get('group_id'),
            'equipment': valve.get('equipment'), 'edge': valve.get('edge'), 'dn': asset.get('dn'),
            'operator': asset.get('operator'), 'package': asset.get('package'), 'version': asset.get('version'),
            'review_status': deepcopy(asset.get('review_status')), 'no_contact_geometry': asset.get('no_contact_geometry'),
            'actual_model': {'root_body': model.body(root).name, 'prefix': prefix,
                             'ngeom': int(model.ngeom), 'njnt': int(model.njnt),
                             'package_model_xml': asset.get('model_xml'),
                             'source': "Use the caller's loaded MjModel and verify native joint names, ranges and geoms"},
            'status': 'eligible' if not reasons else 'ineligible', 'eligible': not reasons,
            'reason': '；'.join(reasons) or 'Static operation geometry passes with crouching allowed',
            **stance, 'grip_basis': grip_basis, 'motion_available': True, 'sweep_clear': not hits,
            'sweep_obstacles': sorted({item['obstacle'] for item in hits.values()}),
            'sweep': {'method': 'Use native joints and 5-degree samples. AABBs screen before checking actual moving geoms and the 25 mm margin.',
                      'samples': actual_samples, 'expected_samples': count, 'max_joint_step_deg': max(abs(j['closed']-j['open']) for j in joints.values())/(count-1)*180/math.pi,
                      'ranges': {name: {'actual_joint': prefix+name, 'open_rad': j['open'], 'closed_rad': j['closed']} for name, j in joints.items()},
                      'moving_geoms': len(moving), 'all_package_geoms': len(own_ids),
                      'obstacles_total': len(all_other), 'obstacles_checked': len(other),
                      'broad_phase_all_angle_box': envelope.tolist(), 'obstacles': list(hits.values()),
                      'actual_distance_queries': distance_queries, 'zero_native_fallback_rechecks': fallback_rechecks,
                      'minimum_actual_distances_by_pair': list(measurements.values()),
                      'expected_interface_contacts': list(touches.values()), 'swept_bbox': _union(swept_per_geom),
                      'self_assembly_exclusion': 'Only the assembly of this native valve is exempt from self-checks',
                      'physics_steps': 0},
            'frozen_operation_side': _frozen_hint(valve, scene, stance['stance'])}


def check_operations(graph, scene, model, data):
    """Report per-valve geometric eligibility; never a 100%-room gate.

    ``graph.valves`` contains placed asset records (or ``asset`` records)
    with id/role/level and optionally the native attach ``prefix``. Missing
    actual motion is an explicit error row, never an invented sweep box.
    """
    environment = require_environment()
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    work.mocap_pos[:] = data.mocap_pos
    work.mocap_quat[:] = data.mocap_quat
    original_qpos = work.qpos.copy()
    mujoco.mj_kinematics(model, work)
    obstacles, measured = _environment(scene, model, work)
    valves = graph.get('valves', [])
    nav, boards = _navigation(scene, measured, obstacles, valves)
    rows = []
    for valve in valves:
        work.qpos[:] = original_qpos
        mujoco.mj_kinematics(model, work)
        try:
            row = _check_one(valve, scene, model, work, original_qpos, obstacles, nav, boards)
        except (OperationUnavailable, KeyError, TypeError, ValueError) as exc:
            row = {'id': valve.get('id'), 'name': valve.get('name'), 'role': valve.get('role', 'unspecified'),
                   'status': 'error', 'eligible': False, 'motion_available': False,
                   'grip': None, 'stance': None, 'sweep_clear': None,
                   'reason': f'Physical motion or operation interface is unavailable: {type(exc).__name__}: {exc}'}
        rows.append(row)
    by_role = {}
    for row in rows:
        group = by_role.setdefault(row['role'], {'total': 0, 'eligible': 0, 'errors': 0})
        group['total'] += 1
        group['eligible'] += int(row['eligible'])
        group['errors'] += int(row['status'] == 'error')
    for group in by_role.values():
        group['fraction'] = group['eligible']/group['total']
    total, eligible = len(rows), sum(row['eligible'] for row in rows)
    return {'status': 'reported', 'total': total, 'eligible': eligible,
            'fraction': eligible/total if total else None,
            'errors': sum(row['status'] == 'error' for row in rows), 'by_role': by_role, 'rows': rows,
            'scope': 'P1 and P2 static operation geometry uses 5-degree native joint samples and a 25 mm clearance. Report valve availability and package status separately.',
            'environment': environment, 'physics_steps': 0, 'source_mutated': False}
