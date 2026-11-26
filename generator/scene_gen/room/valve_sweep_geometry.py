"""Check moving valve parts for interference with nearby geometry."""
from dataclasses import dataclass
from functools import cached_property
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from .export import nums
from .piping_operations import _grips, _mapping
from .valve_position_rules import EPS


GRIP_PREFERRED = .05
BARE_CLEARANCE = .010
SKIN_THICKNESS = .05


@dataclass
class GripClearance:
    distance: float = math.inf

    @property
    def tight(self):
        return self.distance < GRIP_PREFERRED - EPS


@dataclass
class Sweep:
    ids: np.ndarray
    positions: np.ndarray
    matrices: np.ndarray
    bounds: np.ndarray
    moving: np.ndarray
    grip: np.ndarray
    name: tuple

    @cached_property
    def physical(self):
        # The clipped lever grip is a measuring surface. The native lever
        # mesh, present separately, supplies the actual collision geometry.
        return np.asarray([name != 'hold' for name in self.name])

    @cached_property
    def span(self):
        return np.r_[self.bounds[..., :3].min(axis=(0, 1)), self.bounds[..., 3:].max(axis=(0, 1))]

    @cached_property
    def operator_center(self):
        """Mean center of the native operator solids through the sampled travel."""
        operator = np.array([name.startswith(('visual_lever', 'lever_', 'visual_handwheel',
                                             'rim_', 'spoke_')) for name in self.name])
        active = operator & self.moving & self.physical
        if not active.any():
            active = self.moving & self.physical
        centers = self.positions[:, active].mean(axis=1)
        weights = np.ones(len(centers))
        weights[[0, -1]] = .5
        return np.average(centers, axis=0, weights=weights)

    def placed(self, matrix, origin):
        positions = self.positions @ matrix.T + origin
        centers = (self.bounds[..., :3] + self.bounds[..., 3:]) / 2
        half = (self.bounds[..., 3:] - self.bounds[..., :3]) / 2 @ abs(matrix.T)
        centers = centers @ matrix.T + origin
        return Sweep(self.ids, positions, matrix @ self.matrices,
                     np.concatenate((centers - half, centers + half), axis=-1),
                     self.moving, self.grip, self.name)


def select_geometries(model, ids, moving):
    """Use visual parts and the package's individually modelled rim and spokes."""
    selected = {}
    names = {model.geom(i).name.rsplit('__', 1)[-1] for i in ids}
    for gid in ids:
        name = model.geom(gid).name.rsplit('__', 1)[-1]
        if 'ball' in name or name.startswith(('visual_seats', 'visual_seat_pair')):
            continue
        if name == 'visual_handwheel' and any(n.startswith('rim_') for n in names):
            continue
        if name.startswith(('lever_grip', 'lever_neck')) and 'visual_lever' in names:
            continue
        if name.startswith(('collar_', 'gearbox', 'stem_', 'travel_stop_plate_', 'lever_root_hub_')):
            continue  # These duplicate the package's external visual solids.
        selected[gid] = (name, gid in moving, name.startswith('rim_'))
    return selected


def lever_hold(asset):
    """Cut the outer third from the real handle surfaces in the open pose."""
    model = mujoco.MjModel.from_xml_path(asset['model_xml'])
    data = mujoco.MjData(model)
    _, _, root, joints, _, moving = _mapping({'id': 'hold', 'prefix': '', **asset}, model)
    for joint in joints.values():
        data.qpos[joint['address']] = joint['open']
    mujoco.mj_kinematics(model, data)
    joint = joints[asset['motion']['operator_joint']]
    pivot, axis = data.xanchor[joint['id']].copy(), data.xaxis[joint['id']].copy()
    side = np.mean(_grips(asset, data, root)[0], axis=0) - pivot
    side -= axis * (side @ axis)
    side /= np.linalg.norm(side)
    triangles = []
    for gid in moving:
        name = model.geom(gid).name
        if not name.startswith(('visual_lever', 'visual_grip', 'visual_handle')) or any(
                word in name for word in ('screw', 'mount', 'socket')):
            continue
        mesh = model.geom_dataid[gid]
        if mesh < 0:
            continue
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        first, faces = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        points = (model.mesh_vert[start:start+count] @ data.geom_xmat[gid].reshape(3, 3).T
                  + data.geom_xpos[gid])
        triangles.append(points[model.mesh_face[first:first+faces]])
    if not triangles:
        raise ValueError('Package lacks an outer lever grip surface that can be extracted: ' + asset['package_id'])
    triangles = np.vstack(triangles)
    distances = (triangles - pivot) @ side
    distances -= distances.max() * (2 / 3)
    cut = [triangles[distances >= 0.]]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        crossed = (distances[:, a] < 0.) != (distances[:, b] < 0.)
        edges = triangles[crossed]
        d0, d1 = distances[crossed, a], distances[crossed, b]
        cut.append(edges[:, a] + (edges[:, b] - edges[:, a]) * (d0 / (d0 - d1))[:, None])
    points = np.unique(np.vstack(cut).round(9), axis=0)
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    # MuJoCo needs consistently wound triangles.
    inner = points[hull.vertices].mean(axis=0)
    a, b, c = points[faces].transpose(1, 0, 2)
    flip = np.einsum('ij,ij->i', np.cross(b-a, c-a), a-inner) < 0
    faces[flip] = faces[flip, ::-1]
    return points, faces, pivot, axis, joint['closed'] - joint['open']


def component_bounds(model, gid, positions, matrices):
    """Axis extrema of the actual sampled surface, in the valve frame."""
    kind = model.geom_type[gid]
    size = model.geom_size[gid]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start+count]
        lo, hi = [], []
        for index in range(0, len(positions), 8):
            points = positions[index:index+8, None] + vertices @ np.swapaxes(matrices[index:index+8], -1, -2)
            lo.extend(points.min(axis=1))
            hi.extend(points.max(axis=1))
        return np.c_[lo, hi]
    axial = abs(matrices[:, :, 2])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        half = axial * size[1] + np.sqrt(np.maximum(0., 1. - axial**2)) * size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half = axial * size[1] + size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        half = np.linalg.norm(matrices * size, axis=-1)
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        half = abs(matrices) @ size
    else:
        raise ValueError('Unknown valve collision shape type: ' + model.geom(gid).name)
    return np.c_[positions - half, positions + half]


class Geometry:
    def __init__(self, source, assets, insulated, parts):
        root = ET.parse(source / 'room.xml').getroot()
        asset_node, world = root.find('asset'), root.find('worldbody')
        for element in list(asset_node):
            if element.tag == 'model':
                asset_node.remove(element)
        covered = set()
        for element in list(world):
            if element.tag != 'geom' or element.get('name') not in parts:
                world.remove(element)
                continue
            part = parts[element.get('name')]
            skin = part['id'] in insulated
            if skin and part['shape'] in ('cylinder', 'ellipsoid', 'mesh'):
                covered.add(part['id'])
            if skin and part['shape'] == 'cylinder':
                element.set('size', str(part['radius'] + SKIN_THICKNESS))
            elif skin and part['shape'] == 'ellipsoid':
                element.set('size', nums(np.asarray(part['size']) + SKIN_THICKNESS))
        for element in root.iter():
            if 'file' in element.attrib:
                element.set('file', str((source / element.attrib['file']).resolve()))
        ET.SubElement(world, 'geom', name='valve_sweep_probe_pipe', type='cylinder', size='.1 .1', pos='0 0 -20')
        ET.SubElement(world, 'geom', name='valve_sweep_probe_box', type='box', size='.1 .1 .1', pos='0 0 -20')
        self.covered = covered
        self.skin = {name for name in covered if parts[name]['shape'] == 'mesh'}
        prefixes, holds = {}, {}
        for index, (key, asset) in enumerate(sorted(assets.items())):
            prefix = f'valve_sweep_probe_{index}__'
            label = f'valve_sweep_probe_asset_{index}'
            ET.SubElement(asset_node, 'model', name=label, file=asset['model_xml'])
            for side in range(2):
                frame = ET.SubElement(world, 'frame', pos='0 0 -20')
                ET.SubElement(frame, 'attach', model=label, body=asset['root_body'], prefix=f'{side}_{prefix}')
            if asset['operator'] == 'lever':
                points, faces, *motion = lever_hold(asset)
                mesh = 'hold_mesh_' + prefix
                ET.SubElement(asset_node, 'mesh', name=mesh, vertex=nums(points.ravel()),
                              face=' '.join(map(str, faces.ravel())))
                for side in range(2):
                    ET.SubElement(world, 'geom', name=f'{side}_{prefix}hold', type='mesh', mesh=mesh)
                holds[key] = motion
            prefixes[key] = prefix
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, self.data)
        self.fixed = {name: gid for name in parts
                      if (gid := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0}
        self.pipe_id = self.model.geom('valve_sweep_probe_pipe').id
        self.box_id = self.model.geom('valve_sweep_probe_box').id
        self.assets = {key: self._sweep(assets[key], prefix, holds.get(key)) for key, prefix in prefixes.items()}
        self.second = {int(one): int(two) for view, pairs in self.assets.values() for one, two in zip(view.ids, pairs)}
        self.peer_cache = {}

    def _sweep(self, asset, prefix, hold):
        model, data = self.model, self.data
        _, _, root, joints, ids, moving = _mapping({'id': prefix, 'prefix': '0_' + prefix, **asset}, model)
        parts = select_geometries(model, ids, moving)
        if hold is not None:
            gid = model.geom('0_' + prefix + 'hold').id
            parts[gid] = ('hold', True, True)
            initial_center = data.geom_xpos[gid].copy()
            initial_matrix = data.geom_xmat[gid].reshape(3, 3).copy()
        ids = np.asarray(list(parts))
        moving = np.array([parts[i][1] for i in ids])
        origin = data.xpos[root].copy()
        changes = [(j['address'], j['open'], math.copysign(min(abs(j['closed'] - j['open']), math.tau),
                                                          j['closed'] - j['open']))
                   for j in joints.values()]
        steps = max(1, math.ceil(max(abs(arc) for _, _, arc in changes) / math.radians(5)))
        points, matrices = [], []
        for fraction in np.linspace(0., 1., steps + 1):
            for address, start, arc in changes:
                data.qpos[address] = start + fraction * arc
            mujoco.mj_kinematics(model, data)
            p = data.geom_xpos[ids] - origin
            m = data.geom_xmat[ids].reshape(-1, 3, 3).copy()
            if hold is not None:
                pivot, axis, arc = hold
                quat = np.empty(4)
                orient = np.empty(9)
                mujoco.mju_axisAngle2Quat(quat, axis, fraction * arc)
                mujoco.mju_quat2Mat(orient, quat)
                rotation = orient.reshape(3, 3)
                p[-1] = pivot + rotation @ (initial_center - pivot)
                m[-1] = rotation @ initial_matrix
            points.append(p)
            matrices.append(m)
        points, matrices = np.asarray(points), np.asarray(matrices)
        bounds = np.empty((len(points), len(ids), 6))
        for index, gid in enumerate(ids):
            selected = slice(None) if moving[index] else slice(0, 1)
            bounds[:, index] = component_bounds(model, gid, points[selected, index], matrices[selected, index])
        pairs = [model.geom('1_' + prefix + parts[i][0]).id for i in ids]
        names = tuple(parts[i][0] for i in ids)
        view = Sweep(ids, points, matrices, bounds, moving, np.array([parts[i][2] or parts[i][0] == 'visual_handwheel' for i in ids]), names)
        return view, np.asarray(pairs)

    def valve(self, asset, matrix, origin):
        return self.assets[(asset['dn'], asset['operator'])][0].placed(matrix, origin)

    def peer(self, valve):
        key = valve['id']
        row = self.peer_cache.get(key)
        if row is None or row[0] is not valve:
            self.peer_cache[key] = (valve, self.valve(valve, np.asarray(valve['matrix']), np.asarray(valve['origin'])))
        return self.peer_cache[key][1]

    def peer_hit(self, first, second, grip=None, *, check_hard=True):
        limit = max(BARE_CLEARANCE, grip.distance if grip is not None else 0.)
        if np.any(np.minimum(first.span[3:] + limit, second.span[3:]) -
                  np.maximum(first.span[:3] - limit, second.span[:3]) <= EPS):
            return None
        al, ah = first.bounds[..., :3], first.bounds[..., 3:]
        bl, bh = second.bounds[..., :3], second.bounds[..., 3:]
        soft = (first.grip[:, None] & second.physical[None, :]
                if grip is not None else np.zeros((len(first.ids), len(second.ids)), dtype=bool))
        hard = check_hard & first.physical[:, None] & second.physical[None, :]
        search = np.where(soft, limit, BARE_CLEARANCE)
        low = np.maximum(al.min(axis=0)[:, None, :] - search[..., None], bl.min(axis=0))
        high = np.minimum(ah.max(axis=0)[:, None, :] + search[..., None], bh.max(axis=0))
        for i, j in np.argwhere((hard | soft) & np.all(high - low > EPS, axis=-1)):
            a = slice(None) if first.moving[i] else slice(0, 1)
            b = slice(None) if second.moving[j] else slice(0, 1)
            threshold = BARE_CLEARANCE if hard[i, j] else 0.
            limit = max(threshold, grip.distance if soft[i, j] else 0.)
            low = np.maximum(al[a, i, None, :] - limit, bl[b, j])
            high = np.minimum(ah[a, i, None, :] + limit, bh[b, j])
            for f, g in np.argwhere(np.all(high - low > EPS, axis=-1)):
                one, two = first.ids[i], self.second[second.ids[j]]
                self._pose(one, first.positions[f, i], first.matrices[f, i])
                self._pose(two, second.positions[g, j], second.matrices[g, j])
                distance = self._distance(one, two, max(threshold, grip.distance if soft[i, j] else 0.))
                if soft[i, j] and math.isfinite(distance):
                    grip.distance = min(grip.distance, distance)
                if hard[i, j] and distance < threshold - EPS:
                    return f'{first.name[i]} / {second.name[j]}'
        return None

    def box_hit(self, sweep, lo, hi, matrix, origin, *, moving_only=True, grip=None):
        half = (hi - lo) / 2
        center = origin + matrix @ ((lo + hi) / 2)
        half_world = abs(matrix) @ half
        self._pose(self.box_id, center, matrix)
        self.model.geom_size[self.box_id, :3] = half
        return self._static_hit(sweep, self.box_id, np.r_[center - half_world, center + half_world],
                                0., 0., moving_only, grip)

    def _pose(self, gid, position, matrix):
        self.data.geom_xpos[gid] = position
        self.data.geom_xmat[gid] = matrix.ravel()

    def _distance(self, one, two, limit):
        ceiling = limit + .002 if math.isfinite(limit) else math.inf
        return mujoco.mj_geomDistance(self.model, self.data, int(one), int(two), ceiling, None)

    def obstacles(self, sweep, names, boxes, *, mating=(), grip=None):
        results = []
        for name, box in zip(names, boxes):
            name = str(name)
            skin = SKIN_THICKNESS if name in self.skin else 0.
            own = name in mating
            hard = skin if own or name in self.covered else BARE_CLEARANCE
            # The caller's bounding box already includes any predicted skin.
            if self._static_hit(sweep, self.fixed[name], box, hard, skin, own, grip):
                results.append(name)
        return results

    def _static_hit(self, sweep, gid, box, threshold, skin, moving_only, grip):
        soft = sweep.grip if grip is not None else np.zeros_like(sweep.grip)
        hard = sweep.physical
        # Boxes supplied by the caller already include the insulation allowance.
        broad = np.where(soft, max(0., threshold-skin, grip.distance if grip is not None else 0.),
                         max(0., threshold-skin))
        region = sweep.bounds
        near = np.all(np.minimum(region[..., 3:] + broad[None, :, None], box[3:]) -
                      np.maximum(region[..., :3] - broad[None, :, None], box[:3]) > EPS, axis=-1)
        near &= hard | soft
        near[1:, ~sweep.moving] = False
        if moving_only:
            near[:, ~sweep.moving] = False
        for frame, part in np.argwhere(near):
            first = sweep.ids[part]
            self._pose(first, sweep.positions[frame, part], sweep.matrices[frame, part])
            limit = max(threshold, skin + grip.distance if soft[part] else 0.)
            distance = self._distance(first, gid, limit)
            if soft[part]:
                grip.distance = min(grip.distance, distance - skin)
            if hard[part] and distance < threshold - EPS:
                return True
        return False

    def pipe_hit(self, sweep, ends, radius, box, *, mating=False, moving_only=False, insulated=False, grip=None):
        axis = np.asarray(ends[1]) - ends[0]
        length = np.linalg.norm(axis)
        axis /= length
        cross = np.eye(3)[np.argmin(abs(axis))]
        cross -= axis * (cross @ axis)
        cross /= np.linalg.norm(cross)
        matrix = np.column_stack((cross, np.cross(axis, cross), axis))
        self._pose(self.pipe_id, np.mean(ends, axis=0), matrix)
        self.model.geom_size[self.pipe_id, :2] = radius, length / 2
        return self._static_hit(sweep, self.pipe_id, box, 0. if insulated or mating else BARE_CLEARANCE,
                                0., mating or moving_only, grip)
