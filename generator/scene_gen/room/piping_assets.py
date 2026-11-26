"""Load the specified valve model and measure its ports and operator travel."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import itertools
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

from .evaluation_common import ROOT, require_environment
from .runtime_paths import RUNTIME


class AssetUnavailable(ValueError):
    """No exact supported package or a required physical interface is unknown."""


SUPPORTED_DNS = (25, 32, 40, 50, 65, 80, 100, 150, 200, 250, 300)
_LOCKED = {
    **{(dn, 'lever'):(f'bray-7000-dn{dn:03}-npt-lever', 'v0.2')
       for dn in (25, 32, 40, 50, 65)},
    **{(dn, 'lever'):(f'bray-7000-dn{dn:03}-npt-lever', 'v0.6') for dn in (80, 100)},
    **{(dn, 'wheel'):(f'bray-7100-dn{dn:03}-npt-ftg42l', 'v0.3')
       for dn in (50, 65, 80, 100)},
    (150, 'lever'):('bray-f30-dn150-class300-rf-lever', 'v0.9'),
    (200, 'lever'):('bray-f30-dn200-class300-rf-lever', 'v0.9'),
    (150, 'wheel'):('cameron-t31-dn150-class600-rf-dt21', 'v0.6'),
    (200, 'wheel'):('cameron-t31-dn200-class600-rf-dt40', 'v0.7'),
    (250, 'wheel'):('cameron-t31-dn250-class600-rf-dt40', 'v0.3'),
    (300, 'wheel'):('cameron-t31-dn300-class600-rf-dt54', 'v0.4'),
}
_REVIEW_SOURCES = ['Reviewed valve catalogue',
                   'Catalogue geometry and assembly review']
_PIPING_AUTHORITY = 'PG §3.1'


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise AssetUnavailable(f'{label}Missing positive physical dimension')
    return float(value)


def _vector(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AssetUnavailable(f'{label}must have finite three-dimensional coordinates') from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise AssetUnavailable(f'{label}must have finite three-dimensional coordinates')
    return result


def _bbox(center, matrix, aabb):
    c = center + matrix @ aabb[:3]
    half = abs(matrix) @ aabb[3:]
    return np.r_[c-half, c+half].tolist()


def _union(boxes):
    if not boxes:
        raise AssetUnavailable('Package has no inspectable physical geometry')
    values = np.asarray(boxes)
    return np.r_[values[:, :3].min(axis=0), values[:, 3:].max(axis=0)].tolist()


def _ancestors(model, body, root):
    result = []
    while body:
        result.append(body)
        if body == root:
            return result
        body = int(model.body_parentid[body])
    raise AssetUnavailable('Geometry outside valve_base cannot be included by the native attachment')


def _review(package_id, params):
    returned = package_id == 'cameron-t31-dn300-class600-rf-dt54'
    human = package_id.startswith('bray-f30-')
    return {
        'status':'returned' if returned else 'human_appearance_accepted' if human else 'static_review_passed',
        'human_status':'appearance_accepted' if human else 'not_accepted' if returned else 'pending_human_acceptance',
        'review_scope':'Appearance and static assembly only. Contact and forces have separate results.',
        'package_declared_scope':params.get('review_scope', params.get('candidate_scope')),
        'dynamic_contact_validated':False,
        'sources':_REVIEW_SOURCES + (['Manual appearance review'] if human else []),
        'geometry_training_allowed':True,
        'geometry_training_authority':_PIPING_AUTHORITY,
        'not_automatic_room_refusal':True,
        'unresolved':['DT54 handwheel center is 549 mm versus the 512 mm catalogue value, a 37 mm difference. Keep the rejected package intact.'] if returned else [],
    }


def _motion(model, data, root_rotation, root_position, operator, params):
    names = {model.joint(i).name:i for i in range(model.njnt)}
    valve_name = 'valve_hinge' if operator == 'lever' else 'ball_hinge'
    operator_name = valve_name if operator == 'lever' else 'wheel_hinge'
    for name in (valve_name, operator_name):
        if name not in names:
            raise AssetUnavailable(f'Package lacks an operating joint{name}')
    valve = names[valve_name]
    if int(model.jnt_type[valve]) != int(mujoco.mjtJoint.mjJNT_HINGE) or not model.jnt_limited[valve]:
        raise AssetUnavailable('The valve core does not have a finite physical rotation range')
    source = params.get('motion', {})
    if operator == 'lever':
        try:
            opened, closed = (math.radians(float(source[key])) for key in ('open_deg', 'closed_deg'))
        except (KeyError, TypeError, ValueError) as exc:
            raise AssetUnavailable('Lever package lacks open and closed angles') from exc
        declared_axis = source.get('axis')
    else:
        opened = 0.
        closed = math.radians(_positive(source.get('ball_travel_deg'), 'original ball travel'))
        declared_axis = source.get('valve_axis')
    if not np.allclose(_vector(declared_axis, 'original valve shaft axis'), model.jnt_axis[valve], atol=1e-8):
        raise AssetUnavailable('The parameter shaft axis differs from the XML joint axis')
    lo, hi = (float(v) for v in model.jnt_range[valve])
    if not (math.isfinite(opened) and math.isfinite(closed) and lo-1e-8 <= opened <= hi+1e-8
            and lo-1e-8 <= closed <= hi+1e-8 and abs(closed-opened) > 1e-8):
        raise AssetUnavailable('Original open or closed angle falls outside the XML joint range')
    opened_state = {valve_name:opened}; closed_state = {valve_name:closed}; equalities = []
    wheel_range = None
    for i in range(model.neq):
        kind = int(model.eq_type[i]); one = int(model.eq_obj1id[i]); two = int(model.eq_obj2id[i])
        record = {'name':mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i),
                  'type':kind, 'object1':one, 'object2':two, 'data':model.eq_data[i].tolist(),
                  'active_at_start':bool(model.eq_active0[i])}
        if kind == int(mujoco.mjtEq.mjEQ_JOINT):
            record.update(joint1=model.joint(one).name, joint2=model.joint(two).name if two >= 0 else None)
        equalities.append(record)
    if operator == 'wheel':
        wheel = names[operator_name]
        if int(model.jnt_type[wheel]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise AssetUnavailable('The handwheel joint is not a hinge')
        if not np.allclose(_vector(source.get('wheel_axis'), 'original handwheel axis'), model.jnt_axis[wheel], atol=1e-8):
            raise AssetUnavailable('Parameter handwheel axis differs from XML')
        equations = [e for e in equalities if e['type'] == int(mujoco.mjtEq.mjEQ_JOINT)
                     and e['active_at_start'] and {e['object1'], e['object2']} == {valve, wheel}]
        if len(equations) != 1:
            raise AssetUnavailable('Handwheel and core lack a unique physical coupling')
        equation = equations[0]; coefficients = equation['data'][:5]
        if abs(coefficients[1]) < 1e-12 or any(abs(c) > 1e-12 for c in coefficients[2:]):
            raise AssetUnavailable('The adapter supports linear handwheel coupling only')
        if equation['object1'] == valve:
            wheel_open, wheel_closed = [(q-coefficients[0])/coefficients[1] for q in (opened, closed)]
        else:
            wheel_open, wheel_closed = [coefficients[0]+coefficients[1]*q for q in (opened, closed)]
        ratio = abs((wheel_closed-wheel_open)/(closed-opened))
        if not math.isclose(ratio, _positive(source.get('gear_ratio'), 'original gear ratio'), rel_tol=1e-6):
            raise AssetUnavailable('The XML coupling differs from the original gear ratio')
        opened_state[operator_name] = wheel_open; closed_state[operator_name] = wheel_closed
        wheel_range = [wheel_open, wheel_closed]
    joints = []
    for name, i in names.items():
        joints.append({'name':name, 'type':int(model.jnt_type[i]), 'body':model.body(int(model.jnt_bodyid[i])).name,
            'axis':(root_rotation.T @ data.xaxis[i]).tolist(),
            'anchor':(root_rotation.T @ (data.xanchor[i]-root_position)).tolist(),
            'native_body_axis':model.jnt_axis[i].tolist(), 'native_body_anchor':model.jnt_pos[i].tolist(),
            'range_rad':model.jnt_range[i].tolist(), 'limited':bool(model.jnt_limited[i]),
            'qpos_address':int(model.jnt_qposadr[i]), 'dof_address':int(model.jnt_dofadr[i])})
    return {'available':True, 'missing':[], 'params_motion':deepcopy(source), 'joints':joints,
            'equalities':equalities, 'valve_joint':valve_name, 'operator_joint':operator_name,
            'valve_range_rad':[opened, closed], 'operator_range_rad':wheel_range or [opened, closed],
            'operator_range_source':'Finite valve core travel converted through the active linear XML equality. The handwheel joint itself may be unlimited.' if operator == 'wheel' else 'parameter angles and finite lever-joint travel',
            'joint_states':{'open':opened_state, 'closed':closed_state},
            'native_actuator_count':int(model.nu), 'physics_steps':0,
            'scope':'Kinematics from the original XML joints and equalities'}


def _ports(params, geoms):
    geometry = params['geometry']
    length_key = 'face_to_face_mm' if 'face_to_face_mm' in geometry else 'body_length_mm'
    length = _positive(geometry.get(length_key), length_key)/1000
    if 'shaft_from_left_face_mm' in geometry:
        left = -_positive(geometry['shaft_from_left_face_mm'], 'shaft_from_left_face_mm')/1000
        right = left+length
        if right <= 0:
            raise AssetUnavailable('Stem-to-face distance exceeds installed length')
        basis = 'params shaft_from_left_face_mm + '+length_key+'. Asymmetric faces'
    else:
        # The locked families use pipe axis +X at y=z=0. Confirm the actual
        # fixed body/ends instead of silently assuming a midpoint at zero.
        axial = [g['bbox'] for g in geoms if not g['moving'] and
                 g['bbox'][1] <= 1e-8 <= g['bbox'][4]+1e-8 and
                 g['bbox'][2] <= 1e-8 <= g['bbox'][5]+1e-8]
        if not axial:
            raise AssetUnavailable('No fixed axial end geometry to locate the ports')
        left = min(b[0] for b in axial); measured_right = max(b[3] for b in axial)
        if not math.isclose(measured_right-left, length, rel_tol=0., abs_tol=2e-6):
            raise AssetUnavailable(f'measured fixed end span{measured_right-left:g} m versus {length_key}={length:g} m. Port faces cannot be located.')
        # Use exact catalog length after real geometry verified its end span.
        right = left+length
        basis = length_key+' and measured fixed axial end span, including asymmetric end positions'
    return length, {'in':{'position':[left, 0., 0.], 'normal':[-1., 0., 0.]},
                    'out':{'position':[right, 0., 0.], 'normal':[1., 0., 0.]}}, basis


@lru_cache(maxsize=None)
def _inspect(dn, operator):
    package_id, version = _LOCKED[(dn, operator)]
    folder = RUNTIME / 'valve_library/packages' / package_id / version
    relative = Path(os.path.relpath(folder, ROOT))
    try:
        params = json.loads((folder/'params.json').read_text())
        if params['id'] != package_id or params['version'] != version:
            raise AssetUnavailable('Package ID or version differs from the explicit pin')
        stated_dn = params.get('product', {}).get('nominal_size_dn')
        if stated_dn is not None and stated_dn != dn:
            raise AssetUnavailable('Package DN differs from requested DN')
        model = mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, AssetUnavailable):
            raise
        raise AssetUnavailable(f'Could not load the pinned package for this DN{relative}：{exc}') from exc
    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'valve_base')
    if root_body < 0:
        raise AssetUnavailable('Package has no physical valve_base attachment root')
    root_rotation = data.xmat[root_body].reshape(3, 3)
    root_position = data.xpos[root_body]
    geoms = []
    for i in range(model.ngeom):
        ancestry = _ancestors(model, int(model.geom_bodyid[i]), root_body)
        moving_joints = [model.joint(j).name for b in ancestry
                         for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]+model.body_jntnum[b]))]
        center = root_rotation.T @ (data.geom_xpos[i]-root_position)
        matrix = root_rotation.T @ data.geom_xmat[i].reshape(3, 3)
        if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[i]); start = int(model.mesh_vertadr[mesh]); count = int(model.mesh_vertnum[mesh])
            points = np.asarray(model.mesh_vert[start:start+count]) @ matrix.T + center
            box = np.r_[points.min(axis=0), points.max(axis=0)].tolist()
        else:
            box = _bbox(center, matrix, model.geom_aabb[i])
        geoms.append({'name':model.geom(i).name, 'type':int(model.geom_type[i]),
            'body':model.body(int(model.geom_bodyid[i])).name,
            'center':center.tolist(), 'matrix':matrix.tolist(), 'size':model.geom_size[i].tolist(),
            'bbox':box,
            'moving':bool(moving_joints), 'moving_joints':moving_joints,
            'contype':int(model.geom_contype[i]), 'conaffinity':int(model.geom_conaffinity[i]),
            'contact_enabled':bool(model.geom_contype[i] or model.geom_conaffinity[i]),
            'rgba':(model.mat_rgba[model.geom_matid[i]] if model.geom_matid[i] >= 0 else model.geom_rgba[i]).tolist()})
    length, ports, basis = _ports(params, geoms)
    motion = _motion(model, data, root_rotation, root_position, operator, params)
    no_contact = not any(g['contact_enabled'] for g in geoms)
    motion['no_contact_geometry'] = no_contact
    return {'dn':dn, 'operator':operator, 'package':relative.as_posix(), 'package_id':package_id,
        'version':version, 'model_xml':str(folder/'model.xml'), 'params':params,
        'install_length_m':length, 'length':length, 'local_ports':ports, 'port_basis':basis,
        'root_body':'valve_base', 'review_status':_review(package_id, params), 'motion':motion,
        'motion_available':True, 'no_contact_geometry':no_contact,
        'contact_geom_count':sum(g['contact_enabled'] for g in geoms), 'local_geoms':geoms,
        'local_bbox':_union([g['bbox'] for g in geoms]),
        'moving_parts':[g['name'] for g in geoms if g['moving']],
        'fixed_parts':[g['name'] for g in geoms if not g['moving']],
        'native_qpos0':model.qpos0.tolist(), 'scale':[1., 1., 1.],
        'local_pipe_axis':[1., 0., 0.], 'local_stem_direction':[0., 0., 1.],
        'coordinate_basis':'Valve-base coordinates use the original +X pipe axis and +Z stem axis. Joint-axis signs are retained.',
        'bbox_configuration':'Conservative local AABB of all declared geoms at model qpos0',
        'geometry_scope':'Include visual and zero-contact-mask geometry. Geometry without contact masks remains visual.',
        'references':{'params':str(folder/'params.json'), 'model':str(folder/'model.xml'),
                      'package_sources':deepcopy(params.get('sources')), 'review':_REVIEW_SOURCES,
                      'piping_authority':_PIPING_AUTHORITY}, 'physics_steps':0}


def inspect_asset(dn, operator=None):
    """Return a detached JSON-serializable asset record or AssetUnavailable.

    No fallback changes a requested operator or DN. None chooses lever through
    DN200 and the actual locked wheel package for DN250/300.
    """
    require_environment()
    if isinstance(dn, bool) or not isinstance(dn, (int, np.integer)) or int(dn) not in SUPPORTED_DNS:
        raise AssetUnavailable(f'No pinned valve package for DN {dn!r}. Scaling and substitutions across nominal diameters are excluded.')
    dn = int(dn)
    operator = ('wheel' if dn in (250, 300) else 'lever') if operator is None else operator
    if not isinstance(operator, str) or (dn, operator) not in _LOCKED:
        raise AssetUnavailable(f'DN{dn}No pinned {operator!r} package with this operator type')
    return deepcopy(_inspect(dn, operator))


def _world_box(box, matrix, origin):
    corners = np.array(list(itertools.product(*zip(box[:3], box[3:]))))
    points = corners @ matrix.T + origin
    return np.r_[points.min(axis=0), points.max(axis=0)].tolist()


def flow_ports(asset, reversed_ends=False):
    """Return the package ends in process-flow order."""
    ports = asset['local_ports']
    if not reversed_ends:
        return ports
    if not asset['package_id'].startswith(('bray-7000-', 'bray-7100-', 'bray-f30-', 'cameron-t31-')):
        raise AssetUnavailable('The package has no verified interchangeable ends')
    return {'in': ports['out'], 'out': ports['in']}


def place_asset(asset, center, axis, stem, *, reversed_ends=False):
    """Rigidly place the package origin; center is NOT the port midpoint.

    F30's two port distances from this origin can differ. Callers must anchor
    a known local port when joining a pipe, rather than assume +/-length/2.
    """
    require_environment()
    origin = _vector(center, 'center'); x = _vector(axis, 'axis'); z = _vector(stem, 'stem')
    if np.linalg.norm(x) < 1e-12 or np.linalg.norm(z) < 1e-12:
        raise AssetUnavailable('Pipe or stem direction is zero')
    x = x/np.linalg.norm(x); z = z/np.linalg.norm(z)
    if abs(float(np.dot(x, z))) > 1e-6:
        raise AssetUnavailable('Pipe and stem directions must be perpendicular')
    y = np.cross(z, x); y = y/np.linalg.norm(y); z = np.cross(x, y)
    matrix = np.column_stack((x, y, z)); quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix.ravel())
    if asset.get('scale') != [1., 1., 1.] or not asset.get('motion_available'):
        raise AssetUnavailable('An unscaled inspect_asset record with complete physical motion metadata is required')
    # A previous pose carries large native geometry records that are all
    # replaced below. Keep independent metadata without copying stale poses.
    rebuilt = {'center','origin','axis','stem','quat','matrix','world_ports','ports',
               'world_bbox','world_geoms','world_joint_frames','placement_center_semantics'}
    result = deepcopy({k:v for k,v in asset.items() if k not in rebuilt})
    ports = {name:{'position':(origin+matrix @ np.asarray(port['position'])).tolist(),
                   'normal':(matrix @ np.asarray(port['normal'])).tolist()}
             for name, port in flow_ports(asset, reversed_ends).items()}
    result.update(center=origin.tolist(), origin=origin.tolist(), axis=x.tolist(), stem=z.tolist(),
                  quat=quat.tolist(), matrix=matrix.tolist(), world_ports=ports, ports=ports,
                  world_bbox=_world_box(asset['local_bbox'], matrix, origin),
                  placement_center_semantics='Use the original valve_base origin under a rigid transform')
    if reversed_ends:
        result.update(reversed_ends=True, axis=(-x).tolist())
    result['world_geoms'] = [{**deepcopy(g), 'box':_world_box(g['bbox'], matrix, origin),
        'center':(origin+matrix @ np.asarray(g['center'])).tolist(),
        'matrix':(matrix @ np.asarray(g['matrix'])).tolist()} for g in asset['local_geoms']]
    result['world_joint_frames'] = [{'name':j['name'],
        'anchor':(origin+matrix @ np.asarray(j['anchor'])).tolist(),
        'axis':(matrix @ np.asarray(j['axis'])).tolist()} for j in asset['motion']['joints']]
    return result
