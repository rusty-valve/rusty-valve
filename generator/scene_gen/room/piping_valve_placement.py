"""Try valve positions and handle directions along a pipe."""
from __future__ import annotations
from .root_orientation import planning_geoms

from copy import deepcopy
import math
import numpy as np

from .piping_assets import inspect_asset, place_asset
from .piping_build import _unit


EPS = 1e-8


def physical_valve_obstacles(a, exclude_name=None):
    """Installed pieces only, including other planned real-package poses."""
    parts = [c for c in a.scene['components']
             if not c.get('annotation_only') and not c.get('visual_only')]
    for valve in a.all_valves():
        if valve['id'] == exclude_name:
            continue
        parts.extend({'id': valve['prefix']+g['name'], 'kind': 'valve',
                      'shape': 'box', 'box': g['box']} for g in planning_geoms(valve))
    for record in a.scene.get('pullout_reservations', []):
        parts.append({'id': record['id'], 'kind': 'pullout_reservation', 'shape': 'box',
                      'box': [record['rect'][0], record['rect'][1], record['base_z'],
                              record['rect'][2], record['rect'][3], a.bounds[5]]})
    return parts


def obstacle_boxes(obstacles, contact_ids=()):
    excluded = set(contact_ids)
    parts = [part for part in obstacles if part['id'] not in excluded]
    return np.asarray([part['id'] for part in parts]), np.asarray(
        [part['box'] for part in parts], dtype=float).reshape(-1, 6)


def overlapping_obstacle_ids(geoms, obstacle_ids, boxes):
    """Same positive overlap on all three axes as the per-box check."""
    hit = np.zeros(len(boxes), dtype=bool)
    for geom in geoms:
        bb = geom['box']
        hit |= np.all(np.minimum(bb[3:], boxes[:, 3:])-np.maximum(bb[:3], boxes[:, :3]) > 1e-5, axis=1)
    return sorted(set(obstacle_ids[hit].tolist()))


def _real_candidate(a, name, dn, position, axis, stem, medium, role, level, metadata):
    asset = inspect_asset(int(dn))
    # Inspected ports lie on the native +X flow axis. Computing just that
    # datum avoids building a second copy of every native mesh envelope.
    center = np.asarray(position)-axis*asset['local_ports']['in']['position'][0]
    valve = place_asset(asset, center, axis, stem)
    valve.update(id=name, name=name, role=role, dn=int(dn), level=int(level), medium=medium,
                 geom_prefix=name+'_', prefix=name+'_', **deepcopy(metadata))
    valve['root_pipe_radius_m'] = a.radius(dn)
    return valve


def real_valve_candidates(a, name, dn, inlet, axis, preferred_stem, medium, role, level,
                          *, in_positions=None, metadata=None, obstacles=None, contact_ids=()):
    """Return pure real-pose candidates in shared slide-before-turn order.

    in_positions are permissible inlet points on this same straight pipe.
    The caller must keep genuine package/end-spool room and, after choosing,
    update both pipe endpoints before a.reserve_valve(candidate['valve']).
    A physical_fit is an installed-pose box screen, not final collision proof.
    """
    axis = _unit(axis); stem = _unit(preferred_stem); origin = np.asarray(inlet, dtype=float)
    if np.count_nonzero(np.abs(axis)>EPS) != 1 or abs(float(axis@stem))>EPS:
        raise ValueError('Valve candidates need an orthogonal pipe axis and a perpendicular default lever axis')
    positions = [origin] if in_positions is None else [np.asarray(p, dtype=float) for p in in_positions]
    unique = []
    for position in positions:
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Candidate inlet must be finite 3D coordinates')
        delta = position-origin; shift = float(delta@axis)
        if np.linalg.norm(delta-axis*shift)>EPS:
            raise ValueError('An axial offset must stay on its original straight')
        if not any(np.max(abs(position-old))<=EPS for old in unique):
            unique.append(position)
    if not unique:
        raise ValueError('At least one permitted physical valve inlet is required')
    obstacles = physical_valve_obstacles(a, name) if obstacles is None else obstacles
    obstacle_ids, boxes = obstacle_boxes(obstacles, contact_ids)
    result = []; metadata = metadata or {}
    for quarter in range(4):
        for position in unique:
            valve = _real_candidate(a, name, dn, position, axis, stem, medium, role, level, metadata)
            from .rule_identity import annotate_valve
            annotate_valve(a,valve)
            hits = overlapping_obstacle_ids(planning_geoms(valve), obstacle_ids, boxes)
            shift = float((position-origin)@axis)
            evidence = {'method': 'shared_slide_then_handle_rotation_v1',
                        'stage': 'along_pipe' if quarter == 0 else 'handle_rotation',
                        'axial_shift_m': shift, 'stem_rotation_quarters': quarter,
                        'physical_fit': not hits, 'physical_hits': hits,
                        'operation_eligibility_gate': False,
                        'scope': 'Screen native installed geometry. Evaluate travel and stance afterward. Actual geometry determines penetration.'}
            valve['method_valve_placement'] = deepcopy(evidence)
            result.append({'valve': valve, 'in_position': deepcopy(valve['world_ports']['in']['position']),
                           'out_position': deepcopy(valve['world_ports']['out']['position']), **evidence})
        stem = _unit(np.cross(axis, stem))
    return sorted(result,key=lambda c:c['valve']['operation_preference']['rank'])


def forward_pipe_positions(a, dn, inlet, axis, *, max_steps=3):
    """Finite positions along an extendable straight, derived from the real OD/L.

    The caller still checks the complete changed lead and tail. This does not
    move a riser, widen a bundle, add a valve or prescribe a stagger by medium.
    """
    axis = _unit(axis); inlet = np.asarray(inlet, dtype=float)
    asset = inspect_asset(dn)
    gap = a.config['pg_rules']['parallel_pipe_spacing']['protrusions']['min_clearance_mm']/1000.
    step = max(asset['install_length_m'], 2*a.radius(dn))+gap
    points = []
    for index in range(max_steps+1):
        point = inlet+axis*(step*index)
        if all(a.bounds[k]-EPS <= point[k] <= a.bounds[k+3]+EPS for k in range(3)):
            points.append(point.tolist())
    return points


def orthogonal_connection_faces(axis):
    """Try a requested face, then the other transverse horizontal faces.

    Upward/downward roots retain their original axis: callers cannot use this
    to convert a top root into a horizontal valve or add a pre-root elbow.
    """
    axis = _unit(axis)
    if abs(axis[2])>EPS:
        return [axis]
    side = _unit(np.cross([0., 0., 1.], axis))
    return [axis, side, -side, -axis]
