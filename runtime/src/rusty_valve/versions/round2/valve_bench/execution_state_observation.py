"""Read-only physical-state measurements for the reviewed G1 executor.

These functions neither step/forward a simulator nor alter joints, forces,
targets or scores.  They describe current seating and separation; the caller
decides phase changes using the reviewed execution policy.
"""
from __future__ import annotations

import math
from functools import lru_cache

import mujoco
import numpy as np

from .contact_target import load_contact_target_rule
from .forehand_continuity import recess_seating_geometry
from .hook_tool import tool_geometry
from .tool_orientation_context import active_tool_handle_contact


@lru_cache(maxsize=1)
def _allowed_contact_fraction_range():
    rule = load_contact_target_rule()
    return tuple(map(float, rule['target_contact_fraction_range']))


def _finite_handle_recess_intersection(*, tool_origin, tool_rotation, pivot,
        radial, stem, mode, handle_length_m, handle_width_m,
        handle_thickness_m, contact_fraction_range):
    """Convex SAT: the physical allowed handle segment versus recess prism.

    The handle is its actual finite rectangular section swept over the
    permitted axial segment. The recess is the S-profile triangle extruded
    across the actual tool width. Positive intersection is required; merely
    touching a mouth/width boundary is not called nonzero volume. No desired
    contact point is searched, selected or moved.
    """
    cfg, _, plus, minus, _ = tool_geometry()
    triangle = np.asarray(minus[:3] if mode == 'push' else plus[1:4])
    half_tool_width = float(cfg['out_of_plane_width_mm'])*.0005
    rotation = np.asarray(tool_rotation, float)
    base = rotation.T@(np.asarray(pivot, float)-np.asarray(tool_origin, float))
    radial, stem = np.asarray(radial, float), np.asarray(stem, float)
    handle_basis = rotation.T@np.column_stack([radial, np.cross(stem, radial), stem])
    lo, hi = (float(handle_length_m)*f for f in contact_fraction_range)
    handle_vertices = np.asarray([
        base+handle_basis@np.array([r, tangent, thickness])
        for r in (lo, hi)
        for tangent in (-handle_width_m*.5, handle_width_m*.5)
        for thickness in (-handle_thickness_m*.5, handle_thickness_m*.5)])
    prism_vertices = np.asarray([
        [x, y, z] for x in (-half_tool_width, half_tool_width)
        for y, z in triangle])
    edges = np.roll(triangle, -1, axis=0)-triangle
    prism_edges = np.vstack([np.array([1., 0., 0.]),
                            np.column_stack([np.zeros(3), edges])])
    prism_normals = np.vstack([np.array([1., 0., 0.]),
        np.column_stack([np.zeros(3), -edges[:, 1], edges[:, 0]])])
    axes = np.vstack([handle_basis.T, prism_normals,
        np.cross(handle_basis.T[:, None, :], prism_edges[None, :, :]).reshape(-1, 3)])
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 64.*np.finfo(float).eps]
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    handle_projection = handle_vertices@axes.T
    prism_projection = prism_vertices@axes.T
    overlap = (np.minimum(handle_projection.max(axis=0), prism_projection.max(axis=0))
               - np.maximum(handle_projection.min(axis=0), prism_projection.min(axis=0)))
    # Floating-point roundoff only. This is not a physical insertion-depth,
    # contact-error or score threshold.
    numeric_epsilon = 64.*np.finfo(float).eps*max(
        1., float(np.max(np.abs(handle_vertices))), float(np.max(np.abs(prism_vertices))))
    intersects = bool(np.all(overlap > numeric_epsilon))
    return dict(geometry_seated=intersects,
        basis='finite_allowed_handle_box_and_true_recess_prism_sat',
        contact_fraction_range=list(contact_fraction_range),
        observed_radius_range_m=[lo, hi],
        minimum_projected_interval_overlap_m=float(np.min(overlap)),
        numerical_roundoff_epsilon_m=float(numeric_epsilon),
        reason=('physical_allowed_handle_segment_enters_recess' if intersects
                else 'no_nonzero_physical_allowed_handle_segment_recess_overlap'),
        target_contact_point_selected=False)


def _handle_section(env):
    """Use this workstation's physical section, never an installation name."""
    geometry = env.station_spec.valve.get('geometry', {})
    valve = env.station_spec.valve
    width = valve.get('handle_contact_width_m', geometry.get('lever_mid_width_m'))
    thickness = valve.get('handle_contact_thickness_m', geometry.get('lever_mid_thickness_m'))
    if width is None or thickness is None:
        raise ValueError('Workstation must declare the actual handle width and thickness')
    width, thickness = float(width), float(thickness)
    if not all(math.isfinite(v) and v > 0. for v in (width, thickness)):
        raise ValueError('Handle width and thickness must be finite and positive')
    return width, thickness


def _current_working_contacts(env):
    tool, handle = set(env.active_tool_geom_ids), set(env.valve_geom_ids)
    rows = []
    for row in env.current_contacts:
        if row.get('kind') != 'handle' or row.get('robot_part') != 'operation_end':
            continue
        if row.get('robot_geom_id') is not None and int(row['robot_geom_id']) not in tool:
            continue
        if row.get('other_geom_id') is not None and int(row['other_geom_id']) not in handle:
            continue
        # Zero-force true touching is contact too. Positive native margins
        # count only when solved force is nonzero, matching the native helper.
        if (float(row.get('contact_gap_m', float('inf'))) <= 0.
                or float(row.get('force_N', 0.)) > 0.):
            rows.append(row)
    return rows


def measure_seating(env, mode, target_radius_m):
    """Measure the actual finite recess and current correct working contact.

    ``mode`` is supplied by G1 (``push`` or ``pull``), not guessed from the
    task's open/close label. No working-face occupancy, force, dwell or pose
    tolerance is introduced. Unknown faces remain unknown. ``target_radius_m``
    is retained only for the nominal-section diagnostic; current seating uses
    the physical handle segment in the formally allowed fraction range.
    """
    if mode not in ('push', 'pull'):
        raise ValueError('G1 mode must be push or pull')
    radius = float(target_radius_m)
    if not math.isfinite(radius) or not 0. < radius < env.station_spec.handle_length_m:
        raise ValueError('Target radius must lie on the physical handle')
    width, thickness = _handle_section(env)
    body = env.active_hook_body_id
    geometry_kwargs = dict(
        tool_origin=env.data.xpos[body],
        tool_rotation=env.data.xmat[body].reshape(3, 3),
        pivot=env.valve_pivot_world,
        radial=env.station_spec.radial_at(env.valve_angle_rad),
        stem=env.stem_axis_world, mode=mode,
        handle_length_m=env.station_spec.handle_length_m,
        handle_width_m=width, handle_thickness_m=thickness)
    nominal = recess_seating_geometry(**geometry_kwargs, target_radius_m=radius)
    result = _finite_handle_recess_intersection(**geometry_kwargs,
        contact_fraction_range=_allowed_contact_fraction_range())
    touching = active_tool_handle_contact(env, env.data)
    contacts = _current_working_contacts(env)
    faces = sorted({str(row.get('working_face', 'unknown')) for row in contacts})
    expected = {'orange_push'} if mode == 'push' else {'blue_pull', 'blue_hook'}
    correct = bool(touching and expected.intersection(faces))
    result.update(current_correct_face_contact=correct,
        any_handle_contact=bool(touching), working_faces=faces,
        expected_working_faces=sorted(expected),
        ready_to_turn=bool(result['geometry_seated'] and correct),
        handle_width_m=width, handle_thickness_m=thickness,
        target_radius_m=radius, nominal_section_diagnostic=nominal,
        measurement_only=True, physics_steps_run=0)
    return result


def measure_handle_clearance(env):
    """Return native tool/handle separation and touching, without a pass line.

    ``minimum_distance_m`` is signed: negative means overlap, zero touching,
    positive separation. No clearance threshold is selected here.
    """
    tool, handle = set(env.active_tool_geom_ids), set(env.valve_geom_ids)
    if not tool or not handle:
        raise ValueError('Need declared active-tool and handle geometry groups')
    closest = math.inf
    closest_pair = None
    # A very large search cap makes this a measurement of all declared pairs,
    # rather than the old 0.5 m clipped reading used by a historical controller.
    for a in sorted(tool):
        for b in sorted(handle):
            distance = float(mujoco.mj_geomDistance(
                env.model, env.data, int(a), int(b), 1.e6, None))
            if not math.isfinite(distance):
                raise ValueError('Nonfinite native tool/handle distance')
            if distance < closest:
                closest, closest_pair = distance, [int(a), int(b)]
    return dict(minimum_distance_m=float(closest),
        any_handle_contact=bool(active_tool_handle_contact(env, env.data)),
        closest_geom_ids=closest_pair, basis='native_geometry_distance_and_contact',
        measurement_only=True, physics_steps_run=0)
