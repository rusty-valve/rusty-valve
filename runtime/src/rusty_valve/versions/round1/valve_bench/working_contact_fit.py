"""Fit the real S-tool flanks around a finite handle, keeping its rotation.

This is geometry only. It never changes a model, saved pose, force limit or
scoring rule. An exact support construction is not evidence of physical
seating; the existing solved-contact measurement remains authoritative.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from .hook_tool import tool_geometry


@lru_cache(maxsize=1)
def _profile():
    return tool_geometry()


def _vector(value, label):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f'{label} needs three finite coordinates')
    return value


def _unit(value, label):
    value = _vector(value, label)
    length = float(np.linalg.norm(value))
    if length < 1e-12:
        raise ValueError(f'{label} cannot be zero')
    return value/length


def fit_working_contact(pivot, radial, stem, rotation, mode, radius,
                        handle_width, handle_thickness, clearance=0., preload_m=0.):
    """Return a TCP translation fitting both finite physical working flanks.

    The requested radius locates the handle axis in the tool-width mid-plane.
    The actual skew of the handle across the tool's finite width is included,
    so arbitrary saved yaw/roll need not be replaced by an old nominal angle.
    ``clearance`` and ``preload_m`` are normal offsets at each support plane;
    preload is only an optional control target, not an added physical force.
    Both default to zero. Correct actual contact must be measured separately.
    """
    if mode not in ('push', 'pull'):
        raise ValueError('Working contact mode must be push or pull')
    pivot = _vector(pivot, 'pivot')
    stem = _unit(stem, 'stem')
    radial = _vector(radial, 'radial')
    radial = _unit(radial-stem*float(radial@stem), 'radial projected off stem')
    tangent = np.cross(stem, radial)
    matrix = np.asarray(rotation, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T@matrix, np.eye(3), rtol=0., atol=1e-6)
            or not np.isclose(np.linalg.det(matrix), 1., rtol=0., atol=1e-6)):
        raise ValueError('Saved working rotation must be a finite proper matrix')
    scalars = (radius, handle_width, handle_thickness, clearance, preload_m)
    if (not all(math.isfinite(float(value)) for value in scalars)
            or min(float(radius), float(handle_width), float(handle_thickness)) <= 0.
            or float(clearance) < 0. or float(preload_m) < 0.):
        raise ValueError('Contact dimensions must be positive and offsets nonnegative')
    radius, handle_width, handle_thickness, clearance, preload_m = map(float, scalars)
    cfg, _, plus, minus, _ = _profile()
    curve, indices = (minus, (0, 1)) if mode == 'push' else (plus, (1, 2))
    normals, anchors, tangents, lengths = [], [], [], []
    for index in indices:
        a, b = curve[index], curve[index+1]
        length = float(np.linalg.norm(b-a));direction = (b-a)/length
        normal = np.array([-direction[1], direction[0]])
        if mode == 'push':
            normal *= -1.
        normals.append(normal);anchors.append(a);tangents.append(direction);lengths.append(length)
    normals, anchors = np.asarray(normals), np.asarray(anchors)
    u, v, w = matrix.T@radial, matrix.T@tangent, matrix.T@stem
    # A handle axis parallel to the S profile has no isolated section across
    # its width. Do not invent a huge translated target near that singularity.
    if abs(float(u[0])) < 1e-8:
        raise ValueError('Handle axis is parallel to the tool profile; no finite width-section fit')
    along = u[1:]/u[0]
    across = v[1:]-v[0]*along
    thick = w[1:]-w[0]*along
    half_tool = float(cfg['out_of_plane_width_mm'])*.0005
    half_handle, half_thickness = handle_width*.5, handle_thickness*.5
    extent = (half_tool*np.abs(normals@along)
              + half_handle*np.abs(normals@across)
              + half_thickness*np.abs(normals@thick))
    rhs = np.sum(normals*anchors, axis=1)+extent+clearance-preload_m
    center_local = np.r_[0., np.linalg.solve(normals, rhs)]
    center_world = pivot+radius*radial
    origin = center_world-matrix@center_local
    tcp = origin+matrix@np.array([0., 0., .09])
    faces = []
    for k, index in enumerate(indices):
        normal = normals[k]
        x = -half_tool*float(np.sign(normal@along))
        lateral = -half_handle*float(np.sign(normal@across))
        vertical = -half_thickness*float(np.sign(normal@thick))
        point_local = np.r_[x, center_local[1:]+x*along+lateral*across+vertical*thick]
        fraction = float((point_local[1:]-anchors[k])@tangents[k]/lengths[k])
        point_world = origin+matrix@point_local
        faces.append(dict(segment=index,working_face='orange_push' if mode=='push' else 'blue_pull',
            point_local_m=point_local.tolist(),point_world_m=point_world.tolist(),
            outward_normal_local=[0., *normal],outward_normal_world=(matrix@np.r_[0., normal]).tolist(),
            finite_segment_fraction=fraction,
            within_finite_flank=bool(-1e-10 <= fraction <= 1.+1e-10),
            support_normal_gap_m=float(normal@(point_local[1:]-anchors[k])),
            actual_handle_radius_m=float((point_world-pivot)@radial)))
    return dict(tcp=tcp,rotation=matrix.copy(),R=matrix.copy(),origin=origin,
        handle_center=center_world,handle_center_local=center_local,
        working_face_normals_world=np.array([face['outward_normal_world'] for face in faces]),
        face_contacts=faces,finite_working_face_count=sum(face['within_finite_flank'] for face in faces),
        mode=mode,radius_m=radius,clearance_m=clearance,preload_m=preload_m,
        source_rotation_preserved=True,tool_width_included=True,
        kind='finite_working_flank_support_target_not_physical_seating',
        actual_contact_measurement_required=True)
