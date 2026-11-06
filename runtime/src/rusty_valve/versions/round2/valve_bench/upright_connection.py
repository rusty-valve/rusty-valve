"""Smooth approved rest/preparation joints, with no colour-height policy.

The historical module/factory names remain for callers. Shoulder and elbow
references may guide the route. Wrist joints use only the approved endpoints;
an old zero wrist in a lift waypoint cannot turn the wrist back to zero.
Collision and obstacle checks belong to the caller.
"""
from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator


def _smoothstep(u):
    return u*u*u*(10.+u*(-15.+6.*u))


def _arm_spline(points, knots):
    # Shape-preserving slopes avoid overshoot within each reference interval.
    # Explicit shoulder/elbow reversals in the supplied route are retained.
    slopes = PchipInterpolator(knots, points, axis=0).derivative()(knots)
    slopes[0] = 0.
    slopes[-1] = 0.
    return CubicHermiteSpline(knots, points, slopes, axis=0)


def reference_joints(points, fraction):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Reference needs at least two finite joint vectors')
    u = float(fraction)
    if not np.isfinite(u):raise ValueError('Connection fraction must be finite')
    if u <= 0.:return points[0].copy()
    if u >= 1.:return points[-1].copy()
    return np.asarray(_arm_spline(points, np.linspace(0., 1., len(points)))(u))


@dataclass
class UprightConnection:
    reference_points: np.ndarray
    wrist_offsets: np.ndarray
    knots: np.ndarray
    wrist_angles: np.ndarray
    summary: dict

    def __post_init__(self):
        self.reference_points = np.asarray(self.reference_points, dtype=float).copy()
        self.wrist_offsets = np.asarray(self.wrist_offsets, dtype=int).copy()
        self.knots = np.asarray(self.knots, dtype=float).copy()
        self.wrist_angles = np.asarray(self.wrist_angles, dtype=float).copy()
        self._arm = _arm_spline(self.reference_points, self.knots)

    def evaluate(self, fraction):
        u = float(fraction)
        if not np.isfinite(u):raise ValueError('Connection fraction must be finite')
        if u <= 0.:return self.reference_points[0].copy()
        if u >= 1.:return self.reference_points[-1].copy()
        q = np.asarray(self._arm(u)).copy()
        weight = _smoothstep(u)
        q[self.wrist_offsets] = ((1.-weight)*self.reference_points[0, self.wrist_offsets]
            +weight*self.reference_points[-1, self.wrist_offsets])
        return q


def make_upright_connection(model, base_full_qpos, active_qpos_indices,
        active_joint_names, adjustable_joint_names, tool_bodies, active_tool,
        reference_points, *, label):
    """Connect endpoints inside native limits. Tool arguments are unused.

    Intermediate wrist values are ignored. No duration, workstation branch,
    forced-zero wrist, geometry or collision requirement is added here.
    """
    points = np.asarray(reference_points, dtype=float)
    names = list(active_joint_names)
    active = np.asarray(active_qpos_indices, dtype=int)
    base = np.asarray(base_full_qpos, dtype=float)
    if points.ndim != 2 or len(points) < 2 or points.shape[1] != len(names):
        raise ValueError('Reference needs at least two active-joint vectors')
    if not np.isfinite(points).all() or not np.isfinite(base).all():
        raise ValueError('Connection inputs must be finite')
    if base.shape != (model.nq,) or active.shape != (len(names),):
        raise ValueError('Connection qpos dimensions do not match the model')
    if len(set(names)) != len(names) or len(set(active.tolist())) != len(active):
        raise ValueError('Connection active joints must be distinct')
    if np.any(active < 0) or np.any(active >= model.nq):
        raise ValueError('Connection active qpos indices are outside the model')
    wrists = np.asarray([names.index(n) for n in adjustable_joint_names], dtype=int)
    if len(set(wrists.tolist())) != len(wrists):
        raise ValueError('Connection adjustable joints must be distinct')
    joint_ids = np.asarray([model.joint(name).id for name in names], dtype=int)
    if any(int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE)) for j in joint_ids):
        raise ValueError('Connection requires scalar hinge or slide joints')
    if not np.array_equal(model.jnt_qposadr[joint_ids], active):
        raise ValueError('Connection qpos indices do not match named joints')
    limits = np.asarray(model.jnt_range[joint_ids], dtype=float).copy()
    unlimited = ~np.asarray(model.jnt_limited[joint_ids], dtype=bool)
    limits[unlimited, 0] = -np.inf
    limits[unlimited, 1] = np.inf
    knots = np.linspace(0., 1., len(points))
    prepared = points.copy()
    for index, u in enumerate(knots):
        weight = _smoothstep(u)
        prepared[index, wrists] = ((1.-weight)*points[0, wrists]+weight*points[-1, wrists])
    # Intermediate wrist data is not commanded; check the actual reference.
    # Interpolation stays within each reference interval's native bounds.
    if np.any(prepared < limits[:, 0]) or np.any(prepared > limits[:, 1]):
        raise ValueError('Connection reference exceeds native joint limits')
    summary = dict(label=label, policy='natural_to_preparation_continuous_v1',
        adjustable_joint_names=list(adjustable_joint_names), knot_count=len(knots),
        endpoints_unchanged=True, non_wrist_waypoints_retained=True,
        intermediate_wrist_waypoints_ignored=True,
        wrist_interpolation='endpoint_only_quintic_smoothstep',
        arm_interpolation='shape_preserving_hermite_zero_endpoint_velocity',
        no_colour_height_constraint=True, geometry_not_evaluated=True,
        native_joint_ranges_respected=True, zero_endpoint_velocity=True,
        reverse_policy='evaluate_one_minus_fraction', collision_check='caller',
        physics_steps_run=0)
    return UprightConnection(prepared, wrists, knots, prepared[:, wrists].copy(), summary)
