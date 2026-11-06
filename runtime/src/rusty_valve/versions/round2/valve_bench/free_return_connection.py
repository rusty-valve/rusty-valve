"""Local obstacle detours between immutable return keyframes.

The source keyframes, their arrival fractions, the idle joints and the valve
are inputs, never optimization variables.  Only a colliding free-motion edge
gets a small smooth bend, directed away from its measured obstacle surface.
All queries use scratch data and the existing native contact classifier.  No
physics step, scoring rule, model geometry or motor limit is changed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import mujoco
import numpy as np

from .path_obstacles import PathObstacleConflict, PathObstacleGuard


POLICY = 'immutable_knots_local_surface_normal_return_detour_v1'


def quintic(value):
    value = float(np.clip(value, 0., 1.))
    return value**3*(10.+value*(-15.+6.*value))


def endpoint_zero_bend(value):
    """Unit midpoint bend with zero value/first/second derivatives at ends."""
    value = float(np.clip(value, 0., 1.))
    return 64.*(value*(1.-value))**3


@dataclass
class ReturnConnectionPlan:
    anchors: np.ndarray
    active_indices: np.ndarray
    bends: np.ndarray
    report: dict

    def evaluate(self, fraction):
        count = len(self.anchors)-1
        u = float(np.clip(fraction, 0., 1.))*count
        index = min(int(u), count-1)
        local = u-index
        # Exact knots are returned directly, not reconstructed by subtraction.
        if local == 0.:
            return self.anchors[index, self.active_indices].copy()
        if local == 1.:
            return self.anchors[index+1, self.active_indices].copy()
        a, b = self.anchors[index:index+2, self.active_indices]
        result = (1.-quintic(local))*a+quintic(local)*b
        if np.any(self.bends[index]):
            result = result+endpoint_zero_bend(local)*self.bends[index]
        return result

    def full_pose(self, fraction):
        """Inspection helper: inactive joints come unchanged from first knot."""
        q = self.anchors[0].copy()
        q[self.active_indices] = self.evaluate(fraction)
        return q

    @property
    def summary(self):
        return copy.deepcopy(self.report)


class _Geometry:
    def __init__(self, env, anchors, active_indices):
        self.model = env.model
        self.data = mujoco.MjData(self.model)
        self.base = anchors[0].copy()
        self.active = np.asarray(active_indices, dtype=int)
        self.guard = PathObstacleGuard(self.model,
            robot_geom_ids=env.robot_geom_ids, fixture_geom_ids=env.fixture_geom_ids,
            handle_geom_ids=env.valve_geom_ids, active_tool_geom_ids=env.active_tool_geom_ids)
        self.qids = []
        self.dofs = []
        self.low = np.full(len(self.active), -np.inf)
        self.high = np.full(len(self.active), np.inf)
        for col, address in enumerate(self.active):
            matches = np.flatnonzero(self.model.jnt_qposadr == address)
            if len(matches) != 1:
                raise ValueError('Return detours need named single-coordinate active joints')
            jid = int(matches[0])
            if int(self.model.jnt_type[jid]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                                    int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError('Return detours need hinge/slide active joints')
            self.qids.append(jid); self.dofs.append(int(self.model.jnt_dofadr[jid]))
            if self.model.jnt_limited[jid]:
                self.low[col], self.high[col] = self.model.jnt_range[jid]
            for aid in np.flatnonzero(self.model.actuator_ctrllimited):
                if (int(self.model.actuator_trntype[aid]) == int(mujoco.mjtTrn.mjTRN_JOINT)
                        and int(self.model.actuator_trnid[aid, 0]) == jid):
                    self.low[col] = max(self.low[col], self.model.actuator_ctrlrange[aid, 0])
                    self.high[col] = min(self.high[col], self.model.actuator_ctrlrange[aid, 1])
        self.dofs = np.asarray(self.dofs, dtype=int)
        self.calls = 0

    def observe(self, active):
        self.calls += 1
        active = np.asarray(active, dtype=float)
        if (np.any(active < self.low) or np.any(active > self.high)
                or not np.isfinite(active).all()):
            return dict(clear=False, issues=[], max_penetration_m=math.inf,
                        native_joint_limit_conflict=True)
        self.data.qpos[:] = self.base
        self.data.qpos[self.active] = active
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        return self.guard.observe(self.data, stage='return_between_confirmed_keyframes')

    def outward_joint_directions(self, active):
        """Distance-increasing directions at the obstructed reference pose.

        MuJoCo's contact normal points from geometry 1 towards geometry 2.
        J2-J1 also handles self contact, without hand/station-name branches.
        Positive steps retain the side from which the reference meets the
        surface; the opposite side of an obstacle is never tried.
        """
        observation = self.observe(active)
        allowed = {frozenset((row['geom1_id'], row['geom2_id']))
                   for row in observation['issues']}
        rows = []
        for contact in self.data.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if contact.dist >= 0. or frozenset((g1, g2)) not in allowed:
                continue
            jac1 = np.zeros((3, self.model.nv)); jac2 = np.zeros_like(jac1)
            mujoco.mj_jac(self.model, self.data, jac1, None, contact.pos,
                         int(self.model.geom_bodyid[g1]))
            mujoco.mj_jac(self.model, self.data, jac2, None, contact.pos,
                         int(self.model.geom_bodyid[g2]))
            normal = np.asarray(contact.frame[:3])
            gradient = normal@(jac2-jac1)[:, self.dofs]
            norm2 = float(gradient@gradient)
            if norm2 > 1e-16:
                rows.append((gradient/norm2, float(-contact.dist)))
        if not rows:
            return []
        # The average deforms the full contacting part, not just one facet.
        # Individual normals are finite alternatives for corner contacts.
        directions = [np.mean([row[0] for row in rows], axis=0)]
        for vector, _ in sorted(rows, key=lambda row: -row[1]):
            if not any(np.allclose(vector, old, rtol=1e-5, atol=1e-8) for old in directions):
                directions.append(vector)
        return directions[:4]


def _edge(a, b, bend, local):
    if local == 0.:
        return a.copy()
    if local == 1.:
        return b.copy()
    base = (1.-quintic(local))*a+quintic(local)*b
    return base+endpoint_zero_bend(local)*bend if np.any(bend) else base


def _scan(geometry, a, b, bend, intervals):
    first = None; worst = None; deepest = 0.; obstructed = 0
    for index in range(intervals+1):
        fraction = index/intervals
        q = _edge(a, b, bend, fraction)
        observation = geometry.observe(q)
        if not observation['clear']:
            obstructed += 1
            row = dict(local_fraction=fraction, q=q.copy(), observation=observation)
            if first is None:
                first = row
            if observation['max_penetration_m'] >= deepest:
                deepest = observation['max_penetration_m']; worst = row
    return dict(clear=first is None, first=first, worst=worst,
                deepest_m=deepest, obstructed_samples=obstructed,
                checked_samples=intervals+1)


def plan_return_connections(env, full_keyframes, *, active_indices=None,
                            samples_per_segment=301):
    """Retain every confirmed knot and bend only an obstructed return edge.

    ``samples_per_segment`` is a geometry-query resolution, never a collision
    allowance or exam rule. The caller still prechecks the complete resulting
    curve at its actual control cadence, and can perform a denser review.
    """
    anchors = np.asarray(full_keyframes, dtype=float).copy()
    active = np.asarray(env.active_qpos_indices if active_indices is None else active_indices,
                        dtype=int)
    if (anchors.ndim != 2 or anchors.shape[0] < 2 or anchors.shape[1] != env.model.nq
            or not np.isfinite(anchors).all() or active.ndim != 1
            or len(set(active.tolist())) != len(active) or not len(active)):
        raise ValueError('Return planning needs finite full keyframes and active coordinate indices')
    inactive = np.setdiff1d(np.arange(env.model.nq), active)
    if not np.all(anchors[:, inactive] == anchors[0, inactive]):
        raise ValueError('Return keyframes cannot change idle joints or the valve')
    intervals = max(20, int(samples_per_segment)-1)
    geometry = _Geometry(env, anchors, active)
    bends = np.zeros((len(anchors)-1, len(active)))
    reports = []
    for index in range(len(anchors)-1):
        a, b = anchors[index:index+2, active]
        baseline = _scan(geometry, a, b, bends[index], intervals)
        report = dict(segment_index=index, source_knot_fractions=[index/(len(anchors)-1),
                          (index+1)/(len(anchors)-1)], original_clear=baseline['clear'],
                      original_max_penetration_m=float(baseline['deepest_m']),
                      changed=False, candidate_checks=0)
        if baseline['clear']:
            report.update(clear=True, max_extra_joint_deviation_deg=0., bend_joint_vector_rad=bends[index].tolist())
            reports.append(report)
            continue
        first = baseline['first']; worst = baseline['worst']
        if first['local_fraction'] == 0. or not geometry.observe(b)['clear']:
            observation = copy.deepcopy(first['observation'])
            observation.update(return_detour_policy=POLICY, segment_index=index,
                reason='A confirmed endpoint is obstructed; it cannot be moved by connection planning')
            raise PathObstacleConflict(observation)
        directions = geometry.outward_joint_directions(worst['q'])
        depth = float(baseline['deepest_m'])
        # Compensate only for the bend's amplitude at the actual obstruction.
        amplitude_scale = max(endpoint_zero_bend(worst['local_fraction']), 1e-6)
        nominal_distance = depth/amplitude_scale
        accepted = None
        for direction in directions:
            for factor in (1., 1.5, 2., 3., 4., 6.):
                report['candidate_checks'] += 1
                candidate = direction*(nominal_distance*factor)
                check = _scan(geometry, a, b, candidate, intervals)
                if check['clear']:
                    # A grazing candidate can be clear at the coarse samples
                    # yet still clip a facet between them.  Refine the same
                    # unchanged collision test before accepting its geometry.
                    check = _scan(geometry, a, b, candidate, 4*intervals)
                if check['clear']:
                    accepted = candidate
                    report.update(clear=True, changed=True, distance_multiplier=factor,
                        nominal_outward_displacement_m=nominal_distance*factor,
                        max_extra_joint_deviation_deg=float(np.degrees(np.max(np.abs(candidate)))),
                        bend_joint_vector_rad=candidate.tolist(),
                        final_max_penetration_m=0.,
                        escape_basis='positive native contact-distance gradient on original obstacle side')
                    break
            if accepted is not None:
                break
        if accepted is None:
            observation = copy.deepcopy(first['observation'])
            observation.update(return_detour_policy=POLICY, segment_index=index,
                reason='Finite same-surface-side local bends exhausted; confirmed poses and route side retained',
                candidate_checks=report['candidate_checks'])
            raise PathObstacleConflict(observation)
        bends[index] = accepted
        reports.append(report)
    report = dict(policy=POLICY, segment_count=len(reports), segments=reports,
        modified_segments=[r['segment_index'] for r in reports if r['changed']],
        geometry_queries=geometry.calls, samples_per_segment=intervals+1,
        refined_candidate_samples_per_segment=4*intervals+1,
        keyframe_fractions_unchanged=True, confirmed_joint_poses_unchanged=True,
        inactive_joints_and_valve_unchanged=True,
        boundary_value_velocity_acceleration_unchanged=True,
        original_clear_segments_unchanged=True, native_joint_and_motor_limits_unchanged=True,
        no_opposite_side_candidates=True, physics_steps_run=0, score_changed=False,
        collision_criterion='Unchanged native negative-distance contact classification; no overlap allowance')
    return ReturnConnectionPlan(anchors, active.copy(), bends, report)
