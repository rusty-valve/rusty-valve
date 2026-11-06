"""Station-relative S-tool contact geometry and scratch-data arm IK for H1.

No workstation name, world-space valve centre, or recorded arm trajectory is
used here.  ``axis_world`` is the outward stem direction, not necessarily the
positive hinge-coordinate direction. ``direction_sign`` says whether the
requested operation turns right-handed (+1) or left-handed (-1) around it.

The S tool has two inclined orange faces. Its TCP +Z is the V recess axis,
*not* either physical face normal. ``working_face_normals_world`` exposes the
actual normals so a tangent TCP cannot be mistaken for a 90-degree face normal.
Only the separate planning MjData is ever assigned joint positions.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .hook_tool import tool_geometry
from .wp2 import ArmIK


@lru_cache(maxsize=1)
def _tool_profile():
    # Tool construction is fixed within one run; do not reopen its JSON at
    # every 50 Hz contact-target update.
    return tool_geometry()


def _unit(vector, name):
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must contain three finite numbers')
    norm = float(np.linalg.norm(value))
    if norm < 1e-10:
        raise ValueError(f'{name} cannot be zero')
    return value/norm


def contact_geometry(pivot_world, axis_world, radial_world, task, *, mode='push',
                     radius=None, clearance=0., direction_sign=None, yaw_deg=0.,
                     pitch_deg=30., tilt_deg=0., roll_deg=0.,
                     push_contact_geometry='legacy'):
    """Return a TCP target and the actual tool-face geometry in world space.

    ``pivot_world`` is where the stem axis meets the handle centre plane.
    ``radial_world`` is the handle's requested radial direction; it may be the
    measured direction or a small angular lead computed by a feedback pacer.
    All distances are metres, angles in option names are degrees.

    The default turn signs (+ open, - close) reproduce the original benchmark
    convention. A general caller should pass the signs from its description.
    Orange pushing works in either turn direction. Blue hooking likewise uses
    the requested tangent; whether that placement is reachable/clear is left
    to preparation, collision checks and the physical episode.

    Push defaults to the historical depth formula. Opt-in ``support_planes``
    locates the 20 x 6 mm handle section against the two actual orange planes.
    Its clearance moves along the V bisector to separate from both faces.
    Finite faces, true holes and other geometry still require collision and
    physical contact verification; two-plane support is not that verification.
    """
    if task not in ('open', 'close') or mode not in ('push', 'pull'):
        raise ValueError('task must be open/close; mode must be push/pull')
    if push_contact_geometry not in ('legacy', 'support_planes'):
        raise ValueError('push_contact_geometry must be legacy or support_planes')
    pivot = np.asarray(pivot_world, dtype=float)
    if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
        raise ValueError('pivot_world must contain three finite numbers')
    stem = _unit(axis_world, 'axis_world')
    radial = _unit(radial_world, 'radial_world')
    radial = _unit(radial-stem*np.dot(stem, radial), 'radial projected off axis')
    sign = (1. if task == 'open' else -1.) if direction_sign is None else float(direction_sign)
    if sign not in (-1., 1.):
        raise ValueError('direction_sign must be +1 or -1')
    radius = (.14 if mode == 'pull' else .16) if radius is None else float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    for name, value in (('clearance', clearance), ('yaw_deg', yaw_deg),
                        ('pitch_deg', pitch_deg), ('tilt_deg', tilt_deg),
                        ('roll_deg', roll_deg)):
        if not math.isfinite(float(value)):
            raise ValueError(f'{name} must be finite')
    tangent = sign*np.cross(stem, radial)
    handle_center = pivot+radius*radial
    cfg, _, plus, minus, _ = _tool_profile()
    if mode == 'push':
        rotation = np.column_stack([np.cross(stem, tangent), stem, tangent])
        # Historical orange A1 used yaw=20; yaw=0 aligns the V bisector with
        # the circular tangent. Keep yaw explicit rather than call 20 deg 90.
        rotation = Rotation.from_rotvec(stem*math.radians(-sign*yaw_deg)).as_matrix() @ rotation
        rotation = rotation @ Rotation.from_rotvec([math.radians(tilt_deg), 0., 0.]).as_matrix()
        rotation = rotation @ Rotation.from_rotvec([0., 0., math.radians(roll_deg)]).as_matrix()
        if push_contact_geometry == 'legacy':
            depth = .007+.010*abs(math.sin(math.radians(tilt_deg)))
            tcp = handle_center-(depth+float(clearance))*rotation[:, 2]
            origin = tcp-rotation@np.array([0., 0., .09])
        else:
            # The 2D formula assumes both face normals lie in the handle's
            # tangent/stem cross-section. Arbitrary yaw/roll would require
            # the finite longitudinal extent of the handle as a third term.
            if (abs(math.sin(math.radians(yaw_deg))) > 1e-8
                    or abs(math.cos(math.radians(yaw_deg))-1.) > 1e-8
                    or abs(math.sin(math.radians(roll_deg))) > 1e-8):
                raise ValueError('push support_planes requires zero yaw and roll in multiples of 180 degrees')
            normals = []
            face_points = []
            for index in (0, 1):
                a, b = minus[index], minus[index+1]
                side_tangent = (b-a)/np.linalg.norm(b-a)
                # Outward from the metal into the orange V recess.
                normals.append(-np.r_[0., -side_tangent[1], side_tangent[0]])
                face_points.append(np.r_[0., a])
            normals = np.asarray(normals)
            rhs = [normal@point+.010*abs((rotation@normal)@tangent)
                   +.003*abs((rotation@normal)@stem)
                   for normal, point in zip(normals, face_points)]
            local_handle_center = np.r_[0., np.linalg.solve(normals[:, 1:], rhs)]
            retreat = rotation[:, 2]
            if any(float((rotation@normal)@retreat) <= 0. for normal in normals):
                raise ValueError('Push retreat must create clearance from both support planes')
            origin = handle_center-rotation@local_handle_center-float(clearance)*retreat
            tcp = origin+rotation@np.array([0., 0., .09])
        curve, indices = minus, (0, 1)
    else:
        # General form of the proven lower-blue-recess contact construction.
        # Both support planes account for the handle's 20 x 6 mm cross-section.
        # No world-up assumption: stem is the handle thickness direction.
        rotation = np.column_stack([sign*radial, stem, -tangent])
        rotation = rotation @ Rotation.from_rotvec([math.radians(pitch_deg), 0., 0.]).as_matrix()
        # A 180-degree roll preserves the cross-section support construction:
        # both blue support normals stay perpendicular to the handle length.
        # Other rolls would require including the finite longitudinal extent
        # of the lever, which the 20 x 6 mm support formula below does not do.
        if abs(math.sin(math.radians(roll_deg))) > 1e-8:
            raise ValueError('pull roll must be a multiple of 180 degrees')
        rotation = rotation @ Rotation.from_rotvec([0., 0., math.radians(roll_deg)]).as_matrix()
        normals = []
        face_points = []
        for index in (1, 2):
            a, b = plus[index], plus[index+1]
            side_tangent = (b-a)/np.linalg.norm(b-a)
            normals.append(np.r_[0., -side_tangent[1], side_tangent[0]])
            face_points.append(np.r_[0., a])
        normals = np.asarray(normals)
        rhs = [normal@point+.010*abs((rotation@normal)@(-tangent))
               +.003*abs((rotation@normal)@stem)
               for normal, point in zip(normals, face_points)]
        local_handle_center = np.r_[0., np.linalg.solve(normals[:, 1:], rhs)]
        origin = handle_center-rotation@local_handle_center-float(clearance)*tangent
        tcp = origin+rotation@np.array([0., 0., .09])
        curve, indices = plus, (1, 2)
    face_normals = []
    face_centres = []
    for index in indices:
        a, b = curve[index], curve[index+1]
        side_tangent = (b-a)/np.linalg.norm(b-a)
        normal = np.r_[0., -side_tangent[1], side_tangent[0]]
        if mode == 'push':
            normal = -normal
        face_normals.append(rotation@normal)
        face_centres.append(origin+rotation@np.r_[0., (a+b)/2])
    return dict(tcp=tcp, R=rotation, origin=origin, handle_center=handle_center,
                axis_world=stem, radial_world=radial, movement_direction=tangent,
                recess_axis_world=rotation[:, 2],
                working_face_normals_world=np.asarray(face_normals),
                working_face_centres_world=np.asarray(face_centres),
                working_flank_indices=indices, mode=mode,
                contact_geometry_mode=(push_contact_geometry if mode=='push'
                                       else 'pull_support_planes'),
                tool_version=cfg['version'])


def contact_pose(pivot_world, axis_world, radial_world, task, *, mode='push',
                 radius=None, clearance=0., direction_sign=None, yaw_deg=0.,
                 pitch_deg=30., tilt_deg=0., roll_deg=0.,
                 push_contact_geometry='legacy'):
    geometry = contact_geometry(pivot_world, axis_world, radial_world, task,
        mode=mode, radius=radius, clearance=clearance, direction_sign=direction_sign,
        yaw_deg=yaw_deg, pitch_deg=pitch_deg, tilt_deg=tilt_deg, roll_deg=roll_deg,
        push_contact_geometry=push_contact_geometry)
    return geometry['tcp'], geometry['R']


def pull_pitch_deg(opening_fraction, start_deg=30.):
    """Ease hook pitch over the last 50/90 of travel, relative to the stops."""
    fraction = float(np.clip(opening_fraction, 0., 1.))
    u = float(np.clip((fraction-4/9)/(5/9), 0., 1.))
    return float(start_deg)*(1.-u*u*(3.-2.*u))


class H1IK(ArmIK):
    """Official-range arm IK using scratch data; no running-state teleporting.

    ``solve`` has the historical (q, position_error_m, rotation_error_rad)
    return shape. ``solve_best`` additionally uses generic randomized seeds
    and checks preparation contacts. This is waypoint preparation, not a full
    motion planner: collision checking a final pose does not certify a path.
    """
    def __init__(self, env, side=None, solver='dls'):
        self.env = env
        side = env.side if side is None else side
        if side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        names = tuple(f'openarm_{side}_joint{i}' for i in range(1, 8))
        site = 'left_ee_control' if side == 'left' else 'ee_control'
        super().__init__(env.model, names, site)
        self.side = side
        if solver not in ('dls', 'least_squares'):
            raise ValueError('solver must be dls or least_squares')
        self.solver = solver
        self.body = int(self.model.body(f'openarm_{side}_hook').id)
        self.actuator_ids = np.asarray([self.model.actuator(f'{side}_joint{i}_ctrl').id
                                       for i in range(1, 8)], dtype=int)
        # Never command beyond either the upstream joints or upstream motors.
        control_bounds = self.model.actuator_ctrlrange[self.actuator_ids]
        self.bounds[:, 0] = np.maximum(self.bounds[:, 0], control_bounds[:, 0])
        self.bounds[:, 1] = np.minimum(self.bounds[:, 1], control_bounds[:, 1])
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError('joint and actuator bounds do not overlap')
        self.last_preparation: dict[str, Any] | None = None
        self.sync_from(env)

    def sync_from(self, env=None):
        """Read a current episode snapshot into planning storage only."""
        env = self.env if env is None else env
        if env.model is not self.model:
            raise ValueError('IK must be recreated after a model is recompiled')
        self.data.qpos[:] = env.data.qpos
        self.data.qvel[:] = env.data.qvel
        self.data.ctrl[:] = env.data.ctrl
        self.data.time = env.data.time

    def solve(self, pos, rotation, seed, iterations=65):
        target = np.asarray(pos, dtype=float)
        requested_rotation = np.asarray(rotation, dtype=float)
        if target.shape != (3,) or requested_rotation.shape != (3, 3):
            raise ValueError('IK needs a 3-vector position and 3x3 rotation')
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(requested_rotation)):
            raise ValueError('IK target must be finite')
        q = np.clip(np.asarray(seed, dtype=float), self.bounds[:, 0]+1e-5,
                    self.bounds[:, 1]-1e-5)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError('IK seed must contain seven finite joint values')
        if self.solver == 'least_squares':
            return self.solve_precise(target, requested_rotation, q, iterations)
        pe = np.zeros(3); re = np.zeros(3)
        for _ in range(max(1, int(iterations))):
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            actual = self.data.site_xmat[self.site].reshape(3, 3)
            pe = target-self.data.site_xpos[self.site]
            re = Rotation.from_matrix(requested_rotation@actual.T).as_rotvec()
            if np.linalg.norm(pe) < .0005 and np.linalg.norm(re) < .007:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.site)
            jacobian = np.r_[self.jp[:, self.dofs], .16*self.jr[:, self.dofs]]
            error = np.r_[pe, .16*re]
            delta = jacobian.T@np.linalg.solve(jacobian@jacobian.T+1.5e-5*np.eye(6), error)
            q = np.clip(q+np.clip(delta, -.12, .12), self.bounds[:, 0]+1e-5,
                        self.bounds[:, 1]-1e-5)
        # Evaluate the returned q, not the preceding iteration's residual.
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        pe = target-self.data.site_xpos[self.site]
        actual = self.data.site_xmat[self.site].reshape(3, 3)
        re = Rotation.from_matrix(requested_rotation@actual.T).as_rotvec()
        return q.copy(), float(np.linalg.norm(pe)), float(np.linalg.norm(re))

    def solve_precise(self, pos, rotation, seed, iterations=100):
        """Existing blue-hook bounded pose solver, without any A1 geometry.

        Useful near joint limits where clipped DLS can remain on a poor local
        branch. The method and its local nature are recorded by the caller;
        it does not loosen official actuator or joint limits.
        """
        target = np.asarray(pos, dtype=float)
        requested_rotation = np.asarray(rotation, dtype=float)
        seed = np.clip(np.asarray(seed, dtype=float), self.bounds[:, 0]+1e-5,
                       self.bounds[:, 1]-1e-5)
        if (target.shape != (3,) or requested_rotation.shape != (3, 3)
                or seed.shape != (7,) or not np.all(np.isfinite(target))
                or not np.all(np.isfinite(requested_rotation))
                or not np.all(np.isfinite(seed))):
            raise ValueError('IK needs finite position, 3x3 rotation and seven-joint seed')
        def residual(q):
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            actual = self.data.site_xmat[self.site].reshape(3, 3)
            return np.r_[self.data.site_xpos[self.site]-target,
                         .12*(actual-requested_rotation).ravel()]
        fit = least_squares(residual, seed,
            bounds=(self.bounds[:, 0]+1e-5, self.bounds[:, 1]-1e-5),
            max_nfev=max(1, int(iterations)), ftol=1e-8, xtol=1e-8, gtol=1e-8)
        residual(fit.x)
        actual = self.data.site_xmat[self.site].reshape(3, 3)
        return fit.x.copy(), float(np.linalg.norm(self.data.site_xpos[self.site]-target)), float(
            Rotation.from_matrix(requested_rotation@actual.T).magnitude())

    def preparation_contacts(self, q, *, allow_handle=False, margin_m=0.):
        """List geometric contacts in a proposed preparation pose, without force claims."""
        self.data.qpos[self.qadr] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self.model, self.data)
        robot = set(getattr(self.env, 'robot_geom_ids', ()))
        target = set(getattr(self.env, 'valve_geom_ids', ()))
        tool = set(getattr(self.env, 'active_tool_geom_ids', ()))
        fixture = set(getattr(self.env, 'fixture_geom_ids', ()))
        issues = []
        for contact in self.data.contact:
            if float(contact.dist) > float(margin_m):
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            pair = {a, b}
            if not pair & robot:
                continue
            handle_touch = bool(pair & tool and pair & target)
            if allow_handle and handle_touch:
                continue
            if (a in robot and b in robot) or pair & fixture or pair & target:
                issues.append(dict(geom1=self.model.geom(a).name,
                                   geom2=self.model.geom(b).name,
                                   distance_m=float(contact.dist)))
        return issues

    def solve_best(self, pos, rotation, seed, *, iterations=300, attempts=10,
                   random_seed=0, allow_handle=False, position_tolerance_m=.004,
                   rotation_tolerance_rad=.05):
        """Choose a clear reachable seed before starting an episode.

        Generic seeds cover the existing official joint ranges; none are
        workstation-specific saved postures. Report failures rather than
        widening limits or changing the current physical episode.
        """
        candidates = [np.asarray(seed, dtype=float)]
        rng = np.random.default_rng(random_seed)
        span = self.bounds[:, 1]-self.bounds[:, 0]
        candidates.extend(rng.uniform(self.bounds[:, 0]+.03*span,
                                      self.bounds[:, 1]-.03*span,
                                      (max(0, int(attempts)-1), 7)))
        best = None
        diagnostics = []
        preference = getattr(self, 'pose_preference', None)
        accept = getattr(self, 'pose_accept', None)
        for index, candidate in enumerate(candidates):
            q, pe, re = self.solve(pos, rotation, candidate, iterations)
            contacts = self.preparation_contacts(q, allow_handle=allow_handle)
            score = pe+.16*re+sum(.1+min(.1, abs(c['distance_m'])) for c in contacts)
            feasible = bool(pe <= position_tolerance_m and re <= rotation_tolerance_rad
                            and not contacts and (accept is None or accept(q)))
            rank = ((not feasible, *(preference(q) if feasible else (score,)))
                    if preference is not None else (not feasible, score))
            diagnostics.append(dict(seed_index=index, position_error_m=pe,
                                    rotation_error_rad=re, contacts=contacts))
            if best is None or rank < best[0]:
                best = (rank, q, pe, re, contacts)
            if feasible and preference is None:
                break
        _, q, pe, re, contacts = best
        self.last_preparation = dict(position_error_m=pe, rotation_error_rad=re,
            geometric_contacts=contacts, attempts=diagnostics,
            reachable=bool(pe <= position_tolerance_m and re <= rotation_tolerance_rad
                           and not contacts),
            note='Scratch-data waypoint IK; endpoint collision check is not full path planning.')
        return q.copy(), pe, re
