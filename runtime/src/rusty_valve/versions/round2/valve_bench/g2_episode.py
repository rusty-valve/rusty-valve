"""G2 single-contact physical episodes, scored by the environment at every step.

Only reset sets the robot/valve state. During execution all motion comes from
native finite robot motors and contact with the passive valve. Approved G1
poses supply the active arm's raised ready posture and IK seeds, never a
replayed valve motion. Every episode starts with both arms naturally down.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .h1_kinematics import H1IK, contact_geometry, pull_pitch_deg
from .h1_scene import build_env
from .scored_env import ScoredValveEnv
from .slow_open_rule import load_slow_open_rule
from .tool_upright import ToolUprightGuard, ToolUprightConflict
from .path_obstacles import PathObstacleGuard, PathObstacleConflict
from .tool_orientation_context import (POLICY_VERSION as ORIENTATION_POLICY,
    G1OrientationReferences, active_tool_handle_contact, apply_reviewed_rest,
    orientation_scope)

ROOT = Path(__file__).resolve().parents[1]
POSES = ROOT / '试验记录/2026-10-25_G1_自动选位'
APPROVED = ROOT / 'valve_bench/configs/g2_approved_auto.json'
DEFAULTS = dict(r_m=None, approach_clearance_m=.055, pitch_offset_deg=0.,
    tilt_offset_deg=0., approach_angle_deg=0., lead_deg=6., start_lead_deg=.5,
    standby_offset_m=[0., 0., 0.], speed_deg_s=3.7, opening_speed_deg_s=6.5,
    start_speed_deg_s=1.5, approach_s=2.5, reach_s=5., withdraw_s=2.,
    return_s=4., down_hold_s=.5, lift_s=3., chest_hold_s=.4, lower_s=3.,
    lift_waypoint=[.6,0.,0.,.8,0.,0.,0.], force_N=None, force_ramp_s=8., control_dt=.01,
    ik_iterations=20, max_wall_s=180., slow_feedback=False,
    start_integral_limit_rad=.04, start_force_scale=.6,
    start_velocity_damping=.15, pause_integral_decay_s=.5,
    start_brake_speed_deg_s=4., start_brake_release_deg_s=2.,
    start_brake_lead_deg=.15, start_brake_integral_decay_s=.08,
    ik_warm_seed_cost_m=.006, free_path_samples=13)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def interpolate_pose(start_pos, start_rotation, goal_pos, goal_rotation, u):
    """Continuous Cartesian target joining an actual reachable start pose."""
    u=blend(u)
    relative=Rotation.from_matrix(goal_rotation@start_rotation.T).as_rotvec()
    return ((1.-u)*start_pos+u*goal_pos,
            Rotation.from_rotvec(u*relative).as_matrix()@start_rotation)


def blend(u):
    u = float(np.clip(u, 0., 1.))
    return u*u*(3.-2.*u)


class G2Controller:
    def __init__(self, env, variant, method, poses, parameters):
        self.env, self.variant, self.method, self.p = env, variant, method, parameters
        self.direction = -1. if variant == 'close' else 1.
        self.goal = 0. if variant == 'close' else 90.
        self.ik = H1IK(env, solver='dls')
        self.upright_guard = (ToolUprightGuard(env.model, parameters['tool_body_names'])
                              if parameters.get('tool_upright_required') else None)
        if self.upright_guard is not None and parameters.get('tool_orientation_policy') == ORIENTATION_POLICY:
            poses = apply_reviewed_rest(poses, side=env.side,
                path=parameters['reviewed_natural_rest_file'])
        self.orientation_references = (G1OrientationReferences(env, poses,
            parameters['tool_body_names'][env.side]) if self.upright_guard is not None else None)
        self.upright_candidate_checks = []
        self.path_obstacle_guard = (PathObstacleGuard(env.model,
            robot_geom_ids=env.robot_geom_ids,fixture_geom_ids=env.fixture_geom_ids,
            handle_geom_ids=env.valve_geom_ids,active_tool_geom_ids=env.active_tool_geom_ids)
            if parameters.get('path_obstacles_required') else None)
        self.path_obstacle_checks = 0
        self.path_obstacle_rejections = []
        self._lower_path_obstacles_checked = False
        self.active_offsets = [env.robot_joint_names.index(n) for n in env.active_joint_names]
        self.idle_offsets = [env.robot_joint_names.index(n) for n in env.idle_joint_names]
        self.idle_dofs = env.model.jnt_dofadr[[env.model.joint(n).id for n in env.idle_joint_names]]
        # Historical runs use zero-joint down. S2 takes the reviewed standby
        # from G1 and validates its tool orientation instead of forcing zero.
        self.natural = np.asarray(poses[0]['qpos'], dtype=float)[env.robot_qpos_indices].copy()
        if self.upright_guard is None and not np.allclose(self.natural, 0., atol=1e-9):
            raise ValueError('Approved stage 0 must have both arms naturally down at zero joints')
        self.approved_poses = poses
        self.contact_seed_rows = [poses[i] for i in (3,4,5)]
        self.ready = self.natural.copy()  # idle-arm action always stays down
        self.chest = self.natural.copy()
        self.chest[self.active_offsets] = np.asarray(poses[1]['qpos'])[env.active_qpos_indices]
        self.lift_waypoint = np.asarray(parameters['lift_waypoint'], dtype=float)
        if self.lift_waypoint.shape != (7,) or not np.isfinite(self.lift_waypoint).all():
            raise ValueError('lift_waypoint needs seven finite active-arm joint angles')
        self.upright_connection = None
        self.reviewed_free_connections = None
        self.qseed = self.natural[self.active_offsets].copy()
        self.reach_start_s = parameters['down_hold_s']+parameters['lift_s']+parameters['chest_hold_s']
        self.integral = np.zeros(7)
        self.velocity = np.zeros(7)
        self.slow_braking = False
        self.approach_elapsed_s = 0.
        self.feedback_turn_started = None
        self.reference = self.opening
        self.slow_open_rule = load_slow_open_rule()
        self.pause_targets = self.slow_open_rule['pause_angles_deg'] if variant == 'slow_open' else []
        self.pause_index = 0
        self.pause_since = None
        self.withdraw_started = None
        self.return_started = None
        self.lower_started = None
        self.done = False
        self.last_diagnostics = {}
        self.preparation = []
        self._prepare(poses)

    @property
    def opening(self):
        return float((self.env.valve_angle_rad-self.env.q_close_rad)*90./(self.env.q_open_rad-self.env.q_close_rad))

    @property
    def speed(self):
        return float(self.env.valve_speed_rad_s*90./(self.env.q_open_rad-self.env.q_close_rad))

    def target(self, measured, *, lead=0., clearance=0.):
        e, m, p = self.env, self.method, self.p
        q = e.q_close_rad+(e.q_open_rad-e.q_close_rad)*measured/90.
        radial = e.station_spec.radial_at(q)
        lead_q = e.q_close_rad+(e.q_open_rad-e.q_close_rad)*(measured+self.direction*lead)/90.
        lead_radial = e.station_spec.radial_at(lead_q)
        pitch = m['pitch_deg'] if m.get('pitch_constant') else pull_pitch_deg(measured/90., m['pitch_deg'])
        qdirection = np.sign((e.q_close_rad-e.q_open_rad) if self.variant == 'close' else (e.q_open_rad-e.q_close_rad))
        turn_sign = int(qdirection*np.sign(e.valve_hinge_axis_world@e.stem_axis_world))
        g = contact_geometry(e.valve_pivot_world, e.stem_axis_world, radial,
            'close' if self.variant == 'close' else 'open', mode=m['mode'],
            radius=p['r_m'], direction_sign=turn_sign, clearance=0.,
            pitch_deg=pitch+p['pitch_offset_deg'], tilt_deg=m['tilt_deg']+p['tilt_offset_deg'],
            roll_deg=m['roll_deg'],
            push_contact_geometry=m['push_contact_geometry'])
        tcp = g['tcp']+p['r_m']*(lead_radial-radial)
        if clearance:
            away = contact_geometry(e.valve_pivot_world, e.stem_axis_world, radial,
                'close' if self.variant == 'close' else 'open', mode=m['mode'],
                radius=p['r_m'], direction_sign=turn_sign, clearance=clearance,
                pitch_deg=pitch+p['pitch_offset_deg'], tilt_deg=m['tilt_deg']+p['tilt_offset_deg'],
                roll_deg=m['roll_deg'],
                push_contact_geometry=m['push_contact_geometry'])['tcp']-g['tcp']
            turn = Rotation.from_rotvec(e.stem_axis_world*math.radians(p['approach_angle_deg']))
            tcp += turn.apply(away)
        return tcp, g['R']

    def _sync_ik(self):
        self.ik.sync_from(self.env)
        # Never import the old tucked idle arm into any scratch IK scene.
        self.ik.data.qpos[self.env.idle_qpos_indices] = self.natural[self.idle_offsets]
        self.ik.data.qvel[self.idle_dofs] = 0.

    def _approved_contact_seed(self, opening):
        # Use only this approved G1 document; old reference_method/source
        # fields are provenance, never runtime inputs or stored trajectories.
        rows=sorted(self.contact_seed_rows,key=lambda row:row['opening_deg'])
        angles=[row['opening_deg'] for row in rows]
        joints=np.asarray([np.asarray(row['qpos'])[self.env.active_qpos_indices] for row in rows])
        return np.asarray([np.interp(opening,angles,joints[:,i]) for i in range(7)])

    def _solve_contact(self, pos, rotation, seed, iterations, opening, *, allow_alternate=True):
        # A local IK miss is not a rejection. Try the current approved branch
        # as another seed and keep the lower continuous pose/continuity cost.
        self._sync_ik()
        best=self.ik.solve(pos,rotation,seed,iterations)
        candidates=[best]
        if allow_alternate and best[1]+.12*best[2] > self.p['ik_warm_seed_cost_m']:
            alternate=self.ik.solve(pos,rotation,self._approved_contact_seed(opening),max(iterations,40))
            candidates.append(alternate)
            def cost(row):return row[1]+.12*row[2]+.0002*np.linalg.norm(row[0]-seed)
            if cost(alternate)<cost(best):best=alternate
        if self.upright_guard is not None:
            accepted=[]; rejected=[]
            for row in candidates:
                observation=self._upright_for_joints(row[0],
                    getattr(self,'_motion_candidate_stage','contact_candidate'))
                (accepted if observation['all_tools_upright'] else rejected).append((row,observation))
            if not accepted:
                raise ToolUprightConflict(rejected[0][1])
            if not any(row is best for row,_ in accepted):
                best=min(accepted,key=lambda item:item[0][1]+.12*item[0][2]+
                         .0002*np.linalg.norm(item[0][0]-seed))[0]
        if self.path_obstacle_guard is not None:
            accepted=[];rejected=[]
            for row in candidates:
                observation=self._obstacle_for_joints(row[0],
                    getattr(self,'_motion_candidate_stage','contact_candidate'))
                upright=(self.upright_guard is None or
                    self._upright_for_joints(row[0],getattr(self,
                        '_motion_candidate_stage','contact_candidate'))['all_tools_upright'])
                if observation['clear'] and upright:accepted.append(row)
                elif not observation['clear']:rejected.append(observation)
            if not accepted:
                if rejected:raise PathObstacleConflict(rejected[0])
                raise ToolUprightConflict(self._upright_for_joints(best[0],'contact_candidate'))
            if not any(row is best for row in accepted):
                best=min(accepted,key=lambda row:row[1]+.12*row[2]+
                         .0002*np.linalg.norm(row[0]-seed))
        return best

    def _obstacle_for_joints(self,q,stage):
        geometry=self._static_geometry(q)
        observation=geometry['obstacle_observation']
        observation.update(stage=stage,candidate_active_joints_rad=np.asarray(q).tolist(),
            opening_deg=self.opening,phase_scope='planned_pose_not_physical_contact')
        self.path_obstacle_checks+=1
        if not observation['clear']:self.path_obstacle_rejections.append(observation)
        return observation

    def _scan_obstacle_path(self,evaluate,duration,stage):
        count=max(2,int(math.ceil(duration/self.p['control_dt']))+1)
        for index,fraction in enumerate(np.linspace(0.,1.,count)):
            observation=self._obstacle_for_joints(evaluate(float(fraction)),stage)
            if not observation['clear']:
                observation.update(path_fraction=float(fraction),planned_duration_s=float(duration))
                return dict(clear=False,samples_checked=index+1,first_block=observation,
                    sampling='planned_control_times',additional_clearance_m=0.)
        return dict(clear=True,samples_checked=count,first_block=None,
            sampling='planned_control_times',additional_clearance_m=0.)

    def _require_clear_connection(self,reverse=False):
        if self.path_obstacle_guard is None:return
        duration=self.p['lower_s'] if reverse else self.p['lift_s']
        if self.reviewed_free_connections is not None:
            evaluate=lambda u:self.reviewed_free_connections.evaluate(u,reverse=reverse)
        elif self.upright_connection is not None:
            evaluate=lambda u:self.upright_connection.evaluate(1.-u if reverse else u)
        else:
            points=[self.natural[self.active_offsets],self.lift_waypoint,self.chest[self.active_offsets]]
            if reverse:points=list(reversed(points))
            evaluate=lambda u:self._free(u*duration,points,duration)
        stage='lower_to_natural' if reverse else 'lift_to_chest'
        check=self._scan_obstacle_path(evaluate,duration,stage)
        self.preparation.append(dict(name=stage+'_obstacle_precheck',**check))
        if not check['clear']:raise PathObstacleConflict(check['first_block'])

    def _free_connection(self, elapsed, *, reverse=False):
        """Use the reviewed connection only when explicitly selected by input.

        Older jobs keep their original curve and arithmetic.  This selects
        lift/lower data only; the return path still targets the old ready pose.
        """
        duration=self.p['lower_s'] if reverse else self.p['lift_s']
        if self.reviewed_free_connections is not None:
            return self.reviewed_free_connections.evaluate(elapsed/duration,reverse=reverse)
        if self.upright_connection is not None:
            return self.upright_connection.evaluate(1.-elapsed/duration if reverse else elapsed/duration)
        points=[self.natural[self.active_offsets],self.lift_waypoint,self.chest[self.active_offsets]]
        if reverse:points=list(reversed(points))
        return self._free(elapsed,points,duration)

    def _upright_for_joints(self, q, stage):
        self._sync_ik()
        self.ik.data.qpos[self.ik.qadr]=q
        mujoco.mj_kinematics(self.env.model,self.ik.data)
        return self.observe_tool_orientation(self.ik.data,stage=stage,require=False)

    def observe_tool_orientation(self, data, *, stage, require=True):
        """Check only the working hand, retaining strictness while in contact.

        Contact is read from the actual physical state, not the candidate IK
        scene. This does not advance/reorder the existing controller phases.
        """
        contact = active_tool_handle_contact(self.env, self.env.data)
        scope = orientation_scope(stage, active_contact=contact)
        reference = self.orientation_references.at(self.opening,
            self.env.valve_angle_rad, scope=scope)
        call = self.upright_guard.require if require else self.upright_guard.observe
        observation = call(data, stage=stage, scope=scope,
            active_tool=self.env.side, reference_rotation=reference)
        observation['actual_active_tool_handle_contact'] = contact
        observation['reference_source'] = 'G1_tool_in_actual_handle_frame'
        observation['policy_version'] = ORIENTATION_POLICY
        return observation

    def _require_upright_path(self, points, label):
        """Check the existing joint interpolation; never invent a replacement."""
        if self.upright_guard is None:return
        count=max(10,min(20,int(self.p['free_path_samples'])))
        for segment,(start,finish) in enumerate(zip(points[:-1],points[1:])):
            for fraction in np.linspace(0.,1.,count):
                # blend only changes pacing, not the set of joint poses.
                q=(1.-fraction)*start+fraction*finish
                observation=self._upright_for_joints(q,label)
                observation.update(segment=segment,fraction=float(fraction),
                    candidate_active_joints_rad=q.tolist(),
                    phase_scope='planned_joint_interpolation_not_physical_motion')
                self.upright_candidate_checks.append(observation)
                if not observation['all_tools_upright']:
                    raise ToolUprightConflict(observation)

    def _static_geometry(self, q):
        """Scratch geometry preference only; never a physical pass/fail gate."""
        self._sync_ik()
        data, model, env = self.ik.data, self.env.model, self.env
        data.qpos[self.ik.qadr]=q
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        mujoco.mj_collision(model,data)
        issues=[];observation=None
        robot=set(env.robot_geom_ids);fixture=set(env.fixture_geom_ids)
        handle=set(env.valve_geom_ids);tool=set(env.active_tool_geom_ids)
        for contact in (() if self.path_obstacle_guard is not None else data.contact):
            a,b,d=int(contact.geom1),int(contact.geom2),float(contact.dist)
            pair={a,b}
            if d>=0. or not pair & robot:continue
            # Normal tool/handle contact is permitted, including at the end
            # of a free route. Self/fixture overlap is the strong preference.
            if pair & tool and pair & handle:continue
            if (a in robot and b in robot) or pair & fixture or pair & handle:
                issues.append(dict(geom1=model.geom(a).name,geom2=model.geom(b).name,depth_m=-d))
        if self.path_obstacle_guard is not None:
            observation=self.path_obstacle_guard.observe(data)
            issues=observation['issues']
        return dict(position=self.ik.data.site_xpos[self.ik.site].copy(),
            rotation=self.ik.data.site_xmat[self.ik.site].reshape(3,3).copy(),
            penetration_m=sum(row['depth_m'] for row in issues),contacts=issues,
            obstacle_observation=observation)

    def _prepare_target(self, pos, rotation, seeds, label):
        # Keep original approved seeds as candidates: numerical refinement may
        # find a worse local branch and is never allowed to erase a good seed.
        candidates=[];old_solver=self.ik.solver
        try:
            self.ik.solver='least_squares'
            for seed_index,seed in enumerate(seeds):
                self._sync_ik()
                solved=self.ik.solve(pos,rotation,seed,120)[0]
                for kind,q in (('original_seed',np.asarray(seed).copy()),('least_squares',solved)):
                    geometry=self._static_geometry(q)
                    pe=float(np.linalg.norm(geometry['position']-pos))
                    re=float(Rotation.from_matrix(rotation@geometry['rotation'].T).magnitude())
                    cost=pe+.12*re+80.*geometry['penetration_m']
                    candidates.append(dict(seed_index=seed_index,kind=kind,q=q.copy(),
                        position_error_m=pe,rotation_error_rad=re,geometry=geometry,cost=cost))
        finally:self.ik.solver=old_solver
        candidates.sort(key=lambda row:row['cost'])
        admissible=[row for row in candidates if self.path_obstacle_guard is None
                    or row['geometry']['obstacle_observation']['clear']]
        if not admissible:
            observation=candidates[0]['geometry']['obstacle_observation']
            observation.update(stage=label,candidate_active_joints_rad=candidates[0]['q'].tolist())
            raise PathObstacleConflict(observation)
        chosen=admissible[0]
        self.preparation.append(dict(name=label,position_error_m=chosen['position_error_m'],
            rotation_error_rad=chosen['rotation_error_rad'],candidates=[dict(
                seed_index=row['seed_index'],kind=row['kind'],cost=float(row['cost']),
                position_error_m=row['position_error_m'],rotation_error_rad=row['rotation_error_rad'],
                summed_penetration_m=row['geometry']['penetration_m'],
                obstacle_admissible=(any(a is row for a in admissible)
                    if self.path_obstacle_guard is not None else None),
                geometric_contacts=row['geometry']['contacts']) for row in candidates]))
        return chosen['q'].copy(),chosen['position_error_m']+.12*chosen['rotation_error_rad']

    def _path_geometry(self, points):
        count=max(10,min(20,int(self.p['free_path_samples'])))
        samples=[];pairs=set()
        for index,(start,finish) in enumerate(zip(points[:-1],points[1:])):
            for u in np.linspace(0.,1.,count):
                if index and u==0.:continue
                geometry=self._static_geometry((1.-u)*start+u*finish)
                samples.append(geometry['penetration_m'])
                for row in geometry['contacts']:pairs.add((row['geom1'],row['geom2']))
        return dict(sample_count=len(samples),max_summed_penetration_m=float(max(samples)),
            mean_summed_penetration_m=float(np.mean(samples)),
            overlapping_pairs=[list(pair) for pair in sorted(pairs)])

    def _choose_free_path(self,start,finish,origin,rotation,seed,label,endpoint_error=0.):
        yaw=math.radians(self.env.robot_adapter.to_dict()['placement'].get('base_yaw_deg',0.))
        forward=np.array([math.cos(yaw),math.sin(yaw),0.]);side=np.array([-math.sin(yaw),math.cos(yaw),0.])
        up=np.array([0.,0.,1.])
        offsets=[('above',.10*up),('back',-.12*forward),('front',.10*forward),
            ('left',.12*side),('right',-.12*side),('back_above',-.12*forward+.10*up)]
        paths=[dict(name='direct',points=[start.copy(),finish.copy()],pose_error=float(endpoint_error))]
        old_solver=self.ik.solver
        try:
            self.ik.solver='least_squares'
            for name,offset in offsets:
                self._sync_ik()
                q,pe,re=self.ik.solve(origin+offset,rotation,seed,100)
                paths.append(dict(name=name,points=[start.copy(),q,finish.copy()],
                    pose_error=float(max(endpoint_error,pe+.12*re))))
        finally:self.ik.solver=old_solver
        for path in paths:
            geometry=self._path_geometry(path['points'])
            length=sum(float(np.linalg.norm(b-a)) for a,b in zip(path['points'][:-1],path['points'][1:]))
            path['geometry']=geometry
            path['cost']=80.*geometry['max_summed_penetration_m']+16.*geometry['mean_summed_penetration_m']+10.*path['pose_error']+.001*length
        paths.sort(key=lambda row:row['cost'])
        selected=paths[0]
        if self.path_obstacle_guard is not None:
            duration=self.p['reach_s'] if label=='reach_path_ranking' else self.p['return_s']
            for path in paths:
                path['obstacle_precheck']=self._scan_obstacle_path(
                    lambda u,points=path['points']:self._free(u*duration,points,duration),duration,label)
            reference=self.p['route_names_to_preserve'][label]
            matching=[path for path in paths if path['name']==reference]
            if not matching or not matching[0]['obstacle_precheck']['clear']:
                observation=(matching[0]['obstacle_precheck']['first_block'] if matching else
                    dict(stage=label,clear=False,issues=[],reason='reference_route_missing'))
                observation.update(reference_route=reference,
                    clear_alternatives=[path['name'] for path in paths if path['obstacle_precheck']['clear']],
                    alternate_route_requires_g1_review=True)
                raise PathObstacleConflict(observation)
            selected=matching[0]
        self.preparation.append(dict(name=label,free_path_version=self.p['free_path_version'],selected=selected['name'],
            candidates=[dict(name=row['name'],cost=float(row['cost']),pose_error=row['pose_error'],
                geometry=row['geometry'],obstacle_precheck=row.get('obstacle_precheck'),
                joint_waypoints_rad=[q.tolist() for q in row['points']]) for row in paths]))
        return [q.copy() for q in selected['points']]

    def _prepare(self, poses):
        e, p = self.env, self.p
        # Reset is natural down. Optional offsets alter the active chest target
        # in scratch data only; the physical arm must lift to that posture.
        offset = np.asarray(p['standby_offset_m'], float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('standby_offset_m needs three finite values')
        if np.any(offset):
            self._sync_ik()
            self.ik.data.qpos[e.robot_qpos_indices] = self.chest
            mujoco.mj_forward(e.model, self.ik.data)
            pos = self.ik.data.site_xpos[self.ik.site].copy()+offset
            rot = self.ik.data.site_xmat[self.ik.site].reshape(3,3).copy()
            q, pe, re = self.ik.solve(pos, rot, self.chest[self.active_offsets], 100)
            self.chest[self.active_offsets] = q
            self.preparation.append(dict(name='standby_offset',position_error_m=pe,rotation_error_rad=re))
        e.reset(seed=int(p.get('seed', 0)), initial_robot_qpos=self.natural)
        if self.path_obstacle_guard is not None:
            observation=self._obstacle_for_joints(self.natural[self.active_offsets],'approved_standby')
            if not observation['clear']:raise PathObstacleConflict(observation)
        if self.upright_guard is not None:
            self.preparation.append(dict(name='standby_direction_scope',
                **self.observe_tool_orientation(e.data,stage='approved_standby')))
            if p.get('reviewed_free_connection_file'):
                from .reviewed_free_connections import ReviewedFreeConnections
                self.reviewed_free_connections=ReviewedFreeConnections(
                    p['reviewed_free_connection_file'],e,poses)
                self.preparation.append(dict(name='user_reviewed_lift_and_lower_connections',
                    **self.reviewed_free_connections.summary))
            elif p.get('upright_connection_policy') in (
                    'minimum_wrist_change_on_original_arm_reference_v1',
                    'natural_to_preparation_continuous_v1'):
                from .upright_connection import make_upright_connection
                active_tool=next(key for key,name in p['tool_body_names'].items()
                    if name == e.robot_adapter.operation_body_name)
                self.upright_connection=make_upright_connection(e.model,e.data.qpos.copy(),
                    e.active_qpos_indices,e.active_joint_names,
                    p['upright_connection_adjustable_joint_names'],p['tool_body_names'],active_tool,
                    [self.natural[self.active_offsets],self.lift_waypoint,
                     self.chest[self.active_offsets]],label='standby_chest_connection')
                self.preparation.append(dict(name='upright_standby_chest_connection',
                    **self.upright_connection.summary,
                    knots=self.upright_connection.knots.tolist(),
                    wrist_angles_rad=self.upright_connection.wrist_angles.tolist(),
                    return_policy='same_connection_in_reverse'))
            else:
                self._require_upright_path(
                    [self.natural[self.active_offsets],self.lift_waypoint,
                     self.chest[self.active_offsets]],'lift_to_chest')
                self._require_upright_path(
                    [self.chest[self.active_offsets],self.lift_waypoint,
                     self.natural[self.active_offsets]],'lower_to_natural')
        self._require_clear_connection()
        self.qseed = self.ready[self.active_offsets].copy()
        self._sync_ik()
        pos, rot = self.target(self.opening, clearance=p['approach_clearance_m'])
        preseed = np.asarray(poses[2]['qpos'])[e.active_qpos_indices]
        contactseed = self._approved_contact_seed(self.opening)
        pre,endpoint_error=self._prepare_target(pos,rot,[preseed,contactseed],'precontact')
        self.reach_points=self._choose_free_path(self.chest[self.active_offsets],pre,
            pos,rot,pre,'reach_path_ranking',endpoint_error)
        self._require_upright_path(self.reach_points,'reach')
        # The selected pre joint pose may intentionally retain a small target
        # residual to avoid overlap. Start the next phase at its ACTUAL FK,
        # rather than suddenly commanding the ideal, different pre pose.
        actual_pre=self._static_geometry(pre)
        self.approach_start_pos=actual_pre['position'].copy()
        self.approach_start_rotation=actual_pre['rotation'].copy()
        self.preparation.append(dict(name='approach_phase_join',
            execution_revision=self.p['execution_revision'],
            from_selected_pre_fk=True,
            position_to_ideal_pre_m=float(np.linalg.norm(self.approach_start_pos-pos)),
            rotation_to_ideal_pre_rad=float(Rotation.from_matrix(rot@self.approach_start_rotation.T).magnitude()),
            alternate_branch_during_approach=False))
        self.last_free_q = pre.copy()
        self.reference = self.opening

    def _free(self, t, points, duration):
        u = float(np.clip(t/duration, 0., 1.))*(len(points)-1)
        i = min(int(u),len(points)-2)
        a = blend(u-i)
        return points[i]*(1.-a)+points[i+1]*a

    def _pacing(self, t):
        p, opening, speed = self.p, self.opening, self.speed
        intermediate = self.pause_index < len(self.pause_targets)
        goal = self.pause_targets[self.pause_index] if intermediate else self.goal
        remaining = self.direction*(goal-opening)
        rule = self.slow_open_rule
        # A command to stop does not count: the physical hinge must stay near
        # this target and below the settled-speed threshold continuously.
        if intermediate and abs(opening-goal) <= rule['pause_angle_tolerance_deg'] and abs(speed)<rule['pause_speed_deg_s']:
            self.pause_since = t if self.pause_since is None else self.pause_since
            if t-self.pause_since >= rule['pause_s']:
                self.pause_index += 1
                self.pause_since = None
                intermediate = self.pause_index < len(self.pause_targets)
                goal = self.pause_targets[self.pause_index] if intermediate else self.goal
                remaining = self.direction*(goal-opening)
        elif intermediate:
            self.pause_since = None
        early = self.variant != 'close' and opening < 12.
        desired = p['start_speed_deg_s'] if early else (p['speed_deg_s'] if self.variant == 'close' else p['opening_speed_deg_s'])
        if intermediate:
            desired = min(desired,max(.2,remaining*.7))
        rate = float(np.clip(desired+1.2*(desired-self.direction*speed),-15.,desired*1.5))
        maxlead = p['start_lead_deg'] if early else p['lead_deg']
        self.reference += self.direction*rate*p['control_dt']
        self.reference = opening+self.direction*float(np.clip(self.direction*(self.reference-opening),-.5,maxlead))
        # A small seat target keeps contact while the environment times R1.
        finish = goal if intermediate else goal+self.direction*2.
        self.reference = min(self.reference, finish) if self.direction>0 else max(self.reference, finish)
        phase = f'hold_{goal:g}' if intermediate and remaining<.6 else 'turn'
        if abs(opening-self.goal)<=3. and not intermediate:
            phase='hold_final'
        drive = float(np.clip(1.-max(0.,self.direction*speed-desired)/max(desired,1.),0.,1.))
        if intermediate and phase == f'hold_{goal:g}':
            self.reference=goal
            drive=0.
        return self.reference-opening, phase, drive

    def _slow_feedback_active(self, phase):
        # This optional controller feedback never changes the model or scorer.
        # Opening speed is measured, including motion caused during approach.
        p = self.p
        active = (p['slow_feedback'] and self.variant != 'close' and
                  self.opening <= 12. and phase not in ('reach', 'release', 'return'))
        if phase == 'approach':
            active = active and any(c['kind'] == 'handle' and c['force_N'] > 0.
                                    for c in self.env.current_contacts)
        if not active:
            self.slow_braking = False
        elif abs(self.speed) >= p['start_brake_speed_deg_s']:
            self.slow_braking = True
        elif abs(self.speed) <= p['start_brake_release_deg_s']:
            self.slow_braking = False
        return active

    def _braking_target_lead(self):
        # Oppose the actual motion; the small translational lead changes no
        # reviewed tool face or orientation. target() adds travel direction.
        return -float(np.sign(self.speed))*self.direction*self.p['start_brake_lead_deg']

    def command(self):
        e,p = self.env,self.p
        t=float(e.data.time)
        drive=0.; pe=re=0.; slow_active=False
        if self.lower_started is not None:
            if self.path_obstacle_guard is not None and not self._lower_path_obstacles_checked:
                self._require_clear_connection(reverse=True)
                self._lower_path_obstacles_checked=True
            elapsed=t-self.lower_started
            phase='lower_to_natural' if elapsed < p['lower_s'] else 'final_down_hold'
            q=self._free_connection(elapsed,reverse=True)
            self.integral *= math.exp(-p['control_dt']/.5)
            self.done=elapsed>=p['lower_s']+p['down_hold_s']
        elif self.return_started is not None:
            phase='return_chest'
            q=self._free(t-self.return_started,self.return_points,p['return_s'])
            self.integral *= math.exp(-p['control_dt']/.5)
            if t-self.return_started>=p['return_s']+p['chest_hold_s']:
                self.lower_started=t
        elif self.withdraw_started is not None:
            phase='release'
            u=blend((t-self.withdraw_started)/p['withdraw_s'])
            pos,rot=self.target(self.opening,clearance=p['approach_clearance_m']*u)
            self._sync_ik()
            self._motion_candidate_stage='release_candidate'
            q,pe,re=self._solve_contact(pos,rot,self.qseed,p['ik_iterations'],self.opening)
            self.integral *= math.exp(-p['control_dt']/.5)
            if t-self.withdraw_started>=p['withdraw_s']:
                self.return_started=t
                # Compare free return routes in the CURRENT valve geometry.
                self.return_points=self._choose_free_path(q,self.chest[self.active_offsets],
                    pos,rot,q,'return_path_ranking')
                self._require_upright_path(self.return_points,'return_chest')
        elif t < p['down_hold_s']:
            phase='initial_down_hold'
            q=self.natural[self.active_offsets].copy()
        elif t < p['down_hold_s']+p['lift_s']:
            phase='lift_to_chest'
            q=self._free_connection(t-p['down_hold_s'])
        elif t < self.reach_start_s:
            phase='chest_ready'
            q=self.chest[self.active_offsets].copy()
        elif t < self.reach_start_s+p['reach_s']:
            phase='reach'
            q=self._free(t-self.reach_start_s,self.reach_points,p['reach_s'])
        else:
            contact_t=t-self.reach_start_s-p['reach_s']
            feedback_open = bool(p['slow_feedback'] and self.variant != 'close')
            approach_clock = self.approach_elapsed_s if feedback_open else contact_t
            if approach_clock < p['approach_s']:
                phase='approach'
                slow_active=self._slow_feedback_active(phase)
                lead=self._braking_target_lead() if self.slow_braking else 0.
                contact_pos,contact_rot=self.target(self.opening,lead=lead,clearance=0.)
                pos,rot=interpolate_pose(self.approach_start_pos,self.approach_start_rotation,
                    contact_pos,contact_rot,approach_clock/p['approach_s'])
                # Freeze approach advancement while braking rather than
                # resuming with a jump to an elapsed wall-clock waypoint.
                if feedback_open and not self.slow_braking:
                    self.approach_elapsed_s += p['control_dt']
            else:
                lead,phase,drive=self._pacing(t)
                slow_active=self._slow_feedback_active(phase)
                # _pacing returns signed opening difference; target expects a
                # positive travel lead and adds the selected direction itself.
                target_lead=lead*self.direction
                if self.slow_braking:
                    target_lead=self._braking_target_lead()
                    self.reference=self.opening+self.direction*target_lead
                    drive=0.
                elif slow_active:
                    drive *= p['start_force_scale']
                pos,rot=self.target(self.opening,lead=target_lead)
                if feedback_open:
                    if self.feedback_turn_started is None:self.feedback_turn_started=t
                    force_elapsed=t-self.feedback_turn_started
                else:
                    force_elapsed=contact_t-p['approach_s']
                drive *= min(1.,force_elapsed/p['force_ramp_s'])
                if e.scorer.completion_time is not None:
                    self.withdraw_started=t
            self._sync_ik()
            # Continue from the previous joint branch throughout approach;
            # better absolute IK accuracy must not trigger a branch jump at
            # this phase boundary. This is control continuity, not a fail gate.
            self._motion_candidate_stage=phase+'_candidate'
            q,pe,re=self._solve_contact(pos,rot,self.qseed,p['ik_iterations'],self.opening,
                allow_alternate=phase!='approach')
        return self._motor_action(q, phase=phase, position_error=pe,
            rotation_error=re, drive=drive, slow_active=slow_active)

    def _motor_action(self, q, *, phase, position_error=0., rotation_error=0.,
                      drive=0., slow_active=False):
        """Shared, unchanged finite-motor command for legacy and S4 phases."""
        e,p=self.env,self.p
        t=float(e.data.time)
        pe,re=position_error,rotation_error
        self.qseed=q.copy()
        self._sync_ik()
        self.ik.data.qpos[self.ik.qadr]=q
        mujoco.mj_forward(e.model,self.ik.data)
        if self.upright_guard is not None:
            self.observe_tool_orientation(self.ik.data,stage=phase+'_candidate')
        kp=e.model.actuator_gainprm[e.active_actuator_ids,0]
        gravity=np.clip(self.ik.data.qfrc_bias[self.ik.dofs]/kp,-.2,.2)
        if phase not in ('release','return_chest','lower_to_natural','final_down_hold'):
            if p['slow_feedback'] and phase in {f'hold_{angle:g}' for angle in self.pause_targets}:
                self.integral *= math.exp(-p['control_dt']/max(p['pause_integral_decay_s'],1e-6))
            elif slow_active and self.slow_braking:
                self.integral *= math.exp(-p['control_dt']/max(p['start_brake_integral_decay_s'],1e-6))
            else:
                limit=p['start_integral_limit_rad'] if slow_active else .16
                self.integral=np.clip(self.integral+.2*p['control_dt']*(q-e.data.qpos[self.ik.qadr]),-limit,limit)
        self.velocity=.7*self.velocity+.3*e.data.qvel[self.ik.dofs]
        damping=p['start_velocity_damping'] if slow_active else .07
        action=self.ready.copy()
        action[self.active_offsets]=q+gravity+self.integral-damping*self.velocity
        idle_kp=e.model.actuator_gainprm[e.idle_actuator_ids,0]
        action[self.idle_offsets]+=np.clip(e.data.qfrc_bias[self.idle_dofs]/idle_kp,-.2,.2)
        if p['force_N']>0. and drive>0.:
            point=e.data.site_xpos[self.ik.site]
            jp=np.zeros((3,e.model.nv));jr=np.zeros_like(jp)
            mujoco.mj_jac(e.model,e.data,jp,jr,point,e.active_hook_body_id)
            radial=e.station_spec.radial_at(e.valve_angle_rad)
            qdirection=np.sign((e.q_close_rad-e.q_open_rad) if self.variant=='close' else (e.q_open_rad-e.q_close_rad))
            sign=qdirection*np.sign(e.valve_hinge_axis_world@e.stem_axis_world)
            direction=sign*np.cross(e.stem_axis_world,radial)
            action[self.active_offsets]+=jp[:,self.ik.dofs].T@(p['force_N']*drive*direction)/kp
        self.last_diagnostics=dict(time_s=t,phase=phase,opening_deg=self.opening,speed_deg_s=self.speed,
            reference_opening_deg=self.reference,ik_position_error_m=pe,ik_rotation_error_rad=re,
            force_drive_fraction=drive,pause_index=self.pause_index,
            completed_pause_angles_deg=self.pause_targets[:self.pause_index],
            pause_stable_since_s=self.pause_since,
            slow_feedback_active=bool(slow_active),slow_feedback_braking=bool(self.slow_braking),
            integral_max_abs_rad=float(np.max(np.abs(self.integral))),controller_version=p['controller_version'],free_path_version=p['free_path_version'],execution_revision=p['execution_revision'])
        return action


def apply_approved_cameras(env, cameras):
    """Apply frozen camera XML to recording scene and existing model cameras."""
    root=ET.fromstring(env.xml)
    for descriptor in cameras.values():
        node=ET.fromstring(descriptor['xml'])
        name=node.attrib['name']
        parent_body=descriptor.get('parent_body')
        parent=root.find('worldbody') if parent_body in (None,'world','worldbody') else root.find(f".//body[@name='{parent_body}']")
        if parent is None:raise ValueError(f'Missing approved camera parent: {parent_body}')
        old=parent.find(f"camera[@name='{name}']")
        if old is not None:parent.remove(old)
        parent.append(copy.deepcopy(node))
        cid=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_CAMERA,name)
        if cid < 0:raise ValueError(f'Missing model camera: {name}')
        if 'pos' in node.attrib:env.model.cam_pos[cid]=np.fromstring(node.attrib['pos'],sep=' ')
        if 'quat' in node.attrib:
            env.model.cam_quat[cid]=np.fromstring(node.attrib['quat'],sep=' ')
        elif 'xyaxes' in node.attrib:
            axes=np.fromstring(node.attrib['xyaxes'],sep=' ').reshape(2,3)
            x=axes[0]/np.linalg.norm(axes[0]);y=axes[1]-x*(axes[1]@x);y/=np.linalg.norm(y)
            rotation=np.column_stack((x,y,np.cross(x,y)))
            quat=np.zeros(4);mujoco.mju_mat2Quat(quat,rotation.ravel());env.model.cam_quat[cid]=quat
        env.model.cam_fovy[cid]=float(node.attrib['fovy'])
    env.xml=ET.tostring(root,encoding='unicode')
    env.camera_parameters=copy.deepcopy(cameras)


def run_episode(job, output, record=False):
    """Run one job locally; never starts/connects to any GPU or remote host."""
    # Old results remain readable, but cannot silently launch a new episode
    # with the superseded one-pause rule.
    procedure_version=job.get('procedure_version')
    if procedure_version not in ('A1-v2.3','A1-v2.4','A1-v2.5'):
        raise ValueError('New G2 episodes require an explicitly reviewed A1-v2.3/v2.4/v2.5 plan; historical plans are read-only')
    required_procedure=job.get('required_procedure_version')
    if required_procedure is not None and procedure_version!=required_procedure:
        raise ValueError('Job exam version differs from the reviewed batch requirement')
    start=time.monotonic();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    station=job.get('station','A1');variant=job.get('variant',job.get('operation','close'))
    if station not in ('A1','A2') or variant not in ('close','open','slow_open'):
        raise ValueError('G2 currently supports approved A1/A2 close/open/slow_open methods')
    task='close' if variant=='close' else 'open'
    setup_path=Path(job.get('approved_setup',APPROVED))
    if not setup_path.is_absolute():setup_path=ROOT/setup_path
    setup=json.loads(setup_path.read_text(encoding='utf-8'))
    if not setup.get('user_approved') or not setup.get('frozen_for_G2'):
        raise ValueError('G2 requires the user-approved frozen setup')
    approved_station=setup['stations'][station]
    expected=Path(approved_station['accepted_keyposes'][task])
    if not expected.is_absolute():expected=ROOT/expected
    source=Path(job['keyposes']) if job.get('keyposes') else expected
    if not source.is_absolute():source=ROOT/source
    source=source.resolve()
    if source != expected.resolve():
        raise ValueError('G2 keyposes must be the current frozen approved document')
    document=json.loads(source.read_text(encoding='utf-8'))
    placement=copy.deepcopy(approved_station['placement'])
    if document['placement'] != placement or (job.get('approved_placement') is not None and job['approved_placement'] != placement):
        raise ValueError('Placement must match current approved G1 and frozen setup')
    if placement['base_yaw_deg'] != 0:
        raise ValueError('Approved body yaw must stay zero')
    if document['method']['station'] != station or document['method']['task'] != task or document['method']['side'] != 'left':
        raise ValueError('Provided approved keyposes do not match this station/task/left-hand method')
    method=copy.deepcopy(approved_station['methods'][task])
    for key,value in method.items():
        if document['method'].get(key) != value:
            raise ValueError(f'Approved method mismatch: {key}')
    execution_context=None
    if job.get('execution_profile') is not None:
        from .g2_stance_episode import validate_job_context
        execution_context=validate_job_context(job,root=ROOT,require_execution_approval=True)
        placement=copy.deepcopy(execution_context['actual_execution_placement'])
    elif any(key in job for key in ('execution_placement','nominal_chest_distance_m','execution_approval')):
        raise ValueError('Experimental placement fields require an explicit stance execution_profile')
    parameters=dict(DEFAULTS)
    supplied=dict(job.get('parameters',job.get('candidate',{})))
    aliases={'r':'r_m','approach_clearance':'approach_clearance_m','pitch_offset':'pitch_offset_deg',
        'tilt_offset':'tilt_offset_deg','lead':'lead_deg','start_lead':'start_lead_deg','standby_offset':'standby_offset_m','speed':'speed_deg_s'}
    for key,val in supplied.items():parameters[aliases.get(key,key)]=val
    rule=load_slow_open_rule()
    if 'pause_hold_s' in supplied and float(supplied['pause_hold_s']) != rule['pause_s']:
        raise ValueError('pause_hold_s conflicts with the authoritative slow-opening rule; update and review the plan')
    parameters['pause_hold_s']=rule['pause_s']
    parameters['slow_open_rule']=rule
    if 'opening_speed_deg_s' not in supplied:
        parameters['opening_speed_deg_s']=parameters['speed_deg_s']
    reviewed_version=job.get('approved_path_mode')
    reviewed_path_mode=reviewed_version in ('a2_reviewed_forehand_path_v1','a2_reviewed_forehand_path_v2_seated35','a2_forehand_v3_seated_continuous')
    if reviewed_path_mode:
        if setup.get('controller_version')!=reviewed_version or not document.get('approved_path'):
            raise ValueError('Reviewed forehand execution needs its frozen path setup and document')
        parameters['controller_version']=reviewed_version
        parameters['lift_path']='reviewed_approach_without_forced_chest_waypoint'
        parameters['free_path_version']='reviewed_approach_exit_v1'
        parameters['execution_revision']=reviewed_version
        parameters['reviewed_path_source']=str(source)
    else:
        parameters['controller_version']='approved_g1_auto_v4_dual_pause'
        parameters['lift_path']='shoulder_back_then_chest'
        parameters['free_path_version']='geometric_candidates_v1'
        parameters['execution_revision']='continuous_approach_join_v1'
    parameters['feedback_version']='g2_slow_feedback_v1' if parameters['slow_feedback'] else 'disabled'
    parameters['seed']=int(job.get('seed',0))
    if parameters['r_m'] is None:parameters['r_m']=method['radius_m']
    if parameters['force_N'] is None:parameters['force_N']=120. if method['mode']=='pull' else 0.
    offset=np.asarray(job.get('offset_m',[0.,0.,0.]),float)
    scale=float(job.get('resistance_scale',1.))
    if offset.shape!=(3,) or not np.isfinite(offset).all() or not .5<=scale<=1.5:
        raise ValueError('Need finite 3-vector offset and resistance_scale in [0.5,1.5]')
    controller_factory=G2Controller
    if reviewed_path_mode:
        from .t2_forehand_controller import ReviewedForehandController
        controller_factory=ReviewedForehandController
    return _run_prepared_episode(job,out,start=start,procedure_version=procedure_version,
        station=station,variant=variant,setup=setup,setup_path=setup_path,source=source,
        approved_station=approved_station,placement=placement,method=method,
        document=document,parameters=parameters,offset=offset,scale=scale,
        execution_context=execution_context,controller_factory=controller_factory,
        record=record)


def _run_prepared_episode(job,out,*,start,procedure_version,station,variant,setup,
        setup_path,source,approved_station,placement,method,document,parameters,
        offset,scale,execution_context=None,controller_factory=G2Controller,
        workstation=None,input_metadata=None,robot_adapter=None,record=False):
    """Shared physical loop; input loaders do not change controller behavior.

    The legacy entry resolves its historical setup above. The V2 entry resolves
    formal workstation/storyboard data and supplies the same controller inputs.
    """
    # Resistance is the declared condition, set once before reset and unchanged
    # throughout this episode. No easier retry or action-dependent load change.
    from .h0_robot import load_robot_adapter
    adapter=load_robot_adapter('openarm_s_hook' if robot_adapter is None else robot_adapter).to_dict()
    if adapter.get('model_revision'):
        binding=(input_metadata or {}).get('model_binding') or {}
        if (binding.get('model_revision') != adapter['model_revision']
                or binding.get('robot_adapter_id') != adapter['adapter_id']):
            raise ValueError('Versioned robot installation needs matching reviewed input metadata')
    head=approved_station['cameras']['head']['metadata']
    if adapter.get('camera_rig'):
        # The shared rig installs its optical frames and physical parents.
        # A mounted camera's local -Z is not a torso-relative look direction.
        # Frozen descriptors are applied to their true parents just below.
        adapter['cameras']=[]
    else:
        adapter['cameras']=[dict(name=head['name'],parent_body='openarm_body_link0',
            position_local_m=head['local_position_m'],forward_direction_local=head['forward_direction_local'],
            vertical_fov_deg=head['vertical_fov_deg'],profile=setup.get('controller_version','approved_g1_auto_v3'))]
    adapter.pop('camera_profiles',None)
    raw=build_env(station if workstation is None else workstation,variant,robot_adapter=adapter,placement=placement,
        valve_offset=offset,control_dt=parameters['control_dt'],max_time_s=90.)
    apply_approved_cameras(raw,approved_station['cameras'])
    raw.resistance.scale*=scale
    env=ScoredValveEnv(raw,variant,source='G2 native-motor physical episode',
        procedure_version=procedure_version)
    controller=None;reason='not_started';error=None;timeline=[];states=[]
    upright_block=None;upright_physics_samples=0;path_block=None
    try:
        if input_metadata is not None:
            # A saved full-qpos vector is meaningful only in its named model
            # order. Reject a mismatched model instead of moving another joint.
            if method.get('side') != env.side:
                raise ValueError('Formal operating hand differs from the supplied robot adapter')
            model_names=[env.model.joint(i).name for i in range(env.model.njnt)]
            for pose in document['keyposes']:
                if pose['qpos_joint_names'] != model_names or len(pose['qpos']) != env.model.nq:
                    raise ValueError('Formal saved-joint mapping differs from the physical model')
        env.reset(seed=parameters['seed'])
        controller_document=(document['keyposes'] if controller_factory is G2Controller else document)
        controller=controller_factory(env,variant,method,controller_document,parameters)
        if getattr(controller,'upright_guard',None) is not None:
            previous_observer=raw.physics_observer
            def observe_upright(native):
                nonlocal upright_physics_samples
                if previous_observer:previous_observer(native)
                upright_physics_samples+=1
                stage=controller.last_diagnostics.get('phase','initial')
                controller.observe_tool_orientation(native.data,stage=stage+'_actual')
            raw.physics_observer=observe_upright
        if procedure_version in ('A1-v2.4','A1-v2.5'):
            # Constructors prepare/reset at t=0. Record once after that work,
            # before any physical step, so resets do not duplicate the origin.
            env.start_evidence_recording(out/'physical_step_evidence.jsonl.gz')
        reason='calculation_cutoff';last_sample=-1.;last_phase=None
        while env.data.time < env.max_time_s-1e-9:
            if time.monotonic()-start>parameters['max_wall_s']:
                reason='local_wall_time_limit';break
            action=controller.command()
            phase=controller.last_diagnostics['phase']
            env.set_scoring_context(**controller.last_diagnostics, action_finished=controller.done)
            if env.data.time-last_sample>=.02-1e-9:
                timeline.append(controller.last_diagnostics.copy())
                states.append((float(env.data.time),env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy()))
                last_sample=float(env.data.time)
            if phase!=last_phase:
                print(f"{job.get('id',out.name)} {phase} t={env.data.time:.2f}s opening={controller.opening:.2f}",flush=True)
                last_phase=phase
            if getattr(controller,'blocked_reason',None):
                reason='actual_stage_not_completed';break
            env.step(action)
            if not np.isfinite(env.data.qpos).all() or env.warning_count:
                reason='numerical_failure';break
            # Hard collision remains an independent safety mark. Elapsed time
            # only ends computation and never creates an exam failure rule.
            if env.scorer.counts['hard_contact']>0:
                reason='R3_hard_contact';break
            if controller.done:
                reason='controller_completed';break
    except ToolUprightConflict as exc:
        reason='tool_upright_blocked';upright_block=exc.observation
        save(out/'tool_upright_block.json',upright_block)
    except PathObstacleConflict as exc:
        reason='path_obstacle_blocked';path_block=exc.observation
        save(out/'path_obstacle_block.json',path_block)
    except Exception as exc:
        reason='runner_error';error=type(exc).__name__+': '+str(exc)
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        score=env.score()
        result=dict(id=job.get('id',out.name),station=station,variant=variant,offset_m=offset.tolist(),
            procedure_version=procedure_version,
            resistance_scale=scale,parameters=parameters,controller_version=parameters['controller_version'],feedback_version=parameters['feedback_version'],passed=score['passed'],score=score,
            wall_s=time.monotonic()-start,simulation_s=float(env.data.time),stop_reason=reason,error=error,
            final_opening_deg=float((env.valve_angle_rad-env.q_close_rad)*90./(env.q_open_rad-env.q_close_rad)),
            controller_completed=bool(controller and controller.done),data_kind='recorded_simulation',
            lift_path=parameters['lift_path'],lift_waypoint=parameters['lift_waypoint'],free_path_version=parameters['free_path_version'],execution_revision=parameters['execution_revision'],
            initial_posture='both_arms_natural_down_zero_joints_straight_elbows',
            idle_posture='natural_down_throughout_with_native_motor_gravity_compensation',
            stage_timing_s={k:parameters[k] for k in ('down_hold_s','lift_s','chest_hold_s','reach_s','approach_s','withdraw_s','return_s','lower_s')},
            approved_setup=str(setup_path.resolve()),approved_placement=placement,
            approved_method=method,approved_poses=str(source),preparation=controller.preparation if controller else [])
        # The old approval describes the reference only. A new user-reviewed
        # experiment never rewrites that approval or calls its placement old G1.
        result['approved_reference_placement']=copy.deepcopy(approved_station['placement'])
        result['actual_execution_placement']=copy.deepcopy(placement)
        motion_blocked=bool(upright_block or path_block)
        if controller is not None and hasattr(controller,'phase_gate'):
            result['measured_phase_execution']=dict(
                version=controller.phase_version,
                final_phase=controller.phase_gate.phase,
                blocked_reason=controller.blocked_reason,
                events=controller.phase_gate.events,
                controller_settings=controller.phase_settings,
                contact_radius_search_enabled=hasattr(controller,'radius_choice'),score_changed=False)
            save(out/'measured_phase_execution.json',result['measured_phase_execution'])
        if controller is not None and hasattr(controller,'radius_summary'):
            result['contact_radius_selection'] = controller.radius_summary()
            save(out/'contact_radius_selection.json',result['contact_radius_selection'])
        if parameters.get('path_obstacles_required'):
            result['path_obstacle_guard']=dict(enabled=True,blocked=path_block,
                candidate_checks=controller.path_obstacle_checks if controller else None,
                no_additional_clearance=True,score_modified=False,
                route_names_to_preserve=parameters['route_names_to_preserve'])
            if controller is not None:
                save(out/'path_obstacle_rejections.json',controller.path_obstacle_rejections)
        if parameters.get('tool_upright_required'):
            result['initial_posture']=('user_reviewed_shared_bimanual_natural_rest'
                if parameters.get('tool_orientation_policy') == ORIENTATION_POLICY
                else 'user_approved_G1_standby')
            result['tool_upright_guard']=dict(enabled=True,blocked=upright_block,
                actual_physics_samples_checked=upright_physics_samples,
                physics_started=bool(float(env.data.time)>0.),
                policy_version=ORIENTATION_POLICY,
                working_hand_only=True,colour_centre_height_used=False,
                score_modified=False)
            if float(env.data.time)==0.:
                result['data_kind']='planning_check_only_no_physics'
                result['execution_status']='blocked_before_motion' if motion_blocked else 'not_started'
                result['passed']=None
                result['assessment_applicability']='not_applicable_no_physics'
                result['eligible_for_motion_statistics']=False
                save(out/'score_applicability.json',dict(
                    applicable=False,reason='No physical step was executed',
                    raw_scorer_file={'A1-v2.3':'score_v23.json','A1-v2.4':'score_v24.json','A1-v2.5':'score_v25.json'}[procedure_version],
                    raw_score_usage='Initial-state diagnostic only; exclude from motion success/failure counts'))
            else:
                result['execution_status']='blocked_during_motion' if motion_blocked else reason
        if input_metadata is not None:
            result['input_metadata']=copy.deepcopy(input_metadata)
            save(out/'formal_storyboard.json',input_metadata['storyboard_snapshot'])
            save(out/'formal_workstation.json',input_metadata['workstation_snapshot'])
        recording_cameras=copy.deepcopy(approved_station['cameras'])
        if execution_context is not None:
            result['approved_placement']=copy.deepcopy(approved_station['placement'])
            result['execution_context']=copy.deepcopy(execution_context)
            result['execution_profile']=execution_context['execution_profile']
            result['experiment_revision']=execution_context['experiment_revision']
            result['experiment_placement_is_original_G1_approved']=execution_context['experiment_placement_is_original_G1_approved']
            for view,camera in recording_cameras.items():
                metadata=camera['metadata']
                name='robot_head' if view=='head' else 'station_observer'
                cid=env.model.camera(name).id
                metadata['approved_reference_nominal_world_position_m']=copy.deepcopy(metadata.get('nominal_world_position_m'))
                metadata['actual_world_position_m']=env.data.cam_xpos[cid].tolist()
                metadata['actual_world_rotation_matrix']=env.data.cam_xmat[cid].reshape(3,3).tolist()
                metadata['actual_execution_placement']=copy.deepcopy(placement)
                metadata['experiment_placement_is_original_G1_approved']=execution_context['experiment_placement_is_original_G1_approved']
        result['trajectory_path']=None
        if record or score['passed'] is True:
            if not states or abs(states[-1][0]-float(env.data.time))>1e-8:
                states.append((float(env.data.time),env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy()))
            np.savez_compressed(out/'physical_states.npz',time_s=[s[0] for s in states],
                qpos=[s[1] for s in states],qvel=[s[2] for s in states],ctrl=[s[3] for s in states])
            (out/'scene.xml').write_text(env.xml,encoding='utf-8')
            result['trajectory_path']=str(out/'physical_states.npz')
        if procedure_version in ('A1-v2.4','A1-v2.5'):
            result['physical_step_evidence']=env.finish_evidence_recording(
                execution_complete=reason in ('controller_completed','calculation_cutoff','R3_hard_contact'))
        score_name={'A1-v2.3':'score_v23.json','A1-v2.4':'score_v24.json','A1-v2.5':'score_v25.json'}[procedure_version]
        save(out/'result.json',result);save(out/score_name,score);save(out/'timeline.json',timeline)
        save(out/'job.json',job);save(out/'station.json',env.station_spec.to_dict())
        save(out/'robot_adapter.json',env.robot_adapter.to_dict())
        save(out/'approved_setup.json',setup)
        if execution_context is not None:
            save(out/'execution_context.json',execution_context)
        save(out/'recording_cameras.json',recording_cameras)
        env.close()
    print(json.dumps({k:result[k] for k in ('id','passed','stop_reason','final_opening_deg','wall_s')},ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--record',action='store_true');a=p.parse_args()
    run_episode(json.loads(a.job.read_text(encoding='utf-8')),a.output,a.record)


if __name__=='__main__':main()
