"""Track reviewed A2 robot paths through finite motors and measured hinge feedback.

Saved qpos is used only as an IK seed. The physical hinge is never assigned,
except by the normal environment reset before the episode begins.
"""
from __future__ import annotations

import math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .g2_episode import G2Controller, blend
from .h1_kinematics import H1IK
from .slow_open_rule import load_slow_open_rule

VERSION = 'a2_reviewed_forehand_path_v1'
VERSION_35 = 'a2_reviewed_forehand_path_v2_seated35'
VERSION_CONTINUOUS = 'a2_forehand_v3_seated_continuous'
REVIEWED_VERSIONS = (VERSION, VERSION_35, VERSION_CONTINUOUS)


class ReviewedPath:
    def __init__(self, rows, active_qpos_indices, *, contact=False):
        if len(rows)<2:
            raise ValueError('Reviewed path needs at least two samples')
        self.rows=rows
        self.positions=np.asarray([r['tcp_world_m'] for r in rows],float)
        self.rotations=np.asarray([r['rotation_world'] for r in rows],float)
        self.joints=np.asarray([np.asarray(r['qpos'])[active_qpos_indices] for r in rows],float)
        self.angles=np.asarray([r['opening_deg'] for r in rows],float)
        if not (np.isfinite(self.positions).all() and np.isfinite(self.rotations).all() and np.isfinite(self.joints).all()):
            raise ValueError('Reviewed path has nonfinite geometry')
        if contact:
            # Last duplicate retains the reviewed wrist preparation at 90°.
            unique={float(angle):i for i,angle in enumerate(self.angles)}
            indices=[unique[angle] for angle in sorted(unique)]
            self.x=self.angles[indices]
            self.positions=self.positions[indices];self.rotations=self.rotations[indices]
            self.joints=self.joints[indices];self.angles=self.angles[indices]
        else:
            increments=np.linalg.norm(np.diff(self.joints,axis=0),axis=1)
            self.x=np.r_[0.,np.cumsum(np.maximum(increments,1e-6))]
            self.x/=self.x[-1]
        self.slerp=Slerp(self.x,Rotation.from_matrix(self.rotations))

    def at(self, coordinate):
        x=float(np.clip(coordinate,self.x[0],self.x[-1]))
        position=np.asarray([np.interp(x,self.x,self.positions[:,i]) for i in range(3)])
        joints=np.asarray([np.interp(x,self.x,self.joints[:,i]) for i in range(7)])
        return position,self.slerp([x]).as_matrix()[0],joints


class ReviewedForehandController(G2Controller):
    def __init__(self, env, variant, method, document, parameters):
        if variant not in ('close','open') or method.get('station')!='A2' or method.get('side')!='left':
            raise ValueError('Reviewed A2 controller supports left-hand ordinary open/close only')
        self.env,self.variant,self.method,self.p=env,variant,method,parameters
        self.controller_version=parameters.get('controller_version',VERSION)
        self.continuous=self.controller_version==VERSION_CONTINUOUS
        self.seated35=self.controller_version in (VERSION_35,VERSION_CONTINUOUS)
        if self.continuous:
            from .contact_target import target_radius
            self.p=dict(parameters)
            self.p['r_m']=target_radius(env.station_spec.handle_length_m)
            approval=document.get('execution_approval',{})
            if not (approval.get('approved') is True and approval.get('candidate_id')
                    and approval.get('scope')=='complete_approach_contact_exit'
                    and approval.get('controller_version')==VERSION_CONTINUOUS):
                raise ValueError('A2 v3 needs user selection and complete approach/contact/exit approval before execution')
            self.free_plans={};self.seating_state={}
            self.close_exit_reconnection=(document.get('close_exit_reconnection')
                if variant=='close' else None)
        self.seated_since=None
        self.direction=-1. if variant=='close' else 1.
        self.goal=0. if variant=='close' else 90.
        self.ik=H1IK(env,solver='dls')
        self.active_offsets=[env.robot_joint_names.index(n) for n in env.active_joint_names]
        self.idle_offsets=[env.robot_joint_names.index(n) for n in env.idle_joint_names]
        self.idle_dofs=env.model.jnt_dofadr[[env.model.joint(n).id for n in env.idle_joint_names]]
        self.natural=np.asarray(document['natural_start_qpos'],float)[env.robot_qpos_indices].copy()
        if not np.allclose(self.natural,0.,atol=1e-9):
            raise ValueError('Both arms must start naturally down')
        self.ready=self.natural.copy()
        self.approved_poses=document['keyposes']
        path=document['approved_path']
        self.approach_path=ReviewedPath(path['approach'],env.active_qpos_indices)
        self.contact_path=ReviewedPath(path['contact'],env.active_qpos_indices,contact=True)
        self.exit_path=ReviewedPath(path['exit'],env.active_qpos_indices)
        self.contact_seed_rows=path['contact']
        self.nominal_pivot=np.asarray(document['nominal_valve_pivot_world_m'],float)
        if self.seated35:
            if document.get('controller_version')!=self.controller_version:
                raise ValueError('35cm controller needs its new frozen path, never the 20cm document')
            if parameters['pitch_offset_deg'] or parameters['tilt_offset_deg']:
                raise ValueError('The reviewed 35cm wrist orientation must stay unchanged')
            self.contact_path=HandleFrameContactPath(path['contact'],env.active_qpos_indices,
                self.nominal_pivot,env.valve_hinge_axis_world*(env.q_open_rad-env.q_close_rad)/90.)
        self.offset=np.asarray(env.valve_pivot_world)-self.nominal_pivot
        self.approach_duration=parameters['lift_s']+parameters['reach_s']+parameters['approach_s']
        self.exit_duration=parameters['withdraw_s']+parameters['return_s']+parameters['lower_s']
        self.qseed=self.natural[self.active_offsets].copy()
        self.integral=np.zeros(7);self.velocity=np.zeros(7)
        self.reference=self.opening;self.slow_open_rule=load_slow_open_rule()
        self.pause_targets=[];self.pause_index=0;self.pause_since=None
        self.withdraw_started=None;self.exit_start_q=None;self.exit_anchor_rotation=np.eye(3)
        self.done=False;self.last_diagnostics={}
        self.preparation=[dict(name='reviewed_forehand_paths',controller_version=self.controller_version,
            approach_samples=len(path['approach']),contact_samples=len(path['contact']),
            exit_samples=len(path['exit']),contact_policy='measured hinge feedback, no valve state replay',
            free_policy='reviewed approach and exit; no automatic chest return or route ranking',
            translated_valve_offset_m=self.offset.tolist())]
        if self.seated35:
            self.preparation[0]['opening_start_policy']='hold approved seated geometry until a solved orange working face touches; then turn'
        if self.continuous:
            self.preparation[0].update(
                seating_policy='current finite recess geometry and current correct working face; never latched',
                free_policy='one continuous scratch-IK branch, then smooth natural-down joint finish',
                execution_approval=dict(document['execution_approval']))
        env.reset(seed=int(parameters.get('seed',0)),initial_robot_qpos=self.natural)

    def _approved_contact_seed(self, opening):
        return self.contact_path.at(opening)[2]

    def _orange_seat_faces(self):
        return sorted({(c['working_face_geometry'].get('segment'),c['working_face_geometry'].get('side'))
            for c in self.env.current_contacts
            if c.get('kind')=='handle' and c.get('robot_part')=='operation_end'
            and c.get('working_face')=='orange_push' and c.get('force_N',0.)>0.
            and c.get('working_face_geometry',{}).get('segment') is not None})

    def _can_start_turn(self,t):
        if getattr(self,'continuous',False):
            self.seating_state=self._current_seating()
            if not self.seating_state['ready_to_turn']:
                self.seated_since=None
                self.reference=self.opening
                return False
            if self.seated_since is None:
                self.seated_since=t
                self.reference=self.opening
            return True
        if not self.seated35 or self.variant!='open':return True
        if self.seated_since is not None:return True
        if self._orange_seat_faces():
            self.seated_since=t
            self.reference=self.opening
            return True
        return False

    def _current_seating(self):
        from .forehand_continuity import recess_seating_geometry
        from .h0_station import SIX_CONFIG
        import json
        if not hasattr(self,'handle_section'):
            dimensions=json.loads(SIX_CONFIG.read_text())['valve']
            self.handle_section=(dimensions['handle_contact_width_m'],dimensions['handle_contact_thickness_m'])
        e=self.env;body=e.active_hook_body_id
        evidence=recess_seating_geometry(tool_origin=e.data.xpos[body],
            tool_rotation=e.data.xmat[body].reshape(3,3),pivot=e.valve_pivot_world,
            radial=e.station_spec.radial_at(e.valve_angle_rad),stem=e.stem_axis_world,
            mode='pull' if self.variant=='close' else 'push',target_radius_m=self.p['r_m'],
            handle_length_m=e.station_spec.handle_length_m,
            handle_width_m=self.handle_section[0],handle_thickness_m=self.handle_section[1])
        faces=('blue_pull','blue_hook') if self.variant=='close' else ('orange_push',)
        correct=[c for c in e.current_contacts if c.get('kind')=='handle'
            and c.get('robot_part')=='operation_end' and c.get('working_face') in faces
            and c.get('force_N',0.)>0.]
        evidence.update(current_correct_face_contact=bool(correct),
            ready_to_turn=bool(evidence['geometry_seated'] and correct))
        return evidence

    def _solve_contact(self,pos,rotation,seed,iterations,opening,*,allow_alternate=True):
        if not getattr(self,'continuous',False):
            return super()._solve_contact(pos,rotation,seed,iterations,opening,allow_alternate=allow_alternate)
        from .forehand_continuity import bounded_continuation_solve
        self._sync_ik()
        return bounded_continuation_solve(self.ik,pos,rotation,seed,iterations)

    def _orientation_adjustment(self, rotation, weight=1.):
        # Candidate perturbations preserve the approved face relationship;
        # no historical roll/pitch construction is used as a new target.
        adjustment=Rotation.from_rotvec(np.radians([
            self.p['pitch_offset_deg']*weight,self.p['tilt_offset_deg']*weight,0.])).as_matrix()
        return rotation@adjustment

    def target(self, measured, *, lead=0., clearance=0.):
        if clearance:
            raise ValueError('Reviewed forehand release must follow its approved exit route')
        position,rotation,_=self.contact_path.at(measured)
        q=self.env.q_close_rad+(self.env.q_open_rad-self.env.q_close_rad)*measured/90.
        ahead=q+(self.env.q_open_rad-self.env.q_close_rad)*self.direction*lead/90.
        delta=self.p['r_m']*(self.env.station_spec.radial_at(ahead)-self.env.station_spec.radial_at(q))
        return position+self.offset+delta,self._orientation_adjustment(rotation)

    def _free_target(self, path, elapsed, duration, *, exiting=False):
        if getattr(self,'continuous',False):
            return self._continuous_free_target(path,elapsed,duration,exiting=exiting)
        u=float(np.clip(elapsed/duration,0.,1.))
        position,rotation,seed=path.at(blend(u))
        weight=blend(min(1.,2.*(1.-u if exiting else u)))
        if exiting:
            anchor=Rotation.from_rotvec(Rotation.from_matrix(self.exit_anchor_rotation).as_rotvec()*weight).as_matrix()
            position=self.nominal_pivot+anchor@(position-self.nominal_pivot)
            rotation=anchor@rotation
        else:
            # If seating starts moving the passive handle, the near-contact
            # portion follows that measured change instead of driving into
            # the old assigned angle. The far natural-down endpoint stays fixed.
            delta=(self.env.q_open_rad-self.env.q_close_rad)*(self.opening-path.angles[-1])/90.
            anchor=Rotation.from_rotvec(self.env.valve_hinge_axis_world*delta*weight).as_matrix()
            position=self.nominal_pivot+anchor@(position-self.nominal_pivot)
            rotation=anchor@rotation
        position=position+weight*self.offset
        rotation=self._orientation_adjustment(rotation,weight)
        self._sync_ik()
        q,pe,re=self.ik.solve(position,rotation,self.qseed,self.p['ik_iterations'])
        # A reviewed seed is only an alternative solution in scratch IK; it
        # never replaces physical state. Prefer the continuous branch.
        if pe+.12*re>self.p['ik_warm_seed_cost_m']:
            alternate=self.ik.solve(position,rotation,seed,self.p['ik_iterations'])
            cost=lambda r:r[1]+.12*r[2]+.0002*np.linalg.norm(r[0]-self.qseed)
            if cost(alternate)<cost((q,pe,re)):q,pe,re=alternate
        if exiting and u==0. and self.exit_start_q is not None:q=self.exit_start_q.copy()
        if u>=1. and exiting:q=self.natural[self.active_offsets].copy()
        return q,pe,re

    def _continuous_free_target(self,path,elapsed,duration,*,exiting=False):
        from .forehand_continuity import prepare_reviewed_free_path, sample_reviewed_free_path
        if (exiting and self.variant=='close' and
                getattr(self,'close_exit_reconnection',None)):
            return self._released_close_target(path,elapsed,duration)
        key='exit' if exiting else 'approach'
        # The final joint return replaces the old instantaneous all-zero
        # command. It uses the already configured lower_s, not an exam rule.
        settling=min(self.p['lower_s'],duration*.5) if exiting else 0.
        route_duration=duration-settling
        if key not in self.free_plans:
            start=(np.asarray(self.env.data.qpos)[self.ik.qadr].copy() if exiting
                   else self.natural[self.active_offsets].copy())
            delta=(float(Rotation.from_matrix(self.exit_anchor_rotation).as_rotvec()@
                         self.env.valve_hinge_axis_world) if exiting else
                   (self.env.q_open_rad-self.env.q_close_rad)*(self.opening-path.angles[-1])/90.)
            self._sync_ik()
            plan=prepare_reviewed_free_path(self.ik,path,start,nominal_pivot=self.nominal_pivot,
                valve_offset=self.offset,axis_world=self.env.valve_hinge_axis_world,
                angle_delta_rad=delta,exiting=exiting,route_duration=route_duration,
                control_dt=self.p['control_dt'],iterations=self.p['ik_iterations'],
                orientation_adjustment=self._orientation_adjustment)
            self.free_plans[key]=plan
            self.preparation.append(dict(name=f'continuous_{key}_joint_path',
                samples=len(plan.x),physical_steps_run=0,
                max_scratch_position_error_m=float(np.max(plan.errors[:,0])),
                max_scratch_rotation_error_rad=float(np.max(plan.errors[:,1])),
                no_alternate_seed=True,natural_settle_s=settling))
        plan=self.free_plans[key]
        return sample_reviewed_free_path(plan,elapsed,route_duration=route_duration,
            settle_duration=settling,natural=self.natural[self.active_offsets])

    def _released_close_target(self,path,elapsed,duration):
        from .forehand_continuity import prepare_close_release_return
        key='released_close_return'
        policy=self.close_exit_reconnection
        if not math.isclose(float(policy['total_duration_s']),duration,abs_tol=1e-9):
            raise ValueError('Reviewed close return timing differs from approved document')
        if key not in self.free_plans:
            self._sync_ik()
            start=np.asarray(self.env.data.qpos)[self.ik.qadr].copy()
            delta=float(Rotation.from_matrix(self.exit_anchor_rotation).as_rotvec()@
                self.env.valve_hinge_axis_world)
            plan=prepare_close_release_return(self.ik,path,start,policy=policy,
                natural=self.natural[self.active_offsets],nominal_pivot=self.nominal_pivot,
                valve_offset=self.offset,axis_world=self.env.valve_hinge_axis_world,
                angle_delta_rad=delta,control_dt=self.p['control_dt'],
                iterations=self.p['ik_iterations'],orientation_adjustment=self._orientation_adjustment)
            self.free_plans[key]=plan
            self.preparation.append(dict(name=key,policy=dict(policy),
                release_joints_rad=plan.release_joints.tolist(),
                release_velocity_rad_s=plan.release_velocity.tolist(),
                physical_steps_run=0,whole_motion_approval_required=True))
        plan=self.free_plans[key]
        if elapsed>=plan.release_prefix_s and not plan.actual_connected:
            plan.connect_actual_release(np.asarray(self.env.data.qpos)[self.ik.qadr],
                np.asarray(self.env.data.qvel)[self.ik.dofs],elapsed)
        return plan.at(elapsed)

    def command(self):
        e,p=self.env,self.p;t=float(e.data.time);drive=0.;pe=re=0.
        if self.withdraw_started is not None:
            elapsed=t-self.withdraw_started
            phase='reviewed_exit' if elapsed<self.exit_duration else 'final_down_hold'
            q,pe,re=self._free_target(self.exit_path,elapsed,self.exit_duration,exiting=True)
            self.integral*=math.exp(-p['control_dt']/.5)
            self.done=elapsed>=self.exit_duration+p['down_hold_s']
        elif t<p['down_hold_s']:
            phase='initial_down_hold';q=self.natural[self.active_offsets].copy()
        elif t<p['down_hold_s']+self.approach_duration:
            phase='reviewed_approach'
            q,pe,re=self._free_target(self.approach_path,t-p['down_hold_s'],self.approach_duration)
        else:
            if self._can_start_turn(t):
                lead,phase,drive=self._pacing(t)
            else:
                phase=('reviewed_blue_seating' if self.continuous and self.variant=='close'
                       else 'reviewed_orange_seating')
                lead,drive=0.,0.
                self.reference=self.opening
            pos,rot=self.target(self.opening,lead=lead*self.direction)
            q,pe,re=self._solve_contact(pos,rot,self.qseed,p['ik_iterations'],self.opening)
            drive_start=self.seated_since if self.seated_since is not None else p['down_hold_s']+self.approach_duration
            drive*=min(1.,max(0.,t-drive_start)/p['force_ramp_s'])
            if e.scorer.completion_time is not None:
                self.withdraw_started=t
                self.exit_start_q=(np.asarray(e.data.qpos)[self.ik.qadr].copy()
                    if self.continuous else q.copy())
                delta_q=(e.q_open_rad-e.q_close_rad)*(self.opening-self.goal)/90.
                self.exit_anchor_rotation=Rotation.from_rotvec(e.valve_hinge_axis_world*delta_q).as_matrix()
        self.qseed=q.copy();self._sync_ik()
        self.ik.data.qpos[self.ik.qadr]=q;mujoco.mj_forward(e.model,self.ik.data)
        kp=e.model.actuator_gainprm[e.active_actuator_ids,0]
        gravity=np.clip(self.ik.data.qfrc_bias[self.ik.dofs]/kp,-.2,.2)
        if phase not in ('reviewed_exit','final_down_hold'):
            if self.continuous and phase in ('reviewed_orange_seating','reviewed_blue_seating'):
                self.integral*=math.exp(-p['control_dt']/.5)
            else:
                self.integral=np.clip(self.integral+.2*p['control_dt']*(q-e.data.qpos[self.ik.qadr]),-.16,.16)
        self.velocity=.7*self.velocity+.3*e.data.qvel[self.ik.dofs]
        action=self.ready.copy()
        action[self.active_offsets]=q+gravity+self.integral-.07*self.velocity
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
            force_drive_fraction=drive,pause_index=0,completed_pause_angles_deg=[],pause_stable_since_s=None,
            slow_feedback_active=False,slow_feedback_braking=False,
            integral_max_abs_rad=float(np.max(np.abs(self.integral))),controller_version=self.controller_version,
            free_path_version=p['free_path_version'],execution_revision=p['execution_revision'],
            reviewed_path_source=p.get('reviewed_path_source'),
            expected_working_face='blue_hook' if self.variant=='close' else 'orange_push')
        if self.seated35:
            self.last_diagnostics.update(orange_seated_since_s=self.seated_since,
                current_orange_seat_faces=self._orange_seat_faces(),
                contact_interpolation='measured hinge frame arc, not world-space chord')
        if self.continuous:
            self.last_diagnostics.update(seating_state=dict(self.seating_state),
                extra_tangential_force_allowed=bool(drive>0. and self.seating_state.get('ready_to_turn')),
                free_joint_branch='precomputed_continuous_no_runtime_alternate',
                final_down_policy='smooth_joint_return_without_time_boundary_jump')
        return action


class HandleFrameContactPath(ReviewedPath):
    """Interpolate in the moving handle frame so sparse samples remain arcs."""
    def __init__(self, rows, active_qpos_indices, pivot, opening_rotvec_per_deg):
        self.pivot=np.asarray(pivot,float)
        self.opening_rotvec_per_deg=np.asarray(opening_rotvec_per_deg,float)
        local=[]
        for original in rows:
            row=dict(original)
            frame=Rotation.from_rotvec(self.opening_rotvec_per_deg*row['opening_deg']).as_matrix()
            row['tcp_world_m']=(frame.T@(np.asarray(row['tcp_world_m'])-self.pivot)).tolist()
            row['rotation_world']=(frame.T@np.asarray(row['rotation_world'])).tolist()
            local.append(row)
        super().__init__(local,active_qpos_indices,contact=True)

    def at(self, coordinate):
        x=float(np.clip(coordinate,self.x[0],self.x[-1]))
        position,rotation,joints=super().at(x)
        frame=Rotation.from_rotvec(self.opening_rotvec_per_deg*x).as_matrix()
        return self.pivot+frame@position,frame@rotation,joints
