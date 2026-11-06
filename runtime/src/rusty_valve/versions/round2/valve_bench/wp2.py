"""WP2: finite-actuator OpenArm contact trials, not kinematic playback.

IK uses a separate scratch MjData to calculate position references. Episode
states advance only through ValveEnv.step -> mj_step. No valve actuator,
external driving force, weld, gravity compensation force or robot gain change.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import asset_config
from .robot import ARM_JOINTS, EE_SITE


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


class ArmIK:
    def __init__(self, model, joint_names=ARM_JOINTS, site_name=EE_SITE):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(site_name).id
        self.qadr = np.array([model.joint(n).qposadr[0] for n in joint_names], int)
        self.dofs = np.array([model.joint(n).dofadr[0] for n in joint_names], int)
        self.bounds = np.array([model.joint(n).range for n in joint_names])
        self.jp = np.zeros((3, model.nv)); self.jr = self.jp.copy()

    def solve(self, pos, rotation, seed, iterations=65):
        q = np.asarray(seed, float).copy()
        for _ in range(iterations):
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            actual = self.data.site_xmat[self.site].reshape(3, 3)
            pe = np.asarray(pos)-self.data.site_xpos[self.site]
            re = Rotation.from_matrix(rotation @ actual.T).as_rotvec()
            err = np.r_[pe, .16*re]
            if np.linalg.norm(pe) < .0005 and np.linalg.norm(re) < .007:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.site)
            jac = np.r_[self.jp[:, self.dofs], .16*self.jr[:, self.dofs]]
            dq = jac.T @ np.linalg.solve(jac@jac.T + .000015*np.eye(6), err)
            dq = np.clip(dq, -.12, .12)
            q = np.clip(q+dq, self.bounds[:, 0]+.0001, self.bounds[:, 1]-.0001)
        return q, float(np.linalg.norm(pe)), float(np.linalg.norm(re))


def contact_pose(env, angle, style, radius=.16, depth=.024, tilt_deg=-30):
    """Target hand TCP; depth is handle-centre position along hand +Z.

    palm: hand forward axis tangent to valve, fingers open vertically.
    tilted: same with rotation about the handle length.
    finger: top entry, one inner finger flank bears the handle's side.
    """
    cfg = asset_config()['ball']
    m, d = env.model, env.data
    mount = m.body('valve_mount').id
    Rv = d.xmat[mount].reshape(3,3)
    origin = d.xpos[mount]
    radial = np.array([math.cos(angle), math.sin(angle), 0.])
    tangent = np.array([-math.sin(angle), math.cos(angle), 0.])
    stem = np.array([0., 0., 1.])
    if style == 'finger':
        R = np.column_stack([radial, -tangent, -stem])
        # Near-root finger flank, forward depth 42mm.
        local_center = radius*radial + (cfg['lever_highest_m']-.003)*stem
        hand_origin = local_center - .042*R[:,2]
    else:
        R = np.column_stack([-radial, stem, tangent])
        if style == 'tilted':
            R = Rotation.from_rotvec(radial*math.radians(tilt_deg)).as_matrix() @ R
        local_center = radius*radial + (cfg['lever_highest_m']-.003)*stem
        hand_origin = local_center - depth*R[:,2]
    return origin + Rv @ (hand_origin + .09*R[:,2]), Rv @ R


def run_trial(output: Path, *, style='tilted', scale=1., radius=.16, depth=.024,
              base_position=(0, 0, .55), valve_position=(-.01, .125, .70),
              max_time_s=40., video=True, feedback=True):
    from .env import ValveEnv
    output.mkdir(parents=True, exist_ok=False)
    env = ValveEnv(valve='ball', layout='horizontal_top', task='close',
                   resistance_scale=scale, base_position=base_position,
                   valve_position=valve_position, include_image=False,
                   max_time_s=max_time_s)
    obs, info = env.reset(seed=0)
    ik = ArmIK(env.model)
    q = env.data.qpos[ik.qadr].copy()
    # Plan a clear pre-approach pose. This is a command, not an episode reset.
    p0, R0 = contact_pose(env, 0, style, radius, depth)
    approach_axis = R0[:,2]
    pre_pos = p0 - .09*approach_axis + np.array([0.,0.,.10])
    qpre, perr, rerr = ik.solve(pre_pos, R0, q, iterations=300)
    if perr > .025 or rerr > .3:
        # A second standard seed avoids a straight-arm IK singularity.
        other = np.array([.3,-.4,.4,1.5,0.,0.,0.])
        qalt, pe2, re2 = ik.solve(pre_pos, R0, other, iterations=300)
        if pe2+.16*re2 < perr+.16*rerr: qpre,perr,rerr=qalt,pe2,re2
    initial = q.copy()
    rows=[]; contact_rows=[]; video_times=[]; peak=0.; contacted=False
    writer = imageio.get_writer(str(output/'操作_原速.mp4'), fps=25, codec='libx264',
                                quality=7, macro_block_size=1) if video else None
    contact_writer = imageio.get_writer(str(output/'接触近景_原速.mp4'),fps=25,codec='libx264',quality=7,macro_block_size=1) if video else None
    start = time.monotonic(); max_angle=-1.; final_info={}
    (output/'scene.xml').write_text(env.xml,encoding='utf-8')
    config=dict(style=style,feedback=feedback,resistance_scale=scale,radius_m=radius,depth_m=depth,
                base_position_m=base_position,valve_position_m=valve_position,
                controller='Jacobian damped least-squares IK, native stock joint position actuators; Cartesian planned approach then slow quarter turn; no valve drive or weld',
                target_angle_deg=90,control_dt_s=.02,physics_dt_s=.001,video_fps=25,
                seed=0,initial_qpos=env.data.qpos.tolist(),
                max_time_s=max_time_s,camera_names=['robot_overview','robot_contact'],
                initial_ik_position_error_m=perr,initial_ik_orientation_error_rad=rerr)
    config['quality_rules']={'angle_error_deg':2.,'final_speed_deg_s':5.,'hold_s':.5,'no_fixture_contact_above_N':1.,'peak_speed_deg_s':30.}
    config['controller_parameters']={'tilt_deg':-30,'trajectory_speed_rad_s':.09,'reference_lead_rad':.15,'integral_gain_s_inv':.20,'integral_offset_limit_rad':.16,'velocity_filter_old_weight':.7,'velocity_brake_s':.07,'reach_s':3.,'approach_s':3.,'enclose_s':1.,'approach_lift_m':.10}
    snap=output/'source_snapshot';snap.mkdir()
    for name in ['wp2.py','robot.py','env.py','geometry.py','resistance.py']:
        shutil.copy2(Path(__file__).parent/name,snap/name)
    shutil.copytree(Path(__file__).parent/'configs',snap/'configs')
    dump(output/'config.json',config)
    terminated=truncated=False; last_phase=''; next_video=1/25
    if writer is not None:
        writer.append_data(env.render(camera='robot_overview',width=800,height=600))
        contact_writer.append_data(env.render(camera='robot_contact',width=800,height=600))
        video_times.append(0.)
    seed=qpre.copy(); integral=np.zeros(7); filtered_velocity=np.zeros(7)
    try:
        while not (terminated or truncated):
            t=float(env.data.time)
            if t < 3.0:
                phase='reach'; a=t/3.0; a=a*a*(3-2*a)
                command=initial+(qpre-initial)*a
                finger=.044
            elif t < 6.0:
                phase='approach'
                high=p0+np.array([0.,0.,.10])
                if t < 4.5:
                    a=(t-3.)/1.5; a=a*a*(3-2*a)
                    pos=pre_pos+a*(high-pre_pos)
                else:
                    a=(t-4.5)/1.5; a=a*a*(3-2*a)
                    pos=high+a*(p0-high)
                command,perr,rerr=ik.solve(pos,R0,seed,iterations=30); seed=command
                finger=.044
            elif t < 7.0:
                phase='enclose'; command,perr,rerr=ik.solve(p0,R0,seed,iterations=30); seed=command
                finger=.044+(.004-.044)*(t-6.0)
                if style=='finger': finger=.044+(.008-.044)*(t-6.0)
            else:
                phase='turn_hold'
                ref_angle=min(math.pi/2+.08, max(0,t-7.0)*.09)
                if feedback:
                    ref_angle=min(ref_angle,max(0.,env.valve_angle_rad)+.15)
                pos,rot=contact_pose(env,ref_angle,style,radius,depth)
                command,perr,rerr=ik.solve(pos,rot,seed,iterations=25); seed=command
                finger=.004 if style!='finger' else .008
            # Native stiffness implies gravity sag. Supply reference offset only;
            # this changes the command, never robot force limits or dynamics.
            ik.data.qpos[ik.qadr]=command
            mujoco.mj_forward(env.model,ik.data)
            kp=env.model.actuator_gainprm[:7,0]
            gravity_offset=np.clip(ik.data.qfrc_bias[ik.dofs]/kp,-.20,.20)
            if feedback:
                # Outer position-error integral changes finite actuator references
                # only. Each motor still obeys its upstream kp/kv and force cap.
                integral += .20*env.control_dt*(command-env.data.qpos[ik.qadr])
                integral = np.clip(integral,-.16,.16)
            filtered_velocity = .7*filtered_velocity + .3*env.data.qvel[ik.dofs]
            velocity_brake = .07*filtered_velocity if feedback else np.zeros(7)
            action=np.r_[command+gravity_offset+integral-velocity_brake,finger]
            obs, reward, terminated, truncated, info=env.step(action)
            final_info=info
            angle=math.degrees(float(env.data.qpos[env.model.joint('valve_hinge').qposadr[0]]))
            max_angle=max(max_angle,angle)
            # Contact field names come from the reusable environment.
            count=int(info.get('handle_contact_count',0)); contacted |= count>0
            force=sum(c['normal_force_N'] for c in info['handle_contacts']); peak=max(peak,force)
            row={'time_s':float(env.data.time),'action_time_s':t,'phase':phase,'valve_angle_deg':angle,
                 'target_angle_deg':90.,'ik_position_error_m':perr,
                 'ik_orientation_error_rad':rerr,'contact_count':count,
                 'contact_normal_N':force,
                 'valve_friction_Nm':info['valve_friction_torque_Nm'],
                 'handle_contact_torque_Nm':info['handle_normal_torque_Nm'],
                 'maximum_speed_deg_s':info['maximum_valve_speed_deg_s'],
                 'fixture_peak_N':info['fixture_peak_normal_N'],
                 'valve_speed_deg_s':math.degrees(env.valve_speed_rad_s)}
            for i,x in enumerate(env.data.qpos):row[f'qpos_{i}']=float(x)
            for i,x in enumerate(env.data.qvel):row[f'qvel_{i}']=float(x)
            for i,x in enumerate(info['applied_action']):row[f'command_{i}']=float(x)
            for i,x in enumerate(env.data.actuator_force):row[f'actuator_force_{i}']=float(x)
            rows.append(row)
            if count:
                contact_rows.append({'time_s':float(env.data.time),'contact_sample_time_s':info['contact_sample_time_s'],'contacts':info['handle_contacts']})
            if writer is not None and env.data.time+1e-8 >= next_video:
                # Multiple frames only if controller period passes a frame boundary.
                frame=env.render(camera='robot_overview',width=800,height=600)
                writer.append_data(frame)
                contact_writer.append_data(env.render(camera='robot_contact',width=800,height=600))
                video_times.append(float(env.data.time));next_video+=1/25
            if phase!=last_phase:
                print(output.name, phase, 'angle',round(angle,2),'IK',round(perr,4),flush=True)
                last_phase=phase
            if not np.isfinite(env.data.qpos).all():break
        imageio.imwrite(output/'接触近景.png',env.render(camera='robot_contact',width=1000,height=700))
    finally:
        if writer is not None:writer.close()
        if contact_writer is not None:contact_writer.close()
        env.close()
    with (output/'timeline.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(success=bool(final_info['is_success'] and not final_info['has_fixture_contact'] and final_info['maximum_valve_speed_deg_s'] <= 30.),
                angle_goal_reached=bool(final_info['is_success']),final_angle_deg=angle,
                maximum_angle_deg=max_angle,contact_seen=contacted,peak_sampled_contact_normal_N=peak,
                simulation_duration_s=float(env.data.time),wall_time_s=time.monotonic()-start,
                numerical_warnings=int(sum(int(w.number) for w in env.data.warning)),
                final_info={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in final_info.items()})
    with (output/'contacts.jsonl').open('w') as f:
        for entry in contact_rows:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
    dump(output/'video_timestamps.json',video_times)
    dump(output/'result.json',result)
    print(output.name,result,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--style',choices=('palm','tilted','finger'),default='tilted')
    p.add_argument('--scale',type=float,default=1.)
    p.add_argument('--radius',type=float,default=.16)
    p.add_argument('--depth',type=float,default=.024)
    p.add_argument('--max-time',type=float,default=40.)
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--feedback',action='store_true',default=True)
    p.add_argument('--no-feedback',action='store_false',dest='feedback')
    a=p.parse_args()
    run_trial(a.output,style=a.style,scale=a.scale,radius=a.radius,depth=a.depth,
              max_time_s=a.max_time,video=not a.no_video,feedback=a.feedback)

if __name__=='__main__':main()
