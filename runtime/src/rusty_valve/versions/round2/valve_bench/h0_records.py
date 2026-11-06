"""Robot-neutral physical-contact classification and A1-v1.2 recording.

The public scoring record contains only task-side observations. Optional
MuJoCo full states are a separate replay artifact, never a scoring input.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .scoring_v12 import score_record
from .script_state_recording import ScriptStateRecorder


ROBOT_PARTS = ("operation_end", "idle_hand", "robot_other")
VARIANTS = {"close":"A1-关", "open":"A1-开", "slow_open":"A1-缓开"}
PROVISIONAL_SCOPE = "暂用A1 v1.2的通用条款评估，非正式新规程验收"


def _write_json(path, obj, *, compact=False):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, allow_nan=False,
                  indent=None if compact else 2,
                  separators=(",", ":") if compact else None)
        stream.write("\n")


def _geom_ids(model, values):
    return {int(model.geom(x).id) if isinstance(x,str) else int(x) for x in values}


def classify_contacts(model, data, robot_groups, environment_groups):
    """Classify one current solve using adapter declarations, never robot names.

    Robot groups are disjoint ``operation_end/idle_hand/robot_other`` geom-ID
    lists; names are also accepted. Environment groups use ``handle_geom_ids``
    and ``fixture_geom_ids``. Other environment contact remains visible (e.g.
    ground): it can violate idle-hand R6 without being mislabelled pipe R5.
    """
    roles={}
    for role in ROBOT_PARTS:
        for gid in _geom_ids(model,robot_groups.get(role,())):
            if gid in roles:
                raise ValueError("Robot contact groups must be disjoint")
            roles[gid]=role
    handle=_geom_ids(model,environment_groups.get("handle_geom_ids",()))
    fixtures=_geom_ids(model,environment_groups.get("fixture_geom_ids",()))
    if handle & fixtures:
        raise ValueError("Handle and fixture contact groups must be disjoint")
    if set(roles) & (handle|fixtures):
        raise ValueError("Robot and environment contact groups must be disjoint")
    result=[]
    for index in range(int(data.ncon)):
        contact=data.contact[index]
        a,b=int(contact.geom1),int(contact.geom2)
        if a not in roles and b not in roles:
            continue
        force=np.zeros(6)
        mujoco.mj_contactForce(model,data,index,force)
        magnitude=float(np.linalg.norm(force[:3]))
        if not math.isfinite(magnitude):
            raise ValueError("Nonfinite solved contact force")
        if a not in roles:
            a,b=b,a
        kind=("self" if b in roles else "handle" if b in handle
              else "fixture" if b in fixtures else "environment")
        row=dict(kind=kind,robot_part=roles[a],force_N=magnitude,
                 robot_geom=model.geom(a).name or str(a),
                 other_geom=model.geom(b).name or str(b),
                 surface=model.geom(a).name or str(a))
        # Add solved vector evidence without changing any legacy role/force
        # classification. Positive contact wrench acts on geom2 in MuJoCo.
        frame=np.asarray(contact.frame).reshape(3,3)
        sign=1. if b==int(contact.geom2) else -1.
        outward=frame[0]*(1. if a==int(contact.geom1) else -1.)
        row.update(contact_index=index,robot_geom_id=a,other_geom_id=b,
            contact_position_world_m=np.asarray(contact.pos).tolist(),
            robot_outward_contact_normal_world=outward.tolist(),
            force_on_other_world_N=(sign*(frame.T@force[:3])).tolist(),
            contact_moment_on_other_world_Nm=(sign*(frame.T@force[3:])).tolist(),
            contact_gap_m=float(contact.dist),
            force_evidence_basis='original_solved_contact_wrench')
        if kind=="self":
            row["other_robot_part"]=roles[b]
        result.append(row)
    return result


def add_s_tool_face_evidence(env,contacts,*,allow_working_hole_rims=False):
    """Enrich the same solved contacts, never step/forward or infer from mode.

    Caller must establish tool_adapter=s_hook. Tool-local coordinates use its
    actual body frame; torque uses the real hinge axis and contact wrench.
    """
    from .tool_face_evidence import classify_s_face
    data=env.data;rotation=np.asarray(data.xmat[env.active_hook_body_id]).reshape(3,3)
    origin=np.asarray(data.xpos[env.active_hook_body_id])
    delta=(env.q_close_rad-env.q_open_rad) if env.task=='close' else (env.q_open_rad-env.q_close_rad)
    sign=1. if delta>0 else -1.
    axis=sign*np.asarray(data.xaxis[env.valve_joint_id])
    anchor=np.asarray(data.xanchor[env.valve_joint_id])
    omega=sign*float(data.qvel[env.valve_dof_index])
    enriched=[]
    for original in contacts:
        row=dict(original)
        if row.get('kind')=='handle' and row.get('robot_part')=='operation_end':
            required=('contact_position_world_m','robot_outward_contact_normal_world',
                      'force_on_other_world_N','contact_moment_on_other_world_Nm','contact_gap_m')
            if all(key in row for key in required):
                normal=np.asarray(row['robot_outward_contact_normal_world'])
                surface=np.asarray(row['contact_position_world_m'])-.5*row['contact_gap_m']*normal
                point_local=rotation.T@(surface-origin);normal_local=rotation.T@normal
                label=classify_s_face(point_local.tolist(),normal_local.tolist(),
                    allow_working_hole_rims=allow_working_hole_rims)
                row['working_face']=label['face']
                row['working_face_geometry']=label
                row['working_face_evidence']=dict(schema_version=1,
                    basis='solved_contact_point_normal_wrench_v1',
                    tool_surface_position_local_m=point_local.tolist(),
                    tool_outward_normal_local=normal_local.tolist(),
                    contact_position_world_m=row['contact_position_world_m'],
                    valve_axis_point_world_m=anchor.tolist(),requested_axis_world=axis.tolist(),
                    force_on_handle_world_N=row['force_on_other_world_N'],
                    moment_on_handle_world_Nm=row['contact_moment_on_other_world_Nm'],
                    angular_speed_about_requested_axis_rad_s=omega)
        enriched.append(row)
    return enriched


def _plain(value):
    if hasattr(value,"to_dict"):
        return value.to_dict()
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,dict):
        return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_plain(v) for v in value]
    if isinstance(value,np.generic):
        return value.item()
    return value


class H0Recorder:
    """Observe reset plus every physical step from a public task-side provider.

    Provider fields: ``station_spec``, ``physics_dt``, ``valve_angle_rad``,
    ``valve_speed_rad_s``, ``current_contacts`` and ``data.time``; observer
    attachment uses ``physics_observer``. ``runtime_integrity`` explicitly
    declares R8 guarantees. Joint fields are never needed for ``as_record``.
    Full-state replay recording is optional and uses a separate archive.
    """

    def __init__(self,env,variant="close",adapter=None,*,sample_hz=50.,
                 cameras=None,capture_full_state=True,source=None):
        if variant not in VARIANTS:
            raise ValueError("Choose close, open or slow_open")
        if not getattr(env,"_started",False):
            raise ValueError("Reset before attaching H0Recorder")
        if env.physics_observer is not None:
            raise ValueError("Another physical observer is already attached")
        if not math.isfinite(sample_hz) or sample_hz<=0:
            raise ValueError("sample_hz must be finite and positive")
        self.env=env;self.variant=variant;self.spec=env.station_spec
        self.station=self.spec.case
        self.adapter=_plain(adapter if adapter is not None else getattr(env,"robot_adapter",{}))
        if not isinstance(self.adapter,dict):
            raise ValueError("Adapter declaration must be a dictionary or expose to_dict()")
        groups=getattr(env,"robot_groups",None)
        has_idle=self.adapter.get("has_idle_hand")
        if has_idle is None and groups is not None:
            has_idle=groups.get("has_idle_hand",bool(groups.get("idle_hand",())))
        if not isinstance(has_idle,bool):
            raise ValueError("Adapter must explicitly declare has_idle_hand")
        if groups is not None and has_idle != bool(groups.get("idle_hand",())):
            raise ValueError("Adapter has_idle_hand differs from its resolved idle-hand geometry group")
        self.adapter["has_idle_hand"]=has_idle
        self.source=str(source) if source is not None else None
        self.samples=[];self.visual_labels=[]
        self.cameras=_plain(cameras if cameras is not None else
                            getattr(env,"camera_parameters",self.adapter.get("cameras",[])))
        self.physics_dt_s=float(env.physics_dt)
        self._visual_period=1./float(sample_hz);self._next_visual=0.
        self._states=None;self._attached=True
        model=getattr(env,"model",None)
        self._label_data=mujoco.MjData(model) if model is not None else None
        env.physics_observer=self.observe
        try:
            self.observe(env)
            if capture_full_state:
                self._states=ScriptStateRecorder(env,sample_hz=sample_hz)
        except Exception:
            env.physics_observer=None;self._attached=False
            raise

    @property
    def assessment_scope(self):
        return "A1正式规程v1.2，机器人无关公共评分入口" if self.station=="A1" else PROVISIONAL_SCOPE

    def _opening(self):
        scale=90./(self.spec.q_open_rad-self.spec.q_close_rad)
        return ((self.env.valve_angle_rad-self.spec.q_close_rad)*scale,
                self.env.valve_speed_rad_s*scale)

    def _visual(self,time,opening):
        provider=getattr(self.env,"visual_observation",None)
        if callable(provider):
            label=_plain(provider())
        else:
            label=dict(opening_deg=float(opening),handle_pose_world=None,
                       valve_state="closed" if abs(opening)<=2 else "open" if abs(opening-90)<=2 else "intermediate")
            model,data=getattr(self.env,"model",None),self._label_data
            if model is not None:
                # mj_step integrates qpos after its kinematic stage; runtime
                # site_xpos may still describe the previous millisecond. Draw
                # current-qpos labels in scratch data without mutating the run.
                data.qpos[:]=self.env.data.qpos
                data.time=float(time)
                if model.nmocap:
                    data.mocap_pos[:]=self.env.data.mocap_pos
                    data.mocap_quat[:]=self.env.data.mocap_quat
                mujoco.mj_kinematics(model,data)
                sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f"{self.station}_handle_center")
                if sid>=0:
                    quaternion=np.zeros(4)
                    mujoco.mju_mat2Quat(quaternion,data.site_xmat[sid])
                    label["handle_pose_world"]=dict(position_m=data.site_xpos[sid].tolist(),
                                                    quaternion_wxyz=quaternion.tolist())
                    label["pose_source"]="scratch mj_kinematics from the saved current qpos; no force solve"
        label["time_s"]=float(time)
        label.setdefault("opening_deg",float(opening))
        label.setdefault("hinge_q_rad",float(self.env.valve_angle_rad))
        label.setdefault("handle_pose_world",None)
        label.setdefault("valve_state","closed" if abs(opening)<=2 else "open" if abs(opening-90)<=2 else "intermediate")
        return label

    def observe(self,env):
        if env is not self.env:
            raise ValueError("Received another provider")
        time=float(env.data.time)
        if time==0.:
            self.samples.clear();self.visual_labels.clear();self._next_visual=0.
        elif self.samples and time<=self.samples[-1]["time_s"]:
            raise ValueError("Physical evidence times must strictly increase")
        opening,speed=self._opening()
        contacts=_plain(copy.deepcopy(env.current_contacts))
        sample=dict(time_s=time,opening_deg=float(opening),speed_deg_s=float(speed),contacts=contacts)
        sample["hinge_q_rad"]=float(env.valve_angle_rad)
        # Diagnostic solver result is optional and never changes R1's published
        # angle-band test. Do not infer a physical stop from an R0 pass.
        reaction=getattr(env,"current_joint_limit_reaction_Nm",getattr(env,"_stop_reaction",None))
        if reaction is not None:
            sample["hinge_joint_limit_reaction_Nm"]=float(reaction)
        self.samples.append(sample)
        if time>=self._next_visual-1e-9:
            self.visual_labels.append(self._visual(time,opening))
            while self._next_visual<=time+1e-9:
                self._next_visual+=self._visual_period

    def as_record(self):
        runtime=getattr(self.env,"runtime_declaration",None)
        runtime=runtime() if callable(runtime) else getattr(self.env,"runtime_integrity",{})
        return dict(schema_version=1,procedure_version="A1-v1.2",variant=VARIANTS[self.variant],
            actual_station=self.station,actual_variant=f"{self.station}-{VARIANTS[self.variant].split('-')[1]}",
            assessment_scope=self.assessment_scope,
            source=self.source,data_kind="recorded_simulation",robot_adapter=copy.deepcopy(self.adapter),
            sampling=dict(coverage="every_physics_step",physics_dt_s=self.physics_dt_s),
            evidence={r:True for r in ("R4","R5","R6","R7")},
            runtime=_plain(runtime),samples=self.samples,
            visual_labels=self.visual_labels,cameras=self.cameras,
            limitations=["本机MuJoCo受力仿真，非现场实测。"]+([] if self.station=="A1" else [PROVISIONAL_SCOPE]))

    def write_and_score(self,output,extra_metadata=None):
        output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
        paths={k:output/v for k,v in (('record','procedure_record.json'),('score','score.json'),
                                     ('metadata','metadata.json'),('scene','scene.xml'),('trajectory','trajectory.npz'))}
        if any(path.exists() for path in paths.values()):
            raise FileExistsError("Refuse to replace existing H0 evidence")
        record=self.as_record()
        try:
            score=score_record(record)
        except ValueError as exc:
            score=dict(schema_version=1,procedure_version="A1-v1.2",variant=VARIANTS[self.variant],
                       passed=False,verdict="未通过（原评分程序拒绝无效记录）",
                       failed_rules=[],unverified_rules=["record"],invalid_record_reason=str(exc),rules={})
        score.update(actual_station=self.station,actual_variant=record["actual_variant"],
                     assessment_scope=self.assessment_scope,
                     formal_procedure_acceptance=self.station=="A1" and score["passed"])
        if self.station!="A1":
            score["verdict"]="通用条款通过（非正式规程验收）" if score["passed"] else "通用条款未通过（非正式规程验收）"
        _write_json(paths['record'],record,compact=True);_write_json(paths['score'],score)
        states=None
        if self._states is not None:
            states=self._states.write(output)
            with np.load(states['states'],allow_pickle=False) as archive:
                with paths['trajectory'].open('xb') as stream:
                    np.savez_compressed(stream,**{n:archive[n] for n in ('time_s','qpos','qvel','ctrl')})
            mujoco.mj_saveLastXML(str(paths['scene']),self.env.model)
        metadata=dict(schema_version=1,data_kind="recorded_simulation",procedure_version="A1-v1.2",
            task="close" if self.variant=="close" else "open",h1_variant=self.variant,
            actual_station=self.station,actual_variant=record['actual_variant'],
            assessment_scope=self.assessment_scope,robot_adapter=copy.deepcopy(self.adapter),
            station_description=_plain(self.spec),valve_offset=[0.,0.,0.],
            model_transform_note="scene.xml has actual model transforms; do not offset twice.",
            cameras=self.cameras,visual_labels_file=paths['record'].name,
            camera_use="Configurable observations; H0 reference controller reads simulation state.",
            physics_dt_s=self.physics_dt_s,physics_evidence_samples=len(self.samples),
            final_simulation_time_s=float(self.env.data.time))
        metadata.update(_plain(extra_metadata or {}))
        metadata.update(actual_station=self.station,actual_variant=record['actual_variant'],
                        assessment_scope=self.assessment_scope,procedure_version="A1-v1.2",
                        robot_adapter=copy.deepcopy(self.adapter),valve_offset=[0.,0.,0.])
        _write_json(paths['metadata'],metadata)
        return dict(**{k:str(v) for k,v in paths.items() if v.exists()},full_states=states,
                    passed=score['passed'],failed_rules=score['failed_rules'],
                    unverified_rules=score['unverified_rules'],actual_station=self.station,
                    actual_variant=record['actual_variant'],assessment_scope=self.assessment_scope,
                    formal_procedure_acceptance=score['formal_procedure_acceptance'])

    def close(self):
        if self._states is not None:self._states.close()
        if self._attached and self.env.physics_observer==self.observe:
            self.env.physics_observer=None;self._attached=False
