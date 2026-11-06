"""A1 v2.2: terminal holding and observed S-tool driving faces.

No physics is run here. JSON evidence without solved contact geometry/wrenches
cannot establish a working face, even when an old score or mode says otherwise.
The unchanged v2.1 implementation remains available in scoring_v21.
"""
from __future__ import annotations
import math

from .scoring_v21 import OnlineScorer as V21Scorer, ScoreConfig
from .scoring_v2 import ALIASES, VARIANTS, _finite


def _terminal_rule(*,target,last_time,last_angle,start,longest,last_outside,
                   first_time,complete,coarse,hold_s=1.,tolerance=5.):
    near=last_angle is not None and abs(last_angle-target)<=tolerance+1e-9
    duration=0. if not near or start is None else max(0.,last_time-start)
    if last_angle is None:status,reason='unknown','没有末角证据'
    elif not near:status,reason='fail','真实结束角已在目标±5°外，曾经到位不能替代结束到位'
    elif duration>=hold_s-1e-9:
        status,reason='pass','记录的最后连续1秒均在目标±5°内' if complete else '末段在目标内，但完整性尚需确认'
        if not complete:status='unknown'
    elif not complete:status,reason='unknown','末段记录不完整，不能确认最后连续1秒'
    elif coarse and last_outside is not None and last_time-last_outside>=hold_s-1e-9:
        status,reason='unknown','粗采样跨越最后1秒边界，不能把未知入带时刻当作明确未稳住'
    else:status,reason='fail','结束前最后连续稳在目标范围内的时间不足1秒'
    return dict(status=status,target_opening_deg=target,angle_tolerance_deg=tolerance,
        required_hold_s=hold_s,final_time_s=last_time,final_opening_deg=last_angle,
        final_in_target=near,end_hold_s=duration,longest_hold_s=longest,
        last_outside_time_s=last_outside,reason=reason)


def final_target_status(samples,variant,*,observation_dt_s=None,record_complete=True,
                        final_time_s=None,final_opening_deg=None):
    """Assess only the terminal angle window from preserved JSON observations.

    Optional final_time_s/final_opening_deg are a real preserved end observation,
    never interpolated force evidence. Coarse boundary uncertainty stays unknown.
    """
    variant=ALIASES.get(variant,variant)
    if variant not in VARIANTS:raise ValueError('Unknown valve variant')
    rows=[(_finite(r['time_s'],'time_s'),_finite(r['opening_deg'],'opening_deg')) for r in samples]
    if (final_time_s is None)!=(final_opening_deg is None):
        raise ValueError('A preserved final observation needs both time and angle')
    if final_time_s is not None:
        point=(_finite(final_time_s,'final_time_s'),_finite(final_opening_deg,'final_opening_deg'))
        if rows and point[0]<rows[-1][0]-1e-7:raise ValueError('Final observation precedes saved samples')
        if rows and abs(point[0]-rows[-1][0])<=1e-7:rows[-1]=point
        else:rows.append(point)
    if any(b[0]<=a[0] for a,b in zip(rows,rows[1:])):raise ValueError('Times must increase')
    gaps=[b[0]-a[0] for a,b in zip(rows,rows[1:])]
    dt=_finite(observation_dt_s,'observation_dt_s') if observation_dt_s is not None else max(gaps,default=.001)
    if dt<=0:raise ValueError('Observation interval must be positive')
    target=VARIANTS[variant][1];start=None;longest=0.;last_outside=None;last_gap=None
    for i,(t,angle) in enumerate(rows):
        if i and t-rows[i-1][0]>dt+max(1e-7,dt*.01):
            start=None;last_gap=t
        if abs(angle-target)<=5.+1e-9:
            if start is None:start=t
            longest=max(longest,t-start)
        else:start=None;last_outside=t
    return _terminal_rule(target=target,last_time=rows[-1][0] if rows else None,
        last_angle=rows[-1][1] if rows else None,start=start,longest=longest,
        last_outside=last_outside,first_time=rows[0][0] if rows else None,
        complete=bool(record_complete) and (last_gap is None or (rows and rows[-1][0]-last_gap>=1.-1e-9)),
        coarse=dt>.001+1e-9)


class OnlineScorer(V21Scorer):
    def __init__(self,*args,face_constraint_applicable=False,face_evidence_complete=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.face_constraint_applicable=bool(face_constraint_applicable)
        self.face_evidence_complete=bool(face_evidence_complete)
        self._last_outside=None;self._first_time=None
        self.correct_face_drive=False;self.wrong_face_drive=False
        self.unknown_face_drive=False;self.missing_face_evidence=False
        self.face_drive_stats={};self.face_drive_events=[];self._active_drive_faces=set()

    def observe(self,time_s,opening_deg,speed_deg_s,contacts,metadata=None):
        previous_time=self.last_time
        super().observe(time_s,opening_deg,speed_deg_s,contacts,metadata)
        if self._first_time is None:self._first_time=float(time_s)
        if abs(opening_deg-VARIANTS[self.variant][1])>self.config.angle_tolerance_deg+1e-9:
            self._last_outside=float(time_s)
        if not self.face_constraint_applicable:return
        from .tool_face_evidence import assess_contact_drive
        active=set();dt=0. if previous_time is None else float(time_s)-previous_time
        for contact in contacts:
            if contact.get('kind')!='handle' or contact.get('robot_part')!='operation_end':continue
            if float(contact.get('force_N',0.))<=0:continue
            event=assess_contact_drive(contact.get('working_face_evidence'))
            if event['evidence_status']=='unknown':
                # Missing force direction cannot be repaired by a mode, a geom
                # name or a prior peak-force summary.
                self.missing_face_evidence=True
                continue
            if not event['driving']:continue
            face=event['face'];active.add(face)
            entry=self.face_drive_stats.setdefault(face,dict(driving_samples=0,positive_work_J=0.,
                maximum_absolute_torque_Nm=0.,target_direction_drive_samples=0))
            entry['driving_samples']+=1
            entry['positive_work_J']+=max(0.,event['power_W'])*dt
            entry['maximum_absolute_torque_Nm']=max(entry['maximum_absolute_torque_Nm'],abs(event['requested_torque_Nm']))
            if event['requested_torque_Nm']>0:entry['target_direction_drive_samples']+=1
            if face in ('orange_push','blue_pull') and event['requested_torque_Nm']>0:
                self.correct_face_drive=True
            elif face=='nonworking' and event['requested_torque_Nm']>0:
                self.wrong_face_drive=True
            elif face=='unknown' and event['driving_toward_target']:
                self.unknown_face_drive=True
            if face not in self._active_drive_faces and len(self.face_drive_events)<24:
                self.face_drive_events.append(dict(time_s=float(time_s),**event))
        self._active_drive_faces=active

    def result(self,*,final_target_override=None):
        result=super().result();c=self.config
        final=final_target_override or _terminal_rule(target=VARIANTS[self.variant][1],
            last_time=self.last_time,last_angle=self.final_opening,start=self.target_start,
            longest=self.longest_target_hold,last_outside=self._last_outside,first_time=self._first_time,
            complete=self.sample_count>=2 and self.coverage_ok,
            coarse=self.observation_dt_s>self.physics_dt_s+1e-9,
            hold_s=c.final_hold_s,tolerance=c.angle_tolerance_deg)
        if not self.face_constraint_applicable:
            face_status,face_reason='not_applicable','只对明确使用S工具的任务加此规则'
        elif self.wrong_face_drive:
            face_status,face_reason='fail','真实接触点/法线识别为非工作面，且朝任务目标方向做正功带动阀门'
        elif not self.face_evidence_complete or self.missing_face_evidence or self.unknown_face_drive:
            face_status,face_reason='unknown','缺少真实逐步面/受力方向，或驱动接触在几何识别边界，需复核'
        elif self.correct_face_drive:
            face_status,face_reason='pass','实际橙面推或蓝面勾拉对正确转向做正功，未观察到非工作面朝任务目标方向驱动'
        else:
            face_status,face_reason='fail','完整实测证据中未见橙面/蓝面实际带动正确转向'
        face=dict(status=face_status,applicable_to='S_tool_only',reason=face_reason,
            correct_working_face_drive_seen=self.correct_face_drive,
            nonworking_face_drive_seen=self.wrong_face_drive,
            ambiguous_face_drive_seen=self.unknown_face_drive,
            evidence_complete=self.face_evidence_complete and not self.missing_face_evidence,
            qualification_direction='toward_task_target',
            driving_basis='positive solved contact torque times actual valve angular velocity; qualification concerns task-target-direction drive; reverse work is recorded and terminal holding judges rebound; no face-share/force/angle/speed-margin threshold',
            face_stats=self.face_drive_stats,events=self.face_drive_events)
        result.update(schema_version='2.2',procedure_version='A1-v2.2',
            assessment_scope='A1正式规程v2.2' if self.station=='A1' else '暂用A1 v2.2通用条款，非正式新工位验收')
        result['v21_base_outcome']=result['base_outcome']
        result['rules']['ever_target_hold_diagnostic']=result['rules']['target_hold']
        result['rules']['target_hold']=dict(final)
        result['rules']['final_target_hold']=dict(final)
        result['rules']['working_face']=face
        result['rules']['R1']=dict(final,alias_for='final_target_hold')
        rules={key:result['rules'][key]['status'] for key in ('hard_collision','slow_open_wait','final_target_hold','working_face')}
        failed=[key for key,status in rules.items() if status=='fail']
        unknown=[key for key,status in rules.items() if status=='unknown']
        if not self.coverage_ok:unknown.append('continuous_observation')
        passed=False if failed else None if unknown else True
        if final['status']!='pass' and result['grade'] in ('完成','到位但缓开不合格'):
            result['grade']='基本完成' if not final['final_in_target'] and self.maximum_progress>45. else None
            result['grade_unresolved_reason']=None if result['grade'] else final['reason']
        result['result_level']=result['base_outcome']=result['grade']
        label=result['grade'] or '分级待定'
        rebound_at_end=bool(self.physical_target_completion_time is not None and not final['final_in_target'])
        if rebound_at_end:label+='（到位后退回）'
        elif final['status']=='fail':label+='（结束未到位或未稳满1秒）'
        if face_status=='fail':label+='（工作面不符）' if self.wrong_face_drive else '（未见工作面实际带动）'
        elif face_status=='unknown':label+='（操作面证据不足）'
        if result['hard_collision']:label+='（有硬撞）'
        result.update(passed=passed,qualified_completion=passed is True,
            verdict='通过' if passed is True else '未通过' if passed is False else '无法判断',
            failed_rules=failed,unverified_rules=unknown,result_label=label)
        result['metrics'].update(final_target_hold_s=final['end_hold_s'],
            final_target_hold_status=final['status'],working_face_status=face_status,
            arrived_then_rebounded_at_end=rebound_at_end,working_face_mismatch=self.wrong_face_drive,
            correct_working_face_drive_seen=self.correct_face_drive,
            nonworking_face_drive_seen=self.wrong_face_drive)
        result['measurement_coverage']['working_face_evidence_complete']=face['evidence_complete']
        if face_status=='unknown':
            result['measurement_coverage']['verified']=False
            result['measurement_coverage']['reason']='操作面或真实驱动方向证据不足；未把未知改成物理失败'
        result['scoring_notes'] += ['v2.2要求结束仍到位且最后连续1秒在±5°内；范围内抖动不另加速度限制。',
            'S工具按真实接触局部面和已解算力矩/实际角速度识别驱动；无驱动的轻擦、制动或退出接触不因此否决。',
            '未按工作面占比、胸前方向、手腕角度、力余量或普通开关速度另设门槛；几何模糊为未知。']
        return result


def score_record(record,config=None):
    sampling=record.get('sampling',{});physics_dt=sampling.get('physics_dt_s',.001)
    adapter=record.get('robot_adapter',{})
    applicable=record.get('face_constraint_applicable',adapter.get('tool_adapter')=='s_hook')
    evidence=record.get('evidence',{})
    scorer=OnlineScorer(record.get('variant'),physics_dt_s=physics_dt,
        observation_dt_s=sampling.get('observation_dt_s',physics_dt),
        sampling_coverage=sampling.get('coverage','unspecified'),config=config,
        actual_station=record.get('actual_station','A1'),source=record.get('source'),
        contact_evidence=record.get('contact_evidence_complete',False) or all(evidence.get(k) is True for k in ('R5','R7')),
        expected_complete=sampling.get('complete',sampling.get('coverage')=='every_physics_step'),
        face_constraint_applicable=applicable,face_evidence_complete=record.get('face_evidence_complete',False))
    for row in record.get('samples',[]):
        scorer.observe(row['time_s'],row['opening_deg'],row['speed_deg_s'],row.get('contacts',[]),row)
    scorer.add_historical_summary(record.get('old_score_v2') or record.get('preserved_score'))
    final=record.get('final_observation') or {}
    terminal=final_target_status(record.get('samples',[]),record.get('variant'),
        observation_dt_s=sampling.get('observation_dt_s',physics_dt),
        record_complete=sampling.get('complete',sampling.get('coverage')=='every_physics_step'),
        final_time_s=final.get('time_s'),final_opening_deg=final.get('opening_deg'))
    result=scorer.result(final_target_override=terminal)
    result.update(data_kind=record.get('data_kind','unspecified'),historical_comparison_only=True,
        original_procedure_version=record.get('procedure_version','unspecified'),
        runtime_settings=record.get('runtime',{}))
    return result
