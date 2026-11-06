"""Current scoring: sequential 18/45 degree waits, with v2.2 safety rules.

Only preserved observations are consumed. No simulation or interpolation is
performed. Coarse samples that straddle a five-second boundary stay unknown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .scoring_v2 import ALIASES, VARIANTS, _finite
from .scoring_v22 import OnlineScorer as V22Scorer, final_target_status
from .slow_open_rule import load_slow_open_rule


def _rule_value(name):
    return load_slow_open_rule()[name]


@dataclass(frozen=True)
class ScoreConfig:
    # Unchanged v2.2 result and collision thresholds.
    angle_tolerance_deg: float = 5.
    final_hold_s: float = 1.
    hard_contact_limit_N: float = 20.
    episode_cutoff_s: float = 90.
    pause_angles_deg: tuple = field(default_factory=lambda: tuple(_rule_value('pause_angles_deg')))
    pause_s: float = field(default_factory=lambda: _rule_value('pause_s'))
    pause_angle_tolerance_deg: float = field(default_factory=lambda: _rule_value('pause_angle_tolerance_deg'))
    pause_speed_deg_s: float = field(default_factory=lambda: _rule_value('pause_speed_deg_s'))

    @property
    def pause_min_deg(self):
        return self.pause_angles_deg[0]-self.pause_angle_tolerance_deg

    @property
    def pause_max_deg(self):
        return self.pause_angles_deg[0]+self.pause_angle_tolerance_deg

    def __post_init__(self):
        values=(self.angle_tolerance_deg,self.final_hold_s,self.hard_contact_limit_N,
                self.episode_cutoff_s,self.pause_s,self.pause_angle_tolerance_deg,
                self.pause_speed_deg_s,*self.pause_angles_deg)
        if any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in values):
            raise ValueError('Scoring settings must be finite positive numbers')
        if len(self.pause_angles_deg)!=2 or self.pause_angles_deg[0]>=self.pause_angles_deg[1]:
            raise ValueError('Slow opening requires two ordered targets')


class SequentialPauseTracker:
    """Two ordered waits, without accumulating time across separate visits."""
    def __init__(self,config,observation_dt_s,*,record_complete=True,physics_dt_s=.001):
        self.config=config
        self.dt=_finite(observation_dt_s,'observation_dt_s')
        if self.dt<=0:raise ValueError('Observation period must be positive')
        self.coarse=self.dt>physics_dt_s+1e-9
        self.coverage_ok=bool(record_complete)
        self.last_time=None;self.sample_count=0;self.index=0
        self.starts=[None,None];self.lower_starts=[None,None]
        self.pauses=[dict(target_opening_deg=float(a),status='pending',
            longest_wait_s=0.,longest_possible_wait_s=0.,completed_time_s=None,
            crossed_before_verified_wait=False,crossing_time_s=None)
            for a in config.pause_angles_deg]

    def observe(self,time_s,opening_deg,speed_deg_s):
        t,angle,speed=(_finite(time_s,'time_s'),_finite(opening_deg,'opening_deg'),
                       _finite(speed_deg_s,'speed_deg_s'))
        previous=self.last_time
        if previous is None:
            if abs(t)>1e-9:self.coverage_ok=False
        elif t<=previous:raise ValueError('Observation times must strictly increase')
        elif t-previous>self.dt+max(1e-7,self.dt*.01):
            self.coverage_ok=False
            self.starts=[None,None];self.lower_starts=[None,None]
            previous=None
        self.last_time=t;self.sample_count+=1;c=self.config
        while self.index<len(self.pauses):
            i=self.index;row=self.pauses[i];target=row['target_opening_deg']
            eligible=abs(angle-target)<=c.pause_angle_tolerance_deg+1e-9 and abs(speed)<c.pause_speed_deg_s
            if eligible:
                if self.starts[i] is None:
                    self.starts[i]=t
                    self.lower_starts[i]=previous if previous is not None else t
                duration=t-self.starts[i]
                row['longest_wait_s']=max(row['longest_wait_s'],duration)
                row['longest_possible_wait_s']=max(row['longest_possible_wait_s'],t-self.lower_starts[i])
                if duration>=c.pause_s-1e-9:
                    row.update(status='pass',completed_time_s=self.starts[i]+c.pause_s)
                    self.index+=1
                    continue
            else:
                if self.starts[i] is not None:
                    row['longest_possible_wait_s']=max(row['longest_possible_wait_s'],t-self.lower_starts[i])
                self.starts[i]=self.lower_starts[i]=None
            if angle>target+c.pause_angle_tolerance_deg+1e-9:
                row.update(status='unknown' if self.coarse and row['longest_possible_wait_s']>=c.pause_s-1e-9 else 'fail',
                    crossed_before_verified_wait=True,crossing_time_s=t)
                self.index+=1
                continue
            break

    def result(self):
        c=self.config;complete=self.coverage_ok and self.sample_count>=2
        rows=[]
        for original in self.pauses:
            row=dict(original)
            if row['status']=='pending':
                row['status']='unknown' if not complete or (self.coarse and row['longest_possible_wait_s']>=c.pause_s-1e-9) else 'fail'
            if not complete:row['status']='unknown'
            row.update(required_wait_s=c.pause_s,angle_tolerance_deg=c.pause_angle_tolerance_deg,
                       speed_strictly_below_deg_s=c.pause_speed_deg_s)
            rows.append(row)
        status=('unknown' if not complete else 'fail' if any(r['status']=='fail' for r in rows)
                else 'unknown' if any(r['status']=='unknown' for r in rows) else 'pass')
        return dict(status=status,ordered=True,pause_angles_deg=list(c.pause_angles_deg),
            required_wait_s_each=c.pause_s,angle_tolerance_deg=c.pause_angle_tolerance_deg,
            speed_strictly_below_deg_s=c.pause_speed_deg_s,pauses=rows,
            sequence_completed=status=='pass',
            completed_time_s=rows[-1]['completed_time_s'] if status=='pass' else None,
            observed_completed_wait_s=sum(c.pause_s for r in rows if r['status']=='pass'),
            observation_complete=complete,observation_dt_s=self.dt,
            reason='按顺序在18°、45°各停稳5秒' if status=='pass' else
                   '缺少按顺序完成的18°或45°五秒停顿' if status=='fail' else
                   '记录缺口或采样跨过五秒边界，不能确认双停顿')


def slow_open_status(samples,*,observation_dt_s=.001,record_complete=True,
                     physics_dt_s=.001,config=None):
    """Audit only waiting: samples need time_s/opening_deg/speed_deg_s."""
    tracker=SequentialPauseTracker(config or ScoreConfig(),observation_dt_s,
        record_complete=record_complete,physics_dt_s=physics_dt_s)
    for row in samples:
        tracker.observe(row['time_s'],row['opening_deg'],row['speed_deg_s'])
    return tracker.result()


class OnlineScorer(V22Scorer):
    def __init__(self,*args,config=None,**kwargs):
        super().__init__(*args,config=config or ScoreConfig(),**kwargs)
        self.waits=SequentialPauseTracker(self.config,self.observation_dt_s,
            record_complete=self.coverage_ok,physics_dt_s=self.physics_dt_s)

    def observe(self,time_s,opening_deg,speed_deg_s,contacts,metadata=None):
        super().observe(time_s,opening_deg,speed_deg_s,contacts,metadata)
        if self.variant=='A1-缓开':
            self.waits.observe(time_s,opening_deg,speed_deg_s)
            # Override the historical single-band diagnostic before the next
            # observation/result. The base v2.1/v2.2 source remains unchanged.
            states=self.waits.pauses
            self.pause_completion=(states[-1]['completed_time_s']
                if all(r['status']=='pass' for r in states) else None)
            self.pause_start=None
            self.longest_pause=max(r['longest_wait_s'] for r in states)
            self.pause_bypassed=any(r['crossed_before_verified_wait'] for r in states)

    def result(self,*,final_target_override=None):
        result=super().result(final_target_override=final_target_override)
        result.update(schema_version='2.3',procedure_version='A1-v2.3',
            assessment_scope='通用阀门规程v2.3（所有工位缓开统一双停顿）')
        result['scoring_notes']=[s for s in result['scoring_notes'] if '缓开扣除规定5秒' not in s]
        result['scoring_notes'].append('缓开统一18°停5秒、45°停5秒、再全开；等待时间按已核实的两个停顿分别扣除。')
        if self.variant!='A1-缓开':return result
        pause=self.waits.result()
        if not self.coverage_ok:
            pause['status']='unknown';pause['sequence_completed']=False;pause['observation_complete']=False
            pause['reason']='角度记录存在缺口，不能确认完整双停顿'
        result['rules']['slow_open_wait']=pause
        result['rules']['R2']=dict(pause,alias_for='slow_open_wait')
        rules={key:result['rules'][key]['status'] for key in
               ('hard_collision','slow_open_wait','final_target_hold','working_face')}
        failed=[key for key,value in rules.items() if value=='fail']
        unknown=[key for key,value in rules.items() if value=='unknown']
        if not self.coverage_ok:unknown.append('continuous_observation')
        passed=False if failed else None if unknown else True
        final=result['rules']['final_target_hold']
        if final['status']=='pass':
            if pause['status']=='fail':
                result['grade']='到位但缓开不合格';result['grade_unresolved_reason']=None
            elif pause['status']=='unknown':
                result['grade']=None;result['grade_unresolved_reason']='已到位，但双停顿证据不足'
        result['result_level']=result['base_outcome']=result['grade']
        label=result['grade'] or '分级待定'
        if result['metrics']['arrived_then_rebounded_at_end']:label+='（到位后退回）'
        elif final['status']=='fail':label+='（结束未到位或未稳满1秒）'
        face=result['rules']['working_face']
        if face['status']=='fail':label+='（工作面不符）' if self.wrong_face_drive else '（未见工作面实际带动）'
        elif face['status']=='unknown':label+='（操作面证据不足）'
        if result['hard_collision']:label+='（有硬撞）'
        result.update(passed=passed,qualified_completion=passed is True,
            verdict='通过' if passed is True else '未通过' if passed is False else '无法判断',
            failed_rules=failed,unverified_rules=unknown,result_label=label)
        raw=result['metrics'].get('raw_turning_time_s')
        deducted=pause['observed_completed_wait_s'] if raw is not None else 0.
        result['metrics'].update(required_wait_deducted_s=deducted,
            effective_turning_time_s=None if raw is None else max(0.,raw-deducted),
            slow_open_pause_status=pause['status'],slow_open_pause_targets_deg=list(self.config.pause_angles_deg))
        return result


def score_record(record,config=None):
    """Re-score saved evidence with v2.3, preserving old files and labels."""
    sampling=record.get('sampling',{});physics_dt=sampling.get('physics_dt_s',.001)
    adapter=record.get('robot_adapter',{});evidence=record.get('evidence',{})
    applicable=record.get('face_constraint_applicable',adapter.get('tool_adapter')=='s_hook')
    complete=sampling.get('complete',sampling.get('coverage')=='every_physics_step')
    scorer=OnlineScorer(record.get('variant'),physics_dt_s=physics_dt,
        observation_dt_s=sampling.get('observation_dt_s',physics_dt),
        sampling_coverage=sampling.get('coverage','unspecified'),config=config,
        actual_station=record.get('actual_station','A1'),source=record.get('source'),
        contact_evidence=record.get('contact_evidence_complete',False) or all(evidence.get(k) is True for k in ('R5','R7')),
        expected_complete=complete,face_constraint_applicable=applicable,
        face_evidence_complete=record.get('face_evidence_complete',False))
    for row in record.get('samples',[]):
        scorer.observe(row['time_s'],row['opening_deg'],row['speed_deg_s'],row.get('contacts',[]),row)
    scorer.add_historical_summary(record.get('old_score_v2') or record.get('preserved_score'))
    final=record.get('final_observation') or {}
    terminal=final_target_status(record.get('samples',[]),record.get('variant'),
        observation_dt_s=sampling.get('observation_dt_s',physics_dt),record_complete=complete,
        final_time_s=final.get('time_s'),final_opening_deg=final.get('opening_deg'))
    result=scorer.result(final_target_override=terminal)
    result.update(data_kind=record.get('data_kind','unspecified'),historical_comparison_only=True,
        original_procedure_version=record.get('procedure_version','unspecified'),
        runtime_settings=record.get('runtime',{}))
    return result
