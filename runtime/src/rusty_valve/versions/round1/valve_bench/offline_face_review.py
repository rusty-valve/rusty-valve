"""Opt-in offline repair of v2.5 face identification, never an online policy.

No controller or shared recognition code imports this module. The v2.5 task
rules are inherited unchanged. A previously unknown contact can be identified
on a unique finite metal face from position and outward-side evidence; a
30-degree normal cone is not used as a substitute for surface membership.
Existing valid hole lips and nonworking surfaces keep their recorded identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import math
from pathlib import Path
import time

from .scoring_v25 import OnlineScorer as OriginalScorer, ScoreConfig
from .tool_face_evidence import (geometry, classify_s_face, vec, unit, sub,
    dot, cross, add)

REVISION='v25_offline_finite_surface_r2'


def inside_round_hole(point,plane,g):
    """Exclude the real 32-sided void from either face of profile segment 0."""
    if plane['segment']!=0:return False
    for rim in g['hole_rims']:
        delta=sub(point,rim['centre'])
        x,y=dot(delta,rim['u']),dot(delta,rim['v'])
        radial2=x*x+y*y
        if radial2>rim['radius']**2+1e-14:continue
        if radial2<(rim['radius']*math.cos(math.pi/rim['facets']))**2-1e-14:return True
        count=0
        for i in range(rim['facets']):
            a=2*math.pi*i/rim['facets'];b=2*math.pi*(i+1)/rim['facets']
            ax,ay=rim['radius']*math.cos(a),rim['radius']*math.sin(a)
            bx,by=rim['radius']*math.cos(b),rim['radius']*math.sin(b)
            if (bx-ax)*(y-ay)-(by-ay)*(x-ax) < -1e-14:break
            count+=1
        if count==rim['facets']:return True
    return False


def finite_working_rims(point,normal,g):
    """Identify the permitted front lip from its finite physical position.

    Native normals at subdivided collision cells need not lie in the ideal
    face/bore normal cone. Require the real working-side mouth and an outward
    working-side component, with the existing geometric edge band unchanged.
    Bore interiors and the opposite mouth do not acquire a working label.
    """
    matches=[]
    for rim in g['hole_rims']:
        delta=sub(point,rim['centre']);mouth=dot(delta,rim['normal'])
        outward=dot(normal,rim['normal'])
        if abs(mouth)>g['edge_margin_m']+1e-9 or outward<=0.:continue
        xy=(dot(delta,rim['u']),dot(delta,rim['v']));radial=math.hypot(*xy)
        if not rim['radius']*math.cos(math.pi/rim['facets'])-g['edge_margin_m']-1e-9<=radial<=rim['radius']+g['edge_margin_m']+1e-9:continue
        distance=math.inf
        for i in range(rim['facets']):
            a=2*math.pi*i/rim['facets'];b=2*math.pi*(i+1)/rim['facets']
            ax,ay=rim['radius']*math.cos(a),rim['radius']*math.sin(a)
            bx,by=rim['radius']*math.cos(b),rim['radius']*math.sin(b)
            dx,dy=bx-ax,by-ay
            fraction=max(0.,min(1.,((xy[0]-ax)*dx+(xy[1]-ay)*dy)/(dx*dx+dy*dy)))
            distance=min(distance,math.hypot(xy[0]-ax-fraction*dx,xy[1]-ay-fraction*dy))
        if distance<=g['edge_margin_m']+1e-9:
            matches.append(dict(face=rim['face'],region='finite_working_hole_rim',
                segment=rim['segment'],side=rim['side'],plane_distance_m=mouth,
                distance_to_hole_boundary_m=distance,outward_normal_component=outward,
                geometry_version=g['version'],identification_revision=REVISION,
                basis='physical_working_mouth_boundary_and_outward_working_side'))
    return matches


def classify_offline(point,normal,*,recorded_label=None):
    p,n=vec(point),unit(vec(normal));g=geometry()
    old=(recorded_label if recorded_label is not None else
         classify_s_face(list(p),list(n),allow_working_hole_rims=True))
    if old.get('face')!='unknown':
        # Names/paint do not fill the physical hole. This is also a negative
        # control for new position-based membership, not a new region rule.
        if old.get('region')=='flat_face':
            plane=next((q for q in g['planes'] if q['segment']==old.get('segment')
                        and q['side']==old.get('side')),None)
            if plane is not None and inside_round_hole(p,plane,g):
                rims=finite_working_rims(p,n,g)
                if rims and {r['face'] for r in rims}=={old['face']}:return rims[0]
                return dict(face='unknown',reason='point_projects_into_round_bore_not_flat_metal')
        return old
    # At an extrusion edge, outward side and working plane may be ambiguous;
    # retain the existing edge/rim logic rather than granting a new exception.
    if abs(p[0])>=g['half_width']-g['edge_margin_m']:return old
    candidates=finite_working_rims(p,n,g)
    for plane in g['planes']:
        delta=sub(p,plane['a']);along=dot(delta,plane['tangent'])
        distance=dot(delta,plane['normal']);outward=dot(n,plane['normal'])
        if not (g['edge_margin_m']<along<plane['length']-g['edge_margin_m']
                and abs(distance)<=g['plane_tolerance_m'] and outward>0.):continue
        if inside_round_hole(p,plane,g):continue
        candidates.append(dict(face=plane['face'],region='finite_flat_metal',
            segment=plane['segment'],side=plane['side'],
            plane_distance_m=distance,outward_normal_component=outward,
            normal_angle_deg=math.degrees(math.acos(max(-1.,min(1.,outward)))),
            geometry_version=g['version'],identification_revision=REVISION,
            basis='finite_metal_position_and_outward_side_not_normal_cone',
            legacy_face=old.get('face'),legacy_reason=old.get('reason')))
    faces={candidate['face'] for candidate in candidates}
    if len(faces)==1:return min(candidates,key=lambda q:abs(q['plane_distance_m']))
    if len(faces)>1:return dict(face='unknown',reason='multiple_incompatible_finite_surfaces')
    return old


def assess_offline(evidence,*,recorded_label=None):
    if not isinstance(evidence,dict) or evidence.get('basis')!='solved_contact_point_normal_wrench_v1':
        return dict(evidence_status='unknown',face='unknown',driving=None,
                    reason='missing_solved_contact_geometry_and_direction')
    try:
        label=classify_offline(evidence['tool_surface_position_local_m'],
            evidence['tool_outward_normal_local'],recorded_label=recorded_label)
        lever=sub(vec(evidence['contact_position_world_m']),vec(evidence['valve_axis_point_world_m']))
        force=vec(evidence['force_on_handle_world_N']);moment=vec(evidence['moment_on_handle_world_Nm'])
        axis=unit(vec(evidence['requested_axis_world']))
        omega=float(evidence['angular_speed_about_requested_axis_rad_s'])
        if not math.isfinite(omega):raise ValueError('Finite speed required')
        torque=dot(add(cross(lever,force),moment),axis);power=torque*omega
    except (ValueError,TypeError,KeyError,OverflowError):
        return dict(evidence_status='unknown',face='unknown',driving=None,
                    reason='incomplete_or_invalid_contact_vectors')
    return dict(evidence_status='observed',face=label['face'],geometry=label,
        requested_torque_Nm=torque,angular_speed_about_requested_axis_rad_s=omega,
        power_W=power,driving=power>0.,driving_toward_target=power>0. and torque>0.)


class OfflineReviewScorer(OriginalScorer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._labels={};self.repairs=Counter();self.remaining_examples=[]

    def observe(self,time_s,opening_deg,speed_deg_s,contacts,metadata=None):
        self._labels={id(c['working_face_evidence']):c['working_face_geometry']
            for c in contacts if isinstance(c.get('working_face_evidence'),dict)
            and isinstance(c.get('working_face_geometry'),dict)
            and c.get('working_face')==c['working_face_geometry'].get('face')}
        self._review_time=float(time_s)
        try:return super().observe(time_s,opening_deg,speed_deg_s,contacts,metadata)
        finally:self._labels={}

    def _assess_contact_drive(self,evidence):
        label=self._labels.get(id(evidence))
        result=assess_offline(evidence,recorded_label=label)
        if result.get('driving_toward_target'):
            if result.get('geometry',{}).get('identification_revision')==REVISION:
                self.repairs[result['face']]+=1
            elif result.get('face')=='unknown' and len(self.remaining_examples)<8:
                self.remaining_examples.append(dict(time_s=self._review_time,
                    geometry=result.get('geometry'),evidence=evidence))
        return result

    def result(self,*,final_target_override=None):
        score=super().result(final_target_override=final_target_override)
        score.update(offline_face_review_revision=REVISION,offline_review_only=True,
            task_standard_changed=False,online_executor_changed=False)
        score['rules']['working_face']['offline_identification_policy']=REVISION
        score['offline_review_diagnostics']=dict(
            repaired_target_direction_contact_observations=dict(self.repairs),
            remaining_unknown_examples=self.remaining_examples)
        return score


def review_log(path):
    """Stream all saved steps and check the footer before issuing a result."""
    header=footer=scorer=None;count=0;first=last=None;continuous=True
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if footer is not None:raise ValueError('Data follows evidence footer')
            kind=row.get('type')
            if header is None:
                if kind!='header' or row.get('format')!='valve_solved_step_evidence_v1':
                    raise ValueError('Unsupported/missing evidence header')
                if row.get('procedure_version')!='A1-v2.5':
                    raise ValueError('This review only accepts frozen v2.5 records')
                header=row
                scorer=OfflineReviewScorer(row['variant'],actual_station=row['actual_station'],
                    physics_dt_s=row['physics_dt_s'],observation_dt_s=row['observation_dt_s'],
                    face_constraint_applicable=row['face_constraint_applicable'],
                    face_evidence_complete=row['face_evidence_complete'],
                    contact_evidence=row['contact_evidence_complete'],
                    config=ScoreConfig(**row['score_config']),source=str(path))
            elif kind=='sample':
                t=float(row['time_s']);dt=header['physics_dt_s']
                if first is None:first=t;continuous=abs(t)<=1e-9
                elif not (0<t-last<=dt+max(1e-7,dt*.01)):continuous=False
                last=t;count+=1
                scorer.observe(t,row['opening_deg'],row['speed_deg_s'],row['contacts'],row.get('metadata',{}))
            elif kind=='footer':footer=row
            else:raise ValueError('Unexpected evidence record')
    if header is None or footer is None:raise ValueError('Incomplete evidence header/footer')
    if (footer['samples'],footer['first_time_s'],footer['last_time_s'])!=(count,first,last):
        raise ValueError('Evidence footer does not match actual samples')
    complete=bool(footer['record_complete'] and footer['execution_complete']
        and footer['continuous'] and footer['full_contact_vectors_saved'] and continuous and count>=2)
    if not complete:raise ValueError('Incomplete physical observation; no revised score issued')
    score=scorer.result()
    score['physical_step_evidence']=dict(samples=count,first_time_s=first,last_time_s=last,
        record_complete=True,gzip_read_to_eof=True,physical_steps_reconstructed=0)
    return score


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def review_attempt(job):
    row,root,output=job;root,output=Path(root),Path(output)
    folder=root/row['raw_directory'];original=json.loads((folder/'result.json').read_text())
    started=time.monotonic();score=review_log(folder/'physical_step_evidence.jsonl.gz')
    for key in ('final_target_hold','hard_collision','slow_open_wait','elapsed_time'):
        if score['rules'][key]!=original['score']['rules'][key]:
            raise ValueError(f'{row["episode_id"]}: non-face rule changed: {key}')
    if score['grade']!=original['score']['grade']:raise ValueError('Non-face grade changed')
    outcome='qualified' if score['passed'] is True else 'failed' if score['passed'] is False else 'pending_review'
    result=dict(episode_id=row['episode_id'],role=row['role'],task=row['task'],
        original_status=row['status'],original_passed=original['score']['passed'],
        review_status=outcome,review_passed=score['passed'],
        full_return=bool(original.get('controller_completed')),
        original_result=str(folder/'result.json'),evidence=str(folder/'physical_step_evidence.jsonl.gz'),
        original_scores_preserved=True,task_standard='A1-v2.5',implementation=REVISION,
        non_face_rules_unchanged=True,physics_steps_run=0,score=score,
        elapsed_seconds=round(time.monotonic()-started,3))
    group='独立测试复核' if row['role']=='test' else '训练记录复核' if row['status']=='pending_review' else '对照记录复核'
    path=output/group/(row['episode_id']+'.json');save(path,result)
    return {k:v for k,v in result.items() if k!='score'}|dict(review_file=str(path))




