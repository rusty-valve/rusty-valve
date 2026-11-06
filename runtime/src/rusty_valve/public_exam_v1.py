"""Offline, robot-neutral scorer for 项目指挥/规程/公开考试_v1.md.

No simulator, policy, renderer or network is imported. Contact ``kind`` must
describe a robot contact: handle / self / fixture / environment. Include ALL
robot parts and ALL non-handle environment objects, not just named fixtures.
``force_N`` is the magnitude of ONE contact's force, never a sum of contacts.
The historical RL adapter also accepts per-category maxima, which preserve a
threshold test for covered categories but cannot prove omitted categories safe.

Angles are opening degrees (0 closed, 90 open). 'Steady' means continuously
within the rule's 5-degree target band; v1 specifies no additional speed cap.
Only the endpoint (or the saved 90-second cutoff) counts, not earlier arrival.
Time comparisons tolerate 1e-7 seconds of accumulated floating-point error;
angle and force thresholds are used exactly as written in the rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

VERSION = 'public_exam_v1'
AUTHORITY = 'levels/01-handle-ball-valve/#public-exam-v1'
TIME_EPS = 1e-7
CUTOFF = 90.0


def target_angle(variant):
    if variant in ('close', 'A1-关', 'A2-关'):
        return 0.0
    if variant in ('open', 'A1-开', 'A2-开'):
        return 90.0
    raise ValueError('Public v1 accepts open/close only; specify opening degrees.')


class PublicExamScorer:
    """Feed original solved observations; finalize with explicit provenance.

    A known collision can establish failure even if another rule is unknown.
    A pass requires all three rules and physical-contact validity to be proven.
    Incomplete evidence is always exposed, including for a proven failure.
    """

    def __init__(self, variant, physics_dt_s):
        self.target = target_angle(variant)
        self.dt = float(physics_dt_s)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('A finite positive physical timestep is required')
        self.variant = variant
        self.first = self.last = self.initial = self.angle = None
        self.hold_start = None
        self.count = 0
        self.coverage = True
        self.contacts_valid = True
        self.max_progress = self.max_speed = 0.0
        self.peaks = {'environment': 0.0, 'handle': 0.0, 'self': 0.0}
        self.peak_contact = self.first_hard = None
        self.contact_parts = set()
        self.contact_kinds = Counter()
        self.handle_seen = False
        self.unknown_contact_kinds = set()
        self.after_cutoff_samples = 0

    def observe(self, row):
        t, angle = float(row['time_s']), float(row['opening_deg'])
        if not math.isfinite(t) or not math.isfinite(angle) or t < -TIME_EPS:
            raise ValueError('Nonfinite or negative-time observation')
        if t > CUTOFF + TIME_EPS:
            self.after_cutoff_samples += 1
            return
        t = CUTOFF if abs(t-CUTOFF) <= TIME_EPS else t
        gap = False
        if self.last is None:
            self.first = t
            self.initial = angle
            self.coverage = abs(t) <= TIME_EPS
        else:
            if t <= self.last:
                raise ValueError('Observation times must increase strictly')
            gap = abs(t-self.last-self.dt) > TIME_EPS
            self.coverage &= not gap
        if abs(angle-self.target) <= 5.0:
            if self.hold_start is None or gap:
                self.hold_start = t
        else:
            self.hold_start = None
        self.last, self.angle = t, angle
        self.count += 1
        direction = 1.0 if self.target == 90.0 else -1.0
        self.max_progress = max(self.max_progress, direction*(angle-self.initial))
        speed = row.get('speed_deg_s')
        if speed is not None and math.isfinite(float(speed)):
            self.max_speed = max(self.max_speed, abs(float(speed)))
        if not isinstance(row.get('contacts'), list):
            self.contacts_valid = False
        for c in row.get('contacts', []) or []:
            try:
                force = float(c['force_N'])
            except (KeyError, ValueError, TypeError):
                self.contacts_valid = False
                continue
            if not math.isfinite(force) or force < 0:
                self.contacts_valid = False
                continue
            kind = c.get('kind')
            self.contact_kinds[kind] += 1
            if kind == 'handle':
                self.peaks['handle'] = max(self.peaks['handle'], force)
                if force > 0:
                    self.handle_seen = True
                    self.contact_parts.add(str(c.get('robot_geom', c.get('robot_part',
                                           c.get('part', 'unspecified_robot_part')))))
            elif kind == 'self':
                self.peaks['self'] = max(self.peaks['self'], force)
            elif kind in ('fixture', 'environment'):
                detail = {k: c[k] for k in ('kind', 'force_N', 'robot_geom', 'other_geom',
                          'robot_part', 'contact_index') if k in c}
                detail['time_s'] = t
                if force > self.peaks['environment']:
                    self.peaks['environment'] = force
                    self.peak_contact = detail
                if force > 20.0 and self.first_hard is None:
                    self.first_hard = detail
            else:
                # Historical 'idle' is an overlapping arm maximum; it does not
                # say whether the counterpart was handle, self or environment.
                self.contacts_valid = False
                self.unknown_contact_kinds.add(str(kind))

    def finish(self, *, execution_complete, terminal_time_s,
               contact_coverage_complete, physical_contact_only,
               source, provenance, evidence_gaps=()):
        gaps = list(evidence_gaps)
        at_cutoff = self.last is not None and abs(self.last-CUTOFF) <= TIME_EPS
        known_end = at_cutoff or bool(execution_complete and self.last is not None
            and terminal_time_s is not None
            and abs(float(terminal_time_s)-self.last) <= TIME_EPS)
        dense = self.coverage and self.count >= 2
        complete_contacts = bool(dense and known_end and contact_coverage_complete
                                 and self.contacts_valid)
        hold = 0.0 if self.hold_start is None else self.last-self.hold_start
        in_band = self.angle is not None and abs(self.angle-self.target) <= 5.0
        if not known_end:
            goal = 'unverified'
        elif not in_band:
            goal = 'fail'
        elif hold >= 1.0-TIME_EPS:
            goal = 'pass'
        else:
            goal = 'fail' if dense else 'unverified'
        collision = 'fail' if self.first_hard else ('pass' if complete_contacts else 'unverified')
        elapsed = 'pass' if known_end else 'unverified'
        physical = ('pass' if physical_contact_only is True else
                    'invalid' if physical_contact_only is False else 'unverified')
        rules = {
            'target_hold': dict(status=goal, target_opening_deg=self.target,
                tolerance_deg=5.0, final_opening_deg=self.angle, final_in_band=in_band,
                end_hold_s=hold, required_hold_s=1.0, hold_start_s=self.hold_start,
                basis='original solved angles; consecutive in-band samples at the actual endpoint/cutoff'),
            'environment_collision': dict(status=collision, limit_N=20.0,
                comparison='individual force > 20 N; no force summation',
                peak_force_N=self.peaks['environment'], peak_contact=self.peak_contact,
                first_violation=self.first_hard, all_environment_contacts_covered=complete_contacts),
            'deadline': dict(status=elapsed, cutoff_s=CUTOFF, evaluated_end_s=self.last,
                endpoint='90_second_cutoff' if at_cutoff else 'recorded_execution_end',
                ignored_samples_after_cutoff=self.after_cutoff_samples),
        }
        if not known_end:
            gaps.append('Actual execution end / exact 90-second sample is not established.')
        if not dense:
            gaps.append('Continuous every-physics-step evidence from t=0 is missing.')
        if not contact_coverage_complete:
            gaps.append('Not all robot-to-non-handle-environment contacts are covered.')
        if not self.contacts_valid:
            gaps.append('Invalid force values or unclassified contacts: '+str(sorted(self.unknown_contact_kinds)))
        if physical == 'unverified':
            gaps.append('Physical-contact-only execution provenance is missing.')
        if physical == 'invalid':
            gaps.append('Execution is declared nonphysical or the valve is directly driven/rewritten.')
        statuses = [r['status'] for r in rules.values()]
        status = ('invalid' if physical == 'invalid' else 'fail' if 'fail' in statuses
                  else 'pass' if all(s == 'pass' for s in statuses) and physical == 'pass' and not gaps
                  else 'unverified')
        grade = ('completed' if goal == 'pass' else 'mostly_completed' if self.max_progress > 45.0
                 else 'stalled' if self.handle_seen and self.max_progress < 5.0
                 else 'partly_turned' if self.max_progress >= 5.0
                 else 'never_engaged' if complete_contacts and not self.handle_seen else 'unverified')
        return dict(version=VERSION, authority=AUTHORITY, source=str(source),
            result=status, passed=True if status == 'pass' else False if status == 'fail' else None,
            rules=rules, physical_validity=dict(status=physical, provenance=provenance),
            failed_rules=[k for k, r in rules.items() if r['status'] == 'fail'],
            unverified_rules=[k for k, r in rules.items() if r['status'] == 'unverified'],
            evidence_complete=not gaps, evidence_gaps=list(dict.fromkeys(gaps)),
            sampling=dict(samples_scored=self.count, first_time_s=self.first,
                          last_time_s=self.last, physics_dt_s=self.dt, continuous=dense),
            progress_grade=grade, progress_grade_is_not_public_pass=True,
            record_only=dict(maximum_progress_deg=self.max_progress,
                final_progress_deg=None if self.angle is None else (self.angle-self.initial)*(1 if self.target else -1),
                maximum_speed_deg_s=self.max_speed, handle_contact_seen=self.handle_seen,
                handle_contact_parts=sorted(self.contact_parts), handle_peak_force_N=self.peaks['handle'],
                self_contact_peak_force_N=self.peaks['self'], contact_observations=dict(self.contact_kinds)),
            offline_only=True, physical_steps_executed=0)


def score_solved_log(path):
    """Read original gzip JSONL, verify footer and score without state replay."""
    path = Path(path)
    footer = header = scorer = None
    count = 0
    first = last = None
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if footer is not None:
                raise ValueError('Data follows evidence footer')
            if header is None:
                if row.get('type') != 'header' or row.get('format') != 'valve_solved_step_evidence_v1':
                    raise ValueError('Unsupported/missing solved evidence header')
                header = row
                scorer = PublicExamScorer(row['variant'], row['physics_dt_s'])
            elif row.get('type') == 'sample':
                scorer.observe(row)
                t = float(row['time_s'])
                if first is None:
                    first = t
                if last is not None and t <= last:
                    raise ValueError('Unordered source samples')
                last = t
                count += 1
            elif row.get('type') == 'footer':
                footer = row
            else:
                raise ValueError('Unexpected evidence row')
    if scorer is None:
        raise ValueError('Empty evidence')
    footer_valid = bool(footer and (footer.get('samples'), footer.get('first_time_s'),
                                    footer.get('last_time_s')) == (count, first, last))
    original_score = (footer or {}).get('score') or {}
    runtime = original_score.get('runtime_settings', {})
    physical = runtime.get('valve_contact_only')
    if header.get('data_kind') != 'recorded_simulation_every_solved_physics_step':
        physical = None
    gaps = [] if footer_valid else ['Missing or inconsistent evidence footer.']
    result = scorer.finish(execution_complete=bool(footer_valid and footer.get('record_complete')
                           and footer.get('execution_complete')), terminal_time_s=last,
        contact_coverage_complete=bool(footer_valid and header.get('contact_evidence_complete') is True),
        physical_contact_only=physical, source=path,
        provenance=dict(data_kind=header.get('data_kind'), runtime=runtime,
            basis='original solved-step footer plus recorded native controller/runtime declarations'),
        evidence_gaps=gaps)
    result['original_scoring_version'] = header.get('procedure_version')
    return result




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = score_solved_log(args.record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(dict(result=result['result'], output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
