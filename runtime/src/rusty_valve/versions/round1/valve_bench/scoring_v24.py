"""v2.4: distinguish task-driving contact from contact after arrival.

The first step crossing into the existing +/-5 degree goal band is assessed.
Only after an observed correct working face has reached that band are later
contacts in its continuous in-band suffix record-only. An excursion immediately
ends this treatment. Earlier wrong/unknown driving is never erased. Physical
angles, not controller phase names, determine the interval. No face-share,
wrist-angle, work/force minimum, or additional goal-angle threshold is added.

Historical v2.3 is unchanged. This module consumes observations, never physics.
"""
from __future__ import annotations

from collections import Counter

from .scoring_v23 import OnlineScorer as V23Scorer, ScoreConfig
from .scoring_v22 import final_target_status
from .scoring_v2 import VARIANTS
from .tool_face_evidence import assess_contact_drive


def plain_result_label(result):
    """Describe the specific result without the old combined catch-all suffix."""
    closing = result.get('variant') == 'A1-关'
    end = '关到底' if closing else '开到底'
    grade = result.get('grade')
    final = result.get('rules', {}).get('final_target_hold', {})
    face = result.get('rules', {}).get('working_face', {})
    if result.get('metrics', {}).get('arrived_then_rebounded_at_end'):
        label = f'曾经{end}，收手后又被带回'
    elif final.get('status') == 'fail' and final.get('final_in_target'):
        label = f'已经{end}，但结束前没有保持满1秒'
    else:
        label = {'完成': f'已经{end}', '没接上': '没有接上手柄',
                 '拧不动': stalled_contact_description(result.get('seating_diagnostic')),
                 '转了一些': f'转动了5°到45°，还没有{end}',
                 '基本完成': f'转动超过45°，但最后没有{end}',
                 '到位但缓开不合格': '已经开到底，但规定的缓开停顿没有完成'}.get(
                     grade, f'最后没有{end}' if final.get('status') == 'fail' else '现有记录不能判定')
    if result.get('hard_collision'):
        label += '；发生了硬撞'
    if face.get('status') == 'fail' and face.get('nonworking_face_drive_seen'):
        label += '；转动过程中用了非工作面带动手柄'
    elif final.get('status') == 'pass' and face.get('status') == 'unknown':
        label += ('；有一段有效转动的接触面仍无法确定'
                  if face.get('evidence_complete') else '；缺少完整接触记录，不能确认实际工作面')
    elif final.get('status') == 'pass' and face.get('status') == 'fail':
        label += '；没有记录到正确工作面实际带动手柄'
    return label


def stalled_contact_description(diagnostic):
    """Explain low progress without claiming contact is a properly seated tool."""
    d = diagnostic or {}
    if not d.get('supported_geometry_observations'):
        return '碰到了手柄并用力，但转动不足5°；尚未分清是否真正入槽到目标位置'
    if d.get('confirmed_seated_contact_seen'):
        if d.get('geometry_left_recess_after_confirmed'):
            return '曾在目标位置接上正确工作面，但过程中又脱开，最终转动不足5°'
        if d.get('latest_state') == 'seated_with_current_correct_face':
            return '已在目标位置入槽并接上正确工作面，但转动不足5°'
        return '曾在目标位置入槽，但最后未确认正确工作面仍贴住手柄，转动不足5°'
    if d.get('geometry_seated_seen'):
        return '工具到了目标入槽位置，但未确认正确工作面贴住，转动不足5°'
    return '碰到了手柄，但尚未在目标位置入槽贴合，转动不足5°'


class OnlineScorer(V23Scorer):
    def _assess_contact_drive(self, evidence):
        """Keep the historical face policy overridable without global state."""
        return assess_contact_drive(evidence)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._correct_face_target_interval = False
        self.terminal_record_only_contacts = 0
        self.terminal_record_only_drive_samples = 0
        self.terminal_record_only_missing_vectors = 0
        self.face_event_count = 0
        self._seating_counts = Counter()
        self._seating_geometry_seen = False
        self._seating_confirmed_seen = False
        self._seating_left_after_confirmed = False
        self._seating_latest = None

    def _observe_seating_diagnostic(self, time_s, contacts, metadata):
        """Diagnostic only: supplied actual geometry + independent solved face.

        Controller readiness or phase never grants a score. Its actual-geometry
        check is sampled at the control command time, which is saved explicitly;
        the correct contact face is checked again from this solved physics step.
        No dwell time or additional positional/force threshold is introduced.
        """
        metadata = metadata or {}
        state = metadata.get('seating_state')
        if (not isinstance(state, dict) or
                state.get('basis') != 'actual_handle_axis_and_finite_recess' or
                state.get('mode') not in ('push', 'pull') or
                metadata.get('phase') not in ('turn', 'reviewed_blue_seating', 'reviewed_orange_seating')):
            return
        if not all(isinstance(state.get(key), bool) for key in
                   ('geometry_seated', 'in_recess', 'target_section_overlaps_tool')):
            return
        expected = 'blue_pull' if state['mode'] == 'pull' else 'orange_push'
        correct = False
        for contact in contacts:
            if (contact.get('kind') != 'handle' or contact.get('robot_part') != 'operation_end'
                    or float(contact.get('force_N', 0.)) <= 0.):
                continue
            actual = self._assess_contact_drive(contact.get('working_face_evidence'))
            if actual['evidence_status'] == 'observed' and actual['face'] == expected:
                correct = True
        seated = bool(state['geometry_seated'] and state['in_recess'] and state['target_section_overlaps_tool'])
        if self._seating_confirmed_seen and not seated:
            self._seating_left_after_confirmed = True
        self._seating_geometry_seen |= seated
        self._seating_confirmed_seen |= seated and correct
        label = ('seated_with_current_correct_face' if seated and correct else
                 'geometry_seated_without_confirmed_face' if seated else 'not_seated_at_target')
        self._seating_counts[label] += 1
        self._seating_latest = dict(state=label, physics_time_s=float(time_s),
            geometry_sample_time_s=metadata.get('time_s'), target_radius_m=state.get('target_radius_m'),
            current_correct_face_observed=correct,
            controller_ready_claim=state.get('ready_to_turn'),
            basis='controller_actual_geometry_cross_checked_with_current_solved_contact')

    def observe(self, time_s, opening_deg, speed_deg_s, contacts, metadata=None):
        previous_time, previous_angle = self.last_time, self.final_opening
        target = VARIANTS[self.variant][1]
        near = abs(opening_deg-target) <= self.config.angle_tolerance_deg+1e-9
        previous_near = (previous_angle is not None and
                         abs(previous_angle-target) <= self.config.angle_tolerance_deg+1e-9)
        continuous = (previous_time is not None and
                      0 < time_s-previous_time <= self.observation_dt_s + max(1e-7, self.observation_dt_s*.01))
        terminal_only = self._correct_face_target_interval and previous_near and near and continuous
        applicable = self.face_constraint_applicable
        # Keep v2.3 angle, collision and historical slow-open calculations, but
        # supply this version's face attribution instead of sticky v2.2 flags.
        self.face_constraint_applicable = False
        try:
            super().observe(time_s, opening_deg, speed_deg_s, contacts, metadata)
        finally:
            self.face_constraint_applicable = applicable
        self._observe_seating_diagnostic(time_s, contacts, metadata)
        if not applicable:
            return
        active = set()
        dt = 0. if previous_time is None else float(time_s)-previous_time
        for contact in contacts:
            if contact.get('kind') != 'handle' or contact.get('robot_part') != 'operation_end':
                continue
            if float(contact.get('force_N', 0.)) <= 0:
                continue
            event = self._assess_contact_drive(contact.get('working_face_evidence'))
            if terminal_only:
                self.terminal_record_only_contacts += 1
            if event['evidence_status'] == 'unknown':
                if terminal_only:
                    self.terminal_record_only_missing_vectors += 1
                else:
                    self.missing_face_evidence = True
                continue
            if not event['driving']:
                continue
            face = event['face']; active.add(face)
            entry = self.face_drive_stats.setdefault(face, dict(driving_samples=0,
                positive_work_J=0., maximum_absolute_torque_Nm=0.,
                target_direction_drive_samples=0, terminal_record_only_drive_samples=0))
            entry['driving_samples'] += 1
            entry['positive_work_J'] += max(0., event['power_W'])*dt
            entry['maximum_absolute_torque_Nm'] = max(entry['maximum_absolute_torque_Nm'], abs(event['requested_torque_Nm']))
            if event['requested_torque_Nm'] > 0:
                entry['target_direction_drive_samples'] += 1
            if terminal_only:
                self.terminal_record_only_drive_samples += 1
                entry['terminal_record_only_drive_samples'] += 1
            elif face in ('orange_push', 'blue_pull') and event['requested_torque_Nm'] > 0:
                self.correct_face_drive = True
            elif face == 'nonworking' and event['requested_torque_Nm'] > 0:
                self.wrong_face_drive = True
            elif face == 'unknown' and event['driving_toward_target']:
                self.unknown_face_drive = True
            if face not in self._active_drive_faces:
                self.face_event_count += 1
                if len(self.face_drive_events) < 24:
                    self.face_drive_events.append(dict(time_s=float(time_s), opening_deg=float(opening_deg),
                        record_only_after_correct_arrival=bool(terminal_only), **event))
        self._active_drive_faces = active
        # First crossing has already been assessed above. Earlier violations
        # remain sticky even if a later correct contact reaches the goal.
        self._correct_face_target_interval = bool(near and self.correct_face_drive and self.coverage_ok)

    def result(self, *, final_target_override=None):
        result = super().result(final_target_override=final_target_override)
        result.update(schema_version='2.4', procedure_version='A1-v2.4',
                      assessment_scope='所有工位统一：正确面达标后目标区内接触只记录，真实错误面推进仍不合格')
        face = result['rules']['working_face']
        face.update(terminal_record_only_contacts=self.terminal_record_only_contacts,
                    terminal_record_only_drive_samples=self.terminal_record_only_drive_samples,
                    terminal_record_only_missing_vectors=self.terminal_record_only_missing_vectors,
                    event_count=self.face_event_count, events_truncated=self.face_event_count > len(self.face_drive_events),
                    attribution_policy='first target-band crossing assessed; later continuous in-band contact after observed correct drive is record-only; excursion restores assessment; earlier violations retained')
        result['metrics']['terminal_record_only_contacts'] = self.terminal_record_only_contacts
        result['seating_diagnostic'] = dict(
            affects_qualification=False,
            supported_geometry_observations=sum(self._seating_counts.values()),
            observed_state_counts=dict(self._seating_counts),
            geometry_seated_seen=self._seating_geometry_seen,
            confirmed_seated_contact_seen=self._seating_confirmed_seen,
            geometry_left_recess_after_confirmed=self._seating_left_after_confirmed,
            latest_state=self._seating_latest['state'] if self._seating_latest else None,
            latest_observation=self._seating_latest,
            scope='diagnosis only; geometry is controller-supplied actual-state measurement, contact face is independently observed; readiness/phase never determines pass')
        result['result_label'] = plain_result_label(result)
        result['scoring_notes'].append('v2.4只按实际阀角划分正确面到位后的连续目标区段；首次跨入、此前错误/未知驱动仍计分。无新占比、角度或最小做功门槛。')
        return result


def _score_record(record, config=None, *, scorer_class):
    """Assess preserved samples; absent raw evidence is never reconstructed."""
    sampling = record.get('sampling', {}); physics_dt = sampling.get('physics_dt_s', .001)
    complete = sampling.get('complete', sampling.get('coverage') == 'every_physics_step')
    applicable = record.get('face_constraint_applicable', record.get('robot_adapter', {}).get('tool_adapter') == 's_hook')
    scorer = scorer_class(record.get('variant'), physics_dt_s=physics_dt, config=config,
        observation_dt_s=sampling.get('observation_dt_s', physics_dt),
        sampling_coverage=sampling.get('coverage', 'unspecified'), expected_complete=complete,
        actual_station=record.get('actual_station', 'A1'), source=record.get('source'),
        contact_evidence=record.get('contact_evidence_complete', False),
        face_constraint_applicable=applicable, face_evidence_complete=record.get('face_evidence_complete', False))
    for row in record.get('samples', []):
        scorer.observe(row['time_s'], row['opening_deg'], row['speed_deg_s'], row.get('contacts', []), row.get('metadata', row))
    final = record.get('final_observation') or {}
    terminal = final_target_status(record.get('samples', []), record.get('variant'),
        observation_dt_s=sampling.get('observation_dt_s', physics_dt), record_complete=complete,
        final_time_s=final.get('time_s'), final_opening_deg=final.get('opening_deg'))
    result = scorer.result(final_target_override=terminal)
    result.update(data_kind=record.get('data_kind', 'unspecified'), historical_comparison_only=True,
        original_procedure_version=record.get('procedure_version', 'unspecified'), runtime_settings=record.get('runtime', {}))
    return result


def score_record(record, config=None):
    return _score_record(record, config, scorer_class=OnlineScorer)
