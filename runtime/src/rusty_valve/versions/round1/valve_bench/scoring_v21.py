"""A1 v2.1 result levels, with evidence kept separate from qualification.

The online path receives every solved physical step. Historical re-scoring may
use coarser preserved samples, but declares that resolution; it does not invent
missing forces or claim to recover a 1 ms trajectory from a 20 ms recording.
Angles are normalized opening angles: 0 closed, 90 open.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .scoring_v2 import ALIASES, VARIANTS, _finite, normalize_contact


@dataclass(frozen=True)
class ScoreConfig:
    angle_tolerance_deg: float = 5.
    final_hold_s: float = 1.
    pause_min_deg: float = 5.
    pause_max_deg: float = 25.
    pause_s: float = 5.
    pause_speed_deg_s: float = 2.
    hard_contact_limit_N: float = 20.
    episode_cutoff_s: float = 90.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.pause_min_deg >= self.pause_max_deg:
            raise ValueError("pause_min_deg must be below pause_max_deg")


class OnlineScorer:
    """Constant-memory result scorer; elapsed time never creates a failure rule.

    ``metadata`` is optional contextual evidence, not a controller-owned score.
    Known keys: phase, effort_active, effective_handle_contact,
    force_drive_fraction, action_started, action_finished. A positive solved
    operation-end/handle force independently proves physical contact and force.
    """
    def __init__(self, variant, *, physics_dt_s, config=None, actual_station="A1",
                 source=None, contact_evidence=True, expected_complete=True,
                 observation_dt_s=None, sampling_coverage="every_physics_step"):
        self.variant = ALIASES.get(variant, variant)
        if self.variant not in VARIANTS:
            raise ValueError("v2.1 assesses close, open and slow_open")
        self.config = config or ScoreConfig()
        self.physics_dt_s = _finite(physics_dt_s, "physics_dt_s")
        self.observation_dt_s = _finite(observation_dt_s or physics_dt_s, "observation_dt_s")
        if min(self.physics_dt_s, self.observation_dt_s) <= 0:
            raise ValueError("Sampling periods must be positive")
        self.sampling_coverage = sampling_coverage
        self.station, self.source = str(actual_station), source
        self.coverage_ok, self.contact_evidence = bool(expected_complete), bool(contact_evidence)
        self.sample_count = 0
        self.last_time = self.initial_opening = self.final_opening = None
        self.maximum_gap = self.maximum_progress = 0.
        self.target_start = self.pause_start = None
        self.longest_target_hold = self.longest_pause = 0.
        self.physical_target_completion_time = self.completion_time = None
        self.target_entry_time = self.motion_start_time = self.pause_completion = None
        self.motion_time_basis = None
        self.motion_detection_epsilon_deg = .01  # timing only, never a result-level gate
        self._motion_origin = self._previous_opening = None
        self.pause_bypassed = False
        self.maximum_speed = self.maximum_contact = self.hard_peak = 0.
        self.operation_contact_seen = self.effort_contact_seen = False
        self.first_operation_contact_time = None
        self.action_start_time = self.action_end_time = None
        self.action_start_basis = self.action_end_basis = None
        self.counts = {name: 0 for name in ("light_contact", "non_operation_end_handle", "idle_hand_contact", "hard_contact")}
        self._active = {name: set() for name in self.counts}
        self.surface_usage, self._surface_active = {}, set()
        self.contact_count_resolution = "geometry_pairs"
        self.evidence_sources = []
        self.historical_contact_scope_limitations = []

    def observe(self, time_s, opening_deg, speed_deg_s, contacts, metadata=None):
        t, angle, speed = (_finite(time_s, "time_s"), _finite(opening_deg, "opening_deg"),
                           _finite(speed_deg_s, "speed_deg_s"))
        if not isinstance(contacts, list):
            raise ValueError("contacts must be a list")
        metadata = metadata or {}
        if self.last_time is None:
            self.initial_opening = angle
            if abs(t) > 1e-9:
                self.coverage_ok = False
        else:
            dt = t-self.last_time
            if dt <= 0:
                raise ValueError("Observation times must strictly increase")
            self.maximum_gap = max(self.maximum_gap, dt)
            # A shorter terminal interval is allowed; an unobserved long gap
            # cannot establish continuous holding or waiting.
            if dt > self.observation_dt_s+max(1e-7, self.observation_dt_s*.01):
                self.coverage_ok = False
                self.target_start = self.pause_start = None
        previous_opening = self.final_opening
        self.sample_count += 1
        self.last_time, self.final_opening = t, angle
        direction = -1. if self.variant == "A1-关" else 1.
        progress = direction*(angle-self.initial_opening)
        self.maximum_progress = max(self.maximum_progress, progress)
        self.maximum_speed = max(self.maximum_speed, abs(speed))
        phase = metadata.get("phase")
        if self.action_start_time is None and (metadata.get("action_started") is True or
                phase in ("lift_to_chest", "chest_ready", "reach", "approach", "turn", "hold_18", "hold_final")):
            self.action_start_time, self.action_start_basis = t, "recorded_phase_or_action_signal"
        if metadata.get("action_finished") is True or phase == "final_down_hold":
            if self.action_end_time is None:
                self.action_end_time, self.action_end_basis = t, "recorded_return_complete_signal"
        c = self.config
        if self.variant == "A1-缓开":
            eligible = (c.pause_min_deg-1e-9 <= angle <= c.pause_max_deg+1e-9
                        and abs(speed) < c.pause_speed_deg_s and not self.pause_bypassed)
            if eligible:
                if self.pause_start is None:
                    self.pause_start = t
                duration = t-self.pause_start
                self.longest_pause = max(self.longest_pause, duration)
                if duration >= c.pause_s-1e-9 and self.pause_completion is None:
                    self.pause_completion = self.pause_start+c.pause_s
            else:
                self.pause_start = None
            if angle > c.pause_max_deg+1e-9 and self.pause_completion is None:
                self.pause_bypassed = True
        target = VARIANTS[self.variant][1]
        if abs(angle-target) <= c.angle_tolerance_deg+1e-9:
            if self.target_entry_time is None:
                self.target_entry_time = t
            if self.target_start is None:
                self.target_start = t
            duration = t-self.target_start
            self.longest_target_hold = max(self.longest_target_hold, duration)
            if duration >= c.final_hold_s-1e-9:
                when = self.target_start+c.final_hold_s
                if self.physical_target_completion_time is None:
                    self.physical_target_completion_time = when
                if (self.variant != "A1-缓开" or self.pause_completion is not None) and self.completion_time is None:
                    self.completion_time = when
        else:
            self.target_start = None
        active, surfaces = {name: set() for name in self.counts}, set()
        for raw in contacts:
            row = normalize_contact(raw)
            force, kind, role = row["force_N"], row["kind"], row["robot_part"]
            self.maximum_contact = max(self.maximum_contact, force)
            if force <= 0:
                continue
            if "robot_geom" in row and "other_geom" in row:
                pair = tuple(sorted((str(row["robot_geom"]), str(row["other_geom"]))))
            else:
                pair = (kind, role, row["surface"], row.get("other_robot_part", ""))
                self.contact_count_resolution = "legacy_kind_role_surface_groups"
            # Tool/handle contact is explicitly not a hard collision. Robot
            # contact with any other environment geometry is independently
            # counted, including an operation-end/valve-body contact.
            if kind in ("fixture", "self") or (kind == "environment" and role == "operation_end"):
                self.hard_peak = max(self.hard_peak, force)
                active["hard_contact" if force > c.hard_contact_limit_N else "light_contact"].add(pair)
            if kind == "handle" and role != "operation_end":
                active["non_operation_end_handle"].add(pair)
            if role == "idle_hand" or row.get("other_robot_part") == "idle_hand":
                active["idle_hand_contact"].add(pair)
            if kind == "handle" and role == "operation_end":
                self.operation_contact_seen = self.effort_contact_seen = True
                if self.first_operation_contact_time is None:
                    self.first_operation_contact_time = t
                face = row["surface"]
                surfaces.add(face)
                usage = self.surface_usage.setdefault(face, dict(contact_samples=0, contact_episodes=0, maximum_contact_force_N=0.))
                usage["maximum_contact_force_N"] = max(usage["maximum_contact_force_N"], force)
        if metadata.get("effective_handle_contact") is True:
            self.operation_contact_seen = True
            if metadata.get("effort_active") is True:
                self.effort_contact_seen = True
        for name in self.counts:
            self.counts[name] += len(active[name]-self._active[name])
        self._active = active
        for face in surfaces:
            self.surface_usage[face]["contact_samples"] += 1
            self.surface_usage[face]["contact_episodes"] += face not in self._surface_active
        self._surface_active = surfaces
        # Timing begins only after physical tool/handle contact. For preserved
        # coarse traces without contact timing, the controller's recorded turn
        # phase is an explicitly labelled approximation, not force evidence.
        if self.motion_start_time is None:
            if self.operation_contact_seen:
                if self._motion_origin is None:
                    self._motion_origin = previous_opening if previous_opening is not None else angle
                    self.motion_time_basis = "first_observed_motion_after_solved_operation_end_handle_contact"
            elif self.sampling_coverage != "every_physics_step" and phase in ("turn", "hold_18", "hold_final"):
                if self._motion_origin is None:
                    self._motion_origin = angle
                    self.motion_time_basis = "approximate_first_observed_motion_after_recorded_turn_phase_no_contact_timestamps"
            if self._motion_origin is not None and direction*(angle-self._motion_origin) >= self.motion_detection_epsilon_deg-1e-9:
                self.motion_start_time = t

    def add_historical_summary(self, score):
        """Supplement only quantities actually measured by the old scorer.

        Counts/maxima are merged by max, never added to downsampled counts.
        A full old summary can establish that hard collisions were absent;
        angle/pause trajectories remain at their own preserved resolution.
        """
        if not score:
            return
        m, rules = score.get("metrics", {}), score.get("rules", {})
        for name in self.counts:
            self.counts[name] = max(self.counts[name], int(m.get(name+"_episodes", 0)))
        self.maximum_contact = max(self.maximum_contact, float(m.get("maximum_contact_force_N") or 0.))
        self.hard_peak = max(self.hard_peak, float(m.get("maximum_fixture_or_self_contact_force_N") or 0.))
        if self.hard_peak > self.config.hard_contact_limit_N:
            self.counts["hard_contact"] = max(self.counts["hard_contact"], 1)
        old_contact_complete = (score.get("measurement_coverage", {}).get("verified") is True or
                                rules.get("R3", {}).get("status") in ("pass", "fail"))
        self.contact_evidence |= old_contact_complete
        if old_contact_complete and score.get("procedure_version") == "A1-v2.0":
            self.historical_contact_scope_limitations = [
                "旧v2.0摘要只对fixture/self计硬撞；工具碰floor等其余环境若未保存接触记录，不能据摘要排除"]
        for face, usage in m.get("operation_end_surface_usage", {}).items():
            target = self.surface_usage.setdefault(face, {})
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    target[key] = max(target.get(key, 0), value)
            if usage.get("contact_samples", 0) > 0 or usage.get("maximum_contact_force_N", 0) > 0:
                self.operation_contact_seen = self.effort_contact_seen = True
        self.evidence_sources.append(dict(source="preserved_online_score_summary", procedure_version=score.get("procedure_version"),
            quantities=["hard_collision", "contact_counts", "contact_force_maxima", "operation_end_contact_presence"],
            time_resolution_s=score.get("metrics", {}).get("maximum_sample_gap_s"),
            contact_evidence_complete=old_contact_complete))

    def result(self):
        c = self.config
        complete = self.sample_count >= 2 and self.coverage_ok
        target = VARIANTS[self.variant][1]
        direction = -1. if self.variant == "A1-关" else 1.
        final_progress = None if self.initial_opening is None else max(0., direction*(self.final_opening-self.initial_opening))
        progress = None if self.initial_opening is None else self.maximum_progress
        near = self.final_opening is not None and abs(self.final_opening-target) <= c.angle_tolerance_deg+1e-9
        held = self.physical_target_completion_time is not None
        pause_ok = self.variant != "A1-缓开" or self.pause_completion is not None
        hard = self.counts["hard_contact"] > 0
        hard_known = hard or self.contact_evidence
        grade, unresolved = None, None
        # Empty cells in the published five-level table are kept explicit.
        # They are not silently re-labelled "基本完成".
        if held and not pause_ok:
            grade = "到位但缓开不合格"
        elif held and pause_ok and hard_known:
            grade = "完成"
        elif progress is not None and progress > 45. and not near:
            grade = "基本完成"
        elif progress is not None and progress < 5. and self.operation_contact_seen and self.effort_contact_seen:
            grade = "拧不动"
        elif progress is not None and 5. <= progress <= 45.:
            grade = "转了一些"
        elif near and not held:
            unresolved = "已到终点5°以内但未稳满1秒；规程五级表未规定此情况归哪一级"
        elif not self.operation_contact_seen and self.contact_evidence:
            grade = "没接上"
        else:
            unresolved = "现有证据不足以区分五级结果"
        if not complete and grade != "完成":
            unresolved = unresolved or "角度记录存在缺测，结果仅限现有样本"
        known_failure = hard or (complete and (not held or not pause_ok))
        passed = False if known_failure else True if grade == "完成" and complete else None
        if grade == "完成" and not complete:
            grade, unresolved = None, "观察存在缺口，不能确认全程合格"
        elapsed = self.last_time
        start = self.action_start_time if self.action_start_time is not None else 0.
        end = self.action_end_time if self.action_end_time is not None else elapsed
        total = None if end is None else max(0., end-start)
        raw_motion = (None if self.motion_start_time is None or self.target_entry_time is None
                      else max(0., self.target_entry_time-self.motion_start_time))
        deducted = c.pause_s if self.variant == "A1-缓开" and self.pause_completion is not None and raw_motion is not None else 0.
        effective = None if raw_motion is None else max(0., raw_motion-deducted)
        every_step = (self.sampling_coverage == "every_physics_step" and
                      abs(self.observation_dt_s-self.physics_dt_s) <= 1e-9)
        result = dict(schema_version="2.1", procedure_version="A1-v2.1", variant=self.variant,
            actual_station=self.station, source=self.source, data_kind="recorded_simulation",
            assessment_scope="A1正式规程v2.1" if self.station == "A1" else "暂用A1 v2.1通用条款，非正式新工位验收",
            grade=grade, result_level=grade, base_outcome=grade,
            qualified_completion=passed is True, grade_unresolved_reason=unresolved,
            hard_collision=hard if hard_known else None,
            result_label=(("到位" if grade == "完成" and hard else grade)+("（有硬撞）" if hard else "")) if grade else ("结果待定"+("（有硬撞）" if hard else "")),
            passed=passed, verdict="通过" if passed is True else "未通过" if passed is False else "无法判断",
            failed_rules=(["hard_collision"] if hard else [])+(["slow_open_wait"] if complete and not pause_ok else [])+(["target_hold"] if complete and not held else []),
            unverified_rules=(["continuous_observation"] if not complete else [])+(["hard_collision"] if not hard_known else []),
            rules=dict(target_hold=dict(status="pass" if held else "fail" if complete else "unknown", target_opening_deg=target,
                    angle_tolerance_deg=c.angle_tolerance_deg, required_hold_s=c.final_hold_s,
                    longest_hold_s=self.longest_target_hold, completion_time_s=self.physical_target_completion_time),
                slow_open_wait=dict(status="not_applicable" if self.variant != "A1-缓开" else "pass" if pause_ok else "fail" if complete else "unknown",
                    angle_band_deg=[c.pause_min_deg,c.pause_max_deg], speed_strictly_below_deg_s=c.pause_speed_deg_s,
                    required_wait_s=c.pause_s,longest_wait_s=self.longest_pause,completed_time_s=self.pause_completion,
                    opened_beyond_wait_band_before_wait=self.pause_bypassed),
                hard_collision=dict(status="fail" if hard else "pass" if hard_known else "unknown",limit_N=c.hard_contact_limit_N),
                elapsed_time=dict(status="record_only",episode_cutoff_s=c.episode_cutoff_s)),
            metrics=dict(record_duration_s=elapsed,completion_time_s=self.completion_time,
                initial_opening_deg=self.initial_opening,final_opening_deg=self.final_opening,
                angle_progress_deg=progress,maximum_angle_progress_deg=self.maximum_progress,
                final_angle_progress_deg=final_progress,
                rebound_deg=None if final_progress is None else max(0.,self.maximum_progress-final_progress),
                returned_after_progress=final_progress is not None and self.maximum_progress-final_progress >= .01,
                maximum_speed_deg_s=self.maximum_speed,operation_end_contact_seen=self.operation_contact_seen,
                effort_while_contact_seen=self.effort_contact_seen,first_operation_contact_time_s=self.first_operation_contact_time,
                motion_start_time_s=self.motion_start_time,target_first_entry_time_s=self.target_entry_time,
                effective_turning_time_basis=self.motion_time_basis or "unavailable_no_contact_or_turn_phase_timestamps",
                motion_detection_epsilon_deg=self.motion_detection_epsilon_deg,
                effective_turning_time_s=effective,raw_turning_time_s=raw_motion,required_wait_deducted_s=deducted,
                effective_turning_time_complete=raw_motion is not None,
                total_time_s=total,total_time_complete=self.action_end_time is not None,
                total_time_start_s=start,total_time_end_s=end,
                total_time_start_basis=self.action_start_basis or "record_start_no_stretch_signal",
                total_time_end_basis=self.action_end_basis or "last_observation_return_not_confirmed",
                maximum_contact_force_N=self.maximum_contact,maximum_fixture_or_self_contact_force_N=self.hard_peak,
                **{name+"_episodes":count for name,count in self.counts.items()},
                operation_end_surface_usage={k:dict(v) for k,v in self.surface_usage.items()},
                contact_count_resolution=self.contact_count_resolution,observed_samples=self.sample_count,
                maximum_sample_gap_s=self.maximum_gap),
            config=asdict(c),measurement_coverage=dict(verified=complete and self.contact_evidence and every_step,
                record_complete_at_declared_resolution=complete,contact_evidence_complete=self.contact_evidence,
                physics_dt_s=self.physics_dt_s,observation_dt_s=self.observation_dt_s,
                sampling_coverage=self.sampling_coverage,evidence_sources=list(self.evidence_sources),
                historical_contact_scope_limitations=list(self.historical_contact_scope_limitations),
                reason="环境逐物理步算分" if complete and self.contact_evidence and every_step else "按保存记录的时间分辨率重评分；未恢复缺失的物理步"),
            scoring_notes=["普通开关不限速；用时只记录，90秒为计算截止，不产生超时失败。",
                "只有完成且无硬撞才通过；硬撞独立标记；到位但缓开不合格按用户补充口径单列。",
                "有效拧动计时从工具接触后首次可见正确运动开始；无接触时序的旧记录用turn阶段近似并单列依据。",
                "计时用0.01°防抖量，不参与结果分级；缓开扣除规定5秒等待。"])
        # Diagnostic aliases keep consumers from crashing while they migrate.
        # R4 no longer means a 60-second failure rule in this explicit version.
        for old, current in (("R1","target_hold"),("R2","slow_open_wait"),
                             ("R3","hard_collision"),("R4","elapsed_time")):
            result["rules"][old] = dict(result["rules"][current], alias_for=current)
        return result


def score_record(record, config=None):
    """Re-score preserved evidence without stepping physics or changing labels."""
    sampling = record.get("sampling", {})
    physics_dt = sampling.get("physics_dt_s", .001)
    observation_dt = sampling.get("observation_dt_s", physics_dt)
    coverage = sampling.get("coverage", "unspecified")
    contact_evidence = record.get("contact_evidence_complete", False)
    evidence = record.get("evidence", {})
    contact_evidence |= all(evidence.get(k) is True for k in ("R5", "R7"))
    scorer = OnlineScorer(record.get("variant"),physics_dt_s=physics_dt,
        observation_dt_s=observation_dt,sampling_coverage=coverage,config=config,
        actual_station=record.get("actual_station", "A1"),source=record.get("source"),
        contact_evidence=contact_evidence,expected_complete=sampling.get("complete", coverage == "every_physics_step"))
    for row in record.get("samples", []):
        scorer.observe(row["time_s"],row["opening_deg"],row["speed_deg_s"],row.get("contacts", []),row)
    scorer.add_historical_summary(record.get("old_score_v2") or record.get("preserved_score"))
    # A batch may supply a separately documented geometric exclusion for an
    # environment category absent from its preserved force summary. This is
    # explicit, source-bound evidence and never a default for other scenes.
    if record.get("historical_other_environment_excluded") is True:
        evidence_source = record.get("historical_environment_scope_evidence")
        if not isinstance(evidence_source, str) or not evidence_source.strip():
            raise ValueError("A historical environment exclusion requires its evidence source")
        scorer.historical_contact_scope_limitations = []
        scorer.evidence_sources.append(dict(source="batch_specific_geometric_environment_exclusion",
            evidence=evidence_source,quantities=["operation_end_to_other_environment_unreachable"],
            scope="only_the_explicitly_assessed_historical_batch"))
    result = scorer.result()
    result["data_kind"] = record.get("data_kind", "unspecified")
    result["historical_comparison_only"] = True
    result["original_procedure_version"] = record.get("procedure_version", "unspecified")
    result["runtime_settings"] = record.get("runtime", {})
    return result
