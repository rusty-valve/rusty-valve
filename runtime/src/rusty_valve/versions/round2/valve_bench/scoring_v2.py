"""A1 v2.0: four rules, computed online from environment observations.

No controller, trajectory writer or robot-specific joint layout is required.
``score_record`` exists only to compare preserved historical evidence. Missing
historical measurements produce ``passed=None`` (unknown), never a fake fail.
Angles are opening angles: 0 closed, 90 open, normalized by the station.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

VARIANTS = {"A1-关": (90., 0.), "A1-开": (0., 90.), "A1-缓开": (0., 90.)}
ALIASES = {"close": "A1-关", "open": "A1-开", "slow_open": "A1-缓开"}


@dataclass(frozen=True)
class ScoreConfig:
    angle_tolerance_deg: float = 3.
    final_hold_s: float = 2.
    start_window_deg: float = 10.
    start_speed_limit_deg_s: float = 10.
    pause_angle_deg: float = 18.
    pause_s: float = 1.
    pause_speed_deg_s: float = 2.
    hard_contact_limit_N: float = 20.
    timeout_s: float = 60.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _rule(status, reason, **metrics):
    return dict(status=status, reason=reason, **metrics)


def normalize_contact(contact):
    """Accept the preserved v1.1 and robot-neutral v1.2/v2 contact layouts."""
    row = dict(contact)
    kind = row.get("kind")
    if kind not in ("handle", "fixture", "self", "environment", "idle"):
        raise ValueError(f"Unknown contact kind: {kind}")
    force = _finite(row.get("force_N"), "contact.force_N")
    if force < 0:
        raise ValueError("Contact force cannot be negative")
    if "robot_part" not in row:
        row["robot_part"] = ("idle_hand" if row.get("arm") == "idle" or kind == "idle"
                             else "operation_end" if kind == "handle" and row.get("part") in ("orange_push", "blue_pull")
                             else "robot_other")
    if row["robot_part"] not in ("idle_hand", "operation_end", "robot_other"):
        raise ValueError("Contact robot_part must name an adapter role")
    row["surface"] = row.get("surface", row.get("part", "unspecified"))
    row["force_N"] = force
    return row


class OnlineScorer:
    """Constant-memory scorer: observe reset and each solved physical step.

    Contacts are per-contact solved forces, not summed forces over a body.
    A contact episode starts when a geometry pair becomes active. Historical
    records without geometry IDs use kind/role/surface groups, so their episode
    counts are coarser. R1 uses exactly the card's continuous angle-band test;
    no extra velocity or end-stop reaction threshold is imposed.
    """
    def __init__(self, variant, *, physics_dt_s, config=None, actual_station="A1",
                 source=None, contact_evidence=True, expected_complete=True):
        self.variant = ALIASES.get(variant, variant)
        if self.variant not in VARIANTS:
            raise ValueError("v2.0 only assesses close, open and slow_open")
        self.config = config or ScoreConfig()
        self.physics_dt_s = _finite(physics_dt_s, "physics_dt_s")
        if self.physics_dt_s <= 0:
            raise ValueError("physics_dt_s must be positive")
        self.station, self.source = str(actual_station), source
        self.coverage_ok = bool(expected_complete)
        self.contact_evidence = bool(contact_evidence)
        self.sample_count = 0
        self.last_time = None
        self.initial_opening = self.final_opening = None
        self.target_start = self.pause_start = None
        self.longest_target_hold = self.longest_pause = 0.
        self.completion_time = self.pause_completion = None
        self.crossed_pause_before_stopping = False
        self.start_peak = self.maximum_contact = self.hard_peak = 0.
        self.maximum_gap = 0.
        self.counts = {name: 0 for name in ("light_contact", "non_operation_end_handle", "idle_hand_contact", "hard_contact")}
        self._active = {name: set() for name in self.counts}
        self.surface_usage = {}
        self._surface_active = set()
        self.contact_count_resolution = "geometry_pairs"

    def observe(self, time_s, opening_deg, speed_deg_s, contacts):
        t, angle, speed = (_finite(time_s, "time_s"), _finite(opening_deg, "opening_deg"),
                           _finite(speed_deg_s, "speed_deg_s"))
        if not isinstance(contacts, list):
            raise ValueError("contacts must be a list (empty when no contact)")
        if self.last_time is None:
            if abs(t) > 1e-9:
                self.coverage_ok = False
            self.initial_opening = angle
        else:
            dt = t-self.last_time
            if dt <= 0:
                raise ValueError("Observation times must strictly increase")
            self.maximum_gap = max(self.maximum_gap, dt)
            if abs(dt-self.physics_dt_s) > 1e-7:
                self.coverage_ok = False
                # A missing interval cannot extend an observed hold.
                self.target_start = self.pause_start = None
        self.sample_count += 1
        self.last_time, self.final_opening = t, angle
        c = self.config
        target = VARIANTS[self.variant][1]
        if abs(angle-target) <= c.angle_tolerance_deg+1e-9:
            if self.target_start is None:
                self.target_start = t
            hold = t-self.target_start
            self.longest_target_hold = max(self.longest_target_hold, hold)
            if hold >= c.final_hold_s-1e-9 and self.completion_time is None:
                self.completion_time = self.target_start+c.final_hold_s
        else:
            self.target_start = None
        if self.variant != "A1-关" and angle <= c.start_window_deg+1e-9:
            self.start_peak = max(self.start_peak, abs(speed))
        if self.variant == "A1-缓开":
            if abs(angle-c.pause_angle_deg) <= c.angle_tolerance_deg+1e-9 and abs(speed) < c.pause_speed_deg_s:
                if self.pause_start is None:
                    self.pause_start = t
                held = t-self.pause_start
                self.longest_pause = max(self.longest_pause, held)
                if held >= c.pause_s-1e-9 and self.pause_completion is None and not self.crossed_pause_before_stopping:
                    self.pause_completion = self.pause_start+c.pause_s
            else:
                self.pause_start = None
            if angle > c.pause_angle_deg+c.angle_tolerance_deg+1e-9 and self.pause_completion is None:
                self.crossed_pause_before_stopping = True
        active = {name: set() for name in self.counts}
        surfaces = set()
        for raw in contacts:
            row = normalize_contact(raw)
            force, kind, part = row["force_N"], row["kind"], row["robot_part"]
            self.maximum_contact = max(self.maximum_contact, force)
            if force <= 0:
                continue
            if "robot_geom" in row and "other_geom" in row:
                pair = tuple(sorted((str(row["robot_geom"]), str(row["other_geom"]))))
            else:
                pair = (kind, part, row["surface"], row.get("other_robot_part", ""))
                self.contact_count_resolution = "legacy_kind_role_surface_groups"
            if kind in ("fixture", "self"):
                self.hard_peak = max(self.hard_peak, force)
                active["hard_contact" if force > c.hard_contact_limit_N else "light_contact"].add(pair)
            if kind == "handle" and part != "operation_end":
                active["non_operation_end_handle"].add(pair)
            if part == "idle_hand" or row.get("other_robot_part") == "idle_hand":
                active["idle_hand_contact"].add(pair)
            if kind == "handle" and part == "operation_end":
                face = row["surface"]
                surfaces.add(face)
                usage = self.surface_usage.setdefault(face, dict(contact_samples=0, contact_episodes=0, maximum_contact_force_N=0.))
                usage["maximum_contact_force_N"] = max(usage["maximum_contact_force_N"], force)
        for name in self.counts:
            self.counts[name] += len(active[name]-self._active[name])
        self._active = active
        for face in surfaces:
            self.surface_usage[face]["contact_samples"] += 1
            self.surface_usage[face]["contact_episodes"] += face not in self._surface_active
        self._surface_active = surfaces

    def result(self):
        c = self.config
        ran = self.sample_count >= 2
        complete = ran and self.coverage_ok
        hold_ok = self.completion_time is not None
        rules = {}
        rules["R1"] = _rule("pass" if hold_ok else "fail" if complete else "unknown",
            "已在终点±3°内连续保持2秒" if hold_ok else "未到位或连续保持不足" if complete else "旧记录不足以判断到位保持",
            target_opening_deg=VARIANTS[self.variant][1], final_opening_deg=self.final_opening,
            longest_hold_s=self.longest_target_hold, completion_time_s=self.completion_time)
        if self.variant == "A1-关":
            rules["R2"] = _rule("not_applicable", "关阀不考开启起步速度和18°停顿")
        else:
            too_fast = self.start_peak > c.start_speed_limit_deg_s+1e-9
            pause_failed = self.variant == "A1-缓开" and self.pause_completion is None
            status = ("fail" if too_fast or (complete and pause_failed) else "pass" if complete else "unknown")
            reason = ("开启前10°超过10°/秒" if too_fast else "18°±3°处未先停满1秒" if complete and pause_failed
                      else "起步及所需停顿合格" if complete else "旧记录不足以排除起步超速或判断停顿")
            rules["R2"] = _rule(status, reason, start_peak_speed_deg_s=self.start_peak,
                pause_required=self.variant == "A1-缓开", pause_completed_time_s=self.pause_completion,
                longest_pause_s=self.longest_pause)
        hard = self.counts["hard_contact"] > 0
        contact_complete = complete and self.contact_evidence
        rules["R3"] = _rule("fail" if hard else "pass" if contact_complete else "unknown",
            "发生超过20 N的管道/支架/阀体或自碰" if hard else "没有超过20 N的硬撞" if contact_complete else "旧记录缺少完整碰撞测量，无法判断",
            hard_contact_episodes=self.counts["hard_contact"], maximum_fixture_or_self_contact_force_N=self.hard_peak)
        in_time = hold_ok and self.completion_time <= c.timeout_s+1e-9
        rules["R4"] = _rule("pass" if in_time else "fail" if complete else "unknown",
            "60秒内满足到位保持" if in_time else "本回合未在60秒内满足到位保持" if complete else "旧记录不足以判断完成时间",
            completion_time_s=self.completion_time, limit_s=c.timeout_s)
        failed = [k for k,v in rules.items() if v["status"] == "fail"]
        unknown = [k for k,v in rules.items() if v["status"] == "unknown"]
        passed = False if failed else None if unknown else True
        return dict(schema_version=2, procedure_version="A1-v2.0", variant=self.variant,
            actual_station=self.station, assessment_scope="A1正式规程v2.0" if self.station == "A1" else "暂用A1 v2.0通用条款，非正式新工位验收",
            source=self.source, data_kind="recorded_simulation", passed=passed,
            verdict="通过" if passed is True else "未通过" if passed is False else "无法判断",
            failed_rules=failed, unverified_rules=unknown, rules=rules, config=asdict(c),
            metrics=dict(record_duration_s=self.last_time, completion_time_s=self.completion_time,
                initial_opening_deg=self.initial_opening, start_peak_speed_deg_s=self.start_peak if self.variant != "A1-关" else None,
                maximum_contact_force_N=self.maximum_contact, maximum_fixture_or_self_contact_force_N=self.hard_peak,
                **{name+"_episodes": count for name,count in self.counts.items()},
                operation_end_surface_usage={k:dict(v) for k,v in self.surface_usage.items()},
                contact_count_resolution=self.contact_count_resolution, observed_samples=self.sample_count,
                maximum_sample_gap_s=self.maximum_gap),
            measurement_coverage=dict(verified=complete and self.contact_evidence,
                reason="环境逐物理步算分" if complete and self.contact_evidence else "旧记录不足；缺测不作为失败条款"),
            scoring_notes=["只有R1–R4决定成绩；轻碰、非操作端碰手柄、闲置手接触仅记分。",
                           "模型力量上限与阀门仅靠接触带动由环境设置保证，不另设考试条目。"])


def score_record(record, config=None):
    """Read-only v2 comparison of historical records; never change old labels."""
    sampling = record.get("sampling", {})
    dt = sampling.get("physics_dt_s")
    dt_valid = isinstance(dt, (float, int)) and not isinstance(dt, bool) and math.isfinite(dt) and dt > 0
    evidence = record.get("evidence", {})
    contact_evidence = all(evidence.get(k) is True for k in ("R5", "R7"))
    if record.get("procedure_version") == "A1-v2.0":
        contact_evidence = record.get("contact_evidence_complete", False)
    scorer = OnlineScorer(record.get("variant"), physics_dt_s=dt if dt_valid else 1., config=config,
        actual_station=record.get("actual_station", "A1"), source=record.get("source"),
        contact_evidence=contact_evidence,
        expected_complete=dt_valid and sampling.get("coverage") == "every_physics_step")
    rows = record.get("samples", [])
    for row in rows:
        scorer.observe(row["time_s"], row["opening_deg"], row["speed_deg_s"], row.get("contacts", []))
        if "contacts" not in row:
            scorer.contact_evidence = False
    result = scorer.result()
    result["data_kind"] = record.get("data_kind", "unspecified")
    result["measurement_coverage"]["reason"] = ("历史逐物理步记录完整，可作新旧对比" if result["measurement_coverage"]["verified"]
                                                 else "历史记录缺测或未执行物理步；相关规则无法判断")
    result["historical_comparison_only"] = True
    result["original_procedure_version"] = record.get("procedure_version", "A1-v1.1")
    result["runtime_settings"] = record.get("runtime", {})
    return result
