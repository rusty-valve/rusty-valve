"""Controller-independent A1 v1.1/v1.2 scoring. Standard library only.

Angles are opening angles: 0 degrees closed, 90 degrees open. This module never
loads a controller, runs physics, changes a trajectory, or reads legacy success
labels. Missing procedure_version keeps historical v1.1 semantics. Explicit
A1-v1.2 uses adapter-declared body roles; see H0_RECORD_FORMAT.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


A1_DIRECTIONS = {
    "human_position": "front; world -X",
    "back": "away from human; world +X",
    "left": "human's left; world +Y",
    "open": "handle left, parallel to pipe; opening=90deg, hinge_q=0deg",
    "closed": "handle back, perpendicular to pipe; opening=0deg, hinge_q=90deg",
    "conversion": "opening_deg = 90 - hinge_q_deg",
}
VARIANTS = {
    "A1-关": (90., 0.),
    "A1-开": (0., 90.),
    "A1-缓开": (0., 90.),
    "A1-回关": (18., 0.),
    "A1-调小": (45., 18.),
}


@dataclass(frozen=True)
class ScoreConfig:
    angle_tolerance_deg: float = 2.
    final_hold_s: float = 2.
    pause_s: float = 5.
    pause_speed_deg_s: float = 1.
    start_window_deg: float = 10.
    # v1.1 makes this provisional cap effective. None still never means infinity.
    start_speed_limit_deg_s: float | None = 10.
    r4_policy: str = "fail"
    r6_policy: str = "fail"
    fixture_force_limit_N: float = 5.
    timeout_s: float = 60.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name.endswith("policy"):
                if value != "fail":
                    raise ValueError(f"{name} must be fail under A1 procedure v1.1")
            elif value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")


def _rule(status, reason, **metrics):
    return dict(status=status, reason=reason, **metrics)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _procedure_version(record):
    """Old records keep v1.1 semantics; new robot-neutral records opt in."""
    version = record.get("procedure_version", "A1-v1.1")
    if version not in ("A1-v1.1", "A1-v1.2"):
        raise ValueError("procedure_version must be A1-v1.1 or A1-v1.2")
    return version


def _validate(record):
    v12 = _procedure_version(record) == "A1-v1.2"
    if record.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if record.get("variant") not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}")
    rows = record.get("samples")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("samples must contain at least two frames, including time_s=0")
    previous = -1.
    for i, row in enumerate(rows):
        t = _finite(row.get("time_s"), f"samples[{i}].time_s")
        if t < 0 or t <= previous or (i == 0 and t != 0):
            raise ValueError("time_s must start at zero and strictly increase")
        previous = t
        _finite(row.get("opening_deg"), "opening_deg")
        _finite(row.get("speed_deg_s"), "speed_deg_s")
        if not isinstance(row.get("contacts"), list):
            raise ValueError("each sample needs a contacts list; missing is not an empty list")
        for contact in row["contacts"]:
            kinds = ("handle", "fixture", "environment", "self") if v12 else ("handle", "fixture", "idle", "self")
            if contact.get("kind") not in kinds:
                raise ValueError(f"contact kind must be one of {kinds}")
            if _finite(contact.get("force_N"), "contact.force_N") < 0:
                raise ValueError("contact.force_N cannot be negative")
            if v12:
                roles = ("operation_end", "idle_hand", "robot_other")
                if contact.get("robot_part") not in roles:
                    raise ValueError("v1.2 contact robot_part must be operation_end, idle_hand or robot_other")
                if contact["kind"] == "self" and contact.get("other_robot_part") not in roles:
                    raise ValueError("v1.2 self contact needs other_robot_part")
                if "surface" in contact and not isinstance(contact["surface"], str):
                    raise ValueError("v1.2 contact surface must be a string")
            elif contact["kind"] == "handle" and contact.get("part") not in ("orange_push", "blue_pull", "other"):
                raise ValueError("handle contact needs orange_push, blue_pull or other part")
    if v12:
        adapter = record.get("robot_adapter", {})
        if not isinstance(adapter, dict) or not isinstance(adapter.get("has_idle_hand"), bool):
            raise ValueError("v1.2 robot_adapter.has_idle_hand must be a boolean")
    evidence = record.get("evidence", {})
    for name in ("R4", "R5", "R6", "R7"):
        if name in evidence and not isinstance(evidence[name], bool):
            raise ValueError(f"evidence.{name} must be a boolean")


def _runs(rows, predicate, earliest=0.):
    """Conservative observed durations: do not interpolate unobserved crossings."""
    start = None
    runs = []
    for row in rows:
        if row["time_s"] < earliest - 1e-9:
            continue
        if predicate(row):
            if start is None:
                start = row["time_s"]
        elif start is not None:
            runs.append((start, previous))
            start = None
        previous = row["time_s"]
    if start is not None:
        runs.append((start, previous))
    return runs


def _episodes(active):
    return sum(value and (i == 0 or not active[i-1]) for i, value in enumerate(active))


def _measurement_coverage(record, rows):
    """v1.1 requires physics-step measurements, not endpoint-only samples.

    A complete trace is simple to verify and does not depend on the controller.
    Interval-maximum writers may be added in a later format; an unsupported
    declaration is kept unverified rather than guessed to be complete.
    """
    sampling = record.get("sampling", {})
    if sampling.get("coverage") != "every_physics_step":
        return False, "缺少每物理步完整测量；控制步末尾采样不能排除帧间违规"
    dt = sampling.get("physics_dt_s")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        return False, "缺少有效的物理步时长"
    if any(abs((b["time_s"] - a["time_s"]) - dt) > 1e-7 for a, b in zip(rows, rows[1:])):
        return False, "每物理步测量有缺帧或时间间隔与物理步不符"
    return True, "含重置时刻与每个物理步的速度和完整接触分类"


def score_record(record: dict, config: ScoreConfig | None = None) -> dict:
    """Return pass/fail plus each rule's measurements, including unknown evidence."""
    config = config or ScoreConfig()
    _validate(record)
    procedure_version = _procedure_version(record)
    v12 = procedure_version == "A1-v1.2"
    rows = record["samples"]
    coverage_ok, coverage_reason = _measurement_coverage(record, rows)
    variant = record["variant"]
    initial, target = VARIANTS[variant]
    tol = config.angle_tolerance_deg
    rules = {}
    initial_ok = abs(rows[0]["opening_deg"] - initial) <= tol + 1e-9
    rules["setup"] = _rule("pass" if initial_ok else "fail", "起点符合题目" if initial_ok else "起点与题目不符",
                           expected_initial_opening_deg=initial, actual_initial_opening_deg=rows[0]["opening_deg"])
    near_target = lambda row: abs(row["opening_deg"] - target) <= tol + 1e-9
    target_runs = _runs(rows, near_target)
    terminal = target_runs[-1] if target_runs and near_target(rows[-1]) else None
    hold = terminal[1] - terminal[0] if terminal else 0.
    r1_ok = hold >= config.final_hold_s - 1e-9
    completion = terminal[0] + config.final_hold_s if r1_ok else None
    rules["R1"] = _rule("pass" if r1_ok else "fail", "终点保持足够" if r1_ok else "未到终点或连续保持不足",
                        target_opening_deg=target, final_opening_deg=rows[-1]["opening_deg"],
                        terminal_hold_s=hold, required_hold_s=config.final_hold_s,
                        completion_time_s=completion)

    pauses = []
    if variant == "A1-缓开":
        cursor = 0.
        for angle in (18., 45.):
            predicate = lambda row, a=angle: (abs(row["opening_deg"] - a) <= tol + 1e-9
                                              and abs(row["speed_deg_s"]) < config.pause_speed_deg_s)
            candidates = _runs(rows, predicate, cursor)
            good = next(((start, end) for start, end in candidates
                         if end - start >= config.pause_s - 1e-9), None)
            crossed = next((row["time_s"] for row in rows
                            if row["time_s"] >= cursor - 1e-9 and row["opening_deg"] > angle + tol + 1e-9), None)
            ok = bool(good and (crossed is None or crossed >= good[0] + config.pause_s - 1e-9))
            pauses.append(dict(target_opening_deg=angle, passed=ok,
                               longest_hold_s=max((end-start for start, end in candidates), default=0.),
                               completed_time_s=good[0] + config.pause_s if ok else None))
            if not ok:
                break
            cursor = good[0] + config.pause_s
        r2_ok = len(pauses) == 2 and all(p["passed"] for p in pauses)
        if r2_ok and completion is not None:
            r2_ok = completion >= cursor
        rules["R2"] = _rule("pass" if r2_ok else "fail", "按顺序停稳" if r2_ok else "18°、45°未按顺序各停稳5秒",
                            pauses=pauses)
    elif variant == "A1-调小":
        stopped_runs = _runs(rows, lambda row: near_target(row) and abs(row["speed_deg_s"]) < config.pause_speed_deg_s)
        stopped = stopped_runs[-1] if stopped_runs and stopped_runs[-1][1] == rows[-1]["time_s"] else None
        stopped_hold = stopped[1] - stopped[0] if stopped else 0.
        r2_ok = stopped_hold >= config.final_hold_s - 1e-9
        if r2_ok and completion is not None:
            completion = max(completion, stopped[0] + config.final_hold_s)
        rules["R2"] = _rule("pass" if r2_ok else "fail", "调小后停稳" if r2_ok else "调小后未停稳",
                            stopped_hold_s=stopped_hold)
    else:
        rules["R2"] = _rule("not_applicable", "此变体没有中途停顿点")

    opening = variant in ("A1-开", "A1-缓开")
    # Only speeds actually measured inside the first ten degrees belong to R3.
    # In particular, later acceleration just beyond 10 degrees is not an R3
    # violation. A sparse trace crossing that boundary is unverified, not safe.
    early_indices = set()
    early_interval_speeds = []
    if opening:
        window_end = initial + config.start_window_deg
        early_indices = {i for i, row in enumerate(rows)
                         if row["opening_deg"] <= window_end + 1e-9}
        for i in range(1, len(rows)):
            a, b = rows[i-1]["opening_deg"], rows[i]["opening_deg"]
            if max(a, b) <= window_end + 1e-9:
                early_interval_speeds.append(abs(b-a) / (rows[i]["time_s"] - rows[i-1]["time_s"]))
    early_point_peak = max((abs(rows[i]["speed_deg_s"]) for i in early_indices), default=0.)
    early_interval_peak = max(early_interval_speeds, default=0.)
    early_peak = max(early_point_peak, early_interval_peak)
    skipped_start_window = opening and (
        any(row["opening_deg"] > initial + config.start_window_deg for row in rows)
        and not any(initial + 1e-9 < row["opening_deg"] <= initial + config.start_window_deg
                    for row in rows))
    if not opening:
        rules["R3"] = _rule("not_applicable", "关阀类变体不考开阀起步速度")
    elif config.start_speed_limit_deg_s is None:
        rules["R3"] = _rule("unknown", "起步速度上限被设为空，不能默认通过",
                            start_peak_speed_deg_s=early_peak, limit_deg_s=None,
                            recorded_point_peak_deg_s=early_point_peak,
                            interval_average_peak_deg_s=early_interval_peak)
    else:
        r3_ok = early_peak <= config.start_speed_limit_deg_s + 1e-9
        r3_covered = coverage_ok and not skipped_start_window
        status = "fail" if not r3_ok else "pass" if r3_covered else "unknown"
        reason = ("开阀前10°速度超限" if not r3_ok else "起步速度合格" if r3_covered
                  else "记录直接跨过起步10°，缺少范围内速度证据" if skipped_start_window else coverage_reason)
        rules["R3"] = _rule(status, reason,
                            start_peak_speed_deg_s=early_peak, limit_deg_s=config.start_speed_limit_deg_s,
                            recorded_point_peak_deg_s=early_point_peak,
                            interval_average_peak_deg_s=early_interval_peak)

    detections = {name: [] for name in ("R4", "R5", "R6", "R7")}
    face_names = (sorted({contact.get("surface", "unspecified")
                         for row in rows for contact in row["contacts"]
                         if contact["kind"] == "handle" and contact.get("robot_part") == "operation_end"})
                  if v12 else ("orange_push", "blue_pull", "other"))
    face_flags = {part: [] for part in face_names}
    face_peak_forces = {part: 0. for part in face_flags}
    peaks = {name: 0. for name in detections}
    maximum_contact = 0.
    for row in rows:
        flags = {name: False for name in detections}
        faces = {part: False for part in face_flags}
        for contact in row["contacts"]:
            kind, force = contact["kind"], contact["force_N"]
            maximum_contact = max(maximum_contact, force)
            if force <= 0:
                continue
            if kind == "handle" and (not v12 or contact["robot_part"] == "operation_end"):
                part = contact.get("surface", "unspecified") if v12 else contact["part"]
                faces[part] = True
                face_peak_forces[part] = max(face_peak_forces[part], force)
            violated = []
            valid_handle_part = (contact.get("robot_part") == "operation_end" if v12
                                 else contact.get("part") in ("orange_push", "blue_pull"))
            if kind == "handle" and not valid_handle_part:
                violated.append("R4")
            if kind == "fixture" and force > config.fixture_force_limit_N:
                violated.append("R5")
            idle_contact = (contact.get("robot_part") == "idle_hand" or contact.get("other_robot_part") == "idle_hand"
                            if v12 else kind == "idle" or contact.get("arm") == "idle")
            if idle_contact:
                violated.append("R6")
            if kind == "self":
                violated.append("R7")
            for name in violated:
                flags[name] = True
                peaks[name] = max(peaks[name], force)
        for name in detections:
            detections[name].append(flags[name])
        for part in face_flags:
            face_flags[part].append(faces[part])
    evidence = record.get("evidence", {})
    labels = {"R4": "S爪工具面以外的部位碰手柄", "R5": "碰管道、支架或阀体超过限值", "R6": "闲置手接触物体", "R7": "自身部件接触"}
    if v12:
        labels["R4"] = "适配文件声明的操作端以外部件碰手柄"
        labels["R6"] = "适配文件声明的另一只手接触物体"
    violation_counts = {}
    for name, flags in detections.items():
        count = _episodes(flags)
        violation_counts[name] = count
        policy = config.r4_policy if name == "R4" else config.r6_policy if name == "R6" else "fail"
        status = "fail" if count else "pass"
        reason = labels[name] if count else "未发现违规接触"
        if v12 and name == "R6" and not record["robot_adapter"]["has_idle_hand"] and not count:
            status, reason = "not_applicable", "适配文件声明为无另一只手的机器人"
        elif not count and not evidence.get(name, False):
            status, reason = "unknown", "缺少该项完整接触记录，不能把缺失当作零次"
        elif not count and not coverage_ok:
            status, reason = "unknown", coverage_reason
        rules[name] = _rule(status, reason, violation_episodes=count, violating_samples=sum(flags),
                            peak_violation_contact_N=peaks[name], policy=policy)

    runtime = record.get("runtime", {})
    runtime_values = [runtime.get("motor_limits_unchanged"), runtime.get("valve_contact_only")]
    r8_status = "fail" if any(v is False for v in runtime_values) else "pass" if all(v is True for v in runtime_values) else "unknown"
    rules["R8"] = _rule(r8_status, "由运行设置保证" if r8_status == "pass" else "运行设置违规或缺少保证",
                        evidence=runtime.get("evidence", "未提供运行设置依据"))

    task_completed = r1_ok and rules["R2"]["status"] in ("pass", "not_applicable")
    if task_completed:
        r9_ok = completion <= config.timeout_s + 1e-9
        rules["R9"] = _rule("pass" if r9_ok else "fail", "在限时内完成" if r9_ok else "完成时已超时",
                            completion_time_s=completion, limit_s=config.timeout_s)
    elif rows[-1]["time_s"] >= config.timeout_s - 1e-9:
        rules["R9"] = _rule("fail", "超过限时未完成", limit_s=config.timeout_s)
    else:
        rules["R9"] = _rule("pass", "记录提前结束，尚未到超时线；到位另按R1、R2判断", limit_s=config.timeout_s)

    failed = [name for name, item in rules.items() if item["status"] == "fail"]
    unknown = [name for name, item in rules.items() if item["status"] == "unknown"]
    passed = not failed and not unknown
    result = {
        "schema_version": 1, "procedure_version": procedure_version, "variant": variant, "source": record.get("source"),
        "data_kind": record.get("data_kind", "unspecified"),
        "passed": passed, "verdict": "通过" if passed else "未通过",
        "failed_rules": failed, "unverified_rules": unknown,
        "rules": rules, "config": asdict(config), "directions": A1_DIRECTIONS,
        "metrics": {
            "record_duration_s": rows[-1]["time_s"], "completion_time_s": completion if task_completed else None,
            "start_peak_speed_deg_s": early_peak if opening else None,
            "maximum_contact_force_N": maximum_contact, "violation_episodes": violation_counts,
            ("operation_end_surface_usage" if v12 else "tool_face_usage"): {part: {
                "contact_samples": sum(flags), "contact_episodes": _episodes(flags),
                "maximum_contact_force_N": face_peak_forces[part],
            } for part, flags in face_flags.items()},
            "maximum_sample_gap_s": max(b["time_s"] - a["time_s"] for a, b in zip(rows, rows[1:])),
        },
        "limitations": list(record.get("limitations", [])),
        "measurement_coverage": {"verified": coverage_ok, "reason": coverage_reason},
        "pending_card_decisions": [],
        "provisional_card_values": ["R3起步限速10°/s", "R4操作端以外接触判失败" if v12 else "R4工具面以外接触判失败", "R6禁止另一只手接触"],
    }
    if v12:
        result["robot_adapter"] = record["robot_adapter"]
        result["actual_station"] = record.get("actual_station", "A1")
        result["assessment_scope"] = record.get("assessment_scope", "A1-v1.2")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-speed-limit", type=float, default=10., help="v1.1 default: 10 degrees/s")
    args = parser.parse_args()
    config = ScoreConfig(start_speed_limit_deg_s=args.start_speed_limit)
    result = score_record(json.loads(args.record.read_text(encoding="utf-8")), config)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
