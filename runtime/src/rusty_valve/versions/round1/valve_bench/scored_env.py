"""Historical run scoring with optional lossless solved-step evidence.

Wrap a H0/H1 environment before reset, or after preparation at t=0. The wrapper
observes each solved physical step through the environment callback. Controllers
only send normal actions. ``score()`` returns the original end-of-episode
report. Public Exam v1 scores the recorded evidence separately through
``rusty_valve.public_exam_v1``.
"""
from __future__ import annotations

import json
from pathlib import Path

from .scoring_v25 import OnlineScorer
from .scoring_v24 import OnlineScorer as OnlineScorerV24
from .scoring_v23 import OnlineScorer as OnlineScorerV23
from .scoring_v21 import OnlineScorer as OnlineScorerV21
from .scoring_v22 import OnlineScorer as OnlineScorerV22
from .scoring_v2 import OnlineScorer as OnlineScorerV20


class ScoredValveEnv:
    def __init__(self, env, variant=None, *, source=None, procedure_version="A1-v2.5"):
        self.env = env
        self.variant = variant or env.variant
        self.source = source
        if procedure_version not in ("A1-v2.0", "A1-v2.1", "A1-v2.2", "A1-v2.3", "A1-v2.4", "A1-v2.5"):
            raise ValueError("Unsupported procedure version")
        self.procedure_version = procedure_version
        self.scoring_context = {}
        # v2.1's 90 seconds is a computation cutoff, not a failure rule.
        if procedure_version in ("A1-v2.1", "A1-v2.2", "A1-v2.3", "A1-v2.4", "A1-v2.5"):
            env.max_time_s = 90.
        self._previous_observer = env.physics_observer
        self._closed = False
        self._evidence_writer = None
        self.scorer = None
        self._reset_score()
        if getattr(env, "_started", False) and abs(float(env.data.time)) > 1e-9:
            raise ValueError("Attach scoring before physical execution, not halfway through an episode")
        env.physics_observer = self._observe
        if getattr(env, "_started", False):
            self._observe(env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _reset_score(self):
        scorer_class = {"A1-v2.0":OnlineScorerV20,"A1-v2.1":OnlineScorerV21,
                        "A1-v2.2":OnlineScorerV22,"A1-v2.3":OnlineScorerV23,
                        "A1-v2.4":OnlineScorerV24,"A1-v2.5":OnlineScorer}[self.procedure_version]
        extra={}
        if self.procedure_version in ("A1-v2.2", "A1-v2.3", "A1-v2.4", "A1-v2.5"):
            adapter=getattr(self.env,"robot_adapter",{})
            adapter=adapter.to_dict() if hasattr(adapter,"to_dict") else adapter
            self._s_tool=bool(isinstance(adapter,dict) and adapter.get("tool_adapter")=="s_hook")
            self._face_provider_available=self._s_tool and all(hasattr(self.env,key) for key in
                ("active_hook_body_id","valve_joint_id","valve_dof_index","q_close_rad","q_open_rad","task"))
            extra=dict(face_constraint_applicable=self._s_tool,face_evidence_complete=self._face_provider_available)
        self.scorer = scorer_class(self.variant, physics_dt_s=self.env.physics_dt,
                                   actual_station=self.env.station, source=self.source,**extra)

    def _observe(self, env):
        if self.procedure_version in ("A1-v2.2", "A1-v2.3", "A1-v2.4", "A1-v2.5") and self._face_provider_available:
            from .h0_records import add_s_tool_face_evidence
            env.current_contacts=add_s_tool_face_evidence(env,env.current_contacts,
                allow_working_hole_rims=self.procedure_version == 'A1-v2.5')
        if self._previous_observer:
            self._previous_observer(env)
        if abs(float(env.data.time)) <= 1e-9:
            self._reset_score()
            self.scoring_context = {}
            self._check_runtime_settings()
        scale = 90./(env.q_open-env.q_close)
        args = (float(env.data.time), (env.valve_angle_rad-env.q_close)*scale,
                env.valve_speed_rad_s*scale, env.current_contacts)
        if self.procedure_version in ("A1-v2.1", "A1-v2.2", "A1-v2.3", "A1-v2.4", "A1-v2.5"):
            self.scorer.observe(*args, metadata=self.scoring_context)
        else:
            self.scorer.observe(*args)
        if self._evidence_writer is not None:
            self._evidence_writer.observe(*args, metadata=self.scoring_context)

    def start_evidence_recording(self, path):
        """Call after zero-time controller preparation, before physical steps.

        Preparation may reset at t=0 more than once. Its final current sample
        starts this log; no preparation reset is fabricated as a physical step.
        """
        if self._evidence_writer is not None:
            raise RuntimeError('An evidence log is already attached to this episode')
        if abs(float(self.env.data.time)) > 1e-9:
            raise RuntimeError('Full evidence recording must start before physical time advances')
        from .score_evidence import EvidenceWriter
        self._evidence_writer = EvidenceWriter(path, variant=self.variant, station=self.env.station,
            physics_dt_s=self.env.physics_dt, procedure_version=self.procedure_version,
            face_constraint_applicable=getattr(self.scorer, 'face_constraint_applicable', False),
            face_evidence_complete=getattr(self.scorer, 'face_evidence_complete', False),
            config=self.scorer.config, source=self.source)
        scale = 90./(self.env.q_open-self.env.q_close)
        self._evidence_writer.observe(float(self.env.data.time),
            (self.env.valve_angle_rad-self.env.q_close)*scale,
            self.env.valve_speed_rad_s*scale, self.env.current_contacts,
            metadata=self.scoring_context)

    def finish_evidence_recording(self, *, execution_complete=True):
        """Close with explicit execution status; incomplete logs never pass."""
        if self._evidence_writer is None:
            return None
        return self._evidence_writer.finish(terminal_time_s=float(self.env.data.time),
            execution_complete=execution_complete, score=self.score())

    def _check_runtime_settings(self):
        runtime = self.env.runtime_guarantees()
        if not all(runtime.get(key) is True for key in ("motor_limits_unchanged", "valve_contact_only")):
            raise RuntimeError("Invalid simulation setup: model motor limits or passive contact-only valve changed")
        return runtime

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def set_scoring_context(self, **metadata):
        """Optional timestamps/phases; physical observations remain env-owned."""
        self.scoring_context = dict(metadata)

    def step(self, action):
        info = self.env.step(action)
        # Historical contact flags remain diagnostics, never v2 fail gates.
        info["v2_hard_contact"] = self.scorer.counts["hard_contact"] > 0
        info["v2_goal_hold_reached"] = self.scorer.completion_time is not None
        info["procedure_version"] = self.procedure_version
        info["calculation_cutoff_reached"] = float(self.env.data.time) >= getattr(self.env,"max_time_s",float("inf"))-1e-9
        return info

    def score(self):
        # A later consumer can replace a callback accidentally. Do not issue
        # a passing score from an earlier prefix after physics has advanced.
        # This is incomplete measurement, never another exam failure rule.
        observation_current = (self.scorer.last_time is not None
            and abs(self.scorer.last_time-float(self.env.data.time)) <= 1e-7)
        if not observation_current:
            self.scorer.coverage_ok = False
        result = self.scorer.result()
        if not observation_current:
            result["measurement_coverage"]["reason"] = "环境计分回调漏掉末尾物理步，相关项目无法判断"
            result["environment_time_s"] = float(self.env.data.time)
        result["runtime_settings"] = self._check_runtime_settings()
        result["scoring_owner"] = "environment_every_physics_step"
        return result

    def write_score(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(self.score(), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return path

    def detach(self):
        if not self._closed and self.env.physics_observer == self._observe:
            self.env.physics_observer = self._previous_observer
        self._closed = True

    def close(self):
        if self._evidence_writer is not None and not self._evidence_writer.closed:
            self.finish_evidence_recording(execution_complete=False)
        self.detach()
        self.env.close()
