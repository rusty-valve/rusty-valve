"""Stage-scoped S-tool orientation relative to a reviewed G1 reference.

The model's fixed S-profile up axis is local +Y; its wrist-to-tip shaft is
local +Z.  A shortest swing aligns the actual shaft with the reference shaft
before their profile directions are compared.  Normal pitch/tilt is therefore
separate from turning the S profile onto its opposite side.  No colour-region
centre heights or horizontal-tool requirement are used, and no pose is changed.

This is a motion-candidate guard, not a task scoring rule.  The caller supplies
the current stage, active hand and reviewed reference explicitly.
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

POLICY_VERSION = 'reviewed_tool_profile_side_v2_stage_scoped'
SCOPES = frozenset(('rest', 'transition', 'preparation', 'work'))
# Numerical ambiguity guards, not additional posture-angle admission limits.
_SIDE_NUMERICAL_EPS = 1e-7
_ANTIPARALLEL_NUMERICAL_EPS = 1e-10
_ROTATION_NUMERICAL_EPS = 1e-6


def _rotation(value):
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if value.shape != (3, 3) or not np.isfinite(value).all():
        return None
    if not np.allclose(value.T @ value, np.eye(3), atol=_ROTATION_NUMERICAL_EPS, rtol=0.):
        return None
    if not math.isclose(float(np.linalg.det(value)), 1., abs_tol=_ROTATION_NUMERICAL_EPS, rel_tol=0.):
        return None
    return value


def _profile_side(actual_value, reference_value):
    result = dict(same_working_side=None, orientation_class='needs_review',
        needs_review=True, needs_adjustment=False,
        working_side_alignment_cos=None, shaft_alignment_cos=None,
        twist_about_shaft_deg=None)
    if reference_value is None:
        return dict(result, reason='missing_reviewed_reference_rotation')
    reference = _rotation(reference_value)
    if reference is None:
        return dict(result, reason='invalid_reviewed_reference_rotation')
    actual = _rotation(actual_value)
    if actual is None:
        return dict(result, reason='invalid_actual_tool_rotation')

    shaft = actual[:, 2] / np.linalg.norm(actual[:, 2])
    reference_shaft = reference[:, 2] / np.linalg.norm(reference[:, 2])
    cosine = float(np.clip(shaft @ reference_shaft, -1., 1.))
    result.update(shaft_alignment_cos=cosine,
        actual_shaft_world=shaft.tolist(), reference_shaft_world=reference_shaft.tolist())
    if 1. + cosine <= _ANTIPARALLEL_NUMERICAL_EPS:
        return dict(result, reason='opposite_shafts_shortest_swing_undefined')

    cross = np.cross(shaft, reference_shaft)
    skew = np.array([[0., -cross[2], cross[1]],
                     [cross[2], 0., -cross[0]],
                     [-cross[1], cross[0], 0.]])
    swing = np.eye(3) + skew + skew @ skew / (1. + cosine)
    actual_up = swing @ actual[:, 1]
    reference_up = reference[:, 1].copy()
    # Remove only numerical residual components along the aligned shaft.
    actual_up -= reference_shaft * (actual_up @ reference_shaft)
    reference_up -= reference_shaft * (reference_up @ reference_shaft)
    actual_up /= np.linalg.norm(actual_up)
    reference_up /= np.linalg.norm(reference_up)
    side_cosine = float(np.clip(actual_up @ reference_up, -1., 1.))
    twist_sine = float(reference_shaft @ np.cross(reference_up, actual_up))
    result.update(working_side_alignment_cos=side_cosine,
        twist_about_shaft_deg=math.degrees(math.atan2(twist_sine, side_cosine)),
        swing_aligned_profile_up_world=actual_up.tolist(),
        reference_profile_up_world=reference_up.tolist())
    if abs(side_cosine) <= _SIDE_NUMERICAL_EPS:
        return dict(result, reason='profile_side_boundary_numerically_ambiguous')
    same_side = side_cosine > 0.
    return dict(result, same_working_side=same_side,
        orientation_class='same_working_side' if same_side else 'opposite_working_side',
        needs_review=False, needs_adjustment=not same_side,
        reason='reviewed_profile_side' if same_side else 'profile_flipped_to_opposite_side')


class ToolUprightConflict(ValueError):
    def __init__(self, observation):
        self.observation = copy.deepcopy(observation)
        bad = ', '.join(f"{key}: {value['reason']}" for key, value in observation['tools'].items()
                        if not value['accepted'])
        super().__init__(f"Tool orientation conflicts at {observation['stage']}: {bad or observation['reason']}")


class ToolUprightGuard:
    def __init__(self, model, tool_bodies):
        if not tool_bodies:
            raise ValueError('Explicit robot tool-body bindings are required')
        self.model = model
        self.body_ids = {key: model.body(name).id for key, name in tool_bodies.items()}

    def observe(self, data, *, stage, scope='work', active_tool=None, reference_rotation=None):
        if scope not in SCOPES:
            raise ValueError(f'Unknown tool-orientation scope: {scope}')
        if active_tool is not None and active_tool not in self.body_ids:
            raise ValueError(f'Unknown active tool: {active_tool}')
        guidance = scope in ('preparation', 'work')
        missing_active = guidance and active_tool is None
        tools = {}
        for key, body_id in self.body_ids.items():
            checked = guidance and key == active_tool
            if checked:
                row = _profile_side(data.xmat[body_id].reshape(3, 3), reference_rotation)
                row['accepted'] = scope != 'work' or row['same_working_side'] is True
            else:
                reason = ('scope_does_not_constrain_tool_orientation' if not guidance
                          else 'active_tool_not_declared' if missing_active
                          else 'idle_tool_not_orientation_checked')
                row = dict(accepted=True, same_working_side=None, orientation_class='not_checked',
                    needs_review=False, needs_adjustment=False, reason=reason,
                    working_side_alignment_cos=None, shaft_alignment_cos=None,
                    twist_about_shaft_deg=None)
            row.update(checked=checked, scope=scope,
                blue_below_orange=row['same_working_side'],
                deprecated_aliases={'blue_below_orange':
                    'Alias of same_working_side, not a colour-centre height comparison; null means unchecked or ambiguous.'})
            tools[key] = row
        accepted = (scope != 'work' or not missing_active) and all(v['accepted'] for v in tools.values())
        reason = ('active_tool_not_declared' if missing_active else
                  'active_working_tool_accepted' if scope == 'work' and accepted else
                  'active_working_tool_requires_adjustment_or_review' if scope == 'work' else
                  'preparation_direction_guidance_only' if scope == 'preparation' else
                  'scope_does_not_constrain_tool_orientation')
        return dict(stage=stage, scope=scope, active_tool=active_tool, tools=tools,
            accepted=bool(accepted), all_tools_upright=bool(accepted), reason=reason,
            needs_review=missing_active or any(v['needs_review'] for v in tools.values()),
            needs_adjustment=any(v['needs_adjustment'] for v in tools.values()),
            deprecated_aliases={'all_tools_upright':
                'Alias of accepted for the current scope and active tool; idle tools are not checked.'},
            policy_version=POLICY_VERSION,
            kind='motion_candidate_guard_not_a_scoring_rule',
            criterion_source='Fixed S-profile +Y and shaft +Z compared with reviewed G1 rotation',
            requires_horizontal_tool=False, uses_colour_centre_world_height=False)

    def require(self, data, *, stage, scope='work', active_tool=None, reference_rotation=None):
        observation = self.observe(data, stage=stage, scope=scope, active_tool=active_tool,
                                   reference_rotation=reference_rotation)
        if not observation['accepted']:
            raise ToolUprightConflict(observation)
        return observation

    def inspect_saved_pose(self, full_qpos, *, stage, qpos_joint_names, scope='work',
                           active_tool=None, reference_rotation=None):
        names = [self.model.joint(i).name for i in range(self.model.njnt)]
        q = np.asarray(full_qpos, dtype=float)
        if names != qpos_joint_names or q.shape != (self.model.nq,) or not np.isfinite(q).all():
            raise ValueError('Saved G1 joints do not match the supplied robot scene')
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = q
        mujoco.mj_kinematics(self.model, scratch)
        return self.observe(scratch, stage=stage, scope=scope, active_tool=active_tool,
                            reference_rotation=reference_rotation)
