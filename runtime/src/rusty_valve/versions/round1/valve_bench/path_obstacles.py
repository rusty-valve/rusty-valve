"""Classify native scratch contacts without moving the robot or changing rules.

The caller supplies an already populated MuJoCo contact buffer. Only actual
negative distances count as overlap; no additional clearance is imposed. The
native model's collision masks and excluded pairs are never reconstructed or
overridden. Permitted active-tool/handle contact is not proof of a correct
working face or a successful physical operation; task scoring remains separate.
"""
from __future__ import annotations

import copy
from numbers import Integral


CATEGORIES = ('self_collision', 'fixture_collision', 'non_working_handle_collision')


class PathObstacleConflict(ValueError):
    def __init__(self, observation):
        self.observation = copy.deepcopy(observation)
        pairs = ', '.join(
            f"{item['geom1']} / {item['geom2']} ({item['category']})"
            for item in observation['issues'][:6])
        super().__init__(f"Path obstacle at {observation['stage']}: {pairs}")


def _geom_ids(model, values, label):
    result = frozenset(values)
    if any(not isinstance(value, Integral) or not 0 <= value < model.ngeom for value in result):
        raise ValueError(f'{label} must contain valid model geometry IDs')
    return result


class PathObstacleGuard:
    """Reusable contact reader with explicit geometry bindings, no station logic."""

    def __init__(self, model, *, robot_geom_ids, fixture_geom_ids,
                 handle_geom_ids, active_tool_geom_ids):
        self.model = model
        self.robot = _geom_ids(model, robot_geom_ids, 'robot_geom_ids')
        self.fixtures = _geom_ids(model, fixture_geom_ids, 'fixture_geom_ids')
        self.handle = _geom_ids(model, handle_geom_ids, 'handle_geom_ids')
        self.active_tool = _geom_ids(model, active_tool_geom_ids, 'active_tool_geom_ids')
        if not self.active_tool <= self.robot:
            raise ValueError('active_tool_geom_ids must be included in robot_geom_ids')

    def observe(self, data, *, stage='unspecified'):
        """Read contacts already computed by the caller; no FK or collision call."""
        issues, permitted = [], []
        counts = dict.fromkeys(CATEGORIES, 0)
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            distance = float(contact.dist)
            if distance >= 0. or not (a in self.robot or b in self.robot):
                continue
            active_handle = ((a in self.active_tool and b in self.handle)
                             or (b in self.active_tool and a in self.handle))
            if active_handle:
                category = 'permitted_active_tool_handle_contact'
            elif a in self.robot and b in self.robot:
                category = 'self_collision'
            elif a in self.handle or b in self.handle:
                category = 'non_working_handle_collision'
            elif a in self.fixtures or b in self.fixtures:
                category = 'fixture_collision'
            else:
                # Retain the existing _static_geometry scope: only explicit
                # robot, fixture and handle bindings are classified here.
                continue
            row = dict(category=category, geom1=self.model.geom(a).name,
                geom2=self.model.geom(b).name, geom1_id=a, geom2_id=b,
                distance_m=distance, depth_m=-distance,
                position_world_m=[float(value) for value in contact.pos])
            if active_handle:
                permitted.append(row)
            else:
                issues.append(row)
                counts[category] += 1
        return dict(stage=stage, clear=not issues, issues=issues,
            issue_counts=counts, penetration_m=sum(row['depth_m'] for row in issues),
            max_penetration_m=max((row['depth_m'] for row in issues), default=0.),
            permitted_tool_handle_contacts=permitted, native_contact_count=int(data.ncon),
            kind='native_scratch_contact_check_not_a_scoring_rule',
            criterion_source='Existing _static_geometry negative-distance contact categories',
            physics_steps_run=0, additional_clearance_m=0.,
            permitted_contact_note='Active tool/handle overlap is exempt here only; correct working face and physical success require their separate evidence.')

    def require(self, data, *, stage='unspecified'):
        observation = self.observe(data, stage=stage)
        if not observation['clear']:
            raise PathObstacleConflict(observation)
        return observation


def assess_contacts(model, data, *, robot_geom_ids, fixture_geom_ids,
                    handle_geom_ids, active_tool_geom_ids, stage='unspecified'):
    """Classify the supplied native contact buffer with explicit geometry sets."""
    guard = PathObstacleGuard(model, robot_geom_ids=robot_geom_ids,
        fixture_geom_ids=fixture_geom_ids, handle_geom_ids=handle_geom_ids,
        active_tool_geom_ids=active_tool_geom_ids)
    return guard.observe(data, stage=stage)


def require_clear(model, data, *, robot_geom_ids, fixture_geom_ids,
                  handle_geom_ids, active_tool_geom_ids, stage='unspecified'):
    """Raise PathObstacleConflict on a real classified overlap; never move data."""
    guard = PathObstacleGuard(model, robot_geom_ids=robot_geom_ids,
        fixture_geom_ids=fixture_geom_ids, handle_geom_ids=handle_geom_ids,
        active_tool_geom_ids=active_tool_geom_ids)
    return guard.require(data, stage=stage)
