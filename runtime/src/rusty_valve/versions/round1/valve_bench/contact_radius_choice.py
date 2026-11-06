"""Finite contact-radius choices from the single formal proportion rule.

The preferred radius is preserved exactly when it is in range. Alternatives
are existing inside/midpoint/outside implementation samples, ordered by their
distance from that preferred point. These are candidate values, not scores,
new contact tolerances, physical retries, or a claim that every valid radius
has been searched. No simulator, workstation name or motion is used here.
"""
from __future__ import annotations

import math

from .contact_target import RULE_PATH, load_contact_target_rule, target_radii


POLICY_VERSION = 'preferred_then_finite_allowed_radius_candidates_v1'


def _finite_radius(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError('Preferred contact radius must be a finite number')
    return float(value)


def _same_numerical_radius(a, b):
    # Remove alternate encodings such as .175 and .2*.875. This ULP-only
    # equality is for list deduplication, never admission or a physical gate.
    return abs(a-b) <= 4.*max(math.ulp(a), math.ulp(b))


def radius_candidates(handle_length_m, preferred_radius_m):
    """Return the exact preferred value, then nearest formal sample values.

    A preferred point outside the current formal range is an input error; it
    is never silently clipped into range. At equal distance the farther-out
    point is tried first, consistent with moving outward when leverage helps.
    Sampling does not change the caller's target or move an actual robot.
    """
    samples = target_radii(handle_length_m)  # includes formal length validation
    preferred = _finite_radius(preferred_radius_m)
    lo, hi = min(samples), max(samples)
    if not lo <= preferred <= hi:
        raise ValueError(f'Preferred radius {preferred!r} is outside the formal '
                         f'contact range [{lo!r}, {hi!r}] m')
    alternatives = []
    for radius in samples:
        if _same_numerical_radius(preferred, radius):
            continue
        if any(_same_numerical_radius(radius, existing) for existing in alternatives):
            continue
        alternatives.append(radius)
    alternatives.sort(key=lambda radius: (abs(radius-preferred), -radius))
    return [preferred]+alternatives


class ContactRadiusChoice:
    """A finite, monotonic candidate cursor with caller-supplied reasons.

    ``advance`` only records a reported need for another candidate. It does
    not assess a failure, run a retry, loosen a rule, or reset itself. Once
    exhausted it stays exhausted, preventing an implicit endless retry loop.
    """
    def __init__(self, handle_length_m, preferred_radius_m):
        self._candidates = tuple(radius_candidates(handle_length_m, preferred_radius_m))
        self.handle_length_m = float(handle_length_m)
        self.preferred_radius_m = float(preferred_radius_m)
        rule = load_contact_target_rule()
        self.allowed_fraction_range = tuple(rule.get('target_contact_fraction_range',
            [rule['target_contact_fraction']]*2))
        self.rule_version = rule.get('policy_version')
        self._index = 0
        self._history = []

    @property
    def candidates(self):
        return list(self._candidates)

    @property
    def exhausted(self):
        return self._index >= len(self._candidates)

    @property
    def current_radius_m(self):
        return None if self.exhausted else self._candidates[self._index]

    @property
    def selected_radius_m(self):
        return self.current_radius_m

    def advance(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('A nonempty reason is required to change contact candidate')
        if self.exhausted:
            return None
        previous = self.current_radius_m
        previous_index = self._index
        self._index += 1
        self._history.append(dict(from_index=previous_index,
            from_radius_m=previous, to_radius_m=self.current_radius_m,
            reason=reason, exhausted=self.exhausted))
        return self.current_radius_m

    def summary(self):
        return dict(policy_version=POLICY_VERSION, rule_source=str(RULE_PATH),
            rule_version=self.rule_version,
            handle_length_m=self.handle_length_m,
            preferred_radius_m=self.preferred_radius_m,
            preferred_validation='inside_current_formal_range_without_clipping',
            allowed_fraction_range=list(self.allowed_fraction_range),
            allowed_radius_range_m=[self.handle_length_m*f for f in self.allowed_fraction_range],
            candidate_radii_m=self.candidates,
            selected_radius_m=self.current_radius_m,
            candidate_index=self._index if not self.exhausted else None,
            exhausted=self.exhausted,
            selection_reason=(self._history[-1]['reason'] if self._history
                              else 'preserved_preferred_radius'),
            selection_history=[dict(row) for row in self._history],
            sample_policy='preferred_then_nearest_inside_midpoint_outside_samples',
            all_continuous_radii_searched=False, physics_steps_run=0,
            physical_retry_started=False, scoring_changed=False)
