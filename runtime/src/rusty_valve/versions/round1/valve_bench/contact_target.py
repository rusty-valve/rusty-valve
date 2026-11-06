"""Resolve the user-defined contact fraction from the sole rule source.

The length is shaft-axis to handle tip along the handle, never overall valve
body length or tool TCP distance. Historical frozen radii remain untouched.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RULE_PATH = Path(__file__).parent / 'configs/contact_target.json'


def load_contact_target_rule():
    rule = json.loads(RULE_PATH.read_text(encoding='utf-8'))
    fraction = rule.get('target_contact_fraction')
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('Contact fraction must be finite and strictly between axis and tip')
    limits = rule.get('target_contact_fraction_range', [fraction, fraction])
    if (len(limits) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
            or not math.isfinite(x) for x in limits) or not 0 < limits[0] <= limits[1] < 1):
        raise ValueError('Contact range must lie between shaft axis and handle tip')
    return rule


def target_radius(handle_length_m):
    """Historical single-reference resolver. New unified skills use target_radii."""
    if isinstance(handle_length_m, bool) or not isinstance(handle_length_m, (int, float)) or not math.isfinite(handle_length_m) or handle_length_m <= 0:
        raise ValueError('Need positive shaft-axis-to-handle-tip length in metres')
    return float(handle_length_m) * load_contact_target_rule()['target_contact_fraction']


def target_radii(handle_length_m):
    """Common initial search candidates: inside, midpoint, outside of allowed range.

    The range is the user's rule; three samples are an implementation choice,
    not an additional exam requirement or a claim that other points are invalid.
    """
    target_radius(handle_length_m)  # same length validation
    rule = load_contact_target_rule()
    lo, hi = rule.get('target_contact_fraction_range',
                      [rule['target_contact_fraction']]*2)
    return [float(handle_length_m)*fraction
            for fraction in dict.fromkeys((lo, (lo+hi)/2., hi))]
