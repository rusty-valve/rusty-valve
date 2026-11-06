"""Read the sole current slow-opening rule; never substitute historical defaults."""
from __future__ import annotations

import json
import math
from pathlib import Path

RULE_RELATIVE_PATH = Path('configs/inactive_compatibility.json')
RULE_PATH = Path(__file__).parent / RULE_RELATIVE_PATH


def load_slow_open_rule() -> dict:
    """Return the authoritative parameters, including their provenance fields.

    Deployment must preserve RULE_RELATIVE_PATH under the deployed project root.
    Missing or malformed rules stop preparation instead of reverting to a
    single stop, another angle, or a controller-specific waiting duration.
    """
    try:
        rule = json.loads(RULE_PATH.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f'Missing authoritative slow-opening rule: {RULE_PATH}. '
            f'Include {RULE_RELATIVE_PATH} in the deployment bundle.') from exc
    angles = rule.get('pause_angles_deg')
    if not isinstance(angles, list) or not angles:
        raise ValueError('Slow-opening rule needs pause_angles_deg')
    angles = [float(value) for value in angles]
    if any(not math.isfinite(value) or not 0 < value < 90 for value in angles):
        raise ValueError('Slow-opening pause angles must be finite and between closed and open')
    if any(left >= right for left, right in zip(angles, angles[1:])):
        raise ValueError('Slow-opening pause angles must be strictly increasing')
    rule['pause_angles_deg'] = angles
    for name in ('pause_s', 'pause_angle_tolerance_deg', 'pause_speed_deg_s'):
        value = float(rule[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'Slow-opening rule requires positive finite {name}')
        rule[name] = value
    return rule
