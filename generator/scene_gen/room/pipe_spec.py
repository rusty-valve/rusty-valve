"""Load pipe sizes, colors and the range of operating heights."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

PATH = Path(__file__).with_suffix('.json')

def load():
    return json.loads(PATH.read_text())

def color(medium):
    spec=load()
    return deepcopy(spec['colors'].get(medium,spec['colors']['unknown']))

def select_dn(key, *, tier=None, rng=None, minimum=None):
    spec=load()
    if key in ('material','ordinary_material'):
        return int(spec['material_by_tier'][tier])
    if key in ('vapor','process_vapor'):
        choices=spec['process_vapor_candidates']
    elif key in ('condensate','condenser_condensate'):
        choices=spec['condenser_condensate_candidates']
    else:
        role, _, field=key.partition('.')
        value=spec['services'][role][field or 'device_dn']
        choices=value if isinstance(value,list) else [value]
    if minimum is not None:choices=[dn for dn in choices if dn>=minimum]
    if not choices:raise ValueError('pipe_spec has no eligible DN for '+key)
    return int(rng.choice(choices) if rng is not None else choices[0])

def p1_services():
    """Legacy planner schema, with every numeric DN read from this file."""
    spec=load();s=spec['services'];n=s['nitrogen']
    result={}
    for role,key in [('steam','steam'),('water','cold_source'),('water_return','cold_return'),('offgas','offgas')]:
        row=s[key]
        result[role]={'main_dn':row['main_dn'],'branch_dn':row['header_dn'],
                      'temporary_dn':row.get('device_dn',row.get('condenser_dn')),
                      'main_dn_candidates':[row['main_dn']]}
    result['nitrogen']={'main_dn':n['main_candidates'][-1],
        'branch_dn':n['header_max_dn'],'temporary_dn':n['device_candidates'][0],
        'main_dn_candidates':[dn for dn in n['main_candidates'] if dn>n['header_max_dn']]}
    # Kept only to reject obsolete steam-return demands with a clear message.
    result['steam_return']={'main_dn':s['steam']['main_dn'],
        'branch_dn':s['steam_condensate']['device_dn'],
        'temporary_dn':s['steam_condensate']['device_dn']}
    return result
