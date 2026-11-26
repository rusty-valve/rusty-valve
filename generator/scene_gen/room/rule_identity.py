"""Identify which rules affect placement and which results are recorded."""
from .pipe_spec import load

IDENTITIES = {
    'height': 'preference', 'stance': 'preference', 'reach': 'preference',
    'equipment_top_access': 'record', 'route_cost': 'preference',
    'condenser_height': 'input',
    'penetration': 'hard', 'connections': 'record', 'required_valves': 'record',
    'orthogonality': 'hard', 'equipment_dimensions': 'hard', 'floor_limit': 'hard',
    'maintenance': 'hard', 'pullout': 'hard', 'pump_rows': 'hard',
    'connection_method': 'hard', 'pipe_spec': 'hard',
}

def identity(rule):
    return IDENTITIES[rule]

def band():
    return load()['operation_band_m']

def in_band(height):
    low, high = band()
    return low-1e-8 <= float(height) <= high+1e-8

def height_preference(grips, floors, levels=None, top=False):
    levels = list(levels) if levels else list(range(len(floors)))
    limits = band()
    low, high = limits
    eligible = [i for i in levels if all(low-1e-8 <= float(p[2]-floors[i]) <= high+1e-8 for p in grips)]
    level = eligible[-1] if eligible else min(levels, key=lambda i: min(abs(p[2]-floors[i]) for p in grips))
    lowest = min(float(p[2]) for p in grips)
    return {'identity': identity('equipment_top_access' if top else 'height'),
            'level': level, 'within_band': bool(eligible), 'lowest_grip_z_m': lowest,
            'band_m': limits, 'challenge': not bool(eligible),
            'rank': [0,0.] if top else [0 if eligible else 1, 0. if eligible else lowest],
            'note': 'The existing platform has no direct operation position. Access equipment is pending.' if top and not eligible else
                    'When all usable grips are above the preferred height, choose the lowest one' if not eligible else ''}

def equipment_top_port(port):
    source=port.get('source_port',port)
    return (float(source.get('normal',[0,0,0])[2])>.9 or
            source.get('nozzle_zone') in ('upper_shell','upper_channel_box','shell_top'))


def annotate_valve(a, valve, port=None):
    import numpy as np
    from .piping_material_templates import _grip_points
    grips=[np.asarray(valve['origin'])+np.asarray(valve['matrix'])@g for g in _grip_points(valve)]
    top=bool(port and equipment_top_port(port))
    top=top or bool(valve.get('equipment_top'))
    reference=(port or {}).get('operation_reference',valve.get('operation_reference',{}))
    levels=[reference['level']] if reference.get('status')=='standing_face_reference' else None
    report=height_preference(grips,a.floors,levels,top)
    valve.update(operation_preference=report,equipment_top=top,level=report['level'],
                 operating_floor_z_m=a.floors[report['level']],
                 challenge_position=report['challenge'],operation_grip_points_m=[p.tolist() for p in grips])
    return tuple(report['rank'])


def classify(*, completed, hard_errors=(), exception=None, timed_out=False, input_valid=True):
    if not input_valid:return 'invalid input'
    if timed_out or exception or not completed:return 'generation_incomplete'
    return 'noncompliant' if hard_errors else 'compliant'
