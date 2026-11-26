"""Arrange equipment and connect it with utility and process pipes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
import traceback

from .process import write_json
from .pipe_spec import PATH as SPEC_PATH


def _read(path):
    return json.loads(Path(path).read_text())


def _numbers(measured):
    material = measured.get('material', {})
    pc, mc = measured.get('connections', {}), material.get('connections', {})
    roles=measured.get('valve_roles',{})
    pv, mv = measured.get('manual_valves', {}), material.get('manual_valves', {})
    return {
        'connections': {'connected': pc.get('connected', 0)+mc.get('connected', 0),
                        'expected': pc.get('expected', 0)+mc.get('expected', 0)},
        'manual_valves': {'installed': sum(bool(row.get('actual_valve')) and bool(row.get('actual_ports')) for row in roles['rows']) if roles else pv.get('installed', 0)+mv.get('installed', 0),
                          'expected': roles['expected'] if roles else pv.get('expected', 0)+mv.get('expected', 0),
                          'role_position_order_verified':roles.get('verified')},
        'actual_material_penetration_count': measured.get('actual_material_penetration_count'),
        'measurement_status': 'measured' if mc.get('expected', 0) else 'P1_only_material_unmeasured',
        'scope': 'actual exported pipe and native valve packages; missing work remains in denominator',
    }


def generate(recipe, shell_input, seed, folder, images=False, *, storage=None, equipment_reference=None):
    from .building_equipment_generation import generate as layout_generate
    from .piping_method import generate_platform
    started = perf_counter(); folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/'input.json', {'recipe': deepcopy(recipe), 'shell': deepcopy(shell_input),
                                    'seed': seed, 'stage': 'full', 'piping_method': 'end-bundle'})
    (folder/'pipe_spec_snapshot.json').write_bytes(SPEC_PATH.read_bytes())
    layout_folder, model_folder = folder/'layout', folder/'model'
    result = {'room': folder.name, 'seed': seed, 'generation': 'partial',
              'stage': 'full', 'piping_method': 'end-bundle', 'version': 1,
              'source': 'coordinate-free recipe and equipment catalogue',
              'equipment_reference':str(equipment_reference) if equipment_reference else None,
              'layout_dir': str(layout_folder.resolve()), 'model_dir': str(model_folder.resolve()),
              'numbers': {'measurement_status': 'not_measured'}, 'known_problems': [],
              'scope': 'Equipment, piping and valve generation results'}
    try:
        # Existing generic generator, with a small explicit internal budget.
        # These are retained layout candidates, not successive material versions.
        layout = layout_generate(recipe, shell_input, seed, layout_folder, False, attempts=8, reference=equipment_reference,
                                 reserve_pipe_departures=True)
        write_json(layout_folder/'result.json', layout)
        result['layout_generation'] = layout['generation']
        if layout['generation'] != 'success':
            result['known_problems'].append({'stage': 'layout', 'reason': layout.get('reason'),
                                             'causes': layout.get('causes', {})})
        else:
            measured = generate_platform(layout_folder, model_folder, include_material=True, storage=storage)
            from .batch_outcomes import measured_outcome,challenges
            result.update(measured_outcome(measured,_read(model_folder/'piping.json')))
            result['challenges']=challenges(measured,_read(model_folder/'process.json'))
            result['numbers'] = _numbers(measured)
            result['measurement'] = str((model_folder/'measure.json').resolve())
            result['known_problems'] = (measured.get('method_failures', [])+
                _read(model_folder/'piping.json').get('material', {}).get('failures', []))
            if measured.get('status') == 'pass' and measured.get('material', {}).get('status') == 'pass':
                result['generation'] = 'success'
            if result['generation'] != 'success' and not result['known_problems']:
                result['known_problems'] = [{'stage': 'actual_measurement',
                    'codes': sorted({e.get('code', 'unknown') for e in measured.get('errors', [])+
                                     measured.get('material', {}).get('errors', [])})}]
    except Exception as error:
        result['substatus']='stage_exception'
        result['known_problems'].append({'stage': 'generation_or_measurement',
            'type': type(error).__name__, 'reason': str(error)})
        write_json(folder/'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
    result.setdefault('status','generation_incomplete')
    result.setdefault('stops',result['known_problems'])
    result['elapsed_seconds'] = perf_counter()-started
    write_json(folder/'result.json', result)
    return result
