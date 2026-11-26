"""Define which process recipes and equipment the generator supports."""
import json
from .storage_vessel import is_horizontal
from pathlib import Path
PATH=Path(__file__).with_suffix('.json')


def canonical_transfer_mode(mode):
    # Older frozen requests distinguish two supposed drivers for the same pumpless line.
    return 'no_pump' if mode in ('gravity', 'nitrogen') else mode


def load():
    return json.loads(PATH.read_text())


def ensure_supported(recipe, error_type=ValueError):
    support=load()
    for unit in recipe.get('vessels',[]):
        condenser=unit.get('reflux_condenser')
        if condenser and condenser.get('form',condenser.get('orientation','vertical')) not in support['reflux_condenser_forms']:
            raise error_type('This reflux condenser form is unsupported')
        has_jacket=unit.get('jacket') or 'jacket_medium' in unit
        if is_horizontal(unit) and (has_jacket or condenser or unit.get('reactor')):
            raise error_type(unit.get('id','')+'Horizontal storage vessels do not support jackets, reflux condensers or reactor service')
        if has_jacket and unit.get('jacket_medium') not in support['jacket_media']:
            raise error_type('Supported jacket media: '+','.join(support['jacket_media']))
    for transfer in recipe.get('transfers',[]):
        mode=canonical_transfer_mode(transfer.get('mode'))
        if mode not in support['transfer_modes']:
            raise error_type(support['out_of_scope'].get(mode,'Unsupported transfer mode: '+str(mode)))
        if transfer.get('source')==transfer.get('target') and mode not in support['self_return_modes']:
            raise error_type('Recirculation requires a pump')
        for value in transfer.get('via',[]):
            kind=value if isinstance(value,str) else value.get('kind')
            if kind not in support['transfer_via']:
                key='transfer_exchanger' if kind=='exchanger' else kind
                raise error_type(support['not_implemented'].get(key,'Use a pump-inlet protection_filter instead of an inline via filter'))
        if (transfer.get('recycle_via') or transfer.get('protection_filter')) and mode!='pump':
            raise error_type('Pump-inlet filtration and recirculation heating require pumped transfer')
        for aux in transfer.get('recycle_via',[]):
            if not any(aux.get('kind')==r['kind'] and aux.get('service')==r['service'] and
                       aux.get('form',aux.get('orientation','vertical')) in r['forms'] for r in support['recycle_auxiliaries']):
                raise error_type('Recirculation supports horizontal or vertical steam heaters only')
    return support
