"""Validate process recipes and their equipment and connection choices."""
from __future__ import annotations

from copy import deepcopy
import math

from .material_support import canonical_transfer_mode
from .process import random_recipes
from .storage_vessel import normalize_vessel


class InputRejected(ValueError):
    """A readable rejection of a recipe or shell outside the supported inputs."""


class CandidateRefused(ValueError):
    """Only known capacity or no-route results may reject a candidate."""


def _keys(value, allowed, where):
    if not isinstance(value, dict):
        raise InputRejected(f'{where} must be an object')
    extra = set(value) - set(allowed)
    if extra:
        raise InputRejected(f'{where} does not accept {sorted(extra)}. Specify equipment forms without coordinates or rotation.')


def _form(row, where, default='vertical'):
    for key in ('form', 'orientation'):
        if key in row and row[key] not in ('vertical', 'horizontal'):
            raise InputRejected(f'{where}.{key} must be vertical or horizontal. The generator selects rotation.')
    return row.get('form', row.get('orientation', default))


def _tier(row, where):
    if 'size_tier' in row and row['size_tier'] not in ('small', 'medium', 'large'):
        raise InputRejected(f'{where}.size_tier must be small, medium, or large')


def _auxiliary(value, where):
    row = {'kind': value} if isinstance(value, str) else deepcopy(value)
    _keys(row, ('kind', 'form', 'orientation', 'size_tier', 'service'), where)
    kind = row.get('kind')
    if kind not in ('filter', 'exchanger', 'fixed_bed'):
        raise InputRejected('via accepts filter, exchanger, or fixed_bed')
    row['form'] = _form(row, where)
    _tier(row, where)
    if kind == 'fixed_bed' and row['form'] != 'vertical':
        raise InputRejected(f'{kind} only supports the vertical form')
    row.pop('orientation', None)  # Preserve its meaning in the canonical form.
    return row


def validate_recipe(recipe):
    recipe = deepcopy(recipe)
    _keys(recipe, ('id', 'description', 'relation', 'vessels', 'transfers'), 'Process recipe')
    vessels = recipe.get('vessels')
    if not isinstance(vessels, list) or not vessels:
        raise InputRejected('The recipe must contain at least one reactor or storage vessel')
    identifiers = []
    for i, unit in enumerate(vessels):
        _keys(unit, ('id', 'jacket', 'jacket_medium', 'reflux_condenser', 'reactor', 'pressurizable', 'size', 'dn', 'form', 'orientation', 'size_tier'), f'Vessel {i+1}')
        name = unit.get('id')
        if not isinstance(name, str) or not name or not name.replace('_', '').isalnum():
            raise InputRejected('Equipment IDs must contain only letters, numbers, or underscores')
        if name.startswith(('X_', 'U_', 'J_')):
            raise InputRejected('The equipment ID conflicts with a generated port prefix')
        identifiers.append(name)
        unit['form'] = _form(unit, f'Vessel {i+1}')
        _tier(unit, f'Vessel {i+1}')
        unit.pop('orientation', None)
        scale = unit.get('size', 1.)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise InputRejected('Equipment size must be positive')
        for key in ('jacket', 'reactor', 'pressurizable'):
            if key in unit and not isinstance(unit[key], bool):
                raise InputRejected(f'{name}.{key} must be a boolean')
        normalize_vessel(unit)
        if unit.get('jacket') and unit.get('jacket_medium') not in ('steam','cold_source','hot_water'):
            raise InputRejected(name+' requires an explicit jacket_medium')
        if 'jacket_medium' in unit:
            if unit['jacket_medium'] not in ('steam','cold_source','hot_water'):raise InputRejected('Unknown jacket medium')
            unit['jacket']=True
        if 'reflux_condenser' in unit:
            condenser=unit['reflux_condenser'];_keys(condenser,('form','size_tier'),'reflux_condenser')
            condenser['form']=_form(condenser,'reflux_condenser');_tier(condenser,'reflux_condenser')
    if len(set(identifiers)) != len(identifiers):
        raise InputRejected('Equipment IDs must be unique')
    transfers = recipe.get('transfers', [])
    if not isinstance(transfers, list):
        raise InputRejected('transfers must be a list of transfer connections')
    for i, transfer in enumerate(transfers):
        _keys(transfer, ('id', 'source', 'target', 'mode', 'dn', 'via', 'protection_filter', 'recycle_via', 'form', 'orientation', 'size_tier'), f'Transfer{i+1}')
        if transfer.get('source') not in identifiers:
            raise InputRejected(f'Transfer{i+1} refers to an unknown source vessel')
        target = transfer.get('target', '')
        if not isinstance(target, str) or (target not in identifiers and not target.startswith('EXPORT')):
            raise InputRejected(f'Transfer{i+1} refers to an unknown target vessel')
        mode = transfer['mode'] = canonical_transfer_mode(transfer.get('mode'))
        if mode not in ('pump', 'no_pump', 'distillation'):
            raise InputRejected(f'Unsupported transfer mode: {mode}')
        if any(key in transfer for key in ('form', 'orientation', 'size_tier')):
            if mode != 'distillation':
                raise InputRejected('Transfer-level form and size_tier apply only to distillation condensers. Use via for inline equipment.')
            transfer['form'] = _form(transfer, f'Transfer{i+1} condenser')
            _tier(transfer, f'Transfer{i+1} condenser')
            transfer.pop('orientation', None)
        via = transfer.get('via', [])
        if not isinstance(via, list):
            raise InputRejected('via must list inline equipment')
        if 'via' in transfer:
            transfer['via'] = [_auxiliary(value, f'Transfer{i+1}.via[{j}]') for j, value in enumerate(via)]
        if via and mode != 'pump':
            raise InputRejected('Inline filters, exchangers, and fixed beds require a pumped transfer')
        # Every pump gets upstream protection; omitted input is canonicalized
        # before sizing and graph generation, not added only by a viewer.
        if mode == 'pump':
            transfer.setdefault('protection_filter', {'form':'horizontal'})
        if 'protection_filter' in transfer:
            if mode!='pump':raise InputRejected('Pump protection filters require a pumped transfer')
            pf=transfer['protection_filter'];_keys(pf,('form','size_tier'),'protection_filter')
            pf['form']=_form(pf,'protection_filter',default='horizontal');_tier(pf,'protection_filter')
        if 'recycle_via' in transfer:
            if mode!='pump' or not isinstance(transfer['recycle_via'],list):raise InputRejected('recycle_via must list equipment in a pumped recirculation line')
            transfer['recycle_via']=[_auxiliary(v,'recycle_via') for v in transfer['recycle_via']]
            if any(v['kind']!='exchanger' or v.get('service')!='steam_heater' for v in transfer['recycle_via']):
                raise InputRejected('Recirculation equipment must be an exchanger with service=steam_heater')
        if target == transfer['source'] and mode != 'pump':
            raise InputRejected('Recirculation to the same vessel requires a pump')
        if mode == 'distillation' and target.startswith('EXPORT'):
            raise InputRejected('Distillation requires a receiving vessel')
    from .pipe_spec import load
    allowed={int(k) for k in load()['pipe_outer_diameter_mm']}
    for row in [*vessels, *transfers]:
        if 'dn' in row and (isinstance(row['dn'],bool) or row['dn'] not in allowed):
            raise InputRejected('DN must be listed in the pipe specification')
    from .material_support import ensure_supported
    ensure_supported(recipe, InputRejected)
    return deepcopy(recipe)


def batch_recipes(seed, count):
    """Stratified process strata with source-led sizing, fixed before generation."""
    from .equipment_recipes import batch_recipes as balanced_recipes
    return balanced_recipes(seed, count)


def legacy_batch_recipes(seed, count):
    """Historical process draws for legacy fixtures."""
    recipes = random_recipes(seed, max(9, count))[:count]
    return [({key: deepcopy(value) for key, value in row.items() if key not in ('seed',)}, row['seed'])
            for row in recipes]
