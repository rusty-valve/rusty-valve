"""Select equipment sizes from the recipe before laying out the building."""
from __future__ import annotations

from copy import deepcopy
import random

from .process import build_graph, validate, ProcessInputError
from .recipe import validate_recipe, InputRejected, CandidateRefused
from .equipment_catalog import resolve_spec

TIERS = ('small', 'medium', 'large')
PHYSICAL_KINDS = {'vessel', 'pump', 'filter', 'exchanger', 'condenser', 'fixed_bed', 'tower'}
COMPANIONS = {'filter', 'exchanger', 'condenser', 'fixed_bed'}


def _violation(code, detail, equipment=None, service_owner=None):
    return {'stage':'sizing','code':code,'equipment':equipment,
            'service_owner':service_owner,'detail':detail}


def _primary_tiers(recipe, rng):
    """Complete omitted tiers while respecting distillation receiver order."""
    units = {unit['id']:unit for unit in recipe['vessels']}
    bounds = {uid:([TIERS.index(unit['size_tier'])]*2 if 'size_tier' in unit else [0,2])
              for uid,unit in units.items()}
    distill = [t for t in recipe['transfers'] if t['mode']=='distillation']
    def propagate():
        changed=True
        while changed:
            changed=False
            for transfer in distill:
                source,target=transfer['source'],transfer['target']
                low=max(bounds[source][0],bounds[target][0])
                high=min(bounds[target][1],bounds[source][1])
                if low!=bounds[source][0]:bounds[source][0]=low;changed=True
                if high!=bounds[target][1]:bounds[target][1]=high;changed=True
                if any(a>b for a,b in bounds.values()):
                    return _violation('DISTILLATION_RECEIVER_LARGER',
                        'The distillation receiver cannot exceed the source reactor size tier. Keep the explicit input.',target,source)
        return None
    failure=propagate()
    if failure:return [failure]
    for uid,unit in units.items():
        if unit.get('size',1.)!=1.:
            return [_violation('SCALAR_SIZE_NOT_CATALOG_TIER',
                'Legacy size ratios cannot replace catalogue tiers. Manufacturer dimensions are fixed.',uid)]
        if 'size_tier' not in unit:
            low,high=bounds[uid]
            selected=rng.choice(list(range(low,high+1)))
            bounds[uid]=[selected,selected]
            failure=propagate()
            if failure:return [failure]
            unit['size_tier']=TIERS[selected]
    return []


def _select_companion(row, owner, rng, label):
    allowed=('small',) if owner['size_tier']=='small' else ('large',) if owner['size_tier']=='large' else ('small','large')
    if 'size_tier' in row and row['size_tier'] not in allowed:
        return _violation('COMPANION_TIER_MISMATCH',
            f'{label}explicit size tier{row["size_tier"]}does not follow source reactor{owner["size_tier"]}. Allowed: {list(allowed)}. Keep the input',
            label,owner['id'])
    if 'size_tier' not in row:
        row['size_tier']=rng.choice(allowed)
    return None


def select_equipment(recipe, seed):
    """Return sizing.json data, or a documented sizing rejection.

    Input/schema/capability rejections are records. Unexpected programming or
    file errors are deliberately not caught and cannot become capacity labels.
    """
    result={'stage':'sizing','status':'rejected','recipe':deepcopy(recipe),'graph':None,
            'equipment':[],'service_relations':[],'main_equipment_count':0,
            'checks':{},'violations':[]}
    try:
        canonical=validate_recipe(recipe)
    except InputRejected as exc:
        result['violations']=[_violation('INPUT_REJECTED',str(exc))]
        result['checks']['input']={'status':'fail'}
        return result
    canonical.setdefault('id','equipment_selection')
    canonical.setdefault('transfers',[])
    result['recipe']=canonical
    result['checks']['input']={'status':'pass'}
    rng=random.Random(seed)
    violations=_primary_tiers(canonical,rng)
    if violations:
        result['violations']=violations
        result['checks']['pairing']={'status':'fail'}
        return result
    owners={unit['id']:unit for unit in canonical['vessels']}
    for owner in canonical['vessels']:
        if owner.get('reflux_condenser'):
            failure=_select_companion(owner['reflux_condenser'],owner,rng,'reflux condenser returning to the source reactor')
            if failure:violations.append(failure)
    for number,transfer in enumerate(canonical['transfers'],1):
        owner=owners[transfer['source']]
        if transfer['mode']=='distillation':
            failure=_select_companion(transfer,owner,rng,f'transfer{number}condenser')
            if failure:violations.append(failure)
        for index,aux in enumerate(transfer.get('via',[]),1):
            failure=_select_companion(aux,owner,rng,f'transfer{number}auxiliary equipment{index}({aux["kind"]})')
            if failure:violations.append(failure)
        for aux in [*transfer.get('recycle_via',[]),*([transfer['protection_filter']] if transfer.get('protection_filter') else [])]:
            failure=_select_companion(aux,owner,rng,'pump unit auxiliary')
            if failure:violations.append(failure)
    if violations:
        result['violations']=violations
        result['checks']['pairing']={'status':'fail'}
        return result
    try:
        graph=build_graph({**canonical,'seed':seed})
    except ProcessInputError as exc:
        result['violations']=[_violation('PROCESS_INPUT_REJECTED',str(exc))]
        result['checks']['process']={'status':'fail'}
        return result
    process=validate(graph,stage='layout')
    result['graph']=graph
    result['checks']['process']=process
    if process['status']!='pass':
        result['violations']=[_violation('PROCESS_INVALID',str(process['errors']))]
        return result
    relations=[]
    for transfer in graph['transfers']:
        source=transfer['source']
        if transfer['mode']=='distillation':
            relations.append({'relation':'distillation_receiver','equipment':transfer['target'],
                'service_owner':source,'source_size_tier':owners[source]['size_tier'],
                'selected_size_tier':owners[transfer['target']]['size_tier'],'transfer':transfer['id']})
        for uid in dict.fromkeys([*transfer['path_units'],*transfer.get('recycle_path_units',[])]):
            unit=graph['units'].get(uid)
            if unit is not None and unit['kind'] in COMPANIONS:
                relations.append({'relation':'companion','equipment':uid,'kind':unit['kind'],
                    'service_owner':source,'source_size_tier':owners[source]['size_tier'],
                    'selected_size_tier':unit['size_tier'],'transfer':transfer['id']})
    for circuit in graph.get('reflux_circuits',[]):
        unit=graph['units'][circuit['condenser']];source=circuit['owner']
        relations.append({'relation':'companion','equipment':unit['id'],'kind':'condenser',
            'service_owner':source,'source_size_tier':owners[source]['size_tier'],
            'selected_size_tier':unit['size_tier'],'circuit':'reflux'})
    owner_by_unit={}
    for relation in relations:
        if relation['relation']=='companion':
            uid=relation['equipment'];source=relation['service_owner']
            if uid in owner_by_unit and owner_by_unit[uid]!=source:
                violations.append(_violation('SERVICE_OWNER_AMBIGUOUS','The auxiliary unit serves multiple reactors and its size cannot be inferred',uid))
            owner_by_unit[uid]=source
    for unit in graph['units'].values():
        kind=unit['kind']
        if kind not in PHYSICAL_KINDS:continue
        owner=unit.get('source_vessel') if kind=='pump' else owner_by_unit.get(unit['id'])
        if owner is not None:unit['service_owner']=owner
        if kind=='pump':
            spec={'kind':'pump','form':'horizontal','size_tier':'N/A','estimated':True,
                  'dimensions_m':{'width_m':1.10,'depth_m':.68,'height_m':.77},
                  'source':'Coarse physical pump, motor and base envelope. It is independent of the main reactor scale.'}
            relations.append({'relation':'pump_source','equipment':unit['id'],'service_owner':owner})
            unit.update(form='horizontal',orientation='horizontal',size_tier='N/A',dimension_spec=deepcopy(spec))
        else:
            try:
                spec=resolve_spec(unit)
            except CandidateRefused as exc:
                violations.append(_violation('EQUIPMENT_CATALOG_UNSUPPORTED',str(exc),unit['id'],owner))
                spec=None
        result['equipment'].append({'id':unit['id'],'kind':kind,
            'form':spec['form'] if spec is not None else unit.get('form'),
            'size_tier':spec['size_tier'] if spec is not None else unit.get('size_tier'),
            'dimension_spec':spec,'service_owner':owner})
        if spec is not None and kind!='pump':
            unit['dimension_spec']=deepcopy(spec)
            unit['catalog_key']=spec['catalog_key']
    result['service_relations']=relations
    result['main_equipment_count']=sum(row['kind']!='pump' for row in result['equipment'])
    if result['main_equipment_count']<2:
        violations.append(_violation('TOO_FEW_MAIN_EQUIPMENT','Provide at least two main equipment units other than pumps'))
    result['checks']['pairing']={'status':'pass','relations':relations}
    result['checks']['catalog']={'status':'fail' if any(row['dimension_spec'] is None for row in result['equipment']) else 'pass'}
    result['checks']['main_count']={'status':'pass' if result['main_equipment_count']>=2 else 'fail',
                                  'count':result['main_equipment_count'],'minimum':2}
    result['violations']=violations
    result['status']='rejected' if violations else 'pass'
    return result
