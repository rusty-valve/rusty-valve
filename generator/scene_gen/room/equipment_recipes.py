"""Create example recipes with different equipment sizes and transfer arrangements."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True

STRATA = (
    {'mode':'pump','relation':'one_to_one','count':2},
    {'mode':'pump','relation':'one_to_many','count':3},
    {'mode':'pump','relation':'many_to_one','count':3},
    {'mode':'nitrogen','relation':'one_to_one','count':2},
    {'mode':'nitrogen','relation':'one_to_many','count':1},
    {'mode':'nitrogen','relation':'many_to_one','count':1},
    {'mode':'gravity','relation':'one_to_one','count':2},
    {'mode':'gravity','relation':'one_to_many','count':1},
    {'mode':'gravity','relation':'many_to_one','count':1},
    {'mode':'distillation','relation':'one_to_one','count':2},
    {'mode':'distillation','relation':'one_to_many','count':1},
    {'mode':'distillation','relation':'many_to_one','count':1},
)

# These are primary-device requests, not room-specific geometry. A profile is
# attached to its process stratum before names, seeds or positions exist.
PRIMARY_PROFILES = {
    'pump':[
        [('vertical','small'),('vertical','small')],
        [('vertical','medium'),('horizontal','small')],
        [('vertical','large'),('vertical','small'),('horizontal','small')],
        [('vertical','small'),('vertical','small'),('horizontal','small')],
        [('vertical','medium'),('vertical','small'),('horizontal','small')],
        [('vertical','small'),('horizontal','small'),('vertical','small')],
        [('vertical','large'),('horizontal','small'),('vertical','small')],
        [('vertical','medium'),('horizontal','small'),('vertical','small')]],
    'nitrogen':[
        [('vertical','large'),('horizontal','small')],
        [('vertical','small'),('horizontal','small')],
        [('vertical','medium'),('vertical','small'),('horizontal','small')],
        [('vertical','large'),('horizontal','small'),('vertical','medium')]],
    'gravity':[
        [('vertical','small'),('horizontal','small')],
        [('vertical','large'),('vertical','small')],
        [('vertical','medium'),('vertical','small'),('horizontal','small')],
        [('vertical','large'),('horizontal','small'),('vertical','small')]],
    'distillation':[
        [('vertical','small'),('vertical','small')],
        [('vertical','large'),('horizontal','small')],
        [('vertical','medium'),('vertical','small'),('horizontal','small')],
        [('vertical','small'),('vertical','medium'),('horizontal','small')]],
}


def balanced_templates():
    """Construct all 20 process slots; leave companion tiers to sizing."""
    templates=[];mode_index=Counter()
    for stratum in STRATA:
        mode,relation=stratum['mode'],stratum['relation']
        for _ in range(stratum['count']):
            index=mode_index[mode];mode_index[mode]+=1
            profiles=PRIMARY_PROFILES[mode][index]
            vessels=[{'id':f'R{i+1}','form':form,'size_tier':tier,'reactor':form=='vertical','dn':50}
                     for i,(form,tier) in enumerate(profiles)]
            pairs=([(0,1)] if relation=='one_to_one' else
                   [(0,1),(0,2)] if relation=='one_to_many' else [(0,2),(1,2)])
            transfers=[{'source':vessels[a]['id'],'target':vessels[b]['id'],'mode':mode,'dn':50}
                       for a,b in pairs]
            if mode=='pump':
                # Six source-led fixed-bed trains, two exchanger trains. Only
                # the first transfer needs this optional treatment unit.
                kind='fixed_bed' if index<6 else 'exchanger'
                form='vertical' if kind=='fixed_bed' else 'horizontal'
                transfers[0]['via']=[{'kind':kind,'form':form}]
            elif mode=='distillation':
                # Three horizontal-condenser process slots and one vertical.
                # Tier is deliberately absent: the source determines it.
                for transfer in transfers:
                    transfer['form']='horizontal' if index<3 else 'vertical'
            templates.append({'description':'Stratified by transfer mode and unit count. Size main equipment first and derive auxiliary size from its source reactor.',
                              'relation':relation,'vessels':vessels,'transfers':transfers})
    return templates


def legacy_batch_recipes(seed,count):
    """Return canonical [(recipe, room_seed)] without any layout generation."""
    from .equipment_sizing import select_equipment
    if isinstance(count,bool) or not isinstance(count,int) or count<1:
        raise ValueError('Count must be a positive integer')
    rng=random.Random(seed);requests=[]
    while len(requests)<count:
        block=balanced_templates();rng.shuffle(block)
        for raw in block:
            if len(requests)==count:break
            raw=deepcopy(raw);raw['id']=f'room_{len(requests)+1:02}'
            room_seed=rng.randrange(1,10_000_000)
            sizing=select_equipment(raw,room_seed)
            # Keep rejections too. No replacement draw or alternate sizing is
            # allowed, and production will record the same rejection reason.
            requests.append((sizing['recipe'],room_seed))
    return requests


def requested_coverage(sizing):
    categories=set();forms=[]
    for row in sizing['equipment']:
        kind,form,tier=row['kind'],row['form'],row['size_tier']
        if kind=='pump':continue
        forms.append((kind,form,tier))
        if kind=='vessel' and form=='vertical' and tier in ('small','large'):
            categories.add('small_reactor' if tier=='small' else 'large_reactor')
        if kind=='vessel' and form=='horizontal':categories.add('horizontal_vessel')
        if kind=='fixed_bed':categories.add('slender')
        if kind in ('exchanger','condenser') and form=='horizontal':categories.add('horizontal_auxiliary')
    return categories,forms


def manifest(seed=20261111,count=20):
    from .equipment_sizing import select_equipment
    requests=batch_recipes(seed,count);categories=Counter();forms=Counter();main_counts=Counter()
    selected=[]
    for recipe,room_seed in requests:
        sizing=select_equipment(recipe,room_seed)
        found,typed=requested_coverage(sizing)
        categories.update(found);forms.update('|'.join(map(str,row)) for row in typed)
        main_counts[sizing['main_equipment_count']]+=1
        selected.append({'recipe':deepcopy(recipe),'seed':room_seed,'sizing':sizing})
    return {'version':'equipment_sampling_v1','stage':'sizing','batch_seed':seed,'count':count,
            'sampling':'Build twenty transfer-mode and unit-count slots with main equipment form and size. Derive auxiliary sizes from the source reactor, then shuffle to assign seeds without layout feedback or resampling.',
            'strata':[deepcopy(row) for row in STRATA],
            'coverage_counts_are_requests_only':True,'requested_coverage_rooms':dict(categories),
            'requested_equipment_forms_and_tiers':dict(forms),
            'main_equipment_count_distribution':{str(key):value for key,value in sorted(main_counts.items())},
            'rooms_with_at_least_three_main_equipment':sum(value for key,value in main_counts.items() if key>=3),
            'selection_counts':dict(Counter(row['sizing']['status'] for row in selected)),
            'actual_coverage':'No room is generated at this stage. Five-category coverage counts only main equipment validated in completed rooms.',
            'requests':selected}


def main():
    from .evaluation_common import require_environment
    require_environment()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise SystemExit('Choose a new empty directory to keep the saved inputs')
    output.mkdir(parents=True,exist_ok=True)
    data=manifest(args.seed,args.count)
    (output/'definitions.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# Frozen recipe inputs and equipment selection','',f'Development seed{args.seed}；{args.count} inputs. Equipment selection only.','',
           'Select the main equipment form and size first. Condensers, exchangers and fixed beds follow the source reactor size. Small uses small, large uses large, and medium may use small or large.',
           'Reject invalid size overrides and oversized distillation receivers. A filter without a catalogue tier is rejected. Pumps use the existing approximation with no size tier.','',
           '| Transfer mode | Unit-count relation | Requested rooms |','|---|---|---:|']
    lines += [f'| {r["mode"]} | {r["relation"]} | {r["count"]} |' for r in STRATA]
    lines += ['','| Equipment input category | Room count |','|---|---:|']
    lines += [f'| {key} | {value} |' for key,value in data['requested_coverage_rooms'].items()]
    lines += ['',f'Main equipment count excluding pumps: {data["main_equipment_count_distribution"]}. Rooms with at least three units: {data["rooms_with_at_least_three_main_equipment"]} rooms.',
              f'Selection results: {data["selection_counts"]}. See definitions.json for recipes, seeds, dimensions, dependencies and checks.',
              'Counts describe saved requests and equipment choices. Actual room completion and coverage are measured after generation. Rejected inputs remain fixed for layout retries.','']
    (output/'sampling-notes.md').write_text('\n'.join(lines))
    print(json.dumps({key:data[key] for key in ('version','count','requested_coverage_rooms',
        'main_equipment_count_distribution','rooms_with_at_least_three_main_equipment','selection_counts')},ensure_ascii=False))




def batch_recipes(seed, count):
    from .supported_recipes import batch_recipes as supported
    return supported(seed,count)

if __name__=='__main__':
    main()
