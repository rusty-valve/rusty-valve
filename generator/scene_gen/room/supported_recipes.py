"""Choose example recipes with different transfer modes and equipment."""
from collections import Counter
from copy import deepcopy
import random
from .material_support import load,ensure_supported,canonical_transfer_mode
from .storage_vessel import HORIZONTAL_VESSEL_TIER


def batch_recipes(seed,count):
    support=load();rng=random.Random(seed);templates=[]
    assert set(support['batch_jacket_media'])<=set(support['jacket_media'])
    for stratum in support['batch_strata']:
        mode=stratum['mode'];self_return=stratum['self_return']
        assert mode in support['transfer_modes']
        for i in range(stratum['count']):
            n=(1+i%2) if self_return else (2+i%2)
            tier=('small','medium','large','small','medium')[i%5]
            vessels=[]
            for j in range(n):
                form='vertical' if j==0 or self_return else ('horizontal' if (i+j)%2 else 'vertical')
                selected=tier if j==0 else rng.choice(('small','medium','large'))
                if form == 'horizontal':
                    selected = HORIZONTAL_VESSEL_TIER
                v={'id':f'R{j+1}','form':form,'size_tier':selected,'reactor':form=='vertical','jacket':False}
                if form=='vertical' and (i+j)%2==0:
                    v.update(jacket=True,jacket_medium=support['batch_jacket_media'][(i+j)//2%len(support['batch_jacket_media'])])
                vessels.append(v)
            # Companion tiers remain source-led in the existing sizing step.
            if i%3!=2 or self_return:
                vessels[0]['reflux_condenser']={'form':support['reflux_condenser_forms'][i%2]}
            pairs=([(j,j) for j in range(n)] if self_return else
                   [(0,1)] if n==2 else [(0,1),(0,2)] if i%4==1 else [(0,2),(1,2)])
            transfers=[]
            for j,(a,b) in enumerate(pairs):
                t={'source':vessels[a]['id'],'target':vessels[b]['id'],'mode':mode}
                if mode=='pump':
                    t['protection_filter']={'form':'horizontal'}
                    if j==0:
                        allowed=support['recycle_auxiliaries'][0]
                        t['recycle_via']=[{'kind':allowed['kind'],'service':allowed['service'],'form':allowed['forms'][i%2]}]
                transfers.append(t)
            recipe={'id':'pending','relation':'self_return' if self_return else ('one_to_one' if n==2 else 'one_to_many' if i%4==1 else 'many_to_one'),
                    'description':'Sample supported transfer modes, equipment counts, sizes and forms. The generator chooses positions.',
                    'vessels':vessels,'transfers':transfers}
            ensure_supported(recipe);templates.append(recipe)
    result=[]
    while len(result)<count:
        block=deepcopy(templates);rng.shuffle(block)
        for recipe in block:
            if len(result)>=count:break
            recipe['id']=f'room_{len(result)+1:02}'
            result.append((recipe,rng.randrange(1,10_000_000)))
    return result


def coverage(requests):
    counters=Counter();rows=[]
    for recipe,seed in requests:
        flags=set()
        for t in recipe['transfers']:
            flags.add('self_return' if t['source']==t['target'] else canonical_transfer_mode(t['mode']))
            for aux in t.get('recycle_via',[]):flags.add('heater_'+aux['form'])
        for v in recipe['vessels']:
            if v.get('reflux_condenser'):flags.add('condenser_'+v['reflux_condenser']['form'])
            if v.get('jacket'):flags.add('jacket_'+v['jacket_medium'])
            flags.add('vessel_'+v['form']);flags.add('tier_'+v['size_tier'])
        flags.add('vessel_count_'+str(len(recipe['vessels'])))
        counters.update(flags);rows.append({'id':recipe['id'],'seed':seed,'coverage':sorted(flags)})
    required=['pump','self_return','no_pump','heater_horizontal','heater_vertical',
              'condenser_horizontal','condenser_vertical','jacket_steam','jacket_cold_source']
    return {'scope':'Input coverage','counts':dict(counters),'required':required,
            'missing':[k for k in required if not counters[k]],'rooms':rows}
