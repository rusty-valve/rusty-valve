"""Choose two compatible routes for a condenser's cooling pipes."""
from copy import deepcopy
from collections import Counter
import math

from .piping_header_taps import adjacent_reservation


def _shift(a, point, header, tap, taps, distance):
    axis=header['axis_index'];result=deepcopy(tap)
    original=tap['position'][axis];value=original+distance
    ends=sorted((header['start'][axis],header['end'][axis]))
    margin=.039+tap['reservation']['end_reserve_m']
    low,high=ends[0]+margin,ends[1]-margin
    for pid in header['point_ids']:
        if pid==point['id']:continue
        other=taps[pid][1];position=other['position'][axis]
        gap=adjacent_reservation(tap['reservation'],other['reservation'])['required_center_distance_m']
        if position<original:low=max(low,position+gap)
        else:high=min(high,position-gap)
    if not low-1e-8<=value<=high+1e-8:return None
    result['position'][axis]=value
    result['selected_projection_m']=tap['selected_projection_m']+distance
    return result


def connect_groups(a,work,taps,headers,heights):
    from .piping_method import _connect_service
    guides=[];failures=[];done=set()
    for point in work:
        if point['id'] in done:continue
        from .generation_progress import checkpoint
        checkpoint(a,{'stage':'service_connection','edge':point['id'],'endpoint':point['endpoint']})
        unit=a.frozen['graph']['units'][point['equipment']]
        pair=[p for p in work if p['equipment']==point['equipment'] and
              p['medium'] in ('cold_source','cold_return','water','water_return')]
        grouped=(point['medium'] in ('cold_source','cold_return','water','water_return')
            and unit['kind']=='condenser' and len(pair)==2 and len({p['dn'] for p in pair})==1)
        if not grouped:
            guide,failure=_connect_service(a,point,*taps[point['id']],headers,heights)
            if guide:guides.append(guide)
            if failure:failures.append(failure)
            done.add(point['id']);continue
        initial=deepcopy(a.__dict__);choices=[];records=[]
        step=8*a.radius(point['dn'])+.05
        shifts=[(0.,0.),(step,0.),(-step,0.),(0.,step),(0.,-step),(step,-step),(-step,step)]
        for shift in shifts:
            selected={p['id']:_shift(a,p,*taps[p['id']],taps,delta) for p,delta in zip(pair,shift)}
            if any(t is None for t in selected.values()):continue
            for order in (pair,pair[::-1]):
                for extra in ((0.,step) if shift==(0.,0.) else (0.,)):
                    a.__dict__=deepcopy(initial);trial=[];errors=[]
                    for index,p in enumerate(order):
                        h=taps[p['id']][0];tap=selected[p['id']]
                        a.header_tap_plans[h['id']]['device_taps'][p['id']]=deepcopy(tap)
                        guide,failure=_connect_service(a,p,h,tap,headers,heights,top_entry_extra_m=extra if index==0 else 0.)
                        if guide:trial.append(guide)
                        if failure:errors.append(failure)
                    bends=sum(g['diagnostics']['actual_bends'] for g in trial)
                    length=sum(math.dist(x,y) for g in trial for arm in ('lead','tail') for x,y in zip(g[arm],g[arm][1:]))
                    row={'order':[p['id'] for p in order],'tee_axis_shifts_m':list(shift),'first_top_entry_extra_m':extra,
                         'connected':len(trial),'actual_bends':bends,'length_m':length,
                         'failures':[e['reason'] for e in errors]}
                    records.append(row)
                    choices.append(((-len(trial),bends,length,len(choices)),deepcopy(a.__dict__),trial,errors,selected,row))
        _,state,trial,errors,selected,chosen=min(choices,key=lambda row:row[0])
        a.__dict__=state
        for p in pair:
            h=taps[p['id']][0];tap=selected[p['id']]
            taps[p['id']]=(h,deepcopy(tap))
            h['method_joint_taps']['device_taps'][p['id']]=deepcopy(tap)
        a.scene.setdefault('cooling_group_selection',[]).append({'equipment':point['equipment'],
            'selected':chosen,'candidates':records,'scope':'joint finite search; same real nozzles, tee order and spacing'})
        guides.extend(trial);failures.extend(errors);done.update(p['id'] for p in pair)
    counts=Counter(row['requirement'] for row in [*guides,*failures])
    expected={point['id'] for point in work}
    coverage={'expected':sorted(expected),'processed_counts':dict(counts),
              'exactly_once':set(counts)==expected and all(n==1 for n in counts.values())}
    a.scene['service_connection_coverage']=coverage
    if not coverage['exactly_once']:raise RuntimeError('Required service taps were omitted or duplicated: '+str(coverage))
    return guides,failures
