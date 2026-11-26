"""Arrange a filter and leave room for its bypass pipes."""
import numpy as np
from .piping_assets import inspect_asset
from .pipe_spec import load


def geometry(pin,pout,dn):
    first,last=np.asarray(pin,float),np.asarray(pout,float)
    axis=(last-first)/np.linalg.norm(last-first)
    radius=load()['pipe_outer_diameter_mm'][str(dn)]/2000
    length=inspect_asset(dn)['install_length_m']
    margin=3*length+4*radius+.05
    return {'axis':axis,'split':first-axis*margin,'merge':last+axis*margin,
            'radius':radius,'valve_length':length}


def upstream_entry(group,source_normal,has_root,direction=None):
    # Same terminal reservation as _bridge: a downward root needs its real
    # three valve lengths on the final run, followed by a real R=3r turn.
    r=group['radius'];straight=2*r+.05
    if has_root and source_normal[2]<-.5:
        straight=max(straight,3*group['valve_length']+.02)
    direction=-group['axis'] if direction is None else np.asarray(direction)
    start=group['split']+direction*(straight+3*r)
    return {'start':start.tolist(),'end':group['split'].tolist(),
            'straight_m':straight,'elbow_radius_m':3*r,
            'box':np.r_[np.minimum(start,group['split'])-r,
                        np.maximum(start,group['split'])+r].tolist()}


def definitions(graph):
    rows={}
    for uid,u in graph['units'].items():
        if u['kind'] not in ('filter','protective_filter'):continue
        incoming=next((e for e in graph['edges'] if e['b']==uid+'.in'),None)
        outgoing=next((e for e in graph['edges'] if e['a']==uid+'.out'),None)
        if not incoming or not outgoing:continue
        left,right=incoming['a'].split('.')[0],outgoing['b'].split('.')[0]
        if not any(e['a'].split('.')[0]==left and e['b'].split('.')[0]==right for e in graph['edges']):continue
        source=next((e for e in graph['edges'] if e['b'].split('.')[0]==left),None)
        if source is None:continue
        owner,port=source['a'].split('.')
        physical=graph['units'][owner]['ports'][port]
        rows[uid]={'dn':incoming['dn'],'source':source['a'],
                   'source_normal':physical['normal'],
                   'has_root':any(v['at']==source['a'] for v in graph['valves'])}
    return rows
