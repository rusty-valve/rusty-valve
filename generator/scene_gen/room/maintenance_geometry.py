"""Measure the space an equipment unit needs for maintenance."""
from __future__ import annotations

import math
import numpy as np

from .process import RULES
from .rule_identity import band

EPS = 1e-8


def _cylinder(component):
    if '_actual_geom_type' in component:
        import mujoco
        if component['_actual_geom_type'] != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            return None
        axis=np.asarray(component['_actual_matrix'])[:,2]
        centre=np.asarray(component['_actual_center'])
        half=component['_actual_size'][1]
        ends=[(centre-axis*half).tolist(),(centre+axis*half).tolist()]
    elif component.get('shape')=='cylinder' and len(component.get('ends',[]))==2:
        ends=component['ends']
        direction=np.asarray(ends[1])-ends[0]
        length=float(np.linalg.norm(direction))
        if length<=EPS:return None
        axis=direction/length
    else:return None
    return {'ends':ends,'vertical':abs(float(axis[2]))>=1-EPS,
            'length':math.dist(*ends)}


def _support(row,scene):
    lugs=[c for c in row.get('parts',[]) if c['kind']=='lug']
    shoes=[c for c in row.get('parts',[]) if c['kind']=='support']
    evidence=[]
    if not lugs:return None,evidence
    z=lugs[0]['box'][2]
    for lug in lugs:
        b=lug['box']
        contacts=[c['id'] for c in shoes if abs(c['box'][5]-b[2])<=EPS and
            all(min(c['box'][i+3],b[i+3])-max(c['box'][i],b[i])>EPS for i in (0,1))]
        evidence.append({'lug':lug['id'],'underside_z_m':b[2],'contacting_shoes':contacts})
        if abs(b[2]-z)>EPS or not contacts:return None,evidence
    aligned=[height for height in scene['floors'][1:] if abs(height-z)<=EPS]
    return (float(aligned[0]) if aligned else None),evidence


def _nearest_upper(scene,z):
    shell=scene.get('shell',{})
    count=shell.get('platform_limit',shell.get('platforms',len(scene['floors'])-1))
    step=shell.get('storey_height_m')
    floors=([scene['floors'][0]+i*step for i in range(count+1)] if step is not None else scene['floors'])
    low,high=band()
    distances=[max(floor+low-z,z-floor-high,0.) for floor in floors]
    nearest=min(distances)
    return [i for i,distance in enumerate(distances) if abs(distance-nearest)<=EPS]


def _point(value):
    return isinstance(value,(list,tuple)) and len(value)==3 and all(
        isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in value)


def level_maintenance(row,unit,scene,level,rules,planning=False):
    """Use the same physical cross-section during placement and measurement."""
    from .building_equipment_checks import maintenance_width_for
    base={'rect':list(row['rect']) if row.get('rect') is not None else None,
          'width_m':maintenance_width_for(unit,rules),'component_ids':list(row.get('component_ids',[])),
          'z_band_m':None,'applies':False,'reason':'original full equipment projection and service clearance',
          'support_z_m':None,'purposes':{},'support_evidence':[],'upper_candidate_levels':[]}
    if level!=0 or unit.get('kind')!='vessel' or not row.get('rect'):
        return base
    bodies=[_cylinder(c) for c in row.get('parts',[]) if c['kind']=='vessel']
    bodies=[c for c in bodies if c is not None]
    if not bodies or not max(bodies,key=lambda c:c['length'])['vertical']:
        base['reason']='Not the physical main cylinder of a vertical reactor';return base
    support,evidence=_support(row,scene)
    base.update(support_z_m=support,support_evidence=evidence)
    if support is None or support<=scene['floors'][0]+EPS:
        base['reason']='Physical lugs or shoes do not align with a platform bearing face above ground';return base
    ports=[p['pos'] for p in unit.get('ports',{}).values() if _point(p.get('pos'))]
    if not ports:
        base['reason']='Missing physical port geometry';return base
    nearest=_nearest_upper(scene,max(p[2] for p in ports))
    base['upper_candidate_levels']=nearest
    if 0 in nearest:
        base['reason']='Ground level is among the nearest upper-end operating candidates. Keep maintenance clearance at the body level.';return base
    if 0 not in _nearest_upper(scene,min(p[2] for p in ports)):
        base['reason']='The lower port is not used from ground level. Keep the full-unit service envelope.';return base
    uses={'lower':0,'upper_candidates':nearest,'source':'Measured port elevations, bearing plane and ground clearance band'}
    lo=scene['floors'][0];hi=lo+RULES['geometry_screen']['walk_height_m']
    parts=[c for c in row.get('parts',[]) if c['box'][5]>lo+EPS and c['box'][2]<hi-EPS]
    base.update(z_band_m=[lo,hi],purposes=uses)
    if not parts:
        base['reason']='No physical component intersects the clearance band. Keep the full footprint and width.';return base
    rect=[min(c['box'][0] for c in parts),min(c['box'][1] for c in parts),
          max(c['box'][3] for c in parts),max(c['box'][4] for c in parts)]
    base.update(rect=rect,width_m=rules['ground_bottom_maintenance_width_m'],
        component_ids=[c['id'] for c in parts],applies=True,
        reason='For a vertical vessel supported above ground with upper operation above ground and lower operation at ground, retain the complete XY projection of parts intersecting the clearance band.')
    return base
