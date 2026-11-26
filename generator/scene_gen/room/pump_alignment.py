"""Check that pumps on a floor sit in a properly spaced row."""
from __future__ import annotations

import math
import numpy as np


def _actual(part):
    required=('_actual_geom_type','_actual_matrix','_actual_size','_actual_center')
    return all(key in part for key in required)


def _extent(part,direction):
    """Exact projection support of the stage's actual box/cylinder primitives."""
    import mujoco
    matrix=np.asarray(part['_actual_matrix']);size=np.asarray(part['_actual_size'])
    centre=np.asarray(part['_actual_center']);local=matrix.T@np.asarray(direction)
    kind=part['_actual_geom_type']
    if kind==int(mujoco.mjtGeom.mjGEOM_BOX):half=float(np.dot(np.abs(local),size))
    elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half=float(size[0]*math.hypot(local[0],local[1])+size[1]*abs(local[2]))
    elif kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):half=float(size[0])
    elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):half=float(size[0]+size[1]*abs(local[2]))
    elif kind==int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):half=float(np.linalg.norm(local*size))
    else:raise ValueError('Pump row has a part whose projection cannot be measured from the XML primitives: '+part['id'])
    at=float(np.dot(centre,direction));return [at-half,at+half]


def _union_projection(parts,direction):
    ranges=[_extent(part,direction) for part in parts]
    return [min(r[0] for r in ranges),max(r[1] for r in ranges)]


def check_pump_alignment(rows,instances,width):
    """Use real shaft, motor, foundation face and complete solid outline.

    ``instances`` remains in the caller contract, but its row, axis, front and
    motor declarations never decide the result. The stage's actual ground
    datum is Z=0; independent support/maintenance checks remain in check_layout.
    """
    import mujoco
    from .building_equipment_checks import EPS
    measured=[];violations=[];records=[]
    for row in rows:
        if row['kind']!='pump':continue
        parts=row.get('parts',[])
        bodies=[p for p in parts if p['kind']=='pump' and _actual(p)
                and p['_actual_geom_type']==int(mujoco.mjtGeom.mjGEOM_CYLINDER)]
        motors=[p for p in parts if p['kind']=='motor' and _actual(p)]
        bases=[p for p in parts if p['kind'] in ('plinth','foundation','base') and _actual(p)]
        if len(bodies)!=1 or not motors or not bases or not parts or any(not _actual(p) for p in parts):
            violations.append({'code':'PUMP_GEOMETRY_INCOMPLETE','equipment':row['id'],
                'message':'Measure pump shaft, motor, base and full envelope from physical XML geometry'})
            continue
        body=bodies[0];shaft3=np.asarray(body['_actual_matrix'])[:,2].copy()
        if abs(shaft3[2])>EPS or np.linalg.norm(shaft3[:2])<=EPS:
            violations.append({'code':'PUMP_AXES_NOT_HORIZONTAL','equipment':row['id'],'actual_axis':shaft3.tolist()})
            continue
        axis=shaft3[:2]/np.linalg.norm(shaft3[:2])
        if axis[0]<-EPS or (abs(axis[0])<=EPS and axis[1]<0):axis=-axis
        centre=np.asarray(body['_actual_center']);motor_centres=[np.asarray(p['_actual_center']) for p in motors]
        motor_offsets=[float(np.dot(p[:2]-centre[:2],axis)) for p in motor_centres]
        signs={1 if value>EPS else -1 if value<-EPS else 0 for value in motor_offsets}
        motor_side=next(iter(signs)) if len(signs)==1 else 0
        if not motor_side:
            violations.append({'code':'PUMP_MOTOR_SIDES_DIFFER','equipment':row['id'],
                               'message':'The physical motor is not on an identifiable common axial end','actual_offsets_m':motor_offsets})
        directed=axis*(motor_side or 1);normal=np.array([-directed[1],directed[0]])
        e=np.array([*directed,0.]);n=np.array([*normal,0.])
        base_faces=[];valid_faces=True
        for base in bases:
            if base['_actual_geom_type']!=int(mujoco.mjtGeom.mjGEOM_BOX):valid_faces=False;continue
            matrix=np.asarray(base['_actual_matrix'])
            aligned=max(abs(float(np.dot(matrix[:,j],e))) for j in range(3))
            if abs(aligned-1)>EPS:valid_faces=False
            bounds=_extent(base,e)
            base_faces.append({'component':base['id'],'head_end_coordinate_m':bounds[0],
                               'actual_face_normal_towards_motor':e.tolist(),
                               'face_perpendicular_to_shaft':abs(aligned-1)<=EPS})
        if not valid_faces:
            violations.append({'code':'PUMP_BASE_FACE_NOT_PERPENDICULAR','equipment':row['id']})
        try:
            axial=_union_projection(parts,e);transverse=_union_projection(parts,n)
            base_axial=_union_projection(bases,e);base_transverse=_union_projection(bases,n)
        except ValueError as exc:
            violations.append({'code':'PUMP_GEOMETRY_INCOMPLETE','equipment':row['id'],'message':str(exc)})
            continue
        levels=row.get('occupied_levels',[]);bottom=min(_extent(base,np.array([0.,0.,1.]))[0] for base in bases)
        if not levels or levels[0]!=0 or abs(bottom)>EPS:
            violations.append({'code':'PUMP_NOT_ON_GROUND','equipment':row['id'],
                               'actual_occupied_levels':levels,'actual_base_bottom_z_m':bottom,'ground_z_m':0.})
        result={'id':row['id'],'level':levels[0] if levels else None,'axis':axis.tolist(),
                'actual_shaft_axis_3d':shaft3.tolist(),'motor_side':motor_side,
                'head_to_motor_axis':directed.tolist(),'row_direction':normal.tolist(),
                'actual_pump_centre_m':centre.tolist(),'actual_motor_centres_m':[p.tolist() for p in motor_centres],
                'base_front_m':base_axial[0],'base_head_end_faces':base_faces,
                'complete_axial_interval_m':axial,'complete_transverse_interval_m':transverse,
                'actual_base_bottom_z_m':bottom,'footprint':row.get('rect'),
                'component_ids':[p['id'] for p in parts],
                'measurement_source':'loaded XML geom_type/size/position/matrix; row/front/ends metadata ignored'}
        measured.append(result)
        records.append({'row':result,'parts':parts,'bases':bases,
                        'signature':[float(body['_actual_size'][0]),float(body['_actual_size'][1]),
                                     base_axial[1]-base_axial[0],base_transverse[1]-base_transverse[0]]})
    for level in sorted({r['row']['level'] for r in records if r['row']['level'] is not None}):
        group=[r for r in records if r['row']['level']==level];reference=group[0]['row']
        axis=np.asarray(reference['axis']);directed=np.asarray(reference['head_to_motor_axis'])
        e=np.array([*directed,0.]);n=np.array([-directed[1],directed[0],0.])
        if any(abs(abs(float(np.dot(r['row']['axis'],axis)))-1)>EPS for r in group):
            violations.append({'code':'PUMP_AXES_NOT_PARALLEL','level':level})
            continue
        if any(r['row']['motor_side']!=reference['motor_side'] or not r['row']['motor_side'] for r in group):
            violations.append({'code':'PUMP_MOTOR_SIDES_DIFFER','level':level})
        fronts=[]
        for record in group:
            front=_union_projection(record['bases'],e)[0]
            side=_union_projection(record['parts'],n)
            record['row'].update(common_head_end_front_m=front,common_transverse_interval_m=side)
            fronts.append(front)
        if max(fronts)-min(fronts)>EPS:
            violations.append({'code':'PUMP_BASE_FRONTS_NOT_ALIGNED','level':level,'fronts_m':fronts,
                               'message':'A floor allows one pump row. Foundation pump-end faces must align on actual shafts.'})
        for index,first in enumerate(group):
            for second in group[index+1:]:
                same_size=all(abs(a-b)<=EPS for a,b in zip(first['signature'],second['signature']))
                displacement=np.asarray(second['row']['actual_pump_centre_m'][:2])-first['row']['actual_pump_centre_m'][:2]
                along=float(np.dot(displacement,axis))
                if same_size and abs(along)>EPS:
                    violations.append({'code':'PUMP_NOT_SIDE_BY_SIDE','level':level,
                        'pumps':[first['row']['id'],second['row']['id']],
                        'actual_axial_centre_offset_m':along,
                        'message':'Same-size pump centers must lie in a line perpendicular to actual shafts'})
        ordered=sorted(group,key=lambda r:r['row']['common_transverse_interval_m'][0])
        for first,second in zip(ordered,ordered[1:]):
            gap=second['row']['common_transverse_interval_m'][0]-first['row']['common_transverse_interval_m'][1]
            first['row']['next_side_gap_m']=gap;first['row']['next_side_pump']=second['row']['id']
            if gap<width-EPS:
                violations.append({'code':'PUMP_SIDE_GAP_SHORT','level':level,
                    'pumps':[first['row']['id'],second['row']['id']],
                    'actual_gap_m':gap,'required_m':width,
                    'message':'The complete pump, motor and base envelopes are too close laterally'})
    return {'status':'fail' if violations else 'pass','rows':measured,'violations':violations,
            'allowed_rows_per_level':1,'minimum_side_gap_m':width,
            'scope':'Use actual XML shaft, motor and base faces. Pumps sit in one side-by-side row per floor.'}
