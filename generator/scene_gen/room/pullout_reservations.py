"""Keep space clear above reactors for lifting out the agitator."""
from __future__ import annotations
from .runtime_paths import DATA

import json

EPS=1e-8
REFERENCE=DATA/'reactor_f_series_dimensions.json'
STRUCTURE={'column','beam','equipment_beam','floor','ground','walk_surface','step','landing','stringer','rail'}


def _rect(box):return [box[0],box[1],box[3],box[4]]


def _overlap(a,b):
    return all(min(a[i+2],b[i+2])-max(a[i],b[i])>EPS for i in (0,1))


def _same(a,b):
    return isinstance(a,(list,tuple)) and len(a)==len(b) and all(abs(x-y)<=EPS for x,y in zip(a,b))


def drive_mount_flange(model_key,reference=None):
    """Return a sourced central-mount flange OD, explicitly estimating if absent.

    The estimate is a scene envelope, not a manufacturer flange dimension.
    Neither the shaft diameter dN nor internal blade dimensions d4/d5 are used.
    """
    reference=json.loads(REFERENCE.read_text()) if reference is None else reference
    if model_key not in reference.get('models',{}):
        raise ValueError(f'PULLOUT_MODEL_REFERENCE_MISSING: missing verifiable manufacturer model: {model_key!r}')
    mount=reference['models'][model_key].get('drive_mount',{})
    dn=mount.get('dn')
    if isinstance(dn,bool) or not isinstance(dn,(int,float)) or dn<=0:
        raise ValueError(f'PULLOUT_CENTRAL_DN_MISSING: {model_key}Missing central drive_mount.dn')
    values=[]
    for key,scale in (('flange_outer_diameter_m',1.),('flange_od_m',1.),
                      ('flange_outer_diameter_mm',.001),('flange_od_mm',.001)):
        value=mount.get(key)
        if value is not None:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or value<=0:
                raise ValueError(f'PULLOUT_FLANGE_OD_INVALID: {model_key}.drive_mount.{key}')
            values.append((key,float(value)*scale))
    if values and any(abs(value-values[0][1])>EPS for _,value in values):
        raise ValueError(f'PULLOUT_FLANGE_OD_CONFLICT: {model_key}Central flange OD fields disagree')
    estimated=not values
    diameter=values[0][1] if values else .001*float(dn)+.160
    if diameter<=.001*float(dn):
        raise ValueError(f'PULLOUT_FLANGE_OD_INVALID: {model_key}Flange outside diameter must exceed nominal diameter')
    return {'central_mount_dn':dn,'flange_outer_diameter_m':diameter,
            'flange_outer_diameter_estimated':estimated,
            'factory_reference':str(REFERENCE),
            'nominal_dn_field':f'models.{model_key}.drive_mount.dn',
            'flange_od_field':f'models.{model_key}.drive_mount.{values[0][0]}' if values else None,
            'flange_od_basis':('Estimated flange OD as 0.001 times central port DN plus 0.160 m'
                               if estimated else 'manufacturer central mounting flange OD'),
            'estimate_formula':'0.001 * central_mount_dn + 0.160' if estimated else None}


def expected_pullout(row,unit,scene,rules):
    """Derive the axis, body top and fixed cross-section from actual parts."""
    from .building_equipment_checks import _main_body
    if unit.get('kind',row['kind']) not in ('vessel','tank'):return None
    _,body_measurement=_main_body(row)
    drives=[c for c in row.get('parts',[]) if c['kind']=='drive_reservation']
    shells=[c for c in row.get('parts',[]) if c['kind'] in ('vessel','tank','head')]
    if not body_measurement or body_measurement['form']!='vertical' or not drives or not shells:return None
    centre=[sum(p[i] for p in body_measurement['ends'])/2 for i in (0,1)]
    top=max(c['box'][5] for c in shells)
    drive_span=max(c['box'][i+3]-c['box'][i] for c in drives for i in (0,1))
    original=json.loads(REFERENCE.read_text())
    matched=[(key,value) for key,value in original['models'].items()
             if abs(value['dimensions_mm']['d2']/1000-body_measurement['diameter_m'])<=EPS]
    if len(matched)!=1:
        raise ValueError('PULLOUT_MODEL_REFERENCE_MISSING: physical vessel diameter does not uniquely identify a manufacturer mounting model')
    raw=matched[0][1]['dimensions_mm'];model_key=matched[0][0]
    declared_model=(unit.get('dimension_spec') or unit.get('catalog') or {}).get('model')
    if declared_model is not None and declared_model!=model_key:
        raise ValueError(f'PULLOUT_MODEL_REFERENCE_MISMATCH: declared {declared_model} differs from the model indicated by measured diameter {model_key} differ')
    flange=drive_mount_flange(model_key,original);span=flange['flange_outer_diameter_m']
    total_margin=.1
    side=span+total_margin
    rect=[centre[0]-side/2,centre[1]-side/2,centre[0]+side/2,centre[1]+side/2]
    for component in row.get('parts',[]):
        if (component.get('kind')=='equipment_nozzle' and component['box'][5]>top+EPS
                and _overlap(rect,_rect(component['box']))):
            raise ValueError(f'PULLOUT_COVERS_TOP_NOZZLE: central flange extraction reservation covers a top port: {component["id"]}')
    data=matched[0][1]
    support=min(c['box'][2] for c in shells)-(raw['h4']-raw['h1'])/1000
    base=support+(raw['h4']+raw['h2'])/1000
    base_source='Use the physical body and catalogue h1/h4 for the bearing plane, and h4 plus h2 for drive installation height'
    drive_top=support+(raw['h4']+raw['h2']+data['reference_drive_and_overall_mm']['h5'])/1000
    allowed_drive={'id':row['id']+'_drive_reservation','owner':row['id'],
                   'relative_xy_box':[-.16,-.16,.16,.16],'z_range_m':[top,drive_top],
                   'scope':'Allow only the original fixed 0.32 m drive mount box with matching geometry, ID and bounds'}
    return {'id':row['id']+'_pullout','owner':row['id'],'rect':rect,'axis_xy':centre,
        'base_z':base,'open_top':True,'side_m':side,
        'allowed_drive_installation':allowed_drive,
        'source':{'rule':'BL-6E','factory_reference':str(REFERENCE),'factory_model_from_actual_d2':model_key,
            **flange,'reference_span_m':span,'actual_drive_span_m':drive_span,
            'total_margin_m':total_margin,'edge_margin_m':total_margin/2,
            'basis':'Central agitator flange OD plus 0.1 m overall. Estimate from DN only when OD is unavailable.',
            'base_source':base_source,'actual_body_top_m':top,
            'scope':'Open upward from the drive mount reference plane. This is a clearance volume.'},
        'actual_drive_components':[c['id'] for c in drives],
        'actual_body_components':[c['id'] for c in shells]}


def _allowed_drive(record,component):
    allowed=record.get('allowed_drive_installation',{})
    if (component.get('id')!=allowed.get('id') or component.get('owner')!=allowed.get('owner')
            or component.get('kind')!='drive_reservation'):
        return False
    if '_actual_geom_type' in component:
        import mujoco
        if component['_actual_geom_type']!=int(mujoco.mjtGeom.mjGEOM_BOX):return False
    elif component.get('shape')!='box':return False
    x,y=record['axis_xy'];r=allowed['relative_xy_box'];lo,hi=allowed['z_range_m']
    expected=[x+r[0],y+r[1],lo,x+r[2],y+r[3],hi]
    return _same(component['box'],expected)


def placement_clear(record,components,owner=None):
    """Conservative full-XY-solid test above base, without a finite upper cap."""
    for component in components:
        if _allowed_drive(record,component):continue
        if component['box'][5]>record['base_z']+EPS and _overlap(record['rect'],_rect(component['box'])):
            return False
    return True


def _all_xml_components(scene,model):
    """Every real XML geom is an obstacle, irrespective of display/contact masks."""
    import mujoco
    from .checks import geom_bounds
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    bounds=geom_bounds(model,data,list(range(model.ngeom)))
    source={c['id']:c for c in scene['components']};result=[]
    for gid in range(model.ngeom):
        name=model.geom(gid).name or f'xml_geom_{gid}'
        item={**source.get(name,{'id':name,'kind':'unknown','owner':None})}
        item.update(box=bounds[gid].tolist(),_actual_geom_type=int(model.geom_type[gid]),
            _actual_size=model.geom_size[gid].tolist(),_actual_center=data.geom_xpos[gid].tolist(),
            _actual_matrix=data.geom_xmat[gid].reshape(3,3).tolist(),
            _actual_contype=int(model.geom_contype[gid]),_actual_conaffinity=int(model.geom_conaffinity[gid]))
        result.append(item)
    return result


def _covered_segment(lo,hi,segments):
    end=lo
    for a,b in sorted(segments):
        if b<=end+EPS:continue
        if a>end+EPS:return False
        end=max(end,b)
    return end>=hi-EPS


def _opening_edges(rect,boards):
    result={}
    for name,axis,sign in (('x_min',0,-1),('x_max',0,1),('y_min',1,-1),('y_max',1,1)):
        other=1-axis;edge=rect[axis if sign<0 else axis+2]
        segments=[]
        for board in boards:
            b=_rect(board['box'])
            board_edge=b[axis+2 if sign<0 else axis]
            if abs(board_edge-edge)>EPS:continue
            lo,hi=max(rect[other],b[other]),min(rect[other+2],b[other+2])
            if hi-lo>EPS:segments.append([lo,hi])
        result[name]={'coordinate_m':edge,'actual_board_edge_segments_m':segments,
                      'covered':_covered_segment(rect[other],rect[other+2],segments)}
    return result


def check_pullout_reservations(graph,scene,model=None):
    """Remeasure saved XML; never use a declared small channel as the ruler."""
    if model is None:
        return {'status':'fail','rows':[],'violations':[{'code':'PULLOUT_XML_MODEL_REQUIRED',
                'message':'Remeasure the extraction passage from the actual XML.'}]}
    from .building_equipment_checks import read_layout_rules
    rules=read_layout_rules();components=_all_xml_components(scene,model)
    footprints=[]
    for uid,unit in graph['units'].items():
        if unit['kind'] not in ('vessel','tank'):continue
        parts=[c for c in components if c.get('owner')==uid or c['id']==uid]
        footprints.append({'id':uid,'kind':unit['kind'],'parts':parts})
    declared=scene.get('pullout_reservations',[])
    measurements=[];errors=[];owners=set()
    for row in footprints:
        expected=expected_pullout(row,graph['units'][row['id']],scene,rules)
        if expected is None:continue
        uid=row['id'];owners.add(uid);local=[]
        records=[r for r in declared if r.get('owner')==uid]
        record=records[0] if len(records)==1 else None
        valid=bool(record and record.get('id')==expected['id'] and record.get('open_top') is True
                   and _same(record.get('rect'),expected['rect']) and _same(record.get('axis_xy'),expected['axis_xy'])
                   and isinstance(record.get('base_z'),(int,float)) and abs(record['base_z']-expected['base_z'])<=EPS)
        if not valid:local.append({'code':'PULLOUT_RESERVATION_INVALID','equipment':uid,
                                   'declared':records,'expected':expected})
        obstacles=[]
        for component in components:
            if _allowed_drive(expected,component):continue
            if component['box'][5]>expected['base_z']+EPS and _overlap(expected['rect'],_rect(component['box'])):
                obstacles.append({'id':component['id'],'owner':component.get('owner'),
                                  'kind':component['kind'],'actual_box':component['box'],
                                  'actual_contype':component['_actual_contype'],
                                  'actual_conaffinity':component['_actual_conaffinity']})
        if obstacles:local.append({'code':'PULLOUT_BLOCKED','equipment':uid,'obstacles':obstacles})
        actual_holes=[]
        for level,z in enumerate(scene['floors'][1:],1):
            if z<=expected['base_z']+EPS:continue
            holes=scene['holes'].get(level,scene['holes'].get(str(level),[]))
            registered=[h for h in scene.get('opening_owners',[]) if h.get('level')==level and h.get('owner')==uid
                        and h.get('kind')=='pullout' and h.get('reservation_id')==expected['id']]
            exact=[h for h in registered if _same(h.get('rect'),expected['rect']) and any(_same(h['rect'],p) for p in holes)]
            boards=[c for c in components if c['kind'] in ('floor','ground','walk_surface') and abs(c['box'][5]-z)<=EPS]
            remaining=[c['id'] for c in boards if _overlap(expected['rect'],_rect(c['box']))]
            edges=_opening_edges(expected['rect'],boards)
            valid_hole=len(exact)==1 and not remaining and all(v['covered'] for v in edges.values())
            evidence={'level':level,'floor_z_m':z,'expected_rect':expected['rect'],'registered':registered,
                      'actual_board_intrusions':remaining,'actual_edge_measurements':edges,'valid':valid_hole}
            actual_holes.append(evidence)
            if not valid_hole:local.append({'code':'PULLOUT_OPENING_INVALID','equipment':uid,'measurement':evidence})
        # Extra or oversize purpose labels cannot authorize another void.
        for opening in scene.get('opening_owners',[]):
            if opening.get('owner')==uid and opening.get('kind')=='pullout':
                level=opening.get('level')
                if (not isinstance(level,int) or not 0<level<len(scene['floors']) or
                    scene['floors'][level]<=expected['base_z']+EPS or
                    opening.get('reservation_id')!=expected['id'] or not _same(opening.get('rect'),expected['rect'])):
                    local.append({'code':'PULLOUT_OPENING_INVALID','equipment':uid,'opening':opening})
        errors.extend(local)
        measurements.append({'equipment':uid,'status':'fail' if local else 'pass',
            'expected':expected,'declared':record,'dimensions_match':valid,
            'actual_holes':actual_holes,'obstacles':obstacles,'source':expected['source']})
    for record in declared:
        if record.get('owner') not in owners:
            errors.append({'code':'PULLOUT_WITHOUT_ACTUAL_DRIVEN_VESSEL','record':record})
    return {'status':'fail' if errors else 'pass','rows':measurements,'violations':errors,
            'scope':'Measure XML centerline, shell top, drive and central flange OD, estimating from DN where needed. Remeasure opening edges on the physical deck.'}
