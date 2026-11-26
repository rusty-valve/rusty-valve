"""Check equipment and building parts for physical intersections."""
from __future__ import annotations
from .runtime_paths import DATA


PENETRATION_LIMIT_M = 1e-5  # Existing checks.geometry_checks threshold.
STRUCTURAL_KINDS = {'floor','ground','walk_surface','column','beam','equipment_beam',
                    'step','landing','stair','rail','curb'}
EQUIPMENT_KINDS = {'vessel','tank','head','jacket_shell','pump','motor','filter',
                   'exchanger','condenser','fixed_bed','tower','drive_reservation',
                   'equipment_nozzle','nozzle','support','lug','saddle','skirt','plinth'}


def _overlap_box(a,b):
    return [max(a[i],b[i]) for i in range(3)]+[min(a[i+3],b[i+3]) for i in range(3)]


def _volume(box):
    return max(0.,box[3]-box[0])*max(0.,box[4]-box[1])*max(0.,box[5]-box[2])


def _rect(box):
    return [box[0],box[1],box[3],box[4]]


def _contained_rect(inner,outer,eps):
    return inner[0]>=outer[0]-eps and inner[1]>=outer[1]-eps and inner[2]<=outer[2]+eps and inner[3]<=outer[3]+eps


def _horizontal_box(record,model,data,eps):
    import mujoco
    if record['geom_type']!=int(mujoco.mjtGeom.mjGEOM_BOX):return False
    axis=data.geom_xmat[record['geom_index']].reshape(3,3)[:,2]
    return abs(axis[0])<=eps and abs(axis[1])<=eps and abs(abs(axis[2])-1)<=eps


def _taiji_local_profile(shoe,lug,body,plane,model,data,graph,eps):
    """Match the unchanged Taiji coarse outlines to independent factory d2/d3.

    Expected rectangles use the frozen _taiji_lugs construction estimates.
    Both real quarter-turn geometries are tested; no rotation label is trusted.
    """
    import json
    import math
    import mujoco
    unit=graph['units'].get(shoe['owner'],{})
    declaration=unit.get('dimension_spec',unit.get('catalog',{}))
    if declaration.get('series')!='Taiji F':return None
    models={'small':'F5000','medium':'F10000','large':'F20000'}
    key=models.get(unit.get('size_tier'))
    if key is None:return None
    source=DATA/'reactor_f_series_dimensions.json'
    reference=json.loads(source.read_text());raw=reference['models'][key]['dimensions_mm']
    index=body['geom_index']
    if body['geom_type']!=int(mujoco.mjtGeom.mjGEOM_CYLINDER):return None
    axis=data.geom_xmat[index].reshape(3,3)[:,2]
    radius=raw['d2']/2000;hole_radius=raw['d3']/2000
    if abs(abs(axis[2])-1)>eps or abs(float(model.geom_size[index,0])-radius)>eps:return None
    cx,cy=map(float,data.geom_xpos[index,:2])
    for bearing in reference['lug_bearings_deg']:
        angle=math.radians(bearing);hx,hy=hole_radius*math.sin(angle),hole_radius*math.cos(angle)
        sign=1 if hx>0 else -1
        inner=sign*(radius/math.sqrt(2)-.10)
        outer=sign*max(abs(hx)+.18,radius+.24)
        x0,x1=sorted((inner,outer));sx0,sx1=sorted((sign*(radius+.12),outer))
        lug_rect=[x0,hy-.16,x1,hy+.16];shoe_rect=[sx0,hy-.12,sx1,hy+.12]
        for rotation in (0,90):
            def rotated(rect):
                return rect if rotation==0 else [-rect[3],rect[0],-rect[1],rect[2]]
            lr,sr=rotated(lug_rect),rotated(shoe_rect)
            expected_lug=[cx+lr[0],cy+lr[1],plane,cx+lr[2],cy+lr[3],plane+.10]
            expected_shoe=[cx+sr[0],cy+sr[1],plane-.10,cx+sr[2],cy+sr[3],plane]
            if (all(abs(a-b)<=eps for a,b in zip(lug['box'],expected_lug))
                    and all(abs(a-b)<=eps for a,b in zip(shoe['box'],expected_shoe))):
                return {'factory_model':key,'factory_source':str(source),'d2_mm':raw['d2'],'d3_mm':raw['d3'],
                        'bearing_deg':bearing,'actual_matching_rotation_deg':rotation,
                        'expected_lug_box':expected_lug,'expected_shoe_box':expected_shoe,
                        'outline_source':'frozen equipment_catalog._taiji_lugs coarse outline estimates; not manufacturer fabrication details'}
    return None


def _local_bearing_joint(a,b,records,model,data,graph,eps):
    """Recognize only the real shoe volume embedded under its lug/slab face."""
    import mujoco
    from .platform_coverage import _uncovered
    if a['kind']=='support' and b['kind'] in ('floor','ground'):
        shoe,board=a,b
    elif b['kind']=='support' and a['kind'] in ('floor','ground'):
        shoe,board=b,a
    else:return None
    if shoe['role']!='equipment' or not shoe['owner'] or board['role']!='structure':return None
    if not _horizontal_box(shoe,model,data,eps) or not _horizontal_box(board,model,data,eps):return None
    sb,bb=shoe['box'],board['box'];plane=bb[5]
    if abs(sb[5]-plane)>eps or sb[2]<bb[2]-eps:return None
    lugs=[c for c in records if c['role']=='equipment' and c['owner']==shoe['owner'] and c['kind']=='lug'
          and _horizontal_box(c,model,data,eps)
          and abs(c['box'][2]-sb[5])<=eps and _contained_rect(_rect(sb),_rect(c['box']),eps)]
    declared=shoe['component'].get('lug_id')
    if declared is not None:lugs=[c for c in lugs if c['id']==declared]
    if len(lugs)!=1:return None
    lug=lugs[0]
    if mujoco.mj_geomDistance(model,data,shoe['geom_index'],lug['geom_index'],.01,None)>eps:return None
    bodies=[c for c in records if c['owner']==shoe['owner'] and c['kind']=='vessel' and c['id']==shoe['owner']
            and mujoco.mj_geomDistance(model,data,lug['geom_index'],c['geom_index'],.01,None)<=eps]
    if not bodies:return None
    profile=_taiji_local_profile(shoe,lug,bodies[0],plane,model,data,graph,eps)
    if profile is None:return None
    # A local shoe cannot claim an entire plate or span an unplated opening.
    slabs=[_rect(c['box']) for c in records if c['kind'] in ('floor','ground') and c['role']=='structure'
           and abs(c['box'][5]-plane)<=eps]
    missing=[r for r in _uncovered(_rect(sb),slabs) if r[2]-r[0]>eps and r[3]-r[1]>eps]
    if missing:return None
    return {'type':'local_shoe_in_slab_under_real_lug','shoe':shoe['id'],'lug':lug['id'],'board':board['id'],
            'actually_touching_own_body_parts':[c['id'] for c in bodies],
            'frozen_local_geometry_match':profile,
            'actual_shoe_box':sb,'actual_lug_box':lug['box'],'actual_board_box':bb,
            'actual_bearing_plane_z_m':plane,'intersection_box':_overlap_box(sb,bb),
            'scope':'Only the local physical shoe-to-deck contact is allowed. Check beam, column, equipment and main-body penetration independently.'}


def _legacy_bearing_joint(a,b,records,model,data,graph,eps):
    """Identify the unchanged AE/BE local embedded-ear coarse construction.

    Its ear straddles the slab top by .05m and the matching shoe lies below
    the ear. These are the existing _side_lugs geometry dimensions, not new
    clearance allowances. An enlarged/moved support cannot qualify by label.
    """
    import mujoco
    if a['kind'] in ('support','lug') and b['kind'] in ('floor','ground'):
        part,board=a,b
    elif b['kind'] in ('support','lug') and a['kind'] in ('floor','ground'):
        part,board=b,a
    else:return None
    if part['role']!='equipment' or board['role']!='structure' or not part['owner']:return None
    unit=graph['units'].get(part['owner'],{})
    spec=unit.get('dimension_spec',unit.get('catalog',{}))
    if spec.get('series') not in ('Pfaudler AE','GMM Pfaudler BE'):return None
    if not _horizontal_box(part,model,data,eps) or not _horizontal_box(board,model,data,eps):return None
    bodies=[c for c in records if c['owner']==part['owner'] and c['id']==part['owner']
            and c['kind']=='vessel' and c['geom_type']==int(mujoco.mjtGeom.mjGEOM_CYLINDER)]
    if len(bodies)!=1:return None
    body=bodies[0];body_index=body['geom_index']
    axis=data.geom_xmat[body_index].reshape(3,3)[:,2]
    if abs(abs(axis[2])-1)>eps:return None
    radius=float(model.geom_size[body_index,0]);cx,cy=data.geom_xpos[body_index,:2];plane=board['box'][5]
    lugs=[part] if part['kind']=='lug' else [c for c in records if c['kind']=='lug' and c['owner']==part['owner']]
    shoes=[part] if part['kind']=='support' else [c for c in records if c['kind']=='support' and c['owner']==part['owner']]
    for lug in lugs:
        if not _horizontal_box(lug,model,data,eps):continue
        if abs((lug['box'][2]+lug['box'][5])/2-plane)>eps:continue
        if mujoco.mj_geomDistance(model,data,lug['geom_index'],body_index,.01,None)>eps:continue
        for shoe in shoes:
            if not _horizontal_box(shoe,model,data,eps):continue
            if abs(shoe['box'][5]-lug['box'][2])>eps:continue
            if shoe['box'][2]<board['box'][2]-eps or not _contained_rect(_rect(shoe['box']),_rect(lug['box']),eps):continue
            if mujoco.mj_geomDistance(model,data,lug['geom_index'],shoe['geom_index'],.01,None)>eps:continue
            if _volume(_overlap_box(shoe['box'],board['box']))<=eps:continue
            profile=False
            for direction in (0,1):
                for sign in (-1,1):
                    centre=[float(cx),float(cy)];centre[direction]+=sign*(radius+.10)
                    lh=[.14,.14];lh[direction]=.16
                    sh=[.13,.13];sh[direction]=.12
                    expected_lug=[centre[0]-lh[0],centre[1]-lh[1],plane-.05,
                                  centre[0]+lh[0],centre[1]+lh[1],plane+.05]
                    expected_shoe=[centre[0]-sh[0],centre[1]-sh[1],plane-.10,
                                   centre[0]+sh[0],centre[1]+sh[1],plane-.05]
                    if all(abs(x-y)<=eps for x,y in zip(lug['box'],expected_lug)) and all(
                            abs(x-y)<=eps for x,y in zip(shoe['box'],expected_shoe)):
                        profile=True
            if not profile:continue
            return {'type':'legacy_local_embedded_ear_and_shoe','part':part['id'],'lug':lug['id'],
                    'shoe':shoe['id'],'body':body['id'],'board':board['id'],
                    'actual_part_box':part['box'],'actual_lug_box':lug['box'],'actual_shoe_box':shoe['box'],
                    'actual_board_box':board['box'],'actual_bearing_plane_z_m':plane,
                    'intersection_box':_overlap_box(part['box'],board['box']),
                    'geometry_source':'unchanged equipment_catalog._side_lugs AE/BE coarse ear/shoe dimensions',
                    'scope':'Allow only the documented local historical lug or shoe contact with its deck'}
    return None


def check_solid_collisions(graph,scene,model):
    """Check actual cross-owner equipment and equipment/structure penetrations."""
    import mujoco
    from .building_equipment_checks import EPS
    from .checks import geom_bounds
    if model is None:
        return {'status':'fail','rows':[],'violations':[{'code':'SOLID_MODEL_REQUIRED',
                'message':'Solid collision checks require the actual XML model'}]}
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    components={c['id']:c for c in scene['components']};records=[];violations=[];rows=[]
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)]
    boxes=geom_bounds(model,data,list(range(model.ngeom)))
    for index,name in enumerate(names):
        component=components.get(name)
        if component is None:
            component={'id':name or f'unnamed_geom_{index}','kind':'unregistered_solid','owner':None}
            violations.append({'code':'SOLID_GEOMETRY_UNREGISTERED','component':component['id'],
                               'geom_index':index,'message':'The XML geom lacks component ownership'})
        owner=component.get('owner');kind=component['kind']
        if not owner and component['id'] in graph['units'] and kind in EQUIPMENT_KINDS:owner=component['id']
        role=('structure' if kind in STRUCTURAL_KINDS else
              'equipment' if owner is not None or kind in EQUIPMENT_KINDS else 'other')
        records.append({'id':component['id'],'kind':kind,'owner':owner,'role':role,'component':component,
                        'geom_index':index,'geom_type':int(model.geom_type[index]),'box':boxes[index].tolist(),
                        'contype':int(model.geom_contype[index]),'conaffinity':int(model.geom_conaffinity[index])})
    present=set(names)
    for component in scene['components']:
        if component['id'] not in present:
            violations.append({'code':'SOLID_COMPONENT_MISSING','component':component['id'],
                               'message':'The registered component is missing from the physical XML'})
    tested=0;assembly_pairs=0;legacy_distance_queries=0;distance_model=None;distance_data=None
    def public(record):
        return {key:record[key] for key in ('id','kind','owner','role','geom_index','geom_type','box','contype','conaffinity')}
    for index,first in enumerate(records):
        for second in records[index+1:]:
            if first['role']!='equipment' and second['role']!='equipment':continue
            # No structural slab/column/beam is exempt merely because a caller
            # has copied an equipment owner label onto it.
            if (first['role']==second['role']=='equipment' and first['owner'] is not None
                    and first['owner']==second['owner']):
                assembly_pairs+=1;continue
            intersection=_overlap_box(first['box'],second['box'])
            if _volume(intersection)<=0:continue
            tested+=1
            native_distance=float(mujoco.mj_geomDistance(model,data,first['geom_index'],second['geom_index'],.01,None))
            distance=native_distance;algorithm='MuJoCo native mj_geomDistance'
            # Native CCD can return exactly zero for deep, concentric convex
            # overlaps. Recheck only that degenerate result with MuJoCo's
            # bundled libccd distance on a private model copy. Shapes, masks,
            # poses and the penetration threshold remain unchanged.
            if abs(native_distance)<=EPS:
                if distance_model is None:
                    from copy import deepcopy
                    distance_model=deepcopy(model)
                    distance_model.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_NATIVECCD
                    distance_data=mujoco.MjData(distance_model)
                    mujoco.mj_copyData(distance_data,distance_model,data)
                alternative=float(mujoco.mj_geomDistance(distance_model,distance_data,
                    first['geom_index'],second['geom_index'],.01,None))
                legacy_distance_queries+=1;distance=min(native_distance,alternative)
                algorithm='MuJoCo native; zero-result rechecked by bundled libccd'
            row={'first':public(first),'second':public(second),'distance_m':distance,
                 'native_distance_m':native_distance,'distance_algorithm':algorithm,
                 'penetration_m':max(0.,-distance),'status':'clear','joint':None}
            if distance < -PENETRATION_LIMIT_M:
                joint=(_local_bearing_joint(first,second,records,model,data,graph,EPS)
                       or _legacy_bearing_joint(first,second,records,model,data,graph,EPS))
                if joint:
                    row.update(status='declared_local_bearing_joint',joint=joint)
                else:
                    row['status']='collision'
                    violations.append({'code':'SOLID_COLLISION','first':public(first),'second':public(second),
                                       'distance_m':distance,'penetration_m':-distance,
                                       'native_distance_m':native_distance,'distance_algorithm':algorithm,
                                       'actual_intersection_box':intersection})
            rows.append(row)
    return {'status':'fail' if violations else 'pass','rows':rows,'violations':violations,
            'penetration_limit_m':PENETRATION_LIMIT_M,'distance_source':'actual XML geom_bounds and MuJoCo mj_geomDistance',
            'collision_masks_used':False,'distance_tested_pairs':tested,'same_equipment_assembly_pairs':assembly_pairs,
            'zero_native_distance_rechecks':legacy_distance_queries,
            'ccd_reference':'https://mujoco.readthedocs.io/en/stable/computation/index.html#convex-collisions',
            'zero_result_note':'Exact zero penetration was reproduced locally in MuJoCo 3.16.0. A second calculation uses its built-in libccd.',
            'scope':'Check cross-owner solids and equipment against framing, slabs and stairs. Exceptions cover only native assemblies and measured local bearing contacts.'}
