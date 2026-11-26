"""Search for equipment positions with room for operation and maintenance."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import combinations, product
import math
import random

import numpy as np

from .batch_layout import _vessel
from .layout import add_box, add_pipe
from .process import RULES
from .recipe import CandidateRefused
from .shell import _frame_parts, add_structure
from .platform_coverage import _uncovered
from .pump_maintenance import envelope, long_sides, ORDINARY_GAP_M


PRIMARY = ('vessel', 'pump', 'filter', 'exchanger', 'condenser', 'fixed_bed')
EXCLUDED = ('equipment_beam', 'pipe', 'valve', 'rail', 'hanger')


def _overlaps(a, b):
    return min(a[2],b[2])-max(a[0],b[0]) > 1e-8 and min(a[3],b[3])-max(a[1],b[1]) > 1e-8


def _expanded(rect, distance):
    return [rect[0]-distance,rect[1]-distance,rect[2]+distance,rect[3]+distance]


def _projection(components, floors):
    body=[c for c in components if c['kind'] not in EXCLUDED and not c.get('visual_only')]
    if not body:
        raise ValueError('The coarse equipment model has no projectable body parts')
    rect=[min(c['box'][0] for c in body),min(c['box'][1] for c in body),
          max(c['box'][3] for c in body),max(c['box'][4] for c in body)]
    low=min(c['box'][2] for c in body);high=max(c['box'][5] for c in body)
    occupied=[i for i,z in enumerate(floors)
              if high > z+1e-8 and (i==len(floors)-1 or low < floors[i+1]-1e-8)]
    return rect,occupied,[low,high]


def _prototype(unit, graph, shell, rng, rotation_deg=None):
    if unit['kind'] != 'pump':
        from .equipment_catalog import prototype
        return prototype(unit,graph,shell,rng,rotation_deg=rotation_deg)
    scene={'id':graph['id'],'components':[],'holes':defaultdict(list),
           'floors':list(shell['floors'])}
    z=scene['floors'][unit['level']]
    if unit['kind']=='vessel':
        _vessel(unit,scene,shell,(0.,0.),rng)
        for component in scene['components']:
            if component['kind']=='nozzle':
                component['kind']='equipment_nozzle'
                component['scope']='equipment-integral stub'
    elif unit['kind']=='pump':
        # Existing simple pump/base dimensions, with an explicit motor cap so
        # the physical assembly also records which side carries the motor.
        add_box(scene,unit['id']+'_base',[-.55,-.34,z,.55,.34,z+.22],
                'plinth',(.43,.44,.43),unit['id'])
        add_pipe(scene,unit['id'],[-.30,0,z+.55],[.35,0,z+.55],
                 .22,'pump',(.24,.47,.57),unit['id'])
        add_pipe(scene,unit['id']+'_motor',[.15,0,z+.55],[.35,0,z+.55],
                 .18,'motor',(.19,.32,.39),unit['id'])
        add_box(scene,unit['id']+'_foot',[-.20,-.15,z+.22,.20,.15,z+.40],
                'support',(.3,.4,.44),unit['id'])
        unit.update(center=[0.,0.,z+.55],box=[-.55,-.34,z,.55,.34,z+.85])
        unit['ports']['in'].update(pos=[-.45,0,z+.55],normal=[-1,0,0],stem=[0,0,1])
        unit['ports']['out'].update(pos=[.45,0,z+.55],normal=[1,0,0],stem=[0,0,1])
        # Same approved pump assembly, rigidly turned so the existing x rows
        # are perpendicular to its shaft: side-by-side rather than end-to-end.
        from .equipment_catalog import _rotate
        scene['extraction_options']=[]
        _rotate(scene,unit,90)
        unit['rotation_deg']=90
    else:
        low,high=z+.5,z+1.8
        add_pipe(scene,unit['id'],[0.,0.,low],[0.,0.,high],.25,
                 unit['kind'],(.72,.62,.40),unit['id'])
        for side in (-1,1):
            add_box(scene,unit['id']+'_leg'+str(side),
                    [side*.2-.04,-.05,z,side*.2+.04,.05,low+.3],
                    'support',(.27,.38,.44),unit['id'])
        unit.update(center=[0.,0.,(low+high)/2],box=[-.25,-.25,low,.25,.25,high],
                    orientation='vertical',bottom=low,top=high)
        for name,port in unit['ports'].items():
            side=1 if name.startswith('utility') else -1
            height=low+.2 if name in ('in','condensate','utility_in') else high-.2
            port.update(pos=[0.,side*.25,height],normal=[0,side,0],stem=[0,0,1])
    # A jacket is part of the vessel, rather than a second maintenance object.
    jackets=[j for j in graph['units'].values() if j['kind']=='jacket' and j.get('owner')==unit['id']]
    for jacket in jackets:
        radius=unit['radius']+.045
        add_pipe(scene,jacket['id'],[0.,0.,unit['bottom']+.2],[0.,0.,z-.25],radius,
                 'jacket_shell',(.56,.68,.72),unit['id'])
    scene['components']=[c for c in scene['components']
                         if c['kind'] not in ('safety','check_valve','gauge','header','pipe','valve')]
    rect,occupied,z_range=_projection(scene['components'],shell['floors'])
    return {'unit':unit,'components':scene['components'],'holes':dict(scene['holes']),
            'rect':rect,'occupied_levels':occupied,'z_range_m':z_range}


def _shift_rect(rect,x,y):
    return [rect[0]+x,rect[1]+y,rect[2]+x,rect[3]+y]


def _preview_boards(graph,shell):
    """Measure the actual shell once, without adding it to the output scene."""
    from .building_equipment_checks import standing_board_rectangles, _structural_obstacles
    preview={'components':[],'holes':defaultdict(list),'surfaces':[],'stairs':[],
             'floors':list(shell['floors']),'footprint':list(shell['footprint_m'])}
    add_structure(preview,graph,shell)
    return ({level:standing_board_rectangles(preview,level) for level in range(len(shell['floors']))},
            {level:_structural_obstacles(preview['components'],shell['floors'],level)
             for level in range(len(shell['floors']))},preview['components'])


def _standing_limits(prototype,scene,shell):
    """Candidate limits come from boards/rail inner faces, not a new inset."""
    width,depth=shell['footprint_m']
    bounds=[0.,0.,width,depth]
    for level in prototype['occupied_levels']:
        boards=scene['_placement_boards'][level]
        if not boards:
            raise CandidateRefused('No physical standing slab on the equipment level')
        level_bounds=[min(r[0] for r in boards),min(r[1] for r in boards),
                      max(r[2] for r in boards),max(r[3] for r in boards)]
        bounds=[max(bounds[0],level_bounds[0]),max(bounds[1],level_bounds[1]),
                min(bounds[2],level_bounds[2]),min(bounds[3],level_bounds[3])]
    return bounds


def _level_profile(item,level):
    profiles=item.get('level_footprints',{})
    return profiles.get(level,profiles.get(str(level),{
        'rect':item['rect'],'width_m':item.get('maintenance_width_m')}))


def _boxes_overlap(a,b):
    return all(min(a[i+3],b[i+3])-max(a[i],b[i])>1e-8 for i in range(3))


def _trap_spaces(owners,scene):
    from .steam_trap_layout import trap_layout
    from .pipe_spec import load
    definitions=scene.get('_trap_definitions',[]);rows=[]
    for definition in definitions:
        endpoint=definition['source_endpoint'];uid,name=endpoint.split('.',1)
        if uid not in owners:continue
        owner=owners[uid]
        placement=trap_layout(owner['ports'][name],owner['rect'],scene['floors'],load()['services']['steam_condensate']['device_dn'])
        rows.extend({**row,'owner':uid,'automatic':definition['id'],'source_endpoint':endpoint,
            'scope':'required trap body / physical inlet and outlet space; not operation stance'} for row in placement['reservations'])
    return rows


def _candidate_trap_spaces(prototype,xy,scene):
    owners={};x,y=xy
    for original in [prototype['unit'],*prototype.get('companions',[])]:
        unit=deepcopy(original)
        for port in unit['ports'].values():
            if 'pos' in port:port['pos']=[port['pos'][0]+x,port['pos'][1]+y,port['pos'][2]]
        local=prototype['rect'] if original is prototype['unit'] else [original['box'][i] for i in (0,1,3,4)]
        unit['rect']=_shift_rect(local,x,y);owners[unit['id']]=unit
    return _trap_spaces(owners,scene)


def _can_place(prototype,xy,placed,scene,shell,maintenance):
    if 'port_envelopes' in prototype:
        from .layout_port_envelopes import in_room
        _, outside = in_room(prototype['port_envelopes'], xy, shell)
        if outside:
            return False
    if prototype['unit']['kind'] != 'pump':
        return _can_place_side(prototype,xy,placed,scene,shell,maintenance)
    for side in long_sides(prototype['rect']):
        for profile in prototype['level_footprints'].values():
            profile['pump_maintenance_side'] = side
        if _can_place_side(prototype,xy,placed,scene,shell,maintenance):
            return True
    return False


def _can_place_side(prototype,xy,placed,scene,shell,maintenance):
    x,y=xy;rect=_shift_rect(prototype['rect'],x,y)
    definition=scene.get('_filter_connections',{}).get(prototype['unit']['id'])
    if definition:
        from .filter_group_layout import geometry,upstream_entry
        ports=prototype['unit']['ports'];shift=np.array([x,y,0.])
        group=geometry(np.asarray(ports['in']['pos'])+shift,
                       np.asarray(ports['out']['pos'])+shift,definition['dn'])
        side=np.cross([0.,0.,1.],group['axis']);width,depth=shell['footprint_m']
        spaces=[upstream_entry(group,definition['source_normal'],definition['has_root'],direction)
                for direction in (-group['axis'],side,-side)]
        if not any(box[0]>=-1e-8 and box[1]>=-1e-8 and box[3]<=width+1e-8 and box[4]<=depth+1e-8
                   and not any(_boxes_overlap(box,c['box']) for c in scene['components']
                               if c.get('owner')!=prototype['unit']['id'])
                   for box in (space['box'] for space in spaces)):return False
    world_boxes=[[v+(x if i%3==0 else y if i%3==1 else 0.) for i,v in enumerate(c['box'])] for c in prototype['components']]
    for reserved in scene.get('required_installation_spaces',[]):
        if reserved['owner']==prototype['unit']['id']:continue
        if any(_boxes_overlap(box,reserved['box']) for box in world_boxes):return False
    for reserved in _candidate_trap_spaces(prototype,xy,scene):
        if any(_boxes_overlap(c['box'],reserved['box']) for c in scene['components']
               if c.get('owner')!=reserved['owner']):return False
    levels=set(prototype['occupied_levels'])
    width,depth=shell['footprint_m']
    for level in levels:
        profile=_level_profile(prototype,level)
        local=_shift_rect(profile['rect'],x,y)
        ring=envelope(local,profile,maintenance)
        if ring[0]<-1e-8 or ring[1]<-1e-8 or ring[2]>width+1e-8 or ring[3]>depth+1e-8:
            return False
        if _uncovered(ring,scene['_placement_boards'][level]):return False
        if any(_overlaps(ring,o['rect']) for o in scene['_placement_structural'][level]):return False
        for other in placed:
            if level not in other['occupied_levels']:continue
            theirs=_level_profile(other,level)
            other_rect=theirs['rect']
            if (_overlaps(ring,other_rect) or
                    _overlaps(local,envelope(other_rect,theirs,maintenance))):
                return False
            if other.get('extraction_rect') and _overlaps(local,other['extraction_rect']):return False
        for opening in scene['opening_owners']:
            if opening['level']==level and opening['owner']!=prototype['unit']['id'] and _overlaps(ring,opening['rect']):
                return False
    # The complete ground pump, including its motor/base, cannot enter the
    # maximum body shadow. This is independent of level maintenance widths.
    pump_ground=prototype['unit']['kind']=='pump' and prototype['unit']['level']==0
    body=prototype.get('vessel_body_rect')
    body=_shift_rect(body,x,y) if body is not None else None
    for other in placed:
        if pump_ground and other.get('vessel_body_rect') is not None and _overlaps(rect,other['vessel_body_rect']):return False
        if body is not None and other.get('ground_pump') and _overlaps(body,other['rect']):return False
    for _,box,kind in _frame_parts(shell):
        if kind=='beam':
            for component in prototype['components']:
                if component['kind'] not in ('vessel','head','motor','drive_reservation','exchanger','condenser','fixed_bed'):continue
                c=component['box']
                if min(c[5],box[5])-max(c[2],box[2])>1e-8 and _overlaps(
                        _shift_rect([c[0],c[1],c[3],c[4]],x,y),[box[0],box[1],box[3],box[4]]):return False
    # Dedicated, upward-open agitator removal space is not a physical body
    # or an extra maintenance circle. Future holes do not excuse beams/stairs.
    if prototype.get('pullout_reservations') or scene.get('pullout_reservations'):
        from .pullout_reservations import placement_clear
        structures=[c for c in scene.get('_placement_components',[]) if c['kind'] not in ('floor','ground','walk_surface')]
        for record in prototype.get('pullout_reservations',[]):
            world={**record,'rect':_shift_rect(record['rect'],x,y),
                   'axis_xy':[record['axis_xy'][0]+x,record['axis_xy'][1]+y]}
            if not placement_clear(world,scene['components']+structures,owner=prototype['unit']['id']):return False
        world_components=[]
        for component in prototype['components']:
            c={**component,'box':[v+(x if i%3==0 else y if i%3==1 else 0.) for i,v in enumerate(component['box'])]}
            world_components.append(c)
        for record in scene.get('pullout_reservations',[]):
            if not placement_clear(record,world_components,owner=record['owner']):return False
    prototype['_chosen_extraction']=None
    options=prototype.get('extraction_options',[])
    if options:
        for option in options:
            area=_shift_rect(option['rect'],x,y);level=option.get('level',prototype['unit']['level'])
            if _uncovered(area,scene['_placement_boards'][level]):continue
            if any(_overlaps(area,o['rect']) for o in scene['_placement_structural'][level]):continue
            if any(_overlaps(area,_level_profile(other,level)['rect']) for other in placed if level in other['occupied_levels']):continue
            if any(kind=='column' and _overlaps(area,[b[0],b[1],b[3],b[4]]) for _,b,kind in _frame_parts(shell)):continue
            if any(o['level']==level and o['owner']!=prototype['unit']['id'] and _overlaps(area,o['rect']) for o in scene['opening_owners']):continue
            prototype['_chosen_extraction']={**option,'rect':area};break
        if prototype['_chosen_extraction'] is None:return False
    return True


def _port_operation_references(unit,instance,scene):
    """Actual board/stance references; this does not claim an installed valve."""
    from .pipe_spec import load
    from .building_equipment_checks import standing_board_rectangles
    band=load()['operation_band_m'];floors=scene['floors']
    hx=RULES['geometry_screen']['walk_width_m']/2
    hy=RULES['geometry_screen']['walk_depth_m']/2
    references=[]
    for name,port in unit['ports'].items():
        if 'pos' not in port:continue
        p=np.asarray(port['pos']);existing=port.get('operation_reference',{})
        level=int(existing.get('level',unit['level']));floor=floors[level]
        boards=standing_board_rectangles(scene,level)
        holes=scene['holes'].get(level,scene['holes'].get(str(level),[]))
        boards=[part for board in boards for part in _uncovered(board,holes)]
        rect=instance['level_footprints'].get(level,instance['level_footprints'].get(str(level),{})).get('rect',instance['rect'])
        x=float(np.clip(p[0],rect[0],rect[2]));y=float(np.clip(p[1],rect[1],rect[3]))
        candidates=[('x_min',[rect[0]-hx-.02,y]),('x_max',[rect[2]+hx+.02,y]),
                    ('y_min',[x,rect[1]-hy-.02]),('y_max',[x,rect[3]+hy+.02])]
        candidates.sort(key=lambda row:math.dist(row[1],p[:2]))
        chosen=None
        for side,center in candidates:
            stance=[center[0]-hx,center[1]-hy,center[0]+hx,center[1]+hy]
            if _uncovered(stance,boards):continue
            volume=[stance[0],stance[1],floor,stance[2],stance[3],floor+band[1]]
            blocked=[c['id'] for c in scene['components'] if c['box'][5]>floor+1.e-8
                and c['box'][2]<floor+band[1]-1.e-8
                and all(min(volume[i+3],c['box'][i+3])-max(volume[i],c['box'][i])>1.e-8 for i in range(3))]
            if blocked:continue
            board_ids=[c['id'] for c in scene['components'] if
                ((c['kind'] in ('floor','ground') and abs(c['box'][5]-floor)<=1.e-8)
                 or (c['kind']=='walk_surface' and c.get('walk_level')==level))
                and min(stance[2],c['box'][3])>max(stance[0],c['box'][0])+1.e-8
                and min(stance[3],c['box'][4])>max(stance[1],c['box'][1])+1.e-8]
            if not board_ids:continue
            chosen={'side':side,'stance_rect_m':stance,'standing_position_m':[center[0],center[1],floor],
                    'standing_component_ids':board_ids}
            break
        row={'endpoint':unit['id']+'.'+name,'level':level,'standing_surface_z_m':floor,
            'operation_band_m':list(band),'allowed_operation_z_m':[floor+band[0],floor+band[1]],
            'interface_height_m':float(p[2]-floor),
            'interface_within_operation_band':floor+band[0]-1.e-8<=p[2]<=floor+band[1]+1.e-8,
            'status':'standing_face_reference' if chosen else 'standing_face_unresolved',
            'scope':'Actual deck and free standing area as geometric reference for operation'}
        if chosen:row.update(chosen)
        port['operation_reference']=row;references.append(row)
    unit['valve_group_operation_references']=references
    instance['valve_group_operation_references']=deepcopy(references)
    scene.setdefault('equipment_valve_operation_references',[]).extend(deepcopy(references))


def _place(prototype,xy,scene,placed,pump_row=None):
    x,y=xy;unit=prototype['unit'];uid=unit['id']
    components=deepcopy(prototype['components'])
    for c in components:
        # The complete-depth support beams stay between the frame edges.
        span=c.get('building_span_axis')
        dx=0. if span==0 else x
        dy=0. if c['kind']=='equipment_beam' and span!=0 else y
        c['box']=[v+(dx if i%3==0 else dy if i%3==1 else 0.) for i,v in enumerate(c['box'])]
        if 'pos' in c:c['pos']=[c['pos'][0]+dx,c['pos'][1]+dy,c['pos'][2]]
        if 'ends' in c:c['ends']=[[p[0]+dx,p[1]+dy,p[2]] for p in c['ends']]
    unit['center']=[unit['center'][0]+x,unit['center'][1]+y,unit['center'][2]]
    if 'box' in unit:
        unit['box']=[v+(x if i%3==0 else y if i%3==1 else 0.) for i,v in enumerate(unit['box'])]
    for port in unit['ports'].values():
        if 'pos' in port:
            port['pos']=[port['pos'][0]+x,port['pos'][1]+y,port['pos'][2]]
        if 'surface_pos' in port:
            port['surface_pos']=[port['surface_pos'][0]+x,port['surface_pos'][1]+y,port['surface_pos'][2]]
    for jacket in prototype.get('companions',[]):
        placed_jacket=deepcopy(jacket)
        for port in placed_jacket['ports'].values():
            for key in ('pos','surface_pos'):
                port[key]=[port[key][0]+x,port[key][1]+y,port[key][2]]
        placed_jacket['center']=[jacket['center'][0]+x,jacket['center'][1]+y,jacket['center'][2]]
        placed_jacket['box']=[v+(x if i%3==0 else y if i%3==1 else 0) for i,v in enumerate(jacket['box'])]
        unit.setdefault('integral_jackets',{})[jacket['id']]=placed_jacket
    scene['components'].extend(components)
    for record in prototype.get('pullout_reservations',[]):
        scene.setdefault('pullout_reservations',[]).append({**record,
            'rect':_shift_rect(record['rect'],x,y),
            'axis_xy':[record['axis_xy'][0]+x,record['axis_xy'][1]+y]})
    for level,holes in prototype['holes'].items():
        for hole in holes:
            rect=_shift_rect(hole,x,y)
            scene['holes'][level].append(rect)
            dedicated=next((o for o in prototype.get('pullout_openings',[]) if o['level']==level and
                all(abs(a-b)<=1e-8 for a,b in zip(o['rect'],hole))),None)
            opening={'level':level,'rect':rect,'owner':uid,'kind':'pullout' if dedicated else 'equipment'}
            if dedicated:opening['reservation_id']=dedicated['reservation_id']
            scene['opening_owners'].append(opening)
    rect=_shift_rect(prototype['rect'],x,y)
    instance={'id':uid,'kind':unit['kind'],'level':unit['level'],
              'ground_pump':unit['kind']=='pump' and unit['level']==0,
              'vessel_body_rect':_shift_rect(prototype['vessel_body_rect'],x,y) if prototype.get('vessel_body_rect') is not None else None,
              'level_footprints':{level:{**profile,'rect':_shift_rect(profile['rect'],x,y)}
                                  for level,profile in prototype.get('level_footprints',{}).items()},
              'occupied_levels':prototype['occupied_levels'],'rect':rect,
              'z_range_m':prototype['z_range_m'],'component_ids':[c['id'] for c in components],
              'form':unit.get('form',unit.get('orientation','horizontal_pump' if unit['kind']=='pump' else 'vertical')),
              'maintenance_width_m':prototype.get('maintenance_width_m')}
    if unit.get('dimension_spec'):instance['dimension_spec']=deepcopy(unit['dimension_spec'])
    if 'port_envelopes' in prototype:
        from .layout_port_envelopes import in_room
        occupied, outside = in_room(prototype['port_envelopes'], xy, scene['shell'])
        if outside:
            raise CandidateRefused('Equipment port, stub or root valve extends beyond the room: '+outside)
        instance['port_envelopes'] = occupied
    if 'nozzle_schedule' in unit:
        instance['nozzle_schedule'] = deepcopy(unit['nozzle_schedule'])
        instance['external_components'] = deepcopy(unit['external_components'])
    if prototype.get('operating_levels') is not None:instance['operating_levels']=list(prototype['operating_levels'])
    chosen=prototype.get('_chosen_extraction')
    if chosen:instance.update(extraction_end=chosen['end'],extraction_rect=chosen['rect'])
    if unit['kind']=='pump':
        instance.update(shaft_axis=[0.,1.,0.],motor_side=[0.,1.,0.],
                        base_front=[x,y-.55,scene['floors'][unit['level']]],pump_row=pump_row)
    _port_operation_references(unit,instance,scene)
    owners={uid:{**unit,'rect':rect}}
    owners.update({key:{**j,'rect':[j['box'][i] for i in (0,1,3,4)]} for key,j in unit.get('integral_jackets',{}).items()})
    scene['required_installation_spaces'].extend(_trap_spaces(owners,scene))
    scene['equipment_instances'].append(instance)
    placed.append(instance)
    if scene.get('_progress_folder') and len(placed)>scene['_best_count']:
        from .generation_progress import layout_checkpoint,atomic_json
        event={**scene['_trial_event'],'last_completed_equipment':uid,'equipment_count':len(placed)}
        layout_checkpoint(scene['_progress_graph'],scene,scene['_progress_folder'],event)
        scene['_best_count']=len(placed)
        atomic_json(scene['_progress_folder']/'best_layout.json',event)


def _placing(scene, equipment):
    if scene.get('_progress_folder') and scene.get('_current_equipment')!=equipment:
        from .generation_progress import atomic_json
        scene['_current_equipment']=equipment
        atomic_json(scene['_progress_folder']/'progress.json',
                    {**scene['_trial_event'],'current_equipment':equipment})


def _layout_checkpoint(prototypes,scene,placed):
    """Save append-only layout data and poses, retaining graph/unit identity."""
    return {'lengths':{key:len(scene[key]) for key in
            ('components','pullout_reservations','opening_owners','equipment_instances','required_installation_spaces')},
            'placed_length':len(placed),'holes':{level:len(rows) for level,rows in scene['holes'].items()},
            'units':[(p['unit'],deepcopy(p['unit'])) for p in prototypes],
            'choices':[(p,'_chosen_extraction' in p,deepcopy(p.get('_chosen_extraction'))) for p in prototypes]}


def _restore_layout(saved,scene,placed):
    for key,length in saved['lengths'].items():del scene[key][length:]
    del placed[saved['placed_length']:]
    for level in list(scene['holes']):
        if level not in saved['holes']:del scene['holes'][level]
        else:del scene['holes'][level][saved['holes'][level]:]
    for unit,pose in saved['units']:
        unit.clear();unit.update(deepcopy(pose))
    for prototype,present,choice in saved['choices']:
        if present:prototype['_chosen_extraction']=deepcopy(choice)
        else:prototype.pop('_chosen_extraction',None)


def _search_can_place(prototype,xy,placed,scene,shell,maintenance,search_stats=None):
    if search_stats is not None:
        limit=search_stats['position_check_limit']
        if limit is not None and search_stats['position_checks']>=limit:
            search_stats['limit_reached']=True
            error=CandidateRefused('The standard position search exhausted its budget. Candidate rejected.')
            error.search_statistics=deepcopy(search_stats)
            raise error
        search_stats['position_checks']+=1
        key='pump_position_checks' if prototype['unit']['kind']=='pump' else 'ordinary_position_checks'
        search_stats[key]+=1
    return _can_place(prototype,xy,placed,scene,shell,maintenance)


def _ordinary_candidates(prototype,shell,scene,maintenance,rng,direction='rear',placed=(),anchored=False):
    limits=_standing_limits(prototype,scene,shell);r=prototype['rect']
    profiles=[_level_profile(prototype,level) for level in prototype['occupied_levels']]
    x0=max(limits[0]+p['width_m']-p['rect'][0] for p in profiles)
    x1=min(limits[2]-p['width_m']-p['rect'][2] for p in profiles)
    y0=max(limits[1]+p['width_m']-p['rect'][1] for p in profiles)
    y1=min(limits[3]-p['width_m']-p['rect'][3] for p in profiles)
    if x1 < x0 or y1 < y0:return []
    # Sample inside each cell before checking placement. Fixed grid points
    # made different seeds select exactly the same room in the frozen
    # 20-seed gravity batch. Keep the same candidate count and all checks.
    xs=np.linspace(x0,x1,12);ys=np.linspace(y0,y1,8)
    candidates=[(rng.uniform(float(xa),float(xb)),rng.uniform(float(ya),float(yb)))
                for xa,xb in zip(xs[:-1],xs[1:])
                for ya,yb in zip(ys[:-1],ys[1:])]
    rng.shuffle(candidates)
    anchors=[]
    if anchored:
        # Random interior points waste narrow legal gaps. Also try exact
        # maintenance-edge alignments, derived from the actual occupied
        # levels and previously placed equipment/structure, without resizing.
        ax,ay={float(x0),float(x1)},{float(y0),float(y1)}
        for level in prototype['occupied_levels']:
            own=_level_profile(prototype,level);p=own['rect']
            own_width=own.get('width_m') or maintenance
            obstacles=[(o['rect'],own_width) for o in scene['_placement_structural'][level]]
            for other in placed:
                if level not in other['occupied_levels']:continue
                theirs=_level_profile(other,level)
                obstacles.append((theirs['rect'],max(own_width,theirs.get('width_m') or maintenance)))
            for box,gap in obstacles:
                ax.update((box[0]-gap-p[2],box[2]+gap-p[0]))
                ay.update((box[1]-gap-p[3],box[3]+gap-p[1]))
        anchors=[(x,y) for x in sorted(ax) for y in sorted(ay) if x0<=x<=x1 and y0<=y<=y1]
        rng.shuffle(anchors)
    # Keep a simple common front approach and leave a lower row for pumps;
    # there is no connection to a particular pump's source vessel.
    if direction=='rear':
        candidates.sort(key=lambda xy:-xy[1]+rng.random()*shell['depth_m']*.18)
    elif direction=='front':
        candidates.sort(key=lambda xy:xy[1]+rng.random()*shell['depth_m']*.18)
    elif direction=='left':
        candidates.sort(key=lambda xy:xy[0]+rng.random()*shell['width_m']*.18)
    else:
        candidates.sort(key=lambda xy:-xy[0]+rng.random()*shell['width_m']*.18)
    # Preserve the old interior candidates after the additional alignments.
    return anchors+candidates


def _legacy_pump_rows(prototypes,shell,placed,scene,maintenance,rng,search_stats=None):
    width,depth=shell['footprint_m'];first=prototypes[0]['rect']
    limits=_standing_limits(prototypes[0],scene,shell)
    xmin,xmax=limits[0]+ORDINARY_GAP_M-first[0],limits[2]-ORDINARY_GAP_M-first[2]
    ymin,ymax=limits[1]+ORDINARY_GAP_M-first[1],limits[3]-ORDINARY_GAP_M-first[3]
    if xmax<xmin or ymax<ymin:
        raise CandidateRefused('Pump and single-side service aisle extend beyond the shell')
    ys=list(map(float,np.linspace(ymin,ymax,9)))
    # All pumps use the same assembly orientation. Only an entire row moves.
    rng.shuffle(ys);ys.sort(key=lambda y:y+rng.random()*.20)
    for row_count in (1,):
        if row_count>len(prototypes):continue
        groups=[prototypes[i::row_count] for i in range(row_count)]
        for row_ys in combinations(ys,row_count):
            proposals=[];tentative=list(placed);valid=True
            for row,(group,y) in enumerate(zip(groups,row_ys)):
                count=len(group)
                body_width=first[2]-first[0]
                if count*body_width+(count+1)*ORDINARY_GAP_M>width+1e-8:
                    valid=False;break
                # Try several whole-row offsets before falling back to two rows.
                xsets=[np.linspace(xmin,xmax,count)] if count>1 else [np.array([width/2])]
                if count==1:
                    xsets += [np.array([x]) for x in np.linspace(xmin,xmax,9)]
                selected=None
                for xs in xsets:
                    row_tentative=list(tentative);row_proposals=[]
                    for prototype,x in zip(group,xs):
                        xy=(float(x),y)
                        if not _search_can_place(prototype,xy,row_tentative,scene,shell,maintenance,search_stats):
                            break
                        row_tentative.append({'rect':_shift_rect(prototype['rect'],*xy),
                                              'occupied_levels':prototype['occupied_levels'],
                                              'level_footprints':{lev:{**prof,'rect':_shift_rect(prof['rect'],*xy)} for lev,prof in prototype['level_footprints'].items()}})
                        row_proposals.append((prototype,xy,row))
                    else:
                        selected=(row_tentative,row_proposals);break
                if selected is None:
                    valid=False;break
                tentative,row_proposals=selected;proposals.extend(row_proposals)
            if valid:
                for prototype,xy,row in proposals:_place(prototype,xy,scene,placed,row)
                return
    raise CandidateRefused('Side-by-side pumps on the floor cannot retain the single-side maintenance aisle. End-to-end placement is excluded.')


def _pump_rows(prototypes,shell,placed,scene,maintenance,rng,extended=False,search_stats=None):
    if not extended:
        return _legacy_pump_rows(prototypes,shell,placed,scene,maintenance,rng,search_stats)
    first=prototypes[0]['rect'];limits=_standing_limits(prototypes[0],scene,shell)
    xmin,xmax=limits[0]+ORDINARY_GAP_M-first[0],limits[2]-ORDINARY_GAP_M-first[2]
    ymin,ymax=limits[1]+ORDINARY_GAP_M-first[1],limits[3]-ORDINARY_GAP_M-first[3]
    if xmax<xmin or ymax<ymin:raise CandidateRefused('Pump and single-side service aisle extend beyond the shell')
    body_width=first[2]-first[0];body_depth=first[3]-first[1]
    ys=list(map(float,np.linspace(ymin,ymax,9)))
    # A pump-first candidate sees an empty floor. Always preferring the front
    # made every such attempt lock the same row before other equipment tried
    # to fit. Sample complete rows in a different order for each attempt.
    rng.shuffle(ys)
    for row_count in (1,):
        if row_count>len(prototypes):continue
        groups=[prototypes[i::row_count] for i in range(row_count)]
        xoptions=[]
        for group in groups:
            count=len(group)
            options=([np.linspace(xmin,xmax,count)] if count>1
                     else [np.array([shell['width_m']/2])])
            if count==1:
                options += [np.array([x]) for x in np.linspace(xmin,xmax,9)]
            else:
                # Share the maintenance strip, then translate the whole row.
                # No pump is moved independently or toward its source vessel.
                pitch=body_width+maintenance
                last_start=xmax-(count-1)*pitch
                if last_start>=xmin:
                    options += [start+np.arange(count)*pitch for start in np.linspace(xmin,last_start,9)]
            # Keep both the full-width and compact rows, but do not always
            # lock the full-width endpoints first on an otherwise empty floor.
            rng.shuffle(options)
            xoptions.append(options)
        for row_ys in combinations(ys,row_count):
            if row_count==2 and abs(row_ys[1]-row_ys[0])-body_depth<maintenance-1e-8:
                continue
            # At most two rows: try their whole-row offsets together rather
            # than locking the first row before the second one has been tested.
            for row_xs in product(*xoptions):
                tentative=list(placed);proposals=[];valid=True
                for row,(group,y,xs) in enumerate(zip(groups,row_ys,row_xs)):
                    for prototype,x in zip(group,xs):
                        xy=(float(x),y)
                        if not _search_can_place(prototype,xy,tentative,scene,shell,maintenance,search_stats):
                            valid=False;break
                        tentative.append({'rect':_shift_rect(prototype['rect'],*xy),
                            'occupied_levels':prototype['occupied_levels'],'maintenance_width_m':maintenance,
                            'level_footprints':{lev:{**prof,'rect':_shift_rect(prof['rect'],*xy)} for lev,prof in prototype['level_footprints'].items()}})
                        proposals.append((prototype,xy,row))
                    if not valid:break
                if valid:
                    for prototype,xy,row in proposals:_place(prototype,xy,scene,placed,row)
                    return
    raise CandidateRefused('Full-width and compact offsets of the side-by-side pump row could not retain the required clearance')


from .input_heights import assign_levels as _assign_levels


def placement_strategy(index):
    return (0,12,48,49,50,51,52,53)[index%8], ('rear','front','left','right')[index%4]


def equipment(graph,shell,attempt=0,reference=None,progress_folder=None,*,reserve_pipe_departures=False):
    """Return one building/equipment candidate; no process-pipe generation."""
    from .building_equipment_checks import read_layout_rules, maintenance_width_for
    rules=read_layout_rules();maintenance=rules['maintenance_width_m']
    strategy_index=attempt
    attempt,direction=placement_strategy(strategy_index)
    rng=random.Random(graph['seed']+strategy_index*104729)
    if not graph.get('structure_assigned'):_assign_levels(graph,shell)
    primary=[u for u in graph['units'].values() if u['kind'] in PRIMARY]
    actual=shell['platforms'] if graph.get('structure_assigned') else max((u['level'] for u in primary),default=0)
    from .equipment_catalog import resolve_spec
    for unit in primary:
        if unit['kind'] in ('pump','filter'):continue
        spec=resolve_spec(unit)
        # A drive reservation occupies space but does not, by itself, create
        # an otherwise unused platform. Real crossed platforms remain solid
        # geometry with their registered equipment/drive penetrations.
        height=unit['level']*shell['storey_height_m']+max(spec.get('support_to_top_m') or 0.,(spec.get('height_m') or 0.) if unit['kind']=='fixed_bed' else 0.)
        actual=max(actual,max((i for i,z in enumerate(shell['floors']) if z<height-1e-8),default=0))
    actual_shell=deepcopy(shell)
    actual_shell['layout_only']=True
    actual_shell['platform_limit']=shell.get('platform_limit',shell['platforms'])
    actual_shell['platforms']=actual
    actual_shell['floors']=list(shell['floors'][:actual+1])
    actual_shell['roof_z_m']=(actual+1)*shell['storey_height_m']
    scene={'id':graph['id'],'components':[],'holes':defaultdict(list),
           'surfaces':[],'stairs':[],'floors':list(actual_shell['floors']),
           'footprint':list(actual_shell['footprint_m']),'shell':actual_shell,
           'equipment_instances':[],'opening_owners':[],'equipment_port_reservations':[],
           'required_installation_spaces':[],
           '_trap_definitions':[deepcopy(u) for u in graph['units'].values() if u['kind']=='automatic_steam_trap'],
           'pullout_reservations':[],
           'stage':'Building and equipment only; own equipment nozzles, no process piping or valves'}
    scene['_progress_folder']=progress_folder
    scene['_progress_graph']=graph
    if progress_folder:
        from pathlib import Path
        import json
        from .generation_progress import atomic_json
        previous=Path(progress_folder)/'best_layout.json'
        scene['_best_count']=json.loads(previous.read_text())['equipment_count'] if previous.is_file() else 0
        scene['_trial_event']={'stage':'equipment_placement','bays':shell['bays'],'attempt':strategy_index}
        atomic_json(Path(progress_folder)/'progress.json',scene['_trial_event'])
    scene['_placement_boards'],scene['_placement_structural'],scene['_placement_components']=_preview_boards(graph,actual_shell)
    reference_units={u['id']:u for u in reference['equipment']} if reference else {}
    prototypes=[_prototype(unit,graph,actual_shell,rng,reference_units.get(unit['id'],{}).get('rotation_deg')) for unit in primary]
    if reserve_pipe_departures:
        from .layout_port_envelopes import envelopes, in_room, requirements
        required = requirements(graph)
        for prototype in prototypes:
            prototype['port_envelopes'] = envelopes(graph, prototype, actual_shell, required)
    from .filter_group_layout import definitions as filter_definitions
    scene['_filter_connections']=filter_definitions(graph)
    from .maintenance_geometry import level_maintenance
    for prototype in prototypes:
        unit=prototype['unit']
        prototype['maintenance_width_m']=maintenance_width_for(unit,rules)
        measured={'id':unit['id'],'kind':unit['kind'],'rect':prototype['rect'],
                  'parts':prototype['components'],'component_ids':[c['id'] for c in prototype['components']],
                  'occupied_levels':prototype['occupied_levels']}
        prototype['level_footprints']={level:level_maintenance(measured,unit,scene,level,rules,planning=True)
                                       for level in prototype['occupied_levels']}
        core=[c['box'] for c in prototype['components'] if c['kind'] in ('vessel','head','jacket_shell')]
        prototype['vessel_body_rect']=([min(c[0] for c in core),min(c[1] for c in core),
            max(c[3] for c in core),max(c[4] for c in core)]
            if core and unit['kind']=='vessel' else None)
    placed=[]
    # Reference poses stay fixed through all attempts, including pumps.
    # Only newly required equipment is placed freely.
    locked=[]
    outside_reference = {}
    for p in prototypes:
        u=p['unit'];old=reference_units.get(u['id'])
        if old:
            _placing(scene,u['id'])
            old_spec=old.get('dimension_spec',{})
            current=u.get('dimension_spec',{})
            saddle_shift=(current['leg_clearance_m']-old_spec['leg_clearance_m']
                if current.get('bottom_design_dn') and current.get('catalog_key')==old_spec.get('catalog_key')
                and u['level']==old['level'] else 0.)
            if abs(old['center'][2]+saddle_shift-u['center'][2])>1e-7:
                raise CandidateRefused('Saved equipment elevation differs from current equipment selection: '+u['id'])
            xy=[old['center'][i]-u['center'][i] for i in (0,1)]
            if reserve_pipe_departures:
                _, outside = in_room(p['port_envelopes'], xy, actual_shell)
                if outside:
                    outside_reference[u['id']] = outside
                    continue
            if not _can_place(p,xy,placed,scene,actual_shell,p['maintenance_width_m']):
                raise CandidateRefused('Physical space conflict with saved positions fixed: '+u['id'])
            _place(p,xy,scene,placed,pump_row=0 if u['kind']=='pump' else None)
            locked.append(u['id'])
    ordinary=[p for p in prototypes if p['unit']['kind']!='pump' and p['unit']['id'] not in locked]
    pump_first=attempt>=48 and attempt%2==0
    # Largest occupied projection first is a small greedy candidate list, not
    # a general constraint solver. A failed candidate remains a real refusal.
    if attempt<48:
        ordinary.sort(key=lambda p:-(p['rect'][2]-p['rect'][0])*(p['rect'][3]-p['rect'][1]))
    else:
        def maintenance_area(prototype):
            rect=prototype['rect'];width=prototype['maintenance_width_m']
            return (rect[2]-rect[0]+2*width)*(rect[3]-rect[1]+2*width)
        priority=(attempt-48)%5
        if priority==0:
            ordinary.sort(key=maintenance_area,reverse=True)
            ordinary_priority='maintenance_area'
        elif priority==1:
            ordinary.sort(key=lambda p:(len(p['occupied_levels']),maintenance_area(p)),reverse=True)
            ordinary_priority='occupied_levels_then_maintenance_area'
        elif priority==2:
            ordinary.sort(key=lambda p:(bool(p.get('extraction_options')),maintenance_area(p)),reverse=True)
            ordinary_priority='extraction_then_maintenance_area'
        elif priority==3:
            ordinary.sort(key=lambda p:(len(p['occupied_levels']),-maintenance_area(p)))
            ordinary_priority='fewest_occupied_levels_first'
        else:
            rng.shuffle(ordinary)
            ordinary_priority='shuffled_equipment_order'
    search_stats={'backtracking':attempt>=48,'position_check_limit':16384 if attempt>=48 else None,
                  'placement_choices_per_equipment':3 if attempt>=48 else 1,'branch_cutoffs':0,
                  'position_checks':0,'ordinary_position_checks':0,'pump_position_checks':0,
                  'ordinary_placements':0,'backtracks':0,'pump_row_searches':0,'pump_row_refusals':0,
                  'limit_reached':False,'equipment_placement_complete':False}
    pumps=[p for p in prototypes if p['unit']['kind']=='pump' and p['unit']['id'] not in locked]
    by_level=defaultdict(list)
    for prototype in pumps:by_level[prototype['unit']['level']].append(prototype)

    def place_pumps():
        for same_level in by_level.values():
            _placing(scene,[p['unit']['id'] for p in same_level])
            search_stats['pump_row_searches']+=1
            _pump_rows(same_level,actual_shell,placed,scene,maintenance,rng,
                       extended=attempt>=12,search_stats=search_stats)

    def fit_ordinary(index):
        if index==len(ordinary):
            if pump_first:return True
            saved=_layout_checkpoint(prototypes,scene,placed)
            try:
                place_pumps()
                return True
            except CandidateRefused:
                _restore_layout(saved,scene,placed)
                search_stats['pump_row_refusals']+=1
                if search_stats['limit_reached']:raise
                return False
        prototype=ordinary[index]
        _placing(scene,prototype['unit']['id'])
        required=prototype['maintenance_width_m']
        choice_present='_chosen_extraction' in prototype
        entry_choice=deepcopy(prototype.get('_chosen_extraction'))

        def restore_choice():
            if choice_present:prototype['_chosen_extraction']=deepcopy(entry_choice)
            else:prototype.pop('_chosen_extraction',None)

        # This list belongs to this prefix. A changed earlier position invokes
        # this function afresh, so its dependent maintenance-edge points change.
        candidates=_ordinary_candidates(prototype,actual_shell,scene,required,rng,direction,
                                        placed=placed,anchored=True)
        choices_tried=0
        for xy in candidates:
            if not _search_can_place(prototype,xy,placed,scene,actual_shell,required,search_stats):
                restore_choice()
                continue
            saved=_layout_checkpoint(prototypes,scene,placed)
            choices_tried+=1
            accepted=False
            try:
                _place(prototype,xy,scene,placed)
                search_stats['ordinary_placements']+=1
                accepted=fit_ordinary(index+1)
                if accepted:return True
            finally:
                if not accepted:
                    _restore_layout(saved,scene,placed)
                    restore_choice()
                    search_stats['backtracks']+=1
            # Spread the bounded search over earlier devices too, rather
            # than spending it on many positions of only the last device.
            if choices_tried>=search_stats['placement_choices_per_equipment']:
                search_stats['branch_cutoffs']+=1
                break
        restore_choice()
        return False

    try:
        if pump_first:place_pumps()
        if attempt>=48:
            if not fit_ordinary(0):
                raise CandidateRefused('Bounded backtracking found no complete layout. Candidate rejected.')
        else:
            for prototype in ordinary:
                _placing(scene,prototype['unit']['id'])
                required=prototype['maintenance_width_m']
                candidates=_ordinary_candidates(prototype,actual_shell,scene,required,rng,direction)
                chosen=next((xy for xy in candidates if _search_can_place(
                    prototype,xy,placed,scene,actual_shell,required,search_stats)),None)
                if chosen is None:
                    raise CandidateRefused('Equipment needs a single-side maintenance aisle on each occupied floor. Try another layout.')
                _place(prototype,chosen,scene,placed)
                search_stats['ordinary_placements']+=1
            place_pumps()
    except CandidateRefused as error:
        error.partial_scene=deepcopy({k:v for k,v in scene.items() if not k.startswith('_')})
        error.partial_graph=deepcopy(graph)
        error.search_statistics=deepcopy(search_stats)
        raise
    for unit in primary:
        for uid,jacket in unit.get('integral_jackets',{}).items():
            graph['units'][uid].update(deepcopy(jacket))
    search_stats['equipment_placement_complete']=True
    for unit in primary:
        for name,port in unit['ports'].items():
            if 'pos' in port:
                scene['equipment_port_reservations'].append({'owner':unit['id'],'port':name,
                    'pos':port['pos'],'normal':port.get('normal'),
                    'scope':'Reserve connecting stubs and root valves for a later generation stage'})
    scene['equipment_layout']={'attempt':strategy_index,'strategy_code':attempt,'actual_platforms':actual,
                              'candidate_direction':direction,'extended_whole_pump_rows':attempt>=12,
                              'platform_limit':shell['platforms'],
                              'method':'Build levels for equipment use. Provide four-side maintenance clearances on each level. Align pumps side by side at the axial faces.'}
    scene['equipment_layout']['search_statistics']=deepcopy(search_stats)
    if reference:
        scene['equipment_layout']['reference_changes']=[{'id':u['id'],
            'old_center':reference_units.get(u['id'],{}).get('center'),'new_center':u['center'],
            'pose_preserved':u['id'] in locked,'new_equipment':u['id'] not in reference_units,
            **({'reference_port_outside': outside_reference[u['id']]} if u['id'] in outside_reference else {})}
            for u in primary]
    if attempt>=48:
        scene['equipment_layout'].update(placement_order='pump_rows_then_ordinary' if pump_first else 'ordinary_then_pump_rows',
                                         ordinary_priority=ordinary_priority)
    for key in ('_progress_folder','_progress_graph','_current_equipment','_best_count','_trial_event'):
        scene.pop(key,None)
    del scene['_trap_definitions']
    del scene['_placement_boards']
    del scene['_placement_structural']
    del scene['_placement_components']
    del scene['_filter_connections']
    return scene
