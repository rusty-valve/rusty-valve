"""Check the shape and fit of assembled equipment flanges."""
from collections import Counter, defaultdict
import math

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .building_equipment_checks import EPS
from .storage_vessel_model import FLANGE_NUT_MM, PL_BORE_MM


class InvalidJoint(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise InvalidJoint(reason)


def _area(poly):
    return abs(float(poly[:,0] @ np.roll(poly[:,1], -1) - poly[:,1] @ np.roll(poly[:,0], -1)))/2


def _boundary_loops(polygons):
    nodes = []
    edges = Counter()
    for poly in polygons:
        indexes = []
        for p in poly:
            idx = next((i for i,q in enumerate(nodes) if np.linalg.norm(p-q) <= EPS), None)
            if idx is None:
                idx = len(nodes)
                nodes.append(p)
            indexes.append(idx)
        edges.update(tuple(sorted(edge)) for edge in zip(indexes,indexes[1:]+indexes[:1]))
    _require(max(edges.values(), default=0) <= 2, 'Duplicate overlapping flange layers')
    outside = defaultdict(set)
    for (a,b), count in edges.items():
        if count == 1:
            outside[a].add(b)
            outside[b].add(a)
    _require(outside and all(len(ns)==2 for ns in outside.values()), 'The physical flange perimeter has open edges')
    loops = []
    while outside:
        start = next(iter(outside))
        cycle = [start]
        prev, node = start, next(iter(outside[start]))
        while node != start:
            _require(node not in cycle, 'Flange section boundaries intersect')
            cycle.append(node)
            prev, node = node, next(i for i in outside[node] if i != prev)
        for i in cycle:
            del outside[i]
        loops.append(np.asarray([nodes[i] for i in cycle]))
    return loops


def _layer(names, actual, model, cache, p, axis, axes, heights, outer, bore, centres, hole):
    from . import piping_checks as common
    polygons = []
    for name in names:
        row = actual.get(name)
        _require(row and row['type']==int(mujoco.mjtGeom.mjGEOM_MESH), 'Flange sector has no physical mesh: '+str(name))
        verts = common._mesh_support_data(row,model,cache)[0]-p
        z = verts @ axis
        low = np.abs(z-heights[0]) <= EPS
        high = np.abs(z-heights[1]) <= EPS
        _require(np.all(low | high) and low.sum()==high.sum() and low.sum()>=3,
                 'Flange sector differs from the saved face or specified thickness: '+name)
        lower, upper = verts[low]@axes.T, verts[high]@axes.T
        _require(np.max(np.min(np.linalg.norm(lower[:,None]-upper, axis=2),axis=1))<=EPS,
                 'The two end profiles of the flange sector differ: '+name)
        polygons.append(lower[ConvexHull(lower).vertices])
    loops = _boundary_loops(polygons)
    loops.sort(key=_area, reverse=True)
    _require(len(loops)==2+len(centres), 'Flange bore or bolt-hole count differs')
    boundary, inner = loops[0], loops[1:]
    # The outer diameter is measured over polygon vertices; clear holes are
    # measured between opposite flats of the circumscribed bore/hole polygons.
    def circle(loop,center,radius,inscribed):
        delta=loop-center
        r=np.linalg.norm(delta,axis=1)
        if len(loop)<3 or np.min(r)<=EPS:
            return False
        angles=np.sort(np.mod(np.arctan2(delta[:,1],delta[:,0]),2*math.pi))
        pitch=2*math.pi/len(loop)
        if np.max(np.abs(np.diff(np.r_[angles,angles[0]+2*math.pi])-pitch))*radius > 2*EPS:
            return False
        expected=radius/math.cos(pitch/2) if inscribed else radius
        return np.max(np.abs(r-expected))<=EPS
    _require(circle(boundary,np.zeros(2),outer,False), 'Assembly diameter or boundary differs from the source nozzle schedule')
    for center,radius in [(np.zeros(2),bore), *[(q,hole) for q in centres]]:
        idx=next((i for i,ring in enumerate(inner) if circle(ring,center,radius,True)),None)
        _require(idx is not None, 'Actual openings differ from the bolt circle, bolt diameter or pipe bore')
        inner.pop(idx)
    net_area=_area(boundary)-sum(_area(loop) for loop in loops[1:])
    _require(abs(net_area-sum(_area(poly) for poly in polygons))<=EPS*2*math.pi*outer,
             'The flange section has missing or overlapping sectors')
    return {'pieces':len(names), 'axial_extent_m':list(heights),'outer_diameter_m':2*outer,
            'bore_diameter_m':2*bore,'bolt_holes':len(centres),'net_section_m2':net_area}


def _actual_cylinder(name, actual, start, end, radius):
    from . import piping_checks as common
    cylinder=common._cylinder(actual[name]) if name in actual else None
    _require(cylinder and abs(cylinder['radius']-radius)<=EPS and any(
        np.linalg.norm(np.asarray(cylinder['ends'][i])-start)<=EPS and
        np.linalg.norm(np.asarray(cylinder['ends'][1-i])-end)<=EPS for i in (0,1)),
        'Physical cylinder face, axis or radius differs: '+str(name))


def _measure_joint(endpoint, schedule, phys, joint, actual, model, cache, originals, used):
    dims=schedule['flange']
    p=np.asarray(phys['pos']);axis=np.asarray(phys['normal'])
    _require(abs(np.linalg.norm(axis)-1)<=EPS and np.count_nonzero(abs(axis)>EPS)==1,
             'Saved axis is not one of the building axes')
    ref=np.eye(3)[np.argmin(abs(axis))]
    axes=np.asarray([np.cross(ref,axis),ref])
    neck=phys['nozzle_component_id'];neck_radius=phys['actual_neck_outer_diameter_m']/2
    _actual_cylinder(neck,actual,np.asarray(phys['surface_pos']),p,neck_radius)
    count=dims['bolt_count'];pitch=dims['bolt_circle_mm']/2000;hole=dims['hole_diameter_mm']/2000
    bore=PL_BORE_MM[schedule['dn']]/2000
    t=dims['plate_thickness_mm']/1000;rf=dims['rf_height_mm']/1000;reach=t+rf
    outer=dims['outer_diameter_mm']/2000;rf_radius=dims['rf_diameter_mm']/2000
    thread=int(dims['bolt_thread'][1:]);shank_r=thread/2000
    flats,nut_height=FLANGE_NUT_MM[thread]
    nut_r=flats/2000;nut_l=nut_height/1000
    ids=schedule['component_ids']
    bolt_sources=[originals[k] for k in ids if originals[k]['kind']=='flange_bolt']
    _require(len(bolt_sources)==count and len(joint['bolts'])==count,'New bolt count differs from the source nozzle schedule')
    centres=[]
    for part in bolt_sources:
        mid=np.mean(part['ends'],axis=0)
        center=mid-axis*((mid-p)@axis)
        _require(abs(np.linalg.norm(center-p)-pitch)<=EPS,'Saved bolt circle radius differs')
        centres.append(center)
    centers2=(np.asarray(centres)-p)@axes.T
    angles=np.sort(np.arctan2(centers2[:,1],centers2[:,0]))
    _require(np.max(abs(np.diff(np.r_[angles,angles[0]+2*math.pi])-2*math.pi/count))*pitch<=EPS,
             'Original bolt holes are not evenly spaced')
    hardware=[]
    for b,center in zip(joint['bolts'],centres):
        for key,low,high,r in [('shank',-reach,reach,shank_r),('head',-reach-nut_l,-reach,nut_r),
                                ('nut',reach,reach+nut_l,nut_r)]:
            _actual_cylinder(b[key],actual,center+low*axis,center+high*axis,r)
            hardware.append(b[key])
    stub=joint['pipe_stub']
    _actual_cylinder(stub,actual,p,p+axis*reach,neck_radius)
    _require(shank_r < hole and neck_radius < bore and np.linalg.norm(joint['end']-(p+axis*reach))<=EPS,
             'No physical clearance for the neck or bolts')
    layers={}
    for tag,heights,r,is_plate in [('equipment_plate_components',(-reach,-rf),outer,True),
            ('equipment_rf_components',(-rf,0.),rf_radius,False),('rf_components',(0.,rf),rf_radius,False),
            ('plate_components',(rf,reach),outer,True)]:
        names=joint[tag];hardware.extend(names)
        layers[tag]=_layer(names,actual,model,cache,p,axis,axes,heights,r,bore,
                           centers2 if is_plate else [],hole)
    _require(len(set(hardware))==len(hardware) and set(hardware)==set(joint['components']) and
             not set(hardware)&used, 'Part IDs are duplicated, unregistered or shared between joints')
    _require(set(joint['replaced_source_components'])==set(ids)-{neck} and
             not (set(ids)-{neck})&actual.keys(),'Original short studs and solid flange were not fully replaced')
    return hardware,stub,{'component':endpoint,'kind':'measured_equipment_flange_pair',
        'pass':True,'source_face_m':p.tolist(),'axis':axis.tolist(),'layers':layers,
        'verified_bolts':count,'shank_diameter_m':2*shank_r,'hole_diameter_m':2*hole,
        'stub':stub,'stub_radius_m':neck_radius,'stub_end_m':(p+axis*reach).tolist(),
        'neck_dn':schedule['dn'],'replaced_components':sorted(set(ids)-{neck}),
        'bolt_radial_clearance_m':hole-shank_r}


def measure_joints(frozen,piping,model,actual):
    from .piping_direction_checks import _frozen_parts
    from .piping_requirements import P1_ROLES, port_level

    result={'hardware':set(),'stubs':set(),'rows':[],'errors':[]}
    levels=piping.get('equipment_flange_scope_levels')
    material=piping.get('material',{})
    scope=material.get('requirements',{}).get('equipment_flange_scope')
    service_records=[r for r in piping.get('reducers',[]) if r.get('equipment_flange_joint')]
    material_records=[r for r in material.get('interfaces',[]) if r.get('equipment_flange_joint')
                      and not r['equipment_flange_joint'].get('catalogue_joint')]
    catalogue=piping.get('catalogue_flange_joints',[])
    if catalogue:
        from .rule_checks import measured_catalogue_joints
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        for joint,check in zip(catalogue,measured_catalogue_joints(catalogue,model,data)):
            endpoint=joint['endpoint']
            result['rows'].append({**check,'component':endpoint,'kind':'measured_catalogue_flange',
                'stub':joint['pipe_stub'],'through_bore':joint['through_bore'],'pipe_dn':joint['pipe_dn']})
            if check['pass']:
                result['hardware'].update(joint['components'])
                result['stubs'].add(joint['pipe_stub'])
            else:
                result['errors'].append({'code':'PG17_EQUIPMENT_FLANGE_GEOMETRY','component':endpoint,
                    'route':None,'reason':'Measured catalogue face, bore or hollow transition differs','measurement':check})
    if levels is None and scope is None and not service_records and not material_records:
        return result
    graph,equipment=_frozen_parts(frozen)
    originals={c['id']:c for c in frozen.get('scene',{}).get('components',[])}
    expected=defaultdict(set)
    for edge in graph.get('edges',[]):
        service=edge.get('role') in P1_ROLES
        phase='P1' if service else 'P2'
        if service and levels is None and not service_records:
            continue
        if not service and (scope is None and not material_records or
                            scope=='service_steam_condensate_only' and edge.get('role')!='steam_condensate'):
            continue
        for endpoint in (edge['a'],edge['b']):
            uid,name=endpoint.split('.',1)
            unit=equipment.get(uid,{})
            if name not in unit.get('ports',{}):
                continue
            port=unit['ports'][name]
            if port.get('catalogue_face') or port.get('catalogue_missing'):
                continue
            if service and levels is not None and port_level(unit['ports'][name],unit) not in levels:
                continue
            if any(r['port']==name and r.get('connected_to_process') and r.get('flange')
                   for r in unit.get('nozzle_schedule',[])):
                expected[phase,endpoint].add(edge['id'])
    actual_records=defaultdict(list)
    for phase,records,edge_key in [('P1',service_records,'source_requirement'),('P2',material_records,'source_edge')]:
        for record in records:
            key=(phase,record.get('endpoint'))
            if record.get(edge_key) not in expected.get(key,()):
                result['errors'].append({'code':'PG17_EQUIPMENT_FLANGE_GEOMETRY','component':key[1],
                    'route':record.get(edge_key),'reason':'Assembly does not match the source process edge and saved nozzle'})
                continue
            actual_records[key].append(record['equipment_flange_joint'])
    cache={}
    for (phase,endpoint),edges in expected.items():
        try:
            matches=actual_records[phase,endpoint]
            _require(len(matches)==1,'Missing a unique installed mating flange')
            uid,name=endpoint.split('.',1);unit=equipment[uid]
            schedule=next(r for r in unit['nozzle_schedule'] if r['port']==name and r.get('connected_to_process'))
            hardware,stub,row=_measure_joint(endpoint,schedule,unit['ports'][name],matches[0],actual,
                                             model,cache,originals,result['hardware'])
            _require(stub not in result['stubs'],'The same stub is shared by more than one connection')
        except (InvalidJoint,KeyError,ValueError,IndexError,TypeError,QhullError) as exc:
            result['errors'].append({'code':'PG17_EQUIPMENT_FLANGE_GEOMETRY','component':endpoint,
                                     'route':None,'reason':str(exc)})
            continue
        row.update(phase=phase,source_edges=sorted(edges))
        result['hardware'].update(hardware);result['stubs'].add(stub);result['rows'].append(row)
    return result
