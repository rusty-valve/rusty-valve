"""Check equipment, flange and pipe connections in the saved model."""
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .catalogue_geometry import mesh
from .catalogue_surfaces import weld
from .material_rack import destination_rule, rack_run_evidence
from .piping_checks import _cylinder, _measured
from .storage_vessel import storage_service_checks
from .reactor_f_series_fittings import _frame


REFERENCE = 'decision and rule references'
PUMP_CONNECTION = 'Vessel bottom to pump'
GRATING_PENETRATIONS = 'Grating penetrations'
FLANGE_CLEARANCE = 'Equipment connection and clearance v2'
EPS = 1e-6


def _row(label, authority, evidence, passes):
    return {'rule':label,'authority':authority,'evidence':evidence,
            'status':'not applicable' if not evidence else 'passed' if all(passes) else 'fallback'}


def _ports_in_rack(frozen,piping,segments):
    graph=frozen['contracts']['sizing']['graph']
    levels={r['id']:r['level'] for r in frozen['contracts']['layout']['equipment']}
    material=piping.get('material',{})
    ports=material.get('requirements',{}).get('ports',{})
    zones=piping['rack_zones']
    results=[]
    for edge in graph['edges']:
        ep=edge['a']
        uid=ep.split('.')[0]
        joints={r['id']:r['position'] for r in material.get('junctions',[]) if r.get('position')}
        def actual_level(endpoint):
            device=endpoint.split('.')[0]
            level=ports.get(endpoint,{}).get('level',levels.get(device))
            if level is None and device in joints:
                level=max((i for i,z in enumerate(frozen['scene']['floors']) if z<=joints[device][2]),default=0)
            return level
        origin=actual_level(ep)
        target=actual_level(edge['b'])
        if origin is None:
            upstream=[e['a'].split('.')[0] for e in graph['edges'] if e['b'].split('.')[0]==uid]
            origin=next((levels[u] for u in upstream if u in levels),None)
        if origin is None or not (rule:=destination_rule(graph,levels,edge,origin,target)):
            continue
        zone=zones.get(rule['level'],zones.get(str(rule['level'])))
        bounds=[zone['bottom_z_m'],zone['top_z_m']]
        runs=[p['ends'] for p in segments.get(edge['id'],[])]
        destinations=[r for r in material.get('destination_rack_routes',[]) if r['edge']==edge['id']]
        last=destinations[0]['points'][-1] if destinations else None
        results.append({**rule,'edge':edge['id'],'rack_band_m':bounds,
                        **rack_run_evidence(runs,bounds,last)})
    return results


def _projected_hit(triangles, positions, normal):
    x,y,_=_frame(normal)
    frame=np.vstack((x,y)).T
    triangles=triangles@frame
    points=np.asarray(positions)@frame
    a=triangles[:,0]; u=triangles[:,1]-a; v=triangles[:,2]-a
    determinant=u[:,0]*v[:,1]-u[:,1]*v[:,0]
    valid=np.abs(determinant)>1e-14
    a,u,v,determinant=(arr[valid] for arr in (a,u,v,determinant))
    delta=points[:,None,:]-a
    s=(delta[:,:,0]*v[:,1]-delta[:,:,1]*v[:,0])/determinant
    t=(u[:,0]*delta[:,:,1]-u[:,1]*delta[:,:,0])/determinant
    return np.any((s>=-1e-7)&(t>=-1e-7)&(s+t<=1.+1e-7),axis=1)


def _mesh_connectivity(triangles):
    vertices, remap = weld(triangles.reshape(-1, 3))
    faces = remap.reshape(-1, 3)
    edges = np.column_stack((faces.ravel(), np.roll(faces, -1, axis=1).ravel()))
    graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(vertices), len(vertices)))
    shells = connected_components(graph, directed=False, return_labels=False)
    _, incidence = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return shells, bool(np.all(incidence == 2))


def measured_equipment_flanges(devices, model, data):
    """Measure one connected annular face on each visible device flange."""
    by_name = {model.geom(i).name: i for i in range(model.ngeom)}
    cache = {}
    angle = np.arange(96) * math.tau / 96
    samples = np.column_stack((np.cos(angle), np.sin(angle)))
    rows = []
    for device in devices:
        flanges = device.get('flange_surfaces', ())
        if not flanges and device.get('ports'):
            rows.append({'equipment': device['id'], 'pass': False, 'reason': 'Saved model lacks an index of physical equipment flanges'})
        for flange in flanges:
            gid = by_name.get(flange['component'])
            entry = {'equipment': device['id'], 'port': flange['source_tag'],
                     'component': flange['component'], 'ring_samples': len(samples),
                     'ring_samples_covered': 0, 'pass': False}
            rows.append(entry)
            if gid is None:
                continue
            material = model.geom_matid[gid]
            alpha = model.mat_rgba[material, 3] if material >= 0 else model.geom_rgba[gid, 3]
            if alpha <= EPS:
                continue
            vertices, faces = mesh(model, gid, cache)
            world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            frame = np.asarray(flange['frame'])
            center = np.asarray(flange['center_m'])
            ring = (samples * np.asarray(flange['radii_m']) * .95) @ frame[:2] + center
            axial = ((world - center) @ frame[2])[faces]
            lo, hi = flange['axial_limits_m']
            is_cap = (np.ptp(axial, axis=1) < 2*EPS)
            planes = np.unique(np.round(axial.mean(axis=1)[is_cap], 6))
            for level in sorted(planes[(planes >= lo-2*EPS) & (planes <= hi+2*EPS)], reverse=True):
                triangles = world[faces[np.max(abs(axial-level), axis=1) < 2*EPS]]
                hits = int(np.count_nonzero(_projected_hit(triangles, ring, frame[2])))
                entry['ring_samples_covered'] = max(entry['ring_samples_covered'], hits)
                if hits == len(samples):
                    shells, _ = _mesh_connectivity(triangles)
                    if shells == 1:
                        entry.update(face_plane_offset_m=float(level), connected_face=True, **{'pass': True})
                        break
            if flange['source_sector_count'] > 1:
                shells, watertight = _mesh_connectivity(world[faces])
                entry.update(saved_surface_components=shells, closed_mesh=watertight)
                entry['pass'] = entry['pass'] and shells == 1 and watertight
    return rows


def measured_catalogue_joints(joints,model,data):
    by_name={model.geom(i).name:i for i in range(model.ngeom)}
    cache={}; rows=[]
    for joint in joints:
        disk=[c for c in joint['components'] if '_plate' in c and c in by_name]
        faces=[];vertices=[]
        for name in disk:
            i=by_name[name]
            v,f=mesh(model,i,cache)
            v=v@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]
            faces.append(v[f]); vertices.append(v)
        point=np.asarray(joint['position']);normal=np.asarray(joint['normal'])
        x,y,_=_frame(normal)
        radius=joint['outer_m']/2-.002
        angles=np.arange(96)*math.tau/96
        ring=point+radius*(np.outer(np.cos(angles),x)+np.outer(np.sin(angles),y))
        covered=0; open_bore=False; axial_gap=None
        if faces:
            hits=_projected_hit(np.concatenate(faces),np.vstack((point,ring)),normal)
            open_bore=not bool(hits[0])
            covered=int(np.count_nonzero(hits[1:]))
            # Raised-face mate: check the first material along the native axis,
            # using both disk and the separately loaded raised rim.
            first=[]
            for name in joint['components']:
                if name in by_name and ('_rf' in name or '_plate' in name):
                    index=by_name[name]
                    v,_=mesh(model,index,cache)
                    v=v@data.geom_xmat[index].reshape(3,3).T+data.geom_xpos[index]
                    first.append(float(np.min((v-point)@normal)))
            axial_gap=min(first) if first else None
        present=bool(joint['source_components']) and all(c in by_name for c in joint['source_components'])
        transition=joint['transition']
        transition_ids=transition['parts']
        throat=[]
        for name in transition_ids:
            if name in by_name:
                index=by_name[name]
                v,f=mesh(model,index,cache)
                throat.append((v@data.geom_xmat[index].reshape(3,3).T+data.geom_xpos[index],f))
        q0,q1=np.asarray(transition['ends'])
        shaped=False
        if len(throat)==len(transition_ids) and throat:
            v=np.concatenate([v for v,_ in throat]); axial=(v-q0)@normal
            radial=np.linalg.norm(v-q0-np.outer(axial,normal),axis=1)
            end_z=float((q1-q0)@normal)
            actual_radii=[float(max(radial[abs(axial-z)<EPS],default=0.)) for z in (0.,end_z)]
            open_throat=not _projected_hit(np.concatenate([v[f] for v,f in throat]),[q0],normal)[0]
            shaped=(abs(axial.min())<EPS and abs(axial.max()-end_z)<EPS and
                    np.allclose(actual_radii,transition['outer_radii_m'],atol=EPS,rtol=0.) and open_throat)
        rows.append({'endpoint':joint['endpoint'],'id':joint['id'],'pipe_mate_disk_parts':len(disk),
                     'mating_axial_gap_m':axial_gap,'central_opening':open_bore,
                     'ring_samples_covered':covered,'ring_samples':len(angles),'source_parts_loaded':present,
                     'actual_hollow_transition':bool(shaped),
                     'pass':bool(open_bore and present and shaped and axial_gap is not None and abs(axial_gap)<EPS)})
    return rows


def check(frozen,scene,piping,model,data):
    actual=_measured(model,data)
    specifications={c['id']:c for c in scene['components']}
    segments={}
    for route in piping['routes']:
        if route['role']!='material_connection':
            continue
        for name in route['components']:
            if name in actual and specifications[name]['kind']=='pipe' and (part:=_cylinder(actual[name])):
                segments.setdefault(route.get('source_edge'),[]).append(part)
    rack=_ports_in_rack(frozen,piping,segments)
    ascending=[r for r in rack if r['kind']=='ascending_material']
    condensate=[r for r in rack if r['kind']=='horizontal_condenser_to_seal']
    seals=piping.get('material',{}).get('liquid_seals',[])
    for item in condensate:
        seal=next((s for s in seals if s['owner']==item['vessel']),None)
        item['seal']=seal
        item['pass']=item['pass'] and bool(seal and seal['geometric_depth_m']>0)
    devices=scene.get('catalogue_devices',[])
    pumps=[]
    for d in devices:
        if d['kind']!='pump':
            continue
        ports=d['ports']; suction=ports['in']; out=ports['out']
        ok=(abs(suction['normal'][2])<EPS and out['normal'][2]>1.-EPS and
            out['position'][2]>suction['position'][2] and suction['source_tag']=='suction')
        pumps.append({'id':d['id'],'suction':suction['normal'],'discharge':out['normal'],'pass':ok})
    flanges=measured_catalogue_joints(scene.get('catalogue_flange_joints',[]),model,data)
    equipment_flanges=measured_equipment_flanges(devices,model,data)
    drawing=[{'id':d['id'],'model':d['catalogue_key'],'drawing':d['facts'].get('manufacturer_drawing'),
              'pass':bool(d['facts'].get('manufacturer_drawing'))} for d in devices]
    boxes={c['id']:c['box'] for c in scene['components']}
    retired_boxes=[c['id'] for c in scene['components']
                   if c['kind']=='catalogue_equipment_envelope' or c['id'].endswith('_mating_envelope')]
    xml_boxes=[model.geom(i).name for i in range(model.ngeom)
               if model.geom(i).name.endswith('_mating_envelope') or model.geom(i).name in retired_boxes]
    avoid_boxes=[{'retired_boxes_in_scene':retired_boxes,'retired_boxes_in_xml':xml_boxes,
                  'pass':not retired_boxes and not xml_boxes}]
    supports=[]
    for s in scene.get('catalogue_supports',[]):
        beam=boxes.get(s['support_beam']);rect=s['shoe_rect_m']
        bottom=s['shoe_top_z_m']-s['shoe_thickness_m']
        contact=bool(beam and abs(beam[5]-bottom)<EPS and all(
            min(beam[i+3],rect[i+2])>max(beam[i],rect[i])+EPS for i in (0,1)))
        supports.append({'shoe':s['old_shoe_id'],'beam':s['support_beam'],'contact':contact,
                         'ear_contact_m2':s['shoe_contact_area_m2'],'pass':contact and s['pass']})
    drains=[]
    for d in devices:
        guide=d['facts'].get('s3_adjustment')
        if guide:
            delta=None if 'bore_low_m' not in guide else abs(guide['bore_low_m']-guide['shell_bottom_m'])
            ok=bool(guide.get('downward_shell_bottom_opening') or (delta is not None and delta<EPS))
            drains.append({'heater':d['id'],'opening':d['ports']['utility_out']['position'],
                           'vertical_gap_m':delta, 'pass':ok,**guide})
    storage=storage_service_checks(frozen['graph'],set(actual))
    for item in storage:
        uid=item.get('id',item.get('equipment',item.get('unit')))
        runs=[r for r in piping['routes'] if r['role']=='vessel_bottom_side'
              and any(ep==f'{uid}.bottom' for ep in (r.get('source',{}).get('endpoint'),r.get('equipment_endpoint')))]
        valves=[name for route in runs for name in route.get('valves',[])]
        item.update(side_runs=[r['id'] for r in runs],side_run_valves=valves,
                    **{'pass':bool(runs and valves and item['status']!='fail')})
    return [
        _row('Long cross-floor process run uses the destination rack and drops by the vessel',REFERENCE+'material equipment connection rules',ascending,[r['pass'] for r in ascending]),
        _row('Horizontal condenser liquid returns through the reactor-level rack to a liquid trap by the reactor',REFERENCE+'material equipment connection rules',condensate,[r['pass'] for r in condensate]),
        _row('Same-port flanges mate at the actual faces and clear bores',FLANGE_CLEARANCE,flanges,[r['pass'] for r in flanges]),
        _row('The bottom outlet elbow routes to the vessel side and its root valve fits the horizontal run',GRATING_PENETRATIONS,storage,[r['pass'] for r in storage]),
        _row('Pump uses a horizontal front inlet and upward upper discharge',PUMP_CONNECTION,pumps,[r['pass'] for r in pumps]),
        _row('Pipe-side flange has a continuous annular face',FLANGE_CLEARANCE,flanges,[r['ring_samples_covered']==r['ring_samples'] for r in flanges]),
        _row('continuous equipment flange face','equipment port and flange acceptance',equipment_flanges,[r['pass'] for r in equipment_flanges]),
        _row('model has a manufacturer drawing source',REFERENCE+'room assembly and site sequence',drawing,[r['pass'] for r in drawing]),
        _row('Heater S3 is at the lowest point of the shell side',REFERENCE+'material equipment connection rules',drains,[r['pass'] for r in drains]),
        _row('each reactor support shoe bears on a beam', 'equipment geometry acceptance',supports,[r['pass'] for r in supports]),
        _row('clearance boxes are hidden', 'equipment port and flange acceptance',avoid_boxes,[r['pass'] for r in avoid_boxes]),
    ]
