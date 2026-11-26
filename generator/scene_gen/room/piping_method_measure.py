"""Measure the installed utility network, fittings and root valves."""
from __future__ import annotations

from collections import Counter
import heapq
import math

import numpy as np

from .building_equipment_checks import EPS
from .piping_end_bundle import (
    DEMAND_ONLY_TERMINALS, DIRECT_CONNECTION_KINDS, station_root_keys,
)


def _plain(value):
    if isinstance(value, dict):
        return {key:_plain(item) for key,item in value.items()}
    if isinstance(value, (tuple,list,np.ndarray)):
        return [_plain(item) for item in value]
    return value.item() if isinstance(value,np.generic) else value


def _inline(ends, face, axis, maximum):
    axis=np.asarray(axis,dtype=float); axis/=np.linalg.norm(axis); face=np.asarray(face)
    return all(np.linalg.norm((d:=np.asarray(end)-face)-float(d@axis)*axis)<=EPS
               and abs(float(d@axis))<=maximum+EPS for end in ends)


def _horizontal_reference(frozen,graph,scene,piping,actual,by_route,native_rows,points,level):
    """Only report: lengths/exemptions never change connection/geometry status."""
    from .process import RULES
    from . import piping_checks as common
    floor=scene['floors'][level]; height=RULES['geometry_screen']['walk_height_m']
    original=frozen['contracts']['sizing']['graph']; edges={e['id']:e for e in original['edges']}
    units=original['units']; native={v['id']:v for v in native_rows if v['status']=='pass'}
    definitions={v['id']:v for v in graph.get('valves',[])}; point_map={p['id']:p for p in points}
    records={}
    for route in piping.get('routes',[]):
        edge=edges.get(route.get('source_requirement'),edges.get(route.get('source_edge'),{}))
        owners=[endpoint.split('.',1)[0] for endpoint in (edge.get('a',''),edge.get('b','')) if '.' in endpoint]
        kinds=[units.get(uid,{}).get('kind') for uid in owners]
        for segment in by_route.get(route['id'],[]):
            name=segment['id']; measured=actual.get(name)
            cylinder=common._cylinder(measured) if measured else None
            if not cylinder or abs(cylinder['ends'][1][2]-cylinder['ends'][0][2])>EPS:
                continue
            box=measured['box']
            if box[2]>=floor+height-EPS or box[5]<=floor+EPS:
                continue
            reason=None; proof=None; radius=cylinder['radius']; length=math.dist(*cylinder['ends'])
            # One valve-space is its actual package L. These derived local
            # extents identify reference exceptions; they never reject a pipe.
            for vid in route.get('valves',[]):
                valve=native.get(vid)
                if valve is None:
                    continue
                definition=definitions[vid]; L=definition['install_length_m']; R=3*radius
                for face in valve['actual_ports'].values():
                    axis=np.asarray(valve['actual_ports']['out'])-valve['actual_ports']['in']
                    if _inline(cylinder['ends'],face,axis,L+R):
                        reason='local_valve_spool';proof={'native_valve':vid,'actual_native_L_m':L,'bend_R_m':R};break
                point=point_map.get(route.get('source_requirement'))
                if reason is None and point and _inline(cylinder['ends'],point['position'],point['normal'],3*L+R+2*radius):
                    reason='normal_port_stub';proof={'original_endpoint':point['endpoint'],'native_L_m':L,
                        'basis':'Outlet estimate uses valve spaces, native valve length, elbow radius and any unequal-size straight'}
                if reason:
                    break
            if reason is None and level==0 and 'pump' in kinds:
                reason='ground_pump_local_connection';proof={'original_edge':edge.get('id'),'original_endpoint_kinds':kinds}
            if reason is None and 'condenser' in kinds and edge.get('role') in ('vapor','process_evaporated_vapor','condensate'):
                reason='condenser_process_direct_connection';proof={'original_edge':edge.get('id'),
                    'scope':'Direct vapor and condensate connections to the original condenser, without an extra length threshold'}
            row={'component':name,'routes':[route['id']],'actual_ends':cylinder['ends'],
                'length_m':length,'actual_lowest_outer_height_above_floor_m':box[2]-floor,
                'exception':reason,'exception_evidence':proof,
                'nonexception_reference':reason is None,'reference_only':True}
            if name in records:
                records[name]['routes'].append(route['id'])
            else:
                records[name]=row
    rows=list(records.values()); nonexception=[r for r in rows if r['nonexception_reference']]
    return {'count':len(nonexception),'total_length_m':sum(r['length_m'] for r in nonexception),
        'lengths_m':[r['length_m'] for r in nonexception],'rows':rows,
        'operating_band_height_m':height,'measurement':'Check actual header edges against the 0 to 2.2 m standing band',
        'scope':'Reference classification for the three direction cases uses physical ports, equipment roles and same-DN native stubs',
        'gate':False}


def _supply_crossing_evidence(source,join,segments,local,header_parts,actual,by_route,piping):
    """Show a real pipe that blocks the measured high run at side-entry Z."""
    from . import piping_checks as common
    from .piping_route_search import _segments_distance
    gap=piping['config']['pg_rules']['routing']['unconnected_crossing_clearance_m']
    excluded=set(local)|{s['id'] for s in header_parts}; obstacles={}
    for pieces in by_route.values():
        for piece in pieces:
            measured=actual.get(piece['id'])
            cylinder=common._cylinder(measured) if measured and piece['id'] not in excluded else None
            if cylinder:obstacles[piece['id']]=cylinder
    evidence=[]
    for piece in local.values():
        measured=actual.get(piece['id']); run=common._cylinder(measured) if measured else None
        if not run:continue
        a,b=map(np.asarray,run['ends']); radius=run['radius']
        if abs(a[2]-b[2])>EPS or a[2]<=join[2]+radius+EPS:continue
        if not (common._connected(source,[a],segments) and common._connected(b,[join],segments)):continue
        low_a=a.copy();low_b=b.copy();low_a[2]=low_b[2]=join[2]
        for name,obstacle in obstacles.items():
            p,q=map(np.asarray,obstacle['ends']); required=radius+obstacle['radius']+gap
            side_distance=_segments_distance(low_a,low_b,p,q)
            high_distance=_segments_distance(a,b,p,q)
            if (side_distance<required-EPS and high_distance>=required-EPS
                    and a[2]-radius>=max(p[2],q[2])+obstacle['radius']+gap-EPS):
                evidence.append({'actual_overpass_component':piece['id'],'actual_overpass_ends':run['ends'],
                    'actual_obstacle_component':name,'actual_obstacle_ends':obstacle['ends'],
                    'side_entry_projection_ends':[low_a.tolist(),low_b.tolist()],
                    'side_entry_outer_clearance_m':side_distance-radius-obstacle['radius'],
                    'actual_overpass_outer_clearance_m':high_distance-radius-obstacle['radius'],
                    'existing_clearance_m':gap})
    return evidence


def _method_corrections(frozen,scene,piping,actual,by_route,natives,points,connections,manual,directions,headers):
    """Read v1.3 corrections from the same loaded model, not guide labels."""
    from . import piping_checks as common
    routes=piping.get('routes',[]); route_map={r['id']:r for r in routes}
    units=frozen['contracts']['sizing']['graph']['units']
    connection_map={r['requirement']:r for r in connections}
    from .piping_requirements import requirements_from_frozen
    order_points=points+requirements_from_frozen(frozen,piping['config']).get('header_demands',[])
    for point in order_points[len(points):]:
        linked=[r for r in routes if r.get('source_edge')==point['id']]
        assigned={r.get('distribution_id') for r in linked}-{None}
        header=route_map.get(next(iter(assigned)),{}) if len(assigned)==1 else {}
        parts=[s for r in linked for s in by_route.get(r['id'],[])]
        connection_map[point['id']]={'assigned_header':header.get('id'),
            'actual_touch_points':common._touch_points(parts,by_route.get(header.get('id'),[]))}
    manual_map={r['requirement']:r for r in manual}
    elbow_rows=[r for r in directions['elbows']['rows'] if r.get('valid')]
    offgas_map={r['requirement']:r for r in directions.get('offgas_entries',{}).get('rows',[])}
    errors=[]; top_roots=[]; entries=[]; orders=[]
    def fail(code,point,**evidence):
        errors.append({'code':code,'requirement':point['id'],'endpoint':point['endpoint'],**evidence})
    for point in points:
        source=np.asarray(point['position']); normal=np.asarray(point['normal'])
        linked=[r for r in routes if r.get('source_requirement')==point['id']]
        local={s['id']:s for r in linked for s in by_route.get(r['id'],[])}
        ids=set(local); connection=connection_map[point['id']]
        roots=manual_map[point['id']]['actual_native_root_ids']
        # An upward frozen mouth must retain a straight vertical native root.
        if abs(normal[0])<=EPS and abs(normal[1])<=EPS and normal[2]>EPS:
            root={'requirement':point['id'],'endpoint':point['endpoint'],'applicable':True,
                  'actual_root_ids':roots,'status':'missing','actual_native_axis_positive_z':False,
                  'actual_pre_root_straight_positive_z':False,'after_root_first_turn':{'status':'missing'}}
            if len(roots)==1 and roots[0] in natives:
                native=natives[roots[0]]; inlet=np.asarray(native['actual_ports']['in'])
                outlet=np.asarray(native['actual_ports']['out']); delta=outlet-inlet
                root.update(actual_native_in=inlet.tolist(),actual_native_out=outlet.tolist(),
                    actual_native_axis_positive_z=bool(abs(delta[0])<=EPS and abs(delta[1])<=EPS and delta[2]>EPS))
                leads=[r for r in linked if common._connected(source,[inlet],by_route.get(r['id'],[]))]
                pre={s['id']:s for r in leads for s in by_route.get(r['id'],[])}
                pre_straight=bool(pre and inlet[2]>source[2]+EPS and common._near(inlet[:2],source[:2])
                    and all(s.get('path_kind')!='actual_elbow_piece' and all(
                        common._near(q[:2],source[:2]) and source[2]-EPS<=q[2]<=inlet[2]+EPS
                        for q in s['ends']) for s in pre.values()))
                root.update(actual_pre_root_route_ids=[r['id'] for r in leads],
                    actual_pre_root_components=list(pre),actual_pre_root_straight_positive_z=pre_straight)
                axial=[s for s in local.values() if s.get('path_kind')!='actual_elbow_piece' and all(
                    common._near(q[:2],outlet[:2]) and q[2]>=outlet[2]-EPS for q in s['ends'])]
                candidates=[]
                for elbow in elbow_rows:
                    if not set(elbow['components'])<=ids:continue
                    for port in elbow['ports'].values():
                        q=np.asarray(port['position'])
                        if (common._near(q[:2],outlet[:2]) and q[2]>=outlet[2]-EPS
                                and common._connected(outlet,[q],axial)):
                            candidates.append((float(q[2]),elbow))
                zone=piping.get('rack_zones',{}).get(point['level'],piping.get('rack_zones',{}).get(str(point['level']),{}))
                bottom=zone.get('bottom_z_m'); top=zone.get('top_z_m')
                first={'status':'missing','actual_rack_bottom_z_m':bottom,
                    'actual_rack_top_z_m':top,'basis':"Measure real elbow edges against that level's piping.rack_zones bounds"}
                if candidates:
                    _,elbow=min(candidates,key=lambda item:item[0])
                    boxes=[actual[name]['box'] for name in elbow['components']]
                    low=min(box[2] for box in boxes); high=max(box[5] for box in boxes)
                    passed=bool(bottom is not None and top is not None
                        and low>=bottom-EPS and high<=top+EPS)
                    first.update(status='pass' if passed else 'fail',elbow=elbow['id'],
                        actual_components=elbow['components'],actual_lowest_outer_z_m=low,actual_highest_outer_z_m=high)
                elif common._connected(outlet,connection['actual_touch_points'],axial):
                    first.update(status='not_applicable',reason='Actual coaxial straight from the valve to the header with no first turn')
                # A local condenser exception is not inferred merely from its
                # name: without a defined actual local extent keep it pending.
                if (first['status'] not in ('pass','not_applicable')
                        and units[point['equipment']].get('kind')=='condenser'):
                    first.update(status='pending',reason='Classify the local condenser connection from its physical scope without a length threshold')
                root['after_root_first_turn']=first
                root['status']='pass' if (manual_map[point['id']]['installed']
                    and root['actual_native_axis_positive_z'] and pre_straight
                    and first['status'] in ('pass','not_applicable')) else ('pending' if first['status']=='pending' else 'fail')
            top_roots.append(root)
            if root['status']!='pass':fail('METHOD_TOP_ROOT_CORRECTION',point,actual=root)
        # Actual last face/tangent at the assigned T, with no compulsory pair
        # of high routing layers. Supply prefers side/below; a proven actual
        # obstruction permits an overhead crossing and direct downward entry.
        terminals=[]
        for q in connection['actual_touch_points']:
            for s in local.values():
                if s.get('path_kind')=='actual_elbow_piece':continue
                for end,other in (s['ends'],list(reversed(s['ends']))):
                    if common._near(end,q) and not common._near(other,q):
                        terminals.append({'component':s['id'],'join':list(q),'actual_approach':list(other),
                            'actual_pipe_radius_m':s['radius'],
                            'direction_into_header':(np.asarray(q)-other).tolist()})
            for elbow in elbow_rows:
                if not set(elbow['components'])<=ids:continue
                for key,port in elbow['ports'].items():
                    if common._near(port['position'],q):
                        tangent=np.asarray(port['tangent'])*(1. if key=='out' else -1.)
                        other=elbow['ports']['in' if key=='out' else 'out']['position']
                        terminals.append({'component':elbow['components'][0],'elbow':elbow['id'],'join':list(q),
                            'actual_approach':other,'actual_pipe_radius_m':elbow['pipe_outer_radius_m'],
                            'direction_into_header':(tangent*elbow['centerline_radius_m']).tolist()})
        entry={'requirement':point['id'],'endpoint':point['endpoint'],'medium':point['medium'],
            'assigned_header':connection['assigned_header'],'actual_terminal_faces':terminals,
            'two_routing_layers_required':False,'status':'missing'}
        if terminals and connection['connected']:
            if point['medium']=='offgas':
                proof=offgas_map.get(point['id'],{}).get('actual_joins',[])
                good=bool(proof) and all(all(join.get(key) for key in (
                    'actual_top_down','above_actual_header','frozen_source_connected','same_frozen_floor',
                    'actual_horizontal_header')) for join in proof)
                good &= all(abs(t['direction_into_header'][0])<=EPS and abs(t['direction_into_header'][1])<=EPS
                            and t['direction_into_header'][2]<-EPS for t in terminals)
                entry.update(required_entry='actual_Z_down',original_offgas_actual_proof=proof)
            else:
                header_parts=by_route.get(connection['assigned_header'],[])
                segments=list(local.values())+[{'ends':list(natives[vid]['actual_ports'].values())}
                    for vid in roots if vid in natives]
                proofs=[]
                for terminal in terminals:
                    q=terminal['join']; vector=np.asarray(terminal['direction_into_header'])
                    axes=np.flatnonzero(abs(vector)>EPS)
                    touched=[s for s in header_parts if common._on_segment(q,*s['ends'])]
                    header_axes={i for s in touched for i in np.flatnonzero(
                        abs(np.asarray(s['ends'][1])-s['ends'][0])>EPS)}
                    genuine=bool(len(axes)==1 and len(header_axes)==1 and header_axes<={0,1}
                        and int(axes[0]) not in header_axes and common._connected(source,[q],segments))
                    downward=bool(len(axes)==1 and axes[0]==2 and vector[2]<-EPS)
                    above=bool(touched and terminal['actual_approach'][2]-terminal['actual_pipe_radius_m']
                        >q[2]+max(s['radius'] for s in touched)-EPS)
                    crossing=(_supply_crossing_evidence(source,q,segments,local,header_parts,actual,by_route,piping)
                        if genuine and downward and above else [])
                    permitted=bool(genuine and (not downward or above and crossing))
                    proofs.append({'component':terminal['component'],'actual_join':q,
                        'actual_direction_into_header':vector.tolist(),'actual_assigned_header_touch':bool(touched),
                        'actual_header_axis_indices':sorted(header_axes),'actual_source_connected':genuine,
                        'entry_kind':'from_above_after_obstruction' if downward else 'side_or_from_below',
                        'actual_approach_above_header':above,'actual_crossing_obstacles':crossing,'pass':permitted})
                good=bool(proofs) and all(proof['pass'] for proof in proofs)
                entry.update(required_entry='actual_side_or_below_or_obstructed_overpass_Z_down',
                    actual_supply_entry_proofs=proofs,
                    scope='Prefer side entry. Downward entry requires measured endpoint, assigned header connectivity and physical clearance evidence.')
            entry['status']='pass' if good else 'fail'
        entries.append(entry)
        if entry['status']!='pass':fail('METHOD_SERVICE_ENTRY_CORRECTION',point,actual=entry)
    for header in (r for r in routes if r.get('role')=='floor_header'):
        members=[p for p in order_points if connection_map[p['id']]['assigned_header']==header['id']]
        pieces=by_route.get(header['id'],[]); order={'header':header['id'],'status':'missing','same_plane_required':False}
        if members and pieces:
            delta=np.asarray(pieces[0]['ends'][1])-pieces[0]['ends'][0]
            axes=[i for i in range(3) if abs(delta[i])>EPS]
            if len(axes)==1 and axes[0]<2:
                axis=axes[0]; ordered=sorted(members,key=lambda p:(p['position'][axis],p['position'][1-axis],
                    p['position'][2],*p['normal'],p['dn']))
                rows=[]
                for point in ordered:
                    touches=connection_map[point['id']]['actual_touch_points']
                    scalars=sorted({float(q[axis]) for q in touches})
                    rows.append({'requirement':point['id'],'source_position':point['position'],
                        'actual_tap_axis_coordinates_m':scalars})
                good=bool(all(len(r['actual_tap_axis_coordinates_m'])==1 for r in rows)
                    and all(a['actual_tap_axis_coordinates_m'][0]<=b['actual_tap_axis_coordinates_m'][0]+EPS
                            for a,b in zip(rows,rows[1:])))
                order.update(status='pass' if good else 'fail',actual_axis_index=axis,source_order=rows)
        orders.append(order)
        if order['status']!='pass':errors.append({'code':'METHOD_TAP_ORDER_CORRECTION','header':header['id'],'actual':order})
    return {'rule_version':'method_v1.4','status':'fail' if errors or any(h['actual']!=1 for h in headers) else 'pass',
        'shared_headers':headers,'top_roots':top_roots,'service_entries':entries,'source_axis_order':orders,
        'errors':errors,'scope':'Check all four corrections on the same actual model. Missing valves, routing or evidence fail the corresponding check.'}


def _device_connection_bends_rows(piping,by_route,natives,valve_parts,points,connections,manual,directions):
    """Report measured device paths; uninstalled reservations are never used."""
    from . import piping_checks as common
    connection_map={r['requirement']:r for r in connections}
    manual_map={r['requirement']:r for r in manual}
    rows=[]
    for point in points:
        connection=connection_map[point['id']]; root_row=manual_map[point['id']]
        linked=[r for r in piping.get('routes',[]) if r.get('source_requirement')==point['id']]
        local={s['id']:s for r in linked for s in by_route.get(r['id'],[])}
        component_ids={name for route in linked for name in route.get('components',[])}
        roots=[vid for vid in root_row['actual_native_root_ids'] if vid in natives]
        segments=list(local.values())+[valve_parts[vid] for vid in roots if vid in valve_parts]
        measured_elbows=[r for r in directions['elbows']['rows']
            if r.get('components') and set(r['components'])<=component_ids]
        elbows=[r for r in measured_elbows if r.get('valid')]
        source=connection['actual_source_position']; nodes=[]
        def node(position):
            index=next((i for i,p in enumerate(nodes) if common._near(p,position)),None)
            if index is None:
                index=len(nodes); nodes.append(list(position))
            return index
        source_index=node(source)
        for segment in segments:
            for position in segment['ends']:node(position)
        taps=[]
        for position in connection['actual_touch_points']:
            if not any(common._near(position,p) for p in taps):taps.append(list(position))
            node(position)
        adjacency=[{} for _ in nodes]
        for segment in segments:
            contained=sorted((i for i,p in enumerate(nodes) if common._on_segment(p,*segment['ends'])),
                key=lambda i:math.dist(nodes[i],segment['ends'][0]))
            for a,b in zip(contained,contained[1:]):
                distance=math.dist(nodes[a],nodes[b])
                adjacency[a][b]=distance; adjacency[b][a]=distance
        distances={source_index:0.}; queue=[(0.,source_index)]
        while queue:
            distance,index=heapq.heappop(queue)
            if distance>distances[index]+EPS:continue
            for other,length in adjacency[index].items():
                candidate=distance+length
                if candidate<distances.get(other,math.inf)-EPS:
                    distances[other]=candidate; heapq.heappush(queue,(candidate,other))
        def distance(position):
            index=next((i for i,p in enumerate(nodes) if common._near(p,position)),None)
            return distances.get(index,math.inf)
        turns=[]
        for elbow in elbows:
            ports=elbow['ports']; inlet=ports['in']; outlet=ports['out']; radius=elbow['centerline_radius_m']
            corner=np.asarray(inlet['position'])+np.asarray(inlet['tangent'])*radius
            ends=sorted((inlet['position'],outlet['position']),key=distance)
            turns.append({'elbow':elbow['id'],'route':elbow['route'],'actual_components':elbow['components'],
                'actual_entry_face_position':ends[0],'actual_exit_face_position':ends[1],
                'actual_tangent_intersection_position':corner.tolist(),'actual_arc_center':elbow['center'],
                'actual_centerline_radius_m':radius,'actual_angle_degrees':elbow['angle_degrees'],
                'source_reachable':math.isfinite(distance(ends[0]))})
        turns.sort(key=lambda row:distance(row['actual_entry_face_position']))
        sequence=[{'kind':'equipment_port','position':source}]
        root_ports=[]; events=[]
        for vid in roots:
            ports=natives[vid]['actual_ports']; root_ports.append({'valve':vid,'actual_ports':ports})
            for face,position in ports.items():
                events.append((distance(position),{'kind':'root_valve_face','valve':vid,'face':face,'position':position}))
        for turn in turns:
            events.append((distance(turn['actual_entry_face_position']),{'kind':'elbow',
                'elbow':turn['elbow'],'position':turn['actual_tangent_intersection_position'],
                'actual_entry_face_position':turn['actual_entry_face_position'],
                'actual_exit_face_position':turn['actual_exit_face_position']}))
        events.extend((distance(position),{'kind':'header_tap','header':connection['assigned_header'],
            'position':position}) for position in taps)
        sequence.extend(dict(event,source_reachable=math.isfinite(d)) for d,event in sorted(events,key=lambda item:item[0]))
        rows.append({'requirement':point['id'],'endpoint':point['endpoint'],'medium':point['medium'],
            'level':point['level'],'actual_route_ids':[r['id'] for r in linked],
            'actual_source_position':source,'actual_root_ports':root_ports,
            'actual_connected':connection['connected'],'actual_root_installed':root_row['installed'],
            'actual_verified_bend_count':len(turns),'actual_turns':turns,
            'unverified_elbow_ids':[r['id'] for r in measured_elbows if not r.get('valid')],
            'actual_header':connection['assigned_header'],'actual_header_tap_positions':taps,
            'source_to_header_sequence':sequence,
            'scope':'Report measured pipes, native valve faces and validated elbows. Infer corners from physical tangent intersections and exclude unused reservations.'})
    return rows


def _public_header_layout_rows(piping,by_route):
    """Report each actual public header axis/elevation/lateral coordinate."""
    rows=[]
    for header in piping.get('routes',[]):
        if header.get('role')!='floor_header':continue
        parts=by_route.get(header['id'],[])
        axes={int(i) for s in parts for i in np.flatnonzero(
            abs(np.asarray(s['ends'][1])-s['ends'][0])>EPS)}
        axis=next(iter(axes)) if len(axes)==1 and axes<={0,1} else None
        positions=np.asarray([p for s in parts for p in s['ends']])
        row={'header':header['id'],'medium':header['medium'],'level':header['level'],
            'actual_components':[s['id'] for s in parts],'actual_axis_index':axis,
            'actual_axis':'XY'[axis] if axis is not None else None,
            'actual_center_z_m':None,'actual_cross_coordinate_m':None,'actual_axis_interval_m':None}
        if axis is not None and len(positions):
            cross=1-axis
            row.update(actual_cross_axis='XY'[cross],
                actual_center_z_m=float(positions[0,2]) if np.ptp(positions[:,2])<=EPS else None,
                actual_cross_coordinate_m=float(positions[0,cross]) if np.ptp(positions[:,cross])<=EPS else None,
                actual_axis_interval_m=[float(positions[:,axis].min()),float(positions[:,axis].max())])
        rows.append(row)
    return rows


def measure_platform(frozen,graph,scene,piping,model,data):
    """Measure this platform's original points/native roots using loaded XML."""
    from .evaluation_common import require_environment
    from .piping_requirements import requirements_from_frozen
    from . import piping_checks as common
    from .piping_direction_checks import check_directions
    from .piping_operations import check_operations
    from .pullout_reservations import check_pullout_reservations
    environment=require_environment()
    levels=piping.get('platform_levels',[piping.get('platform_level')])
    if (not isinstance(levels,(list,tuple)) or not levels or any(isinstance(level,bool)
            or not isinstance(level,int) or not 0<=level<len(scene['floors']) for level in levels)):
        raise ValueError('platform_levels must list actual floor indices. Otherwise use the single platform_level.')
    levels=sorted(set(levels)); primary_level=levels[0] if len(levels)==1 else None
    required=requirements_from_frozen(frozen,piping['config'])
    points=[p for p in required['service_points'] if p['level'] in levels]
    header_points=points+[p for p in required.get('header_demands',[]) if p['level'] in levels]
    actual=common._measured(model,data); components={c['id']:c for c in scene['components']}
    routes=piping.get('routes',[]); route_map={r['id']:r for r in routes}
    by_route,route_errors,pipe_sizes=common._route_segments(routes,components,actual,
        piping['config']['pipe_outer_diameter_mm'],model,data)
    native_rows,native_errors,valve_segments,native_geoms=common._valves(graph,actual)
    natives={v['id']:v for v in native_rows if v['status']=='pass'}
    valve_parts={v['valve']:v for v in valve_segments}; definitions={v['id']:v for v in graph.get('valves',[])}
    errors=list(route_errors)+list(native_errors); connections=[]; manual=[]; headers=[]
    for level,medium in sorted({(p['level'],p['medium']) for p in header_points}):
        matches=[r for r in routes if r.get('role')=='floor_header' and r.get('level')==level and r.get('medium')==medium]
        original_points=[p for p in header_points if p['level']==level and p['medium']==medium]
        assigned={r.get('distribution_id') for r in routes
                  if r.get('source_requirement',r.get('source_edge')) in {p['id'] for p in original_points}}
        headers.append({'medium':medium,'level':level,'actual_headers':[r['id'] for r in matches],
                        'expected':1,'actual':len(matches),
                        'assigned_header_ids':sorted(assigned-{None}),
                        'original_requirements':[p['id'] for p in original_points],
                        'scope':'One common header per independent service and level. Keep supply and return separate.'})
        if len(matches)!=1:
            errors.append({'code':'PLATFORM_HEADER_COUNT','medium':medium,'level':level,'expected':1,'actual':len(matches)})
    expected_pairs={(p['level'],p['medium']) for p in header_points}
    for route in routes:
        if route.get('role')=='floor_header' and (route.get('level'),route.get('medium')) not in expected_pairs:
            errors.append({'code':'PLATFORM_UNREQUIRED_HEADER','header':route['id'],
                           'level':route.get('level'),'medium':route.get('medium')})
    for point in points:
        level=point['level']
        linked=[r for r in routes if r.get('source_requirement')==point['id']]
        hids={r.get('distribution_id') for r in linked}; header=route_map.get(next(iter(hids))) if len(hids)==1 else None
        correct=bool(header and header.get('role')=='floor_header' and header.get('level')==level
                     and header.get('medium')==point['medium'] and linked and all(
                         r.get('role')=='equipment_connection' and r.get('medium')==point['medium']
                         and r.get('level')==level and r.get('dn')==point['dn'] for r in linked))
        local=[s for r in linked for s in by_route.get(r['id'],[])]
        local += [valve_parts[v] for r in linked for v in r.get('valves',[]) if v in valve_parts]
        header_parts=by_route.get(header['id'],[]) if header else []
        touches=common._touch_points(local,header_parts)
        connected=bool(correct and common._connected(point['position'],touches,local))
        # The valve's legacy level is the proposed operator standing level.
        # The linked routes and traversed native ports prove the service floor.
        roots=list(dict.fromkeys(v for r in linked for v in r.get('valves',[]) if v in natives
            and natives[v]['dn']==point['dn'] and definitions[v].get('piping_level',r.get('level'))==level
            and definitions[v].get('equipment',point['equipment'])==point['equipment']
            and definitions[v].get('at',point['endpoint'])==point['endpoint']))
        traversed=[]
        for vid in roots:
            ports=natives[vid]['actual_ports']
            external=[s for s in local if s.get('valve')!=vid]
            if (common._connected(point['position'],[ports['in']],external)
                    and common._connected(ports['out'],touches,external)):
                traversed.append(vid)
        installed=bool(correct and len(roots)==1 and traversed==roots)
        connections.append({'requirement':point['id'],'endpoint':point['endpoint'],'medium':point['medium'],
            'level':level,'assigned_header':header.get('id') if header else None,
            'actual_touch_points':touches,'actual_source_position':point['position'],
            'connected':connected,'scope':'Follow the original port through its physical pipe and native flow path to the assigned floor header'})
        manual.append({'requirement':point['id'],'endpoint':point['endpoint'],'expected_dn':point['dn'],
            'actual_native_root_ids':roots,'actual_traversed_roots':traversed,'installed':installed})
        if not connected:
            errors.append({'code':'PLATFORM_SOURCE_NOT_CONNECTED','requirement':point['id'],'endpoint':point['endpoint']})
        if not installed:
            errors.append({'code':'PLATFORM_ROOT_NOT_INSTALLED','requirement':point['id'],'endpoint':point['endpoint']})
    uses=Counter(v for row in manual for v in row['actual_native_root_ids'])
    for row in manual:
        if any(uses[v]>1 for v in row['actual_native_root_ids']):
            row['installed']=False
            errors.append({'code':'PLATFORM_ROOT_SHARED','requirement':row['requirement'],'valves':row['actual_native_root_ids']})
    geometry=common._geometry(frozen,scene,piping,model,data,actual,native_geoms,by_route,native_rows)
    errors.extend(geometry['errors'])
    pullout=check_pullout_reservations(graph,scene,model)
    errors.extend(pullout['violations'])
    original_directions=check_directions(frozen,graph,scene,piping,model,data)
    direction_errors=[e for e in original_directions['errors'] if e['code'].startswith('PG17_')]
    errors.extend(direction_errors)
    corrections=_method_corrections(frozen,scene,piping,actual,by_route,natives,points,
        connections,manual,original_directions,headers)
    errors.extend(corrections['errors'])
    offgas=[]
    for row in original_directions.get('offgas_entries',{}).get('rows',[]):
        point=next((p for p in points if p['id']==row['requirement']),None)
        if not point:
            continue
        joins=row['actual_joins']
        passed=bool(joins) and all(all(join.get(key) for key in (
            'actual_top_down','above_actual_header','frozen_source_connected','same_frozen_floor',
            'actual_horizontal_header')) for join in joins)
        offgas.append({'requirement':point['id'],'actual_joins':joins,'pass':passed,
                       'header_to_main_scope':'not_applicable_stage1'})
        if not passed:
            errors.append({'code':'PLATFORM_OFFGAS_ENTRY','requirement':point['id'],'actual_joins':joins})
    references=[dict(_horizontal_reference(frozen,graph,scene,piping,actual,by_route,native_rows,points,level),
                     level=level) for level in levels]
    reference_rows={}
    for per_level in references:
        for row in per_level['rows']:
            if row['component'] not in reference_rows:
                reference_rows[row['component']]=dict(row,levels=[per_level['level']])
            else:
                existing=reference_rows[row['component']]
                existing['levels'].append(per_level['level'])
                existing['nonexception_reference'] |= row['nonexception_reference']
    nonexception=[row for row in reference_rows.values() if row['nonexception_reference']]
    reference={'count':len(nonexception),'total_length_m':sum(row['length_m'] for row in nonexception),
        'lengths_m':[row['length_m'] for row in nonexception],'rows':list(reference_rows.values()),
        'by_level':references,'operating_band_height_m':references[0]['operating_band_height_m'],'gate':False,
        'scope':'Classify physical floor operating bands and deduplicate actual component IDs. Report exceptions and distances.'}
    # Deliberately last, report only: inaccessible grips or swept handles do
    # not alter material collision or completed connection measurements.
    operations=check_operations(graph,scene,model,data)
    full_device=len(required['service_points']); floors=len(frozen['scene']['floors'])
    temporary=0
    fixed_obligations=[]
    bundle = piping.get('end_bundle_station')
    for medium,level in station_root_keys(required,bundle,floors):
        users=[p for p in required['service_points'] if p['medium']==medium and p['level']==level]
        supply_headers=[r for r in routes if r.get('role')=='floor_header'
                        and r.get('medium')==medium and r.get('level')==level] if users else []
        fixed_obligations.append({'medium':medium,'level':level,'expected':1,
            'actual_supply_header_ids':[r['id'] for r in supply_headers],
            'unused_outlet_required':not users})
    fixed=sum(row['expected'] for row in fixed_obligations)
    directions={key:original_directions[key] for key in ('actual_straight_total','checked',
        'nonorthogonal_violations','straight_nonorthogonal_violations','native_valve_axis_violations',
        'fitting_axis_violations','rows','coverage','elbows')}
    directions.update(status='fail' if direction_errors else 'pass',errors=direction_errors,
        scope='Reuse PG17 physical direction and fitting checks. Record the final downward off-gas segment separately.',
        downstream_main_not_applicable=[e for e in original_directions['errors'] if e['code']=='PG13_OFFGAS_TOP_VERTICAL_DOWN'])
    return _plain({'stage':'method_v1_3_platform','platform_level':primary_level,'platform_levels':levels,
        'status':'fail' if errors else 'pass',
        'errors':errors,'headers':headers,'connections':{'expected':len(points),'connected':sum(r['connected'] for r in connections),'rows':connections},
        'manual_valves':{'expected':len(points),'installed':sum(r['installed'] for r in manual),
            'actual_attached':len(graph.get('valves',[])),'actual_verified_native':len(natives),
            'rows':manual,'native_packages':native_rows,'pipe_sizes':pipe_sizes},
        'geometry':geometry,'actual_material_penetration_count':sum(e['code']=='P1_SOLID_COLLISION' for e in geometry['errors']),
        'geometry_measurement_error_count':sum(e['code']!='P1_SOLID_COLLISION' for e in geometry['errors']),
        'pullout':pullout,
        'actual_pullout_blocked_equipment_count':sum(e['code']=='PULLOUT_BLOCKED' for e in pullout['violations']),
        'pullout_scope':'Remeasure protected volumes and keep the precise original drive-mount exception. Report protected-space occupancy separately from solid penetration.',
        'directions':directions,'platform_offgas_entries':offgas,'method_corrections':corrections,
        'device_connection_bends_rows':_device_connection_bends_rows(piping,by_route,natives,valve_parts,
            points,connections,manual,original_directions),
        'public_header_layout_rows':_public_header_layout_rows(piping,by_route),
        'operation_band_horizontal_reference':reference,'operation_report':operations,
        'original_whole_room_scope':{'device_roots':full_device,'workstation_temporary':temporary,
            'workstation_fixed':fixed,'whole_original_expected':full_device+temporary+fixed,
            'workstation_fixed_obligations':fixed_obligations,
            'current_platform_expected':len(points),'current_levels_expected':len(points),
            'selected_levels':levels,'remaining_workstation_next_stage':temporary+fixed},
        'operation_ratio_gate':False,'horizontal_reference_gate':False,'physics_steps':0,'environment':environment,
        'scope':'Measure equipment-to-header connections, native root valves, solid penetration and PG17. Report operation access separately.'})


def _end_bundle_link_evidence(header,main,routes,by_route,native_rows,native_segments,graph,
                              elbow_measurements,cap_candidates,closure,bundle,*,deferred_roots=False):
    """Measure the new end connection; no planned point/bend earns a pass."""
    from . import piping_checks as common
    from .piping_service_group_checks import floor_link_evidence
    hid=header['id']; medium=header['medium']; level=header['level']
    row={'header':hid,'main':main['id'] if main else None,'medium':medium,'level':level,
        'method':'end_bundle_v1','bundle_mode':bundle.get('mode'),'connected':False,'pass':False,'valves':[],
        'actual_bend_count':0,'actual_same_vertical_plane':False}
    parts=by_route.get(hid,[]);main_parts=by_route.get(main['id'],[]) if main else []
    if not parts or not main_parts:
        row['reason']='Missing physical header or same-service main';return row
    delta=np.asarray(parts[0]['ends'][1])-parts[0]['ends'][0]
    axes=np.flatnonzero(abs(delta)>EPS)
    if len(axes)!=1 or axes[0] not in (0,1):
        row['reason']='The physical header is not a single-axis horizontal pipe';return row
    axis=int(axes[0]);cross=1-axis; positions=[p for s in parts for p in s['ends']]
    straight=all(s.get('path_kind')!='actual_elbow_piece' and all(
        abs(p[cross]-positions[0][cross])<=EPS and abs(p[2]-positions[0][2])<=EPS
        for p in s['ends']) for s in parts)
    if not straight:
        row['reason']='Physical header components are not on one horizontal line';return row
    ends=[min(positions,key=lambda p:p[axis]),max(positions,key=lambda p:p[axis])]
    main_position=main_parts[0]['ends'][0]
    near_index=min(range(2),key=lambda i:abs(ends[i][axis]-main_position[axis]))
    near=ends[near_index];far=ends[1-near_index]
    outward=np.asarray(far)-near;outward/=np.linalg.norm(outward)
    far_closed=closure(parts,header['dn'],exact=far,outward=outward)
    # The cap inventory was recovered from actual loaded cylinders. Match the
    # measured far cap radius, or the unchanged OD rule if that cap is missing.
    expected_radius=bundle['_pipe_od'][str(header['dn'])]/2000.+.034
    near_caps=[name for name,cylinder in cap_candidates.items()
        if abs(cylinder['radius']-expected_radius)<=EPS
        and any(common._near(p,near) for p in cylinder['ends'])]
    matching=[r for r in routes if r.get('role')=='floor_branch'
        and r.get('distribution_id')==hid and r.get('main_id')==main['id']
        and r.get('medium')==medium and r.get('level')==level
        and r.get('service_group')==header.get('service_group')]
    roots=set(v for r in matching for v in r.get('valves',[]))
    roots.update(v['id'] for v in graph.get('valves',[]) if v.get('role')=='utility_floor_root'
        and v.get('distribution_id')==hid and v.get('main_id')==main['id'])
    native={v['id']:v for v in native_rows if v['status']=='pass'}
    link_parts={s['id']:s for r in matching for s in by_route.get(r['id'],[])}
    declared={name for r in [header,*matching] for name in r.get('components',[])}
    actual_elbows=[e for e in elbow_measurements['rows'] if e.get('components')
        and set(e['components'])<=declared]
    bends=[e for e in actual_elbows if e.get('valid')]
    flow_parts=[*parts,*link_parts.values(),*[s for s in native_segments if s['valve'] in roots]]
    main_vertical=all(common._near(s['ends'][0][:2],s['ends'][1][:2]) for s in main_parts)
    plane=float(main_position[cross])
    deviations=[abs(float(p[cross])-plane) for s in flow_parts for p in s['ends']]
    same_plane=bool(flow_parts and all(d<=EPS for d in deviations))
    direct = medium=='offgas' or (
        bundle.get('terminals_policy')==DEMAND_ONLY_TERMINALS and medium=='water_return')
    if direct:
        link={'connected':bool(not roots and not matching
            and any(common._on_segment(near,*s['ends']) for s in main_parts)
            and common._connected(far,[near],parts)),
            'scope':'The horizontal header end connects directly to the same-service vertical main'}
        main_touches=[near] if link['connected'] else []
        endpoint_join=bool(link['connected']);expected_bends=0
    elif deferred_roots and not roots and len(matching)==1 and matching[0].get('root_valve_pending'):
        branch=list(link_parts.values())
        touches=common._touch_points(branch,parts)
        endpoint_join=bool(touches and all(common._near(p,near) for p in touches))
        main_touches=common._touch_points(branch,main_parts)
        main_ends=[p for segment in main_parts for p in segment['ends']]
        link={'connected':bool(endpoint_join and main_touches
                and common._connected(near,main_ends,branch+main_parts)),
              'root_valve_pending':True,'scope':'The measured header end connects to its riser by a continuous branch. Root valve placement follows.'}
        expected_bends=2
    else:
        link=floor_link_evidence(header,main,near,routes,by_route,native_rows,native_segments,graph)
        outs=[r for r in matching if r['id'] in link.get('out_routes',[])]
        out_parts=[s for r in outs for s in by_route.get(r['id'],[])]
        touches=common._touch_points(out_parts,parts)
        endpoint_join=bool(touches and all(common._near(p,near) for p in touches))
        ins=[r for r in matching if r['id'] in link.get('in_routes',[])]
        main_touches=common._touch_points([s for r in ins for s in by_route.get(r['id'],[])],main_parts)
        expected_bends=2
    root_topology=bool(link.get('root_valve_pending') or
        len(roots)==(0 if direct else 1) and all(vid in native for vid in roots))
    topology=bool(main_vertical and endpoint_join and far_closed['pass'] and not near_caps and same_plane
        and root_topology
        and len(bends)==expected_bends and all(e.get('valid') for e in actual_elbows))
    row.update(route_ids=[r['id'] for r in matching],valves=sorted(roots),
        actual_flow=link,actual_flow_connected=bool(link.get('connected')),
        actual_header_axis_index=axis,actual_plane_axis_index=cross,actual_plane_coordinate_m=plane,
        actual_main_vertical=main_vertical,
        actual_plane_max_deviation_m=max(deviations,default=None),actual_same_vertical_plane=same_plane,
        actual_header_near_point=near,actual_header_far_point=far,
        actual_endpoint_join=endpoint_join,actual_main_tap_positions=main_touches,
        actual_far_end_closure=far_closed,actual_near_end_blind_ids=near_caps,
        actual_bend_count=len(bends),expected_bend_count=expected_bends,
        actual_bends=[{k:e[k] for k in ('id','route','components','ports','center','centerline_radius_m','angle_degrees')} for e in bends],
        unverified_elbow_ids=[e['id'] for e in actual_elbows if not e.get('valid')],
        connected=bool(link.get('connected')),
        scope='Measure the header face, service main, root valve, elbows, coplanarity and far-end blind. Retain equipment tee checks.')
    row['pass']=bool(link.get('connected') and topology)
    return row


def measure_room(frozen,graph,scene,piping,model,data):
    """Add only real workstation obligations to the inherited platform ruler."""
    from . import piping_checks as common
    from .piping_requirements import requirements_from_frozen
    from .piping_direction_checks import _flange_axis
    from .piping_service_group_checks import floor_link_evidence
    result=measure_platform(frozen,graph,scene,piping,model,data)
    bundle=piping.get('end_bundle_station')
    end_bundle=bool(isinstance(bundle,dict) and bundle.get('version')=='end_bundle_v1')
    errors=list(result['errors']); req=requirements_from_frozen(frozen,piping['config'])
    routes=piping.get('routes',[]); route_map={r['id']:r for r in routes}
    actual=common._measured(model,data); components={c['id']:c for c in scene['components']}
    by_route,_,_=common._route_segments(routes,components,actual,piping['config']['pipe_outer_diameter_mm'],model,data)
    native_rows,_,native_segments,native_geom_owner=common._valves(graph,actual)
    natives={v['id']:v for v in native_rows if v['status']=='pass'}
    definitions={v['id']:v for v in graph.get('valves',[])}
    original_ids={c['id'] for c in frozen['scene']['components']}
    cap_candidates={name:c for name,row in actual.items() if name not in original_ids
        and name not in native_geom_owner and (c:=common._cylinder(row)) is not None
        and abs(math.dist(*c['ends'])-.018)<=EPS}
    cache={}; flanges=[]
    for record in scene.get('flange_connections',[]):
        shape=_flange_axis(record,components,actual,model,cache,piping['config']['pipe_outer_diameter_mm'])
        if shape:flanges.append({'id':record['id'],**shape})
    def closure(parts,dn,*,start=None,exact=None,outward=None):
        radius=piping['config']['pipe_outer_diameter_mm'][str(dn)]/2000.; matches=[]
        for name,cylinder in cap_candidates.items():
            if abs(cylinder['radius']-(radius+.034))>EPS:continue
            for index,q in enumerate(cylinder['ends']):
                n=np.asarray(cylinder['ends'][1-index])-q;n/=np.linalg.norm(n)
                if exact is not None and not common._near(q,exact):continue
                if outward is not None and np.linalg.norm(n-np.asarray(outward))>EPS:continue
                touching=[]
                for segment in parts:
                    if segment.get('path_kind')=='actual_elbow_piece':continue
                    for end,other in (segment['ends'],list(reversed(segment['ends']))):
                        delta=np.asarray(other)-q
                        if (common._near(end,q) and float(delta@n)<-EPS
                                and np.linalg.norm(delta-float(delta@n)*n)<=EPS):
                            touching.append(segment['id'])
                if not touching or start is not None and not common._connected(start,[q],parts):continue
                matching_flanges=[f['id'] for f in flanges if abs(f['radius']-(radius+.034))<=EPS
                    and any(common._near(p,q) for p in f['ends'])
                    and np.linalg.norm(np.cross(np.asarray(f['ends'][1])-f['ends'][0],n))<=EPS]
                if len(matching_flanges)==1:
                    matches.append({'cap':name,'actual_face':q,'actual_radius_m':cylinder['radius'],
                        'actual_thickness_m':math.dist(*cylinder['ends']),'actual_flange':matching_flanges[0],
                        'actual_terminal_pipe_geoms':touching})
        return {'pass':len(matches)==1,'actual_closures':matches}
    mains={}; main_rows=[]; main_ends=[]
    for medium in req['active_services']:
        selected=[r for r in routes if r.get('role')=='workstation_main' and r.get('medium')==medium]
        if len(selected)!=1:
            errors.append({'code':'ROOM_MAIN_COUNT','medium':medium,'actual':[r['id'] for r in selected]});continue
        main=selected[0]; mains[medium]=main; pieces=by_route.get(main['id'],[])
        vertical=bool(pieces) and all(common._near(s['ends'][0][:2],s['ends'][1][:2])
            and math.dist(*s['ends'])>EPS for s in pieces)
        main_rows.append({'main':main['id'],'medium':medium,'actual_vertical':vertical,
                          'actual_segments':pieces})
        if not vertical:
            errors.append({'code':'ROOM_MAIN_NOT_VERTICAL','main':main['id'],'medium':medium})
        top=max((p for s in pieces for p in s['ends']),key=lambda p:p[2],default=None)
        closed=closure(pieces,main['dn'],exact=top,outward=[0.,0.,1.]) if top is not None else {'pass':False,'actual_closures':[]}
        main_ends.append({'main':main['id'],'medium':medium,**closed})
        if not closed['pass']:
            errors.append({'code':'ROOM_MAIN_END_NOT_CLOSED','main':main['id'],'medium':medium,'actual':closed})
    for route in routes:
        if route.get('role')=='workstation_main' and route.get('medium') not in req['active_services']:
            errors.append({'code':'ROOM_UNREQUIRED_MAIN','main':route['id'],'medium':route.get('medium')})
    header_proofs={}; header_rows=[]
    for header in (r for r in routes if r.get('role')=='floor_header'):
        main=mains.get(header.get('medium')); positions=[p for s in by_route.get(header['id'],[]) for p in s['ends']]
        proof=(_end_bundle_link_evidence(header,main,routes,by_route,native_rows,native_segments,graph,
                   result['directions']['elbows'],cap_candidates,closure,
                   {**bundle,'_pipe_od':piping['config']['pipe_outer_diameter_mm']},
                   deferred_roots=piping.get('valve_placement')=='role_trace') if end_bundle else
               floor_link_evidence(header,main,positions[0],routes,by_route,native_rows,native_segments,graph)
               if main and positions else {'connected':False,'reason':'No physical main or header for this service'})
        header_proofs[header['id']]=proof;header_rows.append(proof)
        if not proof['connected']:
            errors.append({'code':'ROOM_HEADER_NOT_CONNECTED_TO_MAIN','header':header['id'],
                           'medium':header.get('medium'),'actual':proof})
        if end_bundle and not proof.get('pass'):
            errors.append({'code':'END_BUNDLE_HEADER_TO_MAIN_TOPOLOGY','header':header['id'],
                           'medium':header.get('medium'),'actual':proof})
    device_taps=[]
    takeoff_rows=[]
    takeoff_checks=[]
    if end_bundle and 'planned_takeoff_spacing' in bundle.get('spacing',{}):
        from .piping_end_bundle import takeoff_spacing
        for connection in bundle['connections']:
            own = ([connection['header_id']] if connection['kind'] in DIRECT_CONNECTION_KINDS else
                   [r['id'] for r in routes if r.get('role')=='floor_branch'
                    and r.get('connection_id')==connection['id']])
            main_parts=by_route.get(connection['main_id'],[])
            touches=common._touch_points([part for route in own for part in by_route.get(route,[])],main_parts)
            actual_taps=[]
            for point in touches:
                if not any(common._near(point,old) for old in actual_taps):
                    actual_taps.append(list(point))
            takeoff_rows.append({'id':connection['id'],'main_id':connection['main_id'],
                                 'kind':connection['kind'],'level':connection['level'],
                                 'main_tap':actual_taps[0] if len(actual_taps)==1 else None,
                                 'actual_touch_points':actual_taps})
        takeoff_checks=takeoff_spacing(bundle['risers'],takeoff_rows)
        errors.extend({'code':'ROOM_WORKSTATION_TAKEOFF_SPACING',**row} for row in takeoff_checks if not row['pass'])
        result['workstation_takeoff_spacing']={'takeoffs':takeoff_rows,'pairs':takeoff_checks}
    if end_bundle:
        from .piping_direction_checks import _through_segments
        for connection in result['connections']['rows']:
            header=route_map.get(connection['assigned_header'],{})
            pieces=by_route.get(header.get('id'),[])
            taps=[]
            for position in connection['actual_touch_points']:
                if not any(common._near(position,p) for p in taps):taps.append(position)
            through=bool(taps and all(_through_segments(position,pieces) for position in taps))
            row={'requirement':connection['requirement'],'endpoint':connection['endpoint'],
                'header':header.get('id'),'actual_tap_positions':taps,
                'actual_header_through_both_sides':through,
                'actual_source_connected':connection['connected'],
                'pass':bool(through and connection['connected']),
                'scope':'Equipment taps connect to the header body. Each tee must have continuous header on both sides.'}
            device_taps.append(row)
            if not row['pass']:
                errors.append({'code':'END_BUNDLE_DEVICE_BODY_TEE','requirement':connection['requirement'],'actual':row})
    station_rows=[]
    def unused_station_terminal(medium,level,dn):
        role='floor_branch'
        main=mains.get(medium)
        matching=[r for r in routes if r.get('medium')==medium and r.get('level')==level and r.get('role')==role]
        ids=list(dict.fromkeys(v for r in matching for v in r.get('valves',[]) if v in natives
            and natives[v]['dn']==dn and definitions[v].get('medium')==medium and definitions[v].get('level')==level
            and definitions[v].get('role')=='utility_floor_root'))
        proof={'actual_main_to_in':False,'actual_out_closed':False,'actual_in_routes':[],'actual_out_routes':[]}
        if len(ids)==1 and main:
            vid=ids[0]; own=[r for r in matching if vid in r.get('valves',[]) and r.get('main_id')==main['id']]
            identities={r.get('connection_id') for r in own};ports=natives[vid]['actual_ports']
            ins=[r for r in own if any(common._on_segment(ports['in'],*s['ends']) for s in by_route.get(r['id'],[]))]
            outs=[r for r in own if any(common._on_segment(ports['out'],*s['ends']) for s in by_route.get(r['id'],[]))]
            inp=[s for r in ins for s in by_route.get(r['id'],[])];outp=[s for r in outs for s in by_route.get(r['id'],[])]
            main_parts=by_route.get(main['id'],[])
            identity_ok=bool(len(identities)==1 and definitions[vid].get('main_id',main['id'])==main['id']
                and definitions[vid].get('connection_id',next(iter(identities),None)) in identities
                and not ({r['id'] for r in ins}&{r['id'] for r in outs}))
            linked=bool(ins and identity_ok and common._connected(ports['in'],[p for s in main_parts for p in s['ends']],inp+main_parts))
            closed=closure(outp,dn,start=ports['out'])
            proof.update(actual_main_to_in=linked,actual_out_closed=closed['pass'],actual_closure=closed,
                actual_in_routes=[r['id'] for r in ins],actual_out_routes=[r['id'] for r in outs],
                same_connection=identity_ok,actual_native_in=ports['in'],actual_native_out=ports['out'])
        good=bool(len(ids)==1 and proof['actual_main_to_in'] and proof['actual_out_closed'])
        row={'requirement':f'{medium}/{level}/{role}','medium':medium,'level':level,'role':role,
             'expected_dn':dn,'actual_native_root_ids':ids,'installed':good,'actual_flow':proof}
        station_rows.append(row)
        if not good:errors.append({'code':'ROOM_STATION_TERMINAL_INCOMPLETE',**row})
    root_keys = station_root_keys(req,bundle,len(scene['floors']))
    demand_only = end_bundle and bundle.get('terminals_policy')==DEMAND_ONLY_TERMINALS
    if demand_only:
        expected_roots = set(root_keys)
        errors.extend({'code':'ROOM_UNREQUIRED_STATION_BRANCH','route':route['id']}
                      for route in routes if route.get('role')=='floor_branch'
                      and (route.get('medium'),route.get('level')) not in expected_roots
                      and by_route.get(route['id']))
    for medium,level in root_keys:
        users=[p for p in req['service_points'] if p['medium']==medium and p['level']==level]
        headers=[r for r in routes if r.get('role')=='floor_header' and r.get('medium')==medium and r.get('level')==level]
        if not users and not demand_only:
            unused_station_terminal(medium,level,req['service_templates'][medium]['branch_dn'])
        else:
            if not headers:
                station_rows.append({'requirement':f'{medium}/{level}/floor_branch','installed':False,
                                     'actual_native_root_ids':[]})
            for h in headers:
                proof=header_proofs.get(h['id'],{});ids=proof.get('valves',[])
                good=bool(proof.get('connected') and len(ids)==1)
                station_rows.append({'requirement':f'{medium}/{level}/floor_branch/{h["id"]}',
                    'header':h['id'],'medium':medium,'level':level,'role':'floor_branch',
                    'actual_native_root_ids':ids,'installed':good,'actual_flow':proof})
    source_rows=[]
    old_rows={r['requirement']:r for r in result['connections']['rows']}
    for point in req['service_points']:
        row=dict(old_rows.get(point['id'],{'requirement':point['id'],'endpoint':point['endpoint'],'connected':False}))
        proof=header_proofs.get(row.get('assigned_header'),{})
        row['actual_header_to_main']=bool(proof.get('connected'))
        row['connected']=bool(row['connected'] and row['actual_header_to_main'])
        source_rows.append(row)
        if not row['connected']:errors.append({'code':'ROOM_SOURCE_NOT_CONNECTED','requirement':point['id'],'actual':row})
    missing=req.get('missing_service_points',())
    for point in missing:
        source_rows.append({'requirement':point['id'],'endpoint':point['endpoint'],'medium':point['medium'],
                            'connected':False,'reason':point['reason']})
        errors.append({'code':'CATALOGUE_PORT_SHORTAGE','requirement':point['id'],'actual':point})
    manual=[dict(row) for row in result['manual_valves']['rows']]+station_rows
    manual.extend({'requirement':p['id'],'endpoint':p['endpoint'],'installed':False,
                   'reason':p['reason']} for p in missing)
    assigned=Counter(v for row in manual for v in row.get('actual_native_root_ids',[]))
    for row in manual:
        if any(assigned[v]>1 for v in row.get('actual_native_root_ids',[])):
            row['installed']=False
            errors.append({'code':'ROOM_REQUIRED_ROOT_SHARED','requirement':row['requirement'],
                           'valves':row['actual_native_root_ids']})
    expected=result['original_whole_room_scope']['whole_original_expected']+len(missing)
    if len(manual)!=expected or sum(r['installed'] for r in manual)!=expected:
        errors.append({'code':'ROOM_REQUIRED_VALVE_COUNT','expected':expected,
                       'actual_required_rows':len(manual),'installed':sum(r['installed'] for r in manual)})
    result['original_whole_room_scope'].update(
        workstation_installed=sum(r['installed'] for r in station_rows),
        remaining_workstation_next_stage=sum(not r['installed'] for r in station_rows))
    temporary_ids=[vid for vid in natives if definitions[vid].get('role')=='temporary_outlet']
    temporary_routes=[r for r in routes if r.get('role')=='temporary_outlet']
    temporary_report={'required':0,'actual_verified_native_count':len(temporary_ids),
        'actual_verified_native_ids':temporary_ids,
        'actual_route_ids':[r['id'] for r in temporary_routes if by_route.get(r['id'])],
        'actual_component_ids':sorted({name for r in temporary_routes for name in r.get('components',[]) if name in actual}),
        'scope':'Temporary ports are outside required valve counts. Report any residual ports separately.'}
    result.update(stage='method_v1_3_room',status='fail' if errors else 'pass',errors=errors,
        platform_connections=result['connections'],platform_manual_valves=result['manual_valves'],
        connections={'expected':len(source_rows),'connected':sum(r['connected'] for r in source_rows),'rows':source_rows},
        manual_valves={'expected':expected,'installed':sum(r['installed'] for r in manual),
            'actual_attached':len(graph.get('valves',[])),'actual_verified_native':len(natives),
            'rows':manual,'native_packages':native_rows},
        workstation_mains=main_rows,workstation_main_ends=main_ends,header_to_main=header_rows,
        header_to_main_bends_rows=header_rows if end_bundle else [],
        end_bundle_device_body_taps=device_taps,
        workstation_terminals=station_rows,
        workstation_temporary_outlets=temporary_report,
        scope=('Check whole-room source-level connectivity, mains and root valves. Off-gas and cooling-water return use direct merges.'
               if demand_only else 'Check whole-room source-level connections, mains and supply roots or spare blind outlets'))
    return _plain(result)
