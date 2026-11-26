"""Find a pipe route along one floor while avoiding equipment and other obstacles."""
from __future__ import annotations

from collections import Counter
import itertools
import math
import time

import numpy as np

from .evaluation_common import require_environment
from .piping_elbows import plan_polyline, coarse_radius, ElbowUnavailable, is_axis_aligned
from .piping_port_contacts import flat_pipe_clears, outward_pipe_bounds
from .recipe import CandidateRefused
from .rule_identity import band


EPS=1e-8
_NON_SOLIDS={'operation_stance','standing_reservation','operation_reservation',
             'full_travel_reservation','operation_box','reserved_operation'}
_EXCEPTIONS={None,'local_stub','pump_local','condenser_short'}


class LaneUnavailable(CandidateRefused):
    def __init__(self,code,detail,**evidence):
        self.code=code;self.evidence={'code':code,'detail':detail,**evidence}
        super().__init__(code+': '+detail)


def _point(value,label):
    try:p=np.asarray(value,dtype=float)
    except (TypeError,ValueError) as exc:raise LaneUnavailable('LANE_INPUT',label+'missing finite three-dimensional coordinates') from exc
    if p.shape!=(3,) or not np.isfinite(p).all():raise LaneUnavailable('LANE_INPUT',label+'missing finite three-dimensional coordinates')
    return p


def _positive(value,label,zero=False):
    if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not math.isfinite(value) or value<0 or (not zero and value==0):
        raise LaneUnavailable('LANE_INPUT',label+' must be finite'+('nonnegative' if zero else 'positive')+'measured dimension')
    return float(value)


def _direction(value,label):
    p=_point(value,label);length=float(np.linalg.norm(p))
    if length<=EPS or np.count_nonzero(abs(p/length)>EPS)!=1:
        raise LaneUnavailable('LANE_NON_AXIAL_PORT',label+' must follow a physical building axis')
    axis=int(np.argmax(abs(p)));result=np.zeros(3);result[axis]=1. if p[axis]>0 else -1.
    return result


def _clean(points):
    result=[]
    for point in points:
        p=np.asarray(point,float)
        if result and np.max(abs(p-result[-1]))<=EPS:continue
        while len(result)>=2 and is_axis_aligned(result[-2],result[-1]) and is_axis_aligned(result[-1],p) and is_axis_aligned(result[-2],p) and float((result[-1]-result[-2])@(p-result[-1]))>=0:
            result.pop()
        result.append(p)
    return result


def _segment_distance(a,b,p,q):
    u,v,w=b-a,q-p,a-p
    aa,bb,cc,dd,ee=float(u@u),float(u@v),float(v@v),float(u@w),float(v@w)
    if aa<=1e-24 and cc<=1e-24:return float(np.linalg.norm(a-p))
    if aa<=1e-24:return float(np.linalg.norm(a-p-np.clip(ee/cc,0,1)*v))
    if cc<=1e-24:return float(np.linalg.norm(a+np.clip(-dd/aa,0,1)*u-p))
    pairs=[(0.,np.clip(ee/cc,0,1)),(1.,np.clip((ee+bb)/cc,0,1)),
           (np.clip(-dd/aa,0,1),0.),(np.clip((bb-dd)/aa,0,1),1.)]
    denominator=aa*cc-bb*bb
    if denominator>1e-15*aa*cc:
        t,k=(bb*ee-cc*dd)/denominator,(aa*ee-bb*dd)/denominator
        if 0<=t<=1 and 0<=k<=1:pairs.append((t,k))
    return min(float(np.linalg.norm(a+t*u-p-k*v)) for t,k in pairs)


def _segment_box(a,b,lo,hi):
    lower,upper=0.,1.;delta=b-a
    for axis in range(3):
        if abs(delta[axis])<=EPS:
            if a[axis]<=lo[axis]+EPS or a[axis]>=hi[axis]-EPS:return False
        else:
            one,two=(lo[axis]-a[axis])/delta[axis],(hi[axis]-a[axis])/delta[axis]
            lower=max(lower,min(one,two));upper=min(upper,max(one,two))
    return upper>=lower and upper>EPS and lower<1-EPS


def _obstacles(components):
    from .grating import grating
    result=[];ignored=[]
    for c in components:
        if grating(c):continue
        if c.get('kind') in _NON_SOLIDS or c.get('operation_only') or c.get('full_travel_only'):
            ignored.append(c.get('id'));continue
        item={'id':c['id'],'kind':c.get('kind'),'box':np.asarray(c['box'],float),'round':None,'obb':None}
        if item['box'].shape!=(6,) or not np.isfinite(item['box']).all():
            raise LaneUnavailable('LANE_OBSTACLE_GEOMETRY','Physical component lacks a complete box',component=c['id'])
        if c.get('shape') in ('cylinder','capsule'):
            item['round']=(_point(c['ends'][0],'obstacle.ends'),_point(c['ends'][1],'obstacle.ends'),
                           _positive(c['radius'],'obstacle.radius'))
        if c.get('obb'):
            obb=c['obb'];centre=_point(obb['center'],'obb.center');axes=np.asarray(obb['axes'],float)
            half=_point(obb['half_extents'],'obb.half_extents')
            if axes.shape!=(3,3) or not np.isfinite(axes).all() or np.max(abs(axes.T@axes-np.eye(3)))>EPS or np.any(half<0):
                raise LaneUnavailable('LANE_OBSTACLE_GEOMETRY','Invalid physical OBB axes or dimensions',component=c['id'])
            item['obb']=(centre,axes,half)
            extent=abs(axes)@half;item['box']=np.r_[centre-extent,centre+extent]
        result.append(item)
    return result,ignored


def _hits(a,b,obstacle,radius,clearance):
    # Extraction is a reserved empty volume, not a pipe/column requiring an
    # extra PG14 gap. Preserve its original faces and the actual pipe radius.
    clearance=0. if obstacle['kind']=='pullout_reservation' else clearance
    # A finite flat-ended cylinder is not a capsule. The actual component box
    # truncates its axial extent before the centreline-distance narrow phase;
    # genuine capsule boxes already include their real round end caps.
    box=obstacle['box']
    if not _segment_box(a,b,box[:3]-radius-clearance,box[3:]+radius+clearance):
        return False
    rounded=obstacle['round']
    if rounded is not None:
        p,q,r=rounded
        return _segment_distance(a,b,p,q)<radius+r+clearance-EPS
    if obstacle['obb'] is not None:
        centre,axes,half=obstacle['obb'];a=axes.T@(a-centre);b=axes.T@(b-centre)
        return _segment_box(a,b,-half-radius-clearance,half+radius+clearance)
    box=obstacle['box'];return _segment_box(a,b,box[:3]-radius-clearance,box[3:]+radius+clearance)


def _contact_cut(a,b,obstacle,radius,clearance,source,target,normal,end_direction,first,last):
    """Only the precise first normal joint / final measured tee may touch."""
    if first and obstacle['id'] in source.get('contact_ids',()):
        p=_point(source['position'],'source.position');box=obstacle['box']
        far=float(np.where(normal>0,box[3:],box[:3])@normal)
        if np.max(abs(a-p))<=EPS and far<=float(p@normal)+EPS and float((b-a)@normal)>0:
            return None  # Sole outward first arm; this component stays for every other arm.
    if last and obstacle['id'] in target.get('contact_ids',()):
        p=_point(target['position'],'target.position')
        if np.max(abs(b-p))<=EPS:
            rounded=obstacle['round']
            if rounded is not None:
                one,two,r=rounded;axis=(two-one)/np.linalg.norm(two-one)
                if abs(float(axis@end_direction))<=EPS and _segment_distance(p,p,one,two)<=EPS:
                    # Cut only the final local tee overlap/clearance interval.
                    cut=p-end_direction*(radius+r+clearance)
                    if float((cut-a)@end_direction)<=EPS:return None
                    return a,cut
            box=obstacle['box'];near=float(np.where(end_direction>0,box[:3],box[3:])@end_direction)
            if near>=float(p@end_direction)-EPS:return None
    return a,b


def _clear_plan(plan,obstacles,radius,clearance,bounds,source,target,normal,end_direction,*,is_lead=False,is_tail=False):
    pieces=[]
    for run in plan['runs']:
        a,b=np.asarray(run['start']),np.asarray(run['end'])
        pieces.append((a,b,radius,True))
    for elbow in plan['elbows']:
        points=[np.asarray(p) for p in elbow['centerline_points']]
        angle=math.radians(elbow['angle_degrees'])/(2*(len(points)-1))
        inflated=radius+elbow['centerline_radius_m']*(1-math.cos(angle))
        pieces.extend((a,b,inflated,False) for a,b in zip(points,points[1:]))
    origin=_point(source['position'],'source');finish=_point(target['position'],'target')
    source_contact_ids=set(source.get('contact_ids',()))
    for a,b,r,is_run in pieces:
        if np.any(np.minimum(a,b)-r<bounds[:3]-EPS) or np.any(np.maximum(a,b)+r>bounds[3:]+EPS):
            return 'physical_pipe_bounds'
        # Only the first piece on the real outlet axis can use the mating face.
        source_bounds = (outward_pipe_bounds(a,b,r,origin,normal)
                         if is_lead and is_run and np.max(abs(a-origin))<=EPS else None)
        for obstacle in obstacles:
            own_source=source_bounds is not None and obstacle['id'] in source_contact_ids
            gap=0. if own_source else clearance
            if own_source and source.get('tee'):
                # A branch leaving the centre of a real main has the same local
                # contact as the same straight entering that main in reverse.
                adjusted = _contact_cut(b,a,obstacle,r,gap,{},source,-normal,-normal,False,True)
            else:
                # A flat-ended real pipe may meet the flat original nozzle.
                if own_source and flat_pipe_clears(source_bounds,obstacle['box']):continue
                adjusted=_contact_cut(a,b,obstacle,r,gap,source,target,normal,end_direction,
                    False,is_tail and np.max(abs(b-finish))<=EPS)
            if adjusted is None:continue
            if _hits(*adjusted,obstacle,r,gap):return obstacle['id']
    return None


def _lane_coordinates(a,b,axis,obstacles,radius,clearance,R,bounds):
    values=[float(a[axis]),float(b[axis]),float((a[axis]+b[axis])/2)]
    pitch=max(2*radius+clearance,2*R+2*radius)
    values.extend((a[axis]-pitch,a[axis]+pitch,b[axis]-pitch,b[axis]+pitch))
    for obstacle in obstacles:
        gap=0. if obstacle['kind']=='pullout_reservation' else clearance
        values.extend((obstacle['box'][axis]-radius-gap-R,
                       obstacle['box'][axis+3]+radius+gap+R))
    kept=[]
    for value in sorted(values,key=lambda v:(min(abs(v-a[axis]),abs(v-b[axis])),v)):
        # Keep usable turning faces before applying the finite count. Tiny
        # differences among mesh-sector bounds must not crowd out whole lanes.
        minimum=2*R+2*radius
        if abs(value-a[axis])<minimum-EPS or abs(value-b[axis])<minimum-EPS:continue
        if bounds[axis]+radius-EPS<=value<=bounds[axis+3]-radius+EPS and not any(abs(value-v)<=EPS for v in kept):
            kept.append(float(value))
        if len(kept)>=12:break  # Finite lane resource, not an install threshold.
    return kept


def _source_exit_outside_pullout(exit_point,axis,obstacles,radius,R,rise_top):
    """Extend a real horizontal tail below extraction before turning upward.

    The sharp corner lies pipe-r + real-R beyond the unmodified box face,
    so the whole horizontal-to-vertical elbow can clear the actual volume.
    This changes only the pipe's own straight length, never an equipment port.
    The full route still checks every real solid and the original pullout box.
    """
    original=exit_point.copy();result=exit_point.copy();evidence=[]
    if abs(axis[2])>EPS:return result,evidence
    dimension=int(np.argmax(abs(axis)));other=1-dimension;sign=axis[dimension]
    # A farther extended riser may meet another real extraction box. Each
    # pass moves monotonically to a supplied face; at most one pass per box.
    for _ in range(len(obstacles)+1):
        moved=False
        for obstacle in obstacles:
            if obstacle['kind']!='pullout_reservation':continue
            box=obstacle['box']
            if min(result[2],rise_top)-radius>=box[5]-EPS or max(result[2],rise_top)+radius<=box[2]+EPS:continue
            if not box[other]-radius-EPS<result[other]<box[other+3]+radius+EPS:continue
            if not box[dimension]-radius-R-EPS<result[dimension]<box[dimension+3]+radius+R+EPS:continue
            face=box[dimension+3] if sign>0 else box[dimension]
            coordinate=float(face+sign*(radius+R))
            if (coordinate-result[dimension])*sign<=EPS:continue
            result[dimension]=coordinate;moved=True
            evidence.append({'obstacle':obstacle['id'],'actual_box':box.tolist(),
                'axis_index':dimension,'actual_face_m':float(face),'corner_coordinate_m':coordinate,
                'pipe_radius_m':radius,'real_elbow_R_m':R,'extra_clearance_m':0.,
                'extension_from_original_m':float((result-original)@axis)})
        if not moved:break
    return result,evidence


def _source_exit_candidates(original,first,axis,obstacles,radius,clearance,R,rise_top,bounds,base_evidence):
    """Try longer arms before new elbows; all exits move outward monotonically."""
    if abs(axis[2])>EPS:
        return [{'point':first,'evidence':list(base_evidence)}]
    dimension=int(np.argmax(abs(axis)));sign=axis[dimension]
    values=[float(first[dimension]),float(first[dimension]+sign*(2*radius+clearance)),
            float(first[dimension]+sign*(2*R+2*radius))]
    # Faces that actually block this riser take precedence over unrelated
    # mesh faces elsewhere in the room. This also permits a low side mouth
    # to continue under a real deck until an existing free edge is reached.
    other=1-dimension
    for obstacle in obstacles:
        box=obstacle['box'];gap=0. if obstacle['kind']=='pullout_reservation' else clearance
        if (box[other]-radius-gap<=first[other]<=box[other+3]+radius+gap and
                box[2]-radius-gap<=max(first[2],rise_top) and box[5]+radius+gap>=min(first[2],rise_top)):
            face=box[dimension+3] if sign>0 else box[dimension]
            values.append(float(face+sign*(radius+gap+R)))
    values.extend(_lane_coordinates(first,first,dimension,obstacles,radius,clearance,R,bounds))
    result=[]
    for coordinate in sorted(values,key=lambda v:((v-first[dimension])*sign,v)):
        if (coordinate-first[dimension])*sign<-EPS:continue
        candidate=first.copy();candidate[dimension]=coordinate
        candidate,extra=_source_exit_outside_pullout(candidate,axis,obstacles,radius,R,rise_top)
        if np.any(candidate-radius<bounds[:3]-EPS) or np.any(candidate+radius>bounds[3:]+EPS):continue
        if any(np.max(abs(candidate-row['point']))<=EPS for row in result):continue
        result.append({'point':candidate,'evidence':list(base_evidence)+extra+[
            {'kind':'outward_lane_change','axis':axis.tolist(),
             'extension_from_original_m':float((candidate-original)@axis),
             'actual_pipe_od_m':2*radius,'existing_gap_m':clearance,
             'basis':'one actual pipe pitch or actual obstacle face plus real R; no additional bend'}]})
        if len(result)>=12:break  # Finite layer candidates, not a new rule.
    return result


def _horizontal_guide(start,end,word,coordinate,levels):
    """Move complete runs, inserting a real-R-capable vertical lane change."""
    current=start.copy();points=[current]
    for index,axis in enumerate(word):
        z=levels[axis]
        next_point=current.copy();next_point[2]=z;points.append(next_point);current=next_point
        next_point=current.copy();next_point[axis]=coordinate if index==0 and len(word)==3 else end[axis]
        points.append(next_point);current=next_point
    points.append(end.copy())
    return points


def plan_lane_connection(source,target,components,pipe_radius,bounds,standing_z,rack_top_z,row_axis,*,
                         asset=None,valve_axis=None,target_direction=None,clearance=.05,
                         exception=None,max_candidates=96,outside_radius=None,rack_bottom_z=None,top_entry_extra_m=0.):
    """Return complete ``lead / slot / tail`` before the caller installs native.

    source/target={position,normal?,contact_ids?}; source is normally the actual
    reducer end, not a relocated equipment port. target_direction is the travel
    direction INTO target (PG13 uses -Z). asset is the unchanged native package
    metadata with actual install_length_m. OBB uses {center,axes,half_extents},
    axes as world matrix columns. All masks/alpha values are immaterial.
    This function never emits/loads models or assesses operation eligibility.
    """
    require_environment();started=time.monotonic()
    a=_point(source['position'],'source.position');b=_point(target['position'],'target.position')
    normal=_direction(source['normal'],'source.normal')
    axis=normal if valve_axis is None else _direction(valve_axis,'valve_axis')
    if target_direction is None:
        if 'normal' not in target:
            raise LaneUnavailable('LANE_TARGET_DIRECTION_REQUIRED','Provide the actual target approach direction')
        target_direction=-_direction(target['normal'],'target.normal')
    arrival=_direction(target_direction,'target_direction')
    radius=_positive(pipe_radius,'pipe_radius');clearance=_positive(clearance,'clearance',zero=True)
    outer=radius if outside_radius is None else _positive(outside_radius,'outside_radius')
    if outer<radius-EPS:raise LaneUnavailable('LANE_INPUT','The fitting envelope must be at least the pipe radius')
    boundary=np.asarray(bounds,float)
    if boundary.shape==(2,3):boundary=boundary.ravel()
    if boundary.shape!=(6,) or not np.isfinite(boundary).all() or np.any(boundary[3:]<=boundary[:3]):
        raise LaneUnavailable('LANE_INPUT','bounds must be finite physical limits on this floor')
    if row_axis not in (0,1) or isinstance(row_axis,bool):raise LaneUnavailable('LANE_INPUT','row_axis must be X=0 or Y=1')
    if exception not in _EXCEPTIONS:raise LaneUnavailable('LANE_INPUT','Use only the three explicitly authorized direction cases')
    if isinstance(max_candidates,bool) or not isinstance(max_candidates,int) or not 1<=max_candidates<=96:
        raise LaneUnavailable('LANE_INPUT','The combined route budget must be between 1 and 96')
    standing_z=float(standing_z);rack_top_z=float(rack_top_z)
    if not math.isfinite(standing_z) or not math.isfinite(rack_top_z):raise LaneUnavailable('LANE_INPUT','Floor elevation must be finite')
    obstacles,ignored=_obstacles(components);R=coarse_radius(radius)
    method_v1_3=source.get('method_v1_3') is True
    if method_v1_3 and rack_bottom_z is None:
        raise LaneUnavailable('ACTUAL_RACK_BOTTOM_REQUIRED','Pass the physical rack lower boundary instead of the 2.2 m operating clearance')
    band_bottom=(standing_z+band()[1] if rack_bottom_z is None else float(rack_bottom_z))
    if not math.isfinite(band_bottom) or band_bottom>=rack_top_z:
        raise LaneUnavailable('LANE_INPUT','Invalid rack vertical bounds')
    top_root=bool(method_v1_3 and asset is not None and source.get('vertical_top_root_required'))
    if top_root and (normal[2]<1-EPS or np.max(abs(axis-normal))>EPS):
        raise LaneUnavailable('TOP_ROOT_MUST_BE_VERTICAL','A top port permits a short +Z straight and vertical root valve before any elbow')
    length=0. if asset is None else _positive(asset['install_length_m'],'actual native install_length_m')
    if abs(float(axis@normal))>EPS and float(axis@normal)<0:
        raise LaneUnavailable('LANE_SOURCE_DIRECTION','The route cannot turn back 180 degrees immediately after leaving the original port')
    if np.max(abs(axis-normal))<=EPS:
        inlet=a+normal*length;lead=[a,inlet]
    else:
        # The unchanged elbow builder permits a reducer face to meet a real
        # elbow tangent face directly. Do not invent a full-OD straight here:
        # that extra rise can occupy the original extraction volume.
        corner=a+normal*R;inlet=corner+axis*(R+length);lead=[a,corner,inlet]
    outlet=inlet+axis*length;exit_point=outlet+axis*(length+R)
    slot=None if asset is None else {'in':inlet.tolist(),'out':outlet.tolist(),'axis':axis.tolist(),
        'install_length_m':length,'one_valve_space_each_side_m':length,
        'package':asset.get('package'),'placement_anchor':'actual_in_face',
        'scope':'Reserve only the native installation length. The caller attaches the real valve afterward.'}
    if asset is None:
        lead=[a];outlet=a.copy()
        straight_lead=_positive(source.get('minimum_straight_m',0.),'minimum_straight_m',zero=True)
        exit_point=a+normal*(R+straight_lead)
    if top_root:
        # Keep the original package length and both original valve spaces.
        # A top root must then rise straight into the rack before any turn.
        exit_point[2]=max(exit_point[2],band_bottom+outer+R)
        conflicts=[]
        for obstacle in obstacles:
            if obstacle['kind']!='pullout_reservation':continue
            root_hit=_hits(inlet,outlet,obstacle,radius,0.)
            rise_hit=_hits(outlet,exit_point,obstacle,radius,0.)
            if root_hit or rise_hit:
                conflicts.append({'reservation':obstacle['id'],'original_box':obstacle['box'].tolist(),
                    'root_flow_axis_in_reserved_volume':root_hit,'post_root_straight_in_reserved_volume':rise_hit})
        if conflicts:
            raise LaneUnavailable('TOP_ROOT_EXTRACTION_CONFLICT',
                'The vertical root valve or riser intersects the original pull-out reservation',
                original_endpoint=source.get('original_endpoint'),original_source=source.get('original_position'),
                reducer_end=a.tolist(),actual_native_L_m=length,one_valve_space_m=length,
                planned_root_in=inlet.tolist(),planned_root_out=outlet.tolist(),
                planned_first_rack_corner=exit_point.tolist(),conflicts=conflicts,
                physical_unsatisfiable_claim=False)
    target_radius=_positive(target.get('radius',radius),'target.radius')
    # Keep a finite straight tee arm beyond the complete bend. A bend ending
    # exactly at the clearance boundary is too tight for its sampled outer arc.
    target_arm=R+target_radius+2*radius+clearance
    if arrival[2]<-1+EPS:target_arm+=_positive(top_entry_extra_m,'top_entry_extra_m',zero=True)
    approach=b-arrival*target_arm
    lower=max(band_bottom+outer,
        b[2]+max(2*R+clearance,target_arm) if arrival[2]<0 else band_bottom+outer)
    if axis[2]>0:lower=max(lower,exit_point[2])
    original_exit=exit_point.copy()
    exit_point,exit_extension=_source_exit_outside_pullout(exit_point,axis,obstacles,radius,R,lower)
    separation=max(2*R+2*radius,2*outer+clearance)
    upper=lower+separation
    if abs(arrival[2])<EPS:
        # The last short tee arm stays in its target's plane. The preceding
        # horizontal direction uses that plane too, without a tiny up/down jog.
        if int(np.argmax(abs(arrival)))==row_axis:
            upper=b[2];lower=upper-separation
        else:lower=b[2];upper=lower+separation
    levels={row_axis:lower,1-row_axis:upper}
    lanes={'row_axis':row_axis,'lower_center_z_m':lower,'upper_center_z_m':upper,
        'actual_outer_radius_m':outer,'minimum_outer_bottom_m':band_bottom,
        'center_R_m':R,'lane_separation_m':separation,'rack_top_z_m':rack_top_z,
        'basis':'Use physical rack bounds. Prefer same-level routes and preserve radius, OD and clearance on elevation changes.'}
    exit_candidates=_source_exit_candidates(original_exit,exit_point,axis,obstacles,radius,clearance,R,
                                             lower,boundary,exit_extension)
    # Select distinct physical lanes, not nearly coincident mesh sectors.
    distinct=[]
    for candidate in exit_candidates:
        if not distinct or min(np.linalg.norm(candidate['point']-old['point']) for old in distinct)>=2*radius+clearance-EPS:
            distinct.append(candidate)
    exit_candidates=distinct
    if source.get('edge_riser_xy') is not None:
        edge_exit=exit_point.copy();edge_exit[:2]=source['edge_riser_xy']
        exit_candidates=[{'point':edge_exit,'evidence':[{'kind':'actual_perimeter_floor_crossing',
            'position':edge_exit.tolist(),'source':'caller retains real plate cut and all beams/columns'}]}]
    failure=Counter();examples=[];tested=0;seen=set()
    lead_plan=None
    if len(_clean(lead))>=2:
        try:lead_plan=plan_polyline(_clean(lead),radius)
        except ElbowUnavailable as exc:raise LaneUnavailable(exc.code,str(exc)) from exc
        hit=_clear_plan(lead_plan,obstacles,radius,clearance,boundary,source,target,normal,arrival,is_lead=True)
        if hit:raise LaneUnavailable('LANE_SOURCE_LEAD_BLOCKED','A physical solid blocks the normal outlet or required pre-valve straight',obstacle=hit,lead=[p.tolist() for p in _clean(lead)])
    # Already planned actual lead runs obstruct every later tail run too.
    lead_obstacles=[]
    if lead_plan:
        for i,run in enumerate(lead_plan['runs']):
            one,two=np.asarray(run['start']),np.asarray(run['end']);e=np.full(3,radius)
            lead_obstacles.append({'id':'current_lead_'+str(i),'kind':'pipe',
                'box':np.r_[np.minimum(one,two)-e,np.maximum(one,two)+e],
                'round':(one,two,radius),'obb':None})
    def accept(points,family,exit_candidate,direct_target=False):
        nonlocal tested
        selected_exit=exit_candidate['point']
        points=_clean([outlet,b] if direct_target else [outlet,selected_exit,*points,approach,b])
        key=tuple(tuple(p) for p in points)
        if key in seen:return None
        seen.add(key);tested+=1
        try:plan=plan_polyline(points,radius)
        except ElbowUnavailable as exc:failure[exc.code]+=1;return None
        if top_root and plan['elbows']:
            first=plan['elbows'][0]
            lowest=min(p[2] for p in first['centerline_points'])-radius
            highest=max(p[2] for p in first['centerline_points'])+radius
            if lowest<band_bottom-EPS or highest>rack_top_z+EPS:
                failure['FIRST_TOP_TURN_OUTSIDE_ACTUAL_RACK']+=1;return None
        if axis@_direction(points[1]-points[0],'first tail direction')<1-EPS:
            failure['NATIVE_OUT_TURNBACK']+=1;return None
        if arrival@_direction(points[-1]-points[-2],'last target direction')<1-EPS:
            failure['TARGET_INCIDENCE']+=1;return None
        hit=_clear_plan(plan,obstacles+lead_obstacles,radius,clearance,boundary,source,target,normal,arrival,
                        is_lead=asset is None,is_tail=True)
        if hit:
            failure['ACTUAL_OBSTACLE']+=1
            if len(examples)<8:examples.append({'family':family,'obstacle':hit})
            return None
        bends=len(plan['elbows'])+(len(lead_plan['elbows']) if lead_plan else 0)
        return {'status':'guide_ready_not_actual_model_pass','method':'level_two_direction_lanes_v1_2',
            'lead':[p.tolist() for p in _clean(lead)],'slot':slot,'tail':[p.tolist() for p in points],
            'control_points':{'lead':[p.tolist() for p in _clean(lead)],'native_gap':None if slot is None else [slot['in'],slot['out']],
                              'tail':[p.tolist() for p in points]},
            'source_exit':selected_exit.tolist(),'source_exit_original':original_exit.tolist(),
            'source_exit_extension':exit_candidate['evidence'],'target_approach':approach.tolist(),'lanes':lanes,
            'diagnostics':{'selected_family':family,'tested_candidates':tested,'candidate_budget':max_candidates,
                'rejections':dict(failure),'examples':examples,'ignored_operation_reservations':ignored,
                'actual_bends':bends,'source_exit_candidates':len(exit_candidates),
                'candidate_order':'complete_guides_bends_then_length',
                'source_contact_policy':'first outward straight, declared port components only; actual solids checked',
                'elapsed_s':time.monotonic()-started,'exception':exception,
                'operation_eligibility_gate':False,'global_optimality_claim':False}}
    # A real vertical supply can join the underside of an aligned header
    # without an elbow. Forced approach/exit waypoints must not create a
    # reversal on this already straight connection.
    delta=b-outlet
    if (float(delta@axis)>EPS and np.linalg.norm(np.cross(delta,axis))<=EPS
            and float(arrival@axis)>1-EPS):
        result=accept([], 'direct_collinear_to_header',
                      {'point':outlet,'evidence':[]},direct_target=True)
        if result:return result

    def options(exit_candidate):
        point=exit_candidate['point']
        if method_v1_3:
            # Prefer the fewest turns and then the shortest complete XYZ path.
            # One usable elevation is enough; two layers are only a fallback.
            direct=[]
            for order in itertools.permutations((row_axis,1-row_axis,2)):
                changes=[dimension for dimension in order if abs(point[dimension]-approach[dimension])>EPS]
                if (exception is None and point[2]-outer<band_bottom-EPS
                        and changes and changes[0]!=2):continue
                current=point.copy();guide=[current.copy()]
                for dimension in order:
                    current=current.copy();current[dimension]=approach[dimension];guide.append(current)
                direct.append((guide,'direct_axis_order_v1_3'))
            planes=[approach[2],max(point[2],band_bottom+outer+R),
                    max(point[2],approach[2])+2*R+2*radius]
            for z in sorted(set(planes)):
                if z-outer<band_bottom-EPS or z+outer>min(rack_top_z,boundary[5])+EPS:continue
                for order in ((row_axis,1-row_axis),(1-row_axis,row_axis)):
                    direct.append((_horizontal_guide(point,approach,order,None,{0:z,1:z}),
                                   'single_plane_v1_3'))
            ranked=[]
            for guide,family in direct:
                path=_clean([outlet,point,*guide,approach,b])
                ranked.append((max(0,len(path)-2),sum(math.dist(x,y) for x,y in zip(path,path[1:])),guide,family))
            for _,_,guide,family in sorted(ranked,key=lambda item:item[:2]):
                yield guide,family,exit_candidate
        # A real crossing can obstruct a full-height source riser while a
        # lower rise remains clear. Move laterally at the header elevation,
        # finish the rise on that clear plane, and enter the header from above
        # directly. Do not descend again merely to restore side incidence.
        if arrival[2]<-1+EPS and method_v1_3:
            high=max(approach[2],point[2]);low=b[2]
            if low-outer>=band_bottom-EPS and low>=point[2]-EPS and high+outer<=min(rack_top_z,boundary[5])+EPS:
                reach=2*R+2*radius+clearance
                for dimension in (row_axis,1-row_axis):
                    other=1-dimension
                    coordinates=[approach[dimension],min(point[dimension],approach[dimension])-reach,
                                 max(point[dimension],approach[dimension])+reach]
                    for coordinate in coordinates:
                        q=point.copy();q[2]=low
                        u=q.copy();u[dimension]=coordinate
                        v=u.copy();v[2]=high
                        w=v.copy();w[other]=approach[other]
                        t=w.copy();t[dimension]=approach[dimension]
                        yield [q,u,v,w,t],'obstacle_split_rise_top_entry',exit_candidate
        # A top-down offgas approach may have only a small lateral offset.
        # Detour on one plane above the header instead of forcing two stacked
        # lanes outside the rack. All original physical checks still apply.
        if arrival[2] < -1+EPS:
            z=max(point[2],approach[2])
            if z+outer <= min(rack_top_z,boundary[5])+EPS:
                for dimension in (row_axis,1-row_axis):
                    other=1-dimension
                    reach=2*R+2*radius+clearance
                    for sign in (-1.,1.):
                        coordinate=(min(point[dimension],approach[dimension])-reach if sign<0
                                    else max(point[dimension],approach[dimension])+reach)
                        q=point.copy();q[2]=z
                        u=q.copy();u[dimension]=coordinate
                        v=u.copy();v[other]=approach[other]
                        w=v.copy();w[dimension]=approach[dimension]
                        yield [q,u,v,w], 'offgas_single_plane_top_entry', exit_candidate
                        # If the other horizontal offset is shorter than two
                        # bend radii, use a rectangular return with real legs.
                        for cross_sign in (-1.,1.):
                            cross_coordinate=(min(point[other],approach[other])-reach if cross_sign<0
                                              else max(point[other],approach[other])+reach)
                            v2=u.copy();v2[other]=cross_coordinate
                            w2=v2.copy();w2[dimension]=approach[dimension]
                            t=w2.copy();t[other]=approach[other]
                            yield [q,u,v2,w2,t], 'offgas_rectangular_top_entry', exit_candidate
        # Local exceptions retain actual mouth/low pump height. No diagonal shortcut.
        if exception is not None:
            for order in itertools.permutations(range(3)):
                p=point.copy();guide=[p]
                for dimension in order:p=p.copy();p[dimension]=approach[dimension];guide.append(p)
                yield guide,'local_exception_axis_order',exit_candidate
        if (lower-outer>=band_bottom-EPS
                and upper+outer<=min(rack_top_z,boundary[5])+EPS):
            for order in ((row_axis,1-row_axis),(1-row_axis,row_axis)):
                yield (_horizontal_guide(point,approach,order,None,levels),
                    'lane_default_order' if order[0]==row_axis else 'swap_direction_order',exit_candidate)
            dimensions=(row_axis,1-row_axis)
            coordinates=[_lane_coordinates(point,approach,dimension,obstacles,radius,clearance,R,boundary)
                         for dimension in dimensions]
            # Interleave both ways around an obstacle. A long list of X-Y-X
            # planes must not consume the budget before any Y-X-Y plane.
            for pair in itertools.zip_longest(*coordinates):
                for dimension,coordinate in zip(dimensions,pair):
                    if coordinate is None:continue
                    other=1-dimension
                    minimum=2*R+2*radius
                    if (abs(coordinate-point[dimension])<minimum-EPS
                            or abs(coordinate-approach[dimension])<minimum-EPS):continue
                    yield (_horizontal_guide(point,approach,(dimension,other,dimension),coordinate,levels),
                           'move_local_lane_plane',exit_candidate)
    if upper+outer>min(rack_top_z,boundary[5])+EPS:failure['TWO_LANE_BAND_CAPACITY']+=1
    # Compare complete geometry across all source exits before spending the
    # unchanged finite check budget. A late simple top entry must not lose to
    # an early seven-bend side-entry template merely by iterator order.
    ranked=[]
    for row in exit_candidates:
        for option in options(row):
            guide,family,exit_candidate=option
            path=_clean([outlet,exit_candidate['point'],*guide,approach,b])
            ranked.append((max(0,len(path)-2),sum(math.dist(x,y) for x,y in zip(path,path[1:])),len(ranked),option))
    for _,_,_,option in sorted(ranked,key=lambda row:row[:3]):
        result=accept(*option)
        if result:return result
        if tested>=max_candidates:break
    raise LaneUnavailable('LANE_CONNECTION_UNAVAILABLE','Limited lane, order and plane variants did not connect the route',
        tested_candidates=tested,candidate_budget=max_candidates,rejections=dict(failure),examples=examples,
        lead=[p.tolist() for p in _clean(lead)],slot=slot,lanes=lanes,
        source_exit_original=original_exit.tolist(),source_exit=exit_point.tolist(),
        source_exit_extension=exit_extension,
        source_exit_candidates=[{'point':row['point'].tolist(),'evidence':row['evidence']} for row in exit_candidates],
        ignored_operation_reservations=ignored,elapsed_s=time.monotonic()-started,
        failed_step='Routing method section 3: change lane, order or local plane',physical_unsatisfiable_claim=False)
