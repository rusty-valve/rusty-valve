"""Provide equipment dimensions and build basic equipment shapes."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .layout import add_box, add_pipe
from .recipe import CandidateRefused
from .storage_vessel import HORIZONTAL_VESSEL_TIER


CATALOG = json.loads(Path(__file__).with_suffix('.json').read_text())


def resolve_spec(unit):
    """Resolve one approved item; never scale a different tier to fit a room."""
    kind=unit['kind']
    family='heat_exchanger' if kind in ('exchanger','condenser') else kind
    if family=='tower':family='fixed_bed'
    form=unit.get('form',unit.get('orientation','vertical'))
    tier=unit.get('size_tier','medium' if family=='vessel' and form=='vertical' else 'small')
    if family == 'vessel' and form == 'horizontal':
        tier = HORIZONTAL_VESSEL_TIER
    key=f'{family}.{form}.{tier}'
    if key not in CATALOG['items']:
        raise CandidateRefused(f'The equipment catalogue lacks this form or size tier: {key}. Other sizes are not scaled to substitute')
    spec=deepcopy(CATALOG['items'][key])
    spec.update(catalog_key=key,kind=kind,source=deepcopy(CATALOG['sources'][spec['source_id']]),
                nominal_volume_l=spec.get('nominal_volume_l'),geometry_approximations=deepcopy(CATALOG['construction_estimates']))
    if family=='vessel' and form=='vertical':
        raw=spec['manufacturer_dimensions_mm']
        spec['diameter_m']=raw['d2']/1000
        if spec['series']=='Taiji F':
            upper_datum=raw['h4']/1000
            bottom=upper_datum-raw['h1']/1000
            neck=spec['nozzle_length_m_estimate']
            top=upper_datum-neck
            lower=upper=spec['diameter_m']*spec['head_depth_diameter_fraction_estimate']
            straight=top-bottom-lower-upper
            drive_mount=upper_datum+raw['h2']/1000
            drive=drive_mount+spec['reference_drive_and_overall_mm']['h5']/1000
            offset=0.
            inner_radius=raw['d1']/2000
            geometric=1000*(math.pi*inner_radius**2*straight+4*math.pi/3*inner_radius**2*lower)
            spec.update(reference_planes_m={'upper_datum':upper_datum,'bottom_datum':bottom,
                'jacket_upper_axis':upper_datum-raw['h3']/1000,'drive_mount':drive_mount},
                geometric_estimated_volume_l=geometric,
                geometric_volume_scope='Estimated from catalogue d1 and coaxial sections. This is neither a measured volume nor a nominal or catalogue total.',
                capacity_fields={'nominal_l':spec['nominal_volume_l'],
                    'catalog_total_l':spec['catalog_total_volume_l'],
                    'catalog_calculated_l':spec['catalog_calculated_volume_l'],
                    'geometric_estimated_l':geometric},
                upper_envelope_source='U equals h4. The approximate shell top is U less the 0.12 m neck. U is not the head tangent.',
                bottom_envelope_source='B equals h4 minus h1. The body bottom is estimated from B. The bottom port projects another 0.12 m.')
        elif spec['series']=='Pfaudler AE':
            offset=spec['support_plane_offset_m_estimate']
            lower=spec['lower_head_depth_m_estimate']
            # The table's h3 is the complete body envelope. h3-h1 includes
            # the head and its 10 mm joint allowance, whereas raw h2 alone
            # does not. Keep every raw field rather than losing that 10 mm.
            upper=(raw['h3']-raw['h1'])/1000
            bottom=offset-raw['h1']/1000
            top=offset+upper
            straight=raw['h1']/1000-lower
            drive=top+raw['h5']/1000+raw['h13']/1000+spec['drive_motor_extra_m_estimate']
        else:
            offset=spec['support_plane_offset_m_estimate']
            lower=spec['lower_head_depth_m_estimate']
            upper=spec['upper_head_depth_m_estimate']
            top=offset+raw['h6']/1000
            bottom=top-raw['h3']/1000
            straight=raw['h3']/1000-lower-upper
            drive=top+raw['e_approx']/1000
        spec.update(length_m=None,height_m=top-bottom,main_cylinder_length_m=straight,
                    lower_head_depth_m=lower,upper_head_depth_m=upper,
                    support_to_bottom_m=bottom,support_to_top_m=top,
                    drive_above_support_m=drive,support_plane_offset_m=offset,
                    installation_estimated=True,installed_height_m=drive-bottom)
        if spec['series']=='Pfaudler AE':
            spec['upper_envelope_source']='Catalogue h3 minus h1 includes the head and mating flange elevation. The original h2 is retained separately. The ellipsoid is approximate.'
    elif family=='fixed_bed':
        depth=spec['head_depth_m']
        spec.update(length_m=None,main_cylinder_length_m=spec['height_m']-spec['skirt_height_m']-2*depth,
                    support_to_bottom_m=0.,support_to_top_m=spec['height_m'],
                    drive_above_support_m=None,installed_height_m=spec['height_m'])
    else:
        depth=spec['head_depth_m'];diameter=spec['diameter_m'];length=spec['length_m']
        height=diameter if form=='horizontal' else length+2*depth
        base=spec['leg_clearance_m']
        spec.update(height_m=height,main_cylinder_length_m=length,total_length_m=length+2*depth,
                    support_to_bottom_m=base,support_to_top_m=base+height,
                    drive_above_support_m=None,installed_height_m=base+height)
        if family=='vessel' and form=='horizontal':
            from .storage_vessel_model import installed_height
            spec['installed_height_m']=installed_height(spec['support_to_top_m'])
        if family=='heat_exchanger' and form=='horizontal':
            chamber=spec['channel_box_length_m_estimate']
            plate=spec['tube_sheet_thickness_m_estimate']
            spec.update(total_length_m=length+2*chamber+2*depth,
                tube_shell_length_m=length,channel_box_length_each_m=chamber,
                tube_sheet_thickness_m=plate,
                max_body_diameter_m=diameter+2*spec['tube_sheet_radial_extra_m_estimate'],
                installed_height_m=base+height+spec['tube_sheet_radial_extra_m_estimate'],
                channel_head_ranges_local_m={
                    'left_channel_box':[-length/2-chamber,-length/2-plate],
                    'right_channel_box':[length/2+plate,length/2+chamber]},
                construction_scope='Separate shell, two tube sheets, two channels and heads. Dimensions are estimates.')
    if family=='heat_exchanger' and form=='vertical':
        chamber=spec['channel_box_length_m_estimate']
        height=length+2*chamber+2*depth
        spec.update(height_m=height,total_length_m=height,support_to_top_m=base+height,
                    installed_height_m=base+height,
                    construction_scope='Separate upper and lower channels, two tube sheets and central shell. Dimensions are estimates.')
    spec['dimensions_m']={key:spec[key] for key in ('diameter_m','length_m','height_m',
        'main_cylinder_length_m','support_to_bottom_m','support_to_top_m','drive_above_support_m')}
    return spec


def _ellipsoid(scene,name,center,size,owner,color=(.69,.74,.75)):
    scene['components'].append({'id':name,'kind':'head','shape':'ellipsoid','pos':list(center),'size':list(size),
        'box':[center[i]-size[i] for i in range(3)]+[center[i]+size[i] for i in range(3)],
        'owner':owner,'color':color,'estimated':True,'scope':'Approximate ellipsoidal head envelope'})


def _short_nozzle(scene,unit,name,start,end):
    port=unit['ports'][name]
    from .pipe_spec import load
    dn=port.get('equipment_neck_dn',port.get('dn',50))
    diameter=load()['pipe_outer_diameter_mm'].get(str(dn))
    if diameter is None:raise CandidateRefused(f'equipment neck DN{dn}Missing actual outside diameter. DN is nominal.')
    diameter/=1000
    spec=load();medium=port.get('medium',port.get('medium_identity'))
    role=port.get('process_role',name)
    if medium not in spec['colors']:
        if role=='nitrogen':medium='nitrogen'
        elif role in ('offgas','noncondensable_out'):medium='offgas'
        elif role in ('safety','emergency_discharge'):medium='emergency_discharge'
        elif role in ('vapor','vapor_in'):medium='vapor'
        elif unit['kind']=='condenser' and name in ('utility_in','cold_in'):medium='cold_source'
        elif unit['kind']=='condenser' and name in ('utility_out','cold_out'):medium='cold_return'
        elif unit.get('service')=='steam_heater' and name in ('utility_in','steam_in','hot_in'):medium='steam'
        elif unit.get('service')=='steam_heater' and name in ('utility_out','steam_condensate_out','hot_out'):medium='steam_condensate'
        else:medium='unknown_material'
    component=add_pipe(scene,unit['id']+'_'+name+'_equipment_nozzle',start,end,diameter/2,
                       'equipment_nozzle',spec['colors'][medium],unit['id'])
    if component:
        component.update(estimated=True,medium=medium,color_source='pipe_spec.json',
            scope='Equipment stub projects 0.12 m from the approximate equipment surface. It represents envelope geometry.')
    normal=(np.asarray(end)-start)/math.dist(start,end)
    unit['ports'][name].update(pos=list(end),normal=normal.tolist(),surface_pos=list(start),
        stem=[0.,1.,0.] if abs(normal[2])>.5 else [0.,0.,1.],
        nozzle_component_id=component['id'],component_ids=[component['id']],
        actual_neck_outer_diameter_m=diameter)


def _side_lugs(scene,unit,z,radius):
    for side in (-1,1):
        center=side*(radius+.10)
        lug=add_box(scene,unit['id']+'_lug'+str(side),
            [center-.16,-.14,z-.05,center+.16,.14,z+.05],'lug',(.38,.46,.48),unit['id'])
        lug.update(estimated=True,support_reference_z_m=z)
        # A local shoe bears on the solid board just outside the normal hole.
        support=add_box(scene,unit['id']+'_lug_shoe'+str(side),
            [center-.12,-.13,z-.10,center+.12,.13,z-.05],'support',(.25,.37,.45),unit['id'])
        support.update(estimated=True,support_reference_z_m=z)


def _taiji_lugs(scene,unit,spec,z,radius):
    """Four lug bearing faces; local plate/shoe outlines remain estimates.

    d3 controls reference hole centres on the four 45-degree bearings, not
    the plate envelope. A short cantilever shoe bears outside the existing
    square platform penetration, without changing the shell/opening rules.
    """
    hole_radius=spec['manufacturer_dimensions_mm']['d3']/2000
    for index,bearing in enumerate(spec['lug_bearings_deg'],1):
        angle=math.radians(bearing)
        hx,hy=hole_radius*math.sin(angle),hole_radius*math.cos(angle)
        sign_x=1 if hx>0 else -1
        inner=sign_x*(radius/math.sqrt(2)-.10)
        outer=sign_x*max(abs(hx)+.18,radius+.24)
        x0,x1=sorted((inner,outer))
        lug=add_box(scene,unit['id']+'_lug'+str(index),
            [x0,hy-.16,z,x1,hy+.16,z+.10],'lug',(.38,.46,.48),unit['id'])
        lug.update(estimated=True,support_reference_z_m=z,bearing_deg=bearing,
            manufacturer_hole_center_local_m=[hx,hy,0.],
            scope='Four lug bearing faces sit at datum Z. Hole centers use d3. Local bearing plate envelopes are estimated.')
        xa,xb=sorted((sign_x*(radius+.12),outer))
        shoe=add_box(scene,unit['id']+'_lug_shoe'+str(index),
            [xa,hy-.12,z-.10,xb,hy+.12,z],'support',(.25,.37,.45),unit['id'])
        shoe.update(estimated=True,support_reference_z_m=z,lug_id=lug['id'],
            scope='Support shoe reaches the solid deck outside the existing rectangular opening. Detail geometry is approximate.')


def reactor_nozzle_mount(name):
    # Port schedule for the coarse upright reactor: ring on the upper head,
    # the two independent discharge mouths on the lower head.
    return 'lower_head' if name in ('bottom', 'drain') else 'upper_head'


def _vertical_reactor(scene,unit,spec,z):
    uid=unit['id'];radius=spec['diameter_m']/2
    bottom=z+spec['support_to_bottom_m'];top=z+spec['support_to_top_m']
    lower,upper=spec['lower_head_depth_m'],spec['upper_head_depth_m']
    if spec['series']=='Pfaudler AE':
        cylinder_top=z+spec['support_plane_offset_m']
    else:cylinder_top=top-upper
    cylinder_bottom=bottom+lower
    add_pipe(scene,uid,[0.,0.,cylinder_bottom],[0.,0.,cylinder_top],radius,'vessel',(.69,.74,.75),uid)
    _ellipsoid(scene,uid+'_lower_head',[0.,0.,cylinder_bottom],[radius,radius,lower],uid)
    _ellipsoid(scene,uid+'_upper_head',[0.,0.,cylinder_top],[radius,radius,upper],uid)
    if spec['series']=='Taiji F':_taiji_lugs(scene,unit,spec,z,radius)
    else:_side_lugs(scene,unit,z,radius)
    drive_top=z+spec['drive_above_support_m']
    # Same installed envelope as the original catalogue approximation, without the detailed drive assembly.
    # Width is the preceding coarse motor's estimated diameter (2 * .16).
    half_width=spec.get('drive_width_m_estimate',.32)/2
    drive=add_box(scene,uid+'_drive_reservation',[-half_width,-half_width,top,half_width,half_width,drive_top],
                  'drive_reservation',(.25,.55,.65),uid)
    drive.update(alpha=.22,estimated=True,scope='reserved drive envelope',
                 width_source='Original coarse motor radius 0.16 m, outside diameter 0.32 m',
                 height_source='Catalogue h4 plus h2 plus h5 gives the reference drive envelope' if spec['series']=='Taiji F' else 'catalogue installation height less shell height')
    for name, port in unit['ports'].items():
        port['nozzle_mount'] = reactor_nozzle_mount(name)
    ring_ports=[name for name, port in unit['ports'].items() if port['nozzle_mount'] == 'upper_head']
    for index,name in enumerate(ring_ports):
        angle=math.tau*index/max(1,len(ring_ports))
        x,y=radius*.55*math.cos(angle),radius*.55*math.sin(angle)
        surface=cylinder_top+upper*math.sqrt(max(0.,1-(x*x+y*y)/(radius*radius)))
        _short_nozzle(scene,unit,name,[x,y,surface],[x,y,top+.12])
    for name in ('bottom','drain'):
        if name in unit['ports']:
            dx=0. if name=='bottom' else radius*.30
            if spec['series']=='Taiji F' and name=='bottom':
                port=unit['ports'][name]
                port['line_dn']=port['dn']
                port['equipment_neck_dn']=spec['bottom_discharge_reference']['dn']
                port['equipment_neck_source']='Taiji reference bottom outlet. The neck and external pipe are recorded separately for later connection.'
            _short_nozzle(scene,unit,name,[dx,0.,bottom+.02],[dx,0.,bottom-.12])
    unit.update(center=[0.,0.,(bottom+top)/2],box=[-radius,-radius,bottom,radius,radius,top],
                bottom=bottom,top=top,radius=radius,support_reference_z_m=z)


def _saddles(scene,unit,z,radius,length,center_z):
    for side in (-1,1):
        x=side*length*.28
        component=add_box(scene,unit['id']+'_saddle'+str(side),
            [x-.13,-radius*.75,z,x+.13,radius*.75,center_z],
            'saddle',(.31,.42,.49),unit['id'])
        component.update(estimated=True,scope='Approximate twin saddles on the physical floor or platform')


def horizontal_vessel_nozzles(unit,spec,z):
    """Separate independent vessel ports, without changing its body or support.

    The former two alternating top positions and shared bottom position aliased
    distinct uses. All original names and DN remain. Rows use the same central
    40% top band and 120mm coarse neck; bottom rows stay between the saddles.
    This is an estimated nozzle layout, not a manufacturer nozzle schedule.
    """
    from .pipe_spec import load
    outside=load()['pipe_outer_diameter_mm']
    radius=spec['diameter_m']/2;length=spec['length_m'];head=spec['head_depth_m']
    center=z+spec['leg_clearance_m']+radius
    groups={'left':[],'right':[],'bottom':[],'top':[]}
    for name in unit['ports']:
        role=('left' if name in ('in','vapor') else 'right' if name in ('out','condensate')
              else 'bottom' if name in ('bottom','drain') else 'top')
        groups[role].append(name)
    result={}
    for role,names in groups.items():
        count=len(names)
        if not count:continue
        span=length*(.20 if role=='top' else .12) if role in ('top','bottom') else radius*.3
        positions=[0.] if count==1 else np.linspace(-span,span,count).tolist()
        if role=='top' and 'feed' in names and count>1:
            # Adding the approved top feed must not shift existing gas/safety
            # mouths. Fill a free gap in their previous central top band.
            names=[name for name in names if name!='feed']+['feed']
            existing=[0.] if count==2 else np.linspace(-span,span,count-1).tolist()
            candidates=[-span,span,*[(x+y)/2 for x,y in zip(existing,existing[1:])]]
            feed=max(candidates,key=lambda x:min(abs(x-y) for y in existing))
            positions=existing+[feed]
        for i,name in enumerate(names):
            t=positions[i]
            if role in ('left','right'):
                sign=-1 if role=='left' else 1
                x=sign*(length/2+head*math.sqrt(max(0.,1-(t/radius)**2)))
                start=[x,t,center];end=[x+sign*.12,t,center]
            else:
                sign=-1 if role=='bottom' else 1
                start=[t,0.,center+sign*radius];end=[t,0.,start[2]+sign*.12]
            result[name]=(start,end)
        for first,second in zip(names,names[1:]):
            r1=outside[str(unit['ports'][first].get('equipment_neck_dn',unit['ports'][first]['dn']))]/2000
            r2=outside[str(unit['ports'][second].get('equipment_neck_dn',unit['ports'][second]['dn']))]/2000
            if math.dist(result[first][0],result[second][0])<=r1+r2+1.e-8:
                raise CandidateRefused("The standard port arrangement cannot fit the horizontal vessel's nozzle count and sizes. Keep its nozzles and body position.")
    return result


def horizontal_exchanger_nozzles(unit,spec,z):
    """Figure 20 heater / figure 17 condenser radial faces in local XYZ.

    Local X is the equipment axis; tube boxes are outside the two tube sheets.
    No tube-side connection pierces an axial end cap.  Aliases preserve original
    process-port names rather than creating duplicate physical interfaces.
    """
    length=spec['length_m'];radius=spec['diameter_m']/2
    center=z+spec['leg_clearance_m']+radius
    box_x=length/2+(spec['channel_box_length_m_estimate']+spec['tube_sheet_thickness_m_estimate'])/2
    shell_x=.30*length
    if unit['kind']=='condenser':
        table={
            'vapor':(-shell_x,-1,'shell','left_shell_bottom','vapor_in'),
            'condensate':(shell_x,-1,'shell','right_shell_bottom','condensate_out'),
            'utility_in':(box_x,-1,'tube','right_channel_box','cold_in'),
            'utility_out':(-box_x,1,'tube','left_channel_box','cold_out'),
            'offgas':(0.,1,'shell','shell_top','noncondensable_out')}
        aliases={'vapor_in':'vapor','condensate_out':'condensate','cold_in':'utility_in',
                 'cold_out':'utility_out','noncondensable_out':'offgas'}
    else:
        table={
            'in':(-box_x,-1,'tube','left_channel_box','material_in'),
            'out':(box_x,1,'tube','right_channel_box','material_out'),
            'utility_in':(shell_x,1,'shell','right_shell_top','steam_in'),
            'utility_out':(-shell_x,-1,'shell','left_shell_bottom','steam_condensate_out')}
        aliases={'material_in':'in','material_out':'out','process_in':'in','process_out':'out',
                 'steam_in':'utility_in','hot_in':'utility_in','steam_condensate_out':'utility_out',
                 'hot_out':'utility_out','condensate_out':'utility_out'}
    result={};used=set()
    for name,port in unit['ports'].items():
        key=aliases.get(name,name)
        if key not in table or key in used:
            raise CandidateRefused(f'{unit["kind"]}separate horizontal vessel nozzle{name}No uniquely assigned use. Nozzle positions are neither inferred nor duplicated.')
        used.add(key);x,side,circuit,zone,role=table[key]
        start=[x,0.,center+side*radius];end=[x,0.,start[2]+side*.12]
        result[name]=(start,end)
        port.update(circuit_side=circuit,nozzle_zone=zone,process_role=role,
            nozzle_mount='radial',nozzle_position_source='Connections from figures 17 and 20. Envelope estimates use the catalogue dimensions.',
            tube_side_flow='left_lower_to_right_upper' if unit['kind']!='condenser' else 'right_lower_to_left_upper')
    return result


def _horizontal_exchanger(scene,unit,spec,z):
    """Adjacent physical shell, plates and channel boxes; no label-only split."""
    uid=unit['id'];length=spec['length_m'];radius=spec['diameter_m']/2
    chamber=spec['channel_box_length_m_estimate'];plate=spec['tube_sheet_thickness_m_estimate']
    head=spec['head_depth_m'];center=z+spec['leg_clearance_m']+radius
    add_pipe(scene,uid,[-length/2,0.,center],[length/2,0.,center],radius,
             unit['kind'],(.69,.74,.75),uid)
    chambers=[];plates=[]
    for side,label in ((-1,'left'),(1,'right')):
        a=side*length/2;b=side*(length/2+plate);c=side*(length/2+chamber)
        sheet=add_pipe(scene,uid+'_'+label+'_tube_sheet',[a,0.,center],[b,0.,center],
            radius+spec['tube_sheet_radial_extra_m_estimate'],'tube_sheet',(.38,.46,.48),uid)
        box=add_pipe(scene,uid+'_'+label+'_channel_box',[b,0.,center],[c,0.,center],
            radius,'channel_box',(.63,.72,.73),uid)
        sheet.update(estimated=True,scope='Approximate solid tube sheet. Internal bores and heat-transfer tubes are omitted.')
        box.update(estimated=True,circuit_side='tube',scope='Separate solid channel beside the shell across the tube sheet')
        plates.append(sheet['id']);chambers.append(box['id'])
        _ellipsoid(scene,uid+'_'+label+'_head',[c,0.,center],[head,radius,radius],uid)
    # The shell-bottom nozzles are at +/-0.30L. Keep real saddle blocks away
    # from both selected nozzles instead of placing a full support under them.
    for side in (-1,1):
        x=side*.08*length;half=min(.10,.08*length)
        saddle=add_box(scene,uid+'_saddle'+str(side),
            [x-half,-radius*.75,z,x+half,radius*.75,center],
            'saddle',(.31,.42,.49),uid)
        saddle.update(estimated=True,scope="Ground-mounted twin saddle envelopes clear the shell's lower ports")
    for name,(start,end) in horizontal_exchanger_nozzles(unit,spec,z).items():
        _short_nozzle(scene,unit,name,start,end)
    span=length/2+chamber+head;outer=radius+spec['tube_sheet_radial_extra_m_estimate']
    unit.update(center=[0.,0.,center],box=[-span,-outer,center-outer,span,outer,center+outer],
        bottom=center-outer,top=center+outer,radius=radius,support_reference_z_m=z,
        heat_exchanger_construction={'shell_component':uid,'tube_sheet_components':plates,
            'channel_box_components':chambers,'tube_shell_length_m':length,
            'total_axial_length_m':2*span,'local_axis':[1.,0.,0.],
            'scope':'The two channels, two sheets and central shell have physical envelope geometry. Fluid and heat transfer are outside the scope.'})


def _horizontal(scene,unit,spec,z):
    if unit['kind'] in ('exchanger','condenser'):
        return _horizontal_exchanger(scene,unit,spec,z)
    uid=unit['id'];radius=spec['diameter_m']/2;length=spec['length_m'];head=spec['head_depth_m']
    center_z=z+spec['leg_clearance_m']+radius
    add_pipe(scene,uid,[-length/2,0.,center_z],[length/2,0.,center_z],radius,unit['kind'],(.69,.74,.75),uid)
    for side in (-1,1):
        _ellipsoid(scene,uid+('_left_head' if side<0 else '_right_head'),
                   [side*length/2,0.,center_z],[head,radius,radius],uid)
    if unit['kind']=='vessel':
        from .storage_vessel_model import build_fittings
        build_fittings(scene,unit,spec,z)
    else:
        _saddles(scene,unit,z,radius,length,center_z)
        for index,name in enumerate(unit['ports']):
            if name in ('in','vapor'):
                start=[-length/2-head,0.,center_z];end=[start[0]-.12,0.,center_z]
            elif name in ('out','condensate'):
                start=[length/2+head,0.,center_z];end=[start[0]+.12,0.,center_z]
            elif name in ('bottom','drain'):
                start=[0.,0.,center_z-radius];end=[0.,0.,start[2]-.12]
            else:
                x=(-.2 if index%2 else .2)*length
                start=[x,0.,center_z+radius];end=[x,0.,start[2]+.12]
            _short_nozzle(scene,unit,name,start,end)
    unit.update(center=[0.,0.,center_z],box=[-length/2-head,-radius,center_z-radius,
                length/2+head,radius,center_z+radius],bottom=center_z-radius,top=center_z+radius,
                radius=radius,support_reference_z_m=z)


def _vertical_exchanger(scene,unit,spec,z):
    """Radial mouths on actual channel boxes/shell, per accepted figures 19/20."""
    uid=unit['id'];r=spec['diameter_m']/2;length=spec['length_m']
    chamber=spec['channel_box_length_m_estimate'];plate=spec['tube_sheet_thickness_m_estimate']
    head=spec['head_depth_m'];bottom=z+spec['leg_clearance_m']
    low=bottom+head+chamber;high=low+length
    add_pipe(scene,uid,[0,0,low],[0,0,high],r,unit['kind'],(.69,.74,.75),uid)
    plates=[];boxes=[]
    for side,label,face in ((-1,'lower',low),(1,'upper',high)):
        a=face+side*plate;b=face+side*chamber
        sheet=add_pipe(scene,uid+'_'+label+'_tube_sheet',[0,0,face],[0,0,a],
            r+spec['tube_sheet_radial_extra_m_estimate'],'tube_sheet',(.38,.46,.48),uid)
        box=add_pipe(scene,uid+'_'+label+'_channel_box',[0,0,a],[0,0,b],r,'channel_box',(.63,.72,.73),uid)
        sheet.update(estimated=True);box.update(estimated=True,circuit_side='tube')
        plates.append(sheet['id']);boxes.append(box['id'])
        _ellipsoid(scene,uid+'_'+label+'_head',[0,0,b],[r,r,head],uid)
    # Four legs at diagonals leave cardinal nozzles unobstructed.
    for i in range(4):
        angle=math.pi/4+i*math.pi/2;x=r*.72*math.cos(angle);y=r*.72*math.sin(angle)
        add_box(scene,uid+'_leg'+str(i),[x-.025,y-.025,z,x+.025,y+.025,low],
                'support',(.27,.38,.44),uid)
    boxlow=low-(chamber+plate)/2;boxhigh=high+(chamber+plate)/2
    from .pipe_spec import load
    od=load()['pipe_outer_diameter_mm']
    def clearance(name):return od[str(unit['ports'][name]['dn'])]/2000+.01
    if unit['kind']=='condenser':
        table={'vapor':(-1,low+clearance('vapor'),'shell','lower_shell','shared_vapor_liquid'),
               'offgas':(1,high-clearance('offgas'),'shell','upper_shell','noncondensable_out'),
               'utility_in':(1,boxlow,'tube','lower_channel_box','cold_in'),
               'utility_out':(-1,boxhigh,'tube','upper_channel_box','cold_out')}
    else:
        table={'in':(1,boxlow,'tube','lower_channel_box','material_in'),
               'out':(-1,boxhigh,'tube','upper_channel_box','material_out'),
               'utility_in':(1,high-clearance('utility_in'),'shell','upper_shell','steam_in'),
               'utility_out':(-1,low+clearance('utility_out'),'shell','lower_shell','steam_condensate_out')}
    for name,port in unit['ports'].items():
        if name not in table:raise CandidateRefused('Vertical heat exchanger is missing a uniquely assigned port role: '+uid+'.'+name)
        side,height,circuit,zone,role=table[name]
        port.update(circuit_side=circuit,nozzle_zone=zone,process_role=role,nozzle_mount='radial',
                    nozzle_position_source='Candidate regions from figures 19 and 20. Positions are derived from equipment dimensions.')
        _short_nozzle(scene,unit,name,[side*r,0,height],[side*(r+.12),0,height])
    top=high+chamber+head;outer=r+spec['tube_sheet_radial_extra_m_estimate']
    unit.update(center=[0,0,(bottom+top)/2],box=[-outer,-outer,bottom,outer,outer,top],
        bottom=bottom,top=top,radius=r,support_reference_z_m=z,
        heat_exchanger_construction={'shell_component':uid,'tube_sheet_components':plates,
            'channel_box_components':boxes,'tube_shell_length_m':length,'total_axial_length_m':top-bottom,
            'local_axis':[0,0,1],'scope':'Physical envelope without internal heat-transfer calculations'})


def _jacket_ports(scene,unit,graph,spec,z):
    """The factory d2 body already envelopes the jacket; attach its real mouths."""
    companions=[]
    for original in graph['units'].values():
        if original['kind']!='jacket' or original.get('owner')!=unit['id']:continue
        if spec['form']!='vertical':raise CandidateRefused('Jacket connections are supported on vertical reactors only')
        jacket=deepcopy(original);radius=unit['radius']
        body=next(c for c in scene['components'] if c['id']==unit['id'])
        lo,hi=body['ends'][0][2],body['ends'][1][2]
        from .pipe_spec import load
        od=load()['pipe_outer_diameter_mm']
        for name,port in jacket['ports'].items():
            steam=jacket['medium']=='steam';upper=(name=='utility_in')==steam
            rr=od[str(port['dn'])]/2000
            height=hi-rr-.03 if upper else lo+rr+.03
            side=1 if upper else -1
            port['medium']=('steam' if name=='utility_in' else 'steam_condensate') if steam else ('cold_source' if name=='utility_in' else 'cold_return')
            port['nozzle_mount'] = 'jacket'
            _short_nozzle(scene,jacket,name,[0,side*radius,height],[0,side*(radius+.12),height])
            scene['components'][-1].update(owner=unit['id'],attached_jacket=jacket['id'])
        jacket.update(center=deepcopy(unit['center']),box=deepcopy(unit['box']),level=unit['level'],
                      integral_to=unit['id'],geometry_scope='Catalogue d2 includes the jacket envelope. Add only the separate medium necks.')
        companions.append(jacket)
    return companions


def _vertical_auxiliary(scene,unit,spec,z):
    uid=unit['id'];radius=spec['diameter_m']/2;length=spec['length_m'];head=spec['head_depth_m']
    bottom=z+spec['leg_clearance_m'];cylinder_bottom=bottom+head;cylinder_top=cylinder_bottom+length
    add_pipe(scene,uid,[0.,0.,cylinder_bottom],[0.,0.,cylinder_top],radius,unit['kind'],(.72,.62,.40),uid)
    _ellipsoid(scene,uid+'_lower_head',[0.,0.,cylinder_bottom],[radius,radius,head],uid)
    _ellipsoid(scene,uid+'_upper_head',[0.,0.,cylinder_top],[radius,radius,head],uid)
    for index,angle in enumerate((0.,math.pi/2,math.pi,3*math.pi/2)):
        x,y=radius*.72*math.cos(angle),radius*.72*math.sin(angle)
        part=add_box(scene,uid+'_leg'+str(index),[x-.035,y-.035,z,x+.035,y+.035,cylinder_bottom],
                     'support',(.27,.38,.44),uid)
        part.update(estimated=True,support_type='legs')
    for name in unit['ports']:
        side=1 if name.startswith('utility') else -1
        height=cylinder_bottom+.15 if name in ('in','condensate','utility_in') else cylinder_top-.15
        _short_nozzle(scene,unit,name,[0.,side*radius,height],[0.,side*(radius+.12),height])
    unit.update(center=[0.,0.,(bottom+cylinder_top+head)/2],
                box=[-radius,-radius,bottom,radius,radius,cylinder_top+head],
                bottom=bottom,top=cylinder_top+head,radius=radius,support_reference_z_m=z)


def _fixed_bed(scene,unit,spec):
    uid=unit['id'];radius=spec['diameter_m']/2;head=spec['head_depth_m']
    lower=spec['skirt_height_m'];top=spec['height_m']
    cylinder_bottom=lower+head;cylinder_top=top-head
    add_pipe(scene,uid,[0.,0.,cylinder_bottom],[0.,0.,cylinder_top],radius,unit['kind'],(.63,.72,.76),uid)
    _ellipsoid(scene,uid+'_lower_head',[0.,0.,cylinder_bottom],[radius,radius,head],uid)
    _ellipsoid(scene,uid+'_upper_head',[0.,0.,cylinder_top],[radius,radius,head],uid)
    skirt=add_pipe(scene,uid+'_skirt',[0.,0.,0.],[0.,0.,lower],radius,'skirt',(.28,.40,.46),uid)
    skirt.update(estimated=True,scope='Ground-mounted skirt envelope')
    for name in unit['ports']:
        if name=='in':start,end=[0.,0.,top-.02],[0.,0.,top+.12]
        else:start,end=[0.,0.,lower+.02],[0.,0.,lower-.12]
        _short_nozzle(scene,unit,name,start,end)
    unit.update(center=[0.,0.,(lower+top)/2],box=[-radius,-radius,lower,radius,radius,top],
                bottom=lower,top=top,radius=radius,level=0,support_reference_z_m=0.,
                flow_arrangement='top_in_bottom_out')


def _rotate_rect(rect,matrix):
    points=np.asarray([[x,y] for x in (rect[0],rect[2]) for y in (rect[1],rect[3])])@matrix[:2,:2].T
    return [*points.min(axis=0).tolist(),*points.max(axis=0).tolist()]


def _rotate(scene,unit,angle):
    # Exact quarter-turn matrix avoids tiny artificial holes from sin(pi/2).
    matrix=np.eye(3) if angle==0 else np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    for c in scene['components']:
        rect=_rotate_rect([c['box'][0],c['box'][1],c['box'][3],c['box'][4]],matrix)
        c['box']=[rect[0],rect[1],c['box'][2],rect[2],rect[3],c['box'][5]]
        if 'ends' in c:c['ends']=[(matrix@np.asarray(p)).tolist() for p in c['ends']]
        if 'pos' in c:c['pos']=(matrix@np.asarray(c['pos'])).tolist()
        if c['shape']=='ellipsoid' and angle:
            c['size']=[c['size'][1],c['size'][0],c['size'][2]]
    unit['center']=(matrix@np.asarray(unit['center'])).tolist()
    rect=_rotate_rect([unit['box'][0],unit['box'][1],unit['box'][3],unit['box'][4]],matrix)
    unit['box']=[rect[0],rect[1],unit['box'][2],rect[2],rect[3],unit['box'][5]]
    for port in unit['ports'].values():
        for field in ('pos','normal','stem','surface_pos'):
            if field in port:port[field]=(matrix@np.asarray(port[field])).tolist()
    for level in scene['holes']:
        scene['holes'][level]=[_rotate_rect(rect,matrix) for rect in scene['holes'][level]]
    for option in scene['extraction_options']:
        option['rect']=_rotate_rect(option['rect'],matrix)
        option['axis']=(matrix@np.asarray(option['axis'])).tolist()
    return matrix


def prototype(unit,graph,shell,rng,rotation_deg=None):
    """Build physical equipment at (0,0), without installed process valves."""
    spec=resolve_spec(unit)
    scene={'components':[],'holes':defaultdict(list),'extraction_options':[]}
    floors=shell['floors'];z=floors[unit['level']]
    if unit['kind'] in ('fixed_bed','tower'):
        _fixed_bed(scene,unit,spec)
    elif spec['form']=='horizontal':
        _horizontal(scene,unit,spec,z)
    elif unit['kind']=='vessel':
        _vertical_reactor(scene,unit,spec,z)
    elif unit['kind'] in ('exchanger','condenser'):_vertical_exchanger(scene,unit,spec,z)
    else:_vertical_auxiliary(scene,unit,spec,z)
    companions=_jacket_ports(scene,unit,graph,spec,z) if unit['kind']=='vessel' else []
    # Actual shell sections receive normal local equipment holes at every
    # crossed platform, including a fixed bed whose head rises above the roof.
    for level,floor_z in enumerate(floors[1:],1):
        if unit['bottom'] < floor_z < unit['top']:
            radius=spec['diameter_m']/2
            if spec['form']=='vertical':
                scene['holes'][level].append([-radius-.1,-radius-.1,radius+.1,radius+.1])
            else:
                raise CandidateRefused('The horizontal vessel crosses the platform plane. Change operating levels rather than cutting a body-sized opening.')
        elif not (unit['kind']=='vessel' and spec['form']=='vertical'):
            # Useful decks may cross the drive's reserved headroom. Keep the
            # actual collision box and reserve a local penetration, using the
            # same existing 0.1 m opening allowance as the body penetration.
            for drive in (c for c in scene['components'] if c['kind']=='drive_reservation'):
                b=drive['box']
                if b[2] < floor_z < b[5]:
                    scene['holes'][level].append([b[0]-.1,b[1]-.1,b[3]+.1,b[4]+.1])
    if spec['form']=='horizontal' and unit['kind'] in ('exchanger','condenser'):
        from .building_equipment_checks import read_layout_rules
        band=spec['diameter_m']+read_layout_rules()['extraction_width_extra_m']
        length=spec['length_m']
        for side in (-1,1):
            start=side*(spec['total_length_m']/2);end=start+side*length
            scene['extraction_options'].append({'end':side,'rect':[min(start,end),-band/2,max(start,end),band/2],
                'axis':[float(side),0.,0.],'length_m':length,'width_m':band})
    angle=rng.choice((0,90)) if rotation_deg is None else rotation_deg
    if angle not in (0,90):raise CandidateRefused("The saved equipment orientation is outside the catalogue's supported range")
    matrix=_rotate(scene,unit,angle)
    for companion in companions:
        _rotate({'components':[],'holes':{},'extraction_options':[]},companion,angle)
    for component in scene['components']:
        component.setdefault('estimated',True)
        component['catalog_key']=spec['catalog_key']
    parts=scene['components']
    rect=[min(c['box'][0] for c in parts),min(c['box'][1] for c in parts),
          max(c['box'][3] for c in parts),max(c['box'][4] for c in parts)]
    low=min(c['box'][2] for c in parts);high=max(c['box'][5] for c in parts)
    occupied=[i for i,floor_z in enumerate(floors)
              if high>floor_z+1e-8 and (i==len(floors)-1 or low<floors[i+1]-1e-8)]
    operating=[unit['level']]
    if unit['kind'] in ('fixed_bed','tower'):
        operating=[i for i,floor_z in enumerate(floors) if 0<=floor_z<unit['top']]
    unit.update(catalog=deepcopy(spec),catalog_key=spec['catalog_key'],dimension_spec=deepcopy(spec),
                form=spec['form'],orientation=spec['form'],size_tier=spec['size_tier'],
                rotation_deg=angle,operating_levels=operating,support_type=spec['support_type'])
    from .pipe_spec import load
    band=load()['operation_band_m']
    for name,port in unit['ports'].items():
        if 'pos' not in port:continue
        pz=port['pos'][2]
        eligible=[i for i,floor_z in enumerate(floors) if floor_z+band[0]-1.e-8<=pz<=floor_z+band[1]+1.e-8]
        level=unit['level'] if unit['level'] in eligible else (min(eligible,key=lambda i:abs(i-unit['level'])) if eligible else unit['level'])
        floor_z=floors[level]
        port['operation_reference']={'level':level,'standing_surface_z_m':floor_z,
            'operation_band_m':list(band),'allowed_operation_z_m':[floor_z+band[0],floor_z+band[1]],
            'interface_height_m':pz-floor_z,'interface_within_operation_band':level in eligible,
            'status':'await_actual_standing_face_and_real_valve_placement',
            'scope':'installation level reference'}
    from .pullout_reservations import expected_pullout
    from .building_equipment_checks import read_layout_rules
    pullout=expected_pullout({'id':unit['id'],'kind':unit['kind'],'parts':parts,'rect':rect},
                             unit,{'floors':floors},read_layout_rules())
    pullout_openings=[]
    if pullout:
        for level,floor_z in enumerate(floors[1:],1):
            if floor_z>pullout['base_z']+1e-8:
                hole=list(pullout['rect'])
                scene['holes'][level].append(hole)
                pullout_openings.append({'level':level,'rect':hole,'owner':unit['id'],
                                         'reservation_id':pullout['id'],'kind':'pullout'})
    return {'unit':unit,'companions':companions,'components':parts,'holes':dict(scene['holes']),'rect':rect,
            'occupied_levels':occupied,'z_range_m':[low,high],
            'catalog_key':spec['catalog_key'],'dimension_spec':deepcopy(spec),
            'extraction_options':scene['extraction_options'],'operating_levels':operating,
            'pullout_reservations':[pullout] if pullout else [],'pullout_openings':pullout_openings,
            'axis':(matrix@np.array([1.,0.,0.] if spec['form']=='horizontal' else [0.,0.,1.])).tolist(),'rotation_deg':angle,
            'support_reference_z_m':unit['support_reference_z_m']}
