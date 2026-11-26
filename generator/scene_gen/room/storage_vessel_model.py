"""Build the horizontal storage vessel and its pipe connections."""
import math

import numpy as np

from .layout import add_box, add_pipe
from .recipe import CandidateRefused

TANK = 'Zhejiang Jinfulong horizontal elliptical-head tank 4 m³ nozzle schedule and drawing, D1200 and 3200 mm straight shell. https://zjjfl.cnreagent.com/product_5122.html'
FLANGE = 'HG/T 20592-2009 PN16 PL-RF, pipe series A. https://www.ghfitting.com/news/chinese-flange/426.html'
BLIND = 'HG/T 20592-2009 PN16 BL-RF；https://www.ghfitting.com/news/chinese-flange/461.html'
MANWAY = 'HG/T 21518-2014 PN16 DN450 swing-cover weld-neck manway, Longan catalogue printed pages 92 and 93. https://www.longanflange.com/wp-content/uploads/2024/10/38821c5b46531428e47c.pdf#page=55'
SADDLE = 'NB/T 47065.1-2018 figure 1 and table 2, DN1200 light type A saddle. Envelope adjusted for the specified 540 mm bottom clearance. https://std.samr.gov.cn/hb/search/stdHBDetailedCNF?id=8B1827F24D25BB19E05397BE0A0AB44A'
BODY = "Reference R1 on the second level has diameter 1200, straight length 2800, head depth 180 and bottom clearance 540 mm. Heads follow Jinfulong's profile."
# Steel tube OD, flange OD, bolt circle, hole diameter / count, thread, plate
# thickness, raised face diameter/height and blind thickness, all in mm.
PL = {
    15:(21.3,95,65,14,4,12,14,45,2,16),
    25:(33.7,115,85,14,4,12,16,68,2,18),
    40:(48.3,150,110,18,4,16,18,88,2,18),
    50:(60.3,165,125,18,4,16,19,102,2,18),
    65:(76.1,185,145,18,8,16,20,122,2,18),
    # Human-access assembly has its own DN450 neck and bolting table.
    450:(480.,640,585,30,20,27,40,550,2,40),
}
# A-series pipe fit B, and the existing coarse wrench-flat / axial nut sizes.
PL_BORE_MM = {15:22.5,25:34.5,40:49.5,50:61.5,65:77.5}
FLANGE_NUT_MM = {12:(18,10),16:(24,13),27:(41,22)}
# Tank drawing at 700 px: axial reference marks x=105 and x=616; separate
# y scale uses the dimensioned Ø1200 between the shell silhouettes y=92,275.
AXIAL = 'Map Jinfulong drawing shell ends at x=105 and 616 pixels to the 2800 mm straight length.'
PROJECTION = "Scale face projection using the drawing's 183-pixel shell diameter as 1200 mm."
PLATE_COLOR=(.59,.66,.69)
NECK_COLOR=(.47,.57,.60)
BOLT_COLOR=(.21,.27,.30)

# DN450 manway, Longan pp. 92–93. The handle rod diameter uses the
# measured 3:60 bar-width / dimensioned leg-span ratio on the drawing.
MANWAY_H1_M = .240
MANWAY_H2_M = .120
HANDLE_SPAN_M = .150
HANDLE_RADIUS_M = HANDLE_SPAN_M * (3 / 60) / 2


def installed_height(shell_top_m):
    """Support-to-top distance, ending at the outside of the handle bar."""
    return shell_top_m + MANWAY_H1_M + MANWAY_H2_M + HANDLE_RADIUS_M


def _x(pixel, spec):
    return ((pixel-105)/511-.5)*spec['length_m']


def schedule(unit, spec):
    """Local coordinates: origin under tank centre on support, positive X toward h."""
    ports = unit['ports']
    radius = spec['diameter_m']/2
    center = spec['leg_clearance_m']+radius
    items = [
        # A shared inlet (several streams joined outside the tank) uses the
        # same top feed nozzle under the process name liquid_return.
        ('a','liquid_return' if 'liquid_return' in unit['ports'] else 'feed','feed',50,'top',244),
        ('b','nitrogen','spare port used for blanketing when nitrogen is connected',65,'top',328),
        ('c','offgas','vent or off-gas',40,'top',427),
        ('d','bottom','material discharge',50,'bottom',130),
        ('e','manway','top swing-cover manway',450,'top',167),
        ('f1','level_upper','upper level-gauge port',15,'end',.50),
        ('f2','level_lower','lower level-gauge port',15,'end',-.50),
        ('h','drain','separate drain',50,'bottom',592),
        ('k','safety','safety relief connection',50,'top',473),
        ('m','gauge','pressure gauge connection',25,'top',364),
    ]
    for name in ports:
        if name.startswith('recycle_'):
            if any(r[0]=='(a)' for r in items):
                raise CandidateRefused('The horizontal vessel drawing lists only one optional second inlet')
            items.append(('(a)',name,"optional second inlet for the recipe's independent recirculation",50,'top',286))
    unknown=set(ports)-{row[1] for row in items}
    if unknown:
        raise CandidateRefused('The nozzle schedule lacks a separate port for this service: '+','.join(sorted(unknown)))
    rows=[]
    for tag,name,purpose,dn,direction,parameter in items:
        source=[TANK, MANWAY if tag=='e' else FLANGE]
        if direction=='end':
            nz=parameter/radius
            start=[-spec['length_m']/2-spec['head_depth_m']*math.sqrt(1.-nz*nz),0.,center+parameter]
            normal=[-1.,0.,0.]
            # Both level gauge flange faces lie in one plane. Drawing shows
            # their common face at x=40 and the left head apex at x=54.
            face_x=-spec['total_length_m']/2-14*spec['length_m']/511
            projection=start[0]-face_x
            position='Left head, normal along negative X. Use f1/f2 spacing A=1000 mm and map the common face from its 14-pixel offset.'
        else:
            sign=1 if direction=='top' else -1
            start=[_x(parameter,spec),0.,center+sign*radius]
            normal=[0.,0.,sign]
            projection=(MANWAY_H1_M if name=='manway' else (30 if sign==1 else 23)*spec['diameter_m']/183)
            position=AXIAL+f' Port x={parameter} pixels.'+('Manway neck height follows H1 in the selected manway drawing.' if name=='manway' else PROJECTION)
        if tag=='(a)':
            position+=' The 4 m³ baseline table has no second inlet. Use the optional DN50 station only if the recipe specifies a second feed.'
        face=(np.asarray(start)+np.asarray(normal)*projection).tolist()
        od,diam,pcd,hole,n,thread,thick,rf,rfh,blind=PL[dn]
        flange={'standard':'HG/T 21518-2014 PN16' if tag=='e' else 'HG/T 20592-2009 PN16 PL-RF',
                'outer_diameter_mm':diam,'bolt_circle_mm':pcd,'hole_diameter_mm':hole,
                'bolt_count':n,'bolt_thread':'M'+str(thread),'plate_thickness_mm':thick,
                'rf_diameter_mm':rf,'rf_height_mm':rfh}
        if tag == 'e':
            flange.update(neck_od_mm=480,neck_wall_mm=10,stud_length_mm=155,nut_count=40,
                          cover_thickness_mm=40,hinge_offset_mm=350,hinge_pin_mm=24,
                          handle_offset_mm=175,handle_span_mm=HANDLE_SPAN_M*1000)
        connected=name in ports
        rows.append({'port':name,'drawing_tag':tag,'purpose':purpose,'dn':dn,
                     'line_dn':ports[name]['dn'] if connected else None,
                     'nozzle_mount':'end_head' if direction=='end' else 'shell',
                     'kind':'manway' if tag=='e' else 'instrument' if tag in ('f1','f2','m') else 'spare' if tag=='b' else 'process',
                     'connected_to_process':connected,'face_local_m':face,'surface_local_m':start,
                     'normal_local':normal,'flange':flange, 'position_basis':position,
                     'sources':source,'component_ids':[]})
    return rows


def _saddles(scene, unit, spec, z):
    """Grounded bearing plates, web and four transverse stiffeners per support."""
    uid=unit['id'];radius=spec['diameter_m']/2;center=z+spec['leg_clearance_m']+radius
    # Symmetric spacing from the dimensioned L1=2500, mapped from 3200 onto
    # 2800. Do not add L2 to L1: the manufacturer's table is inconsistent there.
    for sign in (-1,1):
        x=sign*1.25*spec['length_m']/3.2
        name=f'{uid}_saddle{sign}'
        def plate(suffix,box,kind):
            return add_box(scene,name+suffix,box,kind,(.30,.42,.47),uid)
        plate('_base',[x-.085,-.44,z,x+.085,.44,z+.010],'saddle_base')
        # The web primitive also contacts the bearing plane to preserve the
        # support contact representation at the centre of its integral base.
        plate('',[x-.003,-.38,z,x+.003,.38,center-radius+.02],'saddle')
        for i,y in enumerate((-.36,-.12,.12,.36)):
            ceiling=center-math.sqrt(radius**2-y**2)
            plate(f'_rib{i+1}',[x-.076,y-.003,z+.010,x+.076,y+.003,ceiling+.008],'saddle_rib')
        # Faceted curved wearing plate hugging the lower 120° of the shell.
        # Parallel contact strips use cylinder surface coordinates; slight
        # overlap closes the joints between strips in the exterior model.
        span=math.radians(60);segments=12
        for i in range(segments):
            a=-span+i*2*span/segments;b=a+2*span/segments
            curve=lambda t: [x,radius*math.sin(t),center-radius*math.cos(t)]
            y1,z1=curve(a)[1:];y2,z2=curve(b)[1:]
            plate(f'_liner{i+1}',[x-.115,y1,min(z1,z2)-.008,x+.115,y2,max(z1,z2)+.002],'saddle_liner')


def _cylinder(scene,uid,key,base,axis,lo,hi,radius,kind,col):
    return add_pipe(scene,key,(base+axis*lo).tolist(),(base+axis*hi).tolist(),radius,kind,col,uid)


def build_fittings(scene, unit, spec, z):
    """Materialize the schedule and copy its mating faces to the process ports."""
    uid=unit['id'];ports=unit['ports'];rows=schedule(unit,spec)
    _saddles(scene, unit, spec, z)
    for row in rows:
        name=row['port'];tag=row['drawing_tag'];dn=row['dn']
        od,diam,pcd,hole,count,thread,thick,rf,rfh,blind=PL[dn]
        base=np.asarray(row['face_local_m'])+[0.,0.,z];axis=np.asarray(row['normal_local'])
        mouth=np.asarray(row['surface_local_m'])+[0.,0.,z]
        prefix=uid+'_'+name
        first=len(scene['components'])
        # Preserve the machinery contract: native neck and its flange share
        # the mating plane, also when the old process line has a different DN.
        neck=add_pipe(scene,prefix+'_equipment_nozzle',mouth.tolist(),base.tolist(),od/2000,
                      'equipment_nozzle',NECK_COLOR,uid)
        def solid(suffix,lo,hi,radius,kind,color):
            return _cylinder(scene,uid,prefix+'_'+suffix,base,axis,lo,hi,radius,kind,color)
        back=-(thick+rfh)/1000
        solid('flange',back,-rfh/1000,diam/2000,'nozzle_flange',PLATE_COLOR)
        solid('rf',-rfh/1000,0.,rf/2000,'raised_face',(.48,.54,.57))
        # The reference depicts a cover on e; other nozzle faces are shown
        # separately from any field-installed piping or instruments.
        capped=tag=='e'
        front=(blind+rfh)/1000 if capped else 0.
        if capped:
            solid('cover_face',0.,rfh/1000,rf/2000,'raised_face',(.43,.48,.51))
            solid('cover',rfh/1000,front,diam/2000,
                  'manway_cover' if tag=='e' else 'blind_cover',(.56,.62,.63))
        ex=np.array([1.,0.,0.]) if axis[2] else np.array([0.,1.,0.]);ey=np.cross(axis,ex)
        flats,nut=(v/1000 for v in FLANGE_NUT_MM[thread])
        for i in range(count):
            theta=math.tau*(i+.5)/count
            bolt=base+(ex*math.cos(theta)+ey*math.sin(theta))*pcd/2000
            boltbase=prefix+f'_bolt{i+1:02}'
            stud_back=back-nut
            stud_front=front+nut
            if tag=='e':
                mid=(back+front)/2
                stud_back,stud_front=mid-.155/2,mid+.155/2
                _cylinder(scene,uid,boltbase+'_lower_nut',bolt,axis,back-nut,back,flats/2,'bolt_nut',(.30,.35,.38))
            _cylinder(scene,uid,boltbase,bolt,axis,stud_back,stud_front,thread/2000,'flange_bolt',BOLT_COLOR)
            # Nut external envelope; threads and wrench flats omitted.
            _cylinder(scene,uid,boltbase+'_head',bolt,axis,front,front+nut,flats/2,'bolt_head',(.30,.35,.38))
        if tag=='e':
            # Longan, p92 RF drawing: A-type lugs. Orient the hinge toward
            # the left head as in JFL's elevation, handle on the opposite side.
            hinge_x=base[0]-.350
            hinge_z=base[2]+front*.5
            add_pipe(scene,prefix+'_hinge_pin',[hinge_x,-.125,hinge_z],[hinge_x,.125,hinge_z],
                     .012,'manway_hinge',BOLT_COLOR,uid)
            for i,side in enumerate((-1,1)):
                # Lug widths relative to L on the manufacturer's plan.
                y=side*.106
                for suffix,offset in [('lid',-.010),('base',.010)]:
                    add_box(scene,prefix+f'_hinge_{suffix}{i+1}',
                            [hinge_x-.018,y+offset-.008,hinge_z-.018,
                             base[0]-.300,y+offset+.008,hinge_z+.018],
                            'manway_hinge',PLATE_COLOR,uid)
            handle_x=base[0]+.175
            bar_z=base[2]+MANWAY_H2_M
            # The handle diameter is taken as the figure's bar/leg-span ratio
            # (dimensioned 150 mm); it is a drawing-derived exterior envelope.
            bar_r=HANDLE_RADIUS_M
            half_span=HANDLE_SPAN_M/2
            for y in [-half_span,half_span]:
                add_pipe(scene,prefix+('_handle_near' if y<0 else '_handle_far'),
                         [handle_x,y,base[2]+front],[handle_x,y,bar_z],bar_r,'manway_handle',BOLT_COLOR,uid)
            add_pipe(scene,prefix+'_handle',[handle_x,-half_span,bar_z],[handle_x,half_span,bar_z],
                     bar_r,'manway_handle',BOLT_COLOR,uid)
        ids=[c['id'] for c in scene['components'][first:]]
        row['component_ids']=ids
        for c in scene['components'][first:]:
            c['port']=name
        if row['connected_to_process']:
            ports[name].update(nozzle_mount=row['nozzle_mount'],pos=base.tolist(),surface_pos=mouth.tolist(),normal=axis.tolist(),
                 stem=[0.,1.,0.] if axis[2] else [0.,0.,1.],equipment_neck_dn=dn,
                 line_dn=row['line_dn'],nozzle_component_id=neck['id'],component_ids=ids,
                 actual_neck_outer_diameter_m=od/1000, equipment_neck_source=TANK)
    unit['nozzle_schedule']=rows
    unit['external_components']=[
        {'id':c['id'],'kind':c['kind'],'port':c.get('port'),
         'source':(SADDLE if 'saddle' in c['kind'] else BODY if c.get('port') is None else
                   MANWAY if c['port']=='manway' else BLIND if c['kind']=='blind_cover' else
                   TANK if c['kind']=='equipment_nozzle' else FLANGE)}
        for c in scene['components'] if c.get('owner')==uid]
