"""Build reactor nozzles, covers and the oval manway."""
import math

import numpy as np

from .layout import add_pipe
from .piping_equipment_flange import _sections
from .reactor_f_series_shell import PLAN, SECTION, _prism


REFERENCE = "Reference flange profile follows HG/T 20592-2009 PN16 PL-RF series A. The manufacturer's outline does not specify a supplied series."
REFERENCE_URL = 'https://www.ghfitting.com/news/chinese-flange/426.html'
# A-series tube OD, D, K, L, count, stud, C, RF OD, f1, and fit bore B (mm).
FLANGES = {
    80: (88.9,200,160,18,8,16,20,138,2,90.5),
    125: (139.7,250,210,18,8,16,22,188,2,143.5),
    150: (168.3,285,240,22,8,20,24,212,2,170.5),
    200: (219.1,340,295,22,12,20,26,268,2,221.5),
}
SILVER = (.56,.64,.68)
DARK = (.32,.40,.45)
HARDWARE = (.33,.37,.39)


def _frame(normal):
    axis=np.asarray(normal,dtype=float);axis/=np.linalg.norm(axis)
    reference=np.eye(3)[np.argmin(abs(np.eye(3)@axis))]
    x=np.cross(reference,axis);x/=np.linalg.norm(x)
    return x,np.cross(axis,x),axis


def _round_sections(inner,outer,pitch,hole,count,start=0.):
    sector=_sections(inner,outer,pitch,hole,count)
    result=[]
    for n in range(count):
        a=start+n*math.tau/count;c,s=math.cos(a),math.sin(a)
        matrix=np.array([[c,-s],[s,c]])
        result.extend(poly@matrix.T for poly in sector)
    return result


def _ring(scene,folder,owner,id_prefix,kind,face,frame,thickness,offset,polygons,
          color,source,status,note):
    x,y,axis=frame
    base=np.asarray(face)+offset*axis
    ids=[]
    for i,poly in enumerate(polygons):
        lower=base+np.outer(poly[:,0],x)+np.outer(poly[:,1],y)
        part=_prism(scene,folder,f'{id_prefix}_{i:03}',kind,lower,axis*thickness,
                    color,source,status,note,owner)
        ids.append(part['id'])
    return ids


def _cylinder(scene,owner,uid,kind,start,end,radius,color,source,status,note):
    part=add_pipe(scene,uid,np.asarray(start).tolist(),np.asarray(end).tolist(),radius,kind,color,owner)
    if part is None:
        raise ValueError('External component axial length must be positive: '+uid)
    part.update(source=source,evidence_status=status,note=note)
    return part['id']


def _tube(scene,folder,owner,uid,start,finish,frame,outer,source,status):
    axis=frame[2];length=float((finish-start)@axis)
    if length <= 1e-8 or np.linalg.norm((finish-start)-axis*length)>1e-8:
        raise ValueError('Neck must extend outward along the physical port normal: '+uid)
    return _ring(scene,folder,owner,uid,'equipment_neck',start,frame,length,0.,
                 _round_sections(outer*.82,outer,0.,0.,4),SILVER,source,status,
                 'The neck has an exterior bore with an estimated visible ID.')


def _flange(scene,folder,owner,uid,row,face,frame):
    od,outside,pitch,hole,count,thread,plate,rf,raised,bore=FLANGES[row['dn']]
    outer_d,pitch_d,hole_d,rf_d,bore_d=(v/2000 for v in (outside,pitch,hole,rf,bore))
    length=plate/1000;raised=raised/1000
    slot_phase=math.pi/count
    pattern=_round_sections(bore_d,outer_d,pitch_d,hole_d,count,slot_phase)
    ids=_ring(scene,folder,owner,uid+'_plate','nozzle_flange',face,frame,length,-length-raised,
              pattern,SILVER,REFERENCE,'project-standard envelope','Use a DN-matched reference plate with bore and drilled bolt holes.')
    ids+=_ring(scene,folder,owner,uid+'_rf','nozzle_raised_face',face,frame,raised,-raised,
               _round_sections(bore_d,rf_d,0.,0.,count,slot_phase),DARK,REFERENCE,'project-standard envelope','Locate the original mating plane at the outer raised face.')
    plate_face_ids=ids.copy()
    status=row['geometry_status']
    if status=='reference blind cover' or status in ('thermowell end cover','sight glass end cover'):
        x,y,axis=frame
        cap_thickness=length
        ids+=_ring(scene,folder,owner,uid+'_blank_ring','nozzle_cover',face,frame,cap_thickness,0.,
                   pattern,SILVER,REFERENCE,'estimated from the drawing','Reference end cover envelope matches the port bolt circle.')
        glass=status=='sight glass end cover'
        ids.append(_cylinder(scene,owner,uid+'_cover_center','sightglass' if glass else 'nozzle_cover',
                             face,face+cap_thickness*axis,(bore_d/math.cos(math.pi/24))+.0005,
                             (.26,.46,.49) if glass else SILVER,PLAN+'；'+SECTION,
                             'estimated from the drawing','Visible sight glass outline. Cover thickness is unspecified.'))
        nut_radius,nut_length=(.012,.013) if thread==16 else (.015,.016)
        for k in range(count):
            theta=slot_phase+k*math.tau/count
            center=face+(x*math.cos(theta)+y*math.sin(theta))*pitch_d
            start=center-axis*(length+raised);end=center+axis*cap_thickness
            name=f'{uid}_bolt_{k+1:02}'
            ids.append(_cylinder(scene,owner,name,'flange_bolt',start,end,thread/2000,HARDWARE,
                                 REFERENCE,'project-standard envelope','Stud diameter and bolt circle follow the reference flange.'))
            ids.append(_cylinder(scene,owner,name+'_top','bolt_head',end,end+axis*nut_length,nut_radius,
                                 HARDWARE,REFERENCE,'estimated from the drawing','The nut envelope is represented by a cylinder.'))
            ids.append(_cylinder(scene,owner,name+'_back','bolt_head',start-axis*nut_length,start,nut_radius,
                                 HARDWARE,REFERENCE,'estimated from the drawing','The nut envelope is represented by a cylinder.'))
    row['flange_reference']={'standard':'HG/T 20592-2009 PN16 PL-RF A','supplier_confirmation':False,
        'outer_diameter_mm':outside,'bolt_circle_mm':pitch,'hole_diameter_mm':hole,'bolt_count':count,
        'plate_thickness_mm':plate,'rf_height_mm':2,'raised_face_diameter_mm':rf,'fit_bore_mm':bore,
        'reference_url':REFERENCE_URL}
    row['model_flange_face_component_ids']=plate_face_ids
    return ids


def _ellipse_polygons(inner_axes, outer_axes):
    t=np.linspace(0.,math.tau,33)
    rays=np.c_[np.cos(t),np.sin(t)]
    inner=rays*inner_axes/math.cos(math.pi/32);outer=rays*outer_axes
    return [np.asarray((inner[k],outer[k],outer[k+1],inner[k+1])) for k in range(len(t)-1)]


def _manway(scene,folder,owner,uid,row,face,frame):
    x,y,axis=frame
    # Keep the 400 mm major direction across the vessel front in plan.
    across=np.array([1.,0.,0.])-axis[0]*axis
    across/=np.linalg.norm(across);along=np.cross(axis,across)
    frame=across,along,axis
    ring=_ellipse_polygons((.200,.150),(.254,.195))
    ids=_ring(scene,folder,owner,uid+'_neck','manway_neck',face,frame,.117,-.137,
              _ellipse_polygons((.200,.150),(.223,.171)),SILVER,SECTION,'estimated from the drawing',
              'F10000 catalogue opening is 300 by 400 mm. Neck outline and length are fitted from the drawing.')
    ids+=_ring(scene,folder,owner,uid+'_rim','manway_flange',face,frame,.020,-.020,
               ring,DARK,SECTION+'；'+PLAN,'estimated from the drawing','Closed angled manway face with the physical bolt circle.')
    angles=np.arange(48)*math.tau/48
    lid=np.asarray([face+.251*math.cos(t)*across+.192*math.sin(t)*along for t in angles])
    ids.append(_prism(scene,folder,uid+'_lid','manway_cover',lid,.039*axis,SILVER,SECTION+'；'+PLAN,
                      'estimated from the drawing','Keep the oval cover closed. Its thickness is estimated from the drawing.',owner)['id'])
    for sign in (-1,1):
        q=face+sign*.253*across+.017*axis
        suffix='left' if sign<0 else 'right'
        ids.append(_cylinder(scene,owner,uid+'_clamp_'+suffix,'manway_latch',q-.042*along,q+.042*along,
                             .023,HARDWARE,PLAN,'estimated from the drawing','Clamps shown along both long sides in the plan view.'))
    top=face+.039*axis
    ids.append(_cylinder(scene,owner,uid+'_boss','manway_cover_boss',top,top+.024*axis,.107,DARK,
                         PLAN,'estimated from the drawing','Visible circular center seat on the cover.'))
    for i,t in enumerate(np.arange(8)*math.tau/8):
        p=top+.041*axis+.086*(math.cos(t)*across+math.sin(t)*along)
        ids.append(_cylinder(scene,owner,f'{uid}_head{i+1:02}','bolt_head',p,p+.019*axis,.010,HARDWARE,
                             PLAN,'estimated from the drawing','Visible central cover fastener.'))
    return ids


def build_fittings(scene,folder,owner,rows):
    for row in rows:
        tag=row['tag'];uid=owner+'_port_'+tag
        target=row.get('face_m') or row.get('preview_face_m')
        row['component_ids']=[]
        row['model_face_m']=target
        if tag=='d' or target is None or row.get('normal') is None:
            continue
        face=np.asarray(target,float);x,y,axis=_frame(row['normal']);frame=x,y,axis
        if tag.lower()=='c':
            row['component_ids']=_manway(scene,folder,owner,uid,row,face,frame)
        elif row.get('thread'):
            if row.get('surface_m') is not None:
                start=np.asarray(row['surface_m'])
                radius=.018 if row['thread']=='G3/4' else .014
                row['component_ids']=[_cylinder(scene,owner,uid,'jacket_thread_boss',start,face,
                    radius,DARK,SECTION+'；'+PLAN,'estimated from the drawing','Approximate jacket thread seat without thread detail.')]
        elif row.get('dn') in FLANGES:
            od=FLANGES[row['dn']][0]/2000
            # Use the measured axial direction. Small uncertainty in the shell
            # land is absorbed inside the exterior surface, never at the face.
            start=np.asarray(row['surface_m'],float)
            row['component_ids']=_tube(scene,folder,owner,uid+'_neck',start,face,frame,od,
                                       row['face_basis'],row['evidence_status'])
            row['component_ids']+=_flange(scene,folder,owner,uid,row,face,frame)
