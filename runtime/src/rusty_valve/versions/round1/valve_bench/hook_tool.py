"""User-corrected S tool on BOTH wrists; static geometry, no motion script.

Upper concave flat flanks push, lower concave flat flanks hook. Three mitered
extruded pieces reproduce the zigzag, plus its central shaft. No return lip.
Source OpenArm assets and the seven arm actuators on each side are unchanged.
"""
from __future__ import annotations
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

CONFIG = Path(__file__).parent / 'configs' / 's_hook.json'
EE_SITE = 'ee_control'


def _s(values):
    return ' '.join(f'{float(v):.12g}' for v in values)


def tool_geometry():
    cfg=json.loads(CONFIG.read_text())
    points=np.asarray(cfg['profile_yz_mm'],float)
    tangent=np.diff(points,axis=0)
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    normals=np.column_stack((-tangent[:,1],tangent[:,0]))
    offsets=[];half=cfg['plate_thickness_mm']/2
    for i in range(len(points)):
        if i==0:offset=half*normals[0]
        elif i==len(points)-1:offset=half*normals[-1]
        else:
            bisector=normals[i-1]+normals[i]
            bisector/=np.linalg.norm(bisector)
            offset=half*bisector/np.dot(bisector,normals[i])
        offsets.append(offset)
    offsets=np.asarray(offsets)
    shift=cfg['total_forward_extent_mm']-np.max(points[:,1]+np.abs(offsets[:,1]))
    points[:,1]+=shift
    plus=points+offsets;minus=points-offsets
    quads=[np.array([plus[i],plus[i+1],minus[i+1],minus[i]])*.001 for i in range(len(points)-1)]
    return cfg,points*.001,plus*.001,minus*.001,quads


def _vertices(quad,width):
    return np.array([[x,y,z] for x in (-width/2,width/2) for y,z in quad])


def _paint_face(body,side,label,index,a,b,width,coverage=.65):
    # Only a coloured strip on each flat face. It has no collision or mass.
    # Coverage is visual only: extending paint to a bend does not enlarge the
    # contact-classification surface or alter the physical plate underneath.
    a,b=np.asarray(a),np.asarray(b)
    if not np.isfinite(coverage) or not 0.<coverage<=1.:
        raise ValueError('Paint coverage must be in (0,1]')
    centre=(a+b)/2;delta=(b-a)*coverage;length=np.linalg.norm(delta)
    z=np.array([0.,*delta/length]);x=np.array([1.,0.,0.]);y=np.cross(z,x)
    ET.SubElement(body,'geom',name=f'{side}_{label}_face_{index}',type='box',
        pos=_s((0.,*centre)),size=_s((width/2,.00004,length/2)),
        xyaxes=_s((*x,*y)),material=f's_hook_{label}',
        contype='0',conaffinity='0',density='0',group='0')
    # Edge paint on the visible side makes the same flat flank legible even
    # when its actual face is turned away from the inspection camera.
    lo=centre-delta/2;hi=centre+delta/2
    ET.SubElement(body,'geom',name=f'{side}_{label}_face_edge_{index}',type='capsule',
        fromto=_s((width/2+.00005,*lo,width/2+.00005,*hi)),size='.0003',
        material=f's_hook_{label}',contype='0',conaffinity='0',density='0',group='0')


def make_hook_body(robot,side,hand):
    cfg,points,plus,minus,quads=tool_geometry()
    asset=robot.find('asset')
    if asset.find("material[@name='s_hook_metal']") is None:
        for name,rgba in [('metal','.65 .69 .73 1'),('push','.90 .43 .06 1'),('pull','.04 .43 .72 1')]:
            ET.SubElement(asset,'material',name=f's_hook_{name}',rgba=rgba,specular='.15')
    attrs=dict(name=f'openarm_{side}_hook',pos=hand.get('pos','0 0 .1001'))
    if hand.get('quat') is not None:attrs['quat']=hand.get('quat')
    body=ET.Element('body',attrs)
    width=cfg['out_of_plane_width_mm']*.001
    thick=cfg['plate_thickness_mm']*.001
    junction=points[cfg['shaft_junction_index']]
    if cfg.get('through_holes'):
        from .holed_tool import add_holed_parts
        add_holed_parts(asset,body,side,cfg,points,plus,minus,quads)
    else:
        ET.SubElement(body,'geom',name=f'{side}_hook_shaft',type='box',
            pos=_s((0.,junction[0],junction[1]/2)),size=_s((width/2,thick/2,junction[1]/2)),
            **{'class':'collision','group':'0','material':'s_hook_metal','density':str(cfg['density_kg_m3'])})
        for i,quad in enumerate(quads):
            name=f'{side}_hook_bend_{i}'
            ET.SubElement(asset,'mesh',name=name,vertex=_s(_vertices(quad,width).flatten()))
            ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,
                **{'class':'collision','group':'0','material':'s_hook_metal','density':str(cfg['density_kg_m3'])})
        for label,key in [('push','push_flanks'),('pull','pull_flanks')]:
            for i,edge in cfg[key]:
                curve=plus if edge=='left' else minus
                _paint_face(body,side,label,i,curve[i],curve[i+1],width)
    tcp=(0.,0.,.09)
    ET.SubElement(body,'site',name=EE_SITE if side=='right' else 'left_ee_control',
                  pos=_s(tcp),size='.001',rgba='0 0 0 0',group='4')
    # Named markers describe recesses only; they are not validated grasp targets.
    for label,local in [('push_recess',(0.,0.,.091)),('pull_recess',(0.,-.016,.090))]:
        ET.SubElement(body,'site',name=f'{side}_hook_{label}',pos=_s(local),
                      size='.001',rgba='0 0 0 0',group='4')
    return body



def make_wrist_mount(side, hand):
    """Solid steel spacer on a fixed sibling body, with analytic inertia once.

    The original hand origin is 100.1mm beyond joint7, while the upstream wrist
    collision envelope ends at 94.5mm. This fills that interval without moving
    the S-tool frame, TCP or working faces. One geom is visible and collidable;
    the explicit inertia belongs to this new body, never the stock wrist body.
    """
    cfg=json.loads(CONFIG.read_text())['wrist_mount']
    radius=float(cfg['radius_mm'])*.001
    length=float(cfg['length_mm'])*.001
    density=float(cfg['density_kg_m3'])
    if min(radius,length,density)<=0:raise ValueError('Wrist mount dimensions and density must be positive')
    origin=np.fromstring(hand.get('pos','0 0 .1001'),sep=' ')
    if origin.shape!=(3,) or not np.allclose(origin[:2],0.):
        raise ValueError('Wrist mount supports the declared axial OpenArm interface only')
    quat=np.fromstring(hand.get('quat','1 0 0 0'),sep=' ')
    if quat.shape!=(4,) or not np.allclose(quat,[1.,0.,0.,0.]):
        raise ValueError('Wrist mount needs the unchanged OpenArm axial hand orientation')
    origin=origin.copy();origin[2]-=length/2
    mass=density*np.pi*radius**2*length
    transverse=mass*(3*radius**2+length**2)/12
    axial=mass*radius**2/2
    body=ET.Element('body',name=f'openarm_{side}_tool_mount',pos=_s(origin))
    ET.SubElement(body,'inertial',pos='0 0 0',mass=f'{mass:.12g}',
                  diaginertia=_s((transverse,transverse,axial)))
    ET.SubElement(body,'geom',name=f'{side}_tool_mount',type='cylinder',
                  size=_s((radius,length/2)),pos='0 0 0',
                  **{'class':'collision','group':'0','material':'s_hook_metal',
                     'density':str(density),'contype':'1','conaffinity':'1'})
    return body


def install_hook(robot):
    """Replace both original hands in the composed XML, preserving valve qpos."""
    world=robot.find('worldbody')
    hands=[]
    for side in ('left','right'):
        wrist=world.find(f".//body[@name='openarm_{side}_link7']")
        hand=wrist.find(f"body[@name='openarm_{side}_hand']") if wrist is not None else None
        if hand is None:raise ValueError(f'Missing original {side} hand')
        hands.append((side,wrist,hand))
    removed={tag:{node.get('name') for _,_,hand in hands for node in hand.iter(tag)}
             for tag in ('body','joint','site','geom')}
    remove_qpos=set();remove_qvel=set();nq=nv=0
    for joint in world.iter('joint'):
        qn,vn={'free':(7,6),'ball':(4,3)}.get(joint.get('type','hinge'),(1,1))
        if joint.get('name') in removed['joint']:
            remove_qpos.update(range(nq,nq+qn));remove_qvel.update(range(nv,nv+vn))
        nq+=qn;nv+=vn
    actuators=robot.find('actuator')
    remove_ctrl={i for i,a in enumerate(actuators) if a.get('joint') in removed['joint']}
    for key in robot.findall('keyframe/key'):
        for attr,indices in [('qpos',remove_qpos),('qvel',remove_qvel),('ctrl',remove_ctrl)]:
            if attr in key.attrib:key.set(attr,' '.join(v for i,v in enumerate(key.get(attr).split()) if i not in indices))
    removed_tendons=set();tendons=robot.find('tendon')
    if tendons is not None:
        for tendon in list(tendons):
            if any(e.get('joint') in removed['joint'] or e.get('site') in removed['site'] for e in tendon.iter()):
                removed_tendons.add(tendon.get('name'));tendons.remove(tendon)
    for i,a in reversed(list(enumerate(actuators))):
        if i in remove_ctrl:actuators.remove(a)
    references={'body1':removed['body'],'body2':removed['body'],
                'joint1':removed['joint'],'joint2':removed['joint'],
                'tendon1':removed_tendons,'tendon2':removed_tendons,
                'geom1':removed['geom'],'geom2':removed['geom']}
    for section in ('contact','equality'):
        parent=robot.find(section)
        if parent is not None:
            for element in list(parent):
                if any(element.get(a) in names for a,names in references.items()):parent.remove(element)
    for side,wrist,hand in hands:
        wrist.remove(hand)
        wrist.append(make_hook_body(robot,side,hand))
        wrist.append(make_wrist_mount(side,hand))
    cfg,_,_,_,_=tool_geometry()
    return {'version':cfg['version'],'sides':['left','right'],'configuration':cfg,
            'mass_note':('Perforated tool uses disjoint convex cells with true empty bores and trimmed shaft junction; mass/inertia follow remaining steel volume. ' if cfg.get('through_holes') else 'Solid legacy tool sums simple-piece masses; small junction overlap remains. ')+'Each fixed wrist mount has one analytic cylinder inertia; paint is massless.'}
