"""WP2b scene adapter for the upstream OpenArm v1 bimanual model.

The upstream arms, limits, inertias and arm actuators are untouched. The
fixed upstream torso is elevated so its shoulder centre is at shoulder_height.
World axes follow upstream: +X forward, +Y left, +Z up. The stock variant keeps
both official hands; the hook variant replaces both with the corrected S part.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .geometry import build_scene, _camera, _s
from .resistance import ValveResistance

ASSET_ROOT = Path(__file__).resolve().parent / 'assets' / 'openarm_v1'
SOURCE_MODEL = ASSET_ROOT / 'openarm_bimanual.xml'
EE_SITE = 'ee_control'
SOURCE_COMMIT = 'a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c'


def robot_names(tool='stock'):
    if tool not in ('stock','hook'):
        raise ValueError('tool must be stock or hook')
    arms = {s: tuple(f'openarm_{s}_joint{i}' for i in range(1,8)) for s in ('left','right')}
    fingers = {s: tuple(f'openarm_{s}_finger_joint{i}' for i in (1,2)) for s in ('left','right')}
    motors = {s: tuple(f'{s}_joint{i}_ctrl' for i in range(1,8)) for s in ('left','right')}
    return dict(active_joint_names=arms['right'],
                robot_joint_names=arms['left']+(fingers['left'] if tool=='stock' else ())+arms['right']+(fingers['right'] if tool=='stock' else ()),
                actuator_names=motors['left']+(('left_finger1_ctrl',) if tool=='stock' else ())+motors['right']+(('right_finger1_ctrl',) if tool=='stock' else ()),
                active_actuator_names=motors['right'],
                gripper_actuator_name='right_finger1_ctrl' if tool=='stock' else None,
                root_body_names=('openarm_body_link0',),
                active_root_body_name='openarm_right_link0',
                idle_joint_names=arms['left'],
                idle_actuator_names=motors['left'],
                EE_SITE=EE_SITE)


def build_bimanual_scene(valve='ball', layout='horizontal_top',
                         valve_position=(.35,-.28,1.05), shoulder_height=1.4,
                         resistance_scale=1., close_angle_deg=90., tool='stock',
                         installation_yaw_deg=-90.):
    robot_names(tool)
    if not .8 <= float(shoulder_height) <= 2.:
        raise ValueError('shoulder_height must be between 0.8 and 2 metres')
    resistance=ValveResistance.from_config(valve,scale=resistance_scale)
    task=ET.fromstring(build_scene(valve,layout,close_angle_deg,position=valve_position,
                                  frictionloss=resistance.breakaway_Nm*resistance_scale))
    robot=ET.parse(SOURCE_MODEL).getroot()
    robot.set('model',f'openarm_bimanual_{valve}_{layout}_{tool}')
    compiler=robot.find('compiler'); compiler.attrib.pop('meshdir',None);compiler.set('autolimits','true')
    for mesh in robot.findall('asset/mesh'):
        mesh.set('file',str(ASSET_ROOT/'meshes'/mesh.get('file')))
    scoped=copy.deepcopy(task.find('default'));scoped.set('class','valve_task')
    robot.find('default').append(scoped)
    robot.find('option').attrib.update(task.find('option').attrib)
    robot.append(copy.deepcopy(task.find('visual')))
    robot.find('visual/quality').set('offsamples','1')
    robot.find('visual/quality').set('shadowsize','1024')
    robot.find('asset').extend(copy.deepcopy(list(task.find('asset'))))
    world=robot.find('worldbody');base=world.find("body[@name='openarm_body_link0']")
    torso_z=float(shoulder_height)-.697
    base.set('pos',_s((0,0,torso_z)))
    tcp=base.find(".//body[@name='openarm_right_hand_tcp']")
    ET.SubElement(tcp,'site',name=EE_SITE,pos='0 0 0',size='.003',rgba='0 0 0 0',group='4')
    # Rotate the complete installation around the valve centre. Historical task
    # assets used -Y as the viewing/front direction; the humanoid faces +X.
    # A -90-degree yaw places a horizontal_front valve stem toward the robot.
    # Both moving/fixed valve bodies AND ground supports receive this transform.
    yaw=math.radians(float(installation_yaw_deg))
    installation=ET.SubElement(world,'body',name='valve_installation',
        pos=_s(valve_position),quat=_s((math.cos(yaw/2),0,0,math.sin(yaw/2))),
        childclass='valve_task')
    for original in task.find('worldbody'):
        element=copy.deepcopy(original)
        if element.tag=='body':element.set('childclass','valve_task')
        elif element.tag=='geom':element.set('class','valve_task')
        if element.tag=='light' or element.get('name')=='floor':
            world.append(element)
            continue
        # Convert upstream world positions into installation-relative coordinates.
        if 'pos' in element.attrib:
            values=[float(x) for x in element.get('pos').split()]
            element.set('pos',_s(v-c for v,c in zip(values,valve_position)))
        if 'fromto' in element.attrib:
            values=[float(x) for x in element.get('fromto').split()]
            element.set('fromto',_s(v-valve_position[i%3] for i,v in enumerate(values)))
        installation.append(element)
    # A fixed support below the upstream torso, not a free body or extra actuator.
    ET.SubElement(world,'geom',name='torso_pedestal',type='box',**{'class':'valve_task',
                 'pos':_s((0,0,torso_z/2)),'size':_s((.075,.09,torso_z/2)),
                 'rgba':'.17 .20 .23 1'})
    ET.SubElement(world,'geom',name='torso_foot',type='box',**{'class':'valve_task',
                 'pos':'0 0 .018','size':'.19 .22 .018','rgba':'.17 .20 .23 1'})
    for key in robot.findall('keyframe/key'):
        key.set('qpos',key.get('qpos')+' 0')
    target=(.18,-.04,shoulder_height-.19)
    _camera(world,'robot_overview',(1.1,-1.15,shoulder_height+.38),target,47)
    vp=tuple(valve_position)
    _camera(world,'robot_contact',(vp[0]+.40,vp[1]-.48,vp[2]+.42),
            (vp[0]+.07,vp[1],vp[2]+.07),46)
    if tool=='hook':
        from .hook_tool import install_hook
        install_hook(robot)
    ET.indent(robot)
    return ET.tostring(robot,encoding='unicode')
