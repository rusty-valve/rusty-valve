"""Generate a standalone passive workstation, before any robot is selected.

The construction starts from geometry.build_scene (a passive valve primitive),
not from a robot-containing scene. The returned MJCF has one hinge and zero
actuators, and is also the composition input for an independent robot adapter.
"""
from __future__ import annotations

import copy
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .geometry import build_scene, _s, _camera
from .h0_station import load_workstation


def _prefix_names(element, prefix):
    for node in element.iter():
        if node.get('name'):
            node.set('name',f'{prefix}_{node.get("name")}')


def build_workstation_tree(description='A1'):
    spec = load_workstation(description)
    d = spec.to_dict(); v = d['valve']; p = d['physics']
    params = copy.deepcopy(v['geometry'])
    params['lever_reach_m'] = spec.handle_length_m
    params['lever_highest_m'] = spec.handle_plane_offset_m+params['lever_mid_thickness_m']/2
    root = ET.fromstring(build_scene('ball','horizontal_top',
        close_angle_deg=math.degrees(spec.q_close_rad-spec.q_open_rad),position=(0.,0.,0.),
        timestep=p['timestep_s'],gravity=p['gravity_world_m_s2'],
        frictionloss=v['resistance']['breakaway_Nm']*v['resistance']['scale'],params=params))
    root.set('model',f'H0_{spec.case}_independent_workstation')
    option = root.find('option')
    option.set('solver',p['solver']);option.set('iterations',str(p['iterations']))
    option.set('ls_iterations',str(p['ls_iterations']))
    # Keep environment defaults scoped so a future robot retains its own limits,
    # inertias, friction and collision definitions during composition.
    default = root.find('default')
    children = list(default)
    for node in children: default.remove(node)
    scoped = ET.SubElement(default,'default',{'class':'valve_task'})
    scoped.extend(children)
    world = root.find('worldbody')
    fixed = copy.deepcopy(world.find("body[@name='valve_mount']"))
    moving = copy.deepcopy(world.find("body[@name='valve_moving']"))
    for node in list(world):
        if node.tag not in ('light',) and node.get('name') != 'floor':
            world.remove(node)
    world.find("geom[@name='floor']").set('class','valve_task')
    # MuJoCo plane collision is already infinite; zero X/Y extends only drawing.
    world.find("geom[@name='floor']").set('size','0 0 .1')
    root.find('asset/material').set('reflectance','0')
    stem=spec.stem_axis_world; pipe=spec.pipe_axis_world
    R = np.column_stack((pipe,np.cross(stem,pipe),stem))
    quat = np.zeros(4);mujoco.mju_mat2Quat(quat,R.reshape(-1))
    # Installation frame retains the pipe direction. A separate local phase
    # sets the handgrip's q-open radial without changing its hinge or pipe.
    radial_local = R.T@spec.radial_zero_world
    phase = math.atan2(radial_local[1],radial_local[0])-v['closing_direction_about_stem']*spec.q_open_rad
    installation = ET.SubElement(world,'body',name=spec.installation_body_name,
        pos=_s(spec.mount_origin_world_m),childclass='valve_task')
    for body in (fixed,moving):
        _prefix_names(body,spec.case)
        body.set('pos','0 0 0');body.set('quat',_s(quat))
    hinge=moving.find('joint')
    hinge.set('axis',_s((0.,0.,v['closing_direction_about_stem'])))
    hinge.set('range',_s((spec.q_open_rad,spec.q_close_rad)))
    hinge.set('solreffriction',_s(v['resistance']['solref']))
    hinge.set('solimpfriction',_s(v['resistance']['solimp']))
    phase_body=ET.SubElement(moving,'body',name=f'{spec.case}_handle_phase',
        quat=_s((math.cos(phase/2),0.,0.,math.sin(phase/2))))
    for node in list(moving):
        if node.tag in ('geom','site'):
            moving.remove(node);phase_body.append(node)
    inertia=moving.find('inertial')
    iq=np.zeros(4);mujoco.mju_mulQuat(iq,np.array([math.cos(phase/2),0.,0.,math.sin(phase/2)]),
                                    np.fromstring(inertia.get('quat','1 0 0 0'),sep=' '))
    ip=np.fromstring(inertia.get('pos'),sep=' ')
    c,s=math.cos(phase),math.sin(phase)
    inertia.set('pos',_s((c*ip[0]-s*ip[1],s*ip[0]+c*ip[1],ip[2])))
    inertia.set('quat',_s(iq))
    # Same handle-axis and visual-label sites for every participant.
    ET.SubElement(phase_body,'site',name=spec.handle_tip_site_name,
        pos=_s((spec.handle_length_m,0.,spec.handle_plane_offset_m)),size='.001',rgba='0 0 0 0',group='4')
    ET.SubElement(phase_body,'site',name=f'{spec.case}_handle_center',
        pos=_s((spec.handle_length_m*.5,0.,spec.handle_plane_offset_m)),size='.001',rgba='0 0 0 0',group='4')
    ET.SubElement(moving,'site',name=f'{spec.case}_stem_axis_point',
        pos=_s((0.,0.,spec.handle_plane_offset_m)),size='.001',rgba='0 0 0 0',group='4')
    # Pipe dimensions are explicit environment data, not a hidden robot setting.
    half=params['body_length_m']/2;plen=d['pipe']['length_m']/2;radius=d['pipe']['outside_diameter_m']/2
    for side in (-1,1):
        pipe_geom=fixed.find(f"geom[@name='{spec.case}_pipe_{side}']")
        pipe_geom.set('fromto',_s((side*half,0.,0.,side*plen,0.,0.)))
        pipe_geom.set('size',_s((radius,)))
        shadow=fixed.find(f"geom[@name='{spec.case}_pipe_end_shadow_{side}']")
        shadow.set('fromto',_s((side*(plen+.0001),0.,0.,side*(plen+.0003),0.,0.)))
    installation.extend((fixed,moving))
    for g in d['supports']:
        attrs={key:(_s(value) if isinstance(value,list) else str(value))for key,value in g.items()}
        attrs['name']=f"{spec.case}_{g['name']}";attrs['class']='valve_task'
        ET.SubElement(world,'geom',attrs)
    m=spec.mount_origin_world_m
    _camera(world,'environment_overview',m+np.array([-.8,-.8,.65]),m+np.array([0.,0.,.05]),45)
    _camera(world,f'{spec.case}_detail',m+np.array([-.70,-.66,.48]),m+np.array([-.015,0.,.045]),36)
    keyframe=root.find('keyframe');keyframe.clear()
    ET.SubElement(keyframe,'key',name='environment_open',qpos=_s((spec.q_open_rad,)))
    ET.SubElement(keyframe,'key',name='environment_closed',qpos=_s((spec.q_close_rad,)))
    ET.indent(root)
    return root


def build_workstation(description='A1'):
    """Return standalone environment XML; loadable without any robot adapter."""
    return ET.tostring(build_workstation_tree(description),encoding='unicode')


def environment_names(description='A1'):
    spec=load_workstation(description)
    return dict(station=spec.case,valve_joint=spec.valve_joint_name,
        installation_body=spec.installation_body_name,
        moving_body=spec.valve_moving_body_name,mount_body=spec.valve_mount_body_name,
        handle_tip_site=spec.handle_tip_site_name,handle_center_site=f'{spec.case}_handle_center',
        stem_axis_point_site=f'{spec.case}_stem_axis_point')


def environment_geom_groups(model, description='A1'):
    """Resolve environment-side categories after composition with any robot."""
    spec=load_workstation(description)
    moving={int(model.body(spec.valve_moving_body_name).id)}
    for bid in range(1,model.nbody):
        if int(model.body_parentid[bid]) in moving:moving.add(bid)
    handle=[int(i)for i in range(model.ngeom)if int(model.geom_bodyid[i])in moving]
    fixtures=[int(i)for i in range(model.ngeom)
              if (model.geom(i).name or '').startswith(f'{spec.case}_') and i not in handle]
    return dict(handle_geom_ids=handle,fixture_geom_ids=fixtures,
                floor_geom_ids=[int(model.geom('floor').id)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--station',default='A1',help='case name or pure workstation JSON path')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    xml=build_workstation(args.station)
    model=mujoco.MjModel.from_xml_string(xml)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(xml,encoding='utf-8')
    print(json.dumps(dict(output=str(args.output.resolve()),nq=model.nq,
                         actuator_count=model.nu,standalone=True),ensure_ascii=False))


if __name__=='__main__':main()
