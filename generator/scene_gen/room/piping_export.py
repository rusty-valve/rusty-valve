"""Save an assembled pipe network and its supporting geometry."""
from pathlib import Path
import os
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .evaluation_common import ROOT,require_environment
from .export import nums


def export_piping_model(graph,scene,path):
    require_environment();path=Path(path)
    root=ET.Element('mujoco',model=graph['id']);ET.SubElement(root,'compiler',angle='radian')
    ET.SubElement(root,'option',timestep='.001',integrator='implicitfast',cone='elliptic')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='1800',offheight='1200')
    ET.SubElement(visual,'headlight',ambient='.5 .5 .5',diffuse='.7 .7 .7',specular='.1 .1 .1')
    ET.SubElement(visual,'map',znear='.005',zfar='100')
    assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    ET.SubElement(assets,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.86 .91 .94',rgb2='.96 .97 .98',width='512',height='3072')
    ET.SubElement(world,'light',pos='3 -5 14',dir='0 1 -1',diffuse='.8 .8 .8',castshadow='false')
    for c in scene['components']:
        kw={'name':c['id'],'type':c['shape'],'rgba':nums([*c['color'],c.get('alpha',1)]),'group':str(c.get('group',0))}
        if c['shape']=='mesh':
            name='scene_'+c['id'];ET.SubElement(assets,'mesh',name=name,file=os.path.relpath(c['mesh_path'],path.parent))
            kw.update(mesh=name,pos=nums(c.get('pos',[0,0,0])))
            if c.get('quat'):kw['quat']=nums(c['quat'])
            if not c.get('collision_mesh'):kw.update(contype='0',conaffinity='0')
        elif c['shape']=='cylinder':kw.update(fromto=nums(c['ends'][0]+c['ends'][1]),size=str(c['radius']))
        elif c['shape']=='ellipsoid':kw.update(pos=nums(c['pos']),size=nums(c['size']))
        else:
            low=np.asarray(c['box'][:3]);high=np.asarray(c['box'][3:])
            kw.update(pos=nums((low+high)/2),size=nums((high-low)/2))
        if c.get('annotation_only'):kw.update(contype='0',conaffinity='0')
        ET.SubElement(world,'geom',**kw)
    packs={v['package']:f'valve_asset{i}' for i,v in enumerate(graph['valves'])}
    for p,name in packs.items():ET.SubElement(assets,'model',name=name,file=os.path.relpath(ROOT/p/'model.xml',path.parent))
    for v in graph['valves']:
        frame=ET.SubElement(world,'frame',pos=nums(v['center']),quat=nums(v['quat']))
        ET.SubElement(frame,'attach',model=packs[v['package']],body='valve_base',prefix=v['id']+'_')
    ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return model,data
