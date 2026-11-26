"""Write room geometry, images and valve lists."""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from .routing import ROOT
from .process import write_json

def nums(x):return ' '.join(f'{float(v):.17g}' for v in x)

def export_model(g,s,path):
    path=Path(path);root=ET.Element('mujoco',model=g['id']);ET.SubElement(root,'compiler',angle='radian')
    ET.SubElement(root,'option',timestep='.001',integrator='implicitfast',cone='elliptic')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='1800',offheight='1200')
    ET.SubElement(visual,'headlight',ambient='.5 .5 .5',diffuse='.7 .7 .7',specular='.1 .1 .1')
    ET.SubElement(visual,'map',znear='.005',zfar='100')
    assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    ET.SubElement(assets,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.86 .91 .94',rgb2='.96 .97 .98',width='512',height='3072')
    ET.SubElement(world,'light',pos='3 -5 14',dir='0 1 -1',diffuse='.8 .8 .8',castshadow='false')
    for c in s['components']:
        color=c['color'];kw={'name':c['id'],'type':c['shape'],'rgba':nums([*color,c.get('alpha',1)]),'group':str(c.get('group',0))}
        if c['shape']=='mesh':
            meshname='scene_'+c['id'];ET.SubElement(assets,'mesh',name=meshname,file=os.path.relpath(c['mesh_path'],path.parent))
            kw.update(mesh=meshname)
            if not c.get('collision_mesh'):kw.update(contype='0',conaffinity='0')
        elif c['shape']=='cylinder':kw.update(fromto=nums(c['ends'][0]+c['ends'][1]),size=str(c['radius']))
        elif c['shape']=='ellipsoid':kw.update(pos=nums(c['pos']),size=nums(c['size']))
        else:
            low=np.array(c['box'][:3]);high=np.array(c['box'][3:]);kw.update(pos=nums((low+high)/2),size=nums((high-low)/2))
        if c.get('annotation_only'):kw.update(contype='0',conaffinity='0')
        ET.SubElement(world,'geom',**kw)
    packs={v['package']:f'asset{i}' for i,v in enumerate(g['valves'])}
    for p,name in packs.items():ET.SubElement(assets,'model',name=name,file=os.path.relpath(ROOT/p/'model.xml',path.parent))
    for v in g['valves']:
        frame=ET.SubElement(world,'frame',pos=nums(v['center']),quat=nums(v['quat']))
        ET.SubElement(frame,'attach',model=packs[v['package']],body='valve_base',prefix=v['id']+'_')
    ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    # Native attach includes package joints, equalities and collision geometry; no copied valve code.
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return model,data

def font(size):
    for p in ('/System/Library/Fonts/PingFang.ttc','/System/Library/Fonts/STHeiti Medium.ttc'):
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def render(g,s,model,data,path,azimuth=55):
    width,depth=s.get('footprint',RULES_FOOTPRINT())
    floor_z=s.get('floors',[0]);height=s.get('shell',{}).get('roof_z_m',floor_z[-1]+floor_z[1] if len(floor_z)>1 else depth)
    camera=mujoco.MjvCamera();camera.lookat[:]=[width/2,depth*.35,height*.42];camera.distance=max(width,depth,height)*1.9;camera.azimuth=azimuth;camera.elevation=-24
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    # Make decks translucent in overview only; the saved MJCF retains solid collision floors.
    floors=[]
    for c in s['components']:
        if c['id'].startswith('deck_'):
            gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,c['id']);floors.append(gid);model.geom_rgba[gid,3]=.24
    with mujoco.Renderer(model,width=1800,height=1200) as renderer:
        renderer.update_scene(data,camera=camera,scene_option=opt)
        im=Image.fromarray(renderer.render());cam=renderer.scene.camera[0]
        cam_pos=cam.pos.copy();forward=cam.forward.copy();up=cam.up.copy()
        near=cam.frustum_near;top=cam.frustum_top;bottom=cam.frustum_bottom
    for gid in floors:model.geom_rgba[gid,3]=1
    d=ImageDraw.Draw(im);d.rounded_rectangle([22,20,900,116],radius=12,fill=(241,245,246))
    title=g['id']+' · '+('Building and main equipment' if s.get('stage')=='layout' else g['recipe']['relation'])
    d.text((42,32),title,font=font(32),fill=(28,47,60))
    d.text((42,78),f'{width:g} × {depth:g} m · ground + {len(floor_z)-1} steel platforms · translucent deck reveals the interior',font=font(21),fill=(55,73,82))
    right=np.cross(forward,up)
    for u in g['units'].values():
        if u['kind'] not in ('vessel','pump','condenser','exchanger','filter'):continue
        p=np.array(u['center'])-cam_pos;depth=np.dot(p,forward)
        if depth<=0:continue
        sx=900+np.dot(p,right)/depth*(near/(top-bottom))*1200
        sy=600-np.dot(p,up)/depth*(near/(top-bottom))*1200
        text=u['id'];bb=d.textbbox((sx,sy),text,font=font(25));d.rounded_rectangle([bb[0]-8,bb[1]-5,bb[2]+8,bb[3]+5],radius=4,fill=(236,242,243))
        d.text((sx,sy),text,font=font(25),fill=(19,42,55))
    caption='Building and equipment stage. Equipment, supports, deck and stairs. See connections.json for connectivity.' if s.get('stage')=='layout' else 'Silver-gray vessels, brass exchangers and filters, teal material pipes, orange recirculation and bypass, and an external switchback stair at front right'
    d.rectangle([22,1112,1778,1180],fill=(239,243,245));d.text((40,1124),caption,font=font(23),fill=(31,49,59))
    im.save(path)

def RULES_FOOTPRINT():
    from .process import RULES
    size=RULES['floor_rules']['bay_size_m']
    return [size[0]*RULES['floor_rules']['bays'],size[1]]

def schematic(g,path):
    from .diagram import draw_process
    draw_process(g,path)


def lists(g,folder):
    keys=['id','name','role','dn','operator','level','reachable_from_passage','needs_elevation','stance_free','sweep_clear','eligible','reason','bottom_substitute','package']
    with (folder/'valves.csv').open('w',encoding='utf-8-sig',newline='') as f:
        labels=['ID','Name','Role','Nominal diameter DN','Lever or handwheel','Operating level','Reachable from the aisle','Additional elevation required beyond stair access','Standing space clear','Sweep clear','Eligible for tasks','Reason','temporary vessel bottom ball valve','Model package']
        w=csv.writer(f);w.writerow(labels)
        for v in g['valves']:
            row=[]
            for key in keys:
                value=v[key]
                if key=='operator':value='handwheel' if value=='wheel' else 'lever handle'
                elif isinstance(value,(bool,np.bool_)):value='yes' if value else 'no'
                elif key=='level':value='ground level' if value==0 else f'platform{value}'
                row.append(value)
            w.writerow(row)
    tasks=[]
    for v in g['valves']:
        for action in ('open','close'):
            tasks.append({'id':v['id']+'_'+action,'valve':v['id'],'name':v['name'],'action':action,
                'eligible':v['eligible'],'needs_elevation':v['needs_elevation'],'reason':v['reason'],
                'initial_openness':0 if action=='open' else 1,'target_openness':1 if action=='open' else 0})
    write_json(folder/'all_valve_questions.json',tasks)
    write_json(folder/'tasks.json',[t for t in tasks if t['eligible']])
    write_json(folder/'process.json',g)
