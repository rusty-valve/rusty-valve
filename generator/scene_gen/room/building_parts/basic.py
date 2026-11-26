"""Build equipment and pipe shapes used in compact rooms."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from ..process import Builder, RULES, validate, write_json
from ..layout import add_box, add_pipe, quat_from_axes
from ..routing import ROOT, package, reserved_box
from . import utilities as utility
from .. import building
from . import spatial
from ..export import export_model, lists, font
from ..checks import valve_checks, intersects
from .installation_checks import check as installation_check

DEFAULT = Path('generated_room')
STEEL=(.24,.31,.36); SILVER=(.65,.70,.71); FLANGE=(.45,.51,.53)
COLORS={key:utility.color(key) for key in ('nitrogen','offgas','feed','material','recycle','boundary')}

class Mesh:
    def __init__(self): self.v=[]; self.f=[]; self.solids=[]
    def box(self, lo, hi):
        n=len(self.v)
        self.v.extend([[x,y,z] for x,y,z in ((lo[0],lo[1],lo[2]),(hi[0],lo[1],lo[2]),(hi[0],hi[1],lo[2]),(lo[0],hi[1],lo[2]),(lo[0],lo[1],hi[2]),(hi[0],lo[1],hi[2]),(hi[0],hi[1],hi[2]),(lo[0],hi[1],hi[2]))])
        self.f.extend([[n+i for i in f] for f in ((0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7))])
    def revolve(self, center, profile, count=48, scale_y=1):
        self.solids.append((center,profile,scale_y))
        n=len(self.v)
        for r,z in profile:
            for i in range(count):
                a=math.tau*i/count
                self.v.append([center[0]+r*math.cos(a),center[1]+r*math.sin(a)*scale_y,z])
        for k in range(len(profile)-1):
            for i in range(count):
                a=n+k*count+i;b=n+k*count+(i+1)%count;c=b+count;d=a+count
                self.f.extend([[a,b,c],[a,c,d]])
        # Closed planar ends; zero-radius profiles use tiny radii to avoid degenerate faces.
        for k, reverse in ((0,True),(len(profile)-1,False)):
            ci=len(self.v);self.v.append([center[0],center[1],profile[k][1]])
            for i in range(count):
                a=n+k*count+i;b=n+k*count+(i+1)%count
                self.f.append([ci,b,a] if reverse else [ci,a,b])
    def save(self,s,folder,name,kind,color,owner=None):
        path=folder/'meshes'/f'{name}.obj';path.parent.mkdir(exist_ok=True)
        with path.open('w') as f:
            for p in self.v:f.write('v '+' '.join(f'{x:.6f}' for x in p)+'\n')
            for face in self.f:f.write('f '+' '.join(str(x+1) for x in face)+'\n')
        a=np.asarray(self.v)
        c={'id':name,'kind':kind,'shape':'mesh','mesh_path':str(path),'box':[*a.min(0),*a.max(0)],'color':color,'owner':owner,'triangles':len(self.f)}
        s['components'].append(c)
        if kind=='grating':c['visual_only']=True
        if kind in ('nozzle','head'):
            c['visual_only']=True
            for j,(center,profile,sy) in enumerate(self.solids):
                for k,((r0,z0),(r1,z1)) in enumerate(zip(profile,profile[1:])):
                    if abs(z1-z0)<1e-8:continue
                    r=max(r0,r1);name2=f'collision_{name}_{j}_{k}'
                    if sy==1:proxy=add_pipe(s,name2,[*center,z0],[*center,z1],r,kind,color,owner)
                    else:proxy=add_box(s,name2,[center[0]-r,center[1]-r*sy,min(z0,z1),center[0]+r,center[1]+r*sy,max(z0,z1)],kind,color,owner)
                    proxy['group']=3
        return c


def grating(mesh,rect,z):
    x0,y0,x1,y1=rect
    for x in np.arange(x0,x1-.001,.03):mesh.box([x,y0,z-.03],[min(x+.003,x1),y1,z])
    for y in np.arange(y0,y1-.001,.10):mesh.box([x0,y,z-.005],[x1,min(y+.005,y1),z])
    for x in (x0,x1-.005):mesh.box([x,y0,z-.03],[x+.005,y1,z])
    for y in (y0,y1-.005):mesh.box([x0,y,z-.03],[x1,y+.005,z])


def vessel_placements(spec):
    rows=deepcopy(spec.get('vessels',[{'id':'A','center_xy_m':spec['scene_choices']['vessel_center_xy_m']}]))
    if any('center_xy_m' not in r for r in rows):
        opening=spec['drawing']['d2_mm']/1000+.30;gap=(spec['footprint_m'][0]-len(rows)*opening)/(len(rows)+1)
        if gap<RULES['geometry_screen']['walk_width_m']+.05:raise ValueError('Equipment openings leave less than the required aisle width')
        for i,r in enumerate(rows):r['center_xy_m']=[gap+opening/2+i*(gap+opening),spec['scene_choices']['vessel_row_y_m']]
    return rows

def pump_placements(spec):
    return spec.get('pumps',[{'id':'P1','source_vessel':'A','center_xy_m':[4.7,2.3]}])


def structure(s,folder,spec):
    w,d=spec['footprint_m'];z=spec['platform_z_m'];cx,cy=vessel_placements(spec)[0]['center_xy_m']
    hx0,hx1=cx-.8,cx+.8;hy0,hy1=cy-.8,cy+.8
    equipment_holes=[[v['center_xy_m'][0]-.8,v['center_xy_m'][1]-.8,v['center_xy_m'][0]+.8,v['center_xy_m'][1]+.8] for v in vessel_placements(spec)]
    s['holes'][1]=equipment_holes.copy();s['equipment_opening_count']=len(equipment_holes)
    add_box(s,'ground',[-1.8,-.6,-.12,w+.5,d+.5,0],'ground',(.76,.77,.75))
    def beam(name,a,b):
        # Three rectangles make an I section; flange top supports the grating.
        ax=int(np.argmax(abs(np.array(b)-a)));lo=np.minimum(a,b);hi=np.maximum(a,b)
        if ax==2:
            bounds=np.r_[lo-[.11,.105,0],hi+[.11,.105,0]]
            list(building.i_member(s,name,bounds,ax,'column',flange=.020,web=.012))
        else:
            cross=1-ax
            lo[cross]-=.09;hi[cross]+=.09;lo[2]=a[2]-.26;hi[2]=a[2]-.03
            list(building.i_member(s,name,np.r_[lo,hi],ax))
    for i,(x,y) in enumerate(((0,0),(w,0),(0,d),(w,d))):
        beam(f'column{i}',[x,y,0],[x,y,z-.03])
        add_box(s,f'base{i}',[x-.2,y-.2,0,x+.2,y+.2,.035],'column',FLANGE)
    s['frame_spans']=[]
    for i,(a,b) in enumerate((([0,0,z],[w,0,z]),([0,d,z],[w,d,z]),([0,0,z],[0,d,z]),([w,0,z],[w,d,z]))):
        beam(f'perimeter{i}',a,b)
        s['frame_spans'].append({'id':f'perimeter{i}','column_tops':[a,b],'beam_depth_m':.26})
    for i,x in enumerate(sorted({w*.1916666667,w*.825,*[h[k] for h in equipment_holes for k in (0,2)]})):
        for j,r in enumerate(building.subtract([x-.001,0,x+.001,d],[[h[0]+.001,h[1],h[2]-.001,h[3]] for h in equipment_holes],min_size=.0001)):
            beam(f'joist{i}_{j}',[x,r[1],z],[x,r[3],z])
    for i,h in enumerate(equipment_holes):
        for j,y in enumerate((h[1],h[3])):beam(f'hole_trim{i}_{j}',[h[0],y,z],[h[2],y,z])
    rects=[[0,0,hx0,d],[hx1,0,w,d],[hx0,0,hx1,hy0],[hx0,hy1,hx1,d]]
    deck=Mesh()
    for i,rect in enumerate(rects):
        grating(deck,rect,z)
        c=add_box(s,f'walk_surface{i}',[rect[0],rect[1],z-.03,rect[2],rect[3],z],'walk_surface',SILVER);c['group']=3
    deck.save(s,folder,'actual_grating','grating',(.53,.58,.59))
    # Opening curb is below the reactor's support lugs.
    for j,(hx0,hy0,hx1,hy1) in enumerate(equipment_holes):
        for i,rect in enumerate(([hx0-.025,hy0-.025,hx1+.025,hy0],[hx0-.025,hy1,hx1+.025,hy1+.025],[hx0-.025,hy0,hx0,hy1],[hx1,hy0,hx1+.025,hy1])):
            add_box(s,f'hole_curb{j}_{i}',[rect[0],rect[1],z,rect[2],rect[3],z+.06],'curb',STEEL)
    def rail(name,a,b,base=z):
        for j,h in enumerate((.55,1.10)):add_pipe(s,name+f'_bar{j}',[*a,base+h],[*b,base+h],.021,'rail',(.58,.62,.61))
        length=math.dist(a,b)
        for i,t in enumerate(np.linspace(0,1,math.ceil(length/1.25)+1)):
            p=np.array(a)*(1-t)+np.array(b)*t
            add_pipe(s,name+f'_post{i}',[*p,base-.03],[*p,base+1.10],.022,'rail',(.58,.62,.61))
        lo=np.minimum(a,b)-.015;hi=np.maximum(a,b)+.015
        add_box(s,name+'_toe',[*lo,base,*hi,base+.10],'curb',(.39,.44,.44))
    rail('front',[0,0],[w,0]);rail('back',[0,d],[w,d]);rail('right',[w,0],[w,d]);rail('left',[0,0],[0,4.76])
    # One straight external stair. All tread meshes have real open spaces.
    steps=18;rise=z/steps;run=.28;x0,x1=-1.30,-.20;start=0
    stairmesh=Mesh()
    for i in range(steps):
        yy=start+i*run;zz=(i+1)*rise
        grating(stairmesh,[x0,yy,x1,yy+run],zz)
        c=add_box(s,f'stair_collision{i}',[x0,yy,zz-.035,x1,yy+run,zz],'step',STEEL);c['group']=3
    grating(stairmesh,[x0,4.76,0,5.96],z)
    c=add_box(s,'landing_collision',[x0,4.76,z-.03,0,5.96,z],'landing',STEEL);c['group']=3
    stairmesh.save(s,folder,'stair_grating','grating',(.53,.58,.59))
    for x in (x0-.04,x1+.04):
        add_pipe(s,'stringer'+str(x),[x,0,.05],[x,4.9,z-.08],.055,'beam',STEEL)
        for h in (.55,1.1):add_pipe(s,f'stair_rail{x}_{h}',[x,0,rise+h],[x,4.76,z+h],.023,'rail',(.58,.62,.61))
        for i in (0,6,12,17):
            yy=i*run;zz=(i+1)*rise
            add_pipe(s,f'stair_post{x}_{i}',[x,yy,zz-.1],[x,yy,zz+1.1],.022,'rail',(.58,.62,.61))
    for i,(x,y) in enumerate(((x0,4.76),(x0,5.96))):
        beam('landing_leg'+str(i),[x,y,0],[x,y,z-.03])
    beam('landing_outer',[x0,4.76,z],[x0,5.96,z]);beam('landing_front',[x0,4.76,z],[0,4.76,z]);beam('landing_back',[x0,5.96,z],[0,5.96,z])
    rail('landing_side',[x0,4.76],[x0,5.96]);rail('landing_end',[x0,5.96],[0,5.96])
    s['stairs']=[{'rise':rise,'run':run,'width':1.1,'steps':steps,'landing':[x0,4.76,0,5.96],'headroom':'open above'}]
    s['navigation_entries']={'0':[.6,.6],'1':[.5,5.35]}


def graph(spec):
    vessels=vessel_placements(spec);pumps=pump_placements(spec);transfers=spec.get('transfers',[])
    b=Builder({'id':spec['id'],'seed':spec['seed'],'relation':'one_to_one' if transfers else 'self_recycle','vessels':vessels,'transfers':transfers})
    for v in vessels:
        b.unit(v['id'],'vessel',orientation='vertical',reactor=True,pressurizable=True,level=1)
        b.port(v['id'],'gauge',15)
    for p in pumps:b.port(p['source_vessel'],'recycle_1',50)
    utility.graph_network(b,spec)
    fed={t['target'] for t in transfers};drained={p['source_vessel'] for p in pumps}
    for v in vessels:
        uid=v['id']
        if uid not in fed:b.boundary('feed' if uid=='A' else uid+'_feed',uid+'.feed',50,out=False)
        if uid not in drained:
            b.boundary(uid+'_bottom_spare',uid+'.bottom',50)
            b.g['edges'][-1].update(termination=utility.RULES['feed_connections']['unspecified_source'],termination_reason='No downstream equipment was specified. The line is blinded after the root valve.')
    for p in pumps:
        uid=p['id'];source=p['source_vessel'];j='J_'+uid
        b.unit(uid,'pump',source_vessel=source,level=0,dn=50)
        b.edge(source+'.bottom',uid+'.in',50)
        b.junction(j);b.edge(uid+'.out',b.jp(j),50)
        b.edge(b.jp(j),source+'.recycle_1',50,role='recycle',manual=uid+' Recirculation control valve')
        for t in transfers:
            if t['pump']==uid:
                b.edge(b.jp(j),t['target']+'.feed',t['dn'],role='material')
                b.g['edges'][-1]['transfer']=t['id']
    for v in vessels:b.boundary('gauge' if v['id']=='A' else v['id']+'_gauge',v['id']+'.gauge',15)
    b.g['transfers']=[{**t,'path_units':[t['source'],t['pump'],t['target']]} for t in transfers];spatial.resolve_feeds(b.g)
    edges={e['id']:e for e in b.g['edges']};reserved=set(spec['stable_valve_ids'].values());serial=1
    for v in b.g['valves']:
        e=edges[v['edge']]
        key=f"{e['service']}.{e['utility_tier']}.{e['level']}" if v['at']=='branch' and 'utility_tier' in e else ('pump_recycle_control' if v['at']=='branch' else v['at'])
        if key in spec['stable_valve_ids']:v['id']=spec['stable_valve_ids'][key]
        else:
            while f'V{serial:03}' in reserved:serial+=1
            v['id']=f'V{serial:03}';reserved.add(v['id']);serial+=1
        v['operator']='lever'
    b.g['display']={'title':f'Single reactor with stairs and a full-span rack · {spec["footprint_m"][0]:g} × {spec["footprint_m"][1]:g} m',
        'subtitle':'Flush stair exit, crossing pipe clearance, and a full-span beam supported at both sides','look_at':[2.4,2.8,2.2],
        'cameras':{'stairway exit':[-2.4,3.6,4.0],'pipe crossing':[4.8,2.6,6.1],'full-span pipe rack':[9,-7,8.6],'overview':[10,-12,7],'front':[3,-13,3.2],'top':[2.5,3,17],'side':[17,3,3.2],
                   'grating joints':[.2,.0,4.4],'stairway side':[-3.5,1.6,1.9],'drainage channel':[4.6,3.8,1.1],'containment curb ramp':[-1.8,-4.5,2.0],'utility station':[8.4,1.7,5.6],'angled off-gas connection':[6.5,1.0,7.3],'vessel top':[5.6,-.2,5.3],'vessel bottom and pump':[3.7,-4,1.7]},
        'view_targets':{'stairway exit':[-.65,4.8,2.95],'pipe crossing':[3.562,3.7,5.465],'full-span pipe rack':[3,3,5.4],'grating joints':[1.2,1.8,3],'stairway side':[-.75,1.6,1.1],'drainage channel':[3.3,4.95,-.02],'containment curb ramp':[1.1,-1.8,.1],'utility station':[4.9,5.2,3.8],'angled off-gas connection':[3.9,4.85,5.5],'vessel top':[3.1,3.2,3.9],'vessel bottom and pump':[3.7,2.8,1.65]},'opaque_grating':True}
    if transfers:
        b.g['display'].update(title=f'Two-reactor pumped transfer A to B · {spec["footprint_m"][0]:g} × {spec["footprint_m"][1]:g} m',subtitle='Pumped transfer from A to B, with A recirculation and shared nitrogen and off-gas branches')
        b.g['display']['cameras'].update({'reactor A':[3,-1,5.2],'reactor B':[6,-1,5.2],'pump and branch':[5,-3,2.4]})
        b.g['display']['view_targets'].update({'reactor A':[*vessels[0]['center_xy_m'],3.7],'reactor B':[*vessels[-1]['center_xy_m'],3.7],'pump and branch':[*pumps[0]['center_xy_m'],1.2]})
    return b.g


def reactor_equipment(g,s,folder,spec,placement):
    uid=placement["id"];prefix=uid+"_"
    def tube(name,*args,**kw):return add_pipe(s,prefix+name,*args,**kw)
    def box(name,*args,**kw):return add_box(s,prefix+name,*args,**kw)
    x,y=placement['center_xy_m'];zf=spec['scene_choices']['body_flange_z_m'];draw=spec['drawing'];apex=zf+draw['h2_mm']/1000
    r=draw['d2_mm']/2000;bottom=zf-draw['h1_mm']/1000
    body=Mesh();profile=[]
    for t in np.linspace(0,math.pi/2,36):profile.append((max(.015,r*math.sin(t)),bottom+.24*(1-math.cos(t))))
    profile += [(r,zf)]
    body.revolve([x,y],profile);body.save(s,folder,uid,'vessel',(.43,.57,.62),uid)['visual_only']=True
    # A curved vessel needs layered collision proxies; its full rectangular box
    # falsely fills the space below the dished head where a person can stand.
    for i,((r0,z0),(r1,z1)) in enumerate(zip(profile,profile[1:])):
        c=tube(f'collision{i}',[x,y,z0],[x,y,z1],max(r0,r1),'vessel',SILVER,uid);c['group']=3
    head=Mesh();head.revolve([x,y],[(max(.015,.60*math.cos(t)),zf+.35*math.sin(t)) for t in np.linspace(0,math.pi/2,12)])
    head.save(s,folder,prefix+'top_head','head',SILVER,uid)
    def flange(name,p,dn,blind=False,oval=False):
        radius=(dn/2000+.038);h=.023
        f=Mesh();f.revolve(p[:2],[(radius,p[2]-h),(radius,p[2]+h)],32,450/350 if oval else 1);f.save(s,folder,prefix+name,'nozzle',FLANGE,uid)
        for j,a in enumerate(np.linspace(0,math.tau,8,endpoint=False)):
            xx=p[0]+(radius-.014)*math.cos(a);yy=p[1]+(radius-.014)*math.sin(a)*(450/350 if oval else 1)
            tube(name+'_bolt'+str(j),[xx,yy,p[2]+h],[xx,yy,p[2]+h+.015],.007,'nozzle',(.22,.25,.26),uid)
        if blind:tube(name+'_blind',[p[0],p[1],p[2]+h],[p[0],p[1],p[2]+h+.014],radius,'nozzle',SILVER,uid)
    flange('body_split',[x,y,zf],1400)
    for side in (-1,1):
        for j,yy in enumerate((y-.35,y+.35)):
            xx=x+side*.71
            box(f'lug{side}_{j}',[xx-.15,yy-.10,2.955,xx+.15,yy+.10,3.025],'equipment_beam',FLANGE,uid)
            box(f'lug_web{side}_{j}',[xx-.10,yy-.012,2.76,xx+.10,yy+.012,2.97],'equipment_beam',FLANGE,uid)
    schedule=[]
    for p in draw['ports']:
        tag=p['tag']
        if tag=='L':continue
        theta=math.radians(p.get('angle_deg',0));rr=p.get('radius_mm',0)/1000
        px=x+rr*math.sin(theta);py=y-rr*math.cos(theta)
        if tag=='K':
            face=bottom-draw['h7_mm']/1000
            tube('K_neck',[x,y,face],[x,y,bottom+.04],.05,'nozzle',SILVER,uid);flange('K_flange',[x,y,face],100)
            outlet=[x,y,face-.16];normal=[0,0,-1]
            red=Mesh();red.revolve([x,y],[(utility.radius(50),outlet[2]),(.05,face-.025)]);red.save(s,folder,prefix+'K_reducer','nozzle',SILVER,uid)
        else:
            face=apex+p['h9_mm']/1000
            shell=zf+.35*math.sqrt(max(0,1-(rr/.60)**2))
            dn=350 if tag=='N1' else p['dn']
            neck=Mesh();neck.revolve([px,py],[(dn/2000,shell-.02),(dn/2000,face)],32,450/350 if tag=='N1' else 1)
            neck.save(s,folder,prefix+tag+'_neck','nozzle',SILVER,uid);flange(tag+'_flange',[px,py,face],dn,blind=tag in ('N1','N5','N7') or ('process_port' in p and p['process_port'] not in g['units'][uid]['ports']),oval=tag=='N1')
            if tag=='N1':
                tube('L_sightglass',[px,py,face+.04],[px,py,face+.075],.05,'nozzle',(.13,.27,.31),uid)
                box('manway_hinge',[px-.22,py-.09,face,px-.15,py+.09,face+.1],'nozzle',FLANGE,uid)
                tube('manway_handle',[px+.11,py-.06,face+.08],[px+.11,py+.06,face+.08],.012,'nozzle',FLANGE,uid)
            if p.get('process_port') in g['units'][uid]['ports']:
                outlet=[px,py,face+.16];normal=[0,0,1]
                red=Mesh();red.revolve([px,py],[(dn/2000,face+.025),(utility.radius(p['pipe_dn']),outlet[2])]);red.save(s,folder,prefix+tag+'_reducer','nozzle',SILVER,uid)
        record={**p,'unit':uid,'face_center_m':[px,py,face]};schedule.append(record)
        if p.get('process_port') in g['units'][uid]['ports']:
            g['units'][uid]['ports'][p['process_port']].update(pos=outlet,normal=normal,level=0 if tag=='K' else 1,nozzle=tag,nozzle_dn=p['dn'])
    # Central R drive: visible shaft, mechanical-seal pedestal, gear housing and motor.
    rz=apex+.07
    tube('R_shaft',[x,y,rz],[x,y,rz+.507],.028,'motor',FLANGE,uid)
    for i,(dx,dy) in enumerate(((-.12,-.09),(.12,-.09),(-.12,.09),(.12,.09))):
        tube('drive_stand'+str(i),[x+dx,y+dy,rz],[x+dx,y+dy,rz+.507],.018,'motor',STEEL,uid)
    box('gearbox',[x-.20,y-.18,rz+.45,x+.20,y+.18,rz+.68],'motor',(.22,.34,.40),uid)
    tube('agitator_motor',[x,y,rz+.68],[x,y,rz+1.05],.16,'motor',(.23,.35,.42),uid)
    for i in range(12):
        a=math.tau*i/12
        tube(f'motor_fin{i}',[x+.17*math.cos(a),y+.17*math.sin(a),rz+.70],[x+.17*math.cos(a),y+.17*math.sin(a),rz+1.02],.009,'motor',(.29,.40,.46),uid)
    # Jacket nozzles are inactive and blanked; their size is an explicit approximation.
    for j,zz in enumerate((bottom+.4,zf-.3)):
        tube(f'jacket_stub{j}',[x+r-.02,y,zz],[x+r+.14,y,zz],.025,'nozzle',SILVER,uid)
        tube(f'jacket_blind{j}',[x+r+.14,y,zz],[x+r+.165,y,zz],.05,'nozzle',FLANGE,uid)
    g['units'][uid].update(center=[x,y,(bottom+apex)/2],box=[x-r,y-r,bottom,x+r,y+r,apex])
    g.setdefault('nozzle_schedule',[]).extend(schedule+[{**next(p for p in draw['ports'] if p['tag']=='L'),'unit':uid}])
    rotate_equipment_instance(g,s,uid,placement.get('rotation_deg',placement.get('nozzle_rotation_deg',0)))


def rotate_equipment_instance(g,s,uid,degrees):
    """Rigidly orient the complete drawing-backed unit, including lugs and ports."""
    if not degrees:return
    angle=math.radians(degrees);matrix=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    center=np.array(g['units'][uid]['center']);center[2]=0
    def point(p):return ((np.array(p)-center)@matrix.T+center).tolist()
    def bbox(box):
        from itertools import product
        points=np.array([point(p) for p in product(*zip(box[:3],box[3:]))]);return [*points.min(0),*points.max(0)]
    for c in s['components']:
        if c.get('owner')!=uid:continue
        c['box']=bbox(c['box'])
        if 'ends' in c:c['ends']=[point(p) for p in c['ends']]
        if c['shape']=='mesh':
            path=Path(c['mesh_path']);lines=[]
            for line in path.read_text().splitlines():
                if line.startswith('v '):line='v '+' '.join(f'{x:.6f}' for x in point([float(x) for x in line.split()[1:]]))
                lines.append(line)
            path.write_text('\n'.join(lines)+'\n')
    unit=g['units'][uid];unit['box']=bbox(unit['box']);unit['rotation_deg']=degrees
    for port in unit['ports'].values():
        if 'pos' in port:port['pos']=point(port['pos']);port['normal']=(matrix@port['normal']).tolist()
    for nozzle in g['nozzle_schedule']:
        if nozzle['unit']==uid and 'face_center_m' in nozzle:nozzle['face_center_m']=point(nozzle['face_center_m'])


def pump_equipment(g,s,folder,spec,placement):
    # Compact ground-mounted horizontal pump, suction on the front and discharge upward.
    uid=placement['id'];px,py=placement['center_xy_m']
    def tube(name,*args,**kw):return add_pipe(s,uid+'_'+name,*args,**kw)
    def box(name,*args,**kw):return add_box(s,uid+'_'+name,*args,**kw)
    box('pump_plinth',[px-.45,py-.38,0,px+.45,py+.75,.18],'pump',(.63,.65,.64),uid)
    box('pump_base',[px-.35,py-.28,.18,px+.35,py+.65,.23],'pump',STEEL,uid)
    tube(uid,[px,py-.18,.55],[px,py+.15,.55],.22,'pump',(.19,.37,.43),uid)
    tube('pump_coupling',[px,py+.15,.55],[px,py+.3,.55],.08,'pump',FLANGE,uid)
    tube('pump_motor',[px,py+.3,.55],[px,py+.65,.55],.16,'pump',(.19,.32,.39),uid)
    for i,yy in enumerate((py,py+.5)):box('pump_foot'+str(i),[px-.2,yy-.06,.22,px+.2,yy+.06,.4],'pump',STEEL,uid)
    g['units'][uid].update(center=[px,py+.2,.55],box=[px-.45,py-.38,0,px+.45,py+.75,.8])
    g['units'][uid]['ports']['in'].update(pos=[px,py-.23,.55],normal=[0,-1,0],level=0)
    g['units'][uid]['ports']['out'].update(pos=[px,py,.80],normal=[0,0,1],level=0)




def equipment(g,s,folder,spec):
    for v in vessel_placements(spec):reactor_equipment(g,s,folder,spec,v)
    for p in pump_placements(spec):pump_equipment(g,s,folder,spec,p)

def piping(g,s,spec,utility_tips):
    units=g['units'];tips=dict(utility_tips);vmap={v['at']:v for v in g['valves'] if v['at']!='branch'}
    # Actual package length sets one valve space; extra stagger is explicit.
    extras=spec['scene_choices'].get('root_valve_extra_spacing_m',{})
    for ep,v in vmap.items():
        uid,p=ep.split('.');port=units[uid]['ports'][p];pos=np.array(port['pos'],float);axis=np.array(port['normal'],float)
        pack,params=package(v);length=params['geometry'].get('face_to_face_mm',params['geometry'].get('body_length_mm'))/1000
        center=pos+axis*(utility.valve_space(length)+length/2+extras.get(ep,0))
        if units[uid]['kind']=='vessel' and p!='bottom':
            stem=np.array([pos[0]-units[uid]['center'][0],pos[1]-units[uid]['center'][1],0]);stem/=np.linalg.norm(stem)
        else:
            stem=np.array([0,0,1]) if abs(axis[2])<.5 else np.array([0,-1,0])
        q,mat=quat_from_axes(axis,stem)
        v.update(center=center.tolist(),axis=axis.tolist(),quat=q,matrix=mat.tolist(),length=length,package=os.path.relpath(pack, ROOT),level=port['level'])
        v['installation_reference']={'kind':'equipment_port','endpoint':ep}
        edge=next(e for e in g['edges'] if e['id']==v['edge']);color=utility.color(edge['role'])
        add_pipe(s,'connection_'+v['id'],pos,center-axis*length/2,utility.radius(v['dn']),'nozzle',color,uid)
        if edge.get('termination')==utility.RULES['feed_connections']['unspecified_source']:
            end=center+axis*length/2
            # Move only the flange outside the package's full operator envelope.
            operator=params['geometry'].get('handle_length_mm',0)/1000
            straight=max(0,operator+RULES['geometry_screen']['sweep_margin_m']+.01-length/2-.040)
            cap=utility.flange_joint(s,edge['id']+'_feed_joint',end,axis,v['dn'],edge['id'],unused=True,straight_m=straight)
            boundary=edge['a'] if edge['b']==ep else edge['b'];bu,bp=boundary.split('.')
            units[bu].update(center=cap);units[bu]['ports'][bp].update(pos=cap,level=port['level'])
            tips[ep]=end.tolist();tips[boundary]=cap
            points=[end.tolist(),cap] if edge['a']==ep else [cap,end.tolist()]
            s['routing'].append({'edge':edge['id'],'points':points,'components':[],'elbows':0,'constraints':'No supply vessel was specified. The flange is blinded after a short section that clears the lever sweep.'})
            continue
        tip=center+axis*(length/2+max(length,.25))
        add_pipe(s,'tail_'+v['id'],center+axis*length/2,tip,utility.radius(v['dn']),'pipe',color,edge['id']);tips[ep]=tip.tolist()
    for e in g['edges']:
        au,ap=e['a'].split('.');bu,bp=e['b'].split('.')
        if ap=='gauge':
            tip=tips[e['a']];outward=np.array(tip[:2])-units[au]['center'][:2];outward/=np.linalg.norm(outward)
            tips[e['b']]=[*(np.array(tip[:2])+.35*outward),tip[2]]
            units[bu]['center']=tips[e['b']];units[bu]['ports'][bp].update(pos=tips[e['b']],level=units[au]['level'])
        if units[au]['kind']=='pump' and ap=='out' and units[bu]['kind']=='junction':
            branches=[r for r in s['process_risers'] if next(ee for ee in g['edges'] if ee['id']==r['edge'])['a'].split('.')[0]==bu]
            z=tips[e['a']][2];points=[]
            for row in branches:
                branch=next(ee for ee in g['edges'] if ee['id']==row['edge']);point=[*row['xy'],z]
                tips[branch['a']]=point;units[bu]['ports'][branch['a'].split('.')[1]].update(pos=point,level=0);points.append(point)
            tips[e['b']]=points[0];units[bu]['ports'][bp].update(pos=points[0],level=0);units[bu]['center']=points[0]
            for i,(a,b) in enumerate(zip(points,points[1:])):utility.pipe(s,bu+f'_manifold{i}',a,b,e['dn'],'material',bu)
    s['valve_route_reservations']=[reserved_box(v) for v in g['valves'] if 'center' in v]
    for e in g['edges']:
        if e.get('utility_tier') in ('floor_branch','temporary_outlet'):continue
        if e.get('termination')==utility.RULES['feed_connections']['unspecified_source']:continue
        a,b=tips[e['a']],tips[e['b']];role=e['role']
        if role in ('nitrogen','offgas','feed','vapor','process_evaporated_vapor'):
            if role=='offgas':
                pts,count=utility.offgas_route(a,b,s,e['dn'],spec['scene_choices']['utility_station']['offgas_join_drop_m'])
            else:pts,count=utility.least_elbow_route(a,b,s,e['dn'],medium=role)
            utility.run(s,e,pts);s['routing'][-1]['candidate_count']=count
            continue
        elif e['a'].endswith('.bottom'):
            approach_y=b[1]-.35
            pts=[a,[a[0],a[1],b[2]],[a[0],approach_y,b[2]],[b[0],approach_y,b[2]],b]
        elif units[e['a'].split('.')[0]]['kind']=='pump':pts=[a,[b[0],a[1],a[2]],b]
        elif role=='recycle' or e.get('transfer'):
            rx,ry=next(row['xy'] for row in s['process_risers'] if row['edge']==e['id'])
            target_level=units[e['b'].split('.')[0]]['ports'][e['b'].split('.')[1]]['level']
            pts,rack_z=utility.rack_return_route(s,e,a,b,[rx,ry],target_level)
            s.setdefault('upward_rack_routes',[]).append({'edge':e['id'],'level':target_level,'dn':e['dn'],'horizontal':[[rx,ry,rack_z],[b[0],ry,rack_z]]})
            v=next((v for v in g['valves'] if v['edge']==e['id'] and v['at']=='branch'),None)
            if v:
                _,params=package(v);length=params['geometry']['body_length_mm']/1000
                utility.place_valve(v,[rx,ry,max(1.40,a[2]+utility.valve_space(length)+length/2+3*utility.radius(v['dn']))],spec['scene_choices']['return_valve_axis'],[-1,0,0],0)
            utility.run(s,e,pts,v)
            s['routing'][-1]['constraints']='A pump discharge recirculation or transfer line crosses the slab near an edge, rises to the rack at a height set by pipe diameters and routed lines, and returns to the specified process port.'
            continue
        else:pts=[a,[a[0],a[1],b[2]],[a[0],b[1],b[2]],b]
        utility.run(s,e,pts)
    for e in g['edges']:
        if e['a'].endswith('.gauge'):
            p=tips[e['b']];add_pipe(s,e['id']+'_pressure_gauge',[p[0],p[1]-.04,p[2]],[p[0],p[1]+.04,p[2]],.065,'gauge',(.87,.88,.85))
    for i,row in enumerate(s['process_risers']):
        rx,ry=row['xy'];z=spec['platform_z_m']-.15
        # Attach to a real longitudinal joist, rather than assuming its x.
        joists=[c for c in s['components'] if c['id'].startswith('joist') and c['box'][1]<=ry<=c['box'][4]]
        anchor=min(joists,key=lambda c:abs((c['box'][0]+c['box'][3])/2-rx));ax=(anchor['box'][0]+anchor['box'][3])/2
        add_pipe(s,f'riser_support{i}',[ax,ry,z],[rx,ry,z],.022,'hanger',STEEL)
        add_pipe(s,f'riser_clamp{i}',[rx,ry,z-.02],[rx,ry,z+.02],utility.radius(row['dn'])+.006,'hanger',FLANGE)


def penetration_check(g,s,model,data):
    """Check actual primitive distances, allowing only joined edges at a shared end."""
    pipes=[c for c in s['components'] if c['kind']=='pipe']
    solid=[c for c in s['components'] if not c.get('visual_only') and c['kind'] in ('vessel','head','motor','column','beam','equipment_beam','walk_surface','pump')]
    edges={e['id']:e for e in g['edges']};bad=[];clearance=[];crossings=[]
    def joined(c,o):
        shared_end=any(math.dist(a,b)<1e-5 for a in c['ends'] for b in o['ends'])
        if c.get('owner')==o.get('owner') and shared_end:return True
        def endpoints(component):
            owner=component.get('owner');edge=edges.get(owner)
            return {edge['a'].split('.')[0],edge['b'].split('.')[0]} if edge else {owner}
        common=endpoints(c)&endpoints(o)
        return any(g['units'].get(u,{}).get('kind')=='junction' for u in common) and any(math.dist(a,b)<1e-5 for a in c['ends'] for b in o['ends'])
    for i,c in enumerate(pipes):
        for o in solid:
            if not intersects(c['box'],np.array([o['box']]))[0]:continue
            distance=mujoco.mj_geomDistance(model,data,model.geom(c['id']).id,model.geom(o['id']).id,.001,None)
            if distance<-.0001:bad.append({'pipe':c['id'],'obstacle':o['id'],'penetration_m':float(-distance)})
        direction=np.array(c['ends'][1])-c['ends'][0];direction/=np.linalg.norm(direction)
        for o in pipes[i+1:]:
            if joined(c,o):continue
            distance,t,k=utility.segment_distance(*c['ends'],*o['ends']);gap=distance-c['radius']-o['radius']
            if gap<-.0001:bad.append({'pipe':c['id'],'obstacle':o['id'],'penetration_m':float(-gap),'method':'finite_centerline_envelopes'})
            other=np.array(o['ends'][1])-o['ends'][0];other/=np.linalg.norm(other)
            if np.linalg.norm(np.cross(direction,other))>1e-6 and 1e-6<t<1-1e-6 and 1e-6<k<1-1e-6:
                crossings.append({'pipe':c['id'],'other':o['id'],'gap_m':gap})
                if gap<utility.RULES['routing']['unconnected_crossing_clearance_m']-1e-7:clearance.append(crossings[-1])
            cp=sorted(np.dot(p,direction) for p in c['ends']);op=sorted(np.dot(p,direction) for p in o['ends'])
            if np.linalg.norm(np.cross(direction,other))>1e-6 or min(cp[1],op[1])<=max(cp[0],op[0])+1e-7:continue
            gap=np.linalg.norm(np.cross(np.array(c['ends'][0])-o['ends'][0],direction))-c['radius']-o['radius']
            if gap<RULES['geometry_screen']['parallel_pipe_clearance_m']-1e-7:clearance.append({'pipe':c['id'],'other':o['id'],'gap_m':gap})
        for o in s['components']:
            if o['kind']!='column':continue
            gap=math.sqrt(sum(max(c['box'][k]-o['box'][k+3],o['box'][k]-c['box'][k+3],0)**2 for k in range(3)))
            if gap<RULES['geometry_screen']['pipe_post_clearance_m']-1e-7:clearance.append({'pipe':c['id'],'column':o['id'],'gap_m':gap})
    return {'status':'pass' if not bad and not clearance else 'fail','penetrations':bad,'clearance_violations':clearance,'crossing_measurements':crossings,'scope':'Check pipes against vessels, beams, walkable surfaces and pumps. Check disconnected pipes using finite centerlines and outside diameters. Crossing clearance is 50 mm, parallel bare pipe clearance is 50 mm, and clearance to a column is 100 mm. Connected endpoints are excluded from overlap checks. Structural loads are outside the scope.'}


def render(g,model,data,folder):
    for name,look,dist,az,el in [('stair_exit',[-.65,4.8,2.95],3.3,215,-22),('rack_crossing',[3.562,3.7,5.465],1.5,125,-13),('overview',[2.1,1.3,2.25],15.5,125,-20),('nozzles',[3.1,3.25,3.85],3.4,125,-30),('pump_loop',[3.7,2.8,1.65],6.8,90,-1),('workstation',[4.7,5.0,3.0],9.2,45,-9),('offgas_join',[4.1,4.8,5.5],3.6,80,-20),('panel_detail',[1.1,1.9,3.0],2.3,240,-30),('stair_detail',[-.75,1.7,1.1],3.8,180,-10),('drain_detail',[3.3,4.95,-.02],2.8,270,-35),('ramp_detail',[1.1,-1.8,.1],4.5,230,-25)]:
        if name=='rack_crossing':look=g['display']['view_targets']['pipe crossing']
        cam=mujoco.MjvCamera();cam.lookat[:]=look;cam.distance=dist;cam.azimuth=az;cam.elevation=el
        opt=mujoco.MjvOption();opt.geomgroup[3]=0
        with mujoco.Renderer(model,width=1800,height=1200) as r:
            r.update_scene(data,camera=cam,scene_option=opt);im=Image.fromarray(r.render())
        d=ImageDraw.Draw(im);d.rounded_rectangle([22,20,1160,118],12,fill=(241,244,245))
        d.text((42,32),g['display']['title'],font=font(30),fill=(30,48,60))
        caption={'stair_exit':'The stringer ends at the landing beam below the tread, leaving the walking surface clear.','rack_crossing':'Crossing pipes are staggered by outside diameter and clearance. Individual supports sit on the shared beam.','overview':'The beam spans the platform between opposite supports. Nitrogen, off-gas and recirculation share the rack.','panel_detail':'Grating panels have real gaps. One panel is raised 7 mm and another sags 10 mm at the center.','stair_detail':'Open treads overlap by 20 mm, are 30 mm thick, and retain the lower edge profile.','drain_detail':'300 mm channel with individual covers. A 15 mm settlement and a 6 mm lifted edge are marked separately.','ramp_detail':'The plinth extends along the entrance edge. Both ramps are 900 mm long and 150 mm high.','workstation':'The DN65 off-gas line enters the header directly at high level. The ground-level off-gas line ends at a temporary blinded ball valve.','nozzles':'The N10 feed is blinded after the root valve. The equipment port is retained.','pump_loop':'V016 is retained. The recirculation riser passes through the slab and rises to the rack before turning toward the reactor port.','offgas_join':'DN25 off-gas enters a DN65 branch from above at 45 degrees. Parallel lines on the rack retain clearance.'}[name]
        if g.get('transfers') and name=='nozzles':caption="A's unspecified supply port is blinded, B's feed is connected to the pump discharge, and equipment connections retain root valves."
        d.text((42,79),caption,font=font(22),fill=(55,73,82));im.save(folder/(name+'.png'))


def diagram(folder):
    im=Image.new('RGB',(1500,940),(245,247,248));d=ImageDraw.Draw(im)
    ink=(30,48,60);gas=(166,137,12);silver=(94,112,118)
    def box(rect,label,fill):
        d.rounded_rectangle(rect,10,fill=fill,outline=ink,width=2)
        for i,line in enumerate(label.split('\n')):d.text((rect[0]+15,rect[1]+15+i*34),line,font=font(24),fill=ink)
    def arrow(points,c):
        d.line(points,fill=c,width=6);a=np.array(points[-2]);b=np.array(points[-1]);v=(b-a)/np.linalg.norm(b-a);p=np.array([-v[1],v[0]])
        d.polygon([tuple(b),tuple(b-v*16+p*7),tuple(b-v*16-p*7)],fill=c)
    d.text((35,24),'Single reactor, from the utility station riser via a floor branch to the vessel',font=font(35),fill=ink)
    d.text((35,77),'Ground and one platform. The N10 feed has no specified source and is blinded after the root valve.',font=font(24),fill=ink)
    box([45,160,315,610],'Utility station with a DN50 nitrogen riser, a DN65 off-gas riser, a ball valve and a blind flange on each level',(229,234,236))
    box([440,170,800,280],'The DN25 platform nitrogen branch passes through an isolation valve and rises to the top rack.',(243,237,192))
    box([440,330,800,440],'The DN65 platform off-gas branch joins the high header directly without a root valve.',(222,230,232))
    box([1040,245,1435,480],'A, AE 1000 reactor. N2 nitrogen inlet DN25, N3 off-gas outlet DN25, N9 recirculation and K bottom outlet DN50.',(207,223,227))
    arrow([(315,215),(440,215)],gas);arrow([(800,215),(970,215),(970,290),(1040,290)],gas)
    arrow([(1040,390),(800,390)],silver);arrow([(440,390),(315,390)],silver)
    box([440,520,800,625],'No ground-level equipment user. The nitrogen root is blinded and off-gas has a temporary port only.',(229,234,236))
    arrow([(315,550),(440,550)],gas)
    box([990,690,1340,790],'P1, DN50 recirculation pump',(211,225,228))
    arrow([(1190,480),(1190,690)],silver)
    arrow([(1340,740),(1465,740),(1465,440),(1435,440)],silver)
    d.text((1020,565),'Vessel bottom valve to pump suction valve',font=font(23),fill=ink)
    d.text((970,815),'Pump discharge valve through recirculation control valve to the same reactor',font=font(23),fill=ink)
    d.text((40,860),'Yellow marks nitrogen and silver-gray marks unspecified media. The material, recycle and off-gas lines use media-based colors.',font=font(25),fill=ink)
    d.text((40,903),'Diagram shows connectivity. The 3D model and check records give elevations, routing, standing areas and clearances.',font=font(22),fill=ink)
    im.save(folder/'process.png')


def build(spec,folder):
    g=graph(spec);s={'id':g['id'],'components':[],'holes':{},'surfaces':[],'stairs':[],'floors':[0,spec['platform_z_m']],'footprint':spec['footprint_m'],'routing':[]}
    process_edges=[e for e in g['edges'] if e['role']=='recycle' or e.get('transfer')]
    corridor=spec['scene_choices'].get('riser_corridor');s['process_risers']=[]
    for i,e in enumerate(process_edges):
        xy=[corridor['x_m'],corridor['y_start_m']+i*corridor['pitch_m']] if corridor else spec['scene_choices']['return_riser_xy_m']
        s['process_risers'].append({'edge':e['id'],'xy':xy,'dn':e['dn']})
    spatial.define_rack_zones(s,{1:[r['xy'] for r in s['process_risers']]});g['rack_zones']=s['rack_zones']
    structure(s,folder,spec);building.upgrade_frame(s,folder,spec,Mesh);equipment(g,s,folder,spec);utility_tips=utility.build_station(g,s,spec);piping(g,s,spec,utility_tips)
    spatial.build_shared_rack(g,s)
    for row in s['process_risers']:
        rx,ry=row['xy'];r=utility.radius(row['dn'])+.035;s['holes'][1].append([rx-r,ry-r,rx+r,ry+r])
    building.deck(s,folder,spec,Mesh)
    started=time.perf_counter()
    model,data=export_model(g,s,folder/'room.xml')
    s['model_load_seconds']=time.perf_counter()-started
    return g,s,model,data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--spec',type=Path,default=Path(__file__).resolve().parent.parent/'basic_spec.json');args=parser.parse_args()
    spec=json.loads(args.spec.read_text());out=args.output.resolve();folder=out/spec['id'];folder.mkdir(parents=True,exist_ok=True)
    trials=[]
    for attempt in range(13):
        g,s,model,data=build(spec,folder)
        operation=valve_checks(g,s,model,data)
        collisions=[{'valve':v['id'],'at':v['at'],'obstacles':v['sweep_obstacles']} for v in g['valves'] if not v['sweep_clear']]
        trials.append({'attempt':attempt,'root_extra_spacing_m':dict(spec['scene_choices']['root_valve_extra_spacing_m']),'sweep_collisions':collisions})
        involved={row['valve'] for row in collisions}|{name for row in collisions for name in row['obstacles']}
        adjustable=[v for v in g['valves'] if v['id'] in involved and v['at']!='branch' and g['units'][v['at'].split('.')[0]]['kind']=='vessel' and not v['at'].endswith('.bottom')]
        if not adjustable or attempt==12:break
        v=adjustable[0];extras=spec['scene_choices']['root_valve_extra_spacing_m']
        extras[v['at']]=round(extras.get(v['at'],0)+.05,6)
    write_json(folder/'placement_trials.json',trials)
    spatial.add_scale_people(g,s)
    model,data=export_model(g,s,folder/'room.xml')
    process=validate(g,True)
    report={'process':process,'piping_rules':utility.check(g,s,spec),'installation':installation_check(g,s,spec),'operation':operation,'pipe_geometry':penetration_check(g,s,model,data),'mujoco':{'status':'pass','nq':model.nq,'geoms':model.ngeom,'scope':'model load and forward kinematics'},
        'grating':{'opaque':True,'visual_triangles':sum(c.get('triangles',0) for c in s['components'] if c['kind']=='grating'),'collision':'Individual panel collisions and a segmented depression preserve joints, channels and lower tread edges. Fine perforations are simplified.'},
        'building':building.check(s,model,data),'spatial_rules':spatial.check(g,s,model),'scope':'Scope covers connection checks, standing and sweep geometry, and model loading. Fabrication, loads and construction design require separate checks.'}
    if any(report[k]['status']!='pass' for k in ('process','piping_rules','pipe_geometry','installation','building','spatial_rules')):
        for v in g['valves']:v['eligible']=False;v['reason']+='. Room connection or geometry checks failed'
        report['operation']['status']='room_rejected';report['operation']['eligible']=0
        for v in report['operation']['valves']:v['eligible']=False;v['reason']+='. Room connection or geometry checks failed'
    crossing=report['pipe_geometry']['crossing_measurements']
    if crossing:
        row=min(crossing,key=lambda r:r['gap_m']);cs={c['id']:c for c in s['components']};a,b=cs[row['pipe']],cs[row['other']]
        _,t,k=utility.segment_distance(*a['ends'],*b['ends'])
        point=(np.array(a['ends'][0])+t*(np.array(a['ends'][1])-a['ends'][0])+np.array(b['ends'][0])+k*(np.array(b['ends'][1])-b['ends'][0]))/2
        g['display']['view_targets']['pipe crossing']=point.tolist();g['display']['cameras']['pipe crossing']=(point+[1.5,-1.8,.8]).tolist()
    write_json(folder/'checks.json',report);write_json(folder/'geometry.json',s);write_json(folder/'source_spec.json',spec);lists(g,folder)
    if g['transfers']:
        from ..diagram import draw_process
        draw_process(g,folder/'process.png')
    else:diagram(folder)
    render(g,model,data,folder)
    write_json(folder/'building_rules_snapshot.json',building.RULES);write_json(folder/'piping_rules_snapshot.json',utility.RULES);write_json(out/'rules_snapshot.json',RULES);write_json(out/'summary.json',{'rooms':[{'room':g['id'],'modes':['pump'],'relation':g['recipe']['relation']}]})
    print(json.dumps({'folder':str(folder),'process':process,'operation':{k:operation[k] for k in ('status','eligible','all')},'grating':report['grating'],'building':report['building']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
