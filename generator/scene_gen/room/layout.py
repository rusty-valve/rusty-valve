"""Place equipment and its supporting structure inside the building."""
from __future__ import annotations
from collections import defaultdict
import math
import random
import numpy as np
from .process import RULES

H=RULES['construction_choices']['storey_height_m']

def box_of(a,b,r=0):
    return [min(a[i],b[i])-r for i in range(3)]+[max(a[i],b[i])+r for i in range(3)]

def add_box(scene,name,box,kind,color,owner=None,support=None):
    c={'id':name,'kind':kind,'shape':'box','box':list(box),'color':color,'owner':owner,'support':support}
    scene['components'].append(c); return c

def add_pipe(scene,name,a,b,r,kind='pipe',color=(.42,.62,.65),owner=None):
    if math.dist(a,b)<1e-8: return
    direction=(np.array(b)-a)/math.dist(a,b)
    extent=r*np.sqrt(np.maximum(0,1-direction*direction))
    bounds=[min(a[i],b[i])-float(extent[i]) for i in range(3)]+[max(a[i],b[i])+float(extent[i]) for i in range(3)]
    c={'id':name,'kind':kind,'shape':'cylinder','ends':[list(a),list(b)],'radius':r,
       'box':bounds,'color':color,'owner':owner}
    scene['components'].append(c); return c

def choose_levels(g):
    for u in g['units'].values():
        u['level']=1 if u['kind']=='vessel' else 0
    for u in g['units'].values():
        if u['kind']=='condenser': u['level']=g['units'][u['owner']]['level']+1
        if u['kind']=='jacket': u['level']=g['units'][u['owner']]['level']
        if u['level']>2: raise ValueError('The condenser extends above two platforms')

def equipment(g):
    choose_levels(g); rng=random.Random(g['seed']); scene={'id':g['id'],'components':[], 'holes':defaultdict(list), 'surfaces':[], 'stairs':[], 'floors':[0,H,2*H]}
    units=g['units']; vessels=[u for u in units.values() if u['kind']=='vessel']
    # Placement follows transfer destinations and elevation, with bays containing 1–2 vessels.
    levels=defaultdict(list)
    for u in vessels: levels[u['level']].append(u['id'])
    positions={}
    for lev,names in sorted(levels.items()):
        xs={1:[6],2:[2.7,9.1],3:[2.2,6.1,10.1],4:[1.7,4.5,7.4,10.3]}[len(names)]
        if rng.random()<.5: xs=list(reversed(xs))
        for name,x in zip(names,xs):
            preferred=(x+rng.uniform(-.12,.12),3.35+rng.uniform(-.15,.25))
            # Upper vessels must not cover the lower vessel's top nozzles/motor.
            candidates=[preferred]+[(xx,yy) for xx in (2.1,4.2,6.3,8.4,10.5) for yy in (3.1,3.9)]
            positions[name]=next(p for p in candidates if all(math.dist(p,q)>((2.05 if units[name]['orientation']=='vertical' else .5) if units[old]['level']!=lev else 2.45) for old,q in positions.items()))
    for u in vessels:
        x,y=positions[u['id']]; z=H*u['level']; scale=u['size']; r=.64*scale
        if u['orientation']=='vertical':
            lo=z-1.9; hi=z+.9; a=[x,y,lo+r*.3]; b=[x,y,hi-r*.3]
            add_pipe(scene,u['id'],a,b,r,'vessel',(.69,.74,.75),u['id'])
            bounds=[x-r,y-r,lo,x+r,y+r,hi]
            for name,zz in [('lower',lo+r*.3),('upper',hi-r*.3)]:
                scene['components'].append({'id':u['id']+'_'+name,'kind':'head','shape':'ellipsoid','pos':[x,y,zz], 'size':[r,r,r*.3], 'box':[x-r,y-r,zz-r*.3,x+r,y+r,zz+r*.3], 'color':(.69,.74,.75),'owner':u['id']})
            scene['holes'][u['level']].append([x-r-.1,y-r-.1,x+r+.1,y+r+.1])
            for side in (-1,1):
                add_box(scene,u['id']+'_lug'+str(side),[x+side*(r+.14)-.2,y-.14,z-.15,x+side*(r+.14)+.2,y+.14,z+.05],'lug',(.38,.46,.48),u['id'])
                add_box(scene,u['id']+'_beam'+str(side),[x+side*(r+.23)-.09,0,z-.32,x+side*(r+.23)+.09,6,z-.15],'equipment_beam',(.25,.37,.45),u['id'])
            if u.get('reactor'):
                add_pipe(scene,u['id']+'_motor',[x,y,hi],[x,y,hi+.52],.20,'motor',(.21,.34,.43),u['id'])
            u.update(bottom=lo,top=hi,center=[x,y,(lo+hi)/2],radius=r)
        else:
            half=1.05*scale; center=z+1.35; lo=center-r; hi=center+r
            add_pipe(scene,u['id'],[x-half,y,center],[x+half,y,center],r,'vessel',(.66,.74,.76),u['id'])
            bounds=[x-half-r*.18,y-r,lo,x+half+r*.18,y+r,hi]
            for side in (-1,1):
                add_box(scene,u['id']+'_saddle'+str(side),[x+side*.65-.12,y-.48,z,x+side*.65+.12,y+.48,center],'saddle',(.31,.42,.49),u['id'])
            u.update(bottom=lo,top=hi,center=[x,y,center],radius=r)
        u['box']=bounds
        top_ports=[p for p in u['ports'] if p not in ('bottom','drain')]
        ring=[(-.40,-.37),(.0,-.49),(.40,-.37),(.49,0),(.40,.37),(0,.49),(-.40,.37),(-.49,0),(-.25,-.30)]
        for j,p in enumerate(top_ports):
            dx,dy=ring[j%len(ring)]; dx*=scale; dy*=scale
            zz=hi if u['orientation']=='vertical' else z+1.50
            direction=[0,0,1] if u['orientation']=='vertical' else [0,-1,0]
            if u['orientation']=='vertical':
                pos=[x+dx,y+dy,zz]
                surface_z=hi-r*.3+r*.3*math.sqrt(max(0,1-(dx*dx+dy*dy)/(r*r)))
                add_pipe(scene,u['id']+'_'+p+'_shellneck',[x+dx,y+dy,surface_z],pos,u['ports'][p]['dn']/2000,'nozzle',(.4,.6,.62),u['id'])
            else:
                # A horizontal vessel has one real front nozzle bank, not duplicated ring projections.
                dx=((j%4)-1.5)*.42*scale;zz=z+(1.15 if j<4 else 1.70)
                surface_y=y-math.sqrt(max(.01,r*r-(zz-u['center'][2])**2))
                pos=[x+dx,y-r,zz]
                add_pipe(scene,u['id']+'_'+p+'_shellneck',[x+dx,surface_y,zz],pos,u['ports'][p]['dn']/2000,'nozzle',(.4,.6,.62),u['id'])
            stem=[0,-1 if dy<=0 else 1,0] if u['orientation']=='vertical' else [0,0,1]
            u['ports'][p].update(pos=pos,normal=direction,stem=stem)
        # Bottom head → short elbow → exposed root valve (ball substitute), avoiding the shell's shadow.
        for j,p in enumerate(('bottom','drain')):
            dx=.0 if p=='bottom' else .28
            start=[x+dx,y,lo]; elbow=[x+dx,y,lo-.15-j*.18]; end=[x+dx,y-r-.32,elbow[2]]
            add_pipe(scene,u['id']+'_'+p+'_neck1',start,elbow,.028 if j==0 else .012,'nozzle',(.4,.6,.62),u['id'])
            add_pipe(scene,u['id']+'_'+p+'_neck2',elbow,end,.028 if j==0 else .012,'nozzle',(.4,.6,.62),u['id'])
            u['ports'][p].update(pos=end,normal=[0,-1,0],stem=[0,0,1])
        # Confirmed §5 safety device: obstacle, not a manual-valve question.
        add_pipe(scene,u['id']+'_safety',[x+.25,y+.18,hi],[x+.25,y+.18,hi+.3],.065,'safety',(.73,.44,.21),u['id'])
    pumps=[u for u in units.values() if u['kind']=='pump']
    for i,u in enumerate(pumps):
        s=units[u['source_vessel']]; x=s['center'][0]+.95
        same=[p for p in pumps[:i] if p['source_vessel']==u['source_vessel']]
        y=4.65 if len(same)==0 else 2.15
        x=min(10.2,max(.7,x)); z=.55
        add_box(scene,u['id']+'_base',[x-.55,y-.34,0,x+.55,y+.34,.22],'plinth',(.43,.44,.43),u['id'])
        add_pipe(scene,u['id'],[x-.30,y,z],[x+.35,y,z],.22,'pump',(.24,.47,.57),u['id'])
        add_box(scene,u['id']+'_foot',[x-.2,y-.15,.22,x+.2,y+.15,.4],'support',(.3,.4,.44),u['id'])
        u.update(center=[x,y,z],box=[x-.45,y-.3,.22,x+.45,y+.3,.85])
        u['ports']['in'].update(pos=[x-.45,y,z],normal=[-1,0,0],stem=[0,0,1])
        u['ports']['out'].update(pos=[x+.45,y,z],normal=[1,0,0],stem=[0,0,1])
        add_box(scene,u['id']+'_check',[x+.72,y-.07,z-.07,x+.84,y+.07,z+.07],'check_valve',(.7,.49,.25),u['id'])
    aux=[u for u in units.values() if u['kind'] in ('exchanger','filter','condenser')]
    for i,u in enumerate(aux):
        if u['kind']=='condenser':
            src=units[u['owner']]; z=H*u['level']
            candidates=[(src['center'][0],4.5),(src['center'][0],2.25)]+[(xx,4.6) for xx in (1.2,3.5,5.8,8.2,10.8)]
            x,y=next(p for p in candidates if all(math.dist(p,other['center'][:2])>(2.15 if u.get('orientation')=='horizontal' or other['orientation']=='horizontal' else 1.25) for other in vessels if other['level']==u['level']))
            lo=z+.35; hi=z+1.85
            a=[x,y,lo]; b=[x,y,hi]; r=.25
            if u.get('orientation')=='horizontal': a=[x-.8,y,z+.9];b=[x+.8,y,z+.9];lo=z+.65;hi=z+1.15
        else:
            preferred=4.6 if i%2==0 else 7.4
            candidates=[(preferred,4.6),(preferred,2.5)]+[(xx,yy) for xx in (1.2,5.5,8.3,10.6) for yy in (4.6,2.5)]
            def free_aux(p):
                xx,yy=p
                for pump in pumps:
                    px,py,_=pump['center']
                    if px-.9<xx<px+1.95 and abs(py-yy)<.85:return False
                for other in aux[:i]:
                    if other['level']==0 and math.dist(p,other['center'][:2])<1.25:return False
                for vessel in vessels:
                    if vessel['bottom']<1.85 and math.dist(p,vessel['center'][:2])<vessel['radius']+.45:return False
                return True
            x,y=next(p for p in candidates if free_aux(p));z=0;lo=.5;hi=1.8;r=.25
            a=[x,y,lo]; b=[x,y,hi]; u['orientation']='vertical'
        add_pipe(scene,u['id'],a,b,r,u['kind'],(.72,.62,.40),u['id'])
        bounds=box_of(a,b,r);u.update(center=[x,y,(lo+hi)/2],box=bounds,bottom=lo,top=hi)
        for side in (-1,1): add_box(scene,u['id']+'_leg'+str(side),[x+side*.2-.04,y-.05,z,x+side*.2+.04,y+.05,lo+.3],'support',(.27,.38,.44),u['id'])
        for j,p in enumerate(u['ports']):
            utility=p.startswith('utility');side=1 if utility else -1
            zz=lo+.2 if p in ('in','condensate','utility_in') else hi-.2
            if u.get('orientation')=='horizontal':
                if utility:pos=[x+(-.4 if p=='utility_in' else .4),y-r,z+.9];normal=[0,-1,0]
                else:
                    side=-1 if p in ('in','vapor') else 1
                    pos=[x+side*.8,y,z+.9];normal=[side,0,0]
            else:pos=[x,y+side*r,zz];normal=[0,side,0]
            u['ports'][p].update(pos=pos,normal=normal,stem=[0,0,1])
    for u in units.values():
        if u['kind']=='jacket':
            v=units[u['owner']]; x,y,_=v['center']; z=H*v['level']; r=v['radius']
            if v['orientation']=='vertical':
                add_pipe(scene,u['id'],[x,y,v['bottom']+.2],[x,y,z-.25],r+.045,'jacket_shell',(.56,.68,.72),u['id'])
            else:add_pipe(scene,u['id'],[x-.75*v['size'],y,v['center'][2]],[x+.75*v['size'],y,v['center'][2]],r+.045,'jacket_shell',(.56,.68,.72),u['id'])
            for j,p in enumerate(u['ports']):
                if v['orientation']=='vertical':
                    sign=-1 if j==0 else 1;pos=[x+sign*(r+.1),y,z-1.45+j*.6];normal=[sign,0,0];start=[x+sign*(r+.045),y,pos[2]]
                else:
                    pos=[x+(-.5 if j==0 else .5)*v['size'],y+r+.1,z+1.05+j*.6];normal=[0,1,0]
                    start=[pos[0],y+math.sqrt((r+.045)**2-(pos[2]-v['center'][2])**2),pos[2]]
                add_pipe(scene,u['id']+'_'+p+'_neck',start,pos,u['ports'][p]['dn']/2000,'nozzle',(.4,.6,.62),u['id'])
                u['ports'][p].update(pos=pos,normal=normal,stem=[0,0,1])
            u.update(center=v['center'],box=v['box'])
    # Manifolds retain distinct physical tee ports at a common junction point.
    for u in units.values():
        if u['kind']=='junction':
            if u['id'].endswith('_bottom'):
                v=units[u['id'].removesuffix('_bottom')]; p=v['ports']['bottom']['pos']; center=[p[0]-.45,p[1]-.45,p[2]]
            elif u['id'].endswith('_feed'):
                v=units[u['id'].removesuffix('_feed')]; p=v['ports']['feed']['pos']; center=[p[0]-.8,p[1],p[2]+.60]
            else:
                v=units[u['id'].removesuffix('_out')]; x,y,z=v['center'];center=[x+.95,y,z+.7]
            u['center']=center
            for p in u['ports'].values(): p.update(pos=list(center),normal=[0,0,0],stem=[0,0,1])
    # All utility mains occupy the rear service strip. Ports distributed per service and floor.
    headers=[u for u in units.values() if u['kind']=='header']
    for j,u in enumerate(headers):
        y=5.65; z=8.0+j*.28
        u['center']=[6,y,z];u['level']=2
        add_pipe(scene,u['id'],[.4,y,z],[11.6,y,z],.04,'header',(.3,.52,.58),u['id'])
        for i,p in enumerate(u['ports'].values()):
            x=.7+(i+.3)*10.0/max(1,len(u['ports']))+.12*(j%3)
            p.update(pos=[x,y,z],normal=[0,-1,0],stem=[0,0,1])
        for xx in (.4,11.6):
            add_box(scene,u['id']+'_clamp_'+str(xx),[xx-.04,y+.025,z-.025,xx+.04,y+.10,z+.025],'support',(.32,.4,.44),u['id'])
    for xx in (.4,11.6):add_box(scene,'utility_post_'+str(xx),[xx-.04,5.71,0,xx+.04,5.79,9.6],'support',(.3,.4,.45))
    boundaries=[u for u in units.values() if u['kind']=='boundary']
    for i,u in enumerate(boundaries):
        edge=next(e for e in g['edges'] if e['a'].startswith(u['id']+'.') or e['b'].startswith(u['id']+'.'))
        other=edge['b'] if edge['a'].startswith(u['id']+'.') else edge['a']; uid,p=other.split('.'); target=units[uid]['ports'][p]['pos']
        # Gauge is local; drains go to a boundary manifold, never dangling into the passage.
        if 'gauge' in u['id']:
            pos=([target[0]+.3,target[1]+.3,target[2]+.8] if units[uid].get('orientation')=='vertical'
                 else [target[0],target[1]-.8,target[2]+.45])
            add_pipe(scene,u['id']+'_dial',pos,[pos[0],pos[1]-.08,pos[2]],.07,'gauge',(.85,.87,.80),u['id'])
        else: pos=[.35 if i%2==0 else 11.65,1.8+.45*(i//2), .35+(i%6)*.16]
        u['center']=pos;u['ports']['port'].update(pos=pos,normal=[0,0,0],stem=[0,0,1])
    return scene

def steel(scene,g):
    col=(.23,.35,.43); floor=(.53,.58,.60); rail=(.78,.62,.27)
    add_box(scene,'ground',[-1,-3.8,-.16,13,6.5,0],'floor',(.67,.69,.68))
    for x in (0,6,12):
        for y in (0,6): add_box(scene,f'column_{x}_{y}',[x-.10,y-.10,0,x+.10,y+.10,9.6],'column',col)
        add_box(scene,f'roof_crossbeam_{x}',[x-.1,0,9.5,x+.1,6,9.65],'beam',col)
    for y in (0,6):add_box(scene,f'roof_beam_{y}',[0,y-.1,9.5,12,y+.1,9.65],'beam',col)
    # Floors are split around actual equipment/pipe penetrations, preserving front access.
    for level,z in ((1,H),(2,2*H)):
        for y in (0,6): add_box(scene,f'beam_{level}_{y}',[0,y-.11,z-.30,12,y+.11,z-.10],'beam',col)
        for x in (0,6,12): add_box(scene,f'crossbeam_{level}_{x}',[x-.1,0,z-.30,x+.1,6,z-.10],'beam',col)
        holes=scene['holes'][level]
        xs=sorted({0.,12.,*[max(0,min(12,h[k])) for h in holes for k in (0,2)]})
        count=0
        for xa,xb in zip(xs,xs[1:]):
            if xb-xa<1e-6: continue
            cuts=sorted((max(0,h[1]),min(6,h[3])) for h in holes if h[0]<xb-1e-6 and h[2]>xa+1e-6)
            cursor=0
            for ya,yb in cuts+[(6,6)]:
                if ya>cursor+1e-5:
                    b=[xa,cursor,z-.10,xb,ya,z]; name=f'deck_{level}_{count}';count+=1
                    add_box(scene,name,b,'floor',floor);scene['surfaces'].append({'box':b,'level':level})
                cursor=max(cursor,yb)
        # Full edge protection, except the exterior stair doorway.
        for ya,yb in ((0,6),):
            for x in (0,12):
                add_pipe(scene,f'rail_side_{level}_{x}',[x,ya,z+1.1],[x,yb,z+1.1],.025,'rail',rail)
        for y in (0,6):
            xend=10.8 if y==0 else 12
            for zz in (.55,1.1): add_pipe(scene,f'rail_{level}_{y}_{zz}',[0,y,z+zz],[xend,y,z+zz],.023,'rail',rail)
            for x in range(0,int(xend)+1,2): add_pipe(scene,f'baluster_{level}_{y}_{x}',[x,y,z],[x,y,z+1.1],.023,'rail',rail)
            add_box(scene,f'toeboard_{level}_{y}',[0,y-.02,z,xend,y+.02,z+.10],'rail',rail)
        # Ring guard at equipment openings; lugs and shell close central aperture.
        for k,h in enumerate(holes):
            if h[2]-h[0]<.4: continue
            for a,b in [([h[0],h[1],z+.1],[h[2],h[1],z+.1]),([h[0],h[3],z+.1],[h[2],h[3],z+.1])]:
                add_pipe(scene,f'hole_curb_{level}_{k}_{len(scene["components"])}',a,b,.025,'curb',rail)
    scene['surfaces'].append({'box':[0,0,-.16,12,6,0],'level':0})
    # Two flights per storey, each ten equal rises, 1 m clear width.
    # External switchback parallel to the front wall; real landing and stringers.
    for level in (0,1):
        z0=level*H
        for flight in (0,1):
            y=-1.0 if flight==0 else -2.3
            start=11.7 if flight==0 else 8.9; direction=-1 if flight==0 else 1
            for i in range(10):
                xa=start+direction*i*.28;xb=xa+direction*.28;zt=z0+flight*1.8+(i+1)*.18
                add_box(scene,f'stair_{level}_{flight}_{i}',[min(xa,xb),y-.55,zt-.06,max(xa,xb),y+.55,zt],'step',floor)
            a=[start,y,z0+flight*1.8];b=[start+direction*2.8,y,z0+(flight+1)*1.8]
            for side in (-.525,.525):
                add_pipe(scene,f'stringer_{level}_{flight}_{side}',[a[0],y+side,a[2]-.1],[b[0],y+side,b[2]-.1],.06,'stringer',col)
                add_pipe(scene,f'stair_rail_{level}_{flight}_{side}',[a[0],y+side,a[2]+1.05],[b[0],y+side,b[2]+1.05],.025,'rail',rail)
                for i in (0,3,6,9):
                    xx=start+direction*(i+.5)*.28;zz=z0+flight*1.8+(i+1)*.18
                    add_pipe(scene,f'stair_baluster_{level}_{flight}_{side}_{i}',[xx,y+side,zz],[xx,y+side,zz+.96],.022,'rail',rail)
            scene['stairs'].append({'start':a,'end':b,'width':1.,'rise':.18,'run':.28,'steps':10})
        add_box(scene,f'half_landing_{level}',[7.8,-2.85,z0+1.7,8.9,-.45,z0+1.8],'landing',floor)
        add_box(scene,f'landing_{level}',[11.7,-2.85,z0+H-.10,12.8,0.65,z0+H],'landing',floor)
        add_box(scene,f'entry_bridge_{level}',[10.8,-.45,z0+H-.10,11.7,.65,z0+H],'landing',floor)
        edges=[([7.8,-2.85,z0+1.8],[7.8,-.45,z0+1.8]),([7.8,-2.85,z0+1.8],[8.9,-2.85,z0+1.8]),([7.8,-.45,z0+1.8],[8.9,-.45,z0+1.8]),
               ([12.8,-2.85,z0+H],[12.8,.65,z0+H]),([11.7,-2.85,z0+H],[12.8,-2.85,z0+H]),([10.8,-.45,z0+H],[11.7,-.45,z0+H]),([12,.65,z0+H],[12.8,.65,z0+H])]
        for i,(aa,bb) in enumerate(edges):
            for zz in (.55,1.1):add_pipe(scene,f'landing_guard_{level}_{i}_{zz}',[aa[0],aa[1],aa[2]+zz],[bb[0],bb[1],bb[2]+zz],.025,'rail',rail)
            for j,pp in enumerate((aa,bb)):add_pipe(scene,f'landing_post_{level}_{i}_{j}',pp,[pp[0],pp[1],pp[2]+1.1],.025,'rail',rail)
    for x in (7.8,8.9,11.7,12.8):
        for y in (-2.8,-.5): add_box(scene,f'stair_post_{x}_{y}',[x-.06,y-.06,0,x+.06,y+.06,7.2],'column',col)

def quat_from_axes(flow,stem):
    x=np.array(flow,float);x/=np.linalg.norm(x);z=np.array(stem,float)
    z-=np.dot(x,z)*x;z/=np.linalg.norm(z); y=np.cross(z,x)
    mat=np.column_stack([x,y,z]);import mujoco
    q=np.zeros(4);mujoco.mju_mat2Quat(q,mat.flatten());return q.tolist(),mat
