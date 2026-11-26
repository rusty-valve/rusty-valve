"""Place shared racks, pipe connections and scale references."""
import math
import numpy as np
from ..layout import add_box, add_pipe
from .utilities import RULES, radius
from ..building import RULES as BUILDING_RULES, walking_height
from ..routing import reserved_box, line_hits


def resolve_feeds(g):
    for e in g['edges']:
        if e['role']!='feed':continue
        upstream=g['units'][e['a'].split('.')[0]]
        if upstream['kind']=='boundary' and not upstream.get('source_unit'):
            e['termination']=RULES['feed_connections']['unspecified_source']
            upstream.update(termination='blind',name='No feed source specified. The connection is blinded.')


def define_rack_zones(s, required_xy=None):
    cfg=RULES['distribution']['rack_zone'];floors=s['floors'];w,d=s['footprint']
    zones=[]
    for level,base in enumerate(floors):
        if level==len(floors)-1:
            bottom=base+cfg['top_floor_bottom_above_floor_m'];top=bottom+cfg['thickness_m']
        else:
            top=floors[level+1]-cfg['intermediate_floor_beam_depth_m'];bottom=top-cfg['thickness_m']
        points=(required_xy or {}).get(level,[])
        near=min((p[1] for p in points),default=d-.8)-cfg['plan_edge_padding_m']
        zones.append({'level':level,'box':[0,max(0,near),bottom,w,d,top],'thickness_m':top-bottom,
                      'source':'PG-04; floor elevations and required pipe crossings'})
    s['rack_zones']=zones
    return zones


def build_shared_rack(g,s):
    """Opposite edge frames carry full-span beams, selected from actual pipe runs."""
    cfg=RULES['distribution']['rack_support'];level=len(s['floors'])-1
    zone=s['rack_zones'][level]['box'];w,d=s['footprint'];base=s['floors'][level]
    steel=(.24,.31,.36);half=cfg['column_width_m']/2;bw=cfg['crossbeam_width_m']/2
    top=zone[2]-cfg['seat_min_height_m'];bottom=top-cfg['crossbeam_depth_m']
    rack={'level':level,'posts':[],'edge_girders':[],'crossbeams':[],'seats':[]}
    for i,(x,y) in enumerate(((0,0),(w,0),(0,d),(w,d))):
        c=add_box(s,f'shared_rack_post{i}',[x-half,y-half,base-.03,x+half,y+half,zone[5]],'beam',steel)
        rack['posts'].append(c['id'])
    for i,y in enumerate((0,d)):
        c=add_box(s,f'shared_rack_edge{i}',[-half,y-half,bottom-cfg['edge_girder_depth_m'],w+half,y+half,bottom],'beam',steel)
        rack['edge_girders'].append(c['id'])
        add_box(s,f'shared_rack_tie{i}',[-half,y-half,zone[5]-.20,w+half,y+half,zone[5]-.12],'beam',steel)
    pipes=[c for c in s['components'] if c['kind']=='pipe']
    horizontal=[c for c in pipes if abs(c['ends'][0][2]-c['ends'][1][2])<1e-7 and abs(c['ends'][0][1]-c['ends'][1][1])<1e-7 and abs(c['ends'][0][0]-c['ends'][1][0])>.1 and c['ends'][0][2]-c['radius']>=zone[2]-1e-6 and c['ends'][0][2]+c['radius']<=zone[5]+1e-6]
    groups={c['owner'] for c in horizontal};ranges={owner:(min(min(c['ends'][0][0],c['ends'][1][0]) for c in horizontal if c['owner']==owner),max(max(c['ends'][0][0],c['ends'][1][0]) for c in horizontal if c['owner']==owner)) for owner in groups}
    # A beam may carry several groups. Choose enough beams to support every run;
    # additional groups with disjoint x extents need additional full-span beams.
    candidates=[]
    for x in np.arange(.2,w-.19,.025):
        box=[x-bw,-half,bottom,x+bw,d+half,top]
        if any(line_hits(*c['ends'],box,c['radius']+.005) for c in pipes):continue
        covered={o for o,(lo,hi) in ranges.items() if lo+.07<=x<=hi-.07}
        if covered:candidates.append((float(x),covered))
    counts={o:0 for o in groups};selected=[]
    required={o:min(cfg['crossbeam_count'],max(1,int((hi-lo-.14)/.4)+1)) for o,(lo,hi) in ranges.items()}
    while any(n<required[o] for o,n in counts.items()):
        options=[(sum(counts[o]<required[o] for o in covered),min((abs(x-v) for v in selected),default=w),-abs(x-w/2),x,covered) for x,covered in candidates if all(abs(x-v)>2*cfg['crossbeam_width_m'] for v in selected)]
        if not options:raise ValueError('The rack beam cannot clear risers and support every horizontal pipe')
        score,_,_,x,covered=max(options,key=lambda r:r[:4])
        if not score:raise ValueError(f'No supporting location for the rack beam: {[(o,counts[o],required[o],ranges[o]) for o in groups if counts[o]<required[o]]}')
        selected.append(x)
        for o in covered:counts[o]+=1
    for i,x in enumerate(sorted(selected)):
        beam=add_box(s,f'shared_rack_cross{i}',[x-bw,-half,bottom,x+bw,d+half,top],'beam',steel);rack['crossbeams'].append(beam['id'])
        for c in horizontal:
            a,b=c['ends'];lo,hi=sorted((a[0],b[0]))
            if not lo+.001<x<hi-.001:continue
            y=a[1];seat=add_box(s,f'shared_rack_seat{i}_{c["id"]}',[x-bw,y-bw,top,x+bw,y+bw,a[2]-c['radius']],'hanger',steel)
            row={'beam':beam['id'],'seat':seat['id'],'pipe':c['id'],'edge':c['owner']};rack['seats'].append(row)
            edge=next((e for e in g['edges'] if e['id']==c['owner']),{})
            if edge.get('utility_tier')=='floor_branch':s['header_supports'].append({'service':edge['service'],'arm':beam['id'],'seat':seat['id'],'level':level})
    s['shared_rack']=rack


def scale_person(s, lev, p):
    cfg=BUILDING_RULES['human_scale'];h=cfg['height_m']
    start=len(s['components']);name=f'person_L{lev}';scale=h/1.75
    blue=cfg['color_rgb'];dark=(.13,.17,.21);skin=(.73,.60,.46);yellow=(.96,.72,.08)
    def line(label,a,b,r,col):
        return add_pipe(s,name+'_'+label,(np.array(p)+np.array(a)*scale).tolist(),(np.array(p)+np.array(b)*scale).tolist(),r*scale,'scale_person',col)
    def box(label,lo,hi,col):
        return add_box(s,name+'_'+label,[*(np.array(p)+np.array(lo)*scale),*(np.array(p)+np.array(hi)*scale)],'scale_person',col)
    for side in (-1,1):
        sx=.11*side
        box('boot'+str(side),[sx-.065,-.13,0],[sx+.065,.10,.10],dark)
        line('leg'+str(side),[sx,0,.10],[sx,0,.88],.07,blue)
        line('upper_arm'+str(side),[side*.20,0,1.37],[side*.235,0,1.10],.052,blue)
        line('forearm'+str(side),[side*.235,0,1.10],[side*.24,-.015,.87],.045,blue)
        line('hand'+str(side),[side*.24,-.015,.87],[side*.24,-.015,.80],.04,skin)
    box('body',[-.18,-.105,.84],[.18,.105,1.43],blue)
    line('neck',[0,0,1.43],[0,0,1.52],.055,skin)
    for label,zz,size,col in [('head',1.59,[.09,.095,.12],skin),('helmet',1.70,[.12,.13,.05],yellow)]:
        pos=(np.array(p)+np.array([0,0,zz])*scale).tolist();sizes=(np.array(size)*scale).tolist()
        s['components'].append({'id':name+'_'+label,'kind':'scale_person','shape':'ellipsoid','pos':pos,'size':sizes,'box':[*[pos[i]-sizes[i] for i in range(3)],*[pos[i]+sizes[i] for i in range(3)]],'color':col})
    parts=s['components'][start:]
    for c in parts:c.update(visual_only=True,annotation_only=True,level=lev)
    return {'id':name,'level':lev,'feet':p,'height_m':h,'components':[c['id'] for c in parts],'annotation_only':True}


def add_scale_people(g,s):
    from ..checks import navigation, intersects, STEP
    cfg=BUILDING_RULES['human_scale'];h=cfg['height_m'];w=cfg['body_width_m'];d=cfg['body_depth_m']
    obs=[c['box'] for c in s['components'] if not c.get('visual_only') and 'walk_level' not in c]
    obs += [reserved_box(v) for v in g['valves']]
    width,depth=s['footprint'];s['scale_people']=[]
    for lev,_ in enumerate(s['floors']):
        nav,_=navigation(s,obs,lev)
        entry=s['navigation_entries'][str(lev)]
        candidates=[]
        for x in np.arange(.7,width-.5,.4):
            for y in np.arange(.7,depth-.5,.4):
                if not nav[int(round(x/STEP)),int(round(y/STEP))] or math.dist([x,y],entry)<1.2:continue
                if any(v.get('stance') and v['level']==lev and math.dist([x,y],v['stance'][:2])<.8 for v in g['valves']):continue
                z=walking_height(s,lev,x,y,w/2,d/2);box=[x-w/2,y-d/2,z+.004,x+w/2,y+d/2,z+h]
                if any(intersects(box,np.asarray(obs),.04)):continue
                candidates.append((abs(x-width*.25)+abs(y-depth*.22),[float(x),float(y),float(z)]))
        if not candidates:raise ValueError(f'{lev} has no free scale-reference stance')
        s['scale_people'].append(scale_person(s,lev,min(candidates)[1]))
    g['scale_people']=s['scale_people']


def check(g,s,model=None):
    errors=[];rows=[];cs={c['id']:c for c in s['components']}
    rack=s.get('shared_rack');rack_rows=[]
    if not rack:errors.append({'code':'SHARED_RACK_MISSING'})
    else:
        w,d=s['footprint'];base=s['floors'][rack['level']];zone=s['rack_zones'][rack['level']]['box']
        if len(rack['posts'])!=4 or len(rack['edge_girders'])!=2:errors.append({'code':'RACK_OPPOSITE_EDGE_FRAME'})
        for name,(x,y) in zip(rack['posts'],((0,0),(w,0),(0,d),(w,d))):
            c=cs.get(name);box=c['box'] if c else [0]*6
            if not c or abs((box[0]+box[3])/2-x)>1e-6 or abs((box[1]+box[4])/2-y)>1e-6 or not box[2]<=base<=box[5]:errors.append({'code':'RACK_EDGE_POST','part':name})
        for name,y in zip(rack['edge_girders'],(0,d)):
            c=cs.get(name);box=c['box'] if c else [0]*6
            if not c or abs((box[1]+box[4])/2-y)>1e-6 or box[0]>0 or box[3]<w:errors.append({'code':'RACK_OPPOSITE_EDGE_FRAME','part':name})
        for name in rack['crossbeams']:
            beam=cs.get(name);okay=bool(beam)
            if beam:
                box=beam['box'];okay=box[1]<=0 and box[4]>=d
                for edge in rack['edge_girders']:
                    c=cs.get(edge);okay=okay and bool(c) and abs(c['box'][5]-box[2])<1e-6 and c['box'][0]<=box[0] and box[3]<=c['box'][3]
            if not okay:errors.append({'code':'RACK_NOT_FULL_SPAN','part':name})
        coverage={}
        for row in rack['seats']:
            beam=cs.get(row['beam']);seat=cs.get(row['seat']);pipe=cs.get(row['pipe']);okay=all((beam,seat,pipe))
            if okay:
                a,b=pipe['ends'];box=seat['box'];x=(box[0]+box[3])/2
                okay=min(a[0],b[0])<=x<=max(a[0],b[0]) and box[1]<=a[1]<=box[4] and abs(box[5]-(a[2]-pipe['radius']))<1e-6 and abs(box[2]-beam['box'][5])<1e-6
            if not okay:errors.append({'code':'RACK_SEAT_CONTACT','seat':row['seat']})
            else:coverage.setdefault(row['edge'],set()).add(row['beam'])
        expected={c['owner'] for c in s['components'] if c['kind']=='pipe' and abs(c['ends'][0][2]-c['ends'][1][2])<1e-7 and abs(c['ends'][0][1]-c['ends'][1][1])<1e-7 and abs(c['ends'][0][0]-c['ends'][1][0])>.1 and c['ends'][0][2]-c['radius']>=zone[2]-1e-6 and c['ends'][0][2]+c['radius']<=zone[5]+1e-6}
        for edge in expected:
            beams=coverage.get(edge,set())
            rack_rows.append({'edge':edge,'shared_beams':sorted(beams)})
            runs=[c for c in s['components'] if c['kind']=='pipe' and c.get('owner')==edge and abs(c['ends'][0][2]-c['ends'][1][2])<1e-7 and abs(c['ends'][0][1]-c['ends'][1][1])<1e-7 and c['ends'][0][2]-c['radius']>=zone[2]-1e-6]
            span=max(max(p[0] for p in c['ends']) for c in runs)-min(min(p[0] for p in c['ends']) for c in runs)
            required=min(RULES['distribution']['rack_support']['crossbeam_count'],max(1,int((span-.14)/.4)+1))
            if len(beams)<required:errors.append({'code':'RACK_PIPE_SUPPORT_COUNT','edge':edge})
        if not coverage or not any(sum(name in beams for beams in coverage.values())>1 for name in rack['crossbeams']):errors.append({'code':'RACK_NOT_SHARED'})
    for e in g['edges']:
        if e['role']!='feed':continue
        source=g['units'][e['a'].split('.')[0]]
        if source['kind']!='boundary' or source.get('source_unit'):continue
        v=next(v for v in g['valves'] if v['edge']==e['id']);end=np.array(v['center'])+np.array(v['axis'])*v['length']/2
        cap=cs.get(e['id']+'_feed_joint_blind');adapter=cs.get(e['id']+'_feed_joint_thread_adapter')
        tails=[c['id'] for c in s['components'] if c.get('owner')==e['id'] and c['kind']=='pipe']
        okay=bool(cap and adapter) and math.dist(adapter['ends'][0],end)<1e-6 and not tails
        rows.append({'feed':e['id'],'blind':cap['id'] if cap else None,'free_tail':tails,'pass':bool(okay)})
        if not okay:errors.append({'code':'UNASSIGNED_FEED_NOT_BLIND','edge':e['id']})
    crossings=[];edges={e['id']:e for e in g['edges']}
    for route in s['routing']:
        e=edges[route['edge']]
        if e['role'] in ('vapor','process_evaporated_vapor'):continue
        pts=[np.array(p) for p in route['points']]
        for i,(a,b) in enumerate(zip(pts,pts[1:])):
            levels=[lev for lev,z in enumerate(s['floors']) if lev and a[2]<z-1e-6 and b[2]>z+1e-6]
            if not levels:continue
            horizontal=next(((u,v) for u,v in zip(pts[i+1:],pts[i+2:]) if abs(u[2]-v[2])<1e-6 and np.linalg.norm((u-v)[:2])>1e-6),None)
            if horizontal is None:continue  # A continuing vertical riser has no distribution turn here.
            crossings.append({'edge':e['id'],'level':max(levels),'dn':e['dn'],'horizontal':horizontal})
    for crossing in crossings:
        zone=s['rack_zones'][crossing['level']]['box'];r=radius(crossing['dn']);a,b=crossing['horizontal']
        parts=[c for c in s['components'] if c.get('owner')==crossing['edge'] and c['kind']=='pipe']
        actual=any(math.dist(c['ends'][0],a)<1e-6 and math.dist(c['ends'][1],b)<1e-6 for c in parts)
        inside=all(zone[i]-1e-6<=min(a[i],b[i])-r and max(a[i],b[i])+r<=zone[i+3]+1e-6 for i in range(3))
        if not actual or not inside:errors.append({'code':'UPWARD_PIPE_OUTSIDE_RACK','edge':crossing['edge']})
        rows.append({'edge':crossing['edge'],'rack_level':crossing['level'],'horizontal_pipe_z_m':float(a[2]),'pass':bool(actual and inside)})
    people=s.get('scale_people',[])
    for lev,_ in enumerate(s['floors']):
        persons=[p for p in people if p['level']==lev]
        if len(persons)!=BUILDING_RULES['human_scale']['required_per_floor']:errors.append({'code':'SCALE_PERSON_PER_FLOOR','level':lev})
        for p in persons:
            parts=[cs[n] for n in p['components'] if n in cs]
            height=max(c['box'][5] for c in parts)-min(c['box'][2] for c in parts) if parts else 0
            if abs(height-p['height_m'])>1e-6:errors.append({'code':'SCALE_PERSON_HEIGHT','level':lev})
            if parts:
                foot=min(c['box'][2] for c in parts);x,y,_=p['feet'];cfg=BUILDING_RULES['human_scale']
                ground=walking_height(s,lev,x,y,cfg['body_width_m']/2,cfg['body_depth_m']/2)
                if abs(foot-ground)>.004:errors.append({'code':'SCALE_PERSON_NOT_GROUNDED','level':lev})
            if model is not None and any(model.geom(n).contype or model.geom(n).conaffinity for n in p['components']):errors.append({'code':'SCALE_PERSON_HAS_COLLISION','level':lev})
    return {'status':'pass' if not errors else 'fail','errors':errors,'measurements':rows,'shared_rack_supports':rack_rows,'rack_zones':s['rack_zones'],'people_per_floor':[sum(p['level']==lev for p in people) for lev in range(len(s['floors']))],'scale_height_m':BUILDING_RULES['human_scale']['height_m']}
