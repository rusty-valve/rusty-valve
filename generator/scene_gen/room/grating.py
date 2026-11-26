"""Cut openings in platform panels where pipes pass through."""
def grating(component):
    return component.get('kind')=='floor' and component['id'].startswith('deck_')


def overlaps(rect,box):
    return min(rect[2],box[3])-max(rect[0],box[0])>1e-8 and min(rect[3],box[4])-max(rect[1],box[1])>1e-8


def cut_for_parts(a,route,parts):
    from .piping_build import cut_pipe_openings
    holes=[]
    for level,z in enumerate(a.floors[1:],1):
        boards=[c['box'] for c in a.scene['components'] if grating(c) and abs(c['box'][5]-z)<1e-6]
        if not boards:continue
        for c in parts:
            box=c['box']
            if box[2]>=z-1e-8 or box[5]<=min(b[2] for b in boards)+1e-8:continue
            rect=[box[0]-.005,box[1]-.005,box[3]+.005,box[4]+.005]
            if not any(overlaps(rect,b) and box[5]>b[2]+1e-8 for b in boards):continue
            holes.append({'id':'hole_'+c['id']+'_'+str(level),'level':level,'rect':rect,
                'route':route['id'],'component':c['id'],'dn':route['dn'],'phase':route.get('phase','P1'),
                'source':'actual_pipe_grating','kind':'pipe'})
    if holes:
        cut_pipe_openings(a.scene,holes)
        a.openings.extend(holes)
        a.scene['piping_openings']=list(a.openings)
