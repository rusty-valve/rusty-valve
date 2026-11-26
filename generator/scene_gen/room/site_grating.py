"""Build platform grating and check its openings."""
import numpy as np

from . import building
from .checks import geom_bounds
from .layout import add_box
from .building_parts.basic import Mesh
from .saved_geometry import mesh
from .shell import _subtract


DEPTH = .10  # Preserve the deck underside datum on the existing steel beams.
PIPE_GAP = .025
EPS = 1.e-6
STEEL_SUPPORT = {'beam', 'column', 'equipment_beam'}
LAYOUT_ONLY = {'catalogue_equipment_envelope', 'drive_reservation',
               'pullout_reservation', 'maintenance_reservation'}


def area(rect):
    return (rect[2]-rect[0])*(rect[3]-rect[1])


def contains(outer, inner):
    return (min(inner[0]-outer[0], inner[1]-outer[1], outer[2]-inner[2], outer[3]-inner[3]) >= -EPS)


def overlap(a, b):
    return min(a[2], b[2])-max(a[0], b[0]) > EPS and min(a[3], b[3])-max(a[1], b[1]) > EPS


def geometry_index(scene, model, data, excluded=()):
    parts = {p['id']:p for p in scene['components']}
    names = [model.geom(i).name for i in range(model.ngeom)]
    ids = []
    for gid, name in enumerate(names):
        part = parts.get(name, {})
        kind = part.get('kind')
        if name in excluded or kind in LAYOUT_ONLY or part.get('annotation_only'):
            continue
        if part.get('visual_only') and kind != 'insulation':
            continue
        if name.endswith('_mating_envelope'):
            continue
        ids.append(gid)
    return np.asarray(ids, dtype=int), geom_bounds(model, data, ids), names


def world_mesh(model, data, gid, cache):
    vertices, faces = mesh(model, gid, cache)
    return vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid], faces


def slab_section(vertices, faces, lower, upper):
    z = vertices[:, 2]
    if z.min() >= upper-EPS or z.max() <= lower+EPS:
        return None
    points = [vertices[(z >= lower) & (z <= upper), :2]]
    for first, second in ((0, 1), (1, 2), (2, 0)):
        a, b = vertices[faces[:, first]], vertices[faces[:, second]]
        za, zb = a[:, 2], b[:, 2]
        for plane in (lower, upper):
            cut = (za < plane) != (zb < plane)
            fraction = (plane-za[cut])/(zb[cut]-za[cut])
            points.append(a[cut, :2]+fraction[:, None]*(b[cut, :2]-a[cut, :2]))
    xy = np.concatenate(points)
    return np.r_[xy.min(axis=0), xy.max(axis=0)] if len(xy) else None


def sections(scene, model, data, excluded, z, *, bottom=None):
    parts = {p['id']:p for p in scene['components']}
    ids, boxes, names = geometry_index(scene, model, data, excluded)
    cache = {}
    bottom = z-DEPTH if bottom is None else bottom
    equipment_ids = {u['id'] for u in scene.get('equipment_instances', ())}
    for gid in ids[(boxes[:, 2] < z-EPS) & (boxes[:, 5] > bottom+EPS)]:
        name = names[gid]
        part = parts.get(name, {})
        kind = part.get('kind', 'valve')
        if kind in STEEL_SUPPORT:
            continue
        rect = slab_section(*world_mesh(model, data, gid, cache), bottom, z)
        if rect is not None:
            pipe = kind.startswith('piping_') or kind in ('pipe', 'header') or part.get('piping', False)
            if kind == 'insulation':
                pipe = part.get('owner') not in equipment_ids
            yield name, rect, 'pipe' if pipe else 'solid'


def opening_count(rects):
    groups = []
    for rect in rects:
        touches = [group for group in groups if any(overlap(rect, other) for other in group)]
        groups = [group for group in groups if group not in touches]
        groups.append([rect, *(p for group in touches for p in group)])
    return len(groups)


def grid(axes, pitch):
    coords = []
    for a, b in zip(axes, axes[1:]):
        coords.extend(np.linspace(a,b,max(1,round((b-a)/pitch))+1)[:-1])
    return [*coords, axes[-1]]


def trim_support(scene, name, rect, lower, upper, axis, clear, *, section=True):
    names = []
    for i, r in enumerate(building.subtract(rect,clear,min_size=.008)):
        box = [*r[:2],lower,*r[2:],upper]
        if section:
            parts = building.i_member(scene, f'{name}_{i}', box, axis, flange=.008, web=.008)
        else:
            parts = [add_box(scene, f'{name}_{i}', box, 'beam', building.STEEL)]
        names.extend(part['id'] for part in parts)
    return names


def bearing_width(rect, seats):
    return sum(max(0.,min(rect[2],r[2])-max(rect[0],r[0]))
               for r in seats if min(rect[3],r[3])-max(rect[1],r[1]) >= .025-EPS)


def grid_panels(level, xs, ys, gx, gy, openings):
    plates = []
    for i,(x0,x1) in enumerate(zip(xs,xs[1:])):
        for j,(y0,y1) in enumerate(zip(ys,ys[1:])):
            cell = [x0+gx[i]/2,y0+gy[j]/2,x1-gx[i+1]/2,y1-gy[j+1]/2]
            for k, rect in enumerate(building.subtract(cell,openings,min_size=.035)):
                plates.append((f'site_deck_{level}_{i}_{j}_{k}',rect,i,j))
    return plates


def blocked_bearings(plates, holes):
    blocked=[]
    for _,rect,i,j in plates:
        near=[hole for hole in holes if overlap(rect,hole)]
        if not near or not building.subtract(rect,near):
            continue
        for y in (rect[1],rect[3]-.035):
            end=[rect[0],y,rect[2],y+.035]
            remaining=building.subtract(end,near,min_size=.002)
            if not any(r[3]-r[1]>=.025-EPS for r in remaining):
                blocked.append((i,j))
    return blocked


def fit_panel_grid(level, xs, ys, gx, gy, openings, holes):
    """Shift a shared seam when a pipe opening removes all of a board's bearing."""
    axes=[list(xs),list(ys)]
    def layout(grid):
        return grid_panels(level,*grid,gx,gy,openings)
    plates=layout(axes)
    blocked=blocked_bearings(plates,holes)
    for _ in range(len(blocked)):
        seams={(axis,row[axis]+end) for row in blocked for axis in (0,1) for end in (0,1)
               if 0<row[axis]+end<len(axes[axis])-1}
        best=None
        for axis,index in sorted(seams):
            for delta in (.06,-.06,.12,-.12,.18,-.18,.24,-.24):
                grid=[values.copy() for values in axes]
                grid[axis][index]+=delta
                if min(np.diff(grid[axis][index-1:index+2])) < (.35 if axis==0 else .60):
                    continue
                revised=layout(grid)
                pending=blocked_bearings(revised,holes)
                score=(len(pending),abs(delta))
                if len(pending)<len(blocked) and (best is None or score<best[0]):
                    best=(score,grid,revised,pending)
                if not pending:
                    break
            if best is not None and best[0][0]==0:
                break
        if best is None:
            break
        _,axes,plates,blocked=best
        if not blocked:
            break
    return axes,plates


def lay(scene, model, data, excluded, folder, seed):
    width, length = scene['footprint']
    boundary = [0., 0., width, length]
    floors, violations, changes = [], [], []
    nominal_x = grid(scene['shell']['x_axes_m'], .60)
    nominal_y = grid(scene['shell']['y_axes_m'], 1.0)
    for level, z in enumerate(scene['floors'][1:], 1):
        rng = np.random.default_rng(np.random.SeedSequence([level, *seed]))
        openings = []
        for name, rect, kind in sections(scene, model, data, excluded, z+.01, bottom=z-.045):
            gap = PIPE_GAP if kind == 'pipe' else .002
            cut = (rect+[-gap, -gap, gap, gap]).tolist()
            if overlap(cut, boundary):
                openings.append({'part': name, 'kind': kind, 'rect': cut, 'section': rect.tolist()})
        pullouts = [row for row in scene.get('pullout_reservations', ()) if z > row['base_z']+EPS]
        holes = [row['rect'] for row in openings if row['kind'] == 'pipe']
        solids = [row['rect'] for row in openings if row['kind'] == 'solid']
        cover_rects = [row['rect'] for row in pullouts]
        before = len(scene['components'])
        gx = [0., *rng.uniform(.010,.015,len(nominal_x)-2), 0.]
        gy = [0., *rng.uniform(.010,.015,len(nominal_y)-2), 0.]
        (xs,ys),raw_plates=fit_panel_grid(level,nominal_x,nominal_y,gx,gy,solids+cover_rects,holes)
        plates = [(name,rect,False) for name,rect,_,_ in raw_plates]
        plates.extend((f'site_pullout_{level}_{row["id"]}',row['rect'],True) for row in pullouts)
        eligible = [i for i,(_,r,cover) in enumerate(plates) if not cover
                    and r[2]-r[0]>.30 and r[3]-r[1]>.44 and not any(overlap(r,h) for h in holes)]
        count = min(len(eligible), int(rng.integers(0,4)))
        variants = dict(zip(rng.choice(eligible,size=count,replace=False), range(count))) if count else {}
        clearance = [(r+[-.003,-.003,.003,.003]).tolist()
                     for _,r,_ in sections(scene,model,data,excluded,z-.019,bottom=z-.147)]
        for axis, coords in ((1,xs),(0,ys)):
            for i, pos in enumerate(coords):
                line = list(boundary)
                line[1-axis], line[3-axis] = pos-.035, pos+.035
                trim_support(scene,f'site_joist_{level}_{axis}_{i}',line,z-.145,z-.062,axis,clearance)
        mesh, covers, panels, records = Mesh(), [], [], []
        for index, (name,rect,cover) in enumerate(plates):
            cut = holes+solids if cover else holes
            contact = building.subtract(rect,cut)
            if not contact:
                continue
            raised = float(rng.uniform(.005,.010)) if variants.get(index,-1)%2==0 else 0.
            sag = float(rng.uniform(.005,.015)) if index in variants and variants[index]%2 else 0.
            y0,y1 = rect[1],rect[3]
            splits = [y0,y0+.04,(y0+y1)/2,y1-.04,y1] if sag else [y0,y1]
            def top(y):
                return z+raised-sag*max(0.,1-abs(2*(y-y0-.04)/max(y1-y0-.08,.01)-1))
            target = Mesh() if cover else mesh
            pieces = building.panel(scene,folder,Mesh,target,name,rect,top,splits,cut,level)
            if not pieces:
                continue
            if cover:
                target.save(scene,folder,name,'grating',building.SILVER)
                covers.append({'id':name,'rect':rect})
            if raised or sag:
                changes.append({'part':name,'kind':'platform_step' if raised else 'platform_sag',
                                'mm':round(1000*(raised or sag),2)})
            panels.extend(contact)
            seats = []
            trimmed = any(min(abs(edge-y) for y in ys) > .016 for edge in (rect[1],rect[3]))
            if trimmed:
                # Locally cut boards still reach a transverse beam through
                # their longitudinal edge bearers, as on the old platform.
                lo = max((y for y in ys if y <= rect[1]+EPS), default=rect[1])
                hi = min((y for y in ys if y >= rect[3]-EPS), default=rect[3])
                for side,x in enumerate((rect[0],rect[2]-.035)):
                    trim_support(scene,name+f'_edge_bearer{side}',[x,lo,x+.035,hi],
                                 z-.145,z-.062,1,clearance)
            for end, y in enumerate((rect[1],rect[3]-.035)):
                bearing = [rect[0]-.018,y,rect[2]+.018,y+.035]
                # The bearing strip sits on the ties below it; both are trimmed
                # to the frozen occupied sections, with a separate seat height for a raised panel.
                trim_support(scene,name+f'_trimmer{end}',bearing,z-.145,z-.062,0,clearance)
                seat = trim_support(scene,name+f'_seat{end}',bearing,z-.062,z+raised-.03,0,clearance,section=False)
                seats.append(seat)
            if cover:
                for side,x in enumerate((rect[0],rect[2]-.035)):
                    bearing = [x,rect[1],x+.035,rect[3]]
                    trim_support(scene,name+f'_cover_edge{side}',bearing,z-.145,z-.03,1,clearance)
            records.append({'id':name,'plates':pieces,'rect':rect,'seats':seats,'cover':cover})
        if mesh.v:
            mesh.save(scene, folder, f'site_grating_{level}', 'grating', building.SILVER)
        for part in scene['components'][before:]:
            if part['kind']=='walk_surface':
                part['alpha']=0.
        cuts = [row['rect'] for row in openings]
        for opening in openings:
            if any(overlap(opening['rect'], panel) for panel in panels):
                violations.append({'code':'DECK_OPENING_BLOCKED','level':level,'part':opening['part']})
        floors.append({'level':level,'z_m':z,'area_m2':sum(map(area,panels)),
                       'panels':panels,'boards':records,'covers':covers,'openings':openings,
                       'opening_count':opening_count(cuts)})
    return floors, violations, changes


def support_network(scene, model, data):
    """Find the steel physically connected to the main frame."""
    parts = [p for p in scene['components']
             if p['kind'] in STEEL_SUPPORT and p['shape']=='box' and not p.get('pose')]
    bounds = geom_bounds(model,data,[model.geom(p['id']).id for p in parts])
    reached = np.zeros(len(parts),dtype=bool)
    queue = [i for i,p in enumerate(parts) if p['id'].startswith(('beam_','crossbeam_','column_'))]
    reached[queue] = True
    while queue:
        i = queue.pop()
        lo = np.maximum(bounds[i,:3],bounds[:,:3])
        hi = np.minimum(bounds[i,3:],bounds[:,3:])
        width = hi-lo
        adjacent = (~reached & (width >= -EPS).all(axis=1) & ((width > EPS).sum(axis=1)>=2)).nonzero()[0]
        reached[adjacent] = True
        queue.extend(adjacent.tolist())
    return {p['id']:box for p,box,ok in zip(parts,bounds,reached) if ok}


def audit(scene, model, data, floors):
    connected = support_network(scene,model,data)
    cache = {}
    violations = []
    for floor in floors:
        for board in floor['boards']:
            plates = [world_mesh(model,data,model.geom(name).id,cache)[0] for name in board['plates']]
            actual = [np.r_[v.min(axis=0),v.max(axis=0)] for v in plates]
            for end,seats in enumerate(board['seats']):
                attached = []
                for name in seats:
                    if name not in connected:
                        continue
                    box = connected[name]
                    rect = [*box[:2],*box[3:5]]
                    for points, slab in zip(plates,actual):
                        if abs(slab[2]-box[5]) <= 2.e-6 and overlap(rect,[*slab[:2],*slab[3:5]]):
                            close = points[np.abs(points[:,2]-box[5]) < 2.e-6]
                            if len(close) and (np.ptp(close[:,1]) > .025-EPS):
                                attached.append(rect)
                                break
                if bearing_width(board['rect'],attached) <= EPS:
                    violations.append({'code':'PANEL_SUPPORT','part':board['id'],'end':end})
    return violations
