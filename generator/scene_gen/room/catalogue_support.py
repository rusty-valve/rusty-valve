"""Fit support shoes under reactor mounting lugs."""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from .catalogue_geometry import _load
from .grating import grating
from .piping_build import _subtract_rect
from .reactor_f_series_shell import _prism


EPS = 1e-6


@lru_cache(maxsize=3)
def _sole_polygons(xml):
    """Get the bottom faces of the convex shoe-pad sectors from the compiled source."""
    model, data = _load(Path(xml))
    bearings = defaultdict(list)
    for i in range(model.ngeom):
        name = model.geom(i).name
        match = re.fullmatch(r'.+_lug_(\d+)_pad_\d+', name)
        if not match:
            continue
        m = model.geom_dataid[i]
        start = model.mesh_vertadr[m]
        count = model.mesh_vertnum[m]
        verts = model.mesh_vert[start:start+count] @ data.geom_xmat[i].reshape(3,3).T + data.geom_xpos[i]
        bottom = verts[verts[:,2] < verts[:,2].min()+EPS]
        bearings[int(match.group(1))].append(bottom[ConvexHull(bottom[:,:2]).vertices])
    return dict(bearings)


def _area(poly):
    if len(poly) < 3:
        return 0.
    rel = np.asarray(poly) - poly[0]
    return abs(float(rel[:,0]@np.roll(rel[:,1],-1)-rel[:,1]@np.roll(rel[:,0],-1)))/2


def _clip(poly, rect):
    """Intersection of an actual convex pad face with an axis-aligned plate."""
    for axis, limit, side in ((0,rect[0],1), (1,rect[1],1), (0,rect[2],-1), (1,rect[3],-1)):
        if len(poly) < 3:
            return []
        result = []
        a = poly[-1]
        for b in poly:
            inside_a = (a[axis]-limit)*side >= 0
            inside_b = (b[axis]-limit)*side >= 0
            if inside_a != inside_b:
                result.append(a+(b-a)*((limit-a[axis])/(b[axis]-a[axis])))
            if inside_b:
                result.append(b)
            a = b
        poly = result
    return poly


def _rect(box):
    return [box[i] for i in (0,1,3,4)]


def _fit_into_deck(scene, old_id, rect, height):
    """Remove only the overlapping deck volume; the shoe fills it flush to the same datum."""
    parts = []
    for part in scene['components']:
        if part['id'] == old_id:
            continue
        if not grating(part) or abs(part['box'][5]-height)>EPS:
            parts.append(part)
            continue
        box = part['box']
        whole = _rect(box)
        pieces = _subtract_rect(whole, rect)
        if pieces == [whole]:
            parts.append(part)
            continue
        for i, (x0,y0,x1,y1) in enumerate(pieces):
            tile = deepcopy(part)
            tile.update(id=f'{part["id"]}_{old_id}_tile{i}',box=[x0,y0,box[2],x1,y1,box[5]])
            parts.append(tile)
    scene['components'] = parts


def _shoe_sectors(polys, center, diameter, rect):
    # All small-radius vertices, and only these, are on the actual pad's drilled bore.
    points = np.vstack(polys)
    bore = points[np.linalg.norm(points-center, axis=1) < diameter*.55]
    bore = np.unique(np.round(bore,7),axis=0)
    loop = bore[ConvexHull(bore).vertices]
    x0,y0,x1,y1 = rect
    vertices = np.vstack((loop, [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]))
    faces = Delaunay(vertices).simplices
    triangles = [vertices[t] for t in faces if np.any(t>=len(loop))]
    target = (x1-x0)*(y1-y0)-_area(loop)
    if abs(sum(_area(t) for t in triangles)-target)>1.e-7:
        raise ValueError('The support shoe exterior and physical bores are not fully partitioned')
    return triangles


def install_reactor_supports(assembly, device):
    """Fit four Catalogue-sized shoes flush to the deck. Return one measured row per shoe."""
    uid = device['id']
    matrix = np.asarray(device['rotation_matrix'])
    origin = np.asarray(device['origin'])
    bearings = _sole_polygons(device['source'])
    rows = device['facts']['bearing_lugs']
    if len(rows)!=4 or set(bearings)!={r['bearing_deg'] for r in rows}:
        raise ValueError(f'{uid} Missing four physical lugs')
    folder = assembly.folder/'equipment_supports'
    folder.mkdir(exist_ok=True)
    installed = []
    for number, row in enumerate(rows, 1):
        bearing = row['bearing_deg']
        source = [v@matrix.T+origin for v in bearings[bearing]]
        polys = [v[:,:2] for v in source]
        center = np.asarray(row['hole_center_m'])@matrix.T+origin
        xy = center[:2]
        radial = xy-origin[:2]
        if min(abs(radial))>EPS:
            raise ValueError(f'{uid} of {bearing} degree mounting hole does not point toward the midpoint of the platform edge')
        shoe_id = f'{uid}_lug_shoe{number}'
        old = next(p for p in assembly.scene['components'] if p['id']==shoe_id)
        side = np.asarray(old['box'][3:])-old['box'][:3]
        width, length = sorted(side[:2])
        span = np.array([width,length]) if abs(radial[0])>abs(radial[1]) else np.array([length,width])
        rect = np.r_[xy-span/2,xy+span/2].tolist()
        height = float(old['box'][5])
        level_gap = max(abs(height-v[2]) for poly in source for v in poly)
        if level_gap>EPS:
            raise ValueError(f'{shoe_id} The shoe top and lug are not coplanar')
        decks = [p for p in assembly.scene['components'] if grating(p) and abs(p['box'][5]-height)<EPS]
        supported = []
        shoe_footprint = 0.
        for p in decks:
            x0,y0,x1,y1 = _rect(p['box'])
            exposed = sum(_area(_clip(poly,(x0,y0,x1,y1))) for poly in polys)
            if exposed>EPS**2:
                supported.append((p['id'],exposed))
            shoe_footprint += max(0.,min(x1,rect[2])-max(x0,rect[0]))*max(0.,min(y1,rect[3])-max(y0,rect[1]))
        triangles = _shoe_sectors(polys,xy,row['clear_hole_diameter_m'],rect)
        _fit_into_deck(assembly.scene,shoe_id,rect,height)
        shoe_parts = []
        for i,t in enumerate(triangles):
            bottom = np.column_stack((t,np.full(3,height-side[2])))
            part = _prism(assembly.scene,folder,f'CAT_{shoe_id}_sector{i:02}','support',
                          bottom, np.array([0.,0.,side[2]]),old['color'],
                          'original support shoe dimensions and measured lug bearing faces and holes','scene support shoe',
                          'The replacement plate has the original thickness. Mount holes align with the lugs, and the shoe meets the platform.',uid)
            shoe_parts.append(part['id'])
        contact = sum(_area(_clip(poly,rect)) for poly in polys)
        total = sum(_area(poly) for poly in polys)
        covered = sum(v for _,v in supported)
        beam = _support_under_shoe(assembly.scene, shoe_id, rect, height-float(side[2]), radial, uid)
        installed.append({'device':uid, 'source_bearing_deg':bearing,'hole_center_world_m':center.tolist(),
                          'hole_diameter_m':row['clear_hole_diameter_m'], 'old_shoe_id':shoe_id,
                          'shoe_components':shoe_parts, 'shoe_rect_m':rect, 'support_beam':beam, 'shoe_thickness_m':float(side[2]),
                          'shoe_top_z_m':height, 'max_vertical_gap_m':level_gap,
                          'sole_area_m2':total,'shoe_contact_area_m2':contact,
                          'adjacent_deck_contact_area_m2':covered-contact,
                          'bearing_contact_area_m2':covered,'contact_fraction':covered/total,
                          'deck_components_before_inlay':[name for name,_ in supported],
                          'mounting_scope':'Match the existing mounting plane and plate thickness, with the shoe meeting the platform edge. Loads, welds and anchors need separate checks.',
                          'pass':bool(contact>0. and level_gap<EPS)})
    return installed



def _support_under_shoe(scene, shoe_id, rect, shoe_bottom, radial, owner):
    from .layout import add_box

    def overlaps(box):
        return all(min(rect[i+2], box[i+3]) > max(rect[i], box[i]) + EPS for i in (0, 1))

    beams = [c for c in scene['components'] if c['kind'] in ('beam','equipment_beam')]
    for beam in beams:
        if abs(beam['box'][5]-shoe_bottom) < EPS and overlaps(beam['box']):
            return beam['id']

    axis = int(np.argmax(np.abs(radial)))
    cross = 1-axis
    center = (np.asarray(rect[:2])+rect[2:])/2
    outward = 1 if radial[axis]>0 else -1
    candidates = []
    for beam in beams:
        box = beam['box']
        if box[cross]-EPS <= center[cross] <= box[cross+3]+EPS and box[2]<shoe_bottom+EPS and box[5]>shoe_bottom-.24:
            target = np.clip(center[axis],box[axis],box[axis+3])
            if (target-center[axis])*outward >= 0:
                candidates.append((abs(target-center[axis]),float(target)))
    reach = min(candidates)[1] if candidates else center[axis]+outward*.32
    low = np.asarray(rect[:2]).copy()
    high = np.asarray(rect[2:]).copy()
    low[axis] = min(low[axis]-.035,reach)
    high[axis] = max(high[axis]+.035,reach)
    low[cross],high[cross] = center[cross]-.065,center[cross]+.065
    beam = add_box(scene, shoe_id+'_secondary_beam', [*low,shoe_bottom-.16,*high,shoe_bottom],
                   'beam', [.34,.45,.52], owner)
    beam.update(catalogue_shoe_support=True, reached_existing_beam=bool(candidates))
    return beam['id']
