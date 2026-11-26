"""Check equipment clearances, supports and access on each floor."""
from __future__ import annotations

from collections import deque
import json
import math

import numpy as np

from .process import RULES
from .rule_identity import band
from .platform_coverage import _uncovered as _raw_uncovered, check_platform_coverage
from .maintenance_geometry import level_maintenance
from .pump_maintenance import envelope, long_sides, widths as pump_widths, ORDINARY_GAP_M

EPS = 1e-8  # Numerical equality only, never a construction allowance.
from .runtime_paths import DATA
LAYOUT_LIMITS = DATA / 'layout_limits.json'
PRODUCTION_REACTORS = DATA / 'reactor_f_series_dimensions.json'
EQUIPMENT = {'vessel', 'pump', 'filter', 'exchanger', 'condenser', 'tank', 'tower', 'fixed_bed'}


def _uncovered(bounds, covers):
    # Loaded model AABBs may express 3.6 as 3.5999999999999996. Collapse only
    # zero-width numerical strips, otherwise they falsely block an entire path.
    return [r for r in _raw_uncovered(bounds,covers) if r[2]-r[0] > EPS and r[3]-r[1] > EPS]


def read_layout_rules():
    limits = json.loads(LAYOUT_LIMITS.read_text(encoding='utf-8'))
    limits['ground_bottom_height_m'] = RULES['geometry_screen']['walk_height_m']
    return limits


def maintenance_width_for(unit,rules):
    """Fixed before layout: large vessels and slender towers get the upper bound."""
    declaration = unit.get('dimension_spec',unit.get('catalog',unit.get('dimensions',{}))) or {}
    tier = unit.get('size_tier',unit.get('tier',declaration.get('size_tier',declaration.get('tier'))))
    form = unit.get('form',unit.get('orientation',declaration.get('form','vertical')))
    large = (tier == 'large' and unit['kind'] in ('vessel','tank') and form=='vertical') or unit['kind'] in ('fixed_bed','tower')
    return rules['large_maintenance_width_m'] if large else rules['maintenance_width_m']


def _overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) + EPS and min(a[3], b[3]) > max(a[1], b[1]) + EPS


def _rect(box):
    return [float(box[0]), float(box[1]), float(box[3]), float(box[4])]


def _union(boxes):
    return [min(b[i] for b in boxes) for i in range(3)] + [max(b[i] for b in boxes) for i in range(3, 6)]


def _measured_components(scene, model=None):
    rows = []
    data = None
    if model is not None:
        import mujoco
        from .checks import geom_bounds
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    for c in scene['components']:
        reservation = c['kind']=='drive_reservation'
        if (c.get('visual_only') or c.get('annotation_only')) and not reservation:
            continue
        item = dict(c)
        if model is not None:
            try:
                geom = model.geom(c['id'])
                if not (geom.contype[0] or geom.conaffinity[0]) and not reservation:
                    continue
                item['box'] = geom_bounds(model, data, [geom.id])[0].tolist()
                item['_actual_geom_type'] = int(geom.type[0])
                item['_actual_size'] = geom.size.tolist()
                item['_actual_center'] = data.geom_xpos[geom.id].tolist()
                item['_actual_matrix'] = data.geom_xmat[geom.id].reshape(3,3).tolist()
            except KeyError:
                # A missing physical component is not supplied by its metadata.
                continue
        rows.append(item)
    return rows


def equipment_footprints(graph, scene, model=None):
    """Independent owner-solid projection, including lower-head short nozzles."""
    components = _measured_components(scene, model)
    floors = scene['floors']
    roof = scene.get('shell', {}).get('roof_z_m', floors[-1] + RULES['construction_choices']['storey_height_m'])
    rows = []
    for uid, unit in graph['units'].items():
        if unit['kind'] not in EQUIPMENT:
            continue
        parts = [c for c in components
                 if (c.get('owner') == uid or c['id'] == uid)
                 and c['kind'] not in ('equipment_beam', 'pipe', 'valve', 'check_valve', 'rail', 'curb')]
        if not parts:
            rows.append({'id': uid, 'kind': unit['kind'], 'rect': None,
                         'occupied_levels': [], 'component_ids': [], 'z_range_m': None})
            continue
        box = _union([c['box'] for c in parts])
        levels = [i for i, z in enumerate(floors)
                  if box[5] > z + EPS and box[2] < (floors[i+1] if i+1 < len(floors) else roof) - EPS]
        rows.append({'id': uid, 'kind': unit['kind'], 'rect': _rect(box),
                     'occupied_levels': levels, 'component_ids': [c['id'] for c in parts],
                     'z_range_m': [box[2], box[5]], 'parts': parts,'_unit':unit})
    return rows


def _boards(components, level, z):
    return [_rect(c['box']) for c in components
            if (c['kind'] in ('floor', 'ground') and abs(c['box'][5] - z) <= EPS)
            or (c['kind'] == 'walk_surface' and c.get('walk_level') == level)]


def _standing_boards(components, level, z):
    boards = _boards(components, level, z)
    bounds = _bounds(boards)
    if bounds is None or level == 0:
        return boards
    excluded = []
    for c in components:
        b = c['box']
        if c['kind'] != 'rail' or not z < b[2] < z + RULES['geometry_screen']['walk_height_m']:
            continue
        if b[4]-b[1] > b[3]-b[0]:
            if b[0] <= bounds[0] + EPS and b[3] >= bounds[0]-EPS:
                excluded.append([bounds[0],b[1],b[3],b[4]])
            if b[0] <= bounds[2] + EPS and b[3] >= bounds[2]-EPS:
                excluded.append([b[0],b[1],bounds[2],b[4]])
        else:
            if b[1] <= bounds[1] + EPS and b[4] >= bounds[1]-EPS:
                excluded.append([b[0],bounds[1],b[3],b[4]])
            if b[1] <= bounds[3] + EPS and b[4] >= bounds[3]-EPS:
                excluded.append([b[0],b[1],b[3],bounds[3]])
    return [part for board in boards for part in _uncovered(board,excluded)]


def standing_board_rectangles(scene, level, model=None):
    """Board area inside real boundary-rail inner faces, not their centre lines."""
    return _standing_boards(_measured_components(scene,model),level,scene['floors'][level])


def _bounds(rects):
    if not rects:
        return None
    return [min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects)]


def _area(rects):
    return sum((r[2]-r[0])*(r[3]-r[1]) for r in rects)


def _opening_records(scene, level):
    return [r for r in scene.get('opening_owners', []) if r['level'] == level]


def _own_holes(row, scene, level, rules):
    records, errors = [], []
    z = scene['floors'][level]
    # BL-6 already includes the real drive box in equipment occupancy. It
    # needs the same measured penetration contract when a useful deck crosses
    # it; a registration alone or an oversized hole still cannot pass.
    core = [c for c in row['parts'] if (c['kind'] in EQUIPMENT or c['kind']=='head'
            or (c['kind']=='drive_reservation' and c.get('shape')=='box'))
            and c['box'][2] < z + EPS and c['box'][5] > z - EPS]
    section = _rect(_union([c['box'] for c in core])) if core else None
    all_holes = scene['holes'].get(level, scene['holes'].get(str(level), []))
    for record in _opening_records(scene, level):
        if record['owner'] != row['id']:
            continue
        hole = record['rect']
        registered = any(all(abs(a-b) <= EPS for a,b in zip(hole, h)) for h in all_holes)
        if record.get('kind')=='pullout':
            from .pullout_reservations import expected_pullout
            expected=expected_pullout(row,row.get('_unit',{'kind':row['kind']}),scene,rules)
            valid=bool(registered and expected and z>expected['base_z']+EPS
                       and record.get('reservation_id')==expected['id']
                       and all(abs(a-b)<=EPS for a,b in zip(hole,expected['rect'])))
            evidence={'rect':hole,'source':'BL-6E actual axis/body top/drive + independent factory d4/d5',
                      'expected_pullout':expected,'valid':valid,'purpose':'pullout',
                      'scope':'Openings follow the physical passage. Pull-out checks cover actual openings and edges. Standing area and support remain subject to checks.'}
            records.append(evidence)
            if not valid:errors.append({'code':'OWN_OPENING_INVALID','equipment':row['id'],'level':level,
                'message':'The opening does not match the physical pull-out path. A label does not enlarge it.','measurement':evidence})
            continue
        setbacks = ([section[0]-hole[0], section[1]-hole[1], hole[2]-section[2], hole[3]-section[3]]
                    if section else None)
        valid = bool(registered and section and setbacks is not None
                     and all(-EPS <= v <= rules['normal_equipment_opening_clearance_m'] + EPS for v in setbacks))
        records.append({'rect': hole, 'source': 'scene.opening_owners + scene.holes + actual shell section',
                        'actual_penetrating_components':[{'id':c['id'],'kind':c['kind'],'box':c['box']} for c in core],
                        'actual_penetrating_shell_rect': section, 'edge_setbacks_m': setbacks,
                        'construction_source': rules['opening_source'], 'valid': valid})
        if not valid:
            errors.append({'code': 'OWN_OPENING_INVALID', 'equipment': row['id'], 'level': level,
                           'message': 'There is no shell or drive cross-section at this opening. An oversized registered opening cannot exempt missing decking.', 'measurement': records[-1]})
    return [r['rect'] for r in records if r['valid']], records, errors


def _structural_obstacles(components, floors, level):
    lo = floors[level]
    hi = floors[level+1] if level+1 < len(floors) else math.inf
    return [{'id': c['id'], 'kind': c['kind'], 'rect': _rect(c['box'])}
            for c in components if c['kind'] in ('column', 'step', 'landing')
            and c['box'][5] > lo + EPS and c['box'][2] < hi - EPS]


def _side_width(rect, side, boundary, obstacles):
    axis, direction = {'x_min': (0,-1), 'x_max': (0,1), 'y_min': (1,-1), 'y_max': (1,1)}[side]
    other = 1-axis
    edge = rect[axis if direction < 0 else axis+2]
    limit = boundary[axis if direction < 0 else axis+2]
    width = (edge-limit) if direction < 0 else (limit-edge)
    limiting = 'standing_surface_boundary'
    for obstacle in obstacles:
        b = obstacle['rect']
        if min(rect[other+2], b[other+2]) <= max(rect[other], b[other]) + EPS:
            continue
        if direction < 0:
            if b[axis] >= edge-EPS:
                continue
            gap = max(0., edge-b[axis+2])
        else:
            if b[axis+2] <= edge+EPS:
                continue
            gap = max(0., b[axis]-edge)
        if gap < width:
            width, limiting = gap, obstacle['id']
    width = max(0.,float(width))
    return {'minimum_clearance_m': width,'minimum_clear_width_m': width,'limiting_obstacle': limiting}


def _reachability(boards, obstacles, entry):
    """Exact rectangle configuration-space cells, using existing walking size."""
    bounds = _bounds(boards)
    if bounds is None or entry is None:
        return [], [], np.zeros((0,0), bool)
    hx = RULES['geometry_screen']['walk_width_m']/2
    hy = RULES['geometry_screen']['walk_depth_m']/2
    voids = _uncovered(bounds, boards)
    rects = [o['rect'] for o in obstacles] + voids
    low = [bounds[0]+hx, bounds[1]+hy]
    high = [bounds[2]-hx, bounds[3]-hy]
    xs = {low[0], high[0], float(entry[0])}
    ys = {low[1], high[1], float(entry[1])}
    for r in rects:
        xs.update((max(low[0], min(high[0], r[0]-hx)), max(low[0], min(high[0], r[2]+hx))))
        ys.update((max(low[1], min(high[1], r[1]-hy)), max(low[1], min(high[1], r[3]+hy))))
    def cells(values):
        values = sorted(values)
        return sorted(set(values + [(a+b)/2 for a,b in zip(values, values[1:])]))
    xs, ys = cells(xs), cells(ys)
    x, y = np.asarray(xs)[:,None], np.asarray(ys)[None,:]
    free = ((x >= low[0]-EPS) & (x <= high[0]+EPS) & (y >= low[1]-EPS) & (y <= high[1]+EPS))
    for r in rects:
        free &= ~((x > r[0]-hx+EPS) & (x < r[2]+hx-EPS)
                  & (y > r[1]-hy+EPS) & (y < r[3]+hy-EPS))
    reachable = np.zeros_like(free)
    start = (xs.index(float(entry[0])), ys.index(float(entry[1])))
    queue = deque([start]) if free[start] else deque()
    if queue:
        reachable[start] = True
    while queue:
        a,b = queue.popleft()
        for aa,bb in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
            if 0 <= aa < len(xs) and 0 <= bb < len(ys) and free[aa,bb] and not reachable[aa,bb]:
                reachable[aa,bb] = True
                queue.append((aa,bb))
    return xs, ys, reachable


def _reachable_sides(row, width, nav):
    xs, ys, reachable = nav
    if not len(xs):
        return []
    r = row['rect']
    x,y = np.asarray(xs)[:,None], np.asarray(ys)[None,:]
    sides = {'x_min': (x >= r[0]-width-EPS) & (x <= r[0]+EPS) & (y >= r[1]-EPS) & (y <= r[3]+EPS),
             'x_max': (x >= r[2]-EPS) & (x <= r[2]+width+EPS) & (y >= r[1]-EPS) & (y <= r[3]+EPS),
             'y_min': (y >= r[1]-width-EPS) & (y <= r[1]+EPS) & (x >= r[0]-EPS) & (x <= r[2]+EPS),
             'y_max': (y >= r[3]-EPS) & (y <= r[3]+width+EPS) & (x >= r[0]-EPS) & (x <= r[2]+EPS)}
    return [side for side, mask in sides.items() if np.any(reachable & mask)]


def _side_strips(rect,width):
    a,b,c,d = rect
    return {'x_min':[a-width,b-width,a,d+width],'x_max':[c,b-width,c+width,d+width],
            'y_min':[a,b-width,c,b],'y_max':[a,d,c,d+width]}


def _touches_reachable(stance_box,nav):
    """Reachable cell adjoining an already board/obstacle-validated stance."""
    xs,ys,reachable = nav
    if not len(xs):
        return False
    centre = [(stance_box[i]+stance_box[i+3])/2 for i in (0,1)]
    candidates=[]
    for values,p in ((xs,centre[0]),(ys,centre[1])):
        index=int(np.searchsorted(values,p))
        candidates.append([i for i in (index-1,index) if 0<=i<len(values)])
    return any(reachable[x,y] for x in candidates[0] for y in candidates[1])


def _pump_rows(rows, instances, width):
    measurements, errors = [], []
    for row in rows:
        if row['kind'] != 'pump' or row['rect'] is None:
            continue
        body = next((c for c in row['parts'] if c['kind'] == 'pump' and 'ends' in c), None)
        motors = [c for c in row['parts'] if c['kind'] == 'motor']
        bases = [c for c in row['parts'] if c['kind'] in ('plinth','foundation','base')]
        if body is None or not motors or not bases:
            errors.append({'code': 'PUMP_GEOMETRY_INCOMPLETE', 'equipment': row['id'],
                           'message': 'The pump shaft, motor end and foundation front edge require physical geometry.'})
            continue
        axis = np.asarray(body['ends'][1][:2])-body['ends'][0][:2]
        axis /= np.linalg.norm(axis)
        if axis[0] < -EPS or (abs(axis[0]) <= EPS and axis[1] < 0):
            axis = -axis
        normal = np.array([-axis[1], axis[0]])
        centre = (np.asarray(body['box'][:2])+body['box'][3:5])/2
        motor = (np.asarray(motors[0]['box'][:2])+motors[0]['box'][3:5])/2
        motor_side = float(np.dot(motor-centre, axis))
        base = _union([c['box'] for c in bases])
        front = min(float(np.dot([x,y], normal)) for x in (base[0],base[3]) for y in (base[1],base[4]))
        measurements.append({'id': row['id'], 'level': row['occupied_levels'][0], 'axis': axis.tolist(),
                             'motor_side': 1 if motor_side > EPS else (-1 if motor_side < -EPS else 0),
                             'base_front_m': front, 'footprint': row['rect']})
    for level in sorted({r['level'] for r in measurements}):
        pumps = [r for r in measurements if r['level'] == level]
        ref = pumps[0]
        if any(abs(abs(np.dot(r['axis'],ref['axis']))-1) > EPS for r in pumps):
            errors.append({'code':'PUMP_AXES_NOT_PARALLEL','level':level})
        if any(r['motor_side'] != ref['motor_side'] or not r['motor_side'] for r in pumps):
            errors.append({'code':'PUMP_MOTOR_SIDES_DIFFER','level':level})
        fronts = sorted({round(r['base_front_m'],8) for r in pumps})
        if len(fronts) > 2:
            errors.append({'code':'PUMP_BASE_FRONTS_NOT_ALIGNED','level':level,'fronts_m':fronts})
        if len(fronts) == 2:
            # Two rows need a full maintenance strip between their real bodies.
            normal = np.array([-ref['axis'][1],ref['axis'][0]])
            bounds = []
            for front in fronts:
                ps = [p for p in pumps if abs(p['base_front_m']-front) <= EPS]
                values = [float(np.dot([x,y],normal)) for p in ps
                          for x in (p['footprint'][0],p['footprint'][2])
                          for y in (p['footprint'][1],p['footprint'][3])]
                bounds.append((min(values),max(values)))
            if bounds[1][0]-bounds[0][1] < width-EPS:
                errors.append({'code':'PUMP_ROWS_TOO_CLOSE','level':level,
                               'actual_gap_m':bounds[1][0]-bounds[0][1],'required_m':width})
    return {'rows': measurements, 'violations': errors}


def _cylinder_measurement(component):
    """Use loaded cylinder size/axis; scene dimensions are only the offline source."""
    if '_actual_geom_type' in component:
        import mujoco
        if component['_actual_geom_type'] != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            return None
        radius,half_length = component['_actual_size'][:2]
        axis = np.asarray(component['_actual_matrix'])[:,2]
        # A cylinder's physical axis has no inherent sign. Preserve only the
        # producer's +/- end naming, never its claimed length or radius.
        if 'ends' in component:
            named_direction = np.asarray(component['ends'][1])-component['ends'][0]
            if np.dot(axis,named_direction) < 0:
                axis = -axis
        centre = np.asarray(component['_actual_center'])
        a,b = centre-axis*half_length,centre+axis*half_length
        source = 'loaded MuJoCo cylinder geom_size, geom_xpos and geom_xmat'
    else:
        if component.get('shape') != 'cylinder' or 'ends' not in component:
            return None
        a,b = np.asarray(component['ends'][0]),np.asarray(component['ends'][1])
        half_length = float(np.linalg.norm(b-a))/2
        if half_length <= EPS:
            return None
        axis = (b-a)/(2*half_length)
        radius = component['radius']
        source = 'emitted cylinder geometry (offline, no model supplied)'
    return {'diameter_m':float(2*radius),'shell_length_m':float(2*half_length),
            'axis':axis.tolist(),'ends':[a.tolist(),b.tolist()],
            'form':'vertical' if abs(axis[2]) >= 1-EPS else ('horizontal' if abs(axis[2]) <= EPS else 'inclined'),
            'source':source}


def _main_body(row):
    candidates = [c for c in row['parts'] if c['kind'] == row['kind']]
    measured = [(c,_cylinder_measurement(c)) for c in candidates]
    measured = [(c,m) for c,m in measured if m is not None]
    return max(measured,key=lambda p:p[1]['shell_length_m']) if measured else (None,None)


def _vertical_operational(unit,row):
    if unit['kind'] not in ('vessel','fixed_bed','tower','exchanger','condenser'):
        return False
    _,measurement=_main_body(row)
    return measurement is not None and measurement['form']=='vertical'


def _declared_dimensions(unit,instance):
    """Equipment catalogue declarations; Older coarse fixtures have no catalog."""
    declaration = unit.get('dimension_spec') or instance.get('dimension_spec')
    catalog_form = unit['kind'] in ('vessel','tank','fixed_bed','tower','exchanger','condenser') and unit.get('form') in ('vertical','horizontal')
    if catalog_form or unit.get('catalog_key') or unit.get('catalog') or (declaration and declaration.get('catalog_key')):
        from .equipment_catalog import resolve_spec
        return resolve_spec(unit)
    return declaration


def _production_reactor_evidence(unit,row,declaration,scene,components,actual,shell_box):
    """Taiji anchors come from the original reference, never the producer table.

    The approved head/body/nozzle approximations are checked separately from
    manufacturer dimensions. Full volume is reported, not used to force-fit
    these deliberately coarse solids into a manufacturer's internal cavity.
    """
    reference=json.loads(PRODUCTION_REACTORS.read_text())
    errors=[]
    tier=unit.get('size_tier',declaration.get('size_tier'))
    nominal={'small':5000,'medium':10000,'large':20000}.get(tier)
    model_key=next((key for key,value in reference['models'].items()
                    if value['capacity_l']['nominal']==nominal),None)
    if model_key is None:
        return {'status':'fail','source':str(PRODUCTION_REACTORS),'size_tier':tier},[
            {'code':'PRODUCTION_REACTOR_TIER_INVALID','size_tier':tier}]
    ref=reference['models'][model_key]
    raw=ref['dimensions_mm'];capacity=ref['capacity_l']
    anchors=[]
    def compare(field,expected,measured,code='PRODUCTION_REACTOR_ANCHOR_MISMATCH',source='manufacturer'):
        numeric=lambda value:isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        valid=(measured is None if expected is None else
               numeric(measured) and abs(measured-expected)<=EPS)
        anchors.append({'field':field,'expected':expected,'actual':measured,'source':source,'matches':valid})
        if not valid:
            errors.append({'code':code,'field':field,'expected':expected,'actual':measured})
    if declaration.get('model')!=model_key:
        errors.append({'code':'PRODUCTION_REACTOR_MODEL_MISMATCH','expected':model_key,
                       'actual':declaration.get('model')})
    for field,expected in raw.items():
        compare('manufacturer_dimensions_mm.'+field,expected,
                declaration.get('manufacturer_dimensions_mm',{}).get(field))
    for field,expected in ref['reference_drive_and_overall_mm'].items():
        compare('reference_drive_and_overall_mm.'+field,expected,
                declaration.get('reference_drive_and_overall_mm',{}).get(field))
    for field,expected in capacity.items():
        compare('manufacturer_capacities_l.'+field,expected,
                declaration.get('manufacturer_capacities_l',{}).get(field))
    for field,key in (('nominal_volume_l','nominal'),('catalog_total_volume_l','total'),
                      ('catalog_calculated_volume_l','calculated')):
        compare(field,capacity[key],declaration.get(field),'PRODUCTION_REACTOR_CAPACITY_MISMATCH')

    support_level=int(unit.get('level',0))
    support_z=scene['floors'][support_level]
    upper=raw['h4']/1000
    bottom=upper-raw['h1']/1000
    drive_mount=upper+raw['h2']/1000
    diameter=raw['d2']/1000
    # These are existing, explicitly approved coarse-model choices, not new
    # manufacturer facts or relaxed clearance thresholds.
    neck=.12
    head_depth=diameter/4
    body_top=upper-neck
    cylinder_length=body_top-bottom-2*head_depth
    drive_top=drive_mount+ref['reference_drive_and_overall_mm']['h5']/1000
    inner_radius=raw['d1']/2000
    geometric_l=1000*(math.pi*inner_radius**2*actual['shell_length_m']+
        2*math.pi/3*inner_radius**2*sum(actual.get('head_depths_m',[])))
    estimates={'nozzle_length_m_estimate':neck,'head_depth_diameter_fraction_estimate':.25,
               'drive_width_m_estimate':.32}
    for field,expected in estimates.items():
        compare(field,expected,declaration.get(field),source='approved coarse-model estimate')
    independent_dimensions={'diameter_m':diameter,'shell_length_m':cylinder_length,
        'support_to_bottom_m':bottom,'support_to_top_m':body_top,
        'drive_above_support_m':drive_top,'installed_height_m':drive_top-bottom}
    for field,expected in independent_dimensions.items():
        compare('actual.'+field,expected,actual.get(field),'PRODUCTION_REACTOR_GEOMETRY_MISMATCH',
                'original d2/h1/h2/h4/h5 + declared coarse geometry, independently derived')
    for index,measured in enumerate(actual.get('head_depths_m',[])):
        compare('actual.head_depths_m.'+str(index),head_depth,measured,
                'PRODUCTION_REACTOR_GEOMETRY_MISMATCH','d2/4 approved ellipsoid estimate')
    head_rows=[]
    for head in (c for c in row['parts'] if c['kind']=='head'):
        if '_actual_geom_type' in head:
            import mujoco
            ellipsoid=head['_actual_geom_type']==int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
            sizes=head['_actual_size']
        else:
            ellipsoid=head.get('shape')=='ellipsoid'
            sizes=head.get('size',[])
        good=ellipsoid and len(sizes)==3 and all(abs(a-b)<=EPS for a,b in
            zip(sizes,[diameter/2,diameter/2,head_depth]))
        head_rows.append({'component':head['id'],'actual_semiaxes_m':list(sizes),
                          'expected_semiaxes_m':[diameter/2,diameter/2,head_depth],
                          'actual_ellipsoid':ellipsoid,'matches':good})
        if not good:
            errors.append({'code':'PRODUCTION_REACTOR_GEOMETRY_MISMATCH',
                           'dimension':'ellipsoid_semiaxes_m','measurement':head_rows[-1]})
    compare('geometric_estimated_volume_l',geometric_l,declaration.get('geometric_estimated_volume_l'),
            'PRODUCTION_REACTOR_CAPACITY_MISMATCH','d1 + measured coarse cylinder and heads; not measured cavity')
    planes={'upper_datum':upper,'bottom_datum':bottom,
            'jacket_upper_axis':upper-raw['h3']/1000,'drive_mount':drive_mount}
    for field,expected in planes.items():
        compare('reference_planes_m.'+field,expected,declaration.get('reference_planes_m',{}).get(field))

    body_center=(np.asarray(actual['ends'][0])+actual['ends'][1])/2
    lugs=[c for c in row['parts'] if c['kind']=='lug']
    shoes=[c for c in row['parts'] if c['kind']=='support']
    if len(lugs)!=4:
        errors.append({'code':'PRODUCTION_REACTOR_FOUR_LUGS_MISSING','actual_count':len(lugs),'required_count':4})
    angle=math.radians(unit.get('rotation_deg',0.))
    rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    used=set();lug_rows=[]
    def contact(a,b):
        return _overlap(_rect(a['box']),_rect(b['box'])) and all(
            min(a['box'][i+3],b['box'][i+3])>=max(a['box'][i],b['box'][i])-EPS for i in range(3))
    structural=[c for c in components if c.get('owner')!=row['id'] and
                c['kind'] in ('floor','ground','walk_surface','beam','equipment_beam','column','landing')]
    for bearing in reference['lug_bearings_deg']:
        theta=math.radians(bearing)
        local_point=raw['d3']/2000*np.array([math.sin(theta),math.cos(theta)])
        point=rotation@local_point+body_center[:2]
        candidates=[c for c in lugs if c['id'] not in used and
                    all(c['box'][i]-EPS<=point[i]<=c['box'][i+3]+EPS for i in (0,1))]
        lug=min(candidates,key=lambda c:abs(c['box'][2]-support_z)) if candidates else None
        record={'manufacturer_bearing_deg':bearing,'manufacturer_hole_reference_world_m':
                [float(point[0]),float(point[1]),support_z],
                'reference_scope':'d3/2 reference hole centre; not a required drilled hole or plate outline',
                'lug':lug['id'] if lug else None,'supports':[]}
        if lug is None:
            errors.append({'code':'PRODUCTION_REACTOR_LUG_ANCHOR_MISSING','bearing_deg':bearing})
            lug_rows.append(record)
            continue
        used.add(lug['id']);box=lug['box']
        nearest=np.maximum(np.asarray(box[:2])-body_center[:2],0.)+np.minimum(
            np.asarray(box[3:5])-body_center[:2],0.)
        shell_contact=(float(np.linalg.norm(nearest))<=diameter/2+EPS and
                       min(box[5],max(p[2] for p in actual['ends']))>=
                       max(box[2],min(p[2] for p in actual['ends']))-EPS)
        bearing_at_reference=abs(box[2]-support_z)<=EPS
        for shoe in shoes:
            if not contact(lug,shoe) or abs(shoe['box'][5]-box[2])>EPS:
                continue
            members=[c for c in structural if contact(shoe,c)]
            record['supports'].append({'component':shoe['id'],'actual_box':shoe['box'],
                                       'contacted_structure':[{'id':c['id'],'kind':c['kind'],'box':c['box']} for c in members]})
        supported=any(r['contacted_structure'] for r in record['supports'])
        record.update(actual_box=box,underside_at_support_reference=bearing_at_reference,
                      actual_cylinder_contact=shell_contact,real_structure_supported=supported)
        if not (bearing_at_reference and shell_contact and supported):
            errors.append({'code':'PRODUCTION_REACTOR_LUG_NOT_BEARING','measurement':record})
        lug_rows.append(record)

    nozzle_rows=[]
    nozzle_parts=[(c,_cylinder_measurement(c)) for c in row['parts'] if c['kind']=='equipment_nozzle']
    for name,port in unit.get('ports',{}).items():
        lower=name in ('bottom','drain')
        expected_z=support_z+(bottom-neck if lower else upper)
        position=port.get('pos')
        direction=[0.,0.,-1. if lower else 1.]
        matched=[]
        if isinstance(position,(list,tuple)) and len(position)==3:
            for component,measure in nozzle_parts:
                if measure is None:
                    continue
                face=min(measure['ends'],key=lambda p:p[2]) if lower else max(measure['ends'],key=lambda p:p[2])
                if math.dist(face,position)<=EPS:
                    matched.append({'component':component['id'],'actual_face_m':face,
                                    'actual_diameter_m':measure['diameter_m'],'axis':measure['axis']})
        good=bool(matched and position is not None and abs(position[2]-expected_z)<=EPS and
                  math.dist(port.get('normal',[math.inf]*3),direction)<=EPS and
                  all(abs(m['axis'][2])>=1-EPS for m in matched))
        record={'port':name,'end':'lower' if lower else 'upper','interface_position_m':position,
                'expected_face_z_m':expected_z,'physical_nozzles':matched,'matches':good,
                'basis':'B-.12 / U; approved short-neck estimate, not a verified flange face'}
        if name=='bottom':
            from .pipe_spec import load as load_pipe_spec
            expected_dn=ref['bottom_discharge']['dn']
            expected_outer_diameter=load_pipe_spec()['pipe_outer_diameter_mm'][str(expected_dn)]/1000.
            record.update(manufacturer_reference_dn=expected_dn,
                line_dn=port.get('line_dn',port.get('dn')),
                equipment_neck_dn=port.get('equipment_neck_dn'),
                expected_outer_diameter_m=expected_outer_diameter,
                outer_diameter_source='scene_gen/room/pipe_spec.json',
                dn_source=reference['sources'][ref['bottom_discharge']['source_id']])
            if (port.get('equipment_neck_dn')!=expected_dn or not matched or
                    any(abs(m['actual_diameter_m']-expected_outer_diameter)>EPS for m in matched)):
                errors.append({'code':'PRODUCTION_REACTOR_BOTTOM_NECK_MISMATCH','measurement':record})
        if not good:
            errors.append({'code':'PRODUCTION_REACTOR_INTERFACE_BASELINE_MISMATCH','measurement':record})
        nozzle_rows.append(record)
    if not any(r['end']=='upper' for r in nozzle_rows) or not any(r['end']=='lower' for r in nozzle_rows):
        errors.append({'code':'PRODUCTION_REACTOR_END_INTERFACE_MISSING'})

    coarse=[c for c in row['parts'] if c['kind'] in ('vessel','head','lug','support','drive_reservation','equipment_nozzle')]
    provenance=[{'id':c['id'],'kind':c['kind'],'estimated':c.get('estimated'),
                 'scope':c.get('scope') or declaration.get('field_note')} for c in coarse]
    if not declaration.get('geometry_approximations') or any(r['estimated'] is not True or not r['scope'] for r in provenance):
        errors.append({'code':'PRODUCTION_REACTOR_COARSE_PROVENANCE_MISSING','components':provenance})
    result={'status':'fail' if errors else 'pass','source':str(PRODUCTION_REACTORS),
        'manufacturer_model':model_key,'manufacturer_sources':reference['sources'],
        'manufacturer_dimensions_mm':raw,'anchors':anchors,
        'capacities_l':{'nominal':capacity['nominal'],'catalog_total':capacity['total'],
            'catalog_calculated':capacity['calculated'],'geometry_estimated':geometric_l},
        'capacity_scope':'four separate values; d1/coarse axial segments estimate is not an actual inner-cavity measurement',
        'reference_planes_local_m':planes,'reference_planes_scope':'original diagram datums; body/short-neck relations remain estimates',
        'approximate_catalog_overall_height_m':None if ref['reference_drive_and_overall_mm']['h'] is None else ref['reference_drive_and_overall_mm']['h']/1000,
        'catalog_overall_scope':'reference only; not a forced installed-height equality',
        'independent_coarse_dimensions_m':independent_dimensions,'heads':head_rows,'four_lugs':lug_rows,
        'interfaces':nozzle_rows,'coarse_provenance':provenance,
        'manufacturer_unresolved':reference['unresolved']}
    return result,errors


def check_equipment_dimensions(graph,scene,model=None,maintenance_rows=None):
    rules = read_layout_rules()
    components = _measured_components(scene,model)
    rows = equipment_footprints(graph,scene,model)
    instances = {r['id']:r for r in scene.get('equipment_instances',[])}
    errors,measurements = [],[]
    maintenance_rows = maintenance_rows or []
    for row in rows:
        if not row['rect'] or row['kind'] in ('pump','filter'):
            continue
        unit = graph['units'][row['id']]
        declaration = _declared_dimensions(unit,instances.get(row['id'],{}))
        # Frozen legacy layout geometry without a BL-4 catalog stays within its old scope.
        if declaration is None:
            continue
        body,actual = _main_body(row)
        local = []
        if actual is None:
            errors.append({'code':'DIMENSION_BODY_NOT_MEASURABLE','equipment':row['id']})
            continue
        expected_form = declaration.get('form',unit.get('form',unit.get('orientation')))
        if actual['form'] != expected_form:
            local.append({'code':'EQUIPMENT_FORM_MISMATCH','expected':expected_form,'actual':actual['form']})
        expected_dimensions = {'diameter_m':declaration.get('diameter_m'),
                               'shell_length_m':declaration.get('main_cylinder_length_m',declaration.get('shell_length_m',declaration.get('length_m')))}
        for key,expected in expected_dimensions.items():
            if expected is not None and abs(actual[key]-expected) > EPS:
                local.append({'code':'EQUIPMENT_DIMENSION_MISMATCH','dimension':key,
                              'expected_m':expected,'actual_m':actual[key]})
        actual['body_z_range_m'] = [body['box'][2],body['box'][5]]
        heads = sorted([c for c in row['parts'] if c['kind']=='head'],
                       key=lambda c:float(np.dot((np.asarray(c['box'][:3])+c['box'][3:])/2,actual['axis'])))
        expected_heads = [declaration.get('lower_head_depth_m',declaration.get('head_depth_m')),
                          declaration.get('upper_head_depth_m',declaration.get('head_depth_m'))]
        actual['head_depths_m'] = []
        if any(v is not None for v in expected_heads):
            if len(heads)!=2:
                local.append({'code':'EQUIPMENT_HEADS_MISSING','actual_count':len(heads),'required_count':2})
            else:
                for index,head in enumerate(heads):
                    v = np.asarray(actual['axis'])
                    extent = sum(abs(v[i])*(head['box'][i+3]-head['box'][i]) for i in range(3))/2
                    actual['head_depths_m'].append(float(extent))
                    if expected_heads[index] is not None and abs(extent-expected_heads[index])>EPS:
                        local.append({'code':'EQUIPMENT_DIMENSION_MISMATCH','dimension':f'head_depth_{index}_m',
                                      'expected_m':expected_heads[index],'actual_m':float(extent)})
        shell_parts = [c for c in row['parts'] if c['kind'] in EQUIPMENT or c['kind'] in ('head','skirt')]
        shell_box = _union([c['box'] for c in shell_parts])
        actual['total_height_m'] = shell_box[5]-shell_box[2]
        expected_height = declaration.get('height_m',declaration.get('total_height_m'))
        if expected_height is not None and abs(actual['total_height_m']-expected_height) > EPS:
            local.append({'code':'EQUIPMENT_DIMENSION_MISMATCH','dimension':'total_height_m',
                          'expected_m':expected_height,'actual_m':actual['total_height_m']})
        axis = np.asarray(actual['axis'])
        axial_extents = [float(np.dot([x,y,z],axis)) for c in shell_parts
                         for x in (c['box'][0],c['box'][3]) for y in (c['box'][1],c['box'][4])
                         for z in (c['box'][2],c['box'][5])]
        actual['total_axial_length_m'] = max(axial_extents)-min(axial_extents)
        if 'total_length_m' in declaration and abs(actual['total_axial_length_m']-declaration['total_length_m']) > EPS:
            local.append({'code':'EQUIPMENT_DIMENSION_MISMATCH','dimension':'total_axial_length_m',
                          'expected_m':declaration['total_length_m'],'actual_m':actual['total_axial_length_m']})
        support_level = int(unit.get('level',0))
        if 0 <= support_level < len(scene['floors']):
            support_z = scene['floors'][support_level]
            actual['support_reference_z_m'] = support_z
            # The catalog installation control is the equipment assembly,
            # excluding self nozzles. Maintenance/occupied-level projection
            # above intentionally continues to include every self nozzle.
            controlled = [c for c in row['parts'] if c['kind']!='equipment_nozzle']
            actual['installed_height_m'] = max(c['box'][5] for c in controlled)-support_z
            actual['installed_height_components'] = [c['id'] for c in controlled]
            actual['installed_height_excluded_nozzles'] = [c['id'] for c in row['parts'] if c['kind']=='equipment_nozzle']
            if row['kind']=='vessel' and actual['form']=='vertical':
                # Catalog installation coordinates are signed about the real
                # operation-floor reference: the lower end is below it.
                actual['support_to_bottom_m'] = shell_box[2]-support_z
                actual['support_to_top_m'] = shell_box[5]-support_z
                for key in ('support_to_bottom_m','support_to_top_m'):
                    if key in declaration and abs(actual[key]-declaration[key]) > EPS:
                        local.append({'code':'EQUIPMENT_SUPPORT_HEIGHT_MISMATCH','dimension':key,
                                      'expected_m':declaration[key],'actual_m':actual[key]})
                if declaration.get('drive_above_support_m',0) > 0:
                    drives = [c for c in row['parts'] if c['kind'] in ('motor','drive_reservation')]
                    actual['drive_above_support_m'] = max((c['box'][5] for c in drives),default=support_z)-support_z
                    actual['installed_height_m'] = max((c['box'][5] for c in drives),default=shell_box[5])-shell_box[2]
                    if not drives or abs(actual['drive_above_support_m']-declaration['drive_above_support_m']) > EPS:
                        local.append({'code':'EQUIPMENT_DRIVE_HEIGHT_MISMATCH','expected_m':declaration['drive_above_support_m'],
                                      'actual_m':actual['drive_above_support_m']})
            if 'installed_height_m' in declaration and abs(actual['installed_height_m']-declaration['installed_height_m']) > EPS:
                local.append({'code':'EQUIPMENT_DIMENSION_MISMATCH','dimension':'installed_height_m',
                              'expected_m':declaration['installed_height_m'],'actual_m':actual['installed_height_m']})
        if row['kind'] in ('vessel','tank') and actual['form']=='horizontal':
            limits = rules['horizontal_vessel_limits']
        elif row['kind'] in ('exchanger','condenser'):
            limits = rules['exchanger_limits']
        elif row['kind'] in ('fixed_bed','tower'):
            limits = {'diameter_m':rules['slender_limits']['diameter_m']}
            actual['aspect_ratio'] = actual['total_height_m']/actual['diameter_m']
            lo,hi = rules['slender_limits']['height_m']
            if not lo-EPS <= actual['total_height_m'] <= hi+EPS:
                local.append({'code':'SLENDER_HEIGHT_OUT_OF_RANGE','actual_m':actual['total_height_m'],'range_m':[lo,hi]})
            if actual['aspect_ratio'] < rules['slender_limits']['aspect_ratio_min']-EPS:
                local.append({'code':'SLENDER_ASPECT_RATIO_SHORT','actual':actual['aspect_ratio'],
                              'required':rules['slender_limits']['aspect_ratio_min']})
        else:
            limits = {}
        for key,bounds in limits.items():
            measured_key = 'shell_length_m' if key=='length_m' else key
            if not bounds[0]-EPS <= actual[measured_key] <= bounds[1]+EPS:
                local.append({'code':'EQUIPMENT_RANGE_VIOLATION','dimension':measured_key,
                              'actual_m':actual[measured_key],'range_m':bounds})
        support = declaration.get('support_type',declaration.get('support'))
        supports = [c for c in components if c.get('owner')==row['id'] and c['kind'] in ('lug','saddle','skirt','support')]
        actual['support_components'] = [{'id':c['id'],'kind':c['kind'],'box':c['box']} for c in supports]
        if support in ('saddles','double_saddle','two_saddles'):
            saddles = [c for c in supports if c['kind']=='saddle']
            if len(saddles) < 2:
                local.append({'code':'HORIZONTAL_SADDLES_MISSING','actual_count':len(saddles),'required_count':2})
            else:
                axis = np.asarray(actual['axis'][:2])
                positions = [float(np.dot([(c['box'][0]+c['box'][3])/2,(c['box'][1]+c['box'][4])/2],axis)) for c in saddles]
                body_centre = (np.asarray(actual['ends'][0][:2])+actual['ends'][1][:2])/2
                centre_at = float(np.dot(body_centre,axis))
                if max(positions)-min(positions) <= EPS or not min(positions) < centre_at-EPS < max(positions):
                    local.append({'code':'HORIZONTAL_SADDLES_NOT_SEPARATED'})
        elif support in ('side_lugs','lugs'):
            lugs = [c for c in supports if c['kind']=='lug']
            if len(lugs) < 2:
                local.append({'code':'VERTICAL_LUGS_MISSING','actual_count':len(lugs)})
        elif support=='skirt':
            skirts = [c for c in supports if c['kind']=='skirt']
            if not skirts:
                local.append({'code':'SLENDER_SKIRT_MISSING'})
            elif min(c['box'][2] for c in skirts) > scene['floors'][0]+EPS:
                local.append({'code':'SLENDER_SKIRT_NOT_GROUNDED','bottom_m':min(c['box'][2] for c in skirts)})
        # Every support must physically meet the body and a real floor/structural
        # member. A support count or a catalog word is not a contact measurement.
        for c in supports:
            if c['kind']=='support' and support not in ('legs','support'):
                continue
            shell_hit = any(all(min(c['box'][i+3],part['box'][i+3]) >= max(c['box'][i],part['box'][i])-EPS for i in range(3))
                            for part in shell_parts if part['id']!=c['id'] and part['kind']!='skirt')
            if not shell_hit:
                local.append({'code':'EQUIPMENT_SUPPORT_DETACHED','component':c['id']})
            if c['kind']=='saddle':
                floors_hit = [i for i,z in enumerate(scene['floors']) if abs(c['box'][2]-z) <= EPS]
                supported = any(_area(_uncovered(_rect(c['box']),_boards(components,i,scene['floors'][i]))) <= EPS for i in floors_hit)
                if not supported:
                    local.append({'code':'SADDLE_NOT_ON_FLOOR','component':c['id']})
        required_openings = []
        for level,z in enumerate(scene['floors'][1:],start=1):
            crossing = [c for c in shell_parts if c['kind']!='skirt' and c['box'][2] < z-EPS and c['box'][5] > z+EPS]
            if not crossing:
                continue
            own,records,hole_errors = _own_holes(row,scene,level,rules)
            section = _rect(_union([c['box'] for c in crossing]))
            missing = _uncovered(section,own)
            physical_board_overlap = sum((min(section[2],b[2])-max(section[0],b[0]))
                                         *(min(section[3],b[3])-max(section[1],b[1]))
                                         for b in _boards(components,level,z) if _overlap(section,b))
            required_openings.append({'level':level,'actual_shell_section':section,'own_openings':records,
                                      'uncovered_shell_section_m2':_area(missing),'board_inside_shell_m2':physical_board_overlap})
            if missing or physical_board_overlap > EPS:
                local.append({'code':'CROSSING_PLATFORM_OPENING_MISSING','level':level,
                              'uncovered_shell_section_m2':_area(missing),'board_inside_shell_m2':physical_board_overlap})
        if actual['form']=='vertical':
            if row['kind'] in ('fixed_bed','tower'):
                required_operating_levels = [r['level'] for r in required_openings]
            else:
                required_operating_levels = [max(i for i,z in enumerate(scene['floors']) if z <= body['box'][5]+EPS)]
        else:
            required_operating_levels = [int(unit.get('level',row['occupied_levels'][0]))]
        operating = []
        for level in required_operating_levels:
            measured = next((r for r in maintenance_rows if r['id']==row['id'] and r['level']==level),None)
            good = measured is not None and measured['reachable_sides'] and all(
                s['minimum_clearance_m'] >= measured['required_width_m']-EPS for s in measured['sides'].values())
            if good:
                operating.append(level)
            elif maintenance_rows:
                local.append({'code':'EQUIPMENT_OPERATING_SURFACE_MISSING','level':level})
        if row['kind']=='vessel' and actual['form']=='vertical' and declaration.get('series')=='Taiji F':
            production,production_errors=_production_reactor_evidence(
                unit,row,declaration,scene,components,actual,shell_box)
            actual['production_reactor']=production
            local.extend(production_errors)
        for error in local:
            error['equipment'] = row['id']
        errors.extend(local)
        matching = not any(e['code'] in ('EQUIPMENT_FORM_MISMATCH','EQUIPMENT_DIMENSION_MISMATCH','EQUIPMENT_RANGE_VIOLATION')
                           or e['code'].startswith('PRODUCTION_REACTOR_') for e in local)
        actual['body_length_m'] = actual['shell_length_m']
        actual['body_height_m'] = actual['total_height_m']
        actual['support_kind'] = sorted({c['kind'] for c in supports})
        actual['opening_levels'] = [r['level'] for r in required_openings
                                    if r['uncovered_shell_section_m2'] <= EPS and r['board_inside_shell_m2'] <= EPS]
        measurements.append({'id':row['id'],'kind':row['kind'],'declaration':declaration,'actual':actual,
                             'measured_size_tier':declaration.get('size_tier') if matching else None,
                             'dimensions_match':matching,'source':declaration.get('source'),'estimated':declaration.get('estimated'),
                             'required_openings':required_openings,'required_operating_levels':required_operating_levels,
                             'measured_operating_levels':operating,'status':'fail' if local else 'pass'})
    return {'status':'fail' if errors else 'pass','rows':measurements,'violations':errors}


def _extraction(rows, scene, boards_by_level, structural, width_extra, instances, profiles=None):
    results, errors = [], []
    for row in rows:
        if row['kind'] not in ('exchanger','condenser') or row['rect'] is None:
            continue
        body,measurement = _main_body(row)
        if body is None or measurement['form'] != 'horizontal':
            continue
        a,b = np.asarray(measurement['ends'][0]),np.asarray(measurement['ends'][1])
        direction = b-a
        if abs(direction[2]) > EPS or np.linalg.norm(direction[:2]) <= EPS:
            continue
        length = float(np.linalg.norm(direction[:2]))
        axis = direction[:2]/length
        cross = np.array([-axis[1],axis[0]])
        band = measurement['diameter_m']+width_extra
        specified = instances.get(row['id'],{}).get('extraction_end')
        choices = [specified] if specified in (-1,1) else [-1,1]
        attempts = []
        for end in choices:
            start = a[:2] if end < 0 else b[:2]
            tip = start + axis*end*length
            points = [p+cross*s*band/2 for p in (start,tip) for s in (-1,1)]
            rect = [min(p[0] for p in points),min(p[1] for p in points),
                    max(p[0] for p in points),max(p[1] for p in points)]
            causes = []
            for level in row['occupied_levels']:
                obs = structural[level]+[{'id':r['id'],'rect':(profiles or {}).get((r['id'],level),{}).get('rect',r['rect'])} for r in rows
                                          if r['id'] != row['id'] and r['rect'] and level in r['occupied_levels']]
                hits = [o['id'] for o in obs if _overlap(rect,o['rect'])]
                missing = _uncovered(rect,boards_by_level[level])
                if hits or missing:
                    causes.append({'level':level,'obstacles':hits,'unsupported_area_m2':_area(missing)})
            attempts.append({'end':end,'rect':rect,'length_m':length,'width_m':band,'failures':causes})
        selected = next((r for r in attempts if not r['failures']),None)
        results.append({'id':row['id'],'status':'pass' if selected else 'fail','actual_shell':measurement,
                        'selected_end':selected['end'] if selected else None,'attempts':attempts})
        if selected is None:
            errors.append({'code':'EXTRACTION_SPACE_BLOCKED','equipment':row['id'],'attempts':attempts})
    return {'rows':results,'violations':errors}


def _service_tiers(graph):
    units=graph['units'];pairs=set();rows=[];errors=[]
    companion={'condenser','exchanger','filter','fixed_bed','tower'}
    for transfer in graph.get('transfers',[]):
        source=transfer['source']
        for uid in transfer.get('path_units',[]):
            if uid in units and units[uid]['kind'] in companion:
                pairs.add((uid,source,'companion'))
        if transfer.get('condenser') in units:
            pairs.add((transfer['condenser'],source,'companion'))
        if transfer.get('mode')=='distillation' and transfer.get('target') in units:
            pairs.add((transfer['target'],source,'receiver'))
    for uid,unit in units.items():
        if unit['kind']=='condenser' and unit.get('owner') in units:
            pairs.add((uid,unit['owner'],'companion'))
    rank={'small':0,'medium':1,'large':2}
    for equipment,source,relation in sorted(pairs):
        unit,owner=units[equipment],units[source]
        tier,main=unit.get('size_tier'),owner.get('size_tier')
        allowed={'small':{'small'},'medium':{'small','large'},'large':{'large'}}.get(main,set())
        passed=(tier in rank and main in rank and
                (rank[tier]<=rank[main] if relation=='receiver' else tier in allowed))
        record={'equipment':equipment,'service_owner':source,'relation':relation,
                'source_size_tier':main,'equipment_size_tier':tier,'pass':passed}
        rows.append(record)
        if not passed:
            errors.append({'code':'SERVICE_TIER_MISMATCH',**record})
        if unit.get('service_owner') not in (None,source):
            errors.append({'code':'SERVICE_OWNER_MISMATCH','equipment':equipment,
                           'declared':unit['service_owner'],'actual_from_process':source})
    snapshot=graph.get('sizing')
    if not isinstance(snapshot,dict):
        errors.append({'code':'SIZING_SNAPSHOT_MISSING'})
    else:
        selected={u['id']:u for u in snapshot.get('equipment',[])}
        for uid,unit in units.items():
            if unit['kind'] not in EQUIPMENT:
                continue
            original=selected.get(uid)
            if original is None:
                errors.append({'code':'SIZING_EQUIPMENT_MISSING','equipment':uid})
                continue
            changed=[k for k in ('kind','form','size_tier') if k in original and original[k]!=unit.get(k)]
            before=original.get('dimension_spec');after=unit.get('dimension_spec')
            if isinstance(before,dict):
                if not isinstance(after,dict):changed.append('dimension_spec')
                else:
                    changed.extend('dimension_spec.'+k for k in ('diameter_m','main_cylinder_length_m','height_m',
                        'installed_height_m','support_to_bottom_m','support_to_top_m') if before.get(k)!=after.get(k))
            if changed:
                errors.append({'code':'SIZING_CHANGED_DURING_LAYOUT','equipment':uid,'fields':changed})
    main_count=sum(u['kind'] in EQUIPMENT-{'pump'} for u in units.values())
    if main_count<2:
        errors.append({'code':'MAIN_EQUIPMENT_TOO_FEW','actual':main_count,'required':2})
    return {'rows':rows,'violations':errors,'main_equipment_count':main_count}


def _model_scope_and_drives(graph,scene,components):
    from ..checks import box_overlaps
    scope=[];drive_errors=[];rows=[]
    for c in scene['components']:
        owner=graph['units'].get(c.get('owner'),{})
        if c['kind']=='rail' or (c['kind']=='motor' and owner.get('kind')=='vessel'
                                and owner.get('form',owner.get('orientation','vertical'))=='vertical'):
            scope.append({'code':'LAYOUT_DETAIL_FORBIDDEN','component':c['id'],'kind':c['kind']})
    for uid,unit in graph['units'].items():
        if unit['kind']!='vessel' or unit.get('form',unit.get('orientation','vertical'))!='vertical':
            continue
        spec=_declared_dimensions(unit,{})
        if spec is None:
            drive_errors.append({'code':'DRIVE_CATALOG_MISSING','equipment':uid})
            continue
        shells=[c for c in components if c.get('owner')==uid and c['kind'] in ('vessel','head')]
        drives=[c for c in components if c.get('owner')==uid and c['kind']=='drive_reservation']
        if not shells or len(drives)!=1:
            drive_errors.append({'code':'DRIVE_RESERVATION_MISSING','equipment':uid,'actual_count':len(drives)})
            continue
        shell=_union([c['box'] for c in shells]);drive=drives[0];b=drive['box']
        expected_bottom=shell[5];expected_top=shell[2]+spec['installed_height_m']
        # The retained original motor cylinder has r=.16 m; only this existing
        # estimated diameter supplies the new box width, never a fit adjustment.
        required_width=.32
        correct=(abs(b[2]-expected_bottom)<=EPS and abs(b[5]-expected_top)<=EPS
                 and abs(b[3]-b[0]-required_width)<=EPS and abs(b[4]-b[1]-required_width)<=EPS
                 and abs((b[0]+b[3])-(shell[0]+shell[3]))<=EPS
                 and abs((b[1]+b[4])-(shell[1]+shell[4]))<=EPS)
        hits=[c['id'] for c in components if c['id']!=drive['id']
              and c.get('owner')!=uid and c['kind'] not in ('rail','equipment_nozzle')
              and box_overlaps(b,c['box'])]
        record={'equipment':uid,'component':drive['id'],'actual_box':b,
                'expected_bottom_m':expected_bottom,'expected_top_m':expected_top,
                'width_m':required_width,'width_source':'Existing motor cylinder radius .16m; estimate',
                'dimensions_match':correct,'intrusions':hits}
        rows.append(record)
        if not correct:drive_errors.append({'code':'DRIVE_RESERVATION_DIMENSIONS_WRONG',**record})
        if hits:drive_errors.append({'code':'DRIVE_RESERVATION_INTRUDED',**record})
    return {'rows':scope,'violations':scope},{'rows':rows,'violations':drive_errors}


def _shell_selection(scene):
    selection=scene.get('shell_selection');errors=[]
    actual=scene.get('shell',{}).get('bays')
    if not isinstance(selection,dict):
        errors.append({'code':'SHELL_SELECTION_EVIDENCE_MISSING'})
        trials=[]
    else:
        trials=selection.get('trials',[])
        if actual==2:
            smaller=[t for t in trials if t.get('bays')==1]
            if not smaller or not any(t.get('all_rejected') is True and t.get('attempts',0)>0
                                      and t.get('refusal_codes') for t in smaller):
                errors.append({'code':'LARGER_SHELL_WITHOUT_SMALLER_REFUSAL'})
            if any(t.get('all_rejected') is False or t.get('passed',False) for t in smaller):
                errors.append({'code':'SMALLER_SHELL_ALREADY_FITS'})
    return {'selected_bays':actual,'selected_platforms':len(scene['floors'])-1,
            'trials':trials,'rows':trials,'violations':errors}


def _segment_intersects(a,b,box,radius=0.):
    a,b=np.asarray(a,float),np.asarray(b,float);delta=b-a
    length=float(np.linalg.norm(delta))
    if length<=EPS:return all(box[i]+EPS<a[i]<box[i+3]-EPS for i in range(3))
    transverse=radius*np.sqrt(np.maximum(0.,1-(delta/length)**2))
    low=np.asarray(box[:3])-transverse+EPS;high=np.asarray(box[3:])+transverse-EPS
    start,finish=0.,1.
    for i in range(3):
        if abs(delta[i])<=EPS:
            if not low[i]<a[i]<high[i]:return False
        else:
            first,last=sorted(((low[i]-a[i])/delta[i],(high[i]-a[i])/delta[i]))
            start=max(start,first);finish=min(finish,last)
            if finish-start<=EPS:return False
    return finish-start>EPS


def _operation_checks(graph,scene,components,rows,boards,nav,rules,profiles=None):
    from .rule_identity import identity
    records=scene.get('operation_reservations',[])
    records=records.values() if isinstance(records,dict) else records
    reservations={r.get('equipment',r.get('id')):r for r in records}
    measurements=[];errors=[]
    for row in rows:
        uid=row['id'];unit=graph['units'][uid]
        if not _vertical_operational(unit,row):continue
        ports={k:p for k,p in unit['ports'].items() if 'pos' in p}
        if not ports:
            errors.append({'code':'PHYSICAL_PORTS_MISSING','equipment':uid})
            continue
        for end,extreme in (('upper',max),('lower',min)):
            z=extreme(p['pos'][2] for p in ports.values())
            names=[name for name,p in ports.items() if abs(p['pos'][2]-z)<=EPS]
            record=reservations.get(uid,{}).get(end) or {}
            local=[]
            for name in names:
                pos=ports[name]['pos']
                nozzles=[c for c in row['parts'] if c['kind']=='equipment_nozzle' and
                         (c.get('port')==name or '_'+name+'_equipment_nozzle' in c['id'])]
                if not any(m and min(math.dist(p,pos) for p in m['ends'])<=EPS
                           for m in map(_cylinder_measurement,nozzles)):
                    local.append({'code':'OPERATION_INTERFACE_NOT_PHYSICAL','port':name})
            if record:
                port=ports.get(record.get('port'),{})
                interface=record.get('interface') or {}
                if record.get('port') not in names:
                    local.append({'code':'OPERATION_WRONG_DEVICE_END'})
                for key,base,code in (('position','pos','OPERATION_INTERFACE_NOT_REAL'),
                                      ('normal','normal','OPERATION_INTERFACE_NORMAL_NOT_REAL')):
                    value=interface.get(key) if isinstance(interface,dict) else None
                    if (not isinstance(value,(list,tuple)) or len(value)!=3 or base not in port
                            or not np.allclose(value,port[base],atol=EPS,rtol=0)):
                        local.append({'code':code})
            errors.extend({**issue,'equipment':uid,'end':end} for issue in local)
            measurements.append({'equipment':uid,'end':end,'port':record.get('port'),'physical_ports':names,
                'level':record.get('level'),'status':'fail' if local else 'record',
                'rule_identity':identity('equipment_top_access' if end=='upper' else 'stance'),
                'stance_box':record.get('stance_box'),'challenge':record.get('reason'),
                'scope':'Physical ports checked. Operating stance is evaluated after the real valve is fitted.'})
    return {'rows':measurements,'violations':errors,'scope':'Equipment ports and physical stubs. Operation checks are recorded separately.'}


def _occupancy(graph,rows,boards,rules,profiles=None):
    result=[]
    for level,standing in boards.items():
        base=_bounds(standing);standing_area=_area(standing)
        rects=[]
        for row in rows:
            if row['rect'] and level in row['occupied_levels']:
                profile=(profiles or {}).get((row['id'],level))
                r=profile['rect'] if profile else row['rect']
                w=profile['width_m'] if profile else maintenance_width_for(graph['units'][row['id']],rules)
                rects.append(envelope(r,profile or {},w))
        bounds=_bounds(rects)
        occupied=0. if bounds is None else (bounds[2]-bounds[0])*(bounds[3]-bounds[1])-_area(_uncovered(bounds,rects))
        result.append({'level':level,'equipment_and_maintenance_union_area_m2':occupied,
                       'standing_area_m2':standing_area,'ratio':occupied/standing_area if standing_area>EPS else None,
                       'threshold':None,'scope':'Informational ratio. The numerator includes equipment and its own openings. Ratios above one are retained.'})
    return {'rows':result,'violations':[]}


def _pump_forbidden(rows,scene):
    """The entire ground pump stays outside every real vessel/tank body shadow."""
    measurements=[];errors=[];ground=scene['floors'][0]
    vessels=[]
    for row in rows:
        if row['kind'] not in ('vessel','tank') or not row.get('rect'):
            continue
        body=[c for c in row['parts'] if c['kind'] in ('vessel','tank','head')]
        if body:
            vessels.append((row,_rect(_union([c['box'] for c in body])),body))
    for pump in rows:
        if pump['kind']!='pump' or not pump.get('rect'):continue
        grounded=abs(pump['z_range_m'][0]-ground)<=EPS
        if not grounded:
            errors.append({'code':'PUMP_NOT_ON_GROUND','equipment':pump['id'],
                           'actual_bottom_m':pump['z_range_m'][0],'ground_z_m':ground})
            continue
        for vessel,body_rect,body in vessels:
            overlap=_overlap(pump['rect'],body_rect)
            record={'pump':pump['id'],'vessel':vessel['id'],'ground_z_m':ground,
                    'actual_pump_rect':pump['rect'],'pump_component_ids':pump['component_ids'],
                    'actual_body_maximum_rect':body_rect,'body_component_ids':[c['id'] for c in body],
                    'overlap':overlap,'scope':'Maximum projected overlap of pump, motor and base against vessel bodies. Use the full footprint at every applicable level.'}
            measurements.append(record)
            if overlap:errors.append({'code':'PUMP_INSIDE_VESSEL_BODY_SHADOW',**record})
    return {'status':'fail' if errors else 'pass','rows':measurements,'violations':errors}


def _legacy_bare_interfaces(graph,scene,components,rows,boards,nav,model_available,rules):
    from ..checks import box_overlaps
    low,high=band()
    policy=RULES['geometry_screen'];reach=RULES['construction_choices']['stance_reach_proxy_m']
    records=scene.get('operation_reservations',[])
    if isinstance(records,dict):records=list(records.values())
    reservations={r.get('equipment',r.get('id')):r for r in records}
    result=[];errors=[]
    for row in rows:
        unit=graph['units'][row['id']]
        if not _vertical_operational(unit,row):
            continue
        ports=[(name,p) for name,p in unit.get('ports',{}).items() if 'pos' in p]
        if not ports:continue
        device={'equipment':row['id']}
        for end in ('upper','lower'):
            chosen=reservations.get(row['id'],{}).get(end,{}).get('port')
            name,port=next(((n,p) for n,p in ports if n==chosen),
                           sorted(ports,key=lambda p:(p[1]['pos'][2],p[0]),reverse=end=='upper')[0])
            interface=port['pos'];candidates=[]
            for level,z in enumerate(scene['floors']):
                h=interface[2]-z;within=low-EPS<=h<=high+EPS;found=None
                if within and len(nav[level][0]):
                    xs,ys,reachable=nav[level]
                    for xindex,yindex in np.argwhere(reachable):
                        x,y=xs[xindex],ys[yindex]
                        if math.dist([x,y],interface[:2])>reach+EPS:continue
                        box=[x-policy['station_width_m']/2,y-policy['station_depth_m']/2,z,
                             x+policy['station_width_m']/2,y+policy['station_depth_m']/2,z+policy['station_height_m']]
                        footprint=_rect(box)
                        if _overlap(footprint,row['rect']) or _uncovered(footprint,boards[level]):continue
                        volume=[*box[:2],z+EPS,*box[3:]]
                        if any(box_overlaps(volume,c['box']) for c in components if c['kind'] not in ('floor','ground','walk_surface','rail')):continue
                        found=box;break
                candidates.append({'level':level,'height_above_floor':h,'within_0_55_to_1_8':within,
                                   'has_board_reachable_stance':found is not None,'stance_box':found,
                                   'reasons':[] if found is not None else ['Bare port elevation is outside the earlier range'] if not within else ['No reachable standing area outside the physical deck']})
            passed=any(c['within_0_55_to_1_8'] and c['has_board_reachable_stance'] for c in candidates)
            device[end]={'port':name,'interface_z':interface[2],'interface_position':interface,
                         'actual_level_candidates':candidates,'would_pass':passed,
                         'reason':'Bare port has a reachable standing level' if passed else 'Bare port has no standing level reachable under the earlier criteria'}
            if not passed:errors.append({'code':'LEGACY_BARE_INTERFACE_FAILED','equipment':row['id'],'end':end})
        device['would_pass']=device['upper']['would_pass'] and device['lower']['would_pass'];result.append(device)
    return {'comparison':'Decision comparison on the same saved layouts',
            'physical_model_available':model_available,'eligible_for_comparison':model_available and bool(result),
            'would_pass':not errors,'rows':result,'violations':errors}


def check_layout(graph, scene, model=None):
    rules = read_layout_rules()
    width = rules['maintenance_width_m']
    components = _measured_components(scene, model)
    rows = equipment_footprints(graph, scene, model)
    instances = {r['id']:r for r in scene.get('equipment_instances',[])}
    errors, violations, measures = [], [], []
    floors = scene['floors']
    boards = {i:_standing_boards(components,i,z) for i,z in enumerate(floors)}
    structural = {i:_structural_obstacles(components,floors,i) for i in range(len(floors))}
    profiles={(row['id'],level):level_maintenance(row,graph['units'][row['id']],scene,level,rules)
              for row in rows for level in range(len(floors))}
    nav = {}
    for level in range(len(floors)):
        equipment = [{'id':r['id'],'rect':profiles[(r['id'],level)]['rect']} for r in rows
                     if r['rect'] and level in r['occupied_levels']]
        holes = [{'id':f'opening_{j}','rect':h} for j,h in enumerate(scene['holes'].get(level,scene['holes'].get(str(level),[])))]
        nav[level] = _reachability(boards[level],structural[level]+equipment+holes,
                                   scene.get('navigation_entries',{}).get(str(level)))
    for row in rows:
        if row['rect'] is None:
            violations.append({'code':'EQUIPMENT_SOLIDS_MISSING','equipment':row['id']})
            continue
        declared = instances.get(row['id'],{}).get('occupied_levels')
        purposes=set(row['occupied_levels'])
        if declared is not None and (not set(row['occupied_levels'])<=set(declared) or not set(declared)<=purposes):
            violations.append({'code':'OCCUPIED_LEVELS_MISMATCH','equipment':row['id'],
                               'declared':declared,'measured':row['occupied_levels']})
        for level in sorted(purposes):
            profile=profiles[(row['id'],level)]
            r=profile['rect'];width=profile['width_m']
            outer = [r[0]-width,r[1]-width,r[2]+width,r[3]+width]
            own, opening_records, opening_errors = _own_holes(row,scene,level,rules)
            violations.extend(opening_errors)
            other = [{'id':v['id'],'kind':v['kind'],'rect':profiles[(v['id'],level)]['rect']} for v in rows
                     if v['id'] != row['id'] and v['rect'] and level in v['occupied_levels']]
            own_keys = {tuple(h) for h in own}
            other_holes = [{'id':f'opening_{j}','kind':'opening','rect':h}
                           for j,h in enumerate(scene['holes'].get(level,scene['holes'].get(str(level),[])))
                           if tuple(h) not in own_keys]
            obstacles = other+structural[level]+other_holes
            boundary = _bounds(boards[level]) or r
            voids = _uncovered(boundary,boards[level]+[r]+own)
            side_obstacles = obstacles+[{'id':f'unsupported_{i}','rect':v} for i,v in enumerate(voids)]
            sides = {side:_side_width(r,side,boundary,side_obstacles)
                     for side in ('x_min','x_max','y_min','y_max')}
            reachable = _reachable_sides({**row,'rect':r},width,nav[level])
            required = {side:width for side in sides}
            if row['kind']=='pump':
                choices = long_sides(r)
                feasible = [side for side in choices if side in reachable and
                    all(sides[key]['minimum_clearance_m'] >= value-EPS
                        for key,value in pump_widths(side).items()) and
                    not _uncovered(envelope(r,{'pump_maintenance_side':side},width),boards[level]+[r]+own) and
                    not any(_overlap(envelope(r,{'pump_maintenance_side':side},width),o['rect']) for o in obstacles)]
                selected = feasible[0] if feasible else choices[0]
                profile.update(pump_maintenance_side=selected,reason='Physical maintenance aisle along the long pump side, with standard clearances on the other three sides')
                required = pump_widths(selected)
                outer = envelope(r,profile,width)
                reachable = [side for side in reachable if side==selected]
                if not feasible:
                    violations.append({'code':'PUMP_LONG_SIDE_AISLE_MISSING','equipment':row['id'],'level':level,
                                       'long_sides':list(choices),'required_m':width})
            hits = [o for o in obstacles if _overlap(outer,o['rect'])]
            if hits:
                violations.append({'code':'MAINTENANCE_OBSTRUCTED','equipment':row['id'],'level':level,
                                   'obstacles':[{'id':o['id'],'kind':o['kind']} for o in hits]})
            missing = _uncovered(outer,boards[level]+[r]+own)
            if missing:
                violations.append({'code':'MAINTENANCE_UNSUPPORTED','equipment':row['id'],'level':level,
                                   'missing_area_m2':_area(missing),'rectangles':missing})
            for side,measure in sides.items():
                if measure['minimum_clearance_m'] < required[side]-EPS:
                    violations.append({'code':'MAINTENANCE_WIDTH_SHORT','equipment':row['id'],
                                       'level':level,'side':side,'actual_m':measure['minimum_clearance_m'],'required_m':required[side]})
            if not reachable:
                violations.append({'code':'MAINTENANCE_NOT_REACHABLE','equipment':row['id'],'level':level})
            measures.append({'id':row['id'],'kind':row['kind'],'level':level,'footprint':r,'maintenance_rect':outer,
                             'required_width_m':width,'required_side_widths_m':required,
                             'purpose':'physical_equipment' if level in row['occupied_levels'] else 'operation_face',
                             'component_ids':profile['component_ids'],'maintenance_geometry':profile,
                             'sides':sides,'reachable_sides':reachable,
                             'own_openings':opening_records})
    operations=_operation_checks(graph,scene,components,rows,boards,nav,rules,profiles)
    valid_operating={r['level'] for r in operations['rows'] if r['status']=='pass'}
    empty = [i for i in range(1,len(floors)) if not any(i in r['occupied_levels'] for r in rows) and i not in valid_operating]
    errors.extend({'code':'EMPTY_PLATFORM','level':i} for i in empty)
    from .pump_alignment import check_pump_alignment
    pumps = check_pump_alignment(rows,instances,ORDINARY_GAP_M)
    extraction = _extraction(rows,scene,boards,structural,rules['extraction_width_extra_m'],instances,profiles)
    dimensions = check_equipment_dimensions(graph,scene,model,measures)
    service_tiers=_service_tiers(graph)
    model_scope,drive_reservations=_model_scope_and_drives(graph,scene,components)
    shell_selection=_shell_selection(scene)
    occupancy=_occupancy(graph,rows,boards,rules,profiles)
    pump_forbidden=_pump_forbidden(rows,scene)
    from .solid_collision_checks import check_solid_collisions
    solid_collisions=check_solid_collisions(graph,scene,model)
    from .pullout_reservations import check_pullout_reservations
    pullout=check_pullout_reservations(graph,scene,model)
    legacy=_legacy_bare_interfaces(graph,scene,components,rows,boards,nav,model is not None,rules)
    coverage = (check_platform_coverage(scene,model) if len(floors)>1 else
                {'status':'pass','levels':[],'errors':[],'scope':'No platforms are present. Ground maintenance surfaces are still checked per unit.'})
    errors.extend(violations+pumps['violations']+extraction['violations']+dimensions['violations']+coverage['errors'])
    for section in (operations,service_tiers,model_scope,drive_reservations,shell_selection,
                    pump_forbidden,solid_collisions,pullout):errors.extend(section['violations'])
    if any(c['kind'] in ('pipe','valve') for c in components) or graph.get('valves'):
        errors.append({'code':'PHYSICAL_PIPE_OR_VALVE_IN_LAYOUT_STAGE'})
    values = [side['minimum_clearance_m'] for row in measures for side in row['sides'].values()]
    distribution = {'count':len(values),'min':min(values) if values else None,'max':max(values) if values else None,
                    **{f'p{p}':float(np.percentile(values,p)) if values else None for p in (10,50,90)}}
    return {'status':'fail' if errors else 'pass','errors':errors,'rules':rules,
            'minimum_clear_width_m':distribution['min'],'maintenance_violations':violations,
            'maintenance':{'violations':violations,'rows':measures,'width_distribution_m':distribution},
            'empty_platforms':{'count':len(empty),'levels':empty},'pump_alignment':pumps,
            'extraction':extraction,'dimensions':dimensions,'coverage':coverage,
            'operations':operations,'service_tiers':service_tiers,'model_scope':model_scope,
            'drive_reservations':drive_reservations,'shell_selection':shell_selection,'occupancy':occupancy,
            'pump_forbidden':pump_forbidden,'solid_collisions':solid_collisions,
            'pullout':pullout,
            'legacy_v2_bare_interfaces':legacy,
            'used_platforms':[i for i in range(1,len(floors)) if i not in empty],
            'equipment_footprints':[{k:v for k,v in r.items() if k not in ('parts','_unit')} for r in rows],
            'method':'Physical equipment projections, occupied levels, four-side clear widths, union of decks and walkable connections. Process pipes and valves are outside the check.'}
