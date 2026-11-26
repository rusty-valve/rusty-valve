"""Build pump and strainer models from measured dimensions."""
from copy import deepcopy
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .catalogue_flange import _drilled_profile, extruded_profile
from .export import nums
from .reactor_f_series_fittings import _frame


DIMENSIONS = json.loads(Path(__file__).with_name('equipment_models_dimensions.json').read_text())
ASSETS = tuple(DIMENSIONS)
MARK = 'scene_gen dimensional redraw v1'
STEEL = (.57, .65, .69)
MOTOR = (.20, .44, .51)
BASE = (.31, .38, .42)


def specifications():
    specs = deepcopy(DIMENSIONS)
    for key, spec in specs.items():
        note = spec.pop('reference_dimensions')
        spec.update(xml=key+'/redrawn.xml', facts={'geometry_source': MARK, 'dimension_reference': note})
        if key == 'pump':
            spec['primary_families'] = ['new_base', 'new_pedestal', 'new_pump_foot', 'new_motor_foot', 'new_volute',
                                        'new_bearing', 'new_coupling_guard', 'new_motor', 'new_fan_cowl',
                                        'new_motor_fin', 'new_motor_terminal', 'new_in_nozzle', 'new_out_nozzle']
        for role, port in spec['ports'].items():
            name = f'new_{role}_flange'
            port.update(source_geom=name, contact_geoms=[name],
                        description=f'{key} {role}, rebuilt open mating face')
            if key == 'pump':
                port['neck_geoms'] = [f'new_{role}_nozzle']
    return specs


def _geom(world, name, color, **attributes):
    ET.SubElement(world, 'geom', name=name, rgba=nums([*color, 1]),
                  **{'type': attributes.pop('kind')}, **attributes)


def _box(world, name, low, high, color):
    low, high = np.asarray(low), np.asarray(high)
    _geom(world, name, color, kind='box', pos=nums((low+high)/2), size=nums((high-low)/2))


def _cylinder(world, name, start, end, radius, color):
    _geom(world, name, color, kind='cylinder', fromto=nums([*start, *end]), size=str(radius))


def _surface_profile(port, outer, holes):
    center = np.asarray(port['position'])
    frame = _frame(port['normal'])
    # The eccentric SZ03 bore has a larger physical opening than its
    # guaranteed concentric passage. Keep both dimensions independent.
    ellipse = port.get('opening_semi_axes_mm')
    inner = ellipse[0]/1000 if ellipse else port['opening_diameter_m']/2
    offsets, faces = _drilled_profile(frame, center, inner, outer, holes)
    if ellipse:
        bore = np.isclose(np.linalg.norm(offsets, axis=1), inner, rtol=0, atol=1e-10)
        offsets[bore, 2] *= ellipse[1]/ellipse[0]
        offsets[bore] += np.asarray(port['opening_center_m'])-center
    return offsets, faces


def _faces(folder, assets, world, port, role):
    raised = port.get('flange_reference', {}).get('rf_height_mm', 0)/1000
    thick = port['flange_thickness_m']
    outer = port['flange_outer_diameter_m']/2
    holes = [{**hole, 'center_world_m': hole['center_m']} for hole in port['bolt_holes']]
    sections = [(outer, holes, thick, raised)]
    if raised:
        sections.append((port['contact_face_outer_diameter_m']/2, [], raised, 0.))
    _mesh_sections(folder, assets, world, port, f'new_{role}_flange', sections, STEEL)


def _mesh_sections(folder, assets, world, port, name, sections, color):
    """Write one MJCF mesh with each independently extruded annular section."""
    center = np.asarray(port['position'])
    axis = np.asarray(port['normal'])
    vertices, triangles = [], []
    for outer, holes, thickness, depth in sections:
        offsets, faces = _surface_profile(port, outer, holes)
        points, surface = extruded_profile(center-(depth+thickness)*axis, axis, offsets, faces, thickness)
        triangles.extend(surface+len(vertices)+1)
        vertices.extend(points)
    filename = name+'.obj'
    (folder/filename).write_text(''.join('v %.12g %.12g %.12g\n' % tuple(v) for v in vertices)
                                 + ''.join('f %d %d %d\n' % tuple(f) for f in triangles))
    ET.SubElement(assets, 'mesh', name=name, file=filename)
    _geom(world, name, color, kind='mesh', mesh=name)


def _pump(folder, assets, world, spec):
    hi = spec['bounds_m']['max']
    base = spec['base_bounds_m']
    _box(world, 'new_base', base['min'], base['max'], BASE)
    z_base = base['max'][2]
    _box(world, 'new_pedestal', [-.48, -.087, z_base], [-.38, .175, .106], BASE)
    _box(world, 'new_pump_foot', [-.63, -.09, .09], [-.42, .19, .23], STEEL)
    _box(world, 'new_motor_foot', [.36, -.105, z_base], [.67, .193, .244], BASE)
    y, z = .04379, .36417
    _cylinder(world, 'new_volute', [-.761, y, z], [-.524, y, z], .220, STEEL)
    _cylinder(world, 'new_bearing', [-.531, y, z], [.274, y, z], .049, STEEL)
    _cylinder(world, 'new_coupling_guard', [.042, y, z], [.25, y, z], .102, BASE)
    _cylinder(world, 'new_motor', [.266, y, .395], [.737, y, .395], .169, MOTOR)
    _cylinder(world, 'new_fan_cowl', [.721, y, z], [hi[0], y, z], .146, MOTOR)
    for i in range(8):
        x = .305+i*.05
        _cylinder(world, f'new_motor_fin_{i:02}', [x, y, .395], [x+.013, y, .395], .1706, MOTOR)
    _box(world, 'new_motor_terminal', [.54, -.035, .525], [.688, .122, .61], MOTOR)
    for role, port in spec['ports'].items():
        depth = port['flange_thickness_m']
        if role == 'in':
            length, radius = .041, .077
        else:
            # The discharge neck descends directly from the fixed flange into the volute.
            length, radius = port['position'][2]-depth-.435, .081
        _mesh_sections(folder, assets, world, port, f'new_{role}_nozzle',
                       [(radius, [], length, depth)], STEEL)


def _filter(folder, assets, world, spec):
    _mesh_sections(folder, assets, world,
                   {'position': [.097, 0., 0.], 'normal': [1, 0, 0], 'opening_diameter_m': .05},
                   'housing_00', [(.038, [], .194, 0.)], STEEL)
    _cylinder(world, 'housing_01', [.04, 0., .009], [-.033, 0., -.110], .043, STEEL)
    # The removable cap sets the surveyed low point of the Y branch.
    low = spec['bounds_m']['min'][2]
    _cylinder(world, 'housing_02', [-.033, 0., -.107], [-.033, 0., low], .044, BASE)


def write_asset(folder, key, spec):
    folder.mkdir(parents=True, exist_ok=True)
    root = ET.Element('mujoco', model=key+'_dimensional_redraw')
    ET.SubElement(root, 'compiler', angle='radian')
    assets, world = (ET.SubElement(root, name) for name in ('asset', 'worldbody'))
    for role, port in spec['ports'].items():
        _faces(folder, assets, world, port, role)
    if key == 'pump':
        _pump(folder, assets, world, spec)
    else:
        _filter(folder, assets, world, spec)
    ET.indent(root)
    ET.ElementTree(root).write(folder / 'redrawn.xml', encoding='utf-8', xml_declaration=True)
