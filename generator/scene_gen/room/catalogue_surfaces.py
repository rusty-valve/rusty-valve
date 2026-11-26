"""Join the separate surfaces of an equipment flange."""
from collections import Counter, defaultdict
import re

import numpy as np
from scipy.spatial import cKDTree

from .reactor_f_series_fittings import _frame


_SECTOR = re.compile(r'(.+_port_([^_]+))_(plate|rf|neck|rim|blank_ring)_\d+$')
_WELD_M = 5e-7


def weld(points):
    """Share vertices at source seams within compiled mesh precision."""
    points, inverse = np.unique(points, axis=0, return_inverse=True)
    roots = np.arange(len(points))
    for a, b in cKDTree(points).query_pairs(_WELD_M, output_type='ndarray'):
        while roots[a] != a:
            a = roots[a]
        while roots[b] != b:
            b = roots[b]
        roots[max(a, b)] = min(a, b)
    for i, parent in enumerate(roots):
        roots[i] = roots[parent]
    unique, remap = np.unique(roots, return_inverse=True)
    return points[unique], remap[inverse]


def _perimeter_surface(parts, center, frame):
    triangles = (np.concatenate([v[f] for v, f in parts]) - center) @ frame.T
    low, high = triangles[:, :, 2].min(), triangles[:, :, 2].max()
    caps = triangles[np.max(abs(triangles[:, :, 2] - low), axis=1) < 4 * _WELD_M, :, :2]
    signed_area = ((caps[:, 1, 0] - caps[:, 0, 0]) * (caps[:, 2, 1] - caps[:, 0, 1])
                   - (caps[:, 2, 0] - caps[:, 0, 0]) * (caps[:, 1, 1] - caps[:, 0, 1]))
    caps[signed_area < 0] = caps[signed_area < 0, ::-1]
    points, remap = weld(caps[abs(signed_area) > 1e-13].reshape(-1, 2))
    faces = remap.reshape(-1, 3)
    directed = [(int(a), int(b)) for face in faces for a, b in zip(face, np.roll(face, -1))]
    counts = Counter(tuple(sorted(edge)) for edge in directed)
    outline = [edge for edge in directed if counts[tuple(sorted(edge))] == 1]
    if not outline or max(counts.values()) > 2:
        raise ValueError('Flange sectors cannot form a continuous face')
    n = len(points)
    vertices = np.c_[points, np.full(n, low)]
    walls = [(i, j, j + n) for i, j in outline] + [(i, j + n, i + n) for i, j in outline]
    world = np.vstack((vertices, np.c_[points, np.full(n, high)])) @ frame + center
    # Separate cap and wall normals while keeping their geometric edges shared.
    return np.vstack((world, world)), np.vstack((faces[:, ::-1], faces + n, np.asarray(walls) + 2*n))


def flange_surfaces(model, material_groups, spec, origin, matrix, prefix):
    """Return separately named saved surfaces and the positions to verify on reload."""
    sectors = defaultdict(list)
    native = {}
    ports = {p['source_tag']: p for p in spec['ports'].values()}
    native_names = {p['source_geom'] for p in ports.values() if p.get('source_geom')}
    for rgba, parts in material_groups.items():
        ordinary = []
        for vertices, faces, gid in parts:
            name = model.geom(gid).name
            match = _SECTOR.fullmatch(name)
            if match:
                stem, tag, section = match.groups()
                sectors[stem, tag, section].append((vertices, faces, rgba))
            elif name in native_names:
                native[name] = (vertices, faces, rgba)
            else:
                ordinary.append((vertices, faces, gid))
        material_groups[rgba] = ordinary

    output = []
    evidence = []
    frames = {}
    for (stem, tag, section), parts in sectors.items():
        if section not in ('plate', 'rim'):
            continue
        vertices = np.concatenate([v for v, _, _ in parts])
        if tag in ports:
            port = ports[tag]
            center = matrix @ port['position'] + origin
            frame = np.stack(_frame(matrix @ port['normal']))
            radii = np.full(2, port['flange_outer_diameter_m'] / 2)
        else:
            centered = vertices - vertices.mean(axis=0)
            _, axes = np.linalg.eigh(centered.T @ centered)
            normal, major = axes[:, 0], axes[:, 2]
            frame = np.stack((major, np.cross(normal, major), normal))
            local = vertices @ frame.T
            lower, upper = local.min(axis=0), local.max(axis=0)
            center = (lower + upper) / 2 @ frame
            radii = (upper - lower)[:2] / 2
        frames[stem] = center, frame, radii

    for (stem, tag, section), parts in sectors.items():
        center, frame, radii = frames[stem]
        vertices, faces = _perimeter_surface([(v, f) for v, f, _ in parts], center, frame)
        name = prefix + stem + '_' + section + '_surface'
        output.append((name, vertices, faces, parts[0][2]))
        if section in ('plate', 'rim'):
            evidence.append({'source_tag': tag, 'component': name, 'center_m': center.tolist(),
                             'frame': frame.tolist(), 'radii_m': radii.tolist(),
                             'axial_limits_m': [float(v) for v in (
                                 ((vertices - center) @ frame[2]).min(), ((vertices - center) @ frame[2]).max())],
                             'source_sector_count': len(parts)})

    for name, (vertices, faces, rgba) in native.items():
        component = prefix + name + '_surface'
        output.append((component, vertices, faces, rgba))
        for tag, port in ports.items():
            if port.get('source_geom') != name:
                continue
            frame = np.stack(_frame(matrix @ port['normal']))
            center = matrix @ port['position'] + origin
            radius = port['flange_outer_diameter_m'] / 2
            depth = port.get('flange_thickness_m') or port.get('flange_reference', {}).get('plate_thickness_mm', 22)/1000
            evidence.append({'source_tag': tag, 'component': component, 'center_m': center.tolist(),
                             'frame': frame.tolist(), 'radii_m': [radius, radius],
                             'axial_limits_m': [-depth - .004, .001], 'source_sector_count': 1})
    return output, evidence
