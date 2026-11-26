"""Create insulation shells that follow pipes and equipment."""
from dataclasses import dataclass
import math

import numpy as np


THICKNESS = .05
HANDLE_GAP = .01
SECTIONS = 48
EPS = 1e-8
ANGLES = np.arange(SECTIONS)*2*np.pi/SECTIONS


def unit(vector):
    return vector/np.linalg.norm(vector)


def axis_basis(axis):
    side = unit(np.cross(axis, np.eye(3)[np.argmin(abs(axis))]))
    return side, np.cross(axis, side)


@dataclass
class Tube:
    points: np.ndarray
    radii: np.ndarray
    tangent: np.ndarray
    plane: np.ndarray | None = None

    @classmethod
    def straight(cls, ends, radii):
        points = np.asarray(ends, dtype=float)
        direction = unit(points[1]-points[0])
        return cls(points, np.broadcast_to(radii, len(points)), np.tile(direction, (len(points), 1)))

    @classmethod
    def elbow(cls, record):
        points = np.asarray(record['centerline_points'], dtype=float)
        center = np.asarray(record['center'], dtype=float)
        plane = unit(np.asarray(record['plane_normal'], dtype=float))
        tangent = np.cross(plane, points-center)
        if tangent[0]@(points[1]-points[0]) < 0:
            tangent = -tangent
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        return cls(points, np.full(len(points), record['pipe_outer_radius_m']), tangent, plane)

    def bounds(self, thickness):
        r = np.max(self.radii)+thickness
        return np.r_[self.points.min(axis=0)-r, self.points.max(axis=0)+r]

    def distance(self, points, thickness=0.):
        """Distance to the finite cylinders enclosing the saved profile.

        Adjacent elbow stations use the saved arc chords. The outer radius
        makes a reducers' short measuring cylinder conservative.
        """
        best = math.inf
        for index, (start, end) in enumerate(zip(self.points, self.points[1:])):
            length = np.linalg.norm(end-start)
            if length <= EPS:
                continue
            along = (end-start)/length
            delta = points-start
            axial = delta@along
            radial = np.linalg.norm(delta-axial[:, None]*along, axis=1)
            r = max(self.radii[index:index+2])+thickness
            gap = np.hypot(np.maximum(radial-r, 0.), np.maximum.reduce(
                [-axial, axial-length, np.zeros(len(points))]))
            best = min(best, float(np.min(gap)))
        return best

    def skin(self, thickness):
        frames = []
        for direction in self.tangent:
            a, b = axis_basis(direction) if self.plane is None else (self.plane, np.cross(direction, self.plane))
            frames.append(np.cos(ANGLES)[:, None]*a+np.sin(ANGLES)[:, None]*b)
        frames = np.asarray(frames)
        mid = self.points[:, None, :]
        inner = mid+self.radii[:, None, None]*frames
        outer = mid+(self.radii+thickness)[:, None, None]*frames
        count = len(inner)*SECTIONS
        outer_faces = ring_faces(len(inner))
        cap_faces = []
        for first in (0, count-SECTIONS):
            for i in range(SECTIONS):
                a, b = first+i, first+(i+1)%SECTIONS
                cap_faces.extend(((a, b, a+count), (b, b+count, a+count)))
        return np.concatenate((outer.reshape(-1, 3), inner.reshape(-1, 3))), np.vstack(
            (outer_faces, outer_faces[:, ::-1]+count, cap_faces))


def ring_faces(count):
    faces = []
    for layer in range(count-1):
        for i in range(SECTIONS):
            a, b = layer*SECTIONS+i, layer*SECTIONS+(i+1)%SECTIONS
            faces.extend(((a, b, b+SECTIONS), (a, b+SECTIONS, a+SECTIONS)))
    return np.asarray(faces, dtype=int)


def equipment_skin(center, axis, radius, length, thickness, *, head=False, apertures=()):
    """A cylindrical or ellipsoidal half-head surface with open port collars."""
    a, b = axis_basis(axis)
    frame = np.cos(ANGLES)[:, None]*a+np.sin(ANGLES)[:, None]*b
    if head:
        angle = np.linspace(0, np.pi/2-1.e-5, 27)
        axial, radial = length*np.sin(angle), radius*np.cos(angle)
        normals = np.c_[np.sin(angle)/length, np.cos(angle)/radius]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        axial += thickness*normals[:, 0]
        radial += thickness*normals[:, 1]
    else:
        axial = np.linspace(0., length, max(2, math.ceil(length/.025)+1))
        radial = np.full(len(axial), radius+thickness)
    vertices = (center+axial[:, None, None]*axis+radial[:, None, None]*frame).reshape(-1, 3)
    faces = ring_faces(len(axial))
    triangles = vertices[faces]
    preserve = np.ones(len(faces), dtype=bool)
    for pos, normal, r in apertures:
        delta = triangles-pos
        axial_gap = delta@normal
        radial_gap = np.linalg.norm(delta-axial_gap[..., None]*normal, axis=2)
        preserve &= ~np.any((axial_gap > -.4) & (radial_gap < r+.006), axis=1)
    return vertices, faces[preserve]


def valve_skin(valve, gap=.002):
    """Fit the insulated body case below the native bonnet's upper plane."""
    origin = np.asarray(valve['origin'])
    matrix = np.asarray(valve['matrix'])
    bonnet = next(c for c in valve['local_geoms'] if c['name'] == 'bonnet')
    body = [c['bbox'] for c in valve['local_geoms']
            if not c['moving'] and (c['name'].startswith(('visual_center_body', 'visual_right_end',
                'visual_left_end', 'collar_')) or c['name'] == 'bonnet')]
    body = np.asarray(body).reshape(-1, 2, 3)
    lo, hi = body[:, 0].min(axis=0), body[:, 1].max(axis=0)
    local_ports = [(np.asarray(p['position'])-origin)@matrix for p in valve['world_ports'].values()]
    lo[0], hi[0] = sorted(float(p[0]) for p in local_ports)
    cap = float(bonnet['bbox'][5])
    hi[2] = cap
    shaft = max(abs(x) for x in (bonnet['bbox'][0], bonnet['bbox'][1], bonnet['bbox'][3], bonnet['bbox'][4]))+gap
    return origin, matrix, lo, hi, shaft


def valve_case(origin, matrix, lo, hi, shaft):
    # The open square in the upper cladding exposes the entire stem above the bonnet.
    rim = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    hole = [(-shaft, -shaft), (shaft, -shaft), (shaft, shaft), (-shaft, shaft)]
    vertices = np.asarray([(*p, z) for z in (lo[2], hi[2]) for p in rim]+[(*p, hi[2]) for p in hole])
    faces = [(0, 2, 1), (0, 3, 2)]
    for a in range(4):
        b = (a+1)%4
        faces.extend([(a, b, b+4), (a, b+4, a+4), (a+4, b+4, b+8), (a+4, b+8, a+8)])
    return vertices@matrix.T+origin, np.asarray(faces, dtype=int)


def lever_case(origin, matrix, lo, hi, shaft, thickness, end_top):
    """Six sections give straight cuffs and flat shoulders on the valve body."""
    length = hi[0]-lo[0]
    x = lo[0]+length*np.array([0., .12, .29, .71, .88, 1.])
    if not x[2] < -shaft < shaft < x[3]:
        raise ValueError("The stem opening extends outside the straight shell's center section")
    vertices = []
    for index, axis in enumerate(x):
        shoulder = index in (2, 3)
        narrow = 0. if shoulder else .6*thickness
        top = hi[2] if shoulder else min(hi[2], end_top+.2*thickness)
        vertices.extend([
            (axis, lo[1]+narrow, lo[2]+narrow), (axis, hi[1]-narrow, lo[2]+narrow),
            (axis, hi[1]-narrow, top), (axis, lo[1]+narrow, top),
        ])
    faces = [(0, 2, 1), (0, 3, 2), (20, 21, 22), (20, 22, 23)]
    for section in range(5):
        for side in range(4):
            if section == 2 and side == 2:
                continue
            a = 4*section+side
            b = 4*section+(side+1)%4
            faces.extend([(a, b, b+4), (a, b+4, a+4)])
    vertices.extend([(xx, yy, hi[2]) for xx, yy in
                     ((-shaft, lo[1]), (shaft, lo[1]), (shaft, hi[1]), (-shaft, hi[1]),
                      (-shaft, -shaft), (shaft, -shaft), (shaft, shaft), (-shaft, shaft))])
    for a,b,c,d in ((11,24,27,10),(25,15,14,26),(24,25,29,28),(31,30,26,27)):
        faces.extend(((a,b,c),(a,c,d)))
    return np.asarray(vertices)@matrix.T+origin, np.asarray(faces, dtype=int)
