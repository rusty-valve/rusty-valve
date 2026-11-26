"""Check where a valve can fit and how its operator can face."""
import numpy as np


EPS = 1.e-6


def on_straight(inlet, outlet, axis, bounds, blockers, position_ok):
    lo, hi = bounds
    delta = np.asarray([inlet, outlet])-lo
    stations = delta@axis
    if (np.any(np.linalg.norm(delta-stations[:,None]*axis, axis=1) > EPS)
            or stations[0] < -EPS or stations[1] > (hi-lo)@axis+EPS):
        return ['outside the original straight']
    crossed = [name for p, name in blockers if stations[0]+EPS < p < stations[1]-EPS]
    if crossed:
        return crossed
    return [] if position_ok else ['violates the role-specific port, merge or ordering requirement']

def within_floor(envelope, floor, upper):
    if envelope[2] < floor-EPS:
        return ['below the floor surface']
    if envelope[5] > upper+EPS:
        return ['intersects the upper slab']
    return []

def direction_preference(facing, allowed):
    if allowed is None:
        return 0
    sides = outward_quadrants(np.asarray(facing)[:2])
    if not sides:
        return 2
    return int(not any(side @ preferred > 1 - EPS for side in sides for preferred in allowed))


def outward_quadrants(radial):
    """Room-axis sides; diagonal boundaries admit either neighboring side."""
    radial = np.asarray(radial)
    if np.linalg.norm(radial) <= EPS:
        return ()
    largest = max(abs(radial))
    return tuple(np.eye(2)[i]*np.sign(radial[i]) for i in range(2) if abs(radial[i]) >= largest-EPS)

def operator_orientation(stem, installation):
    reasons=[]
    if np.count_nonzero(abs(stem)>EPS) != 1:
        reasons.append('stem is not along an orthogonal axis')
    if stem[2] < -1. + EPS:
        reasons.append('stem faces down')
    if installation[2] < -1. + EPS:
        reasons.append('operator face points down')
    return reasons

def automatic_fixed(before, after):
    return sorted(name for name, geometry in before.items() if after.get(name) != geometry)
