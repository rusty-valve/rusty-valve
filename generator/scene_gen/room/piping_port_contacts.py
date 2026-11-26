"""Measure how a straight pipe meets an equipment port."""
import numpy as np

from .building_equipment_checks import EPS


def outward_pipe_bounds(first, last, radius, position, normal, *, point_probe=False):
    """Return the physical bounds on the connection axis, without a rounded cap."""
    first, last, position, normal = map(np.asarray, (first, last, position, normal))
    if np.max(abs(last-position)) <= EPS:
        first, last = last, first
    if np.max(abs(first-position)) > EPS:
        return None
    delta = last-first
    if not (point_probe and np.max(abs(delta)) <= EPS):
        if float(delta@normal) <= EPS or np.max(abs(np.cross(delta,normal))) > EPS:
            return None
    extent = radius*np.sqrt(np.maximum(0.,1.-normal**2))
    return np.minimum(first,last)-extent, np.maximum(first,last)+extent


def flat_pipe_clears(bounds, box):
    if bounds is None:
        return False
    low, high = bounds
    return bool(np.any(high <= box[:3]+EPS) or np.any(low >= box[3:]-EPS))
