"""Provide room to work along the service side of a pump."""
AISLE_M = .8
ORDINARY_GAP_M = .1  # Modeling value; not an industrial standard.
SIDES = ('x_min', 'y_min', 'x_max', 'y_max')


def long_sides(rect):
    # Complete real pump/base footprint, measured again from XML at check time.
    return ('x_min', 'x_max') if rect[3]-rect[1] >= rect[2]-rect[0] else ('y_min', 'y_max')


def widths(side):
    return {key: AISLE_M if key == side else ORDINARY_GAP_M for key in SIDES}


def envelope(rect, profile, fallback):
    side = profile.get('pump_maintenance_side')
    w = widths(side) if side else {key: profile.get('width_m') or fallback for key in SIDES}
    return [rect[0]-w['x_min'], rect[1]-w['y_min'], rect[2]+w['x_max'], rect[3]+w['y_max']]
