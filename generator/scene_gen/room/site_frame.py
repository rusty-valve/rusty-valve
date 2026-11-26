"""Build the steel columns, beams and braces for the finished room."""
import numpy as np

from . import building
from .layout import add_box
from .operation_reservations import _segment_hits
from .shell import _frame_parts
from .site_grating import geometry_index


def build(scene, model, data, removed):
    parts = {part['id']:part for part in scene['components']}
    _, boxes, _ = geometry_index(scene, model, data, removed)
    scene['frame_spans'] = []
    for name, _, kind in _frame_parts(scene['shell']):
        box = parts[name]['box']
        low, high = np.asarray(box).reshape(2,3)
        axis = 2 if kind == 'column' else int(np.argmax((high-low)[:2]))
        list(building.i_member(scene, name, box, axis, kind))
        center = (low+high)/2
        if kind == 'column':
            add_box(scene, name+'_foot', [*low[:2], low[2]-.02, *high[:2], low[2]+.035],
                    'column', building.STEEL)
        else:
            top = center.copy()
            top[2] = high[2]
            a, b = top.copy(), top.copy()
            a[axis], b[axis] = low[axis], high[axis]
            scene['frame_spans'].append({'id':name, 'column_tops':[a.tolist(),b.tolist()],
                                         'beam_depth_m':float(high[2]-low[2])})
    omitted = []
    for name, start, end in building.knee_targets(scene):
        joint = start.copy()
        joint[2] = end[2]
        for scale in (1., .8, .6):
            a, b = joint+(start-joint)*scale, joint+(end-joint)*scale
            radius = building.RULES['frame']['knee_section_m']*.72
            close = np.all((boxes[:,3:] > np.minimum(a,b)-radius)
                           & (boxes[:,:3] < np.maximum(a,b)+radius), axis=1)
            if not any(_segment_hits(a,b,box,radius) for box in boxes[close]):
                building.i_knee(scene,name,a,b)
                break
        else:
            omitted.append(name)
    return omitted
