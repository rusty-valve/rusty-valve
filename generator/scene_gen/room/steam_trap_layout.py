"""Place a steam trap and locate its inlet and outlet."""
import numpy as np
from .pipe_spec import load
from .piping_assets import inspect_asset


def trap_layout(physical,rect,floors,dn):
    p=np.asarray(physical['pos'],float);normal=np.asarray(physical['normal'],float)
    r=load()['pipe_outer_diameter_mm'][str(dn)]/2000
    ref=physical.get('operation_reference',{})
    level=int(ref['level']) if ref.get('status')=='standing_face_reference' else max((i for i,z in enumerate(floors) if z<=p[2]+1e-8),default=0)
    length,width,height=max(.4,12*r),max(.3,8*r),max(.2,6*r)
    if abs(normal[2])<.5:
        direction=normal.copy();axis=int(np.argmax(abs(direction)))
    else:
        _,axis,sign=min((abs(p[k]-rect[k+(2 if sign>0 else 0)]),k,sign) for k in (0,1) for sign in (-1.,1.))
        direction=np.zeros(3);direction[axis]=sign
    L=inspect_asset(dn)['install_length_m'];R=3*r
    center=p.copy();center[2]=floors[level]+height/2+.05
    if center[2]>=p[2]-1e-8:raise ValueError('Trap inlet cannot sit below the original condensate outlet')
    face=rect[axis+(2 if direction[axis]>0 else 0)]
    center[axis]=face+direction[axis]*(length/2+.12+3*L+2*R+.05)
    required=length/2+.12+4*r+(3*L+.02)+2*R+.05
    if (center[axis]-p[axis])*direction[axis]<required:center[axis]=p[axis]+direction[axis]*required
    ext=np.array([width/2,width/2,height/2]);ext[axis]=length/2
    body=np.r_[center-ext,center+ext].tolist();ports={}
    for name,sign in (('in',-1),('out',1)):
        n=direction*sign;face=center+n*length/2
        ports[name]={'pos':(face+n*max(2*r,.12)).tolist(),'normal':n.tolist(),'body_face':face.tolist()}
    # These boxes reserve required physical pieces, never a standing area.
    # The inlet corridor includes its native isolation length and two turns.
    def tube_box(a,b,outer):
        a,b=np.asarray(a),np.asarray(b)
        return np.r_[np.minimum(a,b)-outer,np.maximum(a,b)+outer].tolist()
    out=np.asarray(ports['out']['pos']);out_end=out+direction*(2*r+2*r+.05+2*R+.05)
    reservations=[{'kind':'trap_body','box':body},
        {'kind':'trap_inlet_and_isolation','box':tube_box(p,ports['in']['pos'],r+.05)},
        {'kind':'trap_outlet_and_first_turn','box':tube_box(ports['out']['body_face'],out_end,r+.05)}]
    return {'center':center,'direction':direction,'axis':axis,'level':level,'length':length,'width':width,'height':height,
            'radius':r,'body_box':body,'ports':ports,'reservations':reservations}
