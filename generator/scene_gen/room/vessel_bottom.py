"""Route a bottom outlet to the side of a horizontal vessel."""
import numpy as np
from .piping_assets import inspect_asset


def applies(unit,endpoint):
    return (unit.get('kind')=='vessel' and unit.get('form',unit.get('orientation'))=='horizontal'
            and endpoint.rsplit('.',1)[-1]=='bottom')


def beyond_side(unit,point,axis):
    """Use the same transverse side and exterior face in placement and reporting."""
    k=1 if unit['rotation_deg']%180==0 else 0
    return (abs(axis[k])>.999 and (point[k]<unit['rect'][k]-1.e-8
            or point[k]>unit['rect'][k+2]+1.e-8))


def emit(a,unit,port,anchor,edge):
    from . import piping_material_build as b
    p=np.asarray(anchor['adapter_end']);dn=edge['dn'];r=a.radius(dn)
    length=inspect_asset(dn)['install_length_m']
    k=1 if unit['rotation_deg']%180==0 else 0
    rect=unit['rect'];down=np.array([0.,0.,-1.])
    # Leave a full straight beyond the side for the valve and its two connections.
    margins=[(rect[k]-a.bounds[k],-1),(a.bounds[k+3]-rect[k+2],1)]
    # A lower receiver needs a free vertical approach to its own top feed.
    # Prefer the vessel flank that leaves that approach clear.
    scores=[]
    for margin,sign in margins:
        end_k=rect[k+(2 if sign>0 else 0)]+sign*(2*length+.09)
        conflicts=0
        for transfer in a.frozen['graph'].get('transfers',[]):
            if transfer.get('source')!=unit['id']:continue
            receiver=(a.requirements['ports'].get(transfer['target']+'.feed')
                      or a.requirements['ports'].get(transfer['target']+'.liquid_return'))
            if not receiver or receiver['normal'][2]<.9:continue
            q=receiver['position']
            if q[2]<p[2] and abs(q[1-k]-p[1-k])<2*r+.05 and min(p[k],end_k)-r<=q[k]<=max(p[k],end_k)+r:
                conflicts+=1
        scores.append((conflicts,-margin,sign))
    failures=[]
    for _,_,sign in sorted(scores):
        side=np.eye(3)[k]*sign
        end=p+down*(3*r+.015)
        end[k]=rect[k+(2 if sign>0 else 0)]+sign*(2*length+.09)
        try:
            points=a.search(p,end,dn,skip=tuple(anchor['components'])+(port['nozzle_component'],),
                port_contacts=b._departure_contacts(anchor),
                start_direction=down,end_direction=side,start_straight_m=.01,
                end_straight_m=3*length+.02,complete_bend_planes=True)
            if len(points)!=3:raise b.PipingRefused('No single-bend route from the vessel bottom directly to the vessel side')
        except b.KNOWN_REFUSALS as exc:
            failures.append(str(exc));continue
        run=a.route('P2_'+edge['id']+'_vessel_side',points,dn,b._medium(edge),'vessel_bottom_side',
            port['level'],equipment=unit['id'],source={'endpoint':port['endpoint'],'position':p.tolist()},
            target={'position':end.tolist()},**b._meta(edge))
        b._join_contacts(a,run,p)
        return {**anchor,'point':end,'normal':side,'bottom_side_run':run['id']}
    raise b.PipingRefused(port['endpoint']+' The two side discharge straights are blocked: '+'；'.join(failures))
