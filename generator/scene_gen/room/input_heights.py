"""Set equipment elevations from catalogue dimensions."""
from .recipe import CandidateRefused

def assign_levels(graph,shell,reference_vessel_levels=None):
    """Choose useful operation levels from physical form and process direction."""
    from .equipment_catalog import resolve_spec
    units=graph['units'];height=shell['storey_height_m'];limit=shell['platforms']
    for unit in units.values():
        unit['level']=0
        if unit['kind']=='vessel':
            spec=resolve_spec(unit)
            default_level = 1 if spec['form']=='vertical' else 0
            unit['level'] = (reference_vessel_levels or {}).get(unit['id'], default_level)
    for unit in units.values():
        if unit['kind']=='condenser':
            source=units[unit['owner']];a=resolve_spec(source);b=resolve_spec(unit)
            source_top=source['level']*height+a['support_to_top_m']
            options=[level for level in range(limit+1)
                     if level*height+b['support_to_bottom_m']>source_top]
            if not options:raise CandidateRefused('The condenser bottom must remain above the source reactor top. The permitted platform count is insufficient.')
            unit['level']=min(options)
        elif unit['kind']=='jacket':unit['level']=units[unit['owner']]['level']
        if unit['level']>limit:
            raise CandidateRefused(f'The process and equipment form require {unit["level"]} platform levels, but the shell permits only {limit} level')


def check(graph,shell):
    from copy import deepcopy
    work=deepcopy(graph)
    try:assign_levels(work,shell)
    except CandidateRefused as error:return {"status":'invalid input',"reason":str(error)}
    return {"status":'valid input',"levels":{k:v["level"] for k,v in work["units"].items()}}
