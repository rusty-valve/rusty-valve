"""Choose a root valve operator direction that clears nearby geometry."""
import numpy as np
from .piping_assets import place_asset
from .rule_identity import annotate_valve


def planning_geoms(valve):
    return [g for g in valve['world_geoms'] if not (valve.get('deferred_operator') and g.get('moving'))]


def finish(a):
    from .piping_valve_placement import obstacle_boxes, overlapping_obstacle_ids, physical_valve_obstacles
    valves=list(a.graph['valves'])+list(getattr(a,'planned_valves',[]))
    failures=[]
    for valve in valves:
        if not valve.get('deferred_operator'):continue
        from .generation_progress import step
        step(a,{'stage':'root_operator_orientation','edge':valve.get('edge'),'valve':valve['id']})
        axis=np.asarray(valve['axis']);stem=np.asarray(valve['stem']);choices=[];trials=[]
        ids, boxes=obstacle_boxes(physical_valve_obstacles(a,valve['id']))
        for quarter in range(4):
            candidate=place_asset(valve,valve['origin'],axis,stem)
            candidate['deferred_operator']=False
            hits=overlapping_obstacle_ids(candidate['world_geoms'],ids,boxes)
            rank=annotate_valve(a,candidate)
            trials.append({'quarter':quarter,'hits':hits})
            if not hits:choices.append((rank,quarter,candidate))
            stem=np.cross(axis,stem)
        if choices:
            _,quarter,candidate=min(choices,key=lambda x:x[:2]);valve.clear();valve.update(candidate)
            valve['operator_selection']={'stage':'after_all_pipes','selected_quarter':quarter,'trials':trials}
        else:
            valve['deferred_operator']=False
            valve['operator_selection']={'stage':'after_all_pipes','selected_quarter':None,'trials':trials}
            failures.append({'code':'ROOT_OPERATOR_NO_CLEAR_ORIENTATION','valve':valve['id'],'edge':valve.get('edge'),'reason':'All four root-valve lever directions intersect physical geometry'})
    return failures
