"""Summarize pipe connections, installed valves and unresolved checks."""
import math


def _count(section,key):
    value=section.get(key) if isinstance(section,dict) else None
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or int(value)!=value:
        return None
    return int(value)


def pipeline_status(measure=None,piping=None,result=None,*,model_present=False):
    """Return pass/fail/incomplete and measured counts, with None for unknowns.

    The pipe export omits the optional collision_breaks key when none were
    removed. Interpret that omission only after the complete piping record is
    present. A missing piping record or measurement never means zero.
    """
    result=result if isinstance(result,dict) else {}
    measured=measure if isinstance(measure,dict) else {}
    pipe=piping if isinstance(piping,dict) else {}
    material=measured.get('material') if isinstance(measured.get('material'),dict) else {}
    pipe_material=pipe.get('material') if isinstance(pipe.get('material'),dict) else {}
    scopes=[measured] if measured else []
    material_required=pipe.get('phase') in ('P1+P2','P2') or result.get('stage')=='full'
    if material_required or isinstance(material.get('connections'),dict):
        scopes.append(material)
    def total(section,key):
        values=[_count(scope.get(section),key) if isinstance(scope,dict) else None for scope in scopes]
        return sum(values) if values and all(value is not None for value in values) else None
    connected,expected=(total('connections',key) for key in ('connected','expected'))
    installed,wanted=(total('manual_valves',key) for key in ('installed','expected'))
    if connected is not None and expected is not None and connected>expected:connected=expected=None
    if installed is not None and wanted is not None and installed>wanted:installed=wanted=None
    complete_pipe=isinstance(pipe.get('routes'),list) and pipe.get('phase') in ('P1','P2','P1+P2')
    cuts=pipe.get('collision_breaks',[]) if complete_pipe else None
    cut_count=len(cuts) if isinstance(cuts,list) else None
    penetrations=_count(measured,'actual_material_penetration_count')
    reasons=[]
    if not model_present:reasons.append('No readable saved model')
    if not complete_pipe:reasons.append('Incomplete piping records')
    if connected is None or expected is None:reasons.append('Connection counts are missing or incomplete')
    if penetrations is None:reasons.append('Solid penetration has not been measured')
    if cut_count is None:reasons.append('Cut-segment evidence is missing or unreadable')
    failed_generation=(result.get('timed_out') is True or result.get('substatus') in ('stage_exception','stage_timeout')
        or result.get('generation') in ('error','timeout','rejected')
        or result.get('status') in ('stage_exception','stage_timeout','invalid input')
        or result.get('layout_generation') not in (None,'success')
        or bool(pipe_material.get('program_error'))
        or any(isinstance(row,dict) and row.get('stage')=='generation_or_measurement'
               for row in (result.get('known_problems') or [])))
    if failed_generation:reasons.append('Generation failed, timed out or did not complete layout')
    state='incomplete' if reasons else 'pass'
    if not reasons:
        if connected!=expected:reasons.append('Some pipes are unconnected')
        if penetrations:reasons.append('Solid penetration was found')
        if cut_count:reasons.append('Segments were cut after penetration was found')
        if reasons:state='fail'
    role_check=measured.get('valve_roles')
    rows=role_check.get('rows') if isinstance(role_check,dict) else None
    review_pending=(sum(bool(row.get('actual_valve')) and row.get('status')=='fail' for row in rows)
                    if isinstance(rows,list) and all(isinstance(row,dict) and row.get('status') in ('pass','fail') for row in rows) else None)
    discarded = len(pipe.get('close_valve_pruning', ()))
    pending = wanted-installed-discarded if installed is not None and wanted is not None else None
    return {'status':state,'label':{'pass':'piping_passed','fail':'piping_failed','incomplete':'piping_incomplete'}[state],
        'connections':{'connected':connected,'expected':expected},
        'breakpoints':expected-connected if connected is not None and expected is not None else None,
        'penetrations':penetrations,'collision_breaks':cut_count,
        'valves':{'installed':installed,'expected':wanted},
        'pending_valves':pending,
        **({'discarded_close_valves':discarded} if 'close_valve_pruning' in pipe else {}),
        'valve_review_pending':review_pending,'reasons':reasons}
