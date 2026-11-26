"""Summarize which room checks passed and which steps remain incomplete."""
from .rule_identity import classify,identity


def measured_outcome(measured,piping):
    failures=measured.get('method_failures',[])+piping.get('material',{}).get('failures',[])
    parts=[measured]+([measured['material']] if 'material' in measured else [])
    missing=any(p.get('connections',{}).get('connected',0)<p.get('connections',{}).get('expected',0) or
                p.get('manual_valves',{}).get('installed',0)<p.get('manual_valves',{}).get('expected',0) for p in parts)
    errors=[e for p in parts for e in p.get('errors',[])]
    if not errors and any(p.get('status')!='pass' for p in parts):errors=[{'code':'HARD_CHECK_FAILED'}]
    stops=[{'stage':f.get('stage','pipe_or_valve'),'edge':f.get('edge',f.get('requirement')),
            'valve':f.get('valve'),'endpoint':f.get('endpoint'),'reason':f.get('reason',f.get('code'))} for f in failures]
    if missing and not errors:errors=[{'code':'MEASURED_REQUIRED_ITEMS_MISSING'}]
    error=piping.get('material',{}).get('program_error')
    return {'status':classify(completed=not failures,hard_errors=errors,exception=error),
            'substatus':'stage_exception' if error else None,
            'stops':stops,'hard_errors':errors,
            'rule_identities':{r:identity(r) for r in ('connections','required_valves','penetration','height','equipment_top_access')}}


def challenges(measured,graph):
    rows=[]
    for v in measured.get('operation_report',{}).get('rows',[]):
        if not v.get('eligible'):
            rows.append({'valve':v['id'],'identity':v.get('rule_identity','preference'),
                'reason':('The existing platform has no direct operating position. Access equipment has not been modeled. ' if v.get('rule_identity')=='record' and not (v.get('stance_free') and v.get('reachable_from_passage')) else '')+v.get('reason',''),
                'height_m':v.get('operator_height_m'),'source':'actual_model_operations'})
    seen={v['id'] for v in measured.get('operation_report',{}).get('rows',[])}
    for v in graph.get('valves',[]):
        p=v.get('operation_preference',{})
        if p.get('challenge') and v['id'] not in seen:
            rows.append({'valve':v['id'],'identity':p['identity'],'reason':p['note'],'source':'candidate_height'})
    return rows
