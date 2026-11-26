"""Choose a building size and arrange its equipment."""
from __future__ import annotations
import json

from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import traceback

from .process import build_graph, validate, write_json, ProcessInputError
from .recipe import InputRejected, CandidateRefused, validate_recipe
from .shell import resolve_shell, add_structure
from .export import export_model, render
from .checks import geometry_checks
from .platform_coverage import check_platform_coverage

DEFAULT_ATTEMPTS = 96
ACCEPTED_CANDIDATE_TARGET = 8
MAIN_KINDS = {'vessel', 'pump', 'filter', 'exchanger', 'condenser', 'fixed_bed'}


def connections(graph):
    equipment = [{key: unit[key] for key in ('id','kind','form','size_tier','level','source_vessel','owner','dimension_spec') if key in unit}
                 for unit in graph['units'].values() if unit['kind'] in MAIN_KINDS]
    transfers = [{key: deepcopy(transfer[key]) for key in
                  ('id','source','target','mode','path_units','pump','condenser') if key in transfer}
                 for transfer in graph['transfers']]
    pumps = []
    for unit in graph['units'].values():
        if unit['kind'] != 'pump':
            continue
        pumps.append({'pump':unit['id'], 'source_vessel':unit['source_vessel'],
                      'discharge_targets':[t['target'] for t in graph['transfers'] if t.get('pump')==unit['id']],
                      'recycle_target':unit['source_vessel']})
    return {'stage':'building_equipment', 'equipment':equipment, 'transfers':transfers,
            'stage_files':{'sizing':'sizing.json','structure':'shell.json','positioning':'layout.json'},
            'pump_services':pumps,
            'transfer_conditions':{'status':'deferred','detail':'Record process routes and equipment. Pump-free transfer is evaluated separately from the raw difference in port elevations.',
                'deferred_checks':deepcopy(graph.get('deferred_process_checks',[]))}, 'scope':'Process connectivity only. No physical piping, valves or robot motion.'}


def room_pass(report):
    return all(report[key]['status']=='pass' for key in ('layout','geometry','platform_coverage','mujoco'))


def check_candidate(graph, scene, model, data):
    from .building_equipment_checks import check_layout
    physical = {**graph, 'units':{key:unit for key,unit in graph['units'].items() if unit['kind'] in MAIN_KINDS},
                'edges':[], 'valves':[]}
    coverage = check_platform_coverage(scene,model) if len(scene['floors'])>1 else {
        'status':'pass','levels':[],'errors':[],'method':'No physical platform. Check the ground and maintenance clearances.'}
    return {'layout':check_layout(graph,scene,model),
            'process_scope':{'status':'deferred','stage':'layout','hydraulic_conditions':'deferred_to_process_piping_stage','scope':'Check whether a pump is used for pump-free transfer. Condensers and equipment space are evaluated separately.',
                             'deferred_checks':deepcopy(graph.get('deferred_process_checks',[]))},
            'geometry':geometry_checks(physical,scene,model,data), 'platform_coverage':coverage,
            'mujoco':{'status':'pass','nq':model.nq,'geoms':model.ngeom,
                      'scope':'Physical load and forward kinematics of the equipment layout'}}


def select_structure(sizing, shell_input, bays, reference=None):
    """Second work step: select useful levels, never resize equipment."""
    from .building_equipment_layout import _assign_levels
    from .equipment_catalog import resolve_spec
    graph=deepcopy(sizing['graph'])
    shell=resolve_shell({**shell_input,'bays':bays})
    vessel_levels = {row['id']:row['level'] for row in reference['equipment']
                     if row['kind']=='vessel'} if reference else {}
    _assign_levels(graph,shell,reference_vessel_levels=vessel_levels)
    actual=max((u.get('level',0) for u in graph['units'].values() if u['kind'] in MAIN_KINDS),default=0)
    purposes={}
    upper_purposes={}
    for unit in graph['units'].values():
        if unit['kind'] not in MAIN_KINDS:continue
        level=unit['level']
        if unit['kind'] not in ('pump','filter'):
            spec=resolve_spec(unit)
            # A drive reservation is headroom, not an operating-floor purpose.
            # Real useful decks crossing it get their own registered opening.
            top=level*shell['storey_height_m']+max(spec.get('support_to_top_m') or 0.,
                spec['height_m'] if unit['kind']=='fixed_bed' else 0.)
            actual=max(actual,max((i for i,z in enumerate(shell['floors']) if z<top-1e-8),default=0))
        purposes.setdefault(str(level),[]).append(unit['id'])
    shell['platform_limit']=shell['platforms'];shell['bays_limit']=shell_input.get('bays',2)
    shell['platforms']=actual;shell['floors']=shell['floors'][:actual+1]
    shell['roof_z_m']=(actual+1)*shell['storey_height_m'];shell['layout_only']=True
    graph['structure_assigned']=True
    graph['sizing']={key:deepcopy(value) for key,value in sizing.items() if key not in ('graph','recipe')}
    graph['source']='Building layout v2.4 with bounded placement backtracking and physical checks'
    graph['deferred_process_checks']=deepcopy(sizing['checks']['process'].get('deferred_checks',[]))
    result={'stage':'structure','status':'pass','shell':shell,'purposes':purposes,
            'upper_operation_purposes':upper_purposes,
            'checks':{'status':'pass','platform_limit_respected':actual<=shell['platform_limit'],
                      'uniform_storey_height_m':shell['storey_height_m']},
            'scope':'Try one bay before two. Keep the equipment selection fixed.'}
    if actual>shell['platform_limit']:raise CandidateRefused('Equipment use exceeds the permitted number of levels')
    return result,graph


def position_equipment(sizing, structure, graph, folder, attempts, prior_shell_trials, reference=None,
                       *, reserve_pipe_departures=False):
    """Third work step: try whole layouts and retain all failures."""
    from .building_equipment_layout import equipment
    from .operation_reservations import plan_operations
    trials=[];best=None;accepted_count=0;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for attempt in range(attempts):
        trial=folder/f'{attempt:02}';trial.mkdir(parents=True,exist_ok=True)
        attempt_started=perf_counter()
        candidate=deepcopy(graph)
        scene=None
        search_statistics=None;statistics_source=None;refused=False
        try:
            scene=equipment(candidate,structure['shell'],attempt,reference=reference,progress_folder=folder.parent.parent,
                            reserve_pipe_departures=reserve_pipe_departures)
            scene['stage']='layout';scene['routing']=[]
            scene['shell_selection']={'trials':deepcopy(prior_shell_trials),
                'chosen_bays':structure['shell']['bays'],'bays_limit':structure['shell']['bays_limit']}
            process_report=validate(candidate,True,stage='layout')
            if process_report['status']!='pass':
                raise CandidateRefused('Process elevation or connection rejected: '+str(process_report['errors']))
            candidate['deferred_process_checks']=deepcopy(process_report.get('deferred_checks',[]))
            candidate['valves']=[]
            add_structure(scene,candidate,scene['shell'])
            model,data=export_model(candidate,scene,trial/'room.xml')
            write_json(trial/'geometry.json',scene);write_json(trial/'process.json',candidate)
            scene['operation_reservations']=plan_operations(candidate,scene)
            from .generation_progress import atomic_json
            atomic_json(folder.parent.parent/'progress.json',{'stage':'layout_checks',
                'bays':structure['shell']['bays'],'attempt':attempt})
            report=check_candidate(candidate,scene,model,data)
            passed=room_pass(report)
            causes=sorted({e.get('code',e.get('kind','layout')) for e in report['layout'].get('errors',[])})
            causes += [key for key in ('geometry','platform_coverage') if report[key]['status']!='pass']
            trial_score=(int(passed),float(report['layout'].get('minimum_clear_width_m') or 0.))
            row={'attempt':attempt,'room_pass':passed,'score':trial_score,'failure_causes':causes,
                 'actual_platforms':len(scene['floors'])-1,'stage':'layout',
                 'candidate_strategy':scene['equipment_layout']}
            write_json(trial/'layout_checks.json',report)
            write_json(trial/'geometry.json',scene)
            if best is None or trial_score>best[0]:best=(trial_score,candidate,scene,attempt,report)
        except CandidateRefused as error:
            row={'attempt':attempt,'room_pass':False,'reason':str(error),'stage':'layout',
                 'exception_type':type(error).__name__}
            refused=True
            partial=getattr(error,'partial_scene',None)
            if partial is not None:
                candidate=getattr(error,'partial_graph',candidate);candidate['valves']=[]
                partial['partial_generation']=True
                add_structure(partial,candidate,partial['shell'])
                export_model(candidate,partial,trial/'room.xml')
                write_json(trial/'geometry.json',partial);write_json(trial/'process.json',candidate)
            search_statistics=getattr(error,'search_statistics',None)
            if search_statistics is not None:statistics_source='CandidateRefused.search_statistics'
        if search_statistics is None and scene is not None:
            search_statistics=scene.get('equipment_layout',{}).get('search_statistics')
            if search_statistics is not None:statistics_source='scene.equipment_layout.search_statistics'
        if search_statistics is not None:
            row['search_statistics']=deepcopy(search_statistics)
            row['search_statistics_source']=statistics_source
        row['elapsed_seconds']=perf_counter()-attempt_started
        if refused:
            write_json(trial/'refusal.json',row)
        trials.append(row);write_json(folder/'placement_trials.json',trials)
        accepted_count+=int(row['room_pass'])
        if accepted_count >= (1 if reference is not None else ACCEPTED_CANDIDATE_TARGET):
            break
    return best,trials


def generate(recipe, shell_input, seed, folder, images=True, attempts=DEFAULT_ATTEMPTS, reference=None,
             *, reserve_pipe_departures=False):
    generation_started=perf_counter()
    from .equipment_sizing import select_equipment
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    timing={'sizing_seconds':None,'structure_trials':[],'positioning_trials':[],
            'finalization_seconds':None}
    def finish(result):
        for phase in ('structure','positioning'):
            rows=timing[phase+'_trials']
            timing[phase+'_seconds_total']=sum(r['elapsed_seconds'] for r in rows) if rows else None
        timing['attempt_budget_per_shell']=attempts
        timing['total_attempts_executed']=sum(r['attempts_executed'] for r in timing['positioning_trials'])
        timing['scope']={
            'clock':'time.perf_counter; actual elapsed wall seconds',
            'sizing_and_structure':'function calls including their internal work/read IO; excludes caller record writes',
            'positioning':'position_equipment call including its record/model IO and per-attempt trial-list writes',
            'attempt_elapsed_seconds':'from before candidate deepcopy through decision and statistics capture; includes earlier model/report IO; excludes refusal/trial-list writes after measurement',
            'finalization':'selected final export/recheck/output files and optional rendering',
            'elapsed_seconds_total':'all normal generation work/IO before timing.json write; excludes timing.json write and return overhead',
            'unexecuted_phase':'null means not executed, not a measured zero',
            'search_statistics':'copied actual layout counters; missing is not zero; selected scene is one attempt, not whole-room totals'}
        timing['elapsed_seconds_total']=perf_counter()-generation_started
        write_json(folder/'timing.json',timing)
        return {**result,'attempts':result.get('attempts',timing['positioning_trials'][-1]['attempts_executed']
                if timing['positioning_trials'] else 0),'attempt_budget':attempts,
                'total_attempts_executed':timing['total_attempts_executed'],'timing':timing}
    write_json(folder/'input.json',{'recipe':deepcopy(recipe),'shell':deepcopy(shell_input),'seed':seed,'stage':'layout'})
    from .generation_progress import atomic_json
    atomic_json(folder/'progress.json',{'stage':'sizing'})
    sizing_started=perf_counter()
    sizing=select_equipment(recipe,seed)
    timing['sizing_seconds']=perf_counter()-sizing_started
    write_json(folder/'sizing.json',sizing)
    if sizing['status']!='pass':
        reason='Equipment selection rejected: '+str(sizing.get('violations',sizing.get('checks')))
        return finish({'room':folder.name,'seed':seed,'generation':'rejected','stage':'layout',
                'failed_stage':'sizing','reason':reason,'causes':{'sizing':1}})
    sizing['graph']['id']=folder.name
    sizing['graph']['seed']=seed
    shell_trials=[];best=None;last_stage='structure'
    accepted_target=1 if reference is not None else ACCEPTED_CANDIDATE_TARGET
    reference_layout=None
    if reference:
        reference=Path(reference)
        reference_layout=json.loads((reference/'layout.json').read_text())
        reference_shell_record=json.loads((reference/'shell.json').read_text())
        reference_shell=reference_shell_record['shell']
        # Preserve the earlier smaller-shell refusal as historical evidence,
        # not as a newly executed positioning trial. This run retains a shell.
        shell_trials.extend({**deepcopy(row),'inherited_from':str(reference/'shell.json'),
            'executed_this_run':False} for row in reference_shell_record.get('trials',[])
            if row['bays']<reference_shell['bays'])
        if reference_layout.get('status')!='pass':raise CandidateRefused('The reference layout must pass building and equipment checks')
        if reference_shell['bays']>shell_input.get('bays',2):raise CandidateRefused('The reference building exceeds the maximum bay count')
    for bays in ([reference_shell['bays']] if reference else range(1,shell_input.get('bays',2)+1)):
        atomic_json(folder/'progress.json',{'stage':'structure','bays':bays})
        structure_started=perf_counter()
        try:
            structure,graph=select_structure(sizing,shell_input,bays,reference=reference_layout)
        except CandidateRefused as error:
            timing['structure_trials'].append({'bays':bays,'status':'rejected',
                                               'elapsed_seconds':perf_counter()-structure_started})
            row={'bays':bays,'stage':'structure','attempts':0,'attempt_budget':attempts,
                 'all_rejected':True,'refusal_codes':[str(error)]}
            shell_trials.append(row)
            write_json(folder/'shell.json',{'stage':'structure','status':'rejected','reason':str(error),'trials':shell_trials})
            continue
        timing['structure_trials'].append({'bays':bays,'status':'pass',
                                           'elapsed_seconds':perf_counter()-structure_started})
        write_json(folder/'shell.json',structure)
        atomic_json(folder/'progress.json',{'stage':'layout','bays':bays})
        last_stage='layout'
        positioning_started=perf_counter()
        selected,trials=position_equipment(sizing,structure,graph,folder/'attempts'/f'bays_{bays}',attempts,shell_trials,
                                           reference=reference_layout, reserve_pipe_departures=reserve_pipe_departures)
        positioning_elapsed=perf_counter()-positioning_started
        accepted=selected is not None and bool(selected[0][0])
        accepted_count=sum(int(row['room_pass']) for row in trials)
        early_stop=accepted_count>=accepted_target and len(trials)<attempts
        stop_reason=('accepted_candidate_target_reached' if early_stop else
                     'accepted_candidate_target_reached_at_budget' if accepted_count>=accepted_target else
                     'attempt_budget_exhausted')
        timing['positioning_trials'].append({'bays':bays,'status':'pass' if accepted else 'rejected',
            'elapsed_seconds':positioning_elapsed,'attempts_executed':len(trials),'attempt_budget':attempts,
            'accepted_count':accepted_count,'accepted_candidate_target':accepted_target,
            'early_stop':early_stop,'early_stop_reason':stop_reason if early_stop else None,'stop_reason':stop_reason,
            'attempt_elapsed_seconds_sum':sum(row['elapsed_seconds'] for row in trials)})
        causes=Counter(c for row in trials if not row.get('room_pass')
                       for c in (row.get('failure_causes') or [row.get('reason','Layout rejected')]))
        shell_trials.append({'bays':bays,'stage':'layout','attempts':len(trials),'attempt_budget':attempts,
                             'accepted_count':accepted_count,'early_stop':early_stop,'stop_reason':stop_reason,
                             'all_rejected':not accepted,
                             'refusal_codes':list(causes),'refusal_counts':dict(causes)})
        write_json(folder/'shell_trials.json',shell_trials)
        if accepted:
            best=selected;break
    if best is None:
        reason=('No compliant placement for new equipment with the reference shell and existing positions fixed. See candidate records.' if reference else
                'Equipment selection held fixed. Both permitted bay counts were tried. See candidate records and stage reports.')
        write_json(folder/'layout.json',{'stage':'layout','status':'rejected','shell_trials':shell_trials})
        return finish({'room':folder.name,'seed':seed,'generation':'rejected','stage':'layout',
                'failed_stage':last_stage,'reason':reason,'equipment_reference':str(reference) if reference else None,'causes':dict(Counter(
                    cause for item in shell_trials for cause in item['refusal_codes'])),'shell_trials':shell_trials})
    finalization_started=perf_counter()
    trial_score,graph,scene,selected,report=best
    scene['shell_selection']['trials']=shell_trials
    model,data=export_model(graph,scene,folder/'room.xml');report=check_candidate(graph,scene,model,data)
    if not room_pass(report):raise RuntimeError('The selected room failed final verification')
    report['placement']={'selected_attempt':selected,'attempts_per_shell':len(trials),
                         'attempts_executed':len(trials),'attempt_budget':attempts,'score':trial_score}
    write_json(folder/'layout_checks.json',report);write_json(folder/'geometry.json',scene)
    write_json(folder/'process.json',graph);write_json(folder/'connections.json',connections(graph))
    final_structure={'stage':'structure','status':'pass','shell':scene['shell'],
                     'checks':report['layout'].get('shell_selection',{}),'trials':shell_trials,
                     'purposes':{str(i):[r['id'] for r in scene['equipment_instances'] if i in r['occupied_levels']]
                                 for i in range(len(scene['floors']))}}
    write_json(folder/'shell.json',final_structure)
    write_json(folder/'layout.json',{'stage':'layout','status':'pass','equipment':[
        {**row,'center':graph['units'][row['id']]['center'],
         'rotation_deg':graph['units'][row['id']].get('rotation_deg',0),
         'ports':deepcopy(graph['units'][row['id']]['ports'])}
        for row in scene['equipment_instances']]+[
            {**deepcopy(unit),'rect':[unit['box'][i] for i in (0,1,3,4)],
             'component_ids':[p['nozzle_component_id'] for p in unit['ports'].values()]}
            for unit in graph['units'].values() if unit['kind']=='jacket'], 'operation_reservations':scene['operation_reservations'],
        'checks':report['layout'],'scope':'Building and equipment layout reserves clearances for piping and valves'})
    if images:render(graph,scene,model,data,folder/'overview.png')
    timing['finalization_seconds']=perf_counter()-finalization_started
    return finish({'room':folder.name,'seed':seed,'generation':'success','stage':'layout','attempts':len(trials),
            'selected_attempt':selected,'actual_platforms':len(scene['floors'])-1,
            'equipment':sum(u['kind'] in MAIN_KINDS for u in graph['units'].values()),
            'minimum_clear_width_m':report['layout'].get('minimum_clear_width_m'),
            'maintenance_violations':report['layout'].get('maintenance_violations',[]),
            'shell':scene['shell'],'scope':'Building and equipment layout passed. Physical pipes and valves are checked later.'})
