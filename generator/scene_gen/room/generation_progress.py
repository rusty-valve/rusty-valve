"""Save generation progress and preserve incomplete results for inspection."""
from copy import deepcopy
from pathlib import Path
import json,os
from .process import json_default, write_json


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp')
    write_json(tmp,value)
    os.replace(tmp,path)


def step(a,event):
    a.current_step=event
    if getattr(a,'progress_folder',None):
        atomic_json(Path(a.progress_folder)/'progress.json',event)


def _save_state(folder, state):
    tmp=folder/'partial_state.tmp'
    # Stream the full diagnostic snapshot. A second in-memory copy of the
    # indented JSON can push a completed room over the host memory limit.
    with tmp.open('w') as stream:
        json.dump(state,stream,ensure_ascii=False,indent=2,default=json_default)
        stream.write('\n')
    os.replace(tmp,folder/'partial_state.json')


def checkpoint(a,event):
    step(a,event)
    folder=getattr(a,'progress_folder',None)
    if folder is None:return
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    installed=a.graph.get('valves',[])
    present={v['id'] for v in installed}
    graph={**a.graph,'valves':[*installed,*(v for v in getattr(a,'planned_valves',[]) if v['id'] not in present)]}
    piping={'routes':a.routes,'allowed_contacts':a.allowed_contacts,'openings':a.openings,
            'blinds':a.blinds,'reducers':a.reducers,'material':getattr(a,'material',{}),
            'end_bundle_station':a.plan.get('end_bundle_station',{}), 'phase':'partial','scope':'Physical elements placed. Measurement is still in progress.'}
    _save_state(folder,{'stop':event,'graph':graph,'scene':a.scene,'piping':piping})


def materialize(folder):
    folder=Path(folder);state=folder/'partial_state.json'
    if not state.exists():return None
    record=json.loads(state.read_text());dest=folder/'partial_model';dest.mkdir(exist_ok=True)
    from .piping_export import export_piping_model
    write_json(dest/'process.json',record['graph']);write_json(dest/'geometry.json',record['scene'])
    write_json(dest/'piping.json',record['piping']);write_json(dest/'stop.json',record['stop'])
    export_piping_model(record['graph'],record['scene'],dest/'room.xml')
    return str(dest.resolve())


def layout_checkpoint(graph,scene,folder,event):
    from .shell import add_structure
    graph=deepcopy(graph);graph['valves']=[]
    for instance in scene['equipment_instances']:
        unit=graph['units'][instance['id']]
        for uid,jacket in unit.get('integral_jackets',{}).items():
            graph['units'][uid].update(jacket)
    scene=deepcopy({k:v for k,v in scene.items() if not k.startswith('_')})
    scene['partial_generation']=True
    add_structure(scene,graph,scene['shell'])
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    _save_state(folder,{'stop':event,'graph':graph,'scene':scene,'piping':{'phase':'layout','routes':[]}})
