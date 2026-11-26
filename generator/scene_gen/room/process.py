"""Describe the equipment, ports and connections required by a process recipe."""
from __future__ import annotations
from collections import defaultdict, deque
from copy import deepcopy
import json
from pathlib import Path
import random

from .storage_vessel import normalize_vessel, storage_service_checks

RULES = json.loads(Path(__file__).with_name('rules.json').read_text())
from .material_support import canonical_transfer_mode
from .pipe_spec import load as _load_pipe_spec
_CURRENT_PIPE_SPEC=_load_pipe_spec()
RULES['colors']=deepcopy(_CURRENT_PIPE_SPEC['colors'])
RULES['unit_rules']['filter']['placement']='before_pump'
EXTRA_UNIT_RULES = {
    'automatic_steam_trap': {'ports':['in','out'],'manual':'none','source':'current piping dimensions, figure 18'},
    'safety_valve': {'ports':['in','out'],'manual':'none','source':'current piping dimensions for emergency relief'},
    'fixed_bed': {'ports': ['in', 'out'], 'manual': 'every_connected_port',
                  'placement': 'after_pump', 'flow': 'top_in_bottom_out',
                  'source': 'Building and equipment rule BL-4. Fixed bed downstream of pump, with top feed and bottom discharge.'},
}
ROLES = {'feed':'feed root', 'nitrogen':'nitrogen root', 'offgas':'off-gas root',
         'bottom':'discharge root', 'recycle':'recirculation root', 'vapor':'steam root',
         'condensate':'condensate root', 'in':'inlet', 'out':'outlet',
         'utility_in':'utility inlet', 'utility_out':'utility outlet',
         'drain':'drain', 'vent':'vent', 'gauge':'pressure gauge root'}

def json_default(value):
    return value.item() if hasattr(value,'item') else list(value)

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default)+'\n')

class ProcessInputError(ValueError):
    """Input identities conflict with the generated process graph."""


def unit_rule(kind):
    """Add a new kind explicitly, without changing any existing rule/threshold."""
    if kind in RULES['unit_rules']:
        return RULES['unit_rules'][kind]
    if kind in EXTRA_UNIT_RULES:
        return EXTRA_UNIT_RULES[kind]
    raise ProcessInputError(f'Unsupported process equipment type: {kind}')



def equipment_traits(spec):
    form = spec.get('form', spec.get('orientation', 'vertical'))
    if form not in ('vertical', 'horizontal'):
        raise ProcessInputError('Equipment form must be vertical or horizontal')
    result = {'form': form, 'orientation': form}
    if 'size_tier' in spec:
        result['size_tier'] = spec['size_tier']
    return result


class Builder:
    def __init__(self, recipe):
        self.g={'id':recipe['id'], 'seed':recipe['seed'], 'recipe':deepcopy(recipe),
                'units':{}, 'edges':[], 'valves':[], 'transfers':[], 'source':'Generated process definition'}
        self.serial=0

    def unit(self, name, kind, **kw):
        if name in self.g['units']:
            raise ProcessInputError(f'Duplicate equipment or interface identifier: {name}. Recipe ID conflicts with a generated equipment ID.')
        ports={p:{'dn':kw.get('dn',50)} for p in unit_rule(kind)['ports']}
        self.g['units'][name]={'id':name,'kind':kind,'ports':ports,**kw}
        return name

    def port(self,u,p,dn=50):
        self.g['units'][u]['ports'].setdefault(p,{'dn':dn})
        return u+'.'+p

    def edge(self,a,b,dn=50,role='material', manual=None):
        self.serial+=1
        e={'id':f'L{self.serial:03}', 'a':a, 'b':b,'dn':dn,'role':role}
        self.g['edges'].append(e)
        for endpoint in (a,b):
            uid,port=endpoint.split('.')
            unit=self.g['units'][uid]
            unit['ports'].setdefault(port,{'dn':dn})
            unit['ports'][port]['dn']=dn
            if (unit_rule(unit['kind'])['manual']=='every_connected_port'
                    and port not in unit.get('no_manual_ports',[])
                    and not (unit['kind']=='header' and unit.get('service')=='offgas')):
                role='recirculation root' if port.startswith('recycle_') else ROLES.get(port,'header branch '+port)
                if unit['kind']=='header':
                    peer=b if endpoint==a else a;peer_unit,peer_port=peer.split('.')
                    role=peer_unit+' '+ROLES.get(peer_port,peer_port).removesuffix('root')+' header branch'
                self.valve(e,endpoint, f'{uid} {role} valve')
        if manual:
            self.valve(e,'branch',manual)
        return e['id']

    def valve(self,e,at,name):
        if at=='branch':role={'recycle':'pump-discharge recirculation control','condensate':'condensate return control',
            'seal_drain':'low-point liquid-trap drain'}.get(e['role'],'filter bypass switching')
        else:
            uid,p=at.split('.');kind=self.g['units'][uid]['kind']
            category={'vessel':'reactor','header':'utility main','pump':'pump','filter':'filter','exchanger':'heat exchanger','condenser':'condenser','jacket':'jacket','fixed_bed':'fixed bed'}[kind]
            role=category+' '+('branch isolation' if kind=='header' else ('recirculation root' if p.startswith('recycle_') else ROLES.get(p,p)))
        self.g['valves'].append({'id':f'V{len(self.g["valves"])+1:03}', 'edge':e['id'],
            'at':at,'name':name,'role':role,'dn':e['dn'],'manual':True,
            'bottom_substitute':at.endswith('.bottom'),
            'rule_source':'Claude§5' if at.endswith(('.drain','.vent','.gauge')) else 'process rule section 1'})

    def header(self,kind,endpoint,dn=20,out=False):
        uid='U_'+kind
        if uid not in self.g['units']: self.unit(uid,'header',service=kind)
        h=self.port(uid,'b'+str(len(self.g['units'][uid]['ports'])+1),dn)
        self.edge(endpoint,h,dn,kind) if out else self.edge(h,endpoint,dn,kind)

    def boundary(self,label,endpoint,dn=50,out=True):
        uid='X_'+label
        self.unit(uid,'boundary',dn=dn)
        self.edge(endpoint,uid+'.port',dn,'boundary') if out else self.edge(uid+'.port',endpoint,dn,'feed')

    def junction(self,name):
        return self.unit(name,'junction')

    def jp(self,u):
        return self.port(u,'p'+str(len(self.g['units'][u]['ports'])+1))

def build_graph(recipe):
    from .material_support import ensure_supported
    recipe = deepcopy(recipe)
    for vessel in recipe['vessels']:
        normalize_vessel(vessel)
    ensure_supported(recipe, ProcessInputError)
    for transfer in recipe['transfers']:
        transfer['mode'] = canonical_transfer_mode(transfer['mode'])
    return _build_current_graph(recipe)


def _return_manifold(graph, uid):
    unit = graph['units'][uid]
    edges, valves = graph['edges'], graph['valves']
    trunk = next(e for e in edges if e['b'] == uid+'.feed')
    junction = graph['units'][trunk['a'].split('.')[0]]
    branches = [e for e in edges if e['b'].split('.')[0] == junction['id']]
    returns = [e for e in edges if e['b'].startswith(uid+'.recycle_')]
    if len(branches)+len(returns) < 2:
        return
    unit['ports']['liquid_return'] = unit['ports'].pop('feed')
    root = next(v for v in valves if v['edge'] == trunk['id'] and v['at'] == trunk['b'])
    trunk['b'] = uid+'.liquid_return'
    root.update(at=trunk['b'], name=uid+' shared-port root valve', role='shared-port root valve', common_port_root=True)
    for edge in returns:
        old = edge['b']
        port = 'p'+str(len(junction['ports'])+1)
        junction['ports'][port] = unit['ports'].pop(old.split('.')[1])
        edge['b'] = junction['id']+'.'+port
        branch_root = next(v for v in valves if v['edge'] == edge['id'] and v['at'] == old)
        branch_root.update(at='branch', original_at=old, shared_return_root=True)
    for edge in branches:
        if edge.get('branch_purpose') == 'direct_return':
            valves.append({'id':f'V{max(int(v["id"][1:]) for v in valves)+1:03}',
                           'edge':edge['id'], 'at':'branch', 'original_at':uid+'.recycle_direct',
                           'name':uid+' direct return branch root valve', 'role':'reactor recirculation branch root valve',
                           'shared_return_root':True, 'dn':edge['dn'], 'manual':True})
    unit['return_manifold'] = {'trunk':trunk['id'], 'junction':junction['id'],
                               'branches':[e['id'] for e in branches+returns]}


def _condenser_offgas(graph, uid):
    units=graph['units']
    condensers=[name for name,unit in units.items()
                if unit['kind']=='condenser' and unit.get('owner')==uid]
    replacement=next((e for e in graph['edges']
                      if e['role']=='offgas' and any(e['a']==f'{name}.offgas' for name in condensers)),None)
    if replacement is None:
        return
    removed=[e for e in graph['edges'] if e['role']=='offgas' and e['a']==uid+'.offgas']
    for edge in removed:
        graph['edges'].remove(edge)
        target,name=edge['b'].split('.')
        units[target]['ports'].pop(name,None)
        graph['valves'][:]=[v for v in graph['valves'] if v['edge']!=edge['id']]
        graph.setdefault('process_connection_changes',[]).append({
            'retired_edge':edge['id'],'replacement_edge':replacement['id'],
            'reason':'Ordinary off-gas leaves a condenser-equipped reactor through its condenser noncondensable outlet'})
    if removed:
        units[uid]['ports'].pop('offgas',None)
        units[uid]['ordinary_offgas_via_condenser']=replacement['a']

def _build_current_graph(recipe):
    """Coordinate-free current process semantics; dimensions only from pipe_spec."""
    from .pipe_spec import load, color
    spec=load();services=spec['services'];b=Builder(recipe);g=b.g;units=g['units']
    g.update(process_schema='current_pipe_spec_v1',pipe_spec_source='scene_gen/room/pipe_spec.json',
             inline_components=[],reflux_circuits=[])
    material={v['id']:spec['material_by_tier'][v['size_tier']] for v in recipe['vessels']}
    def automatic(uid,kind,owner,endpoint,dn):
        b.unit(uid,kind,owner=owner,source_endpoint=endpoint,dn=dn)
        record={'id':uid,'kind':kind,'owner':owner,'source_endpoint':endpoint,
                'dn':dn,'automatic':True,'count_as_hand_valve':False,'ports':{'in':{'dn':dn},'out':{'dn':dn}}}
        g['inline_components'].append(record)
        return uid
    def steam_service(uid):
        dn=services['steam']['device_dn'];b.header('steam',uid+'.utility_in',dn)
        st=automatic('ST_'+uid,'automatic_steam_trap',uid,uid+'.utility_out',dn)
        b.edge(uid+'.utility_out',st+'.in',dn,'steam_condensate')
        b.boundary(uid+'_steam_drain',st+'.out',dn)
        edge=g['edges'][-1];edge.update(role='steam_condensate',boundary_kind='ground_drain_point')
    def cooling(uid):
        dn=services['cold_source']['device_dn']
        b.header('water',uid+'.utility_in',dn);b.header('water_return',uid+'.utility_out',dn,out=True)
    for row in recipe['vessels']:
        uid=row['id'];dn=material[uid]
        jacket=bool(row.get('jacket') or 'jacket_medium' in row)
        b.unit(uid,'vessel',**equipment_traits(row),dn=dn,reactor=row.get('reactor',True),
            pressurizable=row.get('pressurizable',True),jacket=jacket,no_manual_ports=['safety'])
        ndn=services['nitrogen']['device_candidates'][0]
        b.header('nitrogen',uid+'.nitrogen',ndn);b.header('offgas',uid+'.offgas',ndn,out=True)
        b.junction(uid+'_bottom');b.edge(uid+'.bottom',b.jp(uid+'_bottom'),dn)
        b.junction(uid+'_feed');b.edge(b.jp(uid+'_feed'),uid+'.feed',dn)
        emergency=services['offgas']['emergency_dn'];b.port(uid,'safety',emergency)
        sv=automatic('SV_'+uid,'safety_valve',uid,uid+'.safety',emergency)
        b.edge(uid+'.safety',sv+'.in',emergency,'emergency_discharge')
        b.header('offgas',sv+'.out',emergency,out=True)
        g['edges'][-1].update(role='emergency_discharge',service_identity='emergency_discharge')
        if jacket:
            medium=row['jacket_medium'];jid=b.unit('J_'+uid,'jacket',owner=uid,medium=medium)
            steam_service(jid) if medium=='steam' else cooling(jid)
    for index,source in enumerate(recipe['vessels'],1):
        if not source.get('reflux_condenser'):continue
        uid=source['id'];cfg=source['reflux_condenser']
        vapor=spec['process_vapor_candidates'][0];cond=spec['condenser_condensate_candidates'][-1]
        c=b.unit('CND_'+uid,'condenser',owner=uid,service_owner=uid,service='reflux',
            return_to_owner=True,**equipment_traits(cfg),dn=vapor)
        b.port(uid,'vapor',vapor);b.edge(uid+'.vapor',c+'.vapor',vapor,'vapor')
        if cfg['form']=='vertical':
            # One physical shell mouth and one shared pipe, with opposite flows.
            units[c]['ports'].pop('condensate')
            units[c].update(shared_vapor_liquid=True,requires_liquid_seal=False)
            g['edges'][-1].update(shared_vapor_liquid=True,
                flow_semantics=['vapor_to_condenser','condensate_to_owner'])
            circuit={'owner':uid,'condenser':c,'form':'vertical',
                'shared_edge':g['edges'][-1]['id'],'vapor_endpoint':uid+'.vapor',
                'return_endpoint':uid+'.vapor','return_to_owner':True}
        else:
            seal=b.junction(c+'_seal');units[seal]['template_role']='liquid_seal'
            units[seal]['owner']=c;b.edge(c+'.condensate',b.jp(seal),cond,'condensate')
            b.port(uid,'condensate_return',cond)
            b.edge(b.jp(seal),uid+'.condensate_return',cond,'condensate',manual=c+' condensate return control valve')
            b.boundary(c+'_seal_drain',b.jp(seal),cond)
            g['edges'][-1].update(role='seal_drain',boundary_kind='blind_local_drain')
            b.valve(g['edges'][-1],'branch',c+' liquid-trap low-point drain valve')
            units[c].update(liquid_seal_junction=seal,requires_liquid_seal=True)
            circuit={'owner':uid,'condenser':c,'form':'horizontal',
                'vapor_endpoint':uid+'.vapor','return_endpoint':uid+'.condensate_return',
                'seal_junction':seal,'return_to_owner':True}
        cooling(c);b.port(c,'offgas',services['offgas']['condenser_dn'])
        b.header('offgas',c+'.offgas',services['offgas']['condenser_dn'],out=True)
        g['reflux_circuits'].append(circuit)
    for number,raw in enumerate(recipe['transfers'],1):
        t=deepcopy(raw);t['id']='T'+str(number);a=t['source'];target=t['target'];dn=material[a]
        last=a+'_bottom';path=[a];start_index=len(g['edges'])
        if t['mode']=='pump':
            t['protection_filter']=t.get('protection_filter') or {'form':'horizontal','size_tier':units[a].get('size_tier','small')}
            f=b.unit('F'+str(number),'filter',**equipment_traits(t['protection_filter']),
                dn=dn,use='pump_protection',service_owner=a)
            after=b.junction(f+'_out');b.edge(b.jp(last),f+'.in',dn)
            b.edge(f+'.out',b.jp(after),dn);b.edge(b.jp(last),b.jp(after),dn,'bypass',manual=f+' bypass selector valve')
            t['protection_filter']=f;path.append(f);last=after
            p=b.unit('P'+str(number),'pump',source_vessel=a,service_owner=a,dn=dn)
            b.edge(b.jp(last),p+'.in',dn);split=b.junction(p+'_out');b.edge(p+'.out',b.jp(split),dn)
            t['pump']=p;path.append(p);last=split;recycle_start=len(g['edges']);recycle_path=[p]
            previous=b.jp(split)
            for j,aux in enumerate(t.get('recycle_via',[]),1):
                h=b.unit(f'RHX{number}_{j}','exchanger',**equipment_traits(aux),dn=dn,
                    service='steam_heater',service_owner=a,owner=a)
                b.edge(previous,h+'.in',dn,'recycle')
                previous=h+'.out';recycle_path.append(h);steam_service(h)
            rp='recycle_'+str(number);b.port(a,rp,dn)
            b.edge(previous,a+'.'+rp,dn,'recycle',manual=p+' heated reactor-return control valve' if t.get('recycle_via') else p+' recirculation control valve')
            g['edges'][-1]['branch_purpose']='heated_return' if t.get('recycle_via') else 'pump_return'
            t['recycle_path_units']=[*recycle_path,a]
            t['recycle_edge_ids']=[e['id'] for e in g['edges'][recycle_start:] if e['role']=='recycle']
        if target.startswith('EXPORT'):b.boundary(target,b.jp(last),dn)
        elif target==a and t['mode']=='pump':
            b.edge(b.jp(last),b.jp(target+'_feed'),dn,'recycle',manual=p+' direct reactor-return control valve')
            g['edges'][-1]['branch_purpose']='direct_return'
        else:b.edge(b.jp(last),b.jp(target+'_feed'),dn)
        t.update(path_units=[*path,target],edge_ids=[e['id'] for e in g['edges'][start_index:]])
        g['transfers'].append(t)
    incoming={t['target'] for t in recipe['transfers']};outgoing={t['source'] for t in recipe['transfers']}
    for row in recipe['vessels']:
        uid=row['id'];dn=material[uid]
        if uid not in incoming:b.boundary(uid+'_feed',b.jp(uid+'_feed'),dn,out=False)
        if uid not in outgoing:b.boundary(uid+'_product',b.jp(uid+'_bottom'),dn)
    for uid,unit in list(units.items()):
        if unit['kind']=='vessel':
            _condenser_offgas(g,uid)
            _return_manifold(g,uid)
    for valve in g['valves']:
        edge=next(e for e in g['edges'] if e['id']==valve['edge'])
        source=units[edge['a'].split('.')[0]]
        if (valve['at']=='branch' and not valve.get('shared_return_root')
                and edge['role']=='condensate' and source.get('template_role')=='liquid_seal'):
            incoming=next(e for e in g['edges'] if e['b'].split('.')[0]==source['id'] and e['role']=='condensate')
            valve['edge']=incoming['id']
    from .valve_roles import annotate_graph
    annotate_graph(g)
    for edge in g['edges']:
        role=edge['role'];edge['medium_identity']=('steam_condensate' if role=='steam_condensate' else
            'unknown_material' if role in ('material','recycle','bypass','vapor','condensate','seal_drain','feed','boundary') else role)
        edge['dn_source']='pipe_spec.json'
        color_role={'water':'cold_source','water_return':'cold_return','unknown_material':'unknown',
                    'emergency_discharge':'offgas'}.get(edge['medium_identity'],edge['medium_identity'])
        edge['color']=color(color_role)
    for valve in g['valves']:valve['operator']='lever'
    return g

def adjacency(g, exclude=(), material_only=False):
    adj=defaultdict(list)
    for e in g['edges']:
        a=e['a'].split('.')[0]; b=e['b'].split('.')[0]
        if a in exclude or b in exclude: continue
        if material_only and e['role'] not in ('material','recycle','bypass','boundary','feed','vapor','condensate'): continue
        adj[a].append((b,e['id'])); adj[b].append((a,e['id']))
    return adj

def find_path(g,start,end,exclude=(), material_only=True):
    adj=adjacency(g,exclude,material_only); q=deque([(start,[])]) ; seen={start}
    while q:
        u,path=q.popleft()
        if u==end: return path
        for v,e in adj[u]:
            if v not in seen: seen.add(v); q.append((v,path+[e]))
    return None

def validate(g, geometry=False, stage='full'):
    if stage not in ('full','layout'):raise ValueError('validate stage must be full or layout')
    deferred=[]
    errors=[]; units=g['units']; edges=g['edges']; valves=g['valves']; incident=defaultdict(list)
    def err(code,obj,detail): errors.append({'code':code,'object':obj,'detail':detail})
    for row in storage_service_checks(g):
        if row['status'] == 'fail':
            err('HORIZONTAL_STORAGE_SERVICES', row['equipment'], row)
    for e in edges:
        for ep in (e['a'],e['b']):
            u,p=ep.split('.'); incident[ep].append(e)
            if u not in units or p not in units[u]['ports']: err('BAD_ENDPOINT',e['id'],ep)
            elif units[u]['ports'][p]['dn']!=e['dn']: err('PIPE_DN',e['id'],ep)
    if any(e['code']=='BAD_ENDPOINT' for e in errors):return {'status':'fail','errors':errors,'units':len(units),'edges':len(edges),'manual_valves':len(valves)}
    for u in units.values():
        rule=unit_rule(u['kind'])
        required_ports=(['vapor','utility_in','utility_out','offgas'] if u.get('shared_vapor_liquid') else rule['ports'])
        if u['kind']=='vessel':
            required_ports=[('liquid_return' if p=='feed' and u.get('return_manifold') else p)
                            for p in required_ports if p!='offgas' or not u.get('ordinary_offgas_via_condenser')]
        for p in required_ports:
            if p not in u['ports'] or not incident[u['id']+'.'+p]: err('REQUIRED_PORT',u['id'],p)
        for p in u['ports']:
            ep=u['id']+'.'+p
            if not incident[ep]: err('UNCONNECTED_PORT',u['id'],p)
            if len(incident[ep])>1: err('PORT_REUSED',u['id'],p)
            if (rule['manual']=='every_connected_port' and p not in u.get('no_manual_ports',[])
                    and not (u['kind']=='header' and u.get('service')=='offgas')):
                for e in incident[ep]:
                    if not any(v['manual'] and v['edge']==e['id'] and v['at']==ep for v in valves): err('ROOT_MANUAL',ep,'Missing manual root valve')
        if u['kind']=='pump':
            out=incident[u['id']+'.out']; a=u['source_vessel']
            forbidden=[x for x in units if units[x]['kind'] in ('vessel','pump','header','boundary') and x!=a]
            start=other_unit(out[0],u['id']) if out else ''
            path=find_path(g,start,a,exclude=forbidden)
            if path is None or not any(e['id'] in path and e['role']=='recycle' and any(v['edge']==e['id'] and v['at']=='branch' and v['manual'] for v in valves) for e in edges):
                err('PUMP_RETURN',u['id'],'Pump outlet needs an independent manual-valve recirculation line back to the same reactor')
            inp=incident[u['id']+'.in']; start=other_unit(inp[0],u['id']) if inp else ''
            if find_path(g,start,a,exclude=forbidden) is None: err('PUMP_SOURCE',u['id'],'The inlet does not connect to the bottom of its reactor')
        if u['kind']=='filter':
            ei=incident[u['id']+'.in']; eo=incident[u['id']+'.out']
            ends=[other_unit(es[0],u['id']) if es else '' for es in (ei,eo)]
            forbidden=[x for x in units if units[x]['kind'] in ('vessel','pump','header','boundary')]+[u['id']]
            path=find_path(g,*ends,exclude=forbidden)
            if path is None or not any(e['id'] in path and e['role']=='bypass' and any(v['edge']==e['id'] and v['manual'] for v in valves) for e in edges): err('FILTER_BYPASS',u['id'],'Missing switchable bypass')
        if u['kind'] in ('filter','fixed_bed'):
            # Trace actual upstream edges, stopping at vessels, rather than accepting a downstream pump.
            upstream=defaultdict(list)
            for e in edges:
                if e['role'] in ('material','bypass','recycle'):
                    if u['kind']=='filter':upstream[e['a'].split('.')[0]].append(e['b'].split('.')[0])
                    else:upstream[e['b'].split('.')[0]].append(e['a'].split('.')[0])
            todo=[u['id']];seen=set();has_pump=False
            while todo:
                node=todo.pop()
                if node in seen:continue
                seen.add(node)
                if units[node]['kind']=='pump':has_pump=True;break
                if units[node]['kind']=='vessel':continue
                todo.extend(upstream[node])
            if not has_pump:
                err('PROTECTION_FILTER_BEFORE_PUMP' if u['kind']=='filter' else 'FIXED_BED_AFTER_PUMP',u['id'],
                    'Protection filter and bypass return belong upstream of the pump' if u['kind']=='filter' else 'Fixed bed must be downstream of the pump')
        if u['kind']=='fixed_bed':
            if not any(t['mode']=='pump' and u['id'] in t.get('path_units',[]) for t in g['transfers']):
                err('FIXED_BED_IN_TRANSFER',u['id'],'Fixed bed must be in series along the pumped transfer path')
            if any(e['b']!=u['id']+'.in' for e in incident[u['id']+'.in']) or any(e['a']!=u['id']+'.out' for e in incident[u['id']+'.out']):
                err('FIXED_BED_FLOW_DIRECTION',u['id'],'The process path must use in as the inlet and out as the outlet')
            if geometry:
                inlet=u['ports']['in'].get('pos');outlet=u['ports']['out'].get('pos')
                if inlet is None or outlet is None or inlet[2]<=outlet[2]:
                    err('FIXED_BED_TOP_IN_BOTTOM_OUT',u['id'],'The physical inlet must be above the outlet')
        if u['kind'] in ('exchanger','condenser','jacket'):
            for p in ('utility_in','utility_out'):
                es=incident[u['id']+'.'+p]
                if (p=='utility_out' and es and units.get(other_unit(es[0],u['id']),{}).get('kind')=='automatic_steam_trap'):
                    continue
                if not es or units.get(other_unit(es[0],u['id']),{}).get('kind')!='header': err('UTILITY_CONNECTION',u['id'],p)
                elif units[other_unit(es[0],u['id'])].get('service') not in (('water','steam') if p=='utility_in' else ('water_return','steam_return')):err('UTILITY_SERVICE',u['id'],p)
        if u['kind']=='vessel':
            if g.get('process_schema')=='current_pipe_spec_v1':
                safety=incident[u['id']+'.safety']
                if not safety or any(v['at']==u['id']+'.safety' for v in valves):
                    err('EMERGENCY_NO_MANUAL_ISOLATION',u['id'],'Emergency relief piping must have no ordinary manual isolation')
            if 'vent' in u['ports'] and 'offgas' in u['ports']:
                err('VESSEL_DUPLICATE_ORDINARY_VENT',u['id'],
                    'Ordinary off-gas already vents the equipment. Do not add a duplicate vent or piping.')
            for p,service in (('nitrogen','nitrogen'),('offgas','offgas')):
                endpoint=u.get('ordinary_offgas_via_condenser') if p=='offgas' else None
                es=incident[endpoint or u['id']+'.'+p]
                if not es or units.get(other_unit(es[0],(endpoint or u['id']+'.'+p).split('.')[0]),{}).get('service')!=service:err('VESSEL_SERVICE',u['id'],p)
        if geometry and u['kind']=='pump' and u.get('level')!=RULES['floor_rules']['pump_level']:err('PUMP_LEVEL',u['id'],'Pumps must sit at ground level')
        if u.get('shared_vapor_liquid'):
            shared=[e for e in edges if e.get('shared_vapor_liquid') and e['b']==u['id']+'.vapor' and e['a']==u['owner']+'.vapor']
            if len(shared)!=1 or 'condensate' in u['ports'] or u.get('requires_liquid_seal'):
                err('REFLUX_SINGLE_SHARED_PIPE',u['id'],'Vertical condenser needs one shared vapor/liquid pipe')
        if u['kind']=='condenser' and u.get('service')=='reflux':
            target=u['owner']+('.vapor' if u.get('shared_vapor_liquid') else '.condensate_return')
            if target not in incident or find_path(g,u['id'],u['owner'],exclude=[k for k,v in units.items()
                if v['kind']=='vessel' and k!=u['owner']]) is None:
                err('REFLUX_OWN_VESSEL',u['id'],'Reflux condenser must return to its source reactor')
    for v in valves:
        e=next((e for e in edges if e['id']==v['edge']),None)
        if e is None or e['dn']!=v['dn']: err('VALVE_DN',v['id'],'Valve must belong to a real pipe of the same DN')
        if not v['manual'] and not any(x['manual'] and x['edge']==v['edge'] for x in valves): err('AUTO_COMPANION',v['id'],'Automatic valve lacks a manual companion')
    for t in g['transfers']:
        if canonical_transfer_mode(t['mode']) not in RULES['transfer_rules']: err('TRANSFER_MODE',t['id'],t['mode']); continue
        a=t['source']; z=t['target']; forbidden=[x for x in units if units[x]['kind']=='header' or (units[x]['kind']=='vessel' and x not in (a,z))]
        endpoint='X_'+z if z.startswith('EXPORT') else z
        if find_path(g,a,endpoint,exclude=forbidden) is None: err('TRANSFER_PATH',t['id'],'Material path is disconnected')
        if t['mode']=='pump' and (t.get('pump') not in units or units[t['pump']]['kind']!='pump'):
            err('TRANSFER_PUMP_REQUIRED',t['id'],'Pumped transfer lacks a pump')
        if canonical_transfer_mode(t['mode']) == 'no_pump':
            blocked=forbidden+[x for x in units if units[x]['kind']=='pump']
            if any(units.get(k, {}).get('kind') == 'pump' for k in t['path_units']) or find_path(g,a,endpoint,exclude=blocked) is None:err('PASSIVE_TRANSFER_PATH',t['id'],'This transfer mode cannot rely on a pump')
        if geometry and t['mode']=='distillation':
            if units[t['condenser']]['box'][2] <= units[a]['box'][5]: err('CONDENSER_HEIGHT',t['id'],'Condenser is not above the source reactor')
    return {'status':'fail' if errors else 'pass','errors':errors,'units':len(units),'edges':len(edges),'manual_valves':sum(v['manual'] for v in valves),
            'stage':stage,'deferred_checks':deferred,
            'hydraulic_conditions':'deferred_to_process_piping_stage' if stage=='layout' else 'legacy_full_checks'}

def other_unit(e,u):
    return next(ep.split('.')[0] for ep in (e['a'],e['b']) if ep.split('.')[0]!=u)

def figure13():
    return {'id':'figure13','seed':13,'vessels':[{'id':x,'jacket':x=='B','reactor':x in ('A','B')} for x in 'ABCD'],
       'transfers':[{'source':'A','target':'B','mode':'gravity'},
          {'source':'B','target':'C','mode':'distillation'},
          {'source':'B','target':'D','mode':'pump'},
          {'source':'C','target':'EXPORT_C','mode':'pump'},
          {'source':'D','target':'EXPORT_D','mode':'pump'}], 'relation':'one_to_many'}

def random_recipes(seed=20261109,count=10):
    rng=random.Random(seed); recipes=[]
    # Stratified random selection guarantees modes and cardinalities, not ten translations.
    strata=[(m,r) for m in ('pump','nitrogen','gravity') for r in ('one_to_one','many_to_one','one_to_many')]
    rng.shuffle(strata)
    for i in range(count):
        mode,rel=strata[i%len(strata)]; n=2 if rel=='one_to_one' else 3
        names=list('ABCD'[:n]); pairs=[('A','B')] if n==2 else ([('A','C'),('B','C')] if rel=='many_to_one' else [('A','B'),('A','C')])
        dn=rng.choice([32,40,50,65]); vs=[{'id':a,'orientation':rng.choice(['vertical','horizontal']), 'size':rng.choice([.85,1,1.1]),'jacket':rng.random()<.4,'dn':dn} for a in names]
        ts=[{'source':a,'target':b,'mode':mode,'dn':dn} for a,b in pairs]
        if mode=='pump':
            ts[0]['via']=rng.choice([['exchanger'],['filter'],['exchanger','filter']])
        if i==9:
            vs.append({'id':'D','orientation':'vertical','reactor':False,'dn':dn})
            ts.append({'source':names[-1],'target':'D','mode':'distillation','dn':dn,'orientation':'horizontal'})
        recipes.append({'id':f'room_{i+1:02}','seed':rng.randrange(1,10000000),'vessels':vs,'transfers':ts,'relation':rel})
    # Ensure both auxiliary unit families independent of random draws.
    pump=next(r for r in recipes if any(t['mode']=='pump' for t in r['transfers']))
    pump['transfers'][0]['via']=['exchanger','filter']
    return recipes
