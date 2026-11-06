"""Robot-independent physical workstation descriptions for H0.

This module deliberately has no robot, tool, stance or control dependency.
The description represents the task environment shared by every participant.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np

from .geometry import asset_config
from .resistance import CONFIG_PATH as RESISTANCE_CONFIG, ValveResistance

CONFIG_ROOT = Path(__file__).with_name('configs') / 'workstations'
SIX_CONFIG = Path(__file__).with_name('configs') / 'six_workstations.json'


def _v3(value, name, unit=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain three finite values')
    if unit and not np.isclose(np.linalg.norm(result), 1., atol=1e-7):
        raise ValueError(f'{name} must be a unit vector')
    return result


@dataclass(frozen=True)
class WorkstationSpec:
    """Validated environment-only document; read properties do not mutate it."""
    _document: dict

    def __post_init__(self):
        d = self._document
        if d.get('schema_version') != 1:
            raise ValueError('Unsupported H0 workstation schema')
        if not isinstance(d.get('case'), str) or not d['case']:
            raise ValueError('Workstation case must be a nonempty string')
        if any(key in d for key in ('robot', 'robot_stance', 'tool', 'controller', 'adapter')):
            raise ValueError('Workstation descriptions must not select a robot or controller')
        stem = _v3(self.stem_axis_world, 'stem_axis_world', True)
        radial = _v3(self.radial_zero_world, 'handle_zero_radial_world', True)
        pipe = _v3(self.pipe_axis_world, 'pipe.axis_world', True)
        _v3(self.mount_origin_world_m, 'axis_origin_world_m')
        if abs(stem@radial) > 1e-7 or abs(stem@pipe) > 1e-7:
            raise ValueError('Stem must be perpendicular to handle radial and pipe axis')
        if not 0 < self.handle_length_m:
            raise ValueError('Handle length must be positive')
        if not self.q_close_rad > self.q_open_rad:
            raise ValueError('Closing joint coordinate must increase from open to closed')
        if d['valve']['closing_direction_about_stem'] not in (-1, 1):
            raise ValueError('Closing rotation must be +1 or -1 about the outward stem')
        self.make_resistance()._validate()

    @property
    def case(self): return self._document['case']
    @property
    def title(self): return self._document['title']
    @property
    def valve(self): return self._document['valve']
    @property
    def mount_origin_world_m(self): return np.asarray(self.valve['axis_origin_world_m'], dtype=float)
    @property
    def stem_axis_world(self): return np.asarray(self.valve['stem_axis_world'], dtype=float)
    @property
    def radial_zero_world(self): return np.asarray(self.valve['handle_zero_radial_world'], dtype=float)
    @property
    def closing_axis_world(self): return self.valve['closing_direction_about_stem']*self.stem_axis_world
    @property
    def handle_length_m(self): return float(self.valve['handle_length_m'])
    @property
    def handle_plane_offset_m(self): return float(self.valve['handle_plane_offset_m'])
    @property
    def pivot_world_m(self): return self.mount_origin_world_m+self.handle_plane_offset_m*self.stem_axis_world
    @property
    def q_open_rad(self): return float(self.valve['q_open_rad'])
    @property
    def q_close_rad(self): return float(self.valve['q_closed_rad'])
    @property
    def q_closed_rad(self): return self.q_close_rad
    @property
    def pipe_axis_world(self): return np.asarray(self._document['pipe']['axis_world'], dtype=float)
    @property
    def resistance_parameters(self): return copy.deepcopy(self.valve['resistance'])
    @property
    def valve_joint_name(self): return f'{self.case}_valve_hinge'
    @property
    def valve_moving_body_name(self): return f'{self.case}_valve_moving'
    @property
    def valve_mount_body_name(self): return f'{self.case}_valve_mount'
    @property
    def installation_body_name(self): return f'{self.case}_installation'
    @property
    def handle_tip_site_name(self): return f'{self.case}_handle_tip'

    def to_dict(self): return copy.deepcopy(self._document)

    def radial_at(self, q):
        a = self.closing_axis_world; r = self.radial_zero_world
        theta = float(q)-self.q_open_rad
        return math.cos(theta)*r+math.sin(theta)*np.cross(a, r)

    def make_resistance(self):
        p = self.resistance_parameters
        p.update(valve='ball', joint_name=self.valve_joint_name)
        for name in ('solref', 'solimp'):
            p[name] = tuple(p[name])
        return ValveResistance(**p)

    def translated(self, offset, *, move_base_region=False):
        delta = _v3(offset, 'offset')
        d = self.to_dict()
        d['valve']['axis_origin_world_m'] = (self.mount_origin_world_m+delta).tolist()
        for g in d['supports']:
            if 'pos' in g:
                g['pos'] = (np.asarray(g['pos'])+delta).tolist()
            if 'fromto' in g:
                g['fromto'] = (np.asarray(g['fromto']).reshape(2,3)+delta).reshape(-1).tolist()
        if move_base_region:
            region = d['allowed_base_region_world']
            for key in ('xy_min_m', 'xy_max_m'):
                region[key] = (np.asarray(region[key])+delta[:2]).tolist()
        return WorkstationSpec(d)


def load_workstation(case_or_path='A1'):
    if isinstance(case_or_path, WorkstationSpec):
        return case_or_path
    if isinstance(case_or_path, Mapping):
        return WorkstationSpec(copy.deepcopy(dict(case_or_path)))
    path = Path(case_or_path)
    if not path.is_file():
        path = CONFIG_ROOT / f'{case_or_path}.json'
    return WorkstationSpec(json.loads(path.read_text(encoding='utf-8')))


def generate_descriptions():
    """Copy the agreed six installation geometries into independent documents."""
    cfg = json.loads(SIX_CONFIG.read_text(encoding='utf-8'))
    geometry = asset_config()
    resistance = json.loads(RESISTANCE_CONFIG.read_text(encoding='utf-8'))
    params = {k:v for k,v in resistance['common'].items() if k != 'joint_name'}
    params.update(resistance['profiles']['ball']['parameters'])
    params['scale'] = 1.
    documents = []
    for station in cfg['stations']:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(station['installation_quaternion_wxyz']))
        R = R.reshape(3,3); mount = np.asarray(station['position_m'], dtype=float)
        stem = R[:,2]; pipe = R[:,0]
        phase = math.radians(station['handle_phase_deg'])
        radial = R@np.array([math.cos(phase),math.sin(phase),0.])
        supports = []
        if abs(pipe[2]) < .5:
            for side in (-1,1):
                p = mount+side*.27*pipe
                supports.extend([
                    dict(name=f'support_{side}', type='cylinder', fromto=[*p[:2],.025,*p[:2],p[2]-.01685], size=[.012], rgba=[.24,.27,.29,1.]),
                    dict(name=f'foot_{side}',type='box',pos=[*p[:2],.0125],size=[.065,.055,.0125],rgba=[.20,.23,.25,1.])])
        else:
            supports.extend([
                dict(name='vertical_pedestal',type='cylinder',fromto=[*mount[:2],.025,*mount[:2],mount[2]-.35],size=[.022],rgba=[.24,.27,.29,1.]),
                dict(name='vertical_foot',type='box',pos=[*mount[:2],.0125],size=[.1,.1,.0125],rgba=[.20,.23,.25,1.])])
        params2 = copy.deepcopy(params); params2['breakaway_Nm'] = cfg['valve']['breakaway_reference_Nm']
        documents.append(dict(schema_version=1,case=station['case'],title=station['title'],
            units=dict(length='m',angle='rad',force='N',torque='N*m',time='s'),
            valve=dict(kind='ball',axis_origin_world_m=mount.tolist(),stem_axis_world=stem.tolist(),
                handle_zero_radial_world=radial.tolist(),handle_length_m=cfg['valve']['handle_reach_m'],
                handle_plane_offset_m=.084,q_open_rad=0.,q_closed_rad=math.radians(cfg['valve']['closing_limit_deg']),
                closing_direction_about_stem=-1, geometry=copy.deepcopy(geometry['ball']), resistance=params2,
                resistance_provenance=copy.deepcopy(resistance['profiles']['ball'])),
            pipe=dict(axis_world=pipe.tolist(),outside_diameter_m=cfg['valve']['pipe_outside_diameter_m'],length_m=cfg['valve']['pipe_length_m']),
            supports=supports,
            allowed_base_region_world=dict(kind='axis_aligned_rectangle',xy_min_m=[float(mount[0]-.9),float(mount[1]-.8)],
                xy_max_m=[float(mount[0]-.1),float(mount[1]+.8)],floor_z_m=0.,
                interpretation='Allowed fixed-base reference position only; the submitted physical geometry must also remain collision-free.'),
            physics=dict(timestep_s=.001,gravity_world_m_s2=[0.,0.,-9.81],integrator='implicitfast',solver='Newton',iterations=100,ls_iterations=50),
            provenance='Derived from the agreed six_workstations.json environmental geometry. No measured real-valve force data.'))
    return documents


if __name__ == '__main__':
    CONFIG_ROOT.mkdir(parents=True,exist_ok=True)
    for document in generate_descriptions():
        target = CONFIG_ROOT / f"{document['case']}.json"
        target.write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(target)
