"""Compose H1 from a pure H0 workstation and an independent robot adapter.

Workstation geometry never selects a robot or its stance. The optional H1
placement input belongs to the robot adapter and contains no motion trajectory.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .h0_station import WorkstationSpec, load_workstation

PLACEMENT_CONFIG = Path(__file__).with_name('configs') / 'h1_robot_placements.json'


def load_robot_placement(workstation='A1', *, task='close', path=PLACEMENT_CONFIG):
    """Read a robot setup, separately from the environmental description."""
    spec = load_workstation(workstation)
    document = copy.deepcopy(path) if isinstance(path, dict) else json.loads(Path(path).read_text(encoding='utf-8'))
    if document.get('schema_version') != 1:
        raise ValueError('Unsupported H1 robot placement schema')
    row = copy.deepcopy(document['placements'][spec.case])
    by_task = row.pop('task_overrides', {})
    row.update(by_task.get('open' if task == 'slow_open' else task, {}))
    region = spec.to_dict()['allowed_base_region_world']
    xy = np.asarray(row['base_position_xy_m'], dtype=float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError('Robot base position must contain two finite values')
    if np.any(xy < np.asarray(region['xy_min_m'])) or np.any(xy > np.asarray(region['xy_max_m'])):
        raise ValueError('H1 robot setup lies outside the workstation allowed base region')
    return row


def _coerce_workstation(description):
    # Keep the historical H1 developer document readable, but use only its
    # physical fields when explicitly supplied. Its robot stance never enters
    # the new pure environment. New runners load WorkstationSpec directly.
    if hasattr(description, 'robot_stance') and hasattr(description, 'case'):
        original = load_workstation(description.case)
        document = original.translated(np.asarray(description.mount_origin_world_m)
                                       -original.mount_origin_world_m).to_dict()
        if (not np.allclose(description.stem_axis_world,original.stem_axis_world)
                or not np.allclose(description.radial_zero_world,original.radial_zero_world)):
            raise ValueError('Changed installation orientations must use a pure H0 workstation description')
        document['valve'].update(
            axis_origin_world_m=list(description.mount_origin_world_m),
            stem_axis_world=list(description.stem_axis_world),
            handle_zero_radial_world=list(description.radial_zero_world),
            handle_length_m=description.handle_length_m,
            handle_plane_offset_m=description.handle_plane_offset_m,
            q_open_rad=description.q_open_rad,q_closed_rad=description.q_closed_rad)
        return WorkstationSpec(document)
    return load_workstation(description)


def build_env(spec='A1', task='close', *, robot_adapter=None,
              placement=None, placement_config=PLACEMENT_CONFIG,
              valve_offset=(0., 0., 0.), control_dt=.005, max_time_s=60.,
              side='left', contact_mode=None, physics_observer=None,
              require_endstop=False, final_hold_s=2.8, **kwargs):
    """Compose a physical H0 environment for H1's feedback controller.

    With no explicit adapter, use the sample adapter and the separate H1
    placement file. An explicitly supplied adapter keeps its own placement,
    unless ``placement`` is also supplied. No scene coordinates or paths are
    embedded in the controller, and the environment remains robot-independent.
    """
    from .h0_env import H0Env
    from .h0_robot import load_robot_adapter

    nominal = _coerce_workstation(spec)
    actual = nominal.translated(valve_offset)
    if robot_adapter is None:
        document = (copy.deepcopy(placement_config) if isinstance(placement_config,dict)
                    else json.loads(Path(placement_config).read_text(encoding='utf-8')))
        adapter = load_robot_adapter(document.get('robot_adapter','openarm_s_hook'))
        chosen = load_robot_placement(nominal,task=task,path=placement_config) if placement is None else copy.deepcopy(placement)
    else:
        adapter = load_robot_adapter(robot_adapter)
        chosen = None if placement is None else copy.deepcopy(placement)
    if chosen is not None:
        adapter = adapter.with_placement(**chosen)
    env = H0Env(workstation=actual, robot_adapter=adapter, task=task,
        control_dt=control_dt,max_time_s=max_time_s,
        physics_observer=physics_observer,require_endstop=require_endstop,
        final_hold_s=final_hold_s,side=side,contact_mode=contact_mode,**kwargs)
    env.nominal_station_spec = nominal
    env.station_spec = actual
    env.h1_setup = dict(station=actual.to_dict(),requested_task=task,
        valve_offset_m=np.asarray(valve_offset,dtype=float).tolist(),
        robot_adapter=adapter.to_dict(),robot_placement=chosen,
        physics_dt_s=env.physics_dt,
        scene='Pure H0 workstation composed with independent robot adapter',
        control_source='H1 station-frame feedback; no stored joint trajectory')
    return env
