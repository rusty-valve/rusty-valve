"""A fixed observer across the pipe, separate from the robot's visual input.

Placement uses only the stationary workstation and robot setup.  It does not
read a trajectory, follow a moving tool, alter a body, or step the simulation.
The camera uses the earlier A1 oblique viewing direction, moved closer to
frame the upper body, tools and valve. The mounting stand is outside the focus.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from .geometry import _camera
from .h0_station import load_workstation


OBSERVER_CAMERA_NAME = "station_observer"


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return result


def _placement(robot_adapter, position, shoulder_height):
    if robot_adapter is not None:
        if hasattr(robot_adapter, "to_dict"):
            document = robot_adapter.to_dict()
        elif isinstance(robot_adapter, Mapping):
            document = robot_adapter
        else:
            from .h0_robot import load_robot_adapter
            document = load_robot_adapter(robot_adapter).to_dict()
        p = document.get("placement", document)
        if position is None:
            position = p["base_position_m"]
        if shoulder_height is None:
            shoulder_height = p.get("shoulder_height_m")
    if position is None:
        raise ValueError("An adapter or robot_position_world_m is required")
    base = _vector(position, "robot_position_world_m")
    if shoulder_height is None:
        raise ValueError("Declare shoulder_height_m for the observer's upper-body framing")
    shoulder_height = float(shoulder_height)
    if not math.isfinite(shoulder_height):
        raise ValueError("shoulder_height_m must be finite")
    return base, shoulder_height


def observer_camera_config(workstation, robot_adapter=None, *,
                           robot_position_world_m=None, shoulder_height_m=None,
                           vertical_fov_deg=40., aspect_ratio=16 / 9,
                           side_sign=1., across_offset_m=.64,
                           lateral_offset_m=.54, elevation_offset_m=.48,
                           target_height_offset_m=-.06):
    """Close upper-body view based on the earlier A1 inspection direction.

    Offsets are measured from a fixed point between the shoulder and valve,
    relative to the pipe's opposite side. No trajectory or moving pose is used.
    Camera framing is configurable independently of physics and control.
    """
    station = load_workstation(workstation)
    base, shoulder_z = _placement(robot_adapter, robot_position_world_m,
                                  shoulder_height_m)
    fovy, aspect = float(vertical_fov_deg), float(aspect_ratio)
    if not (5. < fovy < 120.) or not (aspect > 0. and math.isfinite(aspect)):
        raise ValueError("Invalid observer field of view or aspect ratio")
    if side_sign not in (-1., 1.):
        raise ValueError("side_sign must be -1 or +1")
    for name, value in (("across_offset_m", across_offset_m),
                        ("lateral_offset_m", lateral_offset_m),
                        ("elevation_offset_m", elevation_offset_m)):
        if not math.isfinite(value) or value <= 0.:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(target_height_offset_m):
        raise ValueError("target_height_offset_m must be finite")
    mount, pivot = station.mount_origin_world_m, station.pivot_world_m
    up = np.array([0., 0., 1.])
    toward_robot = base - mount
    toward_robot[2] = 0.
    pipe_horizontal = station.pipe_axis_world.copy()
    pipe_horizontal[2] = 0.
    pipe_norm = np.linalg.norm(pipe_horizontal)
    if pipe_norm > 1e-8:
        pipe_horizontal /= pipe_norm
        perpendicular = toward_robot - (toward_robot @ pipe_horizontal) * pipe_horizontal
        if np.linalg.norm(perpendicular) > 1e-8:
            toward_robot = perpendicular
    if np.linalg.norm(toward_robot) < 1e-8:
        raise ValueError("Robot and pipe need a distinct horizontal side for this observer")
    across = -toward_robot / np.linalg.norm(toward_robot)
    side = side_sign * np.cross(up, across)
    shoulder = np.array([base[0], base[1], shoulder_z])
    target = .5 * (pivot + shoulder) + target_height_offset_m * up
    position = target + across_offset_m * across + lateral_offset_m * side + elevation_offset_m * up
    if (position - mount) @ toward_robot >= 0.:
        raise ValueError("Computed observer did not reach the opposite side of the pipe")
    return dict(name=OBSERVER_CAMERA_NAME, parent="worldbody", station=station.case,
                role="human action inspection; separate from robot visual observation",
                mode="fixed world camera; no target tracking or per-frame adjustment",
                position_world_m=position.tolist(), target_world_m=target.tolist(),
                vertical_fov_deg=fovy, aspect_ratio=aspect,
                distance_m=float(np.linalg.norm(position-target)),
                across_pipe_direction_world=across.tolist(), side_direction_world=side.tolist(),
                framing_source="earlier A1 overview direction, moved closer to frame upper body and valve",
                offset_m=dict(across=across_offset_m, lateral=lateral_offset_m,
                              elevation=elevation_offset_m, target_height=target_height_offset_m),
                robot_base_position_world_m=base.tolist(), shoulder_height_m=shoulder_z,
                note="Fixed upper-body inspection camera. No framing guarantee for unrecorded motions; actual occlusion is retained.")


def attach_observer_camera(root, workstation, robot_adapter=None, **kwargs):
    """Add/replace only ``station_observer`` under worldbody; return metadata."""
    config = observer_camera_config(workstation, robot_adapter, **kwargs)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Scene has no worldbody")
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "camera" and child.get("name") == OBSERVER_CAMERA_NAME:
                parent.remove(child)
    _camera(world, OBSERVER_CAMERA_NAME, config["position_world_m"],
            config["target_world_m"], config["vertical_fov_deg"])
    world.find(f"camera[@name='{OBSERVER_CAMERA_NAME}']").set("mode", "fixed")
    return config
