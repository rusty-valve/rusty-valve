"""Compose the unmodified official OpenArm v1 with the current valve asset.

The downloaded source model and meshes remain under assets/openarm_v1.
This adapter adds a fixed mounting stand, task cameras and a nonphysical
``ee_control`` site at the stock hand TCP. It does not alter the robot's
inertias, contact meshes, joint ranges, gains or actuator force limits.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .geometry import build_scene, _camera, _s
from .resistance import ValveResistance


ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "openarm_v1"
SOURCE_MODEL = ASSET_ROOT / "openarm.xml"
ARM_JOINTS = tuple(f"openarm_joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"joint{i}_ctrl" for i in range(1, 8))
GRIPPER_ACTUATOR = "finger_ctrl"
EE_SITE = "ee_control"


def build_robot_scene(
    valve: str = "ball",
    layout: str = "horizontal_top",
    valve_position=(-.01, .125, .70),
    base_position=(0, 0, .55),
    base_quat=(1, 0, 0, 0),
    resistance_scale: float = 1.,
    close_angle_deg: float = 90.,
) -> str:
    """Return a complete MJCF scene with 7 arm and 1 stock gripper inputs.

    Valve joint follows the nine robot qpos entries. ``ee_control`` is at
    hand-local (0, 0, .09) with the hand's axes: +Z points to fingertips,
    fingers open along +/-Y; X is the transverse side-entry direction.
    Mesh file references are absolute paths in this installed task checkout,
    so saved generated XML loads from any result directory on this machine.
    Regenerate with this function after moving/copying the task checkout.
    """
    if len(base_position) != 3 or len(base_quat) != 4:
        raise ValueError("Base pose needs 3 position and 4 quaternion values")
    norm = math.sqrt(sum(float(x)*float(x) for x in base_quat))
    if not norm > 0:
        raise ValueError("Base quaternion must be nonzero")
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Official OpenArm model missing: {SOURCE_MODEL}")
    resistance = ValveResistance.from_config(valve, scale=resistance_scale)
    task = ET.fromstring(build_scene(valve, layout, close_angle_deg,
                                     position=valve_position,
                                     frictionloss=resistance.breakaway_Nm*resistance_scale))
    robot = ET.parse(SOURCE_MODEL).getroot()
    robot.set("model", f"openarm_v1_{valve}_{layout}")
    compiler = robot.find("compiler")
    compiler.attrib.pop("meshdir", None)
    compiler.set("autolimits", "true")
    for mesh in robot.findall("asset/mesh"):
        mesh.set("file", str(ASSET_ROOT / "meshes" / mesh.get("file")))

    # Scope all task defaults: no valve damping or solver setting is inherited
    # by the robot. Robot source classes and explicit values are untouched.
    valve_default = copy.deepcopy(task.find("default"))
    valve_default.set("class", "valve_task")
    robot.find("default").append(valve_default)
    robot.find("option").attrib.update(task.find("option").attrib)
    robot.append(copy.deepcopy(task.find("visual")))
    robot.find("asset").extend(copy.deepcopy(list(task.find("asset"))))

    world = robot.find("worldbody")
    base = world.find("body[@name='openarm_link0']")
    base.set("pos", _s(base_position))
    base.set("quat", _s(tuple(float(x)/norm for x in base_quat)))
    tcp = base.find(".//body[@name='openarm_hand_tcp']")
    ET.SubElement(tcp, "site", name=EE_SITE, pos="0 0 0", size=".003",
                  rgba="0 0 0 0", group="4")
    for original in task.find("worldbody"):
        element = copy.deepcopy(original)
        if element.tag == "body":
            element.set("childclass", "valve_task")
        elif element.tag == "geom":
            element.set("class", "valve_task")
        world.append(element)
    if base_position[2] > .025:
        ET.SubElement(world, "geom", name="robot_mount_stand", type="cylinder",
                      **{"class": "valve_task", "fromto": _s((*base_position[:2], .025,
                                                                 *base_position)),
                         "size": ".060", "rgba": ".18 .21 .24 1"})
        ET.SubElement(world, "geom", name="robot_mount_foot", type="box",
                      **{"class": "valve_task", "pos": _s((*base_position[:2], .0125)),
                         "size": ".12 .12 .0125", "rgba": ".18 .21 .24 1"})

    # Preserve official home controls, adding only a passive valve initial qpos.
    for key in robot.findall("keyframe/key"):
        key.set("qpos", key.get("qpos") + " 0")
    valve_joint = task.find(".//joint[@name='valve_hinge']")
    close_q = float(valve_joint.get("range").split()[1])
    for name, q in (("task_open", 0.), ("task_closed", close_q)):
        key = ET.SubElement(robot.find("keyframe"), "key", name=name,
                            qpos=f"0 0 0 1.570796 0 0 0 .044 .044 {q:.12g}",
                            ctrl="0 0 0 1.570796 0 0 0 .044")
    target = tuple((float(a)+float(b))/2 for a, b in zip(base_position, valve_position))
    target = (target[0], target[1], target[2]+.16)
    _camera(world, "robot_overview", (target[0]-.75, target[1]+1.05, target[2]+.65), target, 44)
    _camera(world, "robot_contact", (valve_position[0]-.25, valve_position[1]+.45,
                                    valve_position[2]+.32),
            (valve_position[0]+.08, valve_position[1], valve_position[2]+.075), 44)
    ET.indent(robot)
    return ET.tostring(robot, encoding="unicode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valve", choices=("ball", "globe"), default="ball")
    parser.add_argument("--layout", default="horizontal_top")
    args = parser.parse_args()
    print(build_robot_scene(valve=args.valve, layout=args.layout))
