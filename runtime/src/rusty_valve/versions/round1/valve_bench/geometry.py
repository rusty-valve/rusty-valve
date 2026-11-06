"""WP1 parameterized valve and pipe scenes, generated from simple primitives.

No CAD conversion or extra Python package is needed to generate the XML.
World Z is up. In valve-local coordinates pipe=X, stem=Z, open lever=+X.
The returned model has one passive joint, ``valve_hinge``, and no actuator.
Resistance is configured separately by the simulation runner.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


ASSET_CONFIG = Path(__file__).with_name("configs") / "assets.json"
LAYOUTS = ("horizontal_front", "horizontal_top", "vertical_front", "obstructed")
VALVES = ("ball", "globe")


def asset_config() -> dict:
    return json.loads(ASSET_CONFIG.read_text(encoding="utf-8"))


def _s(values: Iterable[float]) -> str:
    return " ".join(f"{v:.10g}" for v in values)


def _rotate(q: Iterable[float], p: Iterable[float]) -> tuple[float, float, float]:
    w, x, y, z = q
    a, b, c = p
    return (
        (1 - 2 * (y*y + z*z))*a + 2*(x*y-z*w)*b + 2*(x*z+y*w)*c,
        2*(x*y+z*w)*a + (1 - 2*(x*x+z*z))*b + 2*(y*z-x*w)*c,
        2*(x*z-y*w)*a + 2*(y*z+x*w)*b + (1 - 2*(x*x+y*y))*c,
    )


def _geom(parent: ET.Element, name: str, kind: str, *, color: str,
          pos=None, size=None, fromto=None, **attrs) -> ET.Element:
    kw = {"name": name, "type": kind, "rgba": color, **attrs}
    if pos is not None:
        kw["pos"] = _s(pos)
    if size is not None:
        kw["size"] = _s(size)
    if fromto is not None:
        kw["fromto"] = _s(fromto)
    return ET.SubElement(parent, "geom", {k: str(v) for k, v in kw.items()})


def _bar_xz(parent, name, p0, p1, width, thickness, color):
    dx, dz = p1[0]-p0[0], p1[1]-p0[1]
    return _geom(
        parent, name, "box", color=color,
        pos=((p0[0]+p1[0])/2, 0, (p0[1]+p1[1])/2),
        size=(math.hypot(dx, dz)/2, width/2, thickness/2),
        euler=_s((0, -math.atan2(dz, dx), 0)),
    )


def _camera(parent, name, pos, target, fovy):
    def unit(v):
        n = math.sqrt(sum(x*x for x in v))
        return tuple(x/n for x in v)
    z = unit(tuple(a-b for a, b in zip(pos, target)))
    x = unit((-z[1], z[0], 0))
    y = (z[1]*x[2]-z[2]*x[1], z[2]*x[0]-z[0]*x[2], z[0]*x[1]-z[1]*x[0])
    ET.SubElement(parent, "camera", name=name, pos=_s(pos), xyaxes=_s((*x, *y)), fovy=str(fovy))


def _ball(fixed, moving, cfg):
    metal = ".64 .67 .70 1"
    blue = ".025 .35 .67 1"
    half = cfg["body_length_m"]/2
    _geom(fixed, "ball_body", "cylinder", color=metal,
          fromto=(-half, 0, 0, half, 0, 0), size=(cfg["body_radius_m"],))
    for side in (-1, 1):
        x = side*(half-.009)
        _geom(fixed, f"ball_connection_{side}", "cylinder", color=metal,
              fromto=(x-.009, 0, 0, x+.009, 0, 0), size=(cfg["connection_radius_m"],))
    # A three-piece body envelope, not the manufacturer's internal mechanism.
    for y in (-.023, .023):
        for z in (-.023, .023):
            _geom(fixed, f"ball_tie_bolt_{y}_{z}", "cylinder", color=".43 .46 .49 1",
                  fromto=(-half+.008, y, z, half-.008, y, z), size=(.003,))
    mounting_top = cfg["mounting_height_above_pipe_m"]
    _geom(fixed, "ball_stem_bonnet", "cylinder", color=metal,
          fromto=(0, 0, .018, 0, 0, mounting_top-.004), size=(.010,))
    _geom(fixed, "ball_mounting_plate", "box", color=metal,
          pos=(0, 0, mounting_top-.002), size=(.023, .023, .002))
    root_z = mounting_top + .002
    plate_t = cfg["lever_plate_thickness_m"]
    _geom(moving, "handle_hub", "cylinder", color=metal,
          pos=(0, 0, root_z-.0003), size=(.015, plate_t/2))
    _geom(moving, "handle_hub_fastener", "cylinder", color=metal,
          pos=(0, 0, root_z+.0045), size=(.006, .003))
    # One raised flat strip with a thin cover, matching the approved generic style.
    # D (reach), C (top), C1 (mount) use Bray's dimension drawing. Cross sections
    # and bend locations are explicit task assumptions, not extracted CAD.
    width, thickness = cfg["lever_mid_width_m"], cfg["lever_mid_thickness_m"]
    grip_z = cfg["lever_highest_m"]-thickness/2
    bend_start, bend_end = cfg["lever_bend_start_m"], cfg["lever_bend_end_m"]
    _bar_xz(moving, "handle_root_flat", (.008, root_z), (bend_start, root_z),
            width, plate_t, metal)
    _bar_xz(moving, "handle_root_bend", (bend_start-.001, root_z), (bend_end, grip_z),
            width, plate_t, metal)
    _bar_xz(moving, "handle_flat_core", (bend_end-.001, grip_z),
            (cfg["lever_reach_m"]-.001, grip_z), width-.002, plate_t, metal)
    cover_start = cfg["lever_cover_start_m"]
    _geom(moving, "handle_flat_grip", "box", color=blue,
          pos=((cover_start+cfg["lever_reach_m"])/2, 0, grip_z),
          size=((cfg["lever_reach_m"]-cover_start)/2, width/2, thickness/2))


def _globe(fixed, moving, cfg):
    metal = ".52 .56 .6 1"
    body_color = ".20 .27 .31 1"
    wheel_color = ".68 .12 .075 1"
    half = cfg["body_length_m"]/2
    _geom(fixed, "globe_body", "ellipsoid", color=body_color,
          pos=(0, 0, .009), size=(half*.66, cfg["body_radius_m"], .043))
    _geom(fixed, "globe_flow_neck", "cylinder", color=body_color,
          fromto=(-half, 0, 0, half, 0, 0), size=(.022,))
    for side in (-1, 1):
        x = side*(half-.007)
        _geom(fixed, f"globe_flange_{side}", "cylinder", color=metal,
              fromto=(x-.007, 0, 0, x+.007, 0, 0), size=(cfg["flange_radius_m"],))
        for n in range(4):
            a = math.pi/4+n*math.pi/2
            _geom(fixed, f"globe_flange_bolt_{side}_{n}", "cylinder", color=".21 .23 .25 1",
                  fromto=(x-.009, .043*math.cos(a), .043*math.sin(a),
                          x+.009, .043*math.cos(a), .043*math.sin(a)), size=(.0045,))
    _geom(fixed, "globe_bonnet", "cylinder", color=body_color,
          fromto=(0, 0, .03, 0, 0, .087), size=(.018,))
    for x in (-.014, .014):
        _geom(fixed, f"globe_yoke_{x}", "box", color=metal,
              pos=(x, 0, .123), size=(.004, .009, .036))
    _geom(fixed, "globe_yoke_bridge", "box", color=metal,
          pos=(0, 0, .157), size=(.019, .012, .004))
    wh = cfg["wheel_height_above_pipe_m"]
    _geom(fixed, "globe_stem_fixed_proxy", "cylinder", color=metal,
          fromto=(0, 0, .083, 0, 0, wh-.006), size=(.006,))
    _geom(moving, "wheel_hub", "cylinder", color=wheel_color,
          pos=(0, 0, wh), size=(.010, .004))
    tube = cfg["wheel_tube_radius_m"]
    radius = cfg["wheel_outside_diameter_m"]/2-tube
    count = int(cfg["wheel_segments"])
    for n in range(count):
        a = n*math.tau/count
        b = (n+1)*math.tau/count
        _geom(moving, f"wheel_rim_{n:02d}", "capsule", color=wheel_color,
              fromto=(radius*math.cos(a), radius*math.sin(a), wh,
                      radius*math.cos(b), radius*math.sin(b), wh), size=(tube,))
    for n in range(int(cfg["wheel_spokes"])):
        a = n*math.tau/int(cfg["wheel_spokes"])
        _geom(moving, f"wheel_spoke_{n}", "capsule", color=wheel_color,
              fromto=(.008*math.cos(a), .008*math.sin(a), wh,
                      radius*math.cos(a), radius*math.sin(a), wh), size=(.0035,))
    _geom(moving, "wheel_rotation_marker", "sphere", color=".96 .88 .63 1",
          pos=(radius, 0, wh+.0035), size=(.003,), contype="0", conaffinity="0", density="0")


def build_scene(valve: str = "ball", layout: str = "horizontal_front", close_angle_deg: float = 90,
                *, position=(0, 0, .8), timestep: float = .001,
                gravity=(0, 0, -9.81), valve_damping: float = .005,
                frictionloss: float = 0, params: dict | None = None) -> str:
    """Return a complete passive MuJoCo scene for a valve/layout combination.

    ``params`` overrides that valve's numeric configuration. Joint q=0 is open,
    and positive q closes. The ball's upper limit is configurable 87-92 degrees.
    The globe's configured four-turn span does not claim a real shutoff stroke.
    """
    data = asset_config()
    if valve not in VALVES:
        raise ValueError(f"Unknown valve {valve!r}; choose {VALVES}")
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}; choose {LAYOUTS}")
    if not 0 < close_angle_deg <= 180:
        raise ValueError("close_angle_deg must be in (0, 180]")
    if timestep <= 0 or len(position) != 3:
        raise ValueError("timestep must be positive and position must have three coordinates")
    cfg = copy.deepcopy(data[valve])
    if params:
        cfg.update(params)
    q = data["layouts"][layout]["quaternion_wxyz"]
    model = ET.Element("mujoco", model=f"wp1_{valve}_{layout}")
    ET.SubElement(model, "compiler", angle="radian", autolimits="true")
    ET.SubElement(model, "option", timestep=str(timestep), gravity=_s(gravity), integrator="implicitfast",
                  cone="elliptic", iterations="100")
    default = ET.SubElement(model, "default")
    ET.SubElement(default, "geom", friction=".6 .01 .001", condim="4", solref=".004 1",
                  solimp=".95 .99 .001", density="1000")
    ET.SubElement(default, "joint", damping=str(valve_damping), armature=".00001")
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    asset = ET.SubElement(model, "asset")
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker", rgb1=".25 .28 .30",
                  rgb2=".30 .33 .35", width="256", height="256")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="5 5", reflectance=".1")
    world = ET.SubElement(model, "worldbody")
    ET.SubElement(world, "light", name="key", pos="1 -2 3", dir="-.2 .4 -1", directional="true")
    ET.SubElement(world, "light", name="fill", pos="-1 -1 2", dir=".4 .3 -1", diffuse=".5 .5 .5")
    _geom(world, "floor", "plane", color="1 1 1 1", pos=(0, 0, 0), size=(2, 2, .1), material="floor_mat")
    target = (position[0], position[1], position[2]+.02)
    _camera(world, "overview", (position[0]+.5, position[1]-.78, position[2]+.38), target, 42)
    _camera(world, "closeup", (position[0]+.25, position[1]-.42, position[2]+.24), target, 44)
    _camera(world, "handle_side", (position[0]+.09, position[1]-.34, position[2]+.10),
            (position[0]+.09, position[1], position[2]+.045), 38)
    _camera(world, "handle_top", (position[0]+.09, position[1]-.0001, position[2]+.43),
            (position[0]+.09, position[1], position[2]+.02), 38)
    fixed = ET.SubElement(world, "body", name="valve_mount", pos=_s(position), quat=_s(q))
    # The moving body is a sibling of the fixed mount: contact with pipes is enabled.
    moving = ET.SubElement(world, "body", name="valve_moving", pos=_s(position), quat=_s(q))
    upper = math.radians(close_angle_deg) if valve == "ball" else math.tau*cfg["turns_open_to_closed"]
    ET.SubElement(moving, "joint", name="valve_hinge", type="hinge", axis="0 0 1", limited="true",
                  range=f"0 {upper:.12g}", frictionloss=str(frictionloss),
                  solreffriction=".002 1", solimpfriction=".9999 .9999 .001 .5 2",
                  solreflimit=".002 1", solimplimit=".9999 .9999 .001 .5 2")
    inertial_z = cfg["lever_highest_m"]-cfg["lever_mid_thickness_m"]/2 if valve == "ball" else cfg["wheel_height_above_pipe_m"]
    inertial_x = cfg["lever_reach_m"]/2 if valve == "ball" else 0
    ET.SubElement(moving, "inertial", pos=_s((inertial_x, 0, inertial_z)), mass=str(cfg["moving_mass_kg"]),
                  diaginertia=_s(cfg["moving_inertia_kg_m2"]))
    radius = data["pipe"]["outside_diameter_m"]/2
    half = cfg["body_length_m"]/2
    plen = data["pipe"]["length_m"]/2
    for side in (-1, 1):
        _geom(fixed, f"pipe_{side}", "cylinder", color=".50 .54 .57 1",
              fromto=(side*half, 0, 0, side*plen, 0, 0), size=(radius,))
        _geom(fixed, f"pipe_end_shadow_{side}", "cylinder", color=".075 .09 .10 1",
              fromto=(side*(plen+.0001), 0, 0, side*(plen+.0003), 0, 0), size=(.0125,),
              contype="0", conaffinity="0")
    if valve == "ball":
        _ball(fixed, moving, cfg)
    else:
        _globe(fixed, moving, cfg)
    if layout == "obstructed":
        ox, oy, oz = data["layouts"][layout]["obstacle_pipe_offset_local_m"]
        _geom(fixed, "neighbor_pipe", "cylinder", color=".66 .53 .28 1",
              fromto=(ox-.28, oy, oz, ox+.28, oy, oz), size=(radius,))
        for side in (-1, 1):
            local = _rotate(q, (ox+side*.23, oy, oz))
            p = tuple(position[i]+local[i] for i in range(3))
            _geom(world, f"neighbor_support_post_{side}", "cylinder", color=".29 .27 .21 1",
                  fromto=(p[0], p[1], .018, p[0], p[1], p[2]-radius), size=(.010,))
            _geom(world, f"neighbor_support_foot_{side}", "box", color=".24 .22 .18 1",
                  pos=(p[0], p[1], .009), size=(.045, .035, .009))
    # Visible ground-mounted supports explain how the pipe is held fixed.
    pipe_dir = _rotate(q, (1, 0, 0))
    if abs(pipe_dir[2]) < .5:
        for side in (-1, 1):
            p = tuple(position[i]+side*.27*pipe_dir[i] for i in range(3))
            _geom(world, f"support_post_{side}", "cylinder", color=".22 .25 .27 1",
                  fromto=(p[0], p[1], .025, p[0], p[1], p[2]-radius), size=(.012,))
            _geom(world, f"support_foot_{side}", "box", color=".18 .20 .21 1",
                  pos=(p[0], p[1], .0125), size=(.065, .055, .0125))
    else:
        _geom(world, "vertical_pipe_pedestal", "cylinder", color=".22 .25 .27 1",
              fromto=(position[0], position[1], .025, position[0], position[1], position[2]-plen), size=(.022,))
        _geom(world, "vertical_pipe_foot", "box", color=".18 .20 .21 1",
              pos=(position[0], position[1], .0125), size=(.1, .1, .0125))
    ET.SubElement(moving, "site", name="valve_axis", pos="0 0 0", size=".002", rgba="0 0 0 0")
    keyframes = ET.SubElement(model, "keyframe")
    ET.SubElement(keyframes, "key", name="open", qpos="0")
    ET.SubElement(keyframes, "key", name="closed", qpos=f"{upper:.12g}")
    ET.indent(model)
    return ET.tostring(model, encoding="unicode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valve", choices=VALVES, default="ball")
    parser.add_argument("--layout", choices=LAYOUTS, default="horizontal_front")
    parser.add_argument("--close-angle-deg", type=float, default=90)
    args = parser.parse_args()
    print(build_scene(args.valve, args.layout, args.close_angle_deg))
