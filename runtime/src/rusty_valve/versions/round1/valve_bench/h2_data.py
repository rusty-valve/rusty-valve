"""Frozen-v0 dataset preparation; saved states are rendered without physics steps."""


from __future__ import annotations


import json


import os


from pathlib import Path


import time


import xml.etree.ElementTree as ET


import mujoco


import numpy as np


from PIL import Image, ImageDraw


from .camera_rig import hide_camera_housing_in_scene


from .g2_review import replay_kinematics


ROOT = Path(os.environ.get('PRIMATES_VALVE_DATA_ROOT', '.'))


OUTPUT = Path(os.environ.get('PRIMATES_VALVE_LEARNING_ROOT', '.'))


TASKS = ('A2_close', 'A1_open', 'A1_close', 'A2_open')


CAMERAS = {'head': ('robot_head', 256, 144),
           'wrist': ('wrist_left', 256, 160),
           'observer': ('station_observer', 256, 144)}


FPS = 20


WRIST_OPTICAL_REAR_SHIFT_M = .04


JOINTS = [f'openarm_left_joint{i}' for i in range(1, 8)]


TEST_APPEARANCES = [
    dict(id='test_only_01', pipe=[.20, .29, .38], background=[.68, .64, .52],
         handle=[.65, .07, .13], ambient=.42, diffuse=.68),
    dict(id='test_only_02', pipe=[.55, .50, .33], background=[.28, .35, .43],
         handle=[.88, .73, .12], ambient=.61, diffuse=.43)]


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def make_model(source, root=ROOT):
    tree = ET.parse(source / 'scene.xml').getroot()
    assets = Path(os.environ.get('PRIMATES_VALVE_ASSET_ROOT', str(Path(__file__).parent / 'assets')))
    for node in tree.iter():
        filename = node.get('file')
        if filename and 'assets/' in filename:
            path = assets / filename.split('assets/', 1)[-1]
            if not path.is_file(): raise FileNotFoundError(path)
            node.set('file', str(path))
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
    model.vis.global_.offwidth = max(256, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(160, model.vis.global_.offheight)
    return model


def apply_appearance(model, profile):
    for i in range(model.ngeom):
        name = model.geom(i).name or ''
        if '_pipe_' in name and 'shadow' not in name:
            model.geom_rgba[i] = [*profile['pipe'], 1.]
        elif name.endswith('_handle_flat_grip'):
            model.geom_rgba[i] = [*profile['handle'], 1.]
        elif name in ('floor', 'ground', 'environment_floor') or 'backdrop' in name:
            model.geom_rgba[i] = [*profile['background'], 1.]
    model.vis.headlight.ambient[:] = profile['ambient']
    model.vis.headlight.diffuse[:] = profile['diffuse']
    if model.nlight:
        model.light_ambient[:] = profile['ambient'] * .25
        model.light_diffuse[:] = profile['diffuse']


class NativeRenderer:
    """Official MuJoCo render API, one context and native viewport per view."""
    def __init__(self, model):
        self.model = model
        self.gl = mujoco.GLContext(256, 160)
        self.gl.make_current()
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def image(self, data, option, name, width, height):
        self.gl.make_current()
        self.camera.fixedcamid = self.model.camera(name).id
        mujoco.mjv_updateScene(self.model, data, option, None, self.camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        hide_camera_housing_in_scene(self.scene, self.model, name)
        viewport = mujoco.MjrRect(0, 0, width, height)
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_readPixels(pixels, None, viewport, self.context)
        return pixels[::-1].copy()

    def close(self):
        self.context.free()
        self.gl.free()

