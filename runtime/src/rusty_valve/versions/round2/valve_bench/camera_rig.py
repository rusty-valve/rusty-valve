"""Shared visible OpenArm camera modules and optical frames.

The ZED Mini visual mesh comes unchanged from Stereolabs. The S-tool camera
holders and head post are explicit project adaptations. Default additions are
massless and non-colliding: optical/visual integration is not a physical-kit
inertia validation. Existing joint order, gains and contacts stay untouched.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

CONFIG_PATH = Path(__file__).with_name('configs') / 'openarm_camera_rig.json'
CAMERA_NAMES = ('head_left', 'head_right', 'wrist_left', 'wrist_right')
HEAD_MOUNT_BODY = 'openarm_head_camera_mount'
_SOURCE_ROS_TO_MJ_CAMERA = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])


def load_camera_rig():
    document = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if document.get('schema') != 'openarm_camera_rig_v1':
        raise ValueError('Unsupported camera-rig configuration')
    return document


def default_head_settings():
    return copy.deepcopy(load_camera_rig()['head']['default_settings'])


def sensor_image_size(name):
    if name not in (*CAMERA_NAMES, 'robot_head'):
        raise ValueError(f'Unknown robot camera: {name}')
    kind = 'head' if name.startswith('head') or name=='robot_head' else 'wrist'
    return tuple(load_camera_rig()[kind]['image_size_px'])


def _s(value):
    return ' '.join(f'{float(x):.12g}' for x in value)


def _quat(rotation):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(rotation, dtype=float).ravel())
    return q


def _matrix(quaternion):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=float))
    return result.reshape(3,3)


def _optical_rotation(direction, up):
    forward = np.asarray(direction, dtype=float)
    forward /= np.linalg.norm(forward)
    z = -forward
    x = np.cross(np.asarray(up, dtype=float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def _head_pose(settings, shoulder_offset):
    tilt, yaw = math.radians(settings['tilt']), math.radians(settings['yaw'])
    direction = [math.cos(tilt)*math.cos(yaw), math.cos(tilt)*math.sin(yaw), -math.sin(tilt)]
    rotation = _optical_rotation(direction, [0.,0.,1.])
    position = np.array([settings['forward']/100., 0., shoulder_offset+settings['height']/100.])
    return position, rotation


def _support_segments(position, rotation, shoulder_offset, height):
    base = np.array([0.,0.,shoulder_offset+.03])
    corner = np.array([0.,0.,position[2]])
    bottom = position + rotation @ np.array([0.,-height/2,.010])
    return [(base,corner), (corner,bottom)]


def _profile(adapter_document):
    selected = adapter_document.get('camera_rig', {}).get('profile', 'visual_only')
    rig = load_camera_rig()
    if selected not in rig['profiles']:
        raise ValueError(f'Unknown camera-rig profile: {selected}')
    return selected, rig['profiles'][selected]


def _geom(parent, name, kind, *, rgba, profile, pos=None, size=None, fromto=None, quat=None, mesh=None):
    attributes = dict(name=name, type=kind, rgba=rgba, density='0', group='2',
        contype='1' if profile['collision_enabled'] else '0',
        conaffinity='1' if profile['collision_enabled'] else '0')
    for key,value in [('pos',pos),('size',size),('fromto',fromto),('quat',quat)]:
        if value is not None: attributes[key] = _s(value)
    if mesh is not None: attributes['mesh'] = mesh
    return ET.SubElement(parent, 'geom', attributes)


def _camera(parent, name, position, fov):
    return ET.SubElement(parent, 'camera', name=name, pos=_s(position),
        quat='1 0 0 0', fovy=str(fov), mode='fixed')


def _wrist_installation(root, side, rig):
    wrist = root.find(f".//body[@name='openarm_{side}_link7']")
    if wrist is None:
        raise ValueError(f'Missing {side} OpenArm wrist')
    tool = wrist.find(f"body[@name='openarm_{side}_hook']")
    if tool is None:
        tool = wrist.find(f"body[@name='openarm_{side}_hand']")
    if tool is None:
        raise ValueError(f'Missing fixed {side} tool installation frame')
    # Mount to the wrist, independently of a fixed hook installation flip.
    # Both wrists share these axes: +X is the camera side, including on the
    # left-Z180 hook variant. Reusing hook axes would put that camera below it.
    origin = np.fromstring(tool.get('pos', '0 0 0'), sep=' ')
    optical = origin + np.asarray(rig['wrist']['mount_optical_offset_in_wrist_frame_m'],dtype=float)
    target = origin + np.asarray(rig['wrist']['mount_aim_offset_in_wrist_frame_m'],dtype=float)
    local_rotation = _optical_rotation(target-optical, rig['wrist']['mount_up_in_wrist_frame'])
    anchor = origin + np.asarray(rig['wrist']['support_anchor_offset_in_wrist_frame_m'],dtype=float)
    return wrist, optical, local_rotation, anchor


def camera_metadata(adapter_document, *, root=None):
    """Describe optics and the declared model approximation without mutation."""
    rig = load_camera_rig(); profile_name, profile = _profile(adapter_document)
    cameras = []
    for name in (*CAMERA_NAMES, 'robot_head'):
        kind = 'head' if name.startswith('head') or name=='robot_head' else 'wrist'
        specs = rig[kind]
        aspect = specs['model_pinhole_projection']['aspect_ratio']
        fov = specs['model_pinhole_projection']['vertical_fov_deg']
        parent = HEAD_MOUNT_BODY if kind=='head' else f'openarm_{name.split("_")[1]}_wrist_camera_mount'
        position = ([(-.5 if name in ('head_left','robot_head') else .5)*specs['stereo_baseline_m'],0.,0.]
                    if kind=='head' else [0.,0.,0.])
        if root is not None:
            actual = root.find(f".//camera[@name='{name}']")
            if actual is not None:
                fov = float(actual.get('fovy', fov))
                position = np.fromstring(actual.get('pos','0 0 0'),sep=' ').tolist()
        cameras.append(dict(name=name, model=specs['model'], parent_body=parent,
            position_local_m=position, forward_direction_local=[0.,0.,-1.],
            vertical_fov_deg=fov, image_size_px=list(specs['image_size_px']),
            recommended_image_size_px=list(specs['image_size_px']),
            source_nominal_fov_deg=copy.deepcopy(specs['source_nominal_fov_deg']),
            model_pinhole_projection=dict(vertical_fov_deg=fov, aspect_ratio=aspect,
                horizontal_fov_deg=math.degrees(2*math.atan(math.tan(math.radians(fov/2))*aspect)),
                distortion_simulated=False, calibrated_to_real_unit=False),
            profile='openarm_camera_rig_v1', physical_profile=profile_name,
            added_mass_kg=profile['added_mass_kg'], collision_enabled=profile['collision_enabled'],
            role='compatibility alias of head_left' if name=='robot_head' else 'robot optical observation',
            own_housing_optical_path='Hide only this module housing/lens in its own image; keep robot/tool/environment occlusion.',
            installation_status=specs['mount_reference'], geometry_status=specs.get('geometry_status',specs.get('geometry',{}).get('geometry_status')),
            sources=copy.deepcopy(specs['sources'])))
    return cameras


def install_camera_rig(root, adapter_document):
    """Attach visible cameras once, after the stock/S-tool installation step."""
    rig = load_camera_rig(); _, profile = _profile(adapter_document)
    world = root.find('worldbody')
    base = world.find(f".//body[@name='{adapter_document['placement']['root_body']}']")
    if base is None: raise ValueError('Camera rig needs the declared robot root body')
    # Idempotent installation also removes legacy single-eye cameras.
    body_names = {HEAD_MOUNT_BODY, 'openarm_left_wrist_camera_mount', 'openarm_right_wrist_camera_mount'}
    for parent in root.iter():
        for child in list(parent):
            if ((child.tag=='body' and child.get('name') in body_names)
                    or (child.tag=='camera' and child.get('name') in (*CAMERA_NAMES,'robot_head'))
                    or (child.tag=='geom' and (child.get('name','').startswith('camera_rig_head_support_')
                        or child.get('name') in ('camera_rig_left_wrist_support','camera_rig_right_wrist_support')))):
                parent.remove(child)
    head = rig['head']; settings = head['default_settings']
    shoulder = float(adapter_document['placement'].get('shoulder_reference_offset_m', .697))
    position, rotation = _head_pose(settings, shoulder)
    body = ET.SubElement(base,'body',name=HEAD_MOUNT_BODY,pos=_s(position),quat=_s(_quat(rotation)))
    asset = root.find('asset')
    if asset is None: asset = ET.SubElement(root,'asset')
    mesh_name = 'camera_rig_zedmini_official'
    if asset.find(f"mesh[@name='{mesh_name}']") is None:
        mesh_path=(CONFIG_PATH.parent/head['geometry']['mesh_file']).resolve()
        ET.SubElement(asset,'mesh',name=mesh_name,file=str(mesh_path))
    _geom(body,'head_camera_housing','mesh',mesh=mesh_name,quat=_quat(_SOURCE_ROS_TO_MJ_CAMERA),rgba='.18 .19 .21 1',profile=profile)
    # Opaque display glass is only visual; the optical sensor's own housing is
    # excluded from its image because its optical center is inside the case.
    glass_profile = dict(profile,collision_enabled=False)
    for name,offset in [('head_left',-.5*head['stereo_baseline_m']),('head_right',.5*head['stereo_baseline_m'])]:
        _camera(body,name,[offset,0.,0.],settings['fov'])
        _geom(body,name.replace('head_','head_camera_')+'_glass','cylinder',pos=[offset,0.,-.022],size=[.009,.0005],rgba='.025 .065 .085 1',profile=glass_profile)
    _camera(body,'robot_head',[-.5*head['stereo_baseline_m'],0.,0.],settings['fov'])
    for index,(start,end) in enumerate(_support_segments(position,rotation,shoulder,head['housing_dimensions_m'][2])):
        _geom(base,f'camera_rig_head_support_{index}','capsule',fromto=[*start,*end],size=[.007 if index==0 else .005],rgba='.26 .28 .30 1',profile=profile)
    for side in ('left','right'):
        wrist, position, rotation, anchor = _wrist_installation(root,side,rig)
        body=ET.SubElement(wrist,'body',name=f'openarm_{side}_wrist_camera_mount',pos=_s(position),quat=_s(_quat(rotation)))
        prefix=f'{side}_wrist_camera_'
        width,height,board_thickness=rig['wrist']['board_dimensions_m']
        _geom(body,prefix+'housing','box',pos=[0.,0.,.017],size=[width/2+.002,height/2+.002,.007],rgba='.12 .14 .16 1',profile=profile)
        _geom(body,prefix+'board','box',pos=[0.,0.,.025],size=[width/2,height/2,board_thickness/2],rgba='.06 .24 .17 1',profile=glass_profile)
        _geom(body,prefix+'lens_barrel','cylinder',pos=[0.,0.,.007],size=[.009,.006],rgba='.08 .09 .10 1',profile=profile)
        _geom(body,prefix+'glass','cylinder',pos=[0.,0.,.0005],size=[.007,.0003],rgba='.015 .09 .13 1',profile=glass_profile)
        _camera(body,f'wrist_{side}',[0.,0.,0.],rig['wrist']['model_pinhole_projection']['vertical_fov_deg'])
        back=position+rotation@np.array([0.,0.,.025])
        _geom(wrist,f'camera_rig_{side}_wrist_support','capsule',fromto=[*anchor,*back],size=[.004],rgba='.38 .40 .43 1',profile=profile)
    return camera_metadata(adapter_document,root=root)


def _update_segment(model,name,start,end):
    gid=int(model.geom(name).id)
    vector=np.asarray(end)-start
    quat=np.empty(4);mujoco.mju_quatZ2Vec(quat,vector)
    model.geom_pos[gid]=(np.asarray(start)+end)/2
    model.geom_quat[gid]=quat
    model.geom_size[gid,1]=np.linalg.norm(vector)/2


def apply_head_settings_to_model(model,settings,shoulder_offset):
    """Keep both optical eyes, the housing and support attached together."""
    values=np.array([float(settings[key]) for key in ('tilt','yaw','fov','forward','height')])
    if not np.isfinite(values).all(): raise ValueError('Camera settings must be finite')
    position,rotation=_head_pose(settings,float(shoulder_offset))
    bid=int(model.body(HEAD_MOUNT_BODY).id)
    model.body_pos[bid]=position;model.body_quat[bid]=_quat(rotation)
    for name in ('head_left','head_right','robot_head'):
        model.cam_fovy[int(model.camera(name).id)]=float(settings['fov'])
    height=load_camera_rig()['head']['housing_dimensions_m'][2]
    for index,(start,end) in enumerate(_support_segments(position,rotation,float(shoulder_offset),height)):
        _update_segment(model,f'camera_rig_head_support_{index}',start,end)


def camera_hidden_geom_names(model,camera_name):
    if camera_name in ('head_left','head_right','robot_head'):prefix='head_camera_'
    elif camera_name in ('wrist_left','wrist_right'):prefix=f'{camera_name.split("_")[1]}_wrist_camera_'
    else:return ()
    return tuple(model.geom(i).name for i in range(model.ngeom) if (model.geom(i).name or '').startswith(prefix))


def hide_camera_housing_in_scene(scene,model,camera_name):
    """Model transparent optical paths without hiding other scene occlusion."""
    hidden={int(model.geom(name).id) for name in camera_hidden_geom_names(model,camera_name)}
    for index in range(scene.ngeom):
        geom=scene.geoms[index]
        if int(geom.objtype)==int(mujoco.mjtObj.mjOBJ_GEOM) and int(geom.objid) in hidden:
            geom.rgba[3]=0.
