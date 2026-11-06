"""Render actual G2 saved states at original speed; never runs new physics.

The fixed cameras follow the approved G1 settings. Rendering and a saved exam
result are separate: an image or video never changes an episode's grade.
"""
from __future__ import annotations

import argparse
import copy
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETUP = ROOT / 'valve_bench/configs/g2_approved_auto.json'
CONTROLLER_VERSION = 'approved_g1_auto_v3'


def approved_cameras(root, station, setup):
    """Copy frozen G1 camera nodes without aiming at the randomized valve."""
    cameras = {}
    for view, name, expected_parent in (
            ('head', 'robot_head', 'openarm_body_link0'),
            ('side', 'station_observer', 'worldbody')):
        frozen = setup['stations'][station]['cameras'][view]
        camera = ET.fromstring(frozen['xml'])
        parent_name = frozen['parent_body']
        if parent_name != expected_parent or camera.tag != 'camera' or camera.get('name') != name:
            raise ValueError(f'Unexpected approved {view} camera node or parent')
        parent = (root.find('worldbody') if parent_name == 'worldbody'
                  else root.find(f".//body[@name='{parent_name}']"))
        if parent is None:
            raise ValueError(f'Recorded scene lacks approved camera parent {parent_name}')
        for container in root.iter():
            for existing in list(container):
                if existing.tag == 'camera' and existing.get('name') == name:
                    container.remove(existing)
        parent.append(camera)
        cameras[view] = copy.deepcopy(frozen['metadata'])
        cameras[view]['frozen_xml'] = frozen['xml']
        cameras[view]['parent_body'] = parent_name
    return cameras


def replay_kinematics(model, data):
    """Update geometry/camera transforms only; do not recompute contact physics."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)


def render_episode(source, output, *, fps=20., width=1280, height=720, video=True,
                   approved_setup=DEFAULT_SETUP):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new render output directory')
    result = json.loads((source/'result.json').read_text())
    approved_setup = Path(approved_setup).resolve()
    setup = json.loads(approved_setup.read_text())
    if not setup.get('user_approved') or not setup.get('frozen_for_G2'):
        raise ValueError('Camera setup must be user-approved and frozen for G2')
    version = result.get('controller_version', result.get('parameters', {}).get('controller_version'))
    if not version or version != setup.get('controller_version'):
        raise ValueError('Recorded episode and approved setup use different controller versions')
    root = ET.fromstring((source/'scene.xml').read_text())
    cameras = approved_cameras(root, result['station'], setup)
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth=max(width,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(height,model.vis.global_.offheight)
    with np.load(source/'physical_states.npz') as saved:
        times, positions = saved['time_s'].copy(), saved['qpos'].copy()
    if positions.shape!=(len(times),model.nq) or len(times)<2 or not np.all(np.diff(times)>0):
        raise ValueError('Saved physical states do not match the recorded scene')
    if not np.isfinite(positions).all() or not np.isfinite(times).all():
        raise ValueError('Nonfinite recorded states')
    if fps<=0 or width<=0 or height<=0:
        raise ValueError('fps and image dimensions must be positive')
    duration=float(times[-1]-times[0])
    schedule=times[0]+np.arange(max(2,int(np.ceil(duration*fps))))/fps
    schedule=np.minimum(schedule,times[-1]);schedule[-1]=times[-1]
    ids=np.clip(np.searchsorted(times,schedule),0,len(times)-1)
    left=np.maximum(ids-1,0)
    ids=np.where(abs(times[left]-schedule)<abs(times[ids]-schedule),left,ids)
    output.mkdir(parents=True)
    (output/'render_scene.xml').write_text(xml)
    option=mujoco.MjvOption();option.geomgroup[3]=0;option.sitegroup[:]=0
    for view, camera in [('head','robot_head'),('side','station_observer')]:
        writer=imageio.get_writer(str(output/f'{view}_original_speed.mp4'),fps=fps,
            codec='libx264',quality=7,macro_block_size=1) if video else None
        try:
            with mujoco.Renderer(model,height,width) as renderer:
                for index,i in enumerate(ids):
                    if not video and index not in (0,len(ids)//2,len(ids)-1):
                        continue
                    data.qpos[:]=positions[i]
                    data.time=float(times[i])
                    replay_kinematics(model,data)
                    if index == 0:
                        camera_id=model.camera(camera).id
                        cameras[view]['actual_world_position_m']=data.cam_xpos[camera_id].tolist()
                        cameras[view]['actual_world_rotation_matrix']=data.cam_xmat[camera_id].reshape(3,3).tolist()
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    pixels=renderer.render()
                    if writer is not None:writer.append_data(pixels)
                    if index in (0,len(ids)//2,len(ids)-1):
                        Image.fromarray(pixels).save(output/f'{view}_{index:05d}.png')
        finally:
            if writer is not None:writer.close()
    meta=dict(data_kind='camera_replay_of_saved_contact_physics',source=str(source),
        simulation_location='cloud',controller_version=version,
        approved_setup=str(approved_setup),
        score_passed=result['score']['passed'],stop_reason=result.get('stop_reason'),
        simulation_duration_s=duration,video_duration_s=len(ids)/fps if video else None,
        fps=fps,image_size_px=[width,height],cameras=cameras,
        original_speed=True,physics_integrated_during_render=False,render_physics_steps=0,
        camera_policy='Frozen approved G1 nodes; head attached to fixed torso, observer fixed in world; no valve tracking',
        note='Uniform simulation-time schedule; encoding rounds duration by at most one frame.')
    (output/'render_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    if video:
        (output/'原速查看.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>G2 云端受力仿真回放</title><style>body{font:17px/1.7 -apple-system,sans-serif;background:#edf1f4;color:#20313b;margin:24px}main{max-width:1300px;margin:auto}.views{display:flex;gap:12px}video{width:49%}</style><main>'''
            +f'<h1>{html.escape(result["station"])} · {html.escape(result["variant"])} · '+('通过' if result['score']['passed'] is True else '未通过' if result['score']['passed'] is False else '受力面待核查')+'</h1>'
            +'<p>云端受力仿真的实际保存状态，按仿真时间原速播放。回放不重新计算动作；头部与第三视角沿用已确认 G1 镜头，不随随机阀门位置重新取景。</p>'
            +'<div class="views"><video controls src="head_original_speed.mp4"></video><video controls src="side_original_speed.mp4"></video></div></main></html>')
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=float,default=20.);p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=720);p.add_argument('--images-only',action='store_true')
    p.add_argument('--approved-setup',type=Path,default=DEFAULT_SETUP)
    a=p.parse_args()
    print(json.dumps(render_episode(a.source,a.output,fps=a.fps,width=a.width,height=a.height,
        video=not a.images_only,approved_setup=a.approved_setup),ensure_ascii=False))
