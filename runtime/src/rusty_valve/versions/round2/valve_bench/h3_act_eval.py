"""Independent ACT closed-loop exams using frozen physical scenes and v2.5 scoring."""


from __future__ import annotations


import argparse


from concurrent.futures import ProcessPoolExecutor, as_completed


import copy


import json


import multiprocessing


import os


from pathlib import Path


import shutil


import time


import imageio.v2 as imageio


import mujoco


import numpy as np


from PIL import Image, ImageDraw


from .h2_data import (ROOT, OUTPUT, TASKS, CAMERAS, FPS, JOINTS, NativeRenderer,
    WRIST_OPTICAL_REAR_SHIFT_M, TEST_APPEARANCES, apply_appearance, make_model, load, save,
    camera_option, render_three_views)


from .g2_review import replay_kinematics


from .h0_env import H0Env


from .h0_robot import robot_geom_groups


from .h0_scene import environment_geom_groups


from .scored_env import ScoredValveEnv


from .offline_face_review import review_log


EXAMS = ROOT / '08_ACT基线/考试条件'


def exam_deadline_reached(budget_file):
    if budget_file is None:
        return False
    deadline = load(budget_file).get('evaluation_stop_unix')
    return deadline is not None and time.time() >= deadline


def recorded_environment(source, control_fps=FPS):
    """Reuse H0's native actuator/passive-friction loop with the exact recorded model."""
    condition = load(source / 'condition.json')
    station, adapter = load(source / 'station.json'), load(source / 'robot_adapter.json')
    asset_root = Path(os.environ.get('PRIMATES_VALVE_ASSET_ROOT', str(Path(__file__).parent / 'assets')))
    adapter['model_file'] = str(asset_root / adapter['model_file'].split('assets/', 1)[-1])
    env = H0Env(station, adapter, task=condition['task'].split('_')[1], control_dt=1 / control_fps, max_time_s=90.)
    exact = make_model(source)
    if ([env.model.joint(i).name for i in range(env.model.njnt)] !=
            [exact.joint(i).name for i in range(exact.njnt)]):
        raise ValueError('Frozen scene joint order differs from the physical adapter')
    if ([env.model.actuator(i).name for i in range(env.model.nu)] !=
            [exact.actuator(i).name for i in range(exact.nu)]):
        raise ValueError('Frozen scene actuator order differs from the physical adapter')
    if not np.array_equal(env.model.actuator_forcerange, exact.actuator_forcerange):
        raise ValueError('Frozen motor limits differ from the physical adapter')
    env.model, env.data = exact, mujoco.MjData(exact)
    env.robot_groups = robot_geom_groups(exact, env.robot_adapter)
    env.environment_groups = environment_geom_groups(exact, env.station_spec)
    env.robot_geom_ids = set(env.robot_groups['robot_geom_ids'])
    env.active_tool_geom_ids = set(env.robot_groups['operation_end'])
    env.idle_geom_ids = set(env.robot_groups['idle_hand'])
    env.valve_geom_ids = set(env.environment_groups['handle_geom_ids'])
    env.fixture_geom_ids = set(env.environment_groups['fixture_geom_ids'])
    env.active_hook_body_id = int(exact.body(adapter['control']['operation_body_name']).id)
    env.valve_joint_id = int(exact.joint(env.station_spec.valve_joint_name).id)
    env.valve_body_id = int(exact.body(env.station_spec.valve_moving_body_name).id)
    env.valve_mount_body_id = int(exact.body(env.station_spec.valve_mount_body_name).id)
    env.valve_installation_body_id = int(exact.body(env.station_spec.installation_body_name).id)
    env.resistance.scale *= condition['resistance_scale']
    with np.load(source / 'initial_state.npz') as initial:
        rest = initial['qpos'][env.robot_qpos_indices].copy()
    env.reset(seed=condition['seed'], initial_robot_qpos=rest)
    exact.cam_pos[exact.camera('wrist_left').id, 2] += WRIST_OPTICAL_REAR_SHIFT_M
    return env, rest


def render_views(renderer, env, views):
    images = render_three_views(renderer, env.model, env.data, env._act_image_option)
    return {view: images[view] for view in views}


def run_exam(checkpoint, source, profile, output, budget_file, *, device='cuda'):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from .h3_act_train import open_dataset

    output.mkdir(parents=True, exist_ok=False)
    condition = load(source / 'condition.json')
    from lerobot.policies.act.configuration_act import ACTConfig
    config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
    config.pretrained_backbone_weights = None  # Full checkpoint supplies all backbone weights.
    config.device = device
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True).to(device)
    policy.config.device = device
    dataset = open_dataset(condition['task'], policy.config)
    control_fps = float(dataset.meta.fps)
    camera_fps = float(load(OUTPUT / 'manifest.json').get('camera_fps', control_fps))
    camera_stride = round(control_fps / camera_fps)
    if camera_stride < 1 or abs(camera_stride * camera_fps - control_fps) > 1e-7:
        raise ValueError('Camera capture rate must divide the policy control rate')
    pre, post = make_act_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    policy.eval(); policy.reset()
    native, rest = recorded_environment(source, control_fps=control_fps)
    apply_appearance(native.model, profile)
    native._act_image_option = camera_option()
    env = ScoredValveEnv(native, native.variant, source='ACT closed-loop independent exam', procedure_version='A1-v2.5')
    env.start_evidence_recording(output / 'physical_step_evidence.jsonl.gz')
    renderer = NativeRenderer(native.model)
    writer = imageio.get_writer(str(output / 'three_views_original_speed.mp4'), fps=camera_fps,
                                codec='libx264', quality=7, macro_block_size=1,
                                ffmpeg_params=['-preset', 'fast'])
    begin = time.time(); samples = []; rest_since = None
    reason = 'observation_cutoff'; complete = False; full_return = False
    error = None
    views_needed = ['head'] + (['wrist'] if 'observation.images.wrist' in policy.config.input_features else [])
    images = None
    try:
        while native.data.time < 90. - 1e-9:
            if exam_deadline_reached(budget_file):
                reason = 'gpu_budget_cutoff'; break
            if images is None or len(samples) % camera_stride == 0:
                images = render_views(renderer, native, CAMERAS)
                canvas = Image.new('RGB', (768, 280), 'white')
                for column, view in enumerate(CAMERAS):
                    canvas.paste(Image.fromarray(images[view]), (column * 256, 24))
                    ImageDraw.Draw(canvas).text((column * 256 + 6, 6),
                        f'{condition["task"]} {native.data.time:.2f}s {view}', fill='black')
                writer.append_data(np.asarray(canvas))
            observation = {'observation.state': torch.from_numpy(native.data.qpos[native.active_qpos_indices].astype(np.float32))}
            for view in views_needed:
                observation[f'observation.images.{view}'] = torch.from_numpy(images[view].copy()).permute(2, 0, 1).float() / 255.
            with torch.inference_mode():
                action = post(policy.select_action(pre(observation))).reshape(-1).cpu().numpy()
            if action.shape != (7,) or not np.isfinite(action).all(): raise ValueError('ACT returned invalid joint targets')
            command = native.data.ctrl[native.actuator_ids].copy()
            for i, aid in enumerate(native.active_actuator_ids):
                command[np.flatnonzero(native.actuator_ids == aid)[0]] = action[i]
            samples.append(dict(time_s=float(native.data.time), qpos=native.data.qpos.copy(),
                                qvel=native.data.qvel.copy(), ctrl=command.copy()))
            env.step(command)
            if env.scorer.counts['hard_contact']:
                reason = 'hard_collision'; complete = True; break
            max_error = float(np.max(abs(native.data.qpos[native.robot_qpos_indices] - rest)))
            detached = not any(c['kind'] == 'handle' for c in native.current_contacts)
            if max_error <= np.radians(3.) and detached:
                if rest_since is None: rest_since = float(native.data.time)
            else: rest_since = None
            if rest_since is not None and native.data.time - rest_since >= 1.:
                if env.score()['rules']['final_target_hold']['status'] == 'pass':
                    reason = 'goal_held_and_shared_rest'; complete = True; full_return = True; break
        else: complete = True
        if native.data.time >= 90. - 1e-9: complete = True
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        reason = 'runtime_error'
    finally:
        writer.close(); renderer.close()
        env.finish_evidence_recording(execution_complete=complete)
        original_score = env.score()
        env.close()
    save(output / 'original_score_v25.json', original_score)
    if samples:
        np.savez_compressed(output / 'physical_states.npz',
                            **{k: np.asarray([x[k] for x in samples]) for k in samples[0]})
    corrected = review_log(output / 'physical_step_evidence.jsonl.gz') if complete else None
    result = dict(task=condition['task'], condition=condition, appearance=profile,
                  camera_inputs='head_and_wrist' if len(views_needed) == 2 else 'head',
                  checkpoint=str(checkpoint), stop_reason=reason, error=error,
                  execution_complete=complete, full_return=full_return,
                  simulation_s=float(native.data.time), wall_s=round(time.time() - begin, 2),
                  control_fps=control_fps, camera_fps=camera_fps, video_fps=camera_fps,
                  score=corrected, passed=corrected['passed'] if corrected else None,
                  data_kind='new_closed_loop_simulation', teacher_controller_used=False,
                  video='three_views_original_speed.mp4')
    save(output / 'result.json', result)
    return {k: v for k, v in result.items() if k not in ('score', 'condition', 'appearance')}

