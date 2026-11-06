import json
from importlib.resources import files
from rusty_valve.api import make_env, NativeRenderer, camera_option, render_three_views

# Use the bundled, reviewed placement and initial rest pose.
inputs = files('rusty_valve.versions.round2.valve_bench').joinpath('resources/teacher/A2_close')
station = json.loads(inputs.joinpath('workstation.json').read_text())
rest = json.loads(inputs.joinpath('rest.json').read_text())
angles = {name: value for arm in rest['arms'].values()
          for name, value in zip(arm['joint_names'], arm['joint_angles_rad'])}
env = make_env('A2', 'close', robot_adapter='openarm_s_hook_left_mount180_v1',
               placement=station['placement'])
observation = env.reset(seed=1000,
    initial_robot_qpos=[angles[name] for name in env.robot_joint_names])
# env.actuator_ids defines the full native motor-command order.
command = env.data.ctrl[env.actuator_ids].copy()
observation = env.step(command)
renderer = NativeRenderer(env.model)
images = render_three_views(renderer, env.model, env.data, camera_option())
renderer.close()
env.close()
