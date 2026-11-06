# Rusty Valve v0 runtime

A frozen OpenArm + S-hook reference implementation, two ACT inference runtimes, and public exam v1. A1/A2 each provide open/close references; B1/B2/C1/C2 are robot-free challenges without reference controllers. This package contains no RL baseline or slow-opening task.

Code, simulation models, and ACT weights use Apache-2.0. Demonstration datasets, technical documentation, and project-generated exam evidence, state arrays, and recordings use CC BY 4.0. Third-party assets retain their original notices. See `LICENSE`, `LICENSES/`, and `THIRD_PARTY_NOTICES.md`.

## Material layout

This repository contains the installable runtime, dataset descriptions and episode maps, training metadata, exam plans and results, and fixed exam inputs. For data reading and ACT inference, `--materials` points to a complete local material root. It can be separate from the repository and must have this layout:

```
datasets/four_tasks_646/           # full LeRobot dataset, 521 train + 125 validation
datasets/a2_close_coverage_300/    # full LeRobot dataset, 240 train + 60 validation
models/round1/checkpoint/          # complete first-round ACT checkpoint
models/round2/checkpoint/          # complete second-round ACT checkpoint
evaluations/round1/examination_plan.json
evaluations/round2/examination_plan.json
fixtures/                         # fixed inputs referenced by the exam plans
```

Copy `fixtures/` and the two `examination_plan.json` files from this repository into the material root at the paths above. The [exam results page](../evaluations/) describes the archived records.

## Download and place materials

The five v0 downloads are available in separate Hugging Face repositories in the `rusty-valve` organization. Each repository preserves its folder layout.

| Material | Hugging Face repository | Type | Placement under `materials_root/` | License |
|---|---|---|---|---|
| 646 A1/A2 opening and closing demonstrations | [rusty-valve/a1_a2_open_close_646](https://huggingface.co/datasets/rusty-valve/a1_a2_open_close_646) | Dataset | `datasets/four_tasks_646/` | CC BY 4.0 |
| A2 closing at 300 valve positions | [rusty-valve/a2_close_300_positions](https://huggingface.co/datasets/rusty-valve/a2_close_300_positions) | Dataset | `datasets/a2_close_coverage_300/` | CC BY 4.0 |
| ACT round 1 checkpoint | [rusty-valve/a2_close_act_round1](https://huggingface.co/rusty-valve/a2_close_act_round1) | Model | `models/round1/` | Apache-2.0 |
| ACT round 2 checkpoint | [rusty-valve/a2_close_act_round2](https://huggingface.co/rusty-valve/a2_close_act_round2) | Model | `models/round2/` | Apache-2.0 |
| 52 original ACT exam records | [rusty-valve/raw_exam_records_v0](https://huggingface.co/datasets/rusty-valve/raw_exam_records_v0) | Dataset | Merge its `evaluations/` into `evaluations/` | CC BY 4.0 for project-generated records |

Use [`hf download`](https://huggingface.co/docs/huggingface_hub/en/guides/download#download-from-the-cli) with the type and destination above. This example downloads the first dataset into its runtime directory.

```sh
materials_root='/path/to/complete-v0-materials'
hf download rusty-valve/a1_a2_open_close_646 --repo-type dataset --local-dir "$materials_root/datasets/four_tasks_646"
```

Keep each dataset's `manifest.json` beside its inner `datasets/` directory. Download each model repository to its round directory, so its bundled `checkpoint/` ends up at `models/round1/checkpoint/` or `models/round2/checkpoint/`.

Download the raw exam archive into its own local folder first. Merge the contents of its `evaluations/` directory into `materials_root/evaluations/`. Keep both `examination_plan.json` files and the nested case folders at their existing paths. Keep the archive's README and license notices with the download. Use a case's solved-step record with the scoring command below. Use the matching dataset, checkpoint, exam plan, and fixed scenes for an ACT rerun. See [license scope](../LICENSING.md).

## Installation and commands

Use Python 3.12. From the repository root, install into a fresh environment and run the core commands:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install ./runtime

rusty-valve scene --station B1 --output ./B1.xml
rusty-valve teacher --task A2_close --output ./teacher_A2_close
rusty-valve score ./teacher_A2_close/physical_step_evidence.jsonl.gz --output ./public_score.json
```

The `teacher` command also writes `public_score_v1.json` in its output directory. Use fresh output paths. Other reference tasks are `A1_open`, `A1_close`, and `A2_open`. `scene` accepts all six station IDs. After installation, the commands can run from any directory.

## Read data and run ACT

Install the optional dependencies from the repository root. Set `materials_root` to the [complete local material root](#material-layout).

```sh
python -m pip install './runtime[act]'
materials_root='/path/to/complete-v0-materials'

rusty-valve data-check --materials "$materials_root" --output ./data_readback.json
rusty-valve act --materials "$materials_root" --round round1 --case-index 0 --output ./act_round1_case0
rusty-valve act --materials "$materials_root" --round round2 --case-index 0 --output ./act_round2_case0
```

Each ACT round needs its own checkpoint, plan and fixed inputs. During inference, LeRobot reads the matching A2-close training split for control rate and normalization statistics, and the runtime reads the camera capture rate from the dataset manifest. `case-index` follows the zero-based order in the selected round's exam plan. Commands execute locally on CPU. ACT inference may take much longer than simulation time and needs a render-capable local OpenGL context. A headless Linux installation needs a suitable MuJoCo rendering backend.

The original models were trained on Linux with Torch 2.7.1+cu118, LeRobot 0.6.1, MuJoCo 3.16.0, NumPy 2.2.6 and SciPy 1.18.1. The macOS installation uses the cached Torch 2.11.0 CPU / torchvision 0.26.0 distributions with the same saved weights. Download attempts for Torch 2.7.1 and its dependencies timed out; this is a macOS compatibility check, not a validation of the original Linux/CUDA installation. No training setting was changed. Cross-platform floating-point differences are possible; the local installation checks do not replace the frozen published results. See `VALIDATION.md` for actual tested versions and results, including any remaining limitations.

## Data

Each dataset preserves its manifest, per-task episode map and the original LeRobot layout:

```
datasets/<task>/{train,validation}/
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/...
  data/chunk-.../file-....parquet
  videos/observation.images.{head,wrist,observer}/chunk-.../file-....mp4
```

Use `LeRobotDataset(repo_id='primates-valve/v0_A2_close_train', root=absolute_split_path, video_backend='pyav', download_videos=False)` to read a split without a server. The example `repo_id` is a local dataset label and `root` selects the files. The 646-demo set has 619 full sequences and 27 operation/hold-only sequences. The 300-demo set contains full sequences at distinct positions. All original train/validation memberships and labels are retained. Controls and joint states are recorded at 100 Hz. Cameras capture at 20 Hz, retaining the latest image between captures. Image size is 256 by 256 with the original aspect-ratio padding. The datasets contain three camera views. The two ACT checkpoints use head and wrist plus seven active-arm joint angles, predicting seven absolute joint targets in radians. Original independent tests were excluded from training. Manifest source locations use IDs in the [source index](../provenance/recorded_sources.json).

## Robot/task API

```python
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
```

The underlying interface is the frozen `H0Env`. Both `reset()` and `step()` return a dictionary of time, valve angle, contacts and cumulative status. Robot joint angles are available through `env.data.qpos[env.robot_qpos_indices]`. Public pass/fail comes from the separate public exam scorer, not the legacy `goal_hold_reached` field. The low-level constructor alone does not select the reviewed rest pose; use the explicit initial state above. Commands contain all configured motor position targets in radians; ACT fills only the active arm and preserves the idle-arm commands. Robot adapters are under `rusty_valve/versions/<round>/valve_bench/configs/robots/`. A different robot needs its own adapter, actuator order, limits and contact categories. The supplied reference controller and reviewed key poses are OpenArm/S-hook specific; changing the tool does not automatically produce a reference solution.

## Scores and evidence

Public v1 requires the target (0 degrees closed or 90 degrees open) within 5 degrees for the final continuous second, no robot-to-nonhandle-environment contact above 20 N, and a 90-second deadline. Any robot part may contact the handle; robot self-contact is recorded only. The valve must move through native physical contact. Tool working-face checks and full arm return are recorded separately and are not public pass criteria. The legacy v2.5 control-loop termination conditions are retained to reproduce the frozen policies; the public score is computed separately from their full physical evidence.

The score CLI reads `valve_solved_step_evidence_v1` gzip JSONL: one header with `variant`, `physics_dt_s`, `contact_evidence_complete`, `data_kind`; one sample per solved physics step with increasing `time_s`, `opening_deg` and complete `contacts`; and the matching completion footer. Contact entries include `kind` (`handle`, `self`, `fixture`, `environment`) and `force_N` as each individual contact-force magnitude. The footer must retain sample counts/timestamps, completion flags and physical-runtime provenance. Sparse snapshots cannot prove absence of collisions. Results are `pass`, `fail`, `unverified` or `invalid`; gaps stay explicit. For another simulator, use `PublicExamScorer.observe()` with the same fields and supply truthful coverage/physical-validity declarations to `finish()`.

## Portability changes

Only paths, packaging and entry points were adapted. Round-specific controller, physics, camera settings, action cadence, temporal ensemble coefficient (0.01), normalization statistics, weights, labels and data splits remain frozen. The full ACT checkpoint supplies the visual backbone; its constructor's optional pretrained-backbone download is disabled during loading, then all saved weights are loaded. The bundled `h3_act_train.py` only reads local data for ACT inference. The inactive historical compatibility fields remain private implementation details; no slow-opening command is exposed. See [portability notes](PORTABILITY.md) for details.

The frozen reference code and metadata retain some internal Chinese comments and historical paths, which do not affect runtime operation.
