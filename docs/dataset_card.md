# Rusty Valve: Handle Ball Valve Demonstrations

OpenArm uses an S-shaped hook/push tool to open and close a DN25 ball valve in simulation. The first batch contains **646 demonstrations: 619 include complete retraction; 27 end after the valve reaches and holds its target.** A separate coverage batch contains **300 A2-closing demonstrations, all with complete retraction**. Both batches were selected using the experiment's scoring rules, which included S-tool working-face checks.

## First batch: tasks and splits

| Task | Train | Validation | Total |
|---|---:|---:|---:|
| A1 open | 92 | 32 | 124 |
| A1 close | 102 | 31 | 133 |
| A2 open | 103 | 30 | 133 |
| A2 close | 224 | 32 | 256 |
| **Total** | **521** | **125** | **646** |

Train and validation use separate valve-position groups. The 16 held-out test conditions and two test-only appearance settings are kept separate. Failed runs are retained separately from these demonstrations.

## Coverage experiment: A2 closing

The second ACT round uses a separate batch of **300 demonstrations at 300 different valve positions**, with a new random position and resistance for each demonstration. All 300 include complete retraction. The original held-out test conditions are excluded from training.

| Split | Demonstrations | Frames |
|---|---:|---:|
| Train | 240 | 985,095 |
| Validation | 60 | 238,209 |
| **Total** | **300** | **1,223,304** |

This batch is separate from the first 646-demonstration batch. See the [ACT baseline](act_baseline.md) for the two training rounds and their results.

## What each demonstration contains

| Content | Description |
|---|---|
| Three camera views | Head, left wrist, and observer; 256 × 256 RGB |
| Robot state and action | Seven left-arm joint angles and recorded motor targets |
| Valve labels | Opening angle, open/closed/intermediate state, and handle pose |
| Motor torque | Reconstructed from saved simulation states and controls |

ACT uses **head and wrist images plus joint angles**. See [Level 1](../levels/01-handle-ball-valve/) for the task and [the teacher guide](teacher.md) for how demonstrations are generated.

## Format and timing

LeRobot v3.0 format, written with LeRobot 0.6.1: Parquet records and H.264 videos. Each batch has its own `datasets/<task>/train` and `datasets/<task>/validation` folders.

Both batches record state and action at 100 Hz and camera images at 20 Hz:

| Batch | State and action | Camera capture | Availability |
|---|---|---|---|
| First batch: 646 demonstrations; the A2-closing subset is used in ACT round 1 | 100 Hz | 20 Hz | Collected |
| Second batch: 300 A2-closing demonstrations, used in ACT round 2 | 100 Hz | 20 Hz | Collected |

Each action follows its paired observation, and the latest image is repeated between camera captures. All data comes from simulation; colors and lighting vary between demonstrations.

<details>
<summary>Field reference</summary>

| Field | Meaning |
|---|---|
| `observation.images.head`, `.wrist`, `.observer` | RGB video, 256 × 256 × 3 |
| `observation.state` | Seven left-arm joint angles, radians |
| `action` | Seven upcoming left-arm motor targets, radians |
| `observation.effort_reconstructed` | Seven reconstructed motor torques, N·m; not sensor measurements |
| `label.valve_opening_deg`, `label.valve_state` | Degrees; 0 = closed, 1 = open, 2 = intermediate |
| `label.handle_pose` | Position x, y, z in meters; orientation qw, qx, qy, qz |
| `label.partial_operation_only` | 1 = no complete retraction included; 0 = full motion |
| `label.source_time_s`, `label.camera_source_time_s`, `timestamp` | Source, camera-capture, and dataset times in seconds |
| `frame_index`, `episode_index`, `index`, `task_index` | Frame, episode, record, and task indices |

The seven joint entries follow `openarm_left_joint1` through `openarm_left_joint7`. Continuous values use float32; valve-state codes, fragment flags, and indices use int64. Valve labels, observer images, and torque values are not ACT inputs. Source manifests and episode maps retain provenance and split assignments.

</details>

## Reproduction status

Both demonstration datasets (646 and 300 episodes), the checkpoints from both rounds, and all 52 test records are preserved in the complete local materials. This repository includes the [dataset manifests and episode maps](../datasets/), [fixed exam inputs](../fixtures/), and [lightweight exam results](../evaluations/). The full LeRobot splits, checkpoint files, and raw exam records are separate materials.

All ten LeRobot splits passed a first-sample read in a clean macOS CPU environment. See the [validation notes](../runtime/VALIDATION.md) and the [data-check command](../runtime/README.md#read-data-and-run-act), which takes the complete local material root.

## Download, license, and citation

Download the [646-demonstration four-task dataset](https://huggingface.co/datasets/rusty-valve/a1_a2_open_close_646) or the [300-position A2-closing dataset](https://huggingface.co/datasets/rusty-valve/a2_close_300_positions). Follow the [download and placement instructions](../runtime/README.md#download-and-place-materials) to put each dataset in the folder the runtime expects.

The demonstration data and this dataset card are licensed under [CC BY 4.0](../LICENSES/CC-BY-4.0.txt). Attribute them to **Rusty Valve contributors**. Third-party assets retain their original licenses; see [Licensing](../LICENSING.md) and [Third-party notices](../THIRD_PARTY_NOTICES.md).

```bibtex
@misc{rustyvalve2026demonstrations,
  author = {{Rusty Valve contributors}},
  title = {{Rusty Valve: Handle Ball Valve Demonstrations}},
  year = {2026}
}
```

For dataset questions, use this repository’s GitHub Issues tab.
