# How demonstrations are made: the teacher

The **teacher is a Python control program for OpenArm with an S-hook**. It uses human-reviewed key poses and the actual simulation state to produce valve-operation demonstrations. The key poses are prepared separately; the teacher generates the motion between them.

## The five-step workflow

| Step | What happens |
|---|---|
| **1. Set up** | Position the robot and valve station. |
| **2. Prepare key poses** | Automatic search proposes poses; a person reviews and adjusts them. This is geometry, without force simulation. |
| **3. Run once** | The teacher controls a complete motion in physics simulation while the exam scores it. |
| **4. Review the video** | A person checks that the motion is natural, smooth, and free of anomalies. |
| **5. Generate in bulk** | Vary valve position and resistance, repeat, and collect demonstrations that pass the exam. |

## How it responds to the actual state

During reaching and retraction, the teacher moves smoothly between the saved poses along a route that avoids the pipes.

On the handle, it reads the **actual valve angle every 10 ms** and uses the key poses to calculate the next tool position. It places the target slightly ahead in the turning direction and applies force to keep moving the handle. The valve moves through physical contact.

**Each stage waits for the actual result:** engage before turning; re-engage if contact is lost; reach and hold the target before releasing and retracting.

In the reference settings, the target leads the actual angle by 0.1°–0.5° at startup and at most 6° once moving. Tool targets are converted to joint targets, with directional force ramped up gradually. Motor limits stay unchanged.

## Generating the dataset

The standard setup must pass before bulk generation starts. The A1/A2 batch-generation plan covers four tasks—A1 open, A1 close, A2 open, and A2 close—with eight valve conditions per task and 32 motion-parameter settings per condition: 256 combinations per task, 1,024 total. Four additional conditions per task are reserved for testing.

| Setting | Variation |
|---|---|
| Valve position | Up to ±5 cm independently on each axis |
| Valve resistance | 0.5–1.5× the reference value |
| Force, turning speed, and lead angle | 0.75–1.25× the reference settings |
| Time to ramp up force | 0.5–1.5× the reference setting |

Each run is scored. The **round-1 A1/A2 dataset contains 646 records accepted by the scoring rules used during generation: 619 complete demonstrations and 27 that end after the valve reaches and holds its target**. Their saved states are rendered into head, wrist, and observer views with varied lighting and colours, then stored in LeRobot format.

The **round-2 A2-close experiment used a separate set of 300 complete demonstrations**, each with a different random valve position and resistance: 240 for training and 60 for validation. See the [dataset description](dataset_card.md) for the two sets and their splits.

## Where the code is

Run an OpenArm + S-hook reference demonstration with `rusty-valve teacher --task A2_close --output ./teacher_A2_close` after following the [quick start](../runtime/QUICKSTART.md). The [command-line entry point](../runtime/src/rusty_valve/cli.py) uses the bundled round-2 reference implementation. The [round-1 source](../runtime/src/rusty_valve/versions/round1/valve_bench/) is also bundled for the frozen ACT evaluation.

| Purpose | File |
|---|---|
| **Run one demonstration** | [`v2_episode.py`](../runtime/src/rusty_valve/versions/round2/valve_bench/v2_episode.py), function `run_episode` |
| **Follow reviewed key poses** | [`reviewed_sequence_controller.py`](../runtime/src/rusty_valve/versions/round2/valve_bench/reviewed_sequence_controller.py) and [bundled reference inputs](../runtime/src/rusty_valve/versions/round2/valve_bench/resources/teacher/) |
| Motion, inverse kinematics, and force | [`g2_episode.py`](../runtime/src/rusty_valve/versions/round2/valve_bench/g2_episode.py) |
| Switch phases on measured state | [`state_g1_controller.py`](../runtime/src/rusty_valve/versions/round2/valve_bench/state_g1_controller.py) |
| Contact range and re-engagement | [`range_g1_controller.py`](../runtime/src/rusty_valve/versions/round2/valve_bench/range_g1_controller.py) |
| Engagement and release checks | [`execution_state_observation.py`](../runtime/src/rusty_valve/versions/round2/valve_bench/execution_state_observation.py), [`execution_phase_gate.py`](../runtime/src/rusty_valve/versions/round2/valve_bench/execution_phase_gate.py) |
| Record physical evidence and report the original run score | [`scored_env.py`](../runtime/src/rusty_valve/versions/round2/valve_bench/scored_env.py) |
| Score the recorded evidence under Public Exam v1 | [`public_exam_v1.py`](../runtime/src/rusty_valve/public_exam_v1.py) |

The bulk-generation, pose-search and editing, and dataset-building tools used in the original workflow are outside this release package.

A task needs a station file, a key-pose file, motion parameters, and a robot adapter. The four A1/A2 tasks use 15–17 key poses across eight named roles.

## Using another robot or end-effector

The stations, public exam, and overall method can be reused. **Another robot needs its own controller.** Supply its key poses, kinematics, and control interface, plus tool-specific contact targets and checks for engagement and release. A gripper or hand also needs finger-closing, holding, and release commands.

The contact logic uses the S-hook's recess and working faces; the key poses and pose editor use OpenArm.

[Level 1](../levels/01-handle-ball-valve/) · [Dataset](dataset_card.md)

## License

This guide is **CC BY 4.0**. Our controller code and configuration use **Apache-2.0**. Third-party material retains its original terms. See [license scope](../LICENSING.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
