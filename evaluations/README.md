# Original ACT exams

Round 1 has 16 original cases and round 2 has 36. The [frozen Public Exam v1 table](public_results_v1.csv) and the per-case `result.json`, `original_score_v25.json`, and `public_score_v1.json` files are included here. These are actual policy attempts, including failures.

The raw joint and simulator states (`physical_states.npz`), solved-step evidence (`physical_step_evidence.jsonl.gz`), and 20 Hz original-speed three-camera recordings are separate materials. The `record` column in the table points to the raw evidence paths in the complete material root. Download the [original exam records](https://huggingface.co/datasets/rusty-valve/raw_exam_records_v0) and follow the [placement instructions](../runtime/README.md#download-and-place-materials) to put the case folders at their expected paths.

The project-generated solved-step evidence, state arrays, and original-speed recordings from all 52 formal exams use CC BY 4.0. Credit Rusty Valve contributors. Third-party assets retain their original notices. See [license scope](../LICENSING.md).

Round1 public pass counts: seen positions 1/4, independent positions 0/4, test appearances 0/8. Round2: 0/36. Progress grade, full hand return and public pass/fail are distinct. Installation smoke runs are excluded from this table. Original legacy diagnostics remain available beside the public results.

All 52 runs attempt to close the valve at station A2. The original records retain internal Chinese labels. The original scores use the legacy A1 closing label and the procedure ID `A1-v2.5`.

The legacy path `round2/seen_valve_positions/` contains four tests in round 2 at valve positions used in round 1 training. Round 2 used a separate set of training positions that excludes those four. The `unseen_appearance/` group changes colors and lighting at unseen valve positions. A suffix such as `train_episode_0161` identifies the appearance preset. The exam plan’s `source` points to the fixture for the scene and starting state. Use that fixture and the round’s training split to check whether the position was used in training.

The ordered [round 1](round1/examination_plan.json) and [round 2](round2/examination_plan.json) exam plans reference the files under [Fixed exam scenes and starting states](../fixtures/). To run one case again, use the [ACT commands](../runtime/README.md#read-data-and-run-act) with the corresponding checkpoint and dataset. No tests are used for tuning.
