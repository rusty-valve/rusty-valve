# Quick start

Run from the repository root with Python 3.12. These commands use the files in this repository.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install ./runtime

rusty-valve scene --station B1 --output ./B1.xml
rusty-valve teacher --task A2_close --output ./teacher_A2_close
rusty-valve score ./teacher_A2_close/physical_step_evidence.jsonl.gz --output ./public_score.json
```

Use fresh output paths for each run. The reference command writes `public_score_v1.json` alongside the recording. The separate `score` command shows how to score an existing recording.

For [dataset reading and ACT inference](README.md#read-data-and-run-act), install the optional dependencies and point `--materials` at a complete local material root with the datasets, ACT checkpoints, exam plans, and fixed inputs. See [downloads and placement](README.md#download-and-place-materials) for the v0 repositories and their local paths. See the [clean macOS CPU validation](VALIDATION.md) and [frozen benchmark results](../evaluations/public_results_v1.csv).
