# Local installation validation

Completed 2026-11-06 (Asia/Shanghai). A fresh Python 3.12 virtual environment was created on the external disk. Dependencies were installed from cached distribution wheels, then this runtime was built and installed as a normal non-editable wheel. Old project modules and old site-packages were not injected. Checks ran outside the project with isolated Python (`-I`), offline Hugging Face settings and CPU inference. Installation package caches do not form part of the delivered runtime.

| Check | Actual outcome |
|---|---|
| Six standalone stations A1, A2, B1, B2, C1, C2 | All six loaded in MuJoCo |
| Four frozen OpenArm/S-hook teacher references | All four completed; public v1 pass |
| A1 open / close final opening | 88.2564° / 3.3487° |
| A2 open / close final opening | 86.5603° / 2.2728° |
| Official LeRobot reader | First sample of all ten train/validation splits loaded; 646 + 300 episodes accounted for |
| Round1 fixed original case 0 | Ran to completion, public pass, final opening 4.5452° |
| Round2 fixed original case 0 | Ran to 90 s, public fail, final opening 90.0001°; retained unchanged |
| ACT recordings | Both three-view videos are 20 fps; 713 and 1800 frames |
| Public scoring command | Existing solved-step evidence read and scored successfully |
| Documented robot API | Reset, 14-target step and three 256×256 RGB views ran successfully |
| Dependency / installed command check | `pip check` clean; installed `rusty-valve --help` works |

These are installation checks, not a rerun of the benchmark. The 16 + 36 published exam records stay unchanged. A policy failing to turn the valve is still a completed runtime check and is reported as a policy failure. No GPU was started, no training ran and nothing was uploaded.

## Tested versions and limits

macOS 15.6.1 arm64, Python 3.12.15, MuJoCo 3.16.0, LeRobot 0.6.1, Torch 2.11.0, torchvision 0.26.0, NumPy 2.2.6 and SciPy 1.18.1. Exact installed versions and machine-readable outcomes are in `validation/`. The original Linux/CUDA Torch 2.7.1 environment has **not** been revalidated: network package downloads timed out. The Linux dependency selection preserves that Torch version but remains untested in this packaging round. Cross-platform inference is not claimed to be bit-identical.

The macOS PyAV/OpenCV combination emitted duplicate AVFoundation class warnings; both inference runs and video generation finished. LeRobot first reported unavailable saved `cuda` configuration while reading it; the wrapper then explicitly set CPU before constructing/loading the policy. No pretrained model was downloaded.

The first teacher launch failed before physics because a relative rest-file reference was checked against the current directory. Resolving it against the storyboard fixed this packaging issue; the original failure logs remain in the internal handoff workspace. The low-level API example was also corrected to select the bundled reviewed placement and natural-rest state explicitly. Physics, controllers, scores and saved numeric parameters were not changed.

Original installation logs, failed initialization logs, full local smoke evidence and videos are under `_internal/validation/` in the local handoff. That private workspace is excluded from public uploads. Small sanitized check logs are included in `validation/`.
