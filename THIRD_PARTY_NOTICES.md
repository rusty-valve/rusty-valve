# Third-party sources and notices

Rusty Valve's licenses cover its own contributions. The third-party sources below retain their original licenses and notices.

This repository includes rendered media and six standalone valve-station XML files. The valve runtime bundles OpenArm v1 XML and meshes and the ZED Mini camera mesh. The Unitree G1 and MuJoCo Playground notices cover sources used for the climbing preview.

## Sources used in the demonstrations

| Source | Use in Rusty Valve | Original license |
|---|---|---|
| [OpenArm v1](https://github.com/enactic/openarm_mujoco/tree/a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c/v1) | Robot model shown in valve-operation and ACT media, the six-station overview, and S-hook assembly and gripper images | [Apache-2.0](LICENSES/third-party/openarm-LICENSE.txt) |
| [Stereolabs ZED Mini description](https://github.com/stereolabs/zed-ros2-description) | Camera mesh and reference URDF used in the valve simulation's head-camera setup | [Apache-2.0 and the complete supplied notices](LICENSES/third-party/zed-mini-LICENSE.txt) |
| [Unitree G1 model in MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97/unitree_g1) | Robot model shown in the [ladder-climbing GIF](assets/g1-ladder-climbing.gif) | [BSD-3-Clause](LICENSES/third-party/unitree-g1-LICENSE.txt) |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), version 0.2.0 | Framework used to build and train the climbing demonstration | [Apache-2.0](LICENSES/third-party/mujoco-playground-LICENSE.txt) |

### OpenArm

Copyright 2025 Enactic, Inc.

The model source is `enactic/openarm_mujoco`, revision `a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c`, including `v1/openarm.xml`, `v1/openarm_bimanual.xml`, and their referenced meshes. The upstream XML and meshes were retained unchanged in the development assets. Rusty Valve's scene adapters add robot placement, a fixed support, S-hooks, cameras, and the valve task.

### ZED Mini

The reference URDF states: Copyright 2026 Stereolabs.

The source files are `meshes/zedm.stl` and `urdf/zed_macro.urdf.xacro`, retrieved on 2026-10-28. Rusty Valve adds its own camera mounting support and adapts the camera coordinate frame. The complete supplied license is preserved, including the Caffe copyright and redistribution notice appended to it.

### Unitree G1 and MuJoCo Playground

Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"). All rights reserved.

The G1 meshes come from MuJoCo Menagerie revision `1b86ece576591213e2b666ebf59508454200ca97`. Its model description credits Unitree's public G1 MJCF. The climbing task uses the model through MuJoCo Playground's `G1JoystickFlatTerrain`, with a project-built ladder and fixed tools.

The G1 model's BSD-3-Clause license and Playground's Apache-2.0 license are separate. For the framework and MJX model, see [MuJoCo Playground, Zakka et al., 2025](https://arxiv.org/abs/2502.08844).

## Bray reference material

The generic DN25 valve uses selected dimensions and torque references from:

- [Bray Flow-Tek 7000/8000 brochure](https://www.bray.com/docs/default-source/brochures/product-brochures/en_f-2301_s7000_8000_web.pdf), pages 10–11: DN25 dimensions and the 7100 threaded-end model.
- [Bray TB1005 actuator-selection guide](https://www.bray.com/docs/default-source/technical-bulletin/en_tb-1005_act-selectguide-ball-valves.pdf), pages 3 and 11: torque reference and typical torque ratios.

The valve and station files use project-built, simplified geometry. No Bray CAD, converted manufacturer mesh, or handbook PDF is included. These links credit the references; Rusty Valve does not relicense the handbooks.

## Runnable package

See the [runtime's third-party notices](runtime/THIRD_PARTY_NOTICES.md) for the bundled model assets and their license texts. The runtime's installed software dependencies retain their own licenses.

Project-written explanations on this page use **CC BY 4.0**. Upstream notices and license texts retain their original terms. See [license scope](LICENSING.md).
