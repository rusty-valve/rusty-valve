# Official OpenArm v1 source assets

Source: [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco/tree/a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c/v1)

Pinned revision: `a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c`.

`openarm.xml` and its 36 referenced mesh files are copied from the official
repository. They retain their original content. The license is Apache 2.0;
see [LICENSE](LICENSE). No URDF conversion is involved.

The project adapter is `valve_bench/robot.py`. It adds a fixed stand, task
cameras and an invisible `ee_control` site in `openarm_hand_tcp`. The added
site has no mass or collision geometry. The source robot keeps its original
joint ranges, inertias, collision geometry, actuator gains and force limits.

The TCP is at hand-local `(0, 0, 0.09)` metres. Hand +Z points toward the
fingertips; fingers open along +Y and -Y. X is the transverse direction.
The TCP is a control reference between the fingers, not a physical palm pad.

Generated complete scene XML uses absolute mesh paths in the current
checkout. After moving the checkout, regenerate the scene with
`build_robot_scene()`; no model download is needed to run it again.

The source model's motor limits and reflected finger force are simulation
parameters from the upstream model. They do not establish continuous-duty
hardware capability or a safe real-world valve load.
