# Third-party notices

This runtime actually bundles OpenArm v1 XML/meshes and the ZED Mini camera mesh. They retain their upstream rights and notices.

- OpenArm v1: Enactic, Inc., 2025. Source: https://github.com/enactic/openarm_mujoco/tree/a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c/v1 . Apache-2.0; complete text in `LICENSES/third-party/openarm-LICENSE.txt`. Project adapters add the fixed support, S-hook and cameras; the upstream meshes retain their original files.
- ZED Mini: Stereolabs, 2026. Source: https://github.com/stereolabs/zed-ros2-description , `meshes/zedm.stl` and `urdf/zed_macro.urdf.xacro`, retrieved 2026-10-28. Complete Apache-2.0 and appended Caffe notices in `LICENSES/third-party/zed-mini-LICENSE.txt` and the asset directory. Project camera supports and optical-frame adaptations are separate contributions.
- The wrist camera is a project-created simplified geometric representation; no vendor mesh is bundled.
- Generic valve geometry is project-created. Dimension/torque references: Bray Flow-Tek 7000/8000 brochure pages 10–11 and TB1005 actuator-selection guide pages 3 and 11. No Bray CAD or handbook is redistributed.

MuJoCo, NumPy, SciPy, Pillow, ImageIO, PyTorch and LeRobot are installed dependencies, not vendored source. Their own installed distributions retain their licenses. This runtime contains no Unitree or MuJoCo Playground files.
