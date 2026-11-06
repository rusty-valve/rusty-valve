# OpenArm camera model integration

The shared source of installation and optical parameters is
[openarm_camera_rig.json](../../configs/openarm_camera_rig.json).
`h0_robot.build_robot_tree()` installs the camera geometry in MuJoCo. Camera
names are `head_left`, `head_right`, `wrist_left`, and `wrist_right`.
`robot_head` remains an exact alias of the left head eye.

ZED Mini uses Stereolabs' original mesh, kept in `zedmini/` with its source and
licence. Its physical stereo baseline is represented. The Arducam B0495
module is simplified from the documented board size; assembled housing depth
and both S-tool brackets are project adaptations. None of these brackets is
presented as an unchanged OpenArm 2.0 gripper assembly.

Both hand modules and their supports attach directly to their corresponding
J7 wrist bodies, in symmetric positions above the wrists. The fixed tool
installation origin supplies the longitudinal offset, but the holder axes
remain wrist axes. Reversing the left S-tool's fixed mount by 180 degrees
therefore does not turn its camera underneath the wrist. Both modules still
follow all motion of their respective wrists.

The default `visual_only` profile adds no mass or collision geometry. The
optional `collision_envelopes` profile provides simplified outer collisions
but still has no measured complete camera/support inertia model. It is not
used by default or claimed as a physical-kit validation.

Sensor images use an ideal pinhole projection. Manufacturer nominal FOV,
simulation FOV, and fixed output resolution are recorded separately because
the published horizontal and vertical FOV figures do not form one exact
ideal pinhole calibration at every image mode. Lens distortion, focus blur,
sensor noise, and stereo depth estimation are not simulated.

The optical point stays at the lens reference. Only the sensor's own opaque
display housing/glass is hidden in that sensor's rendered optical path.
The arm, tool, opposite camera, fixtures, and valve remain visible and can
occlude the view. Outside views show every camera module.

The [runtime robot API example](../../../../../../../examples/openarm_simulation_step.py)
shows how to render head, wrist, and observer views from the bundled scene.
