# Valve and plant scene models

An open library of valve models and small chemical-plant scenes. We aim to turn site photos, dimensions, and operating videos into reusable simulation models of equipment and the spaces robots work in.

## Included models

The entries below use a **generic DN25 handle ball valve**, simplified from catalog references.

| Entry | Included |
|---|---|
| [DN25 handle ball valve](valves/dn25-handle-ball-valve/) | Geometry and resistance parameters, with sources; valve geometry is included in the station files |
| [Six Level 1 stations](scenes/handle-ball-valve-stations/) | A1–C2 models with valves, pipes, supports, and contact geometry; can be loaded independently of a robot |

See each entry for model details and resistance settings.

## ACT training records and checkpoint downloads

[ACT round 1 training records](round1/) and [ACT round 2 training records](round2/) contain their training metadata and curves. Download the complete [ACT round 1 checkpoint](https://huggingface.co/rusty-valve/a2_close_act_round1) or [ACT round 2 checkpoint](https://huggingface.co/rusty-valve/a2_close_act_round2), including the model and normalization tensors. See [download and placement instructions](../runtime/README.md#download-and-place-materials) for the two model repositories and where to place each checkpoint and its matching dataset.

## File formats

- **MJCF (`.xml`)** — models for MuJoCo simulation.
- **JSON (`.json`)** — dimensions, installation layouts, parameters, and sources.
- **OpenUSD** — planned for exchanging and assembling larger scenes; not yet included.

## Growing from real sites

We will gradually add more valve types, installation layouts, and small chemical-plant scenes based on real equipment. Each entry will include supporting images, dimensions, simulation files, and a short explanation of its sources and simplifications, distinguishing measured values from modeling assumptions.

[S-hook tool](../hardware/s-hook/) · [Level 1](../levels/01-handle-ball-valve/) · [Back to Rusty Valve](../README.md)

## License

Our MJCF and JSON model files are **Apache-2.0**. Technical text and project-created illustrations are **CC BY 4.0**. Third-party material retains its original terms. See [license scope](../LICENSING.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
