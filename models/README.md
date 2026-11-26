# Models

[Rusty Valve](../README.md)

| | What it is |
|---|---|
| [Valves](valves/README.md) | 17 ball valves, DN15–DN200, lever and handwheel, for MuJoCo and Isaac Sim |
| [Chemical plant rooms](scenes/chemical-plant-rooms/README.md) | 15 rooms with equipment, pipes, platforms, and 304 valves, plus the generator that builds them |
| [Level 1 stations](scenes/handle-ball-valve-stations/) | Six installations of a DN25 lever valve used by the exam |
| [DN25 station valve](valves/dn25-handle-ball-valve/) | The generic valve inside the six stations |
| [ACT training records, round 1](round1/) · [round 2](round2/) | Training settings and loss curves for the learned baseline. Weights: [round 1](https://huggingface.co/rusty-valve/a2_close_act_round1) · [round 2](https://huggingface.co/rusty-valve/a2_close_act_round2). [Where to put them](../runtime/README.md#download-and-place-materials) |

## Formats

- **MJCF (`.xml`)** for MuJoCo.
- **OpenUSD (`.usd`)** for Isaac Sim and other USD tools.
- **STEP (`.step`)** CAD geometry of each valve.
- **JSON** dimensions, parameters, recipes, valve lists, and their sources.

## License

Our models and JSON files are Apache-2.0. Text and project-made images are CC BY 4.0. Third-party material keeps its own terms. [License scope](../LICENSING.md) · [Third-party notices](../THIRD_PARTY_NOTICES.md)
