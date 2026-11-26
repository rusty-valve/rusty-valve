# Room generator

Builds a chemical plant room from a process recipe: equipment, pipes, valves, insulation, platforms, stairs, and floor. The output opens in MuJoCo. To use Isaac Sim, download one of the ready-made rooms with OpenUSD files.

How to install it, rebuild one of the 15 rooms, and write your own recipe: [Generate your own room](../models/scenes/chemical-plant-rooms/README.md#generate-your-own-room).

```sh
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install -r scene_gen/requirements.txt
python -B -m scene_gen.room generate --recipe /path/to/room-12/recipe.json --out ./my_room_12
```

The generator needs [`generator_runtime/`](https://huggingface.co/datasets/rusty-valve/chemical-plant-rooms/tree/main/generator_runtime) next to `scene_gen/`, or the `SCENE_GEN_RUNTIME` environment variable pointing at it.

Code: Apache-2.0. Reactor dimension sources: [third-party notices](THIRD_PARTY_NOTICES.md).
