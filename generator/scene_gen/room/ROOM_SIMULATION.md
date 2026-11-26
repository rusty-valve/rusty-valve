# Run a room and its valves in MuJoCo

In the room directory, run `python -B run_room.py --steps 20` to check loading and physics stepping. Run `python -B run_room.py` for the interactive viewer, or `mjpython -B run_room.py` on macOS. The Python environment needs MuJoCo and NumPy.

For a robot or experiment script, load the scene from the room's `run_room.py`.

```python
from run_room import load_room

room = load_room()
# Set robot commands or experiment inputs on room.model and room.data.
room.step()
```

`step()` updates valve resistance before every physics step. It looks up the resistance from the direction of motion and the valve stem angle, then lets the MuJoCo solver apply friction. Opening requires more torque to start and less through the rest of the turn. Closing requires more torque at the start and again near the closed stop. Handwheel resistance accounts for the gear ratio and efficiency. Breakaway torque applies again after a pause.

Open `room.xml` to inspect the assembled MuJoCo model. Use `run_room.py` or `room.step()` for variable-resistance simulation. Open `usd/room.usd` where supplied to view the room in a USD application. Keep the rest of the room folder beside it.

`task_index.json` lists valves by their room-local `V###` IDs. Each ID matches its USD instance and prefixes its MuJoCo body and joint names. `source_valve_id` identifies the source process valve or utility service.

## Part names

| Name | Meaning |
| --- | --- |
| `P1_` | Utility pipes and fittings |
| `P2_` | Process pipes and fittings |
| `CAT_` | Equipment catalogue parts |
| `INS` | Insulation pieces |
| `R` / `HX` / `P` | Reactor / heat exchanger / pump |
| `rf` / `neck` | Raised flange face / short neck |
| `site_` | Building and site parts, such as grating, stairs and rails |
