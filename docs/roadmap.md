# Roadmap

[Rusty Valve](../README.md)

## What you can use today

- **Valves.** 17 ball valves, DN15–DN200, lever and handwheel, with opening and closing resistance, in MuJoCo and Isaac Sim. [Valves](../models/valves/README.md)
- **Rooms.** 15 chemical plant rooms with 304 valves, and a generator that builds new rooms from process recipes. [Rooms](../models/scenes/chemical-plant-rooms/README.md)
- **A valve task with an exam.** Six installations of a DN25 valve, a scripted robot that opens and closes them, 646 demonstrations, a learned baseline, and an exam any robot can take. [Level 1](../levels/01-handle-ball-valve/)

## What we are working on

**Robots turning valves inside the rooms.** Today the rooms tell you where each valve is and where a robot could stand. Next, a robot has to walk there, reach the valve, and turn it to its stop through contact.

We plan to start with the Unitree G1 humanoid: generate whole-body motion with [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/), track it with [SONIC](https://nvlabs.github.io/GEAR-SONIC/), and score every attempt with the same [exam](../levels/01-handle-ball-valve/#public-exam-v1). The hard part is the one our learned baseline already struggles with: keeping the tool on the handle and pushing until the valve stops.

The idea follows Apple's [HALO](https://apple-aiml-research.github.io/ml-halo/), which generates rooms and fits robot motions into them. Our rooms add what a chemical plant needs: equipment, pipes, and valves placed the way a process requires.

## Later

More valve types (gate, globe, butterfly, needle, wrench-operated), torque measured on real valves, stuck valves, and climbing to valves above 2 m. [Climbing preview](climbing.md)
