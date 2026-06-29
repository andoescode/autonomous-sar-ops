# Autonomous SAR Ops Core — Plan

## Goal

Build a small autonomous search-and-rescue simulation project that demonstrates:

* Gym-style environment design
* multi-agent mission logic
* A* path planning
* greedy vs MILP task allocation
* disruption-aware replanning
* Unity-based mission visualisation

---

## System Design

The system includes 2 separated repos, responsible for different sections of system: 

* **LOGIC**: `autonomous-sar-ops-core` = main resume repo, Python environment/planning/optimisation.

* **VISUALISATION**: `autonomous-sar-ops-unity` = Unity visualisation client.

The Python core exports mission state and replay data as JSON. Unity reads the replay file and visualises the mission.

## System Flow

```text
Python env + planner + controller
        ↓
sends agent states, routes, targets, and events
        ↓
Unity visualisation
```

*For v0.1, Python owns the decision logic. Unity is mainly used to visualise agent movement, routes, obstacles, targets, and mission events.*

---

## v0.1 Scope

Implement:

* 2D grid SAR environment
* 2–3 agents
* obstacles and blocked paths
* search/rescue targets
* priority zones
* battery constraints
* A* path planning
* greedy task allocation baseline
* MILP task allocation
* replanning when a path is blocked or a new target appears
* Unity visualisation
* basic experiment metrics

Metrics:

* mission completion time
* total travel distance
* coverage percentage
* rescue/search success rate
* number of replans
* average battery usage

---

## Build Order

1. Create the Gym-style SAR environment.
2. Add scenario loading from YAML.
3. Implement A* path planning.
4. Implement greedy task allocation.
5. Implement MILP task allocation.
6. Add replanning for blocked paths and new targets.
7. Connect Python mission state to Unity visualisation.
8. Run experiments comparing greedy vs MILP.
9. Add README, architecture diagram, and demo GIF/video.

---

## Repo Structure

```
autonomous-sar-ops-core/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── src/
│   └── autonomous_sar_ops/
│       ├── envs/
│       ├── planning/
│       ├── mission/
│       ├── simulation/
│       │   ├── unity_bridge.py
│       │   └── message_schema.py
│       └── utils/
│
├── scenarios/
├── experiments/
├── tests/
├── configs/
└── docs/
    ├── architecture.md
    └── unity_integration.md
```

---

## Later Expansion

### v0.2 — Deep RL

* PPO/DQN baseline
* reward shaping
* planner vs RL comparison

### v0.3 — ROS2/Gazebo

* ROS2 nodes
* Gazebo world
* planner and mission controller integration

### v0.4 — Perception

* synthetic data from Unity
* object/anomaly detection
* detection-triggered replanning

### v0.5 — Dashboard/API

* FastAPI backend
* Streamlit dashboard
* mission replay and metrics view

---

## Target

Final v0.1 bullet:

> Built an autonomous multi-agent search-and-rescue simulation system integrating a Gym-style environment, A* path planning, MILP-based task allocation, disruption-aware replanning, and Unity-based mission visualisation.

Stronger version after experiments:

> Developed an autonomous SAR simulation platform comparing greedy and MILP-based planners across blocked-path and multi-agent scenarios using mission completion time, route distance, coverage, and rescue success metrics.
