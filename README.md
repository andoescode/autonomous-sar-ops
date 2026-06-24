# Autonomous SAR Ops Core

Autonomous SAR Ops Core is a Python-based multi-drone search-and-rescue autonomy project. It combines Gymnasium-style environment design, A* path planning, greedy and MILP-based task allocation, disruption-aware replanning, and experiment-based evaluation.

The Unity visualisation layer is maintained separately so the Python autonomy logic and Unity scene/assets can be versioned independently.

---

## Project Goal

The goal of this project is to build a small but extensible autonomy system where multiple drones search a hazardous area, visit priority zones, avoid obstacles, manage battery constraints, and adapt when the mission environment changes.

This repository focuses on the core autonomy logic:

* environment modelling
* path planning
* task allocation
* replanning
* mission state management
* experiment logging and evaluation

Unity is used as an external visualisation client rather than the main source of decision logic.

---

## Problem

Autonomous drones must search a grid-based environment while handling:

* obstacles and blocked paths
* search/rescue targets
* priority inspection zones
* limited battery
* multiple drones
* dynamic disruptions
* route replanning

The system compares simple heuristic planning against optimisation-based planning.

---

## System Architecture

```
Python Core Repo
├── Gym-style SAR environment
├── A* path planner
├── Greedy task allocator
├── MILP task allocator
├── Replanner
├── Mission controller
└── Experiment logger
        │
        │ exports mission state / replay data as JSON
        ▼
Unity Visualisation Repo
└── Loads mission replay and visualises drones, routes, targets, obstacles, and replanning events
```

*For v0.1, Python owns the decision-making logic. Unity is used mainly to visualise mission execution.*

---

## Core Components

### Environment

A Gymnasium-style SAR grid environment containing:

* drone positions
* obstacles
* targets
* priority zones
* battery state
* mission progress
* scenario loading from YAML

### Path Planning

A* path planning for drone movement through the grid while avoiding blocked or restricted cells.

### Task Allocation

Two allocation methods are planned for comparison:

* greedy nearest-target allocation
* MILP-based allocation using travel cost, target priority, battery feasibility, and missed-target penalties

### Replanning

The replanning module updates mission plans when:

* a path becomes blocked
* a new target appears
* a drone has insufficient battery
* an assigned route becomes invalid

### Unity Visualisation

Unity is maintained in a separate repository. The Python core exports mission replay/state data, which Unity can load and animate.

---

## Methods Compared

Initial v0.1 methods:

* Greedy allocation + A* routing
* MILP allocation + A* routing
* MILP allocation + replanning

Future versions may add:

* PPO/DQN learning agents
* ROS2/Gazebo simulation
* perception-based target detection
* FastAPI/Streamlit mission dashboard

---

## Metrics

The project evaluates planners using:

* mission completion time
* total travel distance
* coverage percentage
* search/rescue success rate
* number of replans
* failed assignment count
* average battery usage
* planning runtime

---

## Repository Structure

```text
autonomous-sar-ops-core/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── env.yaml
│   ├── planner.yaml
│   └── scenario.yaml
│
├── src/
│   └── autonomous_sar_ops/
│       ├── envs/
│       │   ├── sar_grid_env.py
│       │   ├── observations.py
│       │   ├── rewards.py
│       │   └── scenario_loader.py
│       │
│       ├── planning/
│       │   ├── path_planning/
│       │   │   └── astar.py
│       │   ├── task_allocation/
│       │   │   ├── greedy_allocator.py
│       │   │   ├── milp_allocator.py
│       │   │   └── constraints.py
│       │   └── replanning/
│       │       └── replanner.py
│       │
│       ├── mission/
│       │   ├── drone.py
│       │   ├── mission_state.py
│       │   ├── mission_controller.py
│       │   └── metrics.py
│       │
│       ├── simulation/
│       │   ├── unity_bridge.py
│       │   ├── message_schema.py
│       │   └── simulation_runner.py
│       │
│       └── utils/
│           ├── config.py
│           ├── geometry.py
│           └── logging.py
│
├── scenarios/
│   ├── small_map.yaml
│   ├── blocked_path.yaml
│   └── multi_drone.yaml
│
├── experiments/
│   ├── run_baseline.py
│   ├── run_milp_planner.py
│   ├── results/
│   └── plots/
│
├── tests/
│   ├── test_env.py
│   ├── test_astar.py
│   ├── test_greedy_allocator.py
│   ├── test_milp_allocator.py
│   └── test_replanner.py
│
└── docs/
    ├── architecture.md
    └── architecture.png
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run greedy baseline:

```bash
python experiments/run_baseline.py
```

Run MILP planner:

```bash
python experiments/run_milp_planner.py
```

Generated results will be saved under:

```text
experiments/results/
```

---

## Unity Integration

Unity visualisation is maintained in a separate repository:

```text
autonomous-sar-ops-unity
```

The Python core exports mission replay files in JSON format. The Unity project reads these replay files and visualises:

* drone movement
* assigned routes
* obstacles
* blocked paths
* targets
* priority zones
* replanning events

This separation keeps the Python codebase lightweight and avoids Unity-generated files, scene conflicts, and large asset commits in the core repository.

---

## Current Scope

v0.1 focuses on:

* Python SAR environment
* A* path planning
* greedy task allocation
* MILP task allocation
* disruption-aware replanning
* JSON replay export
* Unity visualisation support
* planner comparison metrics

---

## Future Roadmap

### v0.2 — Deep RL

* PPO/DQN baselines
* reward shaping
* planner vs RL comparison

### v0.3 — ROS2/Gazebo

* ROS2 mission nodes
* Gazebo simulation
* robotics middleware integration

### v0.4 — Perception

* synthetic data generation from Unity
* target/anomaly detection
* detection-triggered replanning

### v0.5 — Dashboard/API

* FastAPI backend
* Streamlit dashboard
* mission replay and metrics viewer

---

