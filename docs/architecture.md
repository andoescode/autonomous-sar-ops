# System Architecture

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
