# Run module from src:
# uv run python -m autonomous_sar_ops.envs.sar_grid_env

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import sqrt
from enum import Enum, StrEnum, auto
from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AgentType(StrEnum):
    """
    An agent can be either ugv (ground robot) or uav (drone).

    ugv = ground robot
        - ignores no-fly zones
        - blocked by ground obstacles
        - uses Manhattan/grid-style movement cost
        - movement cost = 1.0 per step

    uav = aerial drone
        - ignores ground obstacles
        - blocked by no-fly zones
        - uses Euclidean-style travel cost for planning    
        - movement cost = 1.0 (if cardinal) | sqrt(2) (if diagonal)
    """
    UGV = auto()
    UAV = auto()

class AgentMode(StrEnum):
    """
    Decide what type(s) of agents will be included in the env.
    """
    UGV = auto() # all ugv
    UAV = auto() # all uav
    BOTH = auto() # mix ugv + uav

class SpawnMode(StrEnum):
    """
    Decide what spawning type(s) will be in the env.
    """
    SINGLE_BASE = auto() # one base spawn all
    RANDOM_DEPLOYED = auto() # multiple bases

class Reward(Enum):
    TARGET_COMPLETED = ("target_completed", 20.00) # reward for finding a target
    ALL_TARGETS_COMPLETED = ("all_targets_completed", 50.00) # reward for finding all targets
    TRAVEL_COST = ("travel_cost", -0.20) # penalty for long travel route
    BATTERY_USAGE = ("battery_usage", -0.05) # penalty for more battery used (i.e. take longer time)
    ROUTE_BLOCKED = ("route_blocked", -10.00) # penalty for planning route with blockage(s)
    BATTERY_DEPLETED = ("battery_depleted", -5.00) # penalty for running out of battery mid way
    IDLE = ("idle", -0.02) # penalty for not progressing
    TIMEOUT = ("timeout", -25.00) # penalty for running out of time (out of steps but havent found all targets yet)

    def __init__(self, label: str, weight: float) -> None:
        self.label = label
        self.weight = weight

@dataclass
class AgentState:
    id: int
    agent_type: AgentType
    position: tuple[int, int]
    base_position: tuple[int, int]
    battery: float
    active: bool = True

    assigned_targets: list[int] = field(default_factory=list)
    route: list[tuple[int, int]] = field(default_factory=list)
    route_index: int = 0

class SARGridExecutionEnv(gym.Env):
    """
    Route-following multi-agent SAR execution environment.
        
    Flow:
    
        Env -> states need planning -> route planning (from planner) -> simulate on gym -> replan if not all requirements met.
    
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        grid_size: tuple[int, int] = (10, 10), # size of env
        agent_mode: AgentMode = AgentMode.BOTH, # ground only | drone only | mixed
        num_ugv: int = 2, # number of ugv
        num_uav: int = 2, # number of uav
        num_targets: int = 3, # number of targets in env
        max_battery: float = 100.0, # max battery of agent
        max_steps: int = 100, # max step agent(s) can take
        obstacle_ratio: float = 0.15, # ratios of ground obs (for ugv)
        no_fly_ratio: float = 0.05, # ratios of no fly zone (for uav)
        base_position: tuple[int, int] = (0, 0), # postion of the starting base for agents
        spawn_mode: SpawnMode = SpawnMode.SINGLE_BASE,
        max_reset_tries: int = 200, # limit for resetting map 
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self._validate_init_args(
            grid_size=grid_size,
            num_targets=num_targets,
            max_battery=max_battery,
            max_steps=max_steps,
            obstacle_ratio=obstacle_ratio,
            no_fly_ratio=no_fly_ratio,
            spawn_mode=spawn_mode,
                )
        
        self.grid_height, self.grid_width = grid_size
        self.agent_mode = agent_mode
        self.num_ugv = num_ugv
        self.num_uav = num_uav
        self.num_targets = num_targets
        self.max_battery = float(max_battery)
        self.max_steps = max_steps
        self.obstacle_ratio = obstacle_ratio
        self.no_fly_ratio = no_fly_ratio
        self.base_position = base_position
        self.spawn_mode = spawn_mode
        self.max_reset_tries = max_reset_tries
        self.render_mode = render_mode       

        self.agent_types = self._build_agent_types(
            agent_mode=self.agent_mode,
            num_ugv=self.num_ugv,
            num_uav=self.num_uav,
        )

        self.num_agents = len(self.agent_types)
        

        if not self._in_bounds(self.base_position):
            raise ValueError("base_position must be inside the grid.")

        # Option B:
        # The env has no allocation/movement action.
        # action=0 means "advance simulation by one tick using current routes".
        self.action_space = spaces.Discrete(1)

        max_coord = max(self.grid_height - 1, self.grid_width - 1)

        self.observation_space = spaces.Dict(
            {
                "obstacle_grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.grid_height, self.grid_width),
                    dtype=np.int8,
                ),
                "no_fly_grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.grid_height, self.grid_width),
                    dtype=np.int8,
                ),
                "agent_positions": spaces.Box(
                    low=0,
                    high=max_coord,
                    shape=(self.num_agents, 2),
                    dtype=np.int32,
                ),
                "agent_batteries": spaces.Box(
                    low=0.0,
                    high=self.max_battery,
                    shape=(self.num_agents,),
                    dtype=np.float32,
                ),
                "agent_type_ids": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_agents,),
                    dtype=np.int8,
                ),
                "target_positions": spaces.Box(
                    low=0,
                    high=max_coord,
                    shape=(self.num_targets, 2),
                    dtype=np.int32,
                ),
                "target_priorities": spaces.Box(
                    low=1,
                    high=3,
                    shape=(self.num_targets,),
                    dtype=np.int32,
                ),
                "target_completed": spaces.MultiBinary(self.num_targets),
                "base_position": spaces.Box(
                    low=0,
                    high=max_coord,
                    shape=(2,),
                    dtype=np.int32,
                ),
            }
        )

        self.obstacle_grid: np.ndarray
        self.no_fly_grid: np.ndarray
        self.agents: list[AgentState]
        self.target_positions: list[tuple[int, int]]
        self.target_priorities: np.ndarray
        self.target_completed: np.ndarray
        self.visited: np.ndarray
        self.step_count: int

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        if options:
            self._apply_reset_options(options)

        self.step_count = 0

        for _ in range(self.max_reset_tries):
            self.obstacle_grid = np.zeros(
                (self.grid_height, self.grid_width),
                dtype=np.int8,
            )
            self.no_fly_grid = np.zeros(
                (self.grid_height, self.grid_width),
                dtype=np.int8,
            )

            self._place_random_obstacles()
            self._place_random_no_fly_zones()

            base_row, base_col = self.base_position
            self.obstacle_grid[base_row, base_col] = 0
            self.no_fly_grid[base_row, base_col] = 0

            reachable_cells = self._get_reachable_cells_for_team()

            candidate_targets = sorted(
                cell for cell in reachable_cells
                if cell != self.base_position
            )

            if len(candidate_targets) >= self.num_targets:
                target_indices = self.np_random.choice(
                    len(candidate_targets),
                    size=self.num_targets,
                    replace=False,
                )

                self.target_positions = [
                    candidate_targets[int(idx)]
                    for idx in target_indices
                ]

                break
        else:
            raise RuntimeError(
                "Could not generate a valid map with enough reachable targets. "
                "Try reducing obstacle_ratio/no_fly_ratio or num_targets."
            )

        self.target_priorities = self.np_random.integers(
            low=1,
            high=4,
            size=self.num_targets,
            dtype=np.int32,
        )

        self.target_completed = np.zeros(self.num_targets, dtype=np.int8)

        self.agents = []
        occupied_starts: set[tuple[int, int]] = set()

        for agent_id, agent_type in enumerate(self.agent_types):
            start_position = self._sample_agent_start_position(
                agent_type=agent_type,
                occupied_starts=occupied_starts,
            )
            occupied_starts.add(start_position)

            self.agents.append(
                AgentState(
                    id=agent_id,
                    agent_type=agent_type,
                    position=start_position,
                    base_position=self.base_position,
                    battery=self.max_battery,
                    active=True,
                )
            )

        self.visited = np.zeros(
            (self.grid_height, self.grid_width),
            dtype=np.int32,
        )

        for agent in self.agents:
            row, col = agent.position
            self.visited[row, col] += 1

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int = 0,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if int(action) != 0:
            raise ValueError("SARGridExecutionEnv only supports action=0.")

        self.step_count += 1

        reward_parts = {key: 0.0 for key in self.REWARDS}
        events: list[dict[str, Any]] = []

        replan_required = False
        completed_this_step = 0
        total_travel_cost = 0.0
        total_battery_used = 0.0
        active_agents_with_routes = 0

        for agent in self.agents:
            if not agent.active:
                continue

            if agent.battery <= 0:
                agent.active = False
                replan_required = True
                reward_parts["battery_depleted"] += self.REWARDS["battery_depleted"]
                events.append(
                    {
                        "type": "battery_depleted",
                        "agent_id": agent.id,
                    }
                )
                continue

            if not agent.route or agent.route_index >= len(agent.route) - 1:
                reward_parts["idle"] += self.REWARDS["idle"]
                continue

            active_agents_with_routes += 1

            current_cell = agent.position
            next_cell = agent.route[agent.route_index + 1]

            if not self._is_valid_cell_for_agent(next_cell, agent.agent_type):
                agent.route = []
                agent.route_index = 0
                agent.assigned_targets = []

                replan_required = True
                reward_parts["route_blocked"] += self.REWARDS["route_blocked"]
                events.append(
                    {
                        "type": "route_blocked",
                        "agent_id": agent.id,
                        "blocked_cell": next_cell,
                    }
                )
                continue

            movement_cost = self._movement_cost(
                start=current_cell,
                end=next_cell,
                agent_type=agent.agent_type,
            )

            if movement_cost > agent.battery:
                agent.active = False
                replan_required = True

                reward_parts["battery_depleted"] += self.REWARDS["battery_depleted"]
                events.append(
                    {
                        "type": "insufficient_battery",
                        "agent_id": agent.id,
                        "required": movement_cost,
                        "remaining": agent.battery,
                    }
                )
                continue

            agent.position = next_cell
            agent.route_index += 1
            agent.battery -= movement_cost

            total_travel_cost += movement_cost
            total_battery_used += movement_cost

            row, col = next_cell
            self.visited[row, col] += 1

            reward_parts["travel_cost"] += self.REWARDS["travel_cost"] * movement_cost
            reward_parts["battery_usage"] += (
                self.REWARDS["battery_usage"] * movement_cost
            )

            for target_id, target_position in enumerate(self.target_positions):
                if self.target_completed[target_id]:
                    continue

                if agent.position == target_position:
                    self.target_completed[target_id] = 1
                    completed_this_step += 1

                    priority = float(self.target_priorities[target_id])
                    reward_parts["target_completed"] += (
                        self.REWARDS["target_completed"] * priority
                    )

                    events.append(
                        {
                            "type": "target_completed",
                            "agent_id": agent.id,
                            "target_id": target_id,
                            "position": target_position,
                            "priority": int(priority),
                        }
                    )

            if agent.battery <= 0:
                agent.battery = 0.0
                agent.active = False
                replan_required = True
                events.append(
                    {
                        "type": "battery_depleted",
                        "agent_id": agent.id,
                    }
                )

        all_targets_completed = bool(np.all(self.target_completed))
        all_agents_inactive = all(not agent.active for agent in self.agents)

        any_agent_has_route = any(
            agent.active
            and agent.route
            and agent.route_index < len(agent.route) - 1
            for agent in self.agents
        )

        if not all_targets_completed and not any_agent_has_route:
            replan_required = True
            events.append(
                {
                    "type": "all_routes_exhausted_mission_incomplete",
                }
            )

        terminated = all_targets_completed or all_agents_inactive
        truncated = self.step_count >= self.max_steps

        if all_targets_completed and completed_this_step > 0:
            reward_parts["all_targets_completed"] += self.REWARDS[
                "all_targets_completed"
            ]

        if truncated and not all_targets_completed:
            reward_parts["timeout"] += self.REWARDS["timeout"]

        reward = float(sum(reward_parts.values()))

        obs = self._get_obs()
        info = self._get_info()
        info["reward_parts"] = reward_parts
        info["events"] = events
        info["replan_required"] = replan_required
        info["metrics"] = {
            "completed_this_step": completed_this_step,
            "total_travel_cost": total_travel_cost,
            "total_battery_used": total_battery_used,
            "active_agents_with_routes": active_agents_with_routes,
            "coverage_percentage": self._get_coverage_percentage(),
        }

        return obs, reward, terminated, truncated, info

    def set_agent_routes(
        self,
        routes_by_agent: dict[int, dict[str, Any]],
    ) -> None:
        """
        Assign externally planned routes to agents.

        Expected format:
            {
                0: {
                    "target_ids": [0, 2],
                    "route": [(0, 0), (0, 1), (0, 2), ...],
                },
                1: {
                    "target_ids": [1],
                    "route": [(3, 4), (3, 5), ...],
                },
            }
        """
        for agent_id, route_data in routes_by_agent.items():
            if agent_id < 0 or agent_id >= self.num_agents:
                raise ValueError(f"Unknown agent id: {agent_id}")

            agent = self.agents[agent_id]

            target_ids = [
                int(target_id)
                for target_id in route_data.get("target_ids", [])
            ]
            route = [
                tuple(cell)
                for cell in route_data.get("route", [])
            ]

            for target_id in target_ids:
                if target_id < 0 or target_id >= self.num_targets:
                    raise ValueError(f"Unknown target id: {target_id}")

            if not route:
                agent.assigned_targets = []
                agent.route = []
                agent.route_index = 0
                continue

            if route[0] != agent.position:
                route = [agent.position] + route

            self._validate_route(
                route=route,
                agent_type=agent.agent_type,
                agent_id=agent.id,
            )

            agent.assigned_targets = target_ids
            agent.route = route
            agent.route_index = 0

    def clear_agent_routes(self) -> None:
        for agent in self.agents:
            agent.assigned_targets = []
            agent.route = []
            agent.route_index = 0

    def get_planning_state(self) -> dict[str, Any]:
        return {
            "grid_size": (self.grid_height, self.grid_width),
            "obstacle_grid": self.obstacle_grid.copy(),
            "no_fly_grid": self.no_fly_grid.copy(),
            "base_position": self.base_position,
            "step_count": self.step_count,
            "agents": [
                {
                    "id": agent.id,
                    "agent_type": agent.agent_type,
                    "position": agent.position,
                    "base_position": agent.base_position,
                    "battery": agent.battery,
                    "active": agent.active,
                    "assigned_targets": list(agent.assigned_targets),
                }
                for agent in self.agents
            ],
            "targets": [
                {
                    "id": target_id,
                    "position": target_position,
                    "priority": int(self.target_priorities[target_id]),
                    "completed": bool(self.target_completed[target_id]),
                    "reachable_by": self._get_agent_types_that_can_reach_cell(
                        target_position
                    ),
                }
                for target_id, target_position in enumerate(self.target_positions)
            ],
        }

    def block_cell(
        self,
        cell: tuple[int, int],
        layer: Literal["obstacle", "no_fly"] = "obstacle",
    ) -> None:
        if not self._in_bounds(cell):
            raise ValueError(f"Cell {cell} is outside the grid.")

        if cell == self.base_position:
            raise ValueError("Cannot block the base position.")

        row, col = cell

        if layer == "obstacle":
            self.obstacle_grid[row, col] = 1
        elif layer == "no_fly":
            self.no_fly_grid[row, col] = 1
        else:
            raise ValueError("layer must be 'obstacle' or 'no_fly'.")

    def get_route(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        agent_type: AgentType,
    ) -> list[tuple[int, int]]:
        _, path = self._shortest_path(start, goal, agent_type)
        return path

    def get_travel_cost(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        agent_type: AgentType,
    ) -> float:
        cost, _ = self._shortest_path(start, goal, agent_type)
        return cost

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None

        canvas = np.full(
            (self.grid_height, self.grid_width),
            " . ",
            dtype=object,
        )

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self.obstacle_grid[row, col] == 1:
                    canvas[row, col] = " # "

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self.no_fly_grid[row, col] == 1:
                    canvas[row, col] = " N "

        base_row, base_col = self.base_position
        canvas[base_row, base_col] = " B "

        for target_id, target_position in enumerate(self.target_positions):
            row, col = target_position
            priority = int(self.target_priorities[target_id])
            canvas[row, col] = " x " if self.target_completed[target_id] else f" T{priority}"

        for agent in self.agents:
            row, col = agent.position
            prefix = "G" if agent.agent_type == "ugv" else "D"
            canvas[row, col] = f" {prefix}{agent.id}"

        output = "\n".join(
            " ".join(str(cell) for cell in row)
            for row in canvas
        )

        print(output)
        print()

        return output

    def _sample_agent_start_position(
        self,
        *,
        agent_type: AgentType,
        occupied_starts: set[tuple[int, int]],
    ) -> tuple[int, int]:
        if self.spawn_mode == "single_base":
            return self.base_position

        if self.spawn_mode == "random_deployed":
            reachable_cells = sorted(
                self._get_reachable_cells_from_base_for_type(agent_type)
            )

            candidates = [
                cell for cell in reachable_cells
                if cell not in occupied_starts
                and cell not in self.target_positions
            ]

            if not candidates:
                return self.base_position

            selected_idx = int(self.np_random.integers(0, len(candidates)))
            return candidates[selected_idx]

        raise ValueError(f"Unknown spawn_mode: {self.spawn_mode}")

    def _validate_route(
        self,
        *,
        route: list[tuple[int, int]],
        agent_type: AgentType,
        agent_id: int,
    ) -> None:
        for cell in route:
            if not self._is_valid_cell_for_agent(cell, agent_type):
                raise ValueError(
                    f"Invalid route for agent {agent_id}. "
                    f"Cell {cell} is blocked for {agent_type}."
                )

        for current_cell, next_cell in zip(route[:-1], route[1:], strict=True):
            self._movement_cost(
                start=current_cell,
                end=next_cell,
                agent_type=agent_type,
            )

    def _movement_cost(
        self,
        *,
        start: tuple[int, int],
        end: tuple[int, int],
        agent_type: AgentType,
    ) -> float:
        start_row, start_col = start
        end_row, end_col = end

        d_row = abs(end_row - start_row)
        d_col = abs(end_col - start_col)

        if agent_type == "ugv":
            if d_row + d_col != 1:
                raise ValueError(
                    f"Invalid UGV move from {start} to {end}. "
                    "UGV can only move one cardinal cell."
                )
            return 1.0

        if agent_type == "uav":
            if max(d_row, d_col) != 1:
                raise ValueError(
                    f"Invalid UAV move from {start} to {end}. "
                    "UAV can only move to one neighbouring cell per tick."
                )

            if d_row == 1 and d_col == 1:
                return sqrt(2)

            return 1.0

        raise ValueError(f"Unknown agent_type: {agent_type}")

    def _shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        agent_type: AgentType,
    ) -> tuple[float, list[tuple[int, int]]]:
        if not self._is_valid_cell_for_agent(start, agent_type):
            return float("inf"), []

        if not self._is_valid_cell_for_agent(goal, agent_type):
            return float("inf"), []

        if start == goal:
            return 0.0, [start]

        frontier: list[tuple[float, tuple[int, int]]] = []
        heappush(frontier, (0.0, start))

        came_from: dict[tuple[int, int], tuple[int, int] | None] = {
            start: None
        }
        cost_so_far: dict[tuple[int, int], float] = {
            start: 0.0
        }

        while frontier:
            current_cost, current = heappop(frontier)

            if current == goal:
                break

            if current_cost > cost_so_far[current]:
                continue

            for next_cell, move_cost in self._get_neighbours(current, agent_type):
                new_cost = current_cost + move_cost

                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    came_from[next_cell] = current
                    heappush(frontier, (new_cost, next_cell))

        if goal not in came_from:
            return float("inf"), []

        path = self._reconstruct_path(came_from, goal)
        return cost_so_far[goal], path

    def _get_neighbours(
        self,
        position: tuple[int, int],
        agent_type: AgentType,
    ) -> list[tuple[tuple[int, int], float]]:
        row, col = position

        if agent_type == "ugv":
            candidates = [
                ((row - 1, col), 1.0),
                ((row + 1, col), 1.0),
                ((row, col - 1), 1.0),
                ((row, col + 1), 1.0),
            ]

        elif agent_type == "uav":
            candidates = [
                ((row - 1, col), 1.0),
                ((row + 1, col), 1.0),
                ((row, col - 1), 1.0),
                ((row, col + 1), 1.0),
                ((row - 1, col - 1), sqrt(2)),
                ((row - 1, col + 1), sqrt(2)),
                ((row + 1, col - 1), sqrt(2)),
                ((row + 1, col + 1), sqrt(2)),
            ]

        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

        return [
            (cell, cost)
            for cell, cost in candidates
            if self._is_valid_cell_for_agent(cell, agent_type)
        ]

    @staticmethod
    def _reconstruct_path(
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [goal]
        current = goal

        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def _apply_reset_options(self, options: dict[str, Any]) -> None:
        if "obstacle_ratio" in options:
            obstacle_ratio = float(options["obstacle_ratio"])
            if not (0.0 <= obstacle_ratio < 1.0):
                raise ValueError("obstacle_ratio must be in [0.0, 1.0).")
            self.obstacle_ratio = obstacle_ratio

        if "no_fly_ratio" in options:
            no_fly_ratio = float(options["no_fly_ratio"])
            if not (0.0 <= no_fly_ratio < 1.0):
                raise ValueError("no_fly_ratio must be in [0.0, 1.0).")
            self.no_fly_ratio = no_fly_ratio

        if "max_steps" in options:
            max_steps = int(options["max_steps"])
            if max_steps <= 0:
                raise ValueError("max_steps must be greater than 0.")
            self.max_steps = max_steps

    def _place_random_obstacles(self) -> None:
        num_cells = self.grid_height * self.grid_width
        num_obstacles = int(num_cells * self.obstacle_ratio)

        placed = 0

        while placed < num_obstacles:
            row = int(self.np_random.integers(0, self.grid_height))
            col = int(self.np_random.integers(0, self.grid_width))

            position = (row, col)

            if position == self.base_position:
                continue

            if self.obstacle_grid[row, col] == 1:
                continue

            self.obstacle_grid[row, col] = 1
            placed += 1

    def _place_random_no_fly_zones(self) -> None:
        num_cells = self.grid_height * self.grid_width
        num_no_fly = int(num_cells * self.no_fly_ratio)

        placed = 0

        while placed < num_no_fly:
            row = int(self.np_random.integers(0, self.grid_height))
            col = int(self.np_random.integers(0, self.grid_width))

            position = (row, col)

            if position == self.base_position:
                continue

            if self.no_fly_grid[row, col] == 1:
                continue

            self.no_fly_grid[row, col] = 1
            placed += 1

    def _get_reachable_cells_for_team(self) -> set[tuple[int, int]]:
        reachable: set[tuple[int, int]] = set()

        for agent_type in set(self.agent_types):
            reachable.update(
                self._get_reachable_cells_from_base_for_type(agent_type)
            )

        return reachable

    def _get_reachable_cells_from_base_for_type(
        self,
        agent_type: AgentType,
    ) -> set[tuple[int, int]]:
        if not self._is_valid_cell_for_agent(self.base_position, agent_type):
            return set()

        queue = deque([self.base_position])
        visited = {self.base_position}

        while queue:
            row, col = queue.popleft()

            for next_cell, _ in self._get_neighbours((row, col), agent_type):
                if next_cell in visited:
                    continue

                visited.add(next_cell)
                queue.append(next_cell)

        return visited

    def _is_valid_cell_for_agent(
        self,
        position: tuple[int, int],
        agent_type: AgentType,
    ) -> bool:
        if not self._in_bounds(position):
            return False

        row, col = position

        if agent_type == "ugv":
            return self.obstacle_grid[row, col] == 0

        if agent_type == "uav":
            return self.no_fly_grid[row, col] == 0

        raise ValueError(f"Unknown agent_type: {agent_type}")

    def _in_bounds(self, position: tuple[int, int]) -> bool:
        row, col = position

        return (
            0 <= row < self.grid_height
            and 0 <= col < self.grid_width
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "obstacle_grid": self.obstacle_grid.copy(),
            "no_fly_grid": self.no_fly_grid.copy(),
            "agent_positions": np.array(
                [agent.position for agent in self.agents],
                dtype=np.int32,
            ),
            "agent_batteries": np.array(
                [agent.battery for agent in self.agents],
                dtype=np.float32,
            ),
            "agent_type_ids": np.array(
                [
                    self.AGENT_TYPE_TO_ID[agent.agent_type]
                    for agent in self.agents
                ],
                dtype=np.int8,
            ),
            "target_positions": np.array(
                self.target_positions,
                dtype=np.int32,
            ),
            "target_priorities": self.target_priorities.copy(),
            "target_completed": self.target_completed.copy(),
            "base_position": np.array(
                self.base_position,
                dtype=np.int32,
            ),
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "agent_mode": self.agent_mode,
            "agent_types": list(self.agent_types),
            "completed_targets": int(np.sum(self.target_completed)),
            "num_targets": self.num_targets,
            "all_targets_completed": bool(np.all(self.target_completed)),
            "coverage_percentage": self._get_coverage_percentage(),
            "agent_states": [
                {
                    "id": agent.id,
                    "agent_type": agent.agent_type,
                    "position": agent.position,
                    "base_position": agent.base_position,
                    "battery": agent.battery,
                    "active": agent.active,
                    "assigned_targets": list(agent.assigned_targets),
                    "route_index": agent.route_index,
                    "route_length": len(agent.route),
                }
                for agent in self.agents
            ],
            "targets": [
                {
                    "id": target_id,
                    "position": target_position,
                    "priority": int(self.target_priorities[target_id]),
                    "completed": bool(self.target_completed[target_id]),
                    "reachable_by": self._get_agent_types_that_can_reach_cell(
                        target_position
                    ),
                }
                for target_id, target_position in enumerate(self.target_positions)
            ],
        }

    def _get_coverage_percentage(self) -> float:
        reachable = self._get_reachable_cells_for_team()

        if not reachable:
            return 0.0

        visited_reachable = sum(
            1
            for cell in reachable
            if self.visited[cell[0], cell[1]] > 0
        )

        return float(visited_reachable / len(reachable))

    def _get_agent_types_that_can_reach_cell(
        self,
        cell: tuple[int, int],
    ) -> list[AgentType]:
        reachable_by: list[AgentType] = []

        for agent_type in set(self.agent_types):
            reachable_cells = self._get_reachable_cells_from_base_for_type(agent_type)

            if cell in reachable_cells:
                reachable_by.append(agent_type)

        return sorted(reachable_by)

    @staticmethod
    def _build_agent_types(
        *,
        agent_mode: AgentMode,
        num_agents: int,
        num_ugv: int,
        num_uav: int,
    ) -> list[AgentType]:
        if agent_mode == "ugv":
            if num_agents <= 0:
                raise ValueError("num_agents must be greater than 0.")
            return ["ugv"] * num_ugv

        if agent_mode == "uav":
            if num_agents <= 0:
                raise ValueError("num_agents must be greater than 0.")
            return ["uav"] * num_uav

        if agent_mode == "both":
            if num_ugv <= 0:
                raise ValueError("num_ugv must be greater than 0 when agent_mode='both'.")

            if num_uav <= 0:
                raise ValueError("num_uav must be greater than 0 when agent_mode='both'.")

            return (["ugv"] * num_ugv) + (["uav"] * num_uav)

        raise ValueError("agent_mode must be one of: 'ugv', 'uav', 'both'.")

    @staticmethod
    def _validate_init_args(
        *,
        grid_size: tuple[int, int],
        agent_mode: AgentMode,
        num_ugv: int,
        num_uav: int,
        num_targets: int,
        max_battery: float,
        max_steps: int,
        obstacle_ratio: float,
        no_fly_ratio: float,
    )
        if grid_size[0] <=1 or grid_size[1] <= 1:
            raise ValueError("grid_size must be at least (2,2).")
    
        if agent_mode == AgentMode.UGV and num_ugv <= 0:
            raise ValueError("In UGV mode: num_ugv must be greater than 0.")
    
        if agent_mode == AgentMode.UAV and num_uav <= 0:
            raise ValueError("In UAV mode: num_uav must be greater than 0.")

        if agent_mode == AgentMode.BOTH and (num_uav <= 0 or num_ugv <= 0):
            raise ValueError("In BOTH mode: requires atleast 1 agent for each of the agent types.")
                    
        if num_targets <= 0:
            raise ValueError("num_targets must be greater than 0.")
    
        if max_battery <= 0:
            raise ValueError("max_battery must be greater than 0.")

        if max_steps <= 0:
            raise ValueError("max_steps must be greater than 0.")

        if obstacle_ratio > 1 or obstacle_ratio < 0 :
            raise ValueError("obstacle_ratio must be in range [0,1] inclusive.")

        if no_fly_ratio > 1 or no_fly_ratio < 0 :
            raise ValueError("no_fly_ratio must be in range [0,1] inclusive.")
            
SARGridEnv = SARGridExecutionEnv

def build_greedy_spread_routes(env: SARGridExecutionEnv) -> dict[int, dict[str, Any]]:
    """
    Assign each active agent to a different incomplete target.

    This is a simple baseline planner:
        - each agent gets at most one target per planning cycle
        - targets are not duplicated
        - candidate score = route_cost / target_priority
        - unreachable or battery-infeasible targets are skipped

    Later, MILP will replace this.
    """
    planning_state = env.get_planning_state()

    incomplete_targets = [
        target
        for target in planning_state["targets"]
        if not target["completed"]
    ]

    active_agents = [
        agent
        for agent in planning_state["agents"]
        if agent["active"]
    ]

    assigned_targets: set[int] = set()
    routes_by_agent: dict[int, dict[str, Any]] = {}

    for agent in active_agents:
        best_target: dict[str, Any] | None = None
        best_route: list[tuple[int, int]] = []
        best_score = float("inf")
        best_cost = float("inf")

        for target in incomplete_targets:
            target_id = int(target["id"])

            if target_id in assigned_targets:
                continue

            route = env.get_route(
                start=agent["position"],
                goal=target["position"],
                agent_type=agent["agent_type"],
            )

            if not route:
                continue

            cost = env.get_travel_cost(
                start=agent["position"],
                goal=target["position"],
                agent_type=agent["agent_type"],
            )

            if not np.isfinite(cost):
                continue

            if cost > float(agent["battery"]):
                continue

            priority = float(target["priority"])

            # Lower score is better.
            # High-priority targets become more attractive.
            score = cost / priority

            if score < best_score:
                best_score = score
                best_cost = cost
                best_target = target
                best_route = route

        if best_target is None:
            continue

        target_id = int(best_target["id"])
        assigned_targets.add(target_id)

        routes_by_agent[int(agent["id"])] = {
            "target_ids": [target_id],
            "route": best_route,
            "route_cost": best_cost,
        }

    return routes_by_agent


def main() -> None:
    env = SARGridExecutionEnv(
        grid_size=(10, 10),
        agent_mode="ugv",
        num_ugv=4,
        num_uav=0,
        num_targets=4,
        max_battery=100.0,
        max_steps=1000,
        obstacle_ratio=0.15,
        no_fly_ratio=0.05,
        spawn_mode="single_base",
        render_mode="ansi",
    )

    obs, info = env.reset(seed=1)

    print("Initial map:")
    env.render()

    # Temporary demo route assignment.
    # Later this should come from greedy_allocator.py or milp_allocator.py.
    routes_by_agent = build_greedy_spread_routes(env)
    env.set_agent_routes(routes_by_agent)
    print("Agents: ", env.agents)
    print("Number of agents: ", env.num_agents)
    print("Number of targets: ", env.num_targets)
    print("Target position: ", env.target_positions)
    print("Routes: ", routes_by_agent)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(0)

        print(f"Reward: {reward}")
        print(f"Events: {info['events']}")
        print(f"Replan required: {info['replan_required']}")
        print(f"Completed targets: {info['completed_targets']}/{info['num_targets']}")

        env.render()

        if info["replan_required"] and not terminated and not truncated:
            print("Replanning would be triggered here.")
            break


if __name__ == "__main__":
    main()