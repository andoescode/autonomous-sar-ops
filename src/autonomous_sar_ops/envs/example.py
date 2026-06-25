# Run module from src:
# uv run python -m autonomous_sar_ops.envs.sar_grid_env

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from math import sqrt
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces


AgentType = Literal["ugv", "uav"]


@dataclass
class AgentState:
    id: int
    agent_type: AgentType
    position: tuple[int, int]
    battery: float
    active: bool = True


@dataclass
class RoutePlan:
    agent_id: int
    agent_type: AgentType
    assigned_targets: list[int]
    completed_targets: list[int]
    route: list[tuple[int, int]]
    route_cost: float
    battery_used: float
    valid: bool
    failure_reason: str | None = None


class SARGridMissionEnv(gym.Env):
    """
    Mission-level multi-agent SAR environment.

    This environment focuses on agent distribution / task allocation.

    Action:
        action[target_id] = assigned_agent_id

        Example with 3 agents and 4 targets:
            action = [0, 2, 1, 3]

        Meaning:
            target 0 -> agent 0
            target 1 -> agent 2
            target 2 -> agent 1
            target 3 -> unassigned

        num_agents is used as the "unassigned" token.

    Agent types:
        ugv:
            - ground robot
            - blocked by obstacle_grid
            - 4-neighbour movement
            - Manhattan-style route cost

        uav:
            - aerial drone
            - blocked by no_fly_grid
            - 8-neighbour movement
            - Euclidean-style route cost

    This is intentionally not a PettingZoo env yet because the current goal is
    centralised mission planning and allocation, not independent agent policies.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    AGENT_TYPE_TO_ID = {
        "ugv": 0,
        "uav": 1,
    }

    ID_TO_AGENT_TYPE = {
        0: "ugv",
        1: "uav",
    }

    REWARDS = {
        "target_completed": 20.0,
        "all_targets_completed": 50.0,
        "travel_cost": -0.20,
        "mission_time": -0.10,
        "battery_usage": -0.05,
        "invalid_assignment": -15.0,
        "unassigned_target": -10.0,
        "inactive_agent_assignment": -10.0,
        "timeout": -25.0,
    }

    def __init__(
        self,
        grid_size: tuple[int, int] = (10, 10),
        agent_types: list[AgentType] | None = None,
        num_targets: int = 3,
        max_battery: float = 50.0,
        max_steps: int = 10,
        obstacle_ratio: float = 0.15,
        no_fly_ratio: float = 0.05,
        base_position: tuple[int, int] = (0, 0),
        max_reset_tries: int = 200,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if agent_types is None:
            agent_types = ["ugv", "ugv"]

        self._validate_agent_types(agent_types)

        if num_targets <= 0:
            raise ValueError("num_targets must be greater than 0.")

        if grid_size[0] <= 1 or grid_size[1] <= 1:
            raise ValueError("grid_size must be at least (2, 2).")

        if max_battery <= 0:
            raise ValueError("max_battery must be greater than 0.")

        if max_steps <= 0:
            raise ValueError("max_steps must be greater than 0.")

        if not (0.0 <= obstacle_ratio < 1.0):
            raise ValueError("obstacle_ratio must be in [0.0, 1.0).")

        if not (0.0 <= no_fly_ratio < 1.0):
            raise ValueError("no_fly_ratio must be in [0.0, 1.0).")

        self.grid_height, self.grid_width = grid_size
        self.agent_types = list(agent_types)
        self.num_agents = len(self.agent_types)
        self.num_targets = num_targets
        self.max_battery = float(max_battery)
        self.max_steps = max_steps
        self.obstacle_ratio = obstacle_ratio
        self.no_fly_ratio = no_fly_ratio
        self.base_position = base_position
        self.max_reset_tries = max_reset_tries
        self.render_mode = render_mode

        if not self._in_bounds(self.base_position):
            raise ValueError("base_position must be inside the grid.")

        # Action format:
        # one assignment decision per target.
        #
        # Values:
        # 0 ... num_agents - 1 = assigned agent id
        # num_agents          = unassigned
        self.unassigned_action = self.num_agents
        self.action_space = spaces.MultiDiscrete(
            [self.num_agents + 1] * self.num_targets
        )

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
                cell
                for cell in reachable_cells
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

        self.agents = [
            AgentState(
                id=agent_id,
                agent_type=agent_type,
                position=self.base_position,
                battery=self.max_battery,
                active=True,
            )
            for agent_id, agent_type in enumerate(self.agent_types)
        ]

        self.visited = np.zeros(
            (self.grid_height, self.grid_width),
            dtype=np.int32,
        )
        base_row, base_col = self.base_position
        self.visited[base_row, base_col] = self.num_agents

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray | list[int],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assignments = np.asarray(action, dtype=np.int64)

        if assignments.shape != (self.num_targets,):
            raise ValueError(
                f"Expected action shape {(self.num_targets,)}, got {assignments.shape}."
            )

        if np.any(assignments < 0) or np.any(assignments > self.num_agents):
            raise ValueError(
                f"Assignments must be in [0, {self.num_agents}], where "
                f"{self.num_agents} means unassigned."
            )

        self.step_count += 1

        reward_parts = {key: 0.0 for key in self.REWARDS}

        targets_by_agent = self._group_targets_by_agent(assignments)

        route_plans: list[RoutePlan] = []
        completed_this_step = 0
        total_travel_cost = 0.0
        total_battery_used = 0.0
        mission_time = 0.0

        # Penalise unassigned incomplete targets.
        for target_id, assigned_agent_id in enumerate(assignments):
            if self.target_completed[target_id]:
                continue

            if assigned_agent_id == self.unassigned_action:
                priority = float(self.target_priorities[target_id])
                reward_parts["unassigned_target"] += (
                    self.REWARDS["unassigned_target"] * priority
                )

        # Compute and execute each agent plan.
        for agent_id, target_ids in targets_by_agent.items():
            agent = self.agents[agent_id]

            if not target_ids:
                continue

            if not agent.active:
                reward_parts["inactive_agent_assignment"] += (
                    self.REWARDS["inactive_agent_assignment"] * len(target_ids)
                )
                route_plans.append(
                    RoutePlan(
                        agent_id=agent.id,
                        agent_type=agent.agent_type,
                        assigned_targets=target_ids,
                        completed_targets=[],
                        route=[],
                        route_cost=0.0,
                        battery_used=0.0,
                        valid=False,
                        failure_reason="assigned_to_inactive_agent",
                    )
                )
                continue

            plan = self._build_nearest_target_route(agent, target_ids)
            route_plans.append(plan)

            total_travel_cost += plan.route_cost
            total_battery_used += plan.battery_used
            mission_time = max(mission_time, plan.route_cost)

            if not plan.valid:
                reward_parts["invalid_assignment"] += (
                    self.REWARDS["invalid_assignment"] * max(1, len(target_ids))
                )

            for completed_target_id in plan.completed_targets:
                if self.target_completed[completed_target_id]:
                    continue

                self.target_completed[completed_target_id] = 1
                completed_this_step += 1

                priority = float(self.target_priorities[completed_target_id])
                reward_parts["target_completed"] += (
                    self.REWARDS["target_completed"] * priority
                )

            if plan.route:
                agent.position = plan.route[-1]

                for row, col in plan.route:
                    self.visited[row, col] += 1

            agent.battery -= plan.battery_used

            if agent.battery <= 0:
                agent.battery = 0.0
                agent.active = False

        reward_parts["travel_cost"] += self.REWARDS["travel_cost"] * total_travel_cost
        reward_parts["mission_time"] += self.REWARDS["mission_time"] * mission_time
        reward_parts["battery_usage"] += self.REWARDS["battery_usage"] * total_battery_used

        all_targets_completed = bool(np.all(self.target_completed))
        all_agents_inactive = all(not agent.active for agent in self.agents)

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
        info["route_plans"] = [self._route_plan_to_dict(plan) for plan in route_plans]
        info["metrics"] = self._get_metrics(
            total_travel_cost=total_travel_cost,
            total_battery_used=total_battery_used,
            mission_time=mission_time,
            completed_this_step=completed_this_step,
        )

        return obs, reward, terminated, truncated, info

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

    def get_travel_cost(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        agent_type: AgentType,
    ) -> float:
        """
        Public helper for greedy/MILP planners.

        Returns inf if no feasible path exists for the given agent type.
        """
        cost, _ = self._shortest_path(start, goal, agent_type)
        return cost

    def get_route(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        agent_type: AgentType,
    ) -> list[tuple[int, int]]:
        """
        Public helper for Unity replay / planner debugging.
        """
        _, path = self._shortest_path(start, goal, agent_type)
        return path

    def get_blocked_cells_for_agent_type(
        self,
        agent_type: AgentType,
    ) -> set[tuple[int, int]]:
        if agent_type == "ugv":
            grid = self.obstacle_grid
        elif agent_type == "uav":
            grid = self.no_fly_grid
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

        rows, cols = np.where(grid == 1)

        return {
            (int(row), int(col))
            for row, col in zip(rows, cols, strict=True)
        }

    def _group_targets_by_agent(
        self,
        assignments: np.ndarray,
    ) -> dict[int, list[int]]:
        targets_by_agent = {
            agent_id: []
            for agent_id in range(self.num_agents)
        }

        for target_id, assigned_agent_id in enumerate(assignments):
            if self.target_completed[target_id]:
                continue

            if assigned_agent_id == self.unassigned_action:
                continue

            targets_by_agent[int(assigned_agent_id)].append(target_id)

        return targets_by_agent

    def _build_nearest_target_route(
        self,
        agent: AgentState,
        target_ids: list[int],
    ) -> RoutePlan:
        """
        Build a simple nearest-neighbour route for all targets assigned to one agent.

        This intentionally keeps route sequencing simple for v0.1.
        MILP/OR-Tools can replace this later.
        """
        current_position = agent.position
        remaining_targets = list(target_ids)

        route: list[tuple[int, int]] = [current_position]
        completed_targets: list[int] = []
        total_cost = 0.0
        battery_used = 0.0

        while remaining_targets:
            best_target_id: int | None = None
            best_cost = float("inf")
            best_path: list[tuple[int, int]] = []

            for target_id in remaining_targets:
                target_position = self.target_positions[target_id]
                cost, path = self._shortest_path(
                    current_position,
                    target_position,
                    agent.agent_type,
                )

                if cost < best_cost:
                    best_target_id = target_id
                    best_cost = cost
                    best_path = path

            if best_target_id is None or not np.isfinite(best_cost) or not best_path:
                return RoutePlan(
                    agent_id=agent.id,
                    agent_type=agent.agent_type,
                    assigned_targets=target_ids,
                    completed_targets=completed_targets,
                    route=route,
                    route_cost=total_cost,
                    battery_used=battery_used,
                    valid=False,
                    failure_reason="unreachable_target",
                )

            if battery_used + best_cost > agent.battery:
                return RoutePlan(
                    agent_id=agent.id,
                    agent_type=agent.agent_type,
                    assigned_targets=target_ids,
                    completed_targets=completed_targets,
                    route=route,
                    route_cost=total_cost,
                    battery_used=battery_used,
                    valid=False,
                    failure_reason="insufficient_battery",
                )

            # Avoid duplicating the current cell when joining paths.
            route.extend(best_path[1:])
            total_cost += best_cost
            battery_used += best_cost

            current_position = self.target_positions[best_target_id]
            completed_targets.append(best_target_id)
            remaining_targets.remove(best_target_id)

        return RoutePlan(
            agent_id=agent.id,
            agent_type=agent.agent_type,
            assigned_targets=target_ids,
            completed_targets=completed_targets,
            route=route,
            route_cost=total_cost,
            battery_used=battery_used,
            valid=True,
            failure_reason=None,
        )

    def _shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        agent_type: AgentType,
    ) -> tuple[float, list[tuple[int, int]]]:
        """
        Dijkstra-style shortest path.

        UGV:
            4-neighbour movement
            blocked by obstacle_grid
            cost = 1 per move

        UAV:
            8-neighbour movement
            blocked by no_fly_grid
            cost = 1 for cardinal, sqrt(2) for diagonal
        """
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
        """
        Allows reset-time difficulty changes that do not alter observation shape.

        Do not change agent_types or num_targets here because Gym spaces are fixed
        after environment creation.
        """
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
                    "battery": agent.battery,
                    "active": agent.active,
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

    def _get_metrics(
        self,
        *,
        total_travel_cost: float,
        total_battery_used: float,
        mission_time: float,
        completed_this_step: int,
    ) -> dict[str, Any]:
        return {
            "total_travel_cost": total_travel_cost,
            "total_battery_used": total_battery_used,
            "mission_time": mission_time,
            "completed_this_step": completed_this_step,
            "completed_targets": int(np.sum(self.target_completed)),
            "num_targets": self.num_targets,
            "success_rate": float(np.sum(self.target_completed) / self.num_targets),
            "coverage_percentage": self._get_coverage_percentage(),
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
    def _route_plan_to_dict(plan: RoutePlan) -> dict[str, Any]:
        return {
            "agent_id": plan.agent_id,
            "agent_type": plan.agent_type,
            "assigned_targets": plan.assigned_targets,
            "completed_targets": plan.completed_targets,
            "route": plan.route,
            "route_cost": plan.route_cost,
            "battery_used": plan.battery_used,
            "valid": plan.valid,
            "failure_reason": plan.failure_reason,
        }

    @staticmethod
    def _validate_agent_types(agent_types: list[AgentType]) -> None:
        if len(agent_types) == 0:
            raise ValueError("agent_types must contain at least one agent.")

        invalid_types = [
            agent_type
            for agent_type in agent_types
            if agent_type not in ("ugv", "uav")
        ]

        if invalid_types:
            raise ValueError(
                f"Invalid agent types: {invalid_types}. "
                "Allowed values are 'ugv' and 'uav'."
            )


# Backwards-compatible alias if your imports still use SARGridEnv.
SARGridEnv = SARGridMissionEnv


def main() -> None:
    env = SARGridMissionEnv(
        grid_size=(10, 10),
        agent_types=["ugv", "ugv", "uav"],
        num_targets=4,
        max_battery=50.0,
        max_steps=5,
        obstacle_ratio=0.15,
        no_fly_ratio=0.05,
        render_mode="ansi",
    )

    obs, info = env.reset(seed=42)

    print("Initial observation keys:")
    print(obs.keys())

    print("\nInitial info:")
    print(info)

    print("\nInitial map:")
    env.render()

    done = False

    while not done:
        # Random allocation:
        # each target assigned to agent 0, 1, 2, or 3 where 3 = unassigned.
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Reward parts: {info['reward_parts']}")
        print(f"Metrics: {info['metrics']}")
        print(f"Completed targets: {info['completed_targets']}/{info['num_targets']}")

        env.render()

        done = terminated or truncated


if __name__ == "__main__":
    main()