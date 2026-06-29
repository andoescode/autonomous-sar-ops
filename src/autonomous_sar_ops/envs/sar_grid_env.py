# Run module from src: uv run python -m  autonomous_sar_ops.envs.sar_grid_env

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

AgentType = Literal["ugv", "uav"] # an agent can be either ugv (ground robot) or uav (drone)
AgentMode = Literal["ugv", "uav", "both"] # env includes: all ugv | all uav | both ugv and uav
SpawnMode = Literal["single_base", "random_deployed"] # extension: allows multiple base started

# Agent types:
#         ugv = ground robot
#               - blocked by ground obstacles
#               - uses Manhattan/grid-style movement cost

#         uav = aerial drone
#               - ignores ground obstacles
#               - blocked by no-fly zones
#               - uses Euclidean-style travel cost for planning

@dataclass
class AgentState:
    """
    State of each agent in env {A_i: (position, battery, active_state)}
    """
    # Features
    id: int
    agent_type: AgentType # agent type
    position: tuple[int, int] # agent position
    base_position: tuple[int, int] # base position
    battery: float # agent battery level
    active: bool = True # agent active state

    # Assignment
    assigned_targets: list[int] = field(default_factory=list)
    route: list[tuple[int, int]] = field(default_factory=list)
    route_index: int = 0

class SARGridAllocatingEnv(gym.Env):
    """
    Route-following multi-agent SAR execution environment.
    
    Flow:

        Env -> states need planning -> route planning (from planner) -> simulate on gym -> replan if not all requirements met.

    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    AGENT_TYPE_TO_ID = {
            "ugv": 0,
            "uav": 1,
        }

    REWARDS = {
        "target_completed": 20.00, # reward for finding a target
        "all_targets_completed": 50.00, # reward for finding all targets
        "travel_cost": -0.20, # penalty for long travel route
        "battery_usage": -0.05, # penalty for more battery used (i.e. take longer time)
        "route_blocked": -10.0, # penalty for planning route with blockage(s)
        "battery_depleted": -5.00, # penalty for running out of battery mid way
        "idle": -0.02, # penalty for not progressing
        "timeout": -25.00, # penalty for running out of time (out of steps but havent found all targets yet)
    }
    
    def __init__(
        self,
        grid_size: tuple[int, int] = (10, 10), # size of env
        agent_mode: AgentMode = "both", # ground only | drone only | mixed
        num_ugv: int = 2, # number of ugv
        num_uav: int = 2, # number of uav
        num_targets: int = 3, # number of targets in env
        max_battery: float = 100.0, # max battery of agent
        max_steps: int = 100, # max step agent(s) can take
        obstacle_ratio: float = 0.15, # ratios of ground obs (for ugv)
        no_fly_ratio: float = 0.05, # ratios of no fly zone (for uav)
        base_position: tuple[int, int] = (0, 0), # postion of the starting base for agents
        spawn_mode: SpawnMode = "single_base",
        max_reset_tries: int = 200, # limit for resetting map 
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if agent_types is None:
            agent_types = ["ugv", "ugv"]

        if len(agent_types) == 0:
            raise ValueError("agent_types must contain at least one agent.")

        if any(agent_type not in ("ugv", "uav") for agent_type in agent_types):
            raise ValueError("agent_types can only contain 'ugv' or 'uav'.")

        self.agent_types = agent_types
        self.num_agents = len(agent_types)

        self.grid_height, self.grid_width = grid_size
        self.num_targets = num_targets
        self.max_battery = max_battery
        self.max_steps = max_steps
        self.render_mode = render_mode

        max_coord = max(self.grid_height - 1, self.grid_width - 1)

        # a discrete action per agent (one in [0,1,2,3,4])
        self.action_space = spaces.MultiDiscrete([5] * self.num_agents) # 5 actions agent can take (0 to 4)

        # dict observation = {grid, agent_positions, batteries, target_positions, target_completed}
        self.observation_space = spaces.Dict(
            {
                # whether that grid is available or not
                "grid" : spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.grid_height, self.grid_width),
                    dtype=np.int8,
                ),
                # position of each agent
                "agent_positions": spaces.Box(
                    low=0,
                    high=max(self.grid_height, self.grid_width),
                    shape=(self.num_agents, 2),
                    dtype=np.int32,
                ),
                # battery level of each agent
                "agent_batteries": spaces.Box(
                    low=0,
                    high=self.max_battery,
                    shape=(self.num_agents,),
                    dtype=np.int32,
                ),
                # position of target
                "target_positions": spaces.Box(
                    low=0,
                    high=max(self.grid_height, self.grid_width),
                    shape=(self.num_targets, 2),
                    dtype=np.int32,
                ),
                # which target is complete (if complete 1, else 0)
                "target_completed": spaces.MultiBinary(self.num_targets),
            }
        )

        self.grid: np.ndarray # grid array representation
        self.agents: list[AgentState] # list of agents (states)
        self.target_positions: list[tuple[int, int]] # list of target positions
        self.target_completed: np.ndarray # Array 
        self.step_count: int # keep track of the step taken

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
        spawn_mode: SpawnMode,
    )
        if grid_size[0] <=1 or grid_size[1] <= 1:
            raise ValueError("grid_size must be at least (2,2).")

        if agent_mode == num_ugv <= 0:
            raise ValueError("num_targets must be greater than 0.")

        if num_uav <= 0:
            raise ValueError("num_targets must be greater than 0.")
        
        if num_targets <= 0:
            raise ValueError("num_targets must be greater than 0.")

        if max_battery <= 0:
            raise ValueError("max_battery must be greater than 0.")
        
    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:

        super().reset(seed=seed)
    

def main() -> None:
    env = SARGridEnv(render_mode="ansi")
    print("In sar grid env!")

    # obs, info = env.reset(seed=42)

    # print("Initial info:")
    # print(info)
    # env.render()

    # done = False

    # while not done:
    #     action = env.action_space.sample()

    #     obs, reward, terminated, truncated, info = env.step(action)

    #     print(f"Action: {action}")
    #     print(f"Reward: {reward}")
    #     print(f"Info: {info}")

    #     env.render()

    #     done = terminated or truncated


if __name__ == "__main__":
    main()