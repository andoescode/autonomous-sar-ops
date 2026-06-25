# Run module from src: uv run python -m  autonomous_sar_ops.envs.sar_grid_env

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

AgentType = Literal["ugv", "uav"]

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
    id: int
    position: tuple[int, int] # agent position
    battery: int # agent battery level
    active: bool = True # agent active state

class SARGridEnv(gym.Env):
    """
    Multi-agent search-and-rescue grid environment.

    Actions per agent:
        0 = stay
        1 = up
        2 = down
        3 = left
        4 = right
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    ACTIONS = {
            "stay": 0,
            "up": 1,
            "down": 2,
            "left": 3,
            "right": 4,
        }

    ID_TO_ACTION = {v: k for k, v in ACTIONS.items()}

    REWARDS = {
        "step": -0.05, # the less step taken the better
        "new_cell": 0.10, # discover new cell
        "invalid_move": -1.00, # move out of grid or bump into obstacle(s)
        "target_completed": 10.00, # find a target
        "all_targets_completed": 20.00, # find all targets
        "battery_depleted": -5.00, # ran out of battery mid way
        "timeout": -10.00, # ran out of time (out of steps but havent found all targets yet)
    }
    
    def __init__(
        self,
        grid_size: tuple[int, int] = (10, 10), # size of env
        # num_agents: int = 2, # number of agents in env
        agent_types: list[AgentType] | None = None, # land or aerial agent(s) or both
        num_targets: int = 3, # number of targets in env
        max_battery: int = 50, # max battery of agent
        max_steps: int = 100, # max step agent(s) can take
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
        self.num_agents = num_agents
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
        self.agents: list[DroneState] # list of agents (states)
        self.target_positions: list[tuple[int, int]] # list of target positions
        self.target_completed: np.ndarray # Array 
        self.step_count: int # keep track of the step taken

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