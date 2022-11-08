"""
An OpenAI Gym environment for the Four Rooms domain. 
This is modified with permission from Christian Diamore's work.
"""

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

rooms = [
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
]


class FourRooms(gym.Env):
    def __init__(self, grid: list = rooms, timeout=459):
        # define the four room as a 2-D array for easy state space reference and
        # visualization 0 represents an empty cell; 1 represents a wall cell

        # NOTE: the origin for a 2-D numpy array is located at top-left
        # while the origin for the FourRooms is at the bottom-left. The following
        # codes performs the re-projection by reversing the rows
        self.grid = np.array(list(reversed(grid)))

        # define the action space
        self.observation_space, self.action_space = (
            spaces.MultiDiscrete(self.grid.shape, dtype=np.int8),
            spaces.Box(low=-1, high=1, shape=(2,), dtype=int),
        )

        # define the start and goal state
        self.agent_state = (0, 0)
        self.start_state = (0, 0)
        self.goal_state = (10, 10)

        self.t = 0
        self.timeout = timeout

    def reset(self):
        """
        Reset the agent's state to the start state [0, 0]
        Return both the start state and reward
        """
        # reset the agent state to be [0, 0]
        self.agent_state, self.t = self.start_state, 0
        return np.array(self.agent_state), {}

    def is_wall(self, state: tuple):
        return (
            not self.observation_space.contains(np.array(state))
            or self.grid[state] == 1
        )

    def step(self, act: tuple):
        """
        :param act:
            a tuple in {[1, 0], [-1, 0], [0, 1], [0, -1]} representing an
            update to the state space

        :returns:
            :next_state: tuple
                x, y integer coordinates of the agent's new state, i.e. (1, 1)
            :reward: int
                1 if the agent reached the goal, else 0
            :done: bool.
                whether the episode is done, either by hitting goal or timeout
                on the episode
        """
        self.t += 1

        # Compute the next state, and only update if we don't hit a wall
        next_state = tuple(np.array(self.agent_state) + np.array(act))
        if not self.is_wall(next_state): self.agent_state = next_state

        # Reward the agent if it hits the goal
        # Done if it was rewarded, or if it times out the episode
        reward = 1.0 if self.agent_state == (10, 10) else -0.5
        done = reward == 1.0 or (self.t >= self.timeout)

        return np.array(self.agent_state), reward, done, False, {}

    def render(self, mode="human"):
        pass


class FourRoomsController(gym.ActionWrapper):
    """Map from a discrete action space into an actual movement in the grid."""

    def __init__(self, environment: gym.Env, controls: dict[int, tuple]):
        assert all(
            environment.action_space.contains(move)
            for move in controls.values()
        )

        super().__init__(environment)
        self.controls = controls
        self.action_space = spaces.Discrete(len(controls))

    def action(self, action):
        return self.controls[action]

    def reverse_action(self, action):
        return -1 * self.controls[action]


class FourRoomsRandomJitter(gym.ActionWrapper):
    """
    With probability p, the agent takes the correct direction.
    With probability 1 - p, the agent takes one of the two perpendicular actions.

    For example, if the correct action is "LEFT", then
        - With probability 0.8, the agent takes action "LEFT";
        - With probability 0.1, the agent takes action "UP";
        - With probability 0.1, the agent takes action "DOWN".
    """
    def __init__(self, environment: gym.Env, p: float = 0.8):
        super().__init__(environment)
        self.p = p

    @staticmethod
    def perpendicular_action(action):
        return np.random.choice([2, 3] if action in [0, 1] else [0, 1], 1)[0]

    def action(self, action):
        return (
            action
            if np.random.uniform() < self.p else
            self.perpendicular_action(action)
        )

    def reverse_action(self, action):
        pass

controls={
    0: (-1, 0),  # 'LEFT'
    1: (1, 0),  # 'RIGHT'
    2: (0, -1),  # 'DOWN'
    3: (0, 1),  # 'UP'
}

timeout = 10_000
four_rooms_env_no_jitter = FourRoomsController(FourRooms(rooms, timeout=timeout), controls=controls)

four_rooms_env_jitter = FourRoomsRandomJitter(
    FourRoomsController(FourRooms(rooms, timeout=timeout), controls=controls), p=0.8
)