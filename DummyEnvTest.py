"""
This is a simple test function to make sure that a random policy can navigate a given environment
"""
import numpy as np
from envs.FourRooms import FourRooms
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_tag_v2


def test_random_policy(env):
    # reset the environment
    state, reward, done = env.reset()
    # run the random policy
    while not done:
        # choose a random action
        act = np.random.choice(list(env.action_space.keys()), 1)[0]
        # take a step
        next_state, reward, done = env.step(state, act)
        # update the state
        state = next_state
    return reward


# TODO THIS WON"T WORK.
# The random policy function is for single agents.

def test_random_policy_simple_spread():
    env = simple_spread_v2.parallel_env()
    test_random_policy(env)


def test_random_policy_simple_tag():
    env = simple_tag_v2.parallel_env()
    test_random_policy(env)


# main function
if __name__ == "__main__":
    test_random_policy(FourRooms())