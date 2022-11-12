import sys

import gym
import numpy as np
import torch

from envs.FourRooms import FourRoomsController, FourRooms
from option_critic.singleAgent import OptionCriticAgent
import torch.optim as optim

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
timeout = 10_000

controls = {
    0: (-1, 0),  # 'LEFT'
    1: (1, 0),  # 'RIGHT'
    2: (0, -1),  # 'DOWN'
    3: (0, 1),  # 'UP'
}


def train(env, learning_rate, num_episodes, num_steps):

    agent = OptionCriticAgent(in_features=env.observation_space.shape[0],
                              num_actions=env.action_space.n,
                              num_options=2,
                              )
    # Use Adam optimizer because it should converge faster than SGD and generalization may not be super important
    oc_optimizer = optim.SGD(agent.parameters(), lr=learning_rate)

    # all episode length
    all_lengths = []
    # average episode length
    average_lengths = []
    # all episode rewards
    all_rewards = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Action counts
    action_counts = {action: 0 for action in controls}
    for episode in range(num_episodes):

        rewards_list = []

        state, _ = env.reset()
        for steps in range(num_steps):

            action, termination, log_probs, Q = agent.forward(state)
            action_counts[action] += 1
            new_state, reward, done, _, _ = env.step(action)

            rewards_list.append(reward)

            state = new_state

            with torch.no_grad():

                td_target = agent.compute_td_target(reward, done, new_state)

            actor_loss = agent.actor_loss(td_target, log_probs, termination, Q)
            critic_loss = agent.critic_loss(td_target, Q)
            # print(f"Actor loss:{actor_loss}")
            # print(f"Critic loss:{critic_loss}")
            loss = actor_loss + critic_loss
            oc_optimizer.zero_grad()
            loss.backward()
            oc_optimizer.step()

            if done or steps == num_steps - 1:
                if episode % 10 == 0:
                    all_rewards.append(np.sum(rewards_list))
                    all_lengths.append(steps)
                    average_lengths.append(np.mean(all_lengths[-10:]))
                    # Where total length is the number of steps taken in the episode and average length is average
                    # steps in all episodes seen
                    sys.stdout.write(
                        "episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,
                                                                                                  np.sum(rewards_list),
                                                                                                  steps,
                                                                                                  average_lengths[-1]))
                    # print(action_counts)
                break

    return all_rewards, all_lengths, average_lengths


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # env = FourRoomsController(FourRooms(rooms, timeout=timeout), controls=controls)
    rewards, lengths, avg_lengths = train(env, learning_rate=0.0005, num_episodes=3000, num_steps=3000)
