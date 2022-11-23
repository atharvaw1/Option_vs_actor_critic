import sys

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from envs.FourRooms import FourRoomsController, FourRooms
from option_critic.singleAgent import OptionCriticAgent
import torch.optim as optim
from experience_replay import ReplayBuffer
import torch.nn as nn

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
timeout = 500

controls = {
    0: (-1, 0),  # 'LEFT'
    1: (1, 0),  # 'RIGHT'
    2: (0, -1),  # 'DOWN'
    3: (0, 1),  # 'UP'
}


def train(env, learning_rate, num_steps):
    agent = OptionCriticAgent(in_features=env.observation_space.shape[0],
                              num_actions=env.action_space.n,
                              num_options=4,
                              )

    # Use Adam optimizer because it should converge faster than SGD and generalization may not be super important
    oc_optimizer = optim.RMSprop(agent.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # No of steps before critic update
    update_frequency = 4
    target_update_frequency = 200

    buffer = ReplayBuffer(10000)

    state, _ = env.reset()
    rewards = []
    train_returns = []
    train_loss = []
    episode = 0
    for steps in range(num_steps):

        action, beta_w, log_prob, entropy, Q = agent.forward(state)

        new_state, reward, done, _, _ = env.step(action)

        buffer.push(state, agent.current_option, reward, new_state, done)

        rewards.append(reward)

        state = new_state

        with torch.no_grad():
            td_target = agent.compute_td_target(reward, done, new_state, beta_w)

        actor_loss = agent.actor_loss(td_target, log_prob, entropy, beta_w, Q)
        critic_loss = torch.tensor([0])
        # print(f"Actor Loss:{actor_loss}")

        if len(buffer) > 32:

            if steps % update_frequency == 0:
                data_batch = buffer.sample(32)
                critic_loss = agent.critic_loss(data_batch)
                # print(f"Critic loss:{critic_loss}")


        # print(f"Actor loss:{actor_loss}")
        loss = actor_loss + critic_loss
        train_loss.append(loss.detach().cpu().numpy())
        oc_optimizer.zero_grad(True)
        loss.backward()
        oc_optimizer.step()

        if steps % target_update_frequency == 0:
            agent.update_target_net()

        if done:
            episode += 1
            G = 0
            for r in reversed(rewards):
                G = r + 0.99 * G
            train_returns.append(G)
            rewards = []
            state, _ = env.reset()
            if True:#episode % 10 == 0:
                print("Eps:", agent.eps)
                # Where total length is the number of steps taken in the episode and average length is average
                # steps in all episodes seen
                sys.stdout.write("episode: {}, return: {} , steps: {}\n".format(episode, G, steps))
    torch.save(agent.state_dict(), "model_checkpoint.pt")
    Vs = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            print(agent.get_Q(agent.get_features(np.array([i, j]))).cpu().detach().numpy())
            Vs[i, j] = agent.get_Q(agent.get_features(np.array([i, j]))).max(dim=-1)[0].cpu().detach().numpy()

    print(Vs)

    plt.imshow(Vs, cmap='hot', interpolation='nearest')
    plt.show()

    return train_returns, train_loss


def visualize_rollout():
    env = FourRoomsController(FourRooms(rooms, timeout=timeout), controls=controls)

    agent = OptionCriticAgent(in_features=env.observation_space.shape[0],
                              num_actions=env.action_space.n,
                              num_options=4,
                              )
    agent.load_state_dict(torch.load("model_checkpoint.pt"))
    arr = rooms.copy()
    done = False
    state, _ = env.reset()
    time = 0
    rewards = []

    while not done:
        # Get next action by greedy
        # arr[h - 1 - state[1], state[0]] = 3
        arr[state[0]][state[1]] = 3
        action, _, _, _, _ = agent.forward(state)
        next_s, reward, done, _, _ = env.step(action)
        state = next_s
        rewards.append(reward)
        time += 1

    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
    print(f"Return: {G} , Time:{time}")
    plt.imshow(arr)
    plt.show()



if __name__ == '__main__':

    # env = FourRoomsController(FourRooms(rooms, timeout=timeout), controls=controls)
    # train_returns, train_loss = train(env, learning_rate=0.0005, num_steps=150_000)
    #
    #
    # np.save("train_returns", train_returns)
    # np.save("train_loss", train_loss)
    #
    # plt.plot(train_returns)
    # plt.plot()
    # plt.xlabel('Episode')
    # plt.ylabel('Return')
    # plt.show()
    #
    # plt.plot(train_loss)
    # plt.xlabel('Steps')
    # plt.ylabel('Loss')
    # plt.show()
    visualize_rollout()
