import sys
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from pettingzoo.mpe import simple_spread_v2

from option_critic.multi_agent.multiAgent import OptionCriticAgent
import torch.optim as optim
from option_critic.utils.experience_replay import ReplayBuffer


def train(env, learning_rate, num_steps):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)
    agent1 = OptionCriticAgent(in_features=env.observation_space(env.possible_agents[0]).shape[0],
                               num_actions=env.action_space(env.possible_agents[0]).n,
                               num_options=8,
                               num_agents=1,
                               device=device
                               )
    agent2 = OptionCriticAgent(in_features=env.observation_space(env.possible_agents[1]).shape[0],
                               num_actions=env.action_space(env.possible_agents[0]).n,
                               num_options=8,
                               num_agents=1,
                               device=device
                               )
    # agent3 = OptionCriticAgent(in_features=env.observation_space(env.possible_agents[2]).shape[0],
    #                            num_actions=env.action_space(env.possible_agents[0]).n,
    #                            num_options=8,
    #                            num_agents=1,
    #                            device=device
    #                            )

    # Use Adam optimizer because it should converge faster than SGD and generalization may not be super important
    oc_optimizer1 = optim.Adam(agent1.parameters(), lr=learning_rate)
    oc_optimizer2 = optim.Adam(agent1.parameters(), lr=learning_rate)
    # oc_optimizer3 = optim.Adam(agent1.parameters(), lr=learning_rate)

    agent1.to(device)
    agent2.to(device)
    # agent3.to(device)

    # No of steps before critic update
    update_frequency = 4
    target_update_frequency = 200
    batch_size = 32

    buffer1 = ReplayBuffer(10000)
    buffer2 = ReplayBuffer(10000)
    # buffer3 = ReplayBuffer(10000)

    torch.autograd.set_detect_anomaly(True)
    rewards = []
    train_returns = []
    train_loss = []
    episode = 0

    obs_dict = env.reset()

    for steps in range(num_steps):

        actions1, options1, beta_w1, log_prob1, entropy1, q1 = agent1.forward(obs_dict["agent_0"])
        actions2, options2, beta_w2, log_prob2, entropy2, q2 = agent2.forward(obs_dict["agent_1"])
        # actions3, options3, beta_w3, log_prob3, entropy3, q3 = agent3.forward(obs_dict["agent_2"])

        # Change actions to dict for sending to environment
        actions_dict = {"agent_0": actions1[0][0], "agent_1": actions2[0][0]}#, "agent_2": actions3[0][0]}

        new_obs_dict, reward_dict, _, dones, _ = env.step(actions_dict)

        reward = sum(reward_dict.values())
        reward = [reward] * env.max_num_agents
        done = not (False in dones.values())  # If all agents are done then end episode

        buffer1.push(obs_dict["agent_0"], agent1.current_options.cpu().numpy(), [reward_dict["agent_0"]],
                     new_obs_dict["agent_0"], dones["agent_0"])
        buffer2.push(obs_dict["agent_1"], agent2.current_options.cpu().numpy(), [reward_dict["agent_1"]],
                     new_obs_dict["agent_1"], dones["agent_1"])
        # buffer3.push(obs_dict["agent_2"], agent3.current_options.cpu().numpy(), [reward_dict["agent_2"]],
        #              new_obs_dict["agent_2"], dones["agent_2"])

        rewards.append(reward[0])

        obs_dict = new_obs_dict

        with torch.no_grad():
            td_target1 = agent1.compute_td_target([reward_dict["agent_0"]], dones["agent_0"], new_obs_dict["agent_0"], beta_w1)
            td_target2 = agent2.compute_td_target([reward_dict["agent_1"]], dones["agent_1"], new_obs_dict["agent_1"], beta_w2)
            # td_target3 = agent3.compute_td_target([reward_dict["agent_2"]], dones["agent_2"], new_obs_dict["agent_2"], beta_w3)

        actor_loss1 = agent1.actor_loss(td_target1, log_prob1, entropy1, beta_w1, q1)
        actor_loss2 = agent2.actor_loss(td_target2, log_prob2, entropy2, beta_w2, q2)
        # actor_loss3 = agent3.actor_loss(td_target3, log_prob3, entropy3, beta_w3, q3)

        critic_loss1 = torch.tensor(0,dtype=torch.float).to(device)
        critic_loss2 = torch.tensor(0,dtype=torch.float).to(device)
        # critic_loss3 = torch.tensor(0,dtype=torch.float).to(device)
        # print(f"Actor Loss:{actor_loss}")

        if len(buffer1) > batch_size:

            if steps % update_frequency == 0:
                data_batch1 = buffer1.sample(batch_size)
                data_batch2 = buffer2.sample(batch_size)
                # data_batch3 = buffer3.sample(batch_size)

                critic_loss1 = agent1.critic_loss(data_batch1)
                critic_loss2 = agent2.critic_loss(data_batch2)
                # critic_loss3 = agent3.critic_loss(data_batch3)
                # print(f"Critic loss:{critic_loss}")

        # print(f"Actor loss:{actor_loss}")
        loss1 = actor_loss1 + critic_loss1
        loss2 = actor_loss2 + critic_loss2
        # loss3 = actor_loss3 + critic_loss3

        oc_optimizer1.zero_grad(True)
        loss1.backward()
        oc_optimizer1.step()

        oc_optimizer2.zero_grad(True)
        loss2.backward()
        oc_optimizer2.step()

        # oc_optimizer3.zero_grad(True)
        # loss3.backward()
        # oc_optimizer3.step()

        if steps % target_update_frequency == 0:
            agent1.update_target_net()
            agent2.update_target_net()
            # agent3.update_target_net()

        if done:
            episode += 1
            G = 0
            for r in reversed(rewards):
                G = r + 0.99 * G
            train_returns.append(G)
            rewards = []
            obs_dict = env.reset()

            if episode % 10 == 0:
                print("Eps:", agent1.eps)
                sys.stdout.write("episode: {}, return: {} , steps: {}\n".format(episode, G, steps))

    return train_returns, train_loss


def visualize_rollout(env):
    agent = OptionCriticAgent(in_features=env.state_space.shape[0],
                              num_actions=env.action_space(env.possible_agents[0]).n,
                              num_options=4,
                              num_agents=env.max_num_agents,
                              device="cpu"
                              )
    agent.load_state_dict(torch.load("outputs/model_checkpoint_n3.pt"))

    done = False
    obs_dict = env.reset()
    state = np.concatenate(list(obs_dict.values())).flatten()
    t = 0
    rewards = []

    while not done:
        # Get next action by greedy
        actions, _, _, _, _, _ = agent.forward(state)
        time.sleep(0.1)
        env.render()

        actions_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}
        next_obs, reward_dict, dones, _, _ = env.step(actions_dict)
        done = not (False in dones.values())

        if done:
            break
        next_state = np.concatenate(list(next_obs.values())).flatten()
        reward = sum(reward_dict.values())

        state = next_state
        rewards.append(reward)
        t += 1

    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
    print(f"Return: {G} , Time:{t}")


def plot_curves(arr_list, legend_list, color_list, xlabel, ylabel, fig_title):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        ylabel (string): label of the Y axis

        Note that, make sure the elements in the arr_list, legend_list and color_list are associated with each other correctly.
        Do not forget to change the ylabel for different plots.
    """
    # set the figure type
    fig, ax = plt.subplots(figsize=(12, 8))

    # PLEASE NOTE: Change the labels for different plots
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # ploth results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # compute the standard error
        print(arr.shape)
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(range(arr.shape[1]), arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err *= 1.96
        ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3,
                        color=color)
        # save the plot handle
        h_list.append(h)

    # plot legends
    ax.set_title(f"{fig_title}")
    ax.legend(handles=h_list)

    # save the figure
    plt.savefig(f"{fig_title}.png", dpi=200)

    plt.show()


if __name__ == '__main__':

    env = simple_spread_v2.parallel_env(N=2, local_ratio=0.0, max_cycles=25, continuous_actions=False,
                                        render_mode=None)

    num_trials = 1
    all_returns = []

    for i in range(num_trials):
        train_returns, train_loss = train(env, learning_rate=0.0005, num_steps=150_000)
        all_returns.append(train_returns)

    np.save("outputs/decentralized_train_returns_n2", all_returns)
    all_returns = np.load("outputs/decentralized_train_returns_n2.npy", allow_pickle=True)

    plot_curves([np.array(all_returns)],
                ["Returns"],
                ["b"],
                "Episodes",
                "Averaged discounted return", "Decentralized Returns (N=2)")

    # visualize_rollout(env)
