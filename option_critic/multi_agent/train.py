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
    agent = OptionCriticAgent(in_features=env.state_space.shape[0],
                              num_actions=env.action_space(env.possible_agents[0]).n,
                              num_options=4,
                              num_agents=env.max_num_agents,
                              device=device
                              )

    # Use Adam optimizer because it should converge faster than SGD and generalization may not be super important
    oc_optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    agent.to(device)
    # No of steps before critic update
    update_frequency = 4
    target_update_frequency = 200
    batch_size = 32

    buffer = ReplayBuffer(10000)
    torch.autograd.set_detect_anomaly(True)
    rewards = []
    train_returns = []
    train_loss = []
    episode = 0

    obs_dict = env.reset()
    # Create state from observations of all agents
    state = np.concatenate(list(obs_dict.values())).flatten()

    for steps in range(num_steps):

        actions, options, beta_w, log_prob, entropy, q = agent.forward(state)
        # Change actions to dict for sending to environment
        actions_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}

        new_obs_dict, reward_dict, _, dones, _ = env.step(actions_dict)

        new_state = np.concatenate(list(new_obs_dict.values())).flatten()
        reward = sum(reward_dict.values())
        done = not (False in dones.values())  # If all agents are done then end episode

        buffer.push(state, agent.current_options.cpu().numpy(), reward, new_state, done)

        rewards.append(reward)

        state = new_state

        with torch.no_grad():
            td_target = agent.compute_td_target(reward, done, new_state, beta_w)

        actor_loss = agent.actor_loss(td_target, log_prob, entropy, beta_w, q)
        critic_loss = torch.tensor(0).to(device)
        # print(f"Actor Loss:{actor_loss}")

        if len(buffer) > batch_size:

            if steps % update_frequency == 0:
                data_batch = buffer.sample(batch_size)
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
            obs_dict = env.reset()
            # Create state from observations of all agents
            state = np.concatenate(list(obs_dict.values()))
            if episode % 10 == 0:
                print("Eps:", agent.eps)
                sys.stdout.write("episode: {}, return: {} , steps: {}\n".format(episode, G, steps))
    torch.save(agent.state_dict(), "outputs/model_checkpoint_n3.pt")

    return train_returns, train_loss


def visualize_rollout(env):
    agent = OptionCriticAgent(in_features=env.state_space.shape[0],
                              num_actions=env.action_space(env.possible_agents[0]).n,
                              num_options=8,
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


def plot_curves(arr_list, legend_list, color_list,xlabel, ylabel, fig_title):
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

    env = simple_spread_v2.parallel_env(N=2, local_ratio=0.5, max_cycles=25, continuous_actions=False,
                                        render_mode=None)

    num_trials = 1
    all_returns = []
    all_losses = []
    for i in range(num_trials):
        train_returns, train_loss = train(env, learning_rate=0.0005, num_steps=150_000)
        all_returns.append(train_returns)
        all_losses.append(train_loss)

    np.save("outputs/train_returns_n3", all_returns)
    np.save("outputs/train_loss_n3", all_losses)
    all_returns = np.load("outputs/train_returns_n3_1.npy", allow_pickle=True)
    all_losses = np.load("outputs/train_loss_n3_1.npy", allow_pickle=True)

    plot_curves([np.array(all_returns)],
                ["Returns Averaged over 5 trials"],
                ["b"],
                "Episodes",
                "Averaged discounted return", "Returns Over 5 Trails (N=3)")

    plot_curves([np.array(all_losses)],
                ["Losses Averaged over 5 trials"],
                ["r"],
                "Time Steps",
                "Averaged loss", "Losses Over 5 Trails (N=3)")
    # visualize_rollout(env)
