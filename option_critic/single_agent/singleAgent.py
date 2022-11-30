import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical


class OptionCriticAgent(nn.Module):
    def __init__(self, in_features, num_actions, num_options):

        super().__init__()
        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.current_option = np.random.randint(num_options)
        self.gamma = 0.99
        self.eps_min = 0.1
        self.eps_start = 1.0
        self.num_steps = 0
        self.duration = 50_000

        self.termination_count = 0
        self.deliberation_cost = 0.01
        self.ent_coef = 0.01
        self.term_reg = 0.01

        self.target_net_features = nn.Sequential(
            nn.Linear(self.in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),

        )

        self.state_features = nn.Sequential(
            nn.Linear(self.in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )

        self.Q = nn.Linear(64, self.num_options)
        self.target_q = nn.Linear(64, self.num_options)
        self.policy = nn.Linear(64, self.num_actions * self.num_options)
        self.termination = nn.Linear(64, self.num_options)

    def forward(self, state):
        """Take in observation from env and do 1 forward pass through network.
            Return: action, option termination
        """
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        # Passing in the input and hidden state into the model and obtaining outputs
        out = self.state_features(state)

        # Get Q
        Q = self.get_Q(out)

        # Get option policy and termination policy over softmax
        self.current_option, beta_w = self.get_option(Q, out)

        # Get policy from policy network
        pi_logits = self.policy(out).reshape(self.num_options, self.num_actions)
        pi_w = pi_logits[self.current_option, :].softmax(-1)

        action_dist = Categorical(pi_w)
        action = action_dist.sample()

        logprob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), beta_w, logprob, entropy, Q

    def get_features(self, state):
        state = Variable(torch.from_numpy(state).float())
        return self.state_features(state)

    def get_Q(self, features):
        return self.Q(features)

    def get_targetQ(self, state):
        state = Variable(torch.from_numpy(state).float())
        features = self.target_net_features(state)
        return self.target_q(features)

    def get_option(self, q, out):
        beta_w = self.termination(out).flatten()[self.current_option].sigmoid()
        termination_w = torch.bernoulli(beta_w)
        new_option = self.current_option
        # Change to e-greedy option if termination is > 0
        self.termination_count = 1 if termination_w > 0 else 0
        if termination_w > 0:
            if np.random.random() > self.eps:
                new_option = q[0].argmax().item()
            else:

                new_option = np.random.randint(0, self.num_options)
        else:
            _ = self.eps
        return new_option, beta_w

    def get_termination_prob(self, next_s):
        features = self.get_features(next_s)
        beta_w = self.termination(features)[:, self.current_option].sigmoid()

        return beta_w

    def compute_td_target(self, reward, done, next_s, beta_w):
        with torch.no_grad():
            features = self.get_features(next_s)
            q_next = self.get_Q(features).flatten()

            no_termination = (1 - beta_w) * q_next[self.current_option]
            termination = beta_w * q_next.max()

            return reward + self.gamma * (no_termination + termination) * (1 - done)

    def actor_loss(self, td_target, logprob, entropy, beta_w, q):

        with torch.no_grad():
            q_s = q.detach().flatten()
            v_s = q_s.max(-1)[0]
            q_sw = q_s[self.current_option]

        # Policy loss
        policy_loss = (-logprob * (td_target.detach() - q_sw)).mean()

        # Entropy loss
        entropy_loss = -entropy.mean()

        # Termination loss
        termination_loss = (beta_w * (q_sw - v_s)).mean() + (self.termination_count * self.deliberation_cost)

        actor_loss = policy_loss + termination_loss + self.ent_coef * entropy_loss

        return actor_loss

    def critic_loss(self, data):

        states, options, rewards, next_states, dones = data

        options = torch.LongTensor(options).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        masks = (1 - torch.LongTensor(dones)).unsqueeze(1)

        with torch.no_grad():
            term_probs = self.get_termination_prob(next_states).detach().unsqueeze(1)

        q = self.get_Q(self.get_features(states))

        # Target network for bootstrap
        next_q = self.get_targetQ(next_states)

        target = rewards + masks * self.gamma \
                 * ((1 - term_probs) * next_q.gather(1, options)
                    + term_probs * next_q.max(dim=-1)[0].unsqueeze(1)
                    )

        loss = 0.5 * (q.gather(1, options) - target.detach()).pow(2).mean()
        return loss

    def update_target_net(self):
        self.target_net_features.load_state_dict(self.state_features.state_dict())
        self.target_q.load_state_dict(self.Q.state_dict())

    @property
    def eps(self):
        """Linear decay for epsilon"""
        self.num_steps += 1

        if self.num_steps > self.duration:
            return self.eps_min
        else:
            return self.eps_start + (((self.eps_min - self.eps_start) / self.duration) * self.num_steps)
