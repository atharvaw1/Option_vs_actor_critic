import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class OptionCriticAgent(nn.Module):
    def __init__(self, in_features, num_actions, num_options):

        super().__init__()
        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.current_option = 0
        self.gamma = 0.99
        self.eps = 0.7
        self.ent_coef = 0.01
        self.term_reg = 0.01

        # self.state_features = nn.Sequential(
        #     nn.Linear(self.in_channels, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU()
        # )
        self.state_features = nn.Sequential(
            nn.Linear(self.in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.Q = nn.Linear(64, self.num_options)
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
        pi_w = torch.softmax(pi_logits[self.current_option, :], -1)

        # Sample action from policy
        a = torch.multinomial(pi_w, 1)
        a = a.squeeze(-1).item()

        # Store log-prob for updates
        logprobs = torch.log_softmax(pi_logits[self.current_option, :], -1)

        # Get entropy
        probs = logprobs.exp()
        entropy = (-(logprobs * probs)).sum(-1)

        return a, beta_w, logprobs, entropy, Q

    def get_features(self, state):
        state = Variable(torch.from_numpy(state).float())
        return self.state_features(state)

    def get_Q(self, features):
        return self.Q(features)

    def get_option(self, q, out):
        beta_w = self.termination(out).flatten()[self.current_option].softmax(dim=-1)
        termination_w = torch.bernoulli(beta_w)

        new_option= self.current_option
        # Change to e-greedy option if termination is > 0
        if termination_w > 0:
            if np.random.random() > self.eps:
                new_option = q[0].argmax(dim=-1).item()
            else:
                new_option = np.random.randint(0, self.num_options)

        return new_option, beta_w

    def get_termination_prob(self, next_s):
        features = self.get_features(next_s)
        beta_w = self.termination(features)[:, self.current_option].softmax(dim=-1)

        return beta_w

    def compute_td_target(self, reward, done, next_s):
        with torch.no_grad():
            features = self.get_features(next_s)
            q_next = self.get_Q(features).flatten()
            beta_w = self.get_termination_prob(next_s.reshape(1, next_s.shape[0]))[0]

            no_termination = (1 - beta_w) * q_next[self.current_option]
            termination = beta_w * q_next.max(dim=-1)[0]

            if done:
                return reward
            else:
                return reward + self.gamma * (no_termination + termination)

    def actor_loss(self, td_target, logprob, entropy, beta_w, q):

        with torch.no_grad():
            q_s = q.detach().flatten()
            v_s = q_s.max(-1)[0]
            q_sw = q_s[self.current_option]

        # Policy loss
        policy_loss = (-logprob * (td_target - q_sw)).mean()

        # Entropy loss
        entropy_loss = (-entropy).mean()

        # Termination loss
        termination_loss = (beta_w * (q_sw - v_s + self.term_reg)).mean()

        actor_loss = policy_loss + termination_loss + self.ent_coef * entropy_loss

        return actor_loss

    def critic_loss(self, data):

        states, options, rewards, next_states, dones = data
        batch_idx = torch.arange(len(options)).long()
        options = torch.LongTensor(options)
        rewards = torch.FloatTensor(rewards)
        masks = 1 - torch.FloatTensor(dones)

        term_probs = self.get_termination_prob(next_states).detach()

        states = self.get_features(states).squeeze(0)
        Q = self.get_Q(states)

        next_states = self.get_features(next_states).squeeze(0)
        next_Q = self.get_Q(next_states)

        target = rewards + masks * self.gamma \
                 * ((1 - term_probs) * next_Q[batch_idx, options]
                    + term_probs * next_Q.max(dim=-1)[0]
                    )

        loss = (Q[batch_idx, options] - target.detach()).pow(2).mul(0.5).mean()
        return loss
