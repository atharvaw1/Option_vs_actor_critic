import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical


class OptionCriticAgent(nn.Module):
    def __init__(self, in_features, num_actions, num_options, num_agents, device, test=False):

        super().__init__()

        self.device = device
        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.num_agents = num_agents
        self.current_options = Variable(
            torch.from_numpy(np.random.randint(num_options, size=num_agents)).long()).unsqueeze(1).to(self.device)
        self.gamma = 0.95
        self.eps_min = 0.01
        self.eps_start = 1.0
        self.num_steps = 0
        self.duration = 150_000

        self.deliberation_cost = 0.001
        self.termination_counts = 0

        self.ent_coef = 0.0
        self.term_reg = 0.001
        self.val_coef = 1

        self.test = test

        self.target_net_features = nn.Sequential(
            nn.Linear(self.in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )

        self.state_features = nn.Sequential(
            nn.Linear(self.in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )

        self.Q = nn.Linear(64, self.num_agents * self.num_options)
        self.target_q = nn.Linear(64, self.num_agents * self.num_options)
        self.policy = nn.Linear(64, self.num_agents * self.num_options * self.num_actions)
        self.termination = nn.Linear(64, self.num_agents * self.num_options)

    def forward(self, state):
        """Take in observation from env and do 1 forward pass through network.
            Return: action, option termination
        """
        state = Variable(torch.from_numpy(state).float().unsqueeze(0).to(self.device))
        # Passing in the input and hidden state into the model and obtaining outputs
        out = self.state_features(state)

        # Get Q
        q = self.get_Q(out)

        # Get option policy and termination policy over softmax
        beta_w, self.termination_counts = self.get_options(q, out)

        # Get policy from policy network
        pi_logits = self.policy(out).reshape(self.num_agents, self.num_options, self.num_actions)

        # Get current options for all agents and apply softmax
        pi_w = torch.empty(self.num_agents, self.num_actions)
        for i in range(pi_logits.shape[0]):
            pi_w[i, :] = (pi_logits[i, self.current_options[i][0], :])

        pi_w = torch.Tensor(pi_w).softmax(-1)
        # pi_w = pi_logits[torch.arange(len(pi_logits)), self.current_options.view(-1)].softmax(-1)

        # Sample actions from distribution
        actions = torch.multinomial(pi_w, 1)

        # Get log probabilities and entropy for gradients
        logprobs = pi_w.log_softmax(-1)
        logprob = logprobs.gather(-1, actions).squeeze(-1)

        probs = logprobs.exp()
        entropy = (-logprobs * probs).sum(-1)

        return actions.cpu().numpy(), self.current_options, beta_w, logprob, entropy, q

    def get_features(self, state):
        state = Variable(torch.from_numpy(state).float().to(self.device))
        return self.state_features(state)

    def get_Q(self, features):
        return self.Q(features)

    def get_targetQ(self, state):
        state = Variable(torch.from_numpy(state).float().to(self.device))
        features = self.target_net_features(state)
        return self.target_q(features)

    def get_options(self, q, out):

        beta_w = self.termination(out) \
            .reshape(self.num_agents, self.num_options) \
            .gather(1, self.current_options) \
            .sigmoid()

        termination_w = torch.bernoulli(beta_w)
        termination_count = termination_w.count_nonzero()
        # Change to e-greedy option if termination is > 0
        potential_options = torch.where(torch.rand_like(self.current_options, dtype=torch.float32).squeeze() < self.eps,
                                        torch.randint_like(self.current_options.squeeze(), 0, self.num_options),
                                        q.reshape(self.num_agents, self.num_options).argmax(-1))

        self.current_options = torch.where(termination_w.squeeze() > 0,
                                           potential_options,
                                           self.current_options.squeeze()).unsqueeze(1)

        return beta_w, termination_count

    def get_termination_prob(self, next_s):

        features = self.get_features(next_s)
        beta_w = self.termination(features).reshape(next_s.shape[0], self.num_agents, self.num_options) \
            [:, torch.arange(self.num_agents), self.current_options.view(-1)].sigmoid()
        return beta_w

    def compute_td_target(self, reward, done, next_s, beta_w):
        with torch.no_grad():
            reward = torch.Tensor(reward).unsqueeze(-1).to(self.device)
            features = self.get_features(next_s)
            q_next = self.get_Q(features).reshape(self.num_agents, self.num_options)
            no_termination = ((1 - beta_w) * q_next.gather(1, self.current_options))
            termination = (beta_w.squeeze() * q_next.max(-1)[0]).unsqueeze(-1)
            return reward + self.gamma * (no_termination + termination) * (1 - done)

    def actor_loss(self, td_target, logprob, entropy, beta_w, q):

        with torch.no_grad():
            q_s = q.detach().reshape(self.num_agents, self.num_options)
            v_s = q_s.max(-1)[0]
            q_sw = q_s.gather(1, self.current_options).squeeze()

        # Policy loss
        policy_loss = (-logprob.to(self.device) * (td_target.squeeze().detach() - q_sw))

        # Entropy loss
        entropy_loss = -entropy.to(self.device)

        # Termination loss
        termination_loss = (beta_w.squeeze() * (q_sw - v_s)) + (self.termination_counts * self.deliberation_cost)

        actor_loss = policy_loss + termination_loss + self.ent_coef * entropy_loss

        return actor_loss

    def critic_loss(self, data):

        states, options, rewards, next_states, dones = data
        options = torch.LongTensor(np.array(options)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).reshape(-1,self.num_agents).to(self.device)
        masks = (1 - torch.LongTensor(dones)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            term_probs = self.get_termination_prob(next_states).detach()

        q = self.get_Q(self.get_features(states)).reshape(-1, self.num_agents, self.num_options)

        # Target network for bootstrap
        next_q = self.get_targetQ(next_states).reshape(-1, self.num_agents, self.num_options)
        # next_q = self.get_Q(self.get_features(next_states)).reshape(-1, self.num_agents, self.num_options)

        target = rewards + masks * self.gamma \
                 * ((1 - term_probs) * next_q.gather(2, options).reshape(-1, self.num_agents)
                    + term_probs * next_q.max(dim=-1)[0]
                    )

        loss = self.val_coef * (q.gather(2, options).reshape(-1, self.num_agents) - target.detach()).pow(2).mean(0)

        return loss

    def update_target_net(self):
        self.target_net_features.load_state_dict(self.state_features.state_dict())
        self.target_q.load_state_dict(self.Q.state_dict())

    @property
    def eps(self):

        """Linear decay for epsilon"""
        self.num_steps += 1

        if self.num_steps > self.duration or self.test:
            return self.eps_min
        else:
            return self.eps_start + (((self.eps_min - self.eps_start) / self.duration) * self.num_steps)
