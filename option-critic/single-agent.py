import torch
import torch.nn as nn


class AgentNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_actions,
                 num_options,
                 hidden_dim,
                 rnn_layers
                 ):
        super(AgentNetwork, self).__init__()

        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.current_option = 0
        self.num_steps = 0

        self.state_features = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.policy = nn.Linear(128, self.num_actions * self.num_options)
        self.termination = nn.Linear(128, self.num_options)

    def step(self, obs, Q, is_done):
        """Take in observation from env and do 1 forward pass through network.
            Return: action, option termination
        """

        # Passing in the input and hidden state into the model and obtaining outputs
        out = self.state_features(obs)

        # Get option policy and termination policy over softmax
        option, beta_w = self.get_option(Q, out)

        pi_w = torch.softmax(self.policy(out).reshape(-1, self.num_options, self.num_actions)[option, :], -1)
        a = torch.multinomial(pi_w, 1)

        return a.squeeze(-1), beta_w

    def get_state(self, obs):
        return self.state_features(obs)

    def get_option(self, Q, out):
        beta_w = self.softmax(self.terminations(out)[self.current_option])
        termination_w = torch.bernoulli(beta_w)
        # Change to greedy option if termination is > 0
        if termination_w > 0:
            self.current_option = Q.argmax(dim=-1).item()
        return self.current_option, beta_w


