import sys
import os
import tty
import termios
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent():
    """ Wrapper class for any time of agent """

    def __init__(self, agent_type, state_space, action_space):
        self.agent_type = agent_type 
        self.state_space = state_space
        self.action_space = action_space
        self.agent = None

        if self.agent_type == "table":
            self.agent = np.zeros((state_space, action_space))
        elif self.agent_type == "network":
            self.agent = QNetwork(state_space, action_space)

    def get_q_values(self, state):
        """ Get agent Q values at a state """
        if self.agent_type == "table":
            return self.agent[state, :]
        elif self.agent_type == "network":
            return self.agent(state).detach().numpy()[0]

    def get_action(self, state, explore_prob):
        """ Get agent actionat state given an explore probability """
        if self.agent_type == "table":
            return np.argmax(self.agent[state, :] + np.random.randn(1, self.action_space) * explore_prob)
        elif self.agent_type == "network":
            return self.agent(state).max(1)[0].view(1, 1)

# define Q-Network
class QNetwork(nn.Module):
    """ Simple Q network """

    def __init__(self, state_space, action_space):
        super(QNetwork, self).__init__()
        self.state_space = state_space
        self.hidden_size = state_space

        self.l1 = nn.Linear(in_features=self.state_space, out_features=self.hidden_size)
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=action_space)

    def forward(self, x):
        """ Model inference  """
        x = self.one_hot_encoding(x)

        # Move to device if needed
        if next(self.parameters()).is_cuda:
            x = x.cuda()
            
        out1 = torch.sigmoid(self.l1(x))
        return self.l2(out1) 

    def one_hot_encoding(self, x):
        """ One-hot encodes the input data, based on the defined state_space. """
        out_tensor = torch.zeros([1, self.state_space])
        out_tensor[0][x] = 1
        return out_tensor