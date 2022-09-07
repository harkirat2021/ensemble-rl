import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.game_utils import *
from src.training import *
from src.models import *
from src.utils import *

# Define ensemble net
class EnsembleQNetwork(nn.Module):
    def __init__(self, agents, state_space, action_space, trajectory_depth, weighted_actions):
        super(EnsembleQNetwork, self).__init__()
        
        self.agents = agents
        self.state_space = state_space
        self.hidden_size = state_space
        self.weighted_actions = weighted_actions

        # Image = (trajectory_depth, state_space), Num channels = num_agents
        self.conv1 = nn.Conv2d(
            len(agents), 1, kernel_size=(trajectory_depth, state_space), padding='same'
        )

        # Fully connected hidden layer
        self.fc1 = nn.Linear(
            in_features=self.state_space * trajectory_depth, out_features=self.hidden_size
        )

        # Output layer
        self.fc2 = nn.Linear(
            in_features=self.hidden_size,
            out_features=len(agents) if weighted_actions else action_space
        )

    """ Generate agent plans """
    def get_agent_trajectory(self, agent, game, state, n, max_steps):
        # Get Q values for action and convert to softmax
        actions_qs = agent.get_q_values(state)
        actions_probs = actions_qs / np.sum(actions_qs)

        # Safety check, if all actions return 0 probability, make action probabillity uniform
        if np.sum(actions_qs) == 0:
            actions_probs = np.ones(game.action_space.n) / 4

        # Set probabilities for new state
        state_prob = np.zeros(game.observation_space.n)

        # Terminate if we reached the max depth
        if n == max_steps - 1:
            return np.array([state_prob])

        sub_trajectories = []
        for i in range(game.action_space.n):
            # Create copy of game state - TODO - need to set same params
            game = Game(living_penalty=-0.04, render=False)
            game = set_game_state(game, state)

            # Apply action
            new_state, reward, game_over, info = game.step(i)

            # Get sub trajectories - recurse
            sub_trajectory = self.get_agent_trajectory(agent, game, new_state, n+1, max_steps)

            # Add sub trajectory
            sub_trajectories.append(sub_trajectory)

        # Return added sub trajectories togethor scaled by action probs
        return np.sum([a * t for a, t in zip(actions_probs, np.array(sub_trajectories))], axis=0)

    """ Generate multiple agent plans """
    def get_multi_agents_trajectories(self, game, state, max_steps):
        trajectories = []
        for a in self.agents:
            trajectories.append(self.get_agent_trajectory(a, game=game, state=state, n=0, max_steps=max_steps))

        return np.stack(trajectories)

    """ Forward pass """
    def forward(self, state, game, trajectory_depth=3, method="ensemble", device=None):
        if method == "ensemble":
            if self.weighted_actions:
                # Input as state id
                x = self.get_multi_agents_trajectories(
                    game, state, max_steps=trajectory_depth
                )
                x = torch.unsqueeze(torch.tensor(x).float(), dim=0)

                # Input = (batch, num agents, trajectory_depth, state_space)
                x = self.conv1(x)
                x = x.flatten(1)
                x = self.fc1(x)
                x = self.fc2(x)

                # Get weighted average of actions
                agent_actions = torch.tensor(np.array([a.get_q_values(state) for a in self.agents]))
                if device:
                    agent_actions.to(device)

                return torch.matmul(torch.sigmoid(x).float(), agent_actions.float())
            else:
                # Input as state id
                x = self.get_multi_agents_trajectories(
                    game, state, max_steps=1
                )
                x = torch.unsqueeze(torch.tensor(x).float(), dim=0)

                # Input = (batch, num agents, trajectory_depth, state_space)
                x = self.conv1(x)
                x = x.flatten(1)
                x = self.fc1(x)
                x = self.fc2(x)

                return torch.sigmoid(x)
        else:
            max_action = -1
            max_reward = -1
            for a in self.agents:
                game = Game(living_penalty=-0.04, render=False)
                game = set_game_state(game, state)
                action_qs = a.get_q_values(state)

                reward, _, _ = game.perform_action(np.argmax(actions_qs))
                if max_action_reward < reward:
                    max_reward  = reward
                    max_action = np.argmax(actions_qs)
            return np.zeros(self.action_size)



    def one_hot_encoding(self, x):
        '''
        One-hot encodes the input data, based on the defined state_space.
        '''
        out_tensor = torch.zeros([1, state_space])
        out_tensor[0][x] = 1
        return out_tensor