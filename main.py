import os, sys
import numpy as np
import gym
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.game_utils import *
from src.training import *
from src.models import *
from src.utils import *
from src.ensemble import *

MODEL_TYPE = "ensemble"
# FrozenLake-v1
ENV_NAME = "Taxi-v3"

# Load configs
with open("configs/agent_configs.json", "rb") as f:
    agent_config = json.load(f)

with open("configs/train_configs.json", "rb") as f:
    train_config = json.load(f)

# Get gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init Game
env = gym.make(ENV_NAME)
#env = Game(living_penalty=-0.04, render=False)

env.reset()

print("Training model...")

start = time.time()
if MODEL_TYPE == "q_net":
    # Init Q Net
    Q = QNetwork(env.observation_space.n, env.action_space.n).to(device)

    # Train Q Net
    Q, losses, rewards = trained_q_net(
        env, Q, **agent_config["basic_q_net_config"], **train_config["basic_train_config"]
    )

if MODEL_TYPE == "q_table":
    # Train Q Table
    Q, losses, rewards = get_trained_q_table(
        env, **agent_config["basic_q_table_config"], **train_config["basic_train_config"]
    )

if MODEL_TYPE == "ensemble":
    print("defining")
    # Define dummy agents, both of which are good on two different parts of the game
    Q_dummy_a = np.zeros((64, 4))
    Q_dummy_a[:8, 2] = 1
    Q_dummy_a[[15, 23, 31, 39, 47, 55, 63], 1] = 1
    Q_dummy_a[16:23, 2] = 1

    Q_dummy_b = np.zeros((64, 4))
    Q_dummy_b[[0, 8, 16, 24, 32, 48, 56], 1] = 1

    print("runngn ensml")

    # Train Ensemble
    game = Game(living_penalty=-0.04, render=False)

    a1 = Agent(agent_type="table", state_space=64, action_space=4)
    a1.agent = Q_dummy_a

    a2 = Agent(agent_type="table", state_space=64, action_space=4)
    a2.agent = Q_dummy_b

    agent = EnsembleQNetwork(
        [a1, a2], state_space=64, action_space=4, trajectory_depth=1, weighted_actions=True
    )

    agent, losses, rewards = trained_q_net(
        game, agent, e=0.1, lr=0.05, y=0.999, num_episodes=20, training_ensemble=True
    )

end = time.time()

print("Done in {:.2f} seconds".format(end - start))

print("Average Score:", str(sum(rewards) / train_config["basic_train_config"]["num_episodes"]))

# Log run metrics
save_logs(losses, rewards, experiment_name="", save_plots=False)
