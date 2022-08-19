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

MODEL_TYPE = "q_table"
ENV_NAME = "FrozenLake-v1"

# Load configs
with open("configs/agent_configs.json", "rb") as f:
    agent_config = json.load(f)

with open("configs/train_configs.json", "rb") as f:
    train_config = json.load(f)

# Get gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init Game
env = gym.make(ENV_NAME, is_slippery=False)
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

else:
    # Train Q Table
    Q, losses, rewards = get_trained_q_table(
        env, **agent_config["basic_q_table_config"], **train_config["basic_train_config"]
    )

end = time.time()

print("Done in {:.2f} seconds".format(end - start))

print("Average Score:", str(sum(rewards) / train_config["basic_train_config"]["num_episodes"]))

# Log run metrics
save_logs(losses, rewards, experiment_name="", save_plots=False)
