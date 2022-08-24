import sys
import os
import tty
import termios
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def set_game_state(env, state):
    """ Set the state of the environment """
    if type(env) == Game:
        env.set_state(state)
        return env
    elif type(env) == gym.Env:
        env.state = env.unwrapped.state = state
        env.steps_beyond_done = None
        return env
    else:
        print("Environment not recognizable")
        return

class GetKey:
    """ Get keyboard input """
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def get_manual_arrow_key():
    """ Get arrow key inputs """
    inkey = GetKey()
    while(1):
        k = inkey()
        if k != '':
            break
    if k == '\x1b[A':
        return 'up'
    elif k == '\x1b[B':
        return 'down'
    elif k == '\x1b[C':
        return 'right'
    elif k == '\x1b[D':
        return 'left'

# define the map
MAPS = {
    "simple": [
        "SFFFFFFF",
        "FFFFFFFH",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "fl1": [
        "SHFFFHFF",
        "FFFHFFFH",
        "FHFHFFFF",
        "FHFFFHFF",
        "FFFHFFHF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

# define some colors for terminal output
RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"

class Game():
    """ Custom game class - Essentially same as frozen lake """
    def __init__(self, living_penalty=0, map_size='8x8', render=True):
        map_layout = np.asarray(MAPS[map_size], dtype='c')
        self.max_row, self.max_col = map_layout.shape

        self.state_space = self.max_row * self.max_col
        self.observation_space = namedtuple('ObservationSpace', 'n')
        self.observation_space.n = self.state_space

        self.actions = ('left', 'down', 'right', 'up')
        self.action_space = namedtuple('ActionSpace', 'n')
        self.action_space.n = 4

        map_layout = map_layout.tolist()
        self.map_layout = [[c.decode('utf-8') for c in line] for line in map_layout]
        self.x_pos, self.y_pos = 0, 0
        self.lastaction = None
        self.game_over = False
        self.render = render
        self.living_penalty = living_penalty

        if self.render:
            self.rendering()

    def reset(self):
        ''' This method resets the game. '''
        self.x_pos, self.y_pos = 0, 0
        self.lastaction = None
        self.game_over = False
        return self.get_state()

    def get_reward(self):
        ''' Returns the reward of the current player position and sets the game_over boolean. '''
        label = self.map_layout[self.x_pos][self.y_pos]
        if label in 'FS':
            return self.living_penalty
        if label in 'H':
            self.game_over = True
            if self.render:
                sys.stdout.write(RED)
                sys.stdout.write('\n Agent walked into a hole! Game Over!')
            return -1
        if label in 'G':
            self.game_over = True
            if self.render:
                sys.stdout.write('\n Agent walked into the goal! You won the Game!')
            return 1

    def get_state(self):
        ''' Returns the current state of the player '''
        return (self.x_pos * 8) + self.y_pos
    
    def set_state(self, state_val):
        self.x_pos = state_val // 8
        self.y_pos = state_val % 8

    def step(self, action):
        '''
        Performs the given action in the environment. 
        
        action -- String representation of the action, e.g. left, down, right or up.
        '''
        if action == 'left':
            self.x_pos = max(self.x_pos, 0)
            self.y_pos = max(self.y_pos-1, 0)
        elif action == 'down':
            self.x_pos = min(self.x_pos+1, self.max_row-1)
            self.y_pos = max(self.y_pos, 0)
        elif action == 'right':
            self.x_pos = max(self.x_pos, 0)
            self.y_pos = min(self.y_pos+1, self.max_col-1)
        elif action == 'up':
            self.x_pos = max(self.x_pos-1, 0)
            self.y_pos = max(self.y_pos, 0)
        self.lastaction = action
        if self.render:
            self.rendering()

        return self.get_state(), self.get_reward(), self.game_over, {} # Info, not added yet

    def rendering(self):
        ''' This method renders the game in the console. For each step the console will be cleared and then reprinted '''
        os.system('cls||clear')
        if self.lastaction is not None:
            sys.stdout.write(BLUE)
            sys.stdout.write("  ({})\n".format(self.lastaction))
        else:
            sys.stdout.write("\n")

        for idx_r, r in enumerate(self.map_layout):
            for idx_c, c in enumerate(r):
                sys.stdout.write(RESET)
                if c in 'H':
                    sys.stdout.write(RED)
                if c in 'G':
                    sys.stdout.write(GREEN)
                if idx_r == self.x_pos and idx_c == self.y_pos:
                    sys.stdout.write(BLUE)
                sys.stdout.write(c + ' ')
            sys.stdout.write('\n')

# define actions
actions = ('left', 'down', 'right', 'up')


def print_policy(agent, game):
    """ Print agent policy """
    policy = [agent(s).argmax(1)[0].detach().item() for s in range(state_space)]
    policy = np.asarray([actions[action] for action in policy])
    policy = policy.reshape((game.max_row, game.max_col))
    print("\n\n".join('\t'.join(line) for line in policy) + "\n")