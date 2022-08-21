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

def get_trained_q_table(game, e, lr, y, num_episodes):
    """ Train Q table """

    state_space = game.observation_space.n
    action_space = game.action_space.n

    # Initialize Q-table with all zeros (state_space, action_space)
    Q = np.zeros((state_space, action_space))

    # create lists to contain total rewards and per episode
    jList = []
    rList = []

    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = game.reset()
        rAll = 0
        j = 0

        # The Q-Table learning algorithm
        while j < 100:
            j += 1
            # Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1, action_space)*(1./(i+1)))

            # Get new state and reward from environment
            s1, r, game_over, info = game.step(a)
            # print(j, s1)

            # Update Q-Table with new knowledge implementing the Bellmann Equation
            Q[s,a] = Q[s,a] + lr* (r + y * (np.max(Q[s1,:]) - Q[s,a]) )

            # Add reward to list
            rAll += r

            # Replace old state with new
            s = s1

            if game_over:
                break

        jList.append(j)
        rList.append(rAll)
    
    return Q, jList, rList

def trained_q_net(game, Qnet, e, lr, y, num_episodes):
    """ Train deep Q learning model """

    # Define State and Action Space
    state_space = game.observation_space.n
    action_space = game.action_space.n

    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []

    # define optimizer and loss
    # optimizer = optim.SGD(agent.parameters(), lr=lr)
    optimizer = optim.Adam(params=Qnet.parameters())
    criterion = nn.SmoothL1Loss()

    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = game.reset()
        rAll = 0
        j = 0

        # The Q-Network learning algorithm
        while j < 99:
            j += 1

            # Choose an action by greedily (with e chance of random action) from the Q-network
            with torch.no_grad():
                # Do a feedforward pass for the current state s to get predicted Q-values
                # for all actions (=> agent(s)) and use the max as action a: max_a Q(s, a)
                a = Qnet(s).max(1)[1].view(1, 1)  # max(1)[1] returns index of highest value

            # e greedy exploration
            if np.random.rand(1) < e:
                a[0][0] = np.random.randint(1, 4)

            # Get new state and reward from environment
            # perform action to get reward r, next state s1 and game_over flag
            # calculate maximum overall network outputs: max_a’ Q(s1, a’).
            s1, r, game_over, info = game.step(a.item())

            # Calculate Q and target Q
            q = Qnet(s).max(1)[0].view(1, 1)
            q1 = Qnet(s1).max(1)[0].view(1, 1)

            with torch.no_grad():
                # Set target Q-value for action to: r + y max_a’ Q(s’, a’)
                target_q = r + y * q1

            # print(q, target_q)
            # Calculate loss
            loss = criterion(q, target_q)
            if j == 1 and i % 100 == 0:
                print("loss and reward: ", i, loss, r)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add reward to list
            rAll += r

            # Replace old state with new
            s = s1

            if game_over:
                # Reduce chance of random action as we train the model.
                e = 1. / ((i / 50) + 10)
                break
            
        rList.append(rAll)
        jList.append(j)

    return Qnet, jList, rList