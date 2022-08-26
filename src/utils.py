import os
import matplotlib.pyplot as plt

def save_logs(agent, losses, rewards, experiment_name="", experiment_number=0, save_plots=False):
    """ Plot training rewards and losses"""
    if save_plots:
        if not os.path.exists("experiments/{}/".format(experiment_name, experiment_number)):
            os.mkdir("experiments/{}/".format(experiment_name, experiment_number))
        if not os.path.exists("experiments/{}/experiment_{}".format(experiment_name, experiment_number)):
            os.mkdir("experiments/{}/experiment_{}".format(experiment_name, experiment_number))
    
    # Save agent
    if save_plots:
        agent.save("experiments/{}/experiment_{}".format(experiment_name, experiment_number))

    # Save plot
    plt.plot(rewards)
    plt.title("Rewards")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    if save_plots:
        plt.savefig("experiments/{}/experiment_{}/rewards.png".format(experiment_name, experiment_number))
    else:
        plt.show()

    plt.cla()
    plt.clf()

    plt.plot(losses)
    plt.title("Losses")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    if save_plots:
        plt.savefig("experiments/{}/experiment_{}/losses.png".format(experiment_name, experiment_number))
    else:
        plt.show()
