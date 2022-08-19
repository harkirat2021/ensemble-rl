import matplotlib.pyplot as plt

def save_logs(losses, rewards, experiment_name="", save_plots=False):
    plt.plot(rewards)
    plt.title("Rewards")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    if save_plots:
        plt.savefig("experiments/{}/rewards.png".format(experiment_name))
    else:
        plt.show()

    plt.plot(losses)
    plt.title("Losses")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    if save_plots:
        plt.savefig("experiments/{}/losses.png".format(experiment_name))
    else:
        plt.show()
