import os
import time
import torch
import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, directory, write, save_freq = 10):
        self.parent_dir = directory
        self.write = write
        self.dir = ""
        self.fig_index = 0
        self.model_index = 0
        self.save_freq = save_freq
        if self.write:
            self.create_dir()

    def plot_res(self, losses, distances):
        if len(losses) == 0 or not self.write:
            return

        plt.subplot(211)
        plt.plot(list(range(len(losses))), losses, color='orange')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training")
        plt.yscale('log')
        plt.subplot(212)
        for dist in distances:
            plt.plot(list(range(len(dist))), dist)
        plt.xlabel("Steps")
        plt.ylabel("Distance change")
        plt.title("Training")

        if self.fig_index > 0:
            os.remove(os.path.join(self.dir, f"res{self.fig_index-1}.png"))
        plt.savefig(os.path.join(self.dir, f"res{self.fig_index}.png"))
        self.fig_index+=1

    def log(self, message):
        print(str(message))
        if self.write:
            with open(os.path.join(self.dir, "logs.txt"), "a") as logs:
                logs.write(str(message) + "\n")

    def create_dir(self):
        f = None
        index = 0
        while os.path.isdir(os.path.join(self.parent_dir, f"experiment_{index}")):
            index += 1
        self.dir = os.path.join(self.parent_dir, f"experiment_{index}")
        os.makedirs(self.dir)
        with open(os.path.join(self.dir, "logs.txt"), "a") as logs:
            logs.write(f"Logs from {str(time.time())} \n")

    def save_model(self, state_dict):
        if not self.write:
            return
        if self.model_index > 0 and self.model_index % self.save_freq != 1:
                os.remove(os.path.join(self.dir, f"dqn{self.model_index-1}.pt"))
        torch.save(state_dict, os.path.join(self.dir, f"dqn{self.model_index}.pt"))
        self.model_index+=1
