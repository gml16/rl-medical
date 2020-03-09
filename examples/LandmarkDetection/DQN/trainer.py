import torch
import torch.nn as nn
import numpy as np
import time
from expreplayTorch import ReplayBuffer
from DQNModelTorch import DQN
import matplotlib.pyplot as plt
import os

class Trainer(object):
    def __init__(self,
                 env,
                 image_size = (45, 45, 45),
                 update_frequency = 4,
                 replay_buffer_size = 1e6,
                 init_memory_size = 5e4,
                 max_episodes = 100,
                 steps_per_epoch = 50,
                 eps = 1,
                 min_eps = 0.1,
                 delta = 0.001,
                 batch_size = 4,
                 gamma = 0.9,
                 number_actions = 6,
                 frame_history = 4,
                 file=None,
                ):
        self.env = env
        self.agents = env.agents
        self.image_size = image_size
        self.update_frequency = update_frequency
        self.replay_buffer_size = replay_buffer_size
        self.init_memory_size = init_memory_size
        self.max_episodes = max_episodes
        self.steps_per_epoch = steps_per_epoch
        self.eps = eps
        self.min_eps = min_eps
        self.delta = delta
        self.batch_size = batch_size
        self.gamma = gamma
        self.number_actions = number_actions
        self.frame_history = frame_history
        self.buffer = ReplayBuffer(self.replay_buffer_size)
        self.dqn = DQN(self.batch_size, self.agents, self.frame_history, type="Network2d")
        self.file = file
        self.figname = None

    def train(self):
        print(self.dqn.q_network)
        self.set_reproducible()
        losses = []
        distances = []
        episode = 1
        acc_steps = 0
        while episode <= self.max_episodes:
            print("episode", episode, "- eps", self.eps)
            if self.file is not None:
                self.file.write(f"episode {episode} - eps {self.eps} \n")

            # Reset the environment for the start of the episode.
            obs = self.env.reset()
            # Observations stacks is a numpy array with shape (agents, frame_history, *image_size)
            obs_stack = np.stack([obs] * self.frame_history, axis=1)
            # Loop over steps within this episode. The episode length here is 20.
            terminal = [False for _ in range(self.agents)]
            for step_num in range(self.steps_per_epoch):
                acc_steps += 1
                acts, q_values = self.get_next_actions(obs_stack)
                # Step the agent once, and get the transition tuple for this step
                next_obs, reward, terminal, info = self.env.step(acts, q_values, terminal)

                if step_num == 0:
                    start_dist = info['distError_0']

                next_obs_stack = np.concatenate((obs_stack[:,1:], np.expand_dims(next_obs, axis=1)), axis=1)
                self.buffer.add(obs_stack/255.0, acts, reward, next_obs_stack/255.0, terminal)
                if len(self.buffer) >= self.init_memory_size:
                    mini_batch = self.buffer.sample(self.batch_size)
                    # print("getting out of mini batch acts", mini_batch[1], "reward", mini_batch[2], "size", mini_batch[0].shape)
                    # for i in range(len(mini_batch[0])):
                    #     print("trans", i, "there are non zeros:", np.count_nonzero(mini_batch[0][i,0,0]), "-", np.count_nonzero(mini_batch[0][i,0,1]), "-",np.count_nonzero(mini_batch[0][i,0,2]), "-",np.count_nonzero(mini_batch[0][i,0,3]), "-")
                    loss = self.dqn.train_q_network(mini_batch, self.gamma)
                    self.eps = max(self.min_eps, self.eps-self.delta)
                    losses.append(loss)
                obs = next_obs
                obs_stack = next_obs_stack
                if all(t for t in terminal):
                    msg = f"Terminating episode after {step_num+1} steps, total of {acc_steps} steps, final distance is {info['distError_0']:.3f}, improved distance by {(start_dist-info['distError_0']):.3f}"
                    print(msg)
                    if self.file is not None:
                        self.file.write(msg + "\n")
                    break
            distances.append(start_dist-info['distError_0'])
            if episode % self.update_frequency == 0:
                self.dqn.copy_to_target_network()
            episode += 1
            self.plot_loss(losses, distances, self.file)
        self.dqn.save_model()
        file.close()


    # Function to get the next action, using whatever method you like
    def get_next_actions(self, obs_stack):
        # epsilon-greedy policy
        if np.random.random() < self.eps:
            q_values = self.dqn.q_network.forward(torch.tensor(obs_stack).unsqueeze(0))
            q_values = q_values.view(self.agents, self.number_actions).data.numpy()
            actions = np.random.randint(self.number_actions, size=self.agents)
        else:
            actions, q_values = self.get_greedy_actions(obs_stack, doubleLearning=True)
        return actions, q_values

    def get_greedy_actions(self, obs_stack, doubleLearning = True):
        inputs = torch.tensor(obs_stack).unsqueeze(0)
        if doubleLearning:
            vals = self.dqn.q_network.forward(inputs).detach()
        else:
            vals = self.dqn.target_network.forward(inputs).detach()
        idx = torch.max(vals, -1)[1]
        greedy_steps = np.array(idx, dtype = np.int32).flatten()

        # The actions are scaled for better training of the DQN
        return greedy_steps, vals.data.numpy()

    def plot_loss(self, losses, distances, file):
        if len(losses) == 0 or file is None:
            return

        if self.figname is not None:
            os.remove(os.path.join(os.path.dirname(__file__), os.path.normpath(self.figname + ".png")))
        self.figname = file.name.split(".")[0] + str(len(losses))


        plt.subplot(211)
        plt.plot(list(range(len(losses))), losses, color='orange')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training")
        plt.yscale('log')
        #plt.show()
        plt.subplot(212)
        plt.plot(list(range(len(distances))), distances, color='orange')
        plt.xlabel("Steps")
        plt.ylabel("Distance change")
        plt.title("Training")
        #plt.show()
        plt.savefig(self.figname)

    def set_reproducible(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
