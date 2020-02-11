import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from expreplayTorch import ReplayBuffer
from DQNModelTorch import DQN

class Trainer(object):
    def __init__(self,
                 env,
                 image_size = (45, 45, 45),
                 update_frequency = 4,
                 replay_buffer_size = 5000,
                 max_episodes = 20,
                 steps_per_epoch = 50,
                 eps = 1,
                 delta = 0.05,
                 batch_size = 4,
                 gamma = 0.9,
                 number_actions = 6,
                 frame_history = 4
                ):
        self.env = env
        self.agents = env.agents
        self.image_size = image_size
        self.update_frequency = update_frequency
        self.replay_buffer_size = replay_buffer_size
        self.max_episodes = max_episodes
        self.steps_per_epoch = steps_per_epoch
        self.eps = eps
        self.delta = delta
        self.batch_size = batch_size
        self.gamma = gamma
        self.number_actions = number_actions
        self.frame_history = frame_history
        self.buffer = ReplayBuffer(self.replay_buffer_size)
        self.dqn = DQN(self.batch_size, self.agents)

    def train(self):
        losses = []
        episode = 1
        # Create a DQN (Deep Q-Network)

        print("number of agents", self.agents)
        # Loop over episodes
        while episode <= self.max_episodes:
            print("episode ", episode, ", eps", self.eps)
            #print("losses", losses)

            # Reset the environment for the start of the episode.
            obs = self.env.reset()
            # Loop over steps within this episode. The episode length here is 20.

            terminal = [False for _ in range(self.agents)]
            for step_num in range(self.steps_per_epoch):
                acts, q_values = self.get_next_actions(obs)
                # Step the agent once, and get the transition tuple for this step
                #print("acts, q_values", acts, q_values)
                next_obs, reward, terminal, info = self.env.step(acts, q_values.data.numpy(), terminal)
                #print("obs, reward, terminal, info", obs.shape, reward, terminal, info)
                self.buffer.add(obs/255, acts, reward, next_obs, terminal)
                if len(self.buffer) >= self.batch_size:
                    mini_batch = self.buffer.sample(self.batch_size)
                    loss = self.dqn.train_q_network(mini_batch, self.gamma)
                    #loss = 1
                    print("loss:", loss)
                    losses.append(loss)
                obs = next_obs
                if all(t for t in terminal):
                    break
            if episode % self.update_frequency == 0:
                self.dqn.copy_to_target_network()
            self.eps = max(0, self.eps-self.delta)
            episode += 1
        plt.plot(list(range(len(losses))), losses)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training")
        plt.yscale('log')
        plt.show()

    # Function to get the next action, using whatever method you like
    def get_next_actions(self, obs):
        # epsilon-greedy policy
        q_values = self.dqn.q_network.forward(torch.tensor(obs).flatten())
        if np.random.random() < self.eps:
            actions = [np.random.choice(6) for _ in range(obs.shape[0])]
            #q_values = actions.detach().squeeze()
        else:
            actions = self.get_greedy_actions(obs, doubleLearning=True)
        return actions, q_values

    def get_greedy_actions(self, obs, doubleLearning = True):
        inputs = torch.tensor(obs).flatten()
        if doubleLearning:
            vals = self.dqn.q_network.forward(inputs).detach()
        else:
            vals = self.dqn.target_network.forward(inputs).detach()
        idx = torch.max(vals, 1)[1]
        greedy_steps = np.array(idx, dtype = np.float32)
        # The actions are scaled for better training of the DQN
        return greedy_steps
