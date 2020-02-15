import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from expreplayTorch import ReplayBuffer
from DQNModelTorch import DQN

class Trainer(object):
    def __init__(self,
                 env,
                 image_size = (45, 45, 45),
                 update_frequency = 4,
                 replay_buffer_size = 1e6,
                 init_memory_size = 5e4,
                 max_episodes = 20,
                 steps_per_epoch = 50,
                 eps = 1,
                 min_eps = 0.1,
                 delta = 0.05,
                 batch_size = 4,
                 gamma = 0.9,
                 number_actions = 6,
                 frame_history = 4,
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
        self.dqn = DQN(self.batch_size, self.agents, self.frame_history, type="Network3d")

    def train(self):
        losses = []
        episode = 1
        # Create a DQN (Deep Q-Network)

        print("number of agents", self.agents)
        # Loop over episodes
        while episode <= self.max_episodes:
            print("episode", episode, "- eps", self.eps)
            #print("losses", losses)

            # Reset the environment for the start of the episode.
            obs = self.env.reset()
            # Observations stacks is a numpy array with shape (agents, frame_history, *image_size)
            obs_stack = np.stack([obs] * self.frame_history, axis=1)
            # Loop over steps within this episode. The episode length here is 20.

            terminal = [False for _ in range(self.agents)]
            for step_num in range(self.steps_per_epoch):
                acts, q_values = self.get_next_actions(obs_stack)
                # Step the agent once, and get the transition tuple for this step
                #tuple(map(tuple, q_values.data.numpy()))
                next_obs, reward, terminal, info = self.env.step(acts, q_values.data.numpy(), terminal)
                #print("obs, reward, terminal, info", obs.shape, reward, terminal, info)
                next_obs_stack = np.concatenate((obs_stack[:,1:], np.expand_dims(next_obs, axis=1)), axis=1)
                self.buffer.add(obs_stack/255, acts, reward, next_obs_stack/255, terminal)
                if len(self.buffer) >= self.init_memory_size:
                    mini_batch = self.buffer.sample(self.batch_size)
                    loss = self.dqn.train_q_network(mini_batch, self.gamma)
                    print("loss:", loss)
                    losses.append(loss)
                obs = next_obs
                obs_stack = next_obs_stack
                if all(t for t in terminal):
                    break
            if episode % self.update_frequency == 0:
                self.dqn.copy_to_target_network()
            self.eps = max(self.min_eps, self.eps-self.delta)
            episode += 1
        plt.plot(list(range(len(losses))), losses)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training")
        plt.yscale('log')
        plt.show()
        torch.save(model.state_dict(), "data/models/test_models/model_test_" + str(int(time.time())) +".pt")

    # Function to get the next action, using whatever method you like
    def get_next_actions(self, obs_stack):
        # epsilon-greedy policy
        q_values = self.dqn.q_network.forward(torch.tensor(obs_stack).unsqueeze(0)).view(self.agents, self.number_actions)
        if np.random.random() < self.eps:
            actions = np.random.randint(6, size=self.agents)
            #print("exploratory choice", actions)
        else:
            actions = self.get_greedy_actions(obs_stack, doubleLearning=True)
            #print("greedy choice", actions)
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
        return greedy_steps
