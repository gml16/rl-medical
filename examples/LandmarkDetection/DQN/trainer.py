import torch
import torch.nn as nn
import numpy as np
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
                 max_episodes = 100,
                 steps_per_epoch = 50,
                 eps = 1,
                 min_eps = 0.1,
                 delta = 0.001,
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
        self.dqn = DQN(self.batch_size, self.agents, self.frame_history, type="Network3d") #Network3d

    def train(self):
        losses = []
        episode = 1
        acc_steps = 0
        while episode <= self.max_episodes:
            print("episode", episode, "- eps", self.eps)
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
                next_obs_stack = np.concatenate((obs_stack[:,1:], np.expand_dims(next_obs, axis=1)), axis=1)
                self.buffer.add(obs_stack/255, acts, reward, next_obs_stack/255, terminal)
                if len(self.buffer) >= self.init_memory_size:
                    mini_batch = self.buffer.sample(self.batch_size)
                    loss = self.dqn.train_q_network(mini_batch, self.gamma)
                    self.eps = max(self.min_eps, self.eps-self.delta)
                    losses.append(loss)
                obs = next_obs
                obs_stack = next_obs_stack
                if all(t for t in terminal):
                    print(f"Terminating episode after {step_num+1} steps, total of {acc_steps} steps, final distance is {info['distError_0']}.")
                    break
            if episode % self.update_frequency == 0:
                self.dqn.copy_to_target_network()
            episode += 1
            self.plot_loss(losses)
        self.dqn.save_model()


    # Function to get the next action, using whatever method you like
    def get_next_actions(self, obs_stack):
        # epsilon-greedy policy
        q_values = self.dqn.q_network.forward(torch.tensor(obs_stack).unsqueeze(0))
        q_values = q_values.view(self.agents, self.number_actions).data.numpy()
        if np.random.random() < self.eps:
            actions = np.random.randint(self.number_actions, size=self.agents)
        else:
            actions = self.get_greedy_actions(obs_stack, doubleLearning=True)
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

    def plot_loss(self, losses):
        import matplotlib.pyplot as plt
        if len(losses) == 0:
            return
        plt.plot(list(range(len(losses))), losses, color='orange')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training")
        plt.yscale('log')
        #plt.show()
        plt.draw()
        plt.pause(0.001)
