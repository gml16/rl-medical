import torch
import torch.nn as nn
import numpy as np
import collections
import matplotlib.pyplot as plt

def train(env):
    # Create a DQN (Deep Q-Network)
    replay_buffer_size = 5000
    buffer = ReplayBuffer(replay_buffer_size)
    dqn = DQN(env.agents*45*45*45, env.agents*6, buffer)
    episode = 1
    max_episodes = 10
    episode_duration = 50
    losses = []
    eps = 1
    delta = 0.05
    batch_size = 4
    gamma = 0.9
    # Loop over episodes
    while episode <= max_episodes:
        print("episode ", episode, ", eps", eps)
        #print("losses", losses)

        # Reset the environment for the start of the episode.
        obs = env.reset()
        # Loop over steps within this episode. The episode length here is 20.

        terminal = [False for _ in range(env.agents)]
        for step_num in range(episode_duration):
            acts, q_values = get_next_actions(obs, dqn, eps)
            # Step the agent once, and get the transition tuple for this step
            #print("acts, q_values", acts, q_values)
            next_obs, reward, terminal, info = env.step(acts, q_values.data.numpy(), terminal)
            #print("obs, reward, terminal, info", obs.shape, reward, terminal, info)
            buffer.add(obs, acts, reward, next_obs, terminal)
            if len(buffer) >= batch_size:
                eps -= delta
                mini_batch = buffer.sample(batch_size)
                print("mini batch shape", len(mini_batch))
                loss = dqn.train_q_network(mini_batch, gamma)
                losses.append(loss)
            obs = next_obs
        eps = max(0, eps-0.05)
        episode += 1
        plt.plot(list(range(len(losses))), losses)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training")
        plt.yscale('log')

# Function to get the next action, using whatever method you like
def get_next_actions(obs, dqn, eps):
    # epsilon-greedy policy
    q_values = dqn.q_network.forward(torch.tensor(obs).flatten())
    if np.random.random() < eps:
        actions = [np.random.choice(6) for _ in range(obs.shape[0])]
        #q_values = actions.detach().squeeze()
    else:
        actions = get_greedy_actions(obs, dqn, doubleLearning=True)
    return actions, q_values

def get_greedy_actions(obs, dqn, doubleLearning = True):
    inputs = torch.tensor(obs).flatten()
    if doubleLearning:
        vals = dqn.q_network.forward(inputs).detach()
    else:
        vals = dqn.target_network.forward(inputs).detach()
    idx = torch.max(vals, 1)[1]
    greedy_steps = np.array(idx, dtype = np.float32)
    # The actions are scaled for better training of the DQN
    return greedy_steps


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        print("obses_t shape", obses_t.shape)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class Network(nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = nn.functional.relu(self.layer_1(input))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output.view(-1, 6)


class DQN:
    # The class initialisation function.
    def __init__(self, input_dimension, output_dimension, buffer):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension, output_dimension)
        self.target_network = Network(input_dimension, output_dimension)
        self.best_network = Network(input_dimension, output_dimension)
        self.copy_to_target_network()
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.RMSprop(self.q_network.parameters(), lr=0.0004, momentum=0)
        self.losses = []
        self.buffer = buffer

    def copy_to_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transitions, discount_factor):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transitions, discount_factor)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Record the loss for visualisation
        # self.losses.append(loss.item())
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, discount_factor):
        batch_inputs = torch.tensor(transitions[0])
        # Labels are the rewards
        batch_labels = torch.tensor(transitions[2])
        get_greedy = torch.tensor([np.append(t[0], pa) for pa in self.agent.possible_actions for t in transitions])
        y = self.target_network.forward(get_greedy).detach().squeeze().resize_((self.agent.batch_size,len(self.agent.possible_actions)))
        # Get the maximum prediction for the next state from the target network
        max_target_net = y.max(1)[0]
        network_prediction = self.q_network.forward(batch_inputs).squeeze()
        # Bellman equation
        batch_labels_tensor = batch_labels + discount_factor * max_target_net
        td_errors = (network_prediction - batch_labels_tensor).detach()
        # Update transitions' weights
        self.buffer.recompute_weights(transitions, td_errors)
        return torch.nn.MSELoss()(network_prediction, batch_labels_tensor)
