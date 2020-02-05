import torch
import torch.nn as nn
import numpy as np
import collections

def train(env):
    # Create a DQN (Deep Q-Network)
    buffer = ReplayBuffer()
    dqn = DQN(env.agents*45*45*45, env.agents*6, buffer)
    episode = 1
    max_episodes = 10
    episode_duration = 50
    losses = []
    eps = 1
    # Loop over episodes
    while episode <= max_episodes:
        print("episode ", episode, ", eps", eps)
        #print("losses", losses)

        # Reset the environment for the start of the episode.
        obs = env.reset()
        print("obs", obs.shape)
        # Loop over steps within this episode. The episode length here is 20.

        terminal = [False for _ in range(env.agents)]
        for step_num in range(episode_duration):
            acts, q_values = get_next_actions(obs, dqn, eps)
            # Step the agent once, and get the transition tuple for this step
            print("acts, q_values", acts, q_values)
            obs, reward, terminal, info = env.step(acts, q_values.data.numpy(), terminal)
            print("obs, reward, terminal, info", obs.shape, reward, terminal, info)
            #loss = dqn.train_q_network(transition)
            #losses.append(loss)
        eps = max(0, eps-0.05)
        episode += 1

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


class ReplayBuffer:

    # TODO: implement binary heap for faster sampling
    def __init__(self, capacity = 3000):
        self.de = collections.deque(maxlen = capacity)
        # Prevents edge case where TD error is 0 and transition never gets sampled
        self.prob_epsilon = 0.0001
        # Factor to accentuate the effect of the prioritised experience replay buffer, 0 is uniform sampling
        self.alpha = 0.5

    def add(self, transition):
        maxProb = max([t[4] for t in self.de], default = 1)
        transition[4] = maxProb
        self.de.append(transition)

    def sample_batch(self, batch_size, replace = False):
        tot = sum([t[4] for t in self.de])
        probs = [t[4]/tot for t in self.de]
        indexes = np.random.choice(len(self.de), batch_size, replace = replace, p = probs)
        return [self.de[i] for i in indexes]

    def recompute_weights(self, transitions, td_errors):
        for i in range(len(transitions)):
            transitions[i][4] = (abs(td_errors[i].item()) + self.prob_epsilon)**self.alpha

    def size(self):
        return len(self.de)

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
        # Inputs are a 1x4 tensor of the agent position and movement
        batch_inputs = torch.tensor([np.append(t[0], t[1]) for t in transitions])
        # Labels are the rewards
        batch_labels = torch.tensor([t[2] for t in transitions])
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
