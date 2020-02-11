import torch
import torch.nn as nn

class MLP(nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(MLP, self).__init__()
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

class Network3D(nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network3D, self).__init__()
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
    def __init__(self, batch_size, agents, number_actions = 6):
        self.batch_size = batch_size
        self.agents = agents
        self.number_actions = number_actions
        input_dimension = self.agents*45*45*45
        output_dimension = self.agents*self.number_actions
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = MLP(input_dimension, output_dimension)
        self.target_network = MLP(input_dimension, output_dimension)
        self.copy_to_target_network()
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.RMSprop(self.q_network.parameters(), lr=0.0004, momentum=0)

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
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, discount_factor):

        batch_inputs = torch.tensor(transitions[0]).view(self.batch_size, -1)

        # Labels are the rewards
        batch_labels = torch.tensor(transitions[2], dtype=torch.float32)
        #get_greedy = torch.tensor([np.append(t[0], pa) for pa in self.agent.possible_actions for t in transitions])

        y = self.target_network.forward(batch_inputs).detach().squeeze().view(self.batch_size, self.agents, self.number_actions)
        # Get the maximum prediction for the next state from the target network
        max_target_net = y.max(2)[0]
        network_prediction = self.q_network.forward(batch_inputs).view(self.batch_size, self.agents, self.number_actions)
        # Bellman equation
        batch_labels_tensor = batch_labels + (discount_factor * max_target_net)
        td_errors = (network_prediction - batch_labels_tensor.unsqueeze(2)).detach()

        index = torch.tensor(transitions[1], dtype=torch.long).unsqueeze(2)
        y_pred = (torch.gather(network_prediction, 2, index)).squeeze()

        # Update transitions' weights
        # self.buffer.recompute_weights(transitions, td_errors)
        return torch.nn.MSELoss()(y_pred, batch_labels_tensor)
