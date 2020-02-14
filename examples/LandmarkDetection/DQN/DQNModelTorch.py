import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(MLP, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = nn.Linear(in_features=100, out_features=100)
        self.output_layer = nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        input = input.view(-1, self.layer_1.in_features)
        layer_1_output = nn.functional.relu(self.layer_1(input))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output.view(-1, 6)

class Network3D(nn.Module):

    def __init__(self, in_channels, out_channels, agents, frame_history):
        super(Network3D, self).__init__()

        self.agents = agents
        self.frame_history = frame_history

        self.conv0 = nn.Conv3d(in_channels=frame_history, out_channels=32, kernel_size=(5,5,5))
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2,2,2))
        self.prelu0 = nn.PReLU()
        self.conv1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(5,5,5))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2,2,2))
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4,4,4))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2,2,2))
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3))
        self.prelu3 = nn.PReLU()

        self.fc1 = [nn.Linear(in_features=512, out_features=256) for _ in range(self.agents)]
        self.prelu4 = [nn.PReLU() for _ in range(self.agents)]
        self.fc2 = [nn.Linear(in_features=256, out_features=128) for _ in range(self.agents)]
        self.prelu5 = [nn.PReLU() for _ in range(self.agents)]
        self.fc3 = [nn.Linear(in_features=128, out_features=6) for _ in range(self.agents)]

    def forward(self, input):
        # Input is a tensor of size (batch_size, agents, frame_history, *image_size)
        for i in range(self.agents):
            # Common layers
            x = input[:, i]
            #print("input.shape", input.shape)
            #print("x.shape",x.shape)
            x = self.conv0(x)
            #print("x.shape conv0",x.shape)
            x = self.prelu0(x)
            x = self.maxpool0(x)
            #print("x.shape maxpool0",x.shape)
            x = self.conv1(x)
            #print("x.shape conv1",x.shape)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            #print("x.shape maxpool1",x.shape)
            x = self.conv2(x)
            #print("x.shape conv2",x.shape)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            #print("x.shape maxpool2",x.shape)
            #x = self.conv3(x)
            #print("x.shape conv3",x.shape)
            #x = self.prelu3(x)
            x = x.view(-1, 512)

            # Individual layers
            x = self.fc1[i](x)
            x = self.prelu4[i](x)
            x = self.fc2[i](x)
            x = self.prelu5[i](x)
            x = self.fc3[i](x)
            if i == 0:
                output = x.unsqueeze(0)
            else:
                output = torch.cat((output, x.unsqueeze(0)), dim=1)
        return output


class DQN:
    # The class initialisation function.
    def __init__(self, batch_size, agents, frame_history, number_actions = 6, type="Network3d"):
        self.batch_size = batch_size
        self.agents = agents
        self.number_actions = number_actions
        self.frame_history = frame_history
        input_dimension = self.agents*self.frame_history*45*45*45
        output_dimension = self.agents*self.number_actions
        # Create a Q-network, which predicts the q-value for a particular state.
        if type == "Network3d":
            self.q_network = Network3D(input_dimension, output_dimension, agents, frame_history)
            self.target_network = Network3D(input_dimension, output_dimension, agents, frame_history)
        else:
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
        # Transitions are tuple of shape obses_t, actions, rewards, obses_tp1, dones
        batch_inputs = torch.tensor(transitions[0])
        #batch_inputs = batch_inputs.view(self.batch_size, -1)

        # Labels are the rewards
        batch_labels = torch.tensor(transitions[2], dtype=torch.float32)
        y = self.target_network.forward(batch_inputs).detach().squeeze()
        y = y.view(self.batch_size, self.agents, self.number_actions)
        # Get the maximum prediction for the next state from the target network
        max_target_net = y.max(-1)[0]
        network_prediction = self.q_network.forward(batch_inputs).view(self.batch_size, self.agents, self.number_actions)
        # Bellman equation
        batch_labels_tensor = batch_labels + (discount_factor * max_target_net)
        td_errors = (network_prediction - batch_labels_tensor.unsqueeze(-1)).detach()

        index = torch.tensor(transitions[1], dtype=torch.long).unsqueeze(-1)
        y_pred = (torch.gather(network_prediction, -1, index)).squeeze()

        # Update transitions' weights
        # self.buffer.recompute_weights(transitions, td_errors)
        return torch.nn.MSELoss()(y_pred, batch_labels_tensor)
