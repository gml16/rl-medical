import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension, agents):
        # Call the initialisation function of the parent class.
        super(MLP, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.agents = agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_1 = nn.Linear(in_features=input_dimension, out_features=100).to(self.device)
        self.layer_2 = nn.Linear(in_features=100, out_features=100).to(self.device)
        self.output_layer = nn.Linear(in_features=100, out_features=output_dimension).to(self.device)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        input = input.to(self.device)
        input = input.view(-1, self.layer_1.in_features)
        layer_1_output = nn.functional.relu(self.layer_1(input))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        output = output.view(-1, self.agents, 6)
        return output.cpu()

class Network2D(nn.Module):

    def __init__(self, agents, frame_history, number_actions):
        super(Network2D, self).__init__()
        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv0 = nn.Conv3d(in_channels=frame_history, out_channels=32, kernel_size=(5,5,5)).to(self.device)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2,2,2)).to(self.device)
        self.prelu0 = nn.PReLU().to(self.device)
        self.conv1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(5,5,5)).to(self.device)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2,2,2)).to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4,4,4)).to(self.device)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2,2,2)).to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3)).to(self.device)
        self.prelu3 = nn.PReLU().to(self.device)

        self.fc1 = nn.Linear(in_features=512, out_features=256).to(self.device)
        self.prelu4 = nn.LeakyReLU().to(self.device)
        self.fc2 = nn.Linear(in_features=256, out_features=128).to(self.device)
        self.prelu5 = nn.LeakyReLU().to(self.device)
        self.fc3 = nn.Linear(in_features=128, out_features=number_actions).to(self.device)


    def forward(self, input):
        """
        # Input is a tensor of size (batch_size, agents, frame_history, *image_size)
        """
        input = input.to(self.device)

        # Common layers
        x = input.squeeze(1)#input[:, 0]

        x = self.conv0(x)
        x = self.prelu0(x)

        x = self.maxpool0(x)

        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 512)

        # Individual layers
        x = self.fc1(x)
        x = self.prelu4(x)
        x = self.fc2(x)
        x = self.prelu5(x)
        x = self.fc3(x)
        output = x.unsqueeze(1)
        return output.cpu()

class Network3D(nn.Module):

    def __init__(self, agents, frame_history, number_actions, xavier = True):
        super(Network3D, self).__init__()

        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv0 = nn.Conv3d(in_channels=frame_history, out_channels=32, kernel_size=(5,5,5)).to(self.device)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2,2,2)).to(self.device)
        self.prelu0 = nn.PReLU().to(self.device)
        self.conv1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(5,5,5)).to(self.device)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2,2,2)).to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4,4,4)).to(self.device)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2,2,2)).to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3)).to(self.device)
        self.prelu3 = nn.PReLU().to(self.device)

        self.fc1 = nn.ModuleList([nn.Linear(in_features=512, out_features=256).to(self.device) for _ in range(self.agents)])
        self.prelu4 = nn.ModuleList([nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc2 = nn.ModuleList([nn.Linear(in_features=256, out_features=128).to(self.device) for _ in range(self.agents)])
        self.prelu5 = nn.ModuleList([nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc3 = nn.ModuleList([nn.Linear(in_features=128, out_features=number_actions).to(self.device) for _ in range(self.agents)])

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, input):
        """
        # Input is a tensor of size (batch_size, agents, frame_history, *image_size)
        # Output is a tensor of size (batch_size, agents, number_actions)
        """
        input = input.to(self.device)
        for i in range(self.agents):
            # Common layers
            x = input[:, i]

            x = self.conv0(x)
            x = self.prelu0(x)

            x = self.maxpool0(x)

            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
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
                output = x.unsqueeze(1)
            else:
                output = torch.cat((output, x.unsqueeze(1)), dim=1)
        return output.cpu()

class CommNet(nn.Module):

    def __init__(self, agents, frame_history, number_actions, xavier = True):
        super(CommNet, self).__init__()

        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv0 = nn.Conv3d(in_channels=frame_history, out_channels=32, kernel_size=(5,5,5)).to(self.device)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2,2,2)).to(self.device)
        self.prelu0 = nn.PReLU().to(self.device)
        self.conv1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(5,5,5)).to(self.device)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2,2,2)).to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4,4,4)).to(self.device)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2,2,2)).to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3)).to(self.device)
        self.prelu3 = nn.PReLU().to(self.device)

        self.fc1 = nn.ModuleList([nn.Linear(in_features=512*2, out_features=256).to(self.device) for _ in range(self.agents)])
        self.prelu4 = nn.ModuleList([nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc2 = nn.ModuleList([nn.Linear(in_features=256*2, out_features=128).to(self.device) for _ in range(self.agents)])
        self.prelu5 = nn.ModuleList([nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc3 = nn.ModuleList([nn.Linear(in_features=128*2, out_features=number_actions).to(self.device) for _ in range(self.agents)])

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, input):
        """
        # Input is a tensor of size (batch_size, agents, frame_history, *image_size)
        # Output is a tensor of size (batch_size, agents, number_actions)
        """
        input1 = input.to(self.device)
        for i in range(self.agents):
            # Common layers
            x = input1[:, i]

            x = self.conv0(x)
            x = self.prelu0(x)

            x = self.maxpool0(x)

            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            x = x.view(-1, 512)
            if i == 0:
                input2 = x.unsqueeze(1)
            else:
                input2 = torch.cat((input2, x.unsqueeze(1)), dim=1)

        comm = torch.mean(input2, axis=1)
        for i in range(self.agents):
            x = input2[:, i]
            x = self.fc1[i](torch.cat((x, comm), axis=-1))
            x = self.prelu4[i](x)
            if i == 0:
                input3 = x.unsqueeze(1)
            else:
                input3 = torch.cat((input3, x.unsqueeze(1)), dim=1)

        comm = torch.mean(input3, axis=1)
        for i in range(self.agents):
            x = input3[:, i]
            x = self.fc2[i](torch.cat((x, comm), axis=-1))
            x = self.prelu5[i](x)
            if i == 0:
                input4 = x.unsqueeze(1)
            else:
                input4 = torch.cat((input4, x.unsqueeze(1)), dim=1)

        comm = torch.mean(input4, axis=1)
        for i in range(self.agents):
            x = input4[:, i]
            x = self.fc3[i](torch.cat((x, comm), axis=-1))
            if i == 0:
                output = x.unsqueeze(1)
            else:
                output = torch.cat((output, x.unsqueeze(1)), dim=1)

        return output.cpu()


class DQN:
    # The class initialisation function.
    def __init__(self, agents, frame_history, logger, number_actions = 6, type="Network3d"):
        self.agents = agents
        self.number_actions = number_actions
        self.frame_history = frame_history
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.log(f"Using {self.device}")
        # Create a Q-network, which predicts the q-value for a particular state.
        if type == "Network3d":
            self.q_network = Network3D(agents, frame_history, number_actions).to(self.device)
            self.target_network = Network3D(agents, frame_history, number_actions).to(self.device)
        elif type == "CommNet":
            self.q_network = CommNet(agents, frame_history, number_actions).to(self.device)
            self.target_network = CommNet(agents, frame_history, number_actions).to(self.device)
        elif type=="Network2d":
            self.q_network = Network2D(agents, frame_history, number_actions).to(self.device)
            self.target_network = Network2D(agents, frame_history, number_actions).to(self.device)
        elif type == "MLP":
            input_dimension = agents*frame_history*45*45*45
            output_dimension = agents*number_actions
            self.q_network = MLP(input_dimension, output_dimension, agents).to(self.device)
            self.target_network = MLP(input_dimension, output_dimension, agents).to(self.device)
        self.copy_to_target_network()
        # Freezes target network
        self.target_network.train(False)
        for p in self.target_network.parameters():
           p.requires_grad = False
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=1e-3, eps=1e-3)

    def copy_to_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self):
        self.logger.save_model(self.q_network.state_dict())

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transitions, discount_factor):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        #self._calculate_loss_bis(transitions, discount_factor)
        loss = self._calculate_loss(transitions, discount_factor)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        return loss.item()


    def _calculate_loss_tf(self, transitions, discount_factor):
        import tensorflow as tf
        curr_state = transitions[0]
        self.predict_value = tf.convert_to_tensor(self.q_network.forward(torch.tensor(curr_state)).view(-1, self.number_actions).detach().numpy(), dtype=tf.float32) # Only works for 1 agent # self.get_DQN_prediction(state)
        reward = tf.squeeze(tf.clip_by_value(tf.convert_to_tensor(transitions[2], dtype=tf.float32), -1, 1), [1])
        next_state = transitions[3]
        action_onehot = tf.squeeze(tf.one_hot(transitions[1], 6, 1.0, 0.0), [1])

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        with tf.variable_scope('target'):
            targetQ_predict_value = tf.convert_to_tensor(self.q_network.forward(torch.tensor(next_state)).view(-1, self.number_actions).detach().numpy(), dtype=tf.float32)   # NxA

        best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        target = reward + discount_factor * tf.stop_gradient(best_v)

        cost = tf.losses.huber_loss(target, pred_action_value,
                                    reduction=tf.losses.Reduction.MEAN)
        with tf.Session() as sess:
            print("cost", cost.eval())

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, discount_factor):
        '''
        Transitions are tuple of shape obses_t, actions, rewards, obses_tp1, dones
        '''
        curr_state = torch.tensor(transitions[0])
        next_state = torch.tensor(transitions[3])
        terminal = torch.tensor(transitions[4]).type(torch.int)

        rewards = torch.clamp(torch.tensor(transitions[2], dtype=torch.float32), -1, 1)

        y = self.target_network.forward(next_state)
        y = y.view(-1, self.agents, self.number_actions) # dim (batch_size, agents, number_actions)
        # Get the maximum prediction for the next state from the target network
        max_target_net = y.max(-1)[0]
        network_prediction = self.q_network.forward(curr_state).view(-1, self.agents, self.number_actions) # dim (batch_size, agents, number_actions)
        isNotOver = (torch.ones(*terminal.shape)-terminal)
        # Bellman equation
        batch_labels_tensor = rewards + isNotOver * (discount_factor * max_target_net.detach())

        #td_errors = (network_prediction - batch_labels_tensor.unsqueeze(-1)).detach() # TODO td error needed for exp replay

        index = torch.tensor(transitions[1], dtype=torch.long).unsqueeze(-1)
        y_pred = (torch.gather(network_prediction, -1, index)).squeeze()

        # Update transitions' weights
        # self.buffer.recompute_weights(transitions, td_errors)

        return torch.nn.SmoothL1Loss()(batch_labels_tensor.flatten(), y_pred.flatten())
