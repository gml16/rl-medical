import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    if type(m) == nn.Linear or type(m)== nn.Conv3d:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class A3C_discrete(torch.nn.Module):
    def __init__(self, num_inputs, action_space, agents):
        super(A3C_discrete, self).__init__()

        self.agents = agents

        self.conv1 = nn.Conv3d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3 * 3, 256)

        #num_outputs = action_space
        num_outputs = action_space.n
        self.critic_linear = nn.ModuleList(
            [nn.Linear(256, 1) for _ in range(self.agents)])
        self.actor_linear = nn.ModuleList(
            [nn.Linear(256, num_outputs) for _ in range(self.agents)])

        self.apply(weights_init)
        for module in self.actor_linear:
            module.weight.data = normalized_columns_initializer(
                module.weight.data, 0.01)
            module.bias.data.fill_(0)

        # self.actor_linear.weight.data = normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)

        for module in self.critic_linear:
            module.weight.data = normalized_columns_initializer(
                module.weight.data, 0.01)
            module.bias.data.fill_(0)
        # self.critic_linear.weight.data = normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()


    def forward(self, inputs):

        inputs, (hxs, cxs) = inputs

        inputs = inputs / 255.0
        values = []
        actions = []
        hxs_next = []
        cxs_next = []
        for i in range(self.agents):
            x = inputs[:,i]
            hx = hxs[:,i]
            cx = cxs[:,i]

            x = F.elu(self.conv1(x))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            x = x.view(-1, 32 * 3 * 3 * 3)

            hx, cx = self.lstm(x, (hx, cx))
            x = hx
            value = self.critic_linear[i](x)
            action = self.actor_linear[i](x)

            values.append(value)
            actions.append(action)
            hxs_next.append(hx)
            cxs_next.append(cx)

        values = torch.stack(values, dim=1)
        actions = torch.stack(actions, dim=1)
        hxs = torch.stack(hxs_next, dim=1)
        cxs = torch.stack(cxs_next, dim=1)

        return values, actions, (hxs, cxs)

class A3C_continuous(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3C_continuous, self).__init__()

        self.conv1 = nn.Conv3d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3 * 3, 256)

        #num_outputs = action_space
        num_outputs = (action_space.n)//2
        print(num_outputs)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_mu = nn.Linear(256, num_outputs)
        self.actor_sigma = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_mu.weight.data = normalized_columns_initializer(
            self.actor_mu.weight.data, 0.01)
        self.actor_mu.bias.data.fill_(0)
        self.actor_sigma.weight.data = normalized_columns_initializer(
            self.actor_sigma.weight.data, 0.01)
        self.actor_sigma.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()


    def forward(self, inputs):

        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3 * 3)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_mu(x), self.actor_sigma(x), (hx, cx)

class A3C_continuous_v2(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3C_continuous_v2, self).__init__()

        self.conv0 = nn.Conv3d(
            in_channels=num_inputs,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.prelu0 = nn.PReLU()
        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 4),
            padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=0)
        self.prelu3 = nn.PReLU()

        self.lstm = nn.LSTMCell(512, 512)

        num_outputs = (action_space.n)//2

        self.critic0 = nn.Linear(in_features=512, out_features=256)
        self.prelu4 = nn.PReLU()
        self.critic1 = nn.Linear(256, 1)

        self.actor_mu0 = nn.Linear(in_features=512, out_features=256)
        self.prelu5 = nn.PReLU()
        self.actor_mu1 = nn.Linear(256, num_outputs)

        self.actor_sigma0 = nn.Linear(in_features=512, out_features=256)
        self.prelu6 = nn.PReLU()
        self.actor_sigma1 = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_mu0.weight.data = normalized_columns_initializer(
            self.actor_mu0.weight.data, 0.01)
        self.actor_mu0.bias.data.fill_(0)
        self.actor_sigma0.weight.data = normalized_columns_initializer(
            self.actor_sigma0.weight.data, 0.01)
        self.actor_sigma0.bias.data.fill_(0)
        self.critic0.weight.data = normalized_columns_initializer(
            self.critic0.weight.data, 1.0)
        self.critic0.bias.data.fill_(0)

        self.actor_mu1.weight.data = normalized_columns_initializer(
            self.actor_mu1.weight.data, 0.01)
        self.actor_mu1.bias.data.fill_(0)
        self.actor_sigma1.weight.data = normalized_columns_initializer(
            self.actor_sigma1.weight.data, 0.01)
        self.actor_sigma1.bias.data.fill_(0)
        self.critic1.weight.data = normalized_columns_initializer(
            self.critic1.weight.data, 1.0)
        self.critic1.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()


    def forward(self, inputs):

        inputs, (hx, cx) = inputs

        #inputs = inputs/ 255.0

        x = self.conv0(inputs)
        x = self.prelu0(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.view(-1, 512)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        value = self.critic0(x)
        value = self.prelu4(value)
        value = self.critic1(value)

        mu = self.actor_mu0(x)
        mu = self.prelu5(mu)
        mu = self.actor_mu1(mu)

        sigma = self.actor_sigma0(x)
        sigma = self.prelu6(sigma)
        sigma = self.actor_sigma1(sigma)

        return value, mu, sigma, (hx, cx)

class A3C_continuous_v3(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3C_continuous_v3, self).__init__()

        self.conv0 = nn.Conv3d(
            in_channels=num_inputs,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 4),
            padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=0)

        self.lstm = nn.LSTMCell(512, 256)

        num_outputs = (action_space.n)//2

        self.critic = nn.Linear(in_features=256, out_features=1)
        #self.prelu4 = nn.PReLU()
        #self.critic1 = nn.Linear(256, 1)

        self.actor_mu = nn.Linear(in_features=256, out_features=num_outputs)
        #self.prelu5 = nn.PReLU()
        #self.actor_mu1 = nn.Linear(256, num_outputs)

        self.actor_sigma = nn.Linear(in_features=256, out_features=num_outputs)
        #self.prelu6 = nn.PReLU()
        #self.actor_sigma1 = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_mu.weight.data = normalized_columns_initializer(
            self.actor_mu.weight.data, 0.01)
        self.actor_mu.bias.data.fill_(0)
        self.actor_sigma.weight.data = normalized_columns_initializer(
            self.actor_sigma.weight.data, 0.01)
        self.actor_sigma.bias.data.fill_(0)
        self.critic.weight.data = normalized_columns_initializer(
            self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)

        '''
        self.actor_mu1.weight.data = normalized_columns_initializer(
            self.actor_mu1.weight.data, 0.01)
        self.actor_mu1.bias.data.fill_(0)
        self.actor_sigma1.weight.data = normalized_columns_initializer(
            self.actor_sigma1.weight.data, 0.01)
        self.actor_sigma1.bias.data.fill_(0)
        self.critic1.weight.data = normalized_columns_initializer(
            self.critic1.weight.data, 1.0)
        self.critic1.bias.data.fill_(0)
        '''

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()


    def forward(self, inputs):

        inputs, (hx, cx) = inputs

        #inputs = inputs/ 255.0

        x = F.elu(self.conv0(inputs))
        x = self.maxpool0(x)
        x = F.elu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.elu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.elu(self.conv3(x))
        x = x.view(-1, 512)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        '''
        value = self.critic0(x)
        value = self.prelu4(value)
        value = self.critic1(value)

        mu = self.actor_mu0(x)
        mu = self.prelu5(mu)
        mu = self.actor_mu1(mu)

        sigma = self.actor_sigma0(x)
        sigma = self.prelu6(sigma)
        sigma = self.actor_sigma1(sigma)
        '''
        return self.critic(x), self.actor_mu(x), self.actor_sigma(x), (hx, cx)

class A3C_continuous_v4(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3C_continuous_v4, self).__init__()

        self.conv0 = nn.Conv3d(
            in_channels=num_inputs,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=2, padding=1)
        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=2, padding=1)
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=2, padding=1)
        self.conv3 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3 * 3, 256)

        num_outputs = (action_space.n)//2

        self.critic = nn.Linear(in_features=256, out_features=1)
        #self.prelu4 = nn.PReLU()
        #self.critic1 = nn.Linear(256, 1)

        self.actor_mu = nn.Linear(in_features=256, out_features=num_outputs)
        #self.prelu5 = nn.PReLU()
        #self.actor_mu1 = nn.Linear(256, num_outputs)

        self.actor_sigma = nn.Linear(in_features=256, out_features=num_outputs)
        #self.prelu6 = nn.PReLU()
        #self.actor_sigma1 = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_mu.weight.data = normalized_columns_initializer(
            self.actor_mu.weight.data, 0.01)
        self.actor_mu.bias.data.fill_(0)
        self.actor_sigma.weight.data = normalized_columns_initializer(
            self.actor_sigma.weight.data, 0.01)
        self.actor_sigma.bias.data.fill_(0)
        self.critic.weight.data = normalized_columns_initializer(
            self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)

        '''
        self.actor_mu1.weight.data = normalized_columns_initializer(
            self.actor_mu1.weight.data, 0.01)
        self.actor_mu1.bias.data.fill_(0)
        self.actor_sigma1.weight.data = normalized_columns_initializer(
            self.actor_sigma1.weight.data, 0.01)
        self.actor_sigma1.bias.data.fill_(0)
        self.critic1.weight.data = normalized_columns_initializer(
            self.critic1.weight.data, 1.0)
        self.critic1.bias.data.fill_(0)
        '''

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):

        inputs, (hx, cx) = inputs

        #inputs = inputs/ 255.0

        x = F.elu(self.conv0(inputs))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.view(-1, 32 * 3 * 3 * 3)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic(x), self.actor_mu(x), self.actor_sigma(x), (hx, cx)

class A3C_continuous_v5(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3C_continuous_v5, self).__init__()

        self.conv0 = nn.Conv3d(
            in_channels=num_inputs,
            out_channels=32,
            kernel_size=(5, 5, 5),
            stride=2,
            padding=1)
        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5, 5),
            stride=2,
            padding=1)
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 4),
            stride=2,
            padding=1)
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=0)

        self.lstm = nn.LSTMCell(512, 256)

        num_outputs = (action_space.n)//2

        self.critic = nn.Linear(in_features=256, out_features=1)
        #self.prelu4 = nn.PReLU()
        #self.critic1 = nn.Linear(256, 1)

        self.actor_mu = nn.Linear(in_features=256, out_features=num_outputs)
        #self.prelu5 = nn.PReLU()
        #self.actor_mu1 = nn.Linear(256, num_outputs)

        self.actor_sigma = nn.Linear(in_features=256, out_features=num_outputs)
        #self.prelu6 = nn.PReLU()
        #self.actor_sigma1 = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_mu.weight.data = normalized_columns_initializer(
            self.actor_mu.weight.data, 0.01)
        self.actor_mu.bias.data.fill_(0)
        self.actor_sigma.weight.data = normalized_columns_initializer(
            self.actor_sigma.weight.data, 0.01)
        self.actor_sigma.bias.data.fill_(0)
        self.critic.weight.data = normalized_columns_initializer(
            self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)

        '''
        self.actor_mu1.weight.data = normalized_columns_initializer(
            self.actor_mu1.weight.data, 0.01)
        self.actor_mu1.bias.data.fill_(0)
        self.actor_sigma1.weight.data = normalized_columns_initializer(
            self.actor_sigma1.weight.data, 0.01)
        self.actor_sigma1.bias.data.fill_(0)
        self.critic1.weight.data = normalized_columns_initializer(
            self.critic1.weight.data, 1.0)
        self.critic1.bias.data.fill_(0)
        '''

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()


    def forward(self, inputs):

        inputs, (hx, cx) = inputs

        #inputs = inputs/ 255.0

        x = F.elu(self.conv0(inputs))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.view(-1, 512)

        print(x.shape)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        '''
        value = self.critic0(x)
        value = self.prelu4(value)
        value = self.critic1(value)

        mu = self.actor_mu0(x)
        mu = self.prelu5(mu)
        mu = self.actor_mu1(mu)

        sigma = self.actor_sigma0(x)
        sigma = self.prelu6(sigma)
        sigma = self.actor_sigma1(sigma)
        '''
        return self.critic(x), self.actor_mu(x), self.actor_sigma(x), (hx, cx)


class ActorCritic:
    # The class initialisation function.
    def __init__(
            self,
            agents,
            workers,
            frame_history,
            logger,
            number_actions=6,
            type="A3C",
            lr=1e-3):
        self.agents = agents
        self.workers = workers
        self.number_actions = number_actions
        self.frame_history = frame_history
        self.logger = logger
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.logger.log(f"Using {self.device}")
        # Create a Q-network, which predicts the q-value for a particular state
        if type == "A3C":
            self.q_network = Network3D(
                agents,
                frame_history,
                number_actions).to(
                self.device)


        # Define the optimiser which is used when updating the Q-network. The
        # learning rate determines how big each gradient step is during
        # backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    def save_model(self, name="dqn.pt", forced=False):
        self.logger.save_model(self.q_network.state_dict(), name, forced)

    # Function that is called whenever we want to train the Q-network. Each
    # call to this function takes in a transition tuple containing the data we
    # use to update the Q-network.
    def train_actor_critic_network(self, transitions, discount_factor):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transitions, discount_factor)
        # Compute the gradients based on this loss, i.e. the gradients of the
        # loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, discount_factor):
        '''
        Transitions are tuple of shape
        (states, actions, rewards, next_states, dones)
        '''
        curr_state = torch.tensor(transitions[0])
        next_state = torch.tensor(transitions[3])
        terminal = torch.tensor(transitions[4]).type(torch.int)

        rewards = torch.clamp(
            torch.tensor(
                transitions[2], dtype=torch.float32), -1, 1)

        y = self.target_network.forward(next_state)
        # dim (batch_size, agents, number_actions)
        y = y.view(-1, self.agents, self.number_actions)
        # Get the maximum prediction for the next state from the target network
        max_target_net = y.max(-1)[0]

        # dim (batch_size, agents, number_actions)
        network_prediction = self.q_network.forward(curr_state).view(
            -1, self.agents, self.number_actions)
        isNotOver = (torch.ones(*terminal.shape) - terminal)
        # Bellman equation
        batch_labels_tensor = rewards + isNotOver * \
            (discount_factor * max_target_net.detach())

        actions = torch.tensor(transitions[1], dtype=torch.long).unsqueeze(-1)
        y_pred = torch.gather(network_prediction, -1, actions).squeeze()

        return torch.nn.SmoothL1Loss()(batch_labels_tensor.flatten(), y_pred.flatten())
