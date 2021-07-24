import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, num_inputs=1, agents=1):
        super(Actor, self).__init__()

        self.agents = agents

        self.conv1 = nn.Conv3d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(32, 32, 3, stride=2, padding=1)

        self.actor1 = nn.ModuleList(
            [nn.Linear(32 * 3 * 3 * 3, 256) for _ in range(self.agents)])
        self.actor2 = nn.ModuleList(
            [nn.Linear(256, action_dim) for _ in range(self.agents)])

        self.max_action = max_action


    def forward(self, state):
        state = state.to(device)
        actions = []

        for i in range(self.agents):
            x =  state[:,i]
            x = F.elu(self.conv1(x))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            x = x.view(-1, 32 * 3 * 3 * 3)

            action = self.actor1[i](x)
            action = self.actor2[i](action)

            actions.append(action)

        actions = torch.stack(actions, dim=1)

        return self.max_action * torch.tanh(actions)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_inputs=1, agents=1):
        super(Critic, self).__init__()

        self.agents = agents

        # Shared conv layers
        self.conv1 = nn.Conv3d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(32, 32, 3, stride=2, padding=1)

        # Q1
        self.critic_11 = nn.ModuleList(
            [nn.Linear(32 * 3 * 3 * 3 + action_dim, 256) for _ in range(self.agents)])
        self.critic_12 = nn.ModuleList(
            [nn.Linear(256, 1) for _ in range(self.agents)])

        # Q2
        self.critic_21 = nn.ModuleList(
            [nn.Linear(32 * 3 * 3 * 3 + action_dim, 256) for _ in range(self.agents)])
        self.critic_22 = nn.ModuleList(
            [nn.Linear(256, 1) for _ in range(self.agents)])


    def forward(self, state, action):
        state = state.to(device)
        action = action.to(device)

        q1_s = []
        q2_s = []

        for i in range(self.agents):
            act = action[:,i]
            x =  state[:,i]
            x = F.elu(self.conv1(x))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            x = x.view(-1, 32 * 3 * 3 * 3)

            sa = torch.cat([x, act], 1)

            q1 = self.critic_11[i](sa)
            q1 = self.critic_12[i](q1)

            q2 = self.critic_21[i](sa)
            q2 = self.critic_22[i](q2)

            q1_s.append(q1)
            q2_s.append(q2)

        q1_s = torch.stack(q1_s, dim=1)
        q2_s = torch.stack(q2_s, dim=1)

        return q1_s, q2_s


    def Q1(self, state, action):
        state = state.to(device)
        action = action.to(device)

        q1_s = []

        for i in range(self.agents):
            act = action[:,i]
            x =  state[:,i]
            x = F.elu(self.conv1(x))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            x = x.view(-1, 32 * 3 * 3 * 3)

            sa = torch.cat([x, act], 1)

            q1 = self.critic_11[i](sa)
            q1 = self.critic_12[i](q1)

            q1_s.append(q1)

        q1_s = torch.stack(q1_s, dim=1)

        return q1_s

class Actor_v2(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, num_inputs=1, agents=1):
        super(Actor_v2, self).__init__()

        self.agents = agents

        # Shared conv layers
        self.conv1 = nn.Conv3d(num_inputs, 32, 3)
        self.avgpool1 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, 3)

        self.conv3 = nn.Conv3d(64, 64, 3)
        self.avgpool2 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.conv4 = nn.Conv3d(64, 128, 3)

        self.conv5 = nn.Conv3d(128, 128, 3)
        self.avgpool3 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.conv6 = nn.Conv3d(128, 256, 2)

        self.actor1 = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(self.agents)])
        self.actor2 = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(self.agents)])
        self.actor3 = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(self.agents)])
        self.actor4 = nn.ModuleList(
            [nn.Linear(256, action_dim) for _ in range(self.agents)])

        self.max_action = max_action


    def forward(self, state):
        state = state.to(device)#/255.0
        actions = []

        for i in range(self.agents):
            x =  state[:,i]

            x = F.relu(self.conv1(x))
            x = self.avgpool1(x)
            x = F.relu(self.conv2(x))

            x = F.relu(self.conv3(x))
            x = self.avgpool2(x)
            x = F.relu(self.conv4(x))

            x = F.relu(self.conv5(x))
            x = self.avgpool3(x)
            x = F.relu(self.conv6(x))

            x = x.view(-1, 256)

            action = F.relu(self.actor1[i](x))
            action = F.relu(self.actor2[i](action))
            action = F.relu(self.actor3[i](action))
            action = self.actor4[i](action)

            actions.append(action)

        actions = torch.stack(actions, dim=1)

        return self.max_action * torch.tanh(actions)


class Critic_v2(nn.Module):
    def __init__(self, state_dim, action_dim, num_inputs=1, agents=1):
        super(Critic_v2, self).__init__()

        self.agents = agents

        # Shared conv layers
        self.conv1 = nn.Conv3d(num_inputs, 32, 3)
        self.avgpool1 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, 3)

        self.conv3 = nn.Conv3d(64, 64, 3)
        self.avgpool2 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.conv4 = nn.Conv3d(64, 128, 3)

        self.conv5 = nn.Conv3d(128, 128, 3)
        self.avgpool3 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.conv6 = nn.Conv3d(128, 256, 2)


        # Q1
        self.critic_11 = nn.ModuleList(
            [nn.Linear(256 + action_dim, 256) for _ in range(self.agents)])
        self.critic_12 = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(self.agents)])
        self.critic_13 = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(self.agents)])
        self.critic_14 = nn.ModuleList(
            [nn.Linear(256, 1) for _ in range(self.agents)])

        # Q2
        self.critic_21 = nn.ModuleList(
            [nn.Linear(256 + action_dim, 256) for _ in range(self.agents)])
        self.critic_22 = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(self.agents)])
        self.critic_23 = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(self.agents)])
        self.critic_24 = nn.ModuleList(
            [nn.Linear(256, 1) for _ in range(self.agents)])


    def forward(self, state, action):

        state = state.to(device)#/255.0
        action = action.to(device)

        q1_s = []
        q2_s = []

        for i in range(self.agents):
            act = action[:,i]
            x =  state[:,i]

            x = F.relu(self.conv1(x))
            x = self.avgpool1(x)
            x = F.relu(self.conv2(x))

            x = F.relu(self.conv3(x))
            x = self.avgpool2(x)
            x = F.relu(self.conv4(x))

            x = F.relu(self.conv5(x))
            x = self.avgpool3(x)
            x = F.relu(self.conv6(x))

            x = x.view(-1, 256)

            sa = torch.cat([x, act], 1)

            q1 = F.relu(self.critic_11[i](sa))
            q1 = F.relu(self.critic_12[i](q1))
            q1 = F.relu(self.critic_13[i](q1))
            q1 = self.critic_14[i](q1)

            q2 = F.relu(self.critic_21[i](sa))
            q2 = F.relu(self.critic_22[i](q2))
            q2 = F.relu(self.critic_23[i](q2))
            q2 = self.critic_24[i](q2)

            q1_s.append(q1)
            q2_s.append(q2)

        q1_s = torch.stack(q1_s, dim=1)
        q2_s = torch.stack(q2_s, dim=1)

        return q1_s, q2_s


    def Q1(self, state, action):
        state = state.to(device)#/255.0
        action = action.to(device)

        q1_s = []

        for i in range(self.agents):
            act = action[:,i]
            x =  state[:,i]

            x = F.relu(self.conv1(x))
            x = self.avgpool1(x)
            x = F.relu(self.conv2(x))

            x = F.relu(self.conv3(x))
            x = self.avgpool2(x)
            x = F.relu(self.conv4(x))

            x = F.relu(self.conv5(x))
            x = self.avgpool3(x)
            x = F.relu(self.conv6(x))
            x = x.view(-1, 256)

            sa = torch.cat([x, act], 1)

            q1 = F.relu(self.critic_11[i](sa))
            q1 = F.relu(self.critic_12[i](q1))
            q1 = F.relu(self.critic_13[i](q1))
            q1 = self.critic_14[i](q1)

            q1_s.append(q1)

        q1_s = torch.stack(q1_s, dim=1)

        return q1_s


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        frame_history=4,
        agents=1,
        logger=None
    ):

        self.actor = Actor_v2(state_dim, action_dim, max_action, num_inputs=frame_history, agents=agents).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        #self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)

        self.critic = Critic_v2(state_dim, action_dim, num_inputs=frame_history, agents=agents).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        #self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.logger = logger


    def select_action(self, state):
        state = state.to(device)
        return self.actor(state).squeeze(0).cpu().data.numpy()


    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        not_done = 1. - done

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(2).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        not_done = torch.FloatTensor(not_done).unsqueeze(2).to(device)

        reward = torch.clamp(
                    reward,
                    -self.max_action,
                    self.max_action)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            #unsqueeze to simulate frame history
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item()

    def save(self, name, forced):
        self.logger.save_model(self.critic.state_dict(), name + "_critic.pt", forced)
        self.logger.save_model(self.critic_optimizer.state_dict(), name + "_critic_optimizer.pt", forced)

        self.logger.save_model(self.actor.state_dict(), name + "_actor.pt", forced)
        self.logger.save_model(self.actor_optimizer.state_dict(), name + "_actor_optimizer.pt", forced)


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
