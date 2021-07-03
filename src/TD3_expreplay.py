import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, agents, max_size=int(1e5)):
        self.max_size = max_size
        self.agents = agents
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, self.agents) + state_dim, dtype='uint8')
        self.action = np.zeros((max_size, self.agents, action_dim), dtype='float32')
        self.next_state = np.zeros((max_size, self.agents) + state_dim, dtype='uint8')
        self.reward = np.zeros((max_size, self.agents, 1), dtype='float32')
        self.not_done = np.zeros((max_size, self.agents, 1), dtype='bool')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
