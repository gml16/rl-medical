import numpy as np
import copy
from collections import deque


class ReplayMemory(object):
    def __init__(self, max_size, state_shape, history_len, agents):
        self.max_size = int(max_size)
        self.state_shape = state_shape
        self.history_len = int(history_len)
        self.agents = agents
        try:
            self.state = np.zeros(
                (self.agents, self.max_size) + state_shape, dtype='uint8')
        except Exception as e:
            print("Please consider reducing the memory usage with the --memory_size flag.")
            raise e
        self.action = np.zeros((self.agents, self.max_size), dtype='int32')
        self.reward = np.zeros((self.agents, self.max_size), dtype='float32')
        self.isOver = np.zeros((self.agents, self.max_size), dtype='bool')

        self._curr_pos = 0
        self._curr_size = 0
        self._hist = deque(maxlen=history_len)

    def append(self, exp):
        """Append the replay memory with experience sample
        Args:
            exp (Experience): contains (state, action, reward, isOver)
        """
        # increase current memory size if it is not full yet
        if self._curr_size < self.max_size:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
            self._curr_size += 1
        else:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size

    def recent_transition(self):
        """ Return hist_len transitions, padded with zeros if needed
        Most recent transition is at the end of the arrays """
        # print("self._curr_pos", self._curr_pos)
        # print("self._curr_size", self._curr_size)
        # print("self.action", self.action)
        # print("isOver", self.isOver)
        return self._encode_sample(self._curr_pos)

    def recent_state(self):
        """ Return a tuple of previous RoI and actions """
        transition = self.recent_transition()
        return transition[0], transition[1]

    def _encode_sample(self, idx):
        """ Sample an experience replay from memory with index idx
        :returns: a tuple of (state, next_state, reward, action, isOver)
                  where state is of shape STATE_SIZE + (history_length,)
        """
        states = self._slice(self.state, idx)
        next_states = self._slice(self.state, idx + 1)
        isOver = self._slice(self.isOver, idx)
        rewards = self._slice(self.reward, idx)
        actions = self._slice(self.action, idx)
        next_actions = self._slice(self.action, idx + 1)

        states = self._pad_sample(states, isOver)
        next_states = self._pad_sample(next_states, isOver)

        return states, actions, rewards, next_states, isOver, next_actions

    def sample_stacked_actions(self, batch_size):
        idxes = [np.random.randint(0, len(self) - 1)
                 for _ in range(batch_size)]
        states = []
        next_states = []
        rewards = []
        actions = []
        next_actions = []
        isOver = []
        for i in idxes:
            exp = self._encode_sample(i)
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            next_states.append(exp[3])
            isOver.append(exp[4])
            next_actions.append(exp[5])
        return ((np.array(states), np.array(actions)), np.array(actions)[:, :, -1],
                np.array(rewards)[:, :, -1], (np.array(next_states), np.array(next_actions)),
                np.array(isOver)[:, :, -1])

    def sample(self, batch_size):
        idxes = [np.random.randint(0, len(self) - 1)
                 for _ in range(batch_size)]
        states = []
        next_states = []
        rewards = []
        actions = []
        isOver = []
        for i in idxes:
            exp = self._encode_sample(i)
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            next_states.append(exp[3])
            isOver.append(exp[4])
        # Only get most recent terminal state
        return (np.array(states), np.array(actions)[:, :, -1],
                np.array(rewards)[:, :, -1], np.array(next_states),
                np.array(isOver)[:, :, -1])

    # the next_state is a different episode if current_state.isOver==True
    def _pad_sample(self, arr, isOver):
        for k in range(self.history_len - 1, -1, -1):
            for i in range(self.agents):
                if isOver[i][k]:
                    arr[i] = copy.deepcopy(arr[i])
                    arr[i][:k + 1].fill(0)
            break
        return arr

    def _slice(self, arr, idx):
        if idx >= self.history_len:
            return arr[:, idx-self.history_len: idx]
        s1 = arr[:, -self.history_len+idx:]
        s2 = arr[:, :idx]
        return np.concatenate((s1, s2), axis=1)

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        for i in range(self.agents):
            self.state[i, pos] = exp[0][i]
            self.action[i, pos] = exp[1][i]
            self.reward[i, pos] = exp[2][i]
            self.isOver[i, pos] = exp[3][i]

    def __str__(self):
        return f"""Replay buffer:
         Current position / current size: {self._curr_pos}/{self._curr_size}
         states {[hash(str(self.state[0, i]))
                    for i in range(len(self.state[0]))]}
         actions {self.action}
         rewards {self.reward}
         isOver {self.isOver}"""
