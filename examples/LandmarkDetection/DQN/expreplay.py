import numpy as np
import copy
from collections import deque

class ReplayBuffer(object): #Inherit replay mem
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def append(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data) # removed deepcopy
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
        res = self._encode_sample(idxes)
        return res

class ReplayMemory(object):
    def __init__(self, max_size, state_shape, history_len, agents):
        self.max_size = int(max_size)
        self.state_shape = state_shape
        self.history_len = int(history_len)
        self.agents=agents

        self.state=np.zeros((self.agents,self.max_size)+state_shape,dtype='uint8')
        self.action=np.zeros((self.agents,self.max_size),dtype='int32')
        self.reward=np.zeros((self.agents,self.max_size),dtype='float32')
        self.isOver=np.zeros((self.agents,self.max_size),dtype='bool')

        self._curr_pos=0
        self._curr_size = 0
        self._hist = deque(maxlen=history_len) #TODO: was maxlen = history_len - 1

    def append(self, exp):
        """Append the replay memory with experience sample
        Args:
            exp (Experience): experience contains (state, action, reward, isOver)
        """
        # increase current memory size if it is not full yet
        if self._curr_size < self.max_size:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
            self._curr_size += 1
        else:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
        if np.all(exp[3]):
            self._hist.clear()
        else:
            self._hist.append(exp)

    def recent_state(self):
        """ return a list of (hist_len,) + STATE_SIZE """
        lst = list(self._hist)
        states=[]
        for i in range(self.agents):
            states_temp=[np.zeros(self.state_shape, dtype='uint8')] * (self._hist.maxlen - len(lst))
            states_temp.extend([k[0][i] for k in lst])
            states.append(states_temp)
        return np.array(states)

    def _encode_sample(self, idx):
        """ Sample an experience replay from memory with index idx
        :returns: a tuple of (state, next_state, reward, action, isOver)
                  where state is of shape STATE_SIZE + (history_length,)
        """
        idx = (self._curr_pos + idx) % self._curr_size
        k = self.history_len

        states=[]
        next_states=[]
        rewards=[]
        actions=[]
        isOver=[]
        for i in range(self.agents):
            if idx + k < self._curr_size:
                states.append(self.state[i,idx: idx + k])
                next_states.append(self.state[i, idx + 1 : idx + k + 1])
                isOver.append(self.isOver[i,idx: idx + k])
            else:
                end = idx + k - self._curr_size
                states.append(self._slice(self.state[i],idx,end))
                next_states.append(self._slice(self.state[i],idx+1,end+1))
                isOver.append(self._slice(self.isOver[i],idx,end))
        rewards.append(self.reward[i, (idx + k - 1) % self._curr_size])
        actions.append(self.action[i, (idx + k - 1) % self._curr_size])
        states_padded = self._pad_sample(states, isOver)
        return states_padded, actions, rewards, next_states, isOver

    def sample(self, batch_size):
        idxes = [np.random.randint(0, len(self) - 1) for _ in range(batch_size)]
        states = []
        next_states=[]
        rewards=[]
        actions=[]
        isOver=[]
        for i in idxes:
            exp = self._encode_sample(i)
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            next_states.append(exp[3])
            isOver.append(exp[4])

        # Only get most recent terminal state
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(isOver)[:,:,-1]

    # the next_state is a different episode if current_state.isOver==True
    def _pad_sample(self,states,isOver):
        for k in range(self.history_len - 1, -1, -1):
            for i in range(self.agents):
                if isOver[i][k]:
                    states[i]=copy.deepcopy(states[i])
                    states[i][:k + 1].fill(0)
            break

        # transpose state
        # rewards_out=[]
        # actions_out=[]
        # isOver_out=[]
        # for i in range(self.agents):
        #     if states[i].ndim==4: # 3D States
        #         states[i]=states[i].transpose(1,2,3,0)
        #     else: # 2D States
        #         states[i]=states[i].transpose(1,2,0)
        #     rewards_out.append(rewards[i][-2])
        #     actions_out.append(actions[i][-2])
        #     isOver_out.append(isOver[i][-2])

        return states #, rewards_out, actions_out,isOver_out

    def _slice(self, arr, start, end):
        s1 = arr[start:]
        s2 = arr[:end]
        return np.concatenate((s1, s2), axis=0)

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        for i in range (self.agents):
            self.state[i,pos]=exp[0][i]
            self.action[i,pos]=exp[1][i]
            self.reward[i,pos]=exp[2][i]
            self.isOver[i,pos]=exp[3][i]

    def __str__(self):
        return f"""Replay buffer:
         Current position / current size: {self._curr_pos}/{self._curr_size}
         states {[hash(str(self.state[0, i])) for i in range(len(self.state[0]))]}
         actions {self.action}
         rewards {self.reward}
         isOver {self.isOver}"""
