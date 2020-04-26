import torch
import numpy as np
from expreplay import ReplayMemory
from DQNModel import DQN
from tqdm import tqdm


class Trainer(object):
    def __init__(self,
                 env,
                 image_size=(45, 45, 45),
                 update_frequency=4,
                 replay_buffer_size=1e6,
                 init_memory_size=5e4,
                 max_episodes=100,
                 steps_per_episode=50,
                 eps=1,
                 min_eps=0.1,
                 delta=0.001,
                 batch_size=4,
                 gamma=0.9,
                 number_actions=6,
                 frame_history=4,
                 model_name="CommNet",
                 logger=None,
                 train_freq=1,
                 ):
        self.env = env
        self.agents = env.agents
        self.image_size = image_size
        self.update_frequency = update_frequency
        self.replay_buffer_size = replay_buffer_size
        self.init_memory_size = init_memory_size
        self.max_episodes = max_episodes
        self.steps_per_episode = steps_per_episode
        self.eps = eps
        self.min_eps = min_eps
        self.delta = delta
        self.batch_size = batch_size
        self.gamma = gamma
        self.number_actions = number_actions
        self.frame_history = frame_history
        self.buffer = ReplayMemory(
            self.replay_buffer_size,
            self.image_size,
            self.frame_history,
            self.agents)
        self.dqn = DQN(
            self.agents,
            self.frame_history,
            logger=logger,
            type=model_name)
        self.logger = logger
        self.train_freq = train_freq

    def train(self):
        self.logger.log(self.dqn.q_network)
        self.set_reproducible()
        self.init_memory()
        episode = 1
        acc_steps = 0
        while episode <= self.max_episodes:
            self.logger.log(
                f"episode {episode} - eps {self.eps:.5f}",
                acc_steps)
            # Reset the environment for the start of the episode.
            obs = self.env.reset()
            # Observations stacks is a numpy array with shape (agents,
            # frame_history, *image_size)
            terminal = [False for _ in range(self.agents)]
            for step_num in range(self.steps_per_episode):
                acc_steps += 1
                acts, q_values = self.get_next_actions(
                    self.buffer.recent_state())
                # Step the agent once, and get the transition tuple for this
                # step
                obs, reward, terminal, info = self.env.step(
                    np.copy(acts), q_values, terminal)

                if step_num == 0:
                    start_dists = [info['distError_' +
                                        str(i)] for i in range(self.agents)]

                self.buffer.append((obs, acts, reward, terminal))

                if acc_steps % self.train_freq == 0:
                    mini_batch = self.buffer.sample(self.batch_size)
                    loss = self.dqn.train_q_network(mini_batch, self.gamma)
                    self.logger.add_loss_board(loss, acc_steps)
                if all(t for t in terminal):
                    self.logger.log(
                        f"""Terminating episode after {step_num+1} steps,
                        total of {acc_steps} steps, final distance for first
                        agent is {info['distError_0']:.3f}, improved distance
                        by {(start_dists[0]-info['distError_0']):.3f}""",
                        acc_steps)
                    break
            self.logger.add_distances_board(start_dists, info, episode)
            if episode % self.update_frequency == 0:
                self.dqn.copy_to_target_network()
            self.eps = max(self.min_eps, self.eps - self.delta)
            # Updates scheduler every epoch
            # TODO: change magic number 728 which is the number of brains in
            # the dataset
            if episode % 728 == 0:
                self.dqn.scheduler.step()
            episode += 1
            self.dqn.save_model()

    def init_memory(self):
        self.logger.log("Initialising memory buffer...")
        pbar = tqdm(desc="Memory buffer", total=self.init_memory_size)
        while len(self.buffer) < self.init_memory_size:
            # Reset the environment for the start of the episode.
            obs = self.env.reset()
            terminal = [False for _ in range(self.agents)]
            steps = 0
            for _ in range(self.steps_per_episode):
                steps += 1
                acts, q_values = self.get_next_actions(
                    self.buffer.recent_state())
                obs, reward, terminal, info = self.env.step(
                    acts, q_values, terminal)
                self.buffer.append((obs, acts, reward, terminal))
                if all(t for t in terminal):
                    break
            pbar.update(steps)
        pbar.close()
        self.logger.log("Memory buffer filled")

    # Function to get the next action, using whatever method you like

    def get_next_actions(self, obs_stack):
        # epsilon-greedy policy
        if np.random.random() < self.eps:
            q_values = np.zeros((self.agents, self.number_actions))
            actions = np.random.randint(self.number_actions, size=self.agents)
        else:
            actions, q_values = self.get_greedy_actions(
                obs_stack, doubleLearning=True)
        return actions, q_values

    def get_greedy_actions(self, obs_stack, doubleLearning=True):
        inputs = torch.tensor(obs_stack).unsqueeze(0)
        if doubleLearning:
            q_vals = self.dqn.q_network.forward(inputs).detach().squeeze(0)
        else:
            q_vals = self.dqn.target_network.forward(
                inputs).detach().squeeze(0)
        idx = torch.max(q_vals, -1)[1]
        greedy_steps = np.array(idx, dtype=np.int32).flatten()
        # The actions are scaled for better training of the DQN
        return greedy_steps, q_vals.data.numpy()

    def set_reproducible(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
