import torch
import numpy as np
from expreplay import ReplayMemory
from TD3_expreplay import ReplayBuffer
from DQNModel import DQN
from evaluator import Evaluator
from tqdm import tqdm
from TD3Model import TD3


class Trainer(object):
    def __init__(self,
                 env,
                 eval_env=None,
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
                 team_reward=False,
                 attention=False,
                 lr=1e-3,
                 scheduler_gamma=0.5,
                 scheduler_step_size=100,
                 expl_noise=0.1,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2
                ):
        self.env = env
        self.eval_env = eval_env
        self.agents = env.agents
        self.image_size = image_size
        self.update_frequency = update_frequency
        self.replay_buffer_size = replay_buffer_size
        self.init_memory_size = init_memory_size
        self.max_episodes = max_episodes
        self.steps_per_episode = steps_per_episode
        self.expl_noise = expl_noise
        self.min_eps = min_eps
        self.delta = delta
        self.batch_size = batch_size
        self.gamma = gamma
        self.number_actions = number_actions
        self.frame_history = frame_history
        self.epoch_length = self.env.files.num_files
        self.best_val_distance = float('inf')
        self.logger = logger
        
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = float(self.env.action_space.high[0])
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

    	kwargs = {
    		"state_dim": self.state_dim,
    		"action_dim": self.action_dim,
    		"max_action": self.max_action,
    		"discount": self.gamma,
    		"tau": self.tau,
            "policy_noise": self.policy_noise * self.max_action,
            "noise_clip": self.noise_clip * self.max_action,
            "policy_freq": self.policy_freq
    	}

        self.logger.log(kwargs)

        self.policy = TD3()

        self.buffer = ReplayBuffer(state_dim, action_dim)

        self.evaluator = Evaluator(eval_env,
                                   self.policy,
                                   logger,
                                   self.agents,
                                   steps_per_episode)
        self.train_freq = train_freq

    def train(self):
        self.logger.log(self.policy.Actor)
        self.logger.log(self.policy.Critic)
        self.init_memory()
        episode = 1
        acc_steps = 0
        epoch_distances = []
        while episode <= self.max_episodes:
            # Reset the environment for the start of the episode.
            obs = self.env.reset()
            terminal = [False for _ in range(self.agents)]
            losses = []
            score = [0] * self.agents
            for step_num in range(self.steps_per_episode):
                acc_steps += 1
                acts = (
				       policy.select_action(np.array(state))
				       + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
			    ).clip(-self.max_action, self.max_action)
                # Step the agent once, and get the transition tuple
                obs, reward, terminal, info = self.env.step(np.copy(acts))
                score = [sum(x) for x in zip(score, reward)]
                self.buffer.append((obs, acts, reward, terminal))
                if acc_steps % self.train_freq == 0:
                    loss = self.TD3.train(self.buffer, self.batch_size)
                    losses.append(loss)
                if all(t for t in terminal):
                    break
            epoch_distances.append([info['distError_' + str(i)]
                                    for i in range(self.agents)])
            self.append_episode_board(info, score, "train", episode)

            self.eps = max(self.min_eps, self.eps - self.delta)
            # Every epoch
            if episode % self.epoch_length == 0:
                self.append_epoch_board(epoch_distances, self.eps, losses,
                                        "train", episode)
                self.validation_epoch(episode)
                self.dqn.save_model(name="latest_dqn.pt", forced=True)
                self.dqn.scheduler.step()
                epoch_distances = []
            episode += 1

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
                acts = (
                        policy.select_action(np.array(state))
				        + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
			    ).clip(-self.max_action, self.max_action)
                obs, reward, terminal, info = self.env.step(acts)
                self.buffer.append((obs, acts, reward, terminal))
                if all(t for t in terminal):
                    break
            pbar.update(steps)
        pbar.close()
        self.logger.log("Memory buffer filled")

    def validation_epoch(self, episode):
        if self.eval_env is None:
            return
        self.dqn.q_network.train(False)
        epoch_distances = []
        for k in range(self.eval_env.files.num_files):
            self.logger.log(f"eval episode {k}")
            (score, start_dists, q_values,
                info) = self.evaluator.play_one_episode()
            epoch_distances.append([info['distError_' + str(i)]
                                    for i in range(self.agents)])

        val_dists = self.append_epoch_board(epoch_distances, name="eval",
                                            episode=episode)
        if (val_dists < self.best_val_distance):
            self.logger.log("Improved new best mean validation distances")
            self.best_val_distance = val_dists
            self.dqn.save_model(name="best_dqn.pt", forced=True)
        self.dqn.q_network.train(True)

    def append_episode_board(self, info, score, name="train", episode=0):
        dists = {str(i):
                 info['distError_' + str(i)] for i in range(self.agents)}
        self.logger.write_to_board(f"{name}/dist", dists, episode)
        scores = {str(i): score[i] for i in range(self.agents)}
        self.logger.write_to_board(f"{name}/score", scores, episode)

    def append_epoch_board(self, epoch_dists, eps=0, losses=[],
                           name="train", episode=0):
        epoch_dists = np.array(epoch_dists)
        if name == "train":
            lr = self.dqn.scheduler.state_dict()["_last_lr"]
            self.logger.write_to_board(name, {"eps": eps, "lr": lr}, episode)
            if len(losses) > 0:
                loss_dict = {"loss": sum(losses) / len(losses)}
                self.logger.write_to_board(name, loss_dict, episode)
        for i in range(self.agents):
            mean_dist = sum(epoch_dists[:, i]) / len(epoch_dists[:, i])
            mean_dist_dict = {str(i): mean_dist}
            self.logger.write_to_board(
                f"{name}/mean_dist", mean_dist_dict, episode)
            min_dist_dict = {str(i): min(epoch_dists[:, i])}
            self.logger.write_to_board(
                f"{name}/min_dist", min_dist_dict, episode)
            max_dist_dict = {str(i): max(epoch_dists[:, i])}
            self.logger.write_to_board(
                f"{name}/max_dist", max_dist_dict, episode)
        return np.array(list(mean_dist_dict.values())).mean()

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
        return greedy_steps, q_vals.data.numpy()
