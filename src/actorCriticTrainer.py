import torch
import numpy as np
from torchsummary import summary
import torch.multiprocessing as mp
import torch.nn.functional as F
import copy
import sys

from DQNModel import DQN
from evaluator import Evaluator
from tqdm import tqdm
from ActorCriticModel import A3C
import shared_adam

def set_reproducible(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


class Trainer(object):
    def __init__(self,
                 env,
                 no_shared=False,
                 num_processes=4,
                 eval_env=None,
                 image_size=(45, 45, 45),
                 max_episodes=100,
                 steps_per_episode=50,
                 eps=1,
                 min_eps=0.1,
                 delta=0.001,
                 batch_size=4,
                 gamma=0.9,
                 number_actions=6,
                 frame_history=4,
                 logger=None,
                 train_freq=1,
                 lr=1e-3,
                 gae_lamda=1.00,
                 max_grad_norm=50,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 seed=None
                ):
        self.seed = seed
        self.env = env
        self.eval_env = eval_env
        self.agents = env.agents
        self.agents = 1
        self.image_size = image_size
        self.max_episodes = max_episodes
        self.steps_per_episode = steps_per_episode
        self.eps = eps
        self.min_eps = min_eps
        self.delta = delta
        self.gamma = gamma
        self.gae_lambda = gae_lamda
        self.number_actions = number_actions
        self.frame_history = frame_history
        self.epoch_length = env.files.num_files
        self.best_val_distance = float('inf')
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        #TODO change to accept 4 frames
        shared_model = A3C(1, self.env.action_space)
        #shared_model = A3C(self.frame_history, self.env.action_space)
        shared_model.share_memory()

        if no_shared:
            optimizer = None
        else:
            optimizer = shared_adam.SharedAdam(shared_model.parameters(), lr=self.lr)
            optimizer.share_memory()

        processes = []

        counter = mp.Value('i', 0)

        lock = mp.Lock()

        self.logger = logger
        self.train_freq = train_freq

        # p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
        # p.start()
        # processes.append(p)
        
        for rank in range(0, num_processes):
            p = mp.Process(target=self.train, args=(rank, shared_model, counter, lock, optimizer))
            #p = mp.Process(target=self.nothing, args=(1, ))
            p.start()
            processes.append(p)
        for p in processes:
            exit_code = p.join()
            print(f"Exit code {exit_code}")
        

        print("Main process")
        sys.stdout.flush()

        # self.evaluator = Evaluator(eval_env,
        #                            self.dqn.q_network,
        #                            logger,
        #                            self.agents,
        #                            steps_per_episode)
        

    def nothing(self, simple_arg):
        print("Now I work")
        sys.stdout.flush()

    def ensure_shared_grads(model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                    shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def train(self, rank, shared_model, counter, lock, optimizer=None):
        #if(self.seed):
        #    set_reproducible(self.seed+rank)
        #self.logger.log(self.dqn.q_network)
        #self.init_memory()

        #shared_model = A3C(self.frame_history, self.env.action_space)
        #TODO Change to accept 4 frames
        model = A3C(1, self.env.action_space)
        #model = A3C(self.frame_history, self.env.action_space)

        env= copy.deepcopy(self.env)
        env.sampled_files = env.files.sample_circular(env.landmarks)

        episode = 1
        acc_steps = 0
        epoch_distances = []
        if optimizer is None:
            optimizer = optim.Adam(shared_model.parameters(), lr=self.lr)

        model.train()
        while episode <= self.max_episodes:
            # Reset the environment for the start of the episode.
            obs = env.reset()
            terminal = [False for _ in range(self.agents)]
            losses = []
            score = [0] * self.agents

            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)

            values = []
            log_probs = []
            rewards = []
            entropies = []

            for step_num in range(self.steps_per_episode):
                acc_steps += 1
                print(f"Obs shape: {obs.shape}")
                sys.stdout.flush()
                       
                value, logit, (hx, cx) = model((torch.tensor(obs).unsqueeze(0),(hx, cx)))
                print("After forward")
                sys.stdout.flush()

                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)

                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)

                obs, reward, terminal, info = env.step(
                    np.copy(action.numpy()), terminal)


                reward = torch.clamp(
                    torch.tensor(
                        reward, dtype=torch.float32), -1, 1)

                score = [sum(x) for x in zip(score, reward)]
                #self.buffer.append((obs, acts, reward, terminal))
                # if acc_steps % self.train_freq == 0:
                #     #mini_batch = self.buffer.sample(self.batch_size)
                #     loss = self.dqn.train_q_network(mini_batch, self.gamma)
                #     losses.append(loss)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if all(t for t in terminal):
                    break
    

            R = torch.zeros(1, 1)
            if not done:
                value, _, _ = model((state.unsqueeze(0), (hx, cx)))
                R = value.detach()

            values.append(R)

            gae = torch.zeros(1, 1)
            value_loss, policy = 0, 0

            for i in reversed(range(len(rewards))):
                R = self.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimation
                delta_t = rewards[i] + self.gamma * \
                    values[i + 1] - values[i]
                gae = gae * self.gamma * self.gae_lambda + delta_t

                policy_loss = policy_loss - \
                    log_probs[i] * gae.detach() - self.entropy_coef * entropies[i]

            optimizer.zero_grad()

            (policy_loss + self.value_loss_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

            ensure_shared_grads(model, shared_model)
            optimizer.step()

            epoch_distances.append([info['distError_' + str(i)]
                                    for i in range(self.agents)])
            self.append_episode_board(info, score, "train", episode)

            self.eps = max(self.min_eps, self.eps - self.delta)
            # Every epoch
            if episode % self.epoch_length == 0:
                self.append_epoch_board(epoch_distances, self.eps, losses,
                                        "train", episode)
                # self.validation_epoch(episode)
                #self.dqn.save_model(name="latest_dqn.pt", forced=True)
                #self.dqn.scheduler.step()
                epoch_distances = []

            episode += 1

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
