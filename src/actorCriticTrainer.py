import torch
import numpy as np
from torchsummary import summary
from math import pi
import torch.multiprocessing as mp
import torch.nn.functional as F
import copy
import sys
import time
from torch.utils.tensorboard import SummaryWriter

from DQNModel import DQN
#from evaluator import Evaluator
from A3C_evaluator import Evaluator
from tqdm import tqdm
from ActorCriticModel import A3C_discrete, A3C_continuous, A3C_continuous_v2, A3C_continuous_v3, A3C_continuous_v4
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
                 gamma=0.99,
                 number_actions=6,
                 frame_history=4,
                 logger=None,
                 train_freq=1,
                 lr=1e-4,
                 gae_lamda=1.00,
                 max_grad_norm=50,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 seed=None,
                 continuous=False,
                 comment='A3C',
                 model_name="A3C",
                ):
        self.model_name=model_name
        self.seed = seed
        self.env = env
        self.eval_env = eval_env
        self.agents = env.agents
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
        self.logger_comment = comment
        self.continuous = continuous

        #TODO change to accept 4 frames
        if self.model_name == "A3C_discrete":
            shared_model = A3C_discrete(1, self.env.action_space, self.agents)
        elif self.model_name == "A3C_continuous":
            shared_model = A3C_continuous(1, self.env.action_space)
        elif self.model_name == "A3C_continuous_v2":
            shared_model = A3C_continuous_v2(1, self.env.action_space)
        elif self.model_name == "A3C_continuous_v3":
            shared_model = A3C_continuous_v3(1, self.env.action_space)
        elif self.model_name == "A3C_continuous_v4":
            shared_model = A3C_continuous_v4(1, self.env.action_space)
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

        ready_processes = torch.zeros(num_processes)
        ready_processes.share_memory_()

        for rank in range(0, num_processes):
            p = mp.Process(target=self.train, args=(rank, shared_model, counter, lock, ready_processes, optimizer))
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


    def ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                    shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def train(self, rank, shared_model, counter, lock, ready_processes, optimizer=None):
        #if(self.seed):
        #    set_reproducible(self.seed+rank)
        #self.logger.log(self.dqn.q_network)
        #self.init_memory()
        #starting_time = time.time()

        self.logger.boardWriter = SummaryWriter(comment=self.logger_comment)

        #shared_model = A3C(self.frame_history, self.env.action_space)
        #TODO Change to accept 4 frames

        if self.model_name == "A3C_discrete":
            model = A3C_discrete(1, self.env.action_space, self.agents)
        elif self.model_name == "A3C_continuous":
            model = A3C_continuous(1, self.env.action_space)
        elif self.model_name == "A3C_continuous_v2":
            self.logger.log("Created model with A3C_continuous_v2 architecture")
            model = A3C_continuous_v2(1, self.env.action_space)
        elif self.model_name == "A3C_continuous_v3":
            model = A3C_continuous_v3(1, self.env.action_space)
        elif self.model_name == "A3C_continuous_v4":
            model = A3C_continuous_v4(1, self.env.action_space)
        #model = A3C(self.frame_history, self.env.action_space)

        env= copy.deepcopy(self.env)
        env.sampled_files = env.files.sample_circular(env.landmarks)
        env.sub_agent = rank
        env.logger = self.logger

        eval_env = copy.deepcopy(self.eval_env)
        eval_env.env.sampled_files = eval_env.env.files.sample_circular(eval_env.env.landmarks)
        eval_env.env.sub_agent = rank
        eval_env.env.logger = self.logger

        evaluator = Evaluator(eval_env,
                              model,
                              self.logger,
                              self.agents,
                              self.steps_per_episode,
                              self.model_name)

        episode = 1
        acc_steps = 0
        epoch_distances = []
        if optimizer is None:
            optimizer = optim.Adam(shared_model.parameters(), lr=self.lr)

        model.train()

        ready_processes[rank] = 1
        terminal = [False for _ in range(self.agents)]

        #print(f"Sub-agent {rank} initialization duration: {time.time()-starting_time}")


        while episode <= self.max_episodes:
            #forward_time = 0

            #episode_starting_time = time.time()

            #while(not torch.all(ready_processes.byte()).item()):
            #    pass
            while(ready_processes.unique().shape[0]!=1):
                pass

            model.load_state_dict(shared_model.state_dict())


            #ready_processes[rank] = 0

            self.logger.log(f"Subagent {rank} episode {episode}")

            if all(t for t in terminal):
                cx = cx.detach()
                hx = hx.detach()
            else:
                if self.model_name == "A3C_discrete":
                    cx = torch.zeros(self.agents, 256).unsqueeze(0)
                    hx = torch.zeros(self.agents, 256).unsqueeze(0)
                elif self.model_name == "A3C_continuous" or \
                    self.model_name == "A3C_continuous_v3" or \
                    self.model_name == "A3C_continuous_v4":
                    cx = torch.zeros(1, 256)
                    hx = torch.zeros(1, 256)
                elif self.model_name == "A3C_continuous_v2":
                    cx = torch.zeros(1, 512)
                    hx = torch.zeros(1, 512)

            # Reset the environment for the start of the episode.
            obs = env.reset()
            terminal = [False for _ in range(self.agents)]
            losses = []
            score = [0] * self.agents

            #cx = torch.zeros(1, 256)
            #hx = torch.zeros(1, 256)
            print(f"training obs shape {obs.shape}")

            values = []
            log_probs = []
            rewards = []
            entropies = []

            #checkpoint_1 = time.time()
            #print(f"Sub-agent {rank} episode initialization duration: {checkpoint_1-episode_starting_time}")

            for step_num in range(self.steps_per_episode):
                #check_1 = time.time()
                acc_steps += 1
                if(not self.continuous):
                    #check_1 = time.time()
                    value, logit, (hx, cx) = model((torch.tensor(obs).unsqueeze(0).unsqueeze(2),(hx, cx)))
                    #check_2 = time.time()
                    #forward_time += check_2 - check_1
                    value = value.squeeze()
                    logit = logit.squeeze(0)

                    prob = F.softmax(logit, dim=-1)
                    log_prob = F.log_softmax(logit, dim=-1)

                    entropy = -(log_prob * prob).sum(1)

                    action = prob.multinomial(num_samples=1).detach()
                    log_prob = log_prob.gather(1, action).squeeze()
                else:
                    #check_1 = time.time()
                    value, mean, sigma, (hx, cx) = \
                            model((torch.tensor(obs).unsqueeze(0),(hx, cx)))
                    #check_2 = time.time()
                    #forward_time += check_2 - check_1
                    sigma = F.softplus(sigma) + 1e-5
                    action = (mean + sigma * torch.randn(*mean.shape)).detach()
                    log_prob = (-(action - mean).pow(2) / (2 * sigma.pow(2)) -\
                               sigma.log() -\
                               0.5 * (torch.Tensor([2 * pi])).log().expand_as(sigma)).sum()
                    #self.logger.log(f"Subagent {rank} action {action}")

                    #self.logger.log(f"Subagent {rank} log_prob shape {log_prob.shape}")

                    entropy = (0.5 +
                               sigma.log() +
                               0.5 * (torch.Tensor([2 * pi])).log().expand_as(sigma)).sum()

                #check_2 = time.time()
                #print(f"Sub-agent {rank} 1-2  duration: {check_2-check_1}")

                obs, reward, terminal, info = env.step(
                    np.copy(action.numpy()), terminal, continuous = self.continuous, rank = rank)

                #check_3 = time.time()
                #print(f"Sub-agent {rank} 2-3  duration: {check_3-check_2}")

                reward = torch.clamp(
                    torch.tensor(
                        reward, dtype=torch.float32), -1, 1)

                score = [sum(x) for x in zip(score, reward)]

                entropies.append(entropy)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if all(t for t in terminal):
                    break
                #check_4 = time.time()
                #print(f"Sub-agent {rank} 3-4  duration: {check_4-check_3}")
                #print(f"Sub-agent {rank} step: {step_num}")

            #checkpoint_2 = time.time()
            #print(f"Sub-agent {rank} episode running duration: {checkpoint_2-checkpoint_1}")

            #self.logger.log(f"Sub-agent {rank} time spent in forward: {forward_time}")
            R = torch.zeros(self.agents,1)

            if not all(t for t in terminal):
                if not  self.continuous:
                    value, _, _ = model((torch.tensor(obs).unsqueeze(0).unsqueeze(2), (hx, cx)))
                else:
                    value, _, _, _ = model((torch.tensor(obs).unsqueeze(0), (hx, cx)))
                value = value.squeeze()
                R = value.detach()

            #self.logger.log(f"Subagent {rank} final reward {R}")

            R = torch.clamp(
                torch.tensor(
                    R, dtype=torch.float32), -1, 1)


            #self.logger.log(f"Subagent {rank} final clipped reward {R}")

            values.append(R)

            gae = torch.zeros(self.agents)
            value_loss, policy_loss = torch.zeros(self.agents), torch.zeros(self.agents)

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

            #checkpoint_3 = time.time()
            #print(f"Sub-agent {rank} grad aggregation duration: {checkpoint_3-checkpoint_2}")

            optimizer.zero_grad()
            #check_3 = time.time()
            (policy_loss + self.value_loss_coef * value_loss).sum().backward()
            #check_4 = time.time()
            #print(f"Sub-agent {rank} time spent in backward: {check_4 - check_3}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            #checkpoint_4 = time.time()
            #print(f"Sub-agent {rank} backprop duration: {checkpoint_4-checkpoint_3}")

            '''
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.logger.log(f"Subagent {rank}, Layer {name}, \
                                        value before optimizer step {param}")
                    break

            '''

            self.ensure_shared_grads(model, shared_model)
            #check_5 = time.time()
            optimizer.step()
            #check_6 = time.time()
            #print(f"Sub-agent {rank} time spent in backward: {check_6 - check_5}")
            #checkpoint_5 = time.time()
            #print(f"Sub-agent {rank} weight update duration: {checkpoint_5-checkpoint_4}")

            '''
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.logger.log(f"Subagent {rank}, Layer {name}, \
                                        value after optimizer step {param}")
                    break
            '''


            epoch_distances.append([info['distError_' + str(i)]
                                    for i in range(self.agents)])
            self.append_episode_board(info, score, "train", episode, rank)

            self.eps = max(self.min_eps, self.eps - self.delta)
            # Every epoch
            #if episode % 1 == 0:
            if episode % self.epoch_length == 0:
                lr = self.get_lr(optimizer)
                self.append_epoch_board(epoch_distances, self.eps, losses,
                                        "train", episode, rank, lr)
                self.validation_epoch(episode, model, evaluator, rank)
                self.save_model(model, name=f"latest_A3C_sub_agent_{rank}.pt", forced=True)
                #self.dqn.scheduler.step()
                epoch_distances = []

            episode += 1
            ready_processes[rank] += 1

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def validation_epoch(self, episode, model, evaluator, rank = 0):
        if self.eval_env is None:
            return
        model.train(False)
        epoch_distances = []
        for k in range(self.eval_env.files.num_files):
            self.logger.log(f"eval episode {k}")
            (score, start_dists, info) = evaluator.play_one_episode(continuous = self.continuous)
            epoch_distances.append([info['distError_' + str(i)]
                                    for i in range(self.agents)])

        val_dists = self.append_epoch_board(epoch_distances, name="eval",
                                            episode=episode, rank = rank)
        if (val_dists < self.best_val_distance):
            self.logger.log("Improved new best mean validation distances")
            self.best_val_distance = val_dists
            self.save_model(model, name=f"best_A3C_sub_agent_{rank}.pt", forced=True)
        model.train(True)

    def save_model(self, model, name="A3C.pt", forced=False):
        self.logger.save_model(model, name, forced)

    def append_episode_board(self, info, score, name="train", episode=0, rank=0):
        dists = {str(i):
                 info['distError_' + str(i)] for i in range(self.agents)}
        self.logger.write_to_board(f"{name}/sub_agent_{rank}/dist", dists, episode)
        scores = {str(i): score[i] for i in range(self.agents)}
        self.logger.write_to_board(f"{name}/sub_agent_{rank}/score", scores, episode)

    def append_epoch_board(self, epoch_dists, eps=0, losses=[],
                           name="train", episode=0, rank = 0, lr = 0):
        epoch_dists = np.array(epoch_dists)
        if name == "train":
            self.logger.write_to_board(name, {f"eps_sub_agent_{rank}": eps, f"lr_sub_agent_{rank}": lr}, episode)
            if len(losses) > 0:
                loss_dict = {"loss_sub_agent_{rank}": sum(losses) / len(losses)}
                self.logger.write_to_board(name, loss_dict, episode)
        for i in range(self.agents):
            mean_dist = sum(epoch_dists[:, i]) / len(epoch_dists[:, i])
            mean_dist_dict = {str(i): mean_dist}
            self.logger.write_to_board(
                f"{name}/mean_dist_sub_agent_{rank}", mean_dist_dict, episode)
            min_dist_dict = {str(i): min(epoch_dists[:, i])}
            self.logger.write_to_board(
                f"{name}/min_dist_sub_agent_{rank}", min_dist_dict, episode)
            max_dist_dict = {str(i): max(epoch_dists[:, i])}
            self.logger.write_to_board(
                f"{name}/max_dist_sub_agent_{rank}", max_dist_dict, episode)
        return np.array(list(mean_dist_dict.values())).mean()
