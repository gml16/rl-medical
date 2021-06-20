import numpy as np
import torch
from itertools import chain
import torch.nn.functional as F

class Evaluator(object):
    def __init__(self, environment, model, logger, agents, max_steps):
        self.env = environment
        self.model = model
        self.logger = logger
        self.agents = agents
        self.max_steps = max_steps

    def play_n_episodes(self, render=False, fixed_spawn=None, silent=False):
        """
        wraps play_one_episode, playing a single episode at a time and logs
        results used when playing demos.
        """
        if fixed_spawn is None:
            num_runs = 1
        else:
            # fixed_spawn should be, for example, [0.5 , 0.5 , 0.5, 0, 0, 0] for 2 runs
            # In the first run agents spawn in the middle and in the second they will spawn from the corner
            fixed_spawn = np.array(fixed_spawn).reshape((-1, 3)) # 3 dimensions
            num_runs = fixed_spawn.shape[0]
            # Set all the agents to the same spawn point
            fixed_spawn = np.stack([fixed_spawn for _ in range(self.agents)], axis=-1)

        num_files = self.env.files.num_files
        self.model.train(False)
        headers = ["number"] + list(chain.from_iterable(zip(
            [f"Filename {i}" for i in range(self.agents)],
            [f"Agent {i} pos x" for i in range(self.agents)],
            [f"Agent {i} pos y" for i in range(self.agents)],
            [f"Agent {i} pos z" for i in range(self.agents)],
            [f"Landmark {i} pos x" for i in range(self.agents)],
            [f"Landmark {i} pos y" for i in range(self.agents)],
            [f"Landmark {i} pos z" for i in range(self.agents)],
            [f"Distance {i}" for i in range(self.agents)])))
        self.logger.write_locations(headers)
        distances = []
        for j in range(num_runs):
            for k in range(num_files):
                (score, start_dists, info)  = self.play_one_episode(render, fixed_spawn=fixed_spawn[j])
                row = [j * num_files + k + 1] + list(chain.from_iterable(zip(
                    [info[f"filename_{i}"] for i in range(self.agents)],
                    [info[f"agent_xpos_{i}"] for i in range(self.agents)],
                    [info[f"agent_ypos_{i}"] for i in range(self.agents)],
                    [info[f"agent_zpos_{i}"] for i in range(self.agents)],
                    [info[f"landmark_xpos_{i}"] for i in range(self.agents)],
                    [info[f"landmark_ypos_{i}"] for i in range(self.agents)],
                    [info[f"landmark_zpos_{i}"] for i in range(self.agents)],
                    [info[f"distError_{i}"] for i in range(self.agents)])))
                distances.append([info[f"distError_{i}"]
                                for i in range(self.agents)])
                self.logger.write_locations(row)
        mean = np.mean(distances, 0)
        std = np.std(distances, 0, ddof=1)
        if not silent:
            self.logger.log(f"mean distances {mean}")
            self.logger.log(f"Std distances {std}")
        return mean, std

    def play_one_episode(self,
                        render=False,
                        frame_history=4,
                        fixed_spawn=None,
                        continuous=False):

        def predict(inputs, continuous):
            """
            Run a full episode, mapping observation to action,
            using greedy policy.
            """
            obs_stack, (hx, cx) = inputs
            inputs = torch.tensor(obs_stack).permute(
                0, 4, 1, 2, 3)

            if not continuous:
                value, logit, (hx, cx) = self.model.forward((inputs, (hx, cx)))
                value = value.detach()
                logit = logit.detach()
                hx = hx.detach()
                cx = cx.detach()
                prob = F.softmax(logit, dim=-1)
                action = prob.multinomial(num_samples=1).detach()
            else:
                value, mean, sigma, (hx, cx) = self.model((inputs,(hx, cx)))
                value = value.detach()
                mean = mean.detach()
                sigma = sigma.detach()
                hx = hx.detach()
                cx = cx.detach()
                sigma = F.softplus(sigma) + 1e-5
                action = mean + sigma * torch.randn(*mean.shape)

            return action, (hx, cx)

        obs_stack = self.env.reset(fixed_spawn)
        print(obs_stack.shape)

        cx = torch.zeros(1, 256)
        hx = torch.zeros(1, 256)
        # Here obs have shape (agent, *image_size, frame_history)
        sum_r = np.zeros((self.agents))
        isOver = [False] * self.agents
        start_dists = None
        steps = 0
        while steps < self.max_steps and not np.all(isOver):
            acts, (hx, cx) = predict((obs_stack, (hx, cx)), continuous)
            obs_stack, r, isOver, info = self.env.step(acts, isOver, continuous)
            steps += 1
            if start_dists is None:
                start_dists = [
                    info['distError_' + str(i)] for i in range(self.agents)]
            if render:
                self.env.render()
            for i in range(self.agents):
                if not isOver[i]:
                    sum_r[i] += r[i]
        return sum_r, start_dists, info
