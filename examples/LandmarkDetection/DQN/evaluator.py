import numpy as np
import torch

class Evaluator(object):
    def __init__(self, environment, model, logger, agents):
        self.env = environment
        self.model = model
        self.logger = logger
        self.agents = agents

    def play_n_episodes(self, render=False):
        """wraps play_one_episode, playing a single episode at a time and logs results
        used when playing demos."""
        self.logger.log("Getting distance errors:")
        for k in range(self.env.files.num_files):
            score, filename, ditance_error, start_dists, q_values, info = self.play_one_episode(render=render)
            self.logger.add_distances_board(start_dists, info, k)
            self.logger.log(f"{k + 1}/{self.env.files.num_files} - {filename} - score {score} - distError {ditance_error} - q_values {q_values}")

    def play_one_episode(self, render=False, frame_history=4):

        def predict(obs_stack):
            """
            Run a full episode, mapping observation to action, using greedy policy.
            """
            inputs = torch.tensor(obs_stack).permute(0, 4, 1, 2, 3).unsqueeze(0)
            q_vals = self.model.forward(inputs).detach().squeeze(0)
            idx = torch.max(q_vals, -1)[1]
            greedy_steps = np.array(idx, dtype = np.int32).flatten()
            return greedy_steps, q_vals.data.numpy()

        obs_stack = self.env.reset()
        # Here obs are of the form (agent, *image_size, frame_history)
        sum_r = np.zeros((self.agents))
        filenames_list = []
        distError_list = []
        isOver = [False] * self.agents
        start_dists = None
        while True:
            acts, q_values = predict(obs_stack)
            obs_stack, r, isOver, info = self.env.step(acts, q_values, isOver)
            if start_dists is None:
                start_dists = [info['distError_' + str(i)] for i in range(self.agents)]
            if render:
                self.env.render()
            for i in range(self.agents):
                if not isOver[i]:
                    sum_r[i] += r[i]
                if np.all(isOver):
                    filenames_list.append(info['filename_{}'.format(i)])
                    distError_list.append(info['distError_{}'.format(i)])
            if np.all(isOver):
                return sum_r, filenames_list, distError_list, start_dists, q_values, info
