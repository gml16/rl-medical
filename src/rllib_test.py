import argparse
import numpy as np
import os
import random

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

from medical_for_rllib import MedicalPlayer, FrameStack
from logger import Logger

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)
parser.add_argument('--load', help='Path to the model to load')
parser.add_argument(
    '--task',
    help='''task to perform,
            must load a pretrained model if task is "play" or "eval"''',
    choices=['play', 'eval', 'train'], default='train')
parser.add_argument(
    '--file_type', help='Type of the training and validation files',
    choices=['brain', 'cardiac', 'fetal'], default='train')
parser.add_argument(
    '--files', type=argparse.FileType('r'), nargs='+',
    help="""Filepath to the text file that contains list of images.
            Each line of this file is a full path to an image scan.
            For (task == train or eval) there should be two input files
            ['images', 'landmarks']""")
parser.add_argument(
    '--val_files', type=argparse.FileType('r'), nargs='+',
    help="""Filepath to the text file that contains list of validation
            images. Each line of this file is a full path to an image scan.
            For (task == train or eval) there should be two input files
            ['images', 'landmarks']""")
parser.add_argument('--saveGif', help='Save gif image of the game',
                    action='store_true', default=False)
parser.add_argument('--saveVideo', help='Save video of the game',
                    action='store_true', default=False)
parser.add_argument(
    '--log_dir', help='Store logs in this directory during training.',
    default='runs', type=str)
parser.add_argument(
    '--log_comment', help='Suffix appended to the name of the log folder name, which is the current time.',
    default='', type=str)
parser.add_argument(
    '--landmarks', nargs='*', help='Landmarks to use in the images',
    type=int, default=[1])
parser.add_argument(
    '--model_name', help='Models implemented are: Network3d, CommNet',
    default="CommNet", choices=['CommNet', 'Network3d', 'Network3d_stacked', 'GraphNet', 'GraphNet_v2', 'SemGCN'], type=str)
parser.add_argument(
    '--graph_type', help='Types of graph layers, only used for GraphNet_v2',
    default="GCNConv", type=str)
parser.add_argument(
    '--batch_size', help='Size of each batch', default=64, type=int)
parser.add_argument(
    '--memory_size',
    help="""Number of transitions stored in exp replay buffer.
            If too much is allocated training may abruptly stop.""",
    default=1e5, type=int)
parser.add_argument(
    '--init_memory_size',
    help='Number of transitions stored in exp replay before training',
    default=3e4, type=int)
parser.add_argument(
    '--discount',
    help='Discount factor used in the Bellman equation',
    default=0.9, type=float)
parser.add_argument(
    '--lr',
    help='Starting learning rate',
    default=1e-3, type=float)
parser.add_argument(
    '--scheduler_gamma',
    help='Multiply the learning rate by this value every scheduler_step_size epochs',
    default=0.5, type=float)
parser.add_argument(
    '--scheduler_step_size',
    help='Every scheduler_step_size epochs, the learning rate is multiplied by scheduler_gamma',
    default=100, type=int)
parser.add_argument(
    '--max_episodes', help='"Number of episodes to train for"',
    default=1e5, type=int)
parser.add_argument(
    '--steps_per_episode', help='Maximum steps per episode',
    default=200, type=int)
parser.add_argument(
    '--target_update_freq',
    help='Number of epochs between each target network update',
    default=10, type=int)
parser.add_argument(
    '--save_freq', help='Saves network every save_freq steps',
    default=1000, type=int)
parser.add_argument(
    '--delta',
    help="""Amount to decreases epsilon each episode,
            for the epsilon-greedy policy""",
    default=1e-4, type=float)
parser.add_argument(
    '--viz', help='Size of the window, None for no visualisation',
    default=0.01, type=float)
parser.add_argument(
    '--multiscale',
    help='Reduces size of voxel around the agent when it oscillates',
    dest='multiscale', action='store_true')
parser.set_defaults(multiscale=False)
parser.add_argument(
    '--write', help='Saves the training logs', dest='write',
    action='store_true')
parser.set_defaults(write=False)
parser.add_argument(
    '--team_reward', help='Refers to adding the (potentially weighted) average reward of all agents to their individiual rewards',
    choices=[None, 'mean', 'attention', 'physical'], default=None)
parser.add_argument(
    '--attention', help='Use attention for communication channel in C-MARL/CommNet', dest='attention',
    action='store_true')
parser.set_defaults(attention=False)
parser.add_argument(
    '--train_freq',
    help="""Number of agent steps between each training step on one
            mini-batch""",
    default=1, type=int)
parser.add_argument(
    '--seed',
    help="Random seed for both training and evaluating. If none is provided, no seed will be set", type=int)
parser.add_argument(
    '--fixed_spawn', nargs='*',  type=float,
    help='Starting position of the agents during rollout. Randomised if not specified.',)
parser.add_argument(
    '--physical_reward', help='Incorporates physical reward', dest='physical_reward',
    action='store_true')
parser.set_defaults(physical_reward=False)
parser.add_argument(
    '--beta', help='Physical reward scaling',
    default=2, type=int)
parser.add_argument(
    '--reward_limiter', help='Scale or clip the agent rewards before adding them. Only applies to physical reward',
    choices=[None, 'clipping', 'scaling'], default=None)


IMAGE_SIZE = (45, 45, 45)
FRAME_HISTORY = 4

class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    logger = Logger(args.log_dir, args.write, args.save_freq, comment=args.log_comment)

    agents = len(args.landmarks)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.torch else CustomModel)

    config = {
        "env": MedicalPlayer,
        "env_config": {
            "directory" : None,
            "viz" : args.viz,
            "task" : args.task,
            "files_list" : args.files,
            "file_type" : args.file_type,
            "landmark_ids" : args.landmarks,
            "screen_dims" : IMAGE_SIZE,
            "history_length" : 20,
            "multiscale" : args.multiscale,
            "max_num_frames" : 200,
            "saveGif" : args.saveGif,
            "saveVideo" : args.saveVideo,
            "agents" : agents,
            "oscillations_allowed" : 4,
            "fixed_spawn" : args.fixed_spawn,
            "logger" : logger,
            "adj" : None,
            "beta" : args.beta,
            "physical_reward" : args.physical_reward,
            "reward_limiter" : args.reward_limiter
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 1,  # parallelism
        "framework": "torch"
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
