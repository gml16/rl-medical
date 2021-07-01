#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Amir Alansary <amiralansary@gmail.com>

import warnings
from evaluator import Evaluator
from logger import Logger
from actorCriticTrainer import Trainer
from ActorCriticModel import A3C_discrete, A3C_continuous
from medical import MedicalPlayer, FrameStack

import argparse
import os
import torch
import numpy as np
import torch.multiprocessing as mp

def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


###############################################################################
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
IMAGE_SIZE = (45, 45, 45)
# how many frames to keep
# in other words, how many observations the network can see
#TODO change this
FRAME_HISTORY = 1
###############################################################################


def get_player(directory=None, files_list=None, landmark_ids=None, viz=False,
               task="play", file_type="brain", saveGif=False, saveVideo=False,
               multiscale=True, history_length=20, agents=1, logger=None,
               stopping_criterion="osc", threshold=0.25):
    env = MedicalPlayer(
        directory=directory,
        screen_dims=IMAGE_SIZE,
        viz=viz,
        saveGif=saveGif,
        saveVideo=saveVideo,
        task=task,
        files_list=files_list,
        file_type=file_type,
        landmark_ids=landmark_ids,
        history_length=history_length,
        multiscale=multiscale,
        agents=agents,
        logger=logger,
        stopping_criterion=stopping_criterion,
        threshold=threshold)
    if task != "train":
        # in training, env will be decorated by ExpReplay, and history
        # is taken care of in expreplay buffer
        # otherwise, FrameStack modifies self.step to save observations into a
        # queue
        env = FrameStack(env, FRAME_HISTORY, agents)
    return env

###############################################################################
###############################################################################

def set_reproducible(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


if __name__ == '__main__':

    mp.set_start_method("spawn")

    os.environ["OMP_NUM_THREADS"] = "1" # make sure numpy uses only one thread for each process
    os.environ["CUDA_VISABLE_DEVICES"] = "" # make sure not to use gpu

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
        '--model_name',
        help='Models implemented are: Network3d, CommNet, A3C_continuous_v3, A3C_continuous_v2, A3C_dicrete, A3C_continuous',
        default="CommNet",
        choices=['CommNet', 'Network3d', 'A3C_discrete', 'A3C_continuous', 'A3C_continuous_v2', 'A3C_continuous_v3', 'A3C_continuous_v4', 'A3C_continuous_v5'], type=str)
    parser.add_argument(
        '--stopping_criterion',
        help='Stopping criterions implemented are: Oscillations, Zero action, Two consecutive zero actions, Unrounded action smaller than threshold',
        default="osc",
        choices=['osc', 'zero_action', 'consec_zero_action', 'threshold'], type=str)
    parser.add_argument(
        '--threshold',
        help="""Threshold value for threshold stopping criterion""",
        default=0.25, type=float)
    parser.add_argument(
        '--batch_size', help='Size of each batch', default=64, type=int)
    parser.add_argument(
        '--memory_size',
        help="""Number of transitions stored in exp replay buffer.
                If too much is allocated training may abruptly stop.""",
        default=1e5, type=int)
    parser.add_argument(
        '--discount',
        help='Discount factor used in the Bellman equation',
        default=0.99, type=float)
    parser.add_argument(
        '--lr',
        help='Starting learning rate',
        default=1e-4, type=float)
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
        choices=[None, 'mean', 'attention'], default=None)
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
        '--no-shared', default=False,
        help='use an optimizer without shared momentum for A3C.')
    parser.add_argument(
        '--num-processes', type=int, default=4,
        help='how many training processes to use for A3C (default: 4)')
    parser.add_argument(
        '--gae-lamda', type=float, default=1.00,
        help='lamda parameter for GAE (default: 1.00)')
    parser.add_argument(
        '--entropy-coef', type=float, default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef', type=float, default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm', type=float, default=50,
        help='value loss coefficient (default: 50)')

    args = parser.parse_args()

    if "continuous" in args.model_name:
        continuous = True
    else:
        continuous = False

    agents = len(args.landmarks)

    # check valid number of agents:
    assert agents > 0

    # check input files
    if args.task == 'play':
        error_message = f"""Wrong input files {len(args.files)} for {args.task}
                            task - should be 1 \'images.txt\' """
        assert len(args.files) == 1, (error_message)
    else:
        error_message = f"""Wrong input files {len(args.files)} for
                            {args.task} task - should be 2 [\'images.txt\',
                            \'landmarks.txt\'] """
        assert len(args.files) == 2, (error_message)

    if args.seed is not None:
        set_reproducible(args.seed)

    logger = Logger(args.log_dir, args.write, args.save_freq, comment=args.log_comment)
    logger.boardWriter = None
    #logger = None
    if args.task != 'train':
        # dqn = DQN(agents, frame_history=FRAME_HISTORY, logger=logger,
        #           type=args.model_name, collective_rewards=args.team_reward, attention=args.attention)
        if args.model_name == "A3C_discrete":
            model = A3C_discrete(FRAME_HISTORY, 6)
        elif args.model_name == "A3C_continuous":
            model = A3C_continuous(FRAME_HISTORY, 3)
        elif args.model_name == "A3C_continuous_v2":
            model = A3C_continuous_v2(FRAME_HISTORY, 3)

        model.load_state_dict(torch.load(args.load, map_location=model.device))
        environment = get_player(files_list=args.files,
                                 file_type=args.file_type,
                                 landmark_ids=args.landmarks,
                                 saveGif=args.saveGif,
                                 saveVideo=args.saveVideo,
                                 task=args.task,
                                 agents=agents,
                                 viz=args.viz,
                                 logger=logger)
        evaluator = Evaluator(environment, model, logger, agents,
                              args.steps_per_episode)
        evaluator.play_n_episodes(fixed_spawn=args.fixed_spawn)
    else:  # train model
        print("creating train env")

        environment = get_player(task='train',
                                 files_list=args.files,
                                 file_type=args.file_type,
                                 landmark_ids=args.landmarks,
                                 agents=agents,
                                 viz=args.viz,
                                 multiscale=args.multiscale,
                                 logger=None,
                                 stopping_criterion=args.stopping_criterion,
                                 threshold=args.threshold)
        environment.sampled_files = None

        if args.val_files is not None:
            print("creating val env")
            eval_env = get_player(task='eval',
                                  files_list=args.val_files,
                                  file_type=args.file_type,
                                  landmark_ids=args.landmarks,
                                  agents=agents,
                                  logger=None,
                                  stopping_criterion=args.stopping_criterion,
                                  threshold=args.threshold)
            eval_env.env.sampled_files = None
            print("Created val env")

        trainer = Trainer(environment,
                          eval_env=eval_env,
                          batch_size=args.batch_size,
                          image_size=IMAGE_SIZE,
                          frame_history=FRAME_HISTORY,
                          gamma=args.discount,
                          steps_per_episode=args.steps_per_episode,
                          max_episodes=args.max_episodes,
                          delta=args.delta,
                          logger=logger,
                          train_freq=args.train_freq,
                          lr=args.lr,
                          gae_lamda=args.gae_lamda,
                          max_grad_norm=args.max_grad_norm,
                          value_loss_coef=args.value_loss_coef,
                          entropy_coef=args.entropy_coef,
                          num_processes=args.num_processes,
                          continuous=continuous,
                          comment=args.log_comment,
                          model_name=args.model_name
                         )
