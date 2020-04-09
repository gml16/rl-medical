#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Amir Alansary <amiralansary@gmail.com>

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import argparse
from collections import deque

from medical import MedicalPlayer, FrameStack
from DQNModel import DQN
from trainer import Trainer
from logger import Logger
from evaluator import Evaluator

###############################################################################
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
IMAGE_SIZE = (45, 45, 45)
# how many frames to keep
# in other words, how many observations the network can see
FRAME_HISTORY = 4
# DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
GAMMA = 0.9 #0.99
# num training epochs in between model evaluations
EPOCHS_PER_EVAL = 2
# the number of episodes to run during evaluation
EVAL_EPISODE = 50

###############################################################################

def get_player(directory=None, files_list=None, viz=False,
               task='play', saveGif=False, saveVideo=False,
               multiscale=True, history_length=28, agents=1,
               reward_strategy=1, logger=None):
    # in atari paper, max_num_frames = 30000
    env = MedicalPlayer(directory=directory, screen_dims=IMAGE_SIZE,
                        viz=viz, saveGif=saveGif, saveVideo=saveVideo,
                        task=task, files_list=files_list, max_num_frames=1500,
                        history_length=history_length, multiscale=multiscale,
                        agents=agents, reward_strategy=reward_strategy, logger=logger)
    if task != 'train':
        # in training, env will be decorated by ExpReplay, and history
        # is taken care of in expreplay buffer
        # otherwise, FrameStack modifies self.step to save observations into a queue
        env = FrameStack(env, FRAME_HISTORY, agents)
    return env

###############################################################################
###############################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform. Must load a pretrained model if task is "play" or "eval"',
                        choices=['play', 'eval', 'train'], default='train')
    # parser.add_argument('--algo', help='algorithm',
    #                     choices=['DQN', 'Double', 'Dueling','DuelingDouble'],
    #                     default='DQN')
    parser.add_argument('--files', type=argparse.FileType('r'), nargs='+',
                        help="""Filepath to the text file that comtains list of images.
                                Each line of this file is a full path to an image scan.
                                For (task == train or eval) there should be two input files ['images', 'landmarks']""")
    parser.add_argument('--saveGif', help='Save gif image of the game',
                        action='store_true', default=False)
    parser.add_argument('--saveVideo', help='Save video of the game',
                        action='store_true', default=False)
    parser.add_argument('--logDir', help='Store logs in this directory during training',
                        default='runs', type=str)
    # parser.add_argument('--name', help='Name of current experiment for logs',
    #                     default='experiment_1')
    parser.add_argument('--agents', help='Number of agents', type=int, default=1)

    parser.add_argument('--model_name', help='Models implemented are: Network3d, CommNet',default="CommNet", choices=['CommNet', 'Network3d'], type=str)
    parser.add_argument('--batch_size', help='Size of each batch', default=64, type=int)
    parser.add_argument('--memory_size', help="""Number of transitions stored in exp replay buffer.
                                                 If too much is allocated training may abruptly stop.""", default=5e4, type=int)
    parser.add_argument('--init_memory_size', help='Number of transitions stored in exp replay buffer before training', default=2048, type=int)
    parser.add_argument('--max_episodes', help='"Number of episodes to train for"', default=1000, type=int)
    parser.add_argument('--steps_per_episode', help='Maximum steps per episode', default=200, type=int)
    parser.add_argument('--target_update_freq', help='Number of episodes between each target network update', default=10, type=int)
    parser.add_argument('--delta', help='Amount to decreases epsilon each step, for the epsilon-greedy policy', default=1e-4, type=float)
    parser.add_argument('--viz', help='Size of the window, None for no visualisation', default=0.01, type=float)
    parser.add_argument('--multiscale', help='Reduces size of voxel around the agent when it oscillates', dest='multiscale', action='store_true')
    parser.set_defaults(multiscale=False)
    parser.add_argument('--write', help='Saves the training logs', dest='write', action='store_true')
    parser.set_defaults(write=False)
    parser.add_argument('--train_freq', help='Number of agent steps between each training step on one mini-batch', default=1, type=int)



    args = parser.parse_args()

    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # check valid number of agents:
    assert args.agents > 0

    # check input files
    if args.task == 'play':
        error_message = """Wrong input files {} for {} task - should be 1 \'images.txt\' """.format(len(args.files), args.task)
        assert len(args.files) == 1, (error_message)
    else:
        error_message = """Wrong input files {} for {} task - should be 2 [\'images.txt\', \'landmarks.txt\'] """.format(len(args.files), args.task)
        assert len(args.files) == 2, (error_message)

    logger = Logger(args.logDir, args.write)

    # load files into env to set num_actions, num_validation_files
    # TODO: is this still necessary?
    init_player = MedicalPlayer(files_list=args.files,
                                screen_dims=IMAGE_SIZE,
                                # TODO: why is this always play?
                                task='play',
                                agents=args.agents,
                                logger=logger)
    NUM_ACTIONS = init_player.action_space.n
    # num_files = init_player.files.num_files

    if args.task != 'train':
        # TODO: refactor DQN to not have to create both a q_network and target_network
        dqn = DQN(args.agents, frame_history=FRAME_HISTORY,
                  logger=logger, type=args.model_name)
        model = dqn.q_network
        model.load_state_dict(torch.load(args.load))
        environment = get_player(files_list=args.files,
                                   saveGif=args.saveGif,
                                   saveVideo=args.saveVideo,
                                   task=args.task,
                                   agents=args.agents,
                                   viz=args.viz,
                                   logger=logger)
        evaluator = Evaluator(environment, model, logger, args.agents)
        evaluator.play_n_episodes()
    else:  # train model
        environment = get_player(task='train',
                                 files_list=args.files,
                                 agents=args.agents,
                                 history_length=20,
                                 viz=args.viz,
                                 multiscale=args.multiscale,
                                 logger=logger)
        trainer = Trainer(environment,
                          batch_size = args.batch_size,
                          image_size = IMAGE_SIZE,
                          frame_history = FRAME_HISTORY,
                          update_frequency = args.target_update_freq,
                          replay_buffer_size = args.memory_size,
                          init_memory_size = args.init_memory_size,
                          gamma = GAMMA,
                          steps_per_episode = args.steps_per_episode,
                          max_episodes = args.max_episodes,
                          delta=args.delta,
                          logger = logger,
                          model_name=args.model_name,
                          train_freq = args.train_freq,
                          ).train()
