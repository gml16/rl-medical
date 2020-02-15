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

import os
import sys
import time
import argparse
from collections import deque

import tensorflow as tf
from medical import MedicalPlayer, FrameStack
from tensorpack.input_source import QueueInput
from tensorpack_medical.models.conv3d import Conv3D
from tensorpack_medical.models.pool3d import MaxPooling3D
from common import Evaluator, eval_model_multithread, play_n_episodes
from DQNModel import Model3D as DQNModel3D
from DQNModel import Model2D as DQNModel2D
from expreplay import ExpReplay

from tensorpack import (PredictConfig, OfflinePredictor, get_model_loader,
                        logger, TrainConfig, ModelSaver, PeriodicTrigger,
                        ScheduledHyperParamSetter, ObjAttrParam,
                        HumanHyperParamSetter, argscope, RunOp, LinearWrap,
                        FullyConnected, PReLU, SimpleTrainer,
                        launch_train_with_config)

from trainer import Trainer


###############################################################################
# BATCH SIZE USED IN NATURE PAPER IS 32 - MEDICAL IS 256
BATCH_SIZE = 48
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
IMAGE_SIZE = (45, 45, 45)
# how many frames to keep
# in other words, how many observations the network can see
FRAME_HISTORY = 4
# the frequency of updating the target network
UPDATE_FREQ = 4
# DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
GAMMA = 0.9 #0.99
# REPLAY MEMORY SIZE - NATURE (1e6) - MEDICAL (1e5 view-patches)
MEMORY_SIZE = 1e5#6
# consume at least 1e6 * 27 * 27 * 27 bytes
INIT_MEMORY_SIZE = MEMORY_SIZE // 20 #5e4
# each epoch is 100k played frames
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ # 10000 // UPDATE_FREQ * 10
# num training epochs in between model evaluations
EPOCHS_PER_EVAL = 2
# the number of episodes to run during evaluation
EVAL_EPISODE = 50

###############################################################################

def get_player(directory=None, files_list= None, viz=False,
               task='play', saveGif=False, saveVideo=False, agents=1, reward_strategy=1):
    # in atari paper, max_num_frames = 30000
    env = MedicalPlayer(directory=directory, screen_dims=IMAGE_SIZE,
                        viz=viz, saveGif=saveGif, saveVideo=saveVideo,
                        task=task, files_list=files_list, max_num_frames=1500, agents=agents, reward_strategy=reward_strategy)
    if task != 'train':
        # in training, env will be decorated by ExpReplay, and history
        # is taken care of in expreplay buffer
        # otherwise, FrameStack modifies self.step to save observations into a queue
        env = FrameStack(env, FRAME_HISTORY, agents)
    return env

###############################################################################

class Model2D(DQNModel2D):
    def __init__(self):
        super(Model2D, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):
        """ image: [0,255]

        :returns predicted Q values"""
        # normalize image values to [0, 1]
        image = image / 255.0

        with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True):
            # core layers of the network
            conv = (LinearWrap(image)
                 .Conv3D('conv0', out_channel=32,
                         kernel_shape=[5,5,5], stride=[1,1,1])
                 .MaxPooling3D('pool0',2)
                 .Conv3D('conv1', out_channel=32,
                         kernel_shape=[5,5,5], stride=[1,1,1])
                 .MaxPooling3D('pool1',2)
                 .Conv3D('conv2', out_channel=64,
                         kernel_shape=[4,4,4], stride=[1,1,1])
                 .MaxPooling3D('pool2',2)
                 .Conv3D('conv3', out_channel=64,
                         kernel_shape=[3,3,3], stride=[1,1,1])
                 # .MaxPooling3D('pool3',2)
                 )

        if 'Dueling' not in self.method:
            lq = (conv
                 .FullyConnected('fc0', 512).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc1', 256).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc2', 128).tf.nn.leaky_relu(alpha=0.01)())
            Q = FullyConnected('fct', lq, self.num_actions, nl=tf.identity)
        else:
            # Dueling DQN or Double Dueling
            # state value function
            lv = (conv
                 .FullyConnected('fc0V', 512).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc1V', 256).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc2V', 128).tf.nn.leaky_relu(alpha=0.01)())
            V = FullyConnected('fctV', lv, 1, nl=tf.identity)
            # advantage value function
            la = (conv
                 .FullyConnected('fc0A', 512).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc1A', 256).tf.nn.leaky_relu(alpha=0.01)
                 .FullyConnected('fc2A', 128).tf.nn.leaky_relu(alpha=0.01)())
            As = FullyConnected('fctA', la, self.num_actions, nl=tf.identity)

            Q = tf.add(As, V - tf.reduce_mean(As, 1, keepdims=True))

        return tf.identity(Q, name='Qvalue')

class Model3D(DQNModel3D):
    def __init__(self, agents=1):
        super(Model3D, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA, agents)

    def _get_DQN_prediction(self, images):
        """ image: [0,255]

        :returns predicted Q values"""
        # normalize image values to [0, 1]

        agents = len(images)

        Q_list = []

        with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True):

            for i in range(agents):
                images[i] = images[i] / 255.0
                with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True):

                    if i == 0:
                        conv_0 = tf.layers.conv3d(images[i], name='conv0',
                                                  filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                        conv_1 = tf.layers.conv3d(max_pool_0, name='conv1',
                                                  filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool1 = tf.layers.max_pooling3d(conv_1, 2, 2, name='max_pool1')
                        conv_2 = tf.layers.conv3d(max_pool1, name='conv2',
                                                  filters=64, kernel_size=[4, 4, 4], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool2 = tf.layers.max_pooling3d(conv_2, 2, 2, name='max_pool2')
                        conv3 = tf.layers.conv3d(max_pool2, name='conv3',
                                                 filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding='same',
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
                                                 bias_initializer=tf.zeros_initializer())
                    else:
                        conv_0 = tf.layers.conv3d(images[i], name='conv0', reuse=True,
                                                  filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                        conv_1 = tf.layers.conv3d(max_pool_0, name='conv1', reuse=True,
                                                  filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool1 = tf.layers.max_pooling3d(conv_1, 2, 2, name='max_pool1')
                        conv_2 = tf.layers.conv3d(max_pool1, name='conv2', reuse=True,
                                                  filters=64, kernel_size=[4, 4, 4], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool2 = tf.layers.max_pooling3d(conv_2, 2, 2, name='max_pool2')
                        conv3 = tf.layers.conv3d(max_pool2, name='conv3', reuse=True,
                                                 filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding='same',
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
                                                 bias_initializer=tf.zeros_initializer())

                ### now for the dense layers##
                if 'Dueling' not in self.method:
                    fc0 = FullyConnected('fc0_{}'.format(i), conv3, 512, activation=tf.nn.relu)
                    fc1 = FullyConnected('fc1_{}'.format(i), fc0, 256, activation=tf.nn.relu)
                    fc2 = FullyConnected('fc2_{}'.format(i), fc1, 128, activation=tf.nn.relu)
                    Q = FullyConnected('fct_{}'.format(i), fc2, self.num_actions, nl=tf.identity)
                    Q_list.append(tf.identity(Q, name='Qvalue_{}'.format(i)))



                else:
                    fc0 = FullyConnected('fc0V_{}'.format(i), conv3, 512, activation=tf.nn.relu)
                    fc1 = FullyConnected('fc1V_{}'.format(i), fc0, 256, activation=tf.nn.relu)
                    fc2 = FullyConnected('fc2V_{}'.format(i), fc1, 128, activation=tf.nn.relu)
                    V = FullyConnected('fctV_{}'.format(i), fc2, 1, nl=tf.identity)

                    fcA0 = FullyConnected('fc0V_{}'.format(i), conv3, 512, activation=tf.nn.relu)
                    fcA1 = FullyConnected('fc1V_{}'.format(i), fcA0, 256, activation=tf.nn.relu)
                    fcA2 = FullyConnected('fc2V_{}'.format(i), fcA1, 128, activation=tf.nn.relu)
                    A = FullyConnected('fctV_{}'.format(i), fcA2, self.num_actions, nl=tf.identity)

                    Q = tf.add(A, V - tf.reduce_mean(A, 1, keepdims=True))
                    Q_list.append(tf.identity(Q, name='Qvalue_{}'.format(i)))

        return Q_list

###############################################################################

def get_config(files_list, agents, reward_strategy):
    """This is only used during training."""
    input_names=['state_' + str(a+1) for a in range(agents)]
    output_names=['Qvalue_' + str(a+1) for a in range(agents)]

    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(task='train', files_list=files_list, agents=agents, reward_strategy=reward_strategy),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY,
        agents=agents
    )

    return TrainConfig(
        # dataflow=expreplay,
        data=QueueInput(expreplay),
        model=Model3D(agents=agents),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel3D.update_target_param, verbose=True),
                # update target network every 10k steps
                every_k_steps=10000 // UPDATE_FREQ),
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(60, 4e-4), (100, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                # 1->0.1 in the first million steps
                [(0, 1), (10, 0.1), (320, 0.01)],
                interp='linear'),
            PeriodicTrigger(
                Evaluator(nr_eval=EVAL_EPISODE, input_names=input_names,
                          output_names=output_names, files_list=files_list,
                          get_player_fn=get_player, agents=agents,
                          reward_strategy=reward_strategy),
                every_k_epochs=EPOCHS_PER_EVAL),
            HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )



###############################################################################
###############################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform. Must load a pretrained model if task is "play" or "eval"',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling','DuelingDouble'],
                        default='DQN')
    parser.add_argument('--files', type=argparse.FileType('r'), nargs='+',
                        help="""Filepath to the text file that comtains list of images.
                                Each line of this file is a full path to an image scan.
                                For (task == train or eval) there should be two input files ['images', 'landmarks']""")
    parser.add_argument('--saveGif', help='save gif image of the game',
                        action='store_true', default=False)
    parser.add_argument('--saveVideo', help='save video of the game',
                        action='store_true', default=False)
    parser.add_argument('--logDir', help='store logs in this directory during training',
                        default='train_log')
    parser.add_argument('--name', help='name of current experiment for logs',
                        default='experiment_1')
    parser.add_argument('--agents', help='Number of agents', type=int, default=1)
    parser.add_argument('--reward_strategy', help='Reward strategies: 1 is simple, 2 is line based, 3 is agent based',default=1)



    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # check valid number of agents:
    assert args.agents > 0

    # check input files
    if args.task == 'play':
        error_message = """Wrong input files {} for {} task - should be 1 \'images.txt\' """.format(len(args.files), args.task)
        assert len(args.files) == 1, (error_message)
    else:
        error_message = """Wrong input files {} for {} task - should be 2 [\'images.txt\', \'landmarks.txt\'] """.format(len(args.files), args.task)
        assert len(args.files) == 2, (error_message)


    METHOD = args.algo
    # load files into env to set num_actions, num_validation_files
    init_player = MedicalPlayer(files_list=args.files,
                                screen_dims=IMAGE_SIZE,
                                # TODO: why is this always play?
                                task='play',
                                agents=args.agents,
                                reward_strategy=args.reward_strategy)
    NUM_ACTIONS = init_player.action_space.n
    num_files = init_player.files.num_files

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model3D(agents=args.agents),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue']))
        # demo pretrained model one episode at a time
        if args.task == 'play' or args.task == 'eval':
            play_n_episodes(get_player(files_list=args.files, viz=0.01,
                                       saveGif=args.saveGif,
                                       saveVideo=args.saveVideo,
                                       task=args.task,
                                       agents=args.agents,
                                       reward_strategy=args.reward_strategy),
                            pred, num_files)
    else:  # train model
        """
        logger_dir = os.path.join(args.logDir, args.name)
        logger.set_logger_dir(logger_dir)
        config = get_config(args.files, args.agents, args.reward_strategy)
        if args.load:  # resume training from a saved checkpoint
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
        """
        environment = get_player(task='train', files_list=args.files, agents=args.agents, reward_strategy=1, viz=False)
        trainer = Trainer(environment,
                          batch_size = BATCH_SIZE, # Is batch size influencing oscillations? How come
                          image_size = IMAGE_SIZE,
                          frame_history = FRAME_HISTORY,
                          update_frequency = UPDATE_FREQ,
                          replay_buffer_size = MEMORY_SIZE,
                          init_memory_size = INIT_MEMORY_SIZE,
                          gamma = GAMMA,
                          steps_per_epoch = STEPS_PER_EPOCH, # TODO: rename to steps per episode?
                          ).train()
