"""
This script is hardcoded to run evaluation on our models using our naming convention.
Do not use as is.
"""

import argparse

import itertools
import numpy as np
import torch

from DQNModel import DQN
from main import get_player
from logger import Logger
from evaluator import Evaluator
from TD3_evaluator import Evaluator as TD3Evaluator
from TD3Model import TD3

FRAME_HISTORY = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--model_files', type=argparse.FileType('r'), nargs='+',
        help="Filepath to the models that must be evaluated")
    parser.add_argument(
        '--files', type=argparse.FileType('r'), nargs='+',
        help="""Filepath to the text file that contains list of images.
                Each line of this file is a full path to an image scan.
                For (task == train or eval) there should be two input files
                ['images', 'landmarks']""")
    parser.add_argument(
        '--viz', help='Size of the window, None for no visualisation',
        default=0, type=float)
    parser.add_argument(
        '--stopping_criterion', help='Stopping criterion: oscillation or threshold',
        default="osc", choices=['osc', 'other'], type=str)
    # parser.add_argument(
    #     '--fixed_spawn', nargs='*',  type=float,
    #     help='Starting position of the agents during rollout. Randomised if not specified.',)

    args = parser.parse_args()
    logger = Logger(None, False, None)
    x = [0.5,0.25,0.75]
    y = [0.5,0.25,0.75]
    z = [0.5,0.25,0.75]
    fixed_spawn = list(np.array(list(itertools.product(x, y, z))).flatten())

    #set to 1 if not actor critic approach
    step=2

    #for f in args.model_files:
    for i in range(0, len(args.model_files), step):
        f = args.model_files[i]
        f_critic = args.model_files[i+1]
        # mypath = os.path.normpath(f)

        # python DQN.py --task eval --load runs/Mar01_04-16-35_monal03.doc.ic.ac.ukbrain10DefaultNetwork3d/best_dqn.pt --files /vol/biomedic2/aa16914/shared/RL_Guy/rl-medical/examples/LandmarkDetection/DQN/data/filenames/brain_test_files.txt /vol/biomedic2/aa16914/shared/RL_Guy/rl-medical/examples/LandmarkDetection/DQN/data/filenames/brain_test_landmarks.txt --file_type brain --landmarks 13 14 0 1 2 3 4 5 6 7 --model_name Network3d --viz 0
        fullName = f.name.split("/")[-2] # e.g. Mar01_04-16-35_monal03.doc.ic.ac.ukbrain10DefaultNetwork3d
        name = fullName.split("doc.ic.ac.uk")[-1]
        continuous = False
        if "CommNet" in name:
            model_name = "CommNet"
        elif "Network3d" in name:
            model_name = "Network3d"
        elif "TD3V2" in name:
            model_name = "TD3V2"
            continuous = True

        # if "3" in name:
        #     agents = 3
        # elif "5" in name:
        #     agents = 5
        # elif "10" in name:
        #     agents = 10
        if "1Agent" in name:
            agents = 1

        if "brain" in name:
            file_type = "brain"
            file_type2 = "brain"
            landmarks = [13, 14, 0, 1, 2, 3, 4, 5, 6, 7]
        elif "cardiac" in name:
            file_type = "cardiac"
            file_type2 = "cardiac"
            landmarks = [4, 5, 0, 1, 2, 3, 4, 5, 6, 7]
        elif "fetal" in name:
            file_type = "fetal"
            file_type2 = "fetalUS"
            landmarks = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
        landmarks = landmarks[:agents]

        files = args.files

        if "team" in name:
            collective_rewards = "attention"

        environment = get_player(files_list=files,
                                 file_type=file_type,
                                 landmark_ids=landmarks,
                                 saveGif=False,
                                 saveVideo=False,
                                 task="eval",
                                 agents=agents,
                                 viz=args.viz,
                                 logger=logger,
                                 continuous=continuous,
                                 stopping_criterion=args.stopping_criterion)

        if model_name == "CommNet" or model_name == "Network3d":
            dqn = DQN(agents, frame_history=FRAME_HISTORY, logger=logger,
                      type=model_name, collective_rewards=collective_rewards)
            model = dqn.q_network
            model.load_state_dict(torch.load(f.name, map_location=device))
        elif model_name == "TD3V2":
            policy = TD3(environment.observation_space.shape,
                        environment.action_space.shape[0],
                        float(environment.action_space.high[0]),
                        agents = agents)
            actor = policy.actor
            critic = policy.critic

            actor.load_state_dict(torch.load(f.name, map_location=device))
            critic.load_state_dict(torch.load(f_critic.name, map_location=device))

        if model_name == "CommNet" or model_name == "Network3d":
            evaluator = Evaluator(environment, model, logger,
                                    agents, 200)
        else:
            evaluator = TD3Evaluator(environment, actor, critic, logger,
                                    agents, 200)
        mean, std = evaluator.play_n_episodes(fixed_spawn=fixed_spawn, silent=True)
        logger.log(f"{fullName}: mean {mean}, std {std}")
