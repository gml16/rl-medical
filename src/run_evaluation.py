"""
Evaluation helper for trained models.

Prefer passing evaluation config explicitly through CLI args.
Legacy checkpoint-name inference is kept as a fallback for old runs.
"""
import argparse

import itertools
import warnings
import numpy as np
import torch

from DQNModel import DQN
from DQN import get_player
from logger import Logger
from evaluator import Evaluator

FRAME_HISTORY = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--model_files', type=str, nargs='+', required=True,
        help="Filepath to the models that must be evaluated")

    parser.add_argument(
        '--files', type=argparse.FileType('r'), nargs='+',
        help="""Filepath to the text file that contains list of images.
                Each line of this file is a full path to an image scan.
                For (task == train or eval) there should be two input files
                ['images', 'landmarks']""")
    
    parser.add_argument(
        '--model_name', choices=['CommNet', 'Network3D'],
        help='Model architecture to evaluate')

    parser.add_argument(
        '--agents', type=int, choices=[3, 5, 10],
        help='Number of agents used by the model')

    parser.add_argument(
        '--file_type', choices=['brain', 'cardiac', 'fetal'],
        help='Dataset type used for evaluation')

    parser.add_argument(
        '--collective_rewards', action='store_true',
        help='Use attention-based collective rewards')

    args = parser.parse_args()

    explicit_args = [args.model_name, args.agents, args.file_type]
    if any(value is not None for value in explicit_args) and not all(
            value is not None for value in explicit_args):
        raise ValueError(
            "Please provide --model_name, --agents and --file_type together, "
            "or omit all three to use legacy name-based inference."
        )
    
    logger = Logger(None, False, None)
    x = [0.5,0.25,0.75]
    y = [0.5,0.25,0.75]
    z = [0.5,0.25,0.75]
    fixed_spawn = list(np.array(list(itertools.product(x, y, z))).flatten())

    for model_path in args.model_files:
        # mypath = os.path.normpath(model_path)

        # python DQN.py --task eval --load runs/Mar01_04-16-35_monal03.doc.ic.ac.ukbrain10DefaultNetwork3d/best_dqn.pt --files /vol/biomedic2/aa16914/shared/RL_Guy/rl-medical/examples/LandmarkDetection/DQN/data/filenames/brain_test_files.txt /vol/biomedic2/aa16914/shared/RL_Guy/rl-medical/examples/LandmarkDetection/DQN/data/filenames/brain_test_landmarks.txt --file_type brain --landmarks 13 14 0 1 2 3 4 5 6 7 --model_name Network3d --viz 0
        if args.model_name and args.agents and args.file_type:
            fullName = model_path.split("/")[-2]
            model_name = args.model_name
            agents = args.agents
            file_type = args.file_type
            collective_rewards = "attention" if args.collective_rewards else False
        else:
            warnings.warn(
                "Inferring evaluation config from checkpoint name is deprecated and fragile. "
                "Please pass --model_name, --agents and --file_type explicitly.",
                RuntimeWarning,
            )

            fullName = model_path.split("/")[-2]
            name = fullName.split("doc.ic.ac.uk")[-1]

            if "CommNet" in name:
                model_name = "CommNet"
            elif "Network3d" in name or "Network3D" in name:
                model_name = "Network3D"
            else:
                raise ValueError("Could not infer model name from checkpoint path: {}".format(model_path))

            if "10" in name:
                agents = 10
            elif "5" in name:
                agents = 5
            elif "3" in name:
                agents = 3
            else:
                raise ValueError("Could not infer number of agents from checkpoint path: {}".format(model_path))

            if "brain" in name:
                file_type = "brain"
                landmarks = [13, 14, 0, 1, 2, 3, 4, 5, 6, 7]
            elif "cardiac" in name:
                file_type = "cardiac"
                landmarks = [4, 5, 0, 1, 2, 3, 4, 5, 6, 7]
            elif "fetal" in name:
                file_type = "fetal"
                landmarks = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise ValueError("Could not infer file type from checkpoint path: {}".format(model_path))

            collective_rewards = "attention" if "team" in name else False

        if args.model_name and args.agents and args.file_type:
            if file_type == "brain":
                landmarks = [13, 14, 0, 1, 2, 3, 4, 5, 6, 7]
            elif file_type == "cardiac":
                landmarks = [4, 5, 0, 1, 2, 3, 4, 5, 6, 7]
            elif file_type == "fetal":
                landmarks = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]

        landmarks = landmarks[:agents]
        files = args.files
        
        dqn = DQN(agents, frame_history=FRAME_HISTORY, logger=logger,
                  type=model_name, collective_rewards=collective_rewards)
        model = dqn.q_network
        model.load_state_dict(torch.load(model_path, map_location=model.device))
        model.eval()
        environment = get_player(files_list=files,
                                 file_type=file_type,
                                 landmark_ids=landmarks,
                                 saveGif=False,
                                 saveVideo=False,
                                 task="eval",
                                 agents=agents,
                                 viz=0,
                                 logger=logger)
        evaluator = Evaluator(environment, model, logger,
                                agents, 200)
        mean, std = evaluator.play_n_episodes(fixed_spawn=fixed_spawn, silent=True)
        logger.log(f"{fullName}: mean {mean}, std {std}")
