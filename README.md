
# RL-Medical

Deep Reinforcement Learning (DRL) agents for anatomical landmark detection using PyTorch.
This is the code for the paper [Communicative Reinforcement Learning Agents for Landmark Detection in Brain Images](https://arxiv.org/abs/2008.08055).

## Introduction

Accurate detection of anatomical landmarks is an essential step in several medical imaging tasks. This repository implements a novel communicative multi-agent reinforcement learning (C-MARL) system to automatically detect landmarks in 3D medical images. C-MARL enables the agents to learn explicit communication channels, as well as implicit communication signals by sharing certain weights of the architecture among all the agents.

In addition to C-MARL, the code also supports single agents and multi-agents with no communication channel (named Network3d).
This code is originally a fork from [Amir Alansary's repository](https://github.com/amiralansary/rl-medical).

10 brain MRI scans each with 20 landmarks annotated from the ADNI dataset are included in the `data` folder for convenience.

## Results
Here are few examples of the learned agents on unseen data:

* An  example  of  our  proposed  C-MARL  system  consisting  of  5  agents.  These agents are looking for 5 different landmarks in a brain MRI scan. Each agent’s ROI is represented by a yellow box and centered around a blue point, while the red point is the target landmark. ROI is sampled with 3 mm spacing at the beginning of every episode. The length of the circumference of red disks denotes the distance between the current and target landmarks in z-axis.
<p align="center">
<img src="./doc/brain_5_agents.gif" width="255">
</p>

* Similarly, 5 C-MARL agents in fetal ultrasounds scans.
<p align="center">
<img src="./images/fetal_5_agents.gif" width="255">
</p>

* Two examples for single agents: [detecting the apex point in short-axis cardiac MRI](doc/cardiac_apex.mp4) and [detecting the anterior commissure (AC) point in adult brain MRI](doc/brain_ac.mp4)


### Train

Example to train 5 C-MARL agents
```
cd src
DQN.py --task train --files data/filenames/image_files.txt data/filenames/landmark_files.txt --model_name CommNet --file_type brain --landmarks 13 14 0 1 2 --multiscale --viz 0 --train_freq 50 --write
```

With the `--write` flag, training will produce logs and a Tensorboard in the `--logDir` directory (runs by default).

Note: the landmarks flag specifies the number of agents and their target landmarks. For example, `--landmarks 0 1 1` means there are agents, one agent looks for landmark 0 while two agents look for the same landmark number 1. 

### Evaluate

* 8 C-MARL agents 
```
cd src
python DQN.py --task eval --load 'data/models/BrainMRI/CommNet8agents.pt' --files 'data/filenames/image_files.txt' 'data/filenames/landmark_files.txt' --file_type brain --landmarks 13 14 0 1 2 3 4 5 --model_name "CommNet"
```

* 5 C-MARL agents 
```
cd src
python DQN.py --task eval --load 'data/models/BrainMRI/CommNet5agents.pt' --files 'data/filenames/image_files.txt' 'data/filenames/landmark_files.txt' --file_type brain --landmarks 13 14 0 1 2 --model_name "CommNet"
```

* 3 C-MARL agents 
```
cd src
python DQN.py --task eval --load 'data/models/BrainMRI/CommNet3agents.pt' --files 'data/filenames/image_files.txt' 'data/filenames/landmark_files.txt' --file_type brain --landmarks 13 14 0 --model_name "CommNet"
```

* 8 Network3d agents 
```
cd src
python DQN.py --task eval --load 'data/models/BrainMRI/Network3d8agents.pt' --files 'data/filenames/image_files.txt' 'data/filenames/landmark_files.txt' --file_type brain --landmarks 13 14 0 1 2 3 4 5 --model_name "Network3d"
```

* Single agent
```
cd src
python DQN.py --task eval --load 'data/models/BrainMRI/SingleAgent.pt' --files 'data/filenames/image_files.txt' 'data/filenames/landmark_files.txt' --file_type brain --landmarks 13 --model_name "Network3d"
```


## Usage
```
[-h] [--load LOAD] [--task {play,eval,train}]
              [--file_type {brain,cardiac,fetal}] [--files FILES [FILES ...]]
              [--val_files VAL_FILES [VAL_FILES ...]] [--saveGif]
              [--saveVideo] [--logDir LOGDIR]
              [--landmarks [LANDMARKS [LANDMARKS ...]]]
              [--model_name {CommNet,Network3d}] [--batch_size BATCH_SIZE]
              [--memory_size MEMORY_SIZE]
              [--init_memory_size INIT_MEMORY_SIZE]
              [--max_episodes MAX_EPISODES]
              [--steps_per_episode STEPS_PER_EPISODE]
              [--target_update_freq TARGET_UPDATE_FREQ]
              [--save_freq SAVE_FREQ] [--delta DELTA] [--viz VIZ]
              [--multiscale] [--write] [--train_freq TRAIN_FREQ]

optional arguments:
  -h, --help            show this help message and exit
  --load LOAD           Path to the model to load (default: None)
  --task {play,eval,train}
                        task to perform, must load a pretrained model if task
                        is "play" or "eval" (default: train)
  --file_type {brain,cardiac,fetal}
                        Type of the training and validation files (default:
                        train)
  --files FILES [FILES ...]
                        Filepath to the text file that contains list of
                        images. Each line of this file is a full path to an
                        image scan. For (task == train or eval) there should
                        be two input files ['images', 'landmarks'] (default:
                        None)
  --val_files VAL_FILES [VAL_FILES ...]
                        Filepath to the text file that contains list of
                        validation images. Each line of this file is a full
                        path to an image scan. For (task == train or eval)
                        there should be two input files ['images',
                        'landmarks'] (default: None)
  --saveGif             Save gif image of the game (default: False)
  --saveVideo           Save video of the game (default: False)
  --logDir LOGDIR       Store logs in this directory during training (default:
                        runs)
  --landmarks [LANDMARKS [LANDMARKS ...]]
                        Landmarks to use in the images (default: [1])
  --model_name {CommNet,Network3d}
                        Models implemented are: Network3d, CommNet (default:
                        CommNet)
  --batch_size BATCH_SIZE
                        Size of each batch (default: 64)
  --memory_size MEMORY_SIZE
                        Number of transitions stored in exp replay buffer. If
                        too much is allocated training may abruptly stop.
                        (default: 100000.0)
  --init_memory_size INIT_MEMORY_SIZE
                        Number of transitions stored in exp replay before
                        training (default: 30000.0)
  --max_episodes MAX_EPISODES
                        "Number of episodes to train for" (default: 100000.0)
  --steps_per_episode STEPS_PER_EPISODE
                        Maximum steps per episode (default: 200)
  --target_update_freq TARGET_UPDATE_FREQ
                        Number of epochs between each target network update
                        (default: 10)
  --save_freq SAVE_FREQ
                        Saves network every save_freq steps (default: 1000)
  --delta DELTA         Amount to decreases epsilon each episode, for the
                        epsilon-greedy policy (default: 0.0001)
  --viz VIZ             Size of the window, None for no visualisation
                        (default: 0.01)
  --multiscale          Reduces size of voxel around the agent when it
                        oscillates (default: False)
  --write               Saves the training logs (default: False)
  --train_freq TRAIN_FREQ
                        Number of agent steps between each training step on
                        one mini-batch (default: 1)
```

## Development

Issues and pull requests are very welcomed.

## Citation

If you use this code in your research, please cite this paper:

```
@article{leroy2020communicative,
  title={Communicative Reinforcement Learning Agents for Landmark Detection in Brain Images},
  author={Leroy, Guy and Rueckert, Daniel and Alansary, Amir},
  journal={arXiv preprint arXiv:2008.08055},
  year={2020}
}
```

## Resources

More information on this project:
- [Master's project presentation video](https://www.youtube.com/watch?v=Q8Sy4_YbTFE)
- [Master's project report](https://gml16.github.io/projects/mastersthesis.pdf)