#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=nb420 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/nb420/miniconda3/bin/:$PATH
source activate rl-medical
source /vol/cuda/10.2.89-cudnn7.6.4.38/setup.sh
#source /vol/cuda/10.0.130/setup.csh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
python --version
uptime
python DQN.py --task train --files /vol/biomedic2/aa16914/shared/RL_Guy/rl-medical/examples/LandmarkDetection/DQN/data/filenames/cardiac_train_files.txt /vol/biomedic2/aa16914/shared/RL_Guy/rl-medical/examples/LandmarkDetection/DQN/data/filenames/cardiac_train_landmarks.txt --val_files /vol/biomedic2/aa16914/shared/RL_Guy/rl-medical/examples/LandmarkDetection/DQN/data/filenames/cardiac_val_files.txt /vol/biomedic2/aa16914/shared/RL_Guy/rl-medical/examples/LandmarkDetection/DQN/data/filenames/cardiac_val_landmarks.txt --file_type cardiac --landmarks 4 5 0 --model_name CommNet --multiscale --viz 0 --batch_size 48 --max_episodes 10000000 --init_memory_size 10000 --memory_size 10000 --steps_per_episode 200 --target_update_freq 10 --train_freq 50 --delta 0.0001 --write --lr 4e-4 --log_comment cardiacMasterCommNet3AgentsFixInit --fix_init
