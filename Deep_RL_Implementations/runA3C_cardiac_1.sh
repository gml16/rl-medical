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
python medical.py --files /vol/biomedic2/aa16914/shared/RL_Guy/rl-medical/examples/LandmarkDetection/DQN/data/filenames/cardiac_train_files.txt /vol/biomedic2/aa16914/shared/RL_Guy/rl-medical/examples/LandmarkDetection/DQN/data/filenames/cardiac_train_landmarks.txt
