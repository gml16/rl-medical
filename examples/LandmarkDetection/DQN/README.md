# Anatomical Landmark Detection

Automatic detection of anatomical landmarks is an important step for a wide range of applications in medical image analysis. In this project, we formulate the landmark detection problem as a sequential decision process navigating in a medical image environment towards the target landmark. We deploy multiple Deep Q-Network (DQN) based architectures to train agents that can learn to identify the optimal path to the point of interest. This code also supports both fixed- and multi-scale search strategies with hierarchical action steps in a coarse-to-fine manner.

<p align="center">
<img style="float: center;" src="images/framework.png" width="465">
<img style="float: center;" src="images/actions.png" width="130">
</p>

---
## Results
Below are the results from the source repository, this forked repository is a work in progress.
Here are few examples of the learned agent for landmark detection on unseen data:

* Detecting the apex point in short-axis cardiac MRI [(HQ video)](videos/cardiac_apex.mp4)
<p align="center">
<img src="./images/cardiac_apex.gif" width="255">
</p>

* Detecting the anterior commissure (AC) point in adult brain MRI [(HQ video)](videos/brain_ac.mp4)
<p align="center">
<img src="./images/brain_ac.gif" width="255">
</p>

* Detecting the cavum septum pellucidum (CSP) point in fetal head ultrasound [(HQ video)](videos/fetal_csp.mp4)
<p align="center">
<img src="./images/fetal_csp.gif" width="255">
</p>
---

### Train
```
python DQN.py --task train --files './data/filenames/image_files.txt' './data/filenames/landmark_files.txt' --agents 3 --model_name "CommNet"
```

### Evaluate
```
python DQN.py --task eval --load 'data/models/BrainMRI/CommNet3agents.pt' --files './data/filenames/image_files.txt' './data/filenames/landmark_files.txt' --agent 3 --model_name "CommNet"
```

### Test
```
python DQN.py --task play --load 'data/models/BrainMRI/CommNet3agents.pt' --files './data/filenames/image_files.txt' --agent 3 --model_name "CommNet"
```


## Usage
```
[--files FILES [FILES ...]] [--saveGif] [--saveVideo]
[--logDir LOGDIR] [--agents AGENTS]
[--model_name {CommNet,Network3d}] [--batch_size BATCH_SIZE]
[--memory_size MEMORY_SIZE]
[--init_memory_size INIT_MEMORY_SIZE]
[--max_episodes MAX_EPISODES]
[--steps_per_episode STEPS_PER_EPISODE]
[--target_update_freq TARGET_UPDATE_FREQ] [--delta DELTA]
[--viz VIZ] [--multiscale] [--write]

optional arguments:
-h, --help            show this help message and exit
--load LOAD           load model
--task {play,eval,train}
          task to perform. Must load a pretrained model if task
          is "play" or "eval"
--files FILES [FILES ...]
          Filepath to the text file that comtains list of
          images. Each line of this file is a full path to an
          image scan. For (task == train or eval) there should
          be two input files ['images', 'landmarks']
--saveGif             Save gif image of the game
--saveVideo           Save video of the game
--logDir LOGDIR       Store logs in this directory during training
--agents AGENTS       Number of agents
--model_name {CommNet,Network3d}
          Models implemented are: Network3d, CommNet
--batch_size BATCH_SIZE
          Size of each batch
--memory_size MEMORY_SIZE
          Number of transitions stored in exp replay buffer. If
          too much is allocated training may abruptly stop.
--init_memory_size INIT_MEMORY_SIZE
          Number of transitions stored in exp replay buffer
          before training
--max_episodes MAX_EPISODES
          "Number of episodes to train for"
--steps_per_episode STEPS_PER_EPISODE
          Maximum steps per episode
--target_update_freq TARGET_UPDATE_FREQ
          Number of episodes between each target network update
--delta DELTA         Amount to decreases epsilon each step, for the
          epsilon-greedy policy
--viz VIZ             Size of the window, None for no visualisation
--multiscale          Reduces size of voxel around the agent when it
          oscillates
--write               Saves the training logs
PS C:\Users\gmler\Git\thesis\rl-medical\examples\Landmark

```

## Citation

If you use this code in your research, please cite this paper:

```
@article{alansary2019evaluating,
  title={{Evaluating Reinforcement Learning Agents for Anatomical Landmark Detection}},
  author={Alansary, Amir and Oktay, Ozan and Li, Yuanwei and Le Folgoc, Loic and
          Hou, Benjamin and Vaillant, Ghislain and Kamnitsas, Konstantinos and
          Vlontzos, Athanasios and Glocker, Ben and Kainz, Bernhard and Rueckert, Daniel},
  journal={Medical Image Analysis},
  year={2019},
  publisher={Elsevier}
}
 ```
