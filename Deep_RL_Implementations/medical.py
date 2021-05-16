from agents.actor_critic_agents.A3C import A3C
from environments.medical import MedicalPlayer
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from logger import Logger

import argparse

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


#parser.add_argument('--load', help='Path to the model to load')
#parser.add_argument(
#    '--task',
#    help='''task to perform,
#            must load a pretrained model if task is "play" or "eval"''',
#    choices=['play', 'eval', 'train'], default='train')
#parser.add_argument(
#    '--file_type', help='Type of the training and validation files',
#    choices=['brain', 'cardiac', 'fetal'], default='train')

parser.add_argument(
    '--files', type=argparse.FileType('r'), nargs='+',
    help="""Filepath to the text file that contains list of images.
            Each line of this file is a full path to an image scan.
            For (task == train or eval) there should be two input files
            ['images', 'landmarks']""")

args = parser.parse_args()

config = Config()
config.seed = 1
IMAGE_SIZE = (45, 45, 45)

# num_possible_states = (height * width) ** (1 + 1*random_goal_place)
# embedding_dimensions = [[num_possible_states, 20]]
# print("Num possible states ", num_possible_states)


directory = None
screen_dims = IMAGE_SIZE
viz = False
saveGif = False
saveVideo = False
task = 'train'
files_list = args.files
file_type = 'cardiac'
landmark_ids = [4]
history_length = 20
multiscale = True
agents = len(landmark_ids)

log_dir = 'runs'
write = True
save_freq = 1000
log_comment = 'cardiacA3C'
logger = Logger(log_dir, write, save_freq, comment = log_comment)

config.environment = MedicalPlayer(
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
    logger=None)

config.num_episodes_to_run = 1000
config.file_to_save_data_results = "Data_and_Graphs/medical.pkl"
config.file_to_save_results_graph = "Data_and_Graphs/medical.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {
    "DQN_Agents": {
        "linear_hidden_units": [30, 10],
        "learning_rate": 0.01,
        "buffer_size": 40000,
        "batch_size": 256,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        #"embedding_dimensions": embedding_dimensions,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": 10,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "tau": 0.01,
        "exploration_cycle_episodes_length": None,
        "learning_iterations": 1,
        "clip_rewards": False
    },

    "SNN_HRL": {
        "SKILL_AGENT": {
            "num_skills": 20,
            "regularisation_weight": 1.5,
            "visitations_decay": 0.9999,
            "episodes_for_pretraining": 300,
            "batch_size": 256,
            "learning_rate": 0.001,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            # "embedding_dimensions": [embedding_dimensions[0],
            #                          [20, 6]],
            "batch_norm": False,
            "gradient_clipping_norm": 2,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False
        },

        "MANAGER": {
            "timesteps_before_changing_skill": 6,
            "linear_hidden_units": [10, 5],
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "batch_size": 256,
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            #"embedding_dimensions": embedding_dimensions,
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 50,
            "discount_rate": 0.99,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False

        }

    },

    "Actor_Critic_Agents": {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],

        "columns_of_data_to_be_embedded": [0],
        #"embedding_dimensions": embedding_dimensions,
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 50.0,
        "normalise_rewards": True,
        "clip_rewards": False

    },


    "DIAYN": {

        "num_skills": 5,
        "DISCRIMINATOR": {
            "learning_rate": 0.01,
            "linear_hidden_units": [20, 10],
            "columns_of_data_to_be_embedded": [0],
            #"embedding_dimensions": embedding_dimensions,
        },

        "AGENT": {
            "learning_rate": 0.01,
            "linear_hidden_units": [20, 10],
        }
    },


    "HRL": {
        "linear_hidden_units": [10, 5],
        "learning_rate": 0.01,
        "buffer_size": 40000,
        "batch_size": 256,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        #"embedding_dimensions": embedding_dimensions,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": 400,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "tau": 0.01

    }


}

if __name__== '__main__':


    AGENTS = [A3C] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
