from gym_env import GameEnv
import preprocess_array

class HyperParametersModel():

    def __init__(self):
        #MODEL HYPERPARAMETERS
        self.state_size = [172, 136, 1] #width, height, channels
        self.action_size = GameEnv().env.action_space
        self.learning_rate = 0.00025

        #TRANING HYPERPARAMETERS
        self.total_episodes = 50
        self.max_steps = 5000
        self.batch_size = 64

        #EXPLORATION PARAMETERS
        self.explore_start = 1.0
        self.explore_stop = 0.01
        self.decay_rate = 0.00001

        #QLEARNING HYPERPARAMETER
        self.gamma = 0.9

        #MEMORY HYPERPARAMETERS
        self.pretrain_lenght = self.batch_size
        self.memory_size = 1000000

        #PREPROCESING PARAMETERS
        self.stack_size = 4

        self.training = False

        self.episode_render = False