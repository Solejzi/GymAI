from skimage.color import rgb2gray
import numpy as np
from gym_env import GameEnv

def preprocess_array(array):

    ''' Takes an observation (3d array) as an input and slice it
    to get rid off unimportant data (boarders of the screen) '''
    print(array.shape)
    # checks if an array is 3 dimensional
    if array.ndim == 3:
        processed_array = array[28:200, 8:144, 1:2]

        return processed_array


def d3_to_d2(d3array):

    ''' turns 3d array into 2d array '''

    # checks if an array is 3 dimensional
    if d3array.ndim == 3:
        pass

if __name__ == '__main__':
    a = GameEnv().env.observation_space
    print(np.array(a))
    b =preprocess_array(a.env.observation_space)
    print(b)