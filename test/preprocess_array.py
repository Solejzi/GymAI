from skimage.color import rgb2gray
import numpy as np


def preprocess_array(array):

    ''' Takes an observation (3d array) as an input and slice it
    to get rid off unimportant data (boarders of the screen) '''

    # checks if an array is 3 dimensional
    if array.ndim == 3:
        processed_array = array[28:200, 8:144, 1:2]
        return processed_array


def d3_to_d2(d3array):

    ''' turns 3d array into 2d array '''

    # checks if an array is 3 dimensional
    if d3array.ndim == 3:
        pass
