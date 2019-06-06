from collections import deque

import numpy as np
from preprocess_array import preprocess_array


stack_size = 4


def stack_arrays(stacked_arrays, state, is_new_episode):

    array = preprocess_array(state)
    print(array.shape)
    if is_new_episode:
        stacked_arrays = deque([np.zeros((172, 136), dtype=np.float16) for i in range(stack_size)], maxlen=4)

        stacked_arrays.append(array)
        stacked_arrays.append(array)
        stacked_arrays.append(array)
        stacked_arrays.append(array)


        stacked_state = np.stack(stacked_arrays, axis=2)

    else:
        stacked_arrays.append(array)

        stacked_state = np.stack(stacked_arrays, axis=2)

    return stacked_state, stacked_arrays
