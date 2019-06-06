from stack_arrays import stack_arrays
import random
import numpy as np
from preprocess_array import preprocess_array

def action_to_memory(env, DQNetwork, memory, stacked_arrays):
    actions = [0, 1, 2]
    for i in range(DQNetwork.pretrain_lenght):
        if i == 0:
            state = env.env.reset()

            state, stacked_arrays = stack_arrays(stacked_arrays, state, True)

        choice = random.randint(1, len(actions)) - 1
        action = actions[choice]
        observation, reward, done, _ = env.env.step(action)
        observation = preprocess_array(observation)

        if done:
            next_state = np.zeros(state.shape)

            memory.add((state, reward, next_state, done))

            state = env.env.reset()

            state, stacked_arrays = stack_arrays(stacked_arrays, state, True)

        else:
            memory.add((state, reward, observation, done))
            state = observation
