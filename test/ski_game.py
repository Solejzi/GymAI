import numpy as np

import gym
from preprocess_array import preprocess_array
import random
import time
import warnings
warnings.filterwarnings('ignore')
listlist = []
env = gym.make('Skiing-v0')

for _ in range(10):


    observation = env.reset()

    for t in range(1000):
        time.sleep(0.05)
        if t >= 6:
            print(preprocess_array(observation).shape)
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(observation, reward)
            print("Episode finished after {} timesteps".format(t + 1))
            break

    env.close()

print(env.observation_space)
print(env.action_space)




