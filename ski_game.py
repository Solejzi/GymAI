import numpy as np

import gym
from stack_arrays import stack_arrays

from action_to_memory import action_to_memory
from gym_env import GameEnv
from DQNetwork import DQNetwork
from memory import Memory
from predict_action import predict_action
import tensorflow as tf
import warnings
from collections import deque
warnings.filterwarnings('ignore')

env = GameEnv()

tf.reset_default_graph()
DQnetwork = DQNetwork()
memory = Memory(max_size=DQnetwork.memory_size)

#tensorboard
writer = tf.summary.FileWriter("tensorboard/dqn/1")
tf.summary.scalar("Loss", DQnetwork.loss)
write_op = tf.summary.merge_all()

stacked_arrays = deque([np.zeros((172, 136), dtype=np.int) for i in range(DQnetwork.stack_size)], maxlen=4)

save_action = action_to_memory(env, DQnetwork, memory, stacked_arrays)
saver = tf.train.Saver()


if DQnetwork.training:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        decay_step = 0
    for episode in range(DQnetwork.total_episodes):
        step = 0
        episode_rewards = []
        state = env.env.reset()

        state, stacked_array = stack_arrays(stacked_array, state, True)

        while step < DQnetwork.max_steps:
            step += 1

            decay_step += 1

            action, explore_probability = predict_action(DQnetwork.explore_start, DQnetwork.explore_start,
                                                         DQnetwork.decay_rate, DQnetwork.decay_step, DQnetwork.state)

            next_state, reward, done, _ = env.env.step(action)

            if DQnetwork.episode_render:
                env.env.render()

            if done:
                next_state = np.zeros((172, 136), dtype=np.uint16)

                next_state, stacked_array = stack_arrays(stacked_array, next_state, False)

                step = DQnetwork.max_steps

                total_reward = np.sum(episode_rewards)

                print( 'EP: {}'.format(episode),
                        'TOTAL REWARD: {}'.format(total_reward),
                        'EXPLORE P: {:.4f}'.format(explore_probability),
                        'Traning Loss {:.4f}'.format(DQnetwork.loss))
                rewards_list.append((episode, total_reward))

                memory.add((state, action, reward, next_state, done))

            else:
                next_state, stacked_array = stack_arrays(stacked_array,
                                                         next_state, False)
                memory.add((state, action, reward, next_state, done))

                state = next_state

##############################################################

        batch = memory.sample(DQnetwork.batch_size)
        print(batch)
        states_mb = np.array([each[0] for each in batch], ndmin=3)

        actions_mb = np.array([each[1] for each in batch])

        rewards_mb = np.array([each[2] for each in batch])

        next_states_mb = np.array([each[3] for each in batch], ndmin=3)

        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []
        Qs_next_state = sess.run(DQnetwork.output, feed_dict= {DQnetwork.inputs_: next_states_mb})

        for i in range(0, len(batch)):
            terminal = dones_mb[i]
            if terminal:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        targets_mb = np.array([each for each in target_Qs_batch])

        loss, _ = sess.run([DQnetwork.loss, DQnetwork.optimizer],
                           feed_dict={DQnetwork.inputs_: states_mb,
                           DQnetwork.target_Q : targets_mb,
                           DQnetwork.actions_ : actions_mb})

        summary = sess.run(write_op, feed_dict={DQnetwork.inputs_: states_mb,
                           DQnetwork.target_Q : targets_mb,
                           DQnetwork.actions_ : actions_mb})

        writer.add_summary(summary, episode)
        writer.flush()

        if episode % 5 == 0:
            save_path = saver.save((sess, "./models/model.ckpt"))
            print('model saved')
####################################################################

with tf.Session() as sess:
    total_test_rewards = []

    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    possible_actions = (0, 1 ,2)
    for episode in range(1):
        total_rewards = 0

        state = env.env.reset()
        state, stacked_array = stack_arrays(stacked_array, state, True)

        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            # Reshape the state
            state = state.reshape((1, *DQnetwork.state_size))
            # Get action from Q-network
            # Estimate the Qs values state
            Qs = sess.run(DQnetwork.output, feed_dict={DQnetwork.inputs_: state})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice]

            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.env.step(action)
            env.env.render()

            total_rewards += reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            next_state, stacked_array = stack_arrays(stacked_array, next_state, False)
            state = next_state

    env.env.close()















# for _ in range(10):
#     observation = env.env.reset()
#
#     for t in range(1000):
#         time.sleep(0.05)
#
#         env.env.render()
#         action = env.env.action_space.sample()
#         observation, reward, done, info = env.env.step(action)
#         if done:
#             print(observation, reward)
#             print("Episode finished after {} timesteps".format(t + 1))
#             break
#
#     env.env.close()
#
# print(env.env.observation_space)
# print(env.env.action_space)




