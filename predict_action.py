import numpy as np
import random


def predict_action(explore_start, explore_stop, decay_rate, decay_step,
                   state, DQNetwork, sess, possible_actions=np.array([0, 1, 2])):

    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(
                                                            -decay_rate*decay_step)
    if explore_probability > exp_exp_tradeoff:
        choice = random.randint(1, len(possible_actions))-1
        action = possible_actions[choice]

    else:
        Qs = sess.run(DQNetwork.output,
                      feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_probability
