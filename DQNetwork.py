from hyperparameters import HyperParametersModel
import tensorflow as tf


class DQNetwork(HyperParametersModel):
    def __init__(self, name='DQNetwork'):
        super().__init__()

        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float16, [None, *self.state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.float16, [None, 3], name="actions")
            self.target_Q = tf.placeholder(tf.float16, [None], name="target")

            self.conv1 = tf.layers.conv2d(  inputs=self.inputs_,
                                            filters=32,
                                            kernel_size=[8, 8],
                                            strides=[4, 4],
                                            padding="valid",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(  inputs= self.conv1_out,
                                            filters=64,
                                            kernel_size=[4, 4],
                                            strides=[2, 2],
                                            padding="valid",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name = "conv2")
            self.conv2_out = tf.nn.elu(self.conv1, name="conv2_out")

            self.conv3 = tf.layers.conv2d(  inputs=self.conv2_out,
                                            filters=64,
                                            kernel_size = [3, 3],
                                            strides=[2, 2],
                                            padding="valid",
                                            kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv3")
            self.conv3_out = tf.nn.elu(self.conv1, name="conv3_out")


            self.flatten = tf.contrib.layers.flatten(self.conv3_out)


            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fcl")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=3,
                                          activation=None)

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize((self.loss))