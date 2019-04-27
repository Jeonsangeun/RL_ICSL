import tensorflow as tf
import numpy as np
import ex_env as Cache

env = Cache.cache_replacement()


class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.Weight_1 = []
        self.W_1 = np.load("Cache_W_1@.npy")
        self.Weight_2 = []
        self.W_2 = np.load("Cache_W_2@.npy")
        self.Weight_3 = []
        self.W_3 = np.load("Cache_W_3@.npy")
        self.Weight_4 = []
        self.W_4 = np.load("Cache_W_4@.npy")
        self.Weight_5 = []
        self.W_5 = np.load("Cache_W_5@.npy")

        self._build_network()

    def _build_network(self, h_size=100, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            #W1 = tf.Variable(tf.constant(self.W_1), dtype=float)
            #b1 = tf.Variable(tf.random_normal([h_size]))
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))
            W2 = tf.get_variable("W2", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            #W2 = tf.Variable(tf.constant(self.W_2), dtype=float)
            #b2 = tf.Variable(tf.random_normal([h_size]))
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2))
            W3 = tf.get_variable("W3", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            #W3 = tf.Variable(tf.constant(self.W_3), dtype=float)
            #b3 = tf.Variable(tf.random_normal([h_size]))
            layer3 = tf.nn.tanh(tf.matmul(layer2, W3))
            W4 = tf.get_variable("W4", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            #W4 = tf.Variable(tf.constant(self.W_4), dtype=float)
            #b4 = tf.Variable(tf.random_normal([h_size]))
            layer4 = tf.nn.tanh(tf.matmul(layer3, W4))
            W5 = tf.get_variable("W8", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            #W8 = tf.Variable(tf.constant(self.W_8), dtype=float)
            #b5 = tf.Variable(tf.random_normal([h_size]))
            layer5 = tf.nn.tanh(tf.matmul(layer4, W5))
            W6 = tf.get_variable("W9", shape=[h_size, self.output_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            #W9 = tf.Variable(tf.constant(self.W_5), dtype=float)
            b6 = tf.Variable(tf.random_normal([self.output_size]))

            self._Qpred = tf.matmul(layer5, W6)

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

        self.Weight_1 = W1
        self.Weight_2 = W2
        self.Weight_3 = W3
        self.Weight_4 = W4
        self.Weight_5 = W5

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
