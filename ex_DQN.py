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
        self.Weight_1 = np.load("Cache_W_11.npy")
        self.Weight_2 = np.load("Cache_W_22.npy")
        self.Weight_3 = np.load("Cache_W_33.npy")
        self.Bias_1 = np.load("B_11.npy")
        self.Bias_2 = np.load("B_22.npy")
        self.Bias_3 = np.load("B_33.npy")

        self._build_network()

    def _build_network(self, h_size=50, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            #W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            #b1 = tf.Variable(tf.random_normal([h_size]))
            W1 = tf.constant(self.Weight_1)
            b1 = tf.constant(self.Bias_1)
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1) + b1)
            #W2 = tf.get_variable("W2", shape=[h_size, h_size],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            #b2 = tf.Variable(tf.random_normal([h_size]))
            W2 = tf.constant(self.Weight_2)
            b2 = tf.constant(self.Bias_2)
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2) + b2)
            #W3 = tf.get_variable("W3", shape=[h_size, self.output_size],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            #b3 = tf.Variable(tf.random_normal([self.output_size]))
            W3 = tf.constant(self.Weight_3)
            b3 = tf.constant(self.Bias_3)

            self._Qpred = tf.matmul(layer2, W3) + b3
        '''
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
        '''
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})
    '''
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
    '''