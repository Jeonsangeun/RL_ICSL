import tensorflow as tf
import numpy as np
import Cache_env as Cache

env = Cache.cache_replacement()

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.dropout_rate = tf.placeholder("float")
        self.W_1 = []
        self.W_2 = []
        self.W_3 = []
        self.W_4 = []
        self.W_5 = []
        self.W_6 = []
        self.W_7 = []
        self.W_8 = []
        self.W_9 = []
        self.W_10 = []
        self.W_11 = []
        self.B_1 = []
        self.B_2 = []
        self.B_3 = []
        self.B_4 = []
        self.B_5 = []
        self.B_6 = []
        self.B_7 = []
        self.B_8 = []
        self.B_9 = []
        self.B_10 = []
        self.B_11 = []


        self._build_network()

    def _build_network(self, h_size=229, l_rate=0.001):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([h_size]))
            layer1 = tf.nn.leaky_relu(tf.matmul(self._X, W1)+b1)
            layer1 = tf.nn.dropout(layer1, rate=self.dropout_rate)
            W2 = tf.get_variable("W2", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([h_size]))
            layer2 = tf.nn.leaky_relu(tf.matmul(layer1, W2)+b2)
            layer2 = tf.nn.dropout(layer2, rate=self.dropout_rate)
            W3 = tf.get_variable("W3", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([h_size]))
            layer3 = tf.nn.leaky_relu(tf.matmul(layer2, W3)+b3)
            layer3 = tf.nn.dropout(layer3, rate=self.dropout_rate)
            W4 = tf.get_variable("W4", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([h_size]))
            layer4 = tf.nn.leaky_relu(tf.matmul(layer3, W4)+b4)
            layer4 = tf.nn.dropout(layer4, rate=self.dropout_rate)
            W5 = tf.get_variable("W5", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([h_size]))
            layer5 = tf.nn.leaky_relu(tf.matmul(layer4, W5)+b5)
            layer5 = tf.nn.dropout(layer5, rate=self.dropout_rate)
            W6 = tf.get_variable("W6", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b6 = tf.Variable(tf.random_normal([h_size]))
            layer6 = tf.nn.leaky_relu(tf.matmul(layer5, W6) + b6)
            layer6 = tf.nn.dropout(layer6, rate=self.dropout_rate)
            W7 = tf.get_variable("W7", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b7 = tf.Variable(tf.random_normal([h_size]))
            layer7 = tf.nn.leaky_relu(tf.matmul(layer6, W7) + b7)
            layer7 = tf.nn.dropout(layer7, rate=self.dropout_rate)
            W8 = tf.get_variable("W8", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b8 = tf.Variable(tf.random_normal([h_size]))
            layer8 = tf.nn.leaky_relu(tf.matmul(layer7, W8) + b8)
            layer8 = tf.nn.dropout(layer8, rate=self.dropout_rate)
            W9 = tf.get_variable("W9", shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b9 = tf.Variable(tf.random_normal([h_size]))
            layer9 = tf.nn.leaky_relu(tf.matmul(layer8, W9) + b9)
            layer9 = tf.nn.dropout(layer9, rate=self.dropout_rate)
            W10 = tf.get_variable("W10", shape=[h_size, h_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b10 = tf.Variable(tf.random_normal([h_size]))
            layer10 = tf.nn.leaky_relu(tf.matmul(layer9, W10) + b10)
            layer10 = tf.nn.dropout(layer10, rate=self.dropout_rate)
            W11 = tf.get_variable("W11", shape=[h_size, self.output_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b11 = tf.Variable(tf.random_normal([self.output_size]))

            self._Qpred = tf.matmul(layer10, W11) + b11

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

        self.W_1 = W1
        self.W_2 = W2
        self.W_3 = W3
        self.W_4 = W4
        self.W_5 = W5
        self.W_6 = W6
        self.W_7 = W7
        self.W_8 = W8
        self.W_9 = W9
        self.W_10 = W10
        self.W_11 = W11
        self.B_1 = b1
        self.B_2 = b2
        self.B_3 = b3
        self.B_4 = b4
        self.B_5 = b5
        self.B_6 = b6
        self.B_7 = b7
        self.B_8 = b8
        self.B_9 = b9
        self.B_10 = b10
        self.B_11 = b11


    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x, self.dropout_rate: 0})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train],
                                feed_dict={self._X: x_stack, self._Y: y_stack, self.dropout_rate: 0})
