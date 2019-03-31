import numpy as np
import random as rd
import tensorflow as tf

class cache_replacement:

    def __init__(self):
        self.Num_file = 100
        self.Num_packet = 10
        self.Memory = 100
        self.L_file = 100
        self.alpha = 1.1
        self.state = np.zeros([2, 100])
        self.cost = 0
        self.count = 0
        self.file_100 = tuple(np.load("100_file.npy").tolist())
        self.rate_100 = np.load("100_zip.npy").tolist()

    def request_file(self):
        pp = np.random.choice(self.file_100, 1, p=self.rate_100)
        return pp[0]

    def reset(self):
        self.state = np.zeros([2, 100])
        '''
        for i in range(2):
            for k in range(2):
                file = cache_replacement.request_file(self)
                done = False
                while not done:
                    if file in self.state[i]:
                        file = cache_replacement.request_file(self)
                        done = False
                    if file not in self.state[i]:
                        self.state[i][2 * k] = file
                        self.state[i][2 * k + 1] = file + 0.1
                        done = True
        '''
        self.cost = 0
        self.count = 0
        file_1 = cache_replacement.request_file(self)
        user = np.random.choice((10, -10), 1,  p=[0.5, 0.5])
        return cache_replacement.make(self, file_1, user), file_1, user

    def random_action(self):
        return rd.randrange(100)

    def step(self, action, file, user):
        done = False
        cost = 0
        reward = 0
        new_state = []
        self.tmp_1 = (self.state[0])[:]
        self.tmp_2 = (self.state[1])[:]
        if user == 10:
            n = self.tmp_1[action]
            self.tmp_1 = np.delete(self.tmp_1, action)
            if file not in self.tmp_1:
                if file not in self.tmp_2:
                    self.tmp_1 = np.append(self.tmp_1, file)
                    cost += 100
                    reward -= 5
                else:
                    self.tmp_1 = np.append(self.tmp_1, file)
                    cost += 25
                    reward += 45
            else:
                self.tmp_1 = np.append(self.tmp_1, n)
                cost += 5
                reward += 100
            self.count += 1
            self.state[0] = self.tmp_1[:]
        if user == -10:
            n = self.tmp_2[action]
            self.tmp_2 = np.delete(self.tmp_2, action)
            if file not in self.tmp_2:
                if file not in self.tmp_1:
                    self.tmp_2 = np.append(self.tmp_2, file)
                    cost += 100
                    reward -= 5
                else:
                    self.tmp_2 = np.append(self.tmp_2, file)
                    cost += 25
                    reward += 45
            else:
                self.tmp_2 = np.append(self.tmp_2, n)
                cost += 5
                reward += 100
            self.count += 1
            self.state[1] = self.tmp_2[:]

        if self.count % 10 != 0:
            file += 1
            new_state = cache_replacement.make(self, file, user)
        if self.count % 10 == 0:
            file = cache_replacement.request_file(self)
            user = np.random.choice((10, -10), 1, p=[0.5, 0.5])
            new_state = cache_replacement.make(self, file, user)
        self.cost += cost

        if self.count == 1000:
            done = True
        return new_state, reward, done, file, user

    def make(self, file, user):
        #with tf.device('/gpu:0'):
        list = np.zeros(202)
        list[0] = file
        '''
        for i in range(20):
            for k in range(5):
                list[(i + 1)] += self.state[0][5 * i + k] * (10 ** (3 * k))
        for i in range(20):
            for k in range(5):
                list[(i + 21)] += self.state[1][5 * i + k] * (10 ** (3 * k))
        '''
        for k in range(2):
            for i in range(self.Memory):
                list[k * 100 + (i+1)] = self.state[k][i]
        '''
        for i in range(self.Memory):
            n = self.state[0][i]
            m = self.state[1][i]
            for k in range(self.Num_file):
                if n // self.Num_packet == k:
                    list[k+1] += 1
            for j in range(self.Num_file):
                if m // self.Num_packet == j:
                    list[j+101] += 1
        '''
        list[201] = user
        return list.tolist()