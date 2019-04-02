import numpy as np
import random as rd

class cache_replacement:

    def __init__(self):
        self.Num_file = 5
        self.Num_packet = 2
        self.Memory = 4
        self.L_file = 5
        self.alpha = 1.0
        self.state = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1]], dtype=float)
        self.cost = 0
        self.count = 0


    def Zipf_law(self, k):
        m = 0.0
        for i in range(1, self.L_file + 1):
            m += 1.0 / (i ** self.alpha)
        rate = 1.0 / (k ** self.alpha) / m

        return rate

    def request_file(self):
        pp = np.random.choice((0.0, 2.0, 4.0, 6.0, 8.0), 1, p=[cache_replacement.Zipf_law(self, 1),
                                                               cache_replacement.Zipf_law(self, 2),
                                                               cache_replacement.Zipf_law(self, 3),
                                                               cache_replacement.Zipf_law(self, 4),
                                                               cache_replacement.Zipf_law(self, 5)])
        return pp[0]

    def reset(self):
        self.state = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1]], dtype=float)
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
        return rd.randrange(4)

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
                    reward -= 50
                else:
                    self.tmp_1 = np.append(self.tmp_1, file)
                    cost += 25
                    reward += 0
            else:
                self.tmp_1 = np.append(self.tmp_1, n)
                cost += 5
                reward += 50
            self.count += 1
            self.state[0] = self.tmp_1[:]
        if user == -10:
            n = self.tmp_2[action]
            self.tmp_2 = np.delete(self.tmp_2, action)
            if file not in self.tmp_2:
                if file not in self.tmp_1:
                    self.tmp_2 = np.append(self.tmp_2, file)
                    cost += 100
                    reward -= 50
                else:
                    self.tmp_2 = np.append(self.tmp_2, file)
                    cost += 25
                    reward += 0
            else:
                self.tmp_2 = np.append(self.tmp_2, n)
                cost += 5
                reward += 50
            self.count += 1
            self.state[1] = self.tmp_2[:]

        if self.count % 2 == 1:
            file += 1
            new_state = cache_replacement.make(self, file, user)
        if self.count % 2 == 0:
            file = cache_replacement.request_file(self)
            user = np.random.choice((10, -10), 1,  p=[0.5, 0.5])
            new_state = cache_replacement.make(self, file, user)
        self.cost += cost

        if self.count == 2000:
            done = True
        return new_state, reward, done, file, user

    def make(self, file, user):

        list = np.zeros(10)
        list[0] = file
        for k in range(2):
            for i in range(self.Memory):
                list[k * 5 + (i + 1)] = self.state[k][i]
        list[9] = user

        '''
        list = np.zeros(12)
        for i in range(5):
            list[i] += (self.state[(i//5)] // 2 == i).sum()
        for k in range(5, 10):
            list[k] += (self.state[(k//5)] // 2 == (k-5)).sum()
        list[10] = file
        list[11] = user
        '''
        return list.tolist()
