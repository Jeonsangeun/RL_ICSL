import numpy as np
import random as rd

class cache_replacement:

    def __init__(self):
        self.Num_file = 5
        self.Num_packet = 2
        self.Memory = 4
        self.L_file = 5
        self.alpha = 1.0
        self.state = np.zeros([2, 4]) - 1
        self.Memo = np.zeros([22])
        self.cost = 0
        self.count = 0
        self.list_A = []
        self.list_B = []
        self.win_reward = 150
        self.draw_reward = 25
        self.loss_reward = 5

    def Zipf_law(self, k):
        m = 0.0
        for i in range(1, self.L_file + 1):
            m += 1.0 / (i ** self.alpha)
        rate = 1.0 / (k ** self.alpha) / m

        return rate

    def request_file(self):
        pp = np.random.choice((0, 2, 4, 6, 8), 1, p=[cache_replacement.Zipf_law(self, 1),
                                                     cache_replacement.Zipf_law(self, 2),
                                                     cache_replacement.Zipf_law(self, 3),
                                                     cache_replacement.Zipf_law(self, 4),
                                                     cache_replacement.Zipf_law(self, 5)])
        return pp[0]

    def reset(self):
        self.state = np.zeros([2, 4]) - 1
        self.Memo = np.zeros([22])
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
        user = np.random.choice((1, -1), 1,  p=[0.5, 0.5])
        self.Memo[20] = file_1
        self.Memo[21] = user
        return self.Memo, file_1, user

    def random_action(self):
        return rd.randrange(10)

    def step(self, action, file, user):
        done = False
        cost = 0
        reward = 0
        new_state = []
        self.tmp_1 = (self.state[0])[:]
        self.tmp_2 = (self.state[1])[:]
        if user == 1:
            action = cache_replacement.compile(self, action, user)
            if action != []:
                n = self.tmp_1[action[0]]
                self.tmp_1 = np.delete(self.tmp_1, action[0])
                if n != -1:
                    self.Memo[int(n)] = 0
                if file not in self.tmp_1:
                    if file not in self.tmp_2:
                        self.tmp_1 = np.append(self.tmp_1, file)
                        self.Memo[file] = 1
                        cost += 100
                        reward += self.loss_reward
                    else:
                        self.tmp_1 = np.append(self.tmp_1, file)
                        self.Memo[file] = 1
                        cost += 25
                        reward += self.draw_reward
                else:
                    self.tmp_1 = np.append(self.tmp_1, n)
                    if n != -1:
                        self.Memo[int(n)] = 1
                    cost += 5
                    reward += self.win_reward
            else:
                if -1 in self.state[0]:
                    if file not in self.tmp_1:
                        if file not in self.tmp_2:
                            self.tmp_1 = np.delete(self.tmp_1, 0)
                            self.tmp_1 = np.append(self.tmp_1, file)
                            self.Memo[file] = 1
                            cost += 100
                            reward += self.loss_reward
                        else:
                            self.tmp_1 = np.delete(self.tmp_1, 0)
                            self.tmp_1 = np.append(self.tmp_1, file)
                            self.Memo[file] = 1
                            cost += 25
                            reward += self.draw_reward
                    cost += 5
                    reward += self.win_reward
                else:
                    if file not in self.tmp_1:
                        if file not in self.tmp_2:
                            cost += 100
                            reward += self.loss_reward
                        else:
                            cost += 25
                            reward += self.draw_reward
                    else:
                        cost += 5
                        reward += self.win_reward
            self.count += 1
            self.state[0] = self.tmp_1[:]
        if user == -1:
            action = cache_replacement.compile(self, action, user)
            if action != []:
                n = self.tmp_2[action[0]]
                self.tmp_2 = np.delete(self.tmp_2, action[0])
                if n != -1:
                    self.Memo[int(n + 10)] = 0
                if file not in self.tmp_2:
                    if file not in self.tmp_1:
                        self.tmp_2 = np.append(self.tmp_2, file)
                        self.Memo[file + 10] = 1
                        cost += 100
                        reward += self.loss_reward
                    else:
                        self.tmp_2 = np.append(self.tmp_2, file)
                        self.Memo[file + 10] = 1
                        cost += 25
                        reward += self.draw_reward
                else:
                    self.tmp_2 = np.append(self.tmp_2, n)
                    if n != -1:
                        self.Memo[int(n + 10)] = 1
                    cost += 5
                    reward += self.win_reward
            else:
                if -1 in self.state[1]:
                    if file not in self.tmp_2:
                        if file not in self.tmp_1:
                            self.tmp_2 = np.delete(self.tmp_2, 0)
                            self.tmp_2 = np.append(self.tmp_2, file)
                            self.Memo[file+10] = 1
                            cost += 100
                            reward += self.loss_reward
                        else:
                            self.tmp_2 = np.delete(self.tmp_2, 0)
                            self.tmp_2 = np.append(self.tmp_2, file)
                            self.Memo[file+10] = 1
                            cost += 25
                            reward += self.draw_reward
                    cost += 5
                    reward += self.win_reward
                else:
                    if file not in self.tmp_2:
                        if file not in self.tmp_1:
                            cost += 100
                            reward += self.loss_reward
                        else:
                            cost += 25
                            reward += self.draw_reward
                    else:
                        cost += 5
                        reward += self.win_reward
            self.count += 1
            self.state[1] = self.tmp_2[:]

        if self.count % 2 == 1:
            file += 1
            self.Memo[20] = file
            self.Memo[21] = user
            new_state = self.Memo
        if self.count % 2 == 0:
            file = cache_replacement.request_file(self)
            user = np.random.choice((1, -1), 1,  p=[0.5, 0.5])
            self.Memo[20] = file
            self.Memo[21] = user
            new_state = self.Memo
        self.cost += cost

        if self.count == 200:
            done = True
        return new_state, reward, done, file, user

    def compile(self, action, user):
        self.move = []
        if user == 1:
            tmp = np.where(self.state[0] == action)
            self.move = tmp[0].tolist()
        if user == -1:
            tmp = np.where(self.state[1] == action)
            self.move = tmp[0].tolist()
        return self.move