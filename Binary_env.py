import numpy as np
import random as rd

class cache_replacement:

    def __init__(self):
        self.Num_file = 20
        self.Num_packet = 8
        self.Memory = 16
        self.F_packet = 160
        self.alpha = 1.0
        self.state = np.zeros([2, self.Memory]) - 1
        self.Memo = np.zeros([2 * self.F_packet + 2])
        self.cost = 0
        self.count = 0
        self.win_reward = 500
        self.draw_reward = 0
        self.loss_reward = -500
        self.file_20 = tuple(np.load("20_file.npy").tolist())
        self.rate_20 = np.load("20_zip.npy").tolist()

    def request_file(self):
        pp = np.random.choice(self.file_20, 1, p=self.rate_20)
        return pp[0]

    def reset(self):
        self.state = np.zeros([2, self.Memory]) - 1
        self.Memo = np.zeros([2 * self.F_packet + 2])
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
        new = cache_replacement.flat(self, file_1, user)
        return new, file_1, user

    def random_action(self, user):
        action = 0
        '''
        if user == 1:
            if -1 in self.state[0]:
                action = -1
            else:
                action = rd.randrange(-1, self.F_packet + 1)
        if user == -1:
            if -1 in self.state[1]:
                action = -1
            else:
                action = rd.randrange(-1, self.F_packet + 1)
        '''
        if user == 1:
            action = np.random.choice(self.state[0])
        if user == -1:
            action = np.random.choice(self.state[0])
        #action = rd.randrange(-1, self.F_packet + 1)
        return int(action)

    def step(self, action, file, user):
        done = False
        cost = 0
        reward = 0
        new_state = []
        self.tmp_1 = (self.state[0])[:]
        self.tmp_2 = (self.state[1])[:]
        if user == 1:
            action = cache_replacement.compile(self, action, user)
            if action != -1:
                n = self.tmp_1[action]
                self.tmp_1 = np.delete(self.tmp_1, action)
                if file not in self.tmp_1:
                    if file not in self.tmp_2:
                        self.tmp_1 = np.append(self.tmp_1, file)
                        cost += 100
                        reward += self.loss_reward
                    else:
                        self.tmp_1 = np.append(self.tmp_1, file)
                        cost += 25
                        reward += self.draw_reward
                else:
                    self.tmp_1 = np.append(self.tmp_1, n)
                    cost += 5
                    reward += self.win_reward
            else:
                if -1 in self.state[0]:
                    aa = np.where(self.state[0] == -1)
                    bb = aa[0].tolist()
                    if file not in self.tmp_1:
                        self.tmp_1 = np.delete(self.tmp_1, bb[0])
                        if file not in self.tmp_2:
                            self.tmp_1 = np.append(self.tmp_1, file)
                            cost += 100
                            reward += self.loss_reward
                        else:
                            self.tmp_1 = np.append(self.tmp_1, file)
                            cost += 25
                            reward += self.draw_reward
                    else:
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
            if action != -1:
                n = self.tmp_2[action]
                self.tmp_2 = np.delete(self.tmp_2, action)
                if file not in self.tmp_2:
                    if file not in self.tmp_1:
                        self.tmp_2 = np.append(self.tmp_2, file)
                        cost += 100
                        reward += self.loss_reward
                    else:
                        self.tmp_2 = np.append(self.tmp_2, file)
                        cost += 25
                        reward += self.draw_reward
                else:
                    self.tmp_2 = np.append(self.tmp_2, n)
                    cost += 5
                    reward += self.win_reward
            else:
                if -1 in self.state[1]:
                    aa = np.where(self.state[1] == -1)
                    bb = aa[0].tolist()
                    if file not in self.tmp_2:
                        self.tmp_2 = np.delete(self.tmp_2, bb[0])
                        if file not in self.tmp_1:
                            self.tmp_2 = np.append(self.tmp_2, file)
                            cost += 100
                            reward += self.loss_reward
                        else:
                            self.tmp_2 = np.append(self.tmp_2, file)
                            cost += 25
                            reward += self.draw_reward
                    else:
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

        if self.count % 8 != 0:
            file += 1
            new_state = cache_replacement.flat(self, file, user)
        if self.count % 8 == 0:
            file = cache_replacement.request_file(self)
            user = np.random.choice((1, -1), 1,  p=[0.5, 0.5])
            new_state = cache_replacement.flat(self, file, user)
        self.cost += cost

        if self.count == 800:
            done = True
        return new_state, reward, done, file, user

    def compile(self, action, user):
        self.move = []
        self.choice = []
        if user == 1:
            tmp = np.where(self.state[0] == action)
            self.move = tmp[0].tolist()
        if user == -1:
            tmp = np.where(self.state[1] == action)
            self.move = tmp[0].tolist()
        if self.move == []:
            self.move.append(-1)
        self.choice = rd.sample(self.move, 1)
        return self.choice[0]

    def flat(self, file, user):
        self.Memo = np.zeros([2 * self.F_packet + 2])
        for i in range(self.Memory):
            a = self.state[0][i]
            b = self.state[1][i]
            if a == -1:
                pass
            else:
                self.Memo[int(a)] = 10
            if b == -1:
                pass
            else:
                self.Memo[int(b + self.F_packet)] = 10
        self.Memo[2 * self.F_packet] = file
        self.Memo[2 * self.F_packet + 1] = user
        return self.Memo