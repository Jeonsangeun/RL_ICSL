import numpy as np
import random as rd

class cache_replacement:

    def __init__(self):
        self.Num_file = 5
        self.Num_packet = 2
        self.Memory = 4
        self.F_packet = self.Num_file * self.Num_packet
        self.alpha = 1.0
        self.Memo = np.zeros([2, self.F_packet])
        self.state_A = []
        self.state_B = []
        self.cost = 0
        self.count = 0
        self.win_reward = 50
        self.draw_reward = 0
        self.loss_reward = -50
        self.Zip_law = []
        self.Setting = tuple(range(0, self.Num_packet * self.Num_file, self.Num_packet))

    def Zip_funtion(self):
        m = np.sum(np.array(range(1, self.Num_file+1))**(-self.alpha))
        self.Zip_law = (np.array(range(1, self.Num_file+1))**(-self.alpha)) / m

    def request_file(self):
        pp = np.random.choice(self.Setting, 1, p=self.Zip_law)
        return pp[0]


    def reset(self):
        self.Memo = np.zeros([2, self.F_packet])
        self.state_A = []
        self.state_B = []
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

    def random_action(self):
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

        if user == 1:
            action = np.random.choice(self.state[0])
        if user == -1:
            action = np.random.choice(self.state[0])
        '''
        #action = rd.randrange(-1, self.F_packet + 1)
        return rd.randrange(0, self.F_packet)

    def step(self, action, file, user):
        done = False
        cost = 0
        reward = 0
        new_state = []
        cache_replacement.state_def(self)
        if user == 1:
            if self.Memo[0][action] == 1:
                self.Memo[0][action] = 0
                if file not in self.state_A:
                    if file not in self.state_B:
                        self.Memo[0][file] = 1
                        cost += 100
                        reward += self.loss_reward
                    else:
                        self.Memo[0][file] = 1
                        cost += 25
                        reward += self.draw_reward
                else:
                    cost += 5
                    reward += self.win_reward
                self.count += 1
            else:
                if (self.Memo[0] == 1).sum() < self.Memory:
                    if file not in self.state_A:
                        if file not in self.state_B:
                            self.Memo[0][file] = 1
                            cost += 100
                            reward += self.loss_reward
                        else:
                            self.Memo[0][file] = 1
                            cost += 25
                            reward += self.draw_reward
                    else:
                        cost += 5
                        reward += self.win_reward
                    self.count += 1
                else:
                    if file not in self.state_A:
                        if file not in self.state_B:
                            cost += 100
                            reward += self.loss_reward
                        else:
                            cost += 25
                            reward += self.draw_reward
                    else:
                        cost += 5
                        reward += self.win_reward
                    self.count += 1
        if user == -1:
            if self.Memo[1][action] == 1:
                self.Memo[1][action] = 0
                if file not in self.state_B:
                    if file not in self.state_A:
                        self.Memo[1][file] = 1
                        cost += 100
                        reward += self.loss_reward
                    else:
                        self.Memo[1][file] = 1
                        cost += 25
                        reward += self.draw_reward
                else:
                    cost += 5
                    reward += self.win_reward
                self.count += 1
            else:
                if (self.Memo[1] == 1).sum() < self.Memory:
                    if file not in self.state_B:
                        if file not in self.state_A:
                            self.Memo[1][file] = 1
                            cost += 100
                            reward += self.loss_reward
                        else:
                            self.Memo[1][file] = 1
                            cost += 25
                            reward += self.draw_reward
                    else:
                        cost += 5
                        reward += self.win_reward
                    self.count += 1
                else:
                    if file not in self.state_B:
                        if file not in self.state_A:
                            cost += 100
                            reward += self.loss_reward
                        else:
                            cost += 25
                            reward += self.draw_reward
                    else:
                        cost += 5
                        reward += self.win_reward
                    self.count += 1

        if self.count % self.Num_packet != 0:
            file += 1
            new_state = cache_replacement.flat(self, file, user)
        if self.count % self.Num_packet == 0:
            file = cache_replacement.request_file(self)
            user = np.random.choice((1, -1), 1,  p=[0.5, 0.5])
            new_state = cache_replacement.flat(self, file, user)
        self.cost += cost

        if self.count == (100 * self.Num_packet):
            done = True
        return new_state, reward, done, file, user

    def flat(self, file, user):
        temp = np.reshape(self.Memo, [1, 2 * self.F_packet])
        temp = np.append(temp, file)
        temp = np.append(temp, user)
        return temp
    '''
    if i in self.state[0] and i not in self.state[1]:
        self.Memo[i] = 1
    if i not in self.state[0] and i in self.state[1]:
        self.Memo[i] = 2
    if i in self.state[0] and i in self.state[1]:
        self.Memo[i] = 3
    '''
    def state_def(self):
        self.state_A = np.where(self.Memo[0] == 1)[0].tolist()
        self.state_B = np.where(self.Memo[1] == 1)[0].tolist()
