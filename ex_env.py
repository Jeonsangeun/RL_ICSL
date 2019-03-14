import numpy as np
import random as rd

class cache_replacement:

    def __init__(self):
        self.Num_file = 5
        self.Num_packet = 2
        self.Memory = 8
        self.L_file = 5
        self.alpha = 1.0
        self.state = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
        self.cost = 0
        self.count = 0


    def Zipf_law(self, k):
        m = 0.0
        for i in range(1, self.L_file + 1):
            m += 1.0 / (i ** self.alpha)
        rate = 1.0 / (k ** self.alpha) / m

        return rate

    def request_file(self):
        pp = np.random.choice((11, 21, 31, 41, 51), 100, p=[cache_replacement.Zipf_law(self, 1),
                                                             cache_replacement.Zipf_law(self, 2),
                                                             cache_replacement.Zipf_law(self, 3),
                                                             cache_replacement.Zipf_law(self, 4),
                                                             cache_replacement.Zipf_law(self, 5)])
        return pp

    def reset(self):
        self.state = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
        self.cost = 0
        self.count = 0
        return [0, 0, 0, 0]

    def random_action(self):
        return rd.randrange(8)

    def step_A(self, file, file_1, action):
        done = False
        tmp = self.state[0][action]
        self.tmp_state = np.delete(self.state[0], action)
        if file not in self.tmp_state:
            if file not in self.state[1]:
                self.tmp_state = np.append(self.tmp_state, file)
                cost = 105
                reward = -100
            else:
                self.tmp_state = np.append(self.tmp_state, file)
                cost = 25
                reward = -25
        else:
            self.tmp_state = np.append(self.tmp_state, tmp)
            cost = 5
            reward = 5

        self.state[0] = self.tmp_state

        if file % 10 != 0:
            new_state = cache_replacement.make_A(self, file-1)
        else:
            new_state = cache_replacement.make_A(self, file_1)

        self.cost += cost
        self.count += 1
        if self.count == 200:
            done = True
        return new_state, reward, done

    def step_B(self, file, file_1, action):
        done = False
        tmp = self.state[1][action]
        self.tmp_state = np.delete(self.state[1], action)
        if file not in self.tmp_state:
            if file not in self.state[0]:
                self.tmp_state = np.append(self.tmp_state, file)
                cost = 105
                reward = -100
            else:
                self.tmp_state = np.append(self.tmp_state, file)
                cost = 25
                reward = -25
        else:
            self.tmp_state = np.append(self.tmp_state, tmp)
            cost = 5
            reward = 5

        self.state[1] = self.tmp_state

        if file % 10 != 0:
            new_state = cache_replacement.make_B(self, file-1)
        else:
            new_state = cache_replacement.make_B(self, file_1)

        self.cost += cost
        self.count += 1
        if self.count == 200:
            done = True
        return new_state, reward, done

    def make_A(self, file):
        list = np.zeros(4)
        list[0] = file
        tmp_1 = np.where(self.state[0] == file)
        tmp_2 = np.where(self.state[1] == file)
        if file not in self.state[0]:
            list[1] = 0
        else:
            list[1] = tmp_1[0][0]
        if file not in self.state[1]:
            list[2] = 0
        else:
            list[2] = tmp_2[0][0]
        if file not in self.state[0]:
            if file not in self.state[1]:
                distance = 3
            else:
                distance = 2
        else:
            distance = 1
        list[3] = distance
        return list.tolist()

    def make_B(self, file):
        list = np.zeros(4)
        list[0] = file
        tmp_1 = np.where(self.state[0] == file)
        tmp_2 = np.where(self.state[1] == file)
        if file not in self.state[0]:
            list[1] = 0
        else:
            list[1] = tmp_1[0][0]
        if file not in self.state[1]:
            list[2] = 0
        else:
            list[2] = tmp_2[0][0]
        if file not in self.state[1]:
            if file not in self.state[0]:
                distance = 3
            else:
                distance = 2
        else:
            distance = 1
        list[3] = distance
        return list.tolist()

'''
    def flat(self):
        present = np.zeros(5)
        for h in range(1, 6):
            k = -1 * h
            if h in self.state[0]:
                present[h - 1] = 1
            if k in self.state[0]:
                present[h - 1] = -1
            if h and k in self.state[0]:
                present[h - 1] = 5
            if h in self.state[1]:
                present[h-1] = 2
            if k in self.state[1]:
                present[h-1] = -2
            if h and k in self.state[1]:
                present[h - 1] = -5
        return present.tolist()
'''
