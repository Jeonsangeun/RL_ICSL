import numpy as np
import random as rd
import copy

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
        pp = np.random.choice((1, 2, 3, 4, 5), 1, p=[cache_replacement.Zipf_law(self, 1),
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

    def step(self, action, file, user):
        done = False
        self.tmp_0 = copy.deepcopy(self.state[0])
        self.tmp_1 = copy.deepcopy(self.state[1])
        n = 0
        if user == 0:
            for i in range(2):
                if file not in self.state[0]:
                    self.tmp_0 = np.delete(self.tmp_0, action)
                    if file not in self.tmp_1:
                        self.tmp_0 = np.append(self.tmp_0, file)
                        cost += 100
                        reward -= 100
                    else:
                        if i == 1:
                            self.tmp_0 = np.append(self.tmp_0, file)
                            cost += 25
                            reward -=25
                        else:
                            self.tmp_0 = np.append(self.tmp_0, file)
                            self.tmp_1[cache_replacement.find(self, 1, file)] = 0
                            cost += 25
                            reward -=25
                else:
                    if i == 1:
                        cost += 5
                        reward -= 5
                    else:
                        n = cache_replacement.find(self, 0, file)
                        cost += 5
                        reward -= 5
                        self.tmp_0[n] = 0
            self.tmp_0[n] = file
            self.state[0] = self.tmp_0
        if user == 1:
            for i in range(2):
                if file not in self.tmp_0:
                    
                    if file not in self.tmp_1:
                        self.tmp_0[action] = file
                        cost += 100
                        reward -= 100
                    else:
                        if i == 1:
                            self.tmp_0[action] = file
                            cost += 25
                            reward -=25
                        else:
                            self.tmp_0[action] = file
                            self.tmp_1[cache_replacement.find(self, 1, file)] = 0
                            cost += 25
                            reward -=25
                else:
                    if i == 1:
                        cost += 5
                        reward -= 5
                    else:
                        n = cache_replacement.find(self, 0, file)
                        self.tmp_0[n] = 0
            self.tmp_0[n] = file
            self.state = self.tmp_0
                        
                 
        
        
        
        
        
        
        
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

    def find(self, num, k):
        aa = []
        for i in range(self.Memory):
            if self.state[num][i] == k:
                aa.append(i)
        return aa[0]
    
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
    
    def random_batch(self):
        state = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
        for i in range(2):
            for k in range(4):
                file = Choice(1)
                done = False
                while not done:
                    if file in state[i]:
                        file = Choice(1)
                        done = False
                    if file not in state[i]:
                        state[i][2*k] = file
                        state[i][2*k+1] = file + 0.1
                        done = True
        return state

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
