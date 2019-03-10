import numpy as np
import random as rd
from itertools import permutations

class base:
    def __init__(self):
        self.state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.done = 0
        self.marker = 1
        self.counter = 1
        self.winner = 0
        self.zero_match = []
        self.state_env = []

    def reset(self):
        self.state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.done = 0
        self.marker = 1
        self.counter = 0
        self.winner = 0
        self.zero_match = []
        return 0

    def reset_(self):
        self.state = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.done = 0
        self.winner = 0
        self.marker = 1
        self.counter = 0
        x_, y_ = base.AI(self)
        self.done, self.winner = base.mark(self, x_, y_)
        self.marker = -1
        self.zero_match = []
        n = base.state_space(self)
        return n

    def state_universe(self):
        list0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        per0 = permutations(list0, 9)
        res0 = list(set(per0))
        list1 = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        per1 = permutations(list1, 9)
        res1 = list(set(per1))
        list2 = [0, 0, 0, 0, 0, 0, 0, -1, 1]
        per2 = permutations(list2, 9)
        res2 = list(set(per2))
        list3 = [0, 0, 0, 0, 0, 0, 1, -1, 1]
        per3 = permutations(list3, 9)
        res3 = list(set(per3))
        list4 = [0, 0, 0, 0, 0, -1, 1, -1, 1]
        per4 = permutations(list4, 9)
        res4 = list(set(per4))
        list5 = [0, 0, 0, 0, 1, -1, 1, -1, 1]
        per5 = permutations(list5, 9)
        res5 = list(set(per5))
        list6 = [0, 0, 0, -1, 1, -1, 1, -1, 1]
        per6 = permutations(list6, 9)
        res6 = list(set(per6))
        list7 = [0, 0, 1, -1, 1, -1, 1, -1, 1]
        per7 = permutations(list7, 9)
        res7 = list(set(per7))
        list8 = [0, -1, 1, -1, 1, -1, 1, -1, 1]
        per8 = permutations(list8, 9)
        res8 = list(set(per8))
        list9 = [1, -1, 1, -1, 1, -1, 1, -1, 1]
        per9 = permutations(list9, 9)
        res9 = list(set(per9))
        self.state_env = res0 + res1 + res2 + res3 + res4 + res4 + res5 + res6 + res7 + res8 + res9
        return len(self.state_env)

    def state_space(self):
        space = np.zeros([9])
        aa = np.where(self.state == 1)
        first_player = 3*aa[0][:] + aa[1][:]
        space[first_player] = 1
        bb = np.where(self.state == -1)
        second_player = 3 * bb[0][:] + bb[1][:]
        space[second_player] = -1
        space_tuple = tuple(space.tolist())
        next_state = self.state_env.index(space_tuple)
        return next_state

    def zero_index(self):
        self.zero_match = np.where(self.state == 0)
        aa = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        bb = 3*self.zero_match[0][:] + self.zero_match[1][:]
        cc = []
        for i in aa:
            if i not in bb:
                cc.append(i)
        return cc

    def action_space(self):
        can = np.where(self.state == 0)
        temp = 3 * can[0][:] + can[1][:]
        return rd.choice(temp)

    def step_1(self, action):
        reward = 0.0
        x = action // 3
        y = action % 3
        done, winner = base.mark(self, x, y)
        if done == 1:
            new_state = base.state_space(self)
            if winner == 1:
                reward += 100.0
                return new_state, reward, True
            if winner == 2:
                reward += 30.0
                return new_state, reward, True
        else:
            x_, y_ = base.AI(self)
            done, winner = base.mark(self, x_, y_)
            new_state = base.state_space(self)
            if done == 1:
                if winner == -1:
                    reward -= 50.0
                    return new_state, reward, True
            if done == 0:
                reward += 5.0
                return new_state, reward, False

    def step_(self, action):
        reward = 0.0
        x = action // 3
        y = action % 3
        done, winner = base.mark(self, x, y)
        if done == 1:
            new_state = base.state_space(self)
            if winner == -1:
                reward += 100.0
                return new_state, reward, True
        if done == 0:
            x_, y_ = base.AI(self)
            done, winner = base.mark(self, x_, y_)
            new_state = base.state_space(self)
            if done == 1:
                if winner == 1:
                    reward -= 50.0
                    return new_state, reward, True
                if winner == 2:
                    reward += 30.0
                    return new_state, reward, True
            if done == 0:
                reward += 5.0
                return new_state, reward, False

    def mark(self, x, y):
        if self.state[x, y] == 0:
            self.state[x, y] = self.marker
            self.counter += 1

            tmp1 = 0
            tmp2 = 0
            for i in range(3):
                if abs(sum(self.state[i, :])) == 3 or abs(sum(self.state[:, i])) == 3:
                    self.done = 1
                    self.winner = self.marker
                tmp1 += self.state[i, i]
                tmp2 += self.state[i, (2 - i)]
            if abs(tmp1) == 3 or abs(tmp2) == 3:
                self.done = 1
                self.winner = self.marker

            if self.done == 0:
                self.marker = self.marker * -1

            if self.counter == 9:
                self.done = 1
                self.winner = 2

        return self.done, self.winner

    def print_state(self):
        for x in range(3):
            for y in range(3):
                print("%4d" % self.state[x, y], end="")
            print()

    def random_auto(self):
        blank = np.where(self.state == 0)
        number = rd.randrange(len(blank[0]))
        return blank[0][number], blank[1][number]

    def AI(self):
        aa = []
        bb = []
        case1 = 0
        case2 = 0
        cc = [0, 2]
        x = rd.choice(cc)
        y = rd.choice(cc)
        if (self.state[1][1] == 0):
            return 1, 1
        if (self.state[1][1] != 0):
            for i in range(0, 3):
                if abs(sum(self.state[i, :])) == 2:
                    for k in range(0, 3):
                        if self.state[i][k] == 0:
                            return i, k
                if abs(sum(self.state[:, i])) == 2:
                    for k in range(0, 3):
                        if self.state[k][i] == 0:
                            return k, i
                case1 += self.state[i][i]
                case2 += self.state[i][2 - i]
            if abs(case1) == 2:
                for i in range(0, 3):
                    if self.state[i][i] == 0:
                        return i, i
            if abs(case2) == 2:
                for i in range(0, 3):
                    if self.state[i][2 - i] == 0:
                        return i, 2 - i

            if (self.state[x][y] != 0):
                for i in range(3):
                    for k in range(3):
                        if (self.state[i][k] == 0):
                            aa.append(i)
                            bb.append(k)
                n = rd.randrange(len(aa))
                x = aa[n]
                y = bb[n]
                return x, y
            return x, y