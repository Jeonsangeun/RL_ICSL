import numpy as np
import copy
import random as rd
alpha = 1.0
L_file = 5
def Zipf_law(k):
    m = 0.0
    for i in range(1, L_file + 1):
        m += 1.0 / (i ** alpha)
    rate = 1.0 / (k ** alpha) / m

    return rate
def Choice(k):
    pp = np.random.choice((1.0, 2.0, 3.0, 4.0, 5.0), k, p=[Zipf_law(1), Zipf_law(2), Zipf_law(3), Zipf_law(4), Zipf_law(5)])

    return pp[0]
def state_def():
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
state = state_def()

def next_state(file, user):
    list = np.zeros(6)
    list[0] = file
    t1 = np.where(state[0] == file)
    t2 = np.where(state[0] == (file+0.1))
    t3 = np.where(state[1] == file)
    t4 = np.where(state[1] == (file+0.1))
    if t1[0].tolist() == []:
        pass
    else:
        list[1] = t1[0][0] + 1
    if t2[0].tolist() == []:
        pass
    else:
        list[2] = t2[0][0] + 1
    if t3[0].tolist() == []:
        pass
    else:
        list[3] = t3[0][0] + 1
    if t4[0].tolist() == []:
        pass
    else:
        list[4] = t4[0][0] + 1
    list[5] = user
    return list.tolist()

def step(action, file, user):
    done = False
    tmp_0 = copy.deepcopy(state[0])
    tmp_1 = copy.deepcopy(state[1])
    num = file
    cost = 0
    reward = 0
    if user == 0:
        for i in range(2):
            n = tmp_0[action]
            tmp_0 = np.delete(tmp_0, action)
            if file not in tmp_0:
                if file not in tmp_1:
                    tmp_0 = np.append(tmp_0, file)
                    cost += 100
                    reward -= 100
                else:
                    tmp_0 = np.append(tmp_0, file)
                    cost += 25
                    reward -= 25
            else:
                tmp_0 = np.append(tmp_0, n)
                cost += 5
                reward -= 5
            file += 0.1
        state[0] = tmp_0[:]
    if user == 1:
        for i in range(2):
            n = tmp_1[action]
            tmp_1 = np.delete(tmp_1, action)
            if file not in tmp_1:
                if file not in tmp_0:
                    tmp_1 = np.append(tmp_1, file)
                    cost += 100
                    reward -= 100
                else:
                    tmp_1 = np.append(tmp_1, file)
                    cost += 25
                    reward -= 25
            else:
                tmp_1 = np.append(tmp_1, n)
                cost += 5
                reward -= 5
            file += 0.1
        state[1] = tmp_1[:]
    next = next_state(num, user)
    return next, reward, cost
for i in range(10):
    print(state)
    file = Choice(1)
    user = rd.randint(0, 1)
    print(file)
    print(user)
    action = int(input("액션을 입력하시오:"))
    next, reward, cost = step(action, file, user)
    print(state)
    print(next)
    print(reward)
    print(cost)
    print("----------------------")




