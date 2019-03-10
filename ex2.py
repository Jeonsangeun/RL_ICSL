import numpy as np
import random as rd

state = np.array([[0,0,0,0], [0,0,0,0]])

def step(file, action):
    done = False
    tmp_1 = np.where(state[0] == file)
    tmp_2 = np.where(state[1] == file)
    if tmp_1[0].tolist() == []:
        if tmp_2[0].tolist() == []:
            state[0][action] = file
            cost = 105
        else:
            state[0][action] = file
            cost = 25
    else:
        cost = 5
    if file < 0:
        done = True
    return state, cost, done
def step_A(file, action):
    done = False
    tmp_1 = np.where(state[0] == file)
    tmp_2 = np.where(state[1] == file)
    if tmp_2[0].tolist() == []:
        if tmp_1[0].tolist() == []:
            state[1][action] = file
            cost = 105
        else:
            state[1][action] = file
            cost = 25
    else:
        cost = 5
    if file < 0:
        done = True
    return state, cost, done

done = False

def Zip_law(x):
    m = 0.0
    for i in range(1, 6):
        m += 1.0 / i ** 0.8
    return 1.0 / (x ** 0.8) / m

file = np.random.choice((1, 2, 3, 4, 5), 1000, p=[Zip_law(1), Zip_law(2), Zip_law(3), Zip_law(4), Zip_law(5)])

total_cost = 0
for i in range(1000):
    user = rd.randrange(2)
    if user == 0:
        for k in range(2):
            action = rd.randrange(4)
            new_state, cost, done = step(file[i], action)
            file[i] *= -1
            total_cost += cost
            print(cost)
        print(state)
    if user == 1:
        for k in range(2):
            action = rd.randrange(4)
            new_state, cost, done = step_A(file[i], action)
            file[i] *= -1
            total_cost += cost
            print(cost)
        print(state)
    print(total_cost)




