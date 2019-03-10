import tmp as main
import numpy as np
import random as rd
import matplotlib.pyplot as plt

env = main.base()

Q = np.zeros([env.state_universe(), 9])

learning_rate = 0.85
dis = 0.91
num_episodes = 4000
rList = []
count = 0
player = int(input("학습할 플레이어를 선택하시오: (선공: 1, 후공: -1)"))


if (player == 1):
    for i in range(num_episodes):
        #print("-------------")
        state = env.reset()
        done = False
        rate = 0

        e = 1. / ((i // 200) + 1)

        while not done:
            if np.random.rand(1) < e:
                action = env.action_space()
            else:
                temp_Q = Q
                temp_Q[state, env.zero_index()] = -1000
                action = np.argmax(temp_Q[state, :])

            new_state, reward, done = env.step_1(action)

            Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * \
                               (reward + dis * np.max(Q[new_state, :]))

            state = new_state
            if env.winner == -1:
                rate += 1

        rList.append(rate)
        count += 1
        #print("episode %3d 승자는 : %5d" % (count, env.winner))
    print("이기는 통계치: " + str(sum(rList)/num_episodes))
    print("Final Q-table Values")
    print(Q)
    np.save('Q(선공)', Q)
    plt.bar(range(len(rList)), rList, color="blue")
    plt.show()

if (player == -1):
    for i in range(num_episodes):
        #print("-------------")
        state = env.reset_()
        done = False
        rate = 0

        e = 1. / ((i // 200) + 1)

        while not done:
            if np.random.rand(1) < e:
                action = env.action_space()
            else:
                temp_Q = Q
                temp_Q[state, env.zero_index()] = -1000
                action = np.argmax(temp_Q[state, :])

            new_state, reward, done = env.step_(action)

            Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * \
                               (reward + dis * np.max(Q[new_state, :]))

            state = new_state
            if env.winner == -1:
                rate += 1

        rList.append(rate)
        count += 1
        #print("episode %3d 승자는 : %5d" % (count, env.winner))

    print("이기는 통계치: " + str(sum(rList)/num_episodes))
    print("Final Q-table Values")
    print(Q)
    np.save('Q(후공)', Q)
    plt.bar(range(len(rList)), rList, color="blue")
    plt.show()






