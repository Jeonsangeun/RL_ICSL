import Black_jack as Black
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
start_time = time.time()

env = Black.Black()

print("Black-Jack Play")
Value = np.zeros([10, 10, 2])
Count = np.zeros([10, 10, 2])
max_episode = 100000
dis = 0

for episode in range(max_episode):
    state = env.reset()
    start = state
    done = False
    Return = 0
    #print(env.deck)
    #print(state)
    #print("<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>")
    while not done:
        '''
        print("딜러의 패", env.Dealer)
        print("딜러 점수:", env.Count_Point(env.Dealer))
        print("플레이어의 패", env.Player)
        print("플레이어 점수:", env.Count_Point(env.Player))
        '''
        if env.Count_Point(env.Player) > 17:
            action = 0
        else:
            action = 1
        '''
        action = int(input("액션 선택 : 0:stop, 1:twist"))
        print("액션:", action)
        '''
        new_state, reward, done = env.step(action)
        '''
        print("-------------------------------------")
        print("딜러의 패", env.Dealer)
        print("딜러 점수:", env.Count_Point(env.Dealer))
        print("플레이어의 패", env.Player)
        print("플레이어 점수:", env.Count_Point(env.Player))
        
        
        Count[state] += 1
        Value[state] = Value[state] + (1 / Count[state]) * (reward + dis * Value[new_state] - Value[state])
        '''
        Return += reward
        if done == True:
            Count[start] += 1
            Value[start] = Value[start] + (1 / Count[start]) * (Return - Value[start])

        state = new_state



        #print("reward:", reward)

    if episode % 1000 == 0:
        print(episode)

x = np.array([9, 0, 1, 2, 3, 4, 5, 6, 7, 8])
y = np.array(range(10))
xx, yy = np.meshgrid(x, y)

z_1 = Value[xx, yy, env.Usable_Ace]

x_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_1 = np.array(range(12, 22, 1))
xx_1, yy_1 = np.meshgrid(x_1, y_1)

print("--- %s seconds ---" % (time.time() - start_time))

fig = plt.figure()
ax = Axes3D(fig)
ax.set_title("Black_jack value-function")
ax.plot_wireframe(xx_1, yy_1, z_1, rstride=1, cstride=1)
plt.show()

