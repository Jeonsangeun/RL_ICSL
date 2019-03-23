import tensorflow as tf
import numpy as np
import random as rd
import Cache_env as cache
import Cache_DQN as DQN
from collections import deque
import matplotlib.pyplot as plt

env = cache.cache_replacement()

input_size = 10
output_size = 4
dis = .90
request = 2000
REPLAY_MEMORY = 5000000
x_layer = []
y_layer = []

def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name = "target", src_scope_name="main"):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def main():
    max_episodes = 25000
    replay_buffer = deque()
    cost = 0

    with tf.Session() as sess:
        mainDQN = DQN.DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        for episode in range(max_episodes):
            e = 1. / ((episode / 500) + 1)
            state, file, user = env.reset()
            for i in range(request):
                #action = int(input("액션을 입력하시오"))

                if np.random.rand(1) < e:
                    action = env.random_action()
                else:
                    action = np.argmax(mainDQN.predict(state))
                '''
                if user == 10:
                    tmp_state = env.state[0].copy()
                    aa = np.where(env.state[0] == file)
                    bb = np.max(env.state[0])
                    if 0. in tmp_state:
                        action = 0
                    else:
                        if bb == file:
                            tmp_state[aa[0][0]] = -1000
                            bb = np.max(tmp_state)
                            cc = np.where(tmp_state == bb)
                            action = cc[0][0]
                        else:
                            cc = np.where(tmp_state == bb)
                            action = cc[0][0]
                if user == -10:
                    tmp_state = env.state[1].copy()
                    aa = np.where(env.state[1] == file)
                    bb = np.max(env.state[1])
                    if 0. in tmp_state:
                        action = 0
                    else:
                        if bb == file:
                            tmp_state[aa[0][0]] = -1000
                            bb = np.max(tmp_state)
                            cc = np.where(tmp_state == bb)
                            action = cc[0][0]
                        else:
                            cc = np.where(tmp_state == bb)
                            action = cc[0][0]
                '''
                #print(env.state)
                #print(action)
                #print(file)
                #print(user)
                #print("---------------------")
                next_state, reward, done, file, user = env.step(action, file, user)
                #print(reward)
                #print(env.state)
                #print(next_state)
                #print(env.cost)

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state

            cost += env.cost
            if episode % 100 == 99:
                x_layer.append(episode / 100)
                y_layer.append(cost / 100)
                print("Episode: {} cost: {}".format(episode, (cost / 100)))
                for _ in range(50):
                    minibatch = rd.sample(replay_buffer, 200)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)
                cost = 0
                sess.run(copy_ops)

        W1 = mainDQN.Weight_1.eval()
        W2 = mainDQN.Weight_2.eval()
        W3 = mainDQN.Weight_3.eval()
        B1 = mainDQN.Bias_1.eval()
        B2 = mainDQN.Bias_2.eval()
        B3 = mainDQN.Bias_3.eval()
        np.save('Cache_W_1', W1)
        np.save('Cache_W_2', W2)
        np.save('Cache_W_3', W3)
        np.save('B_1', B1)
        np.save('B_2', B2)
        np.save('B_3', B3)

    np.save("X_", x_layer)
    np.save("Y_", y_layer)

    plt.plot(x_layer, y_layer)
    plt.show()

if __name__ =="__main__":
    main()

