import tensorflow as tf
import numpy as np
import random as rd
import Cache_env as cache
import Cache_DQN as DQN
from collections import deque
import matplotlib.pyplot as plt
import time
start_time = time.time()

env = cache.cache_replacement()

input_size = env.Memory * 4 + 5
output_size = 4 * env.Memory
dis = .9999999
request = 200
REPLAY_MEMORY = 1000000
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


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def main():
    max_episodes = 20000
    replay_buffer = deque()
    cost = 0
    env.Zip_funtion()

    with tf.Session() as sess:
        mainDQN = DQN.DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        for episode in range(max_episodes):
            e = 1. / ((episode // 1000) + 1)
            state = env.reset()
            file = env.file_request[0]
            user = env.user_location
            for i in range(request * env.Num_packet):
                print("-------------------")
                print("NOW:", env.BS)
                print("file:", file)
                print("user:", user)
                print("state:", state)
                '''
                action = 0
                aa = np.ones([4]) * 1000
                if file in env.BS[0]:
                    aa[0] = env.Distance(env.BS_0, user)
                if file in env.BS[1]:
                    aa[1] = env.Distance(env.BS_1, user)
                if file in env.BS[2]:
                    aa[2] = env.Distance(env.BS_2, user)
                if file in env.BS[3]:
                    aa[3] = env.Distance(env.BS_3, user)
                bb = np.argmin(aa)
                if bb == 0:
                    action = np.argmax(env.BS[0])
                if bb == 1:
                    action = 4 + np.argmax(env.BS[1])
                if bb == 2:
                    action = 8 + np.argmax(env.BS[2])
                if bb == 3:
                    action = 12 + np.argmax(env.BS[3])
                
                if np.random.rand(1) < e:
                    action = env.random_action()
                else:
                    action = np.argmax(mainDQN.predict(state)[0])
                '''
                action = int(input("액션을 입력하시오:"))
                print("action:", action)

                next_state, reward, done, file, user = env.step(action, file, user)
                print("reward:", reward)

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state

            cost += env.cost

            if episode % 50 == 49:
                x_layer.append(episode / 50)
                y_layer.append(cost / 50)
                print("Episode: {} cost: {}".format(episode, (cost / 50)))
                for _ in range(20):
                    minibatch = rd.sample(replay_buffer, 2000)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)
                cost = 0
                sess.run(copy_ops)
            '''
            if episode % 5 == 4:
                x_layer.append(episode / 5)
                y_layer.append(cost / 5)
                print("Episode: {} cost: {}".format(episode, (cost / 5)))
                cost = 0
            '''
        mainDQN.save()

    print("start_time", start_time)
    print("--- %s seconds ---" % (time.time() - start_time))

    np.save("X_(ex)", x_layer)
    np.save("Y_(ex)", y_layer)
    plt.plot(x_layer, y_layer)
    plt.show()


if __name__ == "__main__":
    main()