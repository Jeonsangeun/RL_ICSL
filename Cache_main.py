import tensorflow as tf
import numpy as np
import random as rd
import ex_env as cache
import ex_DQN as DQN
from collections import deque
import matplotlib.pyplot as plt

env = cache.cache_replacement()

input_size = env.F_packet * 2 + 2
output_size = env.F_packet
dis = .999
request = (100 * env.Num_packet)
REPLAY_MEMORY = 500000
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
    max_episodes = 10000
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
            e = 1. / ((episode // 500) + 1)
            state, file, user = env.reset()
            #print(np.argmax(mainDQN.predict(state)))
            for i in range(request):

                if np.random.rand(1) < e:
                    action = env.random_action()
                else:
                    action = np.argmax(mainDQN.predict(state)[0])
                    #if episode > 1000:
                    #    if env.count % 50 == 0:
                    #        print(state)
                    #        print(action)
                    #        print(mainDQN.predict(state)[0])
                '''
                action = 0
                if user == 1:
                    tmp = env.state[0].copy()
                    if -1 in tmp:
                        action = env.Memory
                    else:
                        aa = np.max(tmp)
                        bb = np.argmax(tmp)
                        if aa == file:
                            tmp[np.argmax(tmp)] = -1
                            action = np.argmax(tmp)
                        else:
                            action = bb
                if user == -1:
                    tmp = env.state[1].copy()
                    if -1 in tmp:
                        action = env.Memory
                    else:
                        aa = np.max(tmp)
                        bb = np.argmax(tmp)
                        if aa == file:
                            tmp[np.argmax(tmp)] = -1
                            action = np.argmax(tmp)
                        else:
                            action = bb
                
                #비인기 패킷 제거
                env.state_def()
                print(env.state_A)
                print(env.state_B)
                print("packet", file)
                print(state)
                print("user: ", user)
                #action = int(input("액션을 입력하시오: "))
                print("action", action)
                #print((state == 1).sum())
                print("---------------------")
                '''
                next_state, reward, done, file, user = env.step(action, file, user)
                '''
                print(reward)
                #print(env.state)
                env.state_def()
                print(env.state_A)
                print(env.state_B)
                print(next_state)
                print(env.cost)
                print("-----++++++++++++----------------")
                '''
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state

            #print("--------")

            cost += env.cost
            '''
            if episode % 100 == 99:
                x_layer.append(episode / 100)
                y_layer.append(cost / 100)
                print("Episode: {} cost: {}".format(episode, (cost / 100)))
                cost = 0
            '''
            if episode % 100 == 99:
                x_layer.append(episode / 100)
                y_layer.append(cost / 100)
                print("Episode: {} cost: {}".format(episode, (cost / 100)))
                for _ in range(50):
                    minibatch = rd.sample(replay_buffer, 100)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)
                cost = 0
                sess.run(copy_ops)
        '''
        state, file, user = env.reset()
        for i in range(request):
            action = np.argmax(mainDQN.predict(state))
            print("action", action)
            print(state)
            print("user: ", user)
            print("packet", file)
            print("---------------------")
            next_state, reward, done, file, user = env.step(action, file, user)
            print(reward)
            print(env.cost)
            state = next_state
        '''
        W1 = mainDQN.Weight_1.eval()
        W2 = mainDQN.Weight_2.eval()
        W3 = mainDQN.Weight_3.eval()
        W4 = mainDQN.Weight_4.eval()
        W5 = mainDQN.Weight_5.eval()
        np.save('Cache_W_1@', W1)
        np.save('Cache_W_2@', W2)
        np.save('Cache_W_3@', W3)
        np.save('Cache_W_4@', W4)
        np.save('Cache_W_5@', W5)

    np.save("X_", x_layer)
    np.save("Y_", y_layer)
    plt.plot(x_layer, y_layer)
    plt.show()

if __name__ =="__main__":
    main()
