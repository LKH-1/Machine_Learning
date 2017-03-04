# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
from gym import wrappers
import numpy as np
import random as ran

env = gym.make('Breakout-v3')

# 꺼내서 사용할 리플레이 갯수
REPLAY = 20
# 리플레이를 저장할 리스트
REPLAY_MEMORY = []
# 미니배치
MINIBATCH = 32

INPUT = env.observation_space.shape
OUTPUT = env.action_space.n

# 하이퍼파라미터
LEARNING_LATE = 0.001
DISCOUNT = 0.99
model_path = "save/model.ckpt"


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w1, w2, w1_o, w2_o):
    X = tf.image.rgb_to_grayscale(X)
    X = tf.image.resize_images(X, [110,84])
    l1 = tf.nn.relu(tf.nn.conv2d(X, w1,  # l1a shape=(?, 28, 28, 32)
                                  strides=[1, 4, 4, 1], padding='SAME'))

    l2 = tf.nn.relu(tf.nn.conv2d(l1, w2, strides = [1,2,2,1], padding='SAME'))

    print(l2)
    l3= tf.reshape(l2, [-1, w1_o.get_shape().as_list()[0]])


    l4= tf.nn.relu(tf.matmul(l3, w1_o))
    pyx = tf.matmul(l4,w2_o)

    return pyx


X = tf.placeholder("float", [None, 210, 160, 3])
Y = tf.placeholder("float", [None, OUTPUT])

w1 = init_weights([8, 8, 1, 16])  # 3x3x1 conv, 32 outputs
w2 = init_weights([4, 4, 16, 32])  # FC 128 * 4 * 4 inputs, 625 outputs

w1_o = init_weights([14*11*32, 256])  # FC 625 inputs, 10 outputs (labels)
w2_o = init_weights([256, OUTPUT])  # FC 625 inputs, 10 outputs (labels)

py_x = model(X, w1, w2, w1_o, w2_o)

w1_r = init_weights([8, 8, 1, 16])  # 3x3x1 conv, 32 outputs
w2_r = init_weights([4, 4, 16, 32])  # 3x3x32 conv, 64 outputs

w1_o_r = init_weights([14 * 11 * 32, 256])  # FC 625 inputs, 10 outputs (labels)
w2_o_r = init_weights([256, OUTPUT])  # FC 625 inputs, 10 outputs (labels)

py_x_r = model(X, w1_r, w2_r, w1_o_r, w2_o_r)

# 총 Reward를 저장해놓을 리스트
rlist = [0]
recent_rlist = [0]

episode = 0

# Loss function 정의
cost = tf.reduce_sum(tf.square(Y - py_x))
optimizer = tf.train.AdamOptimizer(LEARNING_LATE)
train = optimizer.minimize(cost)

saver = tf.train.Saver()

# 세션 정의
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())
    # Target 네트워크에 main 네트워크 값을 카피해줌
    sess.run(w1_r.assign(w1))
    sess.run(w2_r.assign(w2))

    sess.run(w1_o_r.assign(w1_o))
    sess.run(w2_o_r.assign(w2_o))

    # 에피소드 시작
    while np.mean(recent_rlist) < 30:
        episode += 1

        # state 초기화
        s = env.reset()
        if len(recent_rlist) > 100:
            del recent_rlist[0]
        # e-greedy
        e = 1. / ((episode / 50) + 1)
        if e < 0.1 :
            e = 0.1
        rall = 0
        d = False
        count = 0

        # 에피소드가 끝나기 전까지 반복
        while not d:

            count += 1
            # state 값의 전처리
            s_t = np.reshape(s, [-1, 210, 160, 3])
            # 현재 상태의 Q값을 에측
            Q = sess.run(py_x, feed_dict={X: s_t})

            # e-greedy 정책으로 랜덤하게 action 결정
            if e > np.random.rand(1):
                a = env.action_space.sample()
            else:
                a = np.argmax(Q)

            # 결정된 action으로 Environment에 입력
            s1, r, d, _ = env.step(a)

            # Environment에서 반환한 Next_state, action, reward, done 값들을
            # Replay_memory에 저장
            REPLAY_MEMORY.append([s_t, a, r, s1, d])

            # 저장된 값들이 50000개 이상 넘어가면 맨 앞 Replay부터 삭제
            if len(REPLAY_MEMORY) > 50000:
                del REPLAY_MEMORY[0]

            # 총 reward 합
            rall += r
            # state를 Next_state로 바꿈
            s = s1

        # 10번의 episode마다 학습
        if episode % 10 == 0 and episode >= 10:

            # 50번의 미니배치로 학습
            for i in range(MINIBATCH):

                # 저장된 리플레이 중에 학습에 사용할 랜덤한 리플레이 샘플들을 가져옴
                for sample in ran.sample(REPLAY_MEMORY, REPLAY):

                    s_t_r, a_r, r_r, s1_r, d_r = sample

                    # 꺼내온 리플레이의 state의 Q값을 예측

                    Q = sess.run(py_x, feed_dict={X: s_t_r})

                    if d_r:
                        # 꺼내온 리플레이의 상태가 끝난 상황이라면 Negative Reward를 부여
                        Q[0, a_r] = -1
                    else:
                        # 끝나지 않았다면 Q값을 업데이트
                        s1_r_t = np.reshape(s1_r, [-1, 210, 160, 3])
                        Q1 = sess.run(py_x_r, feed_dict={X: s1_r_t})
                        Q[0, a_r] = r_r + DISCOUNT * np.max(Q1)

                    # 업데이트 된 Q값으로 main네트워크를 학습
                    _, loss = sess.run([train, cost], feed_dict={X: s_t_r, Y: Q})

            # 10번 마다 target 네트워크에 main 네트워크 값을 copy
            sess.run(w1_r.assign(w1))
            sess.run(w2_r.assign(w2))

            sess.run(w1_o_r.assign(w1_o))
            sess.run(w2_o_r.assign(w2_o))

        # 총 reward의 합을 list에 저장
        recent_rlist.append(rall)
        rlist.append(rall)
        print("Episode:{} steps:{} reward:{} average reward:{} recent reward:{}".format(episode, count, rall,
                                                                                        np.mean(rlist),
                                                                                        np.mean(recent_rlist)))

    save_path = saver.save(sess, model_path)
    print("Model saved in file: ", save_path)

    rlist = []
    recent_rlist = []

'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)

    print("Model restored form file: ", save_path)
    for episode in range(500):
        # state 초기화
        s = env.reset()

        rall = 0
        d = False
        count = 0
        # 에피소드가 끝나기 전까지 반복
        while not d :
            env.render()
            count += 1
            # state 값의 전처리
            s_t = np.reshape(s, [1, INPUT])

            # 현재 상태의 Q값을 에측
            Q = sess.run(Q_pre, feed_dict={x: s_t,dropout: 1})
            a = np.argmax(Q)

            # 결정된 action으로 Environment에 입력
            s, r, d, _ = env.step(a)

            # 총 reward 합
            rall += r


        rlist.append(rall)

        print("Episode : {} steps : {} r={}. averge reward : {}".format(episode, count, rall,
                                                                        np.mean(rlist)))

'''