# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
from gym import wrappers
import numpy as np
import random as ran

from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize

import matplotlib.pyplot as plt

plt.ion()
env = gym.make('BreakoutNoFrameskip-v3')

# 꺼내서 사용할 리플레이 갯수
REPLAY = 32
# 리플레이를 저장할 리스트
REPLAY_MEMORY = deque()

HISTORY_STEP =4
FRAMESKIP = 4
TRAIN_INTERVAL = 4
TRAIN_START = 50000
TARGET_UPDATE = 30000
SAVE_MODEL = 200
MEMORY_SIZE = 1000000
EXPLORATION = 1000000
START_EXPLORATION = 1.
FINAL_EXPLORATION = 0.01


INPUT = env.observation_space.shape
OUTPUT = env.action_space.n

# 하이퍼파라미터
LEARNING_RATE = 0.00025
DISCOUNT = 0.99
e = 1.
frame = 0
model_path = "save/Breakout.ckpt"

# input data 전처리
def pre_proc(X):
    # 바로 전 frame과 비교하여 max를 취함으로써 flickering을 제거
    # x = np.maximum(X, X1)
    # 그레이 스케일링과 리사이징을 하여 데이터 크기 수정
    x = np.uint8(resize(rgb2gray(X), (110, 84)) * 255)
    x = x[17:101,:]
    return x

# DQN 모델
def model(input1, f1, f2, f3, w1, w2):
    c1 = tf.nn.relu(tf.nn.conv2d(input1, f1, strides=[1, 4, 4, 1], padding = "VALID"))
    c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1], padding="VALID"))
    c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1,1,1,1], padding="VALID"))
    l1 = tf.reshape(c3, [1, w1.get_shape().as_list()[0]])
    l2 = tf.nn.relu(tf.matmul(l1, w1))

    pyx = tf.matmul(l2, w2)
    return pyx


X = tf.placeholder("float", [None, 84, 84, 4])
Y = tf.placeholder("float", [None, OUTPUT])
# 메인 네트워크 Variable
f1 = tf.Variable(tf.random_normal([8,8,4,32],stddev= 0.01))
f2 = tf.Variable(tf.random_normal([4,4,32,64],stddev= 0.01))
f3= tf.Variable(tf.random_normal([3,3,64,64],stddev= 0.01))

w1 = tf.Variable(tf.random_normal([7*7*64, 512], stddev=0.01))
w2 = tf.Variable(tf.random_normal([512, OUTPUT], stddev=0.01))

py_x = model(X, f1, f2, f3 , w1, w2)

# 타겟 네트워크 Variable
f1_r = tf.Variable(tf.random_normal([8,8,4,32],stddev= 0.01))
f2_r = tf.Variable(tf.random_normal([4,4,32,64],stddev= 0.01))
f3_r = tf.Variable(tf.random_normal([3,3,64,64],stddev= 0.01))

w1_r = tf.Variable(tf.random_normal([7*7*64, 512], stddev=0.01))
w2_r = tf.Variable(tf.random_normal([512, OUTPUT], stddev=0.01))

py_x_r = model(X, f1_r, f2_r,f3_r, w1_r, w2_r)

# 총 Reward를 저장해놓을 리스트
rlist=[0]
recent_rlist=[0]
epoch_score = deque()
epoch_Q = deque()
epoch = 0
episode = 0

# Loss function 정의
cost = tf.reduce_sum(tf.square(Y-py_x))
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=0.95 , epsilon=0.01)
train = optimizer.minimize(cost)

saver = tf.train.Saver(max_to_keep=None)

# 세션 정의
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())

    # 에피소드 시작
    while np.mean(recent_rlist) < 500 :
        episode += 1
        # 4개의 프레임을 저장할 history
        history = np.zeros([84, 84, 5], dtype=np.uint8)
        last_action = 0
        average_Q = deque()
        # state 초기화
        s = env.reset()

        # 가장 최근의 100개 episode의 total reward
        if len(recent_rlist) > 100:
            del recent_rlist[0]

        rall = 0
        d = False
        count = 0
        reward = 0
        # 에피소드 시작할때 최대 30만큼 동안 아무 행동 하지않음
        # for _ in range(ran.randint(1, NO_STEP)):
        #    s1, _, _, _ = env.step(0)

        # state의 초기화
        x = pre_proc(s)
        for i in range(HISTORY_STEP):
            history[:,:,i] = x

        # 에피소드가 끝나기 전까지 반복
        while not d :
            # env.render()
            frame +=1
            count+=1
            # e-greedy 정책으로 랜덤하게 action 결정
            e -= (START_EXPLORATION-FINAL_EXPLORATION) / EXPLORATION
            if e < FINAL_EXPLORATION:
                e = FINAL_EXPLORATION

            if count % FRAMESKIP == 1: # Frame skip
                # 현재 state로 Q값을 계산
                Q = np.reshape(np.float32(history / 255.0), [1, 84, 84, 5])[:, :, :, 0:4]
                average_Q.append(np.max(Q))

                if e > np.random.rand(1):
                    a = env.action_space.sample()
                else:
                    a = np.argmax(sess.run(py_x, feed_dict = {X : Q}))

                # 결정된 action으로 Environment에 입력
                s1, r, d, _ = env.step(a)
                reward += np.clip(r,-1,1)
                # 마지막 action
                last_action = a

                # next state를 history에 저장
                history[:, :, 4] = pre_proc(s1)

                # 저장된 state를 Experience Replay memory에 저장
                REPLAY_MEMORY.append([history[:, :, 0:5], a, reward, d])
                history[:,:,0:4] = history[:,:,1:5]
                reward = 0
            else:
                # 마지막 action 반복
                s1, r ,d, _ = env.step(last_action)
                reward += np.clip(r, -1, 1)


            # 저장된 Frame이 1백만개 이상 넘어가면 맨 앞 Replay부터 삭제
            if len(REPLAY_MEMORY) > MEMORY_SIZE:
                REPLAY_MEMORY.popleft()
            # 총 reward 합
            rall += r

            # 2만 frame 이상부터 4개의 Frame마다 학습
            if frame > TRAIN_START and frame % TRAIN_INTERVAL == 1:
                for sample in ran.sample(REPLAY_MEMORY, REPLAY):
                    s_r, a_r, r_r, d_r = sample

                    # 꺼내온 리플레이의 state의 Q값을 예측
                    Q = sess.run(py_x, feed_dict={X: np.reshape(np.float32(s_r[:, :, 0:4] / 255.0), [1, 84, 84, 4])})
                    if d_r:
                        # 꺼내온 리플레이의 상태가 끝난 상황이라면 Negative Reward를 부여
                        Q[0, a_r] = r_r
                    else:
                        # 끝나지 않았다면 Q값을 업데이트
                        Q1 = sess.run(py_x_r, feed_dict={X: np.reshape(np.float32(s_r[:, :, 1:5] / 255.0), [1, 84, 84, 4])})
                        Q[0, a_r] = r_r + DISCOUNT * Q1[0, np.argmax(Q)]

                    # 업데이트 된 Q값으로 main네트워크를 학습
                    sess.run(train, feed_dict={X: np.reshape(np.float32(s_r[:, :, 0:4] / 255.0), [1, 84, 84, 4]), Y: Q})
                # 1만개의 Frame마다 타겟 네트워크 업데이트
                if frame % TARGET_UPDATE == 0 :
                    sess.run(w1_r.assign(w1))
                    sess.run(w2_r.assign(w2))
                    sess.run(f1_r.assign(f1))
                    sess.run(f2_r.assign(f2))
                    sess.run(f3_r.assign(f3))

            # epoch(50000 Trained frame) 마다 plot
            if (frame - TRAIN_START) % (50000*TRAIN_INTERVAL) == 0:
                plt.clf()

                epoch_score.append(np.mean(recent_rlist))
                epoch_Q.append(np.mean(average_Q))

                plt.subplot(211)
                plt.plot(epoch_Q)
                plt.axis([0, 50, 0, 4])
                plt.xlabel('Training Epochs')
                plt.ylabel('Average Action Value(Q)')

                plt.subplot(212)
                plt.axis([0, 50, 0, 300])
                plt.xlabel('Training Epochs')
                plt.ylabel('Average Reward per Episode')
                plt.plot(epoch_score, "r")
                epoch += 1
                plt.pause(0.05)
                plt.savefig("graph/{} epoch".format(epoch))

        # 30만 Frame 마다 모델 저장
        if episode % SAVE_MODEL == 0:
            save_path = saver.save(sess, model_path, global_step=episode)
            print("Model(episode :",episode, ") saved in file: ", save_path)

        # 총 reward의 합을 list에 저장
        recent_rlist.append(rall)
        rlist.append(rall)
        print("Episode:{} steps:{} reward:{}  e-greedy:{} average reward:{} recent reward:{} ".format(episode, count, rall,
                                                                                        e, np.mean(rlist),
                                                                                        np.mean(recent_rlist)))

