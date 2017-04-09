# -*- coding: utf-8 -*-
import tensorflow as tf
import gym

import numpy as np
import random as ran

import matplotlib.pyplot as plt

from collections import deque
from skimage.transform import resize
# from skimage.color import rgb2gray

plt.ion()
env = gym.make('BreakoutDeterministic-v3')

# 꺼내서 사용할 리플레이 갯수
REPLAY = 32
# 리플레이를 저장할 리스트
REPLAY_MEMORY = deque()

HISTORY_STEP =4
FRAMESKIP = 4
TRAIN_INTERVAL = 4
TRAIN_START = 50000/FRAMESKIP
TARGET_UPDATE = 30000/FRAMESKIP
SAVE_MODEL = 200
MEMORY_SIZE = 1000000
EXPLORATION = 1000000/FRAMESKIP
START_EXPLORATION = 1.
FINAL_EXPLORATION = 0.01

INPUT = env.observation_space.shape
OUTPUT = 3

# 하이퍼파라미터
LEARNING_RATE = 0.00025
DISCOUNT = 0.99
e = 1.
frame = 0
model_path = "save/Breakout.ckpt"
def cliped_error(x):
    return tf.where(tf.abs(x) < 1.0 , 0.5 * tf.square(x), tf.abs(x)-0.5)

# input data 전처리
def rgb2gray(rgb):
    return np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114])

def pre_proc(X):
    # 바로 전 frame과 비교하여 max를 취함으로써 flickering을 제거
    # x = np.maximum(X, X1)
    # 그레이 스케일링과 리사이징을 하여 데이터 크기 수정
    x = np.uint8(resize(rgb2gray(X), (110,84))[17:101, :])
    return x

# DQN 모델
def model(input1, f1, f2, f3, w1, w2):
    c1 = tf.nn.relu(tf.nn.conv2d(input1, f1, strides=[1, 4, 4, 1],data_format="NHWC", padding = "VALID"))
    c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1],data_format="NHWC", padding="VALID"))
    c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1,1,1,1],data_format="NHWC", padding="VALID"))
    l1 = tf.reshape(c3, [1, w1.get_shape().as_list()[0]])
    l2 = tf.nn.relu(tf.matmul(l1, w1))

    pyx = tf.matmul(l2, w2)
    return pyx


X = tf.placeholder("float", [None, 84, 84, 4])
Y = tf.placeholder("float", [None, OUTPUT])
# 메인 네트워크 Variable
f1 = tf.get_variable("f1", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
f2 = tf.get_variable("f2", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
f3 = tf.get_variable("f3", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

w1 = tf.get_variable("w1", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.get_variable("w2", shape=[512, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

py_x = model(X, f1, f2, f3 , w1, w2)

# 타겟 네트워크 Variable
f1_r = tf.get_variable("f1_r", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
f2_r = tf.get_variable("f2_r", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
f3_r = tf.get_variable("f3_r", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

w1_r = tf.get_variable("w1_r", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
w2_r = tf.get_variable("w2_r", shape=[512, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

py_x_r = model(X, f1_r, f2_r,f3_r, w1_r, w2_r)

# 총 Reward를 저장해놓을 리스트
rlist=[0]
recent_rlist=[0]

episode = 0
epoch = 0
epoch_score = deque()
epoch_Q = deque()
epoch_on = False

# Loss function 정의
error = Y- py_x
cost = tf.reduce_mean(cliped_error(error))
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE,momentum=0.95,epsilon=0.01)
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
        average_Q = deque()
        # state 초기화
        s = env.reset()

        # 가장 최근의 100개 episode의 total reward
        if len(recent_rlist) > 100:
            del recent_rlist[0]

        rall = 0
        d = False
        count = 0

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
            # 최근 4개의 프레임을 현재 프레임으로 바꿔줌

            frame +=1
            count+=1

            # e-greedy 정책으로 랜덤하게 action 결정
            e -= (START_EXPLORATION- FINAL_EXPLORATION) / EXPLORATION
            if e < FINAL_EXPLORATION:
                e = FINAL_EXPLORATION

            # 현재 state로 Q값을 계산
            Q = sess.run(py_x, feed_dict = {X : np.reshape(np.float32(history/255), [1, 84, 84, 5])[:, :, :, 0:4]})
            average_Q.append(np.max(Q))
            if e > np.random.rand(1):
                a = np.random.randint(OUTPUT)
            else:
                a = np.argmax(Q)

            # 결정된 action으로 Environment에 입력
            s1, r, d, _ = env.step(a+1)
            reward= np.clip(r, -1,1)
            # next state를 history에 저장
            history[:, :, 4] = pre_proc(s1)

            # 저장된 state를 Experience Replay memory에 저장
            REPLAY_MEMORY.append([history[:, :, 0:5], a, reward, d])
            history[:,:,0:4] = history[:,:,1:5]

            # 저장된 Frame이 1백만개 이상 넘어가면 맨 앞 Replay부터 삭제
            if len(REPLAY_MEMORY) > MEMORY_SIZE:
                REPLAY_MEMORY.popleft()
            # 총 reward 합
            rall += r

            # 5만 frame 이상부터 4개의 Frame마다 학습
            if frame > TRAIN_START :
                for sample in ran.sample(REPLAY_MEMORY, REPLAY):
                    s_r, a_r, r_r, d_r = sample
                    # 꺼내온 리플레이의 state의 Q값을 예측
                    Q = sess.run(py_x, feed_dict={X: np.reshape(np.float32(s_r[:, :, 0:4]/255), [1, 84, 84, 4])})
                    if d_r:
                        # 꺼내온 리플레이의 상태가 끝난 상황이라면 Negative Reward를 부여
                        Q[0, a_r] = r_r
                    else:
                        # 끝나지 않았다면 Q값을 업데이트
                        Q1 = sess.run(py_x_r, feed_dict={X: np.reshape(np.float32(s_r[:, :, 1:5]/255), [1, 84, 84, 4])})
                        Q[0, a_r] = r_r + DISCOUNT * Q1[0, np.argmax(Q)]

                    # 업데이트 된 Q값으로 main네트워크를 학습
                    sess.run(train, feed_dict={X: np.reshape(np.float32(s_r[:, :, 0:4]/255) , [1, 84, 84, 4]), Y: Q})
                # 3만개의 Frame마다 타겟 네트워크 업데이트
                if frame % TARGET_UPDATE == 0 :
                    sess.run(w1_r.assign(w1))
                    sess.run(w2_r.assign(w2))
                    sess.run(f1_r.assign(f1))
                    sess.run(f2_r.assign(f2))
                    sess.run(f3_r.assign(f3))

            # epoch(50000 Trained frame) 마다 plot
            if (frame - TRAIN_START) % 50000 == 0:
                epoch_on = True

        if epoch_on:
            plt.clf()
            epoch += 1
            epoch_score.append(np.mean(recent_rlist))
            epoch_Q.append(np.mean(average_Q))

            plt.subplot(211)
            plt.axis([0, epoch, 0, np.max(epoch_Q)*6/5])
            plt.xlabel('Training Epochs')
            plt.ylabel('Average Action Value(Q)')
            plt.plot(epoch_Q)

            plt.subplot(212)
            plt.axis([0, epoch,0,np.max(epoch_score)*6/5])
            plt.xlabel('Training Epochs')
            plt.ylabel('Average Reward per Episode')
            plt.plot(epoch_score, "r")

            epoch_on = False
            plt.pause(0.05)
            plt.savefig("graph/{} epoch".format(epoch))

        # 200 episode 마다 모델 저장
        if episode % SAVE_MODEL == 0:
            save_path = saver.save(sess, model_path, global_step=episode)
            print("Model(episode :",episode, ") saved in file: ", save_path)



        # 총 reward의 합을 list에 저장
        recent_rlist.append(rall)
        rlist.append(rall)

        print("Episode:{} steps:{} reward:{}  e-greedy:{} average reward:{} recent reward:{} ".format(episode, count, rall,
                                                                                        e, np.mean(rlist),
                                                                                        np.mean(recent_rlist)))

