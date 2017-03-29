# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
from gym import wrappers
import numpy as np
import random as ran

from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize


env = gym.make('Breakout-v0')

# 꺼내서 사용할 리플레이 갯수
REPLAY = 32
# 리플레이를 저장할 리스트
REPLAY_MEMORY = deque()
# 미니배치
MINIBATCH = 10
NO_STEP=30
HISTORY_STEP =4

INPUT = env.observation_space.shape
OUTPUT = env.action_space.n

# 하이퍼파라미터
LEARNING_RATE = 0.00025
DISCOUNT = 0.99
e = 1.
model_path = "save/model.ckpt"

def pre_proc(X, X1):
    x = np.maximum(X, X1)
    x = np.uint8(resize(rgb2gray(x), (110, 84)) * 255)
    x = x[26:110,:]
    return x


def model(input1, f1, f2,f3, w1, w2):
    c1 = tf.nn.relu(tf.nn.conv2d(input1, f1, strides=[1, 4, 4, 1], padding = "VALID"))

    c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1], padding="VALID"))
    c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1,1,1,1], padding="VALID"))
    l1 = tf.reshape(c3, [1, w1.get_shape().as_list()[0]])
    l2 = tf.nn.relu(tf.matmul(l1, w1))

    pyx = tf.matmul(l2, w2)
    return pyx


X = tf.placeholder("float", [None, 84, 84, 4])
Y = tf.placeholder("float", [None, OUTPUT])

f1 = tf.Variable(tf.random_normal([8,8,4,32],stddev= 0.01))
f2 = tf.Variable(tf.random_normal([4,4,32,64],stddev= 0.01))
f3= tf.Variable(tf.random_normal([3,3,64,64],stddev= 0.01))

w1 = tf.Variable(tf.random_normal([7*7*64, 512], stddev=0.01))

w2 = tf.Variable(tf.random_normal([512, OUTPUT], stddev=0.01))

py_x = model(X, f1, f2, f3 , w1, w2)


f1_r = tf.Variable(tf.random_normal([8,8,4,32],stddev= 0.01))
f2_r = tf.Variable(tf.random_normal([4,4,32,64],stddev= 0.01))
f3_r = tf.Variable(tf.random_normal([3,3,64,64],stddev= 0.01))

w1_r = tf.Variable(tf.random_normal([7*7*64, 512], stddev=0.01))

w2_r = tf.Variable(tf.random_normal([512, OUTPUT], stddev=0.01))

py_x_r = model(X, f1_r, f2_r,f3_r, w1_r, w2_r)

# 총 Reward를 저장해놓을 리스트
rlist=[0]
recent_rlist=[0]

episode = 0

# Loss function 정의
cost = tf.reduce_sum(tf.square(Y-py_x))

optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=0.95, epsilon=0.01)
train = optimizer.minimize(cost)

saver = tf.train.Saver()

# 세션 정의
with tf.Session() as sess:
    # 변수 초기화a
    sess.run(tf.global_variables_initializer())
    # Target 네트워크에 main 네트워크 값을 카피해줌
    sess.run(w1_r.assign(w1))
    sess.run(w2_r.assign(w2))
    sess.run(f1_r.assign(f1))
    sess.run(f3_r.assign(f3))
    sess.run(f2_r.assign(f2))

    # 에피소드 시작
    while np.mean(recent_rlist) < 50 :
        episode += 1
        history = np.zeros([84, 84, 5], dtype=np.uint8)
        # state 초기화
        s1 = env.reset()
        if episode > 500:
            e -= 0.9 / 10000
            if e < 0.1:
                e = 0.1
        if len(recent_rlist) > 100:
            del recent_rlist[0]

        rall = 0
        d = False
        count = 0
        train_P = 0
        train_N = 0

        for _ in range(ran.randint(1, NO_STEP)):
            s = s1
            s1, _, d, _ = env.step(0)

        x = pre_proc(s,s1)
        for i in range(1,5,1):
            history[:,:,i] = x

        # 에피소드가 끝나기 전까지 반복
        while not d :
            # env.render()
            s = s1
            count+=1
            history[:,:,0:4] = history[:,:,1:5]

            Q = np.reshape(np.float32(history/255.0),[1, 84, 84, 5])[:,:,:,0:4]
            # e-greedy 정책으로 랜덤하게 action 결정
            if e > np.random.rand(1):
                a = env.action_space.sample()
            else:
                a = np.argmax(sess.run(py_x, feed_dict = {X : Q}))
                # np.argmax(Q)

            # 결정된 action으로 Environment에 입력
            s1, r, d, _ = env.step(a)

            # Environment에서 반환한 Next_state, action, reward, done 값들을
            # Replay_memory에

            history[:, :, 4] = pre_proc(s,s1)

            REPLAY_MEMORY.append([history[:,:,0:5], a, r, d])
            # 저장된 값들이 50000개 이상 넘어가면 맨 앞 Replay부터 삭제
            if len(REPLAY_MEMORY) > 200000:
                REPLAY_MEMORY.popleft()

            # 총 reward 합
            rall += r
        # 10번의 episode마다 학습

        if episode > 50:
            for i in range(MINIBATCH):
                # 저장된 리플레이 중에 학습에 사용할 랜덤한 리플레이 샘플들을 가져옴
                for sample in ran.sample(REPLAY_MEMORY, REPLAY):
                    s_r, a_r, r_r, d_r = sample
                    if r_r >= 1:
                        train_P+=1
                    r_r = np.clip(r_r,-1,1)
                    # 꺼내온 리플레이의 state의 Q값을 예측
                    Q = sess.run(py_x, feed_dict={X: np.reshape(np.float32(s_r[:, :, 0:4]/255.0), [1, 84, 84, 4])})

                    if d_r:
                        # 꺼내온 리플레이의 상태가 끝난 상황이라면 Negative Reward를 부여
                        Q[0, a_r] = -2
                    else:
                        # 끝나지 않았다면 Q값을 업데이트
                        Q1 = sess.run(py_x_r, feed_dict={X:np.reshape(np.float32(s_r[:, :, 1:5]/255.0), [1, 84, 84, 4])})
                        Q[0, a_r] = r_r + DISCOUNT * np.max(Q1)

                    # 업데이트 된 Q값으로 main네트워크를 학습
                    sess.run(train, feed_dict={X: np.reshape(np.float32(s_r[:, :, 0:4]/255.0), [1, 84, 84, 4]), Y: Q})
            print("train Positive : ", train_P,"  train Negative : ", train_N)

        if episode % 100 == 0:
            # 10번 마다 target 네트워크에 main 네트워크 값을 copy
            sess.run(w1_r.assign(w1))
            sess.run(w2_r.assign(w2))
            sess.run(f1_r.assign(f1))
            sess.run(f2_r.assign(f2))
            sess.run(f3_r.assign(f3))

        if episode % 2000 == 0:
            save_path = saver.save(sess, model_path, global_step=episode)
            print("Model(episode :",episode, ") saved in file: ", save_path)

        # 총 reward의 합을 list에 저장
        recent_rlist.append(rall)
        rlist.append(rall)
        print("Episode:{} steps:{} reward:{}  e-greedy:{} average reward:{} recent reward:{} ".format(episode, count, rall,
                                                                                        e, np.mean(rlist),
                                                                                        np.mean(recent_rlist)))


    rlist=[]
    recent_rlist=[]

'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    save_path = saver.save(sess, model_path)
    saver.restore(sess, model_path)

    print("Model restored form file: ", save_path)
    for episode in range(500):
        # state 초기화
        history = np.zeros([84, 84, 5], dtype=np.uint8)
        # state 초기화
        s = env.reset()

        rall = 0
        d = False
        count = 0
        s = pre_proc(s)
        history[:, :, 4] = s
        # 에피소드가 끝나기 전까지 반복
        while not d :
            env.render()
            count += 1

            history = np.roll(history, -1, axis=2)

            # 현재 상태의 Q값을 에측
            # Q = sess.run(py_x, feed_dict = {X : np.reshape(history,[1, 84, 84, 5])[:,:,:,0:4]})

            # e-greedy 정책으로 랜덤하게 action 결정
            if 0.05 > np.random.rand(1):
                a = env.action_space.sample()
            else:
                a = np.argmax(sess.run(py_x, feed_dict={X: np.reshape(history, [1, 84, 84, 5])[:, :, :, 0:4]}))
                # np.argmax(Q)

            # 결정된 action으로 Environment에 입력
            s1, r, d, _ = env.step(a)

            # Environment에서 반환한 Next_state, action, reward, done 값들을
            # Replay_memory에
            s1 = pre_proc(s1)
            history[:, :, 4] = s1
            # 총 reward 합
            rall += r


        rlist.append(rall)

        print("Episode : {} steps : {} r={}. averge reward : {}".format(episode, count, rall,
                                                                        np.mean(rlist)))
'''