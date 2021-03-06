# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
from gym import wrappers
import numpy as np
import random as ran

from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize


env = gym.make('BreakoutDeterministic-v0')

# 꺼내서 사용할 리플레이 갯수
REPLAY = 32
# 리플레이를 저장할 리스트
REPLAY_MEMORY = deque()
# 미니배치


INPUT = env.observation_space.shape
OUTPUT = env.action_space.n

# 하이퍼파라미터
LEARNING_LATE = 0.0005
DISCOUNT = 0.99
e = 1.
model_path = "save/model.ckpt"

def pre_proc(X):
    x = np.uint8(resize(rgb2gray(X), (84, 84)) * 255)
    return x

def model(input1, f1, f2,f3, w1, w2):
    c1 = tf.nn.relu(tf.nn.conv2d(input1, f1, strides=[1, 4, 4, 1], padding = "SAME"))

    c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1], padding="SAME"))

    c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1,1,1,1],padding="SAME"))
    l1 = tf.reshape(c3, [1, w1.get_shape().as_list()[0]])
    l2 = tf.nn.relu(tf.matmul(l1, w1))

    pyx = tf.matmul(l2, w2)
    return pyx


X = tf.placeholder("float", [None, 84, 84, 4])
Y = tf.placeholder("float", [None, OUTPUT])

f1 = tf.Variable(tf.random_normal([8,8,4,32],stddev= 0.001))
f2 = tf.Variable(tf.random_normal([4,4,32,64],stddev= 0.001))
f3 = tf.Variable(tf.random_normal([3,3,64,64],stddev= 0.001))

w1 = tf.Variable(tf.random_normal([7744, 512], stddev=0.001))

w2 = tf.Variable(tf.random_normal([512, OUTPUT], stddev=0.001))

py_x = model(X, f1, f2, f3, w1, w2)


f1_r = tf.Variable(tf.random_normal([8,8,4,32],stddev= 0.001))
f2_r = tf.Variable(tf.random_normal([4,4,32,64],stddev= 0.001))
f3_r = tf.Variable(tf.random_normal([3,3,64,64],stddev= 0.001))

w1_r = tf.Variable(tf.random_normal([7744, 512], stddev=0.001))

w2_r = tf.Variable(tf.random_normal([512, OUTPUT], stddev=0.001))

py_x_r = model(X, f1_r, f2_r, f3_r, w1_r, w2_r)


# 총 Reward를 저장해놓을 리스트
rlist=[0]
recent_rlist=[0]

episode = 0

# Loss function 정의
cost = tf.reduce_sum(tf.square(Y-py_x))
cost_summ = tf.summary.scalar('cost', cost)

optimizer = tf.train.AdamOptimizer(LEARNING_LATE)
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
    sess.run(f2_r.assign(f2))
    sess.run(f3_r.assign(f3))

    # 에피소드 시작
    while np.mean(recent_rlist) < 30 :
        episode += 1
        history = np.zeros([84, 84, 5])
        # state 초기화
        s = env.reset()

        if len(recent_rlist) > 100:
            del recent_rlist[0]
        # e-greedy
        if episode > 500:
            e -= 0.9 / 10000
        if e < 0.1 :
            e = 0.1

        rall = 0
        d = False
        count = 0
        s = pre_proc(s)
        history[:, :, 4] = s
        # 에피소드가 끝나기 전까지 반복
        while not d :
            #env.render()
            count+=1
            history = np.roll(history, -1, axis=2)
            # 현재 상태의 Q값을 에측
            # Q = sess.run(py_x, feed_dict = {X : np.reshape(history,[1, 84, 84, 5])[:,:,:,0:4]})

            # e-greedy 정책으로 랜덤하게 action 결정
            if e > np.random.rand(1):
                a = env.action_space.sample()
            else:
                a = np.argmax(sess.run(py_x, feed_dict = {X : np.reshape(history,[1, 84, 84, 5])[:,:,:,0:4]}))
                # np.argmax(Q)

            # 결정된 action으로 Environment에 입력
            s1, r, d, _ = env.step(a)
            # print(r)

            # Environment에서 반환한 Next_state, action, reward, done 값들을
            # Replay_memory에
            s1 = pre_proc(s1)
            history[:,:,4] = s1

            REPLAY_MEMORY.append([history[:,:,0:4], a, r, history[:,:,1:5], d])
            # 저장된 값들이 50000개 이상 넘어가면 맨 앞 Replay부터 삭제
            if len(REPLAY_MEMORY) > 400000:
                REPLAY_MEMORY.popleft()

            # 총 reward 합
            rall += r
            # state를 Next_state로 바꿈

        # 총 reward의 합을 list에 저장
        recent_rlist.append(rall)
        rlist.append(rall)
        print("Episode:{} steps:{} reward:{}  e-greedy:{} average reward:{} recent reward:{} ".format(episode, count, rall,
                                                                                        e, np.mean(rlist),
                                                                                        np.mean(recent_rlist)))

        # 10번의 episode마다 학습
        if episode % 4 == 0 :

            # 저장된 리플레이 중에 학습에 사용할 랜덤한 리플레이 샘플들을 가져옴
            for sample in ran.sample(REPLAY_MEMORY, REPLAY):

                s_r, a_r, r_r, s1_r, d_r = sample

                # 꺼내온 리플레이의 state의 Q값을 예측
                Q = sess.run(py_x, feed_dict={X : np.reshape(s_r,[1,84,84,4])})

                if d_r:
                    # 꺼내온 리플레이의 상태가 끝난 상황이라면 Negative Reward를 부여
                    Q[0, a_r] = -1
                else:
                    # 끝나지 않았다면 Q값을 업데이트
                    Q1 = sess.run(py_x_r, feed_dict={X:np.reshape(s1_r,[1,84,84,4])})
                    Q[0, a_r] = r_r * 100 + DISCOUNT * np.max(Q1)

                # 업데이트 된 Q값으로 main네트워크를 학습
                sess.run(train, feed_dict={X: np.reshape(s_r,[1,84,84,4]), Y: Q})

            # 10번 마다 target 네트워크에 main 네트워크 값을 copy
            if episode % 100 == 0:
                sess.run(w1_r.assign(w1))
                sess.run(w2_r.assign(w2))
                sess.run(f1_r.assign(f1))
                sess.run(f2_r.assign(f2))
                sess.run(f3_r.assign(f3))


    save_path = saver.save(sess, model_path)
    print("Model saved in file: ",save_path)


    rlist=[]
    recent_rlist=[]

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