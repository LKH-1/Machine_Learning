# -*- coding: utf-8 -*-
import tensorflow as tf
import gym

import numpy as np
import random as ran
import datetime
import matplotlib.pyplot as plt

from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray

plt.ion()
# DQN paper setting(frameskip = 4, repeat_action_probability = 0)
# {}Deterministic : frameskip = 4
# {}-v3 : repeat_action_probability
env = gym.make('BreakoutDeterministic-v3')

DDQN = False

# 꺼내서 사용할 리플레이 갯수
MINIBATCH = 32
HISTORY_SIZE =4
TRAIN_START = 50000
if DDQN:
    FINAL_EXPLORATION = 0.01
    TARGET_UPDATE = 30000
else:
    FINAL_EXPLORATION = 0.1
    TARGET_UPDATE = 10000

MEMORY_SIZE = 200000
EXPLORATION = 1000000
START_EXPLORATION = 1.
INPUT = env.observation_space.shape
OUTPUT = 3
HEIGHT =84
WIDTH = 84
LEARNING_RATE = 0.00025
DISCOUNT = 0.99

model_path = "save/Breakout.ckpt"

def cliped_error(x):
    return tf.where(tf.abs(x) < 1.0 , 0.5 * tf.square(x), tf.abs(x)-0.5)


def pre_proc(X):
    # 바로 전 frame과 비교하여 max를 취함으로써 flickering을 제거
    # x = np.maximum(X, X1)
    # 그레이 스케일링과 리사이징을 하여 데이터 크기 수정
    x = np.uint8(resize(rgb2gray(X), (HEIGHT,WIDTH))*255)
    return x

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def get_init_state(history, s):
    for i in range(HISTORY_SIZE):
        history[:,:,i] = pre_proc(s)

def get_game_type(count, l, no_life_game, start_live):
    if count == 1:
        start_live = l['ale.lives']
        if start_live == 0:
            no_life_game = True
        else:
            no_life_game = False
    return [no_life_game, start_live]

def get_terminal(start_live, l, reward,no_life_game, ter):
    if no_life_game:
        # 목숨이 없는 게임일 경우 Terminal 처리
        if reward < 0:
            ter = True
    else :
        # 목숨 있는 게임일 경우 Terminal 처리
        if start_live > l['ale.lives']:
            ter = True
            start_live = l['ale.lives']

    return [ter, start_live]

def train_minibatch(mainDQN, targetDQN, minibatch):
    s_stack = []
    a_stack = []
    r_stack = []
    s1_stack = []
    d_stack = []
    y_stack = []

    for s_r, a_r, r_r, d_r in minibatch:
        s_stack.append(s_r[:, :, :4])
        a_stack.append(a_r)
        r_stack.append(r_r)
        s1_stack.append(s_r[:, :, 1:])
        d_stack.append(d_r)

    d_stack = np.array(d_stack) + 0
    y = mainDQN.get_q(np.array(s_stack))
    Q1 = targetDQN.get_q(np.array(s1_stack))
    for i in range(MINIBATCH):
        y[i, a_stack[i]] = r_stack[i] + (1 - d_stack[i]) * DISCOUNT * np.max(Q1[i])

    # 업데이트 된 Q값으로 main네트워크를 학습
    mainDQN.sess.run(mainDQN.train, feed_dict={mainDQN.X: np.float32(np.array(s_stack) / 255.), mainDQN.Y: y})

def plot_data(epoch, epoch_score, average_reward, epoch_Q, average_Q, mainDQN):
    plt.clf()
    epoch += 1
    epoch_score.append(np.mean(average_reward))
    epoch_Q.append(np.mean(average_Q))

    plt.subplot(211)
    plt.axis([0, epoch, 0, np.max(epoch_Q) * 6 / 5])
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Action Value(Q)')
    plt.plot(epoch_Q)

    plt.subplot(212)
    plt.axis([0, epoch, 0, np.max(epoch_score) * 6 / 5])
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Reward per Episode')
    plt.plot(epoch_score, "r")

    plt.pause(0.05)
    plt.savefig("graph/{} epoch".format(epoch - 1))

    save_path = mainDQN.saver.save(mainDQN.sess, model_path, global_step=(epoch - 1))
    print("Model(epoch :", epoch, ") saved in file: ", save_path, " Now time : ", datetime.datetime.now())

class DQNAgent:
    def __init__(self, sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, NAME='main'):
        self.sess = sess
        self.height = HEIGHT
        self.width = WIDTH
        self.history_size = HISTORY_SIZE
        self.output = OUTPUT
        self.name = NAME
        self.e = 1.0
        self.build_network()

    def build_network(self, LEARNING_RATE = 0.00025, MOMENTUM = 0.95, EPSILON = 0.01):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder('float', [None, self.height, self.width, self.history_size])
            self.Y = tf.placeholder('float', [None, self.output])

            f1 = tf.get_variable("f1", shape=[8, 8, 4, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            f2 = tf.get_variable("f2", shape=[4, 4, 32, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            f3 = tf.get_variable("f3", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w1 = tf.get_variable("w1", shape=[7 * 7 * 64, 512], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable("w2", shape=[512, OUTPUT], initializer=tf.contrib.layers.xavier_initializer())

            c1 = tf.nn.relu(tf.nn.conv2d(self.X, f1, strides=[1, 4, 4, 1], padding="VALID"))
            c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1], padding="VALID"))
            c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1, 1, 1, 1], padding='VALID'))

            l1 = tf.reshape(c3, [-1, w1.get_shape().as_list()[0]])
            l2 = tf.nn.relu(tf.matmul(l1, w1))

        self.Q_pre = tf.matmul(l2, w2)
        self.error = self.Y - self.Q_pre
        self.loss = tf.reduce_mean(cliped_error(self.error))
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=EPSILON)
        self.train = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=None)

    def get_q(self, history):
        return self.sess.run(self.Q_pre, feed_dict={self.X: np.reshape(np.float32(history / 255.),
                                                                [-1, 84, 84, 4])})
    def get_action(self, q, e):
        if e > np.random.rand(1):
            action = np.random.randint(self.output)
        else:
            action = np.argmax(q)
        return action

def main():
    with tf.Session() as sess:
        mainDQN = DQNAgent(sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, NAME='main')
        targetDQN = DQNAgent(sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, NAME = 'target')

        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        rlist = [0]
        recent_rlist = deque(maxlen=100)
        e = 1.
        episode, epoch, frame = 0,0,0

        epoch_score, epoch_Q = deque(),deque()
        average_Q, average_reward = deque(),deque()

        epoch_on = False
        no_life_game = False
        REPLAY_MEMORY = deque(maxlen=200000)

        # Train agent during 200 epoch
        while epoch <= 200:
            episode += 1

            history = np.zeros([84, 84, 5], dtype=np.uint8)
            rall, count = 0,0
            d = False
            ter = False
            start_lives = 0
            s = env.reset()

            get_init_state(history, s)

            while not d:
                #env.render()

                frame += 1
                count += 1

                if e > FINAL_EXPLORATION and frame > TRAIN_START:
                    e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

                Q = mainDQN.get_q(history[:,:,:4])
                average_Q.append(np.max(Q))

                action = mainDQN.get_action(Q,e)

                # 액션 개수 줄임
                if action == 0:
                    real_a = 1
                elif action == 1:
                    real_a = 4
                else:
                    real_a = 5

                # s1 : next frame / r : reward / d : done(terminal) / l : info(lives)
                s1, r, d, l = env.step(real_a)
                ter = d
                reward = np.clip(r, -1, 1)

                no_life_game,start_lives= get_game_type(count, l,no_life_game,start_lives)
                ter, start_lives = get_terminal(start_lives,l,reward,no_life_game, ter)

                history[:,:,4] = pre_proc(s1)

                # 메모리 저장 효율을 높이기 위해 5개의 프레임을 가진 히스토리를 저장
                # state와 next_state는 3개의 데이터가 겹침을 이용.
                REPLAY_MEMORY.append((np.copy(history[:,:,:]), action, reward , ter))
                history[:,:,:4] = history[:,:,1:]

                rall += r

                if frame > TRAIN_START :
                    minibatch = ran.sample(REPLAY_MEMORY, MINIBATCH)
                    train_minibatch(mainDQN, targetDQN, minibatch)

                    if frame % TARGET_UPDATE == 0:
                        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                                    src_scope_name="main")
                        sess.run(copy_ops)

                if (frame - TRAIN_START) % 50000 == 0:
                    epoch_on = True

            recent_rlist.append(rall)
            rlist.append(rall)
            average_reward.append(rall)

            print("Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | Reward:{3:3.0f} | e-greedy:{4:.5f} | "
                  "Avg_Max_Q:{5:2.5f} | Recent reward:{6:.5f}  ".format(episode, frame, count, rall, e,
                                                                        np.mean(average_Q),
                                                                        np.mean(recent_rlist)))

            if epoch_on:
                plot_data(epoch, epoch_score, average_reward, epoch_Q, average_Q, mainDQN)
                epoch_on = False
                average_reward = deque()
                average_Q = deque()

if __name__ == "__main__":
    main()







