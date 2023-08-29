import random
import time

import numpy as np
from tqdm import tqdm
from keras.models import Model
import os.path
from keras.utils import np_utils

from collections import deque
from FGEnv import FGEnv
from Agent import Agent
from FGPolicy import FGPolicy
from DataEdit import *


class ForecastGame:
    def __init__(self, lstm=False):
        self.env = FGEnv()
        if not lstm:
            self.pred_size = int(self.env.action_size)
            self.dummy_action = np.ones((1, self.pred_size))
            self.state = self.env.train_x[self.env.state]
            self.input_len, _, self.input_width = self.env.train_x.shape

            self.policy = FGPolicy(esp_total=250)

            self.actor = Agent(1, self.input_width, action_size=self.pred_size)
            self.actor.build_model()
            self.actor.setEnv(self.env)
            self.actor.setPolicy(self.policy)

            self.critic = Agent(1, self.input_width, action_size=self.pred_size)
            self.critic.build_model()
            self.critic.setEnv(self.env)
            self.critic.setPolicy(self.policy)

            self.q = Model(inputs=self.actor.model.input, outputs=self.actor.model.get_layer('output_q').output)

            self.memory = deque(maxlen=400)
            self.rRecord = []
            self.rTotal = 0
            self.update_step = 72
            self.hit = 0
            self.forward = 150
            self.stage = 0
        else:
            self.pred_size = 1
            self.state = self.env.trainxlstm[self.env.state]
            self.input_len, self.input_width = self.env.trainxlstm[0].shape
            self.policy = FGPolicy(esp_total=800)

            self.actor = Agent(1, self.input_width, action_size=self.pred_size)
            self.actor.build_model_lstm()
            self.actor.setEnv(self.env)
            self.actor.setPolicy(self.policy)

            self.critic = Agent(1, self.input_width, action_size=self.pred_size)
            self.critic.build_model_lstm()
            self.critic.setEnv(self.env)
            self.critic.setPolicy(self.policy)

            self.memory = deque(maxlen=400)
            self.rRecord = []
            self.rTotal = 0
            self.hit = 0
            self.forward = 10
            self.stage = 0

    def copy_c_to_a(self):
        c_weights = self.critic.model.get_weights()
        a_weights = self.actor.model.get_weights()
        for i in range(len(c_weights)):
            a_weights[i] = c_weights[i]
        self.actor.model.set_weights(a_weights)

    def explore(self, exp):
        s = self.env.reset()
        self.state = self.env.train_x[s]
        for i in range(exp):
            rand_prediction = self.env.sample_pred()
            next_state, reward = self.env.step(rand_prediction)
            self.remember(self.state, rand_prediction, 0, reward, next_state)

            self.state = self.env.train_x[next_state]
        print('finished')

    def explorelstm(self, exp):
        s = self.env.reset()
        self.state = self.env.trainxlstm[s]
        for i in range(exp):
            # state = self.env.state
            rand_prediction = self.env.sample_predlstm()
            next_state, reward = self.env.steplstm(rand_prediction)
            self.rememberlstm(self.state, rand_prediction, reward, next_state)

            self.state = self.env.trainxlstm[next_state]
        print('finished')

    def remember(self, state, prediction, q_value, reward, next_state):
        self.memory.append([state, prediction, q_value, reward, next_state])

    def rememberlstm(self, state, prediction, reward, next_state):
        self.memory.append([state, prediction, reward, next_state])

    def rand_sample(self, num):
        return np.array(random.sample(self.memory, num))

    def get_q_value(self, state, pred):
        ipt = [state.reshape(1, *state.shape), pred]
        return self.q.predict(ipt, verbose=0)

    def select_action(self, state, step, dummy):
        eps = self.policy.epsilon(step)
        if np.random.rand() < eps:
            return self.env.sample_pred(), 0
        self.hit += 1
        q_values = self.get_q_value(state, dummy)
        return np.argmax(q_values), np.max(q_values)

    def select_actionlstm(self, state, step):
        eps = self.policy.epsilon(step)
        if np.random.rand() < eps:
            return self.env.sample_predlstm()
        self.hit += 1
        q_values = self.actor.model.predict(state, verbose=0)
        return q_values

    def writeRewards(self, reward):
        path = 'rewards.txt'
        path = os.path.join(os.getcwd(), path)
        reward = "%s" % reward

        with open(path, "a") as f:
            f.writelines(reward)
            f.writelines('\n')

    def critic_learn(self, sample_size):
        if len(self.memory) < sample_size:
            return

        samples = self.rand_sample(sample_size)

        s, pred, old_q, r, next_state = zip(*samples)

        s = np.array(s, dtype='float')
        pred = np.array(pred, dtype='float').reshape(-1, 1)
        old_q = np.array(old_q, dtype='float').reshape(-1, 1)
        r = np.array(r, dtype='float').reshape(-1, 1)
        action_oh = np_utils.to_categorical(pred, self.pred_size)

        q = 0
        q_estimate = (1 - self.critic.alpha) * old_q + self.critic.alpha * (r + self.critic.gamma * q)

        history = self.critic.model.fit([s, action_oh], q_estimate, epochs=1, verbose=0)
        return np.mean(history.history['loss'])

    def critic_learnlstm(self, sample_size):
        if len(self.memory) < sample_size:
            return

        history = []

        for itr in range(sample_size):
            s, pred, r, next_state = zip(*self.rand_sample(1))

            s = np.array(s, dtype='float')
            s = s.reshape(s.shape[1], 1, s.shape[2])
            pred = np.array(pred, dtype='float').reshape(-1, 1)
            r = np.array(r, dtype='int').reshape(-1, 1)
            dc = 1
            if self.stage > 800:
                dc = 0.5

            q_estimate = pred + self.critic.sigma * r * pred * dc
            history.append(self.critic.model.fit(s, q_estimate, epochs=1, verbose=0))

        loss = 0

        for h in history:
            loss += np.mean(h.history['loss'])

        return loss / sample_size

    def train(self, epochs, resume=False, stage=0):
        filepath = "critic.h5"
        if os.path.exists(filepath):
            self.critic.model.load_weights(filepath)
            self.copy_c_to_a()
            print('load success')
        pbar = None
        if resume:
            fp = 'temp.h5'
            if os.path.exists(fp):
                self.critic.model.load_weights(fp)
                self.copy_c_to_a()
                print('resume')
            pbar = tqdm(range(stage, epochs + 1))
            self.stage = stage
            s = self.env.sample_state()
            for pp in range(100):
                state = self.env.train_x[s]
                pred, q = self.select_action(state,
                                             self.stage, self.dummy_action)
                s, reward = self.env.step(pred)
                self.remember(state, pred, q, reward, s)
        else:
            pbar = tqdm(range(1, epochs + 1))
        for epoch in pbar:
            start = time.time()
            self.rTotal = 0
            for step in range(self.forward):
                pred, q = self.select_action(self.state,
                                             self.stage, self.dummy_action)
                eps = self.policy.epsilon(self.stage)

                next_state, reward = self.env.step(pred)
                self.remember(self.state, pred, q, reward, next_state)

                loss = self.critic_learn(64)
                self.rTotal += reward
                self.state = self.env.train_x[next_state]

                if (epoch * self.forward + step) % 240 == 239:
                    self.critic.model.save_weights('temp.h5')
                    self.copy_c_to_a()

            self.stage += 1
            self.rRecord.append(self.rTotal)
            self.writeRewards(self.rTotal)
            pbar.set_description(
                'R:{} L:{:.6f} T:{} P:{:.3f} H:{}'.format(self.rTotal, loss, int(time.time() - start), eps, self.hit))
            self.actor.count = 0

        self.critic.model.save_weights(filepath)
        print("saved")

    def trainlstm(self, epochs, resume=False, stage=0):
        filepath = "criticlstm.h5"
        if os.path.exists(filepath):
            self.critic.model.load_weights(filepath)
            self.copy_c_to_a()
            print('load success')
        pbar = None
        if resume:
            fp = 'templstm.h5'
            if os.path.exists(fp):
                self.critic.model.load_weights(fp)
                self.copy_c_to_a()
                print('resume')
            pbar = tqdm(range(stage, epochs + 1))
            self.stage = stage
            s = self.env.sample_state()
            for pp in range(100):
                state = self.env.trainxlstm[s]
                ipt = state.reshape(state.shape[0], 1, state.shape[1])
                pred = self.select_actionlstm(ipt, self.stage)
                s, reward = self.env.steplstm(pred)
                self.rememberlstm(state, pred, reward, s)
        else:
            pbar = tqdm(range(1, epochs + 1))
        for epoch in pbar:
            start = time.time()
            self.rTotal = 0
            for step in range(self.forward):
                state = self.state.reshape(self.state.shape[0], 1, self.state.shape[1])
                pred = self.select_actionlstm(state, self.stage)
                eps = self.policy.epsilon(self.stage)

                next_state, reward = self.env.steplstm(pred)
                self.rememberlstm(self.state, pred, reward, next_state)

                loss = self.critic_learnlstm(3)
                self.rTotal += sum(reward)
                self.state = self.env.trainxlstm[next_state]

                if (epoch * self.forward + step) % 10 == 0:
                    self.critic.model.save_weights('templstm.h5')
                    self.copy_c_to_a()

            self.stage += 1
            self.rRecord.append(self.rTotal)
            self.writeRewards(self.rTotal)
            # print(loss)
            pbar.set_description(
                'R:{} L:{:.6f} T:{} P:{:.3f} H:{}'.format(self.rTotal, loss, int(time.time() - start), eps, self.hit))
            self.actor.count = 0

        self.critic.model.save_weights(filepath)
        print("saved")

    def test(self):
        filename = 'critic.h5'
        if os.path.isfile(filename):
            self.actor.model.load_weights(filename)
        else:
            return

        ipt = [self.env.test_x, np.ones(shape=(len(self.env.test_t), self.pred_size))]

        q = self.q.predict(ipt)
        pred = np.argmax(q, axis=1)

        return pred


# fg = ForecastGame(True)
# fg.explorelstm(200)
# fg.trainlstm(2000)
# fg.critic_learnlstm(10)


# fg = ForecastGame()
# fg.explore(300)
# print(fg.actor.model.predict(fg.state.reshape(1, fg.state.shape[0], fg.state.shape[1]))[0][0])
# fg.train(1)
# print(fg.test())
# fg.train(2000)
# fg.train(700, resume=True, stage=600)

# print(fg.test())
