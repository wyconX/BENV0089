import random

import pandas as pd
from keras.utils import np_utils
import numpy as np
from DataEdit import *


class FGEnv:

    def __init__(self):
        self.data = pd.read_csv('Data.csv')

        self.train_t = self.data[:8760].TotalLoad
        self.test_t = self.data[8760:].TotalLoad
        self.tlstm = self.data.TotalLoad
        self.minn = self.data.TotalLoad.min()

        data = self.normalize(self.data)
        # data = self.data
        self.train_x = data.drop(columns=['TotalLoad', 'HourlyUnix'])[:8760].values
        self.train_y = data[:8760].TotalLoad
        self.test_x = data.drop(columns=['TotalLoad', 'HourlyUnix'])[8760:].values
        self.test_y = data[8760:].TotalLoad

        self.train_x = self.train_x.reshape(self.train_x.shape[0], 1, self.train_x.shape[1])
        self.test_x = self.test_x.reshape(self.test_x.shape[0], 1, self.test_x.shape[1])

        self.train_c = self.loadCato(self.train_t)
        self.test_c = self.loadCato(self.test_t)

        lstmx = tsData(data.drop(columns=['TotalLoad', 'HourlyUnix']).values)
        self.trainxlstm = lstmx[:8760]
        self.testxlstm = lstmx[8760:]

        lstmy = tsDataY(data.TotalLoad.values)
        self.trainylstm = lstmy[:8760]
        self.testylstm = lstmy[8760:]

        self.state = 0
        self.action_size = (self.data.TotalLoad.max() - self.minn) // 2500 + 1

    def normalize(self, data):
        return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    def loadCato(self, data):
        return (data - data.min(axis=0)) // 2500

    def cateRe(self, cato):
        return cato * 2500 + self.minn


    def restore(self, prediction):
        return prediction * (
                self.data.TotalLoad.max(axis=0) - self.data.TotalLoad.min(axis=0)) + self.data.TotalLoad.min(axis=0)

    def reward(self, prediction):
        y = self.train_c[self.state]
        if prediction == y:
            return 1
        else:
            return -1

    def rewardlstm(self, prediction):
        y = self.tlstm[self.state: self.state + 24].values
        rewards = []
        for i in range(24):
            pred = self.restore(prediction[i])
            if abs(pred - y[i:i+1]) < 50:
                rewards.append(0)
            elif pred - y[i] > 50:
                rewards.append(-1)
            else:
                rewards.append(1)
        return rewards

    def forward(self, prediction, state):
        y = self.train_c[state]
        if prediction == y:
            return 1
        else:
            return -1

    def step(self, prediction):
        if prediction == -1:
            current = self.state
            self.state = 0
            return current, 0
        r = self.reward(prediction)
        self.state = self.sample_state()
        return self.state, r

    def steplstm(self, prediction, reset=False):
        if reset:
            current = self.state
            self.state = 0
            return current, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        r = self.rewardlstm(prediction)
        self.state = self.sample_state()
        return self.state, r

    def reset(self):
        state, _ = self.step(-1)
        return state

    def sample_state(self):
        return random.randint(0, len(self.train_t) - 1)

    def sample_pred(self):
        return random.randint(0, self.action_size - 1)

    def sample_predlstm(self):
        preds = []
        for i in range(24):
            preds.append(random.uniform(0, 1))
        return preds

# a = FGEnv()

# a.reward(0.1)
