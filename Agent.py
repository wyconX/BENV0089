from keras.models import Sequential
from keras.models import Model
import keras.backend as K
from keras.layers import *
from tensorflow import keras
from matplotlib import pyplot
import pandas as pd
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Agent:
    def __init__(self, input_length, input_width, action_size=466, alpha=0.5, gamma=0):
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = 0.1
        self.action_size = action_size
        self.input_length = input_length
        self.input_width = input_width
        self.seqLength = 24
        self.model = None
        self.policy = None
        self.env = None
        self.count = 0

    def build_model(self):
        load_input = Input(shape=(self.input_length, self.input_width))
        layer1 = Dense(128, activation='relu')(load_input)
        layer2 = Dense(64, activation='relu')(layer1)
        output = Dense(self.action_size, name='output_q')(layer2)

        prediction_input = Input((self.action_size,), name='prediction')
        q_value = multiply([prediction_input, output])
        q_value = Lambda(lambda l: K.sum(l, axis=1, keepdims=True), name='q_value')(q_value)

        model = Model(inputs=[load_input, prediction_input], outputs=q_value)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def build_model_lstm(self):
        load_input = Input(shape=(self.input_length, self.input_width))
        layer1 = Dense(128)(load_input)
        layer2 = LSTM(128)(layer1)
        output = Dense(1, name='output_q')(layer2)

        model = Model(inputs=load_input, outputs=output)
        model.compile(loss='mse', optimizer='adam', metrics='accuracy')
        self.model = model

    def setPolicy(self, policy):
        self.policy = policy

    def setEnv(self, env):
        self.env = env


def nmlz(data):
    min = data.min(axis=0)
    max = data.max(axis=0)
    return (data - min) / (max - min)


# a = Agent(1, 5)
# a.build_model_lstm()
# dd = pd.read_csv('Data.csv')
# print(dd.mean(axis=0))
# mean = dd.mean()['TotalLoad']
# std = dd.std()['TotalLoad']
# dd = nmlz(dd)
# x = dd.drop(columns='TotalLoad')[:8760].values
# y = dd[:8760].TotalLoad
# xt = dd.drop(columns='TotalLoad')[8760:].values
# yt = dd[8760:].TotalLoad
#
# # print(x.shape[0])
# x = x.reshape(x.shape[0], 1, x.shape[1])
# xt = xt.reshape(xt.shape[0], 1, xt.shape[1])
# q = a.model.predict(x[0:10])
# print(q)
# # print(q*std + mean)
# # print(y[0:10]*std + mean)
#
#
# history = a.model.fit(x, y, epochs=1, batch_size=24, validation_data=(xt, yt))
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
