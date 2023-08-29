from collections import deque
import numpy as np


def tsData(data):
    buffer = deque(maxlen=24)
    tsdata = []
    l = len(data)
    for ts in data[:l - 24]:
        if len(buffer) < 23:
            buffer.append(ts)
        else:
            buffer.append(ts)
            state = np.array(buffer, dtype='float')
            tsdata.append(state)
    return tsdata


def tsDataY(data):
    buffery = deque(maxlen=24)
    tsdatay = []
    l = len(data)
    for ts in data[24:l]:
        if len(buffery) < 23:
            buffery.append(ts)
        else:
            buffery.append(ts)
            state = np.array(buffery, dtype='float')
            tsdatay.append(state)
    return tsdatay
