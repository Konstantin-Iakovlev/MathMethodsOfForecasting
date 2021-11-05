import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import tensorly
from sklearn.preprocessing import StandardScaler

PHONE_DIR = 'wisdm-dataset/raw/phone/'


def get_data_frame(index, device):
    df = pd.read_csv(os.path.join(PHONE_DIR, device, f'data_{index}_{device}_phone.txt'),
            header=None)
    df = df[df[1] == 'G']
    df = df[[3, 4, 5]]
    df.reset_index(inplace=True, drop=True)
    df[5] = df[5].apply(lambda s: s[:-1])
    # mean inside window of lenght 30
    df = df.groupby(df.index // 30).mean()
    return df


def get_data():
    OBJECTS = 10
    TIMESTAPMS = 40
    CHANNELS = 10
    np.random.seed(1)
    t = np.linspace(0, 1, num=TIMESTAPMS)
    X = np.zeros((OBJECTS, TIMESTAPMS, OBJECTS))
    Y = np.zeros((OBJECTS, TIMESTAPMS, OBJECTS))
    sigma = 0.01
    for i in range(OBJECTS):
        for c in range(CHANNELS):
            omega = np.random.rand(1) * 10 + 20
            phase = np.random.rand() * np.pi
            X[i, :, c] = np.sin(omega * t + phase)
        W = np.random.rand(CHANNELS, CHANNELS)
        Y[i] = X[i] @ W
    return X, Y + np.random.randn(*Y.shape) * sigma
