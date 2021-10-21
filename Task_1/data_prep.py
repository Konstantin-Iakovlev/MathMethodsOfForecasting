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
    df = df[df[1] == 'M']
    df = df[[3, 4, 5]]
    df.reset_index(inplace=True, drop=True)
    df[5] = df[5].apply(lambda s: s[:-1])
    # mean inside window of lenght 30
    df = df.groupby(df.index // 30).mean()
    return df


def get_data():
    TIMESTAPMS = 40
    X_acc = []
    X_gyro = []
    scaler = StandardScaler()
    for index in range(1600, 1651):
        df_acc = get_data_frame(index, 'accel')
        df_gyro = get_data_frame(index, 'gyro')
        # scaling data
        X_acc.append(scaler.fit_transform(df_acc)[:TIMESTAPMS])
        X_gyro.append(scaler.fit_transform(df_gyro)[:TIMESTAPMS])
    return np.array(X_acc), np.array(X_gyro)

