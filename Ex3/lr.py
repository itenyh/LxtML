#coding:utf-8

from __future__ import division
import numpy as np
import random

__author__ = 'itensb'

def create_data(size = 1000):

    target_data = []
    for i in range(0, size):

        rd1 = random.randint(-1000, 1000) / 1000
        rd2 = random.randint(-1000, 1000) / 1000
        y = pow(rd1, 2) + pow(rd2, 2) - 0.6
        y = 1 if y > 0 else -1

        target_data.append([rd1, rd2, y])

    # print(target_data)

    noisy_data = []
    for data in target_data:

        is_noise = random.randint(1, 100) / 100.0
        data[-1] = -data[-1] if is_noise <= 0.1 else data[-1]
        noisy_data.append(data)

    # print(noisy_data)
    return noisy_data

def linear_model(data):

    X = [[item[0], item[1], 1] for item in data]
    Y = [item[2]for item in data]

    # X = [[0.5, 0.1, 1], [0.1, 0.2 , 1]]
    # Y = [[1, -1]]

    X_mat = np.mat(X)
    Y_mat = np.mat(Y)

    H = X_mat * (X_mat.T * X_mat).I * X_mat.T
    Y_p = H * Y_mat.T
    sum_error = ((Y_mat - Y_p.T) * (Y_mat - Y_p.T).T) / 1000

    return sum_error

'''
data = create_data()
# data = [[0.5, 0.1, 1], [0.1, 0.2 , -1]]
X = [[item[0], item[1], 1] for item in data]
Y = [item[2]for item in data]

X_mat = np.mat(X)
Y_mat = np.mat(Y)

X1 = [[item[0]] for item in data]
X2 = [[item[1]] for item in data]
X1X2 = [[x1[0] * x2[0]] for x1, x2 in zip(X1, X2)]
X1p = [[x1[0] * x1[0]] for x1 in X1]
X2p = [[x2[0] * x2[0]] for x2 in X2]

new_X = []
for i in range(0, len(X1)):

    new_X.append([X1[i][0], X2[i][0], X1X2[i][0], X1p[i][0], X2p[i][0], 1])

X_mat = np.mat(new_X)
w1 = [-0.05, 0.08, 0.13, 1.5, 15, -1] #36.814
w2 = [-1.5, 0.08, 0.13, 0.05, 1.5, -1] #1.85
w3 = [-1.5, 0.08, 0.13, 0.05, 0.05, -1] #2.7222
w4 = [-1.5, 0.05, 0.13, 1.5, 1.5, -1] #1.31
w5 = [-1.5, 0.05, 0.13, 15, 1.5, -1] #39.42
w_p = [0, 0, 0, 1, 1, -0.6] #39.42

W = np.mat(w_p).T

Y_p = X_mat * W
sum_error = ((Y_mat - Y_p.T) * (Y_mat - Y_p.T).T) / 1000
print(sum_error)
'''











