#coding:utf-8

from __future__ import division
import numpy as np
import random

__author__ = 'itensb'

def create_01data(size = 2):

    target_data = []
    for i in range(0, size):

        rd1 = random.randint(0, 1000) / 1000
        y = pow(rd1, 2)

        target_data.append([rd1, y])

    X = [[1, item[0]] for item in target_data]
    Y = [item[1]for item in target_data]
    return X,Y

def create_data(size = 1000, original = False, is_noisy = True):

    target_data = []
    for i in range(0, size):

        rd1 = random.randint(-1000, 1000) / 1000
        rd2 = random.randint(-1000, 1000) / 1000
        y = pow(rd1, 2) + pow(rd2, 2) - 0.6
        y = 1 if y > 0 else -1

        target_data.append([rd1, rd2, y])

    # print(target_data)



    if is_noisy:

        noisy_data = []
        for data in target_data:

            is_noise = random.randint(1, 100) / 100.0
            data[-1] = -data[-1] if is_noise <= 0.1 else data[-1]
            noisy_data.append(data)

        target_data = noisy_data

    if original:
        return target_data
    else:
        X = [[1, item[0], item[1]] for item in target_data]
        Y = [item[2]for item in target_data]
        return X,Y



def linear_model_hat(X):

    X = np.array(X)
    H = np.dot(X, np.dot(np.linalg.inv(np.dot(X.transpose() , X)), X.transpose()))

    return H

def linear_model_w(X, Y):

    X = np.array(X)

    revX = np.linalg.inv(np.dot(X.transpose() , X))
    X_star = np.dot(revX, X.transpose())

    print(X_star)
    print(Y)

    W = np.dot(X_star, Y)

    return W

def valide_ein(Y_p, Y):

    error = 0
    size = len(Y)

    for i in range(size):

        if Y_p[i] != Y[i]:

            error += 1

    return error / size

def experiment(T, size = 1000, type = 'ein'):

    all_error = 0
    for t in range(T):

        if(t % 100 == 0): print 'Experiment time now : %d' % (t)
        X, Y = create_data(size)

        if(type == 'ein'):

            H = linear_model_hat(X)
            Y_p = np.dot(H, Y)
            Y_p_sign = [1 if i > 0 else -1 for i in Y_p]

            # print(Y_p_sign)

        else:

            Y_p_sign = []
            for i in range(size):

                x1 = X[i][1]
                x2 = X[i][2]
                y = pow(x1, 2) + pow(x2, 2) - 0.6
                y = 1 if y > 0 else -1
                Y_p_sign.append(y)

        all_error += valide_ein(Y_p_sign, Y)
        # print(all_error)

    print  all_error / T

# experiment(1000, type='eout')





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

# W = np.array([0, 0])
# t = 1000
# for i in range(t):
#     X, Y = create_01data(2)
#     W = np.add(linear_model_w(X, Y), W)
#     print(W)
#
# print W / t

# X, Y = create_01data(2)
# print(X)
# print(Y)
X = [[1, 0.50001], [1, 0.5]]
Y = [np.math.pow(0.50001, 2), 0.25]
print linear_model_w(X, Y)

# [[ 1.41708543 -0.41708543]
#  [-2.51256281  2.51256281]]
# [0.027556000000000004, 0.31809599999999993]
# [-0.093624  0.73    ]









