#coding:utf-8

from __future__ import division
import numpy as np
import time
from math import exp, log
from Other.tool import create_muldata_fromfile, error_01

train_file = 'hw3_train.txt'
test_file = 'hw3_test.txt'

def theta(s):

    return 1 / (1 + exp(-s))

def train_logistic_model(X, Y, l_n = 0.01, is_sgd = False):

    now = time.time()

    N, d = X.shape
    W = np.zeros(d)
    i_sgd = 0 #从0开始到len，然后再返回0，这样产生随机算法

    for t in range(2000):

        if(t % 100 == 0): print 'Train time now : %d,  W00: %s' % (t, W[0])

        temp = np.zeros(d)

        if is_sgd:

            temp += theta(-Y[i_sgd] * np.dot(W, X[i_sgd])) * (-Y[i_sgd] * X[i_sgd])
            i_sgd = i_sgd + 1 if i_sgd < N - 1 else 0

        else:

            for i in range(N):

                temp += theta(-Y[i] * np.dot(W, X[i])) * (-Y[i] * X[i])

                # print(temp)

            temp /= N

        W = W - l_n * temp

    print 'Training Cost : %f seconds' % (time.time() - now)

    return W

def predict(W, X, is_zero_one = True):

    Y = np.dot(X, W)

    result = []
    for i in range(len(X)):

        result.append(theta(Y[i]))

    if is_zero_one:

        zo_result = []

        for r in result:

            if r - 0.5 > 0:

                zo_result.append(1)

            else:

                zo_result.append(-1)

        result = zo_result

    return result

X, Y = create_muldata_fromfile(train_file)

# X = np.array([[1, 1, 0], [1, 0, 1]])
# Y = np.array([1, -1])

W = train_logistic_model(X, Y, l_n = 0.01, is_sgd=True)

# print(W)

X, Y = create_muldata_fromfile(test_file)

ra = predict(W, X)

print(error_01(ra, Y))

'''
fix learning rate : n = 0.001 0.475
fix learning rate : n = 0.01 0.22 cost:68s
'''