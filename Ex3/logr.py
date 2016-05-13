#coding:utf-8

from __future__ import division
import numpy as np
import random
from math import exp, log
import time

# X = [[0.5, 0.1, 1], [0.1, 0.2 , 1]]
# Y = [[1], [-1]]

train_file = 'hw3_train.txt'
test_file = 'hw3_test.txt'

def create_muldata_fromfile(filename):

    f = open(filename)
    data = []
    for line in f:

        str_datas = line.split()
        float_datas = [float(i) for i in str_datas]

        data.append(float_datas)

    X = [d[:-1] + [1] for d in data]
    Y = [[d[-1]] for d in data]

    return X, Y

def theta_f(s):

   return  1 / (1 + exp(-s))

def train_logistic_model(X, Y, l_n = 0.01, is_sgd = False):

    now = time.time()

    X_mat = np.mat(X)
    Y_mat = np.mat(Y)

    d = len(X[0])
    N = len(X)

    w0 = [[0] for i in range(d)]
    W_mat = np.mat(w0)

    t = 0
    i_sgd = 0 #从0开始到len，然后再返回0，这样产生随机算法

    while t < 2000:

        temp_sum = np.mat([0 for i in range(d)]).T

        if is_sgd:

            if(t % 100 == 0): print 'Train time now : %d,  W_mat: %s, i_sgd： %d' % (t, W_mat[0, 0], i_sgd)

            degree = theta_f(- Y_mat[i_sgd] * W_mat.T * X_mat[i_sgd].T) * (-Y[i_sgd][0] * X_mat[i_sgd].T)
            i_sgd = i_sgd + 1 if i_sgd < N - 1 else 0

        else:

            if(t % 100 == 0): print 'Train time now : %d,  W_mat: %s' % (t, W_mat[0, 0])

            for i in range(0, N):

                temp_sum = temp_sum + theta_f(-Y_mat[i] * (W_mat.T * X_mat[i].T)) * (-Y[i][0] * X_mat[i].T)

                # print(theta_f(-Y_mat[i] * (W_mat.T * X_mat[i].T)))

            degree = temp_sum / N

        W_mat = W_mat -  l_n * degree

        t = t + 1

    print 'Training Cost : %f seconds' % (time.time() - now)

    return W_mat

def predict(W_mat, X, is_zero_one = True):

    X_mat = np.mat(X)
    Y_mat = W_mat.T * X_mat.T

    result = []
    for i in range(len(X)):

        result.append(theta_f(Y_mat[0, i]))

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
# Y = np.array([1, -1]).reshape(2, -1)

W = train_logistic_model(X, Y, 0.01, True)


X, Y = create_muldata_fromfile(test_file)

ra =  predict(W, X)

print(Y)
print(ra)

error = 0
for i in range(len(Y)):

    if Y[i][0] != ra[i]:

        error += 1

error_rate = error / len(Y) #0.001 0.230333333333 ， 0.01

print(error_rate)

'''
fix learning rate : n = 0.001 0.393333
fix learning rate : n = 0.01 0.220  cost:237s
sgd: n = 0.001 0.473
sgd: n = 0.01 0.205
'''

# ra = np.array(ra)
# print len(ra)
# ra = ra[ra > 0]
# print len(ra)

# X, Y = create_muldata_fromfile(train_file)
# X_mat = np.mat(X)
# Y_mat = np.mat(Y)
#
# d = len(X[0])
# N = len(X)
#
# w0 = [[0] for i in range(d)]
# W_mat = np.mat(w0)
# print(- Y_mat[0] * W_mat.T * X_mat[0].T)





