#coding:utf-8

from __future__ import division
import numpy as np
import time
from math import exp, log
from Other.tool import create_muldata_fromfile, error_01

train_file = 'hw4_train.dat'
test_file = 'hw4_test.dat'

# create_muldata_fromfile(train_file)

def train_ridge_regression_model(X, Y, fi = 10):

    N, d = X.shape

    # temp = np.dot(X.transpose(), X)

    I = np.identity(d) #全1矩阵表现更好!?

    W = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X) + np.dot(fi , I)), X.transpose()), Y)

    return W

train_num = 120

def error_with_fi(fi):

    X, Y = create_muldata_fromfile(train_file)

    W = train_ridge_regression_model(X, Y, fi)
    # print(X.shape, W.shape)
    Y_p = np.dot(X, W).transpose()[0]

    print error_01(Y_p, Y)

    X, Y = create_muldata_fromfile(test_file)

    Y_p = np.dot(X, W).transpose()[0]

    print error_01(Y_p, Y)

def experiment14_17():

    for fi in range(2 , -11, -1):

        X, Y = create_muldata_fromfile(train_file)
        X_train = X[:train_num]
        Y_train = Y[:train_num]
        X_val = X[train_num:]
        Y_val = Y[train_num:]

        W = train_ridge_regression_model(X_train, Y_train, 10 ** fi)
        Y_p = np.dot(X_train, W).transpose()[0]

        etrain = error_01(Y_p, Y_train)

        Y_p = np.dot(X_val, W).transpose()[0]
        eval = error_01(Y_p, Y_val)

        X, Y = create_muldata_fromfile(test_file)
        Y_p = np.dot(X, W).transpose()[0]

        eout_ = error_01(Y_p, Y)

        fi_ex14 = [-4, -6, -2, -10, -8]
        fi_ex15 = [-9, -1, -5, -7, -3]
        fi_ex16 = [-6, -8, -4, -2, 0]
        fi_ex17 = [-3, -8, -6, -9, 0]

        if fi in fi_ex17:
            print 'fi : %d, errortrain -> %f errorval -> %f errorout -> %f' % (fi, etrain, eval, eout_)

cv_num = 5
X, Y = create_muldata_fromfile(train_file)
group_num = len(X) / cv_num

for fi in [-4, 0, -2, -6, -8]:

    all_error = 0

    for i in range(cv_num):

        X_train = np.vstack((X[0: i * group_num], X[(i + 1) * group_num:]))
        Y_train = np.vstack((Y[0: i * group_num], Y[(i + 1) * group_num:]))
        X_val = X[i * group_num:(i + 1) * group_num]
        Y_val = Y[i * group_num:(i + 1) * group_num]

        W = train_ridge_regression_model(X_train, Y_train, 10 ** fi)
        Y_p = np.dot(X_val, W).transpose()[0]
        ecv = error_01(Y_p, Y_val)
        all_error += ecv

    print 'fi : %d, errorcv -> %f ' % (fi, all_error / cv_num)


error_with_fi(1)
