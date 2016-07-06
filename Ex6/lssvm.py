from __future__ import division
import numpy as np
import copy
from Other.tool import error_01

train_data = 'hw2_lssvm_all.dat'

def create_muldata_fromfile(filename):

    f = open(filename)
    data = []
    for line in f:

        str_datas = line.split()
        float_datas = [float(i) for i in str_datas]

        data.append(float_datas)

    X = [[1] + d[:-1] for d in data]
    Y = [d[-1] for d in data]

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def k_rbf(x1, x2, gamma): #rbf kernel

    tempx_x = x1 - x2

    k = np.math.exp(- gamma * np.dot(tempx_x , tempx_x))

    return k

def get_beta(X, Y, gamma, ru):

    N = len(X)
    d = len(X[0])

    X = np.array(X)
    Y = np.array(Y)

    K = []
    for i in range(N):

        x1 = X[i]

        k_row = []

        for j in range(N):

            x2 = X[j]

            k = k_rbf(x1, x2, gamma)

            k_row.append(k)

        K.append(k_row)

    beta = np.dot(np.linalg.inv(ru * np.eye(N) + K) , Y)

    return beta

def predict_by_X(beta, X, train_X, gamma):

    X = np.array(X)

    r = []
    for x_new in X:

        k_v = [k_rbf(x_new, xn, gamma) for xn in train_X]

        # g(x)= w` * Φ(x) = (beta * Φ(X')) * Φ(x) =  beta * (Φ(X') * Φ(x)) = beta * k(x`i, x)
        # (x`, X`代表原数据; Φ(X')代表N * d~ 矩阵; k(x`i, x)代表N * 1向量)
        p_y = np.dot(k_v, beta)

        r.append(p_y)

    return r

def ex_19_20():

    X, Y = create_muldata_fromfile(train_data)

    X_train = X[:400]
    Y_train = Y[:400]
    X_test = X[400:]
    Y_test = Y[400:]

    gammas = [32, 2, 0.125]
    rus = [0.001,1,1000]

    for gamma in gammas:

        for ru in rus:

            beta = get_beta(X_train, Y_train, gamma, ru)

            # X_new = [[4, 6], [3, 5]]

            # print 'Beta get'

            r = predict_by_X(beta, X_test, X_train, gamma)

            print 'gamma|ru|error %f|%f|%f' % (gamma, ru, error_01(r, Y_test))


data = [[1, 2, 1], [2, 4, -1]]
X = [dd[:-1] for dd in data]
Y = [dd[-1] for dd in data]

ex_19_20()




