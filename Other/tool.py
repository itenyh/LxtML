#coding:utf8

from __future__ import division
import math
import sys
import numpy as np

__author__ = 'yanghan'

def create_muldata_fromfile(filename):

    f = open(filename)
    data = []
    for line in f:

        str_datas = line.split()
        float_datas = [float(i) for i in str_datas]

        data.append(float_datas)

    X = [[1] + d[:-1] for d in data]
    Y = [[d[-1]] for d in data]

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def error_01(Yp, Y):

    error = 0
    for i in range(len(Y)):

        y = 1 if Yp[i] > 0 else -1
        if y != Y[i]:

            error += 1

    error_rate = error / len(Y)

    return error_rate

def sample_complexity(d_vc, e, c, n_diff_tolerance = 1):

    result = 0
    old_result = sys.maxsize

    while True:

        if(abs(old_result - result) < n_diff_tolerance):

            break

        else:

            old_result = result
            n = result

        result = 8 / (e ** 2) * math.log(((4 * ((2 * n) ** d_vc)) + 1) / c)

        print(result)

    return result


# print sample_complexity(10, 0.05, 0.05)


