#coding:utf-8

from __future__ import division
import numpy as np
import copy
from Other.tool import error_01

train_file = 'hw2_adaboost_train.txt'
test_file = 'hw2_adaboost_test.txt'

def get_datas(filename):

    f = open(filename)
    data = []
    for line in f:

        str_datas = line.split()
        float_datas = [float(i) for i in str_datas]

        data.append(float_datas)

    return data

def stump_tree(original_data, u = None):

    datas = copy.deepcopy(original_data)

    d = len(datas[0]) - 1
    N = len(datas)

    if u == None:

        u = [1] * N

    min_ein = 1.1
    min_theta = 0
    min_s = 0
    min_i = 0
    min_error_flag_dic = {}

    # [[x, y, u, index], ...... ]
    for index, dd in enumerate(datas):
        dd.append(u[index])
        dd.append(index)

    for i in range(d):

        datas.sort(key=lambda x:x[i])

        X = [dd[:-3] for dd in datas]
        Y = [dd[d] for dd in datas]
        uu = [dd[d + 1] for dd in datas]
        ii = [dd[d + 2] for dd in datas]

        mid_point = [(X[index][i] + X[index + 1][i]) / 2 for index in range(N - 1)]
        mid_point.insert(0, X[0][i] - 1)
        # print mid_point

        s_para = [1, -1]
        for s in s_para:

            for theta in mid_point:

                y_ps = []
                data_left = N
                for x in X:

                    if (x[i] - theta) < 0:

                        data_left -= 1
                        y_p = -1 * s
                        y_ps.append(y_p)

                    else:

                        break

                y_ps.extend([1 * s] * data_left)

                # ein = error_01(Y, y_ps)
                error = 0
                error_flag_dic = {}
                for j in range(len(Y)):

                    if y_ps[j] != Y[j]:

                        error_flag_dic[ii[j]] = 1
                        error += 1 * uu[j]

                    else:

                        error_flag_dic[ii[j]] = 0

                ein = error / sum(u)
                # print(ein)
                if ein < min_ein:

                    min_ein = ein
                    min_i = i
                    min_theta = theta
                    min_s = s
                    min_error_flag_dic = error_flag_dic

                # print "theta|s|i  %f|%d|%d => %f" % (theta, s, i, ein)

    # print "The Best : theta|s|i  %f|%d|%d => %f" % (min_theta, min_s, min_i, min_ein)

    return min_theta, min_s, min_i, min_ein, min_error_flag_dic

def train_stump_tree_adaboost(datas, iter_time):

    N = len(datas)
    u = [1 / N] * N

    models = []
    for T in range(iter_time):

        min_theta, min_s, min_i, min_ein, min_error_flag_dic = stump_tree(datas, u)

        # min_ein = 0.000001 if min_ein == 0 else min_ein
        u_new = [0] * N
        t = np.math.sqrt((1 - min_ein) / min_ein)

        for item in min_error_flag_dic.items():

            if item[1] != 1:

                u_new[item[0]] = u[item[0]] / t

            else:

                u_new[item[0]] = u[item[0]] * t

        u = u_new

        models.append((min_theta, min_s, min_i, np.math.log(t), min_ein))

        print(T, min_i, min_theta, min_s, min_ein, np.math.log(t))
    # print(min_ein, t, u)

    # min_theta, min_s, min_i, min_ein, min_error_flag_dic = stump_tree(datas, u_new)

    return models

def adaboost_model_predict(models, X):

    p_ys = []

    for x in X:

        score = 0
        for theta, s, i, w, e in models:

            score += ((x[i] - theta) * s) * w

        result = 1 if score > 0 else -1
        p_ys.append(result)

    return p_ys

datas = get_datas(train_file)
X = [dd[:-1] for dd in datas]
Y = [dd[-1] for dd in datas]
models = train_stump_tree_adaboost(datas, 300)

# models.sort(key=lambda x:x[-1])
# print(models)

yps = adaboost_model_predict(models, X)
print error_01(Y, yps)  #ein

datas = get_datas(test_file)
X = [dd[:-1] for dd in datas]
Y = [dd[-1] for dd in datas]
yps = adaboost_model_predict(models, X)
print error_01(Y, yps)  #eout

# X = [[1, 25], [0, 3], [2, 5]]
# Y = [1, 1, -1]

# datas = [[1, 25, 1], [0, 3, 1], [2, 5, -1]]



