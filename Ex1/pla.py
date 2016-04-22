#coding:utf-8

__author__ = 'yanghan'

import numpy as np
import pandas as pd
import random, time

filename = '1_15_train.txt';

f_train_18 = '1_18_train.txt';
f_test_18 = '1_18_test.txt';

def get_data(file_name):

    data = pd.read_table(file_name, sep='\s', header=None, engine='python')
    train_data = np.zeros([1, data.shape[1]])

    for index, s in data.iterrows():

        train_data = np.vstack((train_data, s))

    train_data = train_data[1:]

    return train_data

def train_plc(train, target, alpha = 1.0):

    ones = np.ones(train.shape[0])
    train = np.c_[ones, train]   #x0 = 1, w0 = threshold = 0

    w = np.zeros([1, train.shape[1]])

    cycle = 0
    all_itered = False
    all_update_count= 0

    while not all_itered:

        is_update = False
        update_count = 0
        cycle += 1

        for index, row in enumerate(train):

            # print(index)
            dot_result = np.dot(w, row)
            if dot_result == 0:
                dot_result = -1

            y = target[index]

            if(y * dot_result <= 0):

                w = w + y * alpha * row
                is_update = True
                update_count += 1
                all_update_count += 1
                # print '第%d轮，在点%d:%s上调整[dot : %f, y : %d]，调整后的w为：%s' % (cycle + 1, index, row, dot_result, y, w)
                #
                # break

        if not is_update:
            all_itered = True

        # print '第%d轮，调整个数为:%d个' % (cycle, update_count)

    # print(all_update_count)

    return all_update_count

def pla_error_num(w, X, Y):

    num = 0

    for index, row in enumerate(X):

        dot_result = np.dot(w, row)

        if dot_result == 0:
            dot_result = -1

        y = Y[index]

        if(y * dot_result <= 0):

            num += 1

    # print(num)

    return num;

def train_pocket_pla(train_origin, iterate_bound, learn_rate = 1):

    ones = np.ones(train_origin.shape[0])
    train_origin = np.c_[ones, train_origin]   #x0 = 1, w0 = threshold = 0

    train = train_origin[:, :-1]
    target = train_origin[:, -1:]
    data_num, data_dim = train.shape

    p_w = np.zeros([1, train.shape[1]])
    w = np.zeros([1, train.shape[1]])

    iterate = 0
    ErrNum = train.shape[0] + 1

    while iterate < iterate_bound:

        row_index = random.randint(0, data_num - 1)

        row = train[row_index]
        dot_result = np.dot(w, row)

        if dot_result == 0:
            dot_result = -1

        y = target[row_index]

        #有错误就修正
        if(y * dot_result <= 0):

            iterate += 1

            w = w + y * learn_rate * row
            ErrNum_new = pla_error_num(w, train, target)

            if(ErrNum_new < ErrNum):

                ErrNum = ErrNum_new
                p_w = w


    return p_w

def PLA_Cycle(X, Y, eta, IsRandom):
    # X : input set of training data, Y : output set of..., eta : learning ratio (0 ~ 1)
    # IsRandom == False -> Naive Cycle,  IsRandom == True -> Random Cycle
    (dataNum, dataDim) = X.shape
    W = np.zeros(dataDim)
    permutation = range(0, dataNum)
    if IsRandom:
        random.shuffle(permutation)
    else:
        pass
    upDateTimes = 0
    lastUpDateIdx = 0;
    pmtIdx = 0
    dataIdx = 0
    iteCnt = 0
    halt = False
    while not halt:
        dataIdx = permutation[pmtIdx]
        dotProduct = np.dot(W, X[dataIdx])
        if dotProduct * Y[dataIdx] > 0:
            if dataIdx == lastUpDateIdx:
                halt = True
            else:
                pass
        else:
            # PLA update: W(t+1) = W(t) + eta*Y(n)*X(n)
            W += eta * Y[dataIdx] * X[dataIdx]
            upDateTimes += 1
            lastUpDateIdx = dataIdx
        pmtIdx = (pmtIdx + 1) % dataNum
        iteCnt += 1
        # print(iteCnt, W)
    print('upDateTimes: ', upDateTimes, '\n')
    return (W, upDateTimes)

# ============================= main ===============================





origin_train_data = get_data(filename)
cycles = []
count = 0

for i in range(1, 2000, 1):

    count += 1

    np.random.seed(i);
    train_data = np.random.permutation(origin_train_data);
    # train_data = origin_train_data

    train = train_data[:, :-1]
    target = train_data[:, -1:]
    cycle = train_plc(train, target, 1)
    cycles.append(cycle)

    if count % 100 == 0:
        print '实验数量：%d' % count

print(np.array(cycles).mean())

'''

train_18 = get_data(f_train_18)
train = train_18[:, :-1]
target = train_18[:, -1:]

test_18 = get_data(f_test_18)
test_train = test_18[:, :-1]
test_target = test_18[:, -1:]
test_data_num, test_data_dim = test_train.shape

iterate_bound = 1000
learn_rate = 1.0

result = []

seed = 32
np.random.seed(seed)

for experiment_time in range(0, 5):

    if experiment_time % 1 == 0:
        print '实验数量：%d' % experiment_time

    pocket_w = train_pocket_pla(train_18, iterate_bound = 500)

    result.append(pla_error_num(pocket_w, train, target) / float(train.shape[0]))

print(np.array(result).mean())



origin_train_data = get_data(filename)

ones = np.ones(origin_train_data.shape[0])
train_origin = np.c_[ones, origin_train_data]   #x0 = 1, w0 = threshold = 0

train = train_origin[:, :-1]
target = train_origin[:, -1:]

PLA_Cycle(train, target, 1, False)

'''

