# from  __future__ import division
from svmutil import *
import numpy as np
import pandas as pd


train_file = 'hw5_train.txt'
test_file = 'hw5_test.txt'

def get_X_Y_by_data(data):

    X = []
    Y = []

    for i in range(len(data)):

        item = data[i]

        X.append(item[:-1])
        Y.append(item[-1:][0])

    return X, Y


def create_muldata_fromfile(filename, sig = 0):

    f = open(filename)
    data = []
    for line in f:

        str_datas = line.split()
        float_datas = [float(i) for i in str_datas]

        data.append(float_datas)

    X = [d[1:] for d in data]
    Y = [1 if d[0] == sig else 0 for d in data]

    return X, Y

def ex_3():

    X = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
    Y = [-1, -1, -1, 1, 1, 1, 1]

    para = '-s 0 -t 1 -c 1000 -g 1 -d 2 -r 1 -h 0 -e 0.00001'
    model = svm_train(Y, X, para)

    svs = model.get_SV()
    sv_coeffs = model.get_sv_coef()

    print(svs)
    print(sv_coeffs)

    svs_np = np.array([[np.math.sqrt(2) * -1, np.math.sqrt(2) * 0, 0, 1, 0], [np.math.sqrt(2) * 0, np.math.sqrt(2) * 2, 0, 0, 4],
                       [np.math.sqrt(2) * 0, np.math.sqrt(2) * -2, 0, 0, 4], [np.math.sqrt(2) * 0, np.math.sqrt(2) * 1, 0, 0, 1],
                       [np.math.sqrt(2) * 0, np.math.sqrt(2) * -1, 0 , 0, 1]])
    sv_cof_np = np.array([item[0] for item in sv_coeffs])
    print(svs_np)
    print(np.dot(sv_cof_np, svs_np))
    # print(2 * 0.15 + -2 * 0.37 + -0.49 + 0.92)

def ex_15_16():

    '''
    Now i = 0 ==========================================
    Cross Validation Accuracy = 83.6236%
    Now i = 1 ==========================================
    Cross Validation Accuracy = 98.4227%
    Now i = 2 ==========================================
    Cross Validation Accuracy = 89.9739%
    Now i = 3 ==========================================
    Cross Validation Accuracy = 90.9752%
    Now i = 4 ==========================================
    Cross Validation Accuracy = 91.0575%
    Now i = 5 ==========================================
    Cross Validation Accuracy = 92.3742%
    Now i = 6 ==========================================
    Cross Validation Accuracy = 90.8929%
    Now i = 7 ==========================================
    Cross Validation Accuracy = 91.1535%
    Now i = 8 ==========================================
    Cross Validation Accuracy = 92.5662%
    Now i = 9 ==========================================
    Cross Validation Accuracy = 91.1672%
    '''


    for i in range(10):

        X, Y = create_muldata_fromfile(train_file, i)

        print 'Now i = %d ==========================================' % i
        model = svm_train(Y, X, '-s 0 -t 1 -c 0.01 -d 2 -r 1 -g 1')
        sv_coeffs = model.get_sv_coef()
        svs = model.get_SV()
        sv_cof_np = [abs(item[0]) for item in sv_coeffs]
        print sum(sv_cof_np)
        print(svs)

def ex_19():

    x, y = create_muldata_fromfile(test_file, 0)
    X, Y = create_muldata_fromfile(train_file, 0)

    for i in [1, 10, 100, 1000, 10000]:


        print 'Now gamma = %d ==========================================' % i

        para = '-s 0 -t 2 -c 0.1 -g %d' % i
        model = svm_train(Y, X, para)
        p_label, p_acc, p_val = svm_predict(y, x, model, '-b 0')
        # ACC, MSE, SCC = evaluations(y, p_label)

        # print ACC

def ex_20():

    gamma = [1, 10, 100, 1000, 10000]
    X, Y = create_muldata_fromfile(train_file, 0)
    record = {}

    train_data = []
    for i in range(len(X)):

        data = X[i]
        data.append(Y[i])
        train_data.append(data)

    for experiment in range(30):

        print 'Now experiment Time : %d' % (experiment + 1)

        np.random.shuffle(train_data)
        test, training = train_data[:1000], train_data[1000:]

        X_train, Y_train = get_X_Y_by_data(training)
        X_test, Y_test = get_X_Y_by_data(test)

        min_acc = -1
        min_index = -1
        for i in range(len(gamma)):

            g = gamma[i]
            para = '-s 0 -t 2 -c 0.1 -g %d' % g
            model = svm_train(Y_train, X_train, para)
            p_label, p_acc, p_val = svm_predict(Y_test, X_test, model, '-b 0')

            if(p_acc[0] > min_acc):

                min_acc = p_acc[0]
                min_index = i

        key = '%d' % min_index
        if not record.has_key(key):

            record[key] = 1

        else:

            record[key] += 1


    print(record)

def get_data(file_name):

    data = pd.read_table(file_name, sep='\s', header=None, engine='python')
    train_data = np.zeros([1, data.shape[1]])

    for index, s in data.iterrows():

        train_data = np.vstack((train_data, s))

    train_data = train_data[1:]

    return train_data

# ex_3()

with open('train.csv') as f:

    for line in f:

        print line

'''
X, Y = create_muldata_fromfile(train_file)
model = svm_train(Y, X, '-s 0 -t 0 -c 0.01')
svs = model.get_SV()
sv_coeffs = model.get_sv_coef()

svs_np = np.array([[item[1], item[2]] for item in svs])
sv_cof_np = np.array([item[0] for item in sv_coeffs])
print(svs_np.shape)
print(sv_cof_np.shape)
print(np.dot(sv_cof_np, svs_np))

aa = np.dot(sv_cof_np, svs_np)
bb = np.math.sqrt(np.dot(aa, aa))
print(bb)
'''