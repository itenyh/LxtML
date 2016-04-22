#coding:utf8

from __future__ import division
import random


__author__ = 'HanHan'

train_file = 'hw2_train.txt'
test_file = 'hw2_test.txt'

def create_muldata_fromfile(filename):

    f = open(filename)
    data = []
    for line in f:

        str_datas = line.split()
        float_datas = [float(i) for i in str_datas]

        data.append(float_datas)

    f.close()

    train_data = []
    for i in range(0, len(data[0]) - 1):
        train_data.append([])

    for index, item in enumerate(data):

        y = item[-1]

        for i in range(0, len(item) - 1):

            the_list = train_data[i]
            the_list.append((item[i], y))

    return train_data

def create_data(size = 20):


    X = []
    Y = []

    for i in range(0, size):

        x = random.randint(-100, 100) / 100.0
        y = 1 if x > 0 else -1

        is_noise = random.randint(1, 100) / 100.0
        y = -y if is_noise <= 0.2 else y

        X.append(x)
        Y.append(y)

    data = zip(X, Y)
    data = sorted(data)

    return data

def decision_stump_alg(data):

    result = []

    for index, (x_i, y_i) in enumerate(data):

        theta = x_i - 0.00001

        errors1 = []
        errors2 = []

        for index, (x_i, y_i) in enumerate(data):

            the_y1 = 1 if (x_i - theta) > 0 else -1
            the_y2 = 1 if -(x_i - theta) > 0 else -1
            error1 = 0 if the_y1 == y_i else 1
            error2 = 0 if the_y2 == y_i else 1
            errors1.append(error1)
            errors2.append(error2)

        ein1 = sum(errors1) / len(errors1)
        ein2 = sum(errors2) / len(errors2)

        eout1 = 0.5 + 0.3 * 1 * (abs(theta) - 1)
        eout2 = 0.5 + 0.3 * -1 * (abs(theta) - 1)

        # print(ein1)

        result.append((ein1, eout1, theta, 1))
        result.append((ein2, eout2, theta, -1))


    return sorted(result)[0]

# print decision_stump_alg(create_data(20))

def mul_decision_stump_alg(data):

    bests = []
    dim = 0   #记录下所在维度

    for d in data:

        best = decision_stump_alg(d)
        best = list(best)
        best.append(dim)
        bests.append(tuple(best))
        dim += 1

    return sorted(bests)[0]  #ein, eout, theta, s, dim


# print(mul_decision_stump_alg(create_muldata_fromfile(train_file)))

data = create_muldata_fromfile(test_file)

test_data = data[3]
test_data = sorted(test_data)

errorss = []
for index, (x_i, y_i) in enumerate(test_data):

    theta = 1.77399

    errors = []

    the_y = 1 if -(x_i - theta) > 0 else -1
    error = 0 if the_y == y_i else 1
    errorss.append(error)


etest = sum(errorss) / len(errorss)
print(etest)



'''
avg_ein = 0
avg_eout = 0

for i in range(0, 5000):

    result = decision_stump_alg(create_data())
    avg_ein += result[0]
    avg_eout += result[1]

print(avg_ein / 5000)
print(avg_eout / 5000)
'''

