#coding:utf8

from __future__ import division
import random


__author__ = 'yanghan'


X = []
Y = []

for i in range(0, 20):

    x = random.randint(-100, 100) / 100.0
    y = 1 if x > 0 else -1

    is_noise = random.randint(1, 100) / 100.0
    y = -y if is_noise <= 0.2 else y

    X.append(x)
    Y.append(y)

data = zip(X, Y)
data = sorted(data)

result = []

for index, x_i in enumerate(X):

    theta = x_i - 0.00001

    errors1 = []
    errors2 = []

    for index, x_i in enumerate(X):

        the_y1 = 1 if (x_i - theta) > 0 else -1
        the_y2 = 1 if -(x_i - theta) > 0 else -1
        error1 = 0 if the_y1 == Y[index] else 1
        error2 = 0 if the_y2 == Y[index] else 1
        errors1.append(error1)
        errors2.append(error2)

    ein1 = sum(errors1) / len(errors1)
    ein2 = sum(errors2) / len(errors2)

    # print(ein1)

    result.append((ein1, theta))
    result.append((ein2, theta))

# print(sorted(result))

all_e = 0
for item in result:

    all_e += item[0]

print(all_e)

'''
theta = X[0] - 0.00001
s = 1

errors = []
for index, x_i in enumerate(X):

    the_y = 1 if s * (x_i - theta) > 0 else -1
    error = 0 if the_y == Y[index] else 1
    errors.append(error)

print(X)
print(Y)
print(errors)
print(sum(errors) / len(errors))
'''