#coding:utf-8

from __future__ import division
import random

__author__ = 'itensb'

def create_data(size = 1000):

    target_data = []
    for i in range(0, size):

        rd1 = random.randint(-1000, 1000) / 1000
        rd2 = random.randint(-1000, 1000) / 1000
        y = pow(rd1, 2) + pow(rd2, 2) - 0.6
        y = 1 if y > 0 else -1

        target_data.append([rd1, rd2, y])

    # print(target_data)

    noisy_data = []
    for data in target_data:

        is_noise = random.randint(1, 100) / 100.0
        data[-1] = -data[-1] if is_noise <= 0.1 else data[-1]
        noisy_data.append(data)

    # print(noisy_data)
    return noisy_data




