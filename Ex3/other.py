#coding:utf-8

from __future__ import division
import numpy as np
from math import exp, log

__author__ = 'yanghan'

def fun_c(u, v):

    return exp(u) + exp(2 * v) + exp(u * v) + u ** 2 + 2 * u * v + 2 * (v ** 2) - 3 * u - 2 * v

def gradient(u, v):

    aEu = exp(u) + exp(u * v) * v + 2 * u - 2 * v - 3
    aEv = exp(2 * v) * 2 + exp(u * v) * u - 2 * u + 4 * v - 2

    return  - np.mat([aEu, aEv]).T

def gradient_step(gradient_fuc, u, v, step, n = 0.01):

    for i in range(step):

        g_uv = gradient_fuc(u, v)
        u = u + n * g_uv[0, 0]
        v = v + n * g_uv[1, 0]

        print 'Round %i -> %f' % (i + 1, fun_c(u, v))

# gradient_step(0, 0, 6)

def hessian_mat(u, v):

    H_mat = [[exp(u) + v ** 2 * exp(u * v) + 2, exp(u * v) + v * u * exp(u * v) - 2],
             [exp(u * v) + v * u * exp(u * v) - 2, 4 * exp(2 * v) + u ** 2 * exp(u * v) + 4]]

    return np.mat(H_mat)

def newton_direction(u, v):

    return hessian_mat(u, v).I * gradient(u, v)

print(gradient_step(newton_direction, 0, 0, 6, 1))