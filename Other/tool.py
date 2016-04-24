#coding:utf8

import math
import sys

__author__ = 'yanghan'


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


print sample_complexity(10, 0.05, 0.05)


