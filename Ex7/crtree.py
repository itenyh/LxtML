#coding:utf8

from __future__ import division
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from uuid import uuid1
from Other.tool import error_01
import time

# np.random.seed(time.time())

class rf_tree:

    __n_estimator = -1
    __rf_tree = []

    def __init__(self, n_estimator):

        self.__n_estimator = n_estimator

    def bag(self, df):

        N = len(df)
        df_values = df.values
        bag = np.random.random_integers(0, N - 1, N)
        bag_result_value = [df_values[x] for x in bag]
        return pd.DataFrame(bag_result_value)

    def fit(self, df):

        for i in range(0, self.__n_estimator):

            print 'Now training tree : %d' % i

            bagged_df = self.bag(df)
            crtree = crt_tree()
            crtree.fit(bagged_df, -1)
            self.__rf_tree.append(crtree)

    # def predict(self, X):
    #
    #     for i in range(0, self.__n_estimator):
    #
    #         cur_crtree = self.__rf_tree[i]
    #         y_p = cur_crtree.predict(X)
    #         error_rate = error_01(y_p, )

    def avg_error_pertree(self, X, y):

        all_error = 0

        for i in range(0, self.__n_estimator):

            cur_crtree = self.__rf_tree[i]
            y_p = cur_crtree.predict(X)
            all_error += error_01(y_p, y)

        return all_error / self.__n_estimator


class crt_tree:

    __tree = {}

    def sk_learn(self):

        X = all[[0,1]]
        y = all[2]

        X_t = test[[0,1]]
        y_t = test[2]

        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        clf.fit(X, y)

        score = cross_val_score(clf, X_t, y_t, cv=5, scoring='accuracy').mean()
        print(score)

    def gini(self, df):

        N = len(df)
        N_1 = len(df[df[2] == 1])
        N_n1 = N - N_1
        gini = 1 - ((N_1 / N) ** 2 + (N_n1 / N) ** 2)

        return gini

    #node structure:
    #non-leaf => [fea, index, l_index, r_index]
    #leaf => [result]
    def fit(self, df, nid = -1):

        #检查停止条件
        label_sum = df[2].sum()
        if np.abs(label_sum) == len(df):

            if label_sum > 0:
                self.__tree[nid] = [1]
            else:
                self.__tree[nid] = [-1]
            return

        N, dim = df.shape
        node = []
        criterial = 999

        for fea in range(0, dim - 1):

            split_index = np.argsort(df[fea].values)
            split_index = df.iloc[split_index].index

            #j代表在左边共有几个，最多有N-1个
            #node的值，代表小于等于[fea, index]这个元素的所有数据
            for i, index in enumerate(split_index):

                j = i + 1

                l_index = split_index[:j]
                r_index = split_index[j:]

                if(len(r_index) == 0): continue

                l_df = df.loc[l_index]
                r_df = df.loc[r_index]

                l_gini = self.gini(l_df)
                r_gini = self.gini(r_df)

                l_gini_w = (len(l_df)/N) * l_gini
                r_gini_w = (len(r_df)/N) * r_gini

                cur_criterial = l_gini_w + r_gini_w

                if cur_criterial < criterial:

                    criterial = cur_criterial
                    node = [fea, df.loc[index][fea]]

        l_tree = df[df[node[0]] <= node[1]]
        r_tree = df[df[node[0]] > node[1]]

        l_id = str(uuid1())
        r_id = str(uuid1())
        node.extend([l_id, r_id])
        self.__tree[nid] = node

        self.fit(l_tree, l_id)
        self.fit(r_tree, r_id)

    def print_tree(self):

        print(self.__tree)

    #x:[fea1, fea2]
    def __predict_x(self, x, nid):

        cur_node = self.__tree[nid]

        if len(cur_node) == 1:

            return cur_node[0]

        else:

            if x[cur_node[0]] <= cur_node[1]:

                return self.__predict_x(x, cur_node[2])

            else:

                return self.__predict_x(x, cur_node[3])

    def predict(self, X):

        return [self.__predict_x(x, -1) for x in X]


df = pd.read_table('hw7_train.dat', sep='\s+', header=None)
test =  pd.read_table('hw7_test.dat', sep='\s+', header=None)
# df = pd.DataFrame(df)
# clf = crt_tree()
# clf.fit(df)

# X = df[[0, 1]].values
# Y = df[2].values
#
# X_t = test[[0, 1]].values
# Y_t = test[2].values
#
# Y_p = clf.predict(X_t)
# print error_01(Y_p, Y_t)

# df = {0:[3, 2, 1, 4, 5],1:[98,34,73,43,23],2:[1,-1,1,-1,-1]}
# df = pd.DataFrame(df)
# N = len(df)
#
# df_values = df.values
# bag = np.random.random_integers(0, N - 1, N)
# print(bag)
# bag_result_value = [df_values[x] for x in bag]
# print pd.DataFrame(bag_result_value)

rf = rf_tree(n_estimator=300)
rf.fit(df)
print rf.avg_error_pertree(df[[0, 1]].values, df[2].values)