#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 8:57
# @Author  : zhuo_hf@foxmail.com
# @Site    : 
# @File    : 01_kNN.py
# @Software: PyCharm

#################################################
# sk-learn框架实现kNN:
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X_train = np.array([[3.39, 2.33],
                    [3.11, 1.78],
                    [1.34, 3.36],
                    [3.58, 4.68],
                    [2.28, 2.87],
                    [7.42, 4.70],
                    [5.75, 3.53],
                    [9.17, 2.51],
                    [7.79, 3.42],
                    [7.94, 0.79]])

y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

x = np.array([[8.09, 3.37]])

# kNN_classifier = KNeighborsClassifier(n_neighbors=6)
# kNN_classifier.fit(X_train, y_train)
#
# y_predict = kNN_classifier.predict(x)
# print(y_predict[0])


#################################################
# 手动实现kNN:

from math import sqrt
from collections import Counter


class KNNClassifier:

    def __init__(self, k):
        # 初始化kNN分类器
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X):
        y_predict = [self._predict(x) for x in X]
        return np.array(y_predict)

    def _predict(self, x):
        distance = [sqrt(np.sum((x_train-x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distance)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]


knn_clf = KNNClassifier(k=6)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(x)
print(y_predict)
