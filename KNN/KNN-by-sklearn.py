#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/14 15:33
# @Author  : Panta Sun
# @Site    : 
# @File    : KNN-by-sklearn.py
# @Software: PyCharm
"""KNN算法，通过sklearn库实现"""
from sklearn import neighbors, datasets
knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()
print(iris)
knn.fit(iris['data'], iris['target'])

predictY = knn.predict([[5.9,  3.,  5.1, 1.8]])
print(predictY)
