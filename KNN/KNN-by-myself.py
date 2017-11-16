#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/14 15:54
# @Author  : Panta Sun
# @Site    : 
# @File    : KNN-by-myself.py
# @Software: PyCharm
"""KNN算法，自己实现，简单版"""
import csv
import random
import math
import operator


class KNN:

    def __init__(self, k=3):
        """初始化"""
        self.k = k
        self.training_set = []
        self.test_set = []
        self.predictions = []

    def load_data_sets(self, filename, split):
        """读取数据，并将一部分数据拿出来当作测试数据"""
        with open(filename, 'r') as csvfile:
            lines = csv.reader(csvfile)
            data_list = list(lines)
            for data in data_list:
                for j in range(4):
                    data[j] = float(data[j])
                if random.random() < split:
                    self.training_set.append(data)
                else:
                    self.test_set.append(data)

    def euclidean_distance(self, instance1, instance2, dimension):
        """计算训练集中的数据到测试数据的距离"""
        distance = 0
        for x in range(dimension):
            distance += pow((instance1[x]-instance2[x]), 2)
        return math.sqrt(distance)

    def get_neighbors(self, test_instance):
        """获取到某一测试数据的最近的k个邻居"""
        distances = []
        neighbors = []
        dimension = len(test_instance)-1
        for td in self.training_set:
            dis = self.euclidean_distance(td, test_instance, dimension)
            distances.append((td, dis))
        distances.sort(key=operator.itemgetter(1))
        for k in range(self.k):
            neighbors.append(distances[k][0])
        return neighbors

    def get_response(self, neighbors):
        """根据k个最近邻居的类型，按照少数服从多数的原则进行预测"""
        class_vote = {}
        for n in neighbors:
            response = n[-1]
            if response in class_vote:
                class_vote[response] += 1
            else:
                class_vote[response] = 1
        sorted_vote = sorted(class_vote.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_vote[0][0]

    def get_accuracy(self, predictions):
        """计算正确率"""
        correct = 0
        L = len(self.test_set)
        for x in range(L):
            if self.test_set[x][-1] == predictions[x]:
                correct += 1
        return correct/float(L) * 100.0

    def predict(self, test):

        neighbors = self.get_neighbors(test)
        result = self.get_response(neighbors)
        print("预测：" + result)

    def run(self):
        """主函数"""
        self.load_data_sets()
        for test in self.test_set:
            neighbors = self.get_neighbors(test)
            result = self.get_response(neighbors)
            self.predictions.append(result)
            print("预测："+result, "真实值："+test[-1])
        accuracy = self.get_accuracy(self.predictions)
        print("准确率："+str(accuracy)+"%")


knn = KNN()
knn.load_data_sets("irisdata.txt", 1)
knn.predict([5.7,2.8,4.5,1.3])
knn.predict([6.9,3.2,5.7,2.3])

