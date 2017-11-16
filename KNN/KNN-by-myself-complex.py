#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/16 20:01
# @Author  : Panta Sun
# @Site    : 
# @File    : KNN-by-myself-complex.py
# @Software: PyCharm
"""KNN算法，自己实现，复杂版"""
import numpy as np
import operator
from os import listdir

def create_data_set():
    """产生测试数据"""
    group = np.array([[1.0, 1.1],
                     [1.0, 1.0],
                     [0.0, 0.0],
                     [0.0, 0.1]])
    labels = ['a', 'a', 'b', 'b']

    return group, labels


def knn_classify(inx, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(inx, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sum_diff_mat = sq_diff_mat.sum(axis=1)
    sqrt_mat = sum_diff_mat ** 0.5
    sorted_distance = sqrt_mat.argsort()
    class_dict = {}
    for i in range(k):
        label = labels[sorted_distance[i]]
        class_dict[label] = class_dict.get(label, 0) + 1
    sorted_class = sorted(class_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class[0][0]


def file2matrix(filename):
    """从文件读取数据"""
    with open(filename, 'r') as f:
        lines = f.readlines()
        number_lines = len(lines)
        return_mat = np.zeros((number_lines, 3))
        class_label = []
        index = 0
        for line in lines:
            line = line.strip()
            list_line = line.split('\t')
            return_mat[index, :] = list_line[:3]
            class_label.append(int(list_line[-1]))
            index += 1

        return return_mat, class_label


def auto_norm(data_set):
    """将数据归一化"""
    min_values = data_set.min(0)
    max_values = data_set.max(0)
    ranges = max_values - min_values
    norm_data_set = np.zeros(np.shape(data_set))
    m = norm_data_set.shape[0]
    norm_data_set = data_set - np.tile(min_values, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_values


def data_test():
    ho_ratio = 0.10
    data_set, labels = file2matrix("datingTestSet2.txt")
    norm_data_set, ranges, min_values = auto_norm(data_set)
    m = norm_data_set.shape[0]
    num_test_data = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_data):
        classify_result =\
            knn_classify(norm_data_set[i, :], norm_data_set[num_test_data:m, :], labels[num_test_data:m], 3)
        print("the classifier came back with: %d, the real answer is %d" % (classify_result, labels[i]))
        if classify_result != labels[i]:
            error_count += 1
    print("the total error rate is: %f" % ((error_count/float(num_test_data))*100))
    print(error_count, num_test_data)

def img2vector(filename):
    return_mat = np.zeros((1, 1024))
    with open(filename, 'r') as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                return_mat[0,32*i+j] = int(line[j])
    return return_mat

def hand_write_test():
    hwlabels = []
    traing_list = listdir('trainingDigits')
    m = len(traing_list)
    traing_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = traing_list[i]
        file_str = file_name_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        hwlabels.append(class_num)
        traing_mat[i, :] = img2vector('trainingDigits/'+file_name_str)

    test_list = listdir('testDigits')
    error_count = 0.0
    m_test = len(test_list)
    for i in range(m_test):
        file_name_str = test_list[i]
        file_str = file_name_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        vector_test = img2vector('testDigits/'+file_name_str)
        classifier_result = knn_classify(vector_test, traing_mat, hwlabels, 3)
        print("the classifier came back with: %d, the real answer is %d" % (classifier_result, class_num))
        if classifier_result != class_num:
            error_count += 1
    print("the total error rate is: %f" % (error_count/float(m_test)*100))

# knn_classify([0, 0], g, l, 3)


#data_test()
hand_write_test()
###################################################################
#################################
"""画图测试"""
# import matplotlib
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(g[:, 1], g[:, 2], 15.0*np.array(l), 15.0*np.array(l))
# plt.show()
#################################
###################################################################
# labels = ['a', 'a', 'b', 'b', 'c', 'c']
# group = np.array([[12.0, 12.1],
#                      [12.0, 13.0],
#                      [2.0, 2.0],
#                      [10.01, 10.01],
#                      [10.0001, 10.001]])
# min_v = group.min(0)
# print(min_v)
# atest = np.tile(min_v, (group.shape[0], 1))
# print(atest)
# print(group/atest)
# print(atest)
# atest = atest**2
# print(atest)
# atest = atest.sum(axis=1)
# print(atest)
# atest = atest**0.5
# print(atest)
# atestargsort = atest.argsort()
# print(atestargsort)
# dictt = {}
# for i in range(3):
#     label = labels[atestargsort[i]]
#     print(label)
#     dictt[label] = dictt.get(label, 0) + 1
#     # print(atestargsort[i], label)