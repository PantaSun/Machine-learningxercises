#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/20 22:37
# @Author  : Panta Sun
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import matplotlib.pyplot as plt
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.axl.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt, textcoords='axes fraction',
                             va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.axl = plt.subplot(111, frameon=False)
    plot_node(u'决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node(u'叶子节点', (0.8, 0.1), (0.3, 0.8),leaf_node)
    plt.show()


create_plot()