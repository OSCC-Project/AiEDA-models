#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : es_model.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        :
@version      : 0.1
@date         : 2022-03-17 20:03:04
@copyright    : Copyright (c) 2021-2022 PCNL EDA.
'''

from turtle import shape
import numpy as np
from gym import spaces
import torch
import copy


def sigmoid(x):
    x = np.asarray(x)
    return 1/(1 + np.exp(-x))


'''
moveable_row_num : 3, 5, 7
'''


class ESModel(object):
    def __init__(self, cell_num,  moveable_row_num, sigma):
        self.cell_num = cell_num
        self.moveable_row_num = moveable_row_num
        self.weights = np.zeros(shape=cell_num)
        self.section_point = [-3*sigma + i * (6*sigma / moveable_row_num)
                              for i in range(moveable_row_num + 1)]
        self.sections = [(self.section_point[i], self.section_point[i+1], i - (moveable_row_num-1)//2)
                         for i in range(moveable_row_num)]

    def get_action(self):
        actions = copy.deepcopy(self.weights)
        actions = np.asarray(actions)
        bound = (self.moveable_row_num + 1) / 2
        actions[actions < self.sections[0][0]] = - bound
        actions[actions >= self.sections[len(self.sections)-1][1]] = bound
        for (l, r, act) in self.sections:
            actions[(actions >= l) & (actions < r)] = act
        return actions

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
