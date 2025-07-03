#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : es_solver.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        : 
@version      : 0.1
@date         : 2022-03-17 20:04:06
@copyright    : Copyright (c) 2021-2022 PCNL EDA.
'''

from os import path
import sys

from utils.args import Args
from .es_engien import EvolutionStrategy
from .es_model import ESModel
import pickle
import copy

import numpy as np

from gym_env.place_env import PlaceEnv1


best_step = 50


class ESSolver():
    AGENT_HISTORY_LENGTH = 1
    REWARD_SCALE = 20
    SIGMA = 0.1
    LEARNING_RATE = 0.01
    MOVEABLE_ROW_NUM = 5

    def __init__(self, train_num, env: PlaceEnv1, POPULATION_SIZE=20, print_step=1, model_save_path=None, logger=None, saver=None):
        self.env = env
        self.train_num = train_num
        self.POPULATION_SIZE = POPULATION_SIZE
        self.print_step = print_step
        self.model_save_path = model_save_path if model_save_path else './model.pkl'
        self.logger = logger
        self.saver = saver
        self.es = EvolutionStrategy(self.get_reward, self.log_function, self.stop_function,
                                    self.saver,
                                    self.POPULATION_SIZE, self.SIGMA,
                                    self.LEARNING_RATE, num_threads=Args.num_threads)

    def solve(self):
        self.mini_cost = 1e6
        self.max_cost = 0

        self.best_solution = None
        self.best_traces = None
        self.best_reward = -1e6
        self.best_reward_step = 0
        # cell_num,  moveable_row_num, sigma
        self.model = ESModel(
            self.env.cell_num, moveable_row_num=self.MOVEABLE_ROW_NUM, sigma=self.SIGMA)
        self.es.reset()
        self.es.set_weights(self.model.get_weights())

        self.es.run(iterations=self.train_num)

        if self.saver:
            print('save final model!')
            self.saver.save(self.es.get_weights())

    def close(self):
        self.es.close()

    def log_function(self, iteration, weights, reward, rewards, time_duration):
        if self.logger:
            self.logger.update_data({'reward': reward,
                                     'rewards': rewards,
                                     'time': time_duration,
                                     'step': iteration})
            self.logger.display_info()
            self.logger.plot_figure()
            self.logger.save_solution(self)

    def stop_function(self, iteration, weights, reward, rewards, time_duration):
        if self.best_reward > reward:
            self.best_reward = reward
            self.best_reward_step = iteration
            return False
        if self.best_reward <= reward and (iteration - self.best_reward_step) >= best_step:
            return True

        return False

    def get_reward(self, weights):
        self.model.set_weights(weights)
        action = self.model.get_action()
        r = self.env.reset()
        _, reward = self.env.step_all(action)
        print(reward)
        return reward

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()
