#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : es_engien.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        : 
@version      : 0.1
@date         : 2022-03-17 20:03:57
@copyright    : Copyright (c) 2021-2022 PCNL EDA.
'''
import time
import multiprocessing as mp

import numpy as np

np.random.seed(0)


def worker_process(arg):
    get_reward_func, weights = arg
    return get_reward_func(weights)


class EvolutionStrategy(object):
    def __init__(self, get_reward_func, log_function=None, stop_function=None, saver=None, population_size=50,
                 sigma=0.1,
                 learning_rate=0.03, decay=0.999,
                 num_threads=-1):

        self.weights = None
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.log_function = log_function
        self.stop_function = stop_function
        self.num_threads = min(
            mp.cpu_count(), 8) if num_threads == -1 else num_threads
        self.saver = saver

    def reset(self):
        self.best_reward = -1e6
        self.best_weight = None

    def set_weights(self, weights):
        self.weights = weights

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            weights_try.append(w[index] + jittered)
        return weights_try

    def get_weights(self):
        return self.weights

    def _get_population(self):
        population = []

        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
            population.append(x)
        return population

    def _get_rewards(self, pool, population):

        if pool is not None:
            worker_args = ((self.get_reward, self._get_weights_try(
                self.weights, p)) for p in population)
            rewards = pool.map(worker_process, worker_args)

        else:
            rewards = []
            for p in population:
                weights_try = self._get_weights_try(self.weights, p)
                rewards.append(self.get_reward(weights_try))
        rewards = np.array(rewards)
        return rewards

    def _update_weights(self, rewards, population):
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])
            update_factor = self.learning_rate / \
                (self.POPULATION_SIZE * self.SIGMA)
            self.weights[index] = w + update_factor * \
                np.dot(layer_population.T, rewards).T
        self.learning_rate *= self.decay

    def run(self, iterations):
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None

        start_time = time.time()

        for iteration in range(iterations):

            population = self._get_population()
            rewards = self._get_rewards(pool, population)

            self._update_weights(rewards, population)

            used_time = time.time() - start_time
            reward = self.get_reward(self.get_weights())
            if iteration % 1 == 0:
                print('iteration {} best reward : {}'.format(iteration, reward))

            if reward >= self.best_reward:
                self.best_reward = reward
                self.best_weight = self.get_weights()

            if self.log_function:
                self.log_function(iteration, self.weights,
                                  reward, rewards, time_duration=used_time)
            if self.stop_function and self.stop_function(iteration, self.weights, reward, rewards,
                                                         time_duration=used_time):
                break
            if self.saver:
                self.saver.save(self.get_weights(), iteration, reward)

        if pool is not None:
            pool.close()
            pool.join()

        self.weights = self.best_weight

    def close(self):
        pass
