from pyclbr import Class
import numpy as np
import os
from gym.utils import seeding
from gym import Space, spaces, logger
import math
import random
import logging
import gym
import json
import time
import torch.nn.functional as F
import torch


class EnvTO(gym.Env):

    def __init__(self, gnn_embd, label):
        self._INT_MAX = 2147483647

        self._gnn_embd = gnn_embd

        # for temp test. To get rewards
        self._label = label

        self._set_action_space(7)
        # self.reset()
        pass

    def _set_action_space(self, max):
        self._action_num = max
        self._action_space = spaces.Discrete(max)

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

    def step(self, action):
        # reward

        # next state
        return

    def reset(self):
        state = self._gnn_embd[0]
        return state

    def render(self):
        return

    def close(self):
        return

    def seed(self):
        return


class INTER_ENV(object):
    def __init__(self, gnn_embd, label):
        self._gnn_embd = gnn_embd

        # for temp test. To get rewards
        self._label = label

        self._state_idx = 0

        self.set_action_space(7)
        # self.reset()
        pass

    def set_action_space(self, max):
        self._action_num = max
        self._action_space = spaces.Discrete(max)

    def set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

    def reset(self):
        self._state_idx = 0
        state = self._gnn_embd[self._state_idx]
        self._state_idx += 1
        return state

    def next_state(self):
        state = self._gnn_embd[self._state_idx]
        self._state_idx += 1
        return state

    def step(self, action, next_state_transition):
        # reward
        # action.eq
        reward = torch.from_numpy(action).eq(self._label).sum().numpy()

        # next state
        # next_state = self._state_idx.numpy()
        # next_state = self._gnn_embd.detach().numpy()
        state = self._gnn_embd.detach()
        next_state = torch.matmul(next_state_transition, state).numpy()
        done = False
        return next_state, reward, done
