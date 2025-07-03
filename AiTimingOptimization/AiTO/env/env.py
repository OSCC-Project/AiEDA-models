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
import ctypes
from ctypes import *

from util import interval_mapping

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
    def __init__(self, gnn_embd):
        self._gnn_embd = gnn_embd

        self._state_idx = 0

        self.set_action_space(7)
        # self.reset()
        # self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iTO/libiTO.so")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libunwind.so.8")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libstdc++.so.6")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libc.so.6")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/src/third_party/abseil/lib/unix/libabsl_city.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/src/third_party/abseil/lib/unix/libabsl_hash.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/builder/builder/libIdbBuilder.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/third_party/flute3/libflute.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iSTA/api/libista-engine.so")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libpthread.so.0")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iSTA/sta/libsta.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iSTA/netlist/libnetlist.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iSTA/liberty/libliberty.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/src/third_party/glog/libglog.so.0")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/builder/def_builder/def_service/libdef_service.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/builder/lef_builder/lef_service/liblef_service.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/db/libIdb.so")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libm.so.6")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/liblzma.so.5")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libgcc_s.so.1")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/builder/def_builder/def_read/libdef_read.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/builder/def_builder/def_write/libdef_write.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/builder/lef_builder/lef_read/liblef_read.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/builder/data_builder/data_process/libdata_process.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/builder/data_builder/data_service/libdata_service.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/builder/verilog_builder/verilog_read/libverilog_read.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iDB/builder/verilog_builder/verilog_write/libverilog_write.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iSTA/delay/libdelay.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iUtility/string/libstr.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/src/third_party/abseil/lib/unix/libabsl_raw_hash_set.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iSTA/sdc-cmd/libsdc-cmd.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iSTA/verilog-parser/libverilog-parser.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iUtility/stdBase/graph/libgraph.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iSTA/sdc/libsdc.so")
        self._lib = ctypes.CDLL("/usr/local/lib/libyaml-cpp.so.0.6")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iSTA/third-party/libfort/libfort.so.SOVERSION")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iUtility/time/libtime.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iUtility/tcl/libtcl.so")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libtcl8.6.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iSTA/utility/libutility.so")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libgflags.so.2.2")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/src/iDB/builder/def_builder/def/lib/libdef.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/src/iDB/builder/lef_builder/lef/lib/liblef.so")
        self._lib = ctypes.CDLL("/lib/libabsl_time.so")
        self._lib = ctypes.CDLL("/lib/libabsl_time_zone.so")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libdl.so.2")
        self._lib = ctypes.CDLL("/lib/x86_64-linux-gnu/libz.so.1")
        self._lib = ctypes.CDLL("/lib/libabsl_strings.so")
        self._lib = ctypes.CDLL("/lib/libabsl_base.so")
        self._lib = ctypes.CDLL("/lib/libabsl_int128.so")
        self._lib = ctypes.CDLL("/lib/libabsl_raw_logging_internal.so")
        self._lib = ctypes.CDLL("/lib/libabsl_strings_internal.so")
        self._lib = ctypes.CDLL("/lib/libabsl_throw_delegate.so")
        self._lib = ctypes.CDLL("/lib/libabsl_spinlock_wait.so")
        self._lib = ctypes.CDLL("/home/wuhongxi/iEDA-test/iEDA/build/src/iTO/libTEST.so")
        self._lib.init_()
        self._lib.init_data_()

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

    def step(self, action):
        # reward
        # action.eq
        action = np.argmax(action, axis=1)
        # reward = torch.from_numpy(action).eq(
        #     self._label).sum().numpy()/len(self._label)
        with open("test.txt","a") as f:
            for a in action:
                f.write(str(a))
                f.write(" ")
            f.write("\n")

        arr = action.tolist()
        for i, val in enumerate(arr):
            # self._lib.set_solution_(i, val)
            self._lib.set_delta_(i, val)
        self._lib.get_reward_.restype = c_float
        reward = -self._lib.get_reward_()
        print("python reward", reward)
        # action = torch.argmax(action, axis=1)
        # reward = action.eq(self._label).sum()

        # next state
        # next_state = self._state_idx.numpy()
        next_state = self._gnn_embd.detach().numpy()
        # next_state = self._gnn_embd

        done = False
        return next_state, reward, done

    def step(self, action, next_state_transition):
        # reward
        # action.eq
        # action = np.argmax(action, axis=1)
        action = np.squeeze(action, axis=(1,))

        arr = action.tolist()
        # buf_arr = interval_mapping(action.tolist()[122:], 0.5, 20.3)
        # arr[122:] = buf_arr

        with open("delta.txt","a") as f:
            for a in arr:
                f.write(str(a))
                f.write(" ")
            f.write("\n")

        for i, val in enumerate(arr):
            # self._lib.set_solution_(i, val + 15)
            # delta = c_float(val + 15)
            delta = c_float(val)
            self._lib.set_delta_(i, delta)
        self._lib.get_reward_.restype = c_float
        reward = self._lib.get_reward_()
        print("python reward", reward)
        # action = torch.argmax(action, axis=1)
        # reward = action.eq(self._label).sum()

        # next state
        state = self._gnn_embd.detach()
        next_state = torch.matmul(next_state_transition, state).numpy()
        # next_state = self._gnn_embd.detach().numpy()

        done = False
        return next_state, reward, done

    def saveDef(self):
        self._lib.saveDef_()
