import numpy as np
from aimp.aifp.database.replay_buffer.numpy_storage import NumpyStorage
from aimp.aifp.database.data_structure.data_spec import DataSpec

class RolloutStorage(NumpyStorage):
    def __init__(self, data_spec:DataSpec, parallel_nums, step_nums):
        super(RolloutStorage, self).__init__(data_spec, capacity=step_nums, parallel_nums=parallel_nums)
        self._step = 0
        self._step_nums = step_nums

    def append(self, data_dict):
        self._set_parallel_by_idx(data_dict, self._step)
        self._step = (self._step + 1) % self._capacity

    def sample_batch(self, batch_idx):
        data_dict = dict()
        for data_name in self._data_spec.get_specification_dict().keys():
            data_flatten = self._storage[data_name].reshape((-1, ) + self._data_spec.get_specification_dict()[data_name].shape)
            # print('transformed shape', self._storage[data_name].shape)
            data_dict[data_name] = data_flatten[batch_idx]
        if 'adv' in self._storage.keys():
            data_flatten = self._storage['adv'].reshape((-1, ) + (1, ))
            data_dict['adv'] = data_flatten[batch_idx]
        if 'return' in self._storage.keys():
            data_flatten = self._storage['return'].reshape((-1, ) + (1, ))
            data_dict['return'] = data_flatten[batch_idx]
        return data_dict
    
    # def compute_adv_and_return(self, rewards_key, values_key, done_key, value, done, gamma=0.99, gae_lambda=0.95):
    #     assert rewards_key in self._storage.keys()
    #     assert values_key in self._storage.keys()
    #     advantages = np.zeros_like(self._storage[rewards_key])
    #     last_gae_lam = 0
    #     for t in reversed(range(self._step_nums)):
    #         if t == self._step_nums - 1:
    #             next_nonterminal = 1.0 - done
    #             next_values = value#.reshape(1, -1)
    #         else:
    #             next_nonterminal = 1.0 - self._storage[done_key][t + 1]
    #             next_values = self._storage[values_key][t + 1]
    #         delta = self._storage[rewards_key][t] + gamma * next_values * next_nonterminal - self._storage[values_key][t]
    #         advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam
    #     returns = advantages + self._storage[values_key]
    #     self._storage['return'] = returns
    #     self._storage['adv'] = advantages
    #     return advantages, returns

    def compute_adv_and_return(self, rewards_key, values_key, done_key, value, done, gamma=0.99, gae_lambda=0.95):
        assert rewards_key in self._storage.keys()
        assert values_key in self._storage.keys()
        advantages = np.zeros_like(self._storage[rewards_key])
        returns = np.zeros_like(self._storage[rewards_key])
        discounted_reward = np.zeros_like(value)
        for t in reversed(range(self._step_nums)):
            discounted_reward = self._storage[rewards_key][t] + gamma * (1.0 - self._storage[done_key][t]) * discounted_reward
            returns[t] = discounted_reward
            advantages[t] = discounted_reward - self._storage[values_key][t]

        self._storage['return'] = returns
        self._storage['adv'] = advantages
        return advantages, returns