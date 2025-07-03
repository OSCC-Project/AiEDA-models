import numpy as np
from aimp.aifp.database.replay_buffer.numpy_storage import NumpyStorage
from aimp.aifp.database.data_structure.data_spec import DataSpec
from aimp.aifp import setting
np.random.seed(setting.rl_config['numpy_seed'])

class QueueStorage(NumpyStorage):
    def __init__(self, data_spec:DataSpec, capacity:int):
        super(QueueStorage, self).__init__(data_spec, capacity, parallel_nums=None)
        self._rear = 0
        self._size = 0

    @property
    def size(self):
        return self._size

    def append(self, data_dict):
        self._set_by_idx(data_dict, self._rear)
        self._rear = (self._rear + 1) % self.capacity
        if self._size < self._capacity:
            self._size += 1

    def sample_random_batch(self, batch_size):
        assert self._size > 0
        batch_idx = self._make_sample_index(batch_size)
        data_dict = dict()
        for data_name in self._data_spec.get_specification_dict().keys():
            data_dict[data_name] = self._storage[data_name][batch_idx]
        return data_dict

    def save(self, save_path):
        storage_params = np.array([self._size, self._rear], dtype=np.int32)
        self._save_storage(save_path, storage_params)

    def load(self, load_path):
        storage_params = self._load_storage(load_path)
        self._size = storage_params[0]
        self._rear = storage_params[1]

    def _make_sample_index(self, batch_size):
        batch_idx = np.random.randint(self._size, size=batch_size)
        return batch_idx