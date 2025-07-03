import numpy as np
from abc import abstractmethod
from aimp.aifp.database.data_structure.data_spec import DataSpec

class NumpyStorage:
    def __init__(self, data_spec:DataSpec, capacity:int, parallel_nums:int=None):
        self._data_spec = data_spec
        self._capacity = capacity
        self._parallel_nums = parallel_nums
        self._storage = self._init_storage()

    @property
    def data_spec(self):
        return self._data_spec
    
    @property
    def capacity(self):
        return self._capacity

    @abstractmethod
    def append(self, data_dict):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    def _init_storage(self):
        storage = dict()
        data_spec_dict = self._data_spec.get_specification_dict()
        for key in data_spec_dict.keys():
            if self._parallel_nums != None:
                storage[key] = np.zeros((self._capacity, self._parallel_nums, *data_spec_dict[key].shape), dtype=data_spec_dict[key].dtype)
            else:
                storage[key] = np.zeros((self._capacity, *data_spec_dict[key].shape), dtype=data_spec_dict[key].dtype)
            # storage[key] = np.random.random((self._capacity, *data_spec_dict[key].shape))
        return storage
    
    def _set_by_idx(self, data_dict, idx):
        assert 0 <= idx < self._capacity
        for data_name in data_dict.keys():
            if data_name not in self._data_spec.get_specification_dict().keys():
                raise ValueError
            self._storage[data_name][idx] = data_dict[data_name]

    def _set_parallel_by_idx(self, parallel_data_dict, idx):
        assert self._parallel_nums != None
        assert 0 <= idx < self._capacity
        for data_name in parallel_data_dict.keys():
            if data_name not in self._data_spec.get_specification_dict().keys():
                raise ValueError
            assert parallel_data_dict[data_name].shape[0] == self._parallel_nums
            self._storage[data_name][idx] = parallel_data_dict[data_name]

    def _save_storage(self, save_path, storage_params):
        np_arrays = [self._storage[data_name] for data_name in self._storage.keys()]
        np.savez(
            save_path,
            *np_arrays,
            storage_params=storage_params
        )
        print('numpystorage saved to {}'.format(save_path))

    def _load_storage(self, load_path):
        data = np.load(load_path)
        storage_params = data['storage_params']
        for i, data_name in enumerate(self._storage.keys()):
            assert len(data['arr_{}'.format(i)]) == self._capacity
            self._storage[data_name] = data['arr_{}'.format(i)]
        print('numpystorage loaded from {}'.format(load_path))
        return storage_params


