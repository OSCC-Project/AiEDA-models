import numpy as np

class DataSpec:
    def __init__(self):
        self._specification_dict = dict()
    
    def add_data_item(self, name:str, dtype:np.dtype, shape:tuple):
        self._specification_dict[name] = DataInfo(dtype, shape)
    
    def get_specification_dict(self):
        return self._specification_dict

class DataInfo:
    def __init__(self, dtype:np.dtype, shape:tuple):
        self._dtype = dtype
        self._shape = shape
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def shape(self):
        return self._shape