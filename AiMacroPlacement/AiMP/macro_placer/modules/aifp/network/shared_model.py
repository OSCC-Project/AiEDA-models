import torch.nn as nn
from abc import abstractmethod


class SharedModel(nn.Module):
    def __init__(self):
        super(SharedModel, self).__init__()
        self._is_shared = True
    
    @property
    def is_shared(self):
        return self._is_shared

    @abstractmethod
    def policy_value(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def set_device(self, device):
        self._deivce = device