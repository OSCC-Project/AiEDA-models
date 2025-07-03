import torch.nn as nn
from abc import abstractmethod


class IndependentModel(nn.Module):
    def __init__(self):
        super(IndependentModel, self).__init__()
        self._is_shared = False
    
    @property
    def is_shared(self):
        return self._is_shared

    @abstractmethod
    def policy(self):
        pass

    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def get_policy_params(self):
        pass
    
    @abstractmethod
    def get_value_params(self):
        pass

    def set_device(self, device):
        self._deivce = device