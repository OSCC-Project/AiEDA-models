"""
Author: liudec dec_hi@qq.com
Description: DNN model
"""

import torch
from torch import nn


class DNN(nn.Module):
    """DNN

    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(23, 25)
        self.fc2 = nn.Linear(25, 2)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        return out
