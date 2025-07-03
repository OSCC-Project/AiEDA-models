"""
Author: juanyu 291701755@qq.com
Description: dataset for DNN model
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class DNNDataset(Dataset):
    """DNNDataset

    """

    def __init__(self, dataset_dir):
        xy = np.load(dataset_dir)
        self.x_data = torch.from_numpy(xy[:, 0:23])
        self.y_data = torch.from_numpy(xy[:, 23:])
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


