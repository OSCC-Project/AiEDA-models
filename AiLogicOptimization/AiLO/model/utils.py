import random
import torch

def random_sampler(D, train, test_split=0.3):

    len_D = len(D)
    test_size = int(len_D * test_split)
    train_size = len_D - test_size
    train_D = D[:train_size]
    test_D = D[train_size:]

    if train:
        return train_D
    else:
        return test_D