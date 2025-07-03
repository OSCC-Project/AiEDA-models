"""
Author: liudec dec_hi@qq.com / juanyu 291701755@qq.com
Description: data process
"""

import random

import numpy as np
import torch

from skimage.util.shape import view_as_windows
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def load_npy(path):
    """load npy

    Args:
        path: npy file name

    Returns:
        data: the data we read from npy file
    """

    data = np.load(path)
    return data


def get_data_and_labels(feature_maps_files, label_file):
    """get data and label

    Args:
        feature_maps_files: feature map file name list
        label_file: label file name

    Returns:
        feature_maps: feature maps data
        label: label data
    """
    # get feature maps
    feature_maps = np.array([])
    feature_vector_num = 0
    first_flag = True
    for feature_map_file in feature_maps_files:
        feature_map = load_npy(feature_map_file)
        if feature_map.ndim < 3:
            feature_map = np.expand_dims(feature_map, 0)
            feature_vector_num += 1
        else:
            feature_vector_num += feature_map.shape[0]
        if first_flag:
            feature_maps = feature_map
            first_flag = False
            continue
        feature_maps = np.concatenate((feature_maps, feature_map), axis=0)
    feature_maps = feature_maps.reshape(feature_vector_num, -1)
    feature_maps = feature_maps.T

    # get label whose shape is [-1,1]
    # if it has violation, positive is 1, negative is 0
    label = load_npy(label_file)
    label[label >= 1] = 1
    label = label.reshape(-1, 1)
    return feature_maps, label


def get_batch(samples, targets, size=256):
    """get batch from samples and targets

    Args:
        samples: data
        targets: labels
        size: the size of each batch. Defaults to 256.

    Returns:
        two ndarray: a batch of data, a batch of labels
    """
    num_sample = samples.shape[0]
    labels = list(range(num_sample))
    random.shuffle(labels)
    batch = np.zeros((size, samples.shape[1]))
    target = np.zeros((size, 2))
    for ndx in range(size):
        batch[ndx, :] = samples[labels[ndx], :]
        target[ndx, :] = targets[labels[ndx], :]
    return torch.from_numpy(batch), torch.from_numpy(target)

def create_dataset(data, label, save_path, process_mode):
    """Merge features and labels and save as an npy file

    Args:
        data: feature data
        label: labels
        save_path: save path to the generated npy file
        process_mode: data processing mode. Normalize or Standardize

    """
    if process_mode == "MinMax":
        scaler = MinMaxScaler()
    elif process_mode == "Standard":
        scaler = StandardScaler()

    scaler.fit(data)
    data = scaler.transform(data)
    all_data = np.append(data, label, axis=1)
    np.save(save_path + "all_data_" + process_mode + ".npy", all_data)

def window_crop(x, window_size, stride, is_mask):
    """Clipping feature data and labels based on sliding Windows

    Args:
        x: feature data or labels
        window_size: sliding window size
        stride: Sliding window sliding stride
        is_mask: flag bit determines whether x are labels

    """
    if is_mask:
        window_clip = view_as_windows(x, window_size, stride)
        for i in range(window_clip.shape[0]):
            for j in range(window_clip.shape[1]):
                out_window = window_clip[i][j]
                np.save('data/ispd2018/t1/window_crop/mask/window_size_{}_stride_{}_{}_{}_mask',
                        format(window_size, stride, i, j),out_window)
    else:
        window_clip = view_as_windows(x, window_size, stride)
        for i in range(window_clip.shape[1]):
            for j in range(window_clip.shape[2]):
                out_window = window_clip[0][i][j]
                np.save('data/ispd2018/t1/window_crop/imgs/window_size_{}_stride_{}_{}_{}',
                        format(window_size, stride, i, j),out_window)

