from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import random
from aimp.aifp import setting
from aimp.aifp.operation.data_io.adj_reader import read_adj

class PretrainDataset(Dataset):
    def __init__(self, feature_files, label_files):
        super(PretrainDataset, self).__init__()
        assert len(feature_files) == len(label_files)
        self._feature_files = feature_files
        self._label_files = label_files
        self._sparse_adj_i, self._sparse_adj_j, self._sparse_adj_weight = read_adj('chenlu')
        self._macro_idx_to_place = np.array([1], dtype=np.int64)
        print('============= pretrain-dataset length: {} ============='.format(len(self._feature_files)))

    def __len__(self):
        return len(self._feature_files)
    
    def __getitem__(self, idx):
        feature = self._read_features(self._feature_files[idx])
        label = self._read_labels(self._label_files[idx])
        return feature, self._macro_idx_to_place, self._sparse_adj_i, self._sparse_adj_j, self._sparse_adj_weight, label

    def _read_labels(self, label_file_path):
        pd_labels = pd.read_csv(label_file_path)
        return pd_labels.iloc[0, 0].astype(np.float32)

    def _read_features(self, feature_file_path):
        pd_features = pd.read_csv(feature_file_path)
        features = pd_features.iloc[:, 3:5].values.astype(np.float32)
        # features[:, 0] = 1.0
        # features[:, 1:3] /= 600000.0
        features[:, 3:5] /= setting.env_train['max_grid_nums']
        return features

def create_train_and_valid_set():
    # search all feature and label files
    features_dir = os.environ['AIFP_PATH'] + setting.pretrain['features']
    labels_dir = os.environ['AIFP_PATH'] + setting.pretrain['labels']
    feature_file_names = sorted(os.listdir(features_dir))
    random.shuffle(feature_file_names)
    feature_files = []
    label_files = []

    for feature_file_name in feature_file_names:
        label_file_path = labels_dir + feature_file_name.replace('feature', 'label')
        # only if feature and label files both exist, add to dataset
        if not os.path.exists(label_file_path):
            continue
        pd_labels = pd.read_csv(label_file_path)
        reward = pd_labels['reward'][0]
        if reward < -10:
            os.remove(label_file_path)
            os.remove(feature_file_name)
            print('remove noisy file {}, reward = {}'.format(label_file_path, reward))

        feature_file_path = features_dir + feature_file_name
        feature_files.append(feature_file_path)
        label_files.append(label_file_path)

    train_set_len = int(len(feature_files) * (1 - setting.pretrain['valid_properation']))
    
    train_feature_files = feature_files[0:train_set_len]
    train_label_files = label_files[0:train_set_len]
    valid_feature_files = feature_files[train_set_len:]
    valid_label_files = label_files[train_set_len:]

    return PretrainDataset(train_feature_files, train_label_files), PretrainDataset(valid_feature_files, valid_label_files)