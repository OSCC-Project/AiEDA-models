"""
Author: juanyu 291701755@qq.com
Description: data classification
"""

import numpy as np

def modify_label(dataset_id, design_id):
    """modify_label
       0: DRV-away 1:DRV-located 2:DRV-neighbor

    Args:
        dataset_id: dataset name
        design_id: design name

    Returns:
        label: modified label
    """
    label_file = "../data/" + dataset_id + "/" + design_id + "/label.npy"

    label = np.load(label_file)
    label[label >= 1] = 1

    # find out DRV-located
    DRV_loc = np.argwhere(label == 1)

    row_boundary = label.shape[0]
    column_boundary = label.shape[1]

    # find out DRV-neighbor and mark it as 2
    for i in range(DRV_loc.shape[0]):
        row = DRV_loc[i][0]
        column = DRV_loc[i][1]

        if column - 1 >= 0:
            if label[row][column - 1] == 0:
                label[row][column - 1] = 2
            if row - 1 >= 0:
                if label[row - 1][column - 1] == 0:
                    label[row - 1][column - 1] = 2
            if row + 1 < row_boundary:
                if label[row + 1][column - 1] == 0:
                    label[row + 1][column - 1] = 2

        if row - 1 >= 0:
            if label[row - 1][column] == 0:
                label[row - 1][column] = 2
        if row + 1 < row_boundary:
            if label[row + 1][column] == 0:
                label[row + 1][column] = 2

        if column + 1 < column_boundary:
            if label[row][column + 1] == 0:
                label[row][column + 1] = 2
            if row - 1 >= 0:
                if label[row - 1][column + 1] == 0:
                    label[row - 1][column + 1] = 2
            if row + 1 < row_boundary:
                if label[row + 1][column + 1] == 0:
                    label[row + 1][column + 1] = 2

    # Now the sample is labeled as follows
    # 0: DRV-away 1:DRV-located 2:DRV-neighbor
    return label

def get_feature_maps(feature_maps_files):
    """get feature_maps

    Args:
        feature_maps_files: feature map file name list

    Returns:
        feature_maps: feature maps data
    """
    # get feature maps
    feature_maps = np.array([])
    first_flag = True
    for feature_map_file in feature_maps_files:
        feature_map = np.load(feature_map_file)
        if feature_map.ndim < 3:
            feature_map = np.expand_dims(feature_map, 0)
        if first_flag:
            feature_maps = feature_map
            first_flag = False
            continue
        feature_maps = np.concatenate((feature_maps, feature_map), axis=0)

    return feature_maps

def save_data(feature_maps, label, data_dir):
    """save_data
       Save as three npy files according to the three categories of data

    Args:
        feature_maps: all feature map
        label: modified label
        data_dir: path to save the npy file

    """

    data = np.concatenate((feature_maps, np.expand_dims(label, 0)), axis=0)

    DRV_away_first = True
    DRV_located_first = True
    DRV_neighbor_first = True

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            one_instance = data[:, i, j].reshape(-1, 24)
            if one_instance[0][23] == 0:
                if DRV_away_first:
                    DRV_away = one_instance
                    DRV_away_first = False
                else:
                    DRV_away = np.concatenate((DRV_away, one_instance), axis=0)

            if one_instance[0][23] == 1:
                if DRV_located_first:
                    DRV_located = one_instance
                    DRV_located_first = False
                else:
                    DRV_located = np.concatenate((DRV_located, one_instance), axis=0)

            if one_instance[0][23] == 2:
                one_instance[0][23] = 0
                if DRV_neighbor_first:
                    DRV_neighbor = one_instance
                    DRV_neighbor_first = False
                else:
                    DRV_neighbor = np.concatenate((DRV_neighbor, one_instance), axis=0)

    np.save(data_dir + "DRV_away.npy", DRV_away)
    np.save(data_dir + "DRV_located.npy", DRV_located)
    np.save(data_dir + "DRV_neighbor.npy", DRV_neighbor)

def load_data(design_id_list):
    """load_data
       Load three kinds of samples of all designs in a dataset

    Args:
        design_id_list: design id list

    Returns:
        DRV_away: DRV_away samples
        DRV_located: DRV_located samples
        DRV_neighbor: DRV_neighbor samples
    """
    DRV_away_first = True
    DRV_located_first = True
    DRV_neighbor_first = True

    for design_id in design_id_list:
        a = np.load("../data/ispd2019/" + design_id + "/DRV_away.npy")
        l = np.load("../data/ispd2019/" + design_id + "/DRV_located.npy")
        n = np.load("../data/ispd2019/" + design_id + "/DRV_neighbor.npy")
        if DRV_away_first:
            DRV_away = a
            DRV_away_first = False
        else:
            DRV_away = np.concatenate((DRV_away, a), axis=0)

        if DRV_located_first:
            DRV_located = l
            DRV_located_first = False
        else:
            DRV_located = np.concatenate((DRV_located, l), axis=0)

        if DRV_neighbor_first:
            DRV_neighbor = n
            DRV_neighbor_first = False
        else:
            DRV_neighbor = np.concatenate((DRV_neighbor, n), axis=0)

    return DRV_away, DRV_located, DRV_neighbor

if __name__ == "__main__":
    dataset_id = "ispd2019"  # ispd2018 ispd2019
    design_id = "9t10"  # 8t1 8t2 8t3 8t4 8t5 8t6 8t7 8t8 8t9 8t10
                       # 9t1 9t2 9t3 9t4(5 layers) 9t5(5 layers) 9t6 9t7 9t8 9t9 9t10

    data_dir = "../data/" + dataset_id + "/" + design_id + "/"

    feature_maps_files = [
                           "../data/" + dataset_id + "/" + design_id + "/neighbor_pin_density.npy",
                           "../data/" + dataset_id + "/" + design_id + "/net_density.npy",
                           "../data/" + dataset_id + "/" + design_id + "/num_global_net.npy",
                           "../data/" + dataset_id + "/" + design_id + "/num_local_net.npy",
                           "../data/" + dataset_id + "/" + design_id + "/pin_density.npy",
                           "../data/" + dataset_id + "/" + design_id + "/via_capacity.npy",
                           "../data/" + dataset_id + "/" + design_id + "/wire_capacity.npy"
                         ]

    new_label = modify_label(dataset_id, design_id)
    feature_maps = get_feature_maps(feature_maps_files)
    save_data(feature_maps, new_label, data_dir)


