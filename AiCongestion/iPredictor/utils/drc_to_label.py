"""
Author: liudec dec_hi@qq.com
Description: convert drc to label
"""

import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_drc_center(drc_file_name):
    """get violation center points from drc.rpt file

    Args:
        drc_file_name: the name of drc.rpt file

    Returns:
        c_x: the x of center points
        c_y: the y of center points
    """

    # get information from drc.rpt file
    ld_x = []
    ld_y = []
    rt_x = []
    rt_y = []

    re_brackets = re.compile(r'[(](.*?)[)]', re.S)  # 最小匹配
    with open(drc_file_name, 'r', encoding="utf-8") as drc_file:
        for line in drc_file:
            str_list = re.findall(re_brackets, line)
            if len(str_list) == 2:
                str_list_split = str_list[0].split(',')
                if len(str_list_split) == 2:
                    ld_x.append(float(str_list_split[0]))
                    ld_y.append(float(str_list_split[1]))
                str_list_split = str_list[1].split(',')
                if len(str_list_split) == 2:
                    rt_x.append(float(str_list_split[0]))
                    rt_y.append(float(str_list_split[1]))

    # compute center point
    c_x = []
    c_y = []
    size = len(ld_x)
    for i in range(size):
        c_x.append((ld_x[i] + rt_x[i]) / 2)
        c_y.append((ld_y[i] + rt_y[i]) / 2)

    return c_x, c_y


def drc_to_label(drc_file_name, label_file_name, core_size, tile_size):
    """use drc.rpt file converting to label.csv, boundary tiles have been added
    Args:
        drc_file_name: the name of drc.rpt file
        label_file_name: the name of label which will generate, use .csv as
                        filename extension
        core_size: the list, [the width of core area, the height of core area]
        tile_size: the list, [the width of tile area, the height of tile area]
    """

    assert len(core_size) == 2
    assert len(tile_size) == 2

    c_x, c_y = get_drc_center(drc_file_name)

    # convert center points to tile points
    grid_x = []
    grid_y = []
    violation_num = len(c_x)
    for i in range(violation_num):
        grid_x.append(int(c_x[i] // tile_size[0]))
        grid_y.append(int(c_y[i] // tile_size[1]))

    # boundary tiles have been added
    grid_x_num = math.ceil(core_size[0] / tile_size[0])
    grid_y_num = math.ceil(core_size[1] / tile_size[1])

    label = np.zeros((grid_x_num, grid_y_num), dtype=int)
    for i in range(violation_num):
        label[grid_x[i]][grid_y[i]] += 1

    # dump into csv file
    csv_dict = {}
    for i in range(grid_x_num):
        col_str = 'col_' + str(i)
        csv_dict[col_str] = label[i, :].reshape(-1)
    data_frame = pd.DataFrame(csv_dict)
    data_frame.to_csv(label_file_name, index=False, sep=',')


if __name__ == "__main__":
    # test example
    # compute tile size
    core_size = [195.4, 191.52]
    core_size_real = [390800, 383040]

    # compute tile_size
    rate = core_size_real[0] / core_size[0]
    tile_size = [6150 / rate, 6150 / rate]

    # convert drc.rpt file to csv
    drc_to_label(drc_file_name='../data/ispd2018/t1/8t1.drc.rpt',
                 label_file_name='../data/ispd2018/t1/8t1_label.csv',
                 core_size=core_size,
                 tile_size=tile_size)

    # get numpy from csv file, remember to Transpose
    data_read = pd.read_csv('../data/ispd2018/t1/8t1_label.csv')
    label_data = data_read.values.T

    # transform to violation (0 or 1) array
    label_data[label_data >= 1] = 1

    # visualization
    ax = plt.subplot(111)
    for i in range(label_data.shape[0]):
        for j in range(label_data.shape[1]):
            if label_data[i][j] == 1:
                plt.scatter(i, j, c='r', marker='x')
    plt.show()
