"""
Author: liudec dec_hi@qq.com
Description: run the model
"""

import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils.data_process import get_data_and_labels
from model_run.run_svm import run_svm
from model_run.run_rusboost import run_rusboost
from model_run.run_mars import run_mars

if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser(description='run model')
    parser.add_argument('-f',
                        '--feature_maps_file',
                        type=str,
                        nargs='+',
                        help='all feature input',
                        default=[
                            "data/ispd2018/t8/neighbor_pin_density.npy",
                            "data/ispd2018/t8/net_density.npy",
                            "data/ispd2018/t8/num_global_net.npy",
                            "data/ispd2018/t8/num_local_net.npy",
                            "data/ispd2018/t8/pin_density.npy",
                            "data/ispd2018/t8/via_capacity.npy",
                            "data/ispd2018/t8/wire_capacity.npy"
                        ])
    parser.add_argument('-l',
                        '--label_file',
                        type=str,
                        help='violation labels',
                        default="data/ispd2018/t1/8t1_label.csv")
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        help='model choosen',
                        default='net')

    args = parser.parse_args()

    # get data and labels
    all_data, label = get_data_and_labels(args.feature_maps_file,
                                          args.label_file)

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(all_data,
                                                        label,
                                                        test_size=0.2,
                                                        random_state=42)

    # data preprocess
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # choose model to run
    if (args.model == 'svm'):
        run_svm(x_train, x_test, y_train, y_test)
    if (args.model == 'rusboost'):
        run_rusboost(x_train, x_test, y_train, y_test)
    if (args.model == 'mars'):
        run_mars(x_train, x_test, y_train, y_test)