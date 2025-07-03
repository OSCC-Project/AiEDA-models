import matplotlib
from models.args import args
from models.ppo import ppo_batch_train
import matplotlib.pyplot as plt
from os.path import join as joindir
from os import makedirs as mkdir
import pandas as pd
import numpy as np
import argparse
import datetime
import math
from utils.reader import read_data


def ppo_test(args):
    record_dfs = []
    file_name = 'case1'
    # cell_metadata_box, cell_num, rows_num, site_num
    args.data = read_data('../bench/FPGA_STD/' + file_name + '.in')
    args.seed += 1
    reward_record = pd.DataFrame(ppo_batch_train(args))
    reward_record['#parallel_run'] = 1
    record_dfs.append(reward_record)
    record_dfs = pd.concat(record_dfs, axis=0)
    record_dfs.to_csv(
        joindir(RESULT_DIR, 'ppo-record-{}.csv'.format(args.env_name)))


if __name__ == '__main__':
    RESULT_DIR = joindir('../result', '.'.join(__file__.split('.')[:-1]))
    mkdir(RESULT_DIR, exist_ok=True)
    matplotlib.use('agg')
    args.env_name = 'PlaceEnv-v1'
    ppo_test(args)
