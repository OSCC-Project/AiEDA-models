#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : train.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        :
@version      : 0.1
@date         : 2022-03-18 22:05:50
@copyright    : Copyright (c) 2021-2022 PCNL EDA.
'''


import os
import random
import sys
import numpy as np
import sys
import os


def set_seed(params, envs):
    seed = params.seed

    np.random.seed(seed)
    random.seed(seed)

    for env in envs:
        env.seed(seed)
        env.action_space.seed(seed)


def run(Args, loaded_map):
    log_save_dir = './'
    cell_metadata_box, cell_num, rows_num, site_num = Args.data
    row_height = 8
    site_width = 1
    cell_metadata_box.sort(key=lambda t: (
        math.floor(t[1] / row_height + 0.5), t[0] + t[2] // 2))
    env = Env(cell_box=cell_metadata_box, row_num=rows_num, site_num=site_num, row_height=8,
              site_width=1)
    env.reset()

    # set_seed(Args, [env])

    solver = Solver(POPULATION_SIZE=Args.POPULATION_SIZE, train_num=Args.train_num,
                    env=env, print_step=1, model_save_path=log_save_dir, logger=None, saver=None)
    solver.solve()
    solver.close()


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    from gym_env.place_env import PlaceEnv1 as Env
    from models.es_model.es_solver import ESSolver as Solver
    from utils.args import Args
    from utils.reader import *
    exp_name = 'RL For Legalization'
    run(Args, exp_name)
