from .reader import *
import os


class Args(object):
    env_name = 'Hopper-v2'
    file_name = 'case1'
    seed = 1234
    num_episode = 2000
    batch_size = 2048
    max_step_per_round = 13000
    gamma = 0.1
    lamda = 0.97
    log_num_episode = 1
    num_epoch = 10
    minibatch_size = 256
    clip = 0.2
    loss_coeff_value = 0.5
    loss_coeff_entropy = 0.01

    lr = 3e-4
    num_parallel_run = 1
    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = False
    advantage_norm = False
    lossvalue_norm = True
    data = read_data(os.getcwd() + '/../bench/FPGA_STD/' + file_name + '.in')

    design = 'rocket_top'
    version = 'V1'

    # ES args
    num_threads = 1
    POPULATION_SIZE = 40
    train_num = 10000


'''    
'ES_base': {
        'env': "ES_GR",
        'exp_name': 'ES_GR',
        'lr': 1e-3,
        'seed': 0,
        'action_type': 'VonNeumann_4',  # 'Moore_8'
        'width': 7,
        'height': 7,
        'train_num': 1000,
        'POPULATION_SIZE': 40,
        'solver': 'ES',
        'block_width': 0,
        'train_rank': True
    }
    '''
