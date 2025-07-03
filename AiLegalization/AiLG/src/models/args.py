class args(object):
    env_name = 'Hopper-v2'
    seed = 1234
    num_episode = 100
    batch_size = 2048
    max_step_per_round = 13000
    gamma = 0.6
    lamda = 0.97
    log_num_episode = 1
    num_epoch = 10
    minibatch_size = 32
    clip = 0.2
    loss_coeff_value = 0.5
    loss_coeff_entropy = 0.01
    lr = 3e-4
    num_parallel_run = 5
    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = True
    advantage_norm = True
    lossvalue_norm = True
    data = None
