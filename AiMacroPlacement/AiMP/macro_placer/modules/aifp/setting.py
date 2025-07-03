import os 
case_select = 'ariane133'  # ispd15/mgc_edit_dist_a ariane133
out_dir = os.environ['ROOT-PATH'] + 'RLMP_PROJECT'

log = {
    'output_dir':out_dir,
    'run_num': '7',
    'log_dir': 'log/',
    'model_dir': 'model/'
}

pretrain = {
    'valid_properation': 0.2,
    'optimizer': 'Adam',
    'lr': 5e-4,
    'epoch': 1000,
    'batch_size': 512,
    'dataloader_worker': 8,
    'device': 'cuda',
    'use_grad_clip': True,
    'clip_max_norm': 5,
    'clip_norm_type': 2,
    'save_interval': 10,
    'features': 'input/dataset/features/',
    'labels': 'input/dataset/labels/'
}

metadata_config = [
    'macro_nums',
    'grid_nums',
    'core_width',
    'core_height'
]

node_feature_config = [
    'mid_x',
    'mid_y',
    'is_fixed',
    'unfixed',
    'is_macro',
    'is_stdcell_cluster',
    'is_io_cluster',
    # 'is_io_instance',
    # 'is_flipflop',
    # 'is_stdcell',
    'width',
    'height',
    'degree'
]

layout_feature_config = [
    'density'
]

dense_reward_factor = { # useful only if evaluator == 'macro_io_wirelength' and 'use_dense_reward' == True
    'ispd15/mgc_superblue12': 1e8,
    'ispd15/mgc_edit_dist_a': 1e7,
    'ispd15/mgc_des_perf_a': 1e7,
    'chenlu': 1e8,
}
# 0.18196906095742452

# if dreamplace diverge, give a fixed-reward
fixed_reward = {
    'ariane133': -0.11 #-0.19
}

diverge_threshold = {
    'ariane133': -0.11  #-0.19
}

# metis setting, using metis lib interface
metis = {
    'nparts' : 3000, # stdcell cluster num
    'ufactor' : 10, # unbalence factor
    'ncon': 5
}

# hmetis setting, using hmetis-2.0 exe
# hmetis = {
#     'hmetis_exe' : "/root/hmetis-2.0pre1/Linux-x86_64/hmetis2.0pre1",
#     'nparts': 10000, # stdcell cluster num, if nparts <= 0, not use partition.
#     'ufactor': 5, # unbalence factor
#     'nruns': 10, # run num
#     'dbglvl': 0,
#     'seed': 0,
#     'reconst': False,
#     'ptype': 'rb',
#     'ctype': 'gfc1',
#     'rtype': 'moderate',
#     'otype': 'cut'
# }

# graph = {
#     'graph_type': 'clustered_graph',   # clustered_graph, complete_graph
#     'netlist_clustering': True,  # True, False
#     # 'stdcell_cluster_num': 1000, # stdcell-clustering
#     # 'ufactor' : 10,  #300,     # stdcell-clustering
#     'partition_tool': 'hmetis',  # hmetis, metis
#     'io_slice_num' : 100,  # io-clustering
#     'cluster_inst_density': 0.6,  # clustered-instance(soft-macro)'s approximate area
#     'use_net_weight': True, # building clustered net-list
#     'max_fanout': 100,     # ignore nets with too many pins
# }

# simulate_anneal = {
#     'evaluator': 'clustered_dreamplace',
#     'max_num_step': 100,
#     'perturb_per_step': 300,
#     'init_prob': 0.2
# }

env_train = {
    'consider_blockage': False,
    'max_grid_nums': 64, # max grid_nums
    # 'grid_num_x': 23, # actually used grid num x
    # 'grid_num_y': 21, # actually used grid num x
    'grid_num_x': 64, # actually used grid num x
    'grid_num_y': 64, # actually used grid num x
    'graph_max_hop': 1,   # not used now
    'layout_feature_shape': (84, 84), # not used now
    'macro_sort': 'area_desc',
    'placement_reward': 'minus_wirelength',
    'reward_scale': 1,  # not used now...
    'overlap_punishment': False,
    'evaluator': 'clustered_dreamplace',  # dreamplace, clustered_dreamplace, macro_io_wirelength
    'use_tutorial_learning': False,
    'heuristic_reward': None, # [dist_to_edge, dist_to_origin_macro, None...]
    'grpc_max_message_length': 1000000000
}

evaluator = {
    'clustered_dreamplace' : {
        'iteration': 1500, # 1000,
        'target_density':  0.9, # 1.0,
        'learning_rate': 0.01,
        'num_bins_x': 512,
        'num_bins_y': 512,
        'gpu': False,
        'result_dir': '{}/log/{}/run{}/'.format(out_dir, case_select, log['run_num']),
        'legalize_flag': False,
        'stop_overflow': 0.1,
        'routability_opt_flag': False,
        'num_threads': 1,
        'deterministic_flag': False,
        'num_columns': 200,
        'num_rows': 128
    },
    'macro_io_wirelength' : {
        'use_dense_reward': True,  # useful only if evaluator == 'macro_io_wirelength'
        'consider_io': True,   # useful only if evaluator == 'macro_io_wirelength'
    },
    'dreamplace': {}, # use  '/input/case_select/dreamplace.json' to configure
}

env_server = {
    'env_nums': 1,
}

def get_env_nums():
    return env_server['env_nums']

rl_config = {
    'torch_seed': 222,
    'numpy_seed': 222, # used in replaybuffer sample-batch
    'episode_nums': 10,
    'env_nums': get_env_nums(),
    'evaluate': False,
    'evaluate_evaluator': 'clustered_dreamplace',
    'evaluate_nums': 1,
    'evaluate_interval': 1,
    'load_checkpoint': False,
    'save_interval': 1,
    'save_to_dataset': False,
    'device': 'cpu',
    'save_figure': False,
    'save_figure_interval': 5,
}

solver = {
    'ppo': {
        # 'num_episodes_per_iteration': 28 * 30, # must be i * env_nums
        'num_episodes_per_iteration': 1, # must be i * env_nums
        'add_noise': False,
        'dirichlet_aplpha': 0.1,
        'minibatch_size': 512, # google 1024 (128 * 8)
        'update_epochs': 4,
        'learning_rate': 4e-4 * 5,
        'epsilon': 1e-5,
        'value_pred_loss_coef': 0.5,
        'entropy_regularization': 1e-2, # 1e-4, # 0.01,
        'importance_ratio_clipping': 0.2,
        'discount_factor': 1.0,
        'gradient_clipping': 1.0,
        'gamma': 1.0,
        'gae_lambda': 1.0},

    'sac': {}
}