import os
import copy
import numpy as np
import logging
import time
import socket
import multiprocessing
from multiprocessing import Process
import sys
# os.environ['AIFP_PATH'] = '/root/reconstruct-aifp/'
sys.path.append(os.environ)


from aifp.database.data_structure.box_space import BoxSpace
from aifp.database.data_structure.observation_space import ObservationSpace
from aifp.operation.macro_placer.rl_placer.strategy.macro_sort import MacroSort
from aifp.operation.macro_placer.rl_placer.reward.placement_reward import PlacementReward
from aifp.operation.macro_placer.rl_placer.reward.heuristic_reward import HeuristicReward
from aifp.database.rl_env_db.rl_env_db import RLEnvDB
from aifp.operation.evaluation import evaluate_dreamplace
from aifp.operation.evaluation import evaluate_macro_io_wirelength
from aifp.operation.evaluation import evaluate_clustered_dreamplace
from aifp.utility.operators import number_digits
from aifp.utility.draw import draw_layout
import json
from thirdparty.irefactor import py_aifp_cpp as aifp_cpp
from aifp import setting
from aifp.operation.data_io import aifp_db_io


# @ray.remote(num_cpus=1, num_gpus=0)
class LocalEnv:
    def __init__(self, is_evaluate=False):
        case_select = setting.case_select
        self._is_evaluate=is_evaluate
        self._database = self._init_env_database(case_select)
        if self._is_evaluate == False: # may be dreamplace or some fast evaluator
            self._evaluator = self._init_evaluator(setting.env_train['evaluator'], case_select)
        else:  # evaluate with greedy policy
            self._evaluator = self._init_evaluator(setting.rl_config['evaluate_evaluator'], case_select)

        self._observation_space = self._init_observation_space()
        self._action_space = self._init_action_space()

        self._unfixed_macro_indices = self._get_unfixed_macro_indices()
        self._sort_macros(self._unfixed_macro_indices)
        print("===== base env:   macro num {}, unfixed_macro_num: {}, node num {}".format(self.macro_nums, self.unfixed_macro_nums, self.node_nums))

    def set_log_dir(self, env_name:str):
        self._env_name = env_name
        project_dir = os.environ['AIFP_PATH']
        log_dir = setting.log['log_dir']
        model_dir = setting.log['model_dir']
        case_select = setting.case_select
        run_num = setting.log['run_num']
        
        self._log_dir = '{}/{}/{}/run{}'.format(project_dir, log_dir, case_select, run_num)
        self._model_dir = '{}/{}/{}/run{}'.format(project_dir, model_dir, case_select, run_num)
        self._figure_dir = '{}/figure_env_{}'.format(self._log_dir, self._env_name)

        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        if not os.path.exists(self._figure_dir):
            os.makedirs(self._figure_dir)
        print('===== setting env log dir ======')
        self._env_logger = logging.getLogger()
        self._env_logger.setLevel(logging.INFO)
        self._env_logger.addHandler(logging.FileHandler(self._log_dir + '/env_{}_logging.log'.format(self._env_name)))
        self._env_logger.root.name = 'aifp:env_{}'.format(self._env_name)
        self._env_logger.info('========= env server {} log ========='.format(self._env_name))
        self._env_logger.info('node num: {}'.format(self._database.num_nodes))
        self._env_logger.info('edge num: {}'.format(self._database.num_edges))
        self._env_logger.info('macro num: {}'.format(self._database.num_macros))
        self._env_logger.info('blockage num: {}'.format(self._database.num_blockages))
        self._env_logger.info('episode_length: {}'.format(self.episode_length))
        self._start_time = time.time()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def episode_length(self):
        return self.unfixed_macro_nums

    @property
    def node_nums(self):
        return self._database.num_nodes

    def get_node_nums(self):
        return self._database.num_nodes

    @property
    def macro_nums(self):
        return self._database.num_macros
    
    def get_macro_nums(self):
        return self._database.num_macros
    
    @property
    def unfixed_macro_nums(self):
        return len(self._unfixed_macro_indices)

    @property
    def edge_nums(self):
        return self._database.num_edges

    def get_edge_nums(self):
        return self._database.num_edges

    @property
    def grid_nums(self):
        return setting.env_train['max_grid_nums']

    def get_database(self):
        return self._database

    def reset(self, iteration):
        """
        Args:
            iteration: for recording, and some reward-function's computation.
        Returns:
            obs_list:  list of observations to forward model
                obs_list[0]: node_features       numpy.ndarray, shape(node_nums, node_features), dtype=np.float32
                obs_list[1]: macro_idx_to_place  numpy.ndarray, shape(,), dtype=np.int64
                obs_list[2]: sparse_adj_i        numpy.ndarray, shape(edge_nums, ), dtype=np.int64
                obs_list[3]: sparse_adj_j        numpy.ndarray, shape(edge_nums, ), dtype=np.int64
                obs_list[4]: sparse_adj_weight   numpy.ndarray, shape(edge_nums, ), dtype=np.float32
                obs_list[5]: action_mask         numpy.ndarray, shape(grid_nums, grid_nums), dtype=np.float32
        """
        self._macro_overlaped = False  # if overlap, may return constant reward
        self._iteration = iteration # for some learning param ? may not used actually
        self._step = 0
        self._database.reset(iteration)
        np_obs_data_dict = self._database.get_concat_obs_data()
        macro_idx_to_place = self._unfixed_macro_indices[self._step]
        action_mask, self._overlap_flag = self._database.get_action_mask(macro_idx_to_place)
        sparse_adj_i, sparse_adj_j, sparse_adj_weight = self._database.get_adj()

        macro_idx_to_place = np.array(macro_idx_to_place, dtype=self.observation_space.macro_idx_to_place.dtype).reshape(self.observation_space.macro_idx_to_place.shape)
        sparse_adj_i = sparse_adj_i.astype(self.observation_space.sparse_adj_i.dtype)
        sparse_adj_j = sparse_adj_j.astype(self.observation_space.sparse_adj_j.dtype)
        sparse_adj_weight = sparse_adj_weight.astype(self.observation_space.sparse_adj_weight.dtype)
        action_mask = action_mask.astype(self.observation_space.action_mask.dtype)
        node_features = np_obs_data_dict['nodedata'].astype(self.observation_space.node_features.dtype)

        self._last_wl_reward = 0 # used for non-sparse wl reward
        return [node_features, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight, action_mask]

    def step(self, action:int):
        """
        Args:
            action: int discrete action

        Returns:
            obs_list: list of observations to forward model
                obs_list[0]: node_features       numpy.ndarray, shape(node_nums, node_features), dtype=np.float32
                obs_list[1]: macro_idx_to_place  numpy.ndarray, shape(,), dtype=np.int64
                obs_list[2]: sparse_adj_i        numpy.ndarray, shape(edge_nums, ), dtype=np.int64
                obs_list[3]: sparse_adj_j        numpy.ndarray, shape(edge_nums, ), dtype=np.int64
                obs_list[4]: sparse_adj_weight   numpy.ndarray, shape(edge_nums, ), dtype=np.float32
                obs_list[5]: action_mask         numpy.ndarray, shape(grid_nums, grid_nums), dtype=np.float32
            reward: numpy.ndarray, shape(1,), dtype=np.float32
            done:   numpy.ndarray, shape(1,), dtype=np.float32, = 0.0 if episode ends, else 1.0
        """

        try:
            assert self._step < self.episode_length  # can only called when episode not ended
        except:
            print('episode len: ', self.episode_length)
            print('step: ', self._step)
            self._env_logger.info('episode len: ', self.episode_length)
            self._env_logger.info('step: ', self._step)
            raise RuntimeError('step function called after episode ended')

        # place macro
        placing_macro_idx = self._unfixed_macro_indices[self._step]
        # center_grid_x, center_grid_y = self._action_transform(action)
        # self._database.place_macro_to_grid(placing_macro_idx, center_grid_x, center_grid_y)
        self._database.place_macro_to_grid(placing_macro_idx, action)
        
        # next macro to place
        self._step += 1
        np_obs_data_dict = self._database.get_concat_obs_data()

        if self._step < self.episode_length:
            macro_idx_to_place = self._unfixed_macro_indices[self._step]
            action_mask, overlap_flag = self._database.get_action_mask(macro_idx_to_place)
            if overlap_flag == True:
                self._overlap_flag = overlap_flag

            done = 0.0
            reward = self._get_heuristic_reward()
            info = ''
            # if evaluator == EvaluateMacroIoWirelength, and use dense reward...
            if isinstance(self._evaluator, evaluate_macro_io_wirelength.EvaluateMacroIoWirelength) and setting.env_train['use_dense_reward']==True:
                wl = self._evaluate()['wirelength']
                wl_reward = wl / setting.dense_reward_factor[setting.case_select]
                reward +=  -(wl_reward - self._last_wl_reward)
                self._last_wl_reward = wl_reward
                # print('step {} wl_reward: {}'.format(self._step, wl_reward))
        else:
            macro_idx_to_place = -1
            action_mask = np.zeros(self.observation_space.action_mask.shape, dtype=self.observation_space.action_mask.dtype)
            done = 1.0
            reward, score_dict = self._get_placement_reward()
            info = self._get_episode_info(reward, score_dict)

            # at last step, heuristic-reward is not used
            # if evaluator == EvaluateMacroIoWirelength, and use dense reward...
            if isinstance(self._evaluator, evaluate_macro_io_wirelength.EvaluateMacroIoWirelength) and setting.env_train['use_dense_reward']==True:
                wl = self._evaluate()['wirelength']
                wl_reward = wl / setting.dense_reward_factor[setting.case_select]
                reward =  - (wl_reward - self._last_wl_reward)
                self._last_wl_reward = 0
                # print('last step {} wl_reward: {}'.format(self._step, wl_reward))

            self._score_dict = score_dict
            self._log_result(reward, score_dict)
            if setting.rl_config['save_to_dataset']:
                self._save_to_dataset(reward)
            

        macro_idx_to_place = np.array(macro_idx_to_place, dtype=self.observation_space.macro_idx_to_place.dtype).reshape(self.observation_space.macro_idx_to_place.shape)
        node_features = np_obs_data_dict['nodedata'].astype(self.observation_space.node_features.dtype)
        sparse_adj_i, sparse_adj_j, sparse_adj_weight = self._database.get_adj()
        sparse_adj_i = sparse_adj_i.astype(self.observation_space.sparse_adj_i.dtype)
        sparse_adj_j = sparse_adj_j.astype(self.observation_space.sparse_adj_j.dtype)
        sparse_adj_weight = sparse_adj_weight.astype(self.observation_space.sparse_adj_weight.dtype)
        action_mask = action_mask.astype(self.observation_space.action_mask.dtype)
        done = np.array(done, dtype=self.observation_space.done.dtype).reshape(self.observation_space.done.shape)
        reward = np.array(reward, dtype=self.observation_space.reward.dtype).reshape(self.observation_space.reward.shape)

        return node_features, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight, action_mask, reward, done, info
        # return [node_features, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight, action_mask], reward, done, info

    def get_score_dict(self):
        return copy.deepcopy(self._score_dict)

    def _get_episode_info(self, reward, score_dict:dict):
        info = []
        info.append('env_name:{}'.format(self._env_name))
        info.append('iteration:{}'.format(self._iteration))
        info.append('overlap:{}'.format(self._overlap_flag))
        info.append('reward:{}'.format(reward))
        for score_key, score_value in score_dict.items():
            info.append('{}:{}'.format(score_key, score_value))
        return '\n'.join(info)

    def _log_result(self, reward, score_dict):
        self._env_logger.info('================ episode {} : =================='.format(self._iteration))
        self._env_logger.info('time: {} secs'.format(time.time() - self._start_time))
        self._start_time = time.time()
        # self._env_logger.info('overlap: {}'.format(self._overlap_flag))
        # self._env_logger.info('reward: {}'.format(reward))
        # for score_key, score_value in score_dict.items():
        #     self._env_logger.info('{}: {}'.format(score_key, score_value))
        self._env_logger.info(self._get_episode_info(reward, score_dict))
        self._env_logger.info('========== macro_info =============')
        for macro in self._database.get_macro_list():
            self._env_logger.info('name:{},low_x:{},low_y:{},orient:{}'.format(macro.get_name(), macro.get_low_x(), macro.get_low_y(), macro.get_orient()))
        self._env_logger.info('========================================================\n')
        if (setting.rl_config['save_figure']) == True:
            if (self._iteration+1) % setting.rl_config['save_figure_interval'] == 0:
                self._env_logger.info('drawing layout result...\n')
                draw_layout(save_path='{}/iter_{}.png'.format(self._figure_dir, self._iteration),
                            inst_list=self._database.get_macro_list() + self._database.get_io_list(),
                            core=self._database.get_core())


    def _save_to_dataset(self, reward):
        file_name = str(time.time())
        
        feature_write = open(os.environ['AIFP_PATH'] + setting.pretrain['features'] +  '/feature_{}.csv'.format(file_name), 'w')
        feature_write.write('macro_idx,width,height,grid_x,grid_y,low_x,low_y\n')
        for macro in self._database.get_macro_list():
            feature_write.write('{},{},{},{},{},{},{}\n'.format(macro.get_index(), macro.get_width(), macro.get_height(), macro.get_grid_x(), macro.get_grid_y(), macro.get_low_x(), macro.get_low_y()))
        feature_write.close

        label_write = open(os.environ['AIFP_PATH'] + setting.pretrain['labels'] +  '/label_{}.csv'.format(file_name), 'w')
        label_title = ['reward']
        scores = [str(reward)]
        for score_key, score_value in self._score_dict.items():
            label_title.append(score_key)
            scores.append(str(score_value))
        label_write.write(','.join(label_title) + '\n')
        label_write.write(','.join(scores) + '\n')
        label_write.close()

    def _get_unfixed_macro_indices(self):
        inst_list = self._database.get_inst_list()
        return [idx for idx in self._database.get_macro_indices() if inst_list[idx].get_status() == aifp_cpp.InstanceStatus.unfixed]

    def _sort_macros(self, unfixed_macro_indices):
        if setting.env_train['macro_sort'] == 'area_desc':
            MacroSort.sort_area_desc(unfixed_macro_indices, self._database.get_inst_list())
        else:
            raise NotImplementedError

    def _get_heuristic_reward(self):
        # if setting.env_train['heuristic_reward'] == 'dist_to_edge':
        #     return HeuristicReward.dist_to_edge(macro, core, self.node_nums)
        if setting.env_train['heuristic_reward'] == 'dist_to_origin_macro':
            return HeuristicReward.dist_to_origin_macro(
                placed_macro = self._database.get_inst_list()[self._unfixed_macro_indices[self._step-1]],
                origin_macro = self._database.get_origin_inst_list()[self._unfixed_macro_indices[self._step-1]],
                episode_length = self.episode_length)
        elif setting.env_train['heuristic_reward'] == None:
            return 0
        else:
            return None

    def _get_placement_reward(self):
        def get_overlap_reward(reward):
            reward_digits = number_digits(reward)
            if (reward > 0):
                reward = 10**reward_digits
            else:
                reward = - 10**reward_digits
            return reward

        assert self._step == self.episode_length
        
        score_dict, converge_flag = self._evaluate()
        # print('score_dict: ', score_dict)
        # print('converge_flag: ', converge_flag)
        wirelength = score_dict['wirelength']
        reward = - wirelength / (self._database.get_core().get_width() + self._database.get_core().get_height())
        # dreamplace's optimization may diverge, then give a fixed reward
        if setting.case_select in setting.fixed_reward.keys():
            if (reward < setting.diverge_threshold[setting.case_select]):
                reward = setting.fixed_reward[setting.case_select]
        # if setting.env_train['placement_reward'] == 'minus_wirelength':
        #     reward = PlacementReward.minus_wirelength(score_dict, setting.env_train['reward_scale'])
        #     if self._overlap_flag == True and setting.env_train['overlap_punishment'] == True:  # if macro overlapped, return constant overlap-reward.
        #         reward = get_overlap_reward(reward)
        # else:
        #     raise NotImplementedError
        return reward, score_dict

    def _evaluate(self):
        if isinstance(self._evaluator, evaluate_dreamplace.EvaluateDreamplace):
            score_dict, converge_flag = self._evaluator.evaluate(self._database.get_macro_list())
        elif isinstance(self._evaluator, evaluate_clustered_dreamplace.EvaluateClusteredDreamplace):
            score_dict, converge_flag = self._evaluator.evaluate(self._database)
        elif isinstance(self._evaluator, evaluate_macro_io_wirelength.EvaluateMacroIoWirelength):
            score_dict, converge_flag = self._evaluator.evaluate()
        else:
            raise ValueError
        return score_dict, converge_flag

    """ ================= init database, evaluator, action-space, obs-space ==================="""

    def _init_env_database(self, case_select):
        idb_config_path = '{}/input/{}/irefactor_idb_config.json'.format(os.environ['AIFP_PATH'], case_select)
        design_data_dict = aifp_db_io.read_from_aifp_db_and_destroy(idb_config_path)
        print('========== load from aifp-db success ==========')
        return RLEnvDB(design_data_dict)
        # return RLEnvDB(
        #             core=design_data_dict['core'],
        #             inst_list=design_data_dict['inst_list'],
        #             edge_list=design_data_dict['edge_list'],
        #             blockage_list=design_data_dict['blockage_list'],
        #             net_list=design_data_dict['net_list'],
        #             net_weight=design_data_dict['net_weight'])

    def _init_evaluator(self, evaluator_name, case_select):
        if evaluator_name == 'dreamplace':
            print('evaluator: dreamplace')
            evaluator = evaluate_dreamplace.EvaluateDreamplace(
                idb_core=self._database.get_core(),
                dreamplace_config_file=os.environ["AIFP_PATH"] + 'input/' + case_select + '/dreamplace.json',)
        elif evaluator_name == 'clustered_dreamplace':
            print('evaluator: clustered-dreamplace')
            evaluator = evaluate_clustered_dreamplace.EvaluateClusteredDreamplace(env_db=self._database)
        elif evaluator_name == 'macro_io_wirelength':
            evaluator = evaluate_macro_io_wirelength.EvaluateMacroIoWirelength(env_db=self._database)
        else:
            raise NotImplementedError('evaluator name not supported !')
        return evaluator

    def _init_observation_space(self):
        return ObservationSpace(self.node_nums, self.edge_nums)

    def _init_action_space(self):
        return BoxSpace(low=0, high=setting.env_train['max_grid_nums']**2, shape=(1,))


    # def _check_action_validation(self, grid_x, grid_y):
    #     # return self._last_action_mask[grid_x][grid_y] == 1
    #     return self._last_action_mask[grid_y][grid_x] == 1
    
    def __del__(self):
        pass