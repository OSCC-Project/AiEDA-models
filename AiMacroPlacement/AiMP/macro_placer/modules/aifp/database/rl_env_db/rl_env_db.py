import numpy as np
import copy
import torch
import os
import re
import math
import pandas as pd
import time
# from aimp.aifp.operation.data_io.metadata_reader import ReadMetadata
from aimp.aifp.operation.macro_placer.rl_placer.strategy.tutorial_learning import get_valid_margin
from aimp.aifp.database.data_structure.core import PyCore
from aimp.aifp import setting
# from thirdparty.irefactor import py_aifp_cpp as aifp_cpp
from aimp.aifp.database.data_structure import fp_solution

class RLEnvDB(object):
    def __init__(self,
                design_data_dict:dict,
                # core,
                # inst_list,
                # edge_list,
                # blockage_list=None,
                # net_list=None,
                # net_weight=None,
                mode:str='RL'  # 'RA', 'SA'
                # macro_indices:list,
                # io_indices:list,
                # stdcell_indices:list,
                # stdcell_cluster_indices:list=None,
                # fixed_stdcell_indices:list=None,
                ):

        start_time = time.time()
        self._mode = mode
        self._design_data_dict = design_data_dict
        self._init_node_indices()
        for macro_idx in self._macro_indices:  # ispd15 macro status is fixed..., unfix it
            self.get_inst_list()[macro_idx].set_status("unfixed")
            # self.get_inst_list()[macro_idx].set_status("unfixed")


        # core=design_data_dict['core'],
        # inst_list=design_data_dict['inst_list'],
        # edge_list=design_data_dict['edge_list'],
        # blockage_list=design_data_dict['blockage_list'],
        # net_list=design_data_dict['net_list'],
        # net_weight=design_data_dict['net_weight']


        # self.get_core() = core
        # self.get_inst_list() = copy.deepcopy(inst_list)
        # self._init_node_indices()
        # for macro_idx in self._macro_indices:  # ispd15 macro status is fixed..., unfix it
        #     self.get_inst_list()[macro_idx].set_status("unfixed")
        # self._origin_inst_list = copy.deepcopy(self.get_inst_list()) # keep a copy of original inst_list

        # self.get_edge_list() = edge_list
        # self._net_list = net_list # clustered net-list
        # self._net_weight = net_weight

        
        # self._macro_indices = macro_indices
        # self._io_indices = io_indices
        # self._stdcell_indices = stdcell_indices
        # self._stdcell_cluster_indices = stdcell_cluster_indices
        # self._fixed_stdcell_indices = fixed_stdcell_indices
        
        # self._blockage_list = blockage_list
        
        # self._num_grids = setting.env_train['max_grid_nums']
        # self._bin_width = self.get_core().get_width() / self._num_grids
        # self._bin_height = self.get_core().get_height() / self._num_grids

        self._max_grid_nums = setting.env_train['max_grid_nums']
        self._grid_num_x = setting.env_train['grid_num_x']
        self._grid_num_y = setting.env_train['grid_num_y']
        self._grid_offset_x = (self._max_grid_nums - self._grid_num_x) // 2
        self._grid_offset_y = (self._max_grid_nums - self._grid_num_y) // 2
        self._bin_width = self.get_core().get_width() / self._grid_num_x
        self._bin_height = self.get_core().get_height() / self._grid_num_y

        if (mode == 'RL'):
            # print('edge_list_len: ', len(self.get_edge_list()))
            self._sparse_adj_i, self._sparse_adj_j, self._sparse_adj_weight = self._get_sparse_adj(self.get_edge_list())
            # self._init_occupied_grids()
            # self._init_obs_data()

        # elif mode == 'SA':
        #     # assert all macros are initially placed to a grid_loc
        #     for macro_idx in self.get_macro_indices():
        #         macro = self.get_macro_list()[macro_idx]
        #         grid_x, grid_y = self._get_grid_loc(macro.get_halo_center_x(), macro.get_halo_center_y())
        #         self.place_macro_to_grid(grid_x, grid_y)

        self._origin_design_data_dict = copy.deepcopy(self._design_data_dict) # keep a copy of original inst_list
        self.reset(0)
        print('rl_env_db initialized, time: {} secs'.format(time.time() - start_time))

    @property
    def max_grid_nums(self):
        return self._max_grid_nums

    @property
    def num_macros(self):
        return len(self._macro_indices)

    @property
    def num_nodes(self):
        return self.num_macros
        # return len(self.get_inst_list())

    @property
    def num_blockages(self):
        return len(self.get_blockage_list())

    @property
    def num_edges(self):
        return len(self._sparse_adj_j)

    @property
    def grid_num_x(self):
        return self._grid_num_x
    
    @property
    def grid_num_y(self):
        return self._grid_num_y

    @property
    def bin_width(self):
        return self._bin_width
    
    @property
    def bin_height(self):
        return self._bin_height

    @property
    def num_node_feature_dim(self):
        feature_dim = 0
        for feature_name, feature in self._nodedata.items():
            # feature_dim += feature.shape[1]
            feature_dim += 1
        return feature_dim

    def get_core(self):
        # return self.get_core()
        return self._design_data_dict['core']

    def get_inst_list(self):
        return self._design_data_dict['inst_list']
        # return self.get_inst_list()
         
    def get_edge_list(self):
        return self._design_data_dict['edge_list']
        # return self.get_edge_list()
    
    def get_net_list(self):
        return self._design_data_dict['net_list']
        # return self._net_list
    
    def get_net_weight(self):
        return self._design_data_dict['net_weight']
        # return self._net_weight

    def get_blockage_list(self):
        return self._design_data_dict['blockage_list']
        # return self._blockage_list

    def get_macro_list(self):
        return [self.get_inst_list()[idx] for idx in self._macro_indices]

    def get_io_list(self):
        return [self.get_inst_list()[idx] for idx in self._io_indices]

    # def get_stdcell_cluster_list(self):
    #     if self._stdcell_cluster_indices == None:
    #         return None
    #     else:
    #         return [self.get_inst_list()[idx] for idx in self._stdcell_cluster_indices]

    # def get_fixed_stdcell_list(self):
    #     if self._fixed_stdcell_indices == None:
    #         return None
    #     else:
    #         return [self.get_inst_list()[idx] for idx in self._fixed_stdcell_indices]


    def get_macro_indices(self):
        return self._macro_indices

    def get_io_indices(self):
        return self._io_indices

    def get_stdcell_indices(self):
        return self._stdcell_indices

    # def get_stdcell_cluster_indices(self):
    #     return self._stdcell_cluster_indices

    # def get_fixed_stdcell_indices(self):
    #     return self._fixed_stdcell_indices

    # def get_origin_inst_list(self):
    #     return self._origin_inst_list

    def get_adj(self):
        return self._sparse_adj_i.copy(), self._sparse_adj_j.copy(), self._sparse_adj_weight.copy()
    
    def update_origin_macro_info(self, fp_solution:fp_solution.FPSolution):
        """update env_db macro loc info from a macro_info_list"""
        origin_macro_list = [self._origin_design_data_dict['inst_list'][idx] for idx in self.get_macro_indices()]
        for macro in origin_macro_list:
            macro_name = macro.get_name()
            macro_info = fp_solution.get_macro_info(macro_name)
            macro.set_low_x(macro_info.get_low_x())
            macro.set_low_y(macro_info.get_low_y())
            if (macro_info.get_orient() != ''):
                macro.set_orient(macro_info.get_orient())

        # assert len(origin_macro_list) == len(macro_info)
        # for i in range(len(origin_macro_list)):
        #     if origin_macro_list[i].get_name() != macro_info[i]['name']:
        #         print('macro_info doesn\'t match macro list! ')
        #         return False
        # else:
        #     for i in range(len(origin_macro_list)):
        #         origin_macro_list[i].set_low_x(macro_info[i]['low_x'])
        #         origin_macro_list[i].set_low_y(macro_info[i]['low_y'])
        #         if macro_info[i]['orient'] != '':
        #             origin_macro_list[i].set_orient(macro_info[i]['orient'])
        #     print('update env_db macro info succeed...')
        #     return True



    def reset(self, current_iter=0):
        self._last_action_mask = np.ones((self._max_grid_nums, self._max_grid_nums))
        self._current_iter = current_iter
        self._design_data_dict['inst_list'] = copy.deepcopy(self._origin_design_data_dict['inst_list'])
        self._placed_macro_indices = []

        if self._mode == 'RL':
            self._init_occupied_grids()
            self._init_obs_data()

        elif self._mode == 'SA':
            # assert all macros are initially placed to a grid_loc
            for node_idx in self.get_macro_indices():
                macro = self.get_inst_list()[node_idx]
                grid_x, grid_y = self._get_grid_loc(macro.get_halo_center_x(), macro.get_halo_center_y())
                low_x, low_y = self._get_abs_loc(grid_x, grid_y, node_idx)
                # print('grid_x: {}, grid_y: {}'.format(grid_x, grid_y))
                # macro.set_grid_x(grid_x)
                # macro.set_grid_y(grid_y)
                # macro.set_low_x(low_x)
                # macro.set_low_y(low_y)
                # macro.set_status("fixed")
                self._set_macro_grid_loc_without_check(node_idx, grid_x, grid_y, set_fixed=True)
                self._placed_macro_indices.append(node_idx) # set macro placed
                # action = grid_y * self._grid_num_x + grid_x
                # self.place_macro_to_grid(node_idx, action)
                if not self._check_loc_validation(low_x, low_y):
                    print('unvalid loc: low_x: {}, low_y: {}'.format(low_x, low_y))
                    print('core low_x: {}, low_y: {}'.format(self.get_core().get_low_x(), self.get_core().get_low_y()))
                    print('core_width: {}, core_height: {}'.format(self.get_core().get_width(), self.get_core().get_height()))
                    raise ValueError('check location validation failed')

    def move_macro_grid_loc(self, node_idx:int, grid_x, grid_y):
        # try to move a placed macro from to another grid_loc
        if (self._mode != 'SA' or node_idx not in self._placed_macro_indices):
            raise RuntimeError('this method is only called when mode == SA and node has been placed before !')
        
        self._placed_macro_indices.remove(node_idx) # set macro unplaced
        action_mask, overlap_flag = self.get_action_mask(node_idx)
        self._placed_macro_indices.append(node_idx) # set macro placed
        if action_mask[grid_y, grid_x] == 1:
            # move macro to new grid_loc
            # low_x, low_y = self._get_abs_loc(grid_x, grid_y, node_idx)
            # macro.set_grid_x(grid_x)
            # macro.set_grid_y(grid_y)
            # macro.set_low_x(low_x)
            # macro.set_low_y(low_y)
            self._set_macro_grid_loc_without_check(node_idx, grid_x, grid_y)
            return True
        else:
            print('new grid_x, grid_y invalid! do nothing...')
            return False

    def swap_macro_loc(self, node_idx_i:int, node_idx_j:int):
        # try to swap two placed-macro's grid_loc
        if self._mode != 'SA' or node_idx_i not in self._placed_macro_indices or node_idx_j not in self._placed_macro_indices:
            raise RuntimeError('this method is only called when mode == SA and node has been placed before !')
        
        if node_idx_i == node_idx_j:
            print('swap two identical macros, do nothing...')
            return False

        macro_i = self.get_inst_list()[node_idx_i]
        macro_j = self.get_inst_list()[node_idx_j]
        origin_macro_i_grid_x = macro_i.get_grid_x()
        origin_macro_i_grid_y = macro_i.get_grid_y()
        origin_macro_j_grid_x = macro_j.get_grid_x()
        origin_macro_j_grid_y = macro_j.get_grid_y()

        self._placed_macro_indices.remove(node_idx_j) # set macro unplaced
        action_mask_i, overlap_flag = self.get_action_mask(node_idx_i)
        self._placed_macro_indices.append(node_idx_j) # set macro placed

        self._placed_macro_indices.remove(node_idx_i) # set macro unplaced
        action_mask_j, overlap_flag = self.get_action_mask(node_idx_j)
        self._placed_macro_indices.append(node_idx_i) # set macro placed

        if action_mask_i[origin_macro_j_grid_y, origin_macro_j_grid_x] != 1\
            or action_mask_j[origin_macro_i_grid_y, origin_macro_i_grid_x] != 1:
            print('swap failed, invalid!')
            return False

        # swap two macro loc
        self._set_macro_grid_loc_without_check(node_idx_i, origin_macro_j_grid_x, origin_macro_j_grid_y)
        self._set_macro_grid_loc_without_check(node_idx_j, origin_macro_i_grid_x, origin_macro_i_grid_y)
        return True

    def set_macro_orient(self, node_idx:int, new_orient:str):
        if self._mode != 'SA' or node_idx not in self._placed_macro_indices:
            raise RuntimeError('this method is only called when mode == SA and node has been placed before !')
        self.get_inst_list()[node_idx].set_orient(new_orient)


    def place_macro_to_grid(self, node_idx:int, action:int):
        if self._mode != 'RL':
            raise RuntimeError('only called when mode == RL!')
        center_grid_x, center_grid_y = self._action_transform(action)
        self._placed_macro_indices.append(node_idx)
        low_x, low_y = self._get_abs_loc(center_grid_x, center_grid_y, node_idx)

        if not self._check_loc_validation(low_x, low_y):
            print('unvalid loc: low_x: {}, low_y: {}'.format(low_x, low_y))
            print('core low_x: {}, low_y: {}'.format(self.get_core().get_low_x(), self.get_core().get_low_y()))
            print('core_width: {}, core_height: {}'.format(self.get_core().get_width(), self.get_core().get_height()))
            raise ValueError('check location validation failed')
        # update macro's real-location
        self._set_macro_grid_loc_without_check(node_idx, center_grid_x, center_grid_y, set_fixed=True)
        # self.get_inst_list()[node_idx].set_grid_x(center_grid_x)
        # self.get_inst_list()[node_idx].set_grid_y(center_grid_y)
        # self.get_inst_list()[node_idx].set_low_x(low_x)
        # self.get_inst_list()[node_idx].set_low_y(low_y)
        # self.get_inst_list()[node_idx].set_status("fixed")
        # update occupied grids and obs data after setting a macro's location
        self._update_occupied_grids(self.get_inst_list(), node_idx)
        self._update_obs_data(node_idx)

    def _init_node_indices(self):
        self._macro_indices = []
        self._io_indices = []
        self._stdcell_indices = []
        for index, node in enumerate(self.get_inst_list()):
            node_type = node.get_type()
            if node_type == "macro":
                self._macro_indices.append(index)
            elif node_type == "io" or node_type == "io_cluster":
                self._io_indices.append(index)
            else:
                self._stdcell_indices.append(index)

    def _check_loc_validation(self, low_x, low_y):
        if low_x < self.get_core().get_low_x() or low_y < self.get_core().get_low_y():
            return False
        if low_x > self.get_core().get_low_x() + self.get_core().get_width() or low_y > self.get_core().get_low_y() + self.get_core().get_height():
            return False
        return True

    def get_concat_obs_data(self):
        np_obs_data = dict()
        if len(self._metadata) > 0:
            if len(self._metadata) > 1:
                np_obs_data['metadata'] = np.concatenate(list(self._metadata.values()), axis=0)
            else:
                np_obs_data['metadata'] = list(self._metadata.values())[0]
        else:
            np_obs_data['metadata'] = None

        if len(self._layoutdata) > 0:
            if len(self._layoutdata) > 1:
                np_obs_data['layoutdata'] = np.concatenate(list(self._layoutdata.values()), axis=0)
            else:
                np_obs_data['layoutdata'] = list(self._layoutdata.values())[0]
        else:
            np_obs_data['layoutdata'] = None

        if len(self._nodedata) > 0:
            np_obs_data['nodedata'] = np.stack(list(self._nodedata.values()), axis=-1)
        else:
            np_obs_data['nodedata'] = None
        return np_obs_data

    def get_action_mask(self, next_macro_idx):
        restore_flag = False
        # when using 'SA' to get action_mask, unplace macro first
        if self._mode == 'SA' and next_macro_idx in self._placed_macro_indices:
            self._placed_macro_indices.remove(next_macro_idx)
            restore_flag = True

        def margin_occupied_grids(margin_width, margin_height):
            if margin_width < self._bin_width / 2:
                margin_grid_x = 0
            else:
                margin_grid_x = 1 + int((margin_width - self._bin_width / 2) / self._bin_width)
            if margin_height < self._bin_height / 2:
                margin_grid_y = 0
            else:
                margin_grid_y = 1 + int((margin_height - self._bin_height / 2) / self._bin_height)

            return margin_grid_x, margin_grid_y

        def rectangle_dilated_grids(low_x, low_y, high_x, high_y, dilation_x, dilation_y):
            dilated_low_x = max(low_x - dilation_x, self.get_core().get_low_x()) - self.get_core().get_low_x()
            dilated_low_y = max(low_y - dilation_y, self.get_core().get_low_y()) - self.get_core().get_low_y()
            dilated_high_x = min(high_x + dilation_x, self.get_core().get_low_x() + self.get_core().get_width()) - self.get_core().get_low_x()
            dilated_high_y = min(high_y + dilation_y, self.get_core().get_low_y() + self.get_core().get_height()) - self.get_core().get_low_y()
            start_grid_x = math.ceil((dilated_low_x - self._bin_width/2) / self._bin_width)
            start_grid_y = math.ceil((dilated_low_y - self._bin_height/2) / self._bin_height)
            end_grid_x = math.floor((dilated_high_x - self._bin_width/2) / self._bin_width)
            end_grid_y = math.floor((dilated_high_y - self._bin_height/2) / self._bin_height)
            return start_grid_x, end_grid_x, start_grid_y, end_grid_y

        overlap_flag = False
        next_macro_width = self.get_inst_list()[next_macro_idx].get_halo_width()
        next_macro_height = self.get_inst_list()[next_macro_idx].get_halo_height()

        # zero invalid, one valid
        margin_grid_x, margin_grid_y = margin_occupied_grids(next_macro_width / 2, next_macro_height / 2)
        action_mask = np.zeros((self._max_grid_nums, self._max_grid_nums))

        # mask not-used L-shape region

        # mid region (except for margin of macro-size) is valid
        action_mask[margin_grid_y:self._grid_num_y-margin_grid_y, margin_grid_x:self._grid_num_x-margin_grid_x] = 1
        # mask middle area
        if setting.env_train['use_tutorial_learning']:
            obstacle_start_grid_x, obstacle_end_grid_x = get_valid_margin(self._grid_num_x, self._current_iter)
            obstacle_start_grid_y, obstacle_end_grid_y = get_valid_margin(self._grid_num_y, self._current_iter)
            action_mask[obstacle_start_grid_y: obstacle_end_grid_y+1, obstacle_start_grid_x: obstacle_end_grid_x+1] = 0

        # grids occupied by blockages
        for blockage in self.get_blockage_list():
            start_grid_x, end_grid_x, start_grid_y, end_grid_y = rectangle_dilated_grids(
                low_x=blockage.get_low_x(),
                low_y=blockage.get_low_y(),
                high_x=blockage.get_low_x()+blockage.get_width(),
                high_y=blockage.get_low_y()+blockage.get_height(),
                dilation_x=next_macro_width / 2,
                dilation_y=next_macro_height / 2)
            action_mask[start_grid_y:end_grid_y+1, start_grid_x:end_grid_x+1] = 0
        
        # grids occupied by placed macros
        for placed_macro_idx in self._placed_macro_indices:
            # start_grid_x, end_grid_x, start_grid_y, end_grid_y = macro_dilated_grids(placed_macro_idx, next_macro_width / 2, next_macro_height / 2)
            start_grid_x, end_grid_x, start_grid_y, end_grid_y = rectangle_dilated_grids(
                low_x=self.get_inst_list()[placed_macro_idx].get_halo_low_x(),
                low_y=self.get_inst_list()[placed_macro_idx].get_halo_low_y(),
                high_x=self.get_inst_list()[placed_macro_idx].get_halo_low_x()+self.get_inst_list()[placed_macro_idx].get_halo_width(),
                high_y=self.get_inst_list()[placed_macro_idx].get_halo_low_y()+self.get_inst_list()[placed_macro_idx].get_halo_height(),
                dilation_x=next_macro_width / 2,
                dilation_y=next_macro_height / 2)
            action_mask[start_grid_y:end_grid_y+1, start_grid_x:end_grid_x+1] = 0

        if np.sum(action_mask) == 0:  # no valid grid left
            print('no valid grid left...')
            action_mask[margin_grid_y:self._grid_num_y-margin_grid_y, margin_grid_x:self._grid_num_x-margin_grid_x] = 1
            overlap_flag =True

        # now move valid regions to center of max_grid_nums * max-grid_nums region
        mid_action_mask = np.zeros_like(action_mask)
        mid_action_mask[self._grid_offset_y : self._grid_offset_y+self._grid_num_y, self._grid_offset_x : self._grid_offset_x+self._grid_num_x] = action_mask[0:self._grid_num_y, 0:self._grid_num_x] 
        self._last_action_mask = mid_action_mask.copy()

        if (restore_flag == True):
            self._placed_macro_indices.append(next_macro_idx)

        return mid_action_mask, overlap_flag

    """================================ obs related data init ================================"""

    def _init_occupied_grids(self): # init grids occupied by fixed macro
        self._occupied_grids = np.ones((self._grid_num_y, self._grid_num_x), dtype=np.float32)
        for idx in range(len(self.get_blockage_list())):
            self._update_occupied_grids(self.get_blockage_list(), idx)
        for idx in range(len(self.get_inst_list())):
            if self.get_inst_list()[idx].get_status() == "fixed" and self.get_inst_list()[idx].get_type() == "macro":
                self._update_occupied_grids(self.get_inst_list(), idx)

    def _get_sparse_adj(self, edge_list):
        edge_list = np.array(edge_list, dtype=np.float32)
        sparse_adj_i = edge_list[:,0].astype(np.int64)
        sparse_adj_j = edge_list[:,1].astype(np.int64)
        sparse_adj_weight = edge_list[:,2].astype(np.float32)
        sparse_adj_weight /= sparse_adj_weight.max()  # normalize sparse_adj_weight
        return sparse_adj_i, sparse_adj_j, sparse_adj_weight

    def _init_obs_data(self):
        self._init_metadata()
        self._init_nodedata()
        self._init_layoutdata()

    def _init_metadata(self):
        self._metadata = dict()
        pass
        # for metadata_name in setting.metadata_config:
        #     continue # metadata reader not implemented

    def _init_nodedata(self):
        self._nodedata = dict()
        for node_feature_name in setting.node_feature_config:
            if node_feature_name == 'mid_x':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'mid_y':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'is_fixed':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'unfixed':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'is_macro':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'is_stdcell_cluster':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'is_io_cluster':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'is_io_instance':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'is_flipflop':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'is_stdcell':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'width':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'height':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            elif node_feature_name == 'degree':
                self._nodedata[node_feature_name] = np.zeros((self.num_nodes, ), dtype=np.float32)
            else:
                raise NotImplementedError('invalid node feature {}'.format(node_feature_name))
        # for idx in range(len(self.get_inst_list())):
        for idx in range(len(self.get_macro_list())):
            self._update_nodedata(idx)

    def _init_layoutdata(self):
        self._layoutdata = dict()
        for layout_feature_name in setting.layout_feature_config:
            if layout_feature_name == 'density':
                layoutdata_item = np.zeros((self.num_nodes, self._grid_num_y, self._grid_num_x), dtype=np.float32)
            else:
                raise NotImplementedError
            self._layoutdata[layout_feature_name] = layoutdata_item

        for idx in range(len(self.get_inst_list())):
            if self.get_inst_list()[idx].get_status() == "fixed" and self.get_inst_list()[idx].get_type() == "macro":
                self._update_layoutdata(idx)

    """==================== update obs related data after setting a macro's grid loc =========================="""

    def _update_occupied_grids(self, node_list, node_idx):
        # update grids occuipied by macros
        center_x = node_list[node_idx].get_center_x()
        center_y = node_list[node_idx].get_center_y()
        center_grid_x, center_grid_y = self._get_grid_loc(center_x, center_y)
        inst_width = node_list[node_idx].get_halo_width()
        inst_height = node_list[node_idx].get_halo_height()
        scope_x, scope_y = self._calculate_scope(inst_width, inst_height)
        self._occupied_grids[max(0, center_grid_y - scope_y):min(self._grid_num_y, center_grid_y + scope_y + 1),
                            max(0, center_grid_x - scope_x):min(self._grid_num_x, center_grid_x + scope_x + 1)] = 0

    def _update_obs_data(self, node_idx):
        self._update_nodedata(node_idx)
        self._update_layoutdata(node_idx)

    def _update_nodedata(self, idx):
        node = self.get_inst_list()[idx]
        for node_feature_name in setting.node_feature_config:
            if node_feature_name == 'mid_x':
                self._nodedata[node_feature_name][idx] = (node.get_halo_center_x() - self.get_core().get_low_x())/ self.get_core().get_width()
            elif node_feature_name == 'mid_y':
                self._nodedata[node_feature_name][idx] = (node.get_halo_center_y() - self.get_core().get_low_y())/ self.get_core().get_height()
            elif node_feature_name == 'is_fixed':
                if node.get_status() == "fixed":
                    self._nodedata[node_feature_name][idx] = 1.0
            elif node_feature_name == 'unfixed':
                if node.get_status() == "unfixed":
                    self._nodedata[node_feature_name][idx] = 1.0
            elif node_feature_name == 'is_macro':
                if node.get_type() == "macro":
                    self._nodedata[node_feature_name][idx] = 1.0
            elif node_feature_name == 'is_stdcell_cluster':
                if node.get_type() == "stdcell_cluster":
                    self._nodedata[node_feature_name][idx] = 1.0
            elif node_feature_name == 'is_io_cluster':
                if node.get_type() == "io_cluster":
                    self._nodedata[node_feature_name][idx] = 1.0
            elif node_feature_name == 'is_io_instance':
                if node.get_type() == "io_instance":
                    self._nodedata[node_feature_name][idx] = 1.0
            elif node_feature_name == 'is_flipflop':
                if node.get_type() == "flip_flop":
                    self._nodedata[node_feature_name][idx] = 1.0
            elif node_feature_name == 'is_stdcell':
                if node.get_type() == "stdcell":
                    self._nodedata[node_feature_name][idx] = 1.0
            elif node_feature_name == 'width':
                self._nodedata[node_feature_name][idx] = node.get_halo_width() / self.get_core().get_width()
            elif node_feature_name == 'height':
                self._nodedata[node_feature_name][idx] = node.get_halo_height() / self.get_core().get_height()
            elif node_feature_name == 'degree':
                self._nodedata[node_feature_name][idx] = node.get_degree()
            else:
                raise NotImplementedError

    def _update_layoutdata(self, node_idx):
        for layout_feature_name in setting.layout_feature_config:
            if layout_feature_name == 'density':
                center_x = self.get_inst_list()[node_idx].get_halo_center_x()
                center_y = self.get_inst_list()[node_idx].get_halo_center_y()
                inst_width = self.get_inst_list()[node_idx].get_halo_width()
                inst_height = self.get_inst_list()[node_idx].get_halo_height()
                center_grid_x, center_grid_y = self._get_grid_loc(center_x, center_y)
                scope_x, scope_y = self._calculate_scope(inst_width, inst_height)
                for i in range(max(0, center_grid_y - scope_y), min(self._grid_num_y, center_grid_y + scope_y + 1)):
                    for j in range(max(0, center_grid_x - scope_x), min(self._grid_num_x, center_grid_x + scope_x + 1)):
                        self._layoutdata[layout_feature_name][node_idx, i, j] = 1
            else:
                raise NotImplementedError


    """========================= utility functions ========================="""

    def _set_macro_grid_loc_without_check(self, node_idx:int, grid_x, grid_y, set_fixed=False):
        # set grid_loc directly, with out overlap check!
        macro = self.get_macro_list()[node_idx]
        low_x, low_y = self._get_abs_loc(grid_x, grid_y, node_idx)
        macro.set_grid_x(grid_x)
        macro.set_grid_y(grid_y)
        macro.set_low_x(low_x)
        macro.set_low_y(low_y)
        if set_fixed:
            macro.set_status("fixed")

    def _calculate_scope(self, width, height):
        width -= self._bin_width
        height -= self._bin_height
        if width <= 0:
            scope_x = 0
        else:
            scope_x = int(((width / 2) // self._bin_width)) + 1
        if height <= 0:
            scope_y = 0
        else:
            scope_y = int(((height / 2) // self._bin_height)) + 1
        return scope_x, scope_y

    def _get_grid_loc(self, center_x, center_y):
        # grid_x = int(center_x // self._bin_width) + 1
        # grid_y = int(center_y // self._bin_height) + 1

        grid_x = round((center_x - 0.5 * self._bin_width) / self._bin_width)
        grid_y = round((center_y - 0.5 * self._bin_height) / self._bin_height)

        # bug modify
        grid_x += self._grid_offset_x
        grid_y += self._grid_offset_y
        return grid_x, grid_y

    def _get_abs_loc(self, center_grid_x, center_grid_y, inst_list_idx):
        center_grid_x -= self._grid_offset_x
        center_grid_y -= self._grid_offset_y

        low_x = (self._bin_width * (2 * center_grid_x + 1) ) / 2
        low_y = (self._bin_height * (2 * center_grid_y + 1)) / 2
        low_x -= 0.5 * self.get_inst_list()[inst_list_idx].get_width()
        low_y -= 0.5 * self.get_inst_list()[inst_list_idx].get_height()
        low_x += self.get_core().get_low_x()
        low_y += self.get_core().get_low_y()
        return low_x, low_y

    def _get_layoutdata_density(self):
        up = torch.nn.Upsample(size=setting.env_train_config['layout_feature_shape'][0], mode='bilinear', align_corners=False)
        layout_feature = up(torch.from_numpy(self._occupied_grids)).numpy()
        return layout_feature

    def _action_transform(self, action):
        if action < 0 or action >= self._max_grid_nums**2:
            raise ValueError('action {} invalid!'.format(action))
        grid_y = int(action // setting.env_train['max_grid_nums'])
        grid_x = int(action % setting.env_train['max_grid_nums'])
        if self._last_action_mask[grid_y][grid_x] == 0:
            raise RuntimeError('invalid action ({}, {}) is taken..'.format(grid_y, grid_x))
        return grid_x, grid_y