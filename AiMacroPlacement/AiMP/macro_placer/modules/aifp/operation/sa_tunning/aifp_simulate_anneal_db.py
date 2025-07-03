from aimp.aifp.database.rl_env_db.rl_env_db import RLEnvDB
import numpy as np

import logging
from aimp.aifp.operation.data_io import report_io
from aimp.aifp.utility import operators
from aimp.aifp.database.data_structure import fp_solution
from aimp.aifp import setting

class AifpSimulateAnnealDB:
    def __init__(self, env_db:RLEnvDB, evaluator):
        self._env_db = env_db
        self._evaluator = evaluator
        self._init_solution()
        self._perturb_method_list = ['perturb_shift', 'perturb_swap', 'perturb_flip' ] # 'perturb_shift', 'perturb_swap',

    def perturb(self):
        while True:
            # randomly select a perturb method, try until succeed
            rand_method = self._perturb_method_list[np.random.randint(0, len(self._perturb_method_list))]
            if rand_method == 'perturb_shift':
                succeed_flag = self._perturb_shift()
            elif rand_method == 'perturb_swap':
                succeed_flag = self._perturb_swap()
            elif rand_method == 'perturb_flip':
                succeed_flag = self._perturb_flip()
            else:
                raise RuntimeError('method name error!')
            if (succeed_flag == True):
                break

    def update(self):
        pass

    def update_solution(self):
        for macro in self._env_db.get_macro_list():
            self._best_solution.update_macro_info(
                macro_name = macro.get_name(),
                low_x = macro.get_low_x(),
                low_y = macro.get_low_y(),
                orient = macro.get_orient()
            )
        self._best_solution.set_score_dict(self._score_dict)
        logging.info('=========== update solution =========')

    def rollback(self):
        perburb_method = self._perturb_info['method']
        if perburb_method == 'shift':
            self._restore_shift()
        elif perburb_method == 'swap':
            self._restore_swap()
        elif perburb_method == 'flip':
            self._restore_flip()
        else:
            raise RuntimeError('restore_info error, error method : {}'.format(self._perturb_info['method']))

    def evaluate(self):
        self._score_dict, converge_flag = self._evaluator.evaluate(self._env_db)
        wirelength = self._score_dict['wirelength']
        return wirelength

    def reset(self):
        self._env_db.reset()

    def _perturb_shift(self):
        # Shift: Randomly pick a macro, then randomly move to a neighbor location
        try_num = 100
        for i in range(try_num):
            rand_macro_idx = self._get_random_macro_idx()
            action_mask, overlap_flag = self._env_db.get_action_mask(rand_macro_idx)
            macro = self._env_db.get_inst_list()[rand_macro_idx]
            max_grid_nums = self._env_db.max_grid_nums

            macro_grid_x = macro.get_grid_x()
            macro_grid_y = macro.get_grid_y()
            # get valid neighbors
            valid_neighbor_grids = []
            max_shift = 2
            for grid_y in range (np.max((0, macro_grid_y - max_shift)), np.min((max_grid_nums, macro_grid_y + max_shift + 1))):
                for grid_x in range(np.max((0, macro_grid_x - max_shift)), np.min((max_grid_nums, macro_grid_x + max_shift + 1))):
                    if action_mask[grid_y, grid_x] == 1 and not (grid_x == macro_grid_x and grid_y == macro_grid_y):
                        valid_neighbor_grids.append([grid_x, grid_y])

            if len(valid_neighbor_grids) == 0:
                continue

            rand_loc = np.random.randint(0, len(valid_neighbor_grids))
            succeed_flag = self._env_db.move_macro_grid_loc(
                node_idx = rand_macro_idx,
                grid_x = valid_neighbor_grids[rand_loc][0],
                grid_y = valid_neighbor_grids[rand_loc][1])

            if succeed_flag == True:
                self._perturb_info = {
                    'method': 'shift',
                    'node_idx': rand_macro_idx,
                    'grid_x': macro_grid_x,
                    'grid_y': macro_grid_y,
                }
                return True
        return False

    def _perturb_swap(self):
        try_num = 100
        for i in range(try_num):
            # randomly get two macro index
            rand_node_idx_i = self._get_random_macro_idx()
            rand_node_idx_j = self._get_random_macro_idx()
            while rand_node_idx_j == rand_node_idx_i:
                rand_node_idx_j = self._get_random_macro_idx()
            # swap macro locs
            succeed_flag = self._env_db.swap_macro_loc(rand_node_idx_i, rand_node_idx_j)
            if succeed_flag == True:
                self._perturb_info = {
                    'method': 'swap',
                    'node_idx_i': rand_node_idx_i,
                    'node_idx_j': rand_node_idx_j,
                }
                return True
        return False

    def _perturb_flip(self):
        flip_type_list = ['vertical', 'horizontal']
        flip_type = flip_type_list[np.random.randint(0, len(flip_type_list))]
        rand_macro_idx = self._get_random_macro_idx()
        rand_macro = self._env_db.get_inst_list()[rand_macro_idx]

        if flip_type == 'vertical':
            new_orient = operators.flip_vertical(rand_macro.get_orient())
        else:
            new_orient = operators.flip_horizontal(rand_macro.get_orient())
        
        self._env_db.set_macro_orient(rand_macro_idx, new_orient)
        self._perturb_info = {
            'method': 'flip',
            'node_idx': rand_macro_idx,
            'flip_type': flip_type
        }
        return True


    def _permute_flip(self):
        raise NotImplementedError

    def _restore_shift(self):
        if self._perturb_info['method'] != 'shift':
            raise RuntimeError('restore_info error!')
        self._env_db.move_macro_grid_loc(
            node_idx = self._perturb_info['node_idx'],
            grid_x = self._perturb_info['grid_x'],
            grid_y = self._perturb_info['grid_y'])
    
    def _restore_swap(self):
        if self._perturb_info['method'] != 'swap':
            raise RuntimeError('restore_info error!')
        self._env_db.swap_macro_loc(
            node_idx_i = self._perturb_info['node_idx_i'],
            node_idx_j = self._perturb_info['node_idx_j'])

    def _restore_flip(self):
        if self._perturb_info['method'] != 'flip':
            raise RuntimeError('restore_info error!')

        node_idx = self._perturb_info['node_idx']
        orient = self._env_db.get_inst_list()[node_idx].get_orient()
        # flip back
        if self._perturb_info['flip_type'] == 'vertical':
            self._env_db.set_macro_orient(node_idx, operators.flip_vertical(orient))
        else:
            self._env_db.set_macro_orient(node_idx, operators.flip_horizontal(orient))

    def _init_solution(self):
        self._best_solution = fp_solution.FPSolution()
        self._best_solution.set_design_name(setting.case_select)
        for macro in self._env_db.get_macro_list():
            self._best_solution.add_macro_info(
                fp_solution.MacroInfo(
                    macro.get_name(),
                    macro.get_low_x(),
                    macro.get_low_y(),
                    macro.get_orient())
            )

    def _get_random_macro_idx(self):
        return np.random.randint(0, self._env_db.num_macros)
    
