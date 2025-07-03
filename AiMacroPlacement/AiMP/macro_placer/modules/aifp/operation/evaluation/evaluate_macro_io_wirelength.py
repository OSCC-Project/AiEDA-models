import numpy as np
import pandas as pd
import os
import sys
import logging
import time
import copy
from os.path import abspath


from aimp.aifp.operation.evaluation.evaluate_base import EvaluateBase
from aimp.aifp.database.rl_env_db.rl_env_db import RLEnvDB
from aimp.aifp.database.data_structure.instance import PyInstance
# from thirdparty.irefactor import py_aifp_cpp as aifp_cpp

class EvaluateMacroIoWirelength(EvaluateBase):
    def __init__(self, env_db: RLEnvDB):
        super(EvaluateMacroIoWirelength, self).__init__()
        self._env_db = env_db

    def evaluate(self, consider_io:bool=True, consider_fixed_only=True):
        """evaluate hpwl wirelengh between macro-macro & macro-io
            Args:
                consider_io: if False, only considers wirelength between macro-macro
                consider_fixed_only: if True, only consider macro and io-cluster which status is "fixed"
            returns:
                dict of evaluate scores
        """
        inst_list = self._env_db.get_inst_list()
        edge_list = self._env_db.get_edge_list()
        macro_indices = self._env_db.get_macro_indices()
        io_indices = self._env_db.get_io_indices()
        
        wirelength = 0
        macro_macro_weight = 1.0
        macro_io_weight = 1.0

        for start_idx, end_idx, edge_weight in edge_list:
            # ignore edges between unfixed nodes (macro may not fixed...)
            if consider_fixed_only == True:
                if inst_list[start_idx].get_status() == "unfixed" or inst_list[end_idx].get_status() == "unfixed":
                    continue
            # macro-macro edge
            if start_idx in macro_indices and end_idx in macro_indices:
                wirelength += macro_macro_weight * edge_weight * self._manhattan_distance(inst_list[start_idx], inst_list[end_idx])
            # macro_io edge
            elif consider_io == True:
                if (start_idx in macro_indices and end_idx in io_indices) or (start_idx in io_indices and end_idx in macro_indices):
                    wirelength += macro_io_weight * edge_weight * self._manhattan_distance(inst_list[start_idx], inst_list[end_idx])
        converge_flag = True
        return {'wirelength': wirelength}, converge_flag

    def _manhattan_distance(self, inst_a:PyInstance, inst_b:PyInstance):
        return abs(inst_a.get_center_x() - inst_b.get_center_x()) + abs(inst_a.get_center_y() - inst_b.get_center_y())