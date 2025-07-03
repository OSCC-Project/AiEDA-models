import numpy as np
import os
import sys
import logging
import time
import copy


sys.path.append(os.environ['THIRD-PARTY-PATH'] + 'dreamplace')
# print('third_dir: ', os.environ['THIRD-PARTY-PATH'])

import dreamplace.configure as configure
import dreamplace.Params as Params
import dreamplace.PlaceDB as PlaceDB
import dreamplace.NonLinearPlace as NonLinearPlace

from aimp.aifp.operation.evaluation.evaluate_base import EvaluateBase
from aimp.aifp.database.data_structure.instance import PyInstance
from aimp.aifp.database.data_structure.core import PyCore
# from thirdparty.irefactor import py_aifp_cpp as aifp_cpp
from aimp.aifp import setting




class EvaluateDreamplace(EvaluateBase):
    def __init__(self,
                idb_core,
                dreamplace_config_file):

        super(EvaluateDreamplace, self).__init__()
        logging.root.name = 'aifp:evaluator:DREAMPlace'
        logging.basicConfig(level=logging.INFO,format='[%(levelname)-7s] %(name)s - %(message)s',stream=sys.stdout)
        
        self.params = Params.Params()
        self.params.load(dreamplace_config_file)
        self.params.result_dir=setting.evaluator['clustered_dreamplace']['result_dir']

        os.environ["OMP_NUM_THREADS"] = "%d" % (self.params.num_threads)
        self._placedb = PlaceDB.PlaceDB()
        self._placedb(self.params)
        
        self._init_db_blockage()
        self._idb_core = idb_core
        self._dreamplace_core = self.get_core()
        self._get_macro_io_node_ids()

    def evaluate(self, macro_list):         
        self._set_db_macros(macro_list) # set macro coordinate
        print("start dreamplace")
        result = self._place(self.params, self._placedb)
        # (metrics[-1][-1][-1].iteration) < total_iterations
        print("end dreamplace")
        wirelength = float(result[0].hpwl.data)
        overflow = float(result[0].overflow.mean().data)
        converge_flag = True
        print('dreamplace wirelength: ', wirelength)
        return {'wirelength': wirelength, 'overflow': overflow}, converge_flag

    def get_core(self):
        core = PyCore()
        core.set_name("core")
        core.set_low_x(self._placedb.xl)
        core.set_low_y(self._placedb.yl)
        core.set_width(self._placedb.xh - self._placedb.xl)
        core.set_height(self._placedb.yh - self._placedb.yl)
        return core

    def _init_db_blockage(self):
        index = 0
        for key in self._placedb.node_name2id_map.keys():
            if key == 'DREAMPlacePlaceBlockage' + str(index):
                node_id = self._placedb.node_name2id_map[key]
                self._placedb.node_size_x[node_id] = 0
                self._placedb.node_size_y[node_id] = 0
                index = index + 1
    
    def _set_db_macros(self, macro_list):
        for macro in macro_list:
            # if not macro.name.endswith('.DREAMPlace.Shape0'):
            node_id = self._placedb.node_name2id_map[macro.get_name() + '.DREAMPlace.Shape0'] #+ '.DREAMPlace.Shape0'
            if self._idb_core == None:
                dreamplace_low_x, dreamplace_low_y = macro.get_low_x(), macro.get_low_y()
            else:
                # coord need be tranoformed because dreamplace will scale coords in PlaceDB.
                dreamplace_low_x, dreamplace_low_y = self._idb_cord_to_dreamplace_cord(macro.get_low_x(), macro.get_low_y())
            self._placedb.node_x[node_id] = dreamplace_low_x
            self._placedb.node_y[node_id] = dreamplace_low_y


    def _place(self, params, place_db):
        assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
                "CANNOT enable GPU without CUDA compiled"
        np.random.seed(params.random_seed)
        tt = time.time()
        placer = NonLinearPlace.NonLinearPlace(self.params, self._placedb)
        metrics = placer(params, place_db)
        logging.info("non-linear placement     takes %.2f seconds" % (time.time() - tt))
        result = metrics[-3][0]
        return result

    def _get_macro_io_node_ids(self):
        self._macro_start_id = self._placedb.num_movable_nodes
        self._macro_end_id = self._macro_start_id + self._placedb.num_terminals - 1
        self._io_start_id = self._macro_end_id + 1
        self._io_end_id = self._placedb.num_physical_nodes - 1
    

    def _idb_cord_to_dreamplace_cord(self, idb_low_x, idb_low_y):
        assert self._dreamplace_core.get_low_x() == 0
        assert self._dreamplace_core.get_low_y() == 0

        x_coefficent = self._dreamplace_core.get_width() / self._idb_core.get_width()
        y_coefficent = self._dreamplace_core.get_height() / self._idb_core.get_height()
        dreamplace_low_x = (idb_low_x - self._idb_core.get_low_x()) * x_coefficent
        dreamplace_low_y = (idb_low_y - self._idb_core.get_low_y()) * y_coefficent
        return dreamplace_low_x, dreamplace_low_y



    # def get_macro_list(self):
    #     macro_list = []
    #     macro_list_idx = 0
    #     for i in range(0, self._macro_end_id - self._macro_start_id+1):
    #         temp = PyInstance()
    #         # node_id = self._macro_node_map[i]
    #         node_id = self._macro_start_id + i
    #         if str(self._placedb.node_names[node_id]).startswith('b\'DREAMPlacePlaceBlockage'):
    #             continue
    #         temp.set_type(aifp_cpp.InstanceType.macro)
    #         temp.set_name(self._placedb.node_names[node_id])
    #         temp.set_index(macro_list_idx)
    #         temp.set_low_x(self._placedb.node_x[node_id])
    #         temp.set_low_y(self._placedb.node_y[node_id])
    #         temp.set_width(self._placedb.node_size_x[node_id])
    #         temp.set_height(self._placedb.node_size_y[node_id])
    #         temp.set_index(i)
    #         #temp.set_fixed()
    #         macro_list.append(temp)
    #         macro_list_idx += 1
    #     return macro_list