#from pickle import FALSE
import numpy as np
import os
import sys
import logging
import time
import datetime
import copy
from os.path import abspath
import ctypes

import thirdparty.dreamplace.dreamplace.configure as configure
import thirdparty.dreamplace.dreamplace.Params as Params
import thirdparty.dreamplace.dreamplace.PlaceDB as PlaceDB
import thirdparty.dreamplace.dreamplace.NonLinearPlace as NonLinearPlace

from aimp.aifp.operation.evaluation.evaluate_base import EvaluateBase

class EvaluateIEDA(EvaluateBase):
    def __init__(
        self,
        case_select,
        ieda_config_file = None):

        super(EvaluateIEDA, self).__init__()

        logging.root.name = 'aifp:evaluator:iEDA'
        # logging.basicConfig(level=logging.INFO,format='[%(levelname)-7s] %(name)s - %(message)s',stream=sys.stdout)
        
        if ieda_config_file == None:
            ieda_config_file = os.environ['AIFP+PATH'] + '/input/' + case_select + '/ieda.json'
        
        so_location = "./third_party/iEDA_lib"
        ctypes.CDLL(so_location + "/liblef.so", mode=ctypes.RTLD_GLOBAL) # 6
        ctypes.CDLL(so_location + "/libdef.so", mode=ctypes.RTLD_GLOBAL) # 7
        ctypes.CDLL(so_location + "/iDB/libIdb.so", mode=ctypes.RTLD_GLOBAL) # 11
        ctypes.CDLL(so_location + "/iDB/libdef_service.so", mode=ctypes.RTLD_GLOBAL) # 12
        ctypes.CDLL(so_location + "/iDB/libverilog_write.so", mode=ctypes.RTLD_GLOBAL) # 13
        ctypes.CDLL(so_location + "/iDB/libdata_service.so", mode=ctypes.RTLD_GLOBAL) # 15
        ctypes.CDLL(so_location + "/iDB/liblef_read.so", mode=ctypes.RTLD_GLOBAL) # 17
        ctypes.CDLL(so_location + "/iDB/liblef_service.so", mode=ctypes.RTLD_GLOBAL) # 18
        ctypes.CDLL(so_location + "/iDB/libdata_process.so", mode=ctypes.RTLD_GLOBAL) # 16
        ctypes.CDLL(so_location + "/iDB/libdef_write.so", mode=ctypes.RTLD_GLOBAL) # 19
        ctypes.CDLL(so_location + "/iDB/libdef_read.so", mode=ctypes.RTLD_GLOBAL) # 20
        ctypes.CDLL(so_location + "/iDB/libverilog_read.so", mode=ctypes.RTLD_GLOBAL) # 14
        ctypes.CDLL(so_location + "/iDB/libIdbBuilder.so", mode=ctypes.RTLD_GLOBAL) # 25
        ctypes.CDLL(so_location + "/libverilog-parser.so", mode=ctypes.RTLD_GLOBAL) # 5
        ctypes.CDLL(so_location + "/libstr.so", mode=ctypes.RTLD_GLOBAL) # 3
        ctypes.CDLL(so_location + "/libipl-wrapper.so", mode=ctypes.RTLD_GLOBAL) # 26
        ctypes.CDLL(so_location + "/libipl-configurator.so", mode=ctypes.RTLD_GLOBAL) # 27

        self._ipl_placer = ctypes.CDLL(so_location + '/libcall_iPL.so')
        ieda_config_file = ieda_config_file.encode('utf-8')
        self._ipl_placer.set_json_file(ieda_config_file)
        self._ipl_placer.startIPL()
        self._num_macro = self._ipl_placer.get_num_macro()
        self._core = self.get_core()


        # self._macro_node_map = self._init_macro_node_map(macro_node_map_config_file)
        # self._init_db_blockage()
        # self._num_macro = len(self._macro_node_map)

    def get_macro_list(self):
        macro_list = []
        for i in range(self._num_macro):
            temp = Macro()
            temp.set_index(i) 
            temp.set_low_x(self._iPL_placer.get_macro_x(i))
            temp.set_low_y(self._iPL_placer.get_macro_y(i))
            temp.set_width(self._iPL_placer.get_macro_width(i))
            temp.set_height(self._iPL_placer.get_macro_height(i))
            macro_list.append(temp)
        return macro_list

    def set_macro_list(self, macro_list):
        if len(macro_list) != self._num_macro:
            ValueError('macro num out of ', self._num_macro)
        core_llx = self._core.low_x
        core_lly = self._core.low_y
        core_urx = core_llx + self._core.width
        core_ury = core_lly + self._core.height
        print("in set_macro_list----------------------------")
        print("core_llx: ", core_llx, " core_lly: ", core_lly, " core_urx: ", core_urx, " core_ury: ", core_ury)
        for i in range(len(macro_list)):
            macro_llx = macro_list[i].low_x.round()
            macro_lly = macro_list[i].low_y.round()
            macro_llx.restype = ctypes.c_int
            macro_lly.restype = ctypes.c_int
            macro_urx = macro_llx + macro_list[i].width
            macro_ury = macro_lly + macro_list[i].height
            print("macro",i,": ",macro_llx, " ", macro_lly," ",macro_urx," ", macro_ury)
            print("origin urx: ", i, " ", macro_list[i].low_x + macro_list[i].width)
            if (macro_llx < core_llx) or (macro_lly < core_lly) or (macro_urx > core_urx) or (macro_ury > core_ury):
                print("core: ", core_llx, " ", core_lly, " ",core_urx, " ", core_ury)
                print("macro",i,": ",macro_llx, " ", macro_lly," ",macro_urx," ", macro_ury)
                raise ValueError('macro location out of core bound.')
            self._iPL_placer.set_macro_location(macro_list[i].index, int(macro_llx), int(macro_lly))

    def get_core(self):
        core = Macro()
        core.set_low_x(self._iPL_placer.get_core_llx())
        core.set_low_y(self._iPL_placer.get_core_lly())
        core.set_width(self._iPL_placer.get_core_urx() - self._iPL_placer.get_core_llx())
        core.set_height(self._iPL_placer.get_core_ury() - self._iPL_placer.get_core_lly())
        return core
    
    def evaluate(self, macro_list):
        self.set_macro_list(macro_list)
        self._run_gp()
        evaluation_scores = dict()
        evaluation_scores['wirelength'] = self._get_hpwl()
        evaluation_scores['overflow'] = self._get_overflow()
        return evaluation_scores

    def _run_pl(self):
        self._iPL_placer.runPL()

    def _run_gp(self):
        starttime = datetime.datetime.now()
        self._iPL_placer.runGP()
        endtime = datetime.datetime.now()
        print(self._count, " GP runtime: ", (endtime - starttime).seconds)
        self._count = self._count + 1

    def _get_overflow(self):
        self._iPL_placer.get_overflow.restype = ctypes.c_float
        return self._iPL_placer.get_overflow()

    def _get_hpwl(self):
        self._iPL_placer.get_HPWL.restype = ctypes.c_long
        return self._iPL_placer.get_HPWL()
    
    def _write_gds(self, file_name):
        self._iPL_placer.writeGDS(file_name)