#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : main.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        : 
@version      : 0.1
@date         : 2023-11-15 21:08:06
'''


# from aifp.operation.sa_tunning.simulate_anneal_tunner import SimulateAnnealTunner
# from aifp.operation.macro_placer.rl_placer.rl_placer import RLPlacer
# from aifp import setting
from abc import ABC, abstractmethod
from enum import Enum
import os
import sys
root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

os.environ['ROOT-PATH'] = root_dir
os.environ['THIRD-PARTY-PATH'] = root_dir + '/third_party/'

sys.path.append(root_dir)
sys.path.append(root_dir + '/third_party/ieda/iDB/')
sys.path.append(root_dir + '/app')

current_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
path_to_add = f'{root_dir}/ai_infra/tools/iEDA/lib'
os.environ['LD_LIBRARY_PATH'] = f'{current_ld_library_path}:{path_to_add}'
print(os.environ['LD_LIBRARY_PATH'])
# from eda_engine.engine.iEDA.lib import liblog
from eda_engine.engine.iEDA.lib import ieda_py as ieda
from aimp.params import Params
from aimp.macroConfig import *
from aimp.macroPlaceDB import MacroPlaceDB
from eda_engine.ai_infra.data_manager.data_manager import DataManager
from eda_engine.ai_infra.data_manager.path.path_manager import PathManager
from eda_engine.engine.iEDA.module.ieda_engine_dm import EngineDataManager
from eda_engine.engine.iEDA.ieda_engine import EngineIEDA
from aimp.safp.run_safp import SAPlacer
from aimp.aifp.operation.macro_placer.rl_placer.rl_placer import RLPlacer



def imp_macro_place(params, data_manager: DataManager):
    engine_ieda = EngineIEDA(
        design_name=params.design_name,
        path_manager=data_manager.get_path_manager())
    # workspace_path = '/home/zhaoxueyan/code/ai-eda/workspace_aimp/'
    # path_manager = PathManager()
    engine_data_ieda = EngineDataManager(
        params.design_name, data_manager.get_path_manager())
    # engine_data_ieda.read_verilog(
    #     '/home/zhaoxueyan/code/ai-eda/workspace_aimp/ariane133/ariane133/netlist/ariane.v', top_module='ariane')
    engine_data_ieda.read_def(
        "/data/project_share/Testcases/t28/XSTop_3000x2000_1000M/def/XSTop_input.def", False)
    print('read verilog done')
    macroPlaceDB = MacroPlaceDB(engine_data_ieda)
    macroPlaceDB.init_db(params)
    print('init db done')
    # macroPlaceDB -> SA
    placer = SAPlacer(
        max_iters = 1000,
        num_actions = 3000
    )
    # macroPlaceDB -> rl
    # placer = RLPlacer(macroPlaceDB)
    placer.place(macroPlaceDB)
    # SA -> MacroPlaceDB
    placer.writeMPDB(macroPlaceDB)
    # MacroPlaceDB -> iDB
    macroPlaceDB.write_placement_back(params)
    engine_data_ieda.def_save(params.workspace_dir + "/output.def")

def run_RTLMP(params, data_manager: DataManager):
    pass

if __name__ == '__main__':
    params = Params()
    params.workspace_dir = '/home/zhaoxueyan/code/ai-eda/workspace_aimp/workspace_xs_top'
    params.design_name = 'ariane'
    data_manager = DataManager(params.workspace_dir) # data/project_share/Testcases/t28/XSTop_3000x2000_1000M/netlist/XSTop.v
    imp_macro_place(params, data_manager)
    exit(0)
