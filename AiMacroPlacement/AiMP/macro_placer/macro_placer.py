#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : aimp.py
@author       : Yell
@brief        : 
@version      : 0.1
@date         : 2024-02-29 10:08:06
'''
from asyncio import run_coroutine_threadsafe
from data_manager.aimp_dm import AimpDataManager
from macro_placer.parameters.params import Params
from macro_placer.database.macroPlaceDB import MacroPlaceDB
from macro_placer.modules.safp.run_safp import RunSAPlacer
from macro_placer.modules.reffp.run_reffp import RunRefPlacer
from macro_placer.modules.safp.sa_placer import SAPlacer
# from macro_placer.modules.rlFP.rl_placer import RLPlacer
from AutoDMP.tuner.run_tuner_train import RunTunerTrain
from tools.iEDA.module.placement import IEDAPlacement

class MacroPlacer():
    """run aimp flow"""
    def __init__(self, data_manager : AimpDataManager):
        self.data_manager = data_manager
        
    def run(self):
        print("macro_placer.py: run")
        self.init()
        
        # self.run_imp_sa()
        self.run_imp_ieda()
        
        
        # self.run_autoDMP()
        
        # self.run_imp_rl()
        
        # self.run_imp_refinement()
        
        return True
    
    def init(self):
        self.init_param()
        
    def init_param(self):
        workspace_path = self.data_manager.workspace
        workspace_dir = workspace_path.workspace
        design_name = workspace_path.json_workspace.design
        autodmp_dir = self.data_manager.get_aimp_path_manager().get_autodmp_code_dir()
        # self.params = Params(workspace_dir, design_name, autodmp_dir)
            
    # def run_imp_sa(self):
    #     macroPlaceDB = MacroPlaceDB(self.engine_ieda)
    #     macroPlaceDB.init_db(self.params)
    #     print('init db done')
    #     # macroPlaceDB -> SA
    #     placer = SAPlacer(
    #         self.data_manager.workspace
    #         max_iters = 1000,
    #         num_actions = 3000
    #     )
    #     # macroPlaceDB -> rl
    #     # placer = RLPlacer(macroPlaceDB)
    #     placer.place(macroPlaceDB)
    #     # SA -> MacroPlaceDB
    #     placer.writeMPDB(macroPlaceDB)
    #     # MacroPlaceDB -> iDB
    #     macroPlaceDB.write_placement_back(self.params)
        
    def run_imp_ieda(self):
        workspace_path = self.data_manager.workspace
        output_tcl = workspace_path.json_path.mp_tcl
        config = self.data_manager.aimp_path_manager.get_config_mp()
        
        workspace = self.data_manager.workspace
        ieda_placement = IEDAPlacement(dir_workspace = workspace.workspace,
                              input_def = workspace.json_path.def_input_path)
        run_ieda_placer = RunSAPlacer(ieda_placement)
        run_ieda_placer.run(config=config, tcl_path=output_tcl)
        
    def run_imp_refinement(self):
        workspace_path = self.data_manager.workspace
        output_tcl = workspace_path.json_path.mp_tcl
        
        workspace = self.data_manager.workspace
        ieda_placement = IEDAPlacement(dir_workspace = workspace.workspace,
                              input_def = workspace.json_path.def_input_path)
        run_ieda_placer = RunRefPlacer(ieda_placement)
        run_ieda_placer.run(tcl_path=output_tcl)
    
    def run_imp_rl(self):
        pass
        # rl_placer = RLPlacer()
        # rl_placer.run()
        
    def run_autoDMP(self):
        # work_dir = "/data/project_share/benchmark/aimp/Top_high_2100x1500_800M/"
        # self.engine_ieda.get_engine_dm().get_ieda_engine().set_design_workspace(work_dir + "rpt")
        # self.engine_ieda.get_engine_dm().get_ieda_engine().read_netlist(work_dir + "input/Top_place_opt_nopg.v")
        # self.engine_ieda.get_engine_dm().get_ieda_engine().read_liberty([
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140hvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140lvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140mbssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140mblvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140oppssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140opphvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140opplvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140oppuhvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140oppulvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140uhvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140hvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140lvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140mbssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140mblvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140oppssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140opphvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140opplvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140oppuhvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140oppulvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140uhvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp40p140ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp40p140hvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp40p140lvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp40p140mbssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp40p140mbhvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp40p140oppssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp40p140opphvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp40p140opplvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp40p140oppuhvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp40p140uhvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140mbhvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp30p140ulvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/tcbn28hpcplusbwp35p140ulvtssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/mem/ts5n28hpcplvta64x128m2f_130a_ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/mem/ts5n28hpcplvta64x128m2fw_130a_ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/ccslib/ts5n28hpcplvta256x32m4fw_130a_ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/mem/ts5n28hpcplvta128x32m2f_130a_ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/mem/ts5n28hpcplvta128x64m2f_130a_ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/mem/ts5n28hpcplvta128x80m2f_130a_ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/mem/ts5n28hpcplvta128x8m2f_130a_ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/mem/ts5n28hpcplvta512x64m4f_130a_ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/mem/ts5n28hpcplvta64x32m2f_130a_ssg0p81v125c.lib",
        #     "/data/project_share/process_node/T28_lib/mem/ts5n28hpcplvta64x8m2f_130a_ssg0p81v125c.lib"
        # ]                                                               
        # )
        # self.engine_ieda.get_engine_dm().get_ieda_engine().link_design("Top")
        # self.engine_ieda.get_engine_dm().get_ieda_engine().read_sdc(work_dir + "input/Top_SYN_TYP_ieda.sdc")
        # def_path = self.data_manager.get_engine_dm().get_path_manager().get_design().config_path.def_input_path
        # self.engine_ieda.get_engine_dm().read_def(def_path)


        # exit()
        run_tuner = RunTunerTrain(self.data_manager)
        run_tuner.run()