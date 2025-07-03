#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : analysis.py
@author       : Yell
@brief        : 
@version      : 0.1
@date         : 2024-02-28 09:30:20
'''

from data_manager.aimp_dm import AimpDataManager
from macro_placer.macro_placer import MacroPlacer
from analyses.analysis import Analysis
from tools.iEDA.module.io import IEDAIO

from multiprocessing import Process

class AimpTask():
    """"""
    def __init__(self, data_manager : AimpDataManager):
        self.data_manager = data_manager
        
    # def run_aimp_flow(self):
    #     """macro placer main flow"""    
    #     #run imp as independent process
    #     self.run_mp_process()
    
    #     #run backend flow after imp
    #     self.run_backend_flow()
    
    #     #run feature extraction
    #     self.run_feature()
    
    #     #run analyze
    #     return self.run_analyze()
    
    def run_mp(self):
        """run macro placer as process"""
        p = Process(target=self.run_mp_process, args=(self.data_manager, ))
        p.start()
        p.join()
    
    def run_mp_process(self, data_manager : AimpDataManager):
        # build iEDA engine
        workspace = data_manager.workspace
    
        ieda_io = IEDAIO(dir_workspace = workspace.workspace,
                              input_def = workspace.json_path.def_input_path)
        data_manager.set_ieda_io(ieda_io)
        
        #read input def in path.json
        # def_path = data_manager.get_engine_dm().get_path_manager().get_design().config_path.def_input_path
        ieda_io.read_def() # 有必要吗？
        
        """run mp flow"""
        aimp_flow = MacroPlacer(data_manager)
        aimp_flow.run()
        
        #write mp result def
        ieda_io.def_save(self.data_manager.get_aimp_path_manager().get_output_def())

    def run_feature(self):
        pass

    def run_analyze(self):
        aimp_analysis = Analysis(self.data_manager)
        aimp_analysis.run()
        
    def report(self):
        pass