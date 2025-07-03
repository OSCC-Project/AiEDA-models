#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : aimp_dm.py
@author       : Yell
@brief        : 
@version      : 0.1
@date         : 2024-02-28 10:27:06
'''
from data_manager.path.aimp_path import AimpPath
from data_manager.config.config import AimpConfig
from workspace.path import WorkspacePath

class AimpDataManager():
    """manage all AiMP data"""
    def __init__(self, dir_workspace : str):
        self.dir_workspace = dir_workspace
        self.workspace = WorkspacePath(dir_workspace)
        
        #init aimp workspace path manager
        self.aimp_path_manager = AimpPath(self.dir_workspace + "/aimp/")
        
        #init aimp config
        self.aimp_config = AimpConfig(self.aimp_path_manager.get_config_mp())
        
        self.ieda_io = None
        
    def get_task_name(self):
        return self.dir_workspace
    
    def get_aimp_path_manager(self):
        """manage aimp workspace path"""
        return self.aimp_path_manager
    
    def get_config(self):
        return self.aimp_config
    
    def set_ieda_io(self, ieda_io):
        self.ieda_io = ieda_io

    
    
