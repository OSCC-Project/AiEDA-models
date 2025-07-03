#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : workspace_create.py
@author       : Yell
@brief        : 
@version      : 0.1
@date         : 2024-03-28 11:10:16
'''
import sys
import os
import shutil

from workspace.config.json_path import PathParser
from benchmark.task.task_parser import TaskParser
from utility.folder_permission import FolderPermissionManager
from data_manager.aimp_dm import AimpDataManager

class AimpCreateWorkspace():
    def __init__(self, task_dir: str, task_path : str, mp_dir : str):
        self.task_dir = task_dir # aimp task directory
        self.task_path = task_path # task json path
        self.mp_dir = mp_dir # macro placement directory
        self.workspace_example_dir = self.task_dir + "/example"
    
    def get_file_list(self):
        mp_dm = AimpDataManager(self.mp_dir)
        self.file_list = mp_dm.get_aimp_path_manager().get_best_cfgs_tcl()

    def create_workspace(self):
        self.get_file_list()
        
        index = 0
        for file in self.file_list:           
            # copy a workspace
            target_workspace = self.task_dir + "/imp_" + str(index)
            try:
                shutil.copytree(self.workspace_example_dir, target_workspace)
                Success_create = True
            except FileExistsError:
                Success_create = False
                pass
            if Success_create == False:
                index += 1
                continue  
            # share folder
            permission_manager = FolderPermissionManager (target_workspace)
            permission_manager.enable_read_and_write();
            
            # set mp tcl path
            path_json = target_workspace + "/config/path.json"
            parser = PathParser(path_json)
            parser.set_mp_tcl_path(file)
            
            # update task list
            task_parser = TaskParser(self.task_path)
            task_parser.add_task(target_workspace)
            
            index = index + 1