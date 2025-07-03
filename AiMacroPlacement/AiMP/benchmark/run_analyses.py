#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : main.py
@author       : Yell
@brief        : 
@version      : 0.1
@date         : 2024-02-28 09:30:06
'''
import sys
import os

current_dir = os.path.split(os.path.abspath(__file__))[0]
root_dir = current_dir.rsplit('/', 1)[0]
sys.path.append(root_dir)
sys.path.append(root_dir + "/third_party/aieda")

from data_manager.aimp_dm import AimpDataManager
from analyses.analysis import Analysis

def init(task_path : str):
    #init aimp
    data_manager = AimpDataManager(task_path)
    
    return data_manager

def run_analyses(self, data_manager : AimpDataManager):
    analyses = Analysis(data_manager)
    analyses.run()

def report(data_manager : AimpDataManager):
    pass

if __name__ == '__main__':
    """run analyze task"""
    workspace_path = "/home/huangzengrong/gitee/ai-mp/workspace/t28_gcd_1"
    
    data_manager = init(workspace_path)
    
    run_analyses(data_manager)
        
    report(data_manager)
    
