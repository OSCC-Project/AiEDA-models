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

from backend.backend_flow_innovus import BackendFlowInnovus

def run_backend_flow(workspace_path : str):
    #run backend flow after imp
    backend_flow = BackendFlowInnovus(workspace_path)
    
    return backend_flow.run_innovus_flow()

if __name__ == '__main__':
    """run backend flow by innovus or pt"""
    workspace_path = "/data/project_share/workspace_data/t28/openC910/openC910_3500x3500_1000M/test/generate_input"
    
    run_backend_flow(workspace_path)
    
