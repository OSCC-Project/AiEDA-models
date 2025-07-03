#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File : aimp_pr.py
@Author : yell
@Desc : generate PR data
'''

######################################################################################
# import ai-eda as root

######################################################################################
import sys
import os
import shutil

current_dir = os.path.split(os.path.abspath(__file__))[0]
tool_dir = current_dir.rsplit('/', 1)[0]
sys.path.append(tool_dir)
engine_dir = tool_dir + "/third_party/aieda"
sys.path.append(engine_dir)

from data_manager.workspace.workspace_create import AimpCreateWorkspace
            
        
if __name__ == "__main__":
    task_dir = "/data/project_share/benchmark/aimp/test_flow"
    task_path = "/data/project_share/benchmark/aimp/test_flow/aimp_task.json"
    mp_dir = "/data/project_share/benchmark/aimp/autoDMP/workspace_xs_top"
    workspace_creator = AimpCreateWorkspace(task_dir, task_path, mp_dir)
    workspace_creator.create_workspace()

