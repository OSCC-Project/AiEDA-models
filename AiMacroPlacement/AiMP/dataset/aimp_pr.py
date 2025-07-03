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
current_dir = os.path.split(os.path.abspath(__file__))[0]
tool_dir = current_dir.rsplit('/', 1)[0]
sys.path.append(tool_dir)
engine_dir = tool_dir + "/eda_engine"
sys.path.append(engine_dir)

from eda_engine.tools.innovus.src.task.run_task_pr import RunTaskPR

class AimpPR():
    def __init__(self, task_path: str):
        self.task_path = task_path

    def run_generate_data(self):
        task_manager = RunTaskPR(self.task_path)
        task_manager.run_all_tasks()

if __name__ == "__main__":
    aimp_pr = AimpPR("/data/project_share/huangzengrong/workspace_aimp/workspace_ariane/aimp_task.json")
    aimp_pr.run_generate_data()

