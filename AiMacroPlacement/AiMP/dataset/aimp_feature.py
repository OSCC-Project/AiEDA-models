#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File : aimp_trainging.py
@Author : yell
@Desc : AI Macro Placement model training
'''

######################################################################################
# import ai-eda as root

######################################################################################
import sys
import os
current_dir = os.path.split(os.path.abspath(__file__))[0]
tool_dir = current_dir.rsplit('/', 1)[0]
sys.path.append(tool_dir)
engine_dir = tool_dir + "/third_party/aieda"
sys.path.append(engine_dir)

from eda_engine.ai_infra.feature.task.run_task_feature_generate import RunTaskFeatureGenerate
from eda_engine.ai_infra.feature.task.run_task_feature_read import RunTaskFeatureRead
from eda_engine.ai_infra.feature.task.run_task_feature_plot import RunTaskFeaturePlot
from eda_engine.ai_infra.data_manager.data_manager import DataManager

class AiMPFeature():
    def __init__(self, task_path: str):
        self.task_path = task_path

    def run_generate_feature(self):
        task_manager = RunTaskFeatureGenerate(self.task_path)
        task_manager.run_all_tasks()

    def run_read_feature(self):
        task_manager = RunTaskFeatureRead(self.task_path)
        task_manager.run_all_tasks()

    def run_plot_gui(self):
        task_manager = RunTaskFeaturePlot(self.task_path)
        task_manager.run_all_tasks()

if __name__ == "__main__":
    aimp_training = AiMPFeature("/data/project_share/huangzengrong/workspace_aimp/workspace_ariane/aimp_task.json")
    aimp_training.run_generate_feature()
    # aimp_training.run_plot_gui()
    # aimp_training.run_read_feature()

