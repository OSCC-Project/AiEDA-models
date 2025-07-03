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
import multiprocessing
current_dir = os.path.split(os.path.abspath(__file__))[0]
tool_dir = current_dir.rsplit('/', 1)[0]
sys.path.append(tool_dir)
engine_dir = tool_dir + "/eda_engine"
sys.path.append(engine_dir)

from eda_engine.tools.innovus.src.task.run_task_pr import RunTaskPR

class AimpPR():
    def __init__(self, task_paths: list):
        self.task_paths = task_paths

    def run_task(self, task_path):
        task_manager = RunTaskPR(task_path)
        task_manager.run_all_tasks()

    def run_generate_data(self):
        # 创建进程池
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        
        # 映射run_task方法到所有任务路径上
        pool.map(self.run_task, self.task_paths)
        
        # 关闭进程池并等待所有进程完成
        pool.close()
        pool.join()

if __name__ == "__main__":
    # aimp_pr = AimpPR("/data/project_share/workspace_data/ng45/ariane133_1350x1350_770M/data_1/aimp_task.json")
    task_paths = [
        "/data/project_share/workspace_data/t28/XSTop/XSTop_3000x2000_1000M/AutoDMP_0211_0/aimp_task.json",
        "/data/project_share/workspace_data/t28/XSTop/XSTop_3000x2000_1000M/AutoDMP_0211_1/aimp_task.json",
    ]

    aimp_pr = AimpPR(task_paths)

    aimp_pr.run_generate_data()

