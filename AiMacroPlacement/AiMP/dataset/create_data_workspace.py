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
engine_dir = tool_dir + "/eda_engine"
sys.path.append(engine_dir)

from eda_engine.ai_infra.task.run_tasks import RunTasks
from eda_engine.ai_infra.config.json_parser.path_parser import PathParser
from eda_engine.ai_infra.config.json_parser.task_parser import TaskParser
from eda_engine.ai_infra.data_manager.path.permission.folder_permission import FolderPermissionManager

class AimpCreateWorkspace():
    def __init__(self, workspace: str, task_path : str):
        self.workspace = workspace
        self.task_path = task_path

    def create_workspace(self):
        imp_tcl_dir = self.workspace + "/imp_tcl"
        workspace_example_dir = os.path.join(os.path.dirname(self.workspace), "example")

        file_list = os.listdir(imp_tcl_dir)
        
        for file in file_list:
            # 解析文件名以构造目标工作空间名称
            parts = file.split('macro_loc')
            if len(parts) == 2:
                prefix = parts[0].rstrip('_')  # 移除末尾的下划线
                suffix = parts[1].lstrip('_').split('.')[0]  # 移除开头的下划线和扩展名
                target_workspace_name = f"{prefix}_{suffix}"
            else:
                print(f"Unexpected file name format: {file}")
                continue  # 如果文件名不包含'macro_loc'，或格式不符，跳过这个文件
            
            target_workspace = os.path.join(self.workspace, target_workspace_name)
            file_path = os.path.join(imp_tcl_dir, file)
            
            try:
                shutil.copytree(workspace_example_dir, target_workspace)
                # 设置目标工作空间的权限
                permission_manager = FolderPermissionManager(target_workspace)
                permission_manager.enable_read_and_write()
                
                # 设置mp tcl路径
                path_json = os.path.join(target_workspace, "config/path.json")
                parser = PathParser(path_json)
                parser.set_mp_tcl_path(file_path)
                
                # 更新任务列表
                task_parser = TaskParser(self.task_path)
                task_parser.add_task(target_workspace)
            except FileExistsError:
                print(f"Workspace {target_workspace} already exists. Skipping.")
                continue  # 如果目标工作空间已存在，打印信息并跳过
            
if __name__ == "__main__":
    workspaces_info = [
        ("/data/project_share/workspace_data/t28/XSTop/XSTop_3000x2000_1000M/AutoDMP_0211_{}".format(i), 
         "/data/project_share/workspace_data/t28/XSTop/XSTop_3000x2000_1000M/AutoDMP_0211_{}/aimp_task.json".format(i)) 
        for i in range(12)
    ]
    print(workspaces_info)
    for workspace_dir, task_path in workspaces_info:
        aimp_create_workspace = AimpCreateWorkspace(workspace_dir, task_path)
        aimp_create_workspace.create_workspace()


