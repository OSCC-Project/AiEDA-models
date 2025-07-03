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

os.environ['eda_tool'] = "iEDA"

current_dir = os.path.split(os.path.abspath(__file__))[0]
root_dir = current_dir.rsplit('/', 1)[0]
sys.path.append(root_dir)
sys.path.append(root_dir + "/third_party/aieda")

from data_manager.aimp_dm import AimpDataManager
from tasks.aimp_task import AimpTask
import subprocess
 
def get_loaded_libraries(pid):
    cmd = ['lsof', '-n', '-p', str(pid), '-l']
    result = subprocess.check_output(cmd).decode()
    libraries = [line.split()[-1] for line in result.splitlines() if '.so' in line]
    return libraries
 
def my_lib():
    pid = os.getpid()  # 获取当前进程的PID
    loaded_lib = get_loaded_libraries(pid)
    for e in loaded_lib:
        print("loaded_lib=",e)

if __name__ == '__main__':
    """single task"""
    # my_lib()
    # workspace_path = "/data/project_share/benchmark/aimp/ariane133_1350x1350_770M/test1"
    # workspace_path = "/data/project_share/benchmark/aimp/autoDMP/ariane/ariane133_1350x1350_770M/ariane133_1350x1350_770M_test"
    # workspace_path = "/home/zhaoxueyan/code/ai-mp/workspace_aimp/workspace_NutShell"
    workspace_path = "/data/project_share/benchmark/aimp/autoDMP/asic_top/asic_top_2500x2500_900M_test"
    # workspace_path = "/home/zhaoxueyan/code/ai-mp/workspace_aimp/ariane133"
    # workspace_path = "/data/project_share/benchmark/aimp/autoDMP/ariane/ariane133_1350x1350_770M/ariane133_1350x1350_770M_0408"
    # workspace_path = "/data/project_share/benchmark/aimp/autoDMP/ariane/ariane136_1700x1700_770M/ariane136_1700x1700_770M_0408"
    # workspace_path = "/data/project_share/benchmark/aimp/autoDMP/mempool/mempool_tile_wrap_900x900_147M/mempool_tile_wrap_900x900_147M_0408"
    # workspace_path = "/data/project_share/benchmark/aimp/autoDMP/NV_NVDLA_partition_c/NV_NVDLA_partition_c_2350x2350_1111M/NV_NVDLA_partition_c_2350x2350_1111M_0408"
    # workspace_path = "/data/project_share/benchmark/aimp/autoDMP/openC910/openC910_3500x3500_1000M_0614"
    # workspace_path = "/data/project_share/benchmark/aimp/autoDMP/asic_top/asic_top_2500x2500_25M_test"

    #init aimp
    data_manager = AimpDataManager(workspace_path)
    
    aimp_task = AimpTask(data_manager)
    aimp_task.run_mp()
    
    exit(0)
    
