#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : run_feature.py
@author       : Yell
@brief        : 
@version      : 0.1
@date         : 2024-03-15 10:31:01
'''
import sys
import os
os.environ['eda_tool'] = "iEDA"

current_dir = os.path.split(os.path.abspath(__file__))[0]
root_dir = current_dir.rsplit('/', 1)[0]
sys.path.append(root_dir)
sys.path.append(root_dir + "/third_party/aieda")

from tasks.feature_task import FeatureTask, FeatureTaskList, FeatureTaskFileList
from third_party.aieda.tools.innovus.feature.parser.feature_drc import FeatureDRC

from third_party.aieda.tools.iEDA.feature.feature import IEDAFeature
from third_party.aieda.tools.innovus.utility.tcl.generate_tcl import GenerateTCLInnovus

def run_workspace_feature(workspace : str):
    """run a single aimp task"""   
    feature_task = FeatureTask(workspace)
    feature_task.run_feature()

def run_task_feature(task_name : str):
    """single task that including some workspaces"""
    task_list = FeatureTaskList(task_name)
    task_list.run_feature_for_all_tasks()
    
def get_all_task_files(aimp_dir):
    """get all task json files in aimp_dir"""
    task_file_list = []
    for root, dirs, files in os.walk(aimp_dir):                  
        for dir in dirs:
            task_path = root +"/" + dir + "/aimp_task.json"
            if(os.path.exists(task_path)):
                task_file_list.append(task_path)
    
    return task_file_list

def run_all_task_files(task_files : list):
    """run feature for all task files"""
    feature_task_files = FeatureTaskFileList(task_files)
    feature_task_files.run_feature_for_all_files()
    
def run_aimp_all_task():
    # aimp_dir_list = ["/data/project_share/workspace_data/t28/XSTop/XSTop_3000x2000_1000M", 
    #                  "/data/project_share/workspace_data/t28/XSTop/XSTop_high_3000x2000_1000M"]
    aimp_dir_list = ["/data/project_share/workspace_data/t28/XSTop/XSTop_3000x2000_1000M"]
    for aimp_dir in aimp_dir_list:
        task_file_list = get_all_task_files(aimp_dir)
        run_all_task_files(task_file_list)
        
def run_feature_drc():
    report_path = "/data/project_share/huangzhipeng/Flow_T28/pd_data/pr/rpt/iMP_test_0718_86529777664_2610/route/asic_top.verify_drc.rpt"
    drc_json = "/data/project_share/huangzhipeng/Flow_T28/pd_data/pr/rpt/iMP_test_0718_86529777664_2610/workspace/output/iEDA/feature/asic_top.verify_drc.json" 
    feature_drc = FeatureDRC(json_path=drc_json, report_path=report_path, ignore_pdn=True)
    feature_drc.feature()
    
def run_feature_macro_drc_distribution():

    drc_json = "/data/project_share/huangzhipeng/Flow_T28/pd_data/pr/rpt/iMP_test_0718_86529777664_2610/workspace/output/iEDA/feature/asic_top.verify_drc.json" 
    macro_drc_distribution_json = "/data/project_share/huangzhipeng/Flow_T28/pd_data/pr/rpt/iMP_test_0718_86529777664_2610/workspace/output/iEDA/feature/macro_drc_distribution.json" 
    
    workspace_dir = "/data/project_share/huangzhipeng/Flow_T28/pd_data/pr/rpt/iMP_test_0718_86529777664_2610/workspace"
    input_def = "/data/project_share/huangzhipeng/Flow_T28/pd_data/pr/output/asic_top_test.def"
    feature_ieda = IEDAFeature(dir_workspace=workspace_dir,input_def=input_def)
    feature_ieda.read_def(input_def)
    # feature_ieda.feature_summary(output_path="/data/project_share/huangzhipeng/Flow_T28/pd_data/pr/rpt/iMP_test_0718_86529777664_2610/workspace/output/iEDA/feature/summary.json")
    feature_ieda.feature_macro_drc_distribution(path = macro_drc_distribution_json,drc_path=drc_json)
    
    ouput_tcl="/data/project_share/huangzhipeng/Flow_T28/pd_data/pr/rpt/iMP_test_0718_86529777664_2610/workspace/output/iEDA/feature/add_blockage.tcl"
    get_tcl = GenerateTCLInnovus(workspace_dir, "")
    get_tcl.generate_tcl_blockage(macro_drc_json=macro_drc_distribution_json,ouput_tcl=ouput_tcl)


if __name__ == '__main__':
    # single workspace
    #run_workspace_feature("/data/project_share/workspace_data/t28/XSTop/XSTop_high_3000x2000_1000M/AutoDMP_0303_0/XSTop_93308616704_20")
    
    # single task that including some workspaces
    # run_task_feature("/data/project_share/workspace_data/t28/openC910/openC910_3500x3500_1000M/AutoDMP_0616_2/aimp_task.json")
    
    # run feature extraction task for all aimp workspace list
    # run_aimp_all_task()
    
    # run drc
    run_feature_drc()
    run_feature_macro_drc_distribution()
    
    exit(0)

    
