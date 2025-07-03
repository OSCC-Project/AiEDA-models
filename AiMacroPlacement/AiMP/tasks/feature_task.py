#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : analysis.py
@author       : Yell
@brief        : 
@version      : 0.1
@date         : 2024-02-28 09:30:20
'''
import os

from tools.innovus.feature.feature import InnovusFeature
from benchmark.task.run_tasks import RunTasks

from feature.io import FeatureIO
from database.enum import EdaTool, FlowStep, FeatureOption
from flow.flow_db import DbFlow

class FeatureTask():
    """run feature extraction for single task"""
    def __init__(self, task_path : str):
        self.task_path = task_path
   
    def run_feature(self):
        if not os.path.exists(self.task_path):
            return
    
        feature = FeatureIO(dir_workspace = self.task_path,
                            eda_tool = EdaTool.INNOVUS,
                            flow = DbFlow(eda_tool = EdaTool.INNOVUS,step = FlowStep.place))
        feature.generate()


class FeatureTaskList(RunTasks):
    """run feature extraction for all tasks"""   
    def run_feature_for_all_tasks(self):
        """run_all_tasks"""
        for task in self.task_list.task_list :
            # run feature
            self.run_task_feature(task.task_name)
            
    def run_task_feature(self, task_name : str):
        """run feature"""
        feature_task = FeatureTask(task_name)
        feature_task.run_feature()
        
class FeatureTaskFileList():
    """run feature extraction for all task files"""   
    def __init__(self, task_files : list):
        self.task_files = task_files
        
    def run_feature_for_all_files(self):
        """run_all_tasks"""
        print("task number = ", len(self.task_files))
        index = 0
        for file_path in self.task_files : 
            # run task feature
            task_list = FeatureTaskList(file_path)
            task_list.run_feature_for_all_tasks()
            
            index = index + 1
            print("task success, index = ", index)
            
            
            