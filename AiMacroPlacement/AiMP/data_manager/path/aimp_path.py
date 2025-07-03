#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : aimp_path.py
@author       : Yell
@brief        : 
@version      : 0.1
@date         : 2024-02-28 10:30:06
'''
import os

class AimpPath():
    """manage aimp path in workspace"""
    def __init__(self, aimp_dir : str):
        self.aimp_dir = aimp_dir
        
    def get_autodmp_code_dir(self):
        """get directory for autodmp code"""
        current_dir = os.path.split(os.path.abspath(__file__))[0]
        self.resource_path = current_dir.rsplit('/',2)[0]
        return self.resource_path + "/AudoDMP"
        
    def get_aimp_dir(self):
        return self.aimp_dir
    
    #config manager
    def get_dir_config(self):
        return self.aimp_dir + "config/"
    
    def get_config_mp(self):
        return self.get_dir_config() + "mp_config.json"
    
    # output manager
    def get_dir_output(self):
        return self.aimp_dir + "output/"
        
    def get_output_def(self):
        return self.get_dir_output() + "mp.def"
    
    def get_output_verilog(self):
        return self.get_dir_output() + "mp.v"
    
    #auto dmp manager
    #config
    def get_autodmp_tuner_train(self):
        return self.get_dir_config() + "tuner_train.json"
    
    def get_autodmp_ppa(self):
        return self.get_dir_config() + "ppa.json"
    
    def get_autodmp_configspace(self):
        return self.get_dir_config() + "configspace.json"
    
    def get_autodmp_aux(self):
        return self.get_dir_config() + "mp.aux"
    
    def get_autodmp_log_dir(self):
        return self.aimp_dir + "log/"
    
    # best config manager
    def get_dir_best_cfgs(self):
        return self.get_autodmp_log_dir() + "best_cfgs/"
    
    def get_best_cfgs_tcl(self):
        cfg_list = []
        
        cfg_dir = self.get_dir_best_cfgs()
        self.recursive(cfg_dir,cfg_list)
        
        return cfg_list
        
    def recursive(self, cfg_dir : str, cfg_list):            
        for root, dirs, files in os.walk(cfg_dir):           
            for file in files:
                if file.endswith(".tcl"):
                    cfg_list.append(os.path.join(root, file))
                
            for dir in dirs:
                self.recursive(dir, cfg_list)
             
    
