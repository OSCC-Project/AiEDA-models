#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : config.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        : 
@version      : 0.1
@date         : 2023-10-16 19:32:18
'''
from enum import Enum
from abc import ABC, abstractmethod


class SolverConfig():
    def __init__(self, algorithm='hmetis'):
        self.name = None
        pass
    
class BaseSolver():
    def __init__(self):
        pass
    
    @abstractmethod
    def run(self):
        pass

class SAConfig(SolverConfig):
    def __init__(self,
                 representation:str='bstar_tree'):
        self.representation = representation

class AnalyticalConfig(SolverConfig):
    def __init__(self):
        raise NotImplementedError

class RLConfig(SolverConfig):
    def __init__(self):
        raise NotImplementedError

class AiMPConfig():
    def __init__(self,
                 workspace_dir:str,
                 cluster_config,
                 solver_config:SolverConfig):
        self.workspace_dir = workspace_dir
        self.cluster_config = cluster_config
        self.solver_config = solver_config
        