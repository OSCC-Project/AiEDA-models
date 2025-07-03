#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : Clustering.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        : 
@version      : 0.1
@date         : 2023-10-13 11:01:14
'''

class ClusterConfig():
    def __init__(self,
                 nparts:int,
                 seed:int):

        self.nparts = nparts
        self.seed = seed
    
class HmetisConfig(ClusterConfig):
    def __init__(self,
                 nparts:int,
                 seed:int,
                 ufactor:int,
                 nruns:int=1,
                 dbglvl:int=0,
                 ptype:str='rb',
                 ctype:str='gfc1',
                 rtype:str='moderate',
                 otype:str='cut',
                 reconst:bool=False):
 
        super(HmetisConfig, self).__init__(nparts, seed)
        self.ufactor = ufactor
        self.nruns = nruns
        self.dbglvl = dbglvl
        self.ptype = ptype
        self.ctype = ctype
        self.rtype = rtype
        self.otype = otype
        self.reconst = reconst
        
