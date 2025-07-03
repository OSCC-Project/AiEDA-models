#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : config.py
@author       : Yell
@brief        : 
@version      : 0.1
@date         : 2024-02-28 09:30:06
'''

from dataclasses import dataclass
from dataclasses import field
from utility.json_parser import JsonParser

@dataclass
class AimpConfigData(object):
    """database for config"""
    task_name : str = ""

class AimpConfigParser(JsonParser):
    """flow json parser"""
    
    def get_db(self):
        """get data"""
        data = AimpConfigData()
        if self.read() is True:
            #set data
            
            return data
                       
        return None
    
    def set_db(self):
        """set db to json"""
        if self.read() is True:
            data_dict = self.json_data['param']
            #set
            return self.write()
            
        return False


class AimpConfig():
    """config for aimp"""
    def __init__(self, config_path : str):
        self.path = config_path
        parser = AimpConfigParser(self.path)
        self.data = parser.get_db()
        
    def get_data(self):
        return self.data
