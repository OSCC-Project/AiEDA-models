#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : params.py
@author       : Xueyan Zhao (zhaoxueyan131@gmail.com)
@brief        : 
@version      : 0.1
@date         : 2023-11-16 11:20:02
'''

import os
import sys
import json
import math
from collections import OrderedDict
import pdb

class Params:
    """
    @brief Parameter class
    """

    def __init__(self, workspace_dir : str, design_name : str, autodmp_dir : str):
        self.workspace_dir = workspace_dir
        self.design_name = design_name
        self.AutoDMP_root = autodmp_dir
        # self.AutoDMP_root = '/home/zhaoxueyan/code/AutoDMP'
        # self.cmd = './tuner/run_tuner.sh'
        
        self.aux_input = ''
        self.lef_input = ''
        self.def_input = ''
        self.verilog_input = ''
        self.gpu = 0
        self.num_bins_x = 256
        self.num_bins_y = 256
        self.global_place_stages = [{
            'num_bins_x' : 256,
            'num_bins_y' : 256,
            'fence_num_bins_x' : [32, 32],
            'fence_num_bins_y' : [32, 32],
            'iteration' : 2000,
            'learning_rate' : 0.02,
            'wirelength' : 'weighted_average',
            'optimizer' : 'nesterov',
            'Llambda_density_weight_iteration' : 1,
            'Lsub_iteration' : 1
        }]
        self.target_density = 1.0
        self.density_weight = 0.000085
        self.random_seed = 1000
        self.result_dir = 'results'
        self.scale_factor = 1.0
        self.shift_factor = [0.0,0.0]
        self.ignore_net_degree = 100
        self.gp_noise_ratio = 0.025
        self.enable_fillers = 1
        self.global_place_flag = 1
        self.legalize_flag = 1
        self.detailed_place_flag = 0
        self.stop_overflow = 0.1
        self.dtype = 'float32'
        self.detailed_place_engine = '/ntuplace3'
        self.detailed_place_command = ''
        self.plot_flag = 0
        self.RePlAce_ref_hpwl = 350000
        self.RePlAce_LOWER_PCOF = 0.95
        self.RePlAce_UPPER_PCOF = 1.05
        self.gamma = 4.0
        self.RePlAce_skip_energy_flag = 0
        self.random_center_init_flag = 0
        self.sort_nets_by_degree = 0
        self.num_threads = 8
        self.dump_global_place_solution_flag = 0
        self.dump_legalize_solution_flag = 0
        self.routability_opt_flag = 0
        self.route_num_bins_x = 512
        self.route_num_bins_y = 512
        self.node_area_adjust_overflow = 0.15
        self.max_num_area_adjust = 3
        self.adjust_nctugr_area_flag = 0
        self.adjust_rudy_area_flag = 1
        self.adjust_pin_area_flag = 1
        self.area_adjust_stop_ratio = 0.01
        self.route_area_adjust_stop_ratio = 0.01
        self.pin_area_adjust_stop_ratio = 0.05
        self.unit_horizontal_capacity = 1.5625
        self.unit_vertical_capacity = 1.45
        self.unit_pin_capacity = 0.058
        self.max_route_opt_adjust_rate = 2.0
        self.route_opt_adjust_exponent = 2.0
        self.pin_stretch_ratio = 1.414213562
        self.max_pin_opt_adjust_rate = 1.5
        self.deterministic_flag = 1
        self.contest_input = ''
        self.contest_output = ''
        self.check_legal_flag = False

        self.gpu=0
        self.multiobj=2
        self.cfg=3
        self.aux=4
        self.base_ppa=5
        self.reuse_params=6
        self.iterations=7
        self.workers=8
        self.d_ratio=9
        self.c_ratio=10
        self.m_points=11
        self.log_dir=13
        self.auxbase='xx'
        self.cfgSearchFile=''
        self.script_dir='./tuner'
        
        # add additional parameters to fix bugs
        # tbd need a definition checking
        self.macro_halo_x = 0.0
        self.macro_halo_y = 0.0
        self.macro_pin_halo_x = 0.0
        self.macro_pin_halo_y = 0.0
        self.macro_padding_x = 0.0
        self.macro_padding_y = 0.0
        self.bndry_padding_x = 0.0
        self.bndry_padding_y = 0.0
        self.risa_weights = 0.0
        self.pin_density = 0.0
        self.route_info_input = "default"
        
        """
        @brief initialization
        """
        # filename = os.path.join(os.path.dirname(__file__), 'params.json')
        # self.__dict__ = {}
        # params_dict = {}
        # with open(filename, "r") as f:
        #     params_dict = json.load(f, object_pairs_hook=OrderedDict)
        # for key, value in params_dict.items():
        #     if 'default' in value:
        #         self.__dict__[key] = value['default']
        #     else:
        #         self.__dict__[key] = None
        # self.__dict__['params_dict'] = params_dict

    def printWelcome(self):
        """
        @brief print welcome message
        """
        content = """\
========================================================
                       iPL-3D v1.0
            Xueyan Zhao (https://zhaoxueyan.xyz/)
========================================================"""
        print(content)

    def printHelp(self):
        """
        @brief print help message for JSON parameters
        """
        content = self.toMarkdownTable()
        print(content)

    def toMarkdownTable(self):
        """
        @brief convert to markdown table
        """
        key_length = len('JSON Parameter')
        key_length_map = []
        default_length = len('Default')
        default_length_map = []
        description_length = len('Description')
        description_length_map = []

        def getDefaultColumn(key, value):
            if sys.version_info.major < 3: # python 2
                flag = isinstance(value['default'], unicode)
            else: #python 3
                flag = isinstance(value['default'], str)
            if flag and not value['default'] and 'required' in value:
                return value['required']
            else:
                return value['default']

        for key, value in self.params_dict.items():
            key_length_map.append(len(key))
            default_length_map.append(len(str(getDefaultColumn(key, value))))
            description_length_map.append(len(value['descripton']))
            key_length = max(key_length, key_length_map[-1])
            default_length = max(default_length, default_length_map[-1])
            description_length = max(description_length, description_length_map[-1])

        content = "| %s %s| %s %s| %s %s|\n" % (
                'JSON Parameter',
                " " * (key_length - len('JSON Parameter') + 1),
                'Default',
                " " * (default_length - len('Default') + 1),
                'Description',
                " " * (description_length - len('Description') + 1)
                )
        content += "| %s | %s | %s |\n" % (
                "-" * (key_length + 1),
                "-" * (default_length + 1),
                "-" * (description_length + 1)
                )
        count = 0
        for key, value in self.params_dict.items():
            content += "| %s %s| %s %s| %s %s|\n" % (
                    key,
                    " " * (key_length - key_length_map[count] + 1),
                    str(getDefaultColumn(key, value)),
                    " " * (default_length - default_length_map[count] + 1),
                    value['descripton'],
                    " " * (description_length - description_length_map[count] + 1)
                    )
            count += 1
        return content

    def toJson(self):
        """
        @brief convert to json
        """
        data = {}
        for key, value in self.__dict__.items():
            if key != 'params_dict':
                data[key] = value
        return data

    def fromJson(self, data):
        """
        @brief load form json
        """
        for key, value in data.items():
            self.__dict__[key] = value

    def dump(self, filename):
        """
        @brief dump to json file
        """
        with open(filename, 'w') as f:
            json.dump(self.toJson(), f)

    def load(self, filename):
        """
        @brief load from json file
        """
        with open(filename, 'r') as f:
            self.fromJson(json.load(f))

    def __str__(self):
        """
        @brief string
        """
        return str(self.toJson())

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()

    # def design_name(self):
    #     """
    #     @brief speculate the design name for dumping out intermediate solutions
    #     """
    #     if self.aux_input:
    #         design_name = os.path.basename(self.aux_input).replace(".aux", "").replace(".AUX", "")
    #     elif self.verilog_input:
    #         design_name = os.path.basename(self.verilog_input).replace(".v", "").replace(".V", "")
    #     elif self.def_input:
    #         design_name = os.path.basename(self.def_input).replace(".def", "").replace(".DEF", "")
    #     elif self.contest_input:
    #         design_name = os.path.basename(self.contest_input).replace(".contest", "").replace(".CONTEST", "")
    #     return design_name

    def solution_file_suffix(self):
        """
        @brief speculate placement solution file suffix
        """
        if self.def_input is not None and os.path.exists(self.def_input): # LEF/DEF
            return "def"
        elif self.aux_input: # Bookshelf
            return "pl"
        else:
            return "pl"
