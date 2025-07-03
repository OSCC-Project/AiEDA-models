# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import time
import torch
import shutil
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import MOBOHB as MOBOHB

opj = os.path.join

# disable heavy logging from C++
os.environ["DREAMPLACE_DISABLE_PRINT"] = "0"

from AutoDMP.tuner.tuner_worker import AutoDMPWorker
from AutoDMP.tuner.tuner_utils import str2bool
from AutoDMP.tuner.tuner_analyze import get_candidates, plot_pareto

from dataclasses import dataclass

from utility.json_parser import JsonParser
from multiprocessing import Process

import socket
from contextlib import closing

from data_manager.aimp_dm import AimpDataManager

@dataclass
class TrainConfig(object):
    """training config"""
    nameserver_port : int = 0
    tasks = []
    
@dataclass
class TaskConfig(object):
    """task config"""
    state : str = "" # options : cfgSearchFile, work
    multiobj : bool = False
    cfgSearchFile : str = "",
    min_points_in_model : int= 32,
    min_budget : int = 1,
    max_budget : int = 1,
    n_iterations : int = 200,
    n_workers : int = 8,
    n_samples : int = 64,
    congestion_ratio : float = 0.5,
    density_ratio : float = 0.5,
    num_pareto : int = 5,
    log_dir : str = "logs_tuner",
    run_id : str = 0,
    run_args = {}
    worker : bool = False,
    worker_id : int = 0,
    gpu_pool = [-1]
    

class ConfigParser(JsonParser):
    """flow json parser"""
    def find_available_port(self, start_at=1024):
        max_port = 65535
        for port in range(start_at, max_port + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('127.0.0.1', port))
                sock.close()
                return port
            except socket.error as e:
                if e.errno == 98:  # EADDRINUSE
                    continue
                else:
                    raise e
        raise Exception('No available port found')

    
    def get_db(self, data_manager : AimpDataManager):
        """get data"""
        
        if self.read() is True:
            config = TrainConfig()
            # find port
            config.nameserver_port = self.find_available_port()
            if config.nameserver_port <= 0 :
                print("Error, no available port exist.")
                exit(0)
            
            path_manager = data_manager.get_aimp_path_manager()
            
            node_task_dict = self.json_data['tasklist']
            for task_dict in node_task_dict:
                task_config = TaskConfig()
    
                task_config.state = task_dict['state']
                task_config.multiobj = str2bool(task_dict['multiobj'])
                task_config.cfgSearchFile = path_manager.get_autodmp_configspace()
                task_config.min_points_in_model = task_dict['min_points_in_model']
                task_config.min_budget = task_dict['min_budget']
                task_config.max_budget = task_dict['max_budget']
                task_config.n_iterations = task_dict['n_iterations']
                task_config.n_workers = task_dict['n_workers']
                task_config.n_samples = task_dict['n_samples']
                task_config.congestion_ratio = task_dict['congestion_ratio']
                task_config.density_ratio = task_dict['density_ratio']
                task_config.num_pareto = task_dict['num_pareto']
                task_config.log_dir = path_manager.get_autodmp_log_dir()
                task_config.run_id = task_dict['run_id']
                task_config.worker = str2bool(task_dict['worker'])
                task_config.worker_id = task_dict['worker_id']
                task_config.gpu_pool = task_dict['gpu_pool']
                
                node_run_args = task_dict['run_args']
                task_config.run_args["gpu"] = node_run_args['gpu']
                task_config.run_args["gpu_id"]  = node_run_args['gpu_id']
                task_config.run_args["aux_input"] = path_manager.get_autodmp_aux()
                task_config.run_args["base_ppa"] = path_manager.get_autodmp_ppa()
                task_config.run_args["reuse_params"] = node_run_args['reuse_params']
                
                config.tasks.append(task_config)
        
        return config
    
class RunTunerTrain:
    """run autoDMP training api"""
    def __init__(self, data_manager : AimpDataManager):
        self.data_manager = data_manager
        
    def parse_config(self):
        config_path = self.data_manager.get_aimp_path_manager().get_autodmp_tuner_train()
        parser = ConfigParser(config_path)
        return parser.get_db(self.data_manager)
    
    def run(self):
        process_list = []
        
        # parse json to data structure
        self.config = self.parse_config()
        for task_args in self.config.tasks:
            # run task
            if task_args.state == "cfgSearchFile":
                process_list.append(self.run_master_process(task_args))
            elif task_args.state == "work":
                process_list.append(self.run_worker_process(task_args))
            else:
                print("warning : state must be set in the task config.")
                pass
        for p in process_list :
            p.start()
        for p in process_list :
            p.join()
    
    def run_master_process(self, task_args):
        # Worker
        p = Process(target=self.master_process, args=(task_args,))
        return p

    def master_process(self, args):
        # Master
        print("Starting master process")
        print(f"Master args: {args}")
        
        # Create Log directory
        os.makedirs(args.log_dir, exist_ok=True)
        result_logger = hpres.json_result_logger(directory=args.log_dir, overwrite=True)
        
        # Start a nameserver
        NS = hpns.NameServer(run_id=args.run_id, host="127.0.0.1", port=self.config.nameserver_port)
        NS.start()
        
        # Run an optimizer
        if args.multiobj:
            motpe_params = {
                "init_method": "random",
                "num_initial_samples": 10,
                "num_candidates": 24,
                "gamma": 0.10,
            }
            bohb = MOBOHB(
                configspace=AutoDMPWorker.get_configspace(args.cfgSearchFile),
                parameters=motpe_params,
                run_id=args.run_id,
                min_points_in_model=args.min_points_in_model,
                min_budget=args.min_budget,
                max_budget=args.max_budget,
                num_samples=args.n_samples,
                result_logger=result_logger,
                nameserver_port=self.config.nameserver_port,
            )
        else:
            bohb = BOHB(
                configspace=AutoDMPWorker.get_configspace(args.cfgSearchFile),
                run_id=args.run_id,
                min_points_in_model=args.min_points_in_model,
                min_budget=args.min_budget,
                max_budget=args.max_budget,
                num_samples=args.n_samples,
                result_logger=result_logger,
                nameserver_port=self.config.nameserver_port,
            )
        res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)
        
        # Shutdown
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()
        
        # Analysis
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        
        print("A total of %i unique configurations where sampled." % len(id2config.keys()))
        print("A total of %i runs where executed." % len(res.get_all_runs()))
        all_runs = res.get_all_runs()
        print(
            "The run took %.1f seconds to complete."
            % (all_runs[-1].time_stamps["finished"] - all_runs[0].time_stamps["started"])
        )
        
        # Propose Pareto points
        dp_dir, netlist = os.path.split(args.run_args["aux_input"])
        netlist = netlist.replace(".aux", "")
        result = hpres.logged_results_to_HBS_result(args.log_dir)
        candidates, paretos, df = get_candidates(result, num=args.num_pareto)
        print("Pareto candidates are:")
        print(candidates.to_markdown())
        
        # Save Pareto configs and generate DEFs
        print("Generating DEF files")
        dest = opj(args.log_dir, "best_cfgs")
        os.makedirs(dest, exist_ok=True)
        df.to_pickle(opj(dest, f"{netlist}.dataframe.pkl"))
        for _, row in candidates.iterrows():
            cfg_id = "run-" + "_".join([s for s in re.findall(r"\b\d+\b", row["ID"])])
            print(f"Generating DEF for candidate {cfg_id}")
            src_cfg, dest_cfg = opj(args.log_dir, cfg_id), opj(dest, cfg_id)
            shutil.copytree(src_cfg, dest_cfg, dirs_exist_ok=True)
            # generate DEF
            # def_file = opj(dp_dir, f"{netlist}.ref.def")
            # macro_file = opj(dp_dir, f"{netlist}.macros")
            # pl_file = opj(src_cfg, netlist, f"{netlist}.gp.pl")
            # new_def_file = opj(dest_cfg, f"{netlist}.AutoDMP.def")
            # dp_to_def(def_file, pl_file, macro_file, new_def_file)
        
        # Plot Pareto curve
        plot_pareto(df, paretos, candidates, opj(dest, f"{netlist}.pareto.png"))
            
    def run_worker_process(self, task_args):
        # Worker
        p = Process(target=self.worker_process, args=(task_args,))
        # p.start()
        return p

    def worker_process(self, args):
        # Worker
        print(f"Starting worker process number {args.worker_id}")
        time.sleep(5)  # artificial delay to make sure the nameserver is already running
        print(f"Worker args: {args}")
    
        if args.run_args["gpu"] == "1":
            # alternate the gpu_id of workers
            if args.gpu_pool == [-1]:
                available_gpus = range(torch.cuda.device_count())
            else:
                available_gpus = args.gpu_pool
                assert all(g < torch.cuda.device_count() for g in available_gpus)
            gpu_id = available_gpus[args.worker_id % len(available_gpus)]
            args.run_args["gpu_id"] = gpu_id
            print(f"Assigning worker {args.worker_id} to GPU {gpu_id}")
    
        w = AutoDMPWorker(
            nameserver="127.0.0.1",
            run_id=args.run_id,
            log_dir=args.log_dir,
            congestion_ratio=args.congestion_ratio,
            density_ratio=args.density_ratio,
            default_config=args.run_args,
            multiobj=args.multiobj,
            nameserver_port=self.config.nameserver_port,
            data_manager=self.data_manager
        )
        w.run(background=False)
        exit(0)
