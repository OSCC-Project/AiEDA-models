import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from engine.iEDA.ieda_engine import EngineIEDA
from eda_engine.engine.iEDA.module.ieda_engine_dm import DataManager
from engine.iEDA.lib import ieda_py as ieda
# from app import AiMP
from enum import Enum
from abc import ABC, abstractmethod

aifp_root_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(aifp_root_dir)
sys.path.append(aifp_root_dir+'/third_party/ieda/iDB/')
sys.path.append(aifp_root_dir+'/third_party/')

from aifp import setting
from aifp.operation.macro_placer.rl_placer.rl_placer import RLPlacer
from aifp.operation.sa_tunning.simulate_anneal_tunner import SimulateAnnealTunner

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
                 cluster_config:ClusterConfig,
                 solver_config:SolverConfig):
        self.workspace_dir = workspace_dir
        self.cluster_config = cluster_config
        self.solver_config = solver_config


class AiMPFlow():
    def __init__(self, aimp_config:AiMPConfig):
        self.aimp_config = aimp_config
        self.data_manager = DataManager(aimp_config.workspace_dir)
        self.engine_ieda = EngineIEDA(
            design_name = self.data_manager.get_config_manager().get_config_workspace().design,
            path_manager = self.data_manager.get_path_manager())

    def def_save(self, output_path : str, mp_solution = None):
        if mp_solution != None:
            self.engine_ieda.update(mp_solution)
        self.engine_ieda.def_save(output_path)

    def clustering(method:str, params:dict):
        raise NotImplementedError
    
    def evaluate(mp_solution):
        raise NotImplementedError
    
    def create_sa_solver(self, sa_config:SAConfig):
        raise NotImplementedError
    
    def create_analytical_solver(self, analytical_config:AnalyticalConfig):
        raise NotImplementedError
    
    def create_rl_solver(self, rl_config:RLConfig):
        return RLPlacer()
        # rl_placer.run()
    
    
    def run_flow(self):
        # aimp_config = AiMPConfig()
        self.read_design()
        clustered_design = self.clustering(self.aimp_config.cluster_config) # not implemented
        solver = self.create_solver(self.aimp_config.solver_config) # not implemented
        mp_solution = solver.run() # not implemented
        self.def_save('/home/liuyuezuo/aifp_output')
        scores = self.evaluate(mp_solution) # not implemented
        for k, v in scores.items():
            print("{} : {}".format(k, v))


    def create_solver(self, solver_config:SolverConfig):
        if isinstance(solver_config, SAConfig):
            return self.create_sa_solver(solver_config)
        elif isinstance(solver_config, AnalyticalConfig):
            return self.create_analytical_solver(solver_config)
        elif isinstance(solver_config, RLConfig):
            return self.create_rl_solver(solver_config)

if __name__ == '__main__':
    flow = AiMPFlow()
    flow.run_flow()