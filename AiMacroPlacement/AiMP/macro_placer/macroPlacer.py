import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from engine.iEDA.ieda_engine import EngineIEDA
from engine.ai_infra.data_manager.data_manager import DataManager
from engine.iEDA.lib import ieda_py as ieda
# from app import AiMP


aifp_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(aifp_root_dir)
sys.path.append(aifp_root_dir+'/third_party/ieda/iDB/')
sys.path.append(aifp_root_dir+'/third_party/')

from aifp import setting
from aifp.operation.macro_placer.rl_placer.rl_placer import RLPlacer
from aifp.operation.sa_tunning.simulate_anneal_tunner import SimulateAnnealTunner
from aimp.macroPlaceDB import MacroPlaceDB
from aimp.macroConfig import *


class AiMPFlow():
    def __init__(self, aimp_config):
        self.aimp_config = aimp_config
        self.data_manager = DataManager(aimp_config.workspace_dir)
        self.engine_ieda = EngineIEDA(
            design_name = self.data_manager.get_config_manager().get_config_workspace().design,
            path_manager = self.data_manager.get_path_manager())
        self.place_db = MacroPlaceDB(self.engine_ieda)

    # def read_design(self):
    #     self.engine_ieda.read_def()

    def def_save(self, output_path : str, mp_solution = None):
        if mp_solution != None:
            self.engine_ieda.update(mp_solution)
        self.engine_ieda.def_save(output_path)

    def clustering(method:str, params:dict):
        raise NotImplementedError
    
    def evaluate(mp_solution):
        raise NotImplementedError
    
    def create_sa_solver(self, placedb, sa_config:SAConfig):
        raise NotImplementedError
    
    def create_analytical_solver(self, placedb, analytical_config:AnalyticalConfig):
        raise NotImplementedError
    
    def create_rl_solver(self, placedb, rl_config:RLConfig):
        return RLPlacer(placedb, rl_config)
        # rl_placer.run()
    
    
    def run_flow(self):
        # aimp_config = AiMPConfig()
        self.place_db.init_db()
        self.place_db.clustering(self.aimp_config.cluster_config)
        
        solver = self.create_solver(self.aimp_config.solver_config) # not implemented
        mp_solution = solver.run() # not implemented
        # self.def_save('/home/liuyuezuo/aifp_output')
        scores = self.evaluate(mp_solution) # not implemented
        for k, v in scores.items():
            print("{} : {}".format(k, v))


    def create_solver(self, placedb, solver_config:SolverConfig):
        if isinstance(solver_config, SAConfig):
            return self.create_sa_solver(placedb, solver_config)
        elif isinstance(solver_config, AnalyticalConfig):
            return self.create_analytical_solver(placedb, solver_config)
        elif isinstance(solver_config, RLConfig):
            return self.create_rl_solver(placedb, solver_config)

if __name__ == '__main__':
    flow = AiMPFlow()
    flow.run_flow()