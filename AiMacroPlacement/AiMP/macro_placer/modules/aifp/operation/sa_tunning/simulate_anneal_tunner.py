import multiprocessing
import numpy as np
import os

from aimp.aifp.database.rl_env_db.rl_env_db import RLEnvDB
from aimp.aifp.solver.simulate_anneal.simulate_anneal import SimulateAnneal
from aimp.aifp.operation.sa_tunning.aifp_simulate_anneal_db import AifpSimulateAnnealDB
from aimp.aifp import setting
from aimp.aifp.operation.data_io import report_io, aifp_db_io

from aimp.aifp.operation.evaluation import evaluate_dreamplace
from aimp.aifp.operation.evaluation import evaluate_macro_io_wirelength
from aimp.aifp.operation.evaluation import evaluate_clustered_dreamplace

class SimulateAnnealTunner:
    def __init__(self):
        self._sa_db = self._init_sa_db()
        self._sa_solver = SimulateAnneal(
            max_num_step = setting.simulate_anneal['max_num_step'],
            perturb_per_step = setting.simulate_anneal['perturb_per_step'],
            init_pro = setting.simulate_anneal['init_prob'])

    def run(self):
        self._sa_solver.run(
            db = self._sa_db,
            log_dir = '{}/{}/{}/run{}'.format(os.environ['AIFP_PATH'], setting.log['log_dir'], setting.case_select, setting.log['run_num'])
        )

    def _init_sa_db(self):
        idb_config_path = '{}/input/{}/irefactor_idb_config.json'.format(os.environ['AIFP_PATH'], setting.case_select)
        design_data_dict = aifp_db_io.read_from_aifp_db_and_destroy(idb_config_path)
        # macro_info = report_io.extract_macro_info_from_report('/root/reconstruct-aifp/log/ariane133/run2/RLPlacer.rpt')
        fp_solution = report_io.extract_fp_solution_from_report('/root/reconstruct-aifp/log/ariane133/run2/RLPlacer.rpt')


        env_db = RLEnvDB(design_data_dict, mode='SA')
        env_db.update_origin_macro_info(fp_solution)
        env_db.reset(0)

        evaluator = self._init_evaluator(env_db)
        sa_db = AifpSimulateAnnealDB(env_db, evaluator)
        return sa_db

    def _init_evaluator(self, env_db):
        evaluator_name = setting.simulate_anneal['evaluator']
        if evaluator_name == 'clustered_dreamplace':
            return evaluate_clustered_dreamplace.EvaluateClusteredDreamplace(env_db=env_db)
        else:
            raise NotImplementedError