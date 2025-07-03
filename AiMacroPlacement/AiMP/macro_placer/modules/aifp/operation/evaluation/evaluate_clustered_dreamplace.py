import sys
import logging
import time


from aimp.aifp.operation.evaluation.evaluate_base import EvaluateBase
from aimp.aifp.database.rl_env_db.rl_env_db import RLEnvDB
from aimp.aifp.operation.evaluation.dreamplace_util import dreamplace_core
from aimp.aifp.operation.evaluation.dreamplace_util import dreamplace_util
from aimp.aifp import setting
# from thirdparty.irefactor import py_aifp_cpp as aifp_cpp

class EvaluateClusteredDreamplace(EvaluateBase):
    def __init__(self, env_db:RLEnvDB):

        super(EvaluateClusteredDreamplace, self).__init__()
        logging.root.name = 'aifp:evaluator:clustered-DREAMPlace'
        logging.basicConfig(level=logging.INFO,format='[%(levelname)-7s] %(name)s - %(message)s',stream=sys.stdout)
        
        # self._env_db = env_db
        self._dreamplace_params = dreamplace_util.get_dreamplace_params(
            canvas_width=env_db.get_core().get_width(),
            canvas_height=env_db.get_core().get_height(),
            iteration=setting.evaluator['clustered_dreamplace']['iteration'],
            target_density=setting.evaluator['clustered_dreamplace']['target_density'],
            learning_rate=setting.evaluator['clustered_dreamplace']['learning_rate'],
            num_bins_x=setting.evaluator['clustered_dreamplace']['num_bins_x'],
            num_bins_y=setting.evaluator['clustered_dreamplace']['num_bins_y'],
            gpu=setting.evaluator['clustered_dreamplace']['gpu'],
            result_dir=setting.evaluator['clustered_dreamplace']['result_dir'],
            legalize_flag=setting.evaluator['clustered_dreamplace']['legalize_flag'],
            stop_overflow=setting.evaluator['clustered_dreamplace']['stop_overflow'],
            routability_opt_flag=setting.evaluator['clustered_dreamplace']['routability_opt_flag'],
            num_threads=setting.evaluator['clustered_dreamplace']['num_threads'],
            deterministic_flag=setting.evaluator['clustered_dreamplace']['deterministic_flag'],
        )
        self._dreamplace = dreamplace_core.SoftMacroPlacer(params=self._dreamplace_params)

    def evaluate(self, env_db):
        start_time = time.time()
        print("start clustered-dreamplace")
        # update macro-info
        # self._dreamplace.placedb_plc.read_hard_macros_from_env_db(self._env_db)
        metrics, converge_flag = self._dreamplace.place(env_db)
        print('end dreamplace, converge-flag: {}, time: {} secs'.format(converge_flag, time.time() - start_time))
        result = metrics[-3][0]
        wirelength = float(result[0].hpwl.data)
        overflow = float(result[0].overflow.mean().data)
        return {'wirelength': wirelength, 'overflow': overflow}, converge_flag