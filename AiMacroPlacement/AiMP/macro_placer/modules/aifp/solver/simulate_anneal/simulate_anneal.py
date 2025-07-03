import numpy as np
import math
import logging
import os
from aimp.aifp import setting
from torch.utils.tensorboard import SummaryWriter

class SimulateAnneal:
    def __init__(
            self, 
            max_num_step:int = 100,
            perturb_per_step:int = 200,
            cool_rate: float = 0.92,
            init_pro: float = 0.95,
            init_temperature: float = None):
        
        self._max_num_step = max_num_step
        self._perturb_per_step = perturb_per_step
        self._cool_rate = cool_rate
        self._init_pro = init_pro
        self._init_temperature = init_temperature

    def run(self, db, log_dir):
        """db object must implementes method 'evaluate' and method 'perturb' and 'update', 'rollback'."""
        self._init_log(log_dir)
        # compute init temperature
        if self._init_temperature == None:
            cost_mean, temperature = self._compute_init_params(db, run_num=100)
        else:
            cost_mean, temperature = None, self._init_temperature

        last_cost = db.evaluate()
        if (cost_mean != None):
            last_cost /= cost_mean

        best_cost = last_cost
        step = 0
        while step < self._max_num_step:
            for i in range(self._perturb_per_step):
                db.perturb()

                curr_cost = db.evaluate()
                if (cost_mean != None):
                    curr_cost /= cost_mean

                delta_cost = curr_cost - last_cost
                rand_num = np.random.uniform(0, 1)

                if delta_cost < 0 or math.exp(-delta_cost / temperature) > rand_num:
                    db.update()
                    last_cost = curr_cost

                else:
                    db.rollback()
                    self._logger.info('roll_back...')

                if curr_cost < best_cost:
                    db.update_solution()
                    best_cost = curr_cost

                print('step: {}, i: {}, current_cost: {}, best_cost: {}'.format(step, i, last_cost, best_cost))
                self._logger.info('step: {}, i: {}, current_cost: {}, best_cost: {}'.format(step, i, curr_cost, best_cost))

                perturb_num = step * self._perturb_per_step + i
                self._writer.add_scalar('curr_cost', curr_cost, perturb_num)
                self._writer.add_scalar('best_cost', best_cost, perturb_num)
                self._writer.add_scalar('temperature', temperature, perturb_num)
                self._writer.add_scalar('accept_prob', 1 - math.exp(-delta_cost / temperature), perturb_num)

            step += 1
            temperature *= self._cool_rate

    def _compute_init_params(self, db, run_num:int):
        self._logger.info('================== compute init temperature ================')
        cost_list = []
        for i in range(run_num):
            db.perturb()
            cost_list.append(db.evaluate())
            db.update()
    
        db.reset()
        delta_cost = 0
        for i in range(len(cost_list)-1):
            delta_cost += abs(cost_list[i+1] - cost_list[i])
        cost_mean = np.mean(cost_list)
        delta_cost /= cost_mean

        init_temperature = -1 * (delta_cost / (self._perturb_per_step - 1)) / math.log(self._init_pro)
        self._logger.info('========================== init temperature ================')
        self._logger.info('delta_cost: '.format(delta_cost))
        self._logger.info('init_temperature: '.format(init_temperature))
        self._logger.info('============================================================')
        return cost_mean, init_temperature

    def _init_log(self, log_dir):
        self._log_dir = log_dir + '/sa/'
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(logging.FileHandler(self._log_dir + '/simulate_anneal.log'))
        self._logger.root.name = 'aifp:sa_tunning'
        self._logger.info('========= simulate_anneal tunning log =========')
        self._writer = SummaryWriter(self._log_dir)