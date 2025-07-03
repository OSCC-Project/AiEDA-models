# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A soft macro placer using Dreamplace."""
import time
import os
import sys
from absl import logging

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir + "../../../../../../third_party")
sys.path.append(os.environ['THIRD-PARTY-PATH'] + 'dreamplace')

from aimp.aifp import setting
from aimp.aifp.operation.evaluation.dreamplace_util import dreamplace_util
from aimp.aifp.operation.evaluation.dreamplace_util.placedb_plc import PlacedbPlc
from dreamplace import NonLinearPlace
# from third_party.dreamplace.dreamplace import NonLinearPlace

class SoftMacroPlacer(object):
  """A soft macro placer using Dreamplace."""

  def __init__(self, params, hard_macro_order=None):
    self.params = params
    # for io in env_db.get_io_list():
    #   print('io {} , low_x: {}, low_y: {}, width: {}, height: {}'.format(io.get_name(), io.get_low_x(), io.get_low_y(), io.get_width(), io.get_height()))
    # self.placedb_plc = PlacedbPlc(env_db, params, hard_macro_order)
    # self._env_db = env_db

  # NOTE(hqzhu): convergence flag may not be right.
  # We cannot simply check divergence based on #iterations in DP V3.
  def place(self, env_db):
    """Place soft macros.

    Returns:
      metrics, bool indicating if DP converges or not based on checking #iterations.
    """
    # dreamplace will scale node_x, node_y in placedb.initilize() method, so we should create new placedb_plc every time..
    # placedb_plc will create new placedb, converts placedb, and call placedb.initilize() method.
    placedb_plc = PlacedbPlc(env_db, self.params)

    nonlinear_place = NonLinearPlace.NonLinearPlace(self.params,
                                                    placedb_plc.placedb)
    metrics = nonlinear_place(self.params, placedb_plc.placedb)
    logging.info('Last Dreamplace metric: %s', str(metrics[-1][-1][-1]))
    total_iterations = sum([stage['iteration'] for stage in self.params.global_place_stages])
    return metrics, (metrics[-1][-1][-1].iteration) < total_iterations


def dreamplace_main(env_db):
  # canvas_width, canvas_height = plc.get_canvas_width_height()
  # canvas_width = env_db.get_core().get_width()
  # canvas_height = env_db.get_core().get_height()
  # dp_params = dreamplace_util.get_dreamplace_params(
  #       canvas_width=canvas_width,
  #       canvas_height=canvas_height,
  #       gpu=False,
  #       num_bins_x=setting.evaluator['clustered_dreamplace']['num_bins_x'],
  #       num_bins_y=setting.evaluator['clustered_dreamplace']['num_bins_y'],
  #       )
  dp_params = dreamplace_util.get_dreamplace_params(
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

  return optimize_using_dreamplace(env_db, dp_params, False)

def optimize_using_dreamplace(env_db,
                              params,
                              hard_macro_movable=False):
  """Optimzes using Dreamplace."""
  # Initialization, slow but only happens once.
  start_init_time = time.time()
  placer = SoftMacroPlacer(params)
  if hard_macro_movable:
    placer.placedb_plc.update_num_non_movable_macros(
        env_db, num_non_movable_macros=0)
  logging.info('Initializing Dreamplace took %g seconds.', time.time() - start_init_time)

  # Dreamplace optimzation.
  start_opt_time = time.time()
  metrics, converge_flag = placer.place(env_db)
  print('converge-flag: ', converge_flag)
  result = metrics[-3][0]
  wirelength = float(result[0].hpwl.data)
  overflow = float(result[0].overflow.mean().data)
  logging.info('Dreamplace optimization took %g seconds.',time.time() - start_opt_time)
  return wirelength

  # Write the optimized stdcell location back to the plc. This step may be
  # omitted if the Dreamplace reported density can be used in our cost function
  # directly.
  # start_write_time = time.time()
  # placer.placedb_plc.write_movable_locations_to_plc(plc)
  # logging.info('Writing soft macro locations to plc took %g seconds.',
  #              time.time() - start_write_time)

  # The total run time of using Dreamplace to optimize soft macro placement.

  # filename_prefix = 'dreamplace'
  # dreamplace_util.print_and_save_result(plc, duration, 'Dreamplace',
  #                                       filename_prefix, output_dir)
