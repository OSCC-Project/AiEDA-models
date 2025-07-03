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
"""Building connections between dreamplace.placeDB and PlacementCost.

In this file, we do not conduct placement as it will needs torch library.
Realistic placement is done in dreamplace_core.
"""
from absl import logging

from aimp.aifp.operation.evaluation.dreamplace_util import plc_converter

class PlacedbPlc(object):
  """Building connections between dreamplace.placeDB and PlacementCost."""

  def __init__(self, env_db, params, hard_macro_order=None):
    self.params = params
    self.converter = plc_converter.PlcConverter()
    self.placedb = self.converter.convert(env_db, hard_macro_order)
    print('========= place db converte ok ==========')
    self.placedb.initialize(params)
    # self.placedb(self.params)
    print('========= place db initilize ok ==========')


  def read_macro_from_env_db(self, env_db, macro_index):
    """Reads information about a macro in the plc into the placedb."""
    self.converter.update_macro(self.placedb, env_db, macro_index)

  def read_hard_macros_from_env_db(self, env_db):
    """Reads information of the placed hard macros in the plc into the placedb.

    Args:
      plc: The PlacementCost object.
    """
    for macro_index in env_db.get_macro_indices():
      # if not plc.is_node_soft_macro(macro_index):
      self.converter.update_macro(self.placedb, env_db, macro_index)

  def update_num_non_movable_macros(self, env_db, num_non_movable_macros):
    """Updates the number of non-movable hard macros.

    Args:
      plc: The PlacementCost object.
      num_non_movable_macros: Number of non-movable hard macros.
    """
    if num_non_movable_macros != self.placedb.num_non_movable_macros:
      logging.info("Reinitialized the PlaceDB.")
      self.converter.update_num_non_movable_macros(self.placedb, env_db, num_non_movable_macros)
      self.placedb(self.params)

  # def write_movable_locations_to_plc(self, plc):
  #   """Write the locations of the movable nodes back to the plc."""
  #   for node_id, node_index in enumerate(
  #       self.converter.soft_macro_and_stdcell_indices +
  #       self.converter.hard_macro_indices):
  #     # DREAMPlace uses lower left position, while plc uses centered position.
  #     plc.update_node_coords(
  #         node_index,
  #         self.placedb.node_x[node_id] + self.placedb.node_size_x[node_id] / 2,
  #         self.placedb.node_y[node_id] + self.placedb.node_size_y[node_id] / 2)

  # def update_net_weights(self, plc):
  #   """Update net weights in placedb according to the input plc."""
  #   for net_id, pin_index in enumerate(self.converter.driver_pin_indices):
  #     new_weight = plc.get_node_weight(pin_index)
  #     self.placedb.net_weights[net_id] = new_weight
