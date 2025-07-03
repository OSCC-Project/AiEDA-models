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
"""Library to convert a plc into a Dreamplace PlaceDB.

Convention:
 - indices of macros, ports, and pins in plc is named "_index" and "_indices".
 - indices of nodes, pins, and nets in PlaceDB is named "_id" and "ids".
"""
import pickle
import sys
import os
from absl import logging
sys.path.append(os.environ['THIRD-PARTY-PATH'] + 'dreamplace')

from dreamplace import PlaceDB
from aimp.aifp import setting
import numpy as np
# from thirdparty.irefactor import py_aifp_cpp as aifp_cpp
from aimp.aifp.utility import operators
# Internal gfile dependencies

def blockage_area(env_db):
  blockage_area = 0
  for blockage in env_db.get_blockage_list():
    blockage_area += (blockage.get_width() * blockage.get_height())
  return blockage_area

def convert_canvas(dp_db, env_db):
  """Convert canvas information in env_db into dreamplace.PlaceDB.

  Args:
    db: The dreamplace.PlaceDB instance.
    plc: The Rl-Env-db.
  """
  print('==================== plc_converter.py :: convert_canvas ===================')
  dp_db.xl = 0
  dp_db.yl = 0
  aifp_core = env_db.get_core()
  dp_db.xh = aifp_core.get_width()
  dp_db.yh = aifp_core.get_height()
  # num_columns, num_rows = env_db.get_grid_num_column_rows()
  num_columns = setting.evaluator['clustered_dreamplace']['num_columns']
  num_rows = setting.evaluator['clustered_dreamplace']['num_rows']
  dp_db.row_height = dp_db.yh / num_rows
  dp_db.site_width = dp_db.xh / num_columns
  dp_db.rows = []
  for i in range(num_rows):
    dp_db.rows.append([0, i * dp_db.row_height, dp_db.xh, (i + 1) * dp_db.row_height])

def convert_nodes(dp_db, env_db):
  """Convert nodes in plc into PlaceDB.

  Ports, hard macros, soft macros, and stdcells are converted to "nodes" in
  PlaceDB. Ports are considered "non-movable nodes". Soft macros and stdcells
  are considered "movable nodes". By default, we consider all the hard macros
  as "non-movable nodes".

  Node positions are different in two formats.
  Centered position is saved in the PlacementCost instance,
  while lower left position is saved in PlaceDB instance.

  Args:
    db: The PlaceDB instance.
    plc: The PlacementCost instance.
    hard_macro_order: Order of hard macros (excluding fixed ones).

  Returns:
    physical_node_indices: List of node indices in plc.
    node_index_to_node_id_map: Mapping from node index in plc to node id in
        PlaceDB.
    soft_macro_and_stdcell_indices: List of driver pin indices in plc.
    hard_macro_indices: List of hard macro indices.
    non_movable_node_indices: List of non-movable node indices in plc.
    num_blockage_dummy_node: Number of dummy blockages.
  """
  print('================= plc_converter: convert nodes ================')
  dp_db.node_names = []
  dp_db.node_name2id_map = {}
  dp_db.node_x = []
  dp_db.node_y = []
  dp_db.node_orient = []
  # We keep a copy of the node_size_x and y, so when we restore them after
  # calling PlaceDB.__call__ which modifies them, in case if we want to call
  # the function again.
  dp_db.original_node_size_x = []
  dp_db.original_node_size_y = []
  dp_db.macro_mask = []

  # To support, dreamplace mixed-size for crowded blocks, and to avoid
  # converting plc for every dreamplace call, we separate the non-fixed macros
  # and other movable nodes.
  
  # soft_macro_indices = env_db.get_stdcell_cluster_indices()
  soft_macro_indices = env_db.get_stdcell_indices()
  hard_macro_indices = env_db.get_macro_indices()

  # non_movable_node_indices = env_db.get_fixed_stdcell_indices() + env_db.get_io_indices()
  non_movable_node_indices = env_db.get_io_indices() # only contains io_clusters now...

  # DREAMPlace requires nodes to be arragned movable-first, so do that.
  physical_node_indices = (soft_macro_indices + hard_macro_indices + non_movable_node_indices)
  
  aifp_inst_list = env_db.get_inst_list()
  for node_id, node_index in enumerate(physical_node_indices):
    # 'id' is 'dreamplace-id', 'index' is 'aifp-node-index'.
    aifp_node = aifp_inst_list[node_index]
    name = aifp_node.get_name()
    dp_db.node_names.append(name)
    dp_db.node_name2id_map[name] = node_id

    if aifp_node.get_status() == "fixed":
      # x = aifp_node.get_low_x()
      # y = aifp_node.get_low_y()
      x = aifp_node.get_low_x() - env_db.get_core().get_low_x()
      y = aifp_node.get_low_y() - env_db.get_core().get_low_y()
    else:
      x = 0
      y = 0

    if aifp_node.get_type() == "io_cluster":
      # Treat a port as a node with 0 dimension and 'N' orientation.
      dp_db.node_orient.append(b'N')
      dp_db.original_node_size_x.append(0)
      dp_db.original_node_size_y.append(0)
    else:
      # node orientation not supported now...
      if aifp_node.get_orient() == '':
        dp_db.node_orient.append(b'N')
      else:
        dp_db.node_orient.append(aifp_node.get_orient().encode())

      w = aifp_node.get_width()
      h = aifp_node.get_height()
      dp_db.original_node_size_x.append(w)
      dp_db.original_node_size_y.append(h)

    # DREAMPlace uses lower left position
    dp_db.node_x.append(x)
    dp_db.node_y.append(y)
  
  # if the blockage rate is 1, translate it into a dummy fixed node.
  num_blockage_dummy_node = 0
  for b in env_db.get_blockage_list():
    dummy_node_name = 'blockage_dummy_node_' + str(num_blockage_dummy_node)
    dp_db.node_names.append(dummy_node_name)
    # dp_db.node_name2id_map[dummy_node_name] = (
    #     len(physical_node_indices) + num_blockage_dummy_node
    # )
    dp_db.node_name2id_map[dummy_node_name] = len(physical_node_indices) + num_blockage_dummy_node
    # dp_db.node_x.append(b.get_low_x())
    # dp_db.node_y.append(b.get_low_y())
    dp_db.node_x.append(b.get_low_x() - env_db.get_core().get_low_x())
    dp_db.node_y.append(b.get_low_y() - env_db.get_core().get_low_y())
    dp_db.original_node_size_x.append(b.get_width())
    dp_db.original_node_size_y.append(b.get_height())
    dp_db.node_orient.append(b'N')
    num_blockage_dummy_node += 1

  dp_db.num_physical_nodes = len(physical_node_indices) + num_blockage_dummy_node
  dp_db.num_terminals = len(hard_macro_indices) + len(non_movable_node_indices) + num_blockage_dummy_node
  dp_db.macro_mask = [False] * len(soft_macro_indices)

  dp_db.node_size_x = dp_db.original_node_size_x
  dp_db.node_size_y = dp_db.original_node_size_y
  dp_db.num_non_movable_macros = len(hard_macro_indices)

  node_index_to_node_id_map = {
      n: i for i, n in enumerate(physical_node_indices)
  }

  return (
      physical_node_indices,
      node_index_to_node_id_map,
      soft_macro_indices,
      hard_macro_indices,
      non_movable_node_indices,
      num_blockage_dummy_node,
  )

def convert_a_net(dp_db, env_db):
  """Convert a single net in plc into PlaceDB."""
  raise NotImplementedError

def convert_pins_and_nets(dp_db, env_db, physical_node_indices, node_index_to_node_id_map):
  """Convert pins and nets to PlaceDB"""
  print('=================== plc_converter: convert pins and nets =================')
  dp_db.pin_direct = []  # Array, len = number of pins
  dp_db.pin_offset_x = []  # Array, len = number of pins
  dp_db.pin_offset_y = []  # Array, len = number of pins
  dp_db.node2pin_map = []  # Array of array. node id to pin ids
  for _ in range(dp_db.num_physical_nodes):
    dp_db.node2pin_map.append([])
  dp_db.pin2node_map = []  # Array, len = number of pins
  dp_db.net_name2id_map = {}
  dp_db.net_names = []
  dp_db.pin2net_map = []  # Array, len = number of pins
  dp_db.net2pin_map = []
  dp_db.net_weights = []

  # counters = {'pin_id': 0, 'net_id': 0}
  pin_id_to_pin_index = []
  pin_index_to_pin_id = dict()

  net_id = 0 # dreamplace net-id
  pin_id = 0 # dreamplace pin-id
  aifp_net_list = env_db.get_net_list() # list[list[PyPin]]
  aifp_net_weight = env_db.get_net_weight()
  for net_index, pin_list in enumerate(aifp_net_list):
    # convert a net
    net_name = "net_{}".format(net_index)
    dp_db.net_names.append(net_name)
    dp_db.net_name2id_map[net_name] = net_id
    dp_db.net_weights.append(aifp_net_weight[net_index])
    # dp_db.net_weights.append(1)  # use same weight

    pin_ids_of_net = []
    # for all pins
    for i in range(len(pin_list)):
      aifp_pin = pin_list[i]
      dp_db.pin2net_map.append(net_id)
      # assume first pin is driving-pin
      if i == 0:
        dp_db.pin_direct.append('OUTPUT')
      else:
        dp_db.pin_direct.append('INPUT')

      # dp_db.pin_offset_x.append(aifp_pin.get_offset_x())
      # dp_db.pin_offset_y.append(aifp_pin.get_offset_y())
      pin_offset_x, pin_offset_y = get_pin_offset(env_db, aifp_pin)
      dp_db.pin_offset_x.append(pin_offset_x)
      dp_db.pin_offset_y.append(pin_offset_y)

      node_index = aifp_pin.get_node_index()
      node_id = node_index_to_node_id_map[node_index]
      dp_db.node2pin_map[node_id].append(pin_id)
      dp_db.pin2node_map.append(node_id)
      pin_ids_of_net.append(pin_id)
      pin_id_to_pin_index.append(aifp_pin.get_pin_index())
      pin_index_to_pin_id[aifp_pin.get_pin_index()] = pin_id
      pin_id += 1
    dp_db.net2pin_map.append(pin_ids_of_net)
    net_id += 1
  return pin_id_to_pin_index, pin_index_to_pin_id

def get_pin_offset(env_db, aifp_pin):
    node_index = aifp_pin.get_node_index()
    aifp_inst = env_db.get_inst_list()[node_index]
    # origin_orient: orient in def/lef
    origin_orient = aifp_inst.get_origin_orient()
    new_orient = aifp_inst.get_orient()

    # if instance is not macro or orient == origin-orient, use default offset
    if aifp_inst.get_type() != "macro" or new_orient == origin_orient:
      return aifp_pin.get_offset_x(), aifp_pin.get_offset_y()

    # calcuate new offset
    if new_orient == '':
      raise RuntimeError('macro orient error, orient is empty string!')


    new_pin_offset_x, new_pin_offset_y = operators.get_offset(
          origin_orient = origin_orient,
          new_orient = new_orient,
          width = aifp_inst.get_width(),
          height = aifp_inst.get_height(),
          origin_offset_x = aifp_pin.get_offset_x(),
          origin_offset_y =  aifp_pin.get_offset_y())
    # print('origin_orient: ', origin_orient)
    # print('new_orient: ', new_orient)
    # print('origin_offset_x: ', aifp_pin.get_offset_x())
    # print('origin_offset_y: ', aifp_pin.get_offset_y())
    # print('new_offset_x: ', new_pin_offset_x)
    # print('new_offset_y: ', new_pin_offset_y)
    return new_pin_offset_x, new_pin_offset_y


def np_array_of_array(py_list_of_list, dtype):
  """converts a Python list of list into a Numpy array of array."""
  return np.array(
      [np.array(l, dtype=dtype) for l in py_list_of_list], dtype=object
  )

def convert_to_ndarray(db):
  """Converts lists in the PlaceDB into Numpy arrays."""
  db.rows = np.array(db.rows, dtype=db.dtype)
  db.node_names = np.array(db.node_names, dtype=np.string_)
  db.node_x = np.array(db.node_x, dtype=db.dtype)
  db.node_y = np.array(db.node_y, dtype=db.dtype)
  db.node_orient = np.array(db.node_orient, dtype=np.string_)
  db.original_node_size_x = np.array(db.original_node_size_x, dtype=db.dtype)
  db.original_node_size_y = np.array(db.original_node_size_y, dtype=db.dtype)
  db.node_size_x = np.array(db.node_size_x, dtype=db.dtype)
  db.node_size_y = np.array(db.node_size_y, dtype=db.dtype)
  db.node2pin_map = np_array_of_array(db.node2pin_map, dtype=np.int32)
  db.pin_direct = np.array(db.pin_direct, dtype=np.string_)
  db.pin_offset_x = np.array(db.pin_offset_x, dtype=db.dtype)
  db.pin_offset_y = np.array(db.pin_offset_y, dtype=db.dtype)
  db.pin2node_map = np.array(db.pin2node_map, dtype=np.int32)
  db.pin2net_map = np.array(db.pin2net_map, dtype=np.int32)
  db.net_names = np.array(db.net_names, dtype=np.string_)
  db.net_weights = np.array(db.net_weights, dtype=db.dtype)
  db.net2pin_map = np_array_of_array(db.net2pin_map, dtype=np.int32)
  db.flat_node2pin_map, db.flat_node2pin_start_map = db.flatten_nested_map(
      db.pin2node_map, db.node2pin_map
  )
  db.flat_net2pin_map, db.flat_net2pin_start_map = db.flatten_nested_map(
      db.pin2net_map, db.net2pin_map
  )
  db.macro_mask = np.array(db.macro_mask, dtype=np.uint8)

def initialize_placedb_region_attributes(db):
  """Initialize the region-related attributes in PlaceDB instance.

  Args:
    db: The PlaceDB instance.  Assume there is no region constraints in the plc
      format.
  """
  db.regions = []
  db.flat_region_boxes = np.array([], dtype=db.dtype)
  db.flat_region_boxes_start = np.array([0], dtype=np.int32)
  db.node2fence_region_map = np.array([], dtype=np.int32)
  print('============ initialize_placedb_region_attributes ok =============')

class PlcConverter(object):
  """Class that converts a plc into a Dreamplace PlaceDB."""

  def __init__(self):
    # List of movable node (except hard macros) indices in plc.
    # self._soft_macro_and_stdcell_indices = None
    self._soft_macro_indices = None
    # List of hard macros in plc.
    self._hard_macro_indices = None
    # List of non-movable node indices in plc.
    self._non_movable_node_indices = None
    # List of driver pin indices in plc.
    # self._driver_pin_indices = None
    # Mapping from node index in plc to node id in PlaceDB.
    self._node_index_to_node_id_map = None
    # List of pin index in plc to pin id in PlaceDB.
    self._pin_id_to_pin_index = None
  
  @property
  def soft_macro_indices(self):
    return self._soft_macro_indices

  @property
  def hard_macro_indices(self):
    return self._hard_macro_indices

  @property
  def non_movable_node_indices(self):
    return self._non_movable_node_indices

  # @property
  # def driver_pin_indices(self):
  #   return self._driver_pin_indices

  @property
  def node_index_to_node_id_map(self):
    return self._node_index_to_node_id_map

  @property
  def pin_id_to_pin_index(self):
    return self._pin_id_to_pin_index

  def non_movable_macro_area(self, env_db):
    """Returns the area of the non-movable macros.

    Args:
      plc: A PlacementCost object.
      num_non_movable_macros: Optional int. spesifies the number of placed
        macros that should be consider as non-movable.
    """
    # if not num_non_movable_macros:
    #   num_non_movable_macros = len(self._hard_macro_indices)

    # non_movable_macro = self._hard_macro_indices[:num_non_movable_macros]

    # macro_width_height = [
    #     plc.get_node_width_height(m) for m in non_movable_macro
    # ]
    # macro_area = np.sum([w * h for w, h in macro_width_height])
    macro_area = 0
    for macro in env_db.get_macro_list():
      if macro.get_status() == "fixed":
        macro_area += (macro.get_width() * macro.get_height())
    return macro_area

  def convert(self, env_db, hard_macro_order=None):
    """Converts a env_db into a Dreamplace PlaceDB format.

    Args:
      env_db: The PlacementCost instance.
      hard_macro_order: Optional list of macros ordered by how the RL agent will
        place them. If not provided, use the node order of the plc.

    Returns:
      The converted PlaceDB instance.
    """
    dp_db = PlaceDB.PlaceDB()
    dp_db.dtype = np.float32

    # if not hard_macro_order:
    #   hard_macro_order = [
    #       m
    #       for m in env_db.get_macro_indices()
    #       if not (plc.is_node_soft_macro(m) or plc.is_node_fixed(m))
    #   ]

    convert_canvas(dp_db, env_db)

    (
        physical_node_indices,
        self._node_index_to_node_id_map,
        self._soft_macro_and_stdcell_indices,
        self._hard_macro_indices,
        self._non_movable_node_indices,
        self._num_blockage_dummy_node,
    ) = convert_nodes(dp_db, env_db)

    self._pin_id_to_pin_index, self._pin_index_to_pin_id = convert_pins_and_nets(
        dp_db, env_db, physical_node_indices, self._node_index_to_node_id_map)
    dp_db.total_space_area = dp_db.xh * dp_db.yh - blockage_area(env_db) - self.non_movable_macro_area(env_db)
    convert_to_ndarray(dp_db)
    print('========= start to initialize placed region attributes =========')
    initialize_placedb_region_attributes(dp_db)
    print('========= plc_converter convert() ok ===========')
    return dp_db

  def convert_and_dump(self, env_db, path_to_placedb, hard_macro_order=None):
    """Converts a env into a dreamplace.PlaceDB format and dump it.

    Args:
      plc: The PlacementCost instance.
      path_to_placedb: the path to the output file.
      hard_macro_order: Optional list of macros ordered by how the RL agent will
        place them. If not provided, use the node order of the plc.

    Returns:
      The converted PlaceDB instance.
    """
    db = self.convert(env_db, hard_macro_order)
    with open(path_to_placedb, 'wb') as output_file:
      pickle.dump(db, output_file)
    return db
  
  def update_macro(self, dp_db, env_db, macro_index):
    """Updates information about a macro from plc into db."""
    macro = env_db.get_inst_list()[macro_index]
    if not macro.get_status() == "fixed":
      print('[clustered-dreamplace WARNNING]: updating unfixed macro info, ignored....')
      return

    # Update macro location.
    macro_id = self._node_index_to_node_id_map[macro_index]
    # dp_db.node_x[macro_id] = macro.get_low_x()
    # dp_db.node_y[macro_id] = macro.get_low_y()
    print('origin_node_x: ', dp_db.node_x[macro_id])
    print('update node_x: ', macro.get_low_x() - env_db.get_core().get_low_x())
    print('origin_node_y: ', dp_db.node_y[macro_id])
    print('update_node_y: ', macro.get_low_y() - env_db.get_core().get_low_y())
    dp_db.node_x[macro_id] = macro.get_low_x() - env_db.get_core().get_low_x()
    dp_db.node_y[macro_id] = macro.get_low_y() - env_db.get_core().get_low_y()

    # Update macro orientation. not supported now...
    # old_orient = dp_db.node_orient[macro_id].decode()
    # new_orient = macro.get_orient()
    # if new_orient == '':
    #   raise RuntimeError('macro orient error, orient is empty string!')
    # dp_db.node_orient[macro_id] = new_orient.encode()

    # # Update offsets of hard macro pins only when orientation has changed.
    # if old_orient != new_orient and macro.get_status() == aifp_cpp.InstanceType.macro:
    #   macro_pin_list = macro.get_pin_list()
    #   for pin_index in macro_pin_list:
    #     pin_id = self._pin_index_to_pin_id[pin_index]
    #     new_pin_offset_x, new_pin_offset_y = operators.get_offset(
    #       origin_orient = old_orient,
    #       new_orient = new_orient,
    #       width = macro.get_width(),
    #       height = macro.get_height(),
    #       origin_offset_x = dp_db.pin_offset_x[pin_id],
    #       origin_offset_y = dp_db.pin_offset_y[pin_id]
    #     )

    #     dp_db.pin_offset_x[pin_id] = new_pin_offset_x
    #     dp_db.pin_offset_y[pin_id] = new_pin_offset_y


  def update_num_non_movable_macros(self, dp_db, env_db, num_non_movable_macros):
    """Updates PlaceDB parameters give the new num_non_movable_macros."""
    dp_db.num_terminals = num_non_movable_macros + len(self._non_movable_node_indices) + self._num_blockage_dummy_node
    macro_mask = [False] * len(self._soft_macro_indices) + [True ] * (len(self._hard_macro_indices) - num_non_movable_macros)
    dp_db.macro_mask = np.array(macro_mask, dtype=np.uint8)
    dp_db.total_space_area = (
        dp_db.xh * dp_db.yh
        - blockage_area(env_db)
        - self.non_movable_macro_area(env_db, num_non_movable_macros))
    dp_db.node_size_x = dp_db.original_node_size_x
    dp_db.node_size_y = dp_db.original_node_size_y
    dp_db.num_movable_pins = None
    dp_db.num_non_movable_macros = num_non_movable_macros