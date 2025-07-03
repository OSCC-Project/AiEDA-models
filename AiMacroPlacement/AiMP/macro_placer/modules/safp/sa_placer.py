import numpy as np
import copy
import sys
from macro_placer.database.macroPlaceDB import MacroPlaceDB
from tools.iEDA.module.io import IEDAIO

class SAPlacer:
    def __init__(self,
                workspace_dir : str,
                max_iters: int = 1000, 
                num_actions: int = 200,
                cool_rate: float = 0.98,
                pack_left: bool = True,
                pack_bottom: bool = True):
        
        self.workspace_dir = workspace_dir
        self._max_iters = max_iters
        self._num_actions = num_actions
        self._cool_rate = cool_rate
        self._pack_left = pack_left
        self._pack_bottom = pack_bottom
        self._inited = False
        self._placed = False

    def __init_sa_data(self, mp_db:MacroPlaceDB):
        mp_db.sort()
        self._region_lx = int(mp_db.xl)
        self._region_ly = int(mp_db.yl)
        self._region_dx = int(mp_db.width)
        self._region_dy = int(mp_db.height)
        self._num_nodes = mp_db.node_x.shape[0]
        self._num_movable_nodes = mp_db.num_movable_nodes
        self._num_pins = mp_db.pin_offset_x.shape[0]


        self._pin_x_off = copy.deepcopy(mp_db.pin_offset_x).astype(np.int64)
        self._pin_y_off = copy.deepcopy(mp_db.pin_offset_y).astype(np.int64)
        self._lx = copy.deepcopy(mp_db.node_x).astype(np.int64)
        self._ly = copy.deepcopy(mp_db.node_y).astype(np.int64)
        self._dx = copy.deepcopy(mp_db.node_size_x).astype(np.int64)
        self._dy = copy.deepcopy(mp_db.node_size_y).astype(np.int64)
        self._halo_x = np.zeros_like(self._dx)
        self._halo_y = np.zeros_like(self._dy)

        
        self._pin2vertex = np.ndarray(self._num_pins, dtype=np.int64)
        for pin_id, node_id in enumerate(mp_db.pin2node_map):
            self._pin2vertex[pin_id] = node_id
        
        net_start_idx = list(mp_db.flat_net2pin_start_map)
        net_start_idx = sorted(net_start_idx)
        self._net_span = np.array(net_start_idx, dtype=np.int64)
        self._inited = True

    def writeMPDB(self, mp_db:MacroPlaceDB):
        if not self._placed:
            raise RuntimeError("Not placed..")
        mp_db.node_x = self._result_lx.astype(mp_db.dtype)
        mp_db.node_y = self._result_ly.astype(mp_db.dtype)
        print("mp_db.node_x : ", mp_db.node_x)
        print("mp_db.node_y : ", mp_db.node_y)

    def place(self, mp_db:MacroPlaceDB):
        if not self._inited:
            self.__init_sa_data(mp_db)
        self._result_lx = np.ndarray(self._num_nodes, dtype = mp_db.dtype)
        self._result_ly = np.ndarray(self._num_nodes, dtype = mp_db.dtype)
        
        ieda_io = IEDAIO(self.workspace_dir)
        result = ieda_io.get_ieda().SAPlaceSeqPairInt64(
            self._max_iters,
            self._num_actions,
            self._cool_rate,
            self._pin_x_off,
            self._pin_y_off,
            self._lx,
            self._ly,
            self._dx,
            self._dy,
            self._halo_x,
            self._halo_y,
            self._pin2vertex,
            self._net_span,
            self._region_lx,
            self._region_ly,
            self._region_dx,
            self._region_dy,
            self._num_movable_nodes,
            self._pack_left,
            self._pack_bottom)
        
        for i, (x, y) in enumerate(result):
            self._result_lx[i] = x
            self._result_ly[i] = y
        self._placed = True