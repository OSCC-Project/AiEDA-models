import numpy as np
from concurrent import futures
import json
import logging
import time
# import grpc
import multiprocessing
from multiprocessing import Pool
from aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_servicer import RemoteEnvServicer
import aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_pb2 as remote_env_pb2
import aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_pb2_grpc as remote_env_pb2_grpc
from aifp.database.data_structure.box_space import BoxSpace
from aifp.operation.macro_placer.rl_placer.environment.space.observation_space import ObservationSpace
from aifp import setting


class RemoteEnv:
    def __init__(
        self,
        case_select,
        port,
        max_workers):

        self._case_select = case_select
        self._port = port
        self._max_workers = max_workers
        self._observation_space = self._init_observation_space()
        self._action_space = self._init_observation_space()
        self._init_remote_env_server()
        self._init_local_stub()
        self._node_nums, self._edge_nums = self._get_node_edge_nums()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def episode_length(self): 
        return self.node_nums

    @property
    def node_nums(self):
        return self._node_nums
    
    @property
    def edge_nums(self):
        return self._edge_nums
    
    @property
    def grid_nums(self):
        return setting.env_train['max_grid_nums']


    def reset(self):
        grpc_notused_action = remote_env_pb2.Action(action=-1)
        ret = self._local_stub.reset(grpc_notused_action)

        macro_idx_to_place = ret.macro_idx_to_place
        np_obs = np.array(json.loads(ret.np_obs), dtype=np.float32).reshape(self.node_nums, len(self._nodedata_config.nodedata_names))
        action_mask = np.array(json.loads(ret.action_mask), dtype=np.float32).reshape(self._env_config.num_grids, self._env_config.num_grids)
        sparse_adj_i = np.array(json.loads(ret.sparse_adj_i), dtype=np.int64)
        sparse_adj_j = np.array(json.loads(ret.sparse_adj_j), dtype=np.int64)
        sparse_adj_weight = np.array(json.loads(ret.sparse_adj_weight), dtype=np.float32)

        return [np_obs, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight, action_mask]


    def step(self, action:int):
        grpc_action = remote_env_pb2.Action(action=action)
        ret = self._local_stub.step(grpc_action)

        macro_idx_to_place = ret.macro_idx_to_place
        np_obs = np.array(json.loads(ret.np_obs), dtype=np.float32).reshape(self.node_nums, len(self._nodedata_config.nodedata_names))
        action_mask = np.array(json.loads(ret.action_mask), dtype=np.float32).reshape(self._env_config.num_grids, self._env_config.num_grids)
        sparse_adj_i = np.array(json.loads(ret.sparse_adj_i), dtype=np.int64)
        sparse_adj_j = np.array(json.loads(ret.sparse_adj_j), dtype=np.int64)
        sparse_adj_weight = np.array(json.loads(ret.sparse_adj_weight), dtype=np.float32)
        reward = np.array(ret.reward, dtype=np.float32)
        done = np.array(ret.done, dtype=np.float32)

        return [np_obs, macro_idx_to_place, sparse_adj_i, sparse_adj_j, sparse_adj_weight, action_mask], reward, done

    def _init_observation_space(self):
        return ObservationSpace(self.node_nums, self.edge_nums)

    def _init_action_space(self):
        return BoxSpace(low=0, high=setting.env_train['max_grid_nums']**2, shape=(1,))

    def _init_remote_env_server(self):
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=self._max_workers))
        remote_env_pb2_grpc.add_RemoteEnvServicer_to_server(
            RemoteEnvServicer(self._case_select), self._server)
        self._server.add_insecure_port('[::]:{}'.format(self._port))
        self._server.start()
        print("Server started, listening on {}".format(self._port))
    
    def _init_local_stub(self):
        self._channel = grpc.insecure_channel('localhost:{}'.format(self._port))
        self._local_stub = remote_env_pb2_grpc.RemoteEnvStub(self._channel)
    
    def __del__(self):
        self._channel.close()
        self._server.stop(0)

    def _get_node_edge_nums(self):
        unused_param = remote_env_pb2.Number(num=0)
        node_nums = self._local_stub.node_nums(unused_param).num
        edge_nums = self._local_stub.edge_nums(unused_param).num
        return node_nums, edge_nums