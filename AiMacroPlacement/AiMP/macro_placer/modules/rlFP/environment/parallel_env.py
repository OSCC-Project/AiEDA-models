from tkinter import Y
from typing import Tuple
import numpy as np
import json
# import grpc
import asyncio
import socket
import logging

from aifp.utility.singleton import singleton
from aifp.operation.macro_placer.rl_placer.environment.env_cluster_controller import EnvClusterController
from aifp.database.data_structure.observation_space import ObservationSpace
from aifp.database.data_structure.box_space import BoxSpace
import aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_pb2 as remote_env_pb2
import aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_pb2_grpc as remote_env_pb2_grpc
from aifp import setting
import time


@singleton
class ParallelEnv:
    def __init__(self):
        self._case_select = setting.case_select
        self._env_cluster_controller = EnvClusterController()
        self._env_server_socket_list = self._env_cluster_controller.start_clusters()
        time.sleep(10) # wait for servers to start
        # self._env_server_socket_list = self._get_server_sockets()
        self._env_nums = len(self._env_server_socket_list)
        self._node_nums, self._macro_nums, self._edge_nums = self._get_node_edge_nums()
        self._observation_space = self._init_observation_space()
        self._action_space = self._init_action_space()

    @property
    def env_nums(self):
        return self._env_nums

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def episode_length(self): 
        return self._macro_nums

    @property
    def node_nums(self):
        return self._node_nums
    
    @property
    def edge_nums(self):
        return self._edge_nums
    
    @property
    def grid_nums(self):
        return setting.env_train['max_grid_nums']

    def reset(self, episode):
        """
        Returns:
            obs_list:  list of batch_observations to forward model
                - obs_list[0]: node_features       numpy.ndarray, shape(env_nums, node_nums, node_features)
                - obs_list[1]: macro_idx_to_place  numpy.ndarray, shape(env_nums, 1)
                - obs_list[2]: sparse_adj_i        numpy.ndarray, shape(env_nums, edge_nums), dtype=np.int64
                - obs_list[3]: sparse_adj_j        numpy.ndarray, shape(env_nums, edge_nums), dtype=np.int64
                - obs_list[4]: sparse_adj_weight   numpy.ndarray, shape(env_nums, edge_nums), dtype=np.float32
                - obs_list[5]: action_mask         numpy.ndarray, shape(env_nums, grid_nums, grid_nums), dtype=np.float32
        """
        self._step = 0

        parallel_macro_idx_to_place = []
        parallel_np_obs = []
        parallel_sparse_adj_i = []
        parallel_sparse_adj_j = []
        parallel_sparse_adj_weight = []
        parallel_action_mask = []

        parallel_result = [0 for i in range(self._env_nums)]

        async def async_call_reset(parallel_result, env_idx, socket, episode):
            
            async with grpc.aio.insecure_channel(socket, options=[('grpc.max_send_message_length', setting.env_train['grpc_max_message_length']),
                ('grpc.max_receive_message_length', setting.env_train['grpc_max_message_length'])] ) as channel:
                local_stub = remote_env_pb2_grpc.RemoteEnvStub(channel)
                result = await local_stub.reset(remote_env_pb2.Action(action=episode))
            parallel_result[env_idx] = result

        async def call_all_reset(parallel_result, env_nums, episode):
            tasks =  [async_call_reset(parallel_result, i, self._env_server_socket_list[i], episode) for i in range(env_nums)]
            await asyncio.gather(*tasks)

        asyncio.run(call_all_reset(parallel_result, self._env_nums, episode))

        for i in range(self._env_nums):

            macro_idx_to_place = np.frombuffer(parallel_result[i].macro_idx_to_place, dtype=self.observation_space.macro_idx_to_place.dtype).reshape(self.observation_space.macro_idx_to_place.shape)
            np_obs = np.frombuffer(parallel_result[i].np_obs, dtype=self.observation_space.node_features.dtype).reshape(self.observation_space.node_features.shape)
            action_mask = np.frombuffer(parallel_result[i].action_mask, dtype=self.observation_space.action_mask.dtype).reshape(self.observation_space.action_mask.shape)
            sparse_adj_i = np.frombuffer(parallel_result[i].sparse_adj_i, dtype=self.observation_space.sparse_adj_i.dtype).reshape(self.observation_space.sparse_adj_i.shape)
            sparse_adj_j = np.frombuffer(parallel_result[i].sparse_adj_j, dtype=self.observation_space.sparse_adj_j.dtype).reshape(self.observation_space.sparse_adj_j.shape)
            sparse_adj_weight = np.frombuffer(parallel_result[i].sparse_adj_weight, dtype=self.observation_space.sparse_adj_weight.dtype).reshape(self.observation_space.sparse_adj_weight.shape)

            parallel_macro_idx_to_place.append(macro_idx_to_place)
            parallel_np_obs.append(np_obs)
            parallel_sparse_adj_i.append(sparse_adj_i)
            parallel_sparse_adj_j.append(sparse_adj_j)
            parallel_sparse_adj_weight.append(sparse_adj_weight)
            parallel_action_mask.append(action_mask)
        
        parallel_macro_idx_to_place = np.stack(parallel_macro_idx_to_place, axis=0)
        parallel_np_obs = np.stack(parallel_np_obs, axis=0)
        parallel_sparse_adj_i = np.stack(parallel_sparse_adj_i, axis=0)
        parallel_sparse_adj_j = np.stack(parallel_sparse_adj_j, axis=0)
        parallel_sparse_adj_weight = np.stack(parallel_sparse_adj_weight, axis=0)
        parallel_action_mask = np.stack(parallel_action_mask, axis=0)

        return [parallel_np_obs, parallel_macro_idx_to_place, parallel_sparse_adj_i, parallel_sparse_adj_j, parallel_sparse_adj_weight, parallel_action_mask]
    
    def step(self, parallel_action):
        """
        Args:
            action: batch_discrete_action          numpy.ndarray, shape(env_nums,)

        Returns:
            obs_list: list of batch_observations to forward model
                - obs_list[0]: node_features       numpy.ndarray, shape(env_nums, node_nums, node_features)
                - obs_list[1]: macro_idx_to_place  numpy.ndarray, shape(env_nums, 1)
                - obs_list[2]: sparse_adj_i        numpy.ndarray, shape(env_nums, edge_nums), dtype=np.int64
                - obs_list[3]: sparse_adj_j        numpy.ndarray, shape(env_nums, edge_nums), dtype=np.int64
                - obs_list[4]: sparse_adj_weight   numpy.ndarray, shape(env_nums, edge_nums), dtype=np.float32
                - obs_list[5]: action_mask         numpy.ndarray, shape(env_nums, grid_nums, grid_nums), dtype=np.float32
            reward: numpy.ndarray, shape(env_nums, 1), dtype=np.float32
            done:   numpy.ndarray, shape(env_nums, 1), dtype=np.float32, = 1.0 if episode ends, else 0.0
        """
        assert self._step < self.episode_length
        # print('step {}:, parallel action: {}'.format(self._step, parallel_action))
        parallel_macro_idx_to_place = []
        parallel_np_obs = []
        parallel_sparse_adj_i = []
        parallel_sparse_adj_j = []
        parallel_sparse_adj_weight = []
        parallel_action_mask = []
        parallel_reward = []
        parallel_done = []
        parallel_info = []

        parallel_result = [0 for i in range(self._env_nums)]

        async def async_call_step(parallel_result, env_idx, socket, action):
            async with grpc.aio.insecure_channel(socket, options=[('grpc.max_send_message_length', setting.env_train['grpc_max_message_length']),
                ('grpc.max_receive_message_length', setting.env_train['grpc_max_message_length'])]) as channel:
                local_stub = remote_env_pb2_grpc.RemoteEnvStub(channel)
                result = await local_stub.step(remote_env_pb2.Action(action=action))
            parallel_result[env_idx] = result

        async def call_all_step(parallel_result, env_nums, parallel_action):
            tasks =  [async_call_step(parallel_result, i, self._env_server_socket_list[i], parallel_action[i]) for i in range(env_nums)]
            await asyncio.gather(*tasks)

        asyncio.run(call_all_step(parallel_result, self._env_nums, parallel_action))
        self._step += 1

        for i in range(self._env_nums):
            macro_idx_to_place = np.frombuffer(parallel_result[i].macro_idx_to_place, dtype=self.observation_space.macro_idx_to_place.dtype).reshape(self.observation_space.macro_idx_to_place.shape)
            np_obs = np.frombuffer(parallel_result[i].np_obs, dtype=self.observation_space.node_features.dtype).reshape(self.observation_space.node_features.shape)
            action_mask = np.frombuffer(parallel_result[i].action_mask, dtype=self.observation_space.action_mask.dtype).reshape(self.observation_space.action_mask.shape)
            sparse_adj_i = np.frombuffer(parallel_result[i].sparse_adj_i, dtype=self.observation_space.sparse_adj_i.dtype).reshape(self.observation_space.sparse_adj_i.shape)
            sparse_adj_j = np.frombuffer(parallel_result[i].sparse_adj_j, dtype=self.observation_space.sparse_adj_j.dtype).reshape(self.observation_space.sparse_adj_j.shape)
            sparse_adj_weight = np.frombuffer(parallel_result[i].sparse_adj_weight, dtype=self.observation_space.sparse_adj_weight.dtype).reshape(self.observation_space.sparse_adj_weight.shape)

            reward = np.frombuffer(parallel_result[i].reward, dtype=self.observation_space.reward.dtype).reshape(self.observation_space.reward.shape)
            done = np.frombuffer(parallel_result[i].done, dtype=self.observation_space.done.dtype).reshape(self.observation_space.done.shape)
            info = parallel_result[i].info

            parallel_macro_idx_to_place.append(macro_idx_to_place)
            parallel_np_obs.append(np_obs)
            parallel_sparse_adj_i.append(sparse_adj_i)
            parallel_sparse_adj_j.append(sparse_adj_j)
            parallel_sparse_adj_weight.append(sparse_adj_weight)
            parallel_action_mask.append(action_mask)
            parallel_reward.append(reward)
            parallel_done.append(done)
            parallel_info.append(info)

        parallel_macro_idx_to_place = np.stack(parallel_macro_idx_to_place, axis=0) # np.expand_dims(
        parallel_np_obs = np.stack(parallel_np_obs, axis=0)
        parallel_sparse_adj_i = np.stack(parallel_sparse_adj_i, axis=0)
        parallel_sparse_adj_j = np.stack(parallel_sparse_adj_j, axis=0)
        parallel_sparse_adj_weight = np.stack(parallel_sparse_adj_weight, axis=0)
        parallel_action_mask = np.stack(parallel_action_mask, axis=0)
        parallel_reward = np.stack(parallel_reward, axis=0)
        parallel_done = np.stack(parallel_done, axis=0)

        return [parallel_np_obs, parallel_macro_idx_to_place, parallel_sparse_adj_i, parallel_sparse_adj_j, parallel_sparse_adj_weight, parallel_action_mask],\
                parallel_reward, parallel_done, parallel_info

    def _init_remote_envs(self):
        raise NotImplementedError

    def _get_node_edge_nums(self):
        with grpc.insecure_channel(self._env_server_socket_list[0]) as channel:
            local_stub = remote_env_pb2_grpc.RemoteEnvStub(channel)
            unused_param = remote_env_pb2.Number(num=0)
            node_nums = local_stub.node_nums(unused_param).num
            macro_nums = local_stub.macro_nums(unused_param).num
            edge_nums = local_stub.edge_nums(unused_param).num
            print('node_nums: ', node_nums)
            return node_nums, macro_nums, edge_nums
    
    def _init_observation_space(self):
        return ObservationSpace(self.node_nums, self.edge_nums)

    def _init_action_space(self):
        return BoxSpace(low=0, high=setting.env_train['max_grid_nums']**2, shape=(1,))

    def _get_server_sockets(self):
        server_socket_list = []
        local_hostname = socket.gethostname()
        for hostname, server_config in setting.env_server.items():
            for port in self._env_server_ports:
                if hostname == local_hostname:
                    server_socket = '{}:{}'.format('localhost', port)
                else:
                    raise NotImplementedError
                    # server_socket = '{}:{}'.format(server_config['server_ip_address'], server_config['server_start_port'] + i)
                server_socket_list.append(server_socket)
        print('server_socket_list: ', server_socket_list)
        return server_socket_list