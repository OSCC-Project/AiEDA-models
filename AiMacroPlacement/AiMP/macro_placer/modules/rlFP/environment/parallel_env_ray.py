import numpy as np
import json
# # import grpc
import asyncio
import socket
import logging

from aifp.utility.singleton import singleton
from aifp.operation.macro_placer.rl_placer.environment.local_env import LocalEnv
# from aifp.operation.macro_placer.rl_placer.environment.env_cluster_controller import EnvClusterController
from aifp.database.data_structure.observation_space import ObservationSpace
from aifp.database.data_structure.box_space import BoxSpace
# import aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_pb2 as remote_env_pb2
# import aifp.operation.macro_placer.rl_placer.environment.remote.remote_env_pb2_grpc as remote_env_pb2_grpc
from aifp import setting
import time
import ray
import os

@singleton
class ParallelEnv:
    def __init__(self):

        # log_to_driver=False # Task and actor logs will not be copied to the driver stdout.
        self._env_nums = setting.get_env_nums()
        self._init_ray_config()
        RemoteEnv = ray.remote(LocalEnv).options(memory=2e9) # num_cpus=1, num_gpus=0
        self._remote_envs = [RemoteEnv.remote() for i in range(self._env_nums)]
        self._node_nums, self._macro_nums, self._edge_nums = self._get_node_edge_nums()
        self._observation_space = self._init_observation_space()
        self._action_space = self._init_action_space()
        self._set_env_log()

    def _init_ray_config(self):
        runtime_env={"working_dir": "/root/reconstruct-aifp/",
            "env_vars": {"AIFP_PATH": "/root/reconstruct-aifp"}}
        os.environ["RAY_DEDUP_LOGS"] = "0"
        os.environ["RAY_memory_monitor_refresh_ms"] = "0" # the interval to check memory usage and kill tasks or actors if needed. Task killing is disabled when this value is 0. 
        ray.init(runtime_env=runtime_env)


    def _set_env_log(self):
        for i in range(self.env_nums):
            self._remote_envs[i].set_log_dir.remote(i)

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


        reset_futures = [self._remote_envs[i].reset.remote(episode) for i in range(len(self._remote_envs))]
        reset_results = [ray.get(future) for future in reset_futures]
        result_nums = len(reset_results[0])

        parallel_obs = [[] for i in range(result_nums)]

        for env_idx in range(self.env_nums):
            for result_idx in range(result_nums):
                parallel_obs[result_idx].append(reset_results[env_idx][result_idx])
        # stack parallel_obs arrays
        for result_idx in range(result_nums):
            parallel_obs[result_idx] = np.stack(parallel_obs[result_idx], axis=0)
        return parallel_obs

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
            info:   list of string.
        """
        assert self._step < self.episode_length

        step_futures = [self._remote_envs[i].step.remote(parallel_action[i]) for i in range(len(self._remote_envs))]
        step_results = [ray.get(future) for future in step_futures]
        result_nums = len(step_results[0])

        parallel_obs = [[] for i in range(result_nums - 3)]
        for env_idx in range(len(step_results)):
            for result_idx in range(result_nums - 3):
                parallel_obs[result_idx].append(step_results[env_idx][result_idx])
        parallel_info = [step_results[i][-1] for i in range(self.env_nums)]
        parallel_done = [step_results[i][-2] for i in range(self.env_nums)]
        parallel_reward = [step_results[i][-3] for i in range(self.env_nums)]

        # stack np results
        for result_idx in range(result_nums - 3):
            parallel_obs[result_idx] = np.stack(parallel_obs[result_idx], axis=0)
        parallel_reward = np.stack(parallel_reward, axis=0)
        parallel_done = np.stack(parallel_done, axis=0)
        return parallel_obs, parallel_reward, parallel_done, parallel_info

    def _init_remote_envs(self):
        raise NotImplementedError

    def _get_node_edge_nums(self):
        node_nums = ray.get(self._remote_envs[0].get_node_nums.remote())
        edge_nums = ray.get(self._remote_envs[0].get_edge_nums.remote())
        macro_nums = ray.get(self._remote_envs[0].get_macro_nums.remote())
        return node_nums, macro_nums, edge_nums

    def _init_observation_space(self):
        return ObservationSpace(self.node_nums, self.edge_nums)

    def _init_action_space(self):
        return BoxSpace(low=0, high=setting.env_train['max_grid_nums']**2, shape=(1,))
    
    def __del__(self):
        pass
        # ray.shutdown()  # ray is shutdown automatically when python process terminates