import numpy as np

from aimp.aifp.database.data_structure.data_spec import DataInfo
from aimp.aifp.database.data_structure.box_space import BoxSpace
from aimp.aifp import setting

class ObservationSpace:
    def __init__(self, node_nums, edge_nums):
        self._node_features = DataInfo(dtype=np.float32, shape=(node_nums, len(setting.node_feature_config)))
        self._macro_idx_to_place = DataInfo(dtype=np.int64, shape=(1, ))
        self._sparse_adj_i = DataInfo(dtype=np.int64, shape=(edge_nums, ))
        self._sparse_adj_j = DataInfo(dtype=np.int64, shape=(edge_nums, ))
        self._sparse_adj_weight = DataInfo(dtype=np.float32, shape=(edge_nums, ))
        self._action_mask = DataInfo(dtype=np.float32, shape=(setting.env_train['max_grid_nums'], setting.env_train['max_grid_nums']))
        self._reward = DataInfo(dtype=np.float32, shape=(1, ))
        self._done = DataInfo(dtype=np.float32, shape=(1, ))

    @property
    def node_features(self):
        return self._node_features
    
    @property
    def macro_idx_to_place(self):
        return self._macro_idx_to_place
    
    @property
    def sparse_adj_i(self):
        return self._sparse_adj_i
    
    @property
    def sparse_adj_j(self):
        return self._sparse_adj_j
    
    @property
    def sparse_adj_weight(self):
        return self._sparse_adj_weight
    
    @property
    def action_mask(self):
        return self._action_mask
    
    @property
    def reward(self):
        return self._reward
    
    @property
    def done(self):
        return self._done

# class ObservationSpace:
#     def __init__(self, num_nodes, env_config, metadata_config, layoutdata_config, nodedata_config):
#         self._num_nodes = num_nodes
#         self._num_grids = env_config.num_grids
#         self._layoutdata_shape = env_config.layoutdata_shape

#         self._metadata_space = self._init_metadata_space(metadata_config.metadata_names)
#         self._nodedata_space = self._init_nodedata_space(nodedata_config.nodedata_names)
#         self._layoutdata_space = self._init_layoutdata_space(layoutdata_config.layoutdata_names)

#     @property
#     def metadata_space(self):
#         return self._metadata_space
    
#     @property
#     def nodedata_space(self):
#         return self._nodedata_space
    
#     @property
#     def layoutdata_space(self):
#         return self._layoutdata_space

#     def _init_metadata_space(self, metadata_names):
#         metadata_space = {}
#         for metadata_name in metadata_names:
#             metadata_space[metadata_name] = BoxSpace(low=0, high=1, shape=(1,))
#         return metadata_space
    
#     def _init_nodedata_space(self, nodedata_names):
#         nodedata_space = {}
#         for nodedata_name in nodedata_names:
#             if nodedata_name == 'grid_x':
#                 nodedata_space[nodedata_name] = BoxSpace(low=0, high=self._num_grids-1, shape=(self._num_nodes, 1))
#             elif nodedata_name == 'grid_y':
#                 nodedata_space[nodedata_name] = BoxSpace(low=0, high=self._num_grids-1, shape=(self._num_nodes, 1))
#             elif nodedata_name == 'is_placed':
#                 nodedata_space[nodedata_name] = BoxSpace(low=0, high=1, shape=(self._num_nodes, 1))
#             elif nodedata_name == 'width':
#                 nodedata_space[nodedata_name] = BoxSpace(low=0, high=1, shape=(self._num_nodes, 1))
#             elif nodedata_name == 'height':
#                 nodedata_space[nodedata_name] = BoxSpace(low=0, high=1, shape=(self._num_nodes, 1))
#             else:
#                 raise NotImplementedError
#         return nodedata_space
    
#     def _init_layoutdata_space(self, layoutdata_names):
#         layoutdata_space = {}
#         for layoutdata_name in layoutdata_names:
#             if layoutdata_name == 'density':
#                 layoutdata_space[layoutdata_name] = BoxSpace(low=0, high=1, shape=self._layoutdata_shape)
#             else:
#                 raise NotImplementedError
#         return layoutdata_space

#     def __str__(self):
#         observation_space_str = 'observation_space.space(\n'
#         observation_space_str += 'metadata_space:(\n'
#         for key in self._metadata_space.keys():
#             observation_space_str += '{}: {}\n'.format(key, self._metadata_space[key].__str__())
#         observation_space_str += ')\nnodedata_space:(\n'
#         for key in self._nodedata_space.keys():
#             observation_space_str += '{}: {}\n'.format(key, self._nodedata_space[key].__str__())
#         observation_space_str += ')\nlayoutdata_space:\n'
#         for key in self._layoutdata_space.keys():
#             observation_space_str += '{}: {}\n'.format(key, self._layoutdata_space[key].__str__())
#         observation_space_str += ')\n'
#         return observation_space_str