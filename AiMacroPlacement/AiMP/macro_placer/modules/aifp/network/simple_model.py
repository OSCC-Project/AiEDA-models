from readline import set_pre_input_hook
import torch
import torch.nn as nn

from aimp.aifp.network.shared_model import SharedModel
from aimp.aifp.network import cnn
from aimp.aifp import setting


class SimpleModel(SharedModel):
    EPSILON=1E-6

    def __init__(
        self,
        grid_nums,
        episode_len,
        gcn_layer_nums=3,
        edge_fc_layer_nums=1,
        origin_node_dim=2,
        gcn_node_dim=8,
        dirichlet_alpha=0.1,
        policy_noise_weight=0.0,
        ):
        print('origin_node_dim: ', origin_node_dim)

        super(SimpleModel, self).__init__()

        self.set_task('RL')
        # self._node_nums = node_nums
        self._grid_nums = grid_nums
        self._episode_len = episode_len
        self._gcn_layer_nums=gcn_layer_nums
        self._edge_fc_layer_nums = edge_fc_layer_nums
        self._gcn_node_dim=gcn_node_dim
        self._dirichlet_alpha=dirichlet_alpha
        self._policy_noise_weight=policy_noise_weight
        

        self._laytout_feature_encoder_channels = [1, 8, 8]
        # self._metadata_encoder = self._create_metadata_encoder(episode_len, gcn_node_dim)
        self._node_feature_encoder = self._create_node_feature_encoder(origin_node_dim, gcn_node_dim)
        self._layout_feature_encoder = self._create_layout_feature_encoder(self._laytout_feature_encoder_channels)

        self._attention_layer = self._create_multihead_attention_layers(embed_dim=gcn_node_dim, num_heads=4, kdim=gcn_node_dim, vdim=gcn_node_dim)
        self._attention_query_layer = self._create_attention_query_layer(gcn_node_dim)
        self._attention_key_layer = self._create_attention_key_layer(gcn_node_dim)
        self._attention_value_layer = self._create_attention_value_layer(gcn_node_dim)

        self._transpose_convs = self._create_transpose_convs(3*gcn_node_dim, grid_nums)
        self._policy_action_head = self._create_policy_action_head(2 + self._laytout_feature_encoder_channels[-1])
        self._value_head = self._create_value_head(3*gcn_node_dim)

        # self._pretrain_head = self._create_value_head(gcn_node_dim)
        # self._node_prediction_head = self._create_node_prediction_head(gcn_node_dim)


    def get_params(self):
        return self.parameters()

    def set_task(self, task:str='RL'):
        """set model task in [RL, pretrain, node_predition]"""
        self._task = task

    def forward(self, batch_nodedata, current_node_idx, not_used_sparse_adj_i, not_used_sparse_adj_j, not_used_sparse_adj_weight, action_mask):
        """
        Args:
            batch_node_data:   (batch, node_num, node_feature_dim)
            current_node_idx:  (batch, 1)
            sparse_adj_i:      (batch, edge_num)
            sparse_adj_j:      (batch, edge_num)
            sparse_adj_weight: (batch, edge_num)
            action_mask:       (batch, grid_nums, grid_nums)

        Returns:
            action_logits:     (batch, action_dim)
            value:             (batch, 1)
        """
        # step = step * torch.ones_like(current_node_idx)
        # metadata = nn.functional.one_hot(step, num_classes=self._episode_len).squeeze(1).to(torch.float32)
        # metadata = self._metadata_encoder(metadata)
        node_idx_one_hot = torch.eye(batch_nodedata.shape[1], dtype=torch.float32).unsqueeze(0).tile((batch_nodedata.shape[0],1,1))
        # print('node_idx_one_hot: ', node_idx_one_hot.shape)
        # print('batch_nodedata shape: ', batch_nodedata.shape)
        batch_nodedata = torch.concat((batch_nodedata, node_idx_one_hot), dim=-1)
        # print('concat data: ', batch_nodedata.shape)
        h_nodes = self._node_feature_encoder(batch_nodedata)

        # assert len(h_edges.shape) == 3



        h_nodes_mean = h_nodes.mean(dim=1)

        # RL task
        if self._task == 'RL':
            assert action_mask is not None
            h_current_node = torch.gather(input=h_nodes, index=current_node_idx.unsqueeze(-1).tile(1, 1, self._gcn_node_dim), dim=1)#.squeeze(dim=1)
            h_attended = self._self_attention(h_current_node, h_nodes).squeeze(1)

            h = torch.concat((h_nodes_mean, h_attended, h_current_node.squeeze(dim=1)), dim=-1) # shape: (batch, gcn_node_dim + gcn_node_dim + gcn_node_dim)
            # h = torch.concat((metadata, h_all_edges, h_attended, h_current_node.squeeze(dim=1)), dim=-1) # shape: (batch, gcn_node_dim + gcn_node_dim + gcn_node_dim)
            # action_logits = self._policy_action_head(h)
            # action_logits = self._transpose_convs(h)

            h_after_transpose_conv = self._transpose_convs(h) # out_shape: [batch, 2, grid_num, grid_num]
            


            layout_features = self._layout_feature_encoder(action_mask.unsqueeze(1)) # use action_mask as layout_features
            action_logits = self._policy_action_head(torch.concat((h_after_transpose_conv, layout_features), dim=1))
            action_logits = action_logits.view(action_logits.shape[0], -1)

            if setting.solver['ppo']['add_noise'] == True:
                action_logits = self._add_noise(action_logits)
            value = self._value_head(h)

            action_mask = action_mask.reshape(-1, action_mask.shape[1] * action_mask.shape[1])
            action_mask = (1- action_mask) * (1e10)
            action_logits = action_logits - action_mask
            return action_logits, value

        else:
            raise NotImplementedError

    def policy_value(self, batch_nodedata, current_node_idx, sparse_adj_i, sparse_adj_j, sparse_adj_weight, action_mask):
        """
        Args:
            batch_node_data:   (batch, node_num, node_feature_dim)
            current_node_idx:  (batch, 1)
            sparse_adj_i:      (batch, edge_num)
            sparse_adj_j:      (batch, edge_num)
            sparse_adj_weight: (batch, edge_num)
            action_mask:       (batch, grid_nums, grid_nums)

        Returns:
            action_logits:     (batch, action_dim)
            value:             (batch, 1)
        """
        assert self._task == 'RL'
        return self.forward(batch_nodedata, current_node_idx, sparse_adj_i, sparse_adj_j, sparse_adj_weight, action_mask)

    def pretrain_value(self, batch_nodedata, current_node_idx, sparse_adj_i, sparse_adj_j, sparse_adj_weight):
        return self.forward(batch_nodedata, current_node_idx, sparse_adj_i, sparse_adj_j, sparse_adj_weight)


    def _self_attention(self, h_current_node, h_nodes):
        """
        Args:      
            h_current_node: A [batch, 1, h] tensor of the current node embedding.
            h_nodes: A [batch, num_nodes, h] tensor of all node embeddings.

        Returns:
            A [batch, h] tensor of the weighted average of the node embeddings where
            the weight is the attention score with respect to the current node.
        """
        assert len(h_current_node.shape) == 3
        assert h_current_node.shape[1] == 1
        query = self._attention_query_layer(h_current_node)
        keys = self._attention_value_layer(h_nodes)
        values = self._attention_value_layer(h_nodes)
        h_attended, _ = self._attention_layer(query=query, key=keys, value=values, need_weights=False)
        # return h_attend.squeeze(1)
        return h_attended

    def _add_noise(self, logits):
        probs = nn.functional.softmax(logits)
        alphas = self._dirichlet_alpha * torch.ones((probs.shape), dtype=torch.float32)
        dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(alphas)
        noise = dirichlet_distribution.sample()
        print('noise shape: noise.shape')
        noised_probs = ((1.0 - self._policy_noise_weight) * probs + (self._policy_noise_weight) * noise)
        noised_logit = torch.log(noised_probs + self.EPSILON)
        return noised_logit

    def _create_metadata_encoder(self, metadata_features, out_features):
        return nn.Sequential(
            nn.Linear(metadata_features, out_features),
            nn.ReLU()
        )

    def _create_node_feature_encoder(self, node_features, gcn_node_features):
        return nn.Sequential(
            nn.Linear(node_features, 64),
            nn.ReLU(),
            nn.Linear(64, gcn_node_features),
            nn.ReLU()
        )

    def _create_layout_feature_encoder(self, channels:list):
        return cnn.SimpleConv(channels[0], channels[-1])
        # layer_list = []
        # for i in range(len(channels)-1):
        #     layer_list.append(cnn.conv3x3(channels[i], channels[i]+1))
        #     layer_list.append(nn.ReLU())
        # return nn.Sequential(*layer_list)


    def _create_edge_fc_list(self, gcn_layer_nums, edge_fc_layer_nums, gcn_node_features):
        return nn.ModuleList([self._create_edge_fc(edge_fc_layer_nums, gcn_node_features) for i in range(gcn_layer_nums)])

    def _create_edge_fc(self, edge_fc_layer_nums, gcn_node_features):
        edge_fc = []
        edge_fc.append(nn.Linear(2*gcn_node_features+1, gcn_node_features))
        edge_fc.append(nn.ReLU())
        for layer_idx in range(edge_fc_layer_nums-1):
            edge_fc.append(nn.Linear(gcn_node_features, gcn_node_features))
            edge_fc.append(nn.ReLU())
        return nn.Sequential(*edge_fc)

    def _create_value_head(self, in_features):
        return nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    # def _create_node_prediction_head(self, in_features):
    #     return nn.Sequential(
    #         nn.Linear(in_features, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, 8),
    #         nn.ReLU(),
    #         nn.Linear(8, 2) # coords (x, y)
    #     )

    def _create_policy_action_head(self, in_channels):
        return nn.Sequential(
            cnn.conv3x3(in_channels=in_channels, out_channels=4, stride=1),
            nn.ReLU(),
            cnn.conv3x3(in_channels=4, out_channels=1, stride=1)
        )


    def _create_multihead_attention_layers(self, embed_dim, num_heads, kdim, vdim):
        return nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=kdim, vdim=vdim, batch_first=True)

    def _create_attention_query_layer(self, gcn_node_features):
        return nn.Linear(gcn_node_features, gcn_node_features)

    def _create_attention_key_layer(self, gcn_node_features):
        return nn.Linear(gcn_node_features, gcn_node_features)

    def _create_attention_value_layer(self, gcn_node_features):
        return nn.Linear(gcn_node_features, gcn_node_features)

    def _create_transpose_convs(self, in_features, grid_nums):
        return TraverseConvs(in_features, grid_nums)



class TraverseConvs(nn.Module):
    def __init__(self, in_features, grid_nums):
        super(TraverseConvs, self).__init__()
        self._grid_nums = grid_nums
        self._fc = nn.Linear(in_features, grid_nums // 16 * grid_nums // 16 * 32)
        self._relu = nn.ReLU()
        # 1*32*2*2   2*2
        self._transpose1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 1*16*4*4   4*4
        self._transpose2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 1*8*8*8    8*8
        self._transpose3 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 1*4*16*16  16*16
        self._transpose4 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 1*2*32*32  32*32
        # self._transpose5 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        # 1*1*32*32  32*32

    def forward(self, x):
        out = self._relu(self._fc(x))
        out = out.reshape(out.shape[0], 32, self._grid_nums//16, self._grid_nums//16)
        out = self._relu(self._transpose1(out))
        out = self._relu(self._transpose2(out))
        out = self._relu(self._transpose3(out))
        out = self._relu(self._transpose4(out))
        # out = self._transpose5(out)
        # out = out.view(out.shape[0], -1)

        return out

