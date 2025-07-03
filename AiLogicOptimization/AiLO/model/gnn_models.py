import torch
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import GINConv, SAGEConv, GATConv
from torch_geometric.nn import TopKPooling, SAGPooling
from torch_geometric.nn import global_max_pool, global_mean_pool
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn


num_node_features = 5 
# num_edge_features = 2

class NodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()
        self.node_type_embedding = torch.nn.Embedding(num_node_features, emb_dim)
        torch.nn.init.xavier_uniform_(self.node_type_embedding.weight.data)

    def forward(self, x):
        x_embedding = self.node_type_embedding(x)
        x_embedding = x_embedding.reshape(x_embedding.shape[0], -1)
        
        return x_embedding

class GNN(torch.nn.Module):
    def __init__(self, node_encoder, input_dim, emb_dim=128, n_layers=4, dropout=0.5):
        super(GNN, self).__init__()
        self.node_encoder = node_encoder
        self.pool1 = SAGPooling(emb_dim, ratio=0.8)
        self.pool2 = SAGPooling(emb_dim, ratio=0.8)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(n_layers):
            in_channels = input_dim if i == 0 else emb_dim
            self.convs.append(SAGEConv(in_channels, emb_dim))
            self.bns.append(torch.nn.BatchNorm1d(emb_dim))


    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        x = self.node_encoder(x)

        for i in range(len(self.convs)):
            x = F.relu(self.bns[i](self.convs[i](x, edge_index)))
        
        x_global = torch.cat((global_mean_pool(x, batch),global_max_pool(x, batch)),dim=-1)
        return x_global

class GIN(torch.nn.Module):
    def __init__(self, node_encoder, input_dim, emb_dim=128, n_layers=4, dropout=0.5):
        super(GIN, self).__init__()
        self.node_encoder = node_encoder
        self.pool1 = SAGPooling(emb_dim, ratio=0.8)
        self.pool2 = SAGPooling(emb_dim, ratio=0.8)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(n_layers):
            in_channels = input_dim if i == 0 else emb_dim
            mlp = nn.Sequential(
                nn.Linear(in_channels, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )
            self.convs.append(GINConv(mlp, eps=0.0, train_eps=False))
            self.bns.append(nn.BatchNorm1d(emb_dim))
        self.dropout = dropout

    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        x = self.node_encoder(x)
        
        for i in range(len(self.convs)):
            x = F.relu(self.bns[i](self.convs[i](x, edge_index)))
            if i == 0:
                x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            elif i == len(self.convs) - 1:
                x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_global_mean = global_mean_pool(x, batch)
        x_global_max = global_max_pool(x, batch)
        x_global = torch.cat((x_global_mean, x_global_max), dim=-1)
        return x_global
